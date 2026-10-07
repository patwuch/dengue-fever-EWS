# explain_attention.py
"""
Visualise GAT attention weights for a trained STGATGRU model.

Attention (alpha) is extracted directly from GATConv via
return_attention_weights=True — no gradients needed, so this runs over the
whole evaluation split cheaply. Node feature preprocessing (missing-node
embedding substitution) is reproduced inline to match model.forward exactly;
see _prepare_features below.

Inputs:
    config (YAML)        — defines name, device, data paths, ...
    model checkpoint      — the trained .pt to inspect
    best_params.json      — hyperparameters used to build the model

Outputs (results/STGNN/<name>/):
    attention_weights.npz      — raw per-edge attention (gat1/gat2 mean + by-lag)
    attention_graph.png        — spatial graph, edge width/colour = mean attention
    attention_over_time.png    — heatmap of top-K edges' gat2 attention by lag step
    self_attention_vs_degree.png
        — diagnostic scatter: each node's self-loop attention weight against
          its degree. Self-loop attention competes with every neighbour in
          the same softmax, so nothing guarantees it survives as degree
          grows — unlike a residual connection. A negative trend here is
          evidence that high-degree nodes' own signal is getting diluted
          before the GRU ever sees it (the GRU has no other path to a
          node's own current-timestep features).
    attention_graph_interactive.html
        — same spatial graph, but click/hover a node to isolate its edges and
          grey out the rest. Colour is the pairwise imbalance (does the
          selected node attend a neighbour more, or the reverse?), line
          weight is their combined strength. A slider scrubs through lag
          steps so the model's temporal structure isn't collapsed into a
          single average. Self-contained, no server needed — open directly
          in a browser.

Run via Snakemake (script mode) or directly:
    python explain_attention.py --config config.yaml \
        [--model best_model.pt] [--params best_params.json] \
        [--split test] [--top-k 15]
"""

import json
import sys
from pathlib import Path

# Ensure the script's own directory is on sys.path before the stdlib so local
# imports (model, tune, choropleth, test) resolve to the files in this
# directory, not to stdlib packages that share their name.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader

from model import STGATGRU
from tune import STGNNDataset, load_tensors, load_edge_index
from choropleth import _node_names, _load_geometry


def _window_target_dates(csv_path: str, n_windows: int) -> list:
    """Calendar (year, month) of the target prediction for each of the last
    n_windows windows in chronological order — i.e. what month each window's
    prediction is actually for, not just its position in the split."""
    df = pd.read_csv(csv_path, usecols=["year_month"])
    all_dates = sorted(pd.to_datetime(df["year_month"].unique()))
    return all_dates[-n_windows:]


# ── Feature prep (mirrors model.forward's masking logic) ──────────────────────

def _prepare_features(model: STGATGRU, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reproduce model.py's missing-node substitution so features fed to gat1
    during attention extraction match what the model saw during training."""
    B, T, N, F_dim = x.size()
    mask = mask.float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(1).unsqueeze(-1)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(-1)

    missing_emb = torch.tanh(model.missing_emb).view(1, 1, 1, F_dim)
    x = x * mask + missing_emb * (1 - mask)
    x = torch.cat([x, mask.expand(B, T, N, 1)], dim=-1)
    return x


# ── Attention collection ───────────────────────────────────────────────────────

@torch.no_grad()
def collect_attention(model: STGATGRU, dataloader: DataLoader,
                       edge_index: torch.Tensor, device: torch.device):
    """
    Run gat1/gat2 with return_attention_weights=True over every window in the
    dataloader, keeping each window's own (lag, edge) attention separate
    rather than averaging it away — windows in this split are one calendar
    month apart, so pooling across them collapses exactly the temporal detail
    this model exists to capture.

    Returns:
        edges      (2, E)   — edge_index as returned by GATConv (includes the
                               self-loops it adds internally)
        attn1      (E,)     — gat1 overall mean across windows and lags
        attn2      (E,)     — gat2 overall mean across windows and lags
        attn1_by_t (T, E)   — gat1 per-lag mean across windows
        attn2_by_t (T, E)   — gat2 per-lag mean across windows
        attn1_all  (W, T, E) — gat1, one value per window and lag (unpooled)
        attn2_all  (W, T, E) — gat2, one value per window and lag (unpooled)
    """
    model.eval()
    ref_edges = None
    per_window_attn1, per_window_attn2 = [], []

    for x, _, mask in dataloader:
        x, mask = x.to(device), mask.to(device)
        x = _prepare_features(model, x, mask)
        B, T, N, _ = x.size()

        for b in range(B):
            attn1_bt = attn2_bt = None
            for t in range(T):
                x_bt = x[b, t].float()
                h1, (ei1, alpha1) = model.gat1(x_bt, edge_index, return_attention_weights=True)
                alpha1 = alpha1.mean(dim=-1)
                h1 = F.elu(h1)
                h1 = model.mlp(h1)
                _, (ei2, alpha2) = model.gat2(h1, edge_index, return_attention_weights=True)
                alpha2 = alpha2.mean(dim=-1)

                if ref_edges is None:
                    ref_edges = ei1.cpu()
                if attn1_bt is None:
                    attn1_bt = torch.zeros(T, ei1.size(1))
                    attn2_bt = torch.zeros(T, ei2.size(1))

                attn1_bt[t] = alpha1.cpu()
                attn2_bt[t] = alpha2.cpu()

            per_window_attn1.append(attn1_bt)
            per_window_attn2.append(attn2_bt)

    attn1_all = torch.stack(per_window_attn1).numpy()  # (W, T, E)
    attn2_all = torch.stack(per_window_attn2).numpy()

    attn1_mean = attn1_all.mean(axis=(0, 1))   # (E,)  overall mean
    attn2_mean = attn2_all.mean(axis=(0, 1))
    attn1_by_t = attn1_all.mean(axis=0)         # (T, E) per-lag mean across windows
    attn2_by_t = attn2_all.mean(axis=0)

    return ref_edges.numpy(), attn1_mean, attn2_mean, attn1_by_t, attn2_by_t, attn1_all, attn2_all


# ── Plotting ────────────────────────────────────────────────────────────────────

def plot_attention_graph(edges: np.ndarray, attn1: np.ndarray, attn2: np.ndarray,
                          node_names: list, gdf, out_path: Path, top_k: int = None) -> None:
    combined  = (attn1 + attn2) / 2
    centroids = gdf.set_index("name").loc[node_names].geometry.centroid
    xy = np.array([(pt.x, pt.y) for pt in centroids])

    keep = edges[0] != edges[1]  # drop self-loops for the spatial plot
    src, dst, w = edges[0][keep], edges[1][keep], combined[keep]

    if top_k:
        order = np.argsort(w)[::-1][:top_k]
        src, dst, w = src[order], dst[order], w[order]

    segments = [[xy[s], xy[d]] for s, d in zip(src, dst)]
    norm = matplotlib.colors.Normalize(vmin=float(w.min()), vmax=float(w.max()))
    cmap = matplotlib.colormaps["viridis"]
    lc = LineCollection(segments, colors=cmap(norm(w)), linewidths=1 + 6 * norm(w))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], s=25, color="black", zorder=3)
    for name, (x0, y0) in zip(node_names, xy):
        ax.annotate(name, (x0, y0), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Mean GAT attention (gat1+gat2)/2 — edge width/colour = attention", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Mean attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_attention_over_time(edges: np.ndarray, attn2_by_t: np.ndarray,
                              node_names: list, out_path: Path, top_k: int = 15) -> None:
    T = attn2_by_t.shape[0]
    keep = edges[0] != edges[1]
    edges_keep = edges[:, keep]
    attn_keep  = attn2_by_t[:, keep]

    overall = attn_keep.mean(axis=0)
    order   = np.argsort(overall)[::-1][:min(top_k, len(overall))]
    labels  = [f"{node_names[edges_keep[0, i]]} → {node_names[edges_keep[1, i]]}" for i in order]
    data    = attn_keep[:, order].T  # (top_k, T)

    fig, ax = plt.subplots(figsize=(1 + 0.5 * T, 0.35 * len(order) + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"t-{T - 1 - i}" for i in range(T)], fontsize=8)
    ax.set_xlabel("Lag (months before prediction)")
    ax.set_title(f"Top {len(order)} edges — gat2 attention over lag steps", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_self_attention_diagnostics(edges: np.ndarray, attn1: np.ndarray, attn2: np.ndarray,
                                        n_nodes: int) -> tuple:
    """Per-node self-loop attention and degree, aligned by node index.

    GATConv's self-loop weight competes with every geographic neighbour in
    the same softmax, so it isn't guaranteed to survive as degree grows —
    unlike a residual/skip connection, nothing forces the model to keep a
    node's own signal. That matters here because the GRU has no other path
    to a node's own current-timestep features; everything is filtered
    through gat1/gat2 first. Degree is read off edges[1] (softmax's target
    axis) after GATConv's own self-loop insertion, so it exactly matches the
    neighbour count the model normalised over — not a re-derivation from the
    original edge_index.

    Returns:
        degree      (N,) — neighbours per node, self-loop excluded
        self_attn1  (N,) — gat1 self-loop weight per node
        self_attn2  (N,) — gat2 self-loop weight per node
    """
    self_mask  = edges[0] == edges[1]
    self_nodes = edges[0][self_mask]
    degree     = np.bincount(edges[1], minlength=n_nodes) - 1

    self_attn1 = np.zeros(n_nodes)
    self_attn2 = np.zeros(n_nodes)
    self_attn1[self_nodes] = attn1[self_mask]
    self_attn2[self_nodes] = attn2[self_mask]

    return degree, self_attn1, self_attn2


def plot_self_attention_vs_degree(degree: np.ndarray, self_attn1: np.ndarray, self_attn2: np.ndarray,
                                   node_names: list, out_path: Path) -> None:
    """Self-loop attention against degree only has something to say if
    degree actually varies across nodes — on a near-complete graph (every
    node attends nearly every other node, as is the case for this SEA
    province adjacency) degree is constant and a raw scatter is degenerate.
    The question that still has an answer either way is whether a node's
    self-loop weight is above or below its "fair share" if attention were
    split evenly across its degree+1 softmax entries (uniform baseline =
    1/(degree+1)) — that's what's plotted, as a ratio, against degree.
    """
    combined = (self_attn1 + self_attn2) / 2
    uniform  = 1.0 / (degree + 1)
    ratio    = combined / uniform  # >1: self attended more than a fair share; <1: diluted below it

    degree_varies = degree.max() > degree.min()
    r = float(np.corrcoef(degree, combined)[0, 1]) if len(degree) > 1 and degree_varies else float("nan")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(1.0, color="#898781", linestyle="--", linewidth=1, zorder=1,
               label="uniform baseline (1 / (degree+1))")
    ax.scatter(degree, ratio, s=30, color="#2a78d6", zorder=3)
    for name, x0, y0 in zip(node_names, degree, ratio):
        ax.annotate(name, (x0, y0), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Node degree (neighbours in softmax, self-loop excluded)")
    ax.set_ylabel("Self-loop attention ÷ uniform baseline")
    below_frac = float((ratio < 1.0).mean())
    title = f"Self-attention vs. fair share — {below_frac:.0%} of nodes below baseline"
    if degree_varies:
        title += f" (r vs. degree = {r:.3f})"
    else:
        title += " (degree is constant — graph is near-complete)"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return below_frac, r


# ── Interactive HTML graph ──────────────────────────────────────────────────────

_INTERACTIVE_BODY = """
<style>
  html, body { height: 100%; margin: 0; }
  .attn-root {
    --surface-1:      #fcfcfb;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --muted:          #898781;
    --gridline:       #e1e0d9;
    --border:         rgba(11,11,11,0.10);
    --diverge-pos:    #2a78d6; /* selected node attends the neighbour more */
    --diverge-neg:    #e34948; /* neighbour attends the selected node more */
    --diverge-mid:    #c3c2b7; /* balanced / mutual */
    --accent:         #0b0b0b; /* selected node marker */
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
    background: var(--surface-1);
  }
  @media (prefers-color-scheme: dark) {
    .attn-root {
      --surface-1:      #1a1a19;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --muted:          #898781;
      --gridline:       #2c2c2a;
      --border:         rgba(255,255,255,0.10);
      --diverge-pos:    #3987e5;
      --diverge-neg:    #e66767;
      --diverge-mid:    #52514e;
      --accent:         #ffffff;
    }
  }
  :root[data-theme="dark"] .attn-root {
    --surface-1:      #1a1a19;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --border:         rgba(255,255,255,0.10);
    --diverge-pos:    #3987e5;
    --diverge-neg:    #e66767;
    --diverge-mid:    #52514e;
    --accent:         #ffffff;
  }
  :root[data-theme="light"] .attn-root {
    --surface-1:      #fcfcfb;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --muted:          #898781;
    --gridline:       #e1e0d9;
    --border:         rgba(11,11,11,0.10);
    --diverge-pos:    #2a78d6;
    --diverge-neg:    #e34948;
    --diverge-mid:    #c3c2b7;
    --accent:         #0b0b0b;
  }
  .attn-root {
    height: 100%; box-sizing: border-box; padding: 16px;
    display: flex; flex-direction: column; overflow: hidden;
  }
  .attn-header { margin-bottom: 8px; flex: 0 0 auto; }
  .attn-header h1 { font-size: 16px; margin: 0 0 4px; }
  .attn-header p { font-size: 12.5px; color: var(--text-secondary); margin: 0; max-width: 70ch; }
  .attn-toolbar {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    margin: 12px 0; padding-bottom: 12px; border-bottom: 1px solid var(--gridline);
    flex: 0 0 auto;
  }
  .attn-search input {
    font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
    width: 240px;
  }
  .attn-legend { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--text-secondary); }
  .attn-legend .bar {
    width: 140px; height: 8px; border-radius: 4px;
    background: linear-gradient(to right, var(--diverge-neg), var(--diverge-mid), var(--diverge-pos));
  }
  .attn-clear {
    font: inherit; font-size: 12.5px; padding: 5px 10px; border-radius: 6px;
    border: 1px solid var(--border); background: transparent; color: var(--text-secondary);
    cursor: pointer;
  }
  .attn-clear:hover { color: var(--text-primary); border-color: var(--text-secondary); }
  .attn-timeline {
    display: flex; align-items: center; gap: 10px; margin: 0 0 12px;
    font-size: 12.5px; color: var(--text-secondary);
    flex: 0 0 auto;
  }
  .attn-timeline input[type="range"] { flex: 1 1 auto; accent-color: var(--diverge-pos); }
  .attn-timeline .lag-label {
    font-variant-numeric: tabular-nums; color: var(--text-primary); width: 8.5em; flex: none;
  }
  .attn-layout { display: flex; gap: 16px; flex: 1 1 auto; min-height: 0; }
  .attn-canvas-wrap {
    flex: 1 1 auto; min-width: 0; min-height: 0;
    border: 1px solid var(--gridline); border-radius: 8px;
  }
  .attn-canvas-wrap svg { display: block; width: 100%; height: 100%; }
  .attn-node-hit { cursor: pointer; }
  .attn-node-hit:focus { outline: none; }
  .attn-node-hit:focus-visible + circle.attn-node-dot { stroke: var(--text-primary); stroke-width: 2; }
  .attn-node-label { font-size: 9px; fill: var(--text-primary); pointer-events: none; }
  .attn-sidebar {
    flex: 0 0 260px; font-size: 12.5px; color: var(--text-secondary);
    border: 1px solid var(--gridline); border-radius: 8px; padding: 12px;
    height: 100%; overflow-y: auto; box-sizing: border-box;
  }
  .attn-sidebar h2 { font-size: 13.5px; color: var(--text-primary); margin: 0 0 8px; }
  .attn-sidebar h3 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em;
    color: var(--text-secondary); margin: 14px 0 6px;
  }
  .attn-sidebar .placeholder { color: var(--muted); }
  .attn-row {
    display: flex; justify-content: space-between; gap: 8px; padding: 3px 0;
    cursor: pointer; border-radius: 4px;
  }
  .attn-row:hover { background: var(--gridline); }
  .attn-row .name { color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .attn-row .val { font-variant-numeric: tabular-nums; color: var(--text-secondary); flex: none; }
  .attn-row .key { display: inline-block; width: 10px; height: 2px; margin-right: 6px; vertical-align: middle; }
</style>

<div class="attn-root">
  <div class="attn-header">
    <h1>__TITLE__</h1>
    <p>GAT attention here is close to a complete graph — every node attends to nearly every other node, so direction alone isn't informative. Click or focus a node instead to see, for each neighbour, which side of the relationship dominates: colour is the <em>imbalance</em> between how much the selected node attends the neighbour and how much the neighbour attends it back; line weight is their combined attention strength. Everything unrelated fades. Two sliders pick an exact point in the model's attention: which prediction month you're looking at, and which lag step in that month's input lookback — nothing here is averaged across time, since collapsing that structure away would erase the whole reason an ST-GNN exists over a plain GAT.</p>
  </div>
  <div class="attn-toolbar">
    <div class="attn-search">
      <input type="text" id="attn-search-input" list="attn-node-list" placeholder="Jump to a node…" autocomplete="off">
      <datalist id="attn-node-list"></datalist>
    </div>
    <div class="attn-legend">
      <span>Neighbour attends selected more</span>
      <span class="bar"></span>
      <span>Selected attends neighbour more</span>
    </div>
    <button class="attn-clear" id="attn-clear-btn" type="button">Clear selection</button>
  </div>
  <div class="attn-timeline">
    <span>Month:</span>
    <input type="range" id="attn-month-slider" min="0" max="0" value="0" step="1">
    <span class="lag-label" id="attn-month-label">—</span>
  </div>
  <div class="attn-timeline">
    <span>Lag:</span>
    <input type="range" id="attn-lag-slider" min="0" max="0" value="0" step="1">
    <span class="lag-label" id="attn-lag-label">—</span>
  </div>
  <div class="attn-layout">
    <div class="attn-canvas-wrap">
      <svg id="attn-svg" viewBox="0 0 __VB_W__ __VB_H__" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="attn-sidebar" id="attn-sidebar">
      <h2>No node selected</h2>
      <p class="placeholder">Click, tab to, or search for a node to see its attention detail here.</p>
    </div>
  </div>
</div>

<script type="application/json" id="attn-data">__DATA__</script>
<script>
(function () {
  const root = document.querySelector(".attn-root");
  const data = JSON.parse(document.getElementById("attn-data").textContent);
  const nodes = data.nodes;   // [{name, x, y}]
  const edges = data.edges;   // [{s, d, g}] — g is (month, lag) -> attention
  const months = data.months; // [label, ...] one per prediction window

  const svg = document.getElementById("attn-svg");
  const NS = "http://www.w3.org/2000/svg";

  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const VB_W = __VB_W__, VB_H = __VB_H__, PAD = __VB_PAD__;
  const sx = v => PAD + (v - xMin) / (xMax - xMin || 1) * (VB_W - 2 * PAD);
  const sy = v => VB_H - (PAD + (v - yMin) / (yMax - yMin || 1) * (VB_H - 2 * PAD)); // flip: north = up

  // PyG's GATConv returns edge_index as (source, target) pairs with the
  // attention weight indexed the same way: alpha for edge (s, d) is how much
  // the TARGET (d) attends to the SOURCE (s) — i.e. "d attends to s", not the
  // other way round. wMap keys on "attender_attendee"; each record holds the
  // full (month, lag) grid, so att() below always answers "how much does i
  // attend j" for one specific prediction window at one specific lag step —
  // nothing is pooled across either axis.
  const W = months.length;
  const T = edges.length ? edges[0].g[0].length : 0;
  const wMap = new Map();
  edges.forEach(e => wMap.set(e.d + "_" + e.s, e.g));
  function att(i, j, monthIdx, lagIdx) {
    const rec = wMap.get(i + "_" + j);
    if (!rec) return undefined;
    return rec[monthIdx][lagIdx];
  }

  const neighborSet = nodes.map(() => new Set());
  edges.forEach(e => { neighborSet[e.s].add(e.d); neighborSet[e.d].add(e.s); });

  // Attention here is a softmax over ~N neighbours, so raw weights are
  // squeezed into a narrow, right-skewed band (a handful of outliers set the
  // max and compress everything else toward the low end of a linear scale).
  // Rank each pair's combined magnitude against every other pair *within the
  // same (month, lag) frame* instead, so line weight spreads across the full
  // visual range regardless of skew, and stays comparable across frames.
  function buildFrame(monthIdx, lagIdx) {
    const seen = new Set();
    const pairs = [];
    let maxAbsDiff = 0;
    edges.forEach(e => {
      const i = e.d, j = e.s; // i attends j
      const key = Math.min(i, j) + "_" + Math.max(i, j);
      if (seen.has(key)) return;
      seen.add(key);
      const w1 = att(i, j, monthIdx, lagIdx) ?? 0, w2 = att(j, i, monthIdx, lagIdx) ?? 0;
      pairs.push({ i, j, mag: (w1 + w2) / 2 });
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(w1 - w2));
    });
    pairs.sort((a, b) => b.mag - a.mag);
    const sortedMags = pairs.map(p => p.mag).slice().sort((a, b) => a - b);
    return { pairs, sortedMags, maxAbsDiff };
  }
  // W x T frames precomputed up front — cheap (a few million comparisons at
  // most) and keeps every slider move a plain array lookup.
  const frames = [];
  for (let m = 0; m < W; m++) {
    const row = [];
    for (let t = 0; t < T; t++) row.push(buildFrame(m, t));
    frames.push(row);
  }
  let currentMonth = W - 1;
  let currentLag = T - 1;

  function magPercentileAt(w, monthIdx, lagIdx) {
    const mags = frames[monthIdx][lagIdx].sortedMags;
    if (mags.length === 0) return 0.5;
    let lo = 0, hi = mags.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (mags[mid] < w) lo = mid + 1; else hi = mid;
    }
    return lo / mags.length;
  }

  function monthLabel(monthIdx) {
    return months[monthIdx];
  }

  function lagLabel(lagIdx) {
    return `t-${T - 1 - lagIdx}`;
  }

  function hexToRgb(hex) {
    const n = parseInt(hex.replace("#", ""), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function rgbToHex(rgb) {
    return "#" + rgb.map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, "0")).join("");
  }
  function lerpColor(a, b, t) {
    const ca = hexToRgb(a), cb = hexToRgb(b);
    return rgbToHex(ca.map((v, i) => v + (cb[i] - v) * t));
  }
  const style = getComputedStyle(root);
  const cPos = style.getPropertyValue("--diverge-pos").trim();
  const cNeg = style.getPropertyValue("--diverge-neg").trim();
  const cMid = style.getPropertyValue("--diverge-mid").trim();

  // diff > 0: i attends j MORE than j attends i -> cool/blue.
  // diff < 0: j attends i MORE than i attends j -> warm/red.
  // Scale is per-frame (frames[monthIdx][lagIdx].maxAbsDiff) so a given colour
  // means the same imbalance regardless of which node, month, or lag is
  // currently selected.
  function divergingColor(diff, monthIdx, lagIdx) {
    const maxAbsDiff = frames[monthIdx][lagIdx].maxAbsDiff;
    if (maxAbsDiff <= 0) return cMid;
    const t = Math.max(-1, Math.min(1, diff / maxAbsDiff));
    return t >= 0 ? lerpColor(cMid, cPos, t) : lerpColor(cMid, cNeg, -t);
  }

  // With N nodes this graph can have up to N*(N-1) edges (GAT attends over
  // all neighbours, not just a sparse geographic adjacency) — far too many to
  // keep as live DOM elements. Draw a fixed, cheap "backbone" of only the
  // strongest edges for context; a node's own edges are built on demand in
  // highlight() below, so interaction cost depends on that node's degree, not
  // on the size of the whole graph.
  const CONTEXT_EDGE_COUNT = 150;
  let contextEls = [];
  const gContext = document.createElementNS(NS, "g");
  svg.appendChild(gContext);

  function rebuildContext(monthIdx, lagIdx) {
    while (gContext.firstChild) gContext.removeChild(gContext.firstChild);
    contextEls = [];
    frames[monthIdx][lagIdx].pairs.slice(0, CONTEXT_EDGE_COUNT).forEach(p => {
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", sx(nodes[p.i].x));
      line.setAttribute("y1", sy(nodes[p.i].y));
      line.setAttribute("x2", sx(nodes[p.j].x));
      line.setAttribute("y2", sy(nodes[p.j].y));
      line.setAttribute("stroke", "var(--muted)");
      line.setAttribute("stroke-width", "1");
      line.setAttribute("opacity", "0.15");
      gContext.appendChild(line);
      contextEls.push(line);
    });
  }

  const gSel = document.createElementNS(NS, "g");
  svg.appendChild(gSel);

  const gNodes = document.createElementNS(NS, "g");
  const gLabels = document.createElementNS(NS, "g");
  const dotEls = [];
  nodes.forEach((n, i) => {
    const cx = sx(n.x), cy = sy(n.y);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("class", "attn-node-dot");
    dot.setAttribute("cx", cx);
    dot.setAttribute("cy", cy);
    dot.setAttribute("r", "2.5");
    dot.setAttribute("fill", "var(--muted)");
    dot.setAttribute("opacity", "0.35");
    dotEls.push(dot);

    const hit = document.createElementNS(NS, "circle");
    hit.setAttribute("class", "attn-node-hit");
    hit.setAttribute("cx", cx);
    hit.setAttribute("cy", cy);
    hit.setAttribute("r", "9");
    hit.setAttribute("fill", "transparent");
    hit.setAttribute("tabindex", "0");
    hit.setAttribute("role", "button");
    hit.setAttribute("aria-label", n.name);

    hit.addEventListener("mouseenter", () => { if (!pinned) highlight(i); });
    hit.addEventListener("mouseleave", () => { if (!pinned) highlight(null); });
    hit.addEventListener("focus", () => { if (!pinned) highlight(i); });
    hit.addEventListener("click", () => { pinned = (pinned === i) ? null : i; highlight(pinned); });
    hit.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        pinned = (pinned === i) ? null : i;
        highlight(pinned);
      }
    });

    gNodes.appendChild(dot);
    gNodes.appendChild(hit);
  });
  svg.appendChild(gNodes);
  svg.appendChild(gLabels);

  svg.addEventListener("click", (ev) => {
    if (ev.target === svg) { pinned = null; highlight(null); }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { pinned = null; highlight(null); }
  });

  const sidebar = document.getElementById("attn-sidebar");
  let pinned = null;

  function clearLabels() {
    while (gLabels.firstChild) gLabels.removeChild(gLabels.firstChild);
  }

  function addLabel(i, color) {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("class", "attn-node-label");
    t.setAttribute("x", sx(nodes[i].x) + 5);
    t.setAttribute("y", sy(nodes[i].y) - 5);
    t.setAttribute("fill", color);
    t.textContent = nodes[i].name;
    gLabels.appendChild(t);
  }

  function renderSidebar(i) {
    if (i === null) {
      sidebar.innerHTML = "";
      const h2 = document.createElement("h2");
      h2.textContent = "No node selected";
      const p = document.createElement("p");
      p.className = "placeholder";
      p.textContent = "Click, tab to, or search for a node to see its attention detail here.";
      sidebar.appendChild(h2);
      sidebar.appendChild(p);
      return;
    }
    sidebar.innerHTML = "";
    const h2 = document.createElement("h2");
    h2.textContent = nodes[i].name;
    sidebar.appendChild(h2);

    const h3 = document.createElement("h3");
    h3.textContent = "Neighbours, by imbalance";
    sidebar.appendChild(h3);

    const neighbours = Array.from(neighborSet[i]);
    if (neighbours.length === 0) {
      const p = document.createElement("p");
      p.className = "placeholder";
      p.textContent = "No edges.";
      sidebar.appendChild(p);
      return;
    }
    const rows = neighbours.map(j => {
      const toJ   = att(i, j, currentMonth, currentLag) ?? 0; // selected attends neighbour
      const fromJ = att(j, i, currentMonth, currentLag) ?? 0; // neighbour attends selected
      return { j, toJ, fromJ, diff: toJ - fromJ, mag: (toJ + fromJ) / 2 };
    });
    rows.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));

    rows.forEach(r => {
      const row = document.createElement("div");
      row.className = "attn-row";
      const nameSpan = document.createElement("span");
      nameSpan.className = "name";
      const key = document.createElement("span");
      key.className = "key";
      key.style.background = divergingColor(r.diff, currentMonth, currentLag);
      nameSpan.appendChild(key);
      const label = document.createElement("span");
      label.textContent = nodes[r.j].name;
      nameSpan.appendChild(label);
      const val = document.createElement("span");
      val.className = "val";
      val.title = `selected→neighbour ${r.toJ.toFixed(3)}, neighbour→selected ${r.fromJ.toFixed(3)}`;
      val.textContent = (r.diff >= 0 ? "+" : "") + r.diff.toFixed(3);
      row.appendChild(nameSpan);
      row.appendChild(val);
      row.addEventListener("click", () => { pinned = r.j; highlight(pinned); });
      sidebar.appendChild(row);
    });
  }

  const MAX_LABELS = 30; // hub nodes can have ~N-1 neighbours; cap labels so text stays legible

  function highlight(i) {
    clearLabels();
    while (gSel.firstChild) gSel.removeChild(gSel.firstChild);

    if (i === null) {
      contextEls.forEach(el => el.setAttribute("opacity", "0.15"));
      dotEls.forEach(el => { el.setAttribute("opacity", "0.35"); el.setAttribute("fill", "var(--muted)"); el.setAttribute("r", "2.5"); });
      renderSidebar(null);
      return;
    }
    contextEls.forEach(el => el.setAttribute("opacity", "0.03"));

    // Build only this node's own edges — cost scales with its degree, not
    // with the size of the whole (near-complete) attention graph.
    const neighbours = Array.from(neighborSet[i]);
    const rows = neighbours.map(j => {
      const toJ   = att(i, j, currentMonth, currentLag) ?? 0;
      const fromJ = att(j, i, currentMonth, currentLag) ?? 0;
      return { j, diff: toJ - fromJ, mag: (toJ + fromJ) / 2 };
    });

    rows.forEach(r => {
      const n = magPercentileAt(r.mag, currentMonth, currentLag); // rank, not raw magnitude — see note above
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", sx(nodes[i].x));
      line.setAttribute("y1", sy(nodes[i].y));
      line.setAttribute("x2", sx(nodes[r.j].x));
      line.setAttribute("y2", sy(nodes[r.j].y));
      line.setAttribute("stroke", divergingColor(r.diff, currentMonth, currentLag));
      line.setAttribute("opacity", (0.25 + 0.75 * n).toFixed(2));
      line.setAttribute("stroke-width", (0.75 + 6 * n).toFixed(2));
      gSel.appendChild(line);
    });

    const labelOrder = rows.slice().sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff)).slice(0, MAX_LABELS);
    const labelSet = new Map(labelOrder.map(r => [r.j, r.diff]));

    dotEls.forEach((el, idx) => {
      if (idx === i) {
        el.setAttribute("r", "5.5");
        el.setAttribute("fill", "var(--accent)");
        el.setAttribute("opacity", "1");
      } else if (neighborSet[i].has(idx)) {
        const diff = rows.find(r => r.j === idx).diff;
        el.setAttribute("r", "4");
        el.setAttribute("fill", divergingColor(diff, currentMonth, currentLag));
        el.setAttribute("opacity", "1");
        if (labelSet.has(idx)) addLabel(idx, divergingColor(diff, currentMonth, currentLag));
      } else {
        el.setAttribute("r", "2");
        el.setAttribute("fill", "var(--muted)");
        el.setAttribute("opacity", "0.12");
      }
    });

    renderSidebar(i);
  }

  const datalist = document.getElementById("attn-node-list");
  nodes.forEach(n => {
    const opt = document.createElement("option");
    opt.value = n.name;
    datalist.appendChild(opt);
  });
  const searchInput = document.getElementById("attn-search-input");
  function trySelectByName(name) {
    const idx = nodes.findIndex(n => n.name === name);
    if (idx >= 0) { pinned = idx; highlight(idx); }
  }
  searchInput.addEventListener("change", () => trySelectByName(searchInput.value));
  searchInput.addEventListener("input", () => {
    if (nodes.some(n => n.name === searchInput.value)) trySelectByName(searchInput.value);
  });

  document.getElementById("attn-clear-btn").addEventListener("click", () => {
    pinned = null;
    searchInput.value = "";
    highlight(null);
  });

  const monthSlider = document.getElementById("attn-month-slider");
  const monthLabelEl = document.getElementById("attn-month-label");
  const lagSlider = document.getElementById("attn-lag-slider");
  const lagLabelEl = document.getElementById("attn-lag-label");

  monthSlider.min = 0; monthSlider.max = Math.max(0, W - 1); monthSlider.value = currentMonth;
  lagSlider.min = 0; lagSlider.max = Math.max(0, T - 1); lagSlider.value = currentLag;
  monthLabelEl.textContent = monthLabel(currentMonth);
  lagLabelEl.textContent = lagLabel(currentLag);

  monthSlider.addEventListener("input", () => {
    currentMonth = Number(monthSlider.value);
    monthLabelEl.textContent = monthLabel(currentMonth);
    rebuildContext(currentMonth, currentLag);
    if (pinned !== null) highlight(pinned);
  });
  lagSlider.addEventListener("input", () => {
    currentLag = Number(lagSlider.value);
    lagLabelEl.textContent = lagLabel(currentLag);
    rebuildContext(currentMonth, currentLag);
    if (pinned !== null) highlight(pinned);
  });

  rebuildContext(currentMonth, currentLag);
  highlight(null);
})();
</script>
"""


def write_interactive_graph(edges: np.ndarray, attn1_all: np.ndarray, attn2_all: np.ndarray,
                             node_names: list, gdf, out_path: Path, split: str,
                             window_dates: list) -> None:
    """Self-contained HTML: click/hover a node to isolate its edges and grey
    out the rest of the network. Two sliders — calendar month and lag step —
    index into the model's actual (window, lag) attention grid, so nothing is
    pooled away: what you see is the attention for one specific prediction
    window at one specific point in its input lookback. No server or
    external assets required."""
    combined_all = (attn1_all + attn2_all) / 2   # (W, T, E) — gat1/gat2 mean, nothing else pooled
    centroids = gdf.set_index("name").loc[node_names].geometry.centroid
    keep = edges[0] != edges[1]  # drop self-loops, as in the static plot

    nodes_json = [
        {"name": name, "x": float(pt.x), "y": float(pt.y)}
        for name, pt in zip(node_names, centroids)
    ]
    # Round to keep the embedded JSON a manageable size — W x T x E floats adds
    # up fast (W~14, T=12, E~26k is already millions of numbers) — 5
    # significant figures is far below the noise floor of an attention weight.
    grid_kept = combined_all[:, :, keep].transpose(2, 0, 1)  # (E_kept, W, T)
    edges_json = [
        {"s": int(s), "d": int(d),
         "g": [[round(float(v), 5) for v in lag_row] for lag_row in window_grid]}
        for s, d, window_grid in zip(edges[0][keep], edges[1][keep], grid_kept)
    ]

    month_labels = [d.strftime("%b %Y") for d in window_dates]

    payload = json.dumps({
        "nodes": nodes_json, "edges": edges_json, "months": month_labels,
    }).replace("</", "<\\/")

    # viewBox matches the data's lon/lat aspect ratio so the map isn't stretched
    # (equivalent to matplotlib's ax.set_aspect("equal") in the static plot).
    xs = [pt.x for pt in centroids]
    ys = [pt.y for pt in centroids]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    aspect = x_range / y_range
    if aspect >= 1:
        vb_w, vb_h = 1000, round(1000 / aspect)
    else:
        vb_w, vb_h = round(1000 * aspect), 1000

    body = (
        _INTERACTIVE_BODY
        .replace("__DATA__", payload)
        .replace("__TITLE__", "STGNN attention network")
        .replace("__SPLIT__", split)
        .replace("__VB_W__", str(vb_w))
        .replace("__VB_H__", str(vb_h))
        .replace("__VB_PAD__", "40")
    )
    html = (
        "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>STGNN attention network</title>\n</head>\n<body>\n"
        + body +
        "\n</body>\n</html>\n"
    )
    out_path.write_text(html)


# ── Main ────────────────────────────────────────────────────────────────────────

def explain_attention(cfg: dict, params: dict, model_path, out_dir: Path,
                       csv_path: str, geom_path: str,
                       split: str = "test", top_k: int = 15,
                       tensors_path: Path | None = None,
                       edge_index_path: Path | None = None) -> None:
    model_path = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    device      = torch.device(cfg.get("device", "cpu"))
    window_size = params["window_size"]

    tensors = load_tensors(cfg, window_size, tensors_path)
    if split == "test" and "inference_x" in tensors:
        split = "inference"
        print("INFO: using inference split — attention covers all test months.")

    dataset     = STGNNDataset(tensors, split)
    dataloader  = DataLoader(dataset, batch_size=8, shuffle=False)
    edge_index  = load_edge_index(cfg, device, edge_index_path)
    in_channels = dataset.x.shape[-1]

    model = STGATGRU(
        in_channels  = in_channels,
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params.get("mlp_hidden", params["gat1_hidden"] * params["gat1_heads"]),
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params["gru_hidden"],
        pred_horizon = 1,
        dropout      = params["dropout"],
    ).to(device)
    print(f"Loading model from {model_path} ...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    edges, attn1, attn2, attn1_by_t, attn2_by_t, attn1_all, attn2_all = collect_attention(
        model, dataloader, edge_index, device
    )

    node_names   = _node_names(csv_path)
    gdf          = _load_geometry(geom_path)
    window_dates = _window_target_dates(csv_path, attn1_all.shape[0])

    # combined_by_window is (gat1+gat2)/2 per (window, lag, edge) — the same
    # unpooled resolution the interactive HTML uses, saved here so downstream
    # scripts (e.g. attention_shift.py) don't need to reload the model just
    # to recompute it.
    combined_by_window = (attn1_all + attn2_all) / 2
    degree, self_attn1, self_attn2 = compute_self_attention_diagnostics(
        edges, attn1, attn2, len(node_names)
    )
    npz_path = out_dir / "attention_weights.npz"
    np.savez_compressed(
        npz_path,
        edges              = edges,
        attn1_mean         = attn1,
        attn2_mean         = attn2,
        attn1_by_time      = attn1_by_t,
        attn2_by_time      = attn2_by_t,
        combined_by_window = combined_by_window.astype(np.float32),
        window_dates       = np.array([d.isoformat() for d in window_dates]),
        node_names         = np.array(node_names),
        node_degree        = degree,
        self_attn1         = self_attn1,
        self_attn2         = self_attn2,
    )
    print(f"Attention weights saved to {npz_path}")

    graph_path = out_dir / "attention_graph.png"
    plot_attention_graph(edges, attn1, attn2, node_names, gdf, graph_path, top_k=top_k)
    print(f"  -> {graph_path}")

    time_path = out_dir / "attention_over_time.png"
    plot_attention_over_time(edges, attn2_by_t, node_names, time_path, top_k=top_k)
    print(f"  -> {time_path}")

    self_attn_path = out_dir / "self_attention_vs_degree.png"
    below_frac, r = plot_self_attention_vs_degree(degree, self_attn1, self_attn2, node_names, self_attn_path)
    print(f"  -> {self_attn_path}  ({below_frac:.0%} of nodes below uniform self-attention baseline)")

    interactive_path = out_dir / "attention_graph_interactive.html"
    write_interactive_graph(
        edges, attn1_all, attn2_all, node_names, gdf, interactive_path,
        split=split, window_dates=window_dates,
    )
    print(f"  -> {interactive_path}")

    print(f"\nAll outputs saved to {out_dir}")


# ── Entry point (Snakemake script mode or CLI) ────────────────────────────────

def _main_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Visualise GAT attention weights for a trained STGATGRU model.")
    ap.add_argument("--config", required=True, help="Path to config YAML.")
    ap.add_argument("--model", default=None,
                    help="Checkpoint .pt to inspect (default: results dir best_model.pt).")
    ap.add_argument("--params", default=None,
                    help="best_params.json used to build the model (default: results dir).")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which split to compute attention over (default: test).")
    ap.add_argument("--top-k", type=int, default=15,
                    help="Number of top edges to highlight in the plots.")
    ap.add_argument("--csv-path", default=None,
                    help="Path to merged dengue-env CSV (default: computed from project root).")
    ap.add_argument("--geom-path", default=None,
                    help="Path to geometry Parquet (default: computed from project root).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir     = Path("results/STGNN") / cfg["name"]
    params_path = Path(args.params) if args.params else out_dir / "best_params.json"
    with open(params_path) as f:
        params = json.load(f)
    model_path = Path(args.model) if args.model else out_dir / "best_model.pt"
    csv_path, geom_path = _resolve_data_paths(args.csv_path, args.geom_path)
    explain_attention(cfg, params, model_path, out_dir, csv_path, geom_path,
                      split=args.split, top_k=args.top_k)


def _resolve_data_paths(csv_path: str | None, geom_path: str | None) -> tuple[str, str]:
    """Resolve CSV and geometry paths; compute from project root if not given."""
    if csv_path and geom_path:
        return csv_path, geom_path
    _root = Path(__file__).resolve().parent
    for _ in range(6):
        if (_root / ".git").exists():
            break
        _root = _root.parent
    return (
        csv_path or str(_root / "data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv"),
        geom_path or str(_root / "data/processed/dengue-infection/geoparquet/gaul_2024_sea_filtered.parquet"),
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        with open(snakemake.params.cfg) as f:            # noqa: F821
            cfg = yaml.safe_load(f)
        out_dir    = Path(snakemake.params.results_dir)    # noqa: F821
        with open(snakemake.params.best_params) as f:     # noqa: F821
            params = json.load(f)
        model_path = getattr(snakemake.input, "model", None)  # noqa: F821
        model_path = Path(model_path) if model_path else out_dir / "best_model.pt"
        top_k = cfg.get("explain", {}).get("attention_top_k", 15)
        explain_attention(
            cfg, params, model_path, out_dir,
            csv_path=snakemake.params.csv_path,   # noqa: F821
            geom_path=snakemake.params.geom_path,  # noqa: F821
            split="test", top_k=top_k,
            tensors_path    = Path(snakemake.input.tensors),     # noqa: F821
            edge_index_path = Path(snakemake.input.edge_index),  # noqa: F821
        )
    else:
        _main_cli()
