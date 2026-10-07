# explain_shap.py
"""
Node-, timestep-, and feature-specific attribution via SHAP's GradientExplainer
(Expected Gradients — integrated gradients averaged over a background sample
set, rather than a single fixed baseline).

Design notes / limitations:
  - edge_index is a fixed structural input, not perturbed — this attributes
    to node features (x) only, not graph structure. See explain_attention.py
    for edge/attention-based structural importance instead.
  - mask (missingness) is held fixed at each explained window's own mask for
    every interpolation/background step (it broadcasts against any batch size
    because it keeps a leading batch dim of 1 — see _FixedContextModel). So
    this answers "how does this specific prediction depend on x, given its
    own missingness context" rather than jointly attributing to the mask.
  - The model sums predictions across all nodes into a single scalar target
    before attribution, so each explained window yields one (T, N, F)
    attribution tensor aligned with the input — i.e. per-node, per-timestep,
    per-feature importance for "total predicted burden across the network".
  - Expected Gradients needs (background x interpolation steps) forward+
    backward passes per explained window, and STGATGRU's B x T Python loop
    over GAT layers is not vectorised, so this is slow. n_background and
    n_explain_windows are deliberately small by default (see cfg["explain"]).

Inputs:
    config (YAML)     — defines name, device, data paths, features, ...
    model checkpoint  — the trained .pt to explain
    best_params.json  — hyperparameters used to build the model

Outputs (results/STGNN/<name>/):
    shap_attributions.npz       — raw (n_explain, T, N, F) attributions + metadata
    shap_feature_importance.png — mean |attribution| ranked by feature
    shap_node_time_heatmap.png  — mean |attribution| summed over features, per node x lag

Run via Snakemake (script mode) or directly:
    python explain_shap.py --config config.yaml \
        [--model best_model.pt] [--params best_params.json] [--split test] \
        [--n-background 20] [--n-explain 5]
"""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from model import STGATGRU
from tune import STGNNDataset, load_tensors, load_edge_index
from choropleth import _node_names


# ── Feature naming ─────────────────────────────────────────────────────────────

def _feature_names(cfg: dict, in_channels: int) -> list[str]:
    """Best-effort feature labels for plots. quality_dummy_vars are computed
    at preprocessing time from raw categorical values and aren't recoverable
    from the config alone, so any remaining channels are labelled generically."""
    env_vars  = cfg.get("features", {}).get("env_vars", [])
    lulc_vars = cfg.get("features", {}).get("land_use_vars", [])
    names = ["inc"] + list(env_vars) + list(lulc_vars)
    n_extra = in_channels - len(names)
    if n_extra > 0:
        names += [f"quality_{i}" for i in range(n_extra)]
    return names[:in_channels]


# ── Model wrapper ──────────────────────────────────────────────────────────────

class _FixedContextModel(nn.Module):
    """Exposes only node features `x` as the SHAP-explained input; edge_index
    and mask are held fixed. mask keeps a leading batch dim of 1 so it
    broadcasts against whatever batch size GradientExplainer feeds forward()
    (background samples, interpolation steps, ...). Outputs are summed across
    nodes into a scalar so attributions come back with the same (T, N, F)
    shape as the input."""

    def __init__(self, model: STGATGRU, edge_index: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.model      = model
        self.edge_index = edge_index
        self.mask       = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, self.edge_index, mask=self.mask)
        if out.dim() == 3:
            out = out.squeeze(-1)
        return out.sum(dim=1, keepdim=True)  # (B, 1)


# ── Plotting ────────────────────────────────────────────────────────────────────

def plot_feature_importance(attrs: np.ndarray, feature_names: list, out_path: Path) -> None:
    importance = np.abs(attrs).mean(axis=(0, 1, 2))  # (F,)
    order = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.35 * len(feature_names) + 1.5))
    ax.barh(range(len(order)), importance[order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Expected Gradients attribution|")
    ax.set_title("Feature importance (summed over nodes, averaged over windows)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_node_time_heatmap(attrs: np.ndarray, node_names: list, out_path: Path,
                            top_k: int = 20) -> None:
    imp = np.abs(attrs).mean(axis=(0, 3))  # (T, N)
    T, N = imp.shape
    order = np.argsort(imp.sum(axis=0))[::-1][:min(top_k, N)]
    data  = imp[:, order].T  # (top_k, T)

    fig, ax = plt.subplots(figsize=(1 + 0.5 * T, 0.35 * len(order) + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([node_names[i] for i in order], fontsize=7)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"t-{T - 1 - i}" for i in range(T)], fontsize=8)
    ax.set_xlabel("Lag (months before prediction)")
    ax.set_title(f"Top {len(order)} nodes — mean |attribution| (summed over features)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean |attribution|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────

def explain_shap(cfg: dict, params: dict, model_path, out_dir: Path,
                  csv_path: str, split: str = "test", n_background: int = 20,
                  n_explain: int = 5, seed: int = 0,
                  tensors_path: Path | None = None,
                  edge_index_path: Path | None = None) -> None:
    model_path = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    device      = torch.device(cfg.get("device", "cpu"))
    window_size = params["window_size"]

    tensors = load_tensors(cfg, window_size, tensors_path)
    if split == "test" and "inference_x" in tensors:
        split = "inference"
        print("INFO: using inference split — explained windows cover all test months.")

    explain_dataset = STGNNDataset(tensors, split)
    train_dataset   = STGNNDataset(tensors, "train")
    edge_index      = load_edge_index(cfg, device, edge_index_path)
    in_channels     = explain_dataset.x.shape[-1]

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
    model.eval()

    rng = np.random.default_rng(seed)
    n_background = min(n_background, len(train_dataset))
    bg_idx = rng.choice(len(train_dataset), size=n_background, replace=False)
    background = torch.stack([train_dataset.x[i] for i in bg_idx]).to(device)

    n_explain   = min(n_explain, len(explain_dataset))
    explain_idx = np.linspace(0, len(explain_dataset) - 1, n_explain, dtype=int)

    node_names    = _node_names(csv_path)
    feature_names = _feature_names(cfg, in_channels)

    months_t = tensors.get(f"{split}_months")
    all_attrs, all_months = [], []

    for i in explain_idx:
        x_i, _, mask_i = explain_dataset[i]
        x_i    = x_i.unsqueeze(0).to(device)
        mask_i = mask_i.unsqueeze(0).to(device)

        wrapped   = _FixedContextModel(model, edge_index, mask_i)
        explainer = shap.GradientExplainer(wrapped, background)

        sv = explainer.shap_values(x_i)
        if isinstance(sv, list):
            sv = sv[0]
        attr = np.asarray(sv)[0]  # (T, N, F) or (T, N, F, 1) — model output is scalar,
        if attr.ndim == 4:        # but GradientExplainer still appends an output axis.
            attr = attr.squeeze(-1)
        all_attrs.append(attr)

        if months_t is not None:
            m = months_t[i]
            all_months.append(int(m[-1]) if m.dim() > 0 else int(m))

        print(f"  explained window {i} ({len(all_attrs)}/{n_explain})")

    attrs = np.stack(all_attrs)  # (n_explain, T, N, F)

    npz_path = out_dir / "shap_attributions.npz"
    np.savez(
        npz_path,
        attributions  = attrs,
        node_names    = np.array(node_names),
        feature_names = np.array(feature_names),
        explain_idx   = explain_idx,
        months        = np.array(all_months) if all_months else np.array([]),
    )
    print(f"SHAP attributions saved to {npz_path}")

    feat_path = out_dir / "shap_feature_importance.png"
    plot_feature_importance(attrs, feature_names, feat_path)
    print(f"  -> {feat_path}")

    heat_path = out_dir / "shap_node_time_heatmap.png"
    plot_node_time_heatmap(attrs, node_names, heat_path)
    print(f"  -> {heat_path}")

    print(f"\nAll outputs saved to {out_dir}")


# ── Entry point (Snakemake script mode or CLI) ────────────────────────────────

def _main_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Explain STGATGRU predictions via SHAP Expected Gradients.")
    ap.add_argument("--config", required=True, help="Path to config YAML.")
    ap.add_argument("--model", default=None,
                    help="Checkpoint .pt to explain (default: results dir best_model.pt).")
    ap.add_argument("--params", default=None,
                    help="best_params.json used to build the model (default: results dir).")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which split to draw explained windows from (default: test).")
    ap.add_argument("--n-background", type=int, default=20,
                    help="Number of background (reference) windows sampled from train.")
    ap.add_argument("--n-explain", type=int, default=5,
                    help="Number of windows to explain, spread evenly across the split.")
    ap.add_argument("--csv-path", default=None,
                    help="Path to merged dengue-env CSV (default: computed from project root).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir     = Path("results/STGNN") / cfg["name"]
    params_path = Path(args.params) if args.params else out_dir / "best_params.json"
    with open(params_path) as f:
        params = json.load(f)
    model_path = Path(args.model) if args.model else out_dir / "best_model.pt"
    csv_path = _resolve_csv_path(args.csv_path)
    explain_shap(cfg, params, model_path, out_dir, csv_path, split=args.split,
                 n_background=args.n_background, n_explain=args.n_explain)


def _resolve_csv_path(csv_path: str | None) -> str:
    if csv_path:
        return csv_path
    _root = Path(__file__).resolve().parent
    for _ in range(6):
        if (_root / ".git").exists():
            break
        _root = _root.parent
    return str(_root / "data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv")


if __name__ == "__main__":
    if "snakemake" in globals():
        with open(snakemake.params.cfg) as f:            # noqa: F821
            cfg = yaml.safe_load(f)
        out_dir    = Path(snakemake.params.results_dir)    # noqa: F821
        with open(snakemake.params.best_params) as f:     # noqa: F821
            params = json.load(f)
        model_path = getattr(snakemake.input, "model", None)  # noqa: F821
        model_path = Path(model_path) if model_path else out_dir / "best_model.pt"
        explain_cfg = cfg.get("explain", {})
        explain_shap(
            cfg, params, model_path, out_dir,
            csv_path=snakemake.params.csv_path,  # noqa: F821
            split="test",
            n_background = explain_cfg.get("n_background", 20),
            n_explain    = explain_cfg.get("n_explain_windows", 5),
            tensors_path    = Path(snakemake.input.tensors),     # noqa: F821
            edge_index_path = Path(snakemake.input.edge_index),  # noqa: F821
        )
    else:
        _main_cli()
