import json
import os
import yaml
import torch
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Allows the CUDA allocator to use non-contiguous memory segments,
# which prevents OOM from fragmentation on low-VRAM GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from model import STGATGRU
from loss import masked_huber_loss


# ── Dataset Definition ────────────────────────────────────────────────────────

class STGNNDataset(Dataset):
    """
    Wraps windowed spatio-temporal data into a standard PyTorch Dataset.
    Keeps entire matrix configurations intact as continuous tensors.
    """
    def __init__(self, tensors: dict, split: str):
        self.x = tensors[f"{split}_x"]
        self.y = tensors[f"{split}_y"]
        self.mask = tensors[f"{split}_mask"]

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.mask[idx]


# ── Helpers ───────────────────────────────────────────────────────────────────

from preprocess.utils import _find_git_root


def _processed_dir(cfg: dict) -> Path:
    root = _find_git_root(Path(__file__).resolve())
    return root / "data" / "processed" / "machine-learning" / "STGNN" / cfg["name"]


def _sweep_id_path(cfg: dict) -> Path:
    return Path("results/STGNN") / cfg["name"] / "sweep_id.txt"


def load_tensors(cfg: dict, window_size: int, path: Path | None = None) -> dict:
    # Pass `path` (e.g. snakemake.input.tensors) to load the file Snakemake staged;
    # it differs from the git-root path when inputs come from remote storage.
    path = path or _processed_dir(cfg) / f"window_{window_size}" / "tensors.pt"
    return torch.load(path, weights_only=True)


def load_edge_index(cfg: dict, device: torch.device, path: Path | None = None) -> torch.Tensor:
    path = path or _processed_dir(cfg) / "edge_index.pt"
    return torch.load(path, weights_only=True).to(device)


def check_isolated_nodes(edge_index: torch.Tensor, n_nodes: int, device: torch.device) -> bool:
    """
    Warn if any nodes have no edges. Isolated nodes cause NaN in GAT
    (softmax over empty neighbourhood is undefined).
    """
    nodes_with_edges = edge_index.unique()
    all_nodes        = torch.arange(n_nodes, device=device)
    isolated         = all_nodes[~torch.isin(all_nodes, nodes_with_edges)]

    if isolated.numel() > 0:
        print(f"WARNING: {isolated.numel()} isolated nodes at indices: "
              f"{isolated.tolist()} — these will produce NaN in GAT layers.")
        return False

    print(f"Graph OK: all {n_nodes} nodes have edges.")
    return True


# ── Pre-sweep data sanity check ───────────────────────────────────────────────

def _as_NF(x: torch.Tensor):
    """x (num_windows, T, N, F) or (T, N, F) -> (rows, N, F), N, F."""
    if x.dim() == 4:
        Wc, T, N, F = x.shape
        return x.reshape(Wc * T, N, F), N, F
    if x.dim() == 3:
        T, N, F = x.shape
        return x.reshape(T, N, F), N, F
    raise ValueError(f"unexpected x shape {tuple(x.shape)}")


def inspect_split(tensors: dict, split: str, delta: float = 1.0,
                  const_eps: float = 1e-6) -> dict:
    """Pure tensor diagnostics for one split. No model / GPU required."""
    x    = tensors[f"{split}_x"]
    y    = tensors[f"{split}_y"]
    mask = tensors[f"{split}_mask"]

    if y.dim() == x.dim() - 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    n_windows = x.shape[0]

    xr, N, F = _as_NF(x)

    nan_x = int(torch.isnan(xr).sum())
    inf_x = int(torch.isinf(xr).sum())
    nan_y = int(torch.isnan(y).sum())

    valid_per_window = mask.reshape(n_windows, -1).sum(dim=1)
    n_empty = int((valid_per_window == 0).sum())

    cross_node_std     = xr.std(dim=1).mean(dim=0)
    all_zero_feats     = (xr.reshape(-1, F) == 0).all(dim=0)
    const_across_nodes = cross_node_std < const_eps
    feat_absmax        = xr.reshape(-1, F).abs().max(dim=0).values

    rep       = xr[-1]
    uniq_rows = torch.unique(rep.round(decimals=5), dim=0).shape[0]

    node_target_stds, base_mean, base_median, n_used = [], 0.0, 0.0, 0
    yw = y.reshape(n_windows, -1)
    mw = mask.reshape(n_windows, -1).bool()
    for i in range(n_windows):
        m = mw[i]
        if m.sum() == 0:
            continue
        yv = yw[i][m]
        node_target_stds.append(yv.std().item())
        base_mean   += masked_huber_loss(yv.new_full((yv.numel(),), yv.mean()),   yv, m[m], delta).item()
        base_median += masked_huber_loss(yv.new_full((yv.numel(),), yv.median()), yv, m[m], delta).item()
        n_used += 1
    base_mean   /= max(n_used, 1)
    base_median /= max(n_used, 1)
    target_node_std = float(torch.tensor(node_target_stds).mean()) if node_target_stds else 0.0

    n_const = int(const_across_nodes.sum())
    hard, soft = [], []
    if nan_x or nan_y:            hard.append(f"NaN in inputs (x:{nan_x}, y:{nan_y})")
    if inf_x:                     hard.append(f"Inf in x ({inf_x})")
    if n_empty:                   hard.append(f"{n_empty} window(s) have ZERO valid nodes (no gradient)")
    if n_const == F:              hard.append("ALL features are constant across nodes — nodes are indistinguishable")
    if uniq_rows == 1:            hard.append("all nodes share one identical feature vector (rep. slice)")

    if 0 < n_const < F:           soft.append(f"{n_const}/{F} features constant across nodes")
    if target_node_std < 1e-3:    soft.append(f"target barely varies across nodes (std {target_node_std:.2e}) — graph may be pointless")
    if uniq_rows < N:             soft.append(f"only {uniq_rows}/{N} distinct node feature vectors (rep. slice)")
    if int(all_zero_feats.sum()): soft.append(f"{int(all_zero_feats.sum())} all-zero feature column(s)")
    if feat_absmax.max() > 1e3:   soft.append(f"large feature magnitude (max |x| = {feat_absmax.max():.1f}) — check scaling")

    return {
        "split": split, "n_windows": n_windows, "N": N, "F": F,
        "valid_min": int(valid_per_window.min()), "valid_mean": float(valid_per_window.float().mean()),
        "n_const_across_nodes": n_const, "uniq_node_rows": uniq_rows,
        "target_node_std": target_node_std,
        "const_baseline_mean": base_mean, "const_baseline_median": base_median,
        "hard": hard, "soft": soft,
    }


def _print_report(rep: dict):
    floor = min(rep["const_baseline_mean"], rep["const_baseline_median"])
    print(f"\n── data check [{rep['split']}] ──")
    print(f"  windows={rep['n_windows']}  N={rep['N']}  F={rep['F']}  "
          f"valid nodes/window: min {rep['valid_min']}, mean {rep['valid_mean']:.1f}")
    print(f"  features constant across nodes : {rep['n_const_across_nodes']}/{rep['F']}")
    print(f"  distinct node feature vectors  : {rep['uniq_node_rows']}/{rep['N']} (rep. slice)")
    print(f"  target std across valid nodes  : {rep['target_node_std']:.4f}")
    print(f"  CONSTANT-PREDICTOR floor       : mean {rep['const_baseline_mean']:.4f} | "
          f"median {rep['const_baseline_median']:.4f}")
    print(f"    -> any useful model MUST beat ~{floor:.4f} on this split")
    for h in rep["hard"]:
        print(f"  [HARD]  {h}")
    for s in rep["soft"]:
        print(f"  [warn]  {s}")


def _window_sizes_from_cfg(cfg: dict) -> list[int]:
    spec = cfg["tune"]["search_space"]["stgnn"].get("window_size", {})
    if spec.get("type") == "categorical":
        return list(spec["choices"])
    if "choices" in spec:
        return list(spec["choices"])
    if "value" in spec:
        return [spec["value"]]
    raise ValueError("Could not determine window_size(s) from cfg search space.")


def sanity_check_before_sweep(cfg: dict, strict: bool = True) -> None:
    print("\n" + "=" * 64)
    print("PRE-SWEEP DATA CHECK")
    print("=" * 64)

    all_hard = []
    for ws in _window_sizes_from_cfg(cfg):
        try:
            tensors = load_tensors(cfg, ws)
        except FileNotFoundError:
            print(f"\n[window_{ws}] tensors.pt not found — skipping")
            continue
        print(f"\n### window_size = {ws} ###")
        for split in ("train", "val"):
            if f"{split}_x" not in tensors:
                continue
            rep = inspect_split(tensors, split)
            _print_report(rep)
            all_hard += [f"window_{ws}/{split}: {h}" for h in rep["hard"]]

    print("\n" + "=" * 64)
    if all_hard:
        print(f"DATA CHECK FAILED — {len(all_hard)} hard issue(s):")
        for h in all_hard:
            print(f"  - {h}")
        print("=" * 64)
        if strict:
            raise RuntimeError(
                "Pre-sweep data check failed. Fix the inputs above, or set "
                "cfg['tune']['strict_data_check']=false to override."
            )
    else:
        print("DATA CHECK PASSED — no hard issues. (Still beat the floor above.)")
        print("=" * 64)


# ── Execution Loop (Parallelised Sequence Streams) ───────────────────────────

def run_epoch(
    model:       STGATGRU,
    dataloader:  DataLoader,
    edge_index:  torch.Tensor,
    device:      torch.device,
    optimizer:   torch.optim.Optimizer = None,
    scaler:      GradScaler = None,
) -> float:
    """
    Run one full epoch over all tensor mini-batches natively.
    Uses AMP when a GradScaler is provided (CUDA training only).
    """
    training    = optimizer is not None
    use_amp_fwd = device.type == "cuda"          # fp16 forward pass on CUDA always
    use_amp     = scaler is not None and use_amp_fwd  # scaler only needed for backward
    model.train() if training else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for x, y, mask in dataloader:
            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            assert not torch.isnan(x).any(), "NaN in feature map configurations"
            assert not torch.isnan(y).any(), "NaN in ground-truth labels"

            with autocast("cuda", enabled=use_amp_fwd):
                pred = model(x, edge_index, mask=mask)
                if pred.dim() == 3 and y.dim() == 2:
                    pred = pred.squeeze(-1)

                if mask.dim() == 3:
                    target_mask = mask[:, -1, :]
                else:
                    target_mask = mask

                loss = masked_huber_loss(pred, y, target_mask)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    if torch.isnan(total_norm):
                        print("NaN gradient detected — aborting trial.")
                        return float("nan")
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    if torch.isnan(total_norm):
                        print("NaN gradient detected — aborting trial.")
                        return float("nan")
                    optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def build_sweep_config(cfg: dict) -> dict:
    ss = cfg["tune"]["search_space"]["stgnn"]

    def param(spec):
        if spec["type"] == "log_uniform":
            return {"distribution": "log_uniform_values",
                    "min": spec["low"], "max": spec["high"]}
        if spec["type"] == "categorical":
            return {"values": spec["choices"]}
        raise ValueError(f"Unknown search space type: {spec['type']}")

    return {
        "method": "bayes",
        "metric": {"name": cfg["tune"]["metric"], "goal": "minimize"},
        "parameters": {k: param(v) for k, v in ss.items()},
    }


def train_sweep(cfg: dict):
    """Single W&B sweep agent run — called once per trial."""
    run = wandb.init(
        group=cfg["name"],                         
        tags=[cfg["name"]],                        
        config={"experiment_name": cfg["name"]},   
    )
    wc = run.config

    torch.cuda.empty_cache()
    device      = torch.device(cfg["tune"].get("device", "cpu"))
    window_size = wc.window_size
    batch_size  = wc.get("batch_size", 32)  # Extract batch size dynamically from search space

    # ── Data Pipeline ────────────────────────────────────────────────────────
    tensors       = load_tensors(cfg, window_size)
    train_dataset = STGNNDataset(tensors, "train")
    val_dataset   = STGNNDataset(tensors, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2, persistent_workers=True
    )

    edge_index  = load_edge_index(cfg, device)
    in_channels = train_dataset.x.shape[-1]   
    n_nodes     = train_dataset.x.shape[2]     

    # ── Graph sanity check ────────────────────────────────────────────────────
    graph_ok = check_isolated_nodes(edge_index, n_nodes, device)
    if not graph_ok:
        wandb.log({"masked_huber_loss": float("nan")})
        run.finish()
        torch.cuda.empty_cache()
        return

    # ── Model Construction ───────────────────────────────────────────────────
    model = STGATGRU(
        in_channels  = in_channels,
        gat1_hidden  = wc.gat1_hidden,
        gat1_heads   = wc.gat1_heads,
        mlp_hidden   = wc.get("mlp_hidden", wc.gat1_hidden * wc.gat1_heads), # Decoupled fallback
        mlp_layers   = wc.mlp_layers,
        gat2_hidden  = wc.gat2_hidden,
        gat2_heads   = wc.gat2_heads,
        gru_hidden   = wc.get("gru_hidden", wc.gat2_hidden),
        pred_horizon = 1,
        dropout      = wc.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=wc.learning_rate)
    scaler    = GradScaler("cuda") if device.type == "cuda" else None

    max_epochs = cfg["tune"].get("num_epochs", 100)
    patience = cfg["tune"].get("patience", 10)
    patience_counter = 0
    best_val_loss = float("inf")

    def _cleanup():
        nonlocal model, optimizer, scaler
        model = optimizer = scaler = None
        torch.cuda.empty_cache()

    # ── Training Loop ────────────────────────────────────────────────────────
    try:
        for epoch in range(max_epochs):
            train_loss = run_epoch(model, train_loader, edge_index, device, optimizer, scaler)
            val_loss   = run_epoch(model, val_loader,   edge_index, device)

            if train_loss != train_loss or val_loss != val_loss:
                wandb.log({"masked_huber_loss": float("nan")})
                run.finish()
                _cleanup()
                return

            wandb.log({
                "epoch":             epoch + 1,
                "train/huber_loss":  train_loss,
                "val/huber_loss":    val_loss,
                "masked_huber_loss": val_loss,
            })

            # ── Early Stopping Check ──────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.run.summary["best_val_loss"] = best_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best Validation Loss: {best_val_loss:.4f}")
                break

    except torch.OutOfMemoryError:
        print(f"OOM for config: {dict(wc)} — skipping trial.")
        wandb.log({"masked_huber_loss": float("nan")})

    run.finish()
    _cleanup()


def _get_or_create_sweep(cfg: dict) -> str:
    id_path = _sweep_id_path(cfg)
    project = cfg.get("wandb_project")
    entity  = cfg.get("wandb_entity")

    if id_path.exists():
        sweep_id = id_path.read_text().strip()
        print(f"Reattaching to existing sweep: {sweep_id}")
    else:
        sweep_config = build_sweep_config(cfg)
        sweep_config["name"] = cfg["name"]          
        raw_id = wandb.sweep(sweep_config, project=project, entity=entity)
        if entity is None:
            entity = wandb.Api().default_entity
        sweep_id = f"{entity}/{project}/{raw_id}"
        id_path.parent.mkdir(parents=True, exist_ok=True)
        id_path.write_text(sweep_id)
        print(f"Created new sweep: {sweep_id}")

    return sweep_id


def run_sweep(cfg: dict, out_path: str | Path = None):
    """Initialize (or resume) a W&B sweep, then save best params to disk."""
    strict = cfg["tune"].get("strict_data_check", True)
    sanity_check_before_sweep(cfg, strict=strict)

    sweep_id = _get_or_create_sweep(cfg)

    project = cfg.get("wandb_project")
    entity  = cfg.get("wandb_entity")

    try:
        wandb.agent(
            sweep_id,
            function = lambda: train_sweep(cfg),
            count    = cfg["tune"]["n_trials"],
            entity   = entity,
            project  = project,
        )

        metric   = cfg["tune"]["metric"]
        api      = wandb.Api()

        sweep    = api.sweep(sweep_id)
        best_run = sweep.best_run()   # uses sweep config's metric/goal (minimize)
        if best_run is None:
            raise RuntimeError(
                f"Sweep {sweep_id} has no completed runs with a finite metric. "
                "Check W&B for OOM or NaN failures across all trials."
            )
        print(f"Best run ID: {best_run.id}")
        print(f"Best run config: {best_run.config}")
        print(f"Best run summary: {dict(best_run.summary)}")

        best_params = dict(best_run.config)
        best_params["best_val_loss"] = best_run.summary.get("best_val_loss")

        if out_path is None:
            out_path = Path("results/STGNN") / cfg["name"] / "best_params.json"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(best_params, f, indent=2)

        _sweep_id_path(cfg).unlink(missing_ok=True)

    except Exception:
        # Delete sweep_id.txt so the next Snakemake run starts a fresh sweep
        # rather than reattaching to a partially-run one.
        _sweep_id_path(cfg).unlink(missing_ok=True)
        raise


# ── Snakemake entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(snakemake.params.cfg) as f: # noqa: F821
        cfg = yaml.safe_load(f)
    run_sweep(cfg, out_path=snakemake.output.best_params) # noqa: F821