from __future__ import annotations

from pathlib import Path

import yaml
import torch
import json


def _find_git_root(start: Path) -> Path:
    for p in (start,) + tuple(start.parents):
        if (p / ".git").exists():
            return p
    return start


def _output_dir(cfg: dict) -> Path:
    root = _find_git_root(Path(__file__).resolve())
    return root / "data" / "processed" / "machine-learning" / "STGNN" / cfg["name"]


def save_preprocessing_params(scaler_inc, seasonal_means: dict, cfg: dict) -> None:
    """Save incidence scaler params and seasonal means to a single JSON file."""
    path = _output_dir(cfg) / "preprocessing_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scaler_inc": {
            "mean": float(scaler_inc.mean_[0]),
            "std":  float(scaler_inc.scale_[0]),
        } if scaler_inc is not None else None,
        "seasonal_means": {str(k): float(v) for k, v in seasonal_means.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def inference_bundle_path(cfg: dict) -> Path:
    return _output_dir(cfg) / "inference_bundle.json"


def load_preprocessing_params(cfg: dict) -> dict:
    """Load incidence scaler params and seasonal means saved by preprocessing."""
    path = _output_dir(cfg) / "preprocessing_params.json"
    with open(path) as f:
        return json.load(f)


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_tensors(snapshots: dict, cfg: dict, window_size: int,
                 split_months: dict | None = None) -> None:
    out_dir = _output_dir(cfg) / f"window_{window_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split, windows in snapshots.items():
        splits[f"{split}_x"]    = torch.stack([x    for x, _y, _m  in windows])
        splits[f"{split}_y"]    = torch.stack([y    for _x, y, _m  in windows])
        splits[f"{split}_mask"] = torch.stack([mask for _x, _y, mask in windows])

        if split_months and split in split_months:
            all_months = split_months[split]
            # window i targets timestep window_size + i
            splits[f"{split}_months"] = torch.tensor(
                all_months[window_size:], dtype=torch.int32
            )

    torch.save(splits, out_dir / "tensors.pt")

def save_edge_index(edge_index: torch.Tensor, cfg: dict) -> None:
    """Persist edge_index to disk."""
    out_dir = _output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(edge_index, out_dir / "edge_index.pt")


def get_window_sizes(cfg: dict) -> list[int]:
    """Extract window size choices from tuning search space config."""
    return (
        cfg.get("tune", {})
           .get("search_space", {})
           .get("stgnn", {})
           .get("window_size", {})
           .get("choices", [0])
    )