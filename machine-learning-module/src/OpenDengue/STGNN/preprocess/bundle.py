"""
Inference bundle — everything site/scripts/run_inference.py needs to rebuild
the model's input tensor and invert its output, without the training CSV.

Written by pipeline.py next to preprocessing_params.json. Snakemake's
export_inference_bundle rule then copies it, best_model.pt and best_params.json
into results/STGNN/<name>/inference_bundle/, which is what the monthly
GitHub Actions run downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE_FORMAT_VERSION = 1


def _month_of(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.month


def node_table(df: pd.DataFrame, node_index: dict, cfg: dict) -> list[dict]:
    """Nodes in model order, each with its country so inference can match on (country, name)."""
    node_col    = cfg["data"]["unit_column"]
    region_col  = cfg["data"].get("region_column", "adm_0_name")
    countries   = df.groupby(node_col)[region_col].first().to_dict() if region_col in df else {}
    node_order  = sorted(node_index, key=node_index.get)
    return [{"name": name, "country": countries.get(name, "")} for name in node_order]


def target_baselines(history_df: pd.DataFrame, node_index: dict, cfg: dict) -> dict:
    """
    Per-node and SEA-wide calendar-month means of the (log-transformed, not
    deseasonalised) target over the training period. The risk index compares a
    prediction against these, so ">1" means "above what this province usually
    sees in this calendar month during 2011–2018".
    """
    node_col = cfg["data"]["unit_column"]
    time_col = cfg["data"]["time_column"]
    target   = cfg["target_column"]

    df = history_df[[node_col, time_col, target]].copy()
    df["month"] = _month_of(df[time_col])

    by_month = df.groupby("month")[target].mean()
    by_node_month = df.groupby([node_col, "month"])[target].mean()

    def _clean(v):
        return None if pd.isna(v) else float(v)

    return {
        "by_month": {str(m): _clean(v) for m, v in by_month.items()},
        "by_node_month": {
            name: {str(m): _clean(by_node_month.get((name, m), np.nan)) for m in range(1, 13)}
            for name in sorted(node_index, key=node_index.get)
        },
    }


def env_climatology(history_df: pd.DataFrame, env_vars: list[str], cfg: dict) -> dict:
    """SEA-wide calendar-month means of each raw env feature (keys "01".."12")."""
    time_col = cfg["data"]["time_column"]
    months = _month_of(history_df[time_col]).map(lambda m: f"{m:02d}")
    out = {}
    for feat in env_vars:
        if feat in history_df:
            out[feat] = {m: float(v) for m, v in history_df.groupby(months)[feat].mean().items()}
    return out


def missing_quality_pattern(train_df: pd.DataFrame, quality_dummies: list[str], cfg: dict) -> list[float]:
    """
    The quality-dummy vector most often seen on rows whose target is missing.
    Inference gives this to nodes with no reported incidence, so they look like
    the missing rows the model saw in training. Rows with reported incidence get
    all zeros (OBSERVED is the dropped reference category).
    """
    if not quality_dummies:
        return []
    target  = cfg["target_column"]
    missing = train_df[train_df[target].isna()]
    if missing.empty:
        return [0.0] * len(quality_dummies)
    pattern = missing[quality_dummies].astype(float).value_counts().idxmax()
    return [float(v) for v in pattern]


def save_inference_bundle(
    path: Path,
    cfg: dict,
    nodes: list[dict],
    scalers: dict,
    seasonal_means: dict,
    baselines: dict,
    climatology: dict,
    missing_quality: list[float],
    train_period: tuple[str, str],
) -> None:
    prep         = cfg.get("preprocessing", {})
    feats        = cfg.get("features", {})
    env_vars     = feats.get("env_vars", [])
    lulc_vars    = feats.get("land_use_vars", [])
    quality_vars = feats.get("quality_dummy_vars", [])
    incidence    = prep.get("incidence_input", True)

    scaler_inc = scalers.get("inc")
    scaler_env = scalers.get("env")
    inc_scaler = (
        {"mean": float(scaler_inc.mean_[0]), "std": float(scaler_inc.scale_[0])}
        if scaler_inc is not None else None
    )
    # fill_missing_inc writes 0 into the (log1p, deseasonalised) target before
    # scaling, so missing incidence reaches the model as this scaled value.
    inc_fill_value = (0.0 - inc_scaler["mean"]) / inc_scaler["std"] if inc_scaler else 0.0

    bundle = {
        "format_version":  BUNDLE_FORMAT_VERSION,
        "name":            cfg["name"],
        "mode":            "incidence" if incidence else "climate",
        "incidence_input": incidence,
        "target_column":   cfg["target_column"],
        "train_period":    list(train_period),
        "nodes":           nodes,
        "features": {
            "env_vars":           env_vars,
            "lulc_vars":          lulc_vars,
            "quality_dummy_vars": quality_vars,
            "order":              (["inc"] if incidence else []) + env_vars + lulc_vars + quality_vars,
        },
        "env_scaler": (
            {"mean": scaler_env.mean_.tolist(), "scale": scaler_env.scale_.tolist()}
            if scaler_env is not None else None
        ),
        "inc_scaler":       inc_scaler,
        "inc_fill_value":   inc_fill_value,
        "log_transform":    bool(prep.get("log_transform", False)),
        "deseasonalise":    bool(prep.get("deseasonalise", False)),
        "seasonal_means":   {str(k): float(v) for k, v in seasonal_means.items()},
        "missing_quality_dummies": missing_quality,
        "target_baseline":  baselines,
        "env_climatology":  climatology,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Inference bundle saved → {path}")
