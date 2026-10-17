from __future__ import annotations

import pandas as pd


def load_data(cfg: dict) -> pd.DataFrame:
    """Read CSV, filter to relevant columns, and sort by node and time."""
    data_path   = cfg.get("data_path") or cfg.get("data", {}).get("path")
    target_col  = cfg.get("target_column") or cfg.get("target", {}).get("column")
    features    = cfg.get("features", {})
    quality_vars = cfg.get("quality_vars", []) or features.get("quality_vars", [])
    env_vars    = cfg.get("env_vars", []) or features.get("env_vars", [])
    land_use    = cfg.get("lulc_vars", []) or features.get("land_use_vars", [])

    keep_cols = ["adm_0_name", "adm_1_name", "year_month", "dengue_total", "population_sum", "data_quality", target_col] + env_vars + land_use

    df = pd.read_csv(str(data_path))
    df = df.filter(items=keep_cols)
    # Optional first month, e.g. to train without the earliest year.
    start = cfg.get("data", {}).get("start")
    if start:
        df = df[pd.to_datetime(df["year_month"]) >= pd.Timestamp(start)]
    df = df.sort_values(["adm_1_name", "year_month"]).reset_index(drop=True)
    return df


def build_node_index(df: pd.DataFrame, cfg: dict) -> dict:
    node_names = df[cfg["data"]["unit_column"]].unique().tolist()
    return {name: idx for idx, name in enumerate(node_names)}