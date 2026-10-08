import argparse
from pathlib import Path
import pandas as pd
import torch
from utils import set_output_dir, load_config, save_tensors, save_preprocessing_params, save_edge_index, get_window_sizes
from utils import inference_bundle_path
from bundle import node_table, target_baselines, env_climatology, missing_quality_pattern, save_inference_bundle
from dataset import load_data, build_node_index
from features import log_transform, separate_sources, build_masks, fill_missing_inc, reshape_all
from features import add_cyclical_month_features
from features import fit_seasonal_means, apply_seasonal_means
from features import fill_node_from_donors, impute_env_naive, assert_no_nans, encode_quality_flags
from features import diagnose_feature_composition, scale_sources
from graph import build_edge_index
from temporal import create_windows, create_inference_windows, temporal_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess CSV → Pytorch tensor files for Graph Learning"
    )
    p.add_argument("--config", required=True, help="Path to config.yaml")
    return p.parse_args()


def main(config_path: str | None = None, data_path: str | None = None, cfg: dict | None = None):
    cfg  = cfg if cfg is not None else load_config(config_path)
    if data_path is not None:
        cfg["data"]["path"] = data_path
    prep = cfg.get("preprocessing", {})

    df = load_data(cfg)
    print("Data loaded.")
    print(df["data_quality"].value_counts(dropna=False))

    # ── Sanity checks ────────────────────────────────────────────────────────
    node_col = cfg["data"]["unit_column"]
    time_col = cfg["data"]["time_column"]
    target   = cfg["target_column"]

    dupes = df.groupby([time_col, node_col]).size()
    dupes = dupes[dupes > 1]
    if not dupes.empty:
        print(f"WARNING: {len(dupes)} duplicate (time, node) pairs found:")
        print(dupes.to_string())
    else:
        print("No duplicates found.")

    # ── Diagnose raw NaNs (excluding known Labuan issue) ─────────────────────
    env_cols  = [c for c in df.columns if c not in [
        node_col, time_col, target,
        "dengue_total", "population_sum", "IR", "IR_quality",
        "data_quality", "adm_0_name",
    ]]
    labuan_df = df[df[node_col] != "W.P. Labuan"]
    for col in env_cols:
        n_nan = labuan_df[col].isna().sum()
        if n_nan > 0:
            bad = labuan_df[labuan_df[col].isna()][[node_col, time_col, col]]
            print(f"\n{col}: {n_nan} NaNs")
            print(f"  Nodes affected: {bad[node_col].unique().tolist()}")
            print(f"  Time range: {bad[time_col].min()} → {bad[time_col].max()}")
            print(f"  Consecutive? {bad[time_col].nunique() == n_nan // bad[node_col].nunique()}")

    # ── Standard preprocessing ───────────────────────────────────────────────
    # log_transform runs on the full df before the split: it is deterministic
    # (no fitted parameters) so it introduces no data leakage, and applying it
    # before the split ensures NaNs are still present when deseasonalise later
    # computes monthly means (NaN rows are correctly excluded from the mean).
    df = log_transform(df, cfg) if prep.get("log_transform") else df
    print("Log transform applied.")

    # ── Cyclical calendar-month encoding ─────────────────────────────────────
    # Deterministic function of calendar date — no fitted params, so applying
    # it before the split introduces no leakage (same reasoning as log_transform).
    df = add_cyclical_month_features(df, cfg) if prep.get("cyclical_month_features") else df

    # ── Encode categorical quality flags ─────────────────────────────────────
    quality_vars = cfg.get("features", {}).get("quality_vars", [])
    if quality_vars:
        df, quality_dummies = encode_quality_flags(df, quality_vars)
        cfg["features"]["quality_dummy_vars"] = quality_dummies
        print(f"Quality flags encoded: {quality_dummies}")

    node_index = build_node_index(df, cfg)
    print("Node index built.")
    edge_index = build_edge_index(node_index)
    print("Edge index built.")
    save_edge_index(edge_index, cfg)

    train_df, val_df, test_df = temporal_split(df, cfg)

    # ── Inference-bundle statistics (raw, pre-deseasonalisation) ─────────────
    # Taken from train+val: the risk-index baseline and env climatology describe
    # the whole training period and are never fed back into training.
    history_df     = pd.concat([train_df, val_df])
    env_vars       = cfg.get("features", {}).get("env_vars", [])
    baselines      = target_baselines(history_df, node_index, cfg)
    climatology    = env_climatology(history_df, env_vars, cfg)
    missing_q      = missing_quality_pattern(
        train_df, cfg.get("features", {}).get("quality_dummy_vars", []), cfg
    )
    train_period   = (str(history_df[time_col].min())[:7], str(history_df[time_col].max())[:7])

    # ── Deseasonalise: fit on train only, apply to all splits ────────────────
    # Monthly means are computed from train_df while NaNs are still present,
    # so missing positions are correctly excluded from the mean. The same
    # fixed means are then applied to val and test to avoid data leakage.
    seasonal_means: dict = {}
    if prep.get("deseasonalise"):
        seasonal_means = fit_seasonal_means(train_df, cfg)
        train_df = apply_seasonal_means(train_df, seasonal_means, cfg)
        val_df   = apply_seasonal_means(val_df,   seasonal_means, cfg)
        if not test_df.empty:
            test_df = apply_seasonal_means(test_df, seasonal_means, cfg)
        print("Deseasonalisation applied (fitted on train only).")

    sources = separate_sources(train_df, val_df, test_df, cfg)

    # ── Incidence NaN handling ───────────────────────────────────────────────
    # Order matters: masks must be built before filling so that NaN positions
    # are captured while still present, then zeros are filled explicitly before
    # scaling so scale_sources can assume clean input.
    scale_target = prep.get("scale_target", True)
    masks   = build_masks(sources)
    sources = fill_missing_inc(sources)
    scaled  = scale_sources(sources, scale_target=scale_target)

    # Write scaled values back into the DataFrames so reshape_all sees scaled data.
    # scale_sources operates on flat numpy arrays; we put them back in place here.
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    for split_key in scaled["inc"]:
        df_obj = split_dfs[split_key]
        df_obj[target]   = scaled["inc"][split_key].reshape(-1)
        df_obj[env_vars] = scaled["env"][split_key]

    tensors = reshape_all(masks, train_df, val_df, test_df, node_index, cfg)

    # ── Naive imputation strategy ────────────────────────────────────────────
    # Stage 1: Labuan filled — residual NaNs expected (LST_Night etc.)
    tensors = fill_node_from_donors(
        tensors,
        node_index,
        receiver = "W.P. Labuan",
        donors   = ["Sabah", "Brunei And Muara", "Belait", "Temburong", "Tutong"],
    )
    assert_no_nans(tensors, stage="post-labuan-fill", keys=["env", "lulc"],
                raise_on_fail=False)   # ← warn only

    # Stage 2: ffill → bfill → global mean for remaining sparse NaNs
    tensors = impute_env_naive(tensors, node_index, cfg)
    assert_no_nans(tensors, stage="post-imputation",  keys=["env", "lulc"],
                raise_on_fail=True)    # ← hard stop

    # ── Month indices per split (needed for inverse deseasonalisation) ────────
    date_col = cfg["data"]["time_column"]
    split_months = {
        split: [pd.to_datetime(d).month for d in sorted(split_dfs[split][date_col].unique())]
        for split in tensors
    }
    include_incidence = prep.get("incidence_input", True)

    # ── Window creation ──────────────────────────────────────────────────────
    window_sizes = get_window_sizes(cfg)
    for window_size in window_sizes:
        snapshots = create_windows(tensors, window_size, node_index, cfg)

        if window_size == window_sizes[0]:   # only check first window size
            diagnose_feature_composition(snapshots, cfg)

        # Inference windows span the train/val→test boundary so every test
        # month has a prediction.  The month list is padded with window_size
        # zeros so that save_tensors' [window_size:] slice yields the 14
        # test-month numbers correctly.
        # Production runs (test_months: 0) have no test split to cover.
        if "test" in tensors:
            snapshots["inference"] = create_inference_windows(tensors, window_size, include_incidence)
            inference_months = {
                **split_months,
                "inference": [0] * window_size + split_months["test"],
            }
        else:
            inference_months = split_months
        save_tensors(snapshots, cfg, window_size, split_months=inference_months)

    save_preprocessing_params(scaled["scalers"]["inc"], seasonal_means, cfg)
    save_inference_bundle(
        inference_bundle_path(cfg),
        cfg,
        nodes           = node_table(df, node_index, cfg),
        scalers         = scaled["scalers"],
        seasonal_means  = seasonal_means,
        baselines       = baselines,
        climatology     = climatology,
        missing_quality = missing_q,
        train_period    = train_period,
    )

if "snakemake" in dir():
    # All outputs share one directory; anchor it to the declared output so staged paths match.
    set_output_dir(Path(snakemake.output.scaler).parent)
    main(cfg=dict(snakemake.config), data_path=str(snakemake.input[0]))
elif __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)