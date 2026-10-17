from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def log_transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply log1p to the target column in-place."""
    col = cfg.get("target_column")
    df[col] = np.log1p(df[col])
    return df


def fit_scaler(x_train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    return scaler, x_train_scaled

def apply_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(x)

def add_cyclical_month_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add sin/cos calendar-month encodings. Deterministic — no fitted params, no leakage."""
    date_col = cfg["data"]["time_column"]
    months = pd.to_datetime(df[date_col]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * months / 12)
    df["month_cos"] = np.cos(2 * np.pi * months / 12)
    return df


def fit_seasonal_means(train_df: pd.DataFrame, cfg: dict) -> dict:
    """Compute per-month target means from training data only (NaN-aware)."""
    col      = cfg.get("target_column")
    date_col = cfg["data"]["time_column"]
    months   = pd.to_datetime(train_df[date_col]).dt.month
    return train_df.groupby(months)[col].mean().to_dict()   # {1: float, ..., 12: float}


def apply_seasonal_means(df: pd.DataFrame, seasonal_means: dict, cfg: dict) -> pd.DataFrame:
    """Subtract training monthly means from the target column."""
    col      = cfg.get("target_column")
    date_col = cfg["data"]["time_column"]
    months   = pd.to_datetime(df[date_col]).dt.month
    df[col]  = df[col] - months.map(seasonal_means)
    return df


def fit_and_scale(sources: dict) -> dict:
    """Fit scalers on train, apply to val and test. lulc left unscaled."""
    sources = fill_missing_inc(sources)
    return scale_sources(sources)


def fill_missing_inc(sources: dict) -> dict:
    """Fill NaN incidence values with 0 before scaling."""
    return {
        **sources,
        "inc": {
            split: series.fillna(0)
            for split, series in sources["inc"].items()
        },
    }


def scale_sources(sources: dict, scale_target: bool = True) -> dict:
    """Fit scalers on train, apply to val and test. Assumes NaNs already filled.

    scale_target=False leaves the incidence column in its preprocessed space
    (log1p + deseasonalised) without z-scoring, so expm1 alone inverts it.
    env features are always z-scored regardless of this flag.
    """
    # Splits present in sources — "test" is absent for production runs (test_months: 0).
    splits = list(sources["inc"].keys())

    inc = {split: sources["inc"][split].values.reshape(-1, 1) for split in splits}
    if scale_target:
        scaler_inc, _ = fit_scaler(inc["train"])
        inc = {split: apply_scaler(arr, scaler_inc) for split, arr in inc.items()}
        print("inc scaler std:", scaler_inc.scale_)
    else:
        scaler_inc = None
        print("inc scaler: disabled (scale_target=False)")

    env = {split: sources["env"][split].values for split in splits}
    if env["train"].shape[1] > 0:
        scaler_env, _ = fit_scaler(env["train"])
        env = {split: apply_scaler(arr, scaler_env) for split, arr in env.items()}
        print("env scaler std:", scaler_env.scale_)
    else:
        scaler_env = None
        print("env scaler: skipped (no env features)")

    return {
        "inc":     inc,
        "env":     env,
        "lulc":    {split: sources["lulc"][split].values for split in splits},
        "scalers": {"inc": scaler_inc, "env": scaler_env},
    }

def separate_sources(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cfg:      dict,
) -> dict:
    """Separate each split into incidence, env, and lulc source arrays.

    An empty test_df (test_months: 0) is left out rather than carried as an
    empty split, which the scalers would reject.
    """
    target   = cfg["target_column"]
    env_vars = cfg.get("features", {}).get("env_vars", [])
    lulc     = cfg.get("features", {}).get("land_use_vars", [])

    dfs = {"train": train_df, "val": val_df}
    if not test_df.empty:
        dfs["test"] = test_df

    return {
        "inc":  {split: df[target]   for split, df in dfs.items()},
        "env":  {split: df[env_vars] for split, df in dfs.items()},
        "lulc": {split: df[lulc]     for split, df in dfs.items()},
    }


def build_masks(sources: dict) -> dict:
    """Build missingness masks from IR NaNs before any filling."""
    return {
        split: ~series.isna().values
        for split, series in sources["inc"].items()
    }

def reshape_to_tensor(
    df:         pd.DataFrame,
    arr_cols:   list[str],
    node_index: dict,
    cfg:        dict,
) -> torch.Tensor:
    """Reshape to (T, N, F) with explicit node ordering regardless of df sort order."""
    node_col   = cfg["data"]["unit_column"]
    time_col   = cfg["data"]["time_column"]
    all_dates  = sorted(df[time_col].unique())
    node_order = sorted(node_index, key=node_index.get)

    frames = []
    for date in all_dates:
        day = (
            df[df[time_col] == date]
            .set_index(node_col)
            .reindex(node_order)
            [arr_cols]
            .values
        )
        frames.append(day)

    return torch.tensor(np.stack(frames), dtype=torch.float32)

def reshape_all(
    masks:      dict,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    node_index: dict,
    cfg:        dict,
) -> dict:
    target        = cfg["target_column"]
    time_col      = cfg["data"]["time_column"]
    env_vars      = cfg.get("features", {}).get("env_vars", [])
    lulc          = cfg.get("features", {}).get("land_use_vars", [])
    quality_vars  = cfg.get("features", {}).get("quality_dummy_vars", [])
    n             = len(node_index)
    dfs           = {"train": train_df, "val": val_df}
    if not test_df.empty:
        dfs["test"] = test_df
    tensors       = {}

    for split, df in dfs.items():
        t = len(df[time_col].unique())
        tensors[split] = {
            "inc":     reshape_to_tensor(df, [target],      node_index, cfg),
            "env":     reshape_to_tensor(df, env_vars,      node_index, cfg),
            "lulc":    reshape_to_tensor(df, lulc,          node_index, cfg),
            "quality": reshape_to_tensor(df, quality_vars,  node_index, cfg) if quality_vars else None,
            "mask":    torch.tensor(masks[split].reshape(t, n), dtype=torch.bool),
        }

    return tensors


def impute_env_naive(
    tensors:    dict,
    node_index: dict,
    cfg:        dict,
    max_fill_gap: int = 2,
) -> dict:
    """
    Naive ffill/bfill (gap ≤ max_fill_gap) → global column mean imputation
    on env tensor.

    Runs per-split to avoid leakage. Intended to catch residual NaNs
    after fill_labuan_from_sabah (e.g. LST_Night sparse gaps).

    Args:
        tensors:      Output of fill_labuan_from_sabah (or reshape_all).
        node_index:   {node_name: int_index} mapping.
        cfg:          Config dict.
        max_fill_gap: Maximum consecutive NaN timesteps to fill with
                      ffill/bfill. Longer gaps fall through to column mean.

    Returns:
        tensors with env imputed.
    """
    env_vars   = cfg.get("features", {}).get("env_vars", [])
    node_order = sorted(node_index, key=node_index.get)

    imputed_tensors = {}

    for split, data in tensors.items():
        data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        env = data["env"]               # (T, N, F)
        T, N, F = env.shape

        nan_before = torch.isnan(env).sum().item()
        if nan_before == 0:
            print(f"[impute_env]  {split:5s} | no NaNs — skipping.")
            imputed_tensors[split] = data
            continue

        env_np = env.numpy().reshape(T, N * F)
        df_env = pd.DataFrame(env_np)

        # ── Gap-limited ffill / bfill ─────────────────────────────────────
        # limit=max_fill_gap means only runs of NaNs of that length or shorter
        # are filled; longer runs are left as NaN and caught by the mean below.
        df_env = df_env.ffill(axis=0, limit=max_fill_gap)
        df_env = df_env.bfill(axis=0, limit=max_fill_gap)

        # ── Global column mean for remaining long gaps ────────────────────
        df_env = df_env.fillna(df_env.mean(axis=0))

        still_nan = df_env.isna().sum().sum()
        if still_nan > 0:
            print(f"[impute_env]  {split:5s} | WARNING: {still_nan} NaNs "
                  f"remain after all imputation steps.")

        env_imputed = torch.tensor(
            df_env.values.reshape(T, N, F),
            dtype=env.dtype,
        )

        nan_after = torch.isnan(env_imputed).sum().item()
        print(f"[impute_env]  {split:5s} | NaNs before: {nan_before:4d} "
              f"→ after: {nan_after:4d}")

        _report_imputed_nodes(env, env_imputed, node_order, env_vars, split)

        data["env"] = env_imputed
        imputed_tensors[split] = data

    return imputed_tensors

def _report_imputed_nodes(
    env_orig:    torch.Tensor,
    env_imputed: torch.Tensor,
    node_order:  list,
    env_vars:    list,
    split:       str,
) -> None:
    """Print which (node, feature) pairs were imputed and by which method."""
    T, N, F  = env_orig.shape
    was_nan  = torch.isnan(env_orig)

    for n_idx in range(N):
        for f_idx in range(F):
            n_nans = was_nan[:, n_idx, f_idx].sum().item()
            if n_nans == 0:
                continue
            node_name = node_order[n_idx] if n_idx < len(node_order) else n_idx
            feat_name = env_vars[f_idx]   if f_idx < len(env_vars)   else f_idx
            method    = "global_mean" if n_nans == T else "ffill/bfill"
            print(f"  [{split}] {node_name} | {feat_name}: "
                  f"{n_nans}/{T} timesteps → {method}")


def assert_no_nans(
    tensors:  dict,
    stage:    str,
    keys:     list = None,
    raise_on_fail: bool = True,
) -> bool:
    """
    Check for NaNs in tensors. 

    Args:
        tensors:       The tensors dict to check.
        stage:         Label for the error/warning message.
        keys:          Which tensor keys to check. Defaults to all.
        raise_on_fail: If True, raises ValueError on failure.
                       If False, prints warnings and returns False.

    Returns:
        True if all clean, False if NaNs found (only when raise_on_fail=False).
    """
    all_clean = True

    for split, data in tensors.items():
        for key, tensor in data.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if keys is not None and key not in keys:
                continue
            n_nan = torch.isnan(tensor).sum().item()
            if n_nan > 0:
                print(f"[assert_no_nans] FAIL  | {stage} | {split}.{key}: "
                      f"{n_nan} NaNs remaining")
                all_clean = False
            else:
                print(f"[assert_no_nans] OK    | {stage} | {split}.{key}")

    if not all_clean and raise_on_fail:
        raise ValueError(
            f"NaNs remain in tensors after stage '{stage}'. "
            f"See above for details."
        )

    return all_clean
    

def fill_node_from_donors(
    tensors:    dict,
    node_index: dict,
    receiver:   str,
    donors:     list[str],
) -> dict:
    """
    Fill a receiver node's env/lulc NaNs using a priority-ordered donor list.

    For each NaN position, tries donors in order and uses the first one
    that has a valid (non-NaN) value at that position. Falls through to
    the next donor if the current one is also NaN.

    Args:
        tensors:    Output of reshape_all.
        node_index: {node_name: int_index} mapping.
        receiver:   Node name to fill (e.g. 'W.P. Labuan').
        donors:     Ordered list of donor node names, tried left to right.
                    e.g. ['Sabah', 'Brunei And Muara', 'Belait']
    """
    if receiver not in node_index:
        print(f"[fill_node] '{receiver}' not in node_index — skipping.")
        return tensors

    available_donors = [d for d in donors if d in node_index]
    missing_donors   = [d for d in donors if d not in node_index]
    if missing_donors:
        print(f"[fill_node] donors not in node_index (skipped): {missing_donors}")
    if not available_donors:
        print(f"[fill_node] no valid donors available — skipping.")
        return tensors

    receiver_idx  = node_index[receiver]
    donor_indices = [node_index[d] for d in available_donors]

    filled_tensors = {}

    for split, data in tensors.items():
        data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        for key in ("env", "lulc"):
            tensor   = data[key]                           # (T, N, F)
            receiver_slice = tensor[:, receiver_idx, :]   # (T, F)

            total_filled = 0

            for donor_name, donor_idx in zip(available_donors, donor_indices):
                still_nan = torch.isnan(receiver_slice)   # (T, F) bool
                if not still_nan.any():
                    break                                  # fully filled

                donor_slice  = tensor[:, donor_idx, :]    # (T, F)
                can_fill     = still_nan & ~torch.isnan(donor_slice)
                n_filled     = can_fill.sum().item()

                receiver_slice[can_fill] = donor_slice[can_fill]
                total_filled += n_filled

                if n_filled > 0:
                    print(f"[fill_node] {split:5s} | {key:4s} | "
                          f"'{donor_name}' filled {n_filled} positions")

            tensor[:, receiver_idx, :] = receiver_slice
            data[key] = tensor

            remaining = torch.isnan(tensor[:, receiver_idx, :]).sum().item()
            if remaining > 0:
                print(f"[fill_node] {split:5s} | {key:4s} | "
                      f"{remaining} positions still NaN after all donors "
                      f"— will fall through to impute_env_naive")

        filled_tensors[split] = data

    return filled_tensors


def encode_quality_flags(df: pd.DataFrame, quality_vars: list) -> tuple:
    dummy_cols = []
    for col in quality_vars:
        # Set OBSERVED as the reference category (dropped baseline)
        order   = ["OBSERVED"] + sorted(
            [c for c in df[col].unique() if c != "OBSERVED"]
        )
        df[col] = pd.Categorical(df[col], categories=order)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
        dummy_cols.extend(dummies.columns.tolist())
        df = pd.concat([df, dummies], axis=1).drop(columns=[col])
    return df, dummy_cols


def diagnose_feature_composition(snapshots: dict, cfg: dict) -> None:
    """
    Print a breakdown of what's in x and confirm data_quality dummies
    are present, non-NaN, and non-constant across nodes.
    """
    env_vars     = cfg.get("features", {}).get("env_vars",         [])
    lulc_vars    = cfg.get("features", {}).get("land_use_vars",    [])
    quality_vars = cfg.get("features", {}).get("quality_dummy_vars", [])

    # Reconstruct expected feature order — must match create_windows
    inc_names     = ["inc"] if cfg.get("preprocessing", {}).get("incidence_input", True) else []
    feature_names = inc_names + env_vars + lulc_vars + quality_vars
    expected_f    = len(feature_names)

    x, y, mask = snapshots["train"][0]   # (window_size, N, F)
    actual_f   = x.shape[-1]

    print(f"\n[feature check] window shape:    {tuple(x.shape)}")
    print(f"[feature check] expected F:      {expected_f}")
    print(f"[feature check] actual F:        {actual_f}")

    if actual_f != expected_f:
        print(f"[feature check] MISMATCH — check create_windows concatenation order")
        return

    print(f"\n[feature check] Feature index map:")
    for i, name in enumerate(feature_names):
        col        = x[:, :, i]                        # (window_size, N)
        n_nan      = torch.isnan(col).sum().item()
        n_unique   = col[~torch.isnan(col)].unique().numel()
        print(f"  [{i:3d}] {name:<60s} | NaNs: {n_nan:4d} | unique values: {n_unique}")

    # Specifically check quality dummy columns
    if quality_vars:
        print(f"\n[feature check] Quality dummy summary (train split, t=0):")
        q_start = len(inc_names) + len(env_vars) + len(lulc_vars)
        for i, name in enumerate(quality_vars):
            col     = x[0, :, q_start + i]             # (N,) at first timestep
            vals    = col.unique().tolist()
            n_ones  = (col == 1.0).sum().item()
            n_zeros = (col == 0.0).sum().item()
            print(f"  {name}: values={vals}  ones={n_ones}  zeros={n_zeros}  "
                  f"(out of {col.shape[0]} nodes)")
    else:
        print("[feature check] No quality dummy vars found in config.")
