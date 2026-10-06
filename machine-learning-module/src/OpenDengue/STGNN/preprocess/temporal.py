from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def build_snapshot(
    t: int,
    window_size: int,
    incidence_norm: np.ndarray,
    monthly_norm: np.ndarray,
    yearly_expanded_norm: np.ndarray,
) -> torch.Tensor:
    """
    Stack three feature sources into a single node feature matrix [num_regions, num_features]
    at timestep t.
    """
    # Rolling incidence history: [num_regions, window_size]
    incidence_window = incidence_norm[t - window_size:t].T

    # Monthly env at time t: [num_regions, num_monthly_feats]
    monthly_feats = monthly_norm[t]

    # Yearly env at time t (already expanded): [num_regions, num_yearly_feats]
    yearly_feats = yearly_expanded_norm[t]

    x = np.hstack([incidence_window, monthly_feats, yearly_feats])
    return torch.tensor(x, dtype=torch.float)


def temporal_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df chronologically into train, val, and test by date.

    test_months: 0 (production runs) returns an empty test_df, so every month
    goes to train/val.
    """
    test_months = cfg["data"]["split"]["test_months"]
    val_frac    = cfg["data"]["split"]["val_fraction"]
    date_col    = cfg["data"]["time_column"]

    all_dates   = sorted(df[date_col].unique())
    # all_dates[-0] is the first date, not "past the end", so 0 needs its own branch.
    if test_months > 0:
        test_cutoff = all_dates[-test_months]
        train_dates = [d for d in all_dates if d < test_cutoff]
    else:
        test_cutoff = None
        train_dates = all_dates

    n_val = max(1, int(len(train_dates) * val_frac))
    val_cutoff = train_dates[-n_val]

    train_df = df[df[date_col] < val_cutoff].copy()
    if test_cutoff is None:
        val_df  = df[df[date_col] >= val_cutoff].copy()
        test_df = df.iloc[0:0].copy()
    else:
        val_df  = df[(df[date_col] >= val_cutoff) & (df[date_col] < test_cutoff)].copy()
        test_df = df[df[date_col] >= test_cutoff].copy()

    return train_df, val_df, test_df

def _build_feature_tensor(data: dict, include_incidence: bool = True) -> torch.Tensor:
    """Concatenate inc (NaN→0), env, lulc, quality into (T, N, F).

    include_incidence=False drops the autoregressive incidence channel, giving
    a climate-only model that can run without recent case data.
    """
    parts = []
    if include_incidence:
        inc = data["inc"].clone()
        inc[torch.isnan(inc)] = 0.0
        parts.append(inc)
    parts += [data["env"], data["lulc"]]
    if data.get("quality") is not None:
        parts.append(data["quality"])
    return torch.cat(parts, dim=-1)


def create_windows(tensors: dict, window_size: int, node_index: dict, cfg: dict) -> dict:
    snapshots = {}
    include_incidence = cfg.get("preprocessing", {}).get("incidence_input", True)

    for split, data in tensors.items():
        x    = _build_feature_tensor(data, include_incidence)
        inc  = data["inc"].clone()
        inc[torch.isnan(inc)] = 0.0
        mask = data["mask"]

        if window_size >= len(x):
            raise ValueError(
                f"window_size {window_size} >= number of timesteps {len(x)} "
                f"in {split} split."
            )

        snapshots[split] = [
            (x[t-window_size:t], inc[t], mask[t])
            for t in range(window_size, len(x))
        ]

    return snapshots


def create_inference_windows(tensors: dict, window_size: int, include_incidence: bool = True) -> list:
    """Build windows that cover all test months.

    The first window's context is drawn from the tail of the train+val data,
    so every test month gets a prediction (not just the last len(test)-window_size).
    """
    pre_x = torch.cat(
        [_build_feature_tensor(tensors["train"], include_incidence),
         _build_feature_tensor(tensors["val"], include_incidence)],
        dim=0,
    )[-window_size:]                              # (window_size, N, F)

    test_x    = _build_feature_tensor(tensors["test"], include_incidence)   # (T_test, N, F)
    test_inc  = tensors["test"]["inc"].clone()
    test_inc[torch.isnan(test_inc)] = 0.0
    test_mask = tensors["test"]["mask"]

    full_x = torch.cat([pre_x, test_x], dim=0)   # (window_size + T_test, N, F)
    T_test = len(test_x)

    return [
        (full_x[t - window_size:t], test_inc[t - window_size], test_mask[t - window_size])
        for t in range(window_size, window_size + T_test)
    ]