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


def horizon(cfg: dict) -> int:
    """Months ahead the model predicts: 1 → month t only; 2 → months t and t+1."""
    return int(cfg.get("preprocessing", {}).get("horizon", 1))


def _targets(inc: torch.Tensor, mask: torch.Tensor, t: int, h: int):
    """Target and mask for a window ending before month t.

    h == 1 keeps the original shapes, (N, 1) and (N,). h > 1 gives (N, h) for both: lead k
    is month t + k - 1, and months past the end of the series are masked out.
    """
    if h == 1:
        return inc[t], mask[t]
    n = inc.shape[1]
    y = torch.zeros(n, h, dtype=inc.dtype)
    m = torch.zeros(n, h, dtype=torch.bool)
    for k in range(h):
        if t + k < len(inc):
            y[:, k] = inc[t + k, :, 0]
            m[:, k] = mask[t + k]
    return y, m


def create_windows(tensors: dict, window_size: int, node_index: dict, cfg: dict) -> dict:
    snapshots = {}
    include_incidence = cfg.get("preprocessing", {}).get("incidence_input", True)
    h = horizon(cfg)

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

        # Within a split every lead's target must exist, so the last h - 1 months only
        # appear as later leads of earlier windows.
        snapshots[split] = [
            (x[t-window_size:t], *_targets(inc, mask, t, h))
            for t in range(window_size, len(x) - h + 1)
        ]

    return snapshots


def create_inference_windows(tensors: dict, window_size: int, include_incidence: bool = True,
                             h: int = 1) -> list:
    """Build windows that cover all test months.

    The first window's context is drawn from the tail of the train+val data,
    so every test month gets a prediction (not just the last len(test)-window_size).
    With h > 1, leads that fall past the last test month are masked out.
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
        (full_x[t - window_size:t], *_targets(test_inc, test_mask, t - window_size, h))
        for t in range(window_size, window_size + T_test)
    ]