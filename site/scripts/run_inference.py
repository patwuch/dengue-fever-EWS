"""
Monthly STGNN inference, in one of two modes.

  --mode climate     (mode 2) Climate-only model (production_climate_risk). Inputs
                     are the last window_size months of Earth Engine zonal stats,
                     delta-corrected to the 2011-2018 climate by default. Output is
                     a relative risk index only — no incidence estimate.

  --mode incidence   (mode 1) Autoregressive model (production_logIR). Inputs also
                     include the last window_size months of reported incidence for
                     whichever provinces have it (--incidence CSV); provinces with
                     no report get the same fill value the model saw for missing
                     incidence in training. Output is predicted IR plus the risk index.

Both modes forecast one month ahead: inputs for months L-w+1..L predict month
L+1, the same alignment as training (x[t-w:t] → y[t]).

Risk index (both modes): (1 + predicted IR) / (1 + the province's mean IR for
that calendar month over 2011-2018), computed in log1p space as
exp(pred_log1p - baseline_log1p). Above 1 = above the province's usual level
for that time of year. Falls back to the SEA-wide calendar-month baseline for
provinces with no reported cases in training.

Inputs:
    <bundle>/bundle.json, best_params.json, best_model.pt
        from `snakemake results/STGNN/<name>/inference_bundle`
    site/data/zonal_stats/<YYYY-MM>.json      from fetch_zonal_stats.py --backfill
    site/data/climate_deltas.json             from compute_climate_delta.py (optional)
    --incidence CSV (mode 1 only), columns:
        adm_1_name, year_month (YYYY-MM), and either IR (cases per 100,000)
        or dengue_total + population_sum. Optional adm_0_name.

Outputs:
    site/data/predictions/<mode>/<target>.json   archived per month
    site/data/latest_predictions_<mode>.json
    site/data/latest_predictions.json            climate mode only (site risk map)
"""

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = pathlib.Path(__file__).parent
ROOT        = SCRIPTS_DIR.parent.parent
STGNN_DIR   = ROOT / "machine-learning-module/src/OpenDengue/STGNN"
sys.path.insert(0, str(STGNN_DIR))
sys.path.insert(0, str(STGNN_DIR / "preprocess"))
sys.path.insert(0, str(SCRIPTS_DIR))

from model import STGATGRU                                       # noqa: E402
from graph import build_edge_index                               # noqa: E402
from apply_climate_delta import load_deltas, correct_stats       # noqa: E402

DATA_DIR    = SCRIPTS_DIR.parent / "data"
STATS_DIR   = DATA_DIR / "zonal_stats"
ARCHIVE_DIR = DATA_DIR / "predictions"


# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------

def shift_month(month: str, delta: int) -> str:
    dt = datetime.datetime.strptime(month, "%Y-%m")
    idx = dt.year * 12 + (dt.month - 1) + delta
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def latest_archived_month() -> str:
    months = sorted(p.stem for p in STATS_DIR.glob("????-??.json"))
    if not months:
        raise FileNotFoundError(f"No zonal stats in {STATS_DIR} — run fetch_zonal_stats.py first.")
    return months[-1]


# ---------------------------------------------------------------------------
# Bundle & model
# ---------------------------------------------------------------------------

def load_bundle(bundle_dir: pathlib.Path) -> tuple[dict, dict]:
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    params = json.loads((bundle_dir / "best_params.json").read_text())
    if bundle.get("format_version") != 1:
        raise ValueError(f"Unsupported bundle format_version {bundle.get('format_version')!r}")
    return bundle, params


def load_model(bundle_dir: pathlib.Path, params: dict, n_features: int) -> STGATGRU:
    state = torch.load(bundle_dir / "best_model.pt", map_location="cpu", weights_only=True)
    trained_f = state["missing_emb"].shape[0]
    if trained_f != n_features:
        raise ValueError(f"Model was trained on {trained_f} features; bundle describes {n_features}.")
    model = STGATGRU(
        in_channels  = n_features,
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params.get("mlp_hidden", params["gat1_hidden"] * params["gat1_heads"]),
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params.get("gru_hidden", params["gat2_hidden"]),
        pred_horizon = 1,
        dropout      = params.get("dropout", 0.0),
    )
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature construction — mirrors preprocess/pipeline.py
# ---------------------------------------------------------------------------

def load_window_stats(months: list[str], deltas: dict | None) -> list[dict]:
    missing = [m for m in months if not (STATS_DIR / f"{m}.json").exists()]
    if missing:
        raise FileNotFoundError(
            f"Zonal stats missing for {missing}. Run fetch_zonal_stats.py "
            f"--month {months[-1]} --backfill {len(months)}."
        )
    return [correct_stats(json.loads((STATS_DIR / f"{m}.json").read_text()), deltas) for m in months]


def match_nodes(nodes: list[dict], regions: dict) -> list[str | None]:
    """Region key in the zonal stats for each model node (None if absent)."""
    keys = []
    for node in nodes:
        rec = regions.get(node["name"])
        if rec is not None and node["country"] and rec.get("country") not in ("", None, node["country"]):
            print(f"  WARNING: {node['name']!r} is {node['country']!r} in training but "
                  f"{rec.get('country')!r} in zonal stats")
        keys.append(node["name"] if rec is not None else None)
    return keys


def stack_features(window: list[dict], keys: list[str | None], features: list[str]) -> np.ndarray:
    """(T, N, F) raw feature array, NaN where a value is missing."""
    arr = np.full((len(window), len(keys), len(features)), np.nan, dtype=np.float64)
    for t, stats in enumerate(window):
        regions = stats["regions"]
        for n, key in enumerate(keys):
            if key is None:
                continue
            vals = regions[key]
            for f, feat in enumerate(features):
                v = vals.get(feat)
                if isinstance(v, (int, float)):
                    arr[t, n, f] = v
    return arr


def fill_over_time(arr: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ffill → bfill along T per (node, feature), then `fallback` per feature.

    Returns the filled array and an (N,) count of months that had any gap.
    """
    gaps = np.isnan(arr).any(axis=2).sum(axis=0)
    T, N, F = arr.shape
    flat = pd.DataFrame(arr.reshape(T, N * F)).ffill().bfill().to_numpy().reshape(T, N, F)
    flat = np.where(np.isnan(flat), fallback.reshape(1, 1, F), flat)
    return flat, gaps


def incidence_features(
    csv_path: pathlib.Path, months: list[str], bundle: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    (T, N, 1) incidence channel in model space and a (T, N) observed mask.
    Applies the training transforms: log1p → minus calendar-month mean → z-score.
    """
    nodes = [n["name"] for n in bundle["nodes"]]
    df = pd.read_csv(csv_path)
    if "IR" not in df.columns:
        if not {"dengue_total", "population_sum"} <= set(df.columns):
            raise ValueError("Incidence CSV needs an IR column or dengue_total + population_sum.")
        df["IR"] = df["dengue_total"] / df["population_sum"] * 100_000
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.strftime("%Y-%m")
    df = df.dropna(subset=["IR"])

    unknown = sorted(set(df["adm_1_name"]) - set(nodes))
    if unknown:
        print(f"  WARNING: incidence rows for provinces the model doesn't know (ignored): {unknown}")

    table = (
        df[df["year_month"].isin(months) & df["adm_1_name"].isin(nodes)]
        .groupby(["year_month", "adm_1_name"])["IR"].mean()
        .unstack()
        .reindex(index=months, columns=nodes)
    )
    observed = table.notna().to_numpy()
    values = table.to_numpy(dtype=np.float64)

    if bundle["log_transform"]:
        values = np.log1p(values)
    if bundle["deseasonalise"]:
        seasonal = np.array([bundle["seasonal_means"][str(int(m[5:7]))] for m in months])
        values = values - seasonal[:, None]
    if bundle["inc_scaler"]:
        values = (values - bundle["inc_scaler"]["mean"]) / bundle["inc_scaler"]["std"]
    values = np.where(observed, values, bundle["inc_fill_value"])

    coverage = {m: int(observed[t].sum()) for t, m in enumerate(months)}
    return values[..., None], observed, coverage


def quality_features(observed: np.ndarray, bundle: dict) -> np.ndarray:
    """(T, N, Q) data_quality dummies: zeros (OBSERVED) where incidence was reported,
    otherwise the pattern training rows with missing incidence carried."""
    q = len(bundle["features"]["quality_dummy_vars"])
    if q == 0:
        return np.zeros(observed.shape + (0,))
    missing = np.asarray(bundle["missing_quality_dummies"], dtype=np.float64)
    return np.where(observed[..., None], 0.0, missing.reshape(1, 1, q))


# ---------------------------------------------------------------------------
# Output transforms
# ---------------------------------------------------------------------------

def to_target_space(pred: np.ndarray, target_month: str, bundle: dict) -> np.ndarray:
    """Model output → log1p(IR) (or raw IR when the model wasn't log-transformed)."""
    z = pred.astype(np.float64)
    if bundle["inc_scaler"]:
        z = z * bundle["inc_scaler"]["std"] + bundle["inc_scaler"]["mean"]
    if bundle["deseasonalise"]:
        z = z + bundle["seasonal_means"][str(int(target_month[5:7]))]
    return z


def risk_index(z: np.ndarray, target_month: str, bundle: dict) -> np.ndarray:
    m = str(int(target_month[5:7]))
    base = bundle["target_baseline"]
    fallback = base["by_month"].get(m)
    baseline = np.array([
        v if (v := base["by_node_month"].get(n["name"], {}).get(m)) is not None else fallback
        for n in bundle["nodes"]
    ], dtype=np.float64)
    if bundle["log_transform"]:
        return np.exp(z - baseline)
    return z / baseline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["climate", "incidence"])
    ap.add_argument("--bundle", required=True, type=pathlib.Path,
                    help="Directory with bundle.json, best_params.json, best_model.pt.")
    ap.add_argument("--incidence", type=pathlib.Path, default=DATA_DIR / "incidence/recent_incidence.csv",
                    help="Recent incidence CSV (incidence mode).")
    ap.add_argument("--last-month", default=None,
                    help="Last input month YYYY-MM (default: latest archived zonal stats). Forecast is the month after.")
    ap.add_argument("--delta", choices=["auto", "on", "off"], default="auto",
                    help="Climate delta correction. auto = on for climate mode, off for incidence mode.")
    ap.add_argument("--max-missing-nodes", type=float, default=0.2,
                    help="Fail if more than this fraction of provinces has no env data at all.")
    args = ap.parse_args()

    bundle, params = load_bundle(args.bundle)
    if bundle["mode"] != args.mode:
        raise ValueError(f"--mode {args.mode} but bundle {bundle['name']!r} is a {bundle['mode']!r}-mode model.")

    window_size  = int(params["window_size"])
    last_month   = args.last_month or latest_archived_month()
    months       = [shift_month(last_month, -k) for k in reversed(range(window_size))]
    target_month = shift_month(last_month, 1)
    nodes        = bundle["nodes"]
    feats        = bundle["features"]
    print(f"[{args.mode}] model {bundle['name']} (trained {bundle['train_period'][0]}–{bundle['train_period'][1]})")
    print(f"  inputs {months[0]} → {months[-1]}  ⇒  forecast {target_month}")

    # ── Climate delta ────────────────────────────────────────────────────────
    use_delta = args.delta == "on" or (args.delta == "auto" and args.mode == "climate")
    deltas = load_deltas() if use_delta else None
    if use_delta and deltas is None:
        print("  WARNING: delta correction requested but site/data/climate_deltas.json is missing "
              "or empty — running on uncorrected observations.")
    window = load_window_stats(months, deltas)
    bias_corrected = all(s["bias_corrected"] for s in window)

    # ── Env + land use ───────────────────────────────────────────────────────
    keys = match_nodes(nodes, window[-1]["regions"])
    unmatched = [n["name"] for n, k in zip(nodes, keys) if k is None]
    if unmatched:
        print(f"  WARNING: {len(unmatched)} model provinces absent from zonal stats: {unmatched}")

    env_raw = stack_features(window, keys, feats["env_vars"])
    no_env  = np.isnan(env_raw).all(axis=(0, 2))
    if no_env.mean() > args.max_missing_nodes:
        raise RuntimeError(f"{no_env.sum()}/{len(nodes)} provinces have no env data — refusing to publish.")

    scaler   = bundle["env_scaler"]
    env_mean = np.asarray(scaler["mean"])
    env_raw, env_gaps = fill_over_time(env_raw, env_mean)   # training mean → 0 after scaling
    env = (env_raw - env_mean) / np.asarray(scaler["scale"])

    lulc, _ = fill_over_time(stack_features(window, keys, feats["lulc_vars"]),
                             np.zeros(len(feats["lulc_vars"])))

    # ── Incidence + quality flags ────────────────────────────────────────────
    parts, coverage = [], None
    if bundle["incidence_input"]:
        if not args.incidence.exists():
            raise FileNotFoundError(f"Incidence mode needs {args.incidence}.")
        inc, observed, coverage = incidence_features(args.incidence, months, bundle)
        if not observed.any():
            raise RuntimeError(f"No incidence in {args.incidence} for {months[0]}–{months[-1]}.")
        print(f"  incidence coverage (provinces per month): {coverage}")
        parts.append(inc)
    else:
        observed = np.zeros((window_size, len(nodes)), dtype=bool)
    parts += [env, lulc, quality_features(observed, bundle)]

    x = np.concatenate(parts, axis=-1)
    if x.shape[-1] != len(feats["order"]):
        raise ValueError(f"Built {x.shape[-1]} features; bundle expects {len(feats['order'])}.")

    # ── Forward pass ─────────────────────────────────────────────────────────
    model = load_model(args.bundle, params, x.shape[-1])
    x_t   = torch.tensor(x, dtype=torch.float32).unsqueeze(0)        # (1, T, N, F)
    # Training masked nodes whose *target* was unreported; every node is a
    # prediction target here, which is the mask=1 case.
    mask  = torch.ones(1, len(nodes))
    edge_index = build_edge_index({n["name"]: i for i, n in enumerate(nodes)})
    with torch.no_grad():
        pred = model(x_t, edge_index, mask=mask).squeeze(0).squeeze(-1).numpy()

    z    = to_target_space(pred, target_month, bundle)
    risk = risk_index(z, target_month, bundle)

    # ── Output ───────────────────────────────────────────────────────────────
    regions_out = {}
    for i, node in enumerate(nodes):
        rec = {
            "country":           node["country"],
            "risk_index":        round(float(risk[i]), 4),
            "has_data":          bool(not no_env[i]),
            "env_months_imputed": int(env_gaps[i]),
        }
        if bundle["incidence_input"]:
            rec["predicted_ir"] = round(float(np.expm1(z[i]) if bundle["log_transform"] else z[i]), 4)
            rec["incidence_months_observed"] = int(observed[:, i].sum())
        regions_out[node["name"]] = rec

    output = {
        "mode":           args.mode,
        "model":          bundle["name"],
        "train_period":   bundle["train_period"],
        "generated_at":   datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_month":   target_month,
        "input_months":   months,
        "bias_corrected": bias_corrected,
        "risk_index_definition": "(1 + predicted IR) / (1 + 2011-2018 mean IR for this province and calendar month)",
        "regions":        regions_out,
    }
    if coverage is not None:
        output["incidence_coverage"] = coverage

    payload = json.dumps(output, indent=2)
    archive = ARCHIVE_DIR / args.mode / f"{target_month}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(payload)
    (DATA_DIR / f"latest_predictions_{args.mode}.json").write_text(payload)
    if args.mode == "climate":
        (DATA_DIR / "latest_predictions.json").write_text(payload)

    elevated = sum(r["risk_index"] > 1 for r in regions_out.values())
    print(f"\nArchived → {archive}")
    print(f"Provinces above their 2011–2018 baseline (risk_index > 1): {elevated}/{len(nodes)}")


if __name__ == "__main__":
    main()
