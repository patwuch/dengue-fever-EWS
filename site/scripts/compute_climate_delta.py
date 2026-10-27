"""
Per-variable, per-calendar-month climate deltas for climate-mode STGNN inference.

delta[MM][feature] = (recent mean for calendar month MM) − (the model's 2011–2018 mean for MM)

Both sides are SEA-wide means over the model's provinces. The 2011–2018 side is the bundle's
env_climatology (train + validation rows, preprocess/bundle.py); the recent side comes from the
province × month env table (build_env_table.py), which aggregates the EZGEE products exactly as
the training data was aggregated. apply_climate_delta.py subtracts the deltas from each input
month before inference.

The recent period should end at a December so every calendar month averages the same years,
and before the months being forecast, so recent anomalies don't feed their own baseline.

Usage (from the repo root):
    python site/scripts/compute_climate_delta.py \\
        --env    data/processed/env/SEA_env_monthly.csv \\
        --bundle results/STGNN/production_climate_risk/inference_bundle_climate.tar.gz \\
        --start 2019-01 --end 2025-12 -o site/data/climate_deltas.json
"""

import argparse
import json
import pathlib
import tarfile

import pandas as pd


def read_bundle(path: pathlib.Path) -> dict:
    """bundle.json from an inference bundle .tar.gz, its directory, or the file itself."""
    if path.suffix == ".gz":
        with tarfile.open(path) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("bundle.json"))
            return json.load(tar.extractfile(member))
    if path.is_dir():
        path = path / "bundle.json"
    return json.loads(path.read_text())


def compute_deltas(env: pd.DataFrame, bundle: dict, start: str, end: str) -> dict:
    baseline = bundle["env_climatology"]
    # month_sin/month_cos are calendar features, the same in every year.
    features = [f for f in baseline if f not in ("month_sin", "month_cos")]

    nodes   = {(n["country"], n["name"]) for n in bundle["nodes"]}
    is_node = pd.Series([k in nodes for k in zip(env["adm_0_name"], env["adm_1_name"])], index=env.index)
    month   = pd.to_datetime(env["year_month"]).dt.to_period("M")
    rows    = env[is_node & (month >= pd.Period(start, "M")) & (month <= pd.Period(end, "M"))]
    if rows.empty:
        raise RuntimeError(f"No env rows for the model's provinces between {start} and {end}.")
    rows_month = pd.to_datetime(rows["year_month"]).dt.month

    years = rows.groupby(rows_month)["year_month"].agg(lambda d: pd.to_datetime(d).dt.year.nunique())
    if years.nunique() > 1:
        print(f"  WARNING: calendar months cover unequal numbers of years: {years.to_dict()}")
    found = rows[["adm_0_name", "adm_1_name"]].drop_duplicates().shape[0]
    print(f"Recent period {start} → {end}: {found}/{len(nodes)} model provinces, "
          f"{years.min()}–{years.max()} years per calendar month")

    missing = [f for f in features if f not in env.columns]
    if missing:
        raise ValueError(f"Env table lacks features the bundle has a climatology for: {missing}")
    recent = rows.groupby(rows_month)[features].mean()

    deltas = {}
    for m in range(1, 13):
        mm = f"{m:02d}"
        deltas[mm] = {f: round(float(recent.loc[m, f] - baseline[f][mm]), 6) for f in features}
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, type=pathlib.Path, help="Province × month env table (CSV).")
    ap.add_argument("--bundle", required=True, type=pathlib.Path,
                    help="Climate-mode inference bundle (.tar.gz, directory, or bundle.json).")
    ap.add_argument("--start", default="2019-01", metavar="YYYY-MM")
    ap.add_argument("--end", default="2025-12", metavar="YYYY-MM")
    ap.add_argument("-o", "--output", required=True, type=pathlib.Path)
    args = ap.parse_args()

    bundle = read_bundle(args.bundle)
    deltas = compute_deltas(pd.read_csv(args.env), bundle, args.start, args.end)

    print(f"\nAnnual mean delta vs {bundle['name']} ({bundle['train_period'][0]}–{bundle['train_period'][1]}):")
    for feat in ["temperature_2m_mean", "precipitation_mm", "precipitation_sum", "LST_Day_1km_mean", "NDVI_mean"]:
        if feat in deltas["01"]:
            print(f"  {feat:24s} {sum(d[feat] for d in deltas.values()) / 12:+.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(deltas, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
