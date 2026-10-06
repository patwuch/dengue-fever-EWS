"""
Compute per-variable, per-calendar-month climate deltas for STGNN inference.

Takes the wide GeoParquet output(s) from the offline zonal statistics app
(one per product: ERA5_LAND, MODIS_LST, MODIS_NDVI_EVI) covering 2019–2026
and compares them to the 2011–2018 training period baseline, producing a
stable delta file used by site/scripts/apply_climate_delta.py.

Run once offline on the lab machine after the batch GEE fetch completes.
Commit the output (climate_deltas.json) to the repo — it is a small static
artifact (~5 KB) that the monthly CI runner reads without needing the parquets.

Usage:
    python compute_climate_delta.py \\
        --era5    data/runs/sea_2019_2026/results/ERA5_LAND/ERA5_LAND_2019-01-01_to_2026-12-31.parquet \\
        --chirps  data/runs/sea_2019_2026/results/CHIRPS/CHIRPS_2019-01-01_to_2026-12-31.parquet \\
        --lst     data/runs/sea_2019_2026/results/MODIS_LST/MODIS_LST_2019-01-01_to_2026-12-31.parquet \\
        --ndvi    data/runs/sea_2019_2026/results/MODIS_NDVI_EVI/MODIS_NDVI_EVI_2019-01-01_to_2026-12-31.parquet \\
        --output  ../site/data/climate_deltas.json

    Add --bundle <inference_bundle dir or bundle.json> to take the 2011–2018
    baseline from the model's inference bundle instead of the training CSV.

Column naming note:
    The offline app names wide-parquet columns as {band}_{stat_lower}, e.g.:
        temperature_2m_mean, LST_Day_1km_mean, NDVI_mean
    If your batch was run with different stat selections the column names may
    differ — run with --inspect to print discovered columns before computing.
"""

import argparse
import json
import pathlib
import sys

import pandas as pd
import geopandas as gpd

ROOT         = pathlib.Path(__file__).parent.parent
TRAINING_CSV = ROOT / "data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv"
DEFAULT_OUT  = ROOT / "site/data/climate_deltas.json"

# ---------------------------------------------------------------------------
# Column mapping: parquet column name → STGNN feature name
#
# The offline app names columns {band}_{stat_lower}, which is also how the
# STGNN features are named. precipitation_sum is CHIRPS "precipitation" with
# a spatial SUM (merge_data.py takes it from the CHIRPS parquet), so it needs
# the CHIRPS batch, not ERA5-Land total precipitation, whose units differ.
# ---------------------------------------------------------------------------

PARQUET_TO_STGNN: dict[str, str] = {
    # CHIRPS
    "precipitation_sum":             "precipitation_sum",
    # ERA5-Land
    "temperature_2m_mean":           "temperature_2m_mean",
    "temperature_2m_max_mean":       "temperature_2m_max_mean",
    "temperature_2m_min_mean":       "temperature_2m_min_mean",
    "potential_evaporation_sum_mean":                             "potential_evaporation_sum_mean",
    "total_evaporation_sum_mean":                                 "total_evaporation_sum_mean",
    "evaporation_from_bare_soil_sum_mean":                        "evaporation_from_bare_soil_sum_mean",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_mean": "evaporation_from_open_water_surfaces_excluding_oceans_sum_mean",
    "evaporation_from_the_top_of_canopy_sum_mean":                "evaporation_from_the_top_of_canopy_sum_mean",
    "evaporation_from_vegetation_transpiration_sum_mean":         "evaporation_from_vegetation_transpiration_sum_mean",
    # MODIS LST
    "LST_Day_1km_mean":    "LST_Day_1km_mean",
    "LST_Night_1km_mean":  "LST_Night_1km_mean",
    # MODIS NDVI / EVI
    "NDVI_mean": "NDVI_mean",
    "EVI_mean":  "EVI_mean",
}

# STGNN features that must appear in the delta output.
# Missing features get a delta of 0.0 with a warning.
REQUIRED_FEATURES = list(dict.fromkeys(PARQUET_TO_STGNN.values()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_parquet(path: pathlib.Path) -> pd.DataFrame:
    """Load a GeoParquet or plain Parquet, dropping the geometry column."""
    try:
        gdf = gpd.read_parquet(path)
        df  = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    except Exception:
        df = pd.read_parquet(path)
    return df


def extract_calendar_month(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add a 'calendar_month' column (two-digit string '01'..'12')."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["calendar_month"] = df[date_col].dt.month.apply(lambda m: f"{m:02d}")
    return df


def regional_monthly_mean(
    df: pd.DataFrame,
    value_cols: list[str],
    date_col:   str = "Date",
    adm1_col:   str = "ADM1_NAME",
) -> pd.DataFrame:
    """
    Compute the mean of each value column grouped by calendar month,
    averaging across all regions and all years in the dataset.

    Returns a DataFrame indexed by calendar_month with one column per feature.
    """
    df = extract_calendar_month(df, date_col)
    numeric = [c for c in value_cols if c in df.columns]
    grouped = (
        df.groupby("calendar_month")[numeric]
        .mean(numeric_only=True)
    )
    return grouped


def bundle_climatology(bundle_path: pathlib.Path, features: list[str]) -> pd.DataFrame:
    """Per-calendar-month training means stored in an STGNN inference bundle."""
    if bundle_path.is_dir():
        bundle_path = bundle_path / "bundle.json"
    clim = json.loads(bundle_path.read_text())["env_climatology"]
    return pd.DataFrame({f: clim[f] for f in features if f in clim})


def training_climatology(features: list[str]) -> pd.DataFrame:
    """
    Compute per-calendar-month means from the 2011-2018 training CSV.
    Returns a DataFrame indexed by calendar_month.
    """
    if not TRAINING_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAINING_CSV}\n"
            "Run from the repo root or adjust ROOT in this script."
        )
    df = pd.read_csv(TRAINING_CSV, parse_dates=["year_month"])
    df["calendar_month"] = df["year_month"].dt.month.apply(lambda m: f"{m:02d}")
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: training CSV missing columns: {missing}")
    return df.groupby("calendar_month")[available].mean(numeric_only=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def inspect_parquets(paths: list[pathlib.Path]) -> None:
    """Print column names of each parquet so the user can verify the mapping."""
    for p in paths:
        if p is None or not p.exists():
            continue
        df = load_parquet(p)
        print(f"\n{p.name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")


def compute_deltas(
    parquet_paths: list[pathlib.Path],
    output_path:   pathlib.Path,
    bundle_path:   pathlib.Path | None = None,
) -> None:
    # ── Load and merge all batch parquets ────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for p in parquet_paths:
        if not p.exists():
            print(f"  SKIP (not found): {p}")
            continue
        print(f"  Loading {p.name} ...")
        df = load_parquet(p)
        frames.append(df)

    if not frames:
        raise RuntimeError("No parquet files could be loaded. Check paths.")

    # Merge on a common region + date key — outer join so all region×date
    # combinations from any product survive even if another product has gaps.
    batch = frames[0]
    for other in frames[1:]:
        # The SEA region table uses admin/name (filter_shp_convert_parquet.py);
        # joining on Date alone would cross-join every region with every other.
        join_cols = [c for c in ("ADM1_NAME", "ADM0_NAME", "admin", "name", "Date") if c in batch.columns and c in other.columns]
        if not join_cols:
            print("  WARNING: no common join columns found — concatenating instead of merging.")
            batch = pd.concat([batch, other], axis=1)
        else:
            other_new = other[[c for c in other.columns if c not in batch.columns or c in join_cols]]
            batch = batch.merge(other_new, on=join_cols, how="outer")

    print(f"\nMerged batch: {len(batch)} rows, {len(batch.columns)} columns")

    # Rename parquet columns to STGNN feature names
    rename_map = {k: v for k, v in PARQUET_TO_STGNN.items() if k in batch.columns}
    batch = batch.rename(columns=rename_map)
    print(f"Mapped {len(rename_map)} columns to STGNN feature names.")

    unmapped = [c for c in batch.columns if c not in ("ADM1_NAME", "ADM0_NAME", "admin", "name", "Date", "calendar_month") and c not in REQUIRED_FEATURES]
    if unmapped:
        print(f"  Unmapped columns (ignored): {unmapped}")

    # ── Compute batch climatology (2019-2026) ─────────────────────────────────
    print("\nComputing batch climatology ...")
    batch_clim = regional_monthly_mean(batch, REQUIRED_FEATURES)
    print(f"  Months with data: {sorted(batch_clim.index.tolist())}")

    # ── Compute training climatology (2011-2018) ──────────────────────────────
    if bundle_path is not None:
        print(f"\nReading training climatology from {bundle_path} ...")
        train_clim = bundle_climatology(bundle_path, REQUIRED_FEATURES)
    else:
        print("\nComputing training climatology from training CSV ...")
        train_clim = training_climatology(REQUIRED_FEATURES)
    print(f"  Months with data: {sorted(train_clim.index.tolist())}")

    # ── Compute deltas ────────────────────────────────────────────────────────
    # delta[month][feature] = batch_mean - training_mean
    # Positive delta → current climate is warmer/wetter than training baseline.
    # apply_climate_delta.py subtracts this from raw obs to shift them back.
    print("\nComputing deltas ...")
    months = [f"{m:02d}" for m in range(1, 13)]
    deltas: dict[str, dict[str, float]] = {}

    for month in months:
        deltas[month] = {}
        for feat in REQUIRED_FEATURES:
            batch_val = batch_clim.loc[month, feat] if (month in batch_clim.index and feat in batch_clim.columns) else None
            train_val = train_clim.loc[month, feat] if (month in train_clim.index and feat in train_clim.columns) else None

            if batch_val is None or train_val is None or pd.isna(batch_val) or pd.isna(train_val):
                print(f"  WARNING: missing data for {feat} month {month} — delta set to 0.0")
                deltas[month][feat] = 0.0
            else:
                deltas[month][feat] = round(float(batch_val - train_val), 6)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nDelta summary (annual mean across calendar months):")
    for feat in ["temperature_2m_mean", "precipitation_sum", "LST_Day_1km_mean", "NDVI_mean"]:
        vals = [deltas[m].get(feat, 0.0) for m in months]
        annual = sum(vals) / len(vals)
        direction = "warmer/higher" if annual > 0 else "cooler/lower"
        print(f"  {feat:50s}: {annual:+.4f}  ({direction} than 2011-2018 baseline)")

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(deltas, indent=2))
    print(f"\nClimate deltas saved → {output_path}")
    print("Commit this file to the repo — the CI runner reads it monthly.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--era5",   type=pathlib.Path, default=None, help="ERA5-Land wide parquet")
    parser.add_argument("--chirps", type=pathlib.Path, default=None, help="CHIRPS wide parquet (precipitation_sum)")
    parser.add_argument("--lst",    type=pathlib.Path, default=None, help="MODIS LST wide parquet")
    parser.add_argument("--ndvi",   type=pathlib.Path, default=None, help="MODIS NDVI/EVI wide parquet")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT, help="Output JSON path")
    parser.add_argument("--bundle", type=pathlib.Path, default=None,
                        help="STGNN inference bundle (dir or bundle.json) to use as the 2011-2018 baseline.")
    parser.add_argument("--inspect", action="store_true",
                        help="Print column names of input parquets and exit (no delta computed).")
    args = parser.parse_args()

    paths = [p for p in (args.era5, args.chirps, args.lst, args.ndvi) if p is not None]
    if not paths:
        parser.error("Provide at least one of --era5 / --chirps / --lst / --ndvi")

    if args.inspect:
        inspect_parquets(paths)
        sys.exit(0)

    compute_deltas(paths, args.output, args.bundle)


if __name__ == "__main__":
    main()
