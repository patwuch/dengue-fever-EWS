#!/usr/bin/env python3
"""Write an EZGEE Snakemake config for a date range, without the EZGEE web app.

Builds the payload the way EZGEE's backend does (backend/app.py, submit_run): product
settings come from the submodule's PRODUCT_REGISTRY and time chunks from get_time_chunks.
The band and statistic choices are the ones the 2019-01 → 2026-05 "lostyears" batch was
run with, so a monthly run yields the same columns as that batch.

Usage (from the repo root):
    python site/scripts/ezgee_config.py --start 2026-06 --end 2026-06 \\
        --shp regions.parquet --app-dir /work/ezgee --run-id m202606 -o ezgee_config.yaml
    snakemake -s zonal-statistics-module/Snakefile --configfile ezgee_config.yaml \\
        --directory /work/ezgee/data/runs/m202606 --resources gee=10 -j 12

Annual products (MODIS_LULC, WorldPop) are only included with --products; the env table
carries their latest year forward.
"""
import argparse
import calendar
import pathlib
import sys
from datetime import datetime

import yaml

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "zonal-statistics-module"))
from workflow.products import PRODUCT_REGISTRY  # noqa: E402
from workflow.time_chunks import get_time_chunks  # noqa: E402

ALL_STATS  = ["sum", "mean", "min", "max", "std", "variance"]
MODIS_STATS = ["sum", "mean", "min", "max", "std"]

# Band/statistic selection of the lostyears batch (its run.yaml payload).
SELECTION = {
    "CHIRPS": {"bands": ["precipitation"], "stats": ALL_STATS},
    "ERA5_LAND": {
        "bands": [
            "temperature_2m", "temperature_2m_min", "temperature_2m_max",
            "total_precipitation_sum", "total_evaporation_sum", "potential_evaporation_sum",
            "evaporation_from_bare_soil_sum",
            "evaporation_from_open_water_surfaces_excluding_oceans_sum",
            "evaporation_from_the_top_of_canopy_sum",
            "evaporation_from_vegetation_transpiration_sum",
        ],
        "stats": ALL_STATS,
    },
    "MODIS_LST":      {"bands": ["LST_Day_1km", "LST_Night_1km", "LST_Mean"], "stats": MODIS_STATS},
    "MODIS_NDVI_EVI": {"bands": ["NDVI", "EVI"], "stats": MODIS_STATS},
    "WorldPop":       {"bands": ["population"], "stats": ALL_STATS},
    "MODIS_LULC":     {"bands": ["LC_Type1"], "stats": ["histogram"]},
}
MONTHLY_PRODUCTS = ["CHIRPS", "ERA5_LAND", "MODIS_LST", "MODIS_NDVI_EVI"]


def product_task(product: str, date_start: str, date_end: str) -> dict:
    """One product's payload entry, as backend/app.py submit_run builds it."""
    info  = PRODUCT_REGISTRY[product]
    bands = SELECTION[product]["bands"]
    cadence = info["cadence"]
    dt_end = datetime.strptime(date_end, "%Y-%m-%d")
    dt_end = dt_end.replace(day=calendar.monthrange(dt_end.year, dt_end.month)[1])
    content = info.get("content", {})
    return {
        "ee_collection":       info.get("ee_collection"),
        "multi_collections":   info.get("multi_collections"),
        "bands":               bands,
        "statistics":          SELECTION[product]["stats"],
        "scale":               info["scale"],
        "resolution_m":        info.get("resolution_m", info["scale"]),
        "cadence":             cadence,
        "categorical":         info["categorical"],
        "normalize_histogram": info.get("normalize_histogram", False),
        "start_date":          date_start,
        "end_date":            dt_end.strftime("%Y-%m-%d"),
        "time_chunks":         get_time_chunks(date_start, date_end, cadence),
        "gee_weight":          info.get("gee_weight", 1),
        "tile_scale":          info.get("tile_scale", 1),
        "aoi_mode":            info.get("aoi_mode", "polygon"),
        "band_masks":      {b: content.get(b, {}).get("qa_mask") for b in bands},
        "band_transforms": {b: content.get(b, {}).get("band_transform") for b in bands},
        "band_computes":   {b: content.get(b, {}).get("band_compute") for b in bands},
    }


def build_payload(start: str, end: str, products: list[str], shp: str, app_dir: str,
                  run_id: str, gee_concurrency: int = 10, id_column: str = "name") -> dict:
    """start/end are YYYY-MM, inclusive."""
    date_start = f"{start}-01"
    date_end   = f"{end}-01"
    app_dir = pathlib.PurePosixPath(app_dir)
    shp = pathlib.PurePosixPath(shp)
    return {
        "run_id":          run_id,
        "shp_path":        str(shp),
        "products":        {p: product_task(p, date_start, date_end) for p in products},
        "output_dir":      str(app_dir / "data" / "runs" / run_id / "results"),
        "aoi_name":        shp.name,
        "app_dir":         str(app_dir),
        "gee_concurrency": max(1, gee_concurrency),
        "id_column":       id_column,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, metavar="YYYY-MM", help="First month (inclusive).")
    parser.add_argument("--end", required=True, metavar="YYYY-MM", help="Last month (inclusive).")
    parser.add_argument("--products", nargs="+", default=MONTHLY_PRODUCTS, choices=list(SELECTION),
                        help=f"Default: {' '.join(MONTHLY_PRODUCTS)}.")
    parser.add_argument("--shp", required=True, help="AOI file (regions.parquet).")
    parser.add_argument("--app-dir", required=True,
                        help="EZGEE APP_DIR: outputs go to <app-dir>/data/runs/<run-id>/.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gee-concurrency", type=int, default=10)
    parser.add_argument("--id-column", default="name")
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    payload = build_payload(args.start, args.end, args.products, args.shp, args.app_dir,
                            args.run_id, args.gee_concurrency, args.id_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(f"Wrote {args.output}: {', '.join(args.products)} {args.start} → {args.end}")


if __name__ == "__main__":
    main()
