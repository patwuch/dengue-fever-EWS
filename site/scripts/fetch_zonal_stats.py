"""
Fetch monthly zonal statistics for SEA admin-1 regions from Google Earth Engine.

Reuses gee_ops.py and products.py from the zonal-statistics-module so QA masks,
band transforms, collection IDs, and reducers stay in sync with the offline app.

Each month is archived as site/data/zonal_stats/YYYY-MM.json. The STGNN reads a
window of past months (window_size in best_params.json, up to 12), so
--backfill N makes sure the N months ending at the target month are all
archived, fetching only the ones that are missing. The first run backfills
the whole window; later runs fetch one new month.

Aggregation mirrors how the training features were built (offline app +
machine-learning-module/src/OpenDengue/merge_data.py): each daily/composite
image is reduced spatially, then the month's values are averaged. Spatial and
temporal means commute, so this script averages the month first and reduces
once per product.

Outputs:
    site/data/zonal_stats/<YYYY-MM>.json   one file per month
    site/data/latest_zonal_stats.json      copy of the target month

Region geometry, one of:
    GEE_REGIONS_ASSET env var / --regions-asset   an Earth Engine table asset
                                                  (recommended: no upload per call)
    --regions <path>                              local GeoParquet with admin/name
                                                  columns (gaul_2024_sea_filtered.parquet),
                                                  simplified before upload

Required env var (set as GitHub Actions secret):
    GEE_SERVICE_ACCOUNT  — full contents of a GEE service account key JSON

Local usage (with gcloud ADC):
    earthengine authenticate
    python fetch_zonal_stats.py --month 2026-05 --backfill 12
"""

import argparse
import datetime
import json
import os
import pathlib
import shutil
import sys
import tempfile

import ee

ROOT        = pathlib.Path(__file__).parent.parent.parent
ZSM_ROOT    = ROOT / "zonal-statistics-module"          # EZGEE submodule
sys.path.insert(0, str(ZSM_ROOT / "workflow"))

from gee_ops import apply_qa_mask                            # noqa: E402
from products import PRODUCT_REGISTRY                        # noqa: E402

DEFAULT_REGIONS = ROOT / "data/processed/dengue-infection/geoparquet/gaul_2024_sea_filtered.parquet"
DATA_DIR        = pathlib.Path(__file__).parent.parent / "data"
ARCHIVE_DIR     = DATA_DIR / "zonal_stats"
LATEST_PATH     = DATA_DIR / "latest_zonal_stats.json"

# Column names written by dengue-infection-module/src/OpenDengue/filter_shp_convert_parquet.py.
# "name" matches the STGNN node names (adm_1_name in the training CSV).
ADM1_COL = "name"
ADM0_COL = "admin"

# Simplification tolerance (degrees, ~500 m) for local geometry, keeping the
# client-side FeatureCollection under Earth Engine's request-size limit.
SIMPLIFY_TOL_DEG = 0.005

# ── ERA5-Land: STGNN feature name → daily band (spatial mean, monthly mean) ──
ERA5_FEATURES = {
    "temperature_2m_mean":                                            "temperature_2m",
    "temperature_2m_max_mean":                                        "temperature_2m_max",
    "temperature_2m_min_mean":                                        "temperature_2m_min",
    "potential_evaporation_sum_mean":                                 "potential_evaporation_sum",
    "total_evaporation_sum_mean":                                     "total_evaporation_sum",
    "evaporation_from_bare_soil_sum_mean":                            "evaporation_from_bare_soil_sum",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_mean": "evaporation_from_open_water_surfaces_excluding_oceans_sum",
    "evaporation_from_the_top_of_canopy_sum_mean":                    "evaporation_from_the_top_of_canopy_sum",
    "evaporation_from_vegetation_transpiration_sum_mean":             "evaporation_from_vegetation_transpiration_sum",
}

# ── MODIS composites: STGNN feature name → (product, band) ──────────────────
MODIS_FEATURES = {
    "LST_Day_1km_mean":   ("MODIS_LST",      "LST_Day_1km"),
    "LST_Night_1km_mean": ("MODIS_LST",      "LST_Night_1km"),
    "NDVI_mean":          ("MODIS_NDVI_EVI", "NDVI"),
    "EVI_mean":           ("MODIS_NDVI_EVI", "EVI"),
}

# precipitation_sum in training is CHIRPS daily "precipitation" reduced with a
# spatial SUM per day ({band}_{stat} naming in the offline app), then averaged
# over the month in merge_data.py — not ERA5 total precipitation.
PRECIP_FEATURE = "precipitation_sum"

# MODIS LULC is annual — appended as LC_Type1_pct_class{1-17}
LULC_PRODUCT   = "MODIS_LULC"
LULC_BAND      = "LC_Type1"
LULC_N_CLASSES = 17
LULC_YEAR      = int(PRODUCT_REGISTRY[LULC_PRODUCT]["max_date"][:4])

TILE_SCALE = 4


# ---------------------------------------------------------------------------
# Authentication & regions
# ---------------------------------------------------------------------------

def authenticate() -> None:
    sa_json = os.environ.get("GEE_SERVICE_ACCOUNT")
    if sa_json:
        key = json.loads(sa_json)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(key, f)
            key_path = f.name
        credentials = ee.ServiceAccountCredentials(key["client_email"], key_path)
        ee.Initialize(credentials)
    else:
        ee.Authenticate()
        ee.Initialize()


def load_regions(regions_path: pathlib.Path | None, regions_asset: str | None) -> tuple[ee.FeatureCollection, dict]:
    """Return (FeatureCollection, {name: country}) for the admin-1 regions."""
    if regions_asset:
        fc = ee.FeatureCollection(regions_asset).select([ADM0_COL, ADM1_COL])
        rows = fc.reduceColumns(ee.Reducer.toList(2), [ADM1_COL, ADM0_COL]).get("list").getInfo()
        names = {name: country for name, country in rows}
        n_rows = len(rows)
    else:
        import geopandas as gpd
        gdf = gpd.read_parquet(regions_path).to_crs("EPSG:4326")
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
        gdf = gdf[[ADM0_COL, ADM1_COL, "geometry"]]
        fc = ee.FeatureCollection(json.loads(gdf.to_json()))
        names = dict(zip(gdf[ADM1_COL], gdf[ADM0_COL]))
        n_rows = len(gdf)

    # The STGNN node index is keyed by admin-1 name alone, so names must be unique.
    if len(names) != n_rows:
        raise ValueError("Duplicate admin-1 names in the region table; STGNN nodes are keyed by name.")
    return fc, names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_bounds(month: str) -> tuple[str, str]:
    dt = datetime.datetime.strptime(month, "%Y-%m")
    start = dt.strftime("%Y-%m-01")
    end = f"{dt.year + 1}-01-01" if dt.month == 12 else f"{dt.year}-{dt.month + 1:02d}-01"
    return start, end


def shift_month(month: str, delta: int) -> str:
    dt = datetime.datetime.strptime(month, "%Y-%m")
    idx = dt.year * 12 + (dt.month - 1) + delta
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def default_target_month() -> str:
    """Two months ago — most recent month with complete ERA5-Land/CHIRPS/MODIS data."""
    return shift_month(datetime.date.today().strftime("%Y-%m"), -2)


REDUCERS = {"mean": ee.Reducer.mean, "sum": ee.Reducer.sum}


def reduce_to_dict(image: ee.Image, fc: ee.FeatureCollection, reducer: str,
                   scale: float, bands: list[str]) -> dict[str, dict[str, float]]:
    """reduceRegions an (optionally multi-band) image → {region: {band: value}}."""
    result = image.select(bands).reduceRegions(
        collection=fc, reducer=REDUCERS[reducer](), scale=scale, tileScale=TILE_SCALE,
    ).getInfo()["features"]
    out = {}
    for f in result:
        props = f["properties"]
        # A single-band image names the output property after the reducer, not the band.
        out[props.get(ADM1_COL, "")] = (
            {bands[0]: props.get(reducer)} if len(bands) == 1 else {b: props.get(b) for b in bands}
        )
    return out


# ---------------------------------------------------------------------------
# Per-product fetchers
# ---------------------------------------------------------------------------

def fetch_era5(start: str, end: str, fc: ee.FeatureCollection) -> dict[str, dict[str, float]]:
    info  = PRODUCT_REGISTRY["ERA5_LAND"]
    bands = list(ERA5_FEATURES.values())
    img   = ee.ImageCollection(info["ee_collection"]).filterDate(start, end).select(bands).mean()

    # Kelvin → °C etc., per products.py
    for band in bands:
        transform = info["content"][band].get("band_transform")
        if transform:
            img = img.addBands(
                img.select(band).multiply(transform["scale"]).add(transform["offset"]).rename(band),
                overwrite=True,
            )

    raw = reduce_to_dict(img, fc, "mean", info["scale"], bands)
    return {region: {feat: vals.get(band) for feat, band in ERA5_FEATURES.items()} for region, vals in raw.items()}


def fetch_chirps(start: str, end: str, fc: ee.FeatureCollection) -> dict[str, dict[str, float]]:
    info = PRODUCT_REGISTRY["CHIRPS"]
    img  = ee.ImageCollection(info["ee_collection"]).filterDate(start, end).select("precipitation").mean()
    raw  = reduce_to_dict(img, fc, "sum", info["scale"], ["precipitation"])
    return {region: {PRECIP_FEATURE: vals["precipitation"]} for region, vals in raw.items()}


def fetch_modis(start: str, end: str, fc: ee.FeatureCollection) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for feat, (product, band) in MODIS_FEATURES.items():
        info      = PRODUCT_REGISTRY[product]
        band_cfg  = info["content"][band]
        transform = band_cfg.get("band_transform")
        qa_cfg    = band_cfg.get("qa_mask")

        def process(img, band=band, transform=transform, qa_cfg=qa_cfg):
            img = apply_qa_mask(img, qa_cfg) if qa_cfg else img
            img = img.select([band])
            if transform:
                img = img.multiply(transform["scale"]).add(transform["offset"]).rename([band])
            return img

        img = ee.ImageCollection(info["ee_collection"]).filterDate(start, end).map(process).mean()
        print(f"    {product} / {band}")
        for region, vals in reduce_to_dict(img, fc, "mean", info["scale"], [band]).items():
            out.setdefault(region, {})[feat] = vals[band]
    return out


def fetch_lulc(fc: ee.FeatureCollection, year: int = LULC_YEAR) -> dict[str, dict[str, float]]:
    """MODIS LULC histogram for one year → {region: {"LC_Type1_pct_class{k}": fraction}}."""
    info  = PRODUCT_REGISTRY[LULC_PRODUCT]
    image = ee.ImageCollection(info["ee_collection"]).filterDate(f"{year}-01-01", f"{year}-12-31").first()

    result = image.select([LULC_BAND]).reduceRegions(
        collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=info["scale"], tileScale=TILE_SCALE,
    ).getInfo()["features"]

    region_lulc: dict[str, dict[str, float]] = {}
    for feat in result:
        props = feat["properties"]
        hist  = props.get("histogram") or {}
        total = sum(hist.values())
        region_lulc[props.get(ADM1_COL, "")] = {
            f"LC_Type1_pct_class{cls}": (hist.get(str(cls), 0) / total if total else 0.0)
            for cls in range(1, LULC_N_CLASSES + 1)
        }
    return region_lulc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_month(month: str, fc: ee.FeatureCollection, names: dict, lulc: dict) -> dict:
    start, end = month_bounds(month)
    print(f"Fetching {month}  ({start} → {end})")

    regions: dict[str, dict] = {name: {"country": country} for name, country in names.items()}

    print("  ERA5_LAND ...")
    parts = [fetch_era5(start, end, fc)]
    print("  CHIRPS ...")
    parts.append(fetch_chirps(start, end, fc))
    print("  MODIS ...")
    parts.append(fetch_modis(start, end, fc))
    parts.append(lulc)

    for part in parts:
        for region, vals in part.items():
            if region in regions:
                regions[region].update(vals)

    # NaN → None for JSON
    for vals in regions.values():
        for k, v in vals.items():
            if isinstance(v, float) and v != v:
                vals[k] = None

    return {
        "target_month":   month,
        "fetched_at":     datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_regions":      len(regions),
        "lulc_year":      LULC_YEAR,
        "bias_corrected": False,
        "regions":        regions,
    }


def main(target_month: str | None, backfill: int, refresh: bool,
         regions_path: pathlib.Path | None, regions_asset: str | None) -> None:
    target_month = target_month or default_target_month()
    months = [shift_month(target_month, -k) for k in reversed(range(max(backfill, 1)))]
    todo = [m for m in months if refresh or not (ARCHIVE_DIR / f"{m}.json").exists()]
    print(f"Target {target_month}; window {months[0]} → {months[-1]}; fetching {len(todo)} month(s)")

    if todo:
        authenticate()
        print("Loading region boundaries ...")
        fc, names = load_regions(regions_path, regions_asset)
        print(f"  MODIS_LULC / {LULC_BAND} (year {LULC_YEAR}) ...")
        lulc = fetch_lulc(fc)

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        for month in todo:
            stats = fetch_month(month, fc, names, lulc)
            (ARCHIVE_DIR / f"{month}.json").write_text(json.dumps(stats, indent=2))
            print(f"  saved → {ARCHIVE_DIR / f'{month}.json'}")

    shutil.copyfile(ARCHIVE_DIR / f"{target_month}.json", LATEST_PATH)
    print(f"\nLatest → {LATEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=None,
                        help="YYYY-MM to fetch. Defaults to two months ago (data availability lag).")
    parser.add_argument("--backfill", type=int, default=1,
                        help="Ensure this many months ending at --month are archived (use the model window size).")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch months that are already archived.")
    parser.add_argument("--regions", type=pathlib.Path, default=DEFAULT_REGIONS,
                        help="Local GeoParquet of admin-1 regions (admin/name columns).")
    parser.add_argument("--regions-asset", default=os.environ.get("GEE_REGIONS_ASSET"),
                        help="Earth Engine table asset with admin/name columns; overrides --regions.")
    args = parser.parse_args()
    main(args.month, args.backfill, args.refresh, args.regions, args.regions_asset)
