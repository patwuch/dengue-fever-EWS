"""Monthly inference workflow: EZGEE zonal statistics → province × month env table → predictions.

Launched from the repo root, so the job pods receive site/scripts, the machine-learning-module
sources (env_features.py) and the zonal-statistics-module submodule:

    snakemake data/processed/env/SEA_env_monthly.csv --profile profiles/k8s-inference

Training and bundle builds stay in machine-learning-module/Snakefile.
"""
from pathlib import Path

# Relative by default so paths map to the same S3 keys on the controller and in the job pod.
DATA_ROOT = Path(config.get("data_root", "."))
ZS_DIR    = DATA_ROOT / "data" / "processed" / "zonal-statistics"
ENV_DIR   = DATA_ROOT / "data" / "processed" / "env"

# The 2019-01 → 2026-05 EZGEE batch ("lostyears"); monthly runs are stacked on top of it.
BASE = {
    "era5":     ZS_DIR / "ERA5_LAND_2019-01-01_to_2026-05-31.parquet",
    "chirps":   ZS_DIR / "CHIRPS_2019-01-01_to_2026-05-31.parquet",
    "lst":      ZS_DIR / "MODIS_LST_2019-01-01_to_2026-05-31.parquet",
    "ndvi_evi": ZS_DIR / "MODIS_NDVI_EVI_2019-01-01_to_2026-05-31.parquet",
    "lulc":     ZS_DIR / "MODIS_LULC_2019-01-01_to_2023-12-31.parquet",
    "worldpop": ZS_DIR / "WorldPop_2019-01-01_to_2020-12-31.parquet",
}

# Monthly EZGEE runs to stack on the batches, oldest first, as YYYY-MM run months. Each run also
# re-fetches the month before it, and a later run replaces every month it covers.
EZGEE_MONTHS     = config.get("ezgee_months", [])
# Successor products since 2026-09 (climate model v2). Months up to 2026-08 come from the
# zonal_batch files below.
MONTHLY_PRODUCTS = {"era5": "ERA5_LAND", "chirps": "CHIRPS_V3", "lst_day": "VIIRS_LST_DAY",
                    "lst_night": "VIIRS_LST_NIGHT", "vi": "VIIRS_NDVI_EVI"}
# The one monthly run on the retired products (v1 env table only).
MONTHLY_V1_MONTHS   = ["2026-06"]
MONTHLY_V1_PRODUCTS = {"era5": "ERA5_LAND", "chirps": "CHIRPS", "lst": "MODIS_LST",
                       "ndvi_evi": "MODIS_NDVI_EVI"}


# The 165 regions the base batch was extracted for (EZGEE's lostyears/inputs copy). The models'
# 162 nodes are a subset; the repo's dengue-infection GAUL file lacks three Philippine regions,
# which would then drop out of every month a monthly run replaces.
AOI = ZS_DIR / "aoi" / "gaul_2024_sea_filtered.parquet"


def shift_month(month, delta):
    year, mon = map(int, month.split("-"))
    idx = year * 12 + mon - 1 + delta
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def batches(key):
    """v1 env table: base file plus the 2026-06 monthly run, oldest first."""
    runs = [ZS_DIR / "monthly" / m / f"{MONTHLY_V1_PRODUCTS[key]}.parquet"
            for m in MONTHLY_V1_MONTHS] if key in MONTHLY_V1_PRODUCTS else []
    return [str(p) for p in [BASE[key]] + runs]


rule zonal_stats:
    message:
        "EZGEE zonal statistics for {wildcards.month} and the month before..."
    input:
        aoi = str(AOI),
    output:
        **{key: str(ZS_DIR / "monthly" / "{month}" / f"{product}.parquet")
           for key, product in MONTHLY_PRODUCTS.items()},
    wildcard_constraints:
        month = r"\d{4}-\d{2}",
    params:
        start           = lambda w: shift_month(w.month, -1),
        end             = lambda w: w.month,
        products        = MONTHLY_PRODUCTS,
        run_id          = lambda w: f"m{w.month}",
        gee_concurrency = config.get("gee_concurrency", 10),
    # Mostly waiting on Earth Engine; EZGEE runs up to gee_concurrency requests at once.
    threads: 4
    resources:
        mem_mb = 8000,
    script:
        "site/scripts/run_ezgee.py"


# One-off batches of the successor products (EZGEE registry), one Job per product-year so a
# failure only reruns that year. Fetch everything with:  snakemake batches --profile ...
BATCH_FIRST = {"CHIRPS_V3": 2011, "VIIRS_LST_DAY": 2012, "VIIRS_LST_NIGHT": 2012,
               "VIIRS_NDVI_EVI": 2012,
               # Population-weighted rainfall, a candidate replacement for the area mean.
               "CHIRPS_V3_POPW": 2011,
               # ERA5 is unchanged; 2026 fills June–August between the base batch and the
               # first successor-product monthly run.
               "ERA5_LAND": 2026}
BATCH_END   = config.get("batch_end", "2026-08")      # last month of the final year
BATCH_DIR   = ZS_DIR / "batch"


def batch_months(year):
    """First and last month of one batch year; the final year stops at BATCH_END."""
    last = BATCH_END if int(year) == int(BATCH_END[:4]) else f"{year}-12"
    return f"{year}-01", last


rule batches:
    input:
        [str(BATCH_DIR / product / f"{year}.parquet")
         for product, first in BATCH_FIRST.items()
         for year in range(first, int(BATCH_END[:4]) + 1)],


rule zonal_batch:
    message:
        "EZGEE batch {wildcards.product} {wildcards.year}..."
    input:
        aoi = str(AOI),
    output:
        out = str(BATCH_DIR / "{product}" / "{year}.parquet"),
    wildcard_constraints:
        product = "|".join(BATCH_FIRST),
        year    = r"\d{4}",
    params:
        start           = lambda w: batch_months(w.year)[0],
        end             = lambda w: batch_months(w.year)[1],
        products        = lambda w: {"out": w.product},
        run_id          = lambda w: f"b{w.product}_{w.year}",
        gee_concurrency = config.get("gee_concurrency", 10),
    threads: 4
    resources:
        mem_mb = 8000,
    script:
        "site/scripts/run_ezgee.py"


rule env_table:
    message:
        "Building the province × month env table (v1: MODIS and CHIRPS v2, to 2026-06)..."
    input:
        **{key: batches(key) for key in BASE},
    output:
        table = str(ENV_DIR / "SEA_env_monthly.csv"),
    resources:
        mem_mb = 8000,
    script:
        "site/scripts/build_env_table.py"


# v2: successor products (CHIRPS v3, VIIRS) under product-neutral names, for climate model v2.
# Each product: its yearly batches from 2019 (ERA5: the base file plus 2026), then monthly runs.
V2_FROM = 2019


def batches_v2(key):
    product = MONTHLY_PRODUCTS[key]
    if product == "ERA5_LAND":
        files = [BASE["era5"], BATCH_DIR / "ERA5_LAND" / "2026.parquet"]
    else:
        files = [BATCH_DIR / product / f"{y}.parquet"
                 for y in range(V2_FROM, int(BATCH_END[:4]) + 1)]
    files += [ZS_DIR / "monthly" / m / f"{product}.parquet" for m in EZGEE_MONTHS]
    return [str(f) for f in files]


rule env_table_v2:
    message:
        "Building the province × month env table (v2: CHIRPS v3 and VIIRS)..."
    input:
        **{key: batches_v2(key) for key in MONTHLY_PRODUCTS},
        lulc     = str(BASE["lulc"]),
        worldpop = str(BASE["worldpop"]),
    output:
        table = str(ENV_DIR / "SEA_env_monthly_v2.csv"),
    resources:
        mem_mb = 8000,
    script:
        "site/scripts/build_env_table_v2.py"


# Climate-mode bundle from machine-learning-module/Snakefile (on S3 under the same key); point
# bundle_climate at a downloaded release asset to run without the cluster.
BUNDLE_CLIMATE = config.get("bundle_climate",
                            "results/STGNN/production_climate_risk/inference_bundle_climate.tar.gz")
SITE_DATA      = DATA_ROOT / "site" / "data"


rule climate_deltas:
    message:
        "Climate deltas {params.start} → {params.end} vs the climate model's 2011–2018 baseline..."
    input:
        env    = rules.env_table.output.table,
        bundle = BUNDLE_CLIMATE,
    output:
        deltas = str(SITE_DATA / "climate_deltas.json"),
    params:
        # End at a December, before any month being forecast (compute_climate_delta.py).
        start = config.get("delta_start", "2019-01"),
        end   = config.get("delta_end", "2025-12"),
    shell:
        "python site/scripts/compute_climate_delta.py --env {input.env} --bundle {input.bundle}"
        " --start {params.start} --end {params.end} -o {output.deltas}"


rule predict_climate:
    message:
        "Climate-model risk index for {wildcards.target}..."
    input:
        env    = rules.env_table.output.table,
        bundle = BUNDLE_CLIMATE,
        deltas = rules.climate_deltas.output.deltas,
    output:
        prediction = str(SITE_DATA / "predictions" / "climate" / "{target}.json"),
    wildcard_constraints:
        target = r"\d{4}-\d{2}",
    params:
        last_month = lambda w: shift_month(w.target, -1),
    resources:
        mem_mb = 4000,
    shell:
        "python site/scripts/run_inference.py --mode climate --bundle {input.bundle}"
        " --env {input.env} --deltas {input.deltas} --last-month {params.last_month}"
        " --out {output.prediction}"


rule region_shapes:
    message:
        "Simplified province outlines for the site map..."
    input:
        aoi = str(AOI),
    output:
        shapes = str(SITE_DATA / "regions.json"),
    params:
        tolerance = 0.01,
        grid      = 0.001,
    script:
        "site/scripts/make_region_shapes.py"


# Climate model v2 (successor products, two outputs). Deltas against its own baseline; one
# prediction file per run, named by its first target month, holding both leads.
BUNDLE_CLIMATE_V2 = config.get(
    "bundle_climate_v2", "results/STGNN/climate_v2_production/inference_bundle_climate_v2.tar.gz")


rule climate_deltas_v2:
    message:
        "Climate deltas {params.start} → {params.end} vs climate model v2's 2011–2018 baseline..."
    input:
        env    = rules.env_table_v2.output.table,
        bundle = BUNDLE_CLIMATE_V2,
    output:
        deltas = str(SITE_DATA / "climate_deltas_v2.json"),
    params:
        start = config.get("delta_start", "2019-01"),
        end   = config.get("delta_end", "2025-12"),
    shell:
        "python site/scripts/compute_climate_delta.py --env {input.env} --bundle {input.bundle}"
        " --start {params.start} --end {params.end} -o {output.deltas}"


rule predict_climate_v2:
    message:
        "Climate model v2: risk index for {wildcards.target} and the month after..."
    input:
        env    = rules.env_table_v2.output.table,
        bundle = BUNDLE_CLIMATE_V2,
        deltas = rules.climate_deltas_v2.output.deltas,
    output:
        prediction = str(SITE_DATA / "predictions" / "climate_v2" / "{target}.json"),
    wildcard_constraints:
        target = r"\d{4}-\d{2}",
    params:
        last_month = lambda w: shift_month(w.target, -1),
    resources:
        mem_mb = 4000,
    shell:
        "python site/scripts/run_inference.py --mode climate --label climate_v2"
        " --bundle {input.bundle} --env {input.env} --deltas {input.deltas}"
        " --last-month {params.last_month} --out {output.prediction}"
