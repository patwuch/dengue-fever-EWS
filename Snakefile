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

# Monthly EZGEE runs to stack on the base, oldest first, as YYYY-MM run months. Each run also
# re-fetches the month before it, and a later run replaces every month it covers.
EZGEE_MONTHS     = config.get("ezgee_months", [])
MONTHLY_PRODUCTS = {"era5": "ERA5_LAND", "chirps": "CHIRPS", "lst": "MODIS_LST",
                    "ndvi_evi": "MODIS_NDVI_EVI"}


def batches(key):
    """Base file plus the monthly run files for one product, oldest first."""
    runs = [ZS_DIR / "monthly" / m / f"{MONTHLY_PRODUCTS[key]}.parquet"
            for m in EZGEE_MONTHS] if key in MONTHLY_PRODUCTS else []
    return [str(p) for p in [BASE[key]] + runs]


rule env_table:
    message:
        "Building the province × month env table..."
    input:
        **{key: batches(key) for key in BASE},
    output:
        table = str(ENV_DIR / "SEA_env_monthly.csv"),
    resources:
        mem_mb = 8000,
    script:
        "site/scripts/build_env_table.py"
