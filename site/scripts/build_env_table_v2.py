"""Province × month env table on the successor products, for climate model v2 (Snakemake script).

CHIRPS v3 precipitation and VIIRS LST / NDVI / EVI with ERA5-Land and MODIS land cover, under
the product-neutral names of the v2 training table (successor_training_table.py):

    precipitation_sum   CHIRPS v3          lst_day_mean / lst_night_mean   VIIRS VNP21A1D / N
    ndvi_mean, evi_mean VIIRS VNP13A1      ERA5-Land and LC_Type1_pct_class* as in v1

Aggregated by env_features.build_monthly_env, as the training data was. Each product's batches
are stacked oldest first, a later batch replacing the months it covers; annual products are
carried forward past their last year (lulc_year, pop_year).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "machine-learning-module" / "src" / "OpenDengue"))
from env_features import build_monthly_env, read_product  # noqa: E402

RENAME = {"LST_Day_1km_mean": "lst_day_mean", "LST_Night_1km_mean": "lst_night_mean",
          "NDVI_mean": "ndvi_mean", "EVI_mean": "evi_mean"}


def stack(paths) -> pd.DataFrame:
    """Concatenate batches oldest first; a later batch replaces every month it covers."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    out = None
    for path in paths:
        df = read_product(path)
        if out is not None:
            months = set(pd.to_datetime(df["Date"]).dt.to_period("M"))
            out = out[~pd.to_datetime(out["Date"]).dt.to_period("M").isin(months)]
            df = pd.concat([out, df], ignore_index=True)
        out = df
    print(f"  {Path(paths[0]).parent.name or Path(paths[0]).name}: {len(paths)} file(s), {len(out)} rows")
    return out


inp = snakemake.input                                                   # noqa: F821
day = stack(inp.lst_day).rename(columns={"LST_1KM_mean": "LST_Day_1km_mean"})
night = stack(inp.lst_night).rename(columns={"LST_1KM_mean": "LST_Night_1km_mean"})
keys = ["admin", "name", "Date"]
lst = day[keys + ["LST_Day_1km_mean"]].merge(night[keys + ["LST_Night_1km_mean"]], on=keys, how="outer")

env = build_monthly_env(
    era5=stack(inp.era5), chirps=stack(inp.chirps), lst=lst, ndvi_evi=stack(inp.vi),
    lulc=stack(inp.lulc), pop=stack(inp.worldpop), carry_forward_annual=True,
)
env = env.rename(columns={"admin": "adm_0_name", "name": "adm_1_name", "Date": "year_month", **RENAME})
env = env.sort_values(["adm_0_name", "adm_1_name", "year_month"]).reset_index(drop=True)

print(f"\n{env[['adm_0_name', 'adm_1_name']].drop_duplicates().shape[0]} regions, "
      f"{env['year_month'].min():%Y-%m} → {env['year_month'].max():%Y-%m}, {env.shape}")
out = Path(snakemake.output.table)                                      # noqa: F821
out.parent.mkdir(parents=True, exist_ok=True)
env.to_csv(out, index=False)
print(f"Saved: {out}")
