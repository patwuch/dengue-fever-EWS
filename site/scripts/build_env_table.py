"""Province × month env table from the EZGEE base batch plus any monthly runs (Snakemake script).

Same aggregation and column names as SEA_dengue_env_monthly_2011-2018.csv (env_features.py), keyed
by adm_0_name / adm_1_name / year_month so it joins onto the training data. Annual products are
carried forward past their last published year; lulc_year and pop_year record the year used.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "machine-learning-module" / "src" / "OpenDengue"))
from env_features import build_monthly_env, read_product  # noqa: E402


def stack(paths) -> pd.DataFrame:
    """Concatenate batches oldest first; a later batch replaces every month it covers."""
    out = None
    for path in paths:
        df = read_product(path)
        if out is not None:
            months = set(pd.to_datetime(df["Date"]).dt.to_period("M"))
            out = out[~pd.to_datetime(out["Date"]).dt.to_period("M").isin(months)]
            df = pd.concat([out, df], ignore_index=True)
        out = df
    return out


# Snakemake input name → build_monthly_env argument ("pop" is reserved for input names).
ARGS = {"era5": "era5", "chirps": "chirps", "lst": "lst", "ndvi_evi": "ndvi_evi",
        "lulc": "lulc", "worldpop": "pop"}
products = {}
for key, arg in ARGS.items():
    paths = snakemake.input[key]
    paths = [paths] if isinstance(paths, str) else list(paths)
    products[arg] = stack(paths)
    print(f"{key}: {len(paths)} batch(es), {len(products[arg])} rows")

env = build_monthly_env(**products, carry_forward_annual=True)
env = env.rename(columns={"admin": "adm_0_name", "name": "adm_1_name", "Date": "year_month"})
env = env.sort_values(["adm_0_name", "adm_1_name", "year_month"]).reset_index(drop=True)

print(f"\n{env[['adm_0_name', 'adm_1_name']].drop_duplicates().shape[0]} regions, "
      f"{env['year_month'].min():%Y-%m} → {env['year_month'].max():%Y-%m}, {env.shape}")
print(f"lulc_year ≤ {env['lulc_year'].max()}, pop_year ≤ {env['pop_year'].max()}")

out = Path(snakemake.output.table)
out.parent.mkdir(parents=True, exist_ok=True)
env.to_csv(out, index=False)
print(f"Saved: {out}")
