"""Province × month environmental features from EZGEE zonal-statistics product files.

One definition of the monthly aggregation, shared by the 2011–2018 training merge
(merge_data.py) and the inference-side env table, so every batch lands in the same schema:

    admin, name, Date (month start), <env columns of SEA_dengue_env_monthly_2011-2018.csv>

Inputs are the per-product parquets EZGEE writes (one row per region per image/composite);
several batches of the same product can be concatenated before calling build_monthly_env.
"""
import json

import pandas as pd

JOIN_KEYS = ["admin", "name", "year_month"]

LST_COLS  = ["LST_Day_1km_mean", "LST_Night_1km_mean"]
VEG_COLS  = ["NDVI_mean", "EVI_mean"]
POP_COLS  = ["population_sum"]
LULC_BAND = "LC_Type1"


def read_product(path) -> pd.DataFrame:
    """Read an EZGEE product parquet without its geometry column (the bulk of the file).

    Keeps the keys plus the columns the aggregation can use: band statistics ending in
    _sum/_mean, and the land-cover histogram.
    """
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    keep = ["admin", "name", "Date"] + [
        c for c in names if c.endswith(("_sum", "_mean", "_histogram"))]
    return pd.read_parquet(path, columns=[c for c in names if c in keep])


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    return df


def expand_annual_to_monthly(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Repeat annual rows for each month of the year."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["Date"]).dt.year
    df = df.merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    df["year_month"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df[["admin", "name", "year_month"] + value_cols]


def histogram_to_pct(hist) -> dict:
    """Convert a pixel-count histogram dict/string to fractional percentages."""
    if isinstance(hist, str):
        hist = json.loads(hist)
    if not isinstance(hist, dict) or not hist or sum(hist.values()) == 0:
        return {}
    total = sum(hist.values())
    return {int(k): v / total for k, v in hist.items()}


def era5_monthly(era5: pd.DataFrame) -> pd.DataFrame:
    era5 = to_monthly(era5)
    cols = [c for c in era5.columns if c.endswith(("_sum", "_mean"))]
    return era5.groupby(JOIN_KEYS)[cols].mean().reset_index()


def chirps_monthly(chirps: pd.DataFrame) -> pd.DataFrame:
    return to_monthly(chirps).groupby(JOIN_KEYS)["precipitation_sum"].mean().reset_index()


def lst_monthly(lst: pd.DataFrame) -> pd.DataFrame:
    return to_monthly(lst).groupby(JOIN_KEYS)[LST_COLS].mean().reset_index()


def veg_monthly(ndvi_evi: pd.DataFrame) -> pd.DataFrame:
    return to_monthly(ndvi_evi).groupby(JOIN_KEYS)[VEG_COLS].mean().reset_index()


def lulc_annual(lulc: pd.DataFrame) -> pd.DataFrame:
    """One row per region-year: admin, name, Date, LC_Type1_pct_class<k>."""
    hist = lulc[f"{LULC_BAND}_histogram"].apply(histogram_to_pct).apply(pd.Series)
    hist = hist[sorted(hist.columns)]
    hist.columns = [f"{LULC_BAND}_pct_class{c}" for c in hist.columns]
    hist = hist.fillna(0)
    return pd.concat([lulc[["admin", "name", "Date"]].reset_index(drop=True),
                      hist.reset_index(drop=True)], axis=1)


def _annual_carried_forward(annual: pd.DataFrame, value_cols: list[str], months: pd.Series,
                            year_col: str) -> pd.DataFrame:
    """Monthly rows for `months`, each taking the latest available year ≤ its own year.

    Months before the first available year are left out (no backward fill). `year_col`
    records which year was used, so carried-forward values stay visible in the table.
    """
    annual = annual.copy()
    annual[year_col] = pd.to_datetime(annual["Date"]).dt.year
    annual = annual.drop(columns="Date").sort_values(year_col)
    grid = (annual[["admin", "name"]].drop_duplicates()
            .merge(pd.DataFrame({"year_month": sorted(months.unique())}), how="cross"))
    grid["_year"] = grid["year_month"].dt.year
    out = pd.merge_asof(grid.sort_values("_year"), annual,
                        left_on="_year", right_on=year_col, by=["admin", "name"],
                        direction="backward")
    out = out.dropna(subset=[year_col])
    out[year_col] = out[year_col].astype(int)
    return out[JOIN_KEYS + value_cols + [year_col]]


def build_monthly_env(era5: pd.DataFrame, chirps: pd.DataFrame, lst: pd.DataFrame,
                      ndvi_evi: pd.DataFrame, lulc: pd.DataFrame, pop: pd.DataFrame,
                      carry_forward_annual: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Aggregate EZGEE product rows to one row per (admin, name, month).

    carry_forward_annual=False reproduces the training merge exactly: annual products
    (MODIS land cover, WorldPop) are repeated over the months of their own year and the
    products are outer-merged. With True, months past the last published year reuse the
    latest year, recorded in `lulc_year` / `pop_year`; the monthly products then define
    which months appear.
    """
    log = print if verbose else (lambda *a, **k: None)

    monthly = [("CHIRPS", chirps_monthly(chirps)), ("ERA5_LAND", era5_monthly(era5)),
               ("MODIS_LST", lst_monthly(lst)), ("MODIS_NDVI_EVI", veg_monthly(ndvi_evi))]
    lulc_y = lulc_annual(lulc)
    lulc_cols = [c for c in lulc_y.columns if c not in ("admin", "name", "Date")]

    if carry_forward_annual:
        months = pd.concat([df["year_month"] for _, df in monthly])
        annual = [("MODIS_LULC", _annual_carried_forward(lulc_y, lulc_cols, months, "lulc_year")),
                  ("WorldPop",   _annual_carried_forward(pop, POP_COLS, months, "pop_year"))]
        how_annual = "left"
    else:
        annual = [("MODIS_LULC", expand_annual_to_monthly(lulc_y, lulc_cols)),
                  ("WorldPop",   expand_annual_to_monthly(pop, POP_COLS))]
        how_annual = "outer"

    for name, df in monthly + annual:
        log(f"  {name} shape: {df.shape}")

    env = monthly[0][1]
    for name, df in monthly[1:]:
        env = env.merge(df, on=JOIN_KEYS, how="outer")
        log(f"  after {name}: {env.shape}")
    for name, df in annual:
        env = env.merge(df, on=JOIN_KEYS, how=how_annual)
        log(f"  after {name}: {env.shape}")

    env = env.rename(columns={"year_month": "Date"})
    id_cols = ["admin", "name", "Date"]
    return env[id_cols + [c for c in env.columns if c not in id_cols]]
