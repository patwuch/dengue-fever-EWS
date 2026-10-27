"""Training table on the successor products: CHIRPS v3 and VIIRS instead of CHIRPS v2 and MODIS.

Terra MODIS drifts from 2022 and ends around 2027; CHIRPS v2 ends after 2026. The climate model is
retrained on the products that carry on, so live inputs need no conversion. This rewrites the five
affected columns of the 2011–2018 training table (merge_data.py output) and keeps the rest:

    precipitation_sum                       CHIRPS v2 → CHIRPS v3 (covers 2011)
    precipitation_mm   (new)                CHIRPS v3 monthly total in mm, from the spatial mean
                                            (precipitation_sum scales with province area)
    precipitation_popw_mm (new)             the same, population-weighted (CHIRPS_V3_POPW,
                                            GPW v4.11 2015)
    LST_Day_1km_mean   → lst_day_mean       MODIS → VIIRS VNP21A1D
    LST_Night_1km_mean → lst_night_mean     MODIS → VIIRS VNP21A1N
    NDVI_mean          → ndvi_mean          MODIS → VIIRS VNP13A1
    EVI_mean           → evi_mean           MODIS → VIIRS VNP13A1

VIIRS starts in January 2012, so 2011 is converted from MODIS: each province-month's VIIRS
climatology plus fitted shares of that month's MODIS anomaly and, for LST, the ERA5 air-temperature
anomaly, all fitted on 2012–2021 (Terra's orbit is nominal until 2022). <feature>_source records
where each value came from. The coefficients and a leave-one-year-out check go to the conversion
JSON.

Monthly values are the mean of the product's images in the month (daily for CHIRPS and LST,
16-day composites for NDVI/EVI), as in env_features.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_features import monthly_rain_mm, read_product  # noqa: E402

KEYS = ["adm_0_name", "adm_1_name", "year_month"]
GROUP = ["adm_0_name", "adm_1_name", "mon"]

# new column: (snakemake input, product column, MODIS column in the old tables, uses ERA5 anomaly)
VIIRS = {
    "lst_day_mean":   ("viirs_lst_day",   "LST_1KM_mean", "LST_Day_1km_mean",   True),
    "lst_night_mean": ("viirs_lst_night", "LST_1KM_mean", "LST_Night_1km_mean", True),
    "ndvi_mean":      ("viirs_vi",        "NDVI_mean",    "NDVI_mean",          False),
    "evi_mean":       ("viirs_vi",        "EVI_mean",     "EVI_mean",           False),
}
REPLACED = ["precipitation_sum", "LST_Day_1km_mean", "LST_Night_1km_mean", "NDVI_mean", "EVI_mean"]
AIR = "temperature_2m_mean"


def monthly(paths, cols) -> pd.DataFrame:
    """Province × month mean of each column over a product's yearly batch files."""
    d = pd.concat([read_product(p) for p in paths], ignore_index=True)
    d["year_month"] = pd.to_datetime(d["Date"]).dt.to_period("M").dt.to_timestamp()
    d = d.groupby(["admin", "name", "year_month"])[cols].mean().reset_index()
    return d.rename(columns={"admin": "adm_0_name", "name": "adm_1_name"})


def with_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"])
    df["year"] = df["year_month"].dt.year
    df["mon"] = df["year_month"].dt.month
    return df


def fit(pairs: pd.DataFrame, use_air: bool) -> dict:
    """VIIRS climatology per province-month, plus anomaly coefficients by least squares."""
    clim = pairs.groupby(GROUP)[["new", "old", AIR]].mean().add_suffix("_clim")
    a = pairs.join(clim, on=GROUP)
    x = [a["old"] - a["old_clim"]] + ([a[AIR] - a[f"{AIR}_clim"]] if use_air else [])
    X = np.column_stack(x)
    y = (a["new"] - a["new_clim"]).to_numpy()
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    coef = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return {"clim": clim, "beta": float(coef[0]), "gamma": float(coef[1]) if use_air else 0.0}


def predict(model: dict, rows: pd.DataFrame) -> np.ndarray:
    """Converted value; a missing MODIS or ERA5 anomaly contributes nothing."""
    c = rows.join(model["clim"], on=GROUP)
    anom_old = (c["old"] - c["old_clim"]).fillna(0.0)
    anom_air = (c[AIR] - c[f"{AIR}_clim"]).fillna(0.0)
    return (c["new_clim"] + model["beta"] * anom_old + model["gamma"] * anom_air).to_numpy()


def loyo(pairs: pd.DataFrame, use_air: bool) -> dict:
    """Leave-one-year-out RMSE of the conversion, and of the VIIRS climatology alone."""
    conv, clim = [], []
    for year in sorted(pairs["year"].unique()):
        train, test = pairs[pairs["year"] != year], pairs[pairs["year"] == year]
        m = fit(train, use_air)
        conv.append(np.sqrt(np.nanmean((predict(m, test) - test["new"].to_numpy()) ** 2)))
        base = test.join(m["clim"], on=GROUP)["new_clim"].to_numpy()
        clim.append(np.sqrt(np.nanmean((base - test["new"].to_numpy()) ** 2)))
    return {"rmse_conversion": float(np.mean(conv)), "rmse_climatology_only": float(np.mean(clim)),
            "rmse_conversion_worst_year": float(np.max(conv))}


def main(inp, out, fit_years):
    base = pd.read_csv(inp["training"])
    base["year_month"] = pd.to_datetime(base["year_month"])
    modis = pd.concat([
        base[KEYS + [m for *_, m, _ in VIIRS.values()] + [AIR]],
        pd.read_csv(inp["env"], usecols=KEYS + [m for *_, m, _ in VIIRS.values()] + [AIR])
          .assign(year_month=lambda d: pd.to_datetime(d["year_month"])),
    ]).drop_duplicates(KEYS)
    modis = with_month(modis)

    table = base.drop(columns=REPLACED).copy()
    chirps = monthly_rain_mm(monthly(inp["chirps_v3"], ["precipitation_sum", "precipitation_mean"]))
    popw = monthly_rain_mm(monthly(inp["chirps_v3_popw"], ["precipitation_mean"]))
    chirps = chirps.merge(popw.rename(columns={"precipitation_mm": "precipitation_popw_mm"}), on=KEYS, how="left")
    table = table.merge(chirps, on=KEYS, how="left")
    table["precipitation_source"] = np.where(table["precipitation_sum"].notna(), "chirps_v3", "")

    report = {"fit_years": list(fit_years), "method": (
        "VIIRS province × calendar-month climatology + beta × MODIS anomaly "
        "[+ gamma × ERA5 temperature_2m_mean anomaly for LST], anomalies against the same "
        "province × calendar-month climatologies over fit_years"), "features": {}}
    viirs_cache = {}
    for new, (key, col, modis_col, use_air) in VIIRS.items():
        if key not in viirs_cache:
            cols = sorted({c for k, c, *_ in VIIRS.values() if k == key})
            viirs_cache[key] = monthly(inp[key], cols)
        viirs = viirs_cache[key][KEYS + [col]].rename(columns={col: "new"})

        pairs = with_month(viirs.merge(modis[KEYS + [modis_col, AIR]], on=KEYS)
                                .rename(columns={modis_col: "old"}))
        pairs = pairs[pairs["year"].between(*fit_years)].dropna(subset=["new"])
        model = fit(pairs, use_air)
        report["features"][new] = {"from": modis_col, "beta": model["beta"], "gamma": model["gamma"],
                                   "pairs": int(len(pairs)), **loyo(pairs, use_air)}

        table = table.merge(viirs.rename(columns={"new": new}), on=KEYS, how="left")
        table[f"{new.removesuffix('_mean')}_source"] = np.where(table[new].notna(), "viirs", "")

        # Months before VIIRS: convert from MODIS (+ ERA5).
        gap = with_month(table.loc[table[new].isna(), KEYS])
        gap = gap.merge(modis[KEYS + [modis_col, AIR]].rename(columns={modis_col: "old"}),
                        on=KEYS, how="left")
        gap = gap[gap["year_month"] < viirs["year_month"].min() + pd.offsets.MonthBegin(1)]
        values = pd.Series(predict(model, gap), index=gap.set_index(KEYS).index)
        idx = table.set_index(KEYS).index
        fill = values.reindex(idx).to_numpy()
        take = np.isnan(table[new].to_numpy()) & ~np.isnan(fill)
        table.loc[take, new] = fill[take]
        table.loc[take, f"{new.removesuffix('_mean')}_source"] = "modis_converted"
        report["features"][new]["converted_rows"] = int(take.sum())

    table["year_month"] = table["year_month"].dt.strftime("%Y-%m-%d")
    Path(out["table"]).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out["table"], index=False)
    Path(out["conversion"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out["conversion"]).write_text(json.dumps(report, indent=2))

    print(f"Saved: {out['table']} {table.shape}")
    for new, r in report["features"].items():
        print(f"  {new:15s} beta={r['beta']:.3f} gamma={r['gamma']:.3f}  LOYO RMSE "
              f"{r['rmse_conversion']:.3f} (climatology alone {r['rmse_climatology_only']:.3f})  "
              f"converted rows={r['converted_rows']}")
    src = [c for c in table.columns if c.endswith("_source")]
    print(table[src].apply(pd.Series.value_counts).fillna(0).astype(int).to_string())


if "snakemake" in dir():
    main({k: v for k, v in snakemake.input.items()},                     # noqa: F821
         {k: v for k, v in snakemake.output.items()},                    # noqa: F821
         tuple(snakemake.params.fit_years))                              # noqa: F821
