"""
Climate delta bias correction for raw GEE zonal statistics.

The STGNN was trained on 2011-2018 data. Feeding current observations
directly produces covariate shift — features like temperature sit
systematically outside the training distribution. The correction subtracts a
per-variable, per-calendar-month delta (current climatology minus the
2011-2018 training climatology) from each observation before inference.

Deltas come from site/data/climate_deltas.json, written offline by
zonal-statistics-module/compute_climate_delta.py:
    {"01": {"temperature_2m_mean": 0.41, ...}, ..., "12": {...}}

If that file is missing, no correction is applied (and bias_corrected stays
False). An older version fell back to "this month's SEA-wide mean minus the
training mean", but that subtracts this year's anomaly along with the
long-term shift, pinning every month to the 2011-2018 average.

run_inference.py calls correct_regions() for each month in its input window.
Run this file directly to write the corrected latest month for inspection:

    python site/scripts/apply_climate_delta.py
"""

import json
import pathlib

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
DELTAS_PATH = DATA_DIR / "climate_deltas.json"
STATS_PATH  = DATA_DIR / "latest_zonal_stats.json"
OUTPUT_PATH = DATA_DIR / "delta_corrected_stats.json"


def load_deltas(path: pathlib.Path = DELTAS_PATH) -> dict[str, dict[str, float]] | None:
    """Return {"MM": {feature: delta}} or None if no usable delta file exists."""
    if not path.exists():
        return None
    deltas = json.loads(path.read_text())
    usable = {
        month: {k: float(v) for k, v in feats.items()}
        for month, feats in deltas.items()
        if isinstance(feats, dict) and feats
        and all(isinstance(v, (int, float)) for v in feats.values())
    }
    return usable or None


def correct_regions(regions: dict, month_deltas: dict[str, float]) -> dict:
    """Subtract each feature's delta from every region's raw observation."""
    corrected = {}
    for region, vals in regions.items():
        vals = dict(vals)
        for feat, delta in month_deltas.items():
            if isinstance(vals.get(feat), (int, float)):
                vals[feat] = vals[feat] - delta
        corrected[region] = vals
    return corrected


def correct_stats(stats: dict, deltas: dict | None) -> dict:
    """Return a copy of a monthly zonal-stats dict with the matching month's deltas applied."""
    month_key = stats["target_month"][5:7]
    month_deltas = (deltas or {}).get(month_key)
    out = {k: v for k, v in stats.items() if k != "regions"}
    if not month_deltas:
        out["bias_corrected"] = False
        out["regions"] = stats["regions"]
        return out
    out["bias_corrected"] = True
    out["regions"] = correct_regions(stats["regions"], month_deltas)
    return out


def main() -> None:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Zonal stats not found: {STATS_PATH}\nRun fetch_zonal_stats.py first.")
    stats  = json.loads(STATS_PATH.read_text())
    deltas = load_deltas()
    if deltas is None:
        print(f"No usable {DELTAS_PATH.name} — writing uncorrected stats. "
              "Run zonal-statistics-module/compute_climate_delta.py to produce it.")
    corrected = correct_stats(stats, deltas)
    OUTPUT_PATH.write_text(json.dumps(corrected, indent=2))
    print(f"→ {OUTPUT_PATH} (bias_corrected={corrected['bias_corrected']})")


if __name__ == "__main__":
    main()
