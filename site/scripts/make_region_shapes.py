"""Province outlines for the site's risk map, as a small GeoJSON (Snakemake script).

Simplifies the AOI the zonal statistics are extracted for and rounds coordinates, so the
file stays small enough to commit and load in a browser. Features carry `country` and
`name`, the keys the predictions use.
"""
import json
from pathlib import Path

import geopandas as gpd
import shapely

TOLERANCE_DEG = snakemake.params.tolerance     # ~1 km at the equator for 0.01
GRID_DEG      = snakemake.params.grid          # coordinate rounding

aoi = gpd.read_parquet(snakemake.input.aoi)[["admin", "name", "geometry"]]
geom = aoi.geometry.simplify(TOLERANCE_DEG, preserve_topology=True)
aoi["geometry"] = shapely.set_precision(geom.values, GRID_DEG)
aoi = aoi.rename(columns={"admin": "country"})

out = Path(snakemake.output.shapes)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(json.loads(aoi.to_json(drop_id=True)), separators=(",", ":")))
print(f"{len(aoi)} provinces → {out} ({out.stat().st_size / 1e6:.2f} MB)")
