"""One monthly EZGEE run: zonal statistics for a month and the month before it (Snakemake script).

Builds the EZGEE config with ezgee_config.build_payload, runs EZGEE's own Snakefile with
$EZGEE_PYTHON as a separate local Snakemake (its Snakemake 7 never talks to the cluster), and
copies each product's merged parquet to this rule's outputs. EZGEE works in a scratch APP_DIR
that is removed afterwards; only the outputs leave the pod.

Earth Engine credentials come from GOOGLE_APPLICATION_CREDENTIALS, the path of a service-account
key file, as in the EZGEE app. On the cluster k8s/gee-key-for-inference-pods.yaml sets it from the
gee-key Secret; locally, export it.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from ezgee_config import build_payload

ROOT = Path(__file__).resolve().parent.parent.parent
EZGEE_SNAKEFILE = ROOT / "zonal-statistics-module" / "Snakefile"

key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not key or not Path(key).is_file():
    sys.exit("GOOGLE_APPLICATION_CREDENTIALS must be the path of the GEE service-account key "
             f"(got {key!r}).")

month    = snakemake.wildcards.month
products = snakemake.params.products            # output name → EZGEE product
run_id   = f"m{month}"
app_dir  = Path(tempfile.mkdtemp(prefix="ezgee-"))
run_dir  = app_dir / "data" / "runs" / run_id
run_dir.mkdir(parents=True)

payload = build_payload(snakemake.params.start, snakemake.params.end, list(products.values()),
                        shp=str(Path(snakemake.input.aoi).resolve()), app_dir=str(app_dir),
                        run_id=run_id, gee_concurrency=snakemake.params.gee_concurrency)
config_path = run_dir / "config.yaml"
config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
print(f"EZGEE {run_id}: {', '.join(products.values())} "
      f"{snakemake.params.start} → {snakemake.params.end}", flush=True)

# The command backend/app.py runs, minus the app's log handler and run database.
python = os.environ.get("EZGEE_PYTHON", sys.executable)
subprocess.run(
    [python, "-W", "ignore::FutureWarning", "-m", "snakemake",
     "--snakefile", str(EZGEE_SNAKEFILE), "--configfile", str(config_path),
     "--directory", str(run_dir), "-j", "12",
     "--resources", f"gee={snakemake.params.gee_concurrency}",
     "--rerun-triggers", "mtime"],
    check=True, cwd=run_dir,
    env={**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONUTF8": "1"},
)

for name, product in products.items():
    task = payload["products"][product]
    src = Path(payload["output_dir"]) / product / f"{product}_{task['start_date']}_to_{task['end_date']}.parquet"
    dst = Path(snakemake.output[name])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"{product}: {dst}")

shutil.rmtree(app_dir)
