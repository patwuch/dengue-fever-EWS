# Chuang Lab — Taipei Medical University

Spatiotemporally-aware ML/DL dengue fever prediction by jointly modelling Taiwan and Southeast Asia trends and dynamics: integrating climate projections, reported infection cases, and Shared Socioeconomic Pathways (SSPs). 

## Modules

**climate-projection-module** — Processes climate projection models of various SSPs and simulation models from NetCDF into merged NetCDF and easy-to-use TSV outputs.

**dengue-infection-module** — Cleans and standardises dengue case data from OpenDengue and Indonesia MoH sources.

**machine-learning-module** — Trains, evaluates, and creates xAI artefacts for Random Forest, XGBoost, and STGNN models on remote sensing + climate projection + infection data.

## Recommended Use

Copy `.env.example` to `.env` and fill in your W&B key:

```bash
cp .env.example .env
```

For environment management build [Docker](https://docs.docker.com/desktop/setup/install/windows-install/) container from __Dockerfile__ and __requirements.txt__:

```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t dengue-ews .

docker run --gpus all \
  --env-file .env \
  -v /path/to/data:/workspace/data \
  -v /path/to/results:/workspace/machine-learning-module/results \
  dengue-ews \
  snakemake <target> --configfile config/<experiment>.yaml --cores 4 --resources gpu=1
```

`--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. 

### Retraining API

`api/` runs a small local HTTP API on the host (not in the container) that triggers a `docker run ... snakemake ...` job, so you don't have to type the full command each time. It only accepts one job at a time and rejects a second request while one is running.

```bash
pip install -r api/requirements.txt
uvicorn api.server:app --app-dir . --host 127.0.0.1 --port 8756
```

```bash
curl -X POST http://127.0.0.1:8756/jobs \
  -H "Content-Type: application/json" \
  -d '{"target": "results/STGNN/production_logIR/best_model.pt",
       "configfile": "config/OpenDengue/stgnn_logIR_production.yaml",
       "cores": 4, "gpu": 1}'

curl http://127.0.0.1:8756/jobs/<job_id>
curl http://127.0.0.1:8756/jobs/<job_id>/logs
```

The server loads `.env` on startup (same file as above) and passes `WANDB_API_KEY` through to the container. `DENGUE_DATA_DIR` and `DENGUE_RESULTS_DIR` override the default `./data` and `./machine-learning-module/results` volume mounts.

This binds to localhost by design — it shells out to `docker run` with no auth, so don't expose it beyond your own machine without adding one.

### Monthly inference

`.github/workflows/monthly-update.yml` runs on the 1st of each month on a CPU GitHub runner. It fetches the latest Earth Engine zonal statistics, runs the models below, commits the predictions under `site/data/`, and redeploys the site.

Both models share the cyclical design (sin/cos month encoding, 12-month window) and the same architecture, and both are trained on 2011–2018. The climate-only model drives the live risk map: no incidence data exists after 2018, so it runs on the latest Earth Engine data alone, delta-corrected for the 2019–2026 climate shift. The IR + environment model is the study model it was derived from (its hyperparameters come from the `cyclical_seasonal` sweep); it also forecasts incidence in months when recent case data is supplied.

| | **Climate only** (`climate`) — live map | **IR + environment** (`ir_env`) |
|---|---|---|
| Experiment / production config | — / `stgnn_climate_risk_production.yaml` | `cyclical_seasonal` / `stgnn_logIR_production.yaml` |
| Model inputs | weather, land use, month encoding | past IR, weather, land use, month encoding |
| Needs | Earth Engine | recent incidence + Earth Engine |
| Output | risk index | predicted IR + risk index |
| Runs when | every month (delta-corrected by default) | `site/data/incidence/recent_incidence.csv` exists (format: `site/data/incidence/README.md`) |

Both forecast one month ahead of the last input month. Risk index = (1 + predicted IR) / (1 + the province's 2011–2018 mean IR for that calendar month), so values above 1 mean above that province's usual level for the time of year.

**Producing a bundle.** Each model is shipped as an *inference bundle*: `best_model.pt`, `best_params.json`, and `bundle.json`, which holds the node order, scalers, seasonal means and risk baselines the runner needs instead of the training CSV. `export_inference_bundle` packs them into `inference_bundle_<bundle_label>.tar.gz`, named like the release asset.

Bundles are built on the k3s cluster with the `profiles/k8s` profile. Every rule runs as a Job in `dengue-ews` and reads and writes the S3 store from `k8s/storage.yaml`. The store needs only two author-supplied inputs: `data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv` and `results/STGNN/cyclical_seasonal/best_params.json`.

```bash
cd machine-learning-module
# S3 credentials (SNAKEMAKE_STORAGE_S3_ACCESS_KEY / _SECRET_KEY) in the environment,
# kube context = the snakemake ServiceAccount from k8s/rbac.yaml
snakemake results/STGNN/production_climate_risk/inference_bundle_climate.tar.gz \
    --configfile config/OpenDengue/stgnn_climate_risk_production.yaml --profile profiles/k8s
snakemake results/STGNN/production_logIR/inference_bundle_ir_env.tar.gz \
    --configfile config/OpenDengue/stgnn_logIR_production.yaml --profile profiles/k8s
```

The tarballs land in S3 under the same paths (`s3://dengue-ews/results/STGNN/<name>/`); download them from there before uploading to the release.

**One-time setup:**

1. Create a release holding the bundles and region geometry. Re-upload with `--clobber` after retraining.
   ```bash
   cp data/processed/dengue-infection/geoparquet/gaul_2024_sea_filtered.parquet regions.parquet
   gh release create inference-bundles --title "Inference bundles" --notes "Model bundles for monthly inference"
   gh release upload inference-bundles inference_bundle_climate.tar.gz \
       inference_bundle_ir_env.tar.gz \
       regions.parquet --clobber
   ```
   Instead of `regions.parquet`, you can upload the regions as an Earth Engine table and set the repository variable `GEE_REGIONS_ASSET`.
2. Add the `GEE_SERVICE_ACCOUNT` secret: a service-account key JSON for an Earth Engine–registered Cloud project.
3. Produce `site/data/climate_deltas.json` offline with `zonal-statistics-module/compute_climate_delta.py` (pass `--bundle` to use the climate bundle as the 2011–2018 baseline) and commit it. Without it, climate mode runs uncorrected and records `bias_corrected: false`.

## Specs

Tested on:
- OS: Ubuntu 24.04 (WSL2)
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM, compute capability 8.6 / Ampere)
- Host CUDA driver: 13.3, container CUDA runtime: 12.1 (via `--gpus all` / NVIDIA Container Toolkit)
- PyTorch: 2.5.1+cu121
- Docker image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime


GPU acceleration is used only for the machine-learning-module for
- XGBoost tuning, training, and inference
- STGNN tuning, training, inference, and xAI


## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)