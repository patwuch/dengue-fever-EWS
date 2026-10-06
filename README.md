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