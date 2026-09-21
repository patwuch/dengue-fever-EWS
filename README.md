# Chuang Lab — Taipei Medical University

Spatiotemporally-aware ML/DL dengue fever prediction by jointly modelling Taiwan and Southeast Asia trends and dynamics: integrating climate projections, reported infection cases, and Shared Socioeconomic Pathways (SSPs). 

## Modules

**climate-projection-module** — Processes climate projection models of various SSPs and simulation models from NetCDF into merged NetCDF and easy-to-use TSV outputs.

**dengue-infection-module** — Cleans and standardises dengue case data from OpenDengue and Indonesia MoH sources.

**machine-learning-module** — Trains, evaluates, and creates xAI artefacts for Random Forest, XGBoost, and STGNN models on remote sensing + climate projection + infection data.

## Recommended Use

For environment management build [Docker](https://docs.docker.com/desktop/setup/install/windows-install/) container from __Dockerfile__ and __requirements.txt__:

```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t dengue-ews .

docker run --gpus all \
  -e WANDB_API_KEY=your_key \
  -v /path/to/data:/workspace/data \
  -v /path/to/results:/workspace/machine-learning-module/results \
  dengue-ews \
  snakemake <target> --configfile config/<experiment>.yaml --cores 4 --resources gpu=1
```

`--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. On WSL2 this just means Docker Desktop with WSL2 integration and
GPU support enabled — no separate driver install inside the distro itself.

Alternatively build your own [Conda](https://anaconda.org/anaconda/conda) environment with __environment.yml__ but GPU compatibility may vary.


## Specs

Tested on:
- OS: Ubuntu 24.04.4 LTS
- CPU: Intel Xeon W-2235 (6 cores / 12 threads @ 3.80GHz)
- RAM: 32GB
- GPU: NVIDIA GeForce GTX 1050 Ti (4GB VRAM, compute capability 6.1)
- CUDA: 12.2
- PyTorch: 2.5.1+cu121
- Docker image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

Also verified on:
- OS: Ubuntu 24.04 (WSL2)
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM, compute capability 8.6 / Ampere)
- Host CUDA driver: 13.3, container CUDA runtime: 12.1 (via `--gpus all` / NVIDIA Container Toolkit)

Note: GTX 1050 Ti (sm_61) is not explicitly compiled in the above image but 
falls back to sm_60 and runs correctly. Users with sm_75+ (Turing and newer) 
will get fully optimized builds — this includes the RTX 3050 (sm_86, Ampere).

GPU acceleration is used only for the machine-learning-module for
- XGBoost tuning, training, and inference
- STGNN tuning, training, inference, and xAI


## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)