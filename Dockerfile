FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libnetcdf-dev \
    graphviz \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# SSL — portable across machines
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# W&B — set at runtime via -e WANDB_API_KEY=your_key
ENV WANDB_MODE=offline

# match host user to avoid permission issues with mounted volumes.
# Tolerates UID/GID 0 (host user is root, e.g. some WSL setups) where
# group/user 0 already exist in the base image — group/useradd would
# otherwise fail with "GID/UID already exists".
ARG UID=1000
ARG GID=1000
RUN (getent group "$GID" >/dev/null || groupadd -g "$GID" appgroup) && \
    (getent passwd "$UID" >/dev/null || useradd -u "$UID" -g "$GID" -m appuser)

# Python dependencies. requirements-lock.txt pins exact versions (-c adds no packages, only
# pins them), so the image built here matches one built on another machine.
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt

# source code only — data and outputs mounted at runtime.
# The whole machine-learning-module/ tree is copied (Snakefile, rules/,
# src/, config/); data/, logs/, report/, models/, results/, notebooks/ are
# excluded by .dockerignore so this stays source-only.
COPY machine-learning-module/ ./machine-learning-module/

# machine-learning-module/Snakefile resolves PROJECT_ROOT by walking up
# parent directories for a ".git" directory (find_git_root()). .dockerignore
# deliberately excludes .git/, so without this the walk falls through to the
# filesystem root and every data/results path in the Snakefile resolves
# wrong. find_git_root() only checks (path / ".git").exists() — it never
# reads .git's contents — so an empty marker directory one level above
# machine-learning-module/ reproduces the same repo-root anchor.
RUN mkdir -p .git

# Reference the account by numeric id, not name — when UID/GID 0 already
# existed above, the matching name is "root", not "appuser".
RUN chown -R "$UID:$GID" /workspace
USER $UID:$GID

WORKDIR /workspace/machine-learning-module

# build:
#   docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t dengue-ews .
#
# GPU access requires the NVIDIA Container Toolkit on the host:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# On WSL2 this means Docker Desktop with WSL2 integration + GPU support
# enabled and a recent host NVIDIA driver — no separate driver install
# inside the WSL distro itself.
#
# run:
#   docker run --gpus all \
#     -e WANDB_API_KEY=your_key \
#     -v /path/to/data:/workspace/data \
#     -v /path/to/results:/workspace/machine-learning-module/results \
#     dengue-ews \
#     snakemake results/STGNN/production_logIR/best_model.pt \
#       --configfile config/OpenDengue/stgnn_logIR_production.yaml \
#       --cores 4 --resources gpu=1
