# CPU worker image for the monthly inference workflow (repo-root Snakefile). The job pods of
# profiles/k8s-inference run it, and it runs the same workflow locally with --cores.
#
# Two Python environments:
#   /usr/local      ours: the env table and STGNN inference. Where a package is also in the GPU
#                   image (Dockerfile + requirements-lock.txt) the version is the same, so the env
#                   table and predictions come out identical from either image.
#   /opt/ezgee      EZGEE's own, built from zonal-statistics-module/pixi.lock: the exact versions
#                   of the Windows install that ran the 2019–2026 batch (Python 3.10, pandas 2.3.3,
#                   Snakemake 7.32.4). The zonal-stats rule runs EZGEE's Snakefile with
#                   $EZGEE_PYTHON as a separate local Snakemake inside the pod, so its version
#                   doesn't need to match ours.
#
# build (from the repo root; only the two pixi files are copied in):
#   docker build -f inference.Dockerfile -t ghcr.io/patwuch/dengue-ews-inference:v2 .
#   docker push ghcr.io/patwuch/dengue-ews-inference:v2       (needs a token with write:packages)
#
# local run, from the repo root:
#   docker run --rm --user $(id -u):$(id -g) -v $PWD:/workspace ghcr.io/patwuch/dengue-ews-inference:v2 \
#     snakemake data/processed/env/SEA_env_monthly.csv --cores 2
FROM python:3.11-slim

# Links the GHCR package to the repo (shown on the repo page, inherits its access).
LABEL org.opencontainers.image.source=https://github.com/patwuch/dengue-fever-EWS

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# ── EZGEE environment ─────────────────────────────────────────────────────────
# The pixi release the Windows app uses.
ARG PIXI_VERSION=0.66.0
RUN curl -fsSL "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl" \
        -o /usr/local/bin/pixi && chmod +x /usr/local/bin/pixi
WORKDIR /opt/ezgee
COPY zonal-statistics-module/pixi.toml zonal-statistics-module/pixi.lock ./
# --locked: fail instead of re-resolving if the lock doesn't satisfy pixi.toml.
RUN pixi install --locked && pixi clean cache --yes && rm -rf /root/.cache
ENV EZGEE_PYTHON=/opt/ezgee/.pixi/envs/default/bin/python

# ── Our environment ───────────────────────────────────────────────────────────
# CPU build of the torch the models were trained with (the GPU image has 2.5.1+cu121).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Workflow: must match k8s/controller.Dockerfile and requirements-lock.txt.
RUN pip install --no-cache-dir \
    snakemake==9.27.0 \
    snakemake-executor-plugin-kubernetes==0.5.1 \
    snakemake-storage-plugin-s3==0.3.6

RUN pip install --no-cache-dir \
    numpy==2.1.2 \
    pandas==3.0.6 \
    pyarrow==25.0.1 \
    geopandas==1.2.0 \
    shapely==2.1.2 \
    pyogrio==0.13.0 \
    pyyaml==6.0.2 \
    torch-geometric==2.6.1

# --user <uid> has no passwd entry or home; Snakemake, earthengine-api and getpass still want them.
ENV HOME=/tmp USER=snakemake
WORKDIR /workspace
