# Snakemake controller for the tmu k3s cluster: builds the DAG and submits each rule as a Job in
# dengue-ews. No workflow code runs here; the pods run it in the worker images (dengue-ews:k8s-v1).
#
# The Snakemake and plugin versions must match the worker images, since each pod runs the image's
# own Snakemake on the job it is sent. Change them together with requirements-lock.txt.
#
# build (nothing is copied in, so k8s/ is enough as context):
#   docker build -f k8s/controller.Dockerfile -t dengue-ews-controller:v1 k8s/
#
# run from the repo root, as your own user so .snakemake/ and git stay yours:
#   docker run --rm -it --user $(id -u):$(id -g) \
#     --env-file ~/.config/dengue-ews/s3.docker.env \
#     -e KUBECONFIG=/kube/config -v ~/.config/dengue-ews/kubeconfig:/kube/config:ro \
#     -v $PWD:/workspace -w /workspace/machine-learning-module \
#     dengue-ews-controller:v1 <target> --configfile <config> --profile profiles/k8s
FROM python:3.11-slim

# git: Snakemake packs the workflow sources for the pods with `git ls-files`.
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    snakemake==9.27.0 \
    snakemake-executor-plugin-kubernetes==0.5.1 \
    snakemake-storage-plugin-s3==0.3.6

# The repo is mounted, not copied, and is owned by the host user; without this git refuses it
# ("dubious ownership") whenever the container's uid differs from the owner's.
RUN git config --system --add safe.directory '*'

# --user <uid> has no passwd entry or home. Snakemake and git still want a writable HOME, and
# Snakemake's run header calls getpass.getuser(), which reads $USER before the passwd database.
ENV HOME=/tmp USER=snakemake
WORKDIR /workspace
ENTRYPOINT ["snakemake"]
