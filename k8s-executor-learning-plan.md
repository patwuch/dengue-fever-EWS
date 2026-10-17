# Learning plan: Snakemake Kubernetes executor (Snakemake 9)

Not a project requirement — the pipeline runs fine today with `docker run --gpus all` on a
single GPU workstation. This is a scoped learning exercise to get hands-on with the
Kubernetes primitives that tools like Kubeflow Pipelines/Argo Workflows are built on, without
porting the whole repo.

**Scope:** one rule only — `train_stgnn_production` in `machine-learning-module`, the GPU-bound
step. Porting the rest of the DAG adds work without adding anything new to learn.

**Versions:** this plan targets Snakemake 9.x (what `requirements.txt` and the Docker image
use). Snakemake 8 replaced the built-in `--kubernetes` flag and `S3.remote(...)` remote
providers with plugins:

| Snakemake 7 (old)                   | Snakemake 9                                                         |
|-------------------------------------|---------------------------------------------------------------------|
| `--kubernetes [namespace]`          | `--executor kubernetes --kubernetes-namespace <ns>`                 |
| `--default-remote-provider S3`      | `--default-storage-provider s3`                                     |
| `--default-remote-prefix bucket`    | `--default-storage-prefix s3://bucket`                              |
| `S3.remote("bucket/x")` in rules    | plain paths (default storage) or `storage.s3("s3://bucket/x")`      |
| `--restart-times N`                 | `--retries N`                                                       |
| `--container-image`                 | `--container-image` (unchanged)                                     |

Required plugins, on the host **and** inside the job image (the pod runs its own Snakemake):
`snakemake-executor-plugin-kubernetes` (checked against v0.5.1) and
`snakemake-storage-plugin-s3` (v0.3.6).

## Target cluster

The plan runs against the existing two-node k3s cluster (`tmu`), not the single-node k3s on the
WSL2 laptop:

| Node                          | Role          | LAN IP      | Tailnet                                  | GPU                     |
|-------------------------------|---------------|-------------|------------------------------------------|-------------------------|
| `tmu-clhm-nuc15crhu5`         | control-plane | 10.8.1.30   | `tmu-clhm-nuc15crhu5` (100.100.41.64)    | none                    |
| `chuang-hp-z4-g4-workstation` | worker        | 10.5.7.147  | 100.124.149.3                            | GTX 1050 Ti, 4 GB (x1)  |

Snakemake runs on the laptop and reaches the API server over Tailscale:

- **Kubeconfig:** `/root/.kube/k3s-tmu.yaml` (context `tmu`). This is a copy of the
  workstation's `~/.kube/k3s-tmu.yaml` with `server:` changed to
  `https://tmu-clhm-nuc15crhu5:6443`.
- **Why the hostname:** the API server certificate lists the hostname but not the tailnet IP,
  so TLS verifies normally. The laptop can't reach `10.8.1.30` directly.
- **Usage:** `export KUBECONFIG=/root/.kube/k3s-tmu.yaml` before running `kubectl` or
  Snakemake. The executor plugin calls `load_kube_config()`, which honors `KUBECONFIG`.
- **Access level:** this kubeconfig has full admin rights, which is one more reason to do
  Phase 2.

**Pre-flight:** the `kubeflow` namespace has a crash-looping `workflow-controller` and a long
list of `Evicted` `cache-deployer` pods from about 22 days ago. That many evictions usually
means a node ran short of disk or memory. Both nodes show no pressure now (2026-09-29), but
check `kubectl get events -A` and clean up the evicted pods before adding GPU work.

## Phase 0 — Cluster bootstrap

**Status:** done on `tmu` (2026-09-29). Both nodes are `Ready`, the NVIDIA device plugin runs
on the workstation, it advertises `nvidia.com/gpu: 1`, and pods that request a GPU can use it
without setting `runtimeClassName`.

**What was needed — the default runtime.** The executor plugin never sets `runtimeClassName` on
the Jobs it creates, and k3s's default runtime on the workstation was plain `runc`. A probe pod
that requested `nvidia.com/gpu: 1` with no `runtimeClassName` (what the plugin creates):

- **Before the fix:** the pod was scheduled on the workstation and got
  `NVIDIA_VISIBLE_DEVICES`, but had no `/dev/nvidia*` devices and no `nvidia-smi`. So
  `torch.cuda.is_available()` would have been `False`.
- **After the fix:** `/dev/nvidia0`, `/dev/nvidiactl` and the `uvm` devices were present, and
  `nvidia-smi -L` listed the GTX 1050 Ti.

The fix, on the workstation (agent-only nodes have no `/etc/rancher/k3s/` until you create it):

```bash
sudo mkdir -p /etc/rancher/k3s
echo 'default-runtime: nvidia' | sudo tee /etc/rancher/k3s/config.yaml
sudo systemctl restart k3s-agent
```

k3s regenerates containerd's `config.toml` from this on every start, so edit
`config.yaml`, never containerd's file. Running containers survive the restart and keep
their old runtime until they're recreated. Only the workstation is affected; the NUC has its
own containerd. Docker on the workstation already defaulted to `nvidia`, but k3s's
containerd has its own config, so that setting never carried over. To roll back, remove the
line and restart the agent.

**Gotcha seen on restart:** after `k3s-agent` restarted, the device plugin (v0.17.0)
re-registered with the kubelet but marked the GPU unhealthy ("Unknown Error"), so allocatable
`nvidia.com/gpu` dropped to 0 while `nvidia-smi` on the host was fine. Deleting the device
plugin pod, which the DaemonSet recreates, cleared it. After any `k3s-agent` restart, check
`kubectl get node chuang-hp-z4-g4-workstation -o jsonpath='{.status.allocatable}'`. If GPU
jobs sit `Pending` with `Insufficient nvidia.com/gpu`, this is the first thing to check.

**Side effect of a default `nvidia` runtime:** it exposes GPUs based on `NVIDIA_VISIBLE_DEVICES`
wherever that variable comes from. An image that sets `NVIDIA_VISIBLE_DEVICES=all` itself
(common in CUDA base images) gets the GPU without requesting `nvidia.com/gpu`, and the
scheduler doesn't count it. Check the job image with
`docker inspect <image> --format '{{.Config.Env}}'`. With one GPU and one job at a time this is
acceptable; if the GPU is ever shared, switch the device plugin to a device-list strategy
other than `envvar` and set the NVIDIA runtime to ignore the variable in unprivileged
containers.

[k8s/gpu-runtime-policy.yaml](k8s/gpu-runtime-policy.yaml) was the no-sudo alternative (a
`MutatingAdmissionPolicy` adding `runtimeClassName: nvidia` to GPU pods in `dengue-ews`). It's
no longer needed and was never applied, so the `dengue-ews` namespace doesn't exist yet;
Phase 2 creates it.

**Lesson:** GPU access is configured per container runtime, not system-wide — and within
Kubernetes, per *pod*. "Scheduled on a GPU node" and "can see the GPU" are two separate
things.

## Phase 1 — Image distribution

**Status:** done (2026-09-30). `dengue-ews:k8s-v1` is in the workstation's k3s containerd. A
probe pod (GPU request, no pull policy set) got `IfNotPresent`, and its events showed "already
present on machine", so there was no pull. Inside, it had Snakemake 9.27.0,
snakemake-storage-plugin-s3 0.3.6, snakemake-executor-plugin-kubernetes 0.5.1 and
torch-geometric 2.6.1, `torch.cuda.is_available()` was `True`, and `GATConv` forward+backward
ran on the GTX 1050 Ti. The container ran as uid 1000.

How it was done, and why:
- **Build once, then rebuild with a lock:** built and tested on the laptop, then rebuilt on
  the workstation rather than shipped. The laptop→workstation link measured ~2.9 MB/s, so
  sending 12.9 GB would take ~75 min.
- **Lock file:** `requirements-lock.txt` pins the 111 packages `requirements.txt` adds on top
  of the base, via `pip install -c`. Both builds produced the identical 222-package
  environment. Without it they drift: the old `dengue_ews:latest` had torch-geometric 2.7.0
  vs the pinned 2.6.1.
- **Build time:** 54 min on the workstation, 46 of them spent in pip because PyPI downloads
  there ran at ~0.3–0.7 MB/s. The PyTorch base had the same digest on both machines, so it
  didn't need downloading.
- **Old GPU is fine:** the GTX 1050 Ti (sm_61) doesn't need a special image. The torch
  2.5.1+cu121 wheel includes sm_60 code, which runs on it. Keep the base pinned: newer
  PyTorch CUDA builds have dropped Pascal.
- **Laptop build gotcha:** Docker Desktop's `credsStore: desktop.exe` can't run from this
  WSL shell (interop is off), so the build ran with a temporary empty `DOCKER_CONFIG`.

**Implement:**
1. Add `snakemake-executor-plugin-kubernetes` and `snakemake-storage-plugin-s3` to the image.
   The executor sets `auto_deploy_default_storage_provider`, so it will try to `pip install`
   the storage plugin at pod startup if it's missing. Baking it in avoids depending on network
   access from inside the pod.
2. Build with a **non-`latest` tag**, e.g. `dengue-ews:k8s-v1`.
3. Import it on the **workstation**, the only node the GPU pod can land on. Every node keeps
   its own image cache, so importing on the laptop or the NUC does nothing for a pod on the
   workstation. The workstation has Docker, so build there and then run
   `docker save dengue-ews:k8s-v1 | sudo k3s ctr images import -`. Confirm with
   `sudo k3s ctr images ls | grep dengue` (it will be listed as
   `docker.io/library/dengue-ews:k8s-v1`). A registry would avoid importing on each node,
   but with one GPU node a manual import is simpler.

**Lesson:** a cluster doesn't share Docker's local image cache — `docker images` showing the
image means nothing to k3s's containerd. The tag matters too: the executor plugin doesn't set
`imagePullPolicy`, and Kubernetes defaults to `Always` for `:latest`. So even a correctly
imported `dengue-ews:latest` would still go to Docker Hub and fail with
`ErrImagePull`/`ImagePullBackOff`. Pinned tags default to `IfNotPresent`.

## Phase 2 — RBAC for Snakemake ✅ done 2026-10-01

**Implemented:** namespace `dengue-ews`, ServiceAccount `snakemake`, and Role + RoleBinding in
[k8s/rbac.yaml](k8s/rbac.yaml). The verbs are exactly the eight API calls the 0.5.1 plugin
makes, grep'd from the installed package inside `dengue-ews:k8s-v1`:

| Resource      | Verbs          | Plugin call(s)                                           |
|---------------|----------------|----------------------------------------------------------|
| `jobs`        | create, delete | `create_namespaced_job`, `delete_namespaced_job`         |
| `jobs/status` | get            | `read_namespaced_job_status` polling                     |
| `pods`        | list, delete   | `list_namespaced_pod` (by `job-name` label), cleanup     |
| `pods/log`    | get            | `read_namespaced_pod_log` for failed jobs                |
| `secrets`     | create, delete | per-run Secret carrying env vars (S3 credentials)        |

No `get`/`watch` anywhere. Re-check the list when bumping the plugin. The `default` SA has
`automountServiceAccountToken: false`, so job pods carry no API token. Kubeconfig context
`dengue-ews` (user `snakemake@dengue-ews`, 24h token) is in `~/.kube/k3s-tmu.yaml`. Every
`auth can-i` check matched expectations. The deliberate `pods/log` 403 test was skipped.

Then make Snakemake actually *use* that identity. Snakemake authenticates with whatever
kubeconfig context is current, and the `tmu` context is cluster-admin. So a Role you create
but never switch to changes nothing. Mint a token (`k3s kubectl -n dengue-ews create token
snakemake --duration=24h`), add a kubeconfig user + context with it, and switch to that
context before running Snakemake. `--kubernetes-service-account-name` is a different thing:
it sets the SA the *job pods* run as, and those pods don't call the API, so leave it unset.

**Lesson:** Kubernetes access is least-privilege by default, and there are two identities
involved: the *controller* (Snakemake on the host, submitting Jobs) and the *workload* (the
pods). The failure mode to try on purpose: drop `pods/log` from the Role, make a job fail, and
watch the log-collection step return 403 instead of the real error.

**Controller placement:** Snakemake is a control loop (target → observe outputs and Job status →
submit the next Job) running *outside* the cluster. If the laptop sleeps, Jobs already running
finish and upload their outputs, but nothing new is submitted until Snakemake is rerun, and it
then resumes from what's in the bucket. Kubeflow Pipelines/Argo run the workflow controller as
an in-cluster pod, which closes this gap.

## Phase 3 — Artifact staging (the core lesson)

**Implement:**
1. Run an S3 store inside k3s: [k8s/storage.yaml](k8s/storage.yaml), SeaweedFS (the same image
   Kubeflow already runs here, so known-good, instead of MinIO), Deployment + `NodePort` 30900
   + `local-path` PVC. Pin
   it to the NUC with a `nodeSelector` so its data stays on one node's disk. Both the laptop
   Snakemake and the pods need to reach it at the **same** URL, because the host's
   `--storage-s3-endpoint-url` is passed through to the pods unchanged. That's awkward on this
   cluster:
   - The laptop can reach the NUC only over Tailscale (`100.100.41.64`).
   - Pods on the workstation are on the LAN and see the NUC as `10.8.1.30`.
   - So `http://100.100.41.64:30900` is the likely shared URL, since the workstation host has
     `tailscale0` and flannel masquerades off-cluster traffic. **Unverified**, so first run a
     throwaway pod on the workstation that `curl`s that URL.
   - Prefer the IP over the MagicDNS name inside pods. CoreDNS forwards to the node's
     resolver, and the workstation's Tailscale reports a DNS configuration health warning.
2. Create bucket `dengue-ews` and upload the rule's inputs under the keys Snakemake will expect
   (dry-run with `--default-storage-provider s3` first to print the exact keys). The
   laptop↔workstation link measured 300–1600 ms RTT, so if the data is also on the
   workstation or NUC, upload from there instead of pushing it through Tailscale from the
   laptop.
3. Don't rewrite the rule's `input:`/`output:`. With `--default-storage-provider s3
   --default-storage-prefix s3://dengue-ews`, Snakemake maps every plain path onto the bucket:
   the pod downloads inputs into its `emptyDir` `/workdir`, runs, uploads outputs, and exits.
   Workflow sources are deployed the same way (the executor sets `job_deploy_sources`), so
   they don't need to be baked into the image either.
4. Pass S3 credentials as `SNAKEMAKE_STORAGE_S3_ACCESS_KEY` /
   `SNAKEMAKE_STORAGE_S3_SECRET_KEY` env vars. The executor copies these into the per-run
   Secret from Phase 2 and injects them into each pod. `train.py` calls `wandb.init`, so also
   either pass `--envvars WANDB_API_KEY` (same Secret mechanism) or set `WANDB_MODE=offline`.

**Shared-filesystem assumptions — fixed 2026-09-29 (uncommitted), untested in a pod.** Every
fix routes a file through `snakemake.input`/`snakemake.output`. Those are the only paths
Snakemake stages from and to storage, and locally they resolve to the same paths as before,
so the Docker path is unchanged.
- **Data paths hung off `find_git_root`:** the pod gets the sources without `.git`, so host
  and pod resolved different paths. Fix: a `DATA_ROOT` in the Snakefile, taken from
  `--config data_root=.` on remote runs and defaulting to the git root otherwise.
- **`_tensors_for_production_window` read `best_params.json` at DAG-build time:** fails in
  the pod, before any staging happens. Fix: it now prefers `window_size` from the config.
  Set it in `stgnn_logIR_production.yaml`; the value must match
  `baseline_logIR/best_params.json`, which isn't on the laptop or the workstation.
- **`train.py` read `params.cfg`/`params.best_params` and wrote to `params.results_dir`:**
  params are never remapped, so the pod would read or write the wrong place. Fix: it now
  uses `snakemake.config`, `snakemake.input.*` and `snakemake.output.*`. `train()` takes
  optional explicit paths, and so do `load_tensors`/`load_edge_index` in `tune.py`.
- **`edge_index.pt` was never declared:** it wouldn't be staged at all. Fix: it's now an
  output of `preprocess_stgnn` and an input of `train_stgnn_production`.
- **Production rule was unreachable:** it and `train_stgnn` both produce `best_model.pt`, so
  Snakemake raised `AmbiguousRuleException`. Fix: added
  `ruleorder: train_stgnn_production > train_stgnn`.

**Still a trap:** if `baseline_logIR/best_params.json` is missing, Snakemake silently falls back
to `tune_stgnn` + `train_stgnn`, i.e. a full sweep, instead of failing. Check that the dry run
lists `train_stgnn_production` before running for real.

**Lesson:** pods are ephemeral and don't share local disk. The bind-mount model
(`-v data:/workspace/data`) that has worked everywhere so far stops working entirely, and
hidden local-FS assumptions (like finding `.git`, or reading files in input functions) come to
the surface. Every step has to keep its state outside the pod. Kubeflow Pipelines/Argo
Workflows do this for you behind a nicer abstraction. (The executor also offers
`--kubernetes-persistent-volumes <pvc>:<path>` as a shared-disk escape hatch. Skipping it is
the point of this phase.)

## Phase 4 — GPU resource request

**Implement:** change the rule's resources from `gpu = 1` to

```python
resources:
    gpu = 1,
    gpu_manufacturer = "nvidia",
    scale = False,
```

The executor turns this into `requests`/`limits` of `nvidia.com/gpu: 1` on the container, plus
a `nvidia.com/gpu=present:NoSchedule` toleration. It raises an error if `gpu_manufacturer` is
missing.

**`scale = False` is required.** With the default (`scale` unset, i.e. truthy), the plugin sets
only `requests` and no `limits`. The API server rejects that for an extended resource.
Confirmed with a server-side dry run on 2026-09-29: `Limit must be set for non overcommitable
resources`. The Job create would fail, and Snakemake would report `Failed to create pod`.
`scale = False` makes the plugin set limits too (including CPU/memory limits equal to the
requests). The same resource still works for local runs (`--resources gpu=1`), so this doesn't
break the Docker path. Also consider setting `mem_mb` (it becomes a memory request/limit) and
note `--kubernetes-cpu-scalar` (default 0.95), which shrinks CPU requests so a rule asking
for all 12 cores still fits on the workstation.

The GTX 1050 Ti has 4 GB of VRAM. If training runs out of memory, reduce batch or window size
in the production config. The GPU is Pascal (sm_61) with driver 535, which PyTorch 2.5 / CUDA
12.1 still supports.

**Lesson:** `docker run --gpus all` just grabs whatever GPU exists on the one machine. A
scheduler resource request is a declaration the scheduler uses to *place* the pod on a node
that has one. On `tmu` this is a real decision: the NUC has no GPU, so check with
`kubectl get pod -o wide` that the pod lands on the workstation. As a contrast, set `gpu = 2`
and watch the pod stay `Pending` with an `Insufficient nvidia.com/gpu` event. Verify from
inside the pod (`nvidia-smi` / `torch.cuda.is_available()`), not only from `kubectl describe
pod`. Placement without the Phase 0 runtime fix is exactly the silent CPU fallback that bites
people. The 2026-09-29 probe reproduced it.

## Phase 5 — Run and observe failure handling

**Implement:** from `machine-learning-module/` on the laptop, with `KUBECONFIG=~/.kube/k3s-tmu.yaml` and `kubectl config use-context dengue-ews`:

```bash
snakemake results/STGNN/production_logIR/best_model.pt \
    --configfile config/OpenDengue/stgnn_logIR_production.yaml \
    --executor kubernetes --kubernetes-namespace dengue-ews \
    --container-image dengue-ews:k8s-v1 \
    --default-storage-provider s3 --default-storage-prefix s3://dengue-ews \
    --storage-s3-endpoint-url http://100.100.41.64:30900 \
    --jobs 1 --retries 1
```

Then break it on purpose, and before each run predict the outcome from the plugin's
`check_active_jobs`:
- **`kubectl delete pod <pod>` mid-run.** Jobs are created with `backoffLimit: 0` and
  `restartPolicy: Never`, so Kubernetes itself won't retry. Expect the Job to go `Failed`,
  Snakemake to report a job error, and `--retries 1` to resubmit it as a *new* Job (the Job
  name hashes the attempt number).
- **`kubectl delete job <job>` mid-run.** Now the status read returns 404. In v0.5.1 the
  plugin logs the error and `continue`s without reporting a failure or keeping the job
  active. Watch whether Snakemake notices or quietly stalls.
- **`--kubernetes-omit-job-cleanup`.** Rerun a failure with this flag to keep the finished
  Job/pod around for `kubectl describe`/`logs`. By default the plugin deletes them, and only
  the log copy in `.snakemake/kubernetes-logs/` survives.

**Lesson:** locally, a crash is just your process dying, and you see it immediately. In a
cluster, Snakemake polls *Job status* asynchronously and has to reconcile "the pod
disappeared", "the Job disappeared" and "the container exited nonzero" as distinct failure
modes. Retries happen at two layers (Kubernetes `backoffLimit` vs Snakemake `--retries`), and
you have to know which one is in charge. This is the debugging shift that catches people off
guard the first time a real distributed job silently stalls.

---

**Realistic expectation:** Phase 0 is done. Phases 1–2 are
still pure infrastructure setup. The actual learning starts at Phase 3, and most of it is in
the "expected breakage" list: finding and removing the shared-filesystem assumptions that
the local Docker setup hid.
