# Monthly forecast and cluster operations

How the live risk map is produced each month, how model bundles are built on the k3s cluster, how the explanation plots are generated, and the one-time setup the forecast needs. For running things on your own machine, see [Running the pipelines locally](running-locally.md).

## Monthly inference

Each month the repo-root `Snakefile` runs on the k3s cluster (`profiles/k8s-inference`): EZGEE fetches the newest month of Earth Engine zonal statistics (CHIRPS v3, VIIRS, ERA5-Land), `env_table_v2` updates the province × month table, and `predict_climate_v2` writes this month's and next month's risk index to `site/data/predictions/climate_v2/`. Committing that file to `main` redeploys the site (`.github/workflows/deploy-on-push.yml`).

Since October 2026 the live map uses **climate model v2**, retrained on CHIRPS v3 and VIIRS because Terra MODIS drifts from 2022 and ends in early 2027 and CHIRPS v2 ends after 2026 (`machine-learning-module/config/OpenDengue/stgnn_climate_v2_*.yaml`; 2011 is converted from MODIS, `machine-learning-module/results/conversion/modis_to_viirs_2011.json`). It predicts two months from the same 12-month window. The table below describes the v1 models.

Both models share the cyclical design (sin/cos month encoding, 12-month window) and the same architecture, and both are trained on 2011–2018. The climate-only model drives the live risk map: no incidence data exists after 2018, so it runs on the latest Earth Engine data alone, delta-corrected for the 2019–2026 climate shift. The IR + environment model is the study model it was derived from (its hyperparameters come from the `cyclical_seasonal` sweep); it also forecasts incidence in months when recent case data is supplied.

| | **Climate only** (`climate`) — live map | **IR + environment** (`ir_env`) |
|---|---|---|
| Experiment / production config | — / `stgnn_climate_risk_production.yaml` | `cyclical_seasonal` / `stgnn_logIR_production.yaml` |
| Model inputs | weather, land use, month encoding | past IR, weather, land use, month encoding |
| Needs | Earth Engine | recent incidence + Earth Engine |
| Output | risk index | predicted IR + risk index |
| Runs when | every month (delta-corrected by default) | `site/data/incidence/recent_incidence.csv` exists (format: `site/data/incidence/README.md`) |

Both forecast one month ahead of the last input month. Risk index = (1 + predicted IR) / (1 + the province's 2011–2018 mean IR for that calendar month), so values above 1 mean above that province's usual level for the time of year.

## Producing a bundle

Each model is shipped as an *inference bundle*: `best_model.pt`, `best_params.json`, and `bundle.json`, which holds the node order, scalers, seasonal means and risk baselines the runner needs instead of the training CSV. `export_inference_bundle` packs them into `inference_bundle_<bundle_label>.tar.gz`, named like the release asset.

Bundles are built with the `profiles/k8s` profile on the laptop's single-node k3s cluster, the only cluster with a GPU and with `dengue-ews:k8s-v1` imported. Don't point it at tmu: since the GPU workstation was retired (2026-10-27) tmu is the NUC alone, so job pods there fail with ErrImagePull or, needing `gpu=1`, stay Pending. Every rule runs as a Job in `dengue-ews` and reads and writes the S3 store from `k8s/storage.yaml`, which stays on the NUC. The store needs only two author-supplied inputs: `data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv` and `results/STGNN/cyclical_seasonal/best_params.json`.

The Snakemake controller runs from `ghcr.io/patwuch/dengue-ews-controller`, a small public image that holds only Snakemake and its Kubernetes and S3 plugins ([k8s/controller.Dockerfile](../k8s/controller.Dockerfile)). It submits the Jobs, and the pods run the rules in `dengue-ews:k8s-v1`. It needs an env file with `SNAKEMAKE_STORAGE_S3_ACCESS_KEY` / `_SECRET_KEY` (no `export`), and a kubeconfig whose current context is the `snakemake` ServiceAccount from `k8s/rbac.yaml`.

`ews` defaults to the laptop cluster's kubeconfig (`kubeconfig-laptop` from `k8s/sa-kubeconfig.sh`). `--add-host` lets the container resolve `desktop-ijgfcqd`, the name on the laptop API server's certificate. Set `EWS_KUBECONFIG=~/.config/dengue-ews/kubeconfig` to target tmu, which only monthly inference (`profiles/k8s-inference`) should use.

```bash
ews() {   # ews <dir inside the repo> <snakemake args...>
  local dir=$1; shift
  docker run --rm -it --user $(id -u):$(id -g) --network host \
    --add-host desktop-ijgfcqd:host-gateway \
    --env-file ~/.config/dengue-ews/s3.docker.env \
    -e KUBECONFIG=/kube/config \
    -v "${EWS_KUBECONFIG:-$HOME/.config/dengue-ews/kubeconfig-laptop}":/kube/config:ro \
    -v "$PWD":/workspace -w /workspace/$dir ghcr.io/patwuch/dengue-ews-controller:v1 "$@"
}
# from the repo root
ews machine-learning-module results/STGNN/production_climate_risk/inference_bundle_climate.tar.gz \
    --configfile config/OpenDengue/stgnn_climate_risk_production.yaml --profile profiles/k8s
ews machine-learning-module results/STGNN/production_logIR/inference_bundle_ir_env.tar.gz \
    --configfile config/OpenDengue/stgnn_logIR_production.yaml --profile profiles/k8s
```

The ServiceAccount token expires. On `401 Unauthorized`, mint a new one with an admin context and set it on the kubeconfig's `snakemake` user: `kubectl -n dengue-ews create token snakemake --duration=720h`.

The tarballs land in S3 under the same paths (`s3://dengue-ews/results/STGNN/<name>/`); download them from there before uploading to the release.

## Explaining a model

`explain_attention_stgnn`, `attention_shift_stgnn` and `explain_shap_stgnn` write GAT attention and SHAP Expected Gradients attributions (per lead for two-month models) for a config with held-out test months. For climate model v2 those are the seeded evaluation copies `climate_v2_rain_eval_s{1,2,3}`; all three run on CPU, one seed per invocation:

```bash
ews machine-learning-module results/STGNN/climate_v2_rain_eval_s1/shap_feature_importance.json \
    results/STGNN/climate_v2_rain_eval_s1/attention_shift_interactive.html \
    --configfile config/OpenDengue/stgnn_climate_v2_rain_eval.yaml --profile profiles/k8s \
    --config data_root=. name=climate_v2_rain_eval_s1 bundle_label=climate_v2_rain_eval_s1 seed=1
```

The home page reads `site/data/model_skill_climate_v2.json` and `site/data/xai_climate_v2.json`, the means over the three seeds from `site/scripts/combine_seeds.py`, and `site/xai/attention_shift_climate_v2.html` and `site/data/attention_hubs_climate_v2.json` (`attention_shift_centroid.csv` as JSON) from seed 1, the median seed on skill; the study page reads `machine-learning-module/results/STGNN/climate_v2_rain_eval_s1/`.

Until 2026-10-21 climate model v2 used `precipitation_sum`, CHIRPS rainfall summed over each province's pixels, which scales with province area. It now uses `precipitation_mm`, the monthly total from the spatial mean; `precipitation_sum` stays in the tables for the v1 models trained on it.

## One-time setup

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
3. Produce `site/data/climate_deltas.json` offline with `site/scripts/compute_climate_delta.py` (pass `--bundle` to use the climate bundle as the 2011–2018 baseline, and `--end` at a December, e.g. `--end 2025-12`, so every calendar month covers the same full years) and commit it. Without it, climate mode runs uncorrected and records `bias_corrected: false`.
