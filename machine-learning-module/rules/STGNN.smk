# ============================================================
# STGNN rules — OpenDengue SEA
# Scripts: src/OpenDengue/STGNN/
#
# Invoke:
#   snakemake results/STGNN/<name>/metrics.json \
#       --configfile config/<experiment>.yaml \
#       --cores 4 --resources gpu=1
#
# Scripts receive all experiment parameters via snakemake.config.
# ============================================================

import json

_name = config.get("name", "")

_window_sizes = (
    config.get("tune", {})
          .get("search_space", {})
          .get("stgnn", {})
          .get("window_size", {})
          .get("choices", [0])
)

_ML_PROCESSED = DATA_ROOT / "data" / "processed" / "machine-learning"
_RESULTS      = f"results/STGNN/{_name}"


rule preprocess_stgnn:
    message:
        "Preprocessing for STGNN with experiment name '{_name}' and window sizes {_window_sizes}."
    input:
        str(MERGED_DENGUE_ENV_DATA),
    output:
        tensors = expand(
            str(_ML_PROCESSED / "STGNN/{name}/window_{w}/tensors.pt"),
            name=_name,
            w=_window_sizes,
        ),
        scaler  = str(_ML_PROCESSED / f"STGNN/{_name}/preprocessing_params.json"),
        edge_index = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
        bundle  = str(_ML_PROCESSED / f"STGNN/{_name}/inference_bundle.json"),
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        f"{workflow.basedir}/src/OpenDengue/STGNN/preprocess/pipeline.py"

checkpoint tune_stgnn:
    input:
        tensors = expand(
            str(_ML_PROCESSED / "STGNN/{name}/window_{w}/tensors.pt"),
            name = _name,
            w    = _window_sizes,
        ),
        scaler  = str(_ML_PROCESSED / f"STGNN/{_name}/preprocessing_params.json"),
        edge_index = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
    output:
        best_params = f"{_RESULTS}/best_params.json",
    resources:
        gpu = 1,
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        "../src/OpenDengue/STGNN/tune.py"


def _tensors_for_best_window(wildcards):
    """Resolve the tensor file for the window size tune_stgnn actually picked,
    so Snakemake knows about this dependency instead of it being an implicit
    read inside train.py/test.py/explain_*.py."""
    best_params_path = checkpoints.tune_stgnn.get(**wildcards).output.best_params
    with open(best_params_path) as f:
        window = json.load(f)["window_size"]
    return str(_ML_PROCESSED / f"STGNN/{_name}/window_{window}/tensors.pt")


def _tensors_for_production_window(wildcards):
    """Same as above, but for train_stgnn_production: best_params.json comes
    from a previously completed experiment (_params_source), not one produced
    within this workflow run, so no checkpoint indirection is needed.

    Prefer `window_size` from the config: this function runs while the DAG is
    built, before any inputs are staged, so on an executor without a shared
    filesystem (e.g. kubernetes + S3) best_params.json isn't on local disk yet."""
    window = config.get("window_size")
    if window is None:
        with open(f"results/STGNN/{_params_source}/best_params.json") as f:
            window = json.load(f)["window_size"]
    return str(_ML_PROCESSED / f"STGNN/{_name}/window_{window}/tensors.pt")


rule train_stgnn:
    input:
        best_params = f"{_RESULTS}/best_params.json",
        tensors     = _tensors_for_best_window,
        edge_index  = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
    output:
        checkpoint = f"{_RESULTS}/best_model.pt",
        losses     = f"{_RESULTS}/train_val_losses.json",
        loss_curve = f"{_RESULTS}/loss_curves.png",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        results_dir = _RESULTS,
        best_params = f"{_RESULTS}/best_params.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/train.py"


rule test_stgnn:
    input:
        model       = f"{_RESULTS}/best_model.pt",
        best_params = f"{_RESULTS}/best_params.json",
        tensors     = _tensors_for_best_window,
        edge_index  = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
    output:
        predictions = f"{_RESULTS}/test_predictions.npz",
        metrics     = f"{_RESULTS}/metrics.json",
        pred_plot   = f"{_RESULTS}/predictions.png",
        scatter     = f"{_RESULTS}/scatter.png",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        results_dir = _RESULTS,
        best_params = f"{_RESULTS}/best_params.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/test.py"


rule choropleth_stgnn:
    input:
        predictions = f"{_RESULTS}/test_predictions.npz",
        best_params = f"{_RESULTS}/best_params.json",
        geom        = str(GEOM_PATH),
        csv_data    = str(MERGED_DENGUE_ENV_DATA),
    output:
        pred_vs_actual = f"{_RESULTS}/pred_vs_actual.png",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        results_dir = _RESULTS,
        csv_path    = str(MERGED_DENGUE_ENV_DATA),
        geom_path   = str(GEOM_PATH),
    script:
        "../src/OpenDengue/STGNN/choropleth.py"


rule explain_attention_stgnn:
    message:
        "Visualising GAT attention weights for experiment '{_name}'."
    input:
        model       = f"{_RESULTS}/best_model.pt",
        best_params = f"{_RESULTS}/best_params.json",
        geom        = str(GEOM_PATH),
        csv_data    = str(MERGED_DENGUE_ENV_DATA),
        tensors     = _tensors_for_best_window,
        edge_index  = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
    output:
        weights       = f"{_RESULTS}/attention_weights.npz",
        graph         = f"{_RESULTS}/attention_graph.png",
        over_time     = f"{_RESULTS}/attention_over_time.png",
        self_vs_degree = f"{_RESULTS}/self_attention_vs_degree.png",
        interactive   = f"{_RESULTS}/attention_graph_interactive.html",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        results_dir = _RESULTS,
        best_params = f"{_RESULTS}/best_params.json",
        csv_path    = str(MERGED_DENGUE_ENV_DATA),
        geom_path   = str(GEOM_PATH),
    script:
        "../src/OpenDengue/STGNN/explain_attention.py"


rule explain_shap_stgnn:
    message:
        "Computing SHAP Expected Gradients feature attributions for experiment '{_name}'."
    input:
        model       = f"{_RESULTS}/best_model.pt",
        best_params = f"{_RESULTS}/best_params.json",
        csv_data    = str(MERGED_DENGUE_ENV_DATA),
        tensors     = _tensors_for_best_window,
        edge_index  = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
    output:
        attributions = f"{_RESULTS}/shap_attributions.npz",
        feature_plot = f"{_RESULTS}/shap_feature_importance.png",
        heatmap      = f"{_RESULTS}/shap_node_time_heatmap.png",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        results_dir = _RESULTS,
        best_params = f"{_RESULTS}/best_params.json",
        csv_path    = str(MERGED_DENGUE_ENV_DATA),
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/explain_shap.py"


# ---------------------------------------------------------------------------
# Production training rule — only active when best_params_source is set.
# ---------------------------------------------------------------------------

if config.get("best_params_source"):
    _params_source = config["best_params_source"]

    # Both rules produce {_RESULTS}/best_model.pt; without this Snakemake raises
    # AmbiguousRuleException whenever both are viable.
    ruleorder: train_stgnn_production > train_stgnn

    rule train_stgnn_production:
        input:
            best_params = f"results/STGNN/{_params_source}/best_params.json",
            tensors     = _tensors_for_production_window,
            edge_index  = str(_ML_PROCESSED / f"STGNN/{_name}/edge_index.pt"),
            scaler      = str(_ML_PROCESSED / f"STGNN/{_name}/preprocessing_params.json"),
        output:
            checkpoint = f"{_RESULTS}/best_model.pt",
            losses     = f"{_RESULTS}/train_val_losses.json",
            loss_curve = f"{_RESULTS}/loss_curves.png",
        params:
            cfg         = lambda wc: workflow.configfiles[-1],
            results_dir = _RESULTS,
            best_params = f"results/STGNN/{_params_source}/best_params.json",
        resources:
            gpu = 1,
            # Read by the kubernetes executor: gpu_manufacturer maps gpu to
            # nvidia.com/gpu, and scale=False sets limits alongside requests,
            # which the API server requires for extended resources.
            gpu_manufacturer = "nvidia",
            scale = False,
        script:
            "../src/OpenDengue/STGNN/train.py"


# ---------------------------------------------------------------------------
# Inference bundle — what the monthly GitHub Actions run downloads.
#
#   snakemake results/STGNN/<name>/inference_bundle \
#       --configfile config/OpenDengue/<production config>.yaml --cores 4 --resources gpu=1
#
# Then publish it (see README "Monthly inference"):
#   tar -czf inference_bundle_<climate|ir_env|ir_only>.tar.gz -C results/STGNN/<name> inference_bundle
#   gh release upload inference-bundles inference_bundle_<...>.tar.gz --clobber
# ---------------------------------------------------------------------------

_bundle_params = (
    f"results/STGNN/{config['best_params_source']}/best_params.json"
    if config.get("best_params_source") else f"{_RESULTS}/best_params.json"
)

rule export_inference_bundle:
    input:
        model       = f"{_RESULTS}/best_model.pt",
        best_params = _bundle_params,
        bundle      = str(_ML_PROCESSED / f"STGNN/{_name}/inference_bundle.json"),
    output:
        directory(f"{_RESULTS}/inference_bundle"),
    shell:
        "mkdir -p {output} && "
        "cp {input.model} {output}/best_model.pt && "
        "cp {input.best_params} {output}/best_params.json && "
        "cp {input.bundle} {output}/bundle.json"
