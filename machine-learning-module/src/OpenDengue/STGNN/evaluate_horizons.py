"""Score each lead of a multi-month STGNN on the held-out test months (Snakemake script).

For every lead (1 = the month after the input window, 2 = the month after that), compares the
model with the seasonal baseline the risk index divides by: each province's mean log1p(IR) for
that calendar month over train + validation (bundle.json target_baseline). All scores are in
log1p(IR) space, on province-months with reported incidence.

    skill_vs_seasonal = 1 − MSE(model) / MSE(seasonal baseline)
                        > 0: the model beats "the usual level for this month".
    anomaly_r         = correlation of (prediction − baseline) with (observed − baseline):
                        does the model see departures from the usual level coming?

Inputs mirror inference: every node is treated as observed (mask = 1), as run_inference.py does.
"""
import json
from pathlib import Path

import numpy as np
import torch

from model import STGATGRU
from tune import load_tensors, load_edge_index


def baseline_matrix(bundle: dict, months: np.ndarray) -> np.ndarray:
    """(W, N) seasonal baseline in log1p space for the given target calendar months."""
    base = bundle["target_baseline"]
    out = np.empty((len(months), len(bundle["nodes"])))
    for n, node in enumerate(bundle["nodes"]):
        per_node = base["by_node_month"].get(node["name"], {})
        for w, m in enumerate(months):
            v = per_node.get(str(int(m)))
            out[w, n] = v if v is not None else base["by_month"][str(int(m))]
    return out


def main(inp, out, cfg):
    bundle = json.loads(Path(inp["bundle"]).read_text())
    params = json.loads(Path(inp["best_params"]).read_text())
    h = int(bundle.get("horizon", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensors = load_tensors(cfg, params["window_size"], Path(inp["tensors"]))
    if "inference_x" not in tensors:
        raise ValueError("No test split in these tensors; evaluate a config with test_months > 0.")
    x, y, mask = tensors["inference_x"], tensors["inference_y"], tensors["inference_mask"]
    first_months = tensors["inference_months"].numpy()          # calendar month of lead 1
    if h == 1:
        y, mask = y.reshape(*y.shape[:2], 1), mask.reshape(*mask.shape[:2], 1)

    model = STGATGRU(
        in_channels  = x.shape[-1],
        gat1_hidden  = params["gat1_hidden"],
        gat1_heads   = params["gat1_heads"],
        mlp_hidden   = params.get("mlp_hidden", params["gat1_hidden"] * params["gat1_heads"]),
        mlp_layers   = params["mlp_layers"],
        gat2_hidden  = params["gat2_hidden"],
        gat2_heads   = params["gat2_heads"],
        gru_hidden   = params.get("gru_hidden", params["gat2_hidden"]),
        pred_horizon = h,
        dropout      = params.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(torch.load(inp["model"], map_location=device, weights_only=True))
    model.eval()
    edge_index = load_edge_index(cfg, device, Path(inp["edge_index"]))

    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            xb = x[i:i + 8].to(device)
            preds.append(model(xb, edge_index, mask=torch.ones(xb.shape[0], xb.shape[2], device=device)).cpu())
    pred = torch.cat(preds).numpy()                               # (W, N, h), model space

    sc = bundle["inc_scaler"]
    to_log1p = (lambda z: z * sc["std"] + sc["mean"]) if sc else (lambda z: z)
    pred, obs = to_log1p(pred), to_log1p(y.numpy())
    mask = mask.numpy().astype(bool)

    report = {"model": bundle["name"], "horizon": h, "test_windows": int(len(x)), "leads": {}}
    for k in range(h):
        months = (first_months + k - 1) % 12 + 1
        base = baseline_matrix(bundle, months)
        m = mask[..., k]
        p, o, b = pred[..., k][m], obs[..., k][m], base[m]
        mse_model, mse_base = float(np.mean((p - o) ** 2)), float(np.mean((b - o) ** 2))
        report["leads"][str(k + 1)] = {
            "province_months":   int(m.sum()),
            "mse_model":         mse_model,
            "mse_seasonal":      mse_base,
            "mae_model":         float(np.mean(np.abs(p - o))),
            "mae_seasonal":      float(np.mean(np.abs(b - o))),
            "skill_vs_seasonal": 1.0 - mse_model / mse_base,
            "anomaly_r":         float(np.corrcoef(p - b, o - b)[0, 1]),
        }
    report["beats_seasonal_at_every_lead"] = all(v["skill_vs_seasonal"] > 0 for v in report["leads"].values())

    Path(out["metrics"]).parent.mkdir(parents=True, exist_ok=True)
    Path(out["metrics"]).write_text(json.dumps(report, indent=2))
    for k, v in report["leads"].items():
        print(f"lead {k}: skill vs seasonal {v['skill_vs_seasonal']:+.3f}  anomaly r {v['anomaly_r']:+.3f}  "
              f"MAE {v['mae_model']:.3f} (seasonal {v['mae_seasonal']:.3f})  n={v['province_months']}")


if "snakemake" in dir():
    main(dict(snakemake.input.items()), dict(snakemake.output.items()), dict(snakemake.config))  # noqa: F821
