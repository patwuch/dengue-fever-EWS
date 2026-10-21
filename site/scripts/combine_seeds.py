"""
Mean skill and SHAP attributions over the seeded evaluation runs of a model, for the site.

One seed's numbers move with the initialisation (climate model v2 with rainfall in mm: lead-1
skill 0.37–0.50 over three seeds), so the site shows the mean, with the spread alongside.

Inputs are the results/STGNN/<name>_s<seed>/ directories, as downloaded from the S3 store:
    horizon_metrics.json            (evaluate_horizons_stgnn)
    shap_feature_importance.json    (explain_shap_stgnn)

Usage (from the repo root):
    python site/scripts/combine_seeds.py --runs results/STGNN/climate_v2_rain_eval_s{1,2,3} \\
        --skill site/data/model_skill_climate_v2.json --xai site/data/xai_climate_v2.json
"""

import argparse
import json
import pathlib

import numpy as np

SKILL_KEYS = ["mse_model", "mse_seasonal", "mae_model", "mae_seasonal", "skill_vs_seasonal", "anomaly_r"]


def combine_skill(metrics: list[dict], names: list[str]) -> dict:
    leads = {}
    for lead in metrics[0]["leads"]:
        rows = [m["leads"][lead] for m in metrics]
        out = {"province_months": rows[0]["province_months"]}
        for k in SKILL_KEYS:
            vals = [r[k] for r in rows]
            out[k] = float(np.mean(vals))
            if k in ("skill_vs_seasonal", "anomaly_r"):
                out[f"{k}_by_seed"] = vals
        leads[lead] = out
    return {
        "model": "climate model v2",
        "runs": names,
        "evaluated_on": f"{metrics[0]['test_windows']} held-out months, 2017-11 to 2018-12 "
                        "(trained on earlier months)",
        "skill_definition": "1 - MSE(model) / MSE(each province's 2011-2018 mean log IR for that "
                            "calendar month); mean over the runs, per-seed values alongside",
        "leads": leads,
    }


def combine_xai(xais: list[dict], names: list[str]) -> dict:
    if len({tuple(x["window_months"]) for x in xais}) > 1:
        raise ValueError("Runs explained different windows; their shares aren't comparable.")
    leads = {}
    for lead in xais[0]["leads"]:
        per_run = [{r["feature"]: r for r in x["leads"][lead]} for x in xais]
        features = list(per_run[0])
        if any(set(p) != set(features) for p in per_run):
            raise ValueError(f"Runs have different features at lead {lead}.")
        rows = []
        for f in features:
            share = [p[f]["share"] for p in per_run]
            rows.append({
                "feature":      f,
                "share":        float(np.mean(share)),
                "share_sd":     float(np.std(share, ddof=1)) if len(share) > 1 else 0.0,
                "mean_signed":  float(np.mean([p[f]["mean_signed"] for p in per_run])),
                "share_by_lag": np.mean([p[f]["share_by_lag"] for p in per_run], axis=0).tolist(),
            })
        leads[lead] = sorted(rows, key=lambda r: -r["share"])
    return {
        "explained_windows": xais[0]["explained_windows"],
        "window_months":     xais[0]["window_months"],
        "lags":              xais[0]["lags"],
        "leads":             leads,
        "model":             names,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs", nargs="+", required=True, type=pathlib.Path)
    ap.add_argument("--skill", required=True, type=pathlib.Path)
    ap.add_argument("--xai", required=True, type=pathlib.Path)
    args = ap.parse_args()

    names = [r.name for r in args.runs]
    skill = combine_skill([json.loads((r / "horizon_metrics.json").read_text()) for r in args.runs], names)
    xai = combine_xai([json.loads((r / "shap_feature_importance.json").read_text()) for r in args.runs], names)
    args.skill.write_text(json.dumps(skill, indent=2))
    args.xai.write_text(json.dumps(xai, indent=2))
    for lead, row in skill["leads"].items():
        print(f"lead {lead}: skill {row['skill_vs_seasonal']:.3f} "
              f"({', '.join(f'{v:.3f}' for v in row['skill_vs_seasonal_by_seed'])})")
    print("lead 1 top inputs:", ", ".join(f"{r['feature']} {r['share']:.1%}±{r['share_sd']:.1%}"
                                          for r in xai["leads"]["1"][:6]))
    print(f"Saved: {args.skill}, {args.xai}")


if __name__ == "__main__":
    main()
