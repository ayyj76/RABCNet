#!/usr/bin/env python3
"""Create a compact, provenance-aware record for the CMIG reviewer experiments.

The record deliberately keeps negative controls and separates raw fixed-protocol
comparisons from validation-selected operating-point results.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_cmig/v4"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_metric(path: Path) -> dict:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{") and "dice" in line:
            value = ast.literal_eval(line)
            if isinstance(value, dict):
                return value
    raise ValueError(f"No metric dictionary found in {path}")


def fmt_metrics(metric: dict) -> str:
    return "/".join(f"{metric[key]:.4f}" for key in ("acc", "dice", "jac", "sen", "spe"))


def adaptation_rows() -> tuple[list[dict], list[dict]]:
    base = ROOT / "outputs/review_cmig/v2"
    names = ["no_adaptation", "interaction_only", "decoder_only", "backbone_only", "full_network"]
    rows = []
    for name in names:
        checkpoint = (
            ROOT / "outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt"
            if name == "no_adaptation"
            else base / f"adaptation/{name}/checkpoints/best.pt"
        )
        raw = base / f"evaluations/adaptation/{name}_fixed_raw.txt"
        metric = raw_metric(raw)
        rows.append({"name": name, "metric": metric, "checkpoint": str(checkpoint)})

    fair_names = ["rabc_only", "interaction_rabc", "decoder_rabc", "backbone_rabc", "full_rabc"]
    fair_rows = []
    for name in fair_names:
        summary = load_json(ROOT / f"outputs/review_cmig/v3/rabc_adaptation/{name}/summary.json")
        fair_rows.append({"name": name, "metric": summary["best_test"]})

    bn_unfrozen = raw_metric(OUT / "full_network_bn_unfrozen/fixed_raw.txt")
    fair_rows.append({"name": "full_network_bn_unfrozen", "metric": bn_unfrozen})
    return rows, fair_rows


def sensitivity_rows() -> list[dict]:
    paths = {
        "ISIC-2017": OUT / "sensitivity/isic2017_test.json",
        "ISIC-2018": ROOT / "outputs/review_cmig/v2/evaluations/rabc/sensitivity_test.json",
        "PH2": OUT / "sensitivity/ph2_test.json",
    }
    rows = []
    for dataset, path in paths.items():
        result = load_json(path)
        for item in result["settings"]:
            rows.append(
                {
                    "dataset": dataset,
                    "intercept": item["candidate_intercept"],
                    "uncertainty": item["candidate_uncertainty"],
                    **{key: item["metrics"][key]["mean"] for key in ("acc", "dice", "jac", "sen", "spe")},
                }
            )
    return rows


def calibration_width_rows() -> list[dict]:
    rows = []
    for dataset in ("isic2017", "isic2018", "ph2"):
        for width in (1, 3, 5):
            path = OUT / f"calibration_width/{dataset}_w{width}.json"
            result = load_json(path)
            delta = result["paired_image_bootstrap"]["boundary_band"]
            rows.append(
                {
                    "dataset": dataset.upper() if dataset != "isic2018" else "ISIC-2018",
                    "width": width,
                    "ece_delta": delta["ece"]["rabc_minus_no_rabc_mean"],
                    "ece_low": delta["ece"]["ci95"][0],
                    "ece_high": delta["ece"]["ci95"][1],
                    "brier_delta": delta["brier"]["rabc_minus_no_rabc_mean"],
                    "brier_low": delta["brier"]["ci95"][0],
                    "brier_high": delta["brier"]["ci95"][1],
                    "nll_delta": delta["nll"]["rabc_minus_no_rabc_mean"],
                    "nll_low": delta["nll"]["ci95"][0],
                    "nll_high": delta["nll"]["ci95"][1],
                }
            )
    return rows


def reliability_rows() -> list[dict]:
    rows = []
    for dataset in ("isic2017", "isic2018", "ph2"):
        result = load_json(ROOT / f"outputs/review_cmig/v2/evaluations/reliability_transfer/{dataset}_train.json")
        no = result["no_adaptation"]["summary"]["image_macro"]
        yes = result["interaction_only"]["summary"]["image_macro"]
        boot = result["paired_bootstrap"]
        rows.append(
            {
                "dataset": dataset.upper() if dataset != "isic2018" else "ISIC-2018",
                "no_error_auc": no["error_auc"]["mean"],
                "interaction_error_auc": yes["error_auc"]["mean"],
                "delta_error_auc": boot["error_auc"]["interaction_minus_no_adapt_mean"],
                "error_auc_low": boot["error_auc"]["ci95"][0],
                "error_auc_high": boot["error_auc"]["ci95"][1],
                "delta_uncertainty_separation": boot["uncertainty_separation"]["interaction_minus_no_adapt_mean"],
            }
        )
    return rows


def write_xlsx(path: Path, sheets: dict[str, tuple[list[str], list[list[object]]]]) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    first = True
    for sheet_name, (headers, values) in sheets.items():
        worksheet = workbook.active if first else workbook.create_sheet()
        first = False
        worksheet.title = sheet_name[:31]
        worksheet.append(headers)
        for value in values:
            worksheet.append(value)
        worksheet.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main() -> None:
    raw_rows, fair_rows = adaptation_rows()
    sens = sensitivity_rows()
    widths = calibration_width_rows()
    reliability = reliability_rows()

    lines = [
        "# CMIG Reviewer-Response Experiments (v4)",
        "",
        "This record contains the completed reviewer-response experiments. Test masks are used only for post-hoc evaluation; fixed raw comparisons use threshold 0.30, no TTA, and no morphology unless stated otherwise.",
        "",
        "## Completed Evidence",
        "",
        "1. ISIC-2018 adaptation-strategy controls compare no adaptation, interaction-only, decoder-only, backbone-only, and full-network updates under the same two-epoch Stage-2 budget.",
        "2. A fair RABC-enabled matrix repeats the adaptation comparison with RABC enabled in every trained variant.",
        "3. A true BN-running-statistics control allows BN statistics to update during full-network adaptation; it is separated from the earlier BN-frozen run.",
        "4. RABC coefficient sensitivity is evaluated on all three datasets using one fixed checkpoint per dataset and three coefficient pairs.",
        "5. Boundary-band calibration is repeated at widths 1, 3, and 5 pixels on all three datasets with paired image-bootstrap intervals.",
        "",
        "## 1. Adaptation Matrix",
        "",
        "The no-RABC matrix is the primary fixed raw comparison. Values are ACC/DICE/JAC/SEN/SPE.",
        "",
        "| Strategy | ACC/DICE/JAC/SEN/SPE |",
        "|---|---|",
    ]
    for row in raw_rows:
        lines.append(f"| {row['name']} | {fmt_metrics(row['metric'])} |")
    lines += [
        "",
        "The RABC-enabled matrix is a fairness control, not a claim that every added branch is beneficial:",
        "",
        "| Strategy | ACC/DICE/JAC/SEN/SPE |",
        "|---|---|",
    ]
    for row in fair_rows:
        lines.append(f"| {row['name']} | {fmt_metrics(row['metric'])} |")
    lines += [
        "",
        "Interpretation: decoder, backbone, and full-network updates degrade substantially under the short low-resource budget. In the RABC-enabled matrix, RABC-only is marginally higher than interaction+RABC, so the evidence supports a conservative restricted-update design but does not establish interaction-only adaptation as the unique optimum.",
        "The true BN-unfrozen full-network result remains poor (DICE/JAC 0.6336/0.4966), close to the BN-frozen control (0.6342/0.4989). Thus the full-network degradation is not explained by BN running-statistics drift alone.",
        "",
        "## 2. RABC Coefficient Sensitivity",
        "",
        "The three pairs are (candidate intercept, uncertainty weight) = (0.25, 0.75), (0.35, 0.65), and (0.45, 0.55). Only inference coefficients change; the model checkpoint and fixed raw protocol are unchanged.",
        "",
        "| Dataset | Pair | DICE | JAC |",
        "|---|---|---:|---:|",
    ]
    for row in sens:
        lines.append(f"| {row['dataset']} | {row['intercept']:.2f}/{row['uncertainty']:.2f} | {row['dice']:.6f} | {row['jac']:.6f} |")
    lines += [
        "",
        "Interpretation: all three datasets show changes only at approximately 1e-5--1e-4 scale. This supports local numerical robustness of the fixed parameterization, not coefficient optimality or a large metric contribution.",
        "",
        "## 3. Boundary-Band Calibration Stability",
        "",
        "The table reports RABC minus no-RABC paired image-bootstrap mean differences for the ground-truth-defined boundary band. Negative values are better. The boundary band is evaluation-only and is never an inference input.",
        "",
        "| Dataset | Width | ECE delta [95% CI] | Brier delta [95% CI] | NLL delta [95% CI] |",
        "|---|---:|---|---|---|",
    ]
    for row in widths:
        lines.append(
            f"| {row['dataset']} | {row['width']} | {row['ece_delta']:.6f} [{row['ece_low']:.6f}, {row['ece_high']:.6f}] | "
            f"{row['brier_delta']:.6f} [{row['brier_low']:.6f}, {row['brier_high']:.6f}] | "
            f"{row['nll_delta']:.6f} [{row['nll_low']:.6f}, {row['nll_high']:.6f}] |"
        )
    lines += [
        "",
        "Interpretation: boundary-band calibration improvements remain in the same direction across widths 1, 3, and 5 on all three datasets. Whole-image calibration remains dataset-dependent and near-neutral in some cases; it should not be presented as uniformly improved.",
        "",
        "## 4. Reliability-Transfer Diagnostic",
        "",
        "The interaction branch receives target pseudo-labels during training-side evaluation; ground-truth masks are used only to measure whether uncertainty ranks pseudo-label errors after the fact.",
        "",
        "| Dataset | Error-AUC no adaptation | Error-AUC interaction | Delta with 95% CI | Delta uncertainty separation |",
        "|---|---:|---:|---|---:|",
    ]
    for row in reliability:
        lines.append(
            f"| {row['dataset']} | {row['no_error_auc']:.4f} | {row['interaction_error_auc']:.4f} | "
            f"{row['delta_error_auc']:.4f} [{row['error_auc_low']:.4f}, {row['error_auc_high']:.4f}] | "
            f"{row['delta_uncertainty_separation']:.4f} |"
        )
    lines += [
        "",
        "Interpretation: the diagnostic is dataset-dependent: error ranking improves on ISIC-2017, changes only slightly on ISIC-2018, and decreases on PH2. Therefore the paper may claim conditional reliability-transfer evidence, but not universal improvement across domains.",
        "",
        "## 5. Evidence Boundary",
        "",
        "- Supported: broader image-path updates are fragile under the tested short low-resource protocol; learned RABC improves boundary-band calibration consistently across band widths; the reported RABC coefficient setting is locally stable.",
        "- Not supported: interaction-only adaptation as the unique best subset; universal improvement of the interaction reliability diagnostic; uniformly improved whole-image calibration; strict reproduction of the original RPI-Net paper without its official implementation or per-image predictions.",
        "- Important implementation fact: `model(x, None)` bypasses pseudo-label interaction at image-only inference. Consequently, interaction-only adaptation alone leaves deployed logits effectively unchanged; any claim about its deployment effect must be phrased as training-side reliability modeling rather than direct image-path adaptation.",
        "",
        "## 6. Provenance",
        "",
        "- Existing reviewer runs: `outputs/review_cmig/v2/`.",
        "- Fair RABC adaptation matrix: `outputs/review_cmig/v3/rabc_adaptation/`.",
        "- New v4 runs: `outputs/review_cmig/v4/`.",
        "- Evaluators: `scripts/evaluate_rabc_sensitivity.py`, `scripts/evaluate_review_calibration.py`, and `scripts/evaluate_reliability_transfer.py`.",
        "- This summary is generated by `scripts/summarize_cmig_v4_experiments.py`.",
    ]
    out_md = OUT / "CMIG_EXPERIMENTS.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_dir = OUT / "指标"
    for dataset in ("ISIC2017", "ISIC2018", "PH2"):
        sens_rows = [r for r in sens if r["dataset"].replace("-", "").upper() == dataset]
        width_rows = [r for r in widths if r["dataset"].replace("-", "").upper() == dataset]
        rel_rows = [r for r in reliability if r["dataset"].replace("-", "").upper() == dataset]
        sheets = {
            "RABC_sensitivity": (
                ["dataset", "intercept", "uncertainty", "ACC", "DICE", "JAC", "SEN", "SPE"],
                [[r["dataset"], r["intercept"], r["uncertainty"], r["acc"], r["dice"], r["jac"], r["sen"], r["spe"]] for r in sens_rows],
            ),
            "calibration_width": (
                ["dataset", "width", "ECE_delta", "ECE_low", "ECE_high", "Brier_delta", "Brier_low", "Brier_high", "NLL_delta", "NLL_low", "NLL_high"],
                [[r["dataset"], r["width"], r["ece_delta"], r["ece_low"], r["ece_high"], r["brier_delta"], r["brier_low"], r["brier_high"], r["nll_delta"], r["nll_low"], r["nll_high"]] for r in width_rows],
            ),
            "reliability_transfer": (
                ["dataset", "no_error_auc", "interaction_error_auc", "delta_error_auc", "error_auc_low", "error_auc_high", "delta_uncertainty_separation"],
                [[r["dataset"], r["no_error_auc"], r["interaction_error_auc"], r["delta_error_auc"], r["error_auc_low"], r["error_auc_high"], r["delta_uncertainty_separation"]] for r in rel_rows],
            ),
        }
        write_xlsx(metrics_dir / f"{dataset}.xlsx", sheets)

    print(out_md)


if __name__ == "__main__":
    main()
