#!/usr/bin/env python3
"""Summarize reviewer-response experiments with provenance and parameter counts."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rpi_net.model import RPINet
from train_rpi_net import model_kwargs_from_cfg


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_config(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    return payload.get("config", {}) if isinstance(payload, dict) else {}


def parameter_counts(checkpoint: Path, adaptation: bool = True) -> tuple[int, int]:
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model = RPINet(**model_kwargs_from_cfg(config))
    prefixes = tuple(str(x) for x in config.get("train", {}).get("trainable_prefixes", []) if str(x))
    freeze_path = bool(config.get("train", {}).get("freeze_segmentation_path", False))
    total = sum(p.numel() for p in model.parameters())
    trainable = 0
    if not adaptation:
        return 0, total
    for name, parameter in model.named_parameters():
        if not freeze_path or any(name == p or name.startswith(f"{p}.") for p in prefixes):
            trainable += parameter.numel()
    return trainable, total


def raw_metric(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "dice" in line:
            # Training/evaluation scripts print Python dict reprs, which may
            # contain single-quoted strings and Python booleans.
            value = ast.literal_eval(line)
            if not isinstance(value, dict):
                raise ValueError(f"Metric line is not a dictionary in {path}")
            return value
    raise ValueError(f"No raw metric line found in {path}")


def adaptation_rows(root: Path) -> list[dict]:
    base = root / "outputs/review_cmig/v2"
    manifest = load_json(base / "experiment_manifest.json")
    rows = []
    for key, item in manifest["runs"].items():
        if not key.startswith("adaptation/"):
            continue
        name = key.split("/", 1)[1]
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.exists():
            continue
        if name == "no_adaptation":
            raw_path = base / "evaluations/adaptation/no_adaptation_fixed_raw.txt"
        else:
            raw_path = base / f"evaluations/adaptation/{name}_fixed_raw.txt"
        trainable, total = parameter_counts(checkpoint, adaptation=name != "no_adaptation")
        metric = raw_metric(raw_path)
        rows.append({
            "strategy": name,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100.0 * trainable / total if total else 0.0,
            "raw": {k: metric[k] for k in ("acc", "dice", "jac", "sen", "spe")},
            "selected": load_json(base / f"evaluations/adaptation/selection/{name}.json")["summary"]["best"],
            "checkpoint": str(checkpoint),
        })
    return sorted(rows, key=lambda row: ["no_adaptation", "interaction_only", "decoder_only", "backbone_only", "full_network"].index(row["strategy"]))


def sensitivity_rows(root: Path) -> list[dict]:
    path = root / "outputs/review_cmig/v2/evaluations/rabc/sensitivity_test.json"
    result = load_json(path)
    rows = []
    for item in result["settings"]:
        rows.append({
            "intercept": item["candidate_intercept"],
            "uncertainty": item["candidate_uncertainty"],
            "metrics": {k: item["metrics"][k]["mean"] for k in ("acc", "dice", "jac", "sen", "spe")},
        })
    return rows


def calibration_rows(root: Path) -> list[dict]:
    path = root / "outputs/review_cmig/v2/evaluations"
    rows = []
    for dataset in ("isic2017", "isic2018", "ph2"):
        candidates = sorted((path / "rabc").glob(f"{dataset}_calibration_test*.json"))
        if not candidates:
            continue
        # Prefer the explicitly versioned rerun when both the old and new
        # ISIC-2018 records are present.
        result_path = candidates[-1]
        result = load_json(result_path)
        for region in ("full_image", "boundary_band"):
            no = result["no_rabc"]["summary"][region]
            yes = result["rabc"]["summary"][region]
            rows.append({
                "dataset": dataset,
                "region": region,
                "no_rabc": {k: no[k]["macro_mean"] for k in ("ece", "brier", "nll")},
                "rabc": {k: yes[k]["macro_mean"] for k in ("ece", "brier", "nll")},
                "paired_bootstrap": result["paired_image_bootstrap"][region],
            })
    return rows


def format_md(adaptation: list[dict], sensitivity: list[dict], calibration: list[dict]) -> str:
    lines = [
        "# CMIG Reviewer-Response Experiment Record",
        "",
        "This record summarizes experiments completed before manuscript revision. It is not a claim that every alternative adaptation strategy is competitive; negative results are retained.",
        "",
        "## 1. ISIC-2018 Adaptation Strategy Comparison",
        "",
        "Protocol: same source checkpoint, two Stage-2 epochs, same pseudo-label set and loss configuration; raw image-only inference uses threshold 0.30 without TTA or morphology.",
        "",
        "| Strategy | Trainable params | Trainable (%) | ACC | DICE | JAC | SEN | SPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in adaptation:
        raw = row["raw"]
        lines.append(
            f"| {row['strategy']} | {row['trainable_params']:,} | {row['trainable_pct']:.2f} | "
            f"{raw['acc']:.4f} | {raw['dice']:.4f} | {raw['jac']:.4f} | {raw['sen']:.4f} | {raw['spe']:.4f} |"
        )
    lines += [
        "",
        "Interpretation: interaction-only adaptation preserves the no-adaptation operating point and has the best raw DICE/JAC among the tested alternatives. Updating the decoder, backbone, or full network under the same short low-resource budget causes substantial degradation; these negative controls support the parameter-restricted design but do not prove a universal causal explanation for domain shift.",
        "",
        "## 2. RABC Coefficient Sensitivity",
        "",
        "Protocol: one fixed RABC checkpoint, ISIC-2018 test split, threshold 0.30, no TTA or morphology; only the candidate-intercept/uncertainty coefficients are changed at inference.",
        "",
        "| Candidate intercept | Uncertainty weight | ACC | DICE | JAC | SEN | SPE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        m = row["metrics"]
        lines.append(
            f"| {row['intercept']:.2f} | {row['uncertainty']:.2f} | {m['acc']:.6f} | {m['dice']:.6f} | "
            f"{m['jac']:.6f} | {m['sen']:.6f} | {m['spe']:.6f} |"
        )
    lines += [
        "",
        "Interpretation: the tested coefficient pairs produce nearly identical overlap metrics. The current result supports numerical robustness of the chosen parameterization, not a large performance contribution from hand-tuned coefficients.",
        "",
        "## 3. Calibration Evaluation",
        "",
        "Protocol: raw image-only probabilities before thresholding, TTA, or morphology. The boundary band is defined from the ground-truth mask only for evaluation; it is not an inference input. Lower ECE/Brier/NLL is better.",
        "",
        "| Dataset | Region | no-RABC ECE/Brier/NLL | RABC ECE/Brier/NLL |",
        "|---|---|---|---|",
    ]
    for row in calibration:
        no = row["no_rabc"]
        yes = row["rabc"]
        lines.append(
            f"| {row['dataset']} | {row['region']} | "
            f"{no['ece']:.4f}/{no['brier']:.4f}/{no['nll']:.4f} | "
            f"{yes['ece']:.4f}/{yes['brier']:.4f}/{yes['nll']:.4f} |"
        )
    lines += [
        "",
        "Calibration interpretation should remain localized: boundary-band changes are the primary evidence for RABC, while whole-image effects may be near-neutral. Paired image-bootstrap intervals are stored in the JSON files and should be reported if this analysis is included in the manuscript.",
        "",
        "## 4. Provenance",
        "",
        "- Raw experiment artifacts: `outputs/review_cmig/v2/`.",
        "- Generated configurations: `configs/generated/review_cmig/v2/`.",
        "- Adaptation runner: `scripts/train_rpi_net.py`.",
        "- Calibration evaluator: `scripts/evaluate_review_calibration.py`.",
        "- This summary is generated by `scripts/summarize_review_experiments.py`.",
    ]
    return "\n".join(lines) + "\n"


def write_xlsx(path: Path, rows: list[dict], sheet: str, headers: list[str], values: list[list[object]]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required for reviewer experiment metric exports") from exc
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet[:31]
    worksheet.append(headers)
    for value in values:
        worksheet.append(value)
    worksheet.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/review_cmig/v2/reviewer_experiment_summary.md")
    args = parser.parse_args()
    adaptation = adaptation_rows(ROOT)
    sensitivity = sensitivity_rows(ROOT)
    calibration = calibration_rows(ROOT)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_md(adaptation, sensitivity, calibration), encoding="utf-8")

    metrics_root = ROOT / "outputs/review_cmig/v2/metrics"
    write_xlsx(
        metrics_root / "ISIC2018.xlsx",
        adaptation,
        "adaptation",
        ["strategy", "trainable_params", "trainable_pct", "ACC", "DICE", "JAC", "SEN", "SPE"],
        [[r["strategy"], r["trainable_params"], r["trainable_pct"], *[r["raw"][k] for k in ("acc", "dice", "jac", "sen", "spe")]] for r in adaptation],
    )
    write_xlsx(
        metrics_root / "RABC_sensitivity.xlsx",
        sensitivity,
        "sensitivity",
        ["candidate_intercept", "candidate_uncertainty", "ACC", "DICE", "JAC", "SEN", "SPE"],
        [[r["intercept"], r["uncertainty"], *[r["metrics"][k] for k in ("acc", "dice", "jac", "sen", "spe")]] for r in sensitivity],
    )
    for dataset in ("isic2017", "isic2018", "ph2"):
        dataset_rows = [r for r in calibration if r["dataset"] == dataset]
        write_xlsx(
            metrics_root / f"{dataset.upper()}.xlsx",
            dataset_rows,
            "calibration",
            ["dataset", "region", "no_RABC_ECE", "no_RABC_Brier", "no_RABC_NLL", "RABC_ECE", "RABC_Brier", "RABC_NLL"],
            [[r["dataset"], r["region"], r["no_rabc"]["ece"], r["no_rabc"]["brier"], r["no_rabc"]["nll"], r["rabc"]["ece"], r["rabc"]["brier"], r["rabc"]["nll"]] for r in dataset_rows],
        )
    print(output)


if __name__ == "__main__":
    main()
