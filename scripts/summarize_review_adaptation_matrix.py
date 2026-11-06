#!/usr/bin/env python3
"""Summarize the cross-dataset reviewer-response adaptation matrix.

The matrix deliberately uses the fixed raw test protocol from each run:
threshold=0.30, no TTA, and no morphology.  Validation-selected operating
points are retained separately for context and are never mixed into the main
comparison table.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_cmig/v5/adaptation_matrix"


DATASETS = ("ISIC-2017", "ISIC-2018", "PH2")
STRATEGIES = ("no_adaptation", "interaction_only", "decoder_only", "backbone_only", "full_network")
METRICS = ("acc", "dice", "jac", "sen", "spe")

V2_ADAPTATION_NAMES = {
    "ISIC-2018": {
        "decoder_only": "decoder_only",
        "backbone_only": "backbone_only",
        "full_network": "full_network",
    }
}


def read_metric_lines(path: Path) -> dict:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{") and "dice" in line:
            value = ast.literal_eval(line)
            if isinstance(value, dict):
                return value
    raise ValueError(f"No metric dictionary found in {path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_summary(dataset: str) -> tuple[dict, str]:
    if dataset == "ISIC-2017":
        path = ROOT / "outputs/rpi_net_isic2017_unsup_v23_cotlsk_full_bs14/summary.json"
        return load_json(path), str(path)
    if dataset == "PH2":
        path = ROOT / "outputs/rpi_net_ph2_unsup_v17_retrain/summary.json"
        return load_json(path), str(path)
    path = ROOT / "outputs/review_cmig/v2/evaluations/adaptation/no_adaptation_fixed_raw.txt"
    metric = read_metric_lines(path)
    return {"best_test": metric}, str(path)


def interaction_result(dataset: str) -> tuple[dict, str]:
    if dataset == "ISIC-2018":
        path = ROOT / "outputs/review_cmig/v2/evaluations/adaptation/interaction_only_fixed_raw.txt"
        return {"best_test": read_metric_lines(path)}, str(path)
    key = "isic2017" if dataset == "ISIC-2017" else "ph2"
    path = ROOT / f"outputs/review_cmig/v2/adaptation_transfer/{key}/summary.json"
    return load_json(path), str(path)


def matrix_result(dataset: str, strategy: str) -> tuple[dict, str]:
    if strategy == "no_adaptation":
        return source_summary(dataset)
    if strategy == "interaction_only":
        return interaction_result(dataset)
    if dataset == "ISIC-2018":
        name = V2_ADAPTATION_NAMES[dataset][strategy]
        path = ROOT / f"outputs/review_cmig/v2/evaluations/adaptation/{name}_fixed_raw.txt"
        return {"best_test": read_metric_lines(path)}, str(path)
    key = "isic2017" if dataset == "ISIC-2017" else "ph2"
    path = OUT / key / strategy / "summary.json"
    return load_json(path), str(path)


def trainable_counts(config_path: Path) -> tuple[int, int, float, list[str]]:
    import yaml
    import torch
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from rpi_net.model import RPINet
    from train_rpi_net import model_kwargs_from_cfg

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = RPINet(**model_kwargs_from_cfg(cfg))
    prefixes = tuple(str(x) for x in cfg["train"].get("trainable_prefixes", []) if str(x))
    if not bool(cfg["train"].get("freeze_segmentation_path", False)):
        trainable = sum(p.numel() for p in model.parameters())
    else:
        trainable = sum(
            p.numel()
            for name, p in model.named_parameters()
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        )
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100.0 * trainable / total, list(prefixes)


def fmt_metric(row: dict) -> str:
    return "/".join(f"{float(row[m]):.4f}" for m in METRICS)


def main() -> None:
    rows = []
    provenance = []
    for dataset in DATASETS:
        for strategy in STRATEGIES:
            summary, source = matrix_result(dataset, strategy)
            metric = summary["best_test"]
            if strategy in {"decoder_only", "backbone_only", "full_network"}:
                if dataset == "ISIC-2018":
                    config = ROOT / f"configs/generated/review_cmig/v2/adapt_{strategy}.yaml"
                else:
                    key = "isic2017" if dataset == "ISIC-2017" else "ph2"
                    config = ROOT / "configs/generated/review_cmig/v5/adaptation_matrix" / key / f"{strategy}.yaml"
            elif strategy == "interaction_only":
                if dataset == "ISIC-2018":
                    config = ROOT / "configs/generated/review_cmig/v2/adapt_interaction_only.yaml"
                else:
                    key = "isic2017" if dataset == "ISIC-2017" else "ph2"
                    config = ROOT / "configs/generated/review_cmig/v2/adaptation_transfer" / f"interaction_only_{key}.yaml"
            else:
                config = None
            if config is not None:
                trainable, total, ratio, prefixes = trainable_counts(config)
            else:
                trainable, total, ratio, prefixes = 0, None, 0.0, []
            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    **{metric_name: float(metric[metric_name]) for metric_name in METRICS},
                    "trainable_params": trainable,
                    "total_params": total,
                    "trainable_ratio_percent": ratio,
                }
            )
            provenance.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "source": source,
                    "config": str(config) if config is not None else None,
                    "protocol": "threshold=0.30, tta=none, morphology=none",
                    "trainable_prefixes": prefixes,
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {"seed": 42, "adaptation_epochs": 2, "rabc_enabled": False},
        "fixed_raw_protocol": "threshold=0.30, tta=none, morphology=none",
        "datasets": list(DATASETS),
        "strategies": list(STRATEGIES),
        "rows": rows,
        "provenance": provenance,
        "limitations": [
            "The three datasets start from their existing Stage-1 checkpoints; this is a cross-dataset protocol matrix, not one shared checkpoint.",
            "The matrix uses one short two-epoch target-adaptation budget and one seed (42).",
            "The interaction branch is bypassed by image-only inference, so interaction-only results are evidence about training-side reliability modeling rather than direct image-path rewriting.",
            "These controls support robustness under the tested low-resource protocol, not a universal optimum over all adaptation budgets or parameter subsets.",
        ],
    }
    # Keep the protocol representation explicit and JSON-safe.
    payload["protocol"] = {"seed": 42, "adaptation_epochs": 2, "rabc_enabled": False}
    (OUT / "ADAPTATION_MATRIX.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Cross-Dataset Adaptation Strategy Matrix",
        "",
        "Fixed raw test comparison: threshold=0.30, no TTA, no morphology. All trained controls use seed 42 and a two-epoch Stage-2 budget with RABC disabled.",
        "",
        "## Main Results",
        "",
        "Values are ACC/DICE/JAC/SEN/SPE. The no-adaptation and interaction-only rows reuse existing source/adaptation checkpoints under the same raw protocol.",
        "",
        "| Dataset | Strategy | ACC/DICE/JAC/SEN/SPE | Trainable parameters |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        params = f"{row['trainable_params']:,} ({row['trainable_ratio_percent']:.2f}%)"
        lines.append(f"| {row['dataset']} | {row['strategy']} | {fmt_metric(row)} | {params} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Backbone-only and full-network updates reduce DICE/JAC on all three datasets under the tested short low-resource budget.",
        "- Decoder-only adaptation degrades on ISIC-2017 and ISIC-2018 but improves on PH2 in this single-seed run, so its effect is domain dependent rather than uniformly harmful.",
        "- The cross-dataset pattern supports a conservative restricted-update design, while also showing that no trainable subset is universally optimal under every domain and budget.",
        "- The matrix does not establish interaction-only adaptation as a universal optimum: its direct deployment effect is limited because the pseudo-label interaction branch is bypassed by image-only inference.",
        "- The results should be reported as controlled within-study evidence, not as a claim that every trainable subset was globally optimized.",
        "",
        "## Provenance",
        "",
    ]
    for item in provenance:
        lines.append(f"- {item['dataset']} / {item['strategy']}: `{item['source']}`; config `{item['config']}`.")
    lines += [
        "",
        "## Limitations",
        "",
    ] + [f"- {item}" for item in payload["limitations"]]
    (OUT / "ADAPTATION_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "fixed_raw_matrix"
        headers = ["dataset", "strategy", *METRICS, "trainable_params", "total_params", "trainable_ratio_percent"]
        sheet.append(headers)
        for row in rows:
            sheet.append([row.get(header) for header in headers])
        sheet.freeze_panes = "A2"
        workbook.save(OUT / "ADAPTATION_MATRIX.xlsx")
    except ImportError:
        pass

    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
