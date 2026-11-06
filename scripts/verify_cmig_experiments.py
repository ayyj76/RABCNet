#!/usr/bin/env python3
"""Audit the completed CMIG reviewer-response experiments.

This is a provenance and protocol check, not a new statistical analysis.  It
keeps the reviewer-response tables tied to the exact checkpoints, configs,
coefficient grid, paired case IDs, and evaluation protocols that produced
them.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_cmig/v4"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_metric(path: Path) -> dict:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{") and "dice" in line:
            value = ast.literal_eval(line)
            if isinstance(value, dict):
                return value
    raise ValueError(f"No metric dictionary found in {path}")


def finite_metrics(value: dict, keys: tuple[str, ...]) -> bool:
    return all(key in value and math.isfinite(float(value[key])) for key in keys)


def add_check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def audit_adaptation(checks: list[dict], evidence: dict) -> None:
    root = ROOT / "outputs/review_cmig/v2"
    config_root = ROOT / "configs/generated/review_cmig/v2"
    manifest = load_json(root / "experiment_manifest.json")
    source = Path(manifest["source_checkpoint"])
    source_exists = source.exists()
    add_check(checks, "adaptation/source_checkpoint", source_exists, str(source))
    if not source_exists:
        return
    source_hash = sha256(source)
    evidence["adaptation_source"] = {"path": str(source), "sha256": source_hash}

    variants = {
        "no_adaptation": {"config": None, "checkpoint": source},
        "interaction_only": {"config": config_root / "adapt_interaction_only.yaml", "checkpoint": root / "adaptation/interaction_only/checkpoints/best.pt"},
        "decoder_only": {"config": config_root / "adapt_decoder_only.yaml", "checkpoint": root / "adaptation/decoder_only/checkpoints/best.pt"},
        "backbone_only": {"config": config_root / "adapt_backbone_only.yaml", "checkpoint": root / "adaptation/backbone_only/checkpoints/best.pt"},
        "full_network": {"config": config_root / "adapt_full_network.yaml", "checkpoint": root / "adaptation/full_network/checkpoints/best.pt"},
    }
    metric_keys = ("acc", "dice", "jac", "sen", "spe")
    rows = []
    for name, item in variants.items():
        checkpoint = item["checkpoint"]
        checkpoint_exists = checkpoint.exists()
        add_check(checks, f"adaptation/{name}/checkpoint", checkpoint_exists, str(checkpoint))
        config = item["config"]
        if config is not None:
            config = config.resolve()
            config_exists = config.exists()
            add_check(checks, f"adaptation/{name}/config", config_exists, str(config))
            if config_exists:
                cfg = load_yaml(config)
                resume = Path(str(cfg["train"].get("resume_checkpoint", ""))).resolve()
                add_check(
                    checks,
                    f"adaptation/{name}/shared_resume",
                    resume == source.resolve(),
                    f"configured={resume}; expected={source.resolve()}",
                )
                add_check(
                    checks,
                    f"adaptation/{name}/two_epoch_protocol",
                    int(cfg["train"].get("epochs", -1)) == 2 and int(cfg["seed"]) == 42,
                    f"epochs={cfg['train'].get('epochs')}; seed={cfg.get('seed')}",
                )
                add_check(
                    checks,
                    f"adaptation/{name}/rabc_disabled",
                    not bool(cfg["model"].get("use_rabc", False)),
                    f"use_rabc={cfg['model'].get('use_rabc')}",
                )
        raw_path = root / f"evaluations/adaptation/{name}_fixed_raw.txt"
        raw_exists = raw_path.exists()
        add_check(checks, f"adaptation/{name}/raw_metrics", raw_exists, str(raw_path))
        if raw_exists:
            metric = raw_metric(raw_path)
            add_check(checks, f"adaptation/{name}/finite_metrics", finite_metrics(metric, metric_keys), str(metric))
        rows.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
                "sha256": sha256(checkpoint) if checkpoint_exists else None,
                "raw_metrics": metric if raw_exists else None,
            }
        )
    evidence["adaptation"] = rows

    bn_control = OUT / "full_network_bn_unfrozen/fixed_raw.txt"
    add_check(checks, "adaptation/bn_unfrozen_control", bn_control.exists(), str(bn_control))
    if bn_control.exists():
        metric = raw_metric(bn_control)
        add_check(checks, "adaptation/bn_unfrozen_finite_metrics", finite_metrics(metric, metric_keys), str(metric))
        evidence["bn_unfrozen_control"] = metric


def audit_sensitivity(checks: list[dict], evidence: dict) -> None:
    paths = {
        "ISIC-2017": OUT / "sensitivity/isic2017_test.json",
        "ISIC-2018": ROOT / "outputs/review_cmig/v2/evaluations/rabc/sensitivity_test.json",
        "PH2": OUT / "sensitivity/ph2_test.json",
    }
    expected = [(0.25, 0.75), (0.35, 0.65), (0.45, 0.55)]
    rows = []
    for dataset, path in paths.items():
        exists = path.exists()
        add_check(checks, f"sensitivity/{dataset}/result", exists, str(path))
        if not exists:
            continue
        result = load_json(path)
        settings = [(float(x["candidate_intercept"]), float(x["candidate_uncertainty"])) for x in result["settings"]]
        add_check(checks, f"sensitivity/{dataset}/fixed_grid", settings == expected, str(settings))
        checkpoint = Path(result["checkpoint"])
        checkpoint_exists = checkpoint.exists()
        add_check(checks, f"sensitivity/{dataset}/checkpoint", checkpoint_exists, str(checkpoint))
        actual_hash = sha256(checkpoint) if checkpoint_exists else None
        add_check(
            checks,
            f"sensitivity/{dataset}/recorded_hash",
            actual_hash == result.get("checkpoint_sha256"),
            f"recorded={result.get('checkpoint_sha256')}; actual={actual_hash}",
        )
        for item in result["settings"]:
            metrics = item["metrics"]
            add_check(
                checks,
                f"sensitivity/{dataset}/finite_{item['candidate_intercept']}_{item['candidate_uncertainty']}",
                finite_metrics({key: value["mean"] for key, value in metrics.items()}, ("acc", "dice", "jac", "sen", "spe")),
                str(metrics),
            )
        rows.append({"dataset": dataset, "path": str(path), "checkpoint": str(checkpoint), "sha256": actual_hash, "settings": settings})
    evidence["sensitivity"] = rows


def audit_calibration(checks: list[dict], evidence: dict) -> None:
    rows = []
    for dataset in ("isic2017", "isic2018", "ph2"):
        for width in (1, 3, 5):
            path = OUT / f"calibration_width/{dataset}_w{width}.json"
            exists = path.exists()
            add_check(checks, f"calibration/{dataset}/width_{width}/result", exists, str(path))
            if not exists:
                continue
            result = load_json(path)
            add_check(
                checks,
                f"calibration/{dataset}/width_{width}/raw_protocol",
                result.get("protocol") == "raw image-only inference; no threshold, TTA, or morphology",
                str(result.get("protocol")),
            )
            no_cases = result["no_rabc"]["cases"]
            rabc_cases = result["rabc"]["cases"]
            no_ids = [case["id"] for case in no_cases]
            rabc_ids = [case["id"] for case in rabc_cases]
            add_check(checks, f"calibration/{dataset}/width_{width}/paired_ids", no_ids == rabc_ids and len(no_ids) == len(set(no_ids)), f"n={len(no_ids)}")
            for group in ("no_rabc", "rabc"):
                checkpoint = Path(result[group]["checkpoint"])
                checkpoint_exists = checkpoint.exists()
                add_check(checks, f"calibration/{dataset}/width_{width}/{group}_checkpoint", checkpoint_exists, str(checkpoint))
                actual_hash = sha256(checkpoint) if checkpoint_exists else None
                add_check(
                    checks,
                    f"calibration/{dataset}/width_{width}/{group}_hash",
                    actual_hash == result[group].get("checkpoint_sha256"),
                    f"recorded={result[group].get('checkpoint_sha256')}; actual={actual_hash}",
                )
            boundary = result["paired_image_bootstrap"]["boundary_band"]
            add_check(
                checks,
                f"calibration/{dataset}/width_{width}/boundary_improves",
                all(metric["ci95"][1] < 0 for metric in boundary.values()),
                str(boundary),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "width": width,
                    "n_images": len(no_ids),
                    "no_checkpoint": result["no_rabc"]["checkpoint"],
                    "rabc_checkpoint": result["rabc"]["checkpoint"],
                    "boundary_delta": result["paired_image_bootstrap"]["boundary_band"],
                    "full_image_delta": result["paired_image_bootstrap"]["full_image"],
                }
            )
    evidence["calibration"] = rows


def audit_reliability(checks: list[dict], evidence: dict) -> None:
    rows = []
    for dataset in ("isic2017", "isic2018", "ph2"):
        path = ROOT / f"outputs/review_cmig/v2/evaluations/reliability_transfer/{dataset}_train.json"
        exists = path.exists()
        add_check(checks, f"reliability/{dataset}/result", exists, str(path))
        if not exists:
            continue
        result = load_json(path)
        bootstrap = result.get("paired_bootstrap", {})
        add_check(checks, f"reliability/{dataset}/bootstrap", bool(bootstrap), "paired bootstrap present")
        rows.append({"dataset": dataset, "path": str(path), "paired_bootstrap": bootstrap})
    evidence["reliability_transfer"] = rows


def write_report(checks: list[dict], evidence: dict) -> None:
    passed = sum(item["passed"] for item in checks)
    failed = [item for item in checks if not item["passed"]]
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "passed": passed,
        "failed": len(failed),
        "checks": checks,
        "evidence": evidence,
    }
    json_path = OUT / "EXPERIMENT_AUDIT.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# CMIG Experiment Audit",
        "",
        f"Status: **{audit['status']}** ({passed} checks passed, {len(failed)} failed).",
        "",
        "This audit verifies provenance and protocol consistency. It does not turn single-seed results into multi-seed evidence and does not use test masks for model selection.",
        "",
        "## Checks",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")
        if len(detail) > 220:
            detail = detail[:217] + "..."
        lines.append(f"| {status} | `{item['name']}` | {detail} |")
    lines += [
        "",
        "## Evidence Boundary",
        "",
        "- Adaptation controls are single-seed, fixed two-epoch comparisons on ISIC-2018; they support protocol-specific robustness/fragility conclusions, not universal adaptation claims.",
        "- RABC sensitivity changes only inference coefficients on one fixed checkpoint per dataset; it is robustness evidence, not evidence that the coefficients create a large overlap gain.",
        "- Calibration uses raw image-only probabilities, no threshold/TTA/morphology, paired image bootstrap, and a ground-truth-defined evaluation boundary band. The band is used only after inference.",
        "- PH2 has 30 test images; its calibration and sensitivity results should remain descriptive.",
        "",
        "## Reproduction Entry Points",
        "",
        "- Adaptation configs: `configs/generated/review_cmig/v2/adapt_*.yaml`",
        "- RABC coefficient scans: `scripts/evaluate_rabc_sensitivity.py`",
        "- Calibration evaluator: `scripts/evaluate_review_calibration.py`",
        "- Summary generator: `scripts/summarize_cmig_v4_experiments.py`",
        "- Audit command: `PYTHONPATH=src python3 scripts/verify_cmig_experiments.py`",
    ]
    (OUT / "EXPERIMENT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "passed": passed, "failed": len(failed), "json": str(json_path)}, indent=2))


def main() -> None:
    checks: list[dict] = []
    evidence: dict = {}
    audit_adaptation(checks, evidence)
    audit_sensitivity(checks, evidence)
    audit_calibration(checks, evidence)
    audit_reliability(checks, evidence)
    write_report(checks, evidence)
    if any(not item["passed"] for item in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
