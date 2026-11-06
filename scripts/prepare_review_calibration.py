#!/usr/bin/env python3
"""Prepare paired no-RABC/RABC calibration configurations.

The configurations are derived from the embedded checkpoint configurations so
that calibration comparisons use the same data split and image-path
architecture.  The no-RABC side only disables the optional RABC head; the
RABC side uses a checkpoint produced by the isolated RABC transfer experiment.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "isic2017": {
        "no_rabc_checkpoint": ROOT / "outputs/rpi_net_isic2017_unsup_v23_cotlsk_full_bs14/checkpoints/best.pt",
        "rabc_checkpoint": ROOT / "outputs/review_cmig/v2/evaluations/rabc_transfer/runs/isic2017/checkpoints/best.pt",
    },
    "isic2018": {
        "no_rabc_checkpoint": ROOT / "outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt",
        "rabc_checkpoint": ROOT / "outputs/review_cmig/v2/evaluations/rabc_transfer/runs/isic2018/checkpoints/best.pt",
    },
    "ph2": {
        "no_rabc_checkpoint": ROOT / "outputs/rpi_net_ph2_unsup_v17_retrain/checkpoints/best.pt",
        "rabc_checkpoint": ROOT / "outputs/review_cmig/v2/evaluations/rabc_transfer/runs/ph2/checkpoints/best.pt",
    },
}


def embedded_config(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint has no embedded config: {checkpoint}")
    return deepcopy(payload["config"])


def write_config(path: Path, source: Path, use_rabc: bool) -> None:
    cfg = embedded_config(source)
    cfg["output_dir"] = str(path.parent / "runtime")
    cfg.setdefault("runtime", {})["device"] = "cuda"
    cfg.setdefault("model", {})["use_rabc"] = bool(use_rabc)
    cfg["model"]["use_brr"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="configs/generated/review_cmig/v2/calibration",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/review_cmig/v2/evaluations/calibration_manifest.json",
    )
    args = parser.parse_args()

    output_root = (ROOT / args.output_root).resolve()
    manifest_path = (ROOT / args.manifest).resolve()
    manifest = {"datasets": {}}
    for name, spec in DATASETS.items():
        for key in ("no_rabc_checkpoint", "rabc_checkpoint"):
            if not spec[key].exists():
                raise FileNotFoundError(spec[key])
        no_cfg = output_root / f"{name}_no_rabc.yaml"
        rabc_cfg = output_root / f"{name}_rabc.yaml"
        write_config(no_cfg, spec["no_rabc_checkpoint"], use_rabc=False)
        write_config(rabc_cfg, spec["rabc_checkpoint"], use_rabc=True)
        manifest["datasets"][name] = {
            "config_no_rabc": str(no_cfg),
            "checkpoint_no_rabc": str(spec["no_rabc_checkpoint"]),
            "config_rabc": str(rabc_cfg),
            "checkpoint_rabc": str(spec["rabc_checkpoint"]),
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
