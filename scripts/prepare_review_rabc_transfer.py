#!/usr/bin/env python3
"""Generate isolated RABC-only transfer configs from existing checkpoints.

The generated runs keep each checkpoint's image-path architecture unchanged,
add the RABC head, and train only that head. This is intended for reviewer
calibration/attribution experiments, not for replacing the main pipeline.
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
        "checkpoint": ROOT / "outputs/rpi_net_isic2017_unsup_v23_cotlsk_full_bs14/checkpoints/best.pt",
        "losses": (0.02, 0.03, 0.03),
    },
    "isic2018": {
        "checkpoint": ROOT / "outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt",
        "losses": (0.04, 0.02, 0.01),
    },
    "ph2": {
        "checkpoint": ROOT / "outputs/rpi_net_ph2_unsup_v17_retrain/checkpoints/best.pt",
        "losses": (0.045, 0.02, 0.012),
    },
}


def load_checkpoint_config(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint has no embedded config: {checkpoint}")
    return deepcopy(payload["config"])


def make_config(name: str, config_root: Path, run_root: Path, epochs: int) -> tuple[Path, dict]:
    spec = DATASETS[name]
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cfg = load_checkpoint_config(checkpoint)
    cfg["seed"] = 42
    cfg["output_dir"] = str(run_root / name)

    train = cfg.setdefault("train", {})
    train.update(
        {
            "epochs": int(epochs),
            "iterative_rounds": 0,
            "iterative_epochs": 0,
            "source_distill_weight": 0.0,
            "consistency_weight": 0.0,
            "freeze_segmentation_path": True,
            "freeze_bn_stats": True,
            "trainable_prefixes": ["rabc"],
            "resume_checkpoint": str(checkpoint),
            "resume_strict": False,
            "resume_load_optimizer": False,
            "resume_load_scaler": False,
            "resume_reset_epoch": True,
            "resume_reset_best": True,
            "rabc_boundary_weight": float(spec["losses"][0]),
            "rabc_far_bg_weight": float(spec["losses"][1]),
            "rabc_sparsity_weight": float(spec["losses"][2]),
            "rabc_delta_direction_weight": 0.0,
        }
    )
    model = cfg.setdefault("model", {})
    model["use_rabc"] = True
    model["use_brr"] = False
    model["rabc_mixer_mode"] = "mean"
    model["rabc_candidate_intercept"] = 0.35
    model["rabc_candidate_uncertainty"] = 0.65
    runtime = cfg.setdefault("runtime", {})
    runtime.update(
        {
            "save_every": 1,
            "checkpoint_include_optimizer": False,
            "checkpoint_include_scaler": False,
        }
    )

    config_path = config_root / f"rabc_only_{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path, {
        "dataset": name,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "output_dir": str(run_root / name),
        "trainable_prefixes": ["rabc"],
        "epochs": int(epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="configs/generated/review_cmig/v2/rabc_transfer")
    parser.add_argument("--manifest", default="outputs/review_cmig/v2/evaluations/rabc_transfer/manifest.json")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    config_root = (ROOT / args.output_root).resolve()
    run_root = ROOT / "outputs/review_cmig/v2/evaluations/rabc_transfer/runs"
    manifest_path = (ROOT / args.manifest).resolve()
    entries = []
    for name in DATASETS:
        _, entry = make_config(name, config_root, run_root, args.epochs)
        entries.append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"runs": entries}, indent=2), encoding="utf-8")
    print(json.dumps({"runs": entries}, indent=2))


if __name__ == "__main__":
    main()
