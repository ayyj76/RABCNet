#!/usr/bin/env python3
"""Prepare interaction-only adaptation configs for reviewer mechanism checks."""
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
        "pseudo_root": "outputs/pseudo_labels/isic2017_train",
    },
    "isic2018": {
        "checkpoint": ROOT / "outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt",
        "pseudo_root": "outputs/pseudo_labels/isic2018_train",
    },
    "ph2": {
        "checkpoint": ROOT / "outputs/rpi_net_ph2_unsup_v17_retrain/checkpoints/best.pt",
        "pseudo_root": "outputs/pseudo_labels/ph2_train",
    },
}
INTERACTION_PREFIXES = ["pseudo_encoder", "ipc_attn", "pia_attn", "sigma_heads"]


def make_config(name: str, epochs: int) -> tuple[Path, dict]:
    spec = DATASETS[name]
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint has no embedded config: {checkpoint}")
    cfg = deepcopy(payload["config"])
    output_dir = ROOT / "outputs/review_cmig/v2/adaptation_transfer" / name
    cfg["seed"] = 42
    cfg["output_dir"] = str(output_dir)
    cfg["pseudo"]["output_dir"] = spec["pseudo_root"]
    # The reliability-transfer check reads image files directly and must not
    # depend on an optional historical tensor cache.
    cfg["data"]["use_tensor_cache"] = False
    cfg["data"].pop("tensor_cache_dir", None)
    cfg["data"]["num_workers"] = min(int(cfg["data"].get("num_workers", 4)), 4)
    cfg["data"]["eval_num_workers"] = min(int(cfg["data"].get("eval_num_workers", 2)), 2)
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
            "trainable_prefixes": INTERACTION_PREFIXES,
            "resume_checkpoint": str(checkpoint),
            "resume_strict": False,
            "resume_load_optimizer": False,
            "resume_load_scaler": False,
            "resume_reset_epoch": True,
            "resume_reset_best": True,
        }
    )
    cfg.setdefault("model", {})["use_rabc"] = False
    cfg["model"]["use_brr"] = False
    cfg.setdefault("runtime", {}).update(
        {
            "save_every": 1000,
            "checkpoint_include_optimizer": False,
            "checkpoint_include_scaler": False,
        }
    )
    config_path = ROOT / "configs/generated/review_cmig/v2/adaptation_transfer" / f"interaction_only_{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return config_path, {
        "dataset": name,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "pseudo_root": str(ROOT / spec["pseudo_root"]),
        "output_dir": str(output_dir),
        "trainable_prefixes": INTERACTION_PREFIXES,
        "epochs": int(epochs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--manifest",
        default="outputs/review_cmig/v2/adaptation_transfer/manifest.json",
    )
    args = parser.parse_args()
    config_path, entry = make_config(args.dataset, args.epochs)
    manifest = ROOT / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = {"runs": []}
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
    existing["runs"] = [item for item in existing.get("runs", []) if item.get("dataset") != args.dataset]
    existing["runs"].append(entry)
    manifest.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps({"config": str(config_path), "entry": entry}, indent=2))


if __name__ == "__main__":
    main()
