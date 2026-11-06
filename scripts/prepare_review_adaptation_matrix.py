#!/usr/bin/env python3
"""Prepare cross-dataset Stage-2 adaptation strategy controls.

The generated runs share each dataset's existing Stage-1 checkpoint and use
the same short, two-epoch target-adaptation budget.  RABC is disabled so that
the matrix isolates the effect of the trainable subset.  The resulting
configs are intended for reviewer-response evidence, not as replacement main
training recipes.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/review_cmig/v5/adaptation_matrix"
CONFIG_ROOT = ROOT / "configs/generated/review_cmig/v5/adaptation_matrix"

DATASETS = {
    "isic2017": {
        "checkpoint": ROOT / "outputs/rpi_net_isic2017_unsup_v23_cotlsk_full_bs14/checkpoints/best.pt",
        "pseudo_root": "outputs/pseudo_labels/isic2017_train",
    },
    "ph2": {
        "checkpoint": ROOT / "outputs/rpi_net_ph2_unsup_v17_retrain/checkpoints/best.pt",
        "pseudo_root": "outputs/pseudo_labels/ph2_train",
    },
}

INTERACTION_PREFIXES = ["pseudo_encoder", "ipc_attn", "pia_attn", "sigma_heads"]
BACKBONE_PREFIXES = ["encoder"]
DECODER_PREFIXES = [
    "c2_proj", "c2_down", "c3_proj", "c4_proj", "fuse", "msaa_fusion",
    "context", "mfe_context", "sk_context", "se_context", "coord_attn",
    "eca_context", "ema_context", "anchor_context", "focal_context",
    "edge_extractor", "edge_proj", "edge_gate", "detail_proj", "detail_gate",
    "detail_head", "seg_refine", "cbam_decoder", "simam_refine",
    "safm_decoder", "ega_refine", "triplet_decoder", "lsk_decoder",
    "smfa_decoder", "seg_head", "aux_head", "boundary_fuse", "boundary_head",
    "uncertainty_head", "head_guided_refine", "uncertainty_gated_refine",
]


def load_checkpoint_config(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint has no embedded config: {checkpoint}")
    return deepcopy(payload["config"])


def write_config(dataset: str, strategy: str, prefixes: list[str] | None) -> dict:
    spec = DATASETS[dataset]
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    cfg = load_checkpoint_config(checkpoint)
    output_dir = OUTPUT_ROOT / dataset / strategy
    cfg["seed"] = 42
    cfg["output_dir"] = str(output_dir)
    cfg.setdefault("data", {})["use_tensor_cache"] = False
    cfg["data"].pop("tensor_cache_dir", None)
    # Keep reviewer-response runs deterministic and robust on the shared
    # workstation; multiprocessing workers have caused intermittent loader
    # segfaults in the large ISIC-2017 cache path.
    cfg["data"]["num_workers"] = 0
    cfg["data"]["eval_num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg.setdefault("pseudo", {})["output_dir"] = spec["pseudo_root"]

    train = cfg.setdefault("train", {})
    train.update(
        {
            "epochs": 2,
            "iterative_rounds": 0,
            "iterative_epochs": 0,
            "source_distill_weight": 0.0,
            "consistency_weight": 0.0,
            "freeze_segmentation_path": prefixes is not None,
            "freeze_bn_stats": True,
            "trainable_prefixes": prefixes or [],
            "resume_checkpoint": str(checkpoint),
            "resume_strict": False,
            "resume_load_optimizer": False,
            "resume_load_scaler": False,
            "resume_reset_epoch": True,
            "resume_reset_best": True,
        }
    )
    model = cfg.setdefault("model", {})
    model["use_rabc"] = False
    model["use_brr"] = False
    runtime = cfg.setdefault("runtime", {})
    runtime.update(
        {
            "save_every": 1000,
            "checkpoint_include_optimizer": False,
            "checkpoint_include_scaler": False,
        }
    )

    config_path = CONFIG_ROOT / dataset / f"{strategy}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return {
        "dataset": dataset,
        "strategy": strategy,
        "config": str(config_path),
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(output_dir / "checkpoints/best.pt"),
        "output_dir": str(output_dir),
        "trainable_prefixes": prefixes or [],
        "epochs": 2,
        "fixed_raw_protocol": "threshold=0.30, tta=none, morphology=none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/review_cmig/v5/adaptation_matrix/manifest.json")
    args = parser.parse_args()

    strategies = {
        "decoder_only": DECODER_PREFIXES,
        "backbone_only": BACKBONE_PREFIXES,
        "full_network": None,
    }
    runs = []
    for dataset in DATASETS:
        for strategy, prefixes in strategies.items():
            runs.append(write_config(dataset, strategy, prefixes))

    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "seed": 42,
                    "epochs": 2,
                    "rabc_enabled": False,
                    "fixed_raw_inference": "threshold=0.30, tta=none, morphology=none",
                    "purpose": "cross-dataset reviewer-response adaptation strategy controls",
                },
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"runs": runs}, indent=2))


if __name__ == "__main__":
    main()
