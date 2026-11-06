#!/usr/bin/env python3
"""Prepare a fair RABC-enabled Stage-2 adaptation matrix.

All variants start from the same no-RABC Stage-1 checkpoint.  RABC is enabled
in every trained variant, while the Stage-2 trainable subset is changed.  The
matrix is kept separate from the earlier reviewer experiments because those
experiments disabled RABC in the adaptation comparison.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/generated/review_cmig/v2/rabc_default.yaml"
SOURCE_CHECKPOINT = ROOT / "outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt"
OUTPUT_ROOT = ROOT / "outputs/review_cmig/v3/rabc_adaptation"
CONFIG_ROOT = ROOT / "configs/generated/review_cmig/v3/rabc_adaptation"

INTERACTION_PREFIXES = ["pseudo_encoder", "ipc_attn", "pia_attn", "sigma_heads"]
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


def write_config(name: str, trainable_prefixes: list[str], freeze_path: bool) -> Path:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg = deepcopy(cfg)
    cfg["seed"] = 42
    cfg["output_dir"] = str(OUTPUT_ROOT / name)
    cfg["pseudo"]["output_dir"] = "outputs/pseudo_labels/isic2018_train"
    cfg["train"].update(
        {
            "epochs": 2,
            "iterative_rounds": 0,
            "iterative_epochs": 0,
            "freeze_segmentation_path": freeze_path,
            "freeze_bn_stats": True,
            "trainable_prefixes": trainable_prefixes,
            "resume_checkpoint": str(SOURCE_CHECKPOINT),
            "resume_strict": False,
            "resume_load_optimizer": False,
            "resume_load_scaler": False,
            "resume_reset_epoch": True,
            "resume_reset_best": True,
        }
    )
    cfg["model"]["use_rabc"] = True
    cfg["model"]["use_brr"] = False
    cfg["runtime"].update(
        {
            "save_every": 1000,
            "checkpoint_include_optimizer": False,
            "checkpoint_include_scaler": False,
        }
    )
    path = CONFIG_ROOT / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/review_cmig/v3/rabc_adaptation/manifest.json",
    )
    args = parser.parse_args()
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    variants = {
        "rabc_only": ["rabc"],
        "interaction_rabc": INTERACTION_PREFIXES + ["rabc"],
        "decoder_rabc": DECODER_PREFIXES + ["rabc"],
        "backbone_rabc": ["encoder", "rabc"],
        "full_rabc": [],
    }
    manifest = {
        "protocol": {
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "epochs": 2,
            "seed": 42,
            "fixed_raw_inference": "threshold=0.30, tta=none, morphology=none",
            "rabc_enabled_for_all_trained_variants": True,
        },
        "variants": {},
    }
    for name, prefixes in variants.items():
        freeze_path = name != "full_rabc"
        config = write_config(name, prefixes, freeze_path)
        manifest["variants"][name] = {
            "config": str(config),
            "checkpoint": str(OUTPUT_ROOT / name / "checkpoints/best.pt"),
            "trainable_prefixes": prefixes,
            "freeze_segmentation_path": freeze_path,
        }

    manifest_path = ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
