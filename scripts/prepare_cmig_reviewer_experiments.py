#!/usr/bin/env python3
"""Prepare reproducible CMIG reviewer-response experiment configurations.

The generated runs share one ISIC-2018 Stage-1 checkpoint and differ only in
the Stage-2 trainable subset or the RABC candidate-gate coefficients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/rpi_net_isic2018_innovation_rabc.yaml"
SOURCE_CHECKPOINT = ROOT / "outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt"
OUTPUT_ROOT = ROOT / "outputs/review_cmig/v2"
CONFIG_ROOT = ROOT / "configs/generated/review_cmig/v2"

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


def write_config(name: str, cfg: dict) -> Path:
    path = CONFIG_ROOT / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def base_cfg() -> dict:
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    # Match the pseudo-label set used to train the shared source checkpoint.
    cfg["pseudo"]["output_dir"] = "outputs/pseudo_labels/isic2018_train"
    cfg["data"]["num_workers"] = 4
    cfg["data"]["eval_num_workers"] = 2
    cfg["data"]["persistent_workers"] = True
    cfg["data"]["eval_prefetch_factor"] = 2
    # Match the manuscript's Stage-2 protocol rather than the earlier invalid
    # short runs that accidentally ignored the configured resume checkpoint.
    cfg["train"]["epochs"] = 2
    cfg["train"]["iterative_rounds"] = 0
    cfg["train"]["iterative_epochs"] = 0
    cfg["train"]["resume_strict"] = False
    cfg["train"]["resume_load_optimizer"] = False
    cfg["train"]["resume_load_scaler"] = False
    cfg["train"]["resume_reset_epoch"] = True
    cfg["train"]["resume_reset_best"] = True
    cfg["runtime"]["save_every"] = 1000
    cfg["runtime"]["checkpoint_include_optimizer"] = False
    cfg["runtime"]["checkpoint_include_scaler"] = False
    cfg["model"]["use_brr"] = False
    return cfg


def configure_subset(name: str, prefixes: list[str] | None) -> Path:
    cfg = base_cfg()
    cfg["output_dir"] = str(OUTPUT_ROOT / "adaptation" / name)
    cfg["model"]["use_rabc"] = False
    cfg["train"]["freeze_segmentation_path"] = prefixes is not None
    cfg["train"]["trainable_prefixes"] = prefixes or []
    cfg["train"]["rabc_boundary_weight"] = 0.0
    cfg["train"]["rabc_far_bg_weight"] = 0.0
    cfg["train"]["rabc_sparsity_weight"] = 0.0
    cfg["train"]["resume_checkpoint"] = str(SOURCE_CHECKPOINT)
    return write_config(f"adapt_{name}", cfg)


def configure_rabc(name: str, intercept: float, uncertainty: float) -> Path:
    cfg = base_cfg()
    cfg["output_dir"] = str(OUTPUT_ROOT / "rabc_sensitivity" / name)
    cfg["model"]["use_rabc"] = True
    cfg["model"]["rabc_candidate_intercept"] = float(intercept)
    cfg["model"]["rabc_candidate_uncertainty"] = float(uncertainty)
    cfg["train"]["freeze_segmentation_path"] = True
    cfg["train"]["trainable_prefixes"] = INTERACTION_PREFIXES + ["rabc"]
    cfg["train"]["resume_checkpoint"] = str(SOURCE_CHECKPOINT)
    return write_config(f"rabc_{name}", cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(OUTPUT_ROOT / "experiment_manifest.json"))
    args = parser.parse_args()
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)

    paths = {
        "adaptation": {
            "no_adaptation": None,
            "interaction_only": INTERACTION_PREFIXES,
            "decoder_only": DECODER_PREFIXES,
            "backbone_only": BACKBONE_PREFIXES,
            "full_network": [],
        },
        "rabc_sensitivity": {
            "low_uncertainty_weight": [0.45, 0.55],
            "default": [0.35, 0.65],
            "high_uncertainty_weight": [0.25, 0.75],
        },
    }
    generated = {"source_checkpoint": str(SOURCE_CHECKPOINT), "runs": {}}

    for name, prefixes in paths["adaptation"].items():
        if name == "no_adaptation":
            generated["runs"]["adaptation/no_adaptation"] = {
                "config": None,
                "checkpoint": str(SOURCE_CHECKPOINT),
                "trainable_prefixes": [],
            }
            continue
        path = configure_subset(name, prefixes if prefixes else None)
        generated["runs"][f"adaptation/{name}"] = {
            "config": str(path),
            "checkpoint": str(OUTPUT_ROOT / "adaptation" / name / "checkpoints/best.pt"),
            "trainable_prefixes": prefixes,
        }

    for name, pair in paths["rabc_sensitivity"].items():
        path = configure_rabc(name, pair[0], pair[1])
        generated["runs"][f"rabc_sensitivity/{name}"] = {
            "config": str(path),
            "checkpoint": str(OUTPUT_ROOT / "rabc_sensitivity" / name / "checkpoints/best.pt"),
            "candidate_intercept": pair[0],
            "candidate_uncertainty": pair[1],
        }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
