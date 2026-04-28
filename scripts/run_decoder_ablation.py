#!/usr/bin/env python3
"""
Decoder Module Ablation Experiment for ISIC-2017.

Generates configs, trains, evaluates, and aggregates results for
leave-one-out decoder module ablation on ISIC-2017.

Usage:
  # Generate all configs
  python scripts/run_decoder_ablation.py generate

  # Train a single ablation (e.g., no_se_context)
  python scripts/run_decoder_ablation.py train --name no_se_context

  # Train ALL ablations sequentially
  python scripts/run_decoder_ablation.py train-all

  # Evaluate all completed ablations and print summary table
  python scripts/run_decoder_ablation.py evaluate

  # Full pipeline: generate + train-all + evaluate
  python scripts/run_decoder_ablation.py run-all
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Base config: ISIC-2017 from-scratch training with all decoder modules ON
# ---------------------------------------------------------------------------
BASE_CONFIG = ROOT / "configs" / "rpi_net_isic2017_unsup_v17_msaa.yaml"
ABLATION_DIR = ROOT / "configs" / "ablation_decoder"
OUTPUT_ROOT = ROOT / "outputs" / "ablation_decoder"
RESULTS_FILE = OUTPUT_ROOT / "ablation_results.json"

# ---------------------------------------------------------------------------
# Modules to ablate (leave-one-out): flag_name -> human-readable label
# ---------------------------------------------------------------------------
INDIVIDUAL_MODULES = {
    "no_se_context":       ("use_se_context",                "SE"),
    "no_eca_context":      ("use_eca_context",               "ECA"),
    "no_ema_context":      ("use_ema_context",               "EMA"),
    "no_coord_attn":       ("use_coord_attn",                "CoordAttn"),
    "no_sk_context":       ("use_sk_context",                "SK"),
    "no_cbam_decoder":     ("use_cbam_decoder",              "CBAM"),
    "no_simam_refine":     ("use_simam_refine",              "SimAM"),
    "no_safm_decoder":     ("use_safm_decoder",              "SAFM"),
    "no_triplet_decoder":  ("use_triplet_decoder",           "Triplet"),
    "no_msaa_fusion":      ("use_msaa_fusion",               "MSAA"),
    "no_edge_prompt":      ("use_edge_prompt",               "EdgePrompt"),
    "no_detail_fusion":    ("use_detail_fusion",             "DetailFusion"),
    "no_ugr":              ("use_uncertainty_gated_refine",   "UGR"),
}

# Group ablations: disable multiple modules at once
GROUP_MODULES = {
    "no_context_group": {
        "label": "All Context (SE+ECA+EMA+CoordAttn+SK)",
        "flags": ["use_se_context", "use_eca_context", "use_ema_context",
                  "use_coord_attn", "use_sk_context"],
    },
    "no_decoder_group": {
        "label": "All Decoder Attn (CBAM+SimAM+SAFM+Triplet)",
        "flags": ["use_cbam_decoder", "use_simam_refine",
                  "use_safm_decoder", "use_triplet_decoder"],
    },
    "no_boundary_group": {
        "label": "All Boundary/Detail (Edge+Detail+UGR)",
        "flags": ["use_edge_prompt", "use_detail_fusion",
                  "use_uncertainty_gated_refine"],
    },
    "no_all_decoder_modules": {
        "label": "No Decoder Modules (backbone + basic fuse only)",
        "flags": ["use_se_context", "use_eca_context", "use_ema_context",
                  "use_coord_attn", "use_sk_context",
                  "use_cbam_decoder", "use_simam_refine",
                  "use_safm_decoder", "use_triplet_decoder",
                  "use_msaa_fusion",
                  "use_edge_prompt", "use_detail_fusion",
                  "use_uncertainty_gated_refine"],
    },
}


def load_base_config() -> dict:
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def generate_configs():
    """Generate ablation config YAML files."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_config()
    generated = []

    # Fix base config: disable tensor cache (may not exist), reduce workers
    base["data"]["use_tensor_cache"] = False
    base["data"].pop("tensor_cache_dir", None)

    # Individual leave-one-out
    for name, (flag, label) in INDIVIDUAL_MODULES.items():
        cfg = copy.deepcopy(base)
        cfg["output_dir"] = str(OUTPUT_ROOT / name)
        cfg["model"][flag] = False
        # Adjust related loss weights when disabling modules
        if flag == "use_detail_fusion":
            cfg["train"]["detail_consensus_weight"] = 0.0
        if flag == "use_edge_prompt":
            pass  # edge prompt has no dedicated loss weight
        if flag == "use_uncertainty_gated_refine":
            pass  # UGR doesn't have its own loss

        out_path = ABLATION_DIR / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        generated.append((name, label, out_path))
        print(f"  Generated: {out_path.name}  ({label})")

    # Group ablations
    for name, info in GROUP_MODULES.items():
        cfg = copy.deepcopy(base)
        cfg["output_dir"] = str(OUTPUT_ROOT / name)
        for flag in info["flags"]:
            cfg["model"][flag] = False
        # Zero out related losses
        if "use_detail_fusion" in info["flags"]:
            cfg["train"]["detail_consensus_weight"] = 0.0

        out_path = ABLATION_DIR / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        generated.append((name, info["label"], out_path))
        print(f"  Generated: {out_path.name}  ({info['label']})")

    print(f"\nTotal: {len(generated)} ablation configs in {ABLATION_DIR}")
    return generated


def train_one(name: str):
    """Train a single ablation experiment."""
    config_path = ABLATION_DIR / f"{name}.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        print("Run 'generate' first.")
        sys.exit(1)

    output_dir = OUTPUT_ROOT / name
    checkpoint = output_dir / "checkpoints" / "best.pt"
    if checkpoint.exists():
        print(f"SKIP: {name} — checkpoint already exists at {checkpoint}")
        return

    print(f"\n{'='*60}")
    print(f"  Training ablation: {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train_rpi_net.py"),
        "--config", str(config_path),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARNING: Training failed for {name} (exit code {result.returncode})")


def train_all():
    """Train all ablation experiments sequentially."""
    all_names = list(INDIVIDUAL_MODULES.keys()) + list(GROUP_MODULES.keys())
    total = len(all_names)
    for i, name in enumerate(all_names, 1):
        print(f"\n[{i}/{total}] ", end="")
        train_one(name)
    print(f"\nAll {total} ablation experiments completed.")


def evaluate_all():
    """Evaluate all completed ablation checkpoints and produce summary."""
    results = {}

    all_experiments = {}
    for name, (flag, label) in INDIVIDUAL_MODULES.items():
        all_experiments[name] = label
    for name, info in GROUP_MODULES.items():
        all_experiments[name] = info["label"]

    for name, label in all_experiments.items():
        checkpoint = OUTPUT_ROOT / name / "checkpoints" / "best.pt"
        config_path = ABLATION_DIR / f"{name}.yaml"

        if not checkpoint.exists():
            print(f"SKIP: {name} — no checkpoint found")
            continue

        print(f"Evaluating: {name} ({label})...")

        # Evaluate on val with threshold sweep to find best threshold
        cmd_val = [
            sys.executable, str(ROOT / "scripts" / "evaluate_rpi_net.py"),
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--split", "val",
            "--threshold", "0.30",
            "--output-json", str(OUTPUT_ROOT / name / "val_metrics.json"),
        ]
        subprocess.run(cmd_val, cwd=str(ROOT), capture_output=True)

        # Evaluate on test
        cmd_test = [
            sys.executable, str(ROOT / "scripts" / "evaluate_rpi_net.py"),
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--split", "test",
            "--threshold", "0.30",
            "--output-json", str(OUTPUT_ROOT / name / "test_metrics.json"),
        ]
        subprocess.run(cmd_test, cwd=str(ROOT), capture_output=True)

        # Load test results
        test_json = OUTPUT_ROOT / name / "test_metrics.json"
        if test_json.exists():
            with open(test_json) as f:
                metrics = json.load(f)
            results[name] = {
                "label": label,
                "acc": metrics.get("acc", 0),
                "dice": metrics.get("dice", metrics.get("dic", 0)),
                "jac": metrics.get("jac", 0),
                "sen": metrics.get("sen", 0),
                "spe": metrics.get("spe", 0),
            }
        else:
            print(f"  WARNING: No test metrics for {name}")

    # Save results
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print_summary(results)
    return results


def print_summary(results: dict):
    """Print a formatted summary table."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*90}")
    print(f"  Decoder Module Ablation Results (ISIC-2017, test, thr=0.30)")
    print(f"{'='*90}")
    print(f"  {'Ablation':<40s} {'ACC':>7s} {'DIC':>7s} {'JAC':>7s} {'SEN':>7s} {'SPE':>7s}")
    print(f"  {'-'*40:<40s} {'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s} {'-'*7:>7s}")

    # Sort: individual first, then groups
    individual = {k: v for k, v in results.items() if k in INDIVIDUAL_MODULES}
    groups = {k: v for k, v in results.items() if k in GROUP_MODULES}

    for name, r in individual.items():
        print(f"  Full - {r['label']:<32s} {r['acc']:>7.2f} {r['dice']:>7.2f} "
              f"{r['jac']:>7.2f} {r['sen']:>7.2f} {r['spe']:>7.2f}")

    if groups:
        print(f"  {'--- Group Ablations ---':<40s}")
        for name, r in groups.items():
            label = r['label']
            if len(label) > 38:
                label = label[:35] + "..."
            print(f"  Full - {label:<32s} {r['acc']:>7.2f} {r['dice']:>7.2f} "
                  f"{r['jac']:>7.2f} {r['sen']:>7.2f} {r['spe']:>7.2f}")

    print(f"{'='*90}")
    print(f"  Results saved to: {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Decoder Module Ablation Experiment")
    parser.add_argument("action", choices=["generate", "train", "train-all",
                                           "evaluate", "run-all"],
                        help="Action to perform")
    parser.add_argument("--name", type=str, default=None,
                        help="Ablation name (for 'train' action)")
    args = parser.parse_args()

    if args.action == "generate":
        generate_configs()

    elif args.action == "train":
        if not args.name:
            print("ERROR: --name required for 'train' action")
            sys.exit(1)
        train_one(args.name)

    elif args.action == "train-all":
        train_all()

    elif args.action == "evaluate":
        evaluate_all()

    elif args.action == "run-all":
        generate_configs()
        train_all()
        evaluate_all()


if __name__ == "__main__":
    main()
