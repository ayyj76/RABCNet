#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multiple seed checkpoints at a fixed operating point")
    p.add_argument("--seed-configs", type=str, nargs="+", required=True, help="Per-seed yaml configs")
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Per-seed checkpoints")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--tta", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--min-area", type=int, default=0)
    p.add_argument("--keep-largest", action="store_true")
    p.add_argument("--fill-holes", action="store_true")
    p.add_argument("--dilation-iters", type=int, default=0)
    p.add_argument("--compute-distance", action="store_true", help="Collect HD95/ASSD if evaluate script reports them")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def seed_from_cfg(cfg_path: str) -> int:
    d = yaml.safe_load(Path(cfg_path).read_text())
    return int(d.get("seed", -1))


def run_one(
    cfg: str,
    ckpt: str,
    split: str,
    threshold: float,
    tta: str,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
    dilation_iters: int,
    compute_distance: bool,
    workers: int | None,
    batch_size: int | None,
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_rpi_net.py"),
        "--config",
        cfg,
        "--checkpoint",
        ckpt,
        "--split",
        split,
        "--threshold",
        str(threshold),
        "--tta",
        tta,
        "--min-area",
        str(min_area),
    ]
    if keep_largest:
        cmd.append("--keep-largest")
    if fill_holes:
        cmd.append("--fill-holes")
    if dilation_iters > 0:
        cmd.extend(["--dilation-iters", str(dilation_iters)])
    if compute_distance:
        cmd.append("--compute-distance")
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if not lines:
        raise RuntimeError(f"No stdout from evaluate_rpi_net.py for cfg={cfg}")
    last = lines[-1]
    try:
        parsed = ast.literal_eval(last)
    except Exception as e:
        raise RuntimeError(f"Failed to parse final metrics line: {last}") from e
    return parsed


def main() -> None:
    args = parse_args()
    if len(args.seed_configs) != len(args.checkpoints):
        raise ValueError("--seed-configs and --checkpoints must have same length")

    rows = []
    for cfg, ckpt in zip(args.seed_configs, args.checkpoints):
        seed = seed_from_cfg(cfg)
        m = run_one(
            cfg=cfg,
            ckpt=ckpt,
            split=args.split,
            threshold=float(args.threshold),
            tta=args.tta,
            min_area=int(args.min_area),
            keep_largest=bool(args.keep_largest),
            fill_holes=bool(args.fill_holes),
            dilation_iters=int(args.dilation_iters),
            compute_distance=bool(args.compute_distance),
            workers=args.workers,
            batch_size=args.batch_size,
        )
        row = {
            "seed": seed,
            "acc": float(m["acc"] * 100.0),
            "dice": float(m["dice"] * 100.0),
            "jac": float(m["jac"] * 100.0),
            "sen": float(m["sen"] * 100.0),
            "spe": float(m["spe"] * 100.0),
            "threshold": float(m["threshold"]),
            "tta": str(m["tta"]),
            "min_area": int(m["min_area"]),
            "keep_largest": bool(m["keep_largest"]),
            "fill_holes": bool(m["fill_holes"]),
            "dilation_iters": int(m.get("dilation_iters", args.dilation_iters)),
            "config": cfg,
            "checkpoint": ckpt,
        }
        if "hd95" in m:
            row["hd95"] = float(m["hd95"])
        if "assd" in m:
            row["assd"] = float(m["assd"])
        rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
