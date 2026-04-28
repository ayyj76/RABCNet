#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize fixed-operating-point multiseed evaluation JSON")
    p.add_argument("--inputs", nargs="+", required=True, help="JSON files produced by eval_multiseed_fixedop.py")
    p.add_argument("--labels", nargs="+", required=True, help="Display labels matching --inputs")
    p.add_argument("--output", required=True)
    return p.parse_args()


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    metrics = ["acc", "dice", "jac", "sen", "spe"]
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        vals = np.array([float(r[metric]) for r in rows], dtype=np.float64)
        out[metric] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have same length")

    data = {}
    for label, path_str in zip(args.labels, args.inputs):
        path = Path(path_str)
        rows = json.loads(path.read_text(encoding="utf-8"))
        data[label] = {
            "n": len(rows),
            "per_seed": rows,
            "summary": summarize(rows),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
