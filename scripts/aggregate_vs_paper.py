#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate seed-level metrics and compare with paper")
    p.add_argument("--input", type=str, required=True, help="JSON list of seed rows from eval_multiseed_fixedop.py")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--paper-acc", type=float, required=True)
    p.add_argument("--paper-dice", type=float, required=True)
    p.add_argument("--paper-jac", type=float, required=True)
    p.add_argument("--paper-sen", type=float, required=True)
    p.add_argument("--paper-spe", type=float, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("Input JSON must be a non-empty list")

    paper = {
        "acc": float(args.paper_acc),
        "dice": float(args.paper_dice),
        "jac": float(args.paper_jac),
        "sen": float(args.paper_sen),
        "spe": float(args.paper_spe),
    }
    metrics = ["acc", "dice", "jac", "sen", "spe"]

    agg = {}
    all_pass = True
    for m in metrics:
        vals = [float(r[m]) for r in rows]
        mean = float(sum(vals) / len(vals))
        std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        delta = mean - paper[m]
        passed = bool(mean > paper[m])
        agg[m] = {
            "mean_pct": mean,
            "std_pct": std,
            "paper_pct": paper[m],
            "delta_vs_paper_pct": delta,
            "pass_mean_vs_paper": passed,
        }
        all_pass = all_pass and passed

    op = {
        "threshold": rows[0]["threshold"],
        "tta": rows[0]["tta"],
        "min_area": rows[0]["min_area"],
        "keep_largest": rows[0]["keep_largest"],
        "fill_holes": rows[0]["fill_holes"],
    }

    out = {
        "dataset": args.dataset,
        "protocol": "fixed_op",
        "fixed_operating_point": op,
        "seeds": rows,
        "aggregate": agg,
        "all_metrics_pass_mean_vs_paper": all_pass,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
