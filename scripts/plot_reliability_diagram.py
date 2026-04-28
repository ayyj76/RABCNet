#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot reliability diagrams from calibration evaluation JSON")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--region", choices=["full_image", "boundary_band"], default="boundary_band")
    p.add_argument("--title", default="")
    p.add_argument("--output", required=True)
    return p.parse_args()


def load_curve(path: str, region: str) -> tuple[list[float], list[float], list[int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bins = payload["reliability_bins"][region]
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []
    for row in bins:
        if row["count"] <= 0 or row["mean_prob"] is None or row["freq_pos"] is None:
            continue
        xs.append(float(row["mean_prob"]))
        ys.append(float(row["freq_pos"]))
        counts.append(int(row["count"]))
    return xs, ys, counts


def main() -> None:
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    plt.figure(figsize=(5.6, 5.0))
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0, label="Ideal")
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, (path, label) in enumerate(zip(args.inputs, args.labels)):
        xs, ys, counts = load_curve(path, region=args.region)
        plt.plot(xs, ys, marker=markers[idx % len(markers)], linewidth=1.5, markersize=4, label=label)
        if counts:
            biggest = max(range(len(counts)), key=lambda i: counts[i])
            plt.scatter([xs[biggest]], [ys[biggest]], s=18, zorder=3)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted foreground probability")
    plt.ylabel("Empirical foreground frequency")
    if args.title:
        plt.title(args.title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
