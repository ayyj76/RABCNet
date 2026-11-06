#!/usr/bin/env python3
"""Plot boundary-band calibration deltas from the CMIG review runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "outputs/review_cmig/v4/calibration_width"
OUTPUT = ROOT / "paper/figures/figS8_boundary_calibration.png"
DATASETS = ["isic2017", "isic2018", "ph2"]
DISPLAY = {"isic2017": "ISIC-2017", "isic2018": "ISIC-2018", "ph2": "PH2"}
METRICS = ["ece", "brier", "nll"]
COLORS = {"isic2017": "#1b6ca8", "isic2018": "#c65d2e", "ph2": "#4b7f52"}


def load_rows() -> dict[str, dict[str, list[float]]]:
    rows = {metric: {dataset: [] for dataset in DATASETS} for metric in METRICS}
    intervals = {metric: {dataset: [] for dataset in DATASETS} for metric in METRICS}
    for dataset in DATASETS:
        for width in (1, 3, 5):
            path = INPUT_ROOT / f"{dataset}_w{width}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            band = data["paired_image_bootstrap"]["boundary_band"]
            for metric in METRICS:
                entry = band[metric]
                rows[metric][dataset].append(float(entry["rabc_minus_no_rabc_mean"]))
                intervals[metric][dataset].append(tuple(float(x) for x in entry["ci95"]))
    return rows, intervals


def main() -> None:
    rows, intervals = load_rows()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True)
    widths = np.array([1, 3, 5])
    for axis, metric in zip(axes, METRICS):
        for dataset in DATASETS:
            values = np.asarray(rows[metric][dataset])
            ci = np.asarray(intervals[metric][dataset])
            axis.plot(
                widths,
                values,
                marker="o",
                linewidth=1.7,
                markersize=3.4,
                color=COLORS[dataset],
                label=DISPLAY[dataset],
            )
            axis.fill_between(widths, ci[:, 0], ci[:, 1], color=COLORS[dataset], alpha=0.12)
        axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_title(metric.upper(), fontsize=9, fontweight="bold")
        axis.set_xticks(widths)
        axis.grid(axis="y", alpha=0.22, linewidth=0.55)
        axis.tick_params(labelsize=7.5, length=2.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("RABC - no RABC", fontsize=8.5)
    axes[1].set_xlabel("Ground-truth boundary-band width (pixels)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
