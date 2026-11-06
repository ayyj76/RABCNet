#!/usr/bin/env python3
"""Draw the ablation summary figure used in the manuscript."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig8_ablation_visualization.png"

# Values are copied from the source-distillation and decoder ablations.
train_ablation = {
    "labels": [
        "A0\nBaseline",
        "A1\n+ Source\ndistillation",
    ],
    "DICE": [76.37, 82.15],
    "JAC": [67.57, 74.20],
}

decoder_ablation = {
    "labels": [
        "Full\n(all modules)",
        "- Context\nblock",
        "- Detail\nblock",
        "- Boundary\nblock",
        "- All decoder\nmodules",
    ],
    "DICE": [78.81, 68.32, 69.06, 68.69, 70.11],
    "JAC": [68.44, 57.59, 58.07, 57.97, 59.14],
}


def plot_group(ax, data: dict[str, list], title: str, highlight_idx: list[int]) -> None:
    labels = data["labels"]
    dice = np.asarray(data["DICE"], dtype=float)
    jac = np.asarray(data["JAC"], dtype=float)
    x = np.arange(len(labels))
    width = 0.35

    bars_dice = ax.bar(
        x - width / 2,
        dice,
        width,
        color="#2E86C1",
        alpha=0.88,
        label="DICE (%)",
    )
    bars_jac = ax.bar(
        x + width / 2,
        jac,
        width,
        color="#A9D6EF",
        alpha=0.92,
        hatch="///",
        edgecolor="#2E86C1",
        linewidth=0.8,
        label="JAC (%)",
    )

    for bars in (bars_dice, bars_jac):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.45,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#1A5276",
            )

    for idx in highlight_idx:
        for bars in (bars_dice, bars_jac):
            bars[idx].set_edgecolor("#1A252F")
            bars[idx].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(52, 92)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9, frameon=False, loc="upper left")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    plot_group(
        axes[0],
        train_ablation,
        title="(a) Training Strategy Ablation (ISIC-2018)",
        highlight_idx=[1],
    )
    plot_group(
        axes[1],
        decoder_ablation,
        title="(b) Decoder Module Ablation (ISIC-2017)",
        highlight_idx=[0],
    )
    fig.suptitle("Ablation Study", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.5, w_pad=3.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] saved: {OUT}")


if __name__ == "__main__":
    main()
