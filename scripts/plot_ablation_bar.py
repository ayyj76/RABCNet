#!/usr/bin/env python3
"""
图4：消融实验柱状图
(a) 训练策略消融 ISIC-2018（A0→A1→A2）
(b) 解码器模块组消融 ISIC-2017
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "paper" / "figures" / "fig4_ablation.png"

# ── 数据（来自 paper.md 表 5.3 和 表 5.5）────────────────────────────

# 图(a) 训练策略消融 ISIC-2018
train_ablation = {
    "labels": [
        "A0\nTarget only\n(no distill)",
        "A1\n+ Source\ndistillation",
        "A2\n+ Sigma-Only\nadaptation",
    ],
    "DIC": [72.37, 78.15, 85.08],
    "JAC": [61.55, 68.18, 77.83],
}

# 图(b) 解码器模块组消融 ISIC-2017
decoder_ablation = {
    "labels": [
        "Full\n(all modules)",
        "− Context\nEnhancement",
        "− Attention\nRefinement",
        "− Boundary\n& Detail",
        "− All Decoder\nModules",
    ],
    "DIC": [78.81, 70.11, 69.06, 68.34, 70.11],
    "JAC": [68.44, 59.12, 58.07, 57.53, 59.55],
}

# ── 绘图 ──────────────────────────────────────────────────────────────

def plot_group(ax, data: dict, title: str, highlight_idx: list[int]):
    labels = data["labels"]
    dics   = np.array(data["DIC"])
    jacs   = np.array(data["JAC"])
    n      = len(labels)
    x      = np.arange(n)
    w      = 0.35

    bars_dic = ax.bar(x - w/2, dics, w,
                      color="#2E86C1", alpha=0.85,
                      label="DICE (%)")
    bars_jac = ax.bar(x + w/2, jacs, w,
                      color="#85C1E9", alpha=0.85, hatch="///",
                      edgecolor="#2E86C1", linewidth=0.8,
                      label="JAC (%)")

    # 数值标注
    for bar in bars_dic:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom",
                fontsize=7.5, color="#1A5276", fontfamily="Arial")
    for bar in bars_jac:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom",
                fontsize=7.5, color="#1A5276", fontfamily="Arial")

    # 高亮边框
    for idx in highlight_idx:
        for bars in [bars_dic, bars_jac]:
            bar = bars[idx]
            bar.set_edgecolor("#1A252F")
            bar.set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, fontfamily="Arial")
    ax.set_ylabel("Score (%)", fontsize=10, fontfamily="Arial")
    ax.set_ylim(52, 92)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))
    ax.tick_params(axis="y", labelsize=8.5)

    # 去顶框和右框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 浅灰网格
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_title(title, fontsize=11, fontweight="bold",
                 pad=8, fontfamily="Arial")
    ax.legend(fontsize=9, frameon=False, loc="upper left")


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

plot_group(
    axes[0], train_ablation,
    title="(a) Training Strategy Ablation (ISIC-2018)",
    highlight_idx=[2],   # A2 = 最终配置
)

plot_group(
    axes[1], decoder_ablation,
    title="(b) Decoder Module Ablation (ISIC-2017)",
    highlight_idx=[0],   # Full = 最终配置
)

fig.suptitle("Ablation Study", fontsize=13, fontweight="bold",
             y=1.02, fontfamily="Arial")

plt.tight_layout(pad=1.5, w_pad=3.0)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[done] 图4 已保存: {OUT}")
