#!/usr/bin/env python3
"""
图6：三个数据集 per-sample DIC 分布箱线图
展示模型在各数据集上的预测稳定性
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, binary_fill_holes, label as ndi_label
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import load_config

EPS = 1e-7

DATASETS = [
    dict(
        name="ISIC-2017",
        config="configs/rpi_net_isic2017_unsup_v17_msaa.yaml",
        ckpt="outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt",
        threshold=0.02, sigma=1.0, tta="flip4",
        fill_holes=True, keep_largest=True,
    ),
    dict(
        name="ISIC-2018",
        config="configs/rpi_net_isic2018_unsup_unified_v20sigma.yaml",
        ckpt="outputs/rpi_net_isic2018_unsup_unified_v20sigma/checkpoints/best.pt",
        threshold=0.13, sigma=0.0, tta="none",
        fill_holes=False, keep_largest=False,
    ),
    dict(
        name="PH2",
        config="configs/rpi_net_ph2_unsup_unified_v20sigma.yaml",
        ckpt="outputs/rpi_net_ph2_unsup_unified_v20sigma/checkpoints/best.pt",
        threshold=0.10, sigma=1.0, tta="flip4",
        fill_holes=True, keep_largest=True,
    ),
]


# ── 辅助函数 ──────────────────────────────────────────────────────────

def model_kwargs_from_cfg(cfg: dict) -> dict:
    m = cfg["model"]
    keys = [
        "hidden_dim", "dropout", "backbone", "backbone_pretrained",
        "use_boundary_head", "use_uncertainty_head", "refine_with_heads",
        "use_aspp", "use_aux_head", "use_se_context", "use_detail_fusion",
        "use_coord_attn", "use_edge_prompt", "use_dual_supervised_head",
        "use_eca_context", "use_ema_context", "use_cbam_decoder",
        "use_simam_refine", "use_safm_decoder", "use_uncertainty_gated_refine",
        "use_sk_context", "use_triplet_decoder", "use_msaa_fusion",
        "use_ega_refine", "use_cot_context", "use_lsk_decoder",
        "use_rabc", "use_brr",
    ]
    defaults = {"hidden_dim": 128, "dropout": 0.1, "backbone": "resnet50",
                "backbone_pretrained": True}
    kwargs = {}
    for k in keys:
        if k in m:
            v = m[k]
            if isinstance(v, str) and v.lower() in ("true", "false"):
                v = v.lower() == "true"
            kwargs[k] = v
        elif k in defaults:
            kwargs[k] = defaults[k]
        else:
            kwargs[k] = False
    return kwargs


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":  return [(False, False)]
    if mode == "flip2": return [(False, False), (True, False)]
    if mode == "flip4": return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(mode)


def postprocess(prob: np.ndarray, thr: float, sigma: float,
                fill_holes: bool, keep_largest: bool) -> np.ndarray:
    if sigma > 0:
        prob = gaussian_filter(prob, sigma=sigma)
    mask = prob > thr
    if fill_holes:
        mask = binary_fill_holes(mask)
    if keep_largest:
        lbl, n = ndi_label(mask)
        if n > 0:
            areas = np.bincount(lbl.ravel()); areas[0] = 0
            mask = lbl == int(np.argmax(areas))
    return mask.astype(np.uint8)


def compute_per_sample_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt: [N,H,W] uint8 → per-sample metrics dict"""
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)
    tp = (p * g).sum(axis=(1, 2))
    fp = (p * (1 - g)).sum(axis=(1, 2))
    fn = ((1 - p) * g).sum(axis=(1, 2))
    tn = ((1 - p) * (1 - g)).sum(axis=(1, 2))
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "jac":  (tp + EPS) / (tp + fp + fn + EPS),
        "sen":  (tp + EPS) / (tp + fn + EPS),
        "spe":  (tn + EPS) / (tn + fp + EPS),
        "acc":  (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
    }


@torch.inference_mode()
def collect_metrics(ds_cfg: dict, device) -> dict:
    cfg = load_config(ROOT / ds_cfg["config"])
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))
    data_cfg = cfg["data"]

    ds = ISIC2017Dataset(
        root=data_cfg["root"],
        split=data_cfg["test_split"],
        image_size=data_cfg["image_size"],
        augment=False,
        use_hair_removal=False,
        ph2_split_seed=int(data_cfg.get("ph2_split_seed", 42)),
        ph2_split_ratios=list(data_cfg.get("ph2_split_ratios", [0.7, 0.15, 0.15])),
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    ckpt  = torch.load(ROOT / ds_cfg["ckpt"], map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model = RPINet(**model_kwargs_from_cfg(cfg))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    views = tta_views(ds_cfg["tta"])
    all_gts, all_preds = [], []

    for batch in loader:
        imgs_t = batch["image"].to(device)
        gts_t  = batch["mask"]
        if channels_last:
            imgs_t = imgs_t.contiguous(memory_format=torch.channels_last)

        prob_sum = None
        for hf, vf in views:
            xi = imgs_t
            if hf: xi = torch.flip(xi, dims=[3])
            if vf: xi = torch.flip(xi, dims=[2])
            p = torch.sigmoid(model(xi, None)["logits"])
            if vf: p = torch.flip(p, dims=[2])
            if hf: p = torch.flip(p, dims=[3])
            prob_sum = p if prob_sum is None else prob_sum + p

        probs = (prob_sum / len(views))[:, 0].cpu().numpy()
        gts   = (gts_t[:, 0].numpy() > 0.5).astype(np.uint8)

        preds = np.stack([
            postprocess(probs[i], ds_cfg["threshold"], ds_cfg["sigma"],
                        ds_cfg["fill_holes"], ds_cfg["keep_largest"])
            for i in range(len(probs))
        ])
        all_gts.append(gts)
        all_preds.append(preds)

    all_gts   = np.concatenate(all_gts,  axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    metrics   = compute_per_sample_metrics(all_preds, all_gts)
    n = len(all_gts)
    print(f"  n={n}  DIC={metrics['dice'].mean()*100:.2f}%  "
          f"JAC={metrics['jac'].mean()*100:.2f}%  "
          f"SEN={metrics['sen'].mean()*100:.2f}%  "
          f"SPE={metrics['spe'].mean()*100:.2f}%")
    return {k: v * 100 for k, v in metrics.items()}  # 转为百分比


# ── 绘图 ──────────────────────────────────────────────────────────────

def make_figure(all_metrics: list[dict], ds_names: list[str], out_path: Path):
    METRICS    = ["dice", "jac", "sen", "spe", "acc"]
    LABELS     = ["DICE", "JAC", "SEN", "SPE", "ACC"]
    DS_COLORS  = ["#2E86C1", "#1A9E5A", "#C0392B"]   # 蓝/绿/红
    DS_LIGHT   = ["#AED6F1", "#A9DFBF", "#F1948A"]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.5), dpi=150)

    for mi, (metric, label) in enumerate(zip(METRICS, LABELS)):
        ax = axes[mi]
        data    = [m[metric] for m in all_metrics]   # list of [N] arrays
        means   = [d.mean() for d in data]
        medians = [np.median(d) for d in data]

        # 箱线图
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.2, linestyle="--", color="#555"),
            capprops=dict(linewidth=1.5, color="#555"),
            flierprops=dict(marker="o", markersize=3, alpha=0.5,
                            markeredgewidth=0.5),
            notch=False,
        )

        # 填色
        for patch, fc, ec in zip(bp["boxes"], DS_LIGHT, DS_COLORS):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.5)

        for flier, ec in zip(bp["fliers"], DS_COLORS):
            flier.set_markerfacecolor(ec)
            flier.set_markeredgecolor(ec)

        # 均值菱形标记
        for xi, (mean, color) in enumerate(zip(means, DS_COLORS), start=1):
            ax.plot(xi, mean, marker="D", color=color,
                    markersize=6, zorder=5, markeredgecolor="white",
                    markeredgewidth=0.8)

        # 数据集名称 x 轴
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(
            [n.replace("ISIC-", "ISIC\n") for n in ds_names],
            fontsize=8.5
        )
        ax.set_title(label, fontsize=11, fontweight="bold", pad=18)
        ax.set_ylabel("Score (%)" if mi == 0 else "", fontsize=9)

        # y轴范围自适应但留白（SEN/SPE/ACC 数据集中在高值区，多留空间）
        all_vals = np.concatenate(data)
        ymin = max(0,   np.percentile(all_vals, 1)  - 5)
        extra = 18 if mi >= 2 else 12
        ymax = min(100, np.percentile(all_vals, 99) + extra)
        ax.set_ylim(ymin, ymax)

        # 均值数值标注：从 boxplot 对象取须线顶端坐标，再向上偏移
        whisker_tops = [bp["whiskers"][i * 2 + 1].get_ydata()[1] for i in range(len(data))]
        flier_ys = [
            [pt.get_ydata()[0] for pt in bp["fliers"] if abs(pt.get_xdata()[0] - (i + 1)) < 0.1]
            for i in range(len(data))
        ]
        offset = 2.5 if mi >= 2 else 1.2
        for xi, (d, mean, color, wtop, fys) in enumerate(
            zip(data, means, DS_COLORS, whisker_tops, flier_ys), start=1
        ):
            actual_top = max([wtop] + fys) if fys else wtop
            y = min(actual_top + offset, ymax - 0.8)
            ax.text(xi, y, f"{mean:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")

        # 样式
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="gray", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)

    # 图例（数据集）
    legend_items = [
        mpatches.Patch(facecolor=DS_LIGHT[i], edgecolor=DS_COLORS[i],
                       linewidth=1.5, label=ds_names[i])
        for i in range(len(ds_names))
    ]
    legend_items.append(
        plt.Line2D([0], [0], marker="D", color="gray", markersize=6,
                   linestyle="None", label="Mean")
    )
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Per-Sample Metric Distribution on Test Sets",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout(pad=1.2, w_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] 图6 已保存: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    all_metrics, ds_names = [], []
    for ds_cfg in DATASETS:
        print(f"--- {ds_cfg['name']} ---")
        metrics = collect_metrics(ds_cfg, device)
        all_metrics.append(metrics)
        ds_names.append(ds_cfg["name"])

    out = ROOT / "paper" / "figures" / "fig11_metric_boxplot.png"
    make_figure(all_metrics, ds_names, out)


if __name__ == "__main__":
    main()
