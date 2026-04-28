#!/usr/bin/env python3
"""
plot_prob_histogram.py
绘制三个数据集测试集的 sigmoid 概率分布直方图。

目的：直观说明 ISIC-2017 上 τ=0.02 并非任意极端值，
      而是模型 DUL 压缩导致概率分布左偏时的自然分界线。

输出：paper/figures/fig6_prob_hist.png / .pdf
      各数据集一列，前景（GT=1）/背景（GT=0）像素分别绘制，
      标注论文操作点阈值线。

用法：
    python scripts/plot_prob_histogram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import load_config

# ── Dataset configs（与 plot_gps_frontier.py 保持一致）────────────────────────
DATASETS = {
    "ISIC-2017": {
        "config":     ROOT / "configs" / "rpi_net_isic2017_unsup_v17_msaa.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_isic2017_unsup_v17_msaa" / "checkpoints" / "best.pt",
        "paper_thr":  0.02,
        "color_fg":   "#d62728",
        "color_bg":   "#1f77b4",
    },
    "ISIC-2018": {
        "config":     ROOT / "configs" / "rpi_net_isic2018_unsup_unified_v20sigma.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_isic2018_unsup_unified_v20sigma" / "checkpoints" / "best.pt",
        "paper_thr":  0.13,
        "color_fg":   "#d62728",
        "color_bg":   "#1f77b4",
    },
    "PH2": {
        "config":     ROOT / "configs" / "rpi_net_ph2_unsup_unified_v20sigma.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_ph2_unsup_unified_v20sigma" / "checkpoints" / "best.pt",
        "paper_thr":  0.10,
        "color_fg":   "#d62728",
        "color_bg":   "#1f77b4",
    },
}

OUTPUT_DIR = ROOT / "paper" / "figures"


def model_kwargs_from_cfg(cfg: dict) -> dict:
    model_cfg = cfg["model"]
    keys = [
        "hidden_dim", "dropout", "backbone", "backbone_pretrained",
        "use_boundary_head", "use_uncertainty_head", "refine_with_heads",
        "use_aspp", "use_aux_head", "use_se_context", "use_detail_fusion",
        "use_coord_attn", "use_edge_prompt", "use_dual_supervised_head",
        "use_eca_context", "use_ema_context", "use_cbam_decoder",
        "use_simam_refine", "use_safm_decoder", "use_uncertainty_gated_refine",
        "use_sk_context", "use_triplet_decoder", "use_msaa_fusion",
        "use_ega_refine", "use_cot_context", "use_lsk_decoder",
        "use_ucgf_refine", "use_anchor_context", "use_smfa_decoder",
        "use_tff_refine", "use_mfe_context", "use_bg_decouple_head",
        "use_uncertainty_logit_gate", "use_global_logit_calibrator",
        "use_boundary_uncertainty_logit_refine", "use_residual_logit_refine",
        "use_mamba_logit_refine", "use_cgfn_logit_refine",
        "use_recall_recover_refine", "use_recall_candidate_head",
        "use_multiscale_recall_head", "use_rabc", "use_brr", "use_focal_context", "use_cga_fusion",
    ]
    defaults = {"hidden_dim": 128, "dropout": 0.1, "backbone": "resnet50",
                "backbone_pretrained": True}
    result = {}
    for k in keys:
        v = model_cfg.get(k, defaults.get(k, False))
        if k == "hidden_dim":   result[k] = int(v)
        elif k == "dropout":    result[k] = float(v)
        elif k == "backbone":   result[k] = str(v)
        else:                   result[k] = bool(v)
    return result


def build_test_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    ds_kwargs = {
        "root":            data_cfg["root"],
        "split":           data_cfg.get("test_split", "test"),
        "image_size":      data_cfg["image_size"],
        "augment":         False,
        "use_hair_removal": bool(data_cfg.get("use_hair_removal", False)),
        "cache_images":    False,
    }
    if "ph2_split_seed"   in data_cfg: ds_kwargs["ph2_split_seed"]   = int(data_cfg["ph2_split_seed"])
    if "ph2_split_ratios" in data_cfg: ds_kwargs["ph2_split_ratios"] = tuple(data_cfg["ph2_split_ratios"])
    ds = ISIC2017Dataset(**ds_kwargs)
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                      pin_memory=True, persistent_workers=True, prefetch_factor=4)


@torch.inference_mode()
def collect_probs(model, loader, device, channels_last) -> tuple[np.ndarray, np.ndarray]:
    """返回 (probs_flat, masks_flat)，均为 1D 数组（像素级）"""
    model.eval()
    probs_list, masks_list = [], []
    for batch in tqdm(loader, desc="  collecting", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"][:, 0] > 0.5).numpy().astype(np.uint8)   # [B,H,W]
        logits = model(image, None)["logits"]
        p = torch.sigmoid(logits)[:, 0].cpu().numpy()                  # [B,H,W]
        probs_list.append(p.ravel())
        masks_list.append(mask.ravel())
    return np.concatenate(probs_list), np.concatenate(masks_list)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.14, wspace=0.32)

    BINS = np.linspace(0, 1, 101)   # 100 bins in [0,1]

    for idx, (ds_name, ds_info) in enumerate(DATASETS.items()):
        print(f"\n[{ds_name}]")
        cfg = load_config(str(ds_info["config"]))
        channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))

        model = RPINet(**model_kwargs_from_cfg(cfg)).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)

        ckpt = torch.load(str(ds_info["checkpoint"]), map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        model.eval()

        loader = build_test_loader(cfg)
        probs, masks = collect_probs(model, loader, device, channels_last)

        fg_probs = probs[masks == 1]
        bg_probs = probs[masks == 0]
        thr = ds_info["paper_thr"]

        print(f"  fg pixels: {len(fg_probs):,}  | median={np.median(fg_probs):.3f}  mean={fg_probs.mean():.3f}")
        print(f"  bg pixels: {len(bg_probs):,}  | median={np.median(bg_probs):.3f}  mean={bg_probs.mean():.3f}")

        ax = axes[idx]

        # 归一化到密度（面积=1），方便前景/背景在同一轴比较
        fg_counts, _ = np.histogram(fg_probs, bins=BINS, density=True)
        bg_counts, _ = np.histogram(bg_probs, bins=BINS, density=True)
        bin_centers  = (BINS[:-1] + BINS[1:]) / 2

        ax.fill_between(bin_centers, 0, bg_counts,
                        color=ds_info["color_bg"], alpha=0.45, label="Background (GT=0)")
        ax.fill_between(bin_centers, 0, fg_counts,
                        color=ds_info["color_fg"], alpha=0.55, label="Lesion (GT=1)")
        ax.step(bin_centers, bg_counts, color=ds_info["color_bg"], lw=1.2, where="mid")
        ax.step(bin_centers, fg_counts, color=ds_info["color_fg"], lw=1.2, where="mid")

        # 阈值线
        ax.axvline(thr, color="#333333", lw=1.8, ls="--", zorder=5,
                   label=f"Paper threshold $\\tau$={thr}")

        ax.set_title(ds_name, fontsize=13, fontweight="bold", pad=6)
        ax.set_xlabel("Predicted probability $p$", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 说明文字：仅对 ISIC-2017（τ=0.02）特别注解
        if ds_name == "ISIC-2017":
            ax.text(thr + 0.02, ax.get_ylim()[1] * 0.85,
                    f"$\\tau$={thr}\n(natural\nsplit)",
                    fontsize=7.5, color="#333333", va="top")

    fig.suptitle(
        "Predicted probability distributions (test set, per-pixel)\n"
        "Red = lesion pixels (GT=1), Blue = background pixels (GT=0), "
        "Dashed line = paper threshold $\\tau$",
        fontsize=10, y=1.04,
    )

    out = OUTPUT_DIR / "fig6_prob_hist.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
