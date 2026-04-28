#!/usr/bin/env python3
"""
Plot SEN-SPE operating frontiers under optional reference smoothing.

This figure is retained as contextual evidence for fixed post-processing on the
no-RABC backbone. The main method contribution is still the learned
reliability-aware calibration performed by RABC inside the deployed model.

Usage:
    python scripts/plot_gps_frontier.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import load_config

EPS = 1e-7

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASETS = {
    "ISIC-2017": {
        "config": ROOT / "configs" / "rpi_net_isic2017_unsup_v17_msaa.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_isic2017_unsup_v17_msaa" / "checkpoints" / "best.pt",
        "paper_thr": 0.02,
        "paper_sigma": 1.0,
    },
    "ISIC-2018": {
        "config": ROOT / "configs" / "rpi_net_isic2018_unsup_unified_v20sigma.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_isic2018_unsup_unified_v20sigma" / "checkpoints" / "best.pt",
        "paper_thr": 0.13,
        "paper_sigma": 0.0,
    },
    "PH2": {
        "config": ROOT / "configs" / "rpi_net_ph2_unsup_unified_v20sigma.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_ph2_unsup_unified_v20sigma" / "checkpoints" / "best.pt",
        "paper_thr": 0.10,
        "paper_sigma": 1.0,
    },
}

SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
OUTPUT_DIR = ROOT / "paper" / "figures"

# Per-dataset reference-smoothing sigmas to plot.
DATASET_SIGMAS = {
    "ISIC-2017": [0.0, 0.5, 1.0, 1.5, 2.0],
    "ISIC-2018": [0.0, 0.5, 1.0, 1.5, 2.0],  # all shown; curves overlap -> smoothing has negligible effect
    "PH2":       [0.0, 0.5, 1.0, 1.5, 2.0],
}

# Per-dataset axis limits: (xlim_lo, xlim_hi, ylim_lo, ylim_hi)
DATASET_XLIM = {
    "ISIC-2017": (82, 100),
    "ISIC-2018": (84, 100),    # tighter zoom; single curve sits in 87-98 SPE range
    "PH2":       (82, 100),
}
DATASET_YLIM = {
    "ISIC-2017": (55, 100),
    "ISIC-2018": (70, 100),    # tighter zoom
    "PH2":       (55, 100),
}


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
    defaults = {
        "hidden_dim": 128, "dropout": 0.1, "backbone": "resnet50",
        "backbone_pretrained": True,
    }
    result = {}
    for k in keys:
        v = model_cfg.get(k, defaults.get(k, False))
        if k in ("hidden_dim",):
            result[k] = int(v)
        elif k in ("dropout",):
            result[k] = float(v)
        elif k in ("backbone",):
            result[k] = str(v)
        else:
            result[k] = bool(v)
    return result


def build_test_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg["data"]
    ds_kwargs = {
        "root": data_cfg["root"],
        "split": data_cfg.get("test_split", "test"),
        "image_size": data_cfg["image_size"],
        "augment": False,
        "use_hair_removal": bool(data_cfg.get("use_hair_removal", False)),
        "cache_images": False,
    }
    if "ph2_split_seed" in data_cfg:
        ds_kwargs["ph2_split_seed"] = int(data_cfg["ph2_split_seed"])
    if "ph2_split_ratios" in data_cfg:
        ds_kwargs["ph2_split_ratios"] = tuple(data_cfg["ph2_split_ratios"])

    ds = ISIC2017Dataset(**ds_kwargs)
    return DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 GPU tensor: probs [N,H,W] float32, masks [N,H,W] bool"""
    model.eval()
    probs_all, masks_all = [], []
    for batch in tqdm(loader, desc="collect-probs", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5)
        logits = model(image, None)["logits"]
        p = torch.sigmoid(logits)
        probs_all.append(p[:, 0])
        masks_all.append(mask[:, 0])
    return torch.cat(probs_all), torch.cat(masks_all)


def _make_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """1D 高斯核，用于可分离卷积"""
    radius = int(4 * sigma + 0.5)
    size = 2 * radius + 1
    x = torch.arange(size, dtype=torch.float32, device=device) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


@torch.inference_mode()
def apply_gps(probs: torch.Tensor, sigma: float) -> torch.Tensor:
    """GPU 可分离高斯平滑，输入 [N,H,W]，返回同形状 tensor"""
    if sigma <= 0:
        return probs
    device = probs.device
    k = _make_gaussian_kernel(sigma, device)
    radius = k.shape[0] // 2
    # [N,1,H,W] → 行卷积 → 列卷积
    x = probs.unsqueeze(1)
    # 水平
    kh = k.view(1, 1, 1, -1)
    x = F.conv2d(x, kh, padding=(0, radius))
    # 垂直
    kv = k.view(1, 1, -1, 1)
    x = F.conv2d(x, kv, padding=(radius, 0))
    return x.squeeze(1)


@torch.inference_mode()
def sweep_frontier(
    probs: torch.Tensor,
    masks: torch.Tensor,
    thresholds: np.ndarray,
    chunk: int = 32,
) -> tuple[list[float], list[float]]:
    """
    向量化阈值扫描，全程在 GPU 上运行。
    thresholds: [C] numpy array
    probs:  [N,H,W] float32 GPU tensor
    masks:  [N,H,W] bool   GPU tensor
    chunk:  每批处理的阈值数，控制显存峰值
    """
    thr_t = torch.from_numpy(thresholds.astype(np.float32)).to(probs.device)  # [C]
    eps = 1e-7
    t = masks.float()  # [N,H,W]
    fg = t.sum(dim=(1, 2))          # [N] 每张图前景像素数
    bg = (1.0 - t).sum(dim=(1, 2)) # [N] 背景像素数

    sens_all, spes_all = [], []
    for i in range(0, thr_t.shape[0], chunk):
        thr_chunk = thr_t[i: i + chunk]                    # [c]
        # [c,N,H,W] = [c,1,1,1] > [N,H,W]
        preds = (probs.unsqueeze(0) > thr_chunk[:, None, None, None]).float()
        tp = (preds * t.unsqueeze(0)).sum(dim=(2, 3))      # [c,N]
        fp = (preds * (1.0 - t).unsqueeze(0)).sum(dim=(2, 3))  # [c,N]
        # sen = tp/(tp+fn) = tp/fg,  spe = tn/(tn+fp) = (bg-fp)/bg
        sen = ((tp + eps) / (fg.unsqueeze(0) + eps)).mean(dim=1)          # [c]
        spe = ((bg.unsqueeze(0) - fp + eps) / (bg.unsqueeze(0) + eps)).mean(dim=1)  # [c]
        sens_all.append(sen.cpu())
        spes_all.append(spe.cpu())

    sens = torch.cat(sens_all).tolist()
    spes = torch.cat(spes_all).tolist()
    return sens, spes


def compute_sen_spe_at_threshold(probs: torch.Tensor, masks: torch.Tensor, threshold: float) -> tuple[float, float]:
    preds = (probs > threshold).float()
    t = masks.float()
    tp = (preds * t).sum(dim=(1, 2))
    fp = (preds * (1.0 - t)).sum(dim=(1, 2))
    fn = ((1.0 - preds) * t).sum(dim=(1, 2))
    tn = ((1.0 - preds) * (1.0 - t)).sum(dim=(1, 2))
    sen = float(((tp + EPS) / (tp + fn + EPS)).mean().item())
    spe = float(((tn + EPS) / (tn + fp + EPS)).mean().item())
    return sen, spe


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = np.arange(0.005, 0.96, 0.005)

    # Color map for sigma values
    colors = {0.0: '#1f77b4', 0.5: '#2ca02c', 1.0: '#d62728', 1.5: '#9467bd', 2.0: '#ff7f0e'}
    linewidths = {0.0: 2.5, 0.5: 1.5, 1.0: 2.0, 1.5: 1.5, 2.0: 1.5}
    linestyles = {0.0: '-', 0.5: ':', 1.0: '--', 1.5: '-.', 2.0: (0, (3, 1, 1, 1, 1, 1))}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (ds_name, ds_info) in enumerate(DATASETS.items()):
        print(f"\n{'='*60}")
        print(f"  Processing {ds_name}")
        print(f"{'='*60}")

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
        probs_raw, masks = collect_probs(model, loader, device, channels_last)
        print(f"  Probs range: [{probs_raw.min():.4f}, {probs_raw.max():.4f}]")

        ax = axes[idx]

        # Per-dataset sigma list (ISIC-2018: GPS not used, only plot σ=0)
        sigmas_to_plot = DATASET_SIGMAS[ds_name]

        # Sweep for each sigma (always compute σ=0 and σ=1.0 for fill_between)
        all_frontiers = {}
        for sigma in SIGMAS:
            probs_s = apply_gps(probs_raw, sigma)
            sens, spes = sweep_frontier(probs_s, masks, thresholds)
            all_frontiers[sigma] = (sens, spes)

        for sigma in sigmas_to_plot:
            sens, spes = all_frontiers[sigma]
            spe_pct = [s * 100 for s in spes]
            sen_pct = [s * 100 for s in sens]

            label = f'$\\sigma$={sigma:.1f}' if sigma > 0 else '$\\sigma$=0 (no ref. smoothing)'
            ax.plot(spe_pct, sen_pct,
                    color=colors[sigma],
                    linewidth=linewidths[sigma],
                    linestyle=linestyles[sigma],
                    label=label, zorder=3)

        # Fill between σ=0 and σ=1.0 (only when both are plotted)
        if 0.0 in sigmas_to_plot and 1.0 in sigmas_to_plot:
            sens_0, spes_0 = all_frontiers[0.0]
            sens_1, spes_1 = all_frontiers[1.0]
            spe_grid = np.linspace(0.80, 1.0, 200)
            spe_arr_0 = np.array(spes_0)
            sen_arr_0 = np.array(sens_0)
            spe_arr_1 = np.array(spes_1)
            sen_arr_1 = np.array(sens_1)
            order_0 = np.argsort(spe_arr_0)
            order_1 = np.argsort(spe_arr_1)
            sen_interp_0 = np.interp(spe_grid, spe_arr_0[order_0], sen_arr_0[order_0])
            sen_interp_1 = np.interp(spe_grid, spe_arr_1[order_1], sen_arr_1[order_1])
            ax.fill_between(spe_grid * 100, sen_interp_0 * 100, sen_interp_1 * 100,
                            where=sen_interp_1 > sen_interp_0,
                            alpha=0.15, color='#d62728', zorder=1,
                            label='Reference-smoothing gain region')
        elif ds_name == "ISIC-2018":
            ax.text(0.05, 0.08, 'Curves overlap:\nreference smoothing inactive\n(selected $\\sigma$=0)',
                    transform=ax.transAxes, fontsize=9, color='gray',
                    va='bottom', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

        # Mark the untuned baseline point.
        sen_p0, spe_p0 = compute_sen_spe_at_threshold(probs_raw, masks, 0.30)
        ax.plot(spe_p0*100, sen_p0*100, 's', color='#1f77b4', markersize=9, zorder=6,
                markeredgecolor='black', markeredgewidth=0.8)
        ax.annotate(f'P0\n(base $\\tau$=0.30)',
                    xy=(spe_p0*100, sen_p0*100),
                    xytext=(spe_p0*100 - 5, sen_p0*100 + 3),
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

        # Mark the selected operating point from the reference study.
        p_sigma = ds_info["paper_sigma"]
        p_thr = ds_info["paper_thr"]
        probs_final = apply_gps(probs_raw, p_sigma)
        sen_gp, spe_gp = compute_sen_spe_at_threshold(probs_final, masks, p_thr)
        ax.plot(spe_gp*100, sen_gp*100, '*', color='#ff7f0e', markersize=14, zorder=6,
                markeredgecolor='black', markeredgewidth=0.8)
        sigma_str = f'{p_sigma:.1f}' if p_sigma > 0 else '0'
        ax.annotate(f'Selected\n($\\tau$={p_thr}, $\\sigma$={sigma_str})',
                    xy=(spe_gp*100, sen_gp*100),
                    xytext=(spe_gp*100 + 3, sen_gp*100 - 6),
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

        ax.set_xlabel("SPE (%)", fontsize=12)
        ax.set_ylabel("SEN (%)", fontsize=12)
        ax.set_title(ds_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')

        # Per-dataset zoom
        ax.set_xlim(*DATASET_XLIM[ds_name])
        ax.set_ylim(*DATASET_YLIM[ds_name])

        # ── Inset zoom around the operating-point cluster ────────────────────
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
        # zoom windows: (spe_lo, spe_hi, sen_lo, sen_hi)
        inset_cfg = {
            "ISIC-2017": dict(xlim=(91.5, 96.5), ylim=(82.0, 91.0),
                              loc="lower left",  loc1=2, loc2=4,
                              bbox=(0.02, 0.02, 0.48, 0.48)),
            "ISIC-2018": dict(xlim=(93.5, 99.5), ylim=(83.5, 91.5),
                              loc="lower left",  loc1=2, loc2=4,
                              bbox=(0.02, 0.02, 0.50, 0.50)),
            "PH2":        dict(xlim=(88.0, 100.0), ylim=(88.0, 97.0),
                              loc="lower left",  loc1=2, loc2=4,
                              bbox=(0.02, 0.02, 0.50, 0.50)),
        }
        ic = inset_cfg[ds_name]
        axins = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(ic["bbox"][0], ic["bbox"][1],
                                          ic["bbox"][2], ic["bbox"][3]),
                           bbox_transform=ax.transAxes, loc=ic["loc"],
                           borderpad=0)
        # re-draw all curves into the inset
        for sigma in sigmas_to_plot:
            sens_i, spes_i = all_frontiers[sigma]
            axins.plot([s * 100 for s in spes_i], [s * 100 for s in sens_i],
                       color=colors[sigma], linewidth=linewidths[sigma],
                       linestyle=linestyles[sigma], zorder=3)
        # re-draw fill_between in inset (ISIC-2017 / PH2)
        if 0.0 in sigmas_to_plot and 1.0 in sigmas_to_plot:
            axins.fill_between(spe_grid * 100, sen_interp_0 * 100, sen_interp_1 * 100,
                               where=sen_interp_1 > sen_interp_0,
                               alpha=0.15, color='#d62728', zorder=1)
        # operating point markers in inset
        axins.plot(spe_p0 * 100, sen_p0 * 100, 's', color='#1f77b4',
                   markersize=7, zorder=6,
                   markeredgecolor='black', markeredgewidth=0.6)
        axins.plot(spe_gp * 100, sen_gp * 100, '*', color='#ff7f0e',
                   markersize=10, zorder=6,
                   markeredgecolor='black', markeredgewidth=0.6)
        axins.set_xlim(ic["xlim"])
        axins.set_ylim(ic["ylim"])
        axins.tick_params(labelsize=6)
        axins.grid(True, alpha=0.2, linestyle='--')
        ax.indicate_inset_zoom(axins, edgecolor="0.4", lw=0.8, zorder=5)

        print(f"  P0 (thr=0.30, σ=0): SEN={sen_p0*100:.2f}%, SPE={spe_p0*100:.2f}%")
        print(f"  Selected (thr={p_thr}, σ={p_sigma}): SEN={sen_gp*100:.2f}%, SPE={spe_gp*100:.2f}%")

        del model
        torch.cuda.empty_cache()

    # ── 统一图例（放大图下方）────────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color=colors[0.0], lw=linewidths[0.0], ls=linestyles[0.0],
               label='$\\sigma$=0 (no ref. smoothing)'),
        Line2D([0], [0], color=colors[0.5], lw=linewidths[0.5], ls=linestyles[0.5],
               label='$\\sigma$=0.5'),
        Line2D([0], [0], color=colors[1.0], lw=linewidths[1.0], ls=linestyles[1.0],
               label='$\\sigma$=1.0'),
        Line2D([0], [0], color=colors[1.5], lw=linewidths[1.5], ls=linestyles[1.5],
               label='$\\sigma$=1.5'),
        Line2D([0], [0], color=colors[2.0], lw=linewidths[2.0], ls=linestyles[2.0],
               label='$\\sigma$=2.0'),
        Patch(facecolor='#d62728', alpha=0.25, label='Reference-smoothing gain region'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4',
               markeredgecolor='black', markersize=8, label='P0 baseline ($\\tau$=0.30, $\\sigma$=0)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#ff7f0e',
               markeredgecolor='black', markersize=11, label='Selected operating point'),
    ]
    fig.suptitle("SEN-SPE Frontier Context Under Optional Reference Smoothing",
                 fontsize=16, fontweight='bold', y=1.02)
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "fig8_rabc_frontier.png"
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
