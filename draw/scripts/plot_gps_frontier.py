#!/usr/bin/env python3
"""
Plot GPS operating frontier: SEN vs SPE curves comparing threshold-only vs GPS.

Multiple sigma values, zoomed into the high-SPE region, with fill-between
to highlight the GPS gain area.

Usage:
    python scripts/plot_gps_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                      pin_memory=True, persistent_workers=True, prefetch_factor=4)


@torch.inference_mode()
def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all, masks_all = [], []
    for batch in tqdm(loader, desc="collect-probs", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
        logits = model(image, None)["logits"]
        p = torch.sigmoid(logits)
        probs_all.append(p[:, 0].cpu().numpy())
        masks_all.append(mask[:, 0].cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(masks_all).astype(np.uint8)


def apply_gps(probs: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return probs
    return np.stack([gaussian_filter(probs[i], sigma=sigma) for i in range(probs.shape[0])])


def compute_sen_spe_at_threshold(probs: np.ndarray, masks: np.ndarray, threshold: float) -> tuple[float, float]:
    preds = (probs > threshold).astype(np.float32)
    t = masks.astype(np.float32)
    tp = (preds * t).sum(axis=(1, 2))
    fp = (preds * (1.0 - t)).sum(axis=(1, 2))
    fn = ((1.0 - preds) * t).sum(axis=(1, 2))
    tn = ((1.0 - preds) * (1.0 - t)).sum(axis=(1, 2))
    sen = float(np.mean((tp + EPS) / (tp + fn + EPS)))
    spe = float(np.mean((tn + EPS) / (tn + fp + EPS)))
    return sen, spe


def sweep_frontier(probs: np.ndarray, masks: np.ndarray, thresholds: np.ndarray) -> tuple[list[float], list[float]]:
    sens, spes = [], []
    for thr in thresholds:
        sen, spe = compute_sen_spe_at_threshold(probs, masks, thr)
        sens.append(sen)
        spes.append(spe)
    return sens, spes


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

        # Sweep for each sigma
        all_frontiers = {}
        for sigma in SIGMAS:
            probs_s = apply_gps(probs_raw, sigma)
            sens, spes = sweep_frontier(probs_s, masks, thresholds)
            all_frontiers[sigma] = (sens, spes)

            spe_pct = [s * 100 for s in spes]
            sen_pct = [s * 100 for s in sens]

            label = f'$\\sigma$={sigma:.1f}' if sigma > 0 else '$\\sigma$=0 (no GPS)'
            ax.plot(spe_pct, sen_pct,
                    color=colors[sigma],
                    linewidth=linewidths[sigma],
                    linestyle=linestyles[sigma],
                    label=label, zorder=3)

        # Fill between sigma=0 and sigma=1.0 to highlight GPS gain
        sens_0, spes_0 = all_frontiers[0.0]
        sens_1, spes_1 = all_frontiers[1.0]
        # Interpolate both curves onto common SPE grid for fill_between
        spe_grid = np.linspace(0.80, 1.0, 200)
        # For each curve, find SEN as function of SPE (reverse mapping)
        spe_arr_0 = np.array(spes_0)
        sen_arr_0 = np.array(sens_0)
        spe_arr_1 = np.array(spes_1)
        sen_arr_1 = np.array(sens_1)
        # Sort by SPE for interpolation
        order_0 = np.argsort(spe_arr_0)
        order_1 = np.argsort(spe_arr_1)
        sen_interp_0 = np.interp(spe_grid, spe_arr_0[order_0], sen_arr_0[order_0])
        sen_interp_1 = np.interp(spe_grid, spe_arr_1[order_1], sen_arr_1[order_1])
        ax.fill_between(spe_grid * 100, sen_interp_0 * 100, sen_interp_1 * 100,
                        where=sen_interp_1 > sen_interp_0,
                        alpha=0.15, color='#d62728', zorder=1,
                        label='GPS gain region')

        # Mark P0 operating point (thr=0.30, sigma=0)
        sen_p0, spe_p0 = compute_sen_spe_at_threshold(probs_raw, masks, 0.30)
        ax.plot(spe_p0*100, sen_p0*100, 's', color='#1f77b4', markersize=9, zorder=6,
                markeredgecolor='black', markeredgewidth=0.8)
        ax.annotate(f'P0\n($\\tau$=0.30)',
                    xy=(spe_p0*100, sen_p0*100),
                    xytext=(spe_p0*100 - 5, sen_p0*100 + 3),
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

        # Mark GPS final operating point
        p_sigma = ds_info["paper_sigma"]
        p_thr = ds_info["paper_thr"]
        probs_final = apply_gps(probs_raw, p_sigma)
        sen_gp, spe_gp = compute_sen_spe_at_threshold(probs_final, masks, p_thr)
        marker_color = colors.get(p_sigma, '#d62728')
        ax.plot(spe_gp*100, sen_gp*100, '*', color=marker_color, markersize=14, zorder=6,
                markeredgecolor='black', markeredgewidth=0.8)
        sigma_str = f'{p_sigma:.1f}' if p_sigma > 0 else '0'
        ax.annotate(f'Final\n($\\tau$={p_thr}, $\\sigma$={sigma_str})',
                    xy=(spe_gp*100, sen_gp*100),
                    xytext=(spe_gp*100 + 3, sen_gp*100 - 6),
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

        ax.set_xlabel("SPE (%)", fontsize=12)
        ax.set_ylabel("SEN (%)", fontsize=12)
        ax.set_title(ds_name, fontsize=14, fontweight='bold')
        ax.legend(fontsize=7.5, loc='lower left', framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--')

        # Zoom to interesting region
        ax.set_xlim(82, 100)
        ax.set_ylim(55, 100)

        print(f"  P0 (thr=0.30, σ=0): SEN={sen_p0*100:.2f}%, SPE={spe_p0*100:.2f}%")
        print(f"  Final (thr={p_thr}, σ={p_sigma}): SEN={sen_gp*100:.2f}%, SPE={spe_gp*100:.2f}%")

        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    output_path = OUTPUT_DIR / "gps_frontier.png"
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to: {output_path}")


if __name__ == "__main__":
    main()
