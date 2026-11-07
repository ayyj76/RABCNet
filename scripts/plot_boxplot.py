#!/usr/bin/env python3
"""Draw per-sample metric distributions for the current RABC-Net checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, gaussian_filter, label as ndi_label
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
        config="configs/rpi_net_isic2017_innovation_rabc.yaml",
        ckpt="outputs/innovation/rpi_net_isic2017_rabc_v2/checkpoints/best.pt",
        threshold=0.30,
        sigma=0.0,
        tta="flip4",
        fill_holes=True,
        keep_largest=True,
    ),
    dict(
        name="ISIC-2018",
        config="configs/rpi_net_isic2018_innovation_rabc.yaml",
        ckpt="outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt",
        threshold=0.25,
        sigma=0.0,
        tta="flip4",
        fill_holes=True,
        keep_largest=True,
    ),
    dict(
        name="PH2",
        config="configs/rpi_net_ph2_innovation_rabc.yaml",
        ckpt="outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt",
        threshold=0.06,
        sigma=0.0,
        tta="flip4",
        fill_holes=True,
        keep_largest=True,
    ),
]


def model_kwargs_from_cfg(cfg: dict) -> dict:
    m = cfg["model"]
    known_bool_keys = [
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
        "use_multiscale_recall_head", "use_rabc", "use_brr",
    ]
    kwargs = {
        "hidden_dim": int(m.get("hidden_dim", 128)),
        "dropout": float(m.get("dropout", 0.1)),
        "backbone": str(m.get("backbone", "resnet50")),
        # Checkpoints include trained backbone weights; avoid network downloads.
        "backbone_pretrained": False,
    }
    for k in known_bool_keys:
        kwargs[k] = bool(m.get(k, False))
    return kwargs


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(mode)


def postprocess(prob: np.ndarray, thr: float, sigma: float, fill_holes: bool, keep_largest: bool) -> np.ndarray:
    if sigma > 0:
        prob = gaussian_filter(prob, sigma=sigma)
    mask = prob > thr
    if fill_holes:
        mask = binary_fill_holes(mask)
    if keep_largest:
        lbl, n = ndi_label(mask)
        if n > 0:
            areas = np.bincount(lbl.ravel())
            areas[0] = 0
            mask = lbl == int(np.argmax(areas))
    return mask.astype(np.uint8)


def compute_per_sample_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)
    tp = (p * g).sum(axis=(1, 2))
    fp = (p * (1 - g)).sum(axis=(1, 2))
    fn = ((1 - p) * g).sum(axis=(1, 2))
    tn = ((1 - p) * (1 - g)).sum(axis=(1, 2))
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "jac": (tp + EPS) / (tp + fp + fn + EPS),
        "sen": (tp + EPS) / (tp + fn + EPS),
        "spe": (tn + EPS) / (tn + fp + EPS),
        "acc": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
    }


@torch.inference_mode()
def collect_metrics(ds_cfg: dict, device: torch.device) -> dict[str, np.ndarray]:
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
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    ckpt = torch.load(ROOT / ds_cfg["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model = RPINet(**model_kwargs_from_cfg(cfg))
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    views = tta_views(ds_cfg["tta"])
    all_gts, all_preds = [], []
    for batch in loader:
        imgs_t = batch["image"].to(device)
        gts_t = batch["mask"]
        if channels_last:
            imgs_t = imgs_t.contiguous(memory_format=torch.channels_last)

        prob_sum = None
        for hf, vf in views:
            xi = imgs_t
            if hf:
                xi = torch.flip(xi, dims=[3])
            if vf:
                xi = torch.flip(xi, dims=[2])
            p = torch.sigmoid(model(xi, None)["logits"])
            if vf:
                p = torch.flip(p, dims=[2])
            if hf:
                p = torch.flip(p, dims=[3])
            prob_sum = p if prob_sum is None else prob_sum + p

        probs = (prob_sum / len(views))[:, 0].cpu().numpy()
        gts = (gts_t[:, 0].numpy() > 0.5).astype(np.uint8)
        preds = np.stack([
            postprocess(probs[i], ds_cfg["threshold"], ds_cfg["sigma"], ds_cfg["fill_holes"], ds_cfg["keep_largest"])
            for i in range(len(probs))
        ])
        all_gts.append(gts)
        all_preds.append(preds)

    all_gts = np.concatenate(all_gts, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    metrics = compute_per_sample_metrics(all_preds, all_gts)
    print(
        f"  n={len(all_gts)}  DICE={metrics['dice'].mean()*100:.2f}%  "
        f"JAC={metrics['jac'].mean()*100:.2f}%  SEN={metrics['sen'].mean()*100:.2f}%  "
        f"SPE={metrics['spe'].mean()*100:.2f}%  ACC={metrics['acc'].mean()*100:.2f}%"
    )
    return {k: v * 100 for k, v in metrics.items()}


def make_figure(all_metrics: list[dict[str, np.ndarray]], ds_names: list[str], out_path: Path) -> None:
    metrics = ["dice", "jac", "sen", "spe", "acc"]
    labels = ["DICE", "JAC", "SEN", "SPE", "ACC"]
    ds_colors = ["#2E86C1", "#1A9E5A", "#C0392B"]
    ds_light = ["#AED6F1", "#A9DFBF", "#F1948A"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5), dpi=150)
    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[mi]
        data = [m[metric] for m in all_metrics]
        means = [d.mean() for d in data]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.2, linestyle="--", color="#555"),
            capprops=dict(linewidth=1.5, color="#555"),
            flierprops=dict(marker="o", markersize=3, alpha=0.5, markeredgewidth=0.5),
            notch=False,
        )
        for patch, fc, ec in zip(bp["boxes"], ds_light, ds_colors):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.5)
        for flier, ec in zip(bp["fliers"], ds_colors):
            flier.set_markerfacecolor(ec)
            flier.set_markeredgecolor(ec)
        for xi, (mean, color) in enumerate(zip(means, ds_colors), start=1):
            ax.plot(xi, mean, marker="D", color=color, markersize=6, zorder=5, markeredgecolor="white", markeredgewidth=0.8)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([n.replace("ISIC-", "ISIC\n") for n in ds_names], fontsize=8.5)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=18)
        ax.set_ylabel("Score (%)" if mi == 0 else "", fontsize=9)
        all_vals = np.concatenate(data)
        ymin = max(0, np.percentile(all_vals, 1) - 5)
        ymax = min(100, np.percentile(all_vals, 99) + (18 if mi >= 2 else 12))
        ax.set_ylim(ymin, ymax)

        whisker_tops = [bp["whiskers"][i * 2 + 1].get_ydata()[1] for i in range(len(data))]
        offset = 2.5 if mi >= 2 else 1.2
        for xi, (mean, color, wtop) in enumerate(zip(means, ds_colors, whisker_tops), start=1):
            y = min(wtop + offset, ymax - 0.8)
            ax.text(xi, y, f"{mean:.1f}", ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="gray", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)

    legend_items = [
        mpatches.Patch(facecolor=ds_light[i], edgecolor=ds_colors[i], linewidth=1.5, label=ds_names[i])
        for i in range(len(ds_names))
    ]
    legend_items.append(plt.Line2D([0], [0], marker="D", color="gray", markersize=6, linestyle="None", label="Mean"))
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Per-Sample Metric Distribution on Test Sets", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(pad=1.2, w_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw per-sample metric distributions for RABC-Net checkpoints")
    p.add_argument("--output", default=str(ROOT / "paper" / "figures" / "fig9_metric_boxplot.png"), help="Output PNG path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    all_metrics, ds_names = [], []
    for ds_cfg in DATASETS:
        print(f"--- {ds_cfg['name']} ---")
        all_metrics.append(collect_metrics(ds_cfg, device))
        ds_names.append(ds_cfg["name"])
    make_figure(all_metrics, ds_names, Path(args.output))


if __name__ == "__main__":
    main()
