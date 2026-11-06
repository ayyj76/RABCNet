#!/usr/bin/env python3
"""
Fig 2: Qualitative segmentation comparison.
Each row: Image | GT | Raw P0 | RABC-Net | Error Map
3 datasets × 2 samples each = 6 rows total.
Samples are specified by ID from pre-selected candidates.
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
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── 各数据集配置 + 指定样本ID ──────────────────────────────────────────
DATASETS = [
    dict(
        name="ISIC-2017",
        config="configs/rpi_net_isic2017_innovation_rabc.yaml",
        ckpt="outputs/innovation/rpi_net_isic2017_rabc_v2/checkpoints/best.pt",
        threshold=0.30, sigma=0.0, tta="flip4",
        fill_holes=True, keep_largest=True,
        no_tensor_cache=True,
        # 候选: ISIC_0014284(Δ+0.237) ISIC_0015309(Δ+0.180) ISIC_0013739(Δ+0.182) ISIC_0015944(Δ+0.283)
        selected_ids=["ISIC_0014284", "ISIC_0015309"],
    ),
    dict(
        name="ISIC-2018",
        config="configs/rpi_net_isic2018_innovation_rabc.yaml",
        ckpt="outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt",
        threshold=0.25, sigma=0.0, tta="flip4",
        fill_holes=True, keep_largest=True,
        no_tensor_cache=False,
        # 候选: ISIC_0036099(Δ+0.164) ISIC_0036090(Δ+0.162) ISIC_0021147(Δ+0.150) ISIC_0036211(Δ+0.141)
        selected_ids=["ISIC_0015265", "ISIC_0023283"],
    ),
    dict(
        name="PH2",
        config="configs/rpi_net_ph2_innovation_rabc.yaml",
        ckpt="outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt",
        threshold=0.06, sigma=0.0, tta="flip4",
        fill_holes=True, keep_largest=True,
        no_tensor_cache=False,
        # 候选: IMD378(Δ+0.047) IMD090(Δ+0.037) IMD390(Δ+0.031) IMD177(Δ+0.027)
        selected_ids=["IMD378", "IMD090"],
    ),
]


# ── 辅助函数 ──────────────────────────────────────────────────────────

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
        "use_multiscale_recall_head",
        "use_rabc",
        "use_brr",
    ]
    kwargs = {
        "hidden_dim": int(m.get("hidden_dim", 128)),
        "dropout": float(m.get("dropout", 0.1)),
        "backbone": str(m.get("backbone", "resnet50")),
        "backbone_pretrained": bool(m.get("backbone_pretrained", True)),
    }
    for k in known_bool_keys:
        kwargs[k] = bool(m.get(k, False))
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


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(float), gt.astype(float)
    return float((2 * (p * g).sum() + EPS) / (p.sum() + g.sum() + EPS))


def denorm(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def make_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    err = np.zeros((*pred.shape, 3), dtype=np.uint8)
    err[(pred == 1) & (gt == 1)] = [0,   200, 0]    # TP 绿
    err[(pred == 1) & (gt == 0)] = [220, 0,   0]    # FP 红
    err[(pred == 0) & (gt == 1)] = [255, 220, 0]    # FN 黄
    # TN: black (already 0)
    return err


def overlay_boundary(img_rgb: np.ndarray, mask: np.ndarray,
                     color: tuple, alpha: float = 0.40) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion
    out = img_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    for c, v in enumerate(color):
        out[m, c] = out[m, c] * (1 - alpha) + v * alpha
    boundary = binary_dilation(m, iterations=2) & ~binary_erosion(m, iterations=1)
    out[boundary] = [255, 255, 255]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── 数据加载与推理 ─────────────────────────────────────────────────────

@torch.inference_mode()
def collect_selected_samples(ds_cfg: dict, device: torch.device) -> list[dict]:
    cfg = load_config(ROOT / ds_cfg["config"])
    data_cfg = cfg["data"]
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))

    ds = ISIC2017Dataset(
        root=data_cfg["root"],
        split=data_cfg["test_split"],
        image_size=data_cfg["image_size"],
        augment=False,
        use_hair_removal=False,
        ph2_split_seed=int(data_cfg.get("ph2_split_seed", 42)),
        ph2_split_ratios=list(data_cfg.get("ph2_split_ratios", [0.7, 0.15, 0.15])),
    )

    # build id -> index map
    id_to_idx = {s.image_id: i for i, s in enumerate(ds.samples)}
    selected_ids = ds_cfg["selected_ids"]
    missing = [sid for sid in selected_ids if sid not in id_to_idx]
    if missing:
        raise KeyError(f"IDs not found in dataset {ds_cfg['name']}: {missing}")

    # load model
    ckpt = torch.load(ROOT / ds_cfg["ckpt"], map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    kwargs = model_kwargs_from_cfg(cfg)
    # Checkpoints contain trained backbone weights; avoid network downloads during rendering.
    kwargs["backbone_pretrained"] = False
    model = RPINet(**kwargs)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    views_full = tta_views(ds_cfg["tta"])

    results = []
    for sid in selected_ids:
        idx = id_to_idx[sid]
        sample = ds[idx]
        img_t = sample["image"].unsqueeze(0).to(device)
        if channels_last:
            img_t = img_t.contiguous(memory_format=torch.channels_last)
        gt = (sample["mask"].numpy() > 0.5).astype(np.uint8)
        if gt.ndim == 3:
            gt = gt[0]

        # Full pipeline
        prob_sum = None
        for hf, vf in views_full:
            x = img_t
            if hf: x = torch.flip(x, dims=[3])
            if vf: x = torch.flip(x, dims=[2])
            p = torch.sigmoid(model(x, None)["logits"])
            if vf: p = torch.flip(p, dims=[2])
            if hf: p = torch.flip(p, dims=[3])
            prob_sum = p if prob_sum is None else prob_sum + p
        prob_full = (prob_sum / len(views_full))[0, 0].cpu().numpy()
        pred_full = postprocess(prob_full, ds_cfg["threshold"], ds_cfg["sigma"],
                                ds_cfg["fill_holes"], ds_cfg["keep_largest"])

        # Raw P0: tau=0.30, no TTA, no post-processing
        prob_raw = torch.sigmoid(model(img_t, None)["logits"])[0, 0].cpu().numpy()
        pred_p0 = postprocess(prob_raw, 0.30, 0.0, False, False)

        img_np = denorm(sample["image"])

        results.append({
            "id":        sid,
            "img":       img_np,
            "gt":        gt,
            "pred_full": pred_full,
            "pred_p0":   pred_p0,
            "dice_full": dice_score(pred_full, gt),
            "dice_p0":   dice_score(pred_p0,   gt),
        })
        print(f"  {sid}: P0={results[-1]['dice_p0']:.3f}  Full={results[-1]['dice_full']:.3f}")

    return results


# ── 排版与输出 ─────────────────────────────────────────────────────────

def make_figure(all_results: list[tuple[str, list[dict]]], out_path: Path):
    n_cols = 5
    n_rows = sum(len(samples) for _, samples in all_results)

    col_titles = ["Image", "Ground Truth", "Raw P0", "RABC-Net", "Error Map"]
    col_colors = ["#111111", "#1a7a1a", "#d06000", "#1a4fcc", "#111111"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.4),
        dpi=150,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for c, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold",
                             color=color, pad=5)

    row = 0
    for ds_name, samples in all_results:
        n = len(samples)
        for s_idx, s in enumerate(samples):
            gt_ov   = overlay_boundary(s["img"], s["gt"],       (0,   200, 0))
            p0_ov   = overlay_boundary(s["img"], s["pred_p0"],  (255, 140, 0))
            full_ov = overlay_boundary(s["img"], s["pred_full"],(30,  100, 220))
            err     = make_error_map(s["pred_full"], s["gt"])

            for c, cell in enumerate([s["img"], gt_ov, p0_ov, full_ov, err]):
                ax = axes[row, c]
                ax.imshow(cell)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == 2:
                    ax.set_xlabel(f"DIC={s['dice_p0']*100:.1f}%", fontsize=11,
                                  color="#d06000", labelpad=2)
                if c == 3:
                    ax.set_xlabel(f"DIC={s['dice_full']*100:.1f}%", fontsize=11,
                                  color="#1a4fcc", labelpad=2)

            # dataset label on middle row of each group
            if s_idx == n // 2:
                axes[row, 0].set_ylabel(ds_name, fontsize=13, fontweight="bold",
                                        rotation=90, labelpad=6)
            row += 1

    legend_items = [
        mpatches.Patch(color=(0/255,   200/255, 0/255), label="TP"),
        mpatches.Patch(color=(220/255, 0/255,   0/255), label="FP"),
        mpatches.Patch(color=(255/255, 220/255, 0/255), label="FN"),
        mpatches.Patch(color="black",                   label="TN"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.04),
               title="Error Map (RABC-Net vs GT)", title_fontsize=11)

    plt.tight_layout(pad=0.5, h_pad=0.4, w_pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[done] saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    all_results = []
    for ds_cfg in DATASETS:
        print(f"--- {ds_cfg['name']} ---")
        samples = collect_selected_samples(ds_cfg, device)
        all_results.append((ds_cfg["name"], samples))

    out_path = ROOT / "paper" / "figures" / "fig4_qualitative_comparison.png"
    make_figure(all_results, out_path)


if __name__ == "__main__":
    main()
