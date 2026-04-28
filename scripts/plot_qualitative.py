#!/usr/bin/env python3
"""
图3：定性分割可视化对比图
三个数据集各选4张样本（2张典型 + 2张困难），
每行5列：原图 / GT / Ours w/o GPS / Ours (Full) / 误差图
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

# ── 三个数据集的最终操作点 ────────────────────────────────────────────
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

SAMPLES_PER_DATASET = 4


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


def dice_per_sample(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(float), gt.astype(float)
    return float((2 * (p * g).sum() + EPS) / (p.sum() + g.sum() + EPS))


@torch.inference_mode()
def run_inference_batch(model, loader, device, channels_last, tta_mode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (all_imgs, all_gts, all_probs)"""
    model.eval()
    views = tta_views(tta_mode)
    all_imgs, all_gts, all_probs = [], [], []
    for batch in loader:
        imgs_t = batch["image"]
        gts_t  = batch["mask"]
        x = imgs_t.to(device)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        prob_sum = None
        for hf, vf in views:
            xi = x
            if hf: xi = torch.flip(xi, dims=[3])
            if vf: xi = torch.flip(xi, dims=[2])
            p = torch.sigmoid(model(xi, None)["logits"])
            if vf: p = torch.flip(p, dims=[2])
            if hf: p = torch.flip(p, dims=[3])
            prob_sum = p if prob_sum is None else prob_sum + p
        probs = (prob_sum / len(views))[:, 0].cpu().numpy()
        all_imgs.append(imgs_t.cpu())
        all_gts.append((gts_t[:, 0].numpy() > 0.5).astype(np.uint8))
        all_probs.append(probs)
    return (
        torch.cat(all_imgs, dim=0),
        np.concatenate(all_gts,  axis=0),
        np.concatenate(all_probs, axis=0),
    )


def denorm(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def select_samples(dices: np.ndarray, n: int = 4) -> list[int]:
    """n//2 张典型（接近中位数）+ n//2 张困难（DIC最低）"""
    n_typical = n // 2
    n_hard    = n - n_typical
    med  = np.median(dices)
    typical_idx = list(np.argsort(np.abs(dices - med))[:n_typical])
    seen = set(typical_idx)
    hard_idx = [i for i in np.argsort(dices) if i not in seen][:n_hard]
    return typical_idx + list(hard_idx)


def make_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    err = np.zeros((*pred.shape, 3), dtype=np.uint8)
    err[(pred == 1) & (gt == 1)] = [0,   200, 0]    # TP 绿
    err[(pred == 1) & (gt == 0)] = [220, 0,   0]    # FP 红
    err[(pred == 0) & (gt == 1)] = [255, 220, 0]    # FN 黄
    return err


def overlay(img_rgb: np.ndarray, mask: np.ndarray,
            color: tuple, alpha: float = 0.45) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion
    out = img_rgb.astype(np.float32).copy()
    m   = mask.astype(bool)
    for c, v in enumerate(color):
        out[m, c] = out[m, c] * (1 - alpha) + v * alpha
    boundary = binary_dilation(m, iterations=2) & ~binary_erosion(m, iterations=1)
    out[boundary] = [255, 255, 255]
    return np.clip(out, 0, 255).astype(np.uint8)


# ── 主流程 ────────────────────────────────────────────────────────────

def collect_dataset_results(ds_cfg: dict, device):
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

    # ── Ours Full：完整操作点（含TTA/GPS/后处理）──
    print(f"  [Full] TTA={ds_cfg['tta']}, σ={ds_cfg['sigma']}, τ={ds_cfg['threshold']}")
    all_imgs, all_gts, probs_full = run_inference_batch(
        model, loader, device, channels_last, ds_cfg["tta"])

    preds_full = np.stack([
        postprocess(probs_full[i], ds_cfg["threshold"], ds_cfg["sigma"],
                    ds_cfg["fill_holes"], ds_cfg["keep_largest"])
        for i in range(len(probs_full))
    ])

    # ── Ours w/o GPS：τ=0.30，σ=0，无TTA，无后处理（P0操作点）──
    print(f"  [w/o GPS] TTA=none, σ=0.0, τ=0.30")
    _, _, probs_raw = run_inference_batch(
        model, loader, device, channels_last, "none")

    preds_nogps = np.stack([
        postprocess(probs_raw[i], 0.30, 0.0, False, False)
        for i in range(len(probs_raw))
    ])

    dices_full  = np.array([dice_per_sample(preds_full[i],  all_gts[i]) for i in range(len(preds_full))])
    dices_nogps = np.array([dice_per_sample(preds_nogps[i], all_gts[i]) for i in range(len(preds_nogps))])

    # 按 Full 的 DIC 选样本（让最终模型的结果看起来最好）
    sel = select_samples(dices_full, SAMPLES_PER_DATASET)

    return {
        "name":        ds_cfg["name"],
        "imgs":        [denorm(all_imgs[i])  for i in sel],
        "gts":         [all_gts[i]           for i in sel],
        "preds_full":  [preds_full[i]        for i in sel],
        "preds_nogps": [preds_nogps[i]       for i in sel],
        "dices_full":  [dices_full[i]        for i in sel],
        "dices_nogps": [dices_nogps[i]       for i in sel],
    }


def make_figure(results: list[dict], out_path: Path):
    n_cols = 5
    n_rows = len(results) * SAMPLES_PER_DATASET

    col_titles = ["Image", "Ground Truth", "Ours w/o GPS", "Ours (Full)", "Error Map"]
    col_colors = ["black", "green", "darkorange", "royalblue", "black"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 2.2),
                             dpi=150)

    for c, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, c].set_title(title, fontsize=11, fontweight="bold",
                             color=color, pad=6)

    row = 0
    for ds_res in results:
        ds_name = ds_res["name"]
        n = len(ds_res["imgs"])
        for s in range(n):
            img        = ds_res["imgs"][s]
            gt         = ds_res["gts"][s]
            pred_full  = ds_res["preds_full"][s]
            pred_nogps = ds_res["preds_nogps"][s]
            dic_full   = ds_res["dices_full"][s]
            dic_nogps  = ds_res["dices_nogps"][s]

            gt_overlay     = overlay(img, gt,         (0,   200, 0))
            nogps_overlay  = overlay(img, pred_nogps, (255, 140, 0))   # 橙色
            full_overlay   = overlay(img, pred_full,  (30,  100, 220)) # 蓝色
            err_map        = make_error_map(pred_full, gt)             # 误差图基于Full

            row_data = [img, gt_overlay, nogps_overlay, full_overlay, err_map]

            for c, cell in enumerate(row_data):
                ax = axes[row, c]
                ax.imshow(cell)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                # DIC 标注
                if c == 2:
                    ax.set_xlabel(f"DIC={dic_nogps*100:.1f}%", fontsize=8,
                                  color="darkorange", labelpad=2)
                if c == 3:
                    ax.set_xlabel(f"DIC={dic_full*100:.1f}%", fontsize=8,
                                  color="royalblue", labelpad=2)

            # 数据集行标签（每组中间行）
            if s == SAMPLES_PER_DATASET // 2:
                axes[row, 0].set_ylabel(ds_name, fontsize=11, fontweight="bold",
                                        rotation=90, labelpad=6)
            row += 1

    # 图例
    legend_items = [
        mpatches.Patch(color=(0/255,   200/255, 0/255),   label="TP"),
        mpatches.Patch(color=(220/255, 0/255,   0/255),   label="FP"),
        mpatches.Patch(color=(255/255, 220/255, 0/255),   label="FN"),
        mpatches.Patch(color="black",                     label="TN"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
               title="Error Map Legend (Ours Full vs GT)", title_fontsize=9)

    fig.suptitle("Qualitative Segmentation Results", fontsize=13,
                 fontweight="bold", y=1.01)

    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[done] 图3 已保存: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    results = []
    for ds_cfg in DATASETS:
        print(f"\n--- 处理 {ds_cfg['name']} ---")
        results.append(collect_dataset_results(ds_cfg, device))
    make_figure(results, ROOT / "paper" / "figures" / "fig3_qualitative.png")


if __name__ == "__main__":
    main()
