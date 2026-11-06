#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter, label as ndi_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import load_config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EPS = 1e-7


def model_kwargs_from_cfg(cfg: dict) -> dict:
    m = cfg["model"]
    known_bool_keys = [
        "use_boundary_head",
        "use_uncertainty_head",
        "refine_with_heads",
        "use_aspp",
        "use_aux_head",
        "use_se_context",
        "use_detail_fusion",
        "use_coord_attn",
        "use_edge_prompt",
        "use_dual_supervised_head",
        "use_eca_context",
        "use_ema_context",
        "use_cbam_decoder",
        "use_simam_refine",
        "use_safm_decoder",
        "use_uncertainty_gated_refine",
        "use_sk_context",
        "use_triplet_decoder",
        "use_msaa_fusion",
        "use_ega_refine",
        "use_cot_context",
        "use_lsk_decoder",
        "use_ucgf_refine",
        "use_anchor_context",
        "use_smfa_decoder",
        "use_tff_refine",
        "use_mfe_context",
        "use_bg_decouple_head",
        "use_uncertainty_logit_gate",
        "use_global_logit_calibrator",
        "use_boundary_uncertainty_logit_refine",
        "use_residual_logit_refine",
        "use_mamba_logit_refine",
        "use_cgfn_logit_refine",
        "use_recall_recover_refine",
        "use_recall_candidate_head",
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


def denorm(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def dice(mask: np.ndarray, gt: np.ndarray) -> float:
    p = (mask > 0).astype(np.float32)
    g = (gt > 0).astype(np.float32)
    inter = float((p * g).sum())
    return (2.0 * inter + EPS) / (float(p.sum() + g.sum()) + EPS)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    return inter / (union + EPS)


def overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.36) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask > 0
    c = np.array(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    boundary = binary_dilation(m, iterations=2) & (~binary_erosion(m, iterations=1))
    out[boundary] = np.array([255, 255, 255], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def postprocess(prob: np.ndarray, thr: float, sigma: float, fill_h: bool, keep_largest: bool) -> np.ndarray:
    if sigma > 0:
        prob = gaussian_filter(prob, sigma=sigma)
    mask = prob > thr
    if fill_h:
        mask = binary_fill_holes(mask)
    if keep_largest:
        lbl, n = ndi_label(mask)
        if n > 0:
            areas = np.bincount(lbl.ravel())
            areas[0] = 0
            mask = lbl == int(np.argmax(areas))
    return mask.astype(np.uint8)


def tta_views() -> list[tuple[bool, bool]]:
    return [(False, False), (True, False), (False, True), (True, True)]


@torch.inference_mode()
def compute_interaction_consensus_and_final(
    model: RPINet,
    image_t: torch.Tensor,
    pseudo_t: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    img = image_t.unsqueeze(0).to(device)
    if channels_last:
        img = img.contiguous(memory_format=torch.channels_last)
    pseudo = pseudo_t.unsqueeze(0).to(device)

    # IPC/PIA -> sigma maps -> uncertainty-weighted pseudo consensus.
    out_inter = model(img, pseudo)
    sigma_maps = out_inter.get("sigma_maps", [])
    if not isinstance(sigma_maps, list) or len(sigma_maps) != 4:
        raise RuntimeError("Expected 4 sigma maps from IPC/PIA interaction branch.")

    sigma_np = np.stack([s[0, 0].detach().cpu().numpy() for s in sigma_maps], axis=0)
    pseudo_np = pseudo_t.detach().cpu().numpy()
    weights = np.exp(-np.clip(sigma_np, -5.0, 5.0))
    weights /= np.sum(weights, axis=0, keepdims=True) + 1e-8
    consensus_prob = np.sum(weights * pseudo_np, axis=0)
    consensus_mask = (consensus_prob > 0.5).astype(np.uint8)

    # Final output: full inference setting used in paper for ISIC-2017.
    prob_sum = None
    for hf, vf in tta_views():
        x = img
        if hf:
            x = torch.flip(x, dims=[3])
        if vf:
            x = torch.flip(x, dims=[2])
        p = torch.sigmoid(model(x, None)["logits"])
        if vf:
            p = torch.flip(p, dims=[2])
        if hf:
            p = torch.flip(p, dims=[3])
        prob_sum = p if prob_sum is None else (prob_sum + p)
    prob = (prob_sum / 4.0)[0, 0].detach().cpu().numpy()
    final_mask = postprocess(prob, thr=0.30, sigma=0.0, fill_h=True, keep_largest=True)
    return consensus_mask, final_mask


def select_cases(
    dataset: ISIC2017Dataset,
    top_scan: int = 80,
    keep_rows: int = 3,
) -> list[dict]:
    pseudo_dir = dataset.pseudo_dir
    if pseudo_dir is None:
        raise RuntimeError("dataset.pseudo_dir is required.")

    scored = []
    for idx, rec in enumerate(dataset.samples):
        pseudo_path = pseudo_dir / f"{rec.image_id}.npz"
        if not pseudo_path.exists():
            continue
        masks = np.load(pseudo_path)["masks"].astype(np.float32)
        if masks.shape[0] < 2:
            continue
        p1 = (masks[0] > 0.5).astype(np.uint8)
        p2 = (masks[1] > 0.5).astype(np.uint8)
        gt = (dataset._load_mask(rec.mask_path)[0].numpy() > 0.5).astype(np.uint8)
        p1_d = dice(p1, gt)
        p2_d = dice(p2, gt)
        div = 1.0 - iou(p1, p2)
        poor = 1.0 - max(p1_d, p2_d)
        score = div + 0.7 * poor
        scored.append(
            {
                "idx": idx,
                "id": rec.image_id,
                "p1": p1,
                "p2": p2,
                "gt": gt,
                "d1": p1_d,
                "d2": p2_d,
                "div": div,
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(top_scan, keep_rows)]


def build_figure(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=4,
        figsize=(14.0, 3.7 * len(rows)),
        gridspec_kw={"width_ratios": [1.0, 1.8, 1.0, 1.0]},
        dpi=180,
    )
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)

    titles = [
        "Original + GT",
        "Noisy Pseudo Labels (P1 | P2)",
        "IPC/PIA Consensus",
        "Final Output",
    ]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=13, fontweight="bold")

    for r, row in enumerate(rows):
        img = row["img"]
        gt = row["gt"]
        p1 = row["p1"]
        p2 = row["p2"]
        con = row["consensus"]
        pred = row["pred"]

        col1 = overlay(img, gt, (0, 200, 0), alpha=0.20)
        p1_vis = overlay(img, p1, (255, 140, 0), alpha=0.35)
        p2_vis = overlay(img, p2, (0, 190, 255), alpha=0.35)
        divider = 255 * np.ones((img.shape[0], 6, 3), dtype=np.uint8)
        col2 = np.concatenate([p1_vis, divider, p2_vis], axis=1)
        col3 = overlay(img, con, (220, 40, 170), alpha=0.35)
        col4 = overlay(img, pred, (35, 105, 225), alpha=0.35)

        cells = [col1, col2, col3, col4]
        for c, cell in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[r, 1].set_xlabel(
            f"DICE: P1={row['d1']*100:.1f}% | P2={row['d2']*100:.1f}%",
            fontsize=10,
            color="#9a4a00",
            labelpad=4,
        )
        axes[r, 2].set_xlabel(f"DICE={row['dc']*100:.1f}%", fontsize=10, color="#8a0f6f", labelpad=4)
        axes[r, 3].set_xlabel(f"DICE={row['df']*100:.1f}%", fontsize=10, color="#1240a8", labelpad=4)

    plt.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    cfg_path = ROOT / "configs" / "rpi_net_isic2017_innovation_rabc.yaml"
    ckpt_path = ROOT / "outputs" / "innovation" / "rpi_net_isic2017_rabc_v2" / "checkpoints" / "best.pt"

    cfg = load_config(cfg_path)
    data_cfg = cfg["data"]
    pseudo_cfg = cfg["pseudo"]

    dataset = ISIC2017Dataset(
        root=data_cfg["root"],
        split=data_cfg["train_split"],
        image_size=int(data_cfg["image_size"]),
        pseudo_dir=Path(pseudo_cfg["output_dir"]) / data_cfg["train_split"],
        require_pseudo=True,
        augment=False,
        use_hair_removal=bool(data_cfg.get("use_hair_removal", False)),
        cache_images=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    kwargs = model_kwargs_from_cfg(cfg)
    kwargs["backbone_pretrained"] = False
    model = RPINet(**kwargs)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    selected_ids = ["ISIC_0012677", "ISIC_0013879", "ISIC_0014341"]
    id_to_idx = {rec.image_id: idx for idx, rec in enumerate(dataset.samples)}
    candidates = []
    for sid in selected_ids:
        idx = id_to_idx[sid]
        rec = dataset.samples[idx]
        pseudo_path = dataset.pseudo_dir / f"{sid}.npz"
        masks = np.load(pseudo_path)["masks"].astype(np.float32)
        p1 = (masks[0] > 0.5).astype(np.uint8)
        p2 = (masks[1] > 0.5).astype(np.uint8)
        gt = (dataset._load_mask(rec.mask_path)[0].numpy() > 0.5).astype(np.uint8)
        candidates.append({"idx": idx, "id": sid, "p1": p1, "p2": p2, "gt": gt, "d1": dice(p1, gt), "d2": dice(p2, gt), "div": 1.0 - iou(p1, p2)})
    enriched = []
    for cand in candidates:
        sample = dataset[cand["idx"]]
        img = denorm(sample["image"])
        gt = (sample["mask"][0].numpy() > 0.5).astype(np.uint8)
        pseudo = sample["pseudo"]
        con_mask, pred = compute_interaction_consensus_and_final(
            model=model,
            image_t=sample["image"],
            pseudo_t=pseudo,
            device=device,
            channels_last=channels_last,
        )
        dc = dice(con_mask, gt)
        df = dice(pred, gt)
        imp = (df - max(cand["d1"], cand["d2"])) + 0.5 * (df - dc) + 0.3 * cand["div"]
        enriched.append(
            {
                "id": cand["id"],
                "img": img,
                "gt": gt,
                "p1": cand["p1"],
                "p2": cand["p2"],
                "d1": cand["d1"],
                "d2": cand["d2"],
                "consensus": con_mask,
                "dc": dc,
                "pred": pred,
                "df": df,
                "imp": imp,
            }
        )

    enriched.sort(key=lambda x: x["imp"], reverse=True)
    rows = enriched[:3]

    out_path = ROOT / "paper" / "figures" / "fig6_pseudo_flow.png"
    build_figure(rows, out_path)
    print("Saved:", out_path)
    print("Selected IDs:", [r["id"] for r in rows])
    for r in rows:
        print(
            f"{r['id']} | P1={r['d1']*100:.1f}% P2={r['d2']*100:.1f}% "
            f"Consensus={r['dc']*100:.1f}% Final={r['df']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
