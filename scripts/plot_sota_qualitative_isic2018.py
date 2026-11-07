#!/usr/bin/env python3
"""Render a qualitative comparison on ISIC-2018 with strong unsupervised references.

Rows: selected ISIC-2018 test cases.
Cols: Image | GT | RPI-inspired anchor | RABC-Net | Error Map
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import load_config

EPS = 1e-7
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEDSAM_CKPT = Path(os.environ.get("MEDSAM_CKPT", Path.home() / ".cache" / "medsam"))

SELECTED_IDS = [
    "ISIC_0024191",
    "ISIC_0012572",
    "ISIC_0015478",
    "ISIC_0015486",
]

MODELS = {
    "rpinet": {
        "kind": "rpi",
        "title": "RPI-inspired anchor",
        "color": "#c15600",
        "overlay_rgb": (210, 90, 0),
        "config": ROOT / "configs" / "rpi_net_isic2018_unsup_v17_transfer.yaml",
        "checkpoint": ROOT / "outputs" / "rpi_net_isic2018_unsup_v17_retrain" / "checkpoints" / "best.pt",
        "threshold": 0.65,
        "tta": "none",
        "fill_holes": False,
        "keep_largest": False,
    },
    "ours": {
        "kind": "rpi",
        "title": "RABC-Net",
        "color": "#1a4fcc",
        "overlay_rgb": (30, 100, 220),
        "config": ROOT / "configs" / "rpi_net_isic2018_innovation_rabc.yaml",
        "checkpoint": ROOT / "outputs" / "innovation" / "rpi_net_isic2018_rabc" / "checkpoints" / "best.pt",
        "threshold": 0.13,
        "tta": "none",
        "fill_holes": False,
        "keep_largest": False,
    },
}


def model_kwargs_from_cfg(cfg: dict) -> dict:
    model_cfg = cfg["model"]
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
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "backbone": str(model_cfg.get("backbone", "resnet50")),
        "backbone_pretrained": bool(model_cfg.get("backbone_pretrained", True)),
    }
    for key in known_bool_keys:
        kwargs[key] = bool(model_cfg.get(key, False))
    return kwargs


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported TTA mode: {mode}")


def postprocess(prob: np.ndarray, threshold: float, fill_holes: bool, keep_largest: bool) -> np.ndarray:
    mask = prob > threshold
    if fill_holes:
        from scipy.ndimage import binary_fill_holes, label as ndi_label

        mask = binary_fill_holes(mask)
        if keep_largest:
            labeled, n = ndi_label(mask)
            if n > 0:
                areas = np.bincount(labeled.ravel())
                areas[0] = 0
                mask = labeled == int(np.argmax(areas))
    return mask.astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    return float((2.0 * (pred * gt).sum() + EPS) / (pred.sum() + gt.sum() + EPS))


def denorm(img_t: torch.Tensor) -> np.ndarray:
    array = img_t.cpu().numpy().transpose(1, 2, 0)
    array = array * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def overlay_boundary(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.38) -> np.ndarray:
    out = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    for channel, value in enumerate(color):
        out[mask_bool, channel] = out[mask_bool, channel] * (1.0 - alpha) + value * alpha
    boundary = binary_dilation(mask_bool, iterations=2) & ~binary_erosion(mask_bool, iterations=1)
    out[boundary] = [255, 255, 255]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    error = np.zeros((*pred.shape, 3), dtype=np.uint8)
    error[(pred == 1) & (gt == 1)] = [0, 200, 0]
    error[(pred == 1) & (gt == 0)] = [220, 0, 0]
    error[(pred == 0) & (gt == 1)] = [255, 220, 0]
    return error


def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [int(x0), int(y0), int(x1), int(y1)]


def load_model(spec: dict, device: torch.device) -> dict:
    if spec["kind"] == "rpi":
        cfg = load_config(spec["config"])
        channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))
        checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        kwargs = model_kwargs_from_cfg(cfg)
        # The checkpoint contains the trained backbone; avoid network downloads while rendering figures.
        kwargs["backbone_pretrained"] = False
        model = RPINet(**kwargs)
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        model.eval()
        return {"kind": "rpi", "model": model, "channels_last": channels_last}

    if spec["kind"] == "medsam":
        model = SamModel.from_pretrained(str(MEDSAM_CKPT)).to(device)
        processor = SamProcessor.from_pretrained(str(MEDSAM_CKPT))
        model.eval()
        return {"kind": "medsam", "model": model, "processor": processor}

    raise ValueError(f"Unsupported model kind: {spec['kind']}")


@torch.inference_mode()
def predict_rpi(model: torch.nn.Module, image_t: torch.Tensor, spec: dict, channels_last: bool) -> np.ndarray:
    image_t = image_t.unsqueeze(0)
    if channels_last:
        image_t = image_t.contiguous(memory_format=torch.channels_last)
    image_t = image_t.to(next(model.parameters()).device)

    prob_sum = None
    for hflip, vflip in tta_views(spec["tta"]):
        x = image_t
        if hflip:
            x = torch.flip(x, dims=[3])
        if vflip:
            x = torch.flip(x, dims=[2])
        logits = model(x, None)["logits"]
        prob = torch.sigmoid(logits)
        if vflip:
            prob = torch.flip(prob, dims=[2])
        if hflip:
            prob = torch.flip(prob, dims=[3])
        prob_sum = prob if prob_sum is None else prob_sum + prob
    prob = (prob_sum / len(tta_views(spec["tta"]))).cpu().numpy()[0, 0]
    return postprocess(prob, spec["threshold"], spec["fill_holes"], spec["keep_largest"])


@torch.inference_mode()
def predict_medsam(model: SamModel, processor: SamProcessor, image: np.ndarray, gt: np.ndarray, threshold: float) -> np.ndarray:
    bbox = mask_to_bbox(gt.astype(bool))
    if bbox is None:
        return np.zeros_like(gt, dtype=np.uint8)

    image_pil = Image.fromarray(image)
    inputs = processor(images=image_pil, input_boxes=[[[bbox]]], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, multimask_output=False)
    logit = outputs.pred_masks[0, 0, 0]
    prob_t = torch.sigmoid(logit).unsqueeze(0).unsqueeze(0)
    prob = torch.nn.functional.interpolate(
        prob_t,
        size=gt.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def predict(loaded: dict, image_t: torch.Tensor, image: np.ndarray, gt: np.ndarray, spec: dict) -> np.ndarray:
    if loaded["kind"] == "rpi":
        return predict_rpi(loaded["model"], image_t, spec, loaded["channels_last"])
    if loaded["kind"] == "medsam":
        return predict_medsam(loaded["model"], loaded["processor"], image, gt, spec["threshold"])
    raise ValueError(f"Unsupported loaded model kind: {loaded['kind']}")


def collect_cases(device: torch.device) -> list[dict]:
    ours_cfg = load_config(MODELS["ours"]["config"])
    data_cfg = ours_cfg["data"]
    dataset = ISIC2017Dataset(
        root=data_cfg["root"],
        split=data_cfg["test_split"],
        image_size=data_cfg["image_size"],
        augment=False,
        use_hair_removal=False,
        ph2_split_seed=int(data_cfg.get("ph2_split_seed", 42)),
        ph2_split_ratios=list(data_cfg.get("ph2_split_ratios", [0.7, 0.15, 0.15])),
    )
    id_to_index = {sample.image_id: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in SELECTED_IDS if sample_id not in id_to_index]
    if missing:
        raise KeyError(f"Missing selected IDs in dataset: {missing}")

    model_names = list(MODELS.keys())
    loaded = {name: load_model(spec, device) for name, spec in MODELS.items()}
    results: list[dict] = []
    for sample_id in SELECTED_IDS:
        sample = dataset[id_to_index[sample_id]]
        gt = (sample["mask"].numpy() > 0.5).astype(np.uint8)
        if gt.ndim == 3:
            gt = gt[0]
        image = denorm(sample["image"])

        preds = {}
        dices = {}
        for model_name in model_names:
            pred = predict(loaded[model_name], sample["image"], image, gt, MODELS[model_name])
            preds[model_name] = pred
            dices[model_name] = dice_score(pred, gt)

        results.append(
            {
                "id": sample_id,
                "image": image,
                "gt": gt,
                "preds": preds,
                "dices": dices,
            }
        )
        metric_str = "  ".join(f"{name}={dices[name]*100:.2f}%" for name in model_names)
        print(f"{sample_id}: {metric_str}")
    return results


def render(results: list[dict], out_path: Path) -> None:
    n_rows = len(results)
    model_names = list(MODELS.keys())
    col_titles = ["Image", "Ground Truth"] + [MODELS[name]["title"] for name in model_names] + ["Error Map"]
    col_colors = ["#111111", "#1a7a1a"] + [MODELS[name]["color"] for name in model_names] + ["#111111"]

    fig, axes = plt.subplots(n_rows, len(col_titles), figsize=(14.2, 2.35 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col].set_title(title, fontsize=11.2, fontweight="bold", color=color, pad=5)

    for row, case in enumerate(results):
        gt_overlay = overlay_boundary(case["image"], case["gt"], (0, 200, 0))
        pred_overlays = [
            overlay_boundary(case["image"], case["preds"][name], MODELS[name]["overlay_rgb"])
            for name in model_names
        ]
        error_map = make_error_map(case["preds"]["ours"], case["gt"])
        cells = [case["image"], gt_overlay] + pred_overlays + [error_map]

        for col, cell in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for col_idx, model_name in enumerate(model_names, start=2):
            axes[row, col_idx].set_xlabel(
                f"DIC={case['dices'][model_name]*100:.1f}%",
                fontsize=8.5,
                color=MODELS[model_name]["color"],
                labelpad=2,
            )

    legend_items = [
        mpatches.Patch(color=(0 / 255, 200 / 255, 0 / 255), label="TP"),
        mpatches.Patch(color=(220 / 255, 0 / 255, 0 / 255), label="FP"),
        mpatches.Patch(color=(255 / 255, 220 / 255, 0 / 255), label="FN"),
        mpatches.Patch(color="black", label="TN"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    plt.tight_layout(pad=0.35, h_pad=0.32, w_pad=0.18, rect=(0.0, 0.055, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    results = collect_cases(device)
    render(results, ROOT / "paper" / "figures" / "fig5_sota_qualitative.png")


if __name__ == "__main__":
    main()
