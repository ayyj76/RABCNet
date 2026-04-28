#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import get_device, load_config


EPS = 1e-7
OUT_DIR = ROOT / "outputs" / "postprocess_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: Path
    checkpoint: Path
    threshold: float
    sigma: float
    tta: str
    fill_holes: bool
    keep_largest: bool


DATASETS = [
    DatasetSpec(
        name="ISIC-2017",
        config=ROOT / "configs" / "rpi_net_isic2017_unsup_unified_v20sigma.yaml",
        checkpoint=ROOT / "outputs" / "rpi_net_isic2017_unsup_unified_v20sigma" / "checkpoints" / "best.pt",
        threshold=0.02,
        sigma=1.0,
        tta="flip4",
        fill_holes=True,
        keep_largest=True,
    ),
    DatasetSpec(
        name="ISIC-2018",
        config=ROOT / "configs" / "rpi_net_isic2018_unsup_v20_sigma_only.yaml",
        checkpoint=ROOT / "outputs" / "rpi_net_isic2018_unsup_v20_sigma_only" / "checkpoints" / "best.pt",
        threshold=0.13,
        sigma=0.0,
        tta="none",
        fill_holes=False,
        keep_largest=False,
    ),
    DatasetSpec(
        name="PH2",
        config=ROOT / "configs" / "rpi_net_ph2_unsup_v20_sigma_only.yaml",
        checkpoint=ROOT / "outputs" / "rpi_net_ph2_unsup_unified_v20sigma" / "checkpoints" / "best.pt",
        threshold=0.10,
        sigma=1.0,
        tta="flip4",
        fill_holes=True,
        keep_largest=True,
    ),
]


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
        "hidden_dim": 128,
        "dropout": 0.1,
        "backbone": "resnet50",
        "backbone_pretrained": True,
    }
    result = {}
    for key in keys:
        val = model_cfg.get(key, defaults.get(key, False))
        if key == "hidden_dim":
            result[key] = int(val)
        elif key == "dropout":
            result[key] = float(val)
        elif key == "backbone":
            result[key] = str(val)
        else:
            result[key] = bool(val)
    return result


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported TTA mode: {mode}")


def build_loader(cfg: dict, split: str) -> DataLoader:
    data_cfg = cfg["data"]
    ds_kwargs = {
        "root": data_cfg["root"],
        "split": data_cfg[f"{split}_split"],
        "image_size": data_cfg["image_size"],
        "augment": False,
        "use_hair_removal": bool(data_cfg.get("use_hair_removal", False)),
    }
    if Path(data_cfg["root"]).name.lower() == "ph2":
        ds_kwargs["ph2_split_seed"] = int(data_cfg.get("ph2_split_seed", 42))
        ds_kwargs["ph2_split_ratios"] = tuple(data_cfg.get("ph2_split_ratios", [0.7, 0.15, 0.15]))
    dataset = ISIC2017Dataset(**ds_kwargs)
    workers = min(8, int(data_cfg.get("num_workers", 4)))
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict[str, float]:
    preds = preds.astype(bool)
    gts = gts.astype(bool)
    tp = np.logical_and(preds, gts).sum(axis=(1, 2)).astype(np.float64)
    fp = np.logical_and(preds, ~gts).sum(axis=(1, 2)).astype(np.float64)
    fn = np.logical_and(~preds, gts).sum(axis=(1, 2)).astype(np.float64)
    tn = np.logical_and(~preds, ~gts).sum(axis=(1, 2)).astype(np.float64)
    dice = np.mean((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS))
    jac = np.mean((tp + EPS) / (tp + fp + fn + EPS))
    sen = np.mean((tp + EPS) / (tp + fn + EPS))
    spe = np.mean((tn + EPS) / (tn + fp + EPS))
    acc = np.mean((tp + tn + EPS) / (tp + tn + fp + fn + EPS))
    return {
        "acc": float(acc),
        "dice": float(dice),
        "jac": float(jac),
        "sen": float(sen),
        "spe": float(spe),
    }


def surface_distance_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    h, w = pred.shape
    diag = float(np.hypot(h, w))
    if (not pred.any()) and (not gt.any()):
        return 0.0, 0.0
    if (not pred.any()) or (not gt.any()):
        return diag, diag

    st = np.ones((3, 3), dtype=bool)
    pred_surf = pred ^ ndi.binary_erosion(pred, structure=st, border_value=0)
    gt_surf = gt ^ ndi.binary_erosion(gt, structure=st, border_value=0)
    if not pred_surf.any():
        pred_surf = pred
    if not gt_surf.any():
        gt_surf = gt

    d_gt = ndi.distance_transform_edt(~gt_surf)
    d_pred = ndi.distance_transform_edt(~pred_surf)
    d_pred_to_gt = d_gt[pred_surf]
    d_gt_to_pred = d_pred[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0).astype(np.float64)
    hd95 = float(np.percentile(all_d, 95))
    assd = float((d_pred_to_gt.sum() + d_gt_to_pred.sum()) / (d_pred_to_gt.size + d_gt_to_pred.size))
    return hd95, assd


def binary_postprocess(mask: np.ndarray, fill_holes: bool, keep_largest: bool) -> np.ndarray:
    out = mask.astype(bool)
    if fill_holes:
        out = ndi.binary_fill_holes(out)
    if keep_largest:
        labeled, n = ndi.label(out)
        if n > 0:
            areas = np.bincount(labeled.ravel())
            areas[0] = 0
            if areas.max() > 0:
                out = labeled == int(np.argmax(areas))
    return out.astype(np.uint8)


def apply_method(
    probs: np.ndarray,
    threshold: float,
    fill_holes: bool,
    keep_largest: bool,
    method: str,
    params: dict[str, float | int],
) -> np.ndarray:
    preds = []
    for prob in probs:
        work = prob.astype(np.float32, copy=True)
        if method == "baseline":
            mask = work > threshold
        elif method == "gps":
            sigma = float(params["sigma"])
            if sigma > 0:
                work = gaussian_filter(work, sigma=sigma)
            mask = work > threshold
        elif method == "bilateral":
            sigma_color = float(params["sigma_color"])
            sigma_space = float(params["sigma_space"])
            d = int(params.get("diameter", 5))
            work = cv2.bilateralFilter(work, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            mask = work > threshold
        elif method == "dilation":
            mask = work > threshold
            iterations = int(params["iterations"])
            if iterations > 0:
                mask = ndi.binary_dilation(mask, iterations=iterations)
        else:
            raise ValueError(f"Unknown method: {method}")
        preds.append(binary_postprocess(mask, fill_holes=fill_holes, keep_largest=keep_largest))
    return np.stack(preds, axis=0).astype(np.uint8)


def evaluate_predictions(preds: np.ndarray, gts: np.ndarray, compute_distance: bool) -> dict[str, float]:
    result = compute_metrics(preds, gts)
    if compute_distance:
        hd95_vals = []
        assd_vals = []
        for pred, gt in zip(preds, gts):
            hd95, assd = surface_distance_metrics(pred, gt)
            hd95_vals.append(hd95)
            assd_vals.append(assd)
        result["hd95"] = float(np.mean(hd95_vals))
        result["assd"] = float(np.mean(assd_vals))
    return result


@torch.inference_mode()
def collect_probs(model: RPINet, loader: DataLoader, device: torch.device, channels_last: bool, tta: str) -> tuple[np.ndarray, np.ndarray]:
    views = tta_views(tta)
    all_probs = []
    all_gts = []
    model.eval()
    for batch in tqdm(loader, desc="collect"):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = batch["mask"].cpu().numpy()[:, 0]

        prob_sum = None
        for hf, vf in views:
            x = image
            if hf:
                x = torch.flip(x, dims=[3])
            if vf:
                x = torch.flip(x, dims=[2])
            logits = model(x, None)["logits"]
            probs = torch.sigmoid(logits)
            if vf:
                probs = torch.flip(probs, dims=[2])
            if hf:
                probs = torch.flip(probs, dims=[3])
            prob_sum = probs if prob_sum is None else prob_sum + probs
        avg_probs = (prob_sum / float(len(views))).cpu().numpy()[:, 0]
        all_probs.append(avg_probs.astype(np.float32))
        all_gts.append((mask > 0.5).astype(np.uint8))
    return np.concatenate(all_probs, axis=0), np.concatenate(all_gts, axis=0)


def method_grid(spec: DatasetSpec) -> dict[str, list[dict[str, float | int]]]:
    return {
        "baseline": [{}],
        "gps": [{"sigma": s} for s in ([0.0, 0.5, 1.0, 1.5, 2.0] if spec.name != "ISIC-2018" else [0.0, 0.5, 1.0, 1.5])],
        "dilation": [{"iterations": i} for i in [1, 2, 3]],
        "bilateral": [
            {"diameter": 5, "sigma_color": sc, "sigma_space": ss}
            for sc in [0.05, 0.10, 0.15]
            for ss in [3, 5, 7]
        ],
    }


def threshold_grid(spec: DatasetSpec) -> list[float]:
    if spec.name == "ISIC-2017":
        vals = np.concatenate([np.array([0.01, 0.02, 0.03, 0.05]), np.arange(0.08, 0.31, 0.02)])
    elif spec.name == "ISIC-2018":
        vals = np.arange(0.05, 0.31, 0.02)
    else:
        vals = np.arange(0.04, 0.25, 0.02)
    return [float(np.round(v, 2)) for v in vals]


def select_best(val_probs: np.ndarray, val_gts: np.ndarray, spec: DatasetSpec) -> dict[str, dict]:
    selections: dict[str, dict] = {}
    for method, candidates in method_grid(spec).items():
        best = None
        for threshold in threshold_grid(spec):
            for params in candidates:
                preds = apply_method(
                    val_probs,
                    threshold=threshold,
                    fill_holes=spec.fill_holes,
                    keep_largest=spec.keep_largest,
                    method=method,
                    params=params,
                )
                metrics = evaluate_predictions(preds, val_gts, compute_distance=False)
                entry = {"threshold": threshold, "params": params, "val_metrics": metrics}
                if best is None or metrics["jac"] > best["val_metrics"]["jac"]:
                    best = entry
        selections[method] = best
    return selections


def main() -> None:
    device = get_device("cuda")
    results = {"selection_metric": "val_jac", "datasets": {}}

    for spec in DATASETS:
        print(f"\n=== {spec.name} ===")
        cfg = load_config(spec.config)
        model = RPINet(**model_kwargs_from_cfg(cfg))
        state = torch.load(spec.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        model.to(device)
        channels_last = bool(cfg.get("runtime", {}).get("channels_last", False))
        if channels_last:
            model = model.to(memory_format=torch.channels_last)

        cache_stem = spec.name.lower().replace("-", "").replace(" ", "_")
        cache_path = OUT_DIR / f"{cache_stem}_probs_cache.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            val_probs = cached["val_probs"]
            val_gts = cached["val_gts"]
            test_probs = cached["test_probs"]
            test_gts = cached["test_gts"]
        else:
            val_loader = build_loader(cfg, "val")
            test_loader = build_loader(cfg, "test")
            val_probs, val_gts = collect_probs(model, val_loader, device, channels_last, spec.tta)
            test_probs, test_gts = collect_probs(model, test_loader, device, channels_last, spec.tta)
            np.savez_compressed(
                cache_path,
                val_probs=val_probs,
                val_gts=val_gts,
                test_probs=test_probs,
                test_gts=test_gts,
            )

        selections = select_best(val_probs, val_gts, spec)
        dataset_out = {
            "checkpoint": str(spec.checkpoint.relative_to(ROOT)),
            "config": str(spec.config.relative_to(ROOT)),
            "paper_operating_point": {
                "threshold": spec.threshold,
                "sigma": spec.sigma,
                "tta": spec.tta,
                "fill_holes": spec.fill_holes,
                "keep_largest": spec.keep_largest,
            },
            "methods": {},
        }
        for method, sel in selections.items():
            preds = apply_method(
                test_probs,
                threshold=float(sel["threshold"]),
                fill_holes=spec.fill_holes,
                keep_largest=spec.keep_largest,
                method=method,
                params=sel["params"],
            )
            test_metrics = evaluate_predictions(preds, test_gts, compute_distance=True)
            dataset_out["methods"][method] = {
                "selected_threshold": float(sel["threshold"]),
                "selected_params": sel["params"],
                "val_metrics": sel["val_metrics"],
                "test_metrics": test_metrics,
            }
        results["datasets"][spec.name] = dataset_out
        out_json = OUT_DIR / "boundary_postprocess_comparison.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_json = OUT_DIR / "boundary_postprocess_comparison.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Boundary Postprocessor Comparison",
        "",
        "Selection metric: validation JAC with the paper threshold/TTA/morphology fixed for each dataset.",
        "",
    ]
    for dataset, content in results["datasets"].items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | Selected Params | ACC | DICE | JAC | SEN | SPE | HD95 | ASSD |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method in ["baseline", "dilation", "bilateral", "gps"]:
            row = content["methods"][method]
            metrics = row["test_metrics"]
            params = row["selected_params"]
            pieces = [f"thr={row['selected_threshold']:.2f}"]
            if params:
                pieces.extend(f"{k}={v}" for k, v in params.items())
            param_text = ", ".join(pieces)
            lines.append(
                f"| {method} | {param_text} | "
                f"{metrics['acc']*100:.2f} | {metrics['dice']*100:.2f} | {metrics['jac']*100:.2f} | "
                f"{metrics['sen']*100:.2f} | {metrics['spe']*100:.2f} | {metrics['hd95']:.2f} | {metrics['assd']:.2f} |"
            )
        lines.append("")
    out_md = OUT_DIR / "boundary_postprocess_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
