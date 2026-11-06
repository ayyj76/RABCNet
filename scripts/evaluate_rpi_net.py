#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset, ISIC2017TensorCacheDataset
from rpi_net.model import RPINet
from rpi_net.utils import AverageMeter, get_device, load_config


def model_kwargs_from_cfg(cfg: dict) -> dict:
    model_cfg = cfg["model"]
    return {
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "backbone": str(model_cfg.get("backbone", "resnet50")),
        "backbone_pretrained": bool(model_cfg.get("backbone_pretrained", True)),
        "use_boundary_head": bool(model_cfg.get("use_boundary_head", False)),
        "use_uncertainty_head": bool(model_cfg.get("use_uncertainty_head", False)),
        "refine_with_heads": bool(model_cfg.get("refine_with_heads", False)),
        "use_aspp": bool(model_cfg.get("use_aspp", False)),
        "use_aux_head": bool(model_cfg.get("use_aux_head", False)),
        "use_se_context": bool(model_cfg.get("use_se_context", False)),
        "use_detail_fusion": bool(model_cfg.get("use_detail_fusion", False)),
        "use_coord_attn": bool(model_cfg.get("use_coord_attn", False)),
        "use_edge_prompt": bool(model_cfg.get("use_edge_prompt", False)),
        "use_dual_supervised_head": bool(model_cfg.get("use_dual_supervised_head", False)),
        "use_eca_context": bool(model_cfg.get("use_eca_context", False)),
        "use_ema_context": bool(model_cfg.get("use_ema_context", False)),
        "use_cbam_decoder": bool(model_cfg.get("use_cbam_decoder", False)),
        "use_simam_refine": bool(model_cfg.get("use_simam_refine", False)),
        "use_safm_decoder": bool(model_cfg.get("use_safm_decoder", False)),
        "use_uncertainty_gated_refine": bool(model_cfg.get("use_uncertainty_gated_refine", False)),
        "use_sk_context": bool(model_cfg.get("use_sk_context", False)),
        "use_triplet_decoder": bool(model_cfg.get("use_triplet_decoder", False)),
        "use_msaa_fusion": bool(model_cfg.get("use_msaa_fusion", False)),
        "use_ega_refine": bool(model_cfg.get("use_ega_refine", False)),
        "use_cot_context": bool(model_cfg.get("use_cot_context", False)),
        "use_lsk_decoder": bool(model_cfg.get("use_lsk_decoder", False)),
        "use_ucgf_refine": bool(model_cfg.get("use_ucgf_refine", False)),
        "use_anchor_context": bool(model_cfg.get("use_anchor_context", False)),
        "use_smfa_decoder": bool(model_cfg.get("use_smfa_decoder", False)),
        "use_tff_refine": bool(model_cfg.get("use_tff_refine", False)),
        "use_mfe_context": bool(model_cfg.get("use_mfe_context", False)),
        "use_bg_decouple_head": bool(model_cfg.get("use_bg_decouple_head", False)),
        "use_uncertainty_logit_gate": bool(model_cfg.get("use_uncertainty_logit_gate", False)),
        "use_global_logit_calibrator": bool(model_cfg.get("use_global_logit_calibrator", False)),
        "use_boundary_uncertainty_logit_refine": bool(model_cfg.get("use_boundary_uncertainty_logit_refine", False)),
        "use_residual_logit_refine": bool(model_cfg.get("use_residual_logit_refine", False)),
        "use_mamba_logit_refine": bool(model_cfg.get("use_mamba_logit_refine", False)),
        "use_cgfn_logit_refine": bool(model_cfg.get("use_cgfn_logit_refine", False)),
        "use_recall_recover_refine": bool(model_cfg.get("use_recall_recover_refine", False)),
        "use_recall_candidate_head": bool(model_cfg.get("use_recall_candidate_head", False)),
        "use_multiscale_recall_head": bool(model_cfg.get("use_multiscale_recall_head", False)),
        "use_rabc": bool(model_cfg.get("use_rabc", False)),
        "rabc_mixer_mode": str(model_cfg.get("rabc_mixer_mode", "mean")),
        "rabc_candidate_intercept": float(model_cfg.get("rabc_candidate_intercept", 0.35)),
        "rabc_candidate_uncertainty": float(model_cfg.get("rabc_candidate_uncertainty", 0.65)),
        "use_brr": bool(model_cfg.get("use_brr", False)),
        "use_focal_context": bool(model_cfg.get("use_focal_context", False)),
        "use_cga_fusion": bool(model_cfg.get("use_cga_fusion", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained RPI-Net checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tta", type=str, default="none", choices=["none", "flip2", "flip4"])
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--dilation-iters", type=int, default=0,
        help="Binary dilation iterations applied after thresholding (0=disabled)")
    parser.add_argument("--gaussian-sigma", type=float, default=0.0,
        help="Gaussian blur sigma on probability map before thresholding (0=disabled)")
    parser.add_argument("--compute-distance", action="store_true", help="Also compute HD95/ASSD (pixel units)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported tta mode: {mode}")


def postprocess_mask(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
    dilation_iters: int = 0,
) -> np.ndarray:
    mask = prob > threshold
    if dilation_iters > 0:
        mask = ndi.binary_dilation(mask, iterations=int(dilation_iters))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    if min_area > 0 or keep_largest:
        labeled, n = ndi.label(mask)
        if n > 0:
            areas = np.bincount(labeled.ravel())
            areas[0] = 0

            if min_area > 0:
                keep = np.where(areas >= int(min_area))[0]
                if keep.size > 0:
                    mask = np.isin(labeled, keep)
                else:
                    mask = np.zeros_like(mask, dtype=bool)
                    labeled = None
                    n = 0

            if keep_largest:
                if labeled is None:
                    labeled, n = ndi.label(mask)
                    if n > 0:
                        areas = np.bincount(labeled.ravel())
                        areas[0] = 0
                if n > 0 and areas.max() > 0:
                    largest_idx = int(np.argmax(areas))
                    mask = labeled == largest_idx
                else:
                    mask = np.zeros_like(mask, dtype=bool)
    return mask.astype(np.float32)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    eps = 1e-7
    targets = (targets > 0.5).float()
    preds = (preds > 0.5).float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=(1, 2, 3))

    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
    jac = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    sen = ((tp + eps) / (tp + fn + eps)).mean().item()
    spe = ((tn + eps) / (tn + fp + eps)).mean().item()
    acc = ((tp + tn + eps) / (tp + tn + fp + fn + eps)).mean().item()
    return {"acc": acc, "dice": dice, "jac": jac, "sen": sen, "spe": spe}


def surface_distance_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """Return (hd95, assd) in pixel units for one 2D binary pair."""
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


@torch.inference_mode()
def evaluate(
    model: RPINet,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
    channels_last: bool,
    tta: str,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
    compute_distance: bool,
    gaussian_sigma: float = 0.0,
    dilation_iters: int = 0,
) -> dict:
    model.eval()
    meters = {k: AverageMeter() for k in ["acc", "dice", "jac", "sen", "spe"]}
    if compute_distance:
        meters["hd95"] = AverageMeter()
        meters["assd"] = AverageMeter()
    views = tta_views(tta)

    for batch in tqdm(loader, desc="eval"):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = batch["mask"].to(device, non_blocking=True)

        prob_sum = None
        for hf, vf in views:
            x = image
            if hf:
                x = torch.flip(x, dims=[3])
            if vf:
                x = torch.flip(x, dims=[2])

            logits = model(x, None)["logits"]
            p = torch.sigmoid(logits)
            if vf:
                p = torch.flip(p, dims=[2])
            if hf:
                p = torch.flip(p, dims=[3])
            prob_sum = p if prob_sum is None else (prob_sum + p)
        probs = prob_sum / float(len(views))

        if min_area > 0 or keep_largest or fill_holes or gaussian_sigma > 0 or dilation_iters > 0:
            preds_np = np.zeros((probs.shape[0], probs.shape[2], probs.shape[3]), dtype=np.float32)
            probs_np = probs[:, 0].detach().cpu().numpy()
            for i in range(probs_np.shape[0]):
                prob_i = probs_np[i]
                if gaussian_sigma > 0:
                    prob_i = gaussian_filter(prob_i, sigma=gaussian_sigma)
                preds_np[i] = postprocess_mask(
                    prob_i,
                    threshold=threshold,
                    min_area=min_area,
                    keep_largest=keep_largest,
                    fill_holes=fill_holes,
                    dilation_iters=dilation_iters,
                )
            preds = torch.from_numpy(preds_np).unsqueeze(1).to(mask.device)
        else:
            preds = (probs > threshold).float()

        metrics = compute_metrics(preds, mask)
        bs = image.size(0)
        if compute_distance:
            preds_np = (preds[:, 0].detach().cpu().numpy() > 0.5)
            mask_np = (mask[:, 0].detach().cpu().numpy() > 0.5)
            hd95_vals = []
            assd_vals = []
            for i in range(preds_np.shape[0]):
                hd95_i, assd_i = surface_distance_metrics(preds_np[i], mask_np[i])
                hd95_vals.append(hd95_i)
                assd_vals.append(assd_i)
            metrics["hd95"] = float(np.mean(hd95_vals))
            metrics["assd"] = float(np.mean(assd_vals))
        for k, v in metrics.items():
            meters[k].update(v, bs)

    return {k: v.avg for k, v in meters.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime_cfg = cfg.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
    device = get_device(cfg["runtime"].get("device", "cuda"))
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if bool(cfg.get("runtime", {}).get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = RPINet(**model_kwargs_from_cfg(cfg))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    split = cfg["data"]["val_split"] if args.split == "val" else cfg["data"]["test_split"]
    max_key = "max_val_samples" if args.split == "val" else "max_test_samples"
    use_tensor_cache = bool(cfg["data"].get("use_tensor_cache", False))

    if use_tensor_cache:
        cache_dir = Path(cfg["data"]["tensor_cache_dir"])
        cache_file = cache_dir / ("val_cache.pt" if args.split == "val" else "test_cache.pt")
        ds = ISIC2017TensorCacheDataset(
            cache_file=cache_file,
            require_pseudo=False,
            augment=False,
            max_samples=cfg["data"].get(max_key),
        )
    else:
        ds = ISIC2017Dataset(
            root=cfg["data"]["root"],
            split=split,
            image_size=cfg["data"]["image_size"],
            augment=False,
            use_hair_removal=bool(cfg["data"].get("use_hair_removal", True)),
            cache_images=bool(cfg["data"].get("cache_images", False)),
            cache_workers=int(cfg["data"].get("cache_workers", 4)),
            max_samples=cfg["data"].get(max_key),
        )
    eval_cfg = cfg.get("eval", {})
    n_workers = int(
        args.workers
        if args.workers is not None
        else eval_cfg.get("num_workers", cfg["data"].get("eval_num_workers", cfg["data"].get("num_workers", 4)))
    )
    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": bool(eval_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if n_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            eval_cfg.get("persistent_workers", cfg["data"].get("eval_persistent_workers", cfg["data"].get("persistent_workers", True)))
        )
        loader_kwargs["prefetch_factor"] = int(
            eval_cfg.get("prefetch_factor", cfg["data"].get("eval_prefetch_factor", cfg["data"].get("prefetch_factor", 4)))
        )

    loader = DataLoader(
        ds,
        batch_size=int(
            args.batch_size
            if args.batch_size is not None
            else eval_cfg.get("batch_size", cfg["train"]["batch_size"])
        ),
        shuffle=False,
        **loader_kwargs,
    )

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(cfg["train"].get("threshold", 0.3))
    )

    result = evaluate(
        model,
        loader,
        threshold=threshold,
        device=device,
        channels_last=channels_last,
        tta=args.tta,
        min_area=int(args.min_area),
        keep_largest=bool(args.keep_largest),
        fill_holes=bool(args.fill_holes),
        compute_distance=bool(args.compute_distance),
        gaussian_sigma=float(args.gaussian_sigma),
        dilation_iters=int(args.dilation_iters),
    )
    result["threshold"] = threshold
    result["tta"] = args.tta
    result["min_area"] = int(args.min_area)
    result["keep_largest"] = bool(args.keep_largest)
    result["fill_holes"] = bool(args.fill_holes)
    result["dilation_iters"] = int(args.dilation_iters)
    result["gaussian_sigma"] = float(args.gaussian_sigma)
    print(result)


if __name__ == "__main__":
    main()
