#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset, ISIC2017TensorCacheDataset
from rpi_net.model import RPINet
from rpi_net.utils import get_device, load_config

EPS = 1e-7


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
        "use_brr": bool(model_cfg.get("use_brr", False)),
        "use_focal_context": bool(model_cfg.get("use_focal_context", False)),
        "use_cga_fusion": bool(model_cfg.get("use_cga_fusion", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select inference operating point by validation metrics only")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--tta-modes", type=str, default="none,flip2,flip4")
    parser.add_argument("--min-areas", type=str, default="0,64,128,256,512")
    parser.add_argument("--keep-largest-options", type=str, default="false,true")
    parser.add_argument("--fill-holes-options", type=str, default="false,true")
    parser.add_argument("--dilation-iters-list", type=str, default="0",
        help="Binary dilation iterations to search after thresholding (comma-separated)")
    parser.add_argument("--adaptive-threshold", action="store_true",
        help="Enable uncertainty-adaptive per-pixel threshold using prediction entropy")
    parser.add_argument("--adaptive-scales", type=str, default="0.0,0.05,0.1,0.15,0.2,0.25,0.3",
        help="Adaptive threshold scales to search (comma-separated)")
    parser.add_argument("--gaussian-sigmas", type=str, default="0.0",
        help="Gaussian blur sigmas to search on probability map (comma-separated)")
    parser.add_argument(
        "--objective",
        type=str,
        default="jac",
        choices=["jac", "dice", "acc", "sen", "spe", "harmonic_jac_sen", "mean5"],
    )
    parser.add_argument("--min-sen", type=float, default=None, help="Optional minimum SEN constraint on val (0-1)")
    parser.add_argument("--min-spe", type=float, default=None, help="Optional minimum SPE constraint on val (0-1)")
    parser.add_argument("--min-jac", type=float, default=None, help="Optional minimum JAC constraint on val (0-1)")
    parser.add_argument("--min-dice", type=float, default=None, help="Optional minimum DICE constraint on val (0-1)")
    parser.add_argument("--min-acc", type=float, default=None, help="Optional minimum ACC constraint on val (0-1)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save selection results JSON (default: <checkpoint_dir>/select_val.json)",
    )
    return parser.parse_args()


def parse_bool_list(s: str) -> list[bool]:
    out = []
    for x in s.split(","):
        t = x.strip().lower()
        if t in {"1", "true", "yes", "y"}:
            out.append(True)
        elif t in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid bool token: {x}")
    if not out:
        raise ValueError("Empty bool option list")
    return out


def parse_int_list(s: str) -> list[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Empty int option list")
    return out


def parse_float_list(s: str) -> list[float]:
    out = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("Empty float option list")
    return out


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported tta mode: {mode}")


def compute_entropy_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Compute per-pixel entropy-based uncertainty from probability maps.
    probs: [N, H, W] in [0, 1]
    returns: [N, H, W] in [0, 1], normalized entropy
    """
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return entropy / 0.6931471805599453  # normalize by ln(2)


def postprocess_mask(
    prob: np.ndarray,
    threshold: float | np.ndarray,
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
                    mask = labeled == int(np.argmax(areas))
                else:
                    mask = np.zeros_like(mask, dtype=bool)

    return mask.astype(np.uint8)


def metrics_from_preds(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    p = preds.astype(np.float32)
    t = targets.astype(np.float32)
    tp = (p * t).sum(axis=(1, 2))
    fp = (p * (1.0 - t)).sum(axis=(1, 2))
    fn = ((1.0 - p) * t).sum(axis=(1, 2))
    tn = ((1.0 - p) * (1.0 - t)).sum(axis=(1, 2))
    return {
        "acc": float(np.mean((tp + tn + EPS) / (tp + tn + fp + fn + EPS))),
        "dice": float(np.mean((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS))),
        "jac": float(np.mean((tp + EPS) / (tp + fp + fn + EPS))),
        "sen": float(np.mean((tp + EPS) / (tp + fn + EPS))),
        "spe": float(np.mean((tn + EPS) / (tn + fp + EPS))),
    }


def objective_value(m: dict[str, float], objective: str) -> float:
    if objective == "harmonic_jac_sen":
        j = m["jac"]
        s = m["sen"]
        return 2.0 * j * s / (j + s + 1e-12)
    if objective == "mean5":
        return float((m["acc"] + m["dice"] + m["jac"] + m["sen"] + m["spe"]) / 5.0)
    return float(m[objective])


def build_val_loader(cfg: dict, workers_override: int | None, batch_size_override: int | None) -> DataLoader:
    data_cfg = cfg["data"]
    split = data_cfg["val_split"]
    use_tensor_cache = bool(data_cfg.get("use_tensor_cache", False))
    max_key = "max_val_samples"

    if use_tensor_cache:
        cache_dir = Path(data_cfg["tensor_cache_dir"])
        cache_file = cache_dir / "val_cache.pt"
        ds = ISIC2017TensorCacheDataset(
            cache_file=cache_file,
            require_pseudo=False,
            augment=False,
            max_samples=data_cfg.get(max_key),
        )
    else:
        ds = ISIC2017Dataset(
            root=data_cfg["root"],
            split=split,
            image_size=data_cfg["image_size"],
            augment=False,
            use_hair_removal=bool(data_cfg.get("use_hair_removal", True)),
            cache_images=bool(data_cfg.get("cache_images", False)),
            cache_workers=int(data_cfg.get("cache_workers", 4)),
            max_samples=data_cfg.get(max_key),
        )

    eval_cfg = cfg.get("eval", {})
    n_workers = int(
        workers_override
        if workers_override is not None
        else eval_cfg.get("num_workers", data_cfg.get("eval_num_workers", data_cfg.get("num_workers", 4)))
    )
    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": bool(eval_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if n_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            eval_cfg.get("persistent_workers", data_cfg.get("eval_persistent_workers", data_cfg.get("persistent_workers", True)))
        )
        loader_kwargs["prefetch_factor"] = int(
            eval_cfg.get("prefetch_factor", data_cfg.get("eval_prefetch_factor", data_cfg.get("prefetch_factor", 4)))
        )

    return DataLoader(
        ds,
        batch_size=int(
            batch_size_override
            if batch_size_override is not None
            else eval_cfg.get("batch_size", cfg["train"]["batch_size"])
        ),
        shuffle=False,
        **loader_kwargs,
    )


@torch.inference_mode()
def collect_val_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    tta_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    views = tta_views(tta_mode)
    probs_all = []
    masks_all = []
    for batch in tqdm(loader, desc=f"collect-{tta_mode}", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()

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
        probs_all.append(probs[:, 0].cpu().numpy())
        masks_all.append(mask[:, 0].cpu().numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(masks_all, axis=0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    runtime_cfg = cfg.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    device = get_device(cfg.get("runtime", {}).get("device", "cuda"))
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if bool(cfg.get("runtime", {}).get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = RPINet(**model_kwargs_from_cfg(cfg))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    val_loader = build_val_loader(cfg, args.workers, args.batch_size)

    thresholds = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-12:
        thresholds.append(round(t, 6))
        t += args.threshold_step

    tta_modes = [m.strip() for m in args.tta_modes.split(",") if m.strip()]
    min_areas = parse_int_list(args.min_areas)
    keep_opts = parse_bool_list(args.keep_largest_options)
    fill_opts = parse_bool_list(args.fill_holes_options)
    dilation_iters_list = parse_int_list(args.dilation_iters_list)
    min_sen = args.min_sen
    min_spe = args.min_spe
    min_jac = args.min_jac
    min_dice = args.min_dice
    min_acc = args.min_acc
    use_adaptive = bool(args.adaptive_threshold)
    adaptive_scales = parse_float_list(args.adaptive_scales) if use_adaptive else [0.0]
    gaussian_sigmas = parse_float_list(args.gaussian_sigmas)

    rows = []
    best = None

    for tta in tta_modes:
        val_probs, val_masks = collect_val_probs(model, val_loader, device, channels_last, tta)
        # Precompute per-pixel entropy uncertainty for adaptive threshold
        val_uncertainty = compute_entropy_uncertainty(val_probs) if use_adaptive else None
        for g_sigma in gaussian_sigmas:
            # Apply gaussian blur to probability maps if sigma > 0
            if g_sigma > 0:
                val_probs_smooth = np.stack([gaussian_filter(val_probs[i], sigma=g_sigma) for i in range(val_probs.shape[0])])
            else:
                val_probs_smooth = val_probs
            for thr in thresholds:
                for adapt_scale in adaptive_scales:
                    for min_area in min_areas:
                        for keep_largest in keep_opts:
                            for fill_holes in fill_opts:
                                for dilation_iters in dilation_iters_list:
                                    preds = np.zeros_like(val_masks, dtype=np.uint8)
                                    for i in range(val_probs_smooth.shape[0]):
                                        if adapt_scale > 0.0 and val_uncertainty is not None:
                                            # Per-pixel adaptive threshold: lower threshold in uncertain regions
                                            pixel_thr = np.clip(thr - adapt_scale * val_uncertainty[i], 0.01, 0.99)
                                        else:
                                            pixel_thr = thr
                                        preds[i] = postprocess_mask(
                                            val_probs_smooth[i],
                                            threshold=pixel_thr,
                                            min_area=min_area,
                                            keep_largest=keep_largest,
                                            fill_holes=fill_holes,
                                            dilation_iters=dilation_iters,
                                        )
                                    m = metrics_from_preds(preds, val_masks)
                                    if (min_sen is not None) and (m["sen"] < float(min_sen)):
                                        continue
                                    if (min_spe is not None) and (m["spe"] < float(min_spe)):
                                        continue
                                    if (min_jac is not None) and (m["jac"] < float(min_jac)):
                                        continue
                                    if (min_dice is not None) and (m["dice"] < float(min_dice)):
                                        continue
                                    if (min_acc is not None) and (m["acc"] < float(min_acc)):
                                        continue
                                    row = {
                                        "tta": tta,
                                        "threshold": thr,
                                        "adaptive_scale": float(adapt_scale),
                                        "gaussian_sigma": float(g_sigma),
                                        "min_area": int(min_area),
                                        "keep_largest": bool(keep_largest),
                                        "fill_holes": bool(fill_holes),
                                        "dilation_iters": int(dilation_iters),
                                        "val": m,
                                        "objective_value": objective_value(m, args.objective),
                                    }
                                    rows.append(row)
                                    if (best is None) or (row["objective_value"] > best["objective_value"]):
                                        best = row

    if best is None:
        raise RuntimeError("No valid operating point found (check constraints)")

    out = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "objective": args.objective,
        "min_sen_constraint": min_sen,
        "min_spe_constraint": min_spe,
        "min_jac_constraint": min_jac,
        "min_dice_constraint": min_dice,
        "min_acc_constraint": min_acc,
        "n_rows": len(rows),
        "best": best,
    }
    out_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "select_val.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": out, "rows": rows}, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
