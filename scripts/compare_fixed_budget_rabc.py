#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch
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
        "use_brr": bool(model_cfg.get("use_brr", False)),
        "use_focal_context": bool(model_cfg.get("use_focal_context", False)),
        "use_cga_fusion": bool(model_cfg.get("use_cga_fusion", False)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare no-RABC, global threshold shift, binary dilation, and RABC under a fixed inference protocol"
    )
    p.add_argument("--name", required=True)
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--rabc-config", required=True)
    p.add_argument("--rabc-checkpoint", required=True)
    p.add_argument("--base-threshold", type=float, required=True)
    p.add_argument("--tta", default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--min-area", type=int, default=0)
    p.add_argument("--keep-largest", action="store_true")
    p.add_argument("--fill-holes", action="store_true")
    p.add_argument("--match-target", default="sen", choices=["sen", "fg_ratio"])
    p.add_argument("--threshold-grid", default="")
    p.add_argument("--dilation-grid", default="1,2,3")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported tta mode: {mode}")


def build_loader(cfg: dict, split_name: str, workers_override: int | None, batch_size_override: int | None) -> DataLoader:
    data_cfg = cfg["data"]
    split = data_cfg["val_split"] if split_name == "val" else data_cfg["test_split"]
    max_key = "max_val_samples" if split_name == "val" else "max_test_samples"
    use_tensor_cache = bool(data_cfg.get("use_tensor_cache", False))
    if use_tensor_cache:
        cache_dir = Path(data_cfg["tensor_cache_dir"])
        cache_file = cache_dir / ("val_cache.pt" if split_name == "val" else "test_cache.pt")
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
            use_hair_removal=bool(data_cfg.get("use_hair_removal", False)),
            cache_images=bool(data_cfg.get("cache_images", False)),
            cache_workers=int(data_cfg.get("cache_workers", 4)),
            max_samples=data_cfg.get(max_key),
            ph2_split_seed=int(data_cfg.get("ph2_split_seed", 42)),
            ph2_split_ratios=tuple(data_cfg.get("ph2_split_ratios", (0.7, 0.15, 0.15))),
        )

    eval_cfg = cfg.get("eval", {})
    n_workers = int(
        workers_override if workers_override is not None else eval_cfg.get("num_workers", data_cfg.get("eval_num_workers", data_cfg.get("num_workers", 4)))
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
        batch_size=int(batch_size_override if batch_size_override is not None else eval_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )


def load_model(cfg: dict, checkpoint: str, device: torch.device, channels_last: bool) -> RPINet:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = RPINet(**model_kwargs_from_cfg(cfg))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


@torch.inference_mode()
def collect_probs(
    model: RPINet,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    tta: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    views = tta_views(tta)
    ids: list[str] = []
    probs_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []
    for batch in tqdm(loader, desc="collect", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"][:, 0].cpu().numpy() > 0.5).astype(np.uint8)
        ids.extend(list(batch["id"]))

        prob_sum = None
        for hf, vf in views:
            x = image
            if hf:
                x = torch.flip(x, dims=[3])
            if vf:
                x = torch.flip(x, dims=[2])
            logits = model(x, None)["logits"]
            prob = torch.sigmoid(logits)
            if vf:
                prob = torch.flip(prob, dims=[2])
            if hf:
                prob = torch.flip(prob, dims=[3])
            prob_sum = prob if prob_sum is None else prob_sum + prob
        probs = (prob_sum / float(len(views))).cpu().numpy()[:, 0].astype(np.float32)
        probs_all.append(probs)
        masks_all.append(mask)
    return ids, np.concatenate(probs_all, axis=0), np.concatenate(masks_all, axis=0)


def postprocess_mask(prob: np.ndarray, threshold: float, min_area: int, keep_largest: bool, fill_holes: bool, dilation_iters: int = 0) -> np.ndarray:
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


def preds_from_probs(
    probs: np.ndarray,
    threshold: float,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
    dilation_iters: int = 0,
) -> np.ndarray:
    preds = np.zeros_like(probs, dtype=np.uint8)
    for i in range(probs.shape[0]):
        preds[i] = postprocess_mask(
            probs[i],
            threshold=threshold,
            min_area=min_area,
            keep_largest=keep_largest,
            fill_holes=fill_holes,
            dilation_iters=dilation_iters,
        )
    return preds


def metrics_from_preds(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    p = preds.astype(np.float64)
    t = targets.astype(np.float64)
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
        "fg_ratio": float(np.mean(p.mean(axis=(1, 2)))),
    }


def metric_gap(stats: dict[str, float], target_name: str, target_value: float) -> float:
    return abs(float(stats[target_name]) - float(target_value))


def match_threshold_shift(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    target_name: str,
    target_value: float,
    thresholds: list[float],
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
) -> tuple[float, dict]:
    best_thr = thresholds[0]
    best_stats: dict | None = None
    best_gap = None
    for thr in thresholds:
        preds = preds_from_probs(val_probs, thr, min_area, keep_largest, fill_holes, dilation_iters=0)
        stats = metrics_from_preds(preds, val_targets)
        gap = metric_gap(stats, target_name, target_value)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_thr = thr
            best_stats = stats
    assert best_stats is not None
    return best_thr, best_stats


def match_dilation(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    base_threshold: float,
    target_name: str,
    target_value: float,
    dilation_iters: list[int],
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
) -> tuple[int, dict]:
    best_iter = dilation_iters[0]
    best_stats: dict | None = None
    best_gap = None
    for d in dilation_iters:
        preds = preds_from_probs(val_probs, base_threshold, min_area, keep_largest, fill_holes, dilation_iters=d)
        stats = metrics_from_preds(preds, val_targets)
        gap = metric_gap(stats, target_name, target_value)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_iter = d
            best_stats = stats
    assert best_stats is not None
    return best_iter, best_stats


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.base_config)
    rabc_cfg = load_config(args.rabc_config)

    runtime_cfg = base_cfg.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        cpu_threads = max(1, int(cpu_threads * 0.75))
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    device = get_device(base_cfg.get("runtime", {}).get("device", "cuda"))
    channels_last = bool(base_cfg.get("runtime", {}).get("channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))

    base_model = load_model(base_cfg, args.base_checkpoint, device, channels_last)
    rabc_model = load_model(rabc_cfg, args.rabc_checkpoint, device, channels_last)

    val_loader = build_loader(base_cfg, "val", args.workers, args.batch_size)
    test_loader = build_loader(base_cfg, args.split, args.workers, args.batch_size)

    _, base_val_probs, val_targets = collect_probs(base_model, val_loader, device, channels_last, args.tta)
    _, rabc_val_probs, _ = collect_probs(rabc_model, val_loader, device, channels_last, args.tta)

    thresholds = parse_float_list(args.threshold_grid) if args.threshold_grid.strip() else [round(max(0.01, args.base_threshold - 0.12) + 0.01 * i, 2) for i in range(int((min(0.95, args.base_threshold + 0.02) - max(0.01, args.base_threshold - 0.12)) / 0.01) + 1)]
    dilations = parse_int_list(args.dilation_grid)

    rabc_val_preds = preds_from_probs(
        rabc_val_probs,
        args.base_threshold,
        args.min_area,
        args.keep_largest,
        args.fill_holes,
        dilation_iters=0,
    )
    rabc_val_stats = metrics_from_preds(rabc_val_preds, val_targets)
    target_value = float(rabc_val_stats[args.match_target])

    shift_thr, shift_val_stats = match_threshold_shift(
        base_val_probs,
        val_targets,
        args.match_target,
        target_value,
        thresholds,
        args.min_area,
        args.keep_largest,
        args.fill_holes,
    )
    dil_iter, dil_val_stats = match_dilation(
        base_val_probs,
        val_targets,
        args.base_threshold,
        args.match_target,
        target_value,
        dilations,
        args.min_area,
        args.keep_largest,
        args.fill_holes,
    )

    ids, base_test_probs, test_targets = collect_probs(base_model, test_loader, device, channels_last, args.tta)
    _, rabc_test_probs, _ = collect_probs(rabc_model, test_loader, device, channels_last, args.tta)

    variants = {
        "base": preds_from_probs(base_test_probs, args.base_threshold, args.min_area, args.keep_largest, args.fill_holes, 0),
        "threshold_shift": preds_from_probs(base_test_probs, shift_thr, args.min_area, args.keep_largest, args.fill_holes, 0),
        "dilation": preds_from_probs(base_test_probs, args.base_threshold, args.min_area, args.keep_largest, args.fill_holes, dil_iter),
        "rabc": preds_from_probs(rabc_test_probs, args.base_threshold, args.min_area, args.keep_largest, args.fill_holes, 0),
    }

    summary = {}
    per_case = {}
    for name, preds in variants.items():
        summary[name] = metrics_from_preds(preds, test_targets)
        case_metrics = metrics_from_preds(preds, test_targets)
        per_case[name] = case_metrics

    out = {
        "dataset": args.name,
        "base_protocol": {
            "threshold": args.base_threshold,
            "tta": args.tta,
            "min_area": args.min_area,
            "keep_largest": bool(args.keep_largest),
            "fill_holes": bool(args.fill_holes),
        },
        "matched_budget": {
            "match_target": args.match_target,
            "target_value_from_rabc_val": target_value,
            "threshold_shift_threshold": shift_thr,
            "threshold_shift_val_stats": shift_val_stats,
            "dilation_iters": dil_iter,
            "dilation_val_stats": dil_val_stats,
            "rabc_val_stats": rabc_val_stats,
        },
        "summary": summary,
        "n_cases": len(ids),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
