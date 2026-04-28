#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep inference-time post-processing and TTA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.08)
    parser.add_argument("--threshold-max", type=float, default=0.30)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tta-modes", type=str, default="none,flip2,flip4")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def build_loader(
    cfg: dict,
    split_name: str,
    workers_override: int | None = None,
    batch_size_override: int | None = None,
) -> DataLoader:
    data_cfg = cfg["data"]
    use_tensor_cache = bool(data_cfg.get("use_tensor_cache", False))
    max_key = "max_val_samples" if split_name == "val" else "max_test_samples"

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
        split = data_cfg["val_split"] if split_name == "val" else data_cfg["test_split"]
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


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported tta mode: {mode}")


@torch.inference_mode()
def collect_probs_and_masks(
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
            prob = torch.sigmoid(logits)

            if vf:
                prob = torch.flip(prob, dims=[2])
            if hf:
                prob = torch.flip(prob, dims=[3])

            prob_sum = prob if prob_sum is None else (prob_sum + prob)

        prob_avg = prob_sum / float(len(views))
        probs_all.append(prob_avg[:, 0].cpu().numpy())
        masks_all.append(mask[:, 0].cpu().numpy())

    probs = np.concatenate(probs_all, axis=0)
    masks = np.concatenate(masks_all, axis=0)
    return probs.astype(np.float32), masks.astype(np.uint8)


def postprocess_mask(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
) -> np.ndarray:
    mask = prob > threshold
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

            if keep_largest:
                if labeled is None:
                    labeled, n = ndi.label(mask)
                    if n > 0:
                        areas = np.bincount(labeled.ravel())
                        areas[0] = 0
                if labeled is not None and n > 0 and areas.max() > 0:
                    largest_idx = int(np.argmax(areas))
                    mask = labeled == largest_idx
                else:
                    mask = np.zeros_like(mask, dtype=bool)

    return mask.astype(np.uint8)


def metrics_from_preds(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    # preds/targets: [N,H,W], {0,1}
    p = preds.astype(np.float32)
    t = targets.astype(np.float32)

    tp = (p * t).sum(axis=(1, 2))
    fp = (p * (1.0 - t)).sum(axis=(1, 2))
    fn = ((1.0 - p) * t).sum(axis=(1, 2))
    tn = ((1.0 - p) * (1.0 - t)).sum(axis=(1, 2))

    dice = np.mean((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)).item()
    jac = np.mean((tp + EPS) / (tp + fp + fn + EPS)).item()
    sen = np.mean((tp + EPS) / (tp + fn + EPS)).item()
    spe = np.mean((tn + EPS) / (tn + fp + EPS)).item()
    acc = np.mean((tp + tn + EPS) / (tp + tn + fp + fn + EPS)).item()
    return {"acc": acc, "dice": dice, "jac": jac, "sen": sen, "spe": spe}


def sweep_one_dataset(
    probs: np.ndarray,
    masks: np.ndarray,
    thresholds: list[float],
    min_areas: list[int],
    keep_largest_opts: list[bool],
    fill_holes_opts: list[bool],
) -> tuple[list[dict], dict]:
    rows = []
    n = probs.shape[0]
    best = None
    for thr in thresholds:
        for min_area in min_areas:
            for keep_largest in keep_largest_opts:
                for fill_holes in fill_holes_opts:
                    preds = np.zeros_like(masks, dtype=np.uint8)
                    for i in range(n):
                        preds[i] = postprocess_mask(
                            probs[i],
                            threshold=thr,
                            min_area=min_area,
                            keep_largest=keep_largest,
                            fill_holes=fill_holes,
                        )
                    m = metrics_from_preds(preds, masks)
                    rows.append(
                        {
                            "threshold": thr,
                            "min_area": int(min_area),
                            "keep_largest": bool(keep_largest),
                            "fill_holes": bool(fill_holes),
                            "metrics": m,
                        }
                    )
                    if best is None or m["jac"] > best["metrics"]["jac"]:
                        best = rows[-1]
    if best is None:
        raise RuntimeError("No sweep result produced")
    return rows, best


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
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    val_loader = build_loader(
        cfg,
        "val",
        workers_override=args.workers,
        batch_size_override=args.batch_size,
    )
    test_loader = build_loader(
        cfg,
        "test",
        workers_override=args.workers,
        batch_size_override=args.batch_size,
    )

    thresholds = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-12:
        thresholds.append(round(t, 6))
        t += args.threshold_step

    min_areas = [0, 64, 128, 256, 512]
    keep_largest_opts = [False, True]
    fill_holes_opts = [False, True]
    tta_modes = [m.strip() for m in args.tta_modes.split(",") if m.strip()]
    if not tta_modes:
        raise ValueError("No TTA modes provided")
    for mode in tta_modes:
        if mode not in {"none", "flip2", "flip4"}:
            raise ValueError(f"Unsupported tta mode in --tta-modes: {mode}")

    all_results = []
    best = None
    best_per_tta = []
    cached_test_probs: dict[str, np.ndarray] = {}
    cached_test_masks: np.ndarray | None = None

    for tta_mode in tta_modes:
        print(f"Collecting with tta={tta_mode}")
        val_probs, val_masks = collect_probs_and_masks(model, val_loader, device, channels_last, tta_mode)
        test_probs, test_masks = collect_probs_and_masks(model, test_loader, device, channels_last, tta_mode)
        cached_test_probs[tta_mode] = test_probs
        if cached_test_masks is None:
            cached_test_masks = test_masks

        print(f"Sweeping params for tta={tta_mode}")
        val_rows, val_best = sweep_one_dataset(
            val_probs,
            val_masks,
            thresholds,
            min_areas,
            keep_largest_opts,
            fill_holes_opts,
        )
        best_per_tta.append(
            {
                "tta": tta_mode,
                "threshold": val_best["threshold"],
                "min_area": val_best["min_area"],
                "keep_largest": val_best["keep_largest"],
                "fill_holes": val_best["fill_holes"],
                "val": val_best["metrics"],
            }
        )
        all_results.extend(
            [
                {
                    "tta": tta_mode,
                    "threshold": row["threshold"],
                    "min_area": row["min_area"],
                    "keep_largest": row["keep_largest"],
                    "fill_holes": row["fill_holes"],
                    "val": row["metrics"],
                }
                for row in val_rows
            ]
        )
        if best is None or val_best["metrics"]["jac"] > best["val"]["jac"]:
            best = best_per_tta[-1]

    if best is None:
        raise RuntimeError("No result generated")

    # Evaluate test only once on the global best-by-val combo.
    best_tta = best["tta"]
    if cached_test_masks is None or best_tta not in cached_test_probs:
        raise RuntimeError("Missing cached test probs for best TTA")
    test_probs = cached_test_probs[best_tta]
    test_masks = cached_test_masks
    test_preds = np.zeros_like(test_masks, dtype=np.uint8)
    for i in range(test_probs.shape[0]):
        test_preds[i] = postprocess_mask(
            test_probs[i],
            threshold=best["threshold"],
            min_area=best["min_area"],
            keep_largest=best["keep_largest"],
            fill_holes=best["fill_holes"],
        )
    best["test"] = metrics_from_preds(test_preds, test_masks)

    out = {
        "checkpoint": args.checkpoint,
        "n_results": len(all_results),
        "best_per_tta": best_per_tta,
        "best_by_val_jac": best,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": out, "rows": all_results}, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
