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
    p = argparse.ArgumentParser(description="Evaluate probability calibration for RPI-Net variants")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--tta", default="flip4", choices=["none", "flip2", "flip4"])
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--boundary-width", type=int, default=3)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported tta mode: {mode}")


def build_loader(cfg: dict, split: str, workers: int | None, batch_size: int | None) -> DataLoader:
    max_key = "max_val_samples" if split == "val" else "max_test_samples"
    use_tensor_cache = bool(cfg["data"].get("use_tensor_cache", False))
    eval_cfg = cfg.get("eval", {})
    n_workers = int(
        workers
        if workers is not None
        else eval_cfg.get("num_workers", cfg["data"].get("eval_num_workers", cfg["data"].get("num_workers", 4)))
    )
    if use_tensor_cache:
        cache_dir = Path(cfg["data"]["tensor_cache_dir"])
        cache_file = cache_dir / ("val_cache.pt" if split == "val" else "test_cache.pt")
        ds = ISIC2017TensorCacheDataset(
            cache_file=cache_file,
            require_pseudo=False,
            augment=False,
            max_samples=cfg["data"].get(max_key),
        )
    else:
        ds = ISIC2017Dataset(
            root=cfg["data"]["root"],
            split=cfg["data"]["val_split"] if split == "val" else cfg["data"]["test_split"],
            image_size=cfg["data"]["image_size"],
            augment=False,
            use_hair_removal=bool(cfg["data"].get("use_hair_removal", True)),
            cache_images=bool(cfg["data"].get("cache_images", False)),
            cache_workers=int(cfg["data"].get("cache_workers", 4)),
            max_samples=cfg["data"].get(max_key),
            ph2_split_seed=int(cfg["data"].get("ph2_split_seed", 42)),
            ph2_split_ratios=tuple(cfg["data"].get("ph2_split_ratios", (0.7, 0.15, 0.15))),
        )
    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": bool(eval_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if n_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            eval_cfg.get(
                "persistent_workers",
                cfg["data"].get("eval_persistent_workers", cfg["data"].get("persistent_workers", True)),
            )
        )
        loader_kwargs["prefetch_factor"] = int(
            eval_cfg.get(
                "prefetch_factor",
                cfg["data"].get("eval_prefetch_factor", cfg["data"].get("prefetch_factor", 4)),
            )
        )
    return DataLoader(
        ds,
        batch_size=int(batch_size if batch_size is not None else eval_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    surface = np.logical_xor(mask_bool, ndi.binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool), border_value=0))
    if not surface.any():
        surface = mask_bool.copy()
    dist = ndi.distance_transform_edt(~surface)
    return dist <= float(width)


def ece_from_probs(prob: np.ndarray, target: np.ndarray, bins: int) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    ece = 0.0
    rows: list[dict] = []
    n = float(prob.size)
    if n == 0:
        return 0.0, rows
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            sel = (prob >= lo) & (prob <= hi)
        else:
            sel = (prob >= lo) & (prob < hi)
        count = int(sel.sum())
        if count == 0:
            rows.append({
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "count": 0,
                "mean_prob": None,
                "freq_pos": None,
            })
            continue
        mean_prob = float(prob[sel].mean())
        freq_pos = float(target[sel].mean())
        weight = count / n
        ece += abs(mean_prob - freq_pos) * weight
        rows.append({
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "count": count,
            "mean_prob": mean_prob,
            "freq_pos": freq_pos,
        })
    return float(ece), rows


def summarize_metric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def region_metrics(prob_2d: np.ndarray, target_2d: np.ndarray, region_mask: np.ndarray, bins: int) -> tuple[dict, list[dict]]:
    p = prob_2d[region_mask].astype(np.float64)
    y = target_2d[region_mask].astype(np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    brier = float(np.mean((p - y) ** 2)) if p.size else 0.0
    nll = float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))) if p.size else 0.0
    ece, bins_data = ece_from_probs(p, y, bins=bins)
    return {
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "mean_prob": float(p.mean()) if p.size else 0.0,
        "pos_rate": float(y.mean()) if y.size else 0.0,
        "n_pixels": int(p.size),
    }, bins_data


@torch.inference_mode()
def collect_calibration(
    model: RPINet,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    tta: str,
    bins: int,
    boundary_width: int,
) -> dict:
    views = tta_views(tta)
    per_case: list[dict] = []
    full_ece: list[float] = []
    full_brier: list[float] = []
    full_nll: list[float] = []
    band_ece: list[float] = []
    band_brier: list[float] = []
    band_nll: list[float] = []
    global_full_prob: list[np.ndarray] = []
    global_full_target: list[np.ndarray] = []
    global_band_prob: list[np.ndarray] = []
    global_band_target: list[np.ndarray] = []

    model.eval()
    for batch in tqdm(loader, desc="calibration"):
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
        probs = (prob_sum / float(len(views)))[:, 0].detach().cpu().numpy()
        targets = mask[:, 0].detach().cpu().numpy()
        ids = batch["id"]

        for image_id, prob_2d, target_2d in zip(ids, probs, targets):
            target_bin = (target_2d > 0.5).astype(np.float32)
            band_mask = boundary_band(target_bin, width=boundary_width)
            full_metrics, _ = region_metrics(prob_2d, target_bin, np.ones_like(target_bin, dtype=bool), bins=bins)
            band_metrics, _ = region_metrics(prob_2d, target_bin, band_mask, bins=bins)
            per_case.append({
                "id": str(image_id),
                "full": full_metrics,
                "boundary_band": band_metrics,
            })
            full_ece.append(full_metrics["ece"])
            full_brier.append(full_metrics["brier"])
            full_nll.append(full_metrics["nll"])
            band_ece.append(band_metrics["ece"])
            band_brier.append(band_metrics["brier"])
            band_nll.append(band_metrics["nll"])
            global_full_prob.append(prob_2d.reshape(-1).astype(np.float64))
            global_full_target.append(target_bin.reshape(-1).astype(np.float64))
            global_band_prob.append(prob_2d[band_mask].astype(np.float64))
            global_band_target.append(target_bin[band_mask].astype(np.float64))

    full_prob = np.concatenate(global_full_prob, axis=0)
    full_target = np.concatenate(global_full_target, axis=0)
    band_prob = np.concatenate(global_band_prob, axis=0)
    band_target = np.concatenate(global_band_target, axis=0)
    _, full_bins = ece_from_probs(full_prob, full_target, bins=bins)
    _, band_bins = ece_from_probs(band_prob, band_target, bins=bins)

    return {
        "per_case": per_case,
        "summary": {
            "full_image": {
                "ece": summarize_metric(full_ece),
                "brier": summarize_metric(full_brier),
                "nll": summarize_metric(full_nll),
            },
            "boundary_band": {
                "ece": summarize_metric(band_ece),
                "brier": summarize_metric(band_brier),
                "nll": summarize_metric(band_nll),
            },
        },
        "reliability_bins": {
            "full_image": full_bins,
            "boundary_band": band_bins,
        },
    }


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

    loader = build_loader(cfg, split=args.split, workers=args.workers, batch_size=args.batch_size)
    result = collect_calibration(
        model=model,
        loader=loader,
        device=device,
        channels_last=channels_last,
        tta=args.tta,
        bins=int(args.bins),
        boundary_width=int(args.boundary_width),
    )
    result["config"] = args.config
    result["checkpoint"] = args.checkpoint
    result["split"] = args.split
    result["tta"] = args.tta
    result["bins"] = int(args.bins)
    result["boundary_width"] = int(args.boundary_width)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
