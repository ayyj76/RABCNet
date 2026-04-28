#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


@dataclass
class EvalSpec:
    name: str
    config: str
    checkpoint: str
    threshold: float
    tta: str
    min_area: int
    keep_largest: bool
    fill_holes: bool
    dilation_iters: int


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
    p = argparse.ArgumentParser(description="Compare no-RABC, dilation, and RABC under fixed protocols")
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--base-threshold", type=float, required=True)
    p.add_argument("--base-tta", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--base-min-area", type=int, default=0)
    p.add_argument("--base-keep-largest", action="store_true")
    p.add_argument("--base-fill-holes", action="store_true")

    p.add_argument("--dil-threshold", type=float, required=True)
    p.add_argument("--dil-tta", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--dil-min-area", type=int, default=0)
    p.add_argument("--dil-keep-largest", action="store_true")
    p.add_argument("--dil-fill-holes", action="store_true")
    p.add_argument("--dil-iterations", type=int, default=1)

    p.add_argument("--rabc-config", required=True)
    p.add_argument("--rabc-checkpoint", required=True)
    p.add_argument("--rabc-threshold", type=float, required=True)
    p.add_argument("--rabc-tta", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--rabc-min-area", type=int, default=0)
    p.add_argument("--rabc-keep-largest", action="store_true")
    p.add_argument("--rabc-fill-holes", action="store_true")

    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
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
                    mask = labeled == int(np.argmax(areas))
                else:
                    mask = np.zeros_like(mask, dtype=bool)
    return mask.astype(np.float32)


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
        loader_kwargs["persistent_workers"] = bool(eval_cfg.get("persistent_workers", data_cfg.get("eval_persistent_workers", data_cfg.get("persistent_workers", True))))
        loader_kwargs["prefetch_factor"] = int(eval_cfg.get("prefetch_factor", data_cfg.get("eval_prefetch_factor", data_cfg.get("prefetch_factor", 4))))

    return DataLoader(
        ds,
        batch_size=int(batch_size_override if batch_size_override is not None else eval_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )


def metrics_from_masks(preds: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    p = preds.astype(np.float64)
    t = targets.astype(np.float64)
    tp = (p * t).sum(axis=(1, 2))
    fp = (p * (1.0 - t)).sum(axis=(1, 2))
    fn = ((1.0 - p) * t).sum(axis=(1, 2))
    tn = ((1.0 - p) * (1.0 - t)).sum(axis=(1, 2))
    return {
        "acc": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
        "dice": (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS),
        "jac": (tp + EPS) / (tp + fp + fn + EPS),
        "sen": (tp + EPS) / (tp + fn + EPS),
        "spe": (tn + EPS) / (tn + fp + EPS),
    }


def paired_bootstrap(diff: np.ndarray, iters: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    n = diff.shape[0]
    obs = float(diff.mean())
    idx = rng.integers(0, n, size=(iters, n))
    bmeans = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bmeans, [2.5, 97.5]).tolist()
    p_le0 = float(np.mean(bmeans <= 0.0))
    p_ge0 = float(np.mean(bmeans >= 0.0))
    p_two = min(1.0, float(2.0 * min(p_le0, p_ge0)))
    return {"mean_diff": obs, "ci95": [float(lo), float(hi)], "p_two_sided": p_two}


def load_model(cfg: dict, checkpoint: str, device: torch.device, channels_last: bool) -> torch.nn.Module:
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
def collect_probs(model: torch.nn.Module, loader: DataLoader, device: torch.device, channels_last: bool, tta: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    views = tta_views(tta)
    ids: list[str] = []
    probs_all = []
    masks_all = []
    for batch in tqdm(loader, desc=f"eval-{tta}", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].cpu().numpy()[:, 0] > 0.5).astype(np.float32)
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
        probs = (prob_sum / float(len(views)))[:, 0].detach().cpu().numpy().astype(np.float32)
        probs_all.append(probs)
        masks_all.append(mask)
    return ids, np.concatenate(probs_all, axis=0), np.concatenate(masks_all, axis=0)


def apply_spec(probs: np.ndarray, spec: EvalSpec) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float32)
    for i in range(probs.shape[0]):
        out[i] = postprocess_mask(
            probs[i],
            threshold=spec.threshold,
            min_area=spec.min_area,
            keep_largest=spec.keep_largest,
            fill_holes=spec.fill_holes,
            dilation_iters=spec.dilation_iters,
        )
    return out


def summarize_metrics(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(v.mean()) for k, v in metrics.items()}


def main() -> None:
    args = parse_args()

    base_spec = EvalSpec(
        name="no-RABC",
        config=args.base_config,
        checkpoint=args.base_checkpoint,
        threshold=float(args.base_threshold),
        tta=args.base_tta,
        min_area=int(args.base_min_area),
        keep_largest=bool(args.base_keep_largest),
        fill_holes=bool(args.base_fill_holes),
        dilation_iters=0,
    )
    dil_spec = EvalSpec(
        name="Dil.",
        config=args.base_config,
        checkpoint=args.base_checkpoint,
        threshold=float(args.dil_threshold),
        tta=args.dil_tta,
        min_area=int(args.dil_min_area),
        keep_largest=bool(args.dil_keep_largest),
        fill_holes=bool(args.dil_fill_holes),
        dilation_iters=int(args.dil_iterations),
    )
    rabc_spec = EvalSpec(
        name="RABC",
        config=args.rabc_config,
        checkpoint=args.rabc_checkpoint,
        threshold=float(args.rabc_threshold),
        tta=args.rabc_tta,
        min_area=int(args.rabc_min_area),
        keep_largest=bool(args.rabc_keep_largest),
        fill_holes=bool(args.rabc_fill_holes),
        dilation_iters=0,
    )

    cfg_base = load_config(base_spec.config)
    cfg_rabc = load_config(rabc_spec.config)
    if Path(cfg_base["data"]["root"]).resolve() != Path(cfg_rabc["data"]["root"]).resolve():
        raise ValueError("base and rabc configs must point to the same dataset root")

    runtime_cfg = cfg_rabc.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    device = get_device(cfg_rabc.get("runtime", {}).get("device", "cuda"))
    channels_last = bool(cfg_rabc.get("runtime", {}).get("channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if bool(cfg_rabc.get("runtime", {}).get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    loader = build_loader(cfg_rabc, args.split, args.workers, args.batch_size)

    model_base = load_model(cfg_base, base_spec.checkpoint, device, channels_last)
    ids_base, probs_base, targets = collect_probs(model_base, loader, device, channels_last, base_spec.tta)
    del model_base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_rabc = load_model(cfg_rabc, rabc_spec.checkpoint, device, channels_last)
    ids_rabc, probs_rabc, targets_rabc = collect_probs(model_rabc, loader, device, channels_last, rabc_spec.tta)
    del model_rabc
    if ids_base != ids_rabc:
        raise RuntimeError("Sample order mismatch between base and RABC evaluations")
    if not np.array_equal(targets, targets_rabc):
        raise RuntimeError("Target masks mismatch between base and RABC evaluations")

    preds_base = apply_spec(probs_base, base_spec)
    preds_dil = apply_spec(probs_base, dil_spec)
    preds_rabc = apply_spec(probs_rabc, rabc_spec)

    metrics = {
        "no-RABC": metrics_from_masks(preds_base, targets),
        "Dil.": metrics_from_masks(preds_dil, targets),
        "RABC": metrics_from_masks(preds_rabc, targets),
    }
    means = {k: summarize_metrics(v) for k, v in metrics.items()}

    pairs = [("no-RABC", "Dil."), ("no-RABC", "RABC"), ("Dil.", "RABC")]
    pairwise = {}
    for a, b in pairs:
        pairwise[f"{a}_vs_{b}"] = {
            metric: {
                "a_mean": float(metrics[a][metric].mean()),
                "b_mean": float(metrics[b][metric].mean()),
                "paired_bootstrap": paired_bootstrap(metrics[b][metric] - metrics[a][metric], args.bootstrap_iters, args.seed),
            }
            for metric in ["acc", "dice", "jac", "sen", "spe"]
        }

    out = {
        "split": args.split,
        "n_samples": int(targets.shape[0]),
        "base_spec": base_spec.__dict__,
        "dil_spec": dil_spec.__dict__,
        "rabc_spec": rabc_spec.__dict__,
        "means_percent": {name: {k: round(v * 100.0, 4) for k, v in vals.items()} for name, vals in means.items()},
        "pairwise": pairwise,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
