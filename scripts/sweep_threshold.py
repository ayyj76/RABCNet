#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep decision threshold on val and report mapped test metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


@torch.inference_mode()
def collect_probs_and_masks(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs_all = []
    masks_all = []

    for batch in tqdm(loader, desc="collect", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()

        logits = model(image, None)["logits"]
        probs = torch.sigmoid(logits)

        probs_all.append(probs.cpu())
        masks_all.append(mask.cpu())

    return torch.cat(probs_all, dim=0), torch.cat(masks_all, dim=0)


def compute_metrics_from_probs(
    probs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    preds = (probs > threshold).float()

    tp = (preds * masks).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - masks)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * masks).sum(dim=(1, 2, 3))
    tn = ((1.0 - preds) * (1.0 - masks)).sum(dim=(1, 2, 3))

    dice = ((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)).mean().item()
    jac = ((tp + EPS) / (tp + fp + fn + EPS)).mean().item()
    sen = ((tp + EPS) / (tp + fn + EPS)).mean().item()
    spe = ((tn + EPS) / (tn + fp + EPS)).mean().item()
    acc = ((tp + tn + EPS) / (tp + tn + fp + fn + EPS)).mean().item()

    return {"acc": acc, "dice": dice, "jac": jac, "sen": sen, "spe": spe}


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

    print("Collecting val probs...")
    val_probs, val_masks = collect_probs_and_masks(model, val_loader, device, channels_last)
    print("Collecting test probs...")
    test_probs, test_masks = collect_probs_and_masks(model, test_loader, device, channels_last)

    thr_min = args.threshold_min
    thr_max = args.threshold_max
    thr_step = args.threshold_step

    thresholds = []
    t = thr_min
    while t <= thr_max + 1e-12:
        thresholds.append(round(t, 6))
        t += thr_step

    rows = []
    best_row = None
    for thr in thresholds:
        val_m = compute_metrics_from_probs(val_probs, val_masks, thr)
        test_m = compute_metrics_from_probs(test_probs, test_masks, thr)
        row = {
            "threshold": thr,
            "val": val_m,
            "test": test_m,
        }
        rows.append(row)

        if (best_row is None) or (row["val"]["jac"] > best_row["val"]["jac"]):
            best_row = row

    if best_row is None:
        raise RuntimeError("No threshold evaluated")

    out = {
        "checkpoint": args.checkpoint,
        "n_thresholds": len(rows),
        "best_by_val_jac": best_row,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": out, "rows": rows}, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
