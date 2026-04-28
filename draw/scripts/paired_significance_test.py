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
class ModelEvalSpec:
    name: str
    config: str
    checkpoint: str
    threshold: float
    tta: str
    min_area: int
    keep_largest: bool
    fill_holes: bool


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
    p = argparse.ArgumentParser(description="Paired bootstrap significance test for two segmentation models")
    p.add_argument("--name-a", type=str, default="A")
    p.add_argument("--config-a", type=str, required=True)
    p.add_argument("--checkpoint-a", type=str, required=True)
    p.add_argument("--threshold-a", type=float, required=True)
    p.add_argument("--tta-a", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--min-area-a", type=int, default=0)
    p.add_argument("--keep-largest-a", action="store_true")
    p.add_argument("--fill-holes-a", action="store_true")

    p.add_argument("--name-b", type=str, default="B")
    p.add_argument("--config-b", type=str, required=True)
    p.add_argument("--checkpoint-b", type=str, required=True)
    p.add_argument("--threshold-b", type=float, required=True)
    p.add_argument("--tta-b", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--min-area-b", type=int, default=0)
    p.add_argument("--keep-largest-b", action="store_true")
    p.add_argument("--fill-holes-b", action="store_true")

    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, required=True)
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


def build_loader(
    cfg: dict,
    split_name: str,
    workers_override: int | None,
    batch_size_override: int | None,
) -> DataLoader:
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
        batch_size=int(batch_size_override if batch_size_override is not None else eval_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )


@torch.inference_mode()
def evaluate_per_sample(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    threshold: float,
    tta: str,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
) -> tuple[list[str], dict[str, np.ndarray]]:
    model.eval()
    views = tta_views(tta)

    ids: list[str] = []
    acc_all = []
    dice_all = []
    jac_all = []
    sen_all = []
    spe_all = []

    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
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
            prob_sum = prob if prob_sum is None else (prob_sum + prob)
        probs = prob_sum / float(len(views))

        if min_area > 0 or keep_largest or fill_holes:
            preds_np = np.zeros((probs.shape[0], probs.shape[2], probs.shape[3]), dtype=np.float32)
            probs_np = probs[:, 0].detach().cpu().numpy()
            for i in range(probs_np.shape[0]):
                preds_np[i] = postprocess_mask(
                    probs_np[i],
                    threshold=threshold,
                    min_area=min_area,
                    keep_largest=keep_largest,
                    fill_holes=fill_holes,
                )
            preds = torch.from_numpy(preds_np).unsqueeze(1).to(mask.device)
        else:
            preds = (probs > threshold).float()

        tp = (preds * mask).sum(dim=(1, 2, 3)).cpu().numpy()
        fp = (preds * (1.0 - mask)).sum(dim=(1, 2, 3)).cpu().numpy()
        fn = ((1.0 - preds) * mask).sum(dim=(1, 2, 3)).cpu().numpy()
        tn = ((1.0 - preds) * (1.0 - mask)).sum(dim=(1, 2, 3)).cpu().numpy()

        acc = (tp + tn + EPS) / (tp + tn + fp + fn + EPS)
        dice = (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)
        jac = (tp + EPS) / (tp + fp + fn + EPS)
        sen = (tp + EPS) / (tp + fn + EPS)
        spe = (tn + EPS) / (tn + fp + EPS)

        acc_all.append(acc.astype(np.float64))
        dice_all.append(dice.astype(np.float64))
        jac_all.append(jac.astype(np.float64))
        sen_all.append(sen.astype(np.float64))
        spe_all.append(spe.astype(np.float64))

    metrics = {
        "acc": np.concatenate(acc_all, axis=0),
        "dice": np.concatenate(dice_all, axis=0),
        "jac": np.concatenate(jac_all, axis=0),
        "sen": np.concatenate(sen_all, axis=0),
        "spe": np.concatenate(spe_all, axis=0),
    }
    return ids, metrics


def paired_bootstrap(diff: np.ndarray, iters: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    n = diff.shape[0]
    obs = float(diff.mean())
    if iters <= 0:
        return {
            "mean_diff": obs,
            "ci95": [obs, obs],
            "p_one_sided_gt0": float(obs <= 0.0),
            "p_two_sided": 1.0,
        }
    idx = rng.integers(0, n, size=(iters, n))
    bmeans = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bmeans, [2.5, 97.5]).tolist()
    p_le0 = float(np.mean(bmeans <= 0.0))
    p_ge0 = float(np.mean(bmeans >= 0.0))
    p_two = float(2.0 * min(p_le0, p_ge0))
    p_two = min(1.0, p_two)
    return {
        "mean_diff": obs,
        "ci95": [float(lo), float(hi)],
        "p_one_sided_gt0": p_le0,
        "p_two_sided": p_two,
    }


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


def main() -> None:
    args = parse_args()

    spec_a = ModelEvalSpec(
        name=args.name_a,
        config=args.config_a,
        checkpoint=args.checkpoint_a,
        threshold=float(args.threshold_a),
        tta=args.tta_a,
        min_area=int(args.min_area_a),
        keep_largest=bool(args.keep_largest_a),
        fill_holes=bool(args.fill_holes_a),
    )
    spec_b = ModelEvalSpec(
        name=args.name_b,
        config=args.config_b,
        checkpoint=args.checkpoint_b,
        threshold=float(args.threshold_b),
        tta=args.tta_b,
        min_area=int(args.min_area_b),
        keep_largest=bool(args.keep_largest_b),
        fill_holes=bool(args.fill_holes_b),
    )

    cfg_a = load_config(spec_a.config)
    cfg_b = load_config(spec_b.config)
    if Path(cfg_a["data"]["root"]).resolve() != Path(cfg_b["data"]["root"]).resolve():
        raise ValueError("Two models must be evaluated on the same dataset root")

    runtime_cfg = cfg_b.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    device = get_device(cfg_b.get("runtime", {}).get("device", "cuda"))
    channels_last = bool(cfg_b.get("runtime", {}).get("channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if bool(cfg_b.get("runtime", {}).get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    loader = build_loader(
        cfg_b,
        split_name=args.split,
        workers_override=args.workers,
        batch_size_override=args.batch_size,
    )

    model_a = load_model(cfg_a, spec_a.checkpoint, device, channels_last)
    ids_a, m_a = evaluate_per_sample(
        model=model_a,
        loader=loader,
        device=device,
        channels_last=channels_last,
        threshold=spec_a.threshold,
        tta=spec_a.tta,
        min_area=spec_a.min_area,
        keep_largest=spec_a.keep_largest,
        fill_holes=spec_a.fill_holes,
    )
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_b = load_model(cfg_b, spec_b.checkpoint, device, channels_last)
    ids_b, m_b = evaluate_per_sample(
        model=model_b,
        loader=loader,
        device=device,
        channels_last=channels_last,
        threshold=spec_b.threshold,
        tta=spec_b.tta,
        min_area=spec_b.min_area,
        keep_largest=spec_b.keep_largest,
        fill_holes=spec_b.fill_holes,
    )
    del model_b

    if ids_a != ids_b:
        raise RuntimeError("Sample order mismatch between model A and B")

    metrics = ["acc", "dice", "jac", "sen", "spe"]
    diff_stats = {}
    for k in metrics:
        d = m_b[k] - m_a[k]
        diff_stats[k] = {
            "a_mean": float(m_a[k].mean()),
            "b_mean": float(m_b[k].mean()),
            "a_std": float(m_a[k].std(ddof=1)),
            "b_std": float(m_b[k].std(ddof=1)),
            "paired_bootstrap": paired_bootstrap(d, int(args.bootstrap_iters), int(args.seed)),
        }

    out = {
        "split": args.split,
        "n_samples": len(ids_a),
        "bootstrap_iters": int(args.bootstrap_iters),
        "seed": int(args.seed),
        "model_a": {
            "name": spec_a.name,
            "config": spec_a.config,
            "checkpoint": spec_a.checkpoint,
            "operating_point": {
                "threshold": spec_a.threshold,
                "tta": spec_a.tta,
                "min_area": spec_a.min_area,
                "keep_largest": spec_a.keep_largest,
                "fill_holes": spec_a.fill_holes,
            },
        },
        "model_b": {
            "name": spec_b.name,
            "config": spec_b.config,
            "checkpoint": spec_b.checkpoint,
            "operating_point": {
                "threshold": spec_b.threshold,
                "tta": spec_b.tta,
                "min_area": spec_b.min_area,
                "keep_largest": spec_b.keep_largest,
                "fill_holes": spec_b.fill_holes,
            },
        },
        "metrics": diff_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
