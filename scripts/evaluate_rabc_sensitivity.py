#!/usr/bin/env python3
"""Evaluate RABC coefficient sensitivity without retraining the model.

All settings reuse one checkpoint and differ only in the two constructor
coefficients. The script reports fixed-protocol per-image metrics so that
coefficient effects are not confounded with optimization or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import get_device, load_config
from train_rpi_net import model_kwargs_from_cfg

EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tta", default="none", choices=["none", "flip2", "flip4"])
    parser.add_argument("--settings", required=True,
                        help="semicolon-separated intercept,uncertainty pairs")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"unsupported TTA mode: {mode}")


def parse_settings(spec: str) -> list[dict[str, float]]:
    settings = []
    for item in spec.split(";"):
        left, right = item.split(",", maxsplit=1)
        settings.append({
            "candidate_intercept": float(left),
            "candidate_uncertainty": float(right),
        })
    if not settings:
        raise ValueError("no sensitivity settings supplied")
    return settings


def metric_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    tp = (pred * target).sum(axis=(1, 2))
    fp = (pred * (1.0 - target)).sum(axis=(1, 2))
    fn = ((1.0 - pred) * target).sum(axis=(1, 2))
    tn = ((1.0 - pred) * (1.0 - target)).sum(axis=(1, 2))
    return {
        "acc": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
        "dice": (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS),
        "jac": (tp + EPS) / (tp + fp + fn + EPS),
        "sen": (tp + EPS) / (tp + fn + EPS),
        "spe": (tn + EPS) / (tn + fp + EPS),
    }


def build_loader(cfg: dict, split: str) -> DataLoader:
    data_cfg = cfg["data"]
    split_name = data_cfg["val_split"] if split == "val" else data_cfg["test_split"]
    ds = ISIC2017Dataset(
        root=data_cfg["root"],
        split=split_name,
        image_size=int(data_cfg["image_size"]),
        augment=False,
        use_hair_removal=bool(data_cfg.get("use_hair_removal", True)),
        cache_images=bool(data_cfg.get("cache_images", False)),
        cache_workers=int(data_cfg.get("cache_workers", 4)),
        max_samples=data_cfg.get("max_val_samples" if split == "val" else "max_test_samples"),
        ph2_split_seed=int(data_cfg.get("ph2_split_seed", 42)),
        ph2_split_ratios=tuple(data_cfg.get("ph2_split_ratios", (0.7, 0.15, 0.15))),
    )
    workers = int(data_cfg.get("eval_num_workers", data_cfg.get("num_workers", 2)))
    loader_kwargs = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("eval_prefetch_factor", 2))
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(cfg: dict, checkpoint: Path, settings: list[dict[str, float]], split: str,
             threshold: float, tta: str) -> dict:
    runtime = cfg.get("runtime", {})
    cpu_threads = int(runtime.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
    device = get_device(runtime.get("device", "cuda"))
    kwargs = model_kwargs_from_cfg(cfg)
    kwargs["backbone_pretrained"] = False
    model = RPINet(**kwargs).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    if bool(runtime.get("channels_last", True)):
        model = model.to(memory_format=torch.channels_last)
    loader = build_loader(cfg, split)
    views = tta_views(tta)
    per_setting = [dict(setting, metrics={k: [] for k in ["acc", "dice", "jac", "sen", "spe"]})
                   for setting in settings]

    for batch in tqdm(loader, desc=f"RABC sensitivity ({split})"):
        image = batch["image"].to(device, non_blocking=True)
        target = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
        if bool(runtime.get("channels_last", True)):
            image = image.contiguous(memory_format=torch.channels_last)
        for item in per_setting:
            model.rabc.candidate_intercept = item["candidate_intercept"]
            model.rabc.candidate_uncertainty = item["candidate_uncertainty"]
            prob_sum = None
            for hf, vf in views:
                x = image
                if hf:
                    x = torch.flip(x, dims=[3])
                if vf:
                    x = torch.flip(x, dims=[2])
                probs = torch.sigmoid(model(x, None)["logits"])
                if vf:
                    probs = torch.flip(probs, dims=[2])
                if hf:
                    probs = torch.flip(probs, dims=[3])
                prob_sum = probs if prob_sum is None else prob_sum + probs
            pred = (prob_sum / float(len(views)) > threshold).float()
            arrays = metric_arrays(
                pred[:, 0].detach().cpu().numpy(),
                target[:, 0].detach().cpu().numpy(),
            )
            for key, values in arrays.items():
                item["metrics"][key].extend(values.tolist())

    output = []
    for item in per_setting:
        metrics = {}
        for key, values in item["metrics"].items():
            arr = np.asarray(values, dtype=np.float64)
            metrics[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n": int(arr.size),
            }
        output.append({
            "candidate_intercept": item["candidate_intercept"],
            "candidate_uncertainty": item["candidate_uncertainty"],
            "metrics": metrics,
        })
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "split": split,
        "threshold": threshold,
        "tta": tta,
        "settings": output,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = evaluate(
        cfg,
        Path(args.checkpoint),
        parse_settings(args.settings),
        args.split,
        args.threshold,
        args.tta,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
