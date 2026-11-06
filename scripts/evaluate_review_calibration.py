#!/usr/bin/env python3
"""Evaluate full-image and boundary-band probability calibration.

The primary protocol is raw image-only inference (no threshold, TTA, or
morphology). Metrics are reported both as image-macro means and pooled pixel
values; paired image-bootstrap intervals are provided for model differences.
"""
from __future__ import annotations

import argparse
import hashlib
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
sys.path.insert(0, str(ROOT / "scripts"))

from rpi_net.datasets import ISIC2017Dataset
from rpi_net.model import RPINet
from rpi_net.utils import get_device, load_config
from train_rpi_net import model_kwargs_from_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-no-rabc", required=True)
    parser.add_argument("--checkpoint-no-rabc", required=True)
    parser.add_argument("--config-rabc", required=True)
    parser.add_argument("--checkpoint-rabc", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


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
    return DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, **loader_kwargs)


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    surface = np.logical_xor(
        mask.astype(bool),
        ndi.binary_erosion(mask.astype(bool), structure=np.ones((3, 3), dtype=bool), border_value=0),
    )
    if not surface.any():
        surface = mask.astype(bool)
    return ndi.distance_transform_edt(~surface) <= float(width)


def ece(prob: np.ndarray, target: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    n = float(prob.size)
    for idx in range(bins):
        selected = ((prob >= edges[idx]) & (prob <= edges[idx + 1])
                    if idx == bins - 1 else
                    (prob >= edges[idx]) & (prob < edges[idx + 1]))
        if selected.any():
            value += float(selected.mean()) * abs(float(prob[selected].mean()) - float(target[selected].mean()))
    return value


def calibration_triplet(prob: np.ndarray, target: np.ndarray, bins: int) -> dict[str, float]:
    p = np.clip(prob.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = target.astype(np.float64)
    return {
        "ece": ece(p, y, bins),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def collect(cfg: dict, checkpoint: Path, split: str, bins: int, width: int) -> dict:
    runtime = cfg.get("runtime", {})
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
    cases = []
    pooled = {"full_image": {"prob": [], "target": []}, "boundary_band": {"prob": [], "target": []}}

    for batch in tqdm(loader, desc=f"calibration {checkpoint.name}"):
        image = batch["image"].to(device, non_blocking=True)
        if bool(runtime.get("channels_last", True)):
            image = image.contiguous(memory_format=torch.channels_last)
        probs = torch.sigmoid(model(image, None)["logits"])[:, 0].cpu().numpy()
        targets = (batch["mask"][:, 0].numpy() > 0.5).astype(np.float32)
        for image_id, prob, target in zip(batch["id"], probs, targets):
            band = boundary_band(target, width)
            full = calibration_triplet(prob.reshape(-1), target.reshape(-1), bins)
            local = calibration_triplet(prob[band], target[band], bins)
            cases.append({"id": str(image_id), "full_image": full, "boundary_band": local})
            pooled["full_image"]["prob"].append(prob.reshape(-1))
            pooled["full_image"]["target"].append(target.reshape(-1))
            pooled["boundary_band"]["prob"].append(prob[band])
            pooled["boundary_band"]["target"].append(target[band])

    summary = {}
    for region in ["full_image", "boundary_band"]:
        summary[region] = {}
        for metric in ["ece", "brier", "nll"]:
            values = np.asarray([case[region][metric] for case in cases], dtype=np.float64)
            summary[region][metric] = {
                "macro_mean": float(values.mean()),
                "macro_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n_images": int(values.size),
            }
        pooled_prob = np.concatenate(pooled[region]["prob"])
        pooled_target = np.concatenate(pooled[region]["target"])
        summary[region]["pooled"] = calibration_triplet(pooled_prob, pooled_target, bins)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "split": split,
        "bins": bins,
        "boundary_width": width,
        "summary": summary,
        "cases": cases,
    }


def bootstrap_delta(no_rabc: dict, rabc: dict, iters: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(no_rabc["cases"])
    out = {region: {} for region in ["full_image", "boundary_band"]}
    for region in out:
        for metric in ["ece", "brier", "nll"]:
            a = np.asarray([case[region][metric] for case in no_rabc["cases"]], dtype=np.float64)
            b = np.asarray([case[region][metric] for case in rabc["cases"]], dtype=np.float64)
            draws = np.empty(iters, dtype=np.float64)
            for idx in range(iters):
                sample = rng.integers(0, n, size=n)
                draws[idx] = float((b[sample] - a[sample]).mean())
            out[region][metric] = {
                "rabc_minus_no_rabc_mean": float((b - a).mean()),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            }
    return out


def main() -> None:
    args = parse_args()
    cfg_no = load_config(args.config_no_rabc)
    cfg_rabc = load_config(args.config_rabc)
    runtime = cfg_no.get("runtime", {})
    cpu_threads = int(runtime.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, cpu_threads // 2))
        except RuntimeError:
            # The process may already have initialized inter-op workers.
            pass
    no_result = collect(cfg_no, Path(args.checkpoint_no_rabc), args.split, args.bins, args.boundary_width)
    rabc_result = collect(cfg_rabc, Path(args.checkpoint_rabc), args.split, args.bins, args.boundary_width)
    result = {
        "protocol": "raw image-only inference; no threshold, TTA, or morphology",
        "bootstrap_unit": "image",
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.seed),
        "no_rabc": no_result,
        "rabc": rabc_result,
        "paired_image_bootstrap": bootstrap_delta(no_result, rabc_result, args.bootstrap_iters, args.seed),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["paired_image_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
