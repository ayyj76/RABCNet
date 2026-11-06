#!/usr/bin/env python3
"""Evaluate whether interaction-only adaptation improves pseudo-label reliability.

The deployed image-only path intentionally bypasses the pseudo-label branch, so
this script does not use segmentation Dice as the primary adaptation evidence.
Instead, it supplies the same automatically generated target pseudo-labels to
the interaction branch and measures whether its sigma maps identify pseudo-label
errors against the ground-truth masks used only for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
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
    parser.add_argument("--config-no-adapt", required=True)
    parser.add_argument("--checkpoint-no-adapt", required=True)
    parser.add_argument("--config-interaction", required=True)
    parser.add_argument("--checkpoint-interaction", required=True)
    parser.add_argument("--pseudo-root", required=True, help="Root containing <split>/<image_id>.npz")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-pixels", type=int, default=4096)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument(
        "--bootstrap-unit",
        choices=["image", "source"],
        default="image",
        help="Resampling unit for paired bootstrap; image avoids treating pseudo-label sources from one image as independent.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_loader(cfg: dict, split: str, pseudo_root: Path) -> DataLoader:
    data_cfg = cfg["data"]
    pseudo_dir = pseudo_root / split
    split_name = {
        "train": data_cfg["train_split"],
        "val": data_cfg["val_split"],
        "test": data_cfg["test_split"],
    }[split]
    dataset = ISIC2017Dataset(
        root=data_cfg["root"],
        split=split_name,
        image_size=int(data_cfg["image_size"]),
        pseudo_dir=pseudo_dir,
        require_pseudo=True,
        augment=False,
        use_hair_removal=bool(data_cfg.get("use_hair_removal", True)),
        cache_images=False,
        cache_workers=int(data_cfg.get("cache_workers", 4)),
        max_samples=data_cfg.get(
            {"train": "max_train_samples", "val": "max_val_samples", "test": "max_test_samples"}[split]
        ),
        ph2_split_seed=int(data_cfg.get("ph2_split_seed", 42)),
        ph2_split_ratios=tuple(data_cfg.get("ph2_split_ratios", (0.7, 0.15, 0.15))),
    )
    workers = int(data_cfg.get("eval_num_workers", data_cfg.get("num_workers", 2)))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("eval_prefetch_factor", 2))
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )


def load_model(cfg: dict, checkpoint: Path, device: torch.device) -> RPINet:
    kwargs = model_kwargs_from_cfg(cfg)
    kwargs["backbone_pretrained"] = False
    model = RPINet(**kwargs).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    if bool(cfg.get("runtime", {}).get("channels_last", True)):
        model = model.to(memory_format=torch.channels_last)
    return model


def sample_pixels(
    uncertainty: np.ndarray,
    errors: np.ndarray,
    max_pixels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty = uncertainty.reshape(-1).astype(np.float64)
    errors = errors.reshape(-1).astype(np.int8)
    if uncertainty.size > max_pixels:
        indices = rng.choice(uncertainty.size, size=max_pixels, replace=False)
        uncertainty = uncertainty[indices]
        errors = errors[indices]
    return uncertainty, errors


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    u = float(ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0)
    return u / float(n_pos * n_neg)


def pair_metrics(uncertainty: np.ndarray, errors: np.ndarray) -> dict[str, float] | None:
    auc = binary_auc(uncertainty, errors)
    if auc is None:
        return None
    error_u = uncertainty[errors == 1]
    correct_u = uncertainty[errors == 0]
    rho = spearmanr(uncertainty, errors).statistic
    return {
        "error_auc": float(auc),
        "spearman_rho": float(rho) if np.isfinite(rho) else 0.0,
        "error_uncertainty": float(error_u.mean()) if error_u.size else 0.0,
        "correct_uncertainty": float(correct_u.mean()) if correct_u.size else 0.0,
        "uncertainty_separation": float(error_u.mean() - correct_u.mean()) if correct_u.size else 0.0,
        "error_rate": float(errors.mean()),
    }


@torch.inference_mode()
def collect(
    cfg: dict,
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
    max_pixels: int,
    seed: int,
) -> dict:
    model = load_model(cfg, checkpoint, device)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for batch in tqdm(loader, desc=f"reliability {checkpoint.name}"):
        images = batch["image"].to(device, non_blocking=True)
        if bool(cfg.get("runtime", {}).get("channels_last", True)):
            images = images.contiguous(memory_format=torch.channels_last)
        pseudo = batch["pseudo"].to(device, non_blocking=True)
        target = (batch["mask"][:, 0].numpy() > 0.5).astype(np.int8)
        outputs = model(images, pseudo)
        sigma_maps = outputs.get("sigma_maps", [])
        if len(sigma_maps) != pseudo.shape[1]:
            raise RuntimeError(
                f"Expected {pseudo.shape[1]} sigma maps but received {len(sigma_maps)} "
                f"for {checkpoint}"
            )
        sigma = torch.stack([torch.sigmoid(item[:, 0]) for item in sigma_maps], dim=1).cpu().numpy()
        pseudo_np = pseudo.cpu().numpy()
        for row, image_id in enumerate(batch["id"]):
            source_metrics = []
            for source in range(pseudo_np.shape[1]):
                errors = (np.abs(pseudo_np[row, source] - target[row]) > 0.5).astype(np.int8)
                uncertainty, errors = sample_pixels(sigma[row, source], errors, max_pixels, rng)
                metrics = pair_metrics(uncertainty, errors)
                if metrics is not None:
                    source_metrics.append(metrics)
                    records.append({"id": str(image_id), "source": source, **metrics})
            if source_metrics:
                records.append({
                    "id": str(image_id),
                    "source": "macro",
                    **{
                        key: float(np.mean([item[key] for item in source_metrics]))
                        for key in source_metrics[0]
                    },
                })
    source_records = [item for item in records if item["source"] != "macro"]
    summary = {}
    for key in ("error_auc", "spearman_rho", "error_uncertainty", "correct_uncertainty", "uncertainty_separation", "error_rate"):
        values = np.asarray([item[key] for item in source_records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    macro_records = [item for item in records if item["source"] == "macro"]
    summary["image_macro"] = {}
    for key in ("error_auc", "spearman_rho", "error_uncertainty", "correct_uncertainty", "uncertainty_separation", "error_rate"):
        values = np.asarray([item[key] for item in macro_records], dtype=np.float64)
        summary["image_macro"][key] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "summary": summary,
        "records": source_records,
        "image_records": macro_records,
    }


def bootstrap_delta(
    no_adapt: dict,
    interaction: dict,
    iters: int,
    seed: int,
    unit: str,
) -> dict:
    records_key = "image_records" if unit == "image" else "records"
    no = (
        {item["id"]: item for item in no_adapt[records_key]}
        if unit == "image"
        else {(item["id"], item["source"]): item for item in no_adapt[records_key]}
    )
    yes = (
        {item["id"]: item for item in interaction[records_key]}
        if unit == "image"
        else {(item["id"], item["source"]): item for item in interaction[records_key]}
    )
    keys = sorted(set(no) & set(yes))
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ("error_auc", "spearman_rho", "uncertainty_separation"):
        deltas = np.asarray([yes[key][metric] - no[key][metric] for key in keys], dtype=np.float64)
        draws = np.empty(iters, dtype=np.float64)
        for idx in range(iters):
            sample = rng.integers(0, len(deltas), size=len(deltas))
            draws[idx] = float(deltas[sample].mean())
        result[metric] = {
            "interaction_minus_no_adapt_mean": float(deltas.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "n_pairs": int(len(deltas)),
        }
    return result


def main() -> None:
    args = parse_args()
    cfg_no = load_config(args.config_no_adapt)
    cfg_interaction = load_config(args.config_interaction)
    device = get_device(cfg_no.get("runtime", {}).get("device", "cuda"))
    cpu_threads = int(cfg_no.get("runtime", {}).get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, cpu_threads // 2))
        except RuntimeError:
            pass
    pseudo_root = Path(args.pseudo_root)
    loader_no = build_loader(cfg_no, args.split, pseudo_root)
    loader_interaction = build_loader(cfg_interaction, args.split, pseudo_root)
    no_result = collect(
        cfg_no,
        Path(args.checkpoint_no_adapt),
        loader_no,
        device,
        args.max_pixels,
        args.seed,
    )
    interaction_result = collect(
        cfg_interaction,
        Path(args.checkpoint_interaction),
        loader_interaction,
        device,
        args.max_pixels,
        args.seed,
    )
    result = {
        "protocol": "target pseudo-labels generated without ground truth; ground truth used only for post-hoc reliability evaluation",
        "dataset_root": cfg_no["data"]["root"],
        "split": args.split,
        "max_pixels_per_image_source": args.max_pixels,
        "no_adaptation": no_result,
        "interaction_only": interaction_result,
        "bootstrap_unit": args.bootstrap_unit,
        "paired_bootstrap": bootstrap_delta(
            no_result,
            interaction_result,
            args.bootstrap_iters,
            args.seed,
            args.bootstrap_unit,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["paired_bootstrap"], indent=2))


if __name__ == "__main__":
    main()
