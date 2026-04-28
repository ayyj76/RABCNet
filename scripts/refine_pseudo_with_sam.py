#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.pseudo_labels import (
    _center_prior,
    _mean_iou_to_set,
    _postprocess,
    _resolve_sam_checkpoint,
    list_split_images,
)
from rpi_net.preprocess import pil_to_numpy_rgb, remove_hair_artifacts
from rpi_net.utils import ensure_dir, load_config

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine existing pseudo labels with SAM replacement gating")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--input-dir", type=str, default=None, help="Root dir that contains <split>/*.npz")
    parser.add_argument("--output-dir", type=str, required=True, help="Root dir to save refined <split>/*.npz")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_sam_mask(sam_masks: List[dict], image_rgb: np.ndarray) -> np.ndarray | None:
    if not sam_masks:
        return None
    h, w = image_rgb.shape[:2]
    gray = image_rgb.mean(axis=2)
    prior = _center_prior(h, w)

    best_score = -1e9
    best_mask = None
    for item in sam_masks:
        seg = item.get("segmentation", None)
        if seg is None:
            continue
        seg = np.asarray(seg, dtype=bool)
        if seg.shape != (h, w):
            continue
        if seg.sum() == 0:
            continue

        area = float(seg.mean())
        if area < 0.01 or area > 0.95:
            continue

        brightness = float(gray[seg].mean()) / 255.0
        center_score = float(prior[seg].mean())
        pred_iou = float(item.get("predicted_iou", 0.5))
        stability = float(item.get("stability_score", 0.5))
        area_score = 1.0 - abs(area - 0.30)
        score = (
            1.1 * (1.0 - brightness)
            + 0.7 * center_score
            + 0.5 * pred_iou
            + 0.3 * stability
            + 0.2 * area_score
        )
        if score > best_score:
            best_score = score
            best_mask = seg

    if best_mask is None:
        return None
    return _postprocess(best_mask.astype(np.float32))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    pseudo_cfg = cfg.get("pseudo", {})
    split = args.split

    input_root = Path(args.input_dir) if args.input_dir else Path(pseudo_cfg["output_dir"])
    input_split_dir = input_root / split
    if not input_split_dir.exists():
        raise FileNotFoundError(f"input split dir not found: {input_split_dir}")

    output_root = Path(args.output_dir)
    output_split_dir = output_root / split
    ensure_dir(output_split_dir)

    images = list_split_images(
        dataset_root=data_cfg["root"],
        split=split,
    )
    if args.max_samples is not None:
        images = images[: int(args.max_samples)]

    device = cfg.get("runtime", {}).get("device", "cuda")
    image_size = int(data_cfg.get("image_size", 224))
    use_hair_removal = bool(data_cfg.get("use_hair_removal", False))
    sam_model_type = str(pseudo_cfg.get("sam_model_type", "vit_b"))
    sam_points_per_side = int(pseudo_cfg.get("sam_points_per_side", 8))
    sam_pred_iou_thresh = float(pseudo_cfg.get("sam_pred_iou_thresh", 0.88))
    sam_stability_score_thresh = float(pseudo_cfg.get("sam_stability_score_thresh", 0.94))
    sam_min_mask_region_area = int(pseudo_cfg.get("sam_min_mask_region_area", 32))
    replace_margin = float(pseudo_cfg.get("sam_replace_margin", 0.02))

    ckpt = _resolve_sam_checkpoint(
        model_type=sam_model_type,
        checkpoint=pseudo_cfg.get("sam_checkpoint"),
    )
    sam_model = sam_model_registry[sam_model_type](checkpoint=str(ckpt))
    sam_model = sam_model.to(device=device).eval()
    sam_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=max(4, sam_points_per_side),
        pred_iou_thresh=sam_pred_iou_thresh,
        stability_score_thresh=sam_stability_score_thresh,
        min_mask_region_area=max(0, sam_min_mask_region_area),
    )

    targets = images if args.force else [p for p in images if not (output_split_dir / f"{p.stem}.npz").exists()]
    replaced = 0
    unchanged = 0
    missing_source = 0

    pbar = tqdm(targets, desc=f"sam-refine-{split}")
    for image_path in pbar:
        image_id = image_path.stem
        src_path = input_split_dir / f"{image_id}.npz"
        if not src_path.exists():
            missing_source += 1
            continue

        src_masks = np.load(src_path)["masks"].astype(np.float32)
        if src_masks.shape[0] != 4:
            raise ValueError(f"Expected 4-source masks at {src_path}, got shape={src_masks.shape}")

        image = Image.open(image_path).convert("RGB")
        arr = pil_to_numpy_rgb(image)
        if use_hair_removal:
            arr = remove_hair_artifacts(arr)
        arr = np.array(Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)

        sam_mask = None
        try:
            sam_candidates = sam_generator.generate(arr)
            sam_mask = select_sam_mask(sam_candidates, arr)
        except Exception:
            sam_mask = None

        out_masks = src_masks.copy()
        did_replace = False
        if sam_mask is not None:
            src_list = [out_masks[i] for i in range(4)]
            src_scores = []
            for i in range(4):
                others = [src_list[j] for j in range(4) if j != i]
                src_scores.append(_mean_iou_to_set(src_list[i], others))
            worst_idx = int(np.argmin(np.asarray(src_scores)))
            sam_score = _mean_iou_to_set(sam_mask, src_list)
            if sam_score >= src_scores[worst_idx] + replace_margin:
                out_masks[worst_idx] = sam_mask.astype(np.float32)
                did_replace = True

        np.savez_compressed(output_split_dir / f"{image_id}.npz", masks=out_masks.astype(np.float32))
        if did_replace:
            replaced += 1
        else:
            unchanged += 1

    print(
        {
            "split": split,
            "n_targets": len(targets),
            "replaced": replaced,
            "unchanged": unchanged,
            "missing_source": missing_source,
            "output": str(output_split_dir),
        }
    )


if __name__ == "__main__":
    main()
