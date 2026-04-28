#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.pseudo_labels import MultiSourcePseudoLabelGenerator, list_split_images
from rpi_net.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-source pseudo labels for RPI-Net")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root = cfg["data"]["root"]
    image_size = int(cfg["data"]["image_size"])
    device = cfg["runtime"].get("device", "cuda")
    kmeans_clusters = int(cfg["pseudo"].get("kmeans_clusters", 2))
    kmeans_threads = int(cfg["pseudo"].get("kmeans_threads", 4))
    batch_size = int(cfg["pseudo"].get("batch_size", 8))

    output_dir = Path(cfg["pseudo"]["output_dir"]) / args.split
    ensure_dir(output_dir)

    images = list_split_images(root, args.split)
    if args.max_samples is not None:
        images = images[: args.max_samples]

    generator = MultiSourcePseudoLabelGenerator(
        image_size=image_size,
        device=device,
        kmeans_clusters=kmeans_clusters,
        kmeans_threads=kmeans_threads,
        consensus_blend=float(cfg["pseudo"].get("consensus_blend", 0.2)),
        use_sam=bool(cfg["pseudo"].get("use_sam", False)),
        sam_model_type=str(cfg["pseudo"].get("sam_model_type", "vit_b")),
        sam_checkpoint=cfg["pseudo"].get("sam_checkpoint"),
        sam_points_per_side=int(cfg["pseudo"].get("sam_points_per_side", 16)),
        sam_pred_iou_thresh=float(cfg["pseudo"].get("sam_pred_iou_thresh", 0.86)),
        sam_stability_score_thresh=float(cfg["pseudo"].get("sam_stability_score_thresh", 0.92)),
        sam_min_mask_region_area=int(cfg["pseudo"].get("sam_min_mask_region_area", 32)),
        sam_p2_blend=float(cfg["pseudo"].get("sam_p2_blend", 0.35)),
        sam_refine_blend=float(cfg["pseudo"].get("sam_refine_blend", 0.2)),
        sam_min_iou_with_legacy=float(cfg["pseudo"].get("sam_min_iou_with_legacy", 0.10)),
        sam_replace_margin=float(cfg["pseudo"].get("sam_replace_margin", 0.02)),
    )

    targets = images if args.force else [p for p in images if not (output_dir / f"{p.stem}.npz").exists()]
    pbar = tqdm(total=len(targets), desc=f"pseudo-{args.split}")
    for start in range(0, len(targets), batch_size):
        batch_paths = targets[start : start + batch_size]
        pseudos = generator.generate_for_paths(batch_paths)
        import numpy as np

        for image_path, pseudo in zip(batch_paths, pseudos):
            image_id = image_path.stem
            out_path = output_dir / f"{image_id}.npz"
            np.savez_compressed(out_path, masks=pseudo["masks"])
        pbar.update(len(batch_paths))
    pbar.close()

    print(f"Pseudo labels saved to: {output_dir}")


if __name__ == "__main__":
    main()
