#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import SPLIT_MAP
from rpi_net.preprocess import pil_to_numpy_rgb, remove_hair_artifacts
from rpi_net.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tensor cache for ISIC2017 to remove online I/O bottlenecks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    return parser.parse_args()


def _normalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def build_split_cache(
    root: Path,
    split: str,
    image_size: int,
    use_hair_removal: bool,
    pseudo_dir: Path | None,
    out_file: Path,
    workers: int,
    dtype: torch.dtype,
    pseudo_dtype: torch.dtype,
) -> None:
    paths = SPLIT_MAP[split]
    image_dir = root / paths.image_dir
    mask_dir = root / paths.mask_dir

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    def _load_one(image_path: Path):
        image_id = image_path.stem

        image = Image.open(image_path).convert("RGB")
        if use_hair_removal:
            arr = pil_to_numpy_rgb(image)
            arr = remove_hair_artifacts(arr)
            image = Image.fromarray(arr)
        image = image.resize((image_size, image_size), Image.BILINEAR)
        image_t = _normalize(TF.to_tensor(image)).to(dtype)

        mask_path = mask_dir / f"{image_id}_segmentation.png"
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        mask_t = (TF.to_tensor(mask) > 0.5).to(torch.uint8)

        pseudo_t = None
        if pseudo_dir is not None:
            pseudo_path = pseudo_dir / f"{image_id}.npz"
            if not pseudo_path.exists():
                raise FileNotFoundError(f"Missing pseudo label: {pseudo_path}")
            # Keep soft pseudo labels; uint8 would collapse fractional consensus values.
            pseudo = np.load(pseudo_path)["masks"].astype(np.float32)
            pseudo_t = torch.from_numpy(pseudo).to(pseudo_dtype)

        return image_id, image_t, mask_t, pseudo_t

    ids: List[str] = []
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    pseudos: List[torch.Tensor] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for image_id, image_t, mask_t, pseudo_t in tqdm(ex.map(_load_one, image_paths), total=len(image_paths), desc=f"cache-{split}"):
            ids.append(image_id)
            images.append(image_t)
            masks.append(mask_t)
            if pseudo_t is not None:
                pseudos.append(pseudo_t)

    payload = {
        "ids": ids,
        "images": torch.stack(images, dim=0),
        "masks": torch.stack(masks, dim=0),
    }
    if len(pseudos) > 0:
        payload["pseudos"] = torch.stack(pseudos, dim=0)

    torch.save(payload, out_file)
    print(f"saved: {out_file}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    image_size = int(data_cfg["image_size"])
    use_hair_removal = bool(data_cfg.get("use_hair_removal", True))
    cache_dir = Path(data_cfg["tensor_cache_dir"])
    ensure_dir(cache_dir)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    pseudo_dtype = dtype

    train_cache = cache_dir / "train_cache.pt"
    val_cache = cache_dir / "val_cache.pt"
    test_cache = cache_dir / "test_cache.pt"

    if (not args.force) and train_cache.exists() and val_cache.exists() and test_cache.exists():
        print("tensor cache already exists, skip")
        return

    pseudo_train_dir = Path(cfg["pseudo"]["output_dir"]) / data_cfg["train_split"]

    build_split_cache(
        root=root,
        split=data_cfg["train_split"],
        image_size=image_size,
        use_hair_removal=use_hair_removal,
        pseudo_dir=pseudo_train_dir,
        out_file=train_cache,
        workers=args.workers,
        dtype=dtype,
        pseudo_dtype=pseudo_dtype,
    )
    build_split_cache(
        root=root,
        split=data_cfg["val_split"],
        image_size=image_size,
        use_hair_removal=use_hair_removal,
        pseudo_dir=None,
        out_file=val_cache,
        workers=max(1, args.workers // 2),
        dtype=dtype,
        pseudo_dtype=pseudo_dtype,
    )
    build_split_cache(
        root=root,
        split=data_cfg["test_split"],
        image_size=image_size,
        use_hair_removal=use_hair_removal,
        pseudo_dir=None,
        out_file=test_cache,
        workers=max(1, args.workers // 2),
        dtype=dtype,
        pseudo_dtype=pseudo_dtype,
    )


if __name__ == "__main__":
    main()
