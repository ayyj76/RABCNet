from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .preprocess import pil_to_numpy_rgb, remove_hair_artifacts


@dataclass(frozen=True)
class SplitPaths:
    image_dir: str
    mask_dir: str


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_path: Path
    mask_path: Path


ISIC2017_SPLIT_MAP: Dict[str, SplitPaths] = {
    "train": SplitPaths(
        image_dir="ISIC-2017_Training_Data",
        mask_dir="ISIC-2017_Training_Part1_GroundTruth",
    ),
    "val": SplitPaths(
        image_dir="ISIC-2017_Validation_Data",
        mask_dir="ISIC-2017_Validation_Part1_GroundTruth",
    ),
    "test": SplitPaths(
        image_dir="ISIC-2017_Test_v2_Data",
        mask_dir="ISIC-2017_Test_v2_Part1_GroundTruth",
    ),
}

# Backward compatibility for existing imports.
SPLIT_MAP = ISIC2017_SPLIT_MAP

ISIC2018_SPLIT_MAP: Dict[str, SplitPaths] = {
    "train": SplitPaths(
        image_dir="ISIC2018_Task1-2_Training_Input",
        mask_dir="ISIC2018_Task1_Training_GroundTruth",
    ),
    "val": SplitPaths(
        image_dir="ISIC2018_Task1-2_Validation_Input",
        mask_dir="ISIC2018_Task1_Validation_GroundTruth",
    ),
    "test": SplitPaths(
        image_dir="ISIC2018_Task1-2_Test_Input",
        mask_dir="ISIC2018_Task1_Test_GroundTruth",
    ),
}


def _infer_dataset_kind(root: Path) -> str:
    if (root / "ISIC-2017_Training_Data").exists():
        return "isic2017"
    if (root / "ISIC2018_Task1-2_Training_Input").exists():
        return "isic2018"
    if (root / "PH2Dataset" / "PH2 Dataset images").exists():
        return "ph2"
    if (root / "PH2 Dataset images").exists():
        return "ph2"
    raise RuntimeError(f"Cannot infer dataset type from root: {root}")


def _ph2_image_root(root: Path) -> Path:
    nested = root / "PH2Dataset" / "PH2 Dataset images"
    if nested.exists():
        return nested
    flat = root / "PH2 Dataset images"
    if flat.exists():
        return flat
    raise RuntimeError(f"PH2 image directory not found under: {root}")


def _resolve_isic_samples(root: Path, split: str, split_map: Dict[str, SplitPaths]) -> List[SampleRecord]:
    if split not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    paths = split_map[split]
    image_dir = root / paths.image_dir
    mask_dir = root / paths.mask_dir

    image_paths: List[Path] = []
    for patt in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(image_dir.glob(patt))
    image_paths = sorted(set(image_paths))

    # ignore generated derivatives accidentally mixed into image folders
    ignore_suffixes = ("_superpixels",)

    samples: List[SampleRecord] = []
    for image_path in image_paths:
        image_id = image_path.stem
        if any(image_id.endswith(sfx) for sfx in ignore_suffixes):
            continue
        mask_path = mask_dir / f"{image_id}_segmentation.png"
        if not mask_path.exists():
            alt_png = mask_dir / f"{image_id}.png"
            if alt_png.exists():
                mask_path = alt_png
            else:
                raise FileNotFoundError(f"Mask not found for {image_id}: {mask_path}")
        samples.append(SampleRecord(image_id=image_id, image_path=image_path, mask_path=mask_path))
    return samples


def _resolve_ph2_samples(
    root: Path,
    split: str,
    seed: int,
    ratios: tuple[float, float, float],
) -> List[SampleRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported PH2 split: {split}")

    train_ratio, val_ratio, test_ratio = ratios
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        raise ValueError(f"Invalid PH2 split ratios: {ratios}")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    ph2_root = _ph2_image_root(root)
    case_dirs = sorted([p for p in ph2_root.iterdir() if p.is_dir()])
    samples: List[SampleRecord] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        image_path = case_dir / f"{case_id}_Dermoscopic_Image" / f"{case_id}.bmp"
        mask_path = case_dir / f"{case_id}_lesion" / f"{case_id}_lesion.bmp"
        if image_path.exists() and mask_path.exists():
            samples.append(SampleRecord(image_id=case_id, image_path=image_path, mask_path=mask_path))

    if not samples:
        raise RuntimeError(f"No PH2 samples resolved under: {ph2_root}")

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(samples))
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 1)
    n_val = max(1, min(n_val, n - n_train - 1)) if (n - n_train) >= 2 else max(0, n - n_train - 1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    out = [samples[int(i)] for i in split_indices]
    out.sort(key=lambda x: x.image_id)
    return out


def resolve_split_samples(
    root: str | Path,
    split: str,
    ph2_split_seed: int = 42,
    ph2_split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> List[SampleRecord]:
    root_p = Path(root)
    kind = _infer_dataset_kind(root_p)
    if kind == "isic2017":
        return _resolve_isic_samples(root_p, split, ISIC2017_SPLIT_MAP)
    if kind == "isic2018":
        return _resolve_isic_samples(root_p, split, ISIC2018_SPLIT_MAP)
    if kind == "ph2":
        return _resolve_ph2_samples(root_p, split, seed=ph2_split_seed, ratios=ph2_split_ratios)
    raise RuntimeError(f"Unsupported dataset kind: {kind}")


def list_split_image_paths(
    root: str | Path,
    split: str,
    ph2_split_seed: int = 42,
    ph2_split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> List[Path]:
    samples = resolve_split_samples(
        root=root,
        split=split,
        ph2_split_seed=ph2_split_seed,
        ph2_split_ratios=ph2_split_ratios,
    )
    return [s.image_path for s in samples]


class ISIC2017Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 224,
        pseudo_dir: str | Path | None = None,
        require_pseudo: bool = False,
        augment: bool = False,
        use_hair_removal: bool = True,
        cache_images: bool = False,
        cache_workers: int = 4,
        max_samples: Optional[int] = None,
        ph2_split_seed: int = 42,
        ph2_split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.pseudo_dir = Path(pseudo_dir) if pseudo_dir else None
        self.require_pseudo = require_pseudo
        self.augment = augment
        self.use_hair_removal = use_hair_removal
        self.cache_images = cache_images
        self.cache_workers = max(1, int(cache_workers))
        self.samples = resolve_split_samples(
            root=self.root,
            split=split,
            ph2_split_seed=int(ph2_split_seed),
            ph2_split_ratios=ph2_split_ratios,
        )
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No samples found in {self.root} ({split})")

        self._image_cache: Dict[str, torch.Tensor] = {}
        self._mask_cache: Dict[str, torch.Tensor] = {}
        self._pseudo_cache: Dict[str, torch.Tensor] = {}
        if self.cache_images:
            with ThreadPoolExecutor(max_workers=min(self.cache_workers, len(self.samples))) as ex:
                for image_id, image_t, mask_t, pseudo_t in ex.map(self._cache_one, self.samples):
                    self._image_cache[image_id] = image_t
                    self._mask_cache[image_id] = mask_t
                    if pseudo_t is not None:
                        self._pseudo_cache[image_id] = pseudo_t

    def _cache_one(self, sample: SampleRecord) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        image_t = self._load_image(sample.image_path)
        mask_t = self._load_mask(sample.mask_path)
        pseudo_t = None
        if self.pseudo_dir is not None:
            pseudo_t = self._load_pseudo(sample.image_id)
        return sample.image_id, image_t, mask_t, pseudo_t

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.use_hair_removal:
            arr = pil_to_numpy_rgb(image)
            arr = remove_hair_artifacts(arr)
            image = Image.fromarray(arr)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(image)
        return tensor

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()
        return mask_t

    def _load_pseudo(self, image_id: str) -> torch.Tensor:
        if self.pseudo_dir is None:
            raise RuntimeError("pseudo_dir is required when loading pseudo labels")
        pseudo_path = self.pseudo_dir / f"{image_id}.npz"
        if not pseudo_path.exists():
            if self.require_pseudo:
                raise FileNotFoundError(f"Pseudo-label file not found: {pseudo_path}")
            return torch.empty(0)

        pseudo = np.load(pseudo_path)["masks"].astype(np.float32)
        # [4, H, W]
        pseudo_t = torch.from_numpy(pseudo)
        return pseudo_t

    @staticmethod
    def _normalize(image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(3, 1, 1)
        return (image - mean) / std

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        pseudo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            if pseudo.numel() > 0:
                pseudo = torch.flip(pseudo, dims=[2])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            if pseudo.numel() > 0:
                pseudo = torch.flip(pseudo, dims=[1])

        return image, mask, pseudo

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_path = sample.image_path
        image_id = sample.image_id

        if self.cache_images:
            image = self._image_cache[image_id].clone()
            mask = self._mask_cache[image_id].clone()
            pseudo = (
                self._pseudo_cache[image_id].clone()
                if self.pseudo_dir and image_id in self._pseudo_cache
                else torch.empty(0)
            )
        else:
            image = self._load_image(image_path)
            mask = self._load_mask(sample.mask_path)
            pseudo = self._load_pseudo(image_id) if self.pseudo_dir else torch.empty(0)

        if self.augment:
            image, mask, pseudo = self._apply_augmentation(image, mask, pseudo)

        image = self._normalize(image)

        return {
            "id": image_id,
            "image": image,
            "mask": mask,
            "pseudo": pseudo,
        }


class ISIC2017TensorCacheDataset(Dataset):
    def __init__(
        self,
        cache_file: str | Path,
        require_pseudo: bool = False,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.cache_file = Path(cache_file)
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Tensor cache file not found: {self.cache_file}")

        payload = torch.load(self.cache_file, map_location="cpu")
        self.ids: List[str] = payload["ids"]
        self.images: torch.Tensor = payload["images"]  # normalized, float16/float32
        self.masks: torch.Tensor = payload["masks"]  # uint8/float [N,1,H,W]
        self.pseudos: torch.Tensor | None = payload.get("pseudos", None)  # uint8/float [N,4,H,W] or None

        # Convert once at dataset init to avoid per-sample dtype conversion overhead.
        if self.images.dtype != torch.float32:
            self.images = self.images.float()
        if self.masks.dtype != torch.float32:
            self.masks = self.masks.float()
        if self.pseudos is not None and self.pseudos.dtype != torch.float32:
            self.pseudos = self.pseudos.float()

        self.require_pseudo = require_pseudo
        self.augment = augment

        if self.require_pseudo and self.pseudos is None:
            raise RuntimeError(f"Pseudo labels required but missing in {self.cache_file}")

        if max_samples is not None:
            max_samples = int(max_samples)
            self.ids = self.ids[:max_samples]
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]
            if self.pseudos is not None:
                self.pseudos = self.pseudos[:max_samples]

    def __len__(self) -> int:
        return len(self.ids)

    def _apply_augmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        pseudo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            if pseudo.numel() > 0:
                pseudo = torch.flip(pseudo, dims=[2])

        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            if pseudo.numel() > 0:
                pseudo = torch.flip(pseudo, dims=[1])
        return image, mask, pseudo

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        image_id = self.ids[index]
        image = self.images[index]
        mask = (self.masks[index] > 0.5).float()

        if self.pseudos is None:
            pseudo = torch.empty(0)
        else:
            pseudo = self.pseudos[index]

        if self.augment:
            image, mask, pseudo = self._apply_augmentation(image, mask, pseudo)

        return {
            "id": image_id,
            "image": image,
            "mask": mask,
            "pseudo": pseudo,
        }
