from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet50,
    vit_b_16,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as TF

from .datasets import list_split_image_paths
from .preprocess import pil_to_numpy_rgb, remove_hair_artifacts

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception:  # pragma: no cover
    SamAutomaticMaskGenerator = None
    sam_model_registry = None


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _center_prior(h: int, w: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    ys = (ys - (h - 1) / 2.0) / (h / 2.0)
    xs = (xs - (w - 1) / 2.0) / (w / 2.0)
    sigma = 0.6
    prior = np.exp(-(xs**2 + ys**2) / (2.0 * sigma**2))
    return prior


def _select_lesion_cluster(binary_map: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    gray = image_rgb.mean(axis=2)
    prior = _center_prior(binary_map.shape[0], binary_map.shape[1])
    candidates = []

    for cluster in [0, 1]:
        region = (binary_map == cluster)
        area = region.mean()
        if area < 0.01 or area > 0.98:
            score = -1e6
        else:
            brightness = float(gray[region].mean()) / 255.0
            center_score = float(prior[region].mean())
            area_score = 1.0 - abs(area - 0.35)
            score = (1.0 - brightness) + 0.4 * center_score + 0.2 * area_score
        candidates.append(score)

    lesion_cluster = int(np.argmax(candidates))
    return (binary_map == lesion_cluster).astype(np.float32)


def _postprocess(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return mask
    m = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (m > 127).astype(np.float32)


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 0.5
    b_bin = b > 0.5
    inter = float(np.logical_and(a_bin, b_bin).sum())
    union = float(np.logical_or(a_bin, b_bin).sum())
    if union <= 0.0:
        return 0.0
    return inter / union


def _mean_iou_to_set(mask: np.ndarray, others: List[np.ndarray]) -> float:
    if not others:
        return 0.0
    vals = [_binary_iou(mask, o) for o in others]
    return float(np.mean(vals))


def _resolve_sam_checkpoint(
    model_type: str,
    checkpoint: str | Path | None = None,
) -> Path:
    model_type = str(model_type).lower()
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    if model_type not in urls:
        raise ValueError(f"Unsupported SAM model type: {model_type}")

    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")
        return ckpt_path

    cache_dir = Path.home() / ".cache" / "rpi_net_sam"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = urls[model_type].split("/")[-1]
    ckpt_path = cache_dir / filename
    if not ckpt_path.exists():
        torch.hub.download_url_to_file(urls[model_type], str(ckpt_path), progress=True)
    return ckpt_path


class MultiSourcePseudoLabelGenerator:
    def __init__(
        self,
        image_size: int = 224,
        device: str = "cuda",
        kmeans_clusters: int = 2,
        kmeans_threads: int = 4,
        consensus_blend: float = 0.2,
        use_sam: bool = False,
        sam_model_type: str = "vit_b",
        sam_checkpoint: str | Path | None = None,
        sam_points_per_side: int = 16,
        sam_pred_iou_thresh: float = 0.86,
        sam_stability_score_thresh: float = 0.92,
        sam_min_mask_region_area: int = 32,
        sam_p2_blend: float = 0.35,  # backward compatible, unused in replacement mode
        sam_refine_blend: float = 0.2,  # backward compatible, unused in replacement mode
        sam_min_iou_with_legacy: float = 0.10,  # backward compatible, unused in replacement mode
        sam_replace_margin: float = 0.02,
    ) -> None:
        self.image_size = image_size
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_threads = max(1, int(kmeans_threads))
        self.consensus_blend = float(max(0.0, min(0.5, consensus_blend)))
        self.use_sam = bool(use_sam)
        self.sam_replace_margin = float(max(-1.0, min(1.0, sam_replace_margin)))

        self.res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.res50_feat = create_feature_extractor(
            self.res50,
            return_nodes={
                "layer1": "feat",
            },
        )
        self.res18_feat = create_feature_extractor(
            self.res18,
            return_nodes={
                "layer1": "feat",
            },
        )

        self.res50 = self.res50.to(self.device).eval()
        self.res18 = self.res18.to(self.device).eval()
        self.vit = self.vit.to(self.device).eval()
        self.res50_feat = self.res50_feat.to(self.device).eval()
        self.res18_feat = self.res18_feat.to(self.device).eval()

        self._cam_acts = None
        self._cam_grads = None
        layer4 = self.res50.layer4
        layer4.register_forward_hook(self._forward_hook)
        layer4.register_full_backward_hook(self._backward_hook)

        self.sam_mask_generator = None
        if self.use_sam:
            if SamAutomaticMaskGenerator is None or sam_model_registry is None:
                raise ImportError(
                    "use_sam=True requires segment_anything; install it first."
                )
            ckpt = _resolve_sam_checkpoint(model_type=sam_model_type, checkpoint=sam_checkpoint)
            sam_model = sam_model_registry[sam_model_type](checkpoint=str(ckpt))
            sam_model = sam_model.to(device=self.device).eval()
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                model=sam_model,
                points_per_side=max(4, int(sam_points_per_side)),
                pred_iou_thresh=float(sam_pred_iou_thresh),
                stability_score_thresh=float(sam_stability_score_thresh),
                min_mask_region_area=max(0, int(sam_min_mask_region_area)),
            )

    def _forward_hook(self, module, inp, out):
        self._cam_acts = out

    def _backward_hook(self, module, grad_input, grad_output):
        self._cam_grads = grad_output[0]

    def _load_images(self, image_paths: Sequence[Path]) -> tuple[torch.Tensor, List[np.ndarray]]:
        tensors = []
        arrays = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            arr = pil_to_numpy_rgb(image)
            arr = remove_hair_artifacts(arr)
            image = Image.fromarray(arr).resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.array(image, dtype=np.uint8)
            arrays.append(arr)
            tensors.append(TF.to_tensor(image))

        x = torch.stack(tensors, dim=0)
        return x, arrays

    def _kmeans_mask(self, feat_map: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
        c, h, w = feat_map.shape
        pixels = feat_map.reshape(c, -1).T

        km = MiniBatchKMeans(
            n_clusters=self.kmeans_clusters,
            random_state=42,
            n_init=5,
            batch_size=min(2048, pixels.shape[0]),
        )
        if threadpool_limits is not None:
            with threadpool_limits(limits=1):
                labels = km.fit_predict(pixels).reshape(h, w)
        else:
            labels = km.fit_predict(pixels).reshape(h, w)

        if cv2 is not None:
            image_small = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            image_small = np.array(
                Image.fromarray(image_rgb).resize((w, h), Image.BILINEAR),
                dtype=np.uint8,
            )

        lesion = _select_lesion_cluster(labels, image_small)
        if cv2 is not None:
            lesion = cv2.resize(
                lesion.astype(np.float32),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            lesion = np.array(
                Image.fromarray((lesion * 255).astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                ),
                dtype=np.float32,
            ) / 255.0
        return _postprocess(lesion)

    def _cam_masks_batch(self, x_norm: torch.Tensor) -> List[np.ndarray]:
        self.res50.zero_grad(set_to_none=True)
        self._cam_acts = None
        self._cam_grads = None

        logits = self.res50(x_norm)
        cls = logits.argmax(dim=1, keepdim=True)
        score = logits.gather(1, cls).sum()
        score.backward()

        if self._cam_acts is None or self._cam_grads is None:
            raise RuntimeError("CAM hooks did not capture activations/gradients")

        acts = self._cam_acts.detach()
        grads = self._cam_grads.detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        cam_np = cam[:, 0].detach().cpu().numpy()

        masks: List[np.ndarray] = []
        for c in cam_np:
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            if cv2 is not None:
                thr, _ = cv2.threshold(
                    (c * 255).astype(np.uint8),
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                mask = (c * 255 >= thr).astype(np.float32)
            else:
                mask = (c >= 0.5).astype(np.float32)
            masks.append(_postprocess(mask))
        return masks

    def _select_sam_lesion_mask(
        self,
        sam_masks: List[Dict],
        image_rgb: np.ndarray,
    ) -> np.ndarray | None:
        if not sam_masks:
            return None
        h, w = image_rgb.shape[:2]
        gray = image_rgb.mean(axis=2)
        prior = _center_prior(h, w)
        hw = float(max(1, h * w))

        best_score = -1e9
        best_mask = None
        for item in sam_masks:
            seg = item.get("segmentation", None)
            if seg is None:
                continue
            seg = np.asarray(seg, dtype=bool)
            if seg.shape != (h, w):
                continue

            area = float(seg.mean())
            if area < 0.01 or area > 0.95:
                continue
            if seg.sum() <= 0:
                continue

            brightness = float(gray[seg].mean()) / 255.0
            center_score = float(prior[seg].mean())
            pred_iou = float(item.get("predicted_iou", 0.5))
            stability = float(item.get("stability_score", 0.5))
            area_score = 1.0 - abs(area - 0.30)

            # Prefer darker, centered, moderate-size masks with good SAM quality.
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

    def _sam_masks_batch(self, image_rgbs: List[np.ndarray]) -> List[np.ndarray | None]:
        if self.sam_mask_generator is None:
            return [None for _ in image_rgbs]
        out: List[np.ndarray | None] = []
        for arr in image_rgbs:
            try:
                masks = self.sam_mask_generator.generate(arr)
                out.append(self._select_sam_lesion_mask(masks, arr))
            except Exception:
                out.append(None)
        return out

    def _vit_patch_features(self, x_norm: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(x_norm)
        b = x.shape[0]
        cls_token = self.vit.class_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.encoder(x)
        patch = x[:, 1:, :]
        side = int(np.sqrt(patch.shape[1]))
        patch = patch.transpose(1, 2).reshape(b, patch.shape[2], side, side)
        return patch

    def _kmeans_triplet_batch(
        self,
        r50_feat: np.ndarray,
        r18_feat: np.ndarray,
        vit_feat: np.ndarray,
        image_rgbs: List[np.ndarray],
    ) -> List[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        def _single(idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            p1 = self._kmeans_mask(r50_feat[idx], image_rgbs[idx])
            p3 = self._kmeans_mask(r18_feat[idx], image_rgbs[idx])
            p4 = self._kmeans_mask(vit_feat[idx], image_rgbs[idx])
            return p1, p3, p4

        workers = min(self.kmeans_threads, len(image_rgbs))
        if workers <= 1:
            return [_single(i) for i in range(len(image_rgbs))]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_single, range(len(image_rgbs))))

    def generate_for_paths(self, image_paths: Sequence[Path]) -> List[Dict[str, np.ndarray]]:
        if not image_paths:
            return []

        x, image_rgbs = self._load_images(image_paths)
        x = x.to(self.device, non_blocking=True)
        x_norm = _imagenet_normalize(x)

        with torch.no_grad():
            r50_feats = self.res50_feat(x_norm)["feat"]
            r18_feats = self.res18_feat(x_norm)["feat"]
            vit_feats = self._vit_patch_features(x_norm)

        p2_batch = self._cam_masks_batch(x_norm)
        sam_batch = self._sam_masks_batch(image_rgbs) if self.use_sam else [None for _ in image_rgbs]

        triplets = self._kmeans_triplet_batch(
            r50_feats.detach().cpu().numpy(),
            r18_feats.detach().cpu().numpy(),
            vit_feats.detach().cpu().numpy(),
            image_rgbs,
        )

        outputs: List[Dict[str, np.ndarray]] = []
        for i in range(len(image_paths)):
            p1, p3, p4 = triplets[i]
            p2 = p2_batch[i]
            sam_mask = sam_batch[i]
            if sam_mask is not None:
                src = [p1, p2, p3, p4]
                legacy_scores = []
                for j in range(4):
                    others = [src[k] for k in range(4) if k != j]
                    legacy_scores.append(_mean_iou_to_set(src[j], others))
                worst_idx = int(np.argmin(np.asarray(legacy_scores)))
                sam_score = _mean_iou_to_set(sam_mask, src)
                if sam_score >= legacy_scores[worst_idx] + self.sam_replace_margin:
                    src[worst_idx] = sam_mask
                    p1, p2, p3, p4 = src
            if self.consensus_blend > 0.0:
                alpha = self.consensus_blend
                consensus = (p1 + p2 + p3 + p4) / 4.0
                p1 = _postprocess((1.0 - alpha) * p1 + alpha * consensus)
                p2 = _postprocess((1.0 - alpha) * p2 + alpha * consensus)
                p3 = _postprocess((1.0 - alpha) * p3 + alpha * consensus)
                p4 = _postprocess((1.0 - alpha) * p4 + alpha * consensus)
            masks = np.stack([p1, p2, p3, p4], axis=0).astype(np.float32)
            outputs.append({"masks": masks})
        return outputs

    def generate_for_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        return self.generate_for_paths([image_path])[0]


def list_split_images(
    dataset_root: str | Path,
    split: str,
    ph2_split_seed: int = 42,
    ph2_split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> List[Path]:
    return list_split_image_paths(
        dataset_root,
        split,
        ph2_split_seed=ph2_split_seed,
        ph2_split_ratios=ph2_split_ratios,
    )
