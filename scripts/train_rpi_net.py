#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset, ISIC2017TensorCacheDataset
from rpi_net.losses import (
    background_supervision_loss,
    boundary_alignment_loss,
    boundary_aware_focal_tversky_loss,
    boundary_recall_consistency_loss,
    boundary_supervision_loss,
    calibration_sparsity_loss,
    dual_head_consistency_loss,
    dynamic_uncertainty_loss,
    far_background_preservation_loss,
    low_threshold_background_loss,
    pseudo_consensus_tversky_loss,
    rabc_delta_direction_loss,
    recall_candidate_supervision_loss,
    supervised_segmentation_tversky_loss,
    uncertainty_supervision_loss,
)
from rpi_net.metrics import compute_batch_metrics
from rpi_net.model import RPINet
from rpi_net.pseudo_labels import MultiSourcePseudoLabelGenerator, list_split_images
from rpi_net.utils import AverageMeter, ensure_dir, get_device, load_config, save_json, set_seed


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def get_model_state(module: torch.nn.Module) -> Dict:
    return unwrap_model(module).state_dict()


def load_model_state(module: torch.nn.Module, state_dict: Dict, strict: bool = True):
    m = unwrap_model(module)
    if strict:
        info = m.load_state_dict(state_dict, strict=True)
        return {
            "missing_keys": list(info.missing_keys),
            "unexpected_keys": list(info.unexpected_keys),
            "skipped_mismatch": [],
        }

    own = m.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in own and own[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    info = m.load_state_dict(filtered, strict=False)
    return {
        "missing_keys": list(info.missing_keys),
        "unexpected_keys": list(info.unexpected_keys),
        "skipped_mismatch": skipped,
    }


def _name_matches_prefixes(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(f"{p}.") for p in prefixes)


def apply_trainable_prefix_freeze(model: torch.nn.Module, trainable_prefixes: tuple[str, ...]) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for name, p in model.named_parameters():
        requires = _name_matches_prefixes(name, trainable_prefixes)
        p.requires_grad_(requires)
        if requires:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return trainable, frozen


def model_kwargs_from_cfg(cfg: Dict) -> Dict:
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
        "rabc_mixer_mode": str(model_cfg.get("rabc_mixer_mode", "mean")),
        "rabc_candidate_intercept": float(model_cfg.get("rabc_candidate_intercept", 0.35)),
        "rabc_candidate_uncertainty": float(model_cfg.get("rabc_candidate_uncertainty", 0.65)),
        "use_brr": bool(model_cfg.get("use_brr", False)),
        "use_focal_context": bool(model_cfg.get("use_focal_context", False)),
        "use_cga_fusion": bool(model_cfg.get("use_cga_fusion", False)),
    }


def ph2_split_args(data_cfg: Dict) -> Dict:
    ratios = data_cfg.get("ph2_split_ratios", [0.7, 0.15, 0.15])
    if not isinstance(ratios, (list, tuple)) or len(ratios) != 3:
        ratios = [0.7, 0.15, 0.15]
    return {
        "ph2_split_seed": int(data_cfg.get("ph2_split_seed", 42)),
        "ph2_split_ratios": (float(ratios[0]), float(ratios[1]), float(ratios[2])),
    }


def configure_runtime(cfg: Dict, device: torch.device) -> None:
    runtime_cfg = cfg.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if bool(runtime_cfg.get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")


@torch.no_grad()
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    teacher_module = unwrap_model(teacher)
    student_module = unwrap_model(student)
    for t_param, s_param in zip(teacher_module.parameters(), student_module.parameters()):
        t_param.mul_(decay).add_(s_param, alpha=1.0 - decay)
    for t_buf, s_buf in zip(teacher_module.buffers(), student_module.buffers()):
        t_buf.copy_(s_buf)


def rampup_weight(base_weight: float, epoch_idx: int, rampup_epochs: int) -> float:
    if rampup_epochs <= 0:
        return base_weight
    p = min(1.0, max(0.0, float(epoch_idx + 1) / float(rampup_epochs)))
    # standard smooth ramp-up
    return float(base_weight * math.exp(-5.0 * (1.0 - p) * (1.0 - p)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RPI-Net on ISIC2017")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--skip-pseudo-gen", action="store_true")
    return parser.parse_args()


def ensure_pseudo_labels(cfg: Dict, split: str = "train") -> Path:
    pseudo_root = Path(cfg["pseudo"]["output_dir"])
    pseudo_dir = pseudo_root / split
    ensure_dir(pseudo_dir)

    data_cfg = cfg["data"]
    images = list_split_images(
        data_cfg["root"],
        split,
        **ph2_split_args(data_cfg),
    )
    if split == cfg["data"]["train_split"]:
        max_samples = cfg["data"].get("max_train_samples")
        if max_samples is not None:
            images = images[: int(max_samples)]
    missing = [p for p in images if not (pseudo_dir / f"{p.stem}.npz").exists()]

    if not missing and not cfg["pseudo"].get("force_regenerate", False):
        print(f"Pseudo labels already ready: {pseudo_dir}")
        return pseudo_dir

    print(f"Generating pseudo labels for split={split}, missing={len(missing)}")
    generator = MultiSourcePseudoLabelGenerator(
        image_size=int(cfg["data"]["image_size"]),
        device=cfg["runtime"].get("device", "cuda"),
        kmeans_clusters=int(cfg["pseudo"].get("kmeans_clusters", 2)),
        kmeans_threads=int(cfg["pseudo"].get("kmeans_threads", 4)),
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

    targets = missing
    if cfg["pseudo"].get("force_regenerate", False):
        targets = images

    batch_size = int(cfg["pseudo"].get("batch_size", 8))
    pbar = tqdm(total=len(targets), desc=f"pseudo-{split}")
    for start in range(0, len(targets), batch_size):
        batch_paths = targets[start : start + batch_size]
        pseudos = generator.generate_for_paths(batch_paths)
        for image_path, pseudo in zip(batch_paths, pseudos):
            out_path = pseudo_dir / f"{image_path.stem}.npz"
            np.savez_compressed(out_path, masks=pseudo["masks"])
        pbar.update(len(batch_paths))
    pbar.close()

    return pseudo_dir


def make_dataloaders(cfg: Dict, pseudo_dir: Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    ph2_kwargs = ph2_split_args(data_cfg)
    use_tensor_cache = bool(data_cfg.get("use_tensor_cache", False))
    tensor_cache_dir = Path(data_cfg.get("tensor_cache_dir", "")) if use_tensor_cache else None

    if use_tensor_cache:
        if tensor_cache_dir is None:
            raise ValueError("use_tensor_cache=True requires data.tensor_cache_dir")
        train_ds = ISIC2017TensorCacheDataset(
            cache_file=tensor_cache_dir / "train_cache.pt",
            require_pseudo=True,
            augment=True,
            max_samples=data_cfg.get("max_train_samples"),
        )
        val_ds = ISIC2017TensorCacheDataset(
            cache_file=tensor_cache_dir / "val_cache.pt",
            require_pseudo=False,
            augment=False,
            max_samples=data_cfg.get("max_val_samples"),
        )
        test_ds = ISIC2017TensorCacheDataset(
            cache_file=tensor_cache_dir / "test_cache.pt",
            require_pseudo=False,
            augment=False,
            max_samples=data_cfg.get("max_test_samples"),
        )
    else:
        cache_images = bool(data_cfg.get("cache_images", False))
        cache_workers = int(data_cfg.get("cache_workers", 4))
        use_hair_removal = bool(data_cfg.get("use_hair_removal", True))
        train_ds = ISIC2017Dataset(
            root=data_cfg["root"],
            split=data_cfg["train_split"],
            image_size=data_cfg["image_size"],
            pseudo_dir=pseudo_dir,
            require_pseudo=True,
            augment=True,
            use_hair_removal=use_hair_removal,
            cache_images=cache_images,
            cache_workers=cache_workers,
            max_samples=data_cfg.get("max_train_samples"),
            **ph2_kwargs,
        )
        val_ds = ISIC2017Dataset(
            root=data_cfg["root"],
            split=data_cfg["val_split"],
            image_size=data_cfg["image_size"],
            pseudo_dir=None,
            require_pseudo=False,
            augment=False,
            use_hair_removal=use_hair_removal,
            cache_images=cache_images,
            cache_workers=cache_workers,
            max_samples=data_cfg.get("max_val_samples"),
            **ph2_kwargs,
        )
        test_ds = ISIC2017Dataset(
            root=data_cfg["root"],
            split=data_cfg["test_split"],
            image_size=data_cfg["image_size"],
            pseudo_dir=None,
            require_pseudo=False,
            augment=False,
            use_hair_removal=use_hair_removal,
            cache_images=cache_images,
            cache_workers=cache_workers,
            max_samples=data_cfg.get("max_test_samples"),
            **ph2_kwargs,
        )

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))
    persistent_workers = bool(data_cfg.get("persistent_workers", True))
    pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype | None,
    grad_clip: float,
    channels_last: bool,
    teacher_model: torch.nn.Module | None,
    source_teacher_model: torch.nn.Module | None,
    consistency_weight: float,
    consistency_rampup_epochs: int,
    source_distill_weight: float,
    source_distill_rampup_epochs: int,
    ema_decay: float,
    epoch_idx: int,
    agreement_gamma: float,
    pseudo_consensus_weight: float,
    boundary_weight: float,
    boundary_head_weight: float,
    background_supervision_weight: float,
    low_threshold_bg_weight: float,
    low_threshold_bg_threshold: float,
    uncertainty_supervision_weight: float,
    recall_candidate_weight: float,
    recall_candidate_dilate_kernel: int,
    recall_candidate_pos_weight: float,
    multiscale_recall_weight: float,
    multiscale_recall_dilate_kernel: int,
    multiscale_recall_pos_weight: float,
    rabc_boundary_weight: float,
    rabc_far_bg_weight: float,
    rabc_sparsity_weight: float,
    rabc_delta_direction_weight: float,
    rabc_delta_direction_margin: float,
    rabc_delta_direction_fp_weight: float,
    brr_candidate_weight: float,
    brr_boundary_weight: float,
    brr_far_bg_weight: float,
    brr_sparsity_weight: float,
    aux_consensus_weight: float,
    detail_consensus_weight: float,
    dual_head_consistency_weight: float,
    supervised_weight: float,
    labeled_ratio: float,
    pseudo_tversky_weight: float,
    pseudo_tversky_alpha: float,
    pseudo_tversky_beta: float,
    pseudo_tversky_gamma: float,
    supervised_tversky_weight: float,
    supervised_tversky_alpha: float,
    supervised_tversky_beta: float,
    supervised_tversky_gamma: float,
    boundary_recall_weight: float,
    boundary_recall_alpha: float,
    boundary_recall_beta: float,
    boundary_recall_gamma: float,
    boundary_recall_boost: float,
    bn_trainable_prefixes: tuple[str, ...] | None,
) -> Dict[str, float]:
    model.train()
    if bn_trainable_prefixes is not None:
        # Keep frozen-path BatchNorm stats fixed even in train mode.
        for module_name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                if not _name_matches_prefixes(module_name, bn_trainable_prefixes):
                    module.eval()
    if teacher_model is not None:
        teacher_model.eval()

    loss_meter = AverageMeter()
    dul_loss_meter = AverageMeter()
    consistency_loss_meter = AverageMeter()
    source_distill_loss_meter = AverageMeter()
    pseudo_consensus_meter = AverageMeter()
    boundary_loss_meter = AverageMeter()
    boundary_recall_loss_meter = AverageMeter()
    boundary_head_loss_meter = AverageMeter()
    background_sup_loss_meter = AverageMeter()
    low_threshold_bg_loss_meter = AverageMeter()
    uncertainty_sup_loss_meter = AverageMeter()
    recall_candidate_loss_meter = AverageMeter()
    multiscale_recall_loss_meter = AverageMeter()
    rabc_boundary_loss_meter = AverageMeter()
    rabc_far_bg_loss_meter = AverageMeter()
    rabc_sparsity_loss_meter = AverageMeter()
    rabc_delta_direction_loss_meter = AverageMeter()
    brr_candidate_loss_meter = AverageMeter()
    brr_boundary_loss_meter = AverageMeter()
    brr_far_bg_loss_meter = AverageMeter()
    brr_sparsity_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    detail_loss_meter = AverageMeter()
    dual_head_consistency_meter = AverageMeter()
    supervised_loss_meter = AverageMeter()
    agreement_meter = AverageMeter()
    current_consistency_weight = rampup_weight(consistency_weight, epoch_idx, consistency_rampup_epochs)
    current_source_distill_weight = rampup_weight(
        source_distill_weight,
        epoch_idx,
        source_distill_rampup_epochs,
    )

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        pseudo = batch["pseudo"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
            outputs = model(images, pseudo)
            logits = outputs["logits"]
            train_logits = outputs.get("train_logits")
            if not isinstance(train_logits, torch.Tensor):
                train_logits = logits
            sigma_maps = outputs["sigma_maps"]
            dul_loss, dul_log = dynamic_uncertainty_loss(
                train_logits,
                pseudo,
                sigma_maps,
                agreement_gamma=agreement_gamma,
            )
            total_loss = dul_loss

            consistency_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if teacher_model is not None and consistency_weight > 0.0:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images, None)
                    teacher_train_logits = teacher_outputs.get("train_logits")
                    if not isinstance(teacher_train_logits, torch.Tensor):
                        teacher_train_logits = teacher_outputs["logits"]
                student_prob = torch.sigmoid(train_logits)
                teacher_prob = torch.sigmoid(teacher_train_logits.detach())
                consistency_loss = F.mse_loss(student_prob, teacher_prob)
                total_loss = total_loss + current_consistency_weight * consistency_loss

            source_distill_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if source_teacher_model is not None and source_distill_weight > 0.0:
                with torch.no_grad():
                    source_teacher_outputs = source_teacher_model(images, None)
                    source_teacher_train_logits = source_teacher_outputs.get("train_logits")
                    if not isinstance(source_teacher_train_logits, torch.Tensor):
                        source_teacher_train_logits = source_teacher_outputs["logits"]
                source_teacher_prob = torch.sigmoid(source_teacher_train_logits.detach())
                source_distill_loss = F.binary_cross_entropy_with_logits(train_logits, source_teacher_prob)
                total_loss = total_loss + current_source_distill_weight * source_distill_loss

            pseudo_c_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if pseudo_consensus_weight > 0.0:
                pseudo_c_loss = pseudo_consensus_tversky_loss(
                    train_logits,
                    pseudo,
                    tversky_weight=pseudo_tversky_weight,
                    tversky_alpha=pseudo_tversky_alpha,
                    tversky_beta=pseudo_tversky_beta,
                    tversky_gamma=pseudo_tversky_gamma,
                )
                total_loss = total_loss + pseudo_consensus_weight * pseudo_c_loss

            boundary_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if boundary_weight > 0.0:
                boundary_loss = boundary_alignment_loss(train_logits, pseudo)
                total_loss = total_loss + boundary_weight * boundary_loss

            boundary_recall_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if boundary_recall_weight > 0.0:
                boundary_recall_loss = boundary_aware_focal_tversky_loss(
                    train_logits, pseudo,
                    alpha=boundary_recall_alpha,
                    beta=boundary_recall_beta,
                    gamma=boundary_recall_gamma,
                    boundary_boost=boundary_recall_boost,
                )
                total_loss = total_loss + boundary_recall_weight * boundary_recall_loss

            boundary_head_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if boundary_head_weight > 0.0:
                boundary_logits = outputs.get("boundary_logits")
                if isinstance(boundary_logits, torch.Tensor):
                    boundary_head_loss = boundary_supervision_loss(boundary_logits, pseudo)
                    total_loss = total_loss + boundary_head_weight * boundary_head_loss

            background_sup_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if background_supervision_weight > 0.0:
                bg_logits = outputs.get("bg_logits")
                if isinstance(bg_logits, torch.Tensor):
                    background_sup_loss = background_supervision_loss(bg_logits, pseudo)
                    total_loss = total_loss + background_supervision_weight * background_sup_loss

            low_bg_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if low_threshold_bg_weight > 0.0:
                low_bg_loss = low_threshold_background_loss(
                    train_logits,
                    pseudo,
                    threshold=low_threshold_bg_threshold,
                )
                total_loss = total_loss + low_threshold_bg_weight * low_bg_loss

            uncertainty_sup_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if uncertainty_supervision_weight > 0.0:
                uncertainty_logits = outputs.get("uncertainty_logits")
                if isinstance(uncertainty_logits, torch.Tensor):
                    uncertainty_sup_loss = uncertainty_supervision_loss(uncertainty_logits, pseudo)
                    total_loss = total_loss + uncertainty_supervision_weight * uncertainty_sup_loss

            recall_candidate_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if recall_candidate_weight > 0.0:
                recall_candidate_logits = outputs.get("recall_candidate_logits")
                if isinstance(recall_candidate_logits, torch.Tensor):
                    recall_candidate_loss = recall_candidate_supervision_loss(
                        recall_candidate_logits,
                        pseudo,
                        dilate_kernel=recall_candidate_dilate_kernel,
                        pos_weight=recall_candidate_pos_weight,
                    )
                    total_loss = total_loss + recall_candidate_weight * recall_candidate_loss

            multiscale_recall_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if multiscale_recall_weight > 0.0:
                multiscale_recall_logits = outputs.get("multiscale_recall_logits")
                if isinstance(multiscale_recall_logits, torch.Tensor):
                    multiscale_recall_loss = recall_candidate_supervision_loss(
                        multiscale_recall_logits,
                        pseudo,
                        dilate_kernel=multiscale_recall_dilate_kernel,
                        pos_weight=multiscale_recall_pos_weight,
                    )
                    total_loss = total_loss + multiscale_recall_weight * multiscale_recall_loss

            rabc_boundary_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            rabc_far_bg_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            rabc_sparsity_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            rabc_delta_direction = torch.zeros((), device=device, dtype=train_logits.dtype)
            rabc_delta = outputs.get("rabc_delta")
            if isinstance(rabc_delta, torch.Tensor):
                base_train_logits = outputs.get("base_train_logits")
                if not isinstance(base_train_logits, torch.Tensor):
                    base_train_logits = train_logits.detach()
                boundary_logits = outputs.get("boundary_logits")
                uncertainty_logits = outputs.get("uncertainty_logits")
                if rabc_boundary_weight > 0.0:
                    rabc_boundary_loss = boundary_recall_consistency_loss(
                        train_logits,
                        pseudo,
                        base_logits=base_train_logits,
                        boundary_logits=boundary_logits if isinstance(boundary_logits, torch.Tensor) else None,
                        uncertainty_logits=uncertainty_logits if isinstance(uncertainty_logits, torch.Tensor) else None,
                        focus_boost=2.0,
                    )
                    total_loss = total_loss + rabc_boundary_weight * rabc_boundary_loss
                if rabc_delta_direction_weight > 0.0:
                    rabc_delta_direction = rabc_delta_direction_loss(
                        rabc_delta,
                        pseudo,
                        base_logits=base_train_logits,
                        boundary_logits=boundary_logits if isinstance(boundary_logits, torch.Tensor) else None,
                        uncertainty_logits=uncertainty_logits if isinstance(uncertainty_logits, torch.Tensor) else None,
                        margin=rabc_delta_direction_margin,
                        boundary_kernel=5,
                        fp_weight=rabc_delta_direction_fp_weight,
                    )
                    total_loss = total_loss + rabc_delta_direction_weight * rabc_delta_direction
                if rabc_far_bg_weight > 0.0:
                    rabc_far_bg_loss = far_background_preservation_loss(
                        train_logits,
                        pseudo,
                        base_logits=base_train_logits,
                        margin=0.02,
                        band_kernel=5,
                    )
                    total_loss = total_loss + rabc_far_bg_weight * rabc_far_bg_loss
                if rabc_sparsity_weight > 0.0:
                    support = outputs.get("rabc_alpha_map")
                    rabc_sparsity_loss = calibration_sparsity_loss(
                        rabc_delta,
                        support_map=support if isinstance(support, torch.Tensor) else None,
                        outside_weight=1.0,
                    )
                    total_loss = total_loss + rabc_sparsity_weight * rabc_sparsity_loss

            brr_candidate_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            brr_boundary_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            brr_far_bg_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            brr_sparsity_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            brr_candidate_logits = outputs.get("brr_candidate_logits")
            brr_delta = outputs.get("brr_delta")
            brr_support_map = outputs.get("brr_support_map")
            if isinstance(brr_delta, torch.Tensor):
                base_train_logits = outputs.get("base_train_logits")
                if not isinstance(base_train_logits, torch.Tensor):
                    base_train_logits = train_logits.detach()
                boundary_logits = outputs.get("boundary_logits")
                uncertainty_logits = outputs.get("uncertainty_logits")
                if brr_candidate_weight > 0.0 and isinstance(brr_candidate_logits, torch.Tensor):
                    brr_candidate_loss = recall_candidate_supervision_loss(
                        brr_candidate_logits,
                        pseudo,
                        dilate_kernel=max(3, recall_candidate_dilate_kernel),
                        pos_weight=max(1.0, recall_candidate_pos_weight),
                    )
                    total_loss = total_loss + brr_candidate_weight * brr_candidate_loss
                if brr_boundary_weight > 0.0:
                    brr_boundary_loss = boundary_recall_consistency_loss(
                        train_logits,
                        pseudo,
                        base_logits=base_train_logits,
                        boundary_logits=boundary_logits if isinstance(boundary_logits, torch.Tensor) else None,
                        uncertainty_logits=uncertainty_logits if isinstance(uncertainty_logits, torch.Tensor) else None,
                        focus_boost=2.5,
                    )
                    total_loss = total_loss + brr_boundary_weight * brr_boundary_loss
                if brr_far_bg_weight > 0.0:
                    brr_far_bg_loss = far_background_preservation_loss(
                        train_logits,
                        pseudo,
                        base_logits=base_train_logits,
                        margin=0.015,
                        band_kernel=5,
                    )
                    total_loss = total_loss + brr_far_bg_weight * brr_far_bg_loss
                if brr_sparsity_weight > 0.0:
                    brr_sparsity_loss = calibration_sparsity_loss(
                        brr_delta,
                        support_map=brr_support_map if isinstance(brr_support_map, torch.Tensor) else None,
                        outside_weight=1.0,
                    )
                    total_loss = total_loss + brr_sparsity_weight * brr_sparsity_loss

            aux_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if aux_consensus_weight > 0.0:
                aux_logits = outputs.get("aux_logits")
                if isinstance(aux_logits, torch.Tensor):
                    aux_loss = pseudo_consensus_tversky_loss(
                        aux_logits,
                        pseudo,
                        tversky_weight=pseudo_tversky_weight,
                        tversky_alpha=pseudo_tversky_alpha,
                        tversky_beta=pseudo_tversky_beta,
                        tversky_gamma=pseudo_tversky_gamma,
                    )
                    total_loss = total_loss + aux_consensus_weight * aux_loss

            detail_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if detail_consensus_weight > 0.0:
                detail_logits = outputs.get("detail_logits")
                if isinstance(detail_logits, torch.Tensor):
                    detail_loss = pseudo_consensus_tversky_loss(
                        detail_logits,
                        pseudo,
                        tversky_weight=pseudo_tversky_weight,
                        tversky_alpha=pseudo_tversky_alpha,
                        tversky_beta=pseudo_tversky_beta,
                        tversky_gamma=pseudo_tversky_gamma,
                    )
                    total_loss = total_loss + detail_consensus_weight * detail_loss

            dual_head_c_loss = torch.zeros((), device=device, dtype=train_logits.dtype)
            if dual_head_consistency_weight > 0.0:
                sup_logits = outputs.get("supervised_logits")
                if isinstance(sup_logits, torch.Tensor):
                    uncertainty_logits = outputs.get("uncertainty_logits")
                    if not isinstance(uncertainty_logits, torch.Tensor):
                        uncertainty_logits = None
                    dual_head_c_loss = dual_head_consistency_loss(
                        train_logits,
                        sup_logits,
                        pseudo,
                        uncertainty_logits=uncertainty_logits,
                        min_weight=0.05,
                    )
                    total_loss = total_loss + dual_head_consistency_weight * dual_head_c_loss

            supervised_loss = torch.zeros((), device=device, dtype=logits.dtype)
            if supervised_weight > 0.0 and labeled_ratio > 0.0:
                if labeled_ratio >= 1.0:
                    sel = torch.ones(images.size(0), device=device, dtype=torch.bool)
                else:
                    sel = torch.rand(images.size(0), device=device) < labeled_ratio
                if bool(sel.any()):
                    sup_logits = outputs.get("supervised_logits")
                    if isinstance(sup_logits, torch.Tensor):
                        supervised_main = supervised_segmentation_tversky_loss(
                            logits[sel],
                            masks[sel],
                            tversky_weight=supervised_tversky_weight,
                            tversky_alpha=supervised_tversky_alpha,
                            tversky_beta=supervised_tversky_beta,
                            tversky_gamma=supervised_tversky_gamma,
                        )
                        supervised_aux = supervised_segmentation_tversky_loss(
                            sup_logits[sel],
                            masks[sel],
                            tversky_weight=supervised_tversky_weight,
                            tversky_alpha=supervised_tversky_alpha,
                            tversky_beta=supervised_tversky_beta,
                            tversky_gamma=supervised_tversky_gamma,
                        )
                        supervised_loss = 0.5 * (supervised_main + supervised_aux)
                    else:
                        supervised_loss = supervised_segmentation_tversky_loss(
                            logits[sel],
                            masks[sel],
                            tversky_weight=supervised_tversky_weight,
                            tversky_alpha=supervised_tversky_alpha,
                            tversky_beta=supervised_tversky_beta,
                            tversky_gamma=supervised_tversky_gamma,
                        )
                    total_loss = total_loss + supervised_weight * supervised_loss

        scaler.scale(total_loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if teacher_model is not None:
            ema_update(teacher_model, model, ema_decay)

        bs = images.size(0)
        loss_meter.update(total_loss.item(), n=bs)
        dul_loss_meter.update(dul_loss.item(), n=bs)
        consistency_loss_meter.update(consistency_loss.item(), n=bs)
        source_distill_loss_meter.update(source_distill_loss.item(), n=bs)
        pseudo_consensus_meter.update(pseudo_c_loss.item(), n=bs)
        boundary_loss_meter.update(boundary_loss.item(), n=bs)
        boundary_recall_loss_meter.update(boundary_recall_loss.item(), n=bs)
        boundary_head_loss_meter.update(boundary_head_loss.item(), n=bs)
        background_sup_loss_meter.update(background_sup_loss.item(), n=bs)
        low_threshold_bg_loss_meter.update(low_bg_loss.item(), n=bs)
        uncertainty_sup_loss_meter.update(uncertainty_sup_loss.item(), n=bs)
        recall_candidate_loss_meter.update(recall_candidate_loss.item(), n=bs)
        multiscale_recall_loss_meter.update(multiscale_recall_loss.item(), n=bs)
        rabc_boundary_loss_meter.update(rabc_boundary_loss.item(), n=bs)
        rabc_far_bg_loss_meter.update(rabc_far_bg_loss.item(), n=bs)
        rabc_sparsity_loss_meter.update(rabc_sparsity_loss.item(), n=bs)
        rabc_delta_direction_loss_meter.update(rabc_delta_direction.item(), n=bs)
        brr_candidate_loss_meter.update(brr_candidate_loss.item(), n=bs)
        brr_boundary_loss_meter.update(brr_boundary_loss.item(), n=bs)
        brr_far_bg_loss_meter.update(brr_far_bg_loss.item(), n=bs)
        brr_sparsity_loss_meter.update(brr_sparsity_loss.item(), n=bs)
        aux_loss_meter.update(aux_loss.item(), n=bs)
        detail_loss_meter.update(detail_loss.item(), n=bs)
        dual_head_consistency_meter.update(dual_head_c_loss.item(), n=bs)
        supervised_loss_meter.update(supervised_loss.item(), n=bs)
        agreement_meter.update(float(dul_log["agreement_mean"]), n=bs)

    return {
        "train_loss": loss_meter.avg,
        "train_dul_loss": dul_loss_meter.avg,
        "train_consistency_loss": consistency_loss_meter.avg,
        "train_source_distill_loss": source_distill_loss_meter.avg,
        "train_pseudo_consensus_loss": pseudo_consensus_meter.avg,
        "train_boundary_loss": boundary_loss_meter.avg,
        "train_boundary_recall_loss": boundary_recall_loss_meter.avg,
        "train_boundary_head_loss": boundary_head_loss_meter.avg,
        "train_background_supervision_loss": background_sup_loss_meter.avg,
        "train_low_threshold_bg_loss": low_threshold_bg_loss_meter.avg,
        "train_uncertainty_supervision_loss": uncertainty_sup_loss_meter.avg,
        "train_recall_candidate_loss": recall_candidate_loss_meter.avg,
        "train_multiscale_recall_loss": multiscale_recall_loss_meter.avg,
        "train_rabc_boundary_loss": rabc_boundary_loss_meter.avg,
        "train_rabc_far_bg_loss": rabc_far_bg_loss_meter.avg,
        "train_rabc_sparsity_loss": rabc_sparsity_loss_meter.avg,
        "train_rabc_delta_direction_loss": rabc_delta_direction_loss_meter.avg,
        "train_brr_candidate_loss": brr_candidate_loss_meter.avg,
        "train_brr_boundary_loss": brr_boundary_loss_meter.avg,
        "train_brr_far_bg_loss": brr_far_bg_loss_meter.avg,
        "train_brr_sparsity_loss": brr_sparsity_loss_meter.avg,
        "train_aux_consensus_loss": aux_loss_meter.avg,
        "train_detail_consensus_loss": detail_loss_meter.avg,
        "train_dual_head_consistency_loss": dual_head_consistency_meter.avg,
        "train_supervised_loss": supervised_loss_meter.avg,
        "train_consistency_weight": current_consistency_weight,
        "train_source_distill_weight": current_source_distill_weight,
        "train_agreement_mean": agreement_meter.avg,
    }


@torch.inference_mode()
def evaluate(
    model: RPINet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    desc: str,
    channels_last: bool,
) -> Dict[str, float]:
    model.eval()
    meters = {k: AverageMeter() for k in ["acc", "dice", "jac", "sen", "spe"]}

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images, None)["logits"]
        metrics = compute_batch_metrics(logits, masks, threshold=threshold)
        bs = images.size(0)
        for k, v in metrics.items():
            meters[k].update(v, n=bs)

    return {k: m.avg for k, m in meters.items()}


@torch.inference_mode()
def iterative_update_pseudo_labels(
    model: torch.nn.Module,
    cfg: Dict,
    pseudo_dir: Path,
    device: torch.device,
    blend: float,
    channels_last: bool,
) -> Dict[str, float]:
    model.eval()

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    threshold = float(cfg["train"].get("threshold", 0.3))
    iterative_use_hard = bool(cfg["train"].get("iterative_use_hard", False))
    uncertainty_selective = bool(train_cfg.get("iterative_uncertainty_selective", False))
    uncertainty_percentile = float(train_cfg.get("iterative_uncertainty_percentile", 0.8))
    if uncertainty_percentile > 1.0:
        uncertainty_percentile = uncertainty_percentile / 100.0
    uncertainty_percentile = min(max(uncertainty_percentile, 0.0), 0.999)
    use_sigma_uncertainty = bool(train_cfg.get("iterative_uncertainty_from_sigma", True))
    entropy_weight = float(train_cfg.get("iterative_entropy_weight", 0.7))
    entropy_weight = min(max(entropy_weight, 0.0), 1.0)
    blend_high = float(train_cfg.get("iterative_blend_high", blend))
    blend_low = float(train_cfg.get("iterative_blend_low", 0.0))
    blend_high = min(max(blend_high, 0.0), 1.0)
    blend_low = min(max(blend_low, 0.0), 1.0)
    if blend_low > blend_high:
        blend_low, blend_high = blend_high, blend_low

    updated_ratio_sum = 0.0
    uncertainty_mean_sum = 0.0
    blend_mean_sum = 0.0
    n_batches = 0

    if bool(data_cfg.get("use_tensor_cache", False)):
        cache_dir = Path(data_cfg["tensor_cache_dir"])
        train_cache = cache_dir / "train_cache.pt"
        payload = torch.load(train_cache, map_location="cpu")

        if "pseudos" not in payload or payload["pseudos"] is None:
            raise RuntimeError(f"Tensor cache has no pseudos: {train_cache}")

        images: torch.Tensor = payload["images"]
        old_pseudos: torch.Tensor = payload["pseudos"]
        pseudos = old_pseudos.float().clone()

        n_total = images.shape[0]
        max_samples = data_cfg.get("max_train_samples")
        if max_samples is not None:
            n_total = min(n_total, int(max_samples))

        bs = int(cfg["train"]["batch_size"])
        for start in tqdm(range(0, n_total, bs), desc="iterative-update-cache", leave=False):
            end = min(start + bs, n_total)
            batch_images = images[start:end].to(device, non_blocking=True).float()
            batch_old_pseudo = pseudos[start:end].to(device, non_blocking=True)
            if channels_last:
                batch_images = batch_images.contiguous(memory_format=torch.channels_last)

            outputs = model(batch_images, batch_old_pseudo if use_sigma_uncertainty else None)
            logits = outputs["logits"]
            pred = torch.sigmoid(logits)[:, 0]
            if iterative_use_hard:
                pred = (pred > threshold).float()

            p = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
            entropy_u = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / 0.6931471805599453
            uncertainty = entropy_u
            if use_sigma_uncertainty and len(outputs["sigma_maps"]) > 0:
                sigma_u = torch.stack([torch.sigmoid(s[:, 0]) for s in outputs["sigma_maps"]], dim=1).mean(dim=1)
                uncertainty = entropy_weight * entropy_u + (1.0 - entropy_weight) * sigma_u
                uncertainty = torch.clamp(uncertainty, min=0.0, max=1.0)

            if uncertainty_selective:
                thr = torch.quantile(uncertainty.flatten(1), uncertainty_percentile, dim=1)
                high_mask = uncertainty >= thr[:, None, None]
                blend_map = torch.where(
                    high_mask,
                    torch.full_like(uncertainty, blend_high),
                    torch.full_like(uncertainty, blend_low),
                )
            else:
                high_mask = torch.ones_like(uncertainty, dtype=torch.bool)
                blend_map = torch.full_like(uncertainty, blend_high)

            new = (1.0 - blend_map[:, None]) * batch_old_pseudo + blend_map[:, None] * pred[:, None]
            pseudos[start:end] = new.cpu()

            updated_ratio_sum += float(high_mask.float().mean().item())
            uncertainty_mean_sum += float(uncertainty.mean().item())
            blend_mean_sum += float(blend_map.mean().item())
            n_batches += 1

        payload["pseudos"] = pseudos.to(torch.float16)
        tmp_cache = train_cache.with_suffix(".tmp.pt")
        torch.save(payload, tmp_cache)
        tmp_cache.replace(train_cache)
        n = max(1, n_batches)
        return {
            "updated_ratio": updated_ratio_sum / n,
            "uncertainty_mean": uncertainty_mean_sum / n,
            "blend_mean": blend_mean_sum / n,
        }

    ds = ISIC2017Dataset(
        root=data_cfg["root"],
        split=data_cfg["train_split"],
        image_size=data_cfg["image_size"],
        pseudo_dir=pseudo_dir,
        require_pseudo=True,
        augment=False,
        use_hair_removal=bool(data_cfg.get("use_hair_removal", True)),
        cache_images=bool(data_cfg.get("cache_images", False)),
        cache_workers=int(data_cfg.get("cache_workers", 4)),
        max_samples=data_cfg.get("max_train_samples"),
        **ph2_split_args(data_cfg),
    )
    n_workers = int(data_cfg.get("num_workers", 4))
    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if n_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    for batch in tqdm(loader, desc="iterative-update", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        ids = batch["id"]

        old_pseudo_t = batch["pseudo"].to(device, non_blocking=True)
        outputs = model(images, old_pseudo_t if use_sigma_uncertainty else None)
        logits = outputs["logits"]
        pred = torch.sigmoid(logits)[:, 0]
        if iterative_use_hard:
            pred = (pred > threshold).float()
        p = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        entropy_u = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / 0.6931471805599453
        uncertainty = entropy_u
        if use_sigma_uncertainty and len(outputs["sigma_maps"]) > 0:
            sigma_u = torch.stack([torch.sigmoid(s[:, 0]) for s in outputs["sigma_maps"]], dim=1).mean(dim=1)
            uncertainty = entropy_weight * entropy_u + (1.0 - entropy_weight) * sigma_u
            uncertainty = torch.clamp(uncertainty, min=0.0, max=1.0)

        if uncertainty_selective:
            thr = torch.quantile(uncertainty.flatten(1), uncertainty_percentile, dim=1)
            high_mask = uncertainty >= thr[:, None, None]
            blend_map = torch.where(
                high_mask,
                torch.full_like(uncertainty, blend_high),
                torch.full_like(uncertainty, blend_low),
            )
        else:
            high_mask = torch.ones_like(uncertainty, dtype=torch.bool)
            blend_map = torch.full_like(uncertainty, blend_high)

        pred = pred.cpu().numpy()
        blend_map = blend_map.cpu().numpy()
        high_mask = high_mask.float().cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()

        for i, image_id in enumerate(ids):
            pseudo_path = pseudo_dir / f"{image_id}.npz"
            old = np.load(pseudo_path)["masks"].astype(np.float32)  # [4, H, W]
            p = pred[i]
            bmap = blend_map[i]
            new = (1.0 - bmap[None, ...]) * old + bmap[None, ...] * p[None, ...]
            np.savez_compressed(pseudo_path, masks=new.astype(np.float32))

        updated_ratio_sum += float(high_mask.mean())
        uncertainty_mean_sum += float(uncertainty.mean())
        blend_mean_sum += float(blend_map.mean())
        n_batches += 1

    n = max(1, n_batches)
    return {
        "updated_ratio": updated_ratio_sum / n,
        "uncertainty_mean": uncertainty_mean_sum / n,
        "blend_mean": blend_mean_sum / n,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg.get("seed", 42)))

    output_dir = ensure_dir(cfg["output_dir"])
    ckpt_dir = ensure_dir(output_dir / "checkpoints")

    device = get_device(cfg["runtime"].get("device", "cuda"))
    print(f"Using device: {device}")
    configure_runtime(cfg, device)
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))

    if args.skip_pseudo_gen:
        pseudo_dir = Path(cfg["pseudo"]["output_dir"]) / cfg["data"]["train_split"]
    else:
        pseudo_dir = ensure_pseudo_labels(cfg, split=cfg["data"]["train_split"])

    train_loader, val_loader, test_loader = make_dataloaders(cfg, pseudo_dir)

    model = RPINet(**model_kwargs_from_cfg(cfg)).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if bool(cfg.get("runtime", {}).get("compile", False)):
        compile_mode = cfg.get("runtime", {}).get("compile_mode", "default")
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"torch.compile enabled, mode={compile_mode}")
        except Exception as e:
            print(f"torch.compile skipped due to: {e}")

    freeze_segmentation_path = bool(cfg["train"].get("freeze_segmentation_path", False))
    trainable_prefixes_cfg = cfg["train"].get(
        "trainable_prefixes",
        ["pseudo_encoder", "ipc_attn", "pia_attn", "sigma_heads"],
    )
    if not isinstance(trainable_prefixes_cfg, (list, tuple)):
        raise ValueError("train.trainable_prefixes must be a list/tuple of module prefixes")
    trainable_prefixes = tuple(str(x) for x in trainable_prefixes_cfg if str(x))
    freeze_bn_stats = bool(cfg["train"].get("freeze_bn_stats", freeze_segmentation_path))
    bn_trainable_prefixes: tuple[str, ...] | None = None
    if freeze_segmentation_path:
        if len(trainable_prefixes) == 0:
            raise ValueError("freeze_segmentation_path=True requires non-empty trainable_prefixes")
        t_cnt, f_cnt = apply_trainable_prefix_freeze(model, trainable_prefixes)
        bn_trainable_prefixes = trainable_prefixes if freeze_bn_stats else None
        print(
            "Applied freeze_segmentation_path: "
            f"trainable_params={t_cnt}, frozen_params={f_cnt}, "
            f"freeze_bn_stats={freeze_bn_stats}"
        )
    elif freeze_bn_stats:
        # A fully trainable adaptation can still keep running statistics fixed
        # for a small target domain. Affine BN parameters remain trainable.
        bn_trainable_prefixes = ()
        print("Applied freeze_bn_stats to all BatchNorm running statistics")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if len(optim_params) == 0:
        raise RuntimeError("No trainable parameters for optimizer")
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    amp_dtype_name = str(cfg["train"].get("amp_dtype", "fp16")).lower()
    if amp_dtype_name in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
    elif amp_dtype_name in {"fp16", "float16", "half"}:
        amp_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported train.amp_dtype={amp_dtype_name}. Use fp16 or bf16.")

    use_grad_scaler = bool(amp_enabled and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_grad_scaler)
    print(
        f"AMP: enabled={amp_enabled}, dtype={amp_dtype_name}, grad_scaler={use_grad_scaler}"
    )

    use_mean_teacher = bool(cfg["train"].get("use_mean_teacher", False))
    ema_decay = float(cfg["train"].get("ema_decay", 0.99))
    consistency_weight = float(cfg["train"].get("consistency_weight", 0.0))
    consistency_rampup_epochs = int(cfg["train"].get("consistency_rampup_epochs", 10))
    source_distill_weight = float(cfg["train"].get("source_distill_weight", 0.0))
    source_distill_rampup_epochs = int(cfg["train"].get("source_distill_rampup_epochs", 0))
    source_teacher_checkpoint = cfg["train"].get("source_teacher_checkpoint")
    source_teacher_strict = bool(cfg["train"].get("source_teacher_strict", False))
    agreement_gamma = float(cfg["train"].get("agreement_gamma", 0.0))
    pseudo_consensus_weight = float(cfg["train"].get("pseudo_consensus_weight", 0.0))
    boundary_weight = float(cfg["train"].get("boundary_weight", 0.0))
    boundary_recall_weight = float(cfg["train"].get("boundary_recall_weight", 0.0))
    boundary_recall_alpha = float(cfg["train"].get("boundary_recall_alpha", 0.3))
    boundary_recall_beta = float(cfg["train"].get("boundary_recall_beta", 0.7))
    boundary_recall_gamma = float(cfg["train"].get("boundary_recall_gamma", 1.33))
    boundary_recall_boost = float(cfg["train"].get("boundary_recall_boost", 2.0))
    boundary_head_weight = float(cfg["train"].get("boundary_head_weight", 0.0))
    background_supervision_weight = float(cfg["train"].get("background_supervision_weight", 0.0))
    low_threshold_bg_weight = float(cfg["train"].get("low_threshold_bg_weight", 0.0))
    low_threshold_bg_threshold = float(cfg["train"].get("low_threshold_bg_threshold", 0.08))
    uncertainty_supervision_weight = float(cfg["train"].get("uncertainty_supervision_weight", 0.0))
    recall_candidate_weight = float(cfg["train"].get("recall_candidate_weight", 0.0))
    recall_candidate_dilate_kernel = int(cfg["train"].get("recall_candidate_dilate_kernel", 3))
    recall_candidate_pos_weight = float(cfg["train"].get("recall_candidate_pos_weight", 1.5))
    multiscale_recall_weight = float(cfg["train"].get("multiscale_recall_weight", 0.0))
    multiscale_recall_dilate_kernel = int(cfg["train"].get("multiscale_recall_dilate_kernel", 3))
    multiscale_recall_pos_weight = float(cfg["train"].get("multiscale_recall_pos_weight", 1.5))
    rabc_boundary_weight = float(cfg["train"].get("rabc_boundary_weight", 0.0))
    rabc_far_bg_weight = float(cfg["train"].get("rabc_far_bg_weight", 0.0))
    rabc_sparsity_weight = float(cfg["train"].get("rabc_sparsity_weight", 0.0))
    rabc_delta_direction_weight = float(cfg["train"].get("rabc_delta_direction_weight", 0.0))
    rabc_delta_direction_margin = float(cfg["train"].get("rabc_delta_direction_margin", 0.05))
    rabc_delta_direction_fp_weight = float(cfg["train"].get("rabc_delta_direction_fp_weight", 0.5))
    brr_candidate_weight = float(cfg["train"].get("brr_candidate_weight", 0.0))
    brr_boundary_weight = float(cfg["train"].get("brr_boundary_weight", 0.0))
    brr_far_bg_weight = float(cfg["train"].get("brr_far_bg_weight", 0.0))
    brr_sparsity_weight = float(cfg["train"].get("brr_sparsity_weight", 0.0))
    aux_consensus_weight = float(cfg["train"].get("aux_consensus_weight", 0.0))
    detail_consensus_weight = float(cfg["train"].get("detail_consensus_weight", 0.0))
    dual_head_consistency_weight = float(cfg["train"].get("dual_head_consistency_weight", 0.0))
    supervised_weight = float(cfg["train"].get("supervised_weight", 0.0))
    pseudo_tversky_weight = float(cfg["train"].get("pseudo_tversky_weight", 0.0))
    pseudo_tversky_alpha = float(cfg["train"].get("pseudo_tversky_alpha", 0.5))
    pseudo_tversky_beta = float(cfg["train"].get("pseudo_tversky_beta", 0.5))
    pseudo_tversky_gamma = float(cfg["train"].get("pseudo_tversky_gamma", 1.0))
    supervised_tversky_weight = float(cfg["train"].get("supervised_tversky_weight", 0.0))
    supervised_tversky_alpha = float(cfg["train"].get("supervised_tversky_alpha", 0.5))
    supervised_tversky_beta = float(cfg["train"].get("supervised_tversky_beta", 0.5))
    supervised_tversky_gamma = float(cfg["train"].get("supervised_tversky_gamma", 1.0))
    labeled_ratio = float(cfg["train"].get("labeled_ratio", 0.0))
    labeled_ratio = min(max(labeled_ratio, 0.0), 1.0)
    teacher_model: torch.nn.Module | None = None
    source_teacher_model: torch.nn.Module | None = None

    if use_mean_teacher:
        teacher_model = RPINet(**model_kwargs_from_cfg(cfg)).to(device)
        if channels_last:
            teacher_model = teacher_model.to(memory_format=torch.channels_last)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    if source_distill_weight > 0.0:
        if source_teacher_checkpoint is None:
            raise ValueError("train.source_distill_weight > 0 requires train.source_teacher_checkpoint")
        source_teacher_path = Path(str(source_teacher_checkpoint))
        if not source_teacher_path.exists():
            raise FileNotFoundError(f"source teacher checkpoint not found: {source_teacher_path}")
        source_teacher_model = RPINet(**model_kwargs_from_cfg(cfg)).to(device)
        if channels_last:
            source_teacher_model = source_teacher_model.to(memory_format=torch.channels_last)
        source_ckpt = torch.load(source_teacher_path, map_location="cpu")
        source_state = source_ckpt.get("model") if isinstance(source_ckpt, dict) and "model" in source_ckpt else source_ckpt
        source_load_info = load_model_state(source_teacher_model, source_state, strict=source_teacher_strict)
        source_teacher_model.eval()
        for p in source_teacher_model.parameters():
            p.requires_grad_(False)
        if not source_teacher_strict:
            print(
                "Source teacher load strict=False: "
                f"missing={len(source_load_info['missing_keys'])}, "
                f"unexpected={len(source_load_info['unexpected_keys'])}, "
                f"shape_mismatch_skipped={len(source_load_info['skipped_mismatch'])}"
            )

    start_epoch = 0
    best_jac = -1.0
    best_state = None

    resume_path = args.resume or cfg["train"].get("resume_checkpoint")
    if resume_path:
        resume_path = str(resume_path)
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        resume_strict = bool(cfg["train"].get("resume_strict", True))
        load_info = load_model_state(model, ckpt["model"], strict=resume_strict)
        if not resume_strict:
            print(
                "Resume with strict=False: "
                f"missing={len(load_info['missing_keys'])}, "
                f"unexpected={len(load_info['unexpected_keys'])}, "
                f"shape_mismatch_skipped={len(load_info['skipped_mismatch'])}"
            )
        if bool(cfg["train"].get("resume_load_optimizer", True)) and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"Skip optimizer state load due to: {e}")
        if (
            use_grad_scaler
            and bool(cfg["train"].get("resume_load_scaler", True))
            and "scaler" in ckpt
            and ckpt["scaler"] is not None
        ):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"Skip scaler state load due to: {e}")

        if bool(cfg["train"].get("resume_reset_epoch", False)):
            start_epoch = 0
        else:
            start_epoch = int(ckpt["epoch"]) + 1

        if bool(cfg["train"].get("resume_reset_best", False)):
            best_jac = -1.0
        else:
            best_jac = float(ckpt.get("best_jac", -1.0))
        best_state = copy.deepcopy(get_model_state(model))
        print(f"Resumed from {resume_path}, epoch {start_epoch}, best_jac={best_jac:.4f}")
        if teacher_model is not None:
            teacher_state = ckpt.get("teacher_model")
            if teacher_state is not None:
                load_model_state(teacher_model, teacher_state, strict=resume_strict)
            else:
                load_model_state(teacher_model, get_model_state(model), strict=True)
    elif teacher_model is not None:
        load_model_state(teacher_model, get_model_state(model), strict=True)

    history = []
    total_epochs = int(cfg["train"]["epochs"])
    include_teacher_ckpt = bool(cfg["runtime"].get("checkpoint_include_teacher", True))
    include_optimizer_ckpt = bool(cfg["runtime"].get("checkpoint_include_optimizer", False))
    include_scaler_ckpt = bool(cfg["runtime"].get("checkpoint_include_scaler", False))

    def make_ckpt_payload(epoch_idx: int) -> Dict:
        payload: Dict = {
            "epoch": int(epoch_idx),
            "model": get_model_state(model),
            "best_jac": float(best_jac),
            "config": cfg,
        }
        if include_teacher_ckpt:
            payload["teacher_model"] = get_model_state(teacher_model) if teacher_model is not None else None
        if include_optimizer_ckpt:
            payload["optimizer"] = optimizer.state_dict()
        if include_scaler_ckpt:
            payload["scaler"] = scaler.state_dict()
        return payload

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch [{epoch + 1}/{total_epochs}]")
        train_log = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
            channels_last=channels_last,
            teacher_model=teacher_model,
            source_teacher_model=source_teacher_model,
            consistency_weight=consistency_weight if use_mean_teacher else 0.0,
            consistency_rampup_epochs=consistency_rampup_epochs,
            source_distill_weight=source_distill_weight,
            source_distill_rampup_epochs=source_distill_rampup_epochs,
            ema_decay=ema_decay,
            epoch_idx=epoch,
            agreement_gamma=agreement_gamma,
            pseudo_consensus_weight=pseudo_consensus_weight,
            boundary_weight=boundary_weight,
            boundary_head_weight=boundary_head_weight,
            background_supervision_weight=background_supervision_weight,
            low_threshold_bg_weight=low_threshold_bg_weight,
            low_threshold_bg_threshold=low_threshold_bg_threshold,
            uncertainty_supervision_weight=uncertainty_supervision_weight,
            recall_candidate_weight=recall_candidate_weight,
            recall_candidate_dilate_kernel=recall_candidate_dilate_kernel,
            recall_candidate_pos_weight=recall_candidate_pos_weight,
            multiscale_recall_weight=multiscale_recall_weight,
            multiscale_recall_dilate_kernel=multiscale_recall_dilate_kernel,
            multiscale_recall_pos_weight=multiscale_recall_pos_weight,
            rabc_boundary_weight=rabc_boundary_weight,
            rabc_far_bg_weight=rabc_far_bg_weight,
            rabc_sparsity_weight=rabc_sparsity_weight,
            rabc_delta_direction_weight=rabc_delta_direction_weight,
            rabc_delta_direction_margin=rabc_delta_direction_margin,
            rabc_delta_direction_fp_weight=rabc_delta_direction_fp_weight,
            brr_candidate_weight=brr_candidate_weight,
            brr_boundary_weight=brr_boundary_weight,
            brr_far_bg_weight=brr_far_bg_weight,
            brr_sparsity_weight=brr_sparsity_weight,
            aux_consensus_weight=aux_consensus_weight,
            detail_consensus_weight=detail_consensus_weight,
            dual_head_consistency_weight=dual_head_consistency_weight,
            supervised_weight=supervised_weight,
            labeled_ratio=labeled_ratio,
            pseudo_tversky_weight=pseudo_tversky_weight,
            pseudo_tversky_alpha=pseudo_tversky_alpha,
            pseudo_tversky_beta=pseudo_tversky_beta,
            pseudo_tversky_gamma=pseudo_tversky_gamma,
            supervised_tversky_weight=supervised_tversky_weight,
            supervised_tversky_alpha=supervised_tversky_alpha,
            supervised_tversky_beta=supervised_tversky_beta,
            supervised_tversky_gamma=supervised_tversky_gamma,
            boundary_recall_weight=boundary_recall_weight,
            boundary_recall_alpha=boundary_recall_alpha,
            boundary_recall_beta=boundary_recall_beta,
            boundary_recall_gamma=boundary_recall_gamma,
            boundary_recall_boost=boundary_recall_boost,
            bn_trainable_prefixes=bn_trainable_prefixes,
        )
        val_log = evaluate(
            model,
            val_loader,
            device,
            threshold=float(cfg["train"].get("threshold", 0.3)),
            desc="val",
            channels_last=channels_last,
        )

        row = {"epoch": epoch + 1, **train_log, **{f"val_{k}": v for k, v in val_log.items()}}
        history.append(row)
        print(row)

        save_every = int(cfg["runtime"].get("save_every", 5))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(make_ckpt_payload(epoch), ckpt_dir / f"epoch_{epoch + 1}.pt")

        if val_log["jac"] > best_jac:
            best_jac = val_log["jac"]
            best_state = copy.deepcopy(get_model_state(model))
            torch.save(make_ckpt_payload(epoch), ckpt_dir / "best.pt")

    if best_state is not None:
        load_model_state(model, best_state)

    rounds = int(cfg["train"].get("iterative_rounds", 0))
    iter_epochs = int(cfg["train"].get("iterative_epochs", 0))
    blend = float(cfg["train"].get("iterative_blend", 0.35))
    blend_decay = float(cfg["train"].get("iterative_blend_decay", 1.0))

    for r in range(rounds):
        round_blend = blend * (blend_decay ** r)
        print(f"\nIterative round {r + 1}/{rounds}: updating pseudo labels (blend={round_blend:.4f})")
        pseudo_model = teacher_model if teacher_model is not None else model
        update_stats = iterative_update_pseudo_labels(
            pseudo_model,
            cfg,
            pseudo_dir,
            device,
            round_blend,
            channels_last,
        )
        print(f"Iterative update stats: {update_stats}")

        train_loader, val_loader, test_loader = make_dataloaders(cfg, pseudo_dir)

        for e in range(iter_epochs):
            print(f"Iter epoch [{e + 1}/{iter_epochs}] in round {r + 1}")
            train_log = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
                channels_last=channels_last,
                teacher_model=teacher_model,
                source_teacher_model=source_teacher_model,
                consistency_weight=consistency_weight if use_mean_teacher else 0.0,
                consistency_rampup_epochs=consistency_rampup_epochs,
                source_distill_weight=source_distill_weight,
                source_distill_rampup_epochs=source_distill_rampup_epochs,
                ema_decay=ema_decay,
                epoch_idx=total_epochs + (r * iter_epochs + e),
                agreement_gamma=agreement_gamma,
                pseudo_consensus_weight=pseudo_consensus_weight,
                boundary_weight=boundary_weight,
                boundary_head_weight=boundary_head_weight,
                background_supervision_weight=background_supervision_weight,
                low_threshold_bg_weight=low_threshold_bg_weight,
                low_threshold_bg_threshold=low_threshold_bg_threshold,
                uncertainty_supervision_weight=uncertainty_supervision_weight,
                recall_candidate_weight=recall_candidate_weight,
                recall_candidate_dilate_kernel=recall_candidate_dilate_kernel,
                recall_candidate_pos_weight=recall_candidate_pos_weight,
                multiscale_recall_weight=multiscale_recall_weight,
                multiscale_recall_dilate_kernel=multiscale_recall_dilate_kernel,
                multiscale_recall_pos_weight=multiscale_recall_pos_weight,
                rabc_boundary_weight=rabc_boundary_weight,
                rabc_far_bg_weight=rabc_far_bg_weight,
                rabc_sparsity_weight=rabc_sparsity_weight,
                rabc_delta_direction_weight=rabc_delta_direction_weight,
                rabc_delta_direction_margin=rabc_delta_direction_margin,
                rabc_delta_direction_fp_weight=rabc_delta_direction_fp_weight,
                brr_candidate_weight=brr_candidate_weight,
                brr_boundary_weight=brr_boundary_weight,
                brr_far_bg_weight=brr_far_bg_weight,
                brr_sparsity_weight=brr_sparsity_weight,
                aux_consensus_weight=aux_consensus_weight,
                detail_consensus_weight=detail_consensus_weight,
                dual_head_consistency_weight=dual_head_consistency_weight,
                supervised_weight=supervised_weight,
                labeled_ratio=labeled_ratio,
                pseudo_tversky_weight=pseudo_tversky_weight,
                pseudo_tversky_alpha=pseudo_tversky_alpha,
                pseudo_tversky_beta=pseudo_tversky_beta,
                pseudo_tversky_gamma=pseudo_tversky_gamma,
                supervised_tversky_weight=supervised_tversky_weight,
                supervised_tversky_alpha=supervised_tversky_alpha,
                supervised_tversky_beta=supervised_tversky_beta,
                supervised_tversky_gamma=supervised_tversky_gamma,
                boundary_recall_weight=boundary_recall_weight,
                boundary_recall_alpha=boundary_recall_alpha,
                boundary_recall_beta=boundary_recall_beta,
                boundary_recall_gamma=boundary_recall_gamma,
                boundary_recall_boost=boundary_recall_boost,
                bn_trainable_prefixes=bn_trainable_prefixes,
            )
            val_log = evaluate(
                model,
                val_loader,
                device,
                threshold=float(cfg["train"].get("threshold", 0.3)),
                desc="val-iter",
                channels_last=channels_last,
            )
            row = {
                "iter_round": r + 1,
                "iter_epoch": e + 1,
                **train_log,
                **{f"val_{k}": v for k, v in val_log.items()},
            }
            history.append(row)
            print(row)

            if val_log["jac"] > best_jac:
                best_jac = val_log["jac"]
                best_state = copy.deepcopy(get_model_state(model))
                torch.save(make_ckpt_payload(total_epochs + (r * iter_epochs + e)), ckpt_dir / "best.pt")

    if best_state is not None:
        load_model_state(model, best_state)

    val_best = evaluate(
        model,
        val_loader,
        device,
        threshold=float(cfg["train"].get("threshold", 0.3)),
        desc="val-best",
        channels_last=channels_last,
    )
    test_best = evaluate(
        model,
        test_loader,
        device,
        threshold=float(cfg["train"].get("threshold", 0.3)),
        desc="test-best",
        channels_last=channels_last,
    )

    summary = {
        "best_val": val_best,
        "best_test": test_best,
        "best_jac": best_jac,
        "history_len": len(history),
    }
    save_json(summary, output_dir / "summary.json")
    save_json({"history": history}, output_dir / "history.json")

    print("\nTraining complete")
    print(summary)


if __name__ == "__main__":
    main()
