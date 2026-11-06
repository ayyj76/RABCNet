from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

EPS = 1e-7


def pseudo_agreement_map(pseudo_labels: torch.Tensor) -> torch.Tensor:
    """
    pseudo_labels: [B, K, H, W], values in [0, 1]
    returns agreement map in [0, 1], shape [B, 1, H, W]
    """
    p = pseudo_labels.mean(dim=1, keepdim=True)
    entropy = -(p * torch.log(p + 1e-8) + (1.0 - p) * torch.log(1.0 - p + 1e-8))
    entropy = entropy / 0.6931471805599453  # normalize by ln(2)
    agreement = 1.0 - entropy
    return torch.clamp(agreement, min=0.0, max=1.0)


def pseudo_consensus_map(pseudo_labels: torch.Tensor) -> torch.Tensor:
    return pseudo_labels.mean(dim=1, keepdim=True).clamp(0.0, 1.0)


def dynamic_uncertainty_loss(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    sigma_maps: List[torch.Tensor],
    agreement_gamma: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    logits: [B, 1, H, W]
    pseudo_labels: [B, 4, H, W]
    sigma_maps: list of 4 tensors [B, 1, H, W]
    """
    if pseudo_labels.shape[1] != len(sigma_maps):
        raise ValueError("Pseudo label channels and sigma maps count mismatch")

    per_label_losses = []
    alphas = []
    agreement = pseudo_agreement_map(pseudo_labels)
    agreement_weight = torch.pow(torch.clamp(agreement, min=0.05), float(max(0.0, agreement_gamma)))

    for i, sigma in enumerate(sigma_maps):
        target = pseudo_labels[:, i : i + 1]
        sigma = torch.clamp(sigma, min=-5.0, max=5.0)

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if agreement_gamma > 0.0:
            bce = bce * agreement_weight
        uncertainty_weighted = bce / torch.exp(sigma) + 0.5 * sigma

        entropy = torch.pow(F.softplus(sigma), 2.0).mean(dim=(1, 2, 3))
        alpha = 1.0 / (entropy + 1e-8)

        per_label_losses.append(uncertainty_weighted.mean())
        alphas.append(alpha.mean())

    alpha_stack = torch.stack(alphas)
    loss_stack = torch.stack(per_label_losses)
    total = (alpha_stack * loss_stack).sum() / (alpha_stack.sum() + 1e-8)

    log_dict = {
        "dul_loss": float(total.detach().item()),
        "alpha_mean": float(alpha_stack.mean().detach().item()),
        "agreement_mean": float(agreement.mean().detach().item()),
    }
    return total, log_dict


def pseudo_consensus_loss(logits: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, 1, H, W]
    pseudo_labels: [B, K, H, W]
    """
    target = pseudo_consensus_map(pseudo_labels)
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, target)
    inter = (probs * target).sum(dim=(1, 2, 3))
    dice = (2.0 * inter + EPS) / (probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + EPS)
    dice_loss = 1.0 - dice.mean()
    return 0.5 * bce + 0.5 * dice_loss


def soft_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Soft Tversky / Focal-Tversky loss for binary segmentation.
    logits: [B,1,H,W]
    targets: [B,1,H,W] in [0,1]
    """
    targets = targets.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - probs) * targets).sum(dim=dims)
    tversky = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
    base = 1.0 - tversky
    if gamma != 1.0:
        base = torch.pow(torch.clamp(base, min=0.0), gamma)
    return base.mean()


def pseudo_consensus_tversky_loss(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    tversky_gamma: float = 1.0,
) -> torch.Tensor:
    """
    BCE + Dice with optional Tversky term on pseudo-consensus targets.
    """
    target = pseudo_consensus_map(pseudo_labels)
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, target)
    inter = (probs * target).sum(dim=(1, 2, 3))
    dice = (2.0 * inter + EPS) / (probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + EPS)
    dice_loss = 1.0 - dice.mean()
    base = 0.5 * bce + 0.5 * dice_loss

    if tversky_weight <= 0.0:
        return base
    tv = soft_tversky_loss(
        logits,
        target,
        alpha=float(tversky_alpha),
        beta=float(tversky_beta),
        gamma=float(tversky_gamma),
    )
    return base + float(tversky_weight) * tv


def dual_head_consistency_loss(
    train_logits: torch.Tensor,
    aux_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    uncertainty_logits: torch.Tensor | None = None,
    min_weight: float = 0.05,
) -> torch.Tensor:
    """
    Symmetric consistency between two segmentation heads, weighted by
    pseudo-label agreement and optional uncertainty suppression.
    """
    agreement = pseudo_agreement_map(pseudo_labels).to(dtype=train_logits.dtype)
    if uncertainty_logits is not None:
        uncertainty = torch.sigmoid(uncertainty_logits).to(dtype=train_logits.dtype)
        weight = agreement * (1.0 - uncertainty)
    else:
        weight = agreement
    weight = torch.clamp(weight, min=float(min_weight), max=1.0)

    aux_prob = torch.sigmoid(aux_logits).detach()
    train_prob = torch.sigmoid(train_logits).detach()
    loss_train = F.binary_cross_entropy_with_logits(train_logits, aux_prob, reduction="none")
    loss_aux = F.binary_cross_entropy_with_logits(aux_logits, train_prob, reduction="none")
    return (0.5 * (loss_train + loss_aux) * weight).mean()


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    # x: [B,1,H,W]
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + EPS)
    return torch.clamp(mag, 0.0, 1.0)


def boundary_alignment_loss(logits: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """
    Encourage predicted boundaries to align with pseudo-consensus boundaries.
    """
    pred = torch.sigmoid(logits)
    target = pseudo_consensus_map(pseudo_labels)

    pred_b = _sobel_mag(pred)
    tgt_b = _sobel_mag(target)
    return F.l1_loss(pred_b, tgt_b)


def boundary_supervision_loss(boundary_logits: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """
    Supervise explicit boundary prediction head using pseudo-consensus boundary target.
    """
    target = _sobel_mag(pseudo_consensus_map(pseudo_labels))
    return F.binary_cross_entropy_with_logits(boundary_logits, target)


def background_supervision_loss(bg_logits: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """
    Supervise background branch using inverse pseudo-consensus target.
    """
    target = 1.0 - pseudo_consensus_map(pseudo_labels)
    target = torch.clamp(target, min=0.0, max=1.0)
    return F.binary_cross_entropy_with_logits(bg_logits, target)


def low_threshold_background_loss(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    threshold: float = 0.08,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Penalize background-region probabilities above a low operating threshold.
    """
    bg = 1.0 - pseudo_consensus_map(pseudo_labels)
    bg = torch.clamp(bg, min=0.0, max=1.0)
    p = torch.sigmoid(logits)
    exceed = torch.relu(p - float(threshold))
    return (torch.pow(exceed, float(power)) * bg).mean()


def uncertainty_supervision_loss(uncertainty_logits: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """
    Supervise uncertainty head with pseudo disagreement map.
    """
    target = 1.0 - pseudo_agreement_map(pseudo_labels)
    target = torch.clamp(target, min=0.0, max=1.0)
    return F.binary_cross_entropy_with_logits(uncertainty_logits, target)


def supervised_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Supervised segmentation loss with BCE + Dice on available labeled samples.
    logits: [B,1,H,W]
    targets: [B,1,H,W], values in {0,1}
    """
    targets = targets.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    dice = (2.0 * inter + EPS) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + EPS)
    dice_loss = 1.0 - dice.mean()
    return 0.5 * bce + 0.5 * dice_loss


def supervised_segmentation_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    tversky_gamma: float = 1.0,
) -> torch.Tensor:
    """
    BCE + Dice with optional Tversky term on supervised masks.
    """
    targets = targets.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    dice = (2.0 * inter + EPS) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + EPS)
    dice_loss = 1.0 - dice.mean()
    base = 0.5 * bce + 0.5 * dice_loss

    if tversky_weight <= 0.0:
        return base
    tv = soft_tversky_loss(
        logits,
        targets,
        alpha=float(tversky_alpha),
        beta=float(tversky_beta),
        gamma=float(tversky_gamma),
    )
    return base + float(tversky_weight) * tv


def boundary_aware_focal_tversky_loss(
    logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.33,
    boundary_boost: float = 2.0,
) -> torch.Tensor:
    """
    Boundary-aware Focal Tversky loss for recall improvement.

    Key innovations over standard Tversky:
    1. α < β by default: penalizes FN (missed lesion) more than FP → boosts SEN.
    2. Focal weighting (1-tversky)^γ: harder samples (low tversky) get higher loss.
    3. Boundary boosting: FN pixels near pseudo-consensus boundaries receive extra
       penalty via Sobel magnitude weighting, directly targeting the dominant
       source of SEN degradation (boundary under-segmentation).

    Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function with
    Improved Attention U-Net for Lesion Segmentation", ISBI 2019.

    logits: [B,1,H,W]
    pseudo_labels: [B,K,H,W]
    alpha: FP weight (smaller → less FP penalty → more recall)
    beta: FN weight (larger → more FN penalty → more recall)
    gamma: focal exponent (>1 emphasizes hard samples)
    boundary_boost: extra weight for FN in boundary regions
    """
    target = pseudo_consensus_map(pseudo_labels)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)

    # Boundary-weighted FN: pixels near edges get boosted FN penalty
    boundary_map = _sobel_mag(target)  # [B,1,H,W], high at edges
    fn_weight = 1.0 + float(boundary_boost) * boundary_map
    fn_map = (1.0 - probs) * target * fn_weight
    fn = fn_map.sum(dim=dims)

    tversky = (tp + EPS) / (tp + float(alpha) * fp + float(beta) * fn + EPS)
    base = 1.0 - tversky

    # Focal weighting: harder samples (lower tversky) get exponentially higher loss
    if float(gamma) != 1.0:
        base = torch.pow(torch.clamp(base, min=0.0), float(gamma))
    return base.mean()


def recall_candidate_supervision_loss(
    recall_candidate_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    dilate_kernel: int = 3,
    pos_weight: float = 1.5,
) -> torch.Tensor:
    """
    Supervise recall candidate head with a slightly dilated pseudo-consensus target.
    This favors recovering missed lesion pixels without touching main logits directly.
    """
    target = pseudo_consensus_map(pseudo_labels)
    k = max(1, int(dilate_kernel))
    if k % 2 == 0:
        k += 1
    if k > 1:
        target = F.max_pool2d(target, kernel_size=k, stride=1, padding=k // 2)
    target = torch.clamp(target, min=0.0, max=1.0)

    if float(pos_weight) <= 1.0:
        return F.binary_cross_entropy_with_logits(recall_candidate_logits, target)

    weight = 1.0 + (float(pos_weight) - 1.0) * target
    bce = F.binary_cross_entropy_with_logits(
        recall_candidate_logits,
        target,
        reduction="none",
    )
    return (bce * weight).mean()


def boundary_recall_consistency_loss(
    refined_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    boundary_logits: torch.Tensor | None = None,
    uncertainty_logits: torch.Tensor | None = None,
    focus_boost: float = 2.0,
) -> torch.Tensor:
    """
    Encourage refined logits to recover pseudo-consensus positives near
    uncertain boundaries without changing the whole mask uniformly.
    """
    target = pseudo_consensus_map(pseudo_labels)
    boundary_map = _sobel_mag(target)
    if boundary_logits is not None:
        boundary_map = torch.maximum(boundary_map, torch.sigmoid(boundary_logits))
    if uncertainty_logits is not None:
        uncertainty_map = torch.sigmoid(uncertainty_logits)
    else:
        uncertainty_map = torch.zeros_like(refined_logits)

    if base_logits is not None:
        base_prob = torch.sigmoid(base_logits.detach())
    else:
        base_prob = torch.sigmoid(refined_logits.detach())

    candidate = target * boundary_map * (0.5 + 0.5 * uncertainty_map) * (1.0 - base_prob)
    weight = 1.0 + float(focus_boost) * torch.clamp(candidate, min=0.0, max=1.0)
    bce = F.binary_cross_entropy_with_logits(refined_logits, target, reduction="none")
    return (bce * weight).mean()


def rabc_delta_direction_loss(
    delta_map: torch.Tensor,
    pseudo_labels: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    boundary_logits: torch.Tensor | None = None,
    uncertainty_logits: torch.Tensor | None = None,
    margin: float = 0.05,
    boundary_kernel: int = 5,
    fp_weight: float = 0.5,
) -> torch.Tensor:
    """
    Directly supervise the calibration direction of RABC deltas.

    Positive deltas are encouraged for pseudo-positive boundary pixels where
    the frozen base prediction is under-confident, while negative deltas are
    encouraged for pseudo-background boundary pixels where the base prediction
    is over-confident. This makes the calibration branch learn a directional
    correction instead of relying only on an indirect refined-logit BCE.
    """
    target = pseudo_consensus_map(pseudo_labels)
    boundary_map = _sobel_mag(target)
    k = max(1, int(boundary_kernel))
    if k % 2 == 0:
        k += 1
    if k > 1:
        boundary_map = F.max_pool2d(boundary_map, kernel_size=k, stride=1, padding=k // 2)
    if boundary_logits is not None:
        boundary_map = torch.maximum(boundary_map, torch.sigmoid(boundary_logits))
    if uncertainty_logits is not None:
        uncertainty_map = torch.sigmoid(uncertainty_logits)
    else:
        uncertainty_map = torch.zeros_like(delta_map)
    if base_logits is not None:
        base_prob = torch.sigmoid(base_logits.detach())
    else:
        base_prob = torch.sigmoid(delta_map.detach() * 0.0)

    reliability = torch.clamp(boundary_map, 0.0, 1.0) * (0.5 + 0.5 * torch.clamp(uncertainty_map, 0.0, 1.0))
    fn_candidate = target * torch.relu(target - base_prob) * reliability
    fp_candidate = (1.0 - target) * torch.relu(base_prob - target) * reliability

    margin_t = torch.as_tensor(float(margin), device=delta_map.device, dtype=delta_map.dtype)
    pos_loss = F.softplus(margin_t - delta_map) * fn_candidate
    neg_loss = F.softplus(margin_t + delta_map) * fp_candidate
    pos_denom = fn_candidate.sum() + EPS
    neg_denom = fp_candidate.sum() + EPS
    return pos_loss.sum() / pos_denom + float(fp_weight) * neg_loss.sum() / neg_denom


def far_background_preservation_loss(
    refined_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    margin: float = 0.02,
    band_kernel: int = 5,
) -> torch.Tensor:
    """
    Penalize refined predictions that increase confidence in far-background
    regions outside the pseudo-consensus support and boundary band.
    """
    target = pseudo_consensus_map(pseudo_labels)
    boundary = _sobel_mag(target)
    k = max(1, int(band_kernel))
    if k % 2 == 0:
        k += 1
    boundary_band = F.max_pool2d(boundary, kernel_size=k, stride=1, padding=k // 2)
    far_background = (1.0 - target) * (1.0 - boundary_band)

    refined_prob = torch.sigmoid(refined_logits)
    if base_logits is not None:
        base_prob = torch.sigmoid(base_logits.detach())
    else:
        base_prob = torch.zeros_like(refined_prob)

    excess = torch.relu(refined_prob - base_prob - float(margin))
    denom = far_background.sum() + EPS
    return (excess * far_background).sum() / denom


def calibration_sparsity_loss(
    delta_map: torch.Tensor,
    support_map: torch.Tensor | None = None,
    outside_weight: float = 1.0,
) -> torch.Tensor:
    """
    Keep calibration sparse outside the intended boundary-support region.
    """
    if support_map is None:
        return torch.abs(delta_map).mean()
    outside = 1.0 - torch.clamp(support_map, min=0.0, max=1.0)
    return float(outside_weight) * (torch.abs(delta_map) * outside).mean()
