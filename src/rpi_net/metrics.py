from __future__ import annotations

from typing import Dict

import torch


EPS = 1e-7


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.3,
) -> Dict[str, float]:
    preds = _binarize(logits, threshold)
    targets = (targets > 0.5).float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=(1, 2, 3))

    dice = ((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)).mean().item()
    jac = ((tp + EPS) / (tp + fp + fn + EPS)).mean().item()
    sen = ((tp + EPS) / (tp + fn + EPS)).mean().item()
    spe = ((tn + EPS) / (tn + fp + EPS)).mean().item()
    acc = ((tp + tn + EPS) / (tp + tn + fp + fn + EPS)).mean().item()

    return {
        "acc": acc,
        "dice": dice,
        "jac": jac,
        "sen": sen,
        "spe": spe,
    }
