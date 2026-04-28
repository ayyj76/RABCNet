#!/usr/bin/env python3
"""
train_supervised.py
六个监督分割基线模型统一训练 + 测试评估脚本。

支持数据集：ISIC-2017 / ISIC-2018 / PH2（通过 datasets 根目录自动识别）
支持模型：unet / unetplusplus / attdeeplabv3plus / transfuse / fatnet / malunet

用法示例：
  python scripts/train_supervised.py \
      --model unet \
      --data-root datasets/ISIC2017 \
      --output-dir outputs/supervised/unet_isic2017 \
      --epochs 50 --batch-size 16 --lr 1e-4

  # 训练完成后自动在 test split 上评估，结果写入 output-dir/test_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset
from supervised_models import build_model

EPS = 1e-7


# ═══════════════════════════════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════════════════════════════

def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    flat_p = prob.view(prob.size(0), -1)
    flat_t = targets.view(targets.size(0), -1)
    inter = (flat_p * flat_t).sum(1)
    union = flat_p.sum(1) + flat_t.sum(1)
    return (1 - (2 * inter + EPS) / (union + EPS)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dl  = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dl


# ═══════════════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    """preds/gts: [N,H,W] float32 (pred 为 sigmoid 概率, gt 为 0/1)"""
    p = (preds > 0.5).astype(np.float32)
    g = gts.astype(np.float32)
    tp = (p * g).sum(axis=(1, 2))
    fp = (p * (1 - g)).sum(axis=(1, 2))
    fn = ((1 - p) * g).sum(axis=(1, 2))
    tn = ((1 - p) * (1 - g)).sum(axis=(1, 2))
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    jac  = (tp + EPS) / (tp + fp + fn + EPS)
    sen  = (tp + EPS) / (tp + fn + EPS)
    spe  = (tn + EPS) / (tn + fp + EPS)
    acc  = (tp + tn + EPS) / (tp + tn + fp + fn + EPS)
    return dict(
        dice=float(dice.mean() * 100),
        jac =float(jac.mean()  * 100),
        sen =float(sen.mean()  * 100),
        spe =float(spe.mean()  * 100),
        acc =float(acc.mean()  * 100),
    )


# ═══════════════════════════════════════════════════════════════════════
# 训练 / 验证 / 测试
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs   = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(imgs)["logits"]          # [B,1,H,W]
            loss   = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, device, use_amp) -> dict:
    model.eval()
    all_probs, all_gts = [], []
    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"]
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(imgs)["logits"]
        probs = torch.sigmoid(logits)[:, 0].cpu().float().numpy()
        gts   = (masks[:, 0].numpy() > 0.5).astype(np.float32)
        all_probs.append(probs)
        all_gts.append(gts)
    all_probs = np.concatenate(all_probs, axis=0)
    all_gts   = np.concatenate(all_gts,   axis=0)
    return compute_metrics(all_probs, all_gts)


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(data_root: str, split: str, image_size: int, augment: bool):
    return ISIC2017Dataset(
        root=data_root,
        split=split,
        image_size=image_size,
        augment=augment,
        use_hair_removal=False,
        ph2_split_seed=42,
        ph2_split_ratios=(0.7, 0.15, 0.15),
    )


def main():
    parser = argparse.ArgumentParser(description="Supervised segmentation baseline trainer")
    parser.add_argument("--model",       required=True,
                        choices=["unet","unetplusplus","attdeeplabv3plus",
                                 "transfuse","fatnet","malunet"],
                        help="Model name")
    parser.add_argument("--data-root",   required=True,  help="Dataset root directory")
    parser.add_argument("--output-dir",  required=True,  help="Output directory for checkpoints and results")
    parser.add_argument("--image-size",  type=int, default=224)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights")
    parser.add_argument("--amp",         action="store_true", default=True,
                        help="Use automatic mixed precision (default: on)")
    parser.add_argument("--no-amp",      dest="amp", action="store_false")
    parser.add_argument("--patience",    type=int, default=10,
                        help="Early-stopping patience (epochs without val JAC improvement)")
    args = parser.parse_args()

    set_seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp  = args.amp and device.type == "cuda"
    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_supervised] model={args.model}  data={args.data_root}")
    print(f"  device={device}  amp={use_amp}  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")

    # ── 数据集 ────────────────────────────────────────────────────────
    train_ds = make_dataset(args.data_root, "train", args.image_size, augment=True)
    val_ds   = make_dataset(args.data_root, "val",   args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # ── 模型 ──────────────────────────────────────────────────────────
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params={n_params/1e6:.1f}M")

    # ── 优化器 + 调度器 ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 训练循环 ──────────────────────────────────────────────────────
    best_jac    = -1.0
    no_improve  = 0
    history     = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_metrics = evaluate(model, val_loader, device, use_amp)
        scheduler.step()

        row = dict(epoch=epoch, train_loss=round(train_loss, 5), **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()})
        history.append(row)
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"JAC={val_metrics['jac']:.2f}%  DIC={val_metrics['dice']:.2f}%  "
              f"SEN={val_metrics['sen']:.2f}%  SPE={val_metrics['spe']:.2f}%")

        if val_metrics["jac"] > best_jac:
            best_jac = val_metrics["jac"]
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_metrics": val_metrics}, ckpt_dir / "best.pt")
            print(f"    → best checkpoint saved (JAC={best_jac:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # 保存训练历史
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── 测试集评估 ────────────────────────────────────────────────────
    print("\n[test evaluation]")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    test_ds     = make_dataset(args.data_root, "test", args.image_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_metrics = evaluate(model, test_loader, device, use_amp)

    print(f"  n={len(test_ds)}")
    for k, v in test_metrics.items():
        print(f"  {k.upper():<5} = {v:.2f}%")

    results = dict(
        model=args.model,
        data_root=args.data_root,
        best_epoch=int(ckpt["epoch"]),
        best_val_jac=round(best_jac, 4),
        test=test_metrics,
    )
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] 结果已保存: {out_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
