#!/usr/bin/env python3
"""
evaluate_medsam.py
MedSAM 零样本评测脚本（bounding-box prompt，GT mask 外接矩形）

用法：
  python scripts/evaluate_medsam.py --data-root datasets/ISIC2017 --split test
  python scripts/evaluate_medsam.py --data-root datasets/ISIC2018 --split test
  python scripts/evaluate_medsam.py --data-root datasets/PH2    --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset

EPS = 1e-7
MEDSAM_CKPT = "/home/yyj/.cache/medsam"


# ───────────────────────────────────────────────────────────────
# 指标
# ───────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    """preds/gts: [N,H,W] bool/uint8"""
    p = preds.astype(np.float32)
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


# ───────────────────────────────────────────────────────────────
# GT mask → bounding box（[x_min, y_min, x_max, y_max]，图像坐标）
# ───────────────────────────────────────────────────────────────

def mask_to_bbox(mask: np.ndarray) -> list[int] | None:
    """mask: [H,W] bool。返回 [x0,y0,x1,y1] 或 None（空 mask）"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [int(x0), int(y0), int(x1), int(y1)]


# ───────────────────────────────────────────────────────────────
# 评测主函数
# ───────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(data_root: str, split: str, image_size: int,
             batch_size: int, device: torch.device) -> dict:
    # 数据集
    ds = ISIC2017Dataset(
        root=data_root,
        split=split,
        image_size=image_size,
        augment=False,
        use_hair_removal=False,
        ph2_split_seed=42,
        ph2_split_ratios=(0.7, 0.15, 0.15),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)
    print(f"  dataset: {data_root}  split={split}  n={len(ds)}")

    # 模型
    model = SamModel.from_pretrained(MEDSAM_CKPT).to(device)
    processor = SamProcessor.from_pretrained(MEDSAM_CKPT)
    model.eval()
    print(f"  MedSAM loaded from {MEDSAM_CKPT}")

    all_preds, all_gts = [], []

    for batch_idx, batch in enumerate(loader):
        imgs_t  = batch["image"]   # [B,3,H,W]  float32 in [-??, ??] — 我们数据集归一化后
        masks_t = batch["mask"]    # [B,1,H,W]  float32 0/1

        B = imgs_t.shape[0]
        gt_np = (masks_t[:, 0].numpy() > 0.5).astype(np.uint8)  # [B,H,W]

        # 逐样本处理（processor 需要 PIL image + bbox）
        pred_np = np.zeros_like(gt_np)

        for i in range(B):
            # 还原为 PIL image（从归一化 tensor 反变换）
            img_np = imgs_t[i].numpy()  # [3,H,W]
            # 反 ImageNet 归一化
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std  = np.array([0.229, 0.224, 0.225])[:, None, None]
            img_np = (img_np * std + mean).clip(0, 1)
            img_pil = Image.fromarray((img_np.transpose(1, 2, 0) * 255).astype(np.uint8))

            gt_i = gt_np[i]  # [H,W]
            bbox = mask_to_bbox(gt_i.astype(bool))
            if bbox is None:
                # 空 mask → 预测全零
                pred_np[i] = 0
                continue

            # processor 接受 [[x0,y0,x1,y1]] 格式
            inputs = processor(
                images=img_pil,
                input_boxes=[[[bbox]]],
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, multimask_output=False)
            # pred_masks: [1,1,1,256,256] → [256,256]
            logit = outputs.pred_masks[0, 0, 0]  # [256,256]
            # logit: [256,256] → resize 到 image_size
            prob_t = torch.sigmoid(logit).unsqueeze(0).unsqueeze(0)  # [1,1,256,256]
            prob_resized = torch.nn.functional.interpolate(
                prob_t, size=(image_size, image_size), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
            pred_np[i] = (prob_resized > 0.5).astype(np.uint8)

        all_preds.append(pred_np)
        all_gts.append(gt_np)

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] batches processed")

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts   = np.concatenate(all_gts,   axis=0)
    metrics = compute_metrics(all_preds, all_gts)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  required=True)
    parser.add_argument("--split",      default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output",     default=None,
                        help="JSON 输出路径（默认打印到终端）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_medsam]  device={device}")

    metrics = evaluate(args.data_root, args.split, args.image_size,
                       args.batch_size, device)

    print("\n结果：")
    for k, v in metrics.items():
        print(f"  {k.upper():<5} = {v:.2f}%")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"data_root": args.data_root, "split": args.split,
                       "test": metrics}, f, indent=2)
        print(f"\n[done] 已保存: {out}")


if __name__ == "__main__":
    main()
