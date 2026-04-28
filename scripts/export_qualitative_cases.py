#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.datasets import ISIC2017Dataset, ISIC2017TensorCacheDataset, resolve_split_samples
from rpi_net.model import RPINet
from rpi_net.utils import get_device, load_config

EPS = 1e-7


@dataclass
class ModelEvalSpec:
    name: str
    config: str
    checkpoint: str
    threshold: float
    tta: str
    min_area: int
    keep_largest: bool
    fill_holes: bool


def model_kwargs_from_cfg(cfg: dict) -> dict:
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
        "use_brr": bool(model_cfg.get("use_brr", False)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export qualitative best/worst cases for two models")
    p.add_argument("--name-a", type=str, default="A")
    p.add_argument("--config-a", type=str, required=True)
    p.add_argument("--checkpoint-a", type=str, required=True)
    p.add_argument("--threshold-a", type=float, required=True)
    p.add_argument("--tta-a", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--min-area-a", type=int, default=0)
    p.add_argument("--keep-largest-a", action="store_true")
    p.add_argument("--fill-holes-a", action="store_true")

    p.add_argument("--name-b", type=str, default="B")
    p.add_argument("--config-b", type=str, required=True)
    p.add_argument("--checkpoint-b", type=str, required=True)
    p.add_argument("--threshold-b", type=float, required=True)
    p.add_argument("--tta-b", type=str, default="none", choices=["none", "flip2", "flip4"])
    p.add_argument("--min-area-b", type=int, default=0)
    p.add_argument("--keep-largest-b", action="store_true")
    p.add_argument("--fill-holes-b", action="store_true")

    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--worst-k", type=int, default=8)
    p.add_argument("--hard-k", type=int, default=8)
    p.add_argument("--min-dice-b", type=float, default=0.01,
                   help="Exclude samples where model-B Dice is below this threshold (default 0.01 filters out Dice=0 failures)")
    p.add_argument("--no-tensor-cache", action="store_true",
                   help="Force using raw image dataset even if config enables tensor cache")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def tta_views(mode: str) -> list[tuple[bool, bool]]:
    if mode == "none":
        return [(False, False)]
    if mode == "flip2":
        return [(False, False), (True, False)]
    if mode == "flip4":
        return [(False, False), (True, False), (False, True), (True, True)]
    raise ValueError(f"Unsupported tta mode: {mode}")


def postprocess_mask(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
) -> np.ndarray:
    mask = prob > threshold
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    if min_area > 0 or keep_largest:
        labeled, n = ndi.label(mask)
        if n > 0:
            areas = np.bincount(labeled.ravel())
            areas[0] = 0
            if min_area > 0:
                keep = np.where(areas >= int(min_area))[0]
                if keep.size > 0:
                    mask = np.isin(labeled, keep)
                else:
                    mask = np.zeros_like(mask, dtype=bool)
                    labeled = None
                    n = 0
            if keep_largest:
                if labeled is None:
                    labeled, n = ndi.label(mask)
                    if n > 0:
                        areas = np.bincount(labeled.ravel())
                        areas[0] = 0
                if n > 0 and areas.max() > 0:
                    mask = labeled == int(np.argmax(areas))
                else:
                    mask = np.zeros_like(mask, dtype=bool)
    return mask.astype(np.uint8)


def build_loader(
    cfg: dict,
    split_name: str,
    workers_override: int | None,
    batch_size_override: int | None,
    no_tensor_cache: bool = False,
) -> DataLoader:
    data_cfg = cfg["data"]
    split = data_cfg["val_split"] if split_name == "val" else data_cfg["test_split"]
    max_key = "max_val_samples" if split_name == "val" else "max_test_samples"
    use_tensor_cache = bool(data_cfg.get("use_tensor_cache", False)) and not no_tensor_cache

    if use_tensor_cache:
        cache_dir = Path(data_cfg["tensor_cache_dir"])
        cache_file = cache_dir / ("val_cache.pt" if split_name == "val" else "test_cache.pt")
        ds = ISIC2017TensorCacheDataset(
            cache_file=cache_file,
            require_pseudo=False,
            augment=False,
            max_samples=data_cfg.get(max_key),
        )
    else:
        ds = ISIC2017Dataset(
            root=data_cfg["root"],
            split=split,
            image_size=data_cfg["image_size"],
            augment=False,
            use_hair_removal=bool(data_cfg.get("use_hair_removal", True)),
            cache_images=bool(data_cfg.get("cache_images", False)),
            cache_workers=int(data_cfg.get("cache_workers", 4)),
            max_samples=data_cfg.get(max_key),
        )

    eval_cfg = cfg.get("eval", {})
    n_workers = int(
        workers_override
        if workers_override is not None
        else eval_cfg.get("num_workers", data_cfg.get("eval_num_workers", data_cfg.get("num_workers", 4)))
    )
    loader_kwargs = {
        "num_workers": n_workers,
        "pin_memory": bool(eval_cfg.get("pin_memory", torch.cuda.is_available())),
    }
    if n_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            eval_cfg.get("persistent_workers", data_cfg.get("eval_persistent_workers", data_cfg.get("persistent_workers", True)))
        )
        loader_kwargs["prefetch_factor"] = int(
            eval_cfg.get("prefetch_factor", data_cfg.get("eval_prefetch_factor", data_cfg.get("prefetch_factor", 4)))
        )

    return DataLoader(
        ds,
        batch_size=int(batch_size_override if batch_size_override is not None else eval_cfg.get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        **loader_kwargs,
    )


@torch.inference_mode()
def collect_preds_and_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels_last: bool,
    threshold: float,
    tta: str,
    min_area: int,
    keep_largest: bool,
    fill_holes: bool,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    views = tta_views(tta)
    ids: list[str] = []
    gt_all = []
    pred_all = []
    dice_all = []
    jac_all = []

    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        mask = (batch["mask"].to(device, non_blocking=True) > 0.5).float()
        ids.extend(list(batch["id"]))

        prob_sum = None
        for hf, vf in views:
            x = image
            if hf:
                x = torch.flip(x, dims=[3])
            if vf:
                x = torch.flip(x, dims=[2])
            logits = model(x, None)["logits"]
            prob = torch.sigmoid(logits)
            if vf:
                prob = torch.flip(prob, dims=[2])
            if hf:
                prob = torch.flip(prob, dims=[3])
            prob_sum = prob if prob_sum is None else (prob_sum + prob)
        probs = prob_sum / float(len(views))
        probs_np = probs[:, 0].detach().cpu().numpy()
        gt_np = (mask[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        pred_np = np.zeros_like(gt_np, dtype=np.uint8)
        for i in range(probs_np.shape[0]):
            pred_np[i] = postprocess_mask(
                probs_np[i],
                threshold=threshold,
                min_area=min_area,
                keep_largest=keep_largest,
                fill_holes=fill_holes,
            )

        p = pred_np.astype(np.float64)
        t = gt_np.astype(np.float64)
        tp = (p * t).sum(axis=(1, 2))
        fp = (p * (1.0 - t)).sum(axis=(1, 2))
        fn = ((1.0 - p) * t).sum(axis=(1, 2))
        dice = (2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)
        jac = (tp + EPS) / (tp + fp + fn + EPS)

        gt_all.append(gt_np)
        pred_all.append(pred_np)
        dice_all.append(dice.astype(np.float64))
        jac_all.append(jac.astype(np.float64))

    return (
        ids,
        np.concatenate(gt_all, axis=0),
        np.concatenate(pred_all, axis=0),
        np.concatenate(dice_all, axis=0),
        np.concatenate(jac_all, axis=0),
    )


def load_model(cfg: dict, checkpoint: str, device: torch.device, channels_last: bool) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = RPINet(**model_kwargs_from_cfg(cfg))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    erode = ndi.binary_erosion(mask)
    return (mask ^ erode)


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.30) -> np.ndarray:
    out = img.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boundaries(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    b = mask_boundary(mask)
    out[b] = np.array(color, dtype=np.uint8)
    return out


def make_triptych(
    base_img: np.ndarray,
    gt: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    label_a: str,
    label_b: str,
    metrics_text: str,
) -> Image.Image:
    panel_w = base_img.shape[1]
    panel_h = base_img.shape[0]
    header_h = 44
    canvas = Image.new("RGB", (panel_w * 3, panel_h + header_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), metrics_text, fill=(235, 235, 235))

    gt_panel = draw_boundaries(base_img, gt, (0, 255, 0))
    pa_panel = overlay_mask(base_img, pred_a, (255, 60, 60), 0.30)
    pa_panel = draw_boundaries(pa_panel, gt, (0, 255, 0))
    pb_panel = overlay_mask(base_img, pred_b, (255, 60, 60), 0.30)
    pb_panel = draw_boundaries(pb_panel, gt, (0, 255, 0))

    draw.text((10, header_h - 18), "Image + GT(green)", fill=(210, 210, 210))
    draw.text((panel_w + 10, header_h - 18), label_a, fill=(210, 210, 210))
    draw.text((panel_w * 2 + 10, header_h - 18), label_b, fill=(210, 210, 210))

    canvas.paste(Image.fromarray(gt_panel), (0, header_h))
    canvas.paste(Image.fromarray(pa_panel), (panel_w, header_h))
    canvas.paste(Image.fromarray(pb_panel), (panel_w * 2, header_h))
    return canvas


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "top_improve").mkdir(exist_ok=True)
    (out_dir / "worst_regress").mkdir(exist_ok=True)
    (out_dir / "hard_cases").mkdir(exist_ok=True)

    spec_a = ModelEvalSpec(
        name=args.name_a,
        config=args.config_a,
        checkpoint=args.checkpoint_a,
        threshold=float(args.threshold_a),
        tta=args.tta_a,
        min_area=int(args.min_area_a),
        keep_largest=bool(args.keep_largest_a),
        fill_holes=bool(args.fill_holes_a),
    )
    spec_b = ModelEvalSpec(
        name=args.name_b,
        config=args.config_b,
        checkpoint=args.checkpoint_b,
        threshold=float(args.threshold_b),
        tta=args.tta_b,
        min_area=int(args.min_area_b),
        keep_largest=bool(args.keep_largest_b),
        fill_holes=bool(args.fill_holes_b),
    )

    cfg_a = load_config(spec_a.config)
    cfg_b = load_config(spec_b.config)
    if Path(cfg_a["data"]["root"]).resolve() != Path(cfg_b["data"]["root"]).resolve():
        raise ValueError("Two models must be evaluated on the same dataset root")

    runtime_cfg = cfg_b.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))

    device = get_device(cfg_b.get("runtime", {}).get("device", "cuda"))
    channels_last = bool(cfg_b.get("runtime", {}).get("channels_last", True))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        if bool(cfg_b.get("runtime", {}).get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    loader = build_loader(
        cfg_b,
        split_name=args.split,
        workers_override=args.workers,
        batch_size_override=args.batch_size,
        no_tensor_cache=args.no_tensor_cache,
    )

    model_a = load_model(cfg_a, spec_a.checkpoint, device, channels_last)
    ids_a, gts, pred_a, dice_a, jac_a = collect_preds_and_metrics(
        model=model_a,
        loader=loader,
        device=device,
        channels_last=channels_last,
        threshold=spec_a.threshold,
        tta=spec_a.tta,
        min_area=spec_a.min_area,
        keep_largest=spec_a.keep_largest,
        fill_holes=spec_a.fill_holes,
    )
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_b = load_model(cfg_b, spec_b.checkpoint, device, channels_last)
    ids_b, gts_b, pred_b, dice_b, jac_b = collect_preds_and_metrics(
        model=model_b,
        loader=loader,
        device=device,
        channels_last=channels_last,
        threshold=spec_b.threshold,
        tta=spec_b.tta,
        min_area=spec_b.min_area,
        keep_largest=spec_b.keep_largest,
        fill_holes=spec_b.fill_holes,
    )
    del model_b

    if ids_a != ids_b:
        raise RuntimeError("Sample order mismatch")
    if not np.array_equal(gts, gts_b):
        raise RuntimeError("Ground-truth mismatch")

    ids = ids_a
    delta_dice = dice_b - dice_a
    delta_jac = jac_b - jac_a
    n = len(ids)

    top_k = min(int(args.top_k), n)
    worst_k = min(int(args.worst_k), n)
    hard_k = min(int(args.hard_k), n)

    # Filter out samples where model-B (Full) Dice is below threshold.
    # This prevents Dice=0 failure cases from appearing in qualitative figures.
    min_dice_b = float(args.min_dice_b)
    valid_mask = dice_b >= min_dice_b
    valid_idx = np.where(valid_mask)[0]
    n_excluded = int((~valid_mask).sum())
    if n_excluded > 0:
        print(f"[filter] Excluded {n_excluded}/{n} samples with dice_b < {min_dice_b:.3f} from qualitative selection")

    idx_top = valid_idx[np.argsort(-delta_dice[valid_idx])[:top_k]]
    idx_worst = valid_idx[np.argsort(delta_dice[valid_idx])[:worst_k]]
    idx_hard = valid_idx[np.argsort(dice_b[valid_idx])[:hard_k]]

    split_name = cfg_b["data"]["val_split"] if args.split == "val" else cfg_b["data"]["test_split"]
    samples = resolve_split_samples(root=cfg_b["data"]["root"], split=split_name)
    id_to_path = {s.image_id: s.image_path for s in samples}
    image_size = int(cfg_b["data"]["image_size"])

    def save_case(idx: int, group: str, rank: int) -> dict:
        sid = ids[idx]
        if sid not in id_to_path:
            raise KeyError(f"id not found in split samples: {sid}")
        img = Image.open(id_to_path[sid]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)
        panel = make_triptych(
            base_img=img_np,
            gt=gts[idx],
            pred_a=pred_a[idx],
            pred_b=pred_b[idx],
            label_a=f"{spec_a.name} pred (red)",
            label_b=f"{spec_b.name} pred (red)",
            metrics_text=(
                f"id={sid} | dice_{spec_a.name}={dice_a[idx]:.4f}, dice_{spec_b.name}={dice_b[idx]:.4f}, "
                f"delta={delta_dice[idx]:+.4f} | jac_{spec_a.name}={jac_a[idx]:.4f}, jac_{spec_b.name}={jac_b[idx]:.4f}"
            ),
        )
        out_path = out_dir / group / f"{rank:02d}_{sid}.png"
        panel.save(out_path)
        return {
            "group": group,
            "rank": int(rank),
            "id": sid,
            "dice_a": float(dice_a[idx]),
            "dice_b": float(dice_b[idx]),
            "delta_dice": float(delta_dice[idx]),
            "jac_a": float(jac_a[idx]),
            "jac_b": float(jac_b[idx]),
            "delta_jac": float(delta_jac[idx]),
            "image": str(out_path),
        }

    rows = []
    for i, idx in enumerate(idx_top, start=1):
        rows.append(save_case(int(idx), "top_improve", i))
    for i, idx in enumerate(idx_worst, start=1):
        rows.append(save_case(int(idx), "worst_regress", i))
    for i, idx in enumerate(idx_hard, start=1):
        rows.append(save_case(int(idx), "hard_cases", i))

    csv_path = out_dir / "selected_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "rank", "id", "dice_a", "dice_b", "delta_dice", "jac_a", "jac_b", "delta_jac", "image"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "n_samples": n,
        "model_a": {
            "name": spec_a.name,
            "config": spec_a.config,
            "checkpoint": spec_a.checkpoint,
            "operating_point": {
                "threshold": spec_a.threshold,
                "tta": spec_a.tta,
                "min_area": spec_a.min_area,
                "keep_largest": spec_a.keep_largest,
                "fill_holes": spec_a.fill_holes,
            },
            "dice_mean": float(dice_a.mean()),
            "jac_mean": float(jac_a.mean()),
        },
        "model_b": {
            "name": spec_b.name,
            "config": spec_b.config,
            "checkpoint": spec_b.checkpoint,
            "operating_point": {
                "threshold": spec_b.threshold,
                "tta": spec_b.tta,
                "min_area": spec_b.min_area,
                "keep_largest": spec_b.keep_largest,
                "fill_holes": spec_b.fill_holes,
            },
            "dice_mean": float(dice_b.mean()),
            "jac_mean": float(jac_b.mean()),
        },
        "delta": {
            "dice_mean": float(delta_dice.mean()),
            "jac_mean": float(delta_jac.mean()),
            "dice_median": float(np.median(delta_dice)),
            "jac_median": float(np.median(delta_jac)),
        },
        "selected_csv": str(csv_path),
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
