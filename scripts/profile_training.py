#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpi_net.losses import dynamic_uncertainty_loss
from rpi_net.model import RPINet
from rpi_net.utils import get_device, load_config, set_seed
from train_rpi_net import configure_runtime, ensure_pseudo_labels, make_dataloaders


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
    parser = argparse.ArgumentParser(description="Profile data/compute bottlenecks for RPI-Net training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--profiler-steps", type=int, default=12)
    return parser.parse_args()


def fmt_ms(sec: float) -> float:
    return sec * 1000.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg["runtime"].get("device", "cuda"))
    configure_runtime(cfg, device)
    channels_last = bool(cfg.get("runtime", {}).get("channels_last", True))

    t0 = time.perf_counter()
    pseudo_dir = ensure_pseudo_labels(cfg, split=cfg["data"]["train_split"])
    t1 = time.perf_counter()

    train_loader, _, _ = make_dataloaders(cfg, pseudo_dir)
    t2 = time.perf_counter()

    model = RPINet(**model_kwargs_from_cfg(cfg)).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=bool(cfg["train"].get("amp", True) and device.type == "cuda"))
    amp = bool(cfg["train"].get("amp", True) and device.type == "cuda")

    print(f"device={device}")
    print(f"prepare_pseudo_ms={fmt_ms(t1-t0):.1f}")
    print(f"build_loader_ms={fmt_ms(t2-t1):.1f}")

    model.train()
    data_times = []
    compute_times = []
    step_times = []

    it = iter(train_loader)
    for step in range(args.max_steps):
        load_s = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        load_e = time.perf_counter()

        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        pseudo = batch["pseudo"].to(device, non_blocking=True)

        if device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        step_s = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            outputs = model(images, pseudo)
            logits = outputs["logits"]
            sigma_maps = outputs["sigma_maps"]
            loss, _ = dynamic_uncertainty_loss(logits, pseudo, sigma_maps)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            ev_end.record()
            torch.cuda.synchronize()
            compute_ms = ev_start.elapsed_time(ev_end)
        else:
            compute_ms = fmt_ms(time.perf_counter() - step_s)

        step_e = time.perf_counter()

        data_ms = fmt_ms(load_e - load_s)
        total_ms = fmt_ms(step_e - load_s)
        data_times.append(data_ms)
        compute_times.append(compute_ms)
        step_times.append(total_ms)
        print(
            f"step={step+1:02d} data_ms={data_ms:.1f} compute_ms={compute_ms:.1f} total_ms={total_ms:.1f} loss={float(loss.detach().item()):.4f}"
        )

    if not step_times:
        print("no steps profiled")
        return

    def summary(name: str, vals: list[float]) -> None:
        print(
            f"{name}: mean={statistics.mean(vals):.1f} ms, p50={statistics.median(vals):.1f} ms, "
            f"max={max(vals):.1f} ms"
        )

    print("\n=== Step Timing Summary ===")
    summary("data", data_times)
    summary("compute", compute_times)
    summary("total", step_times)

    spikes = [
        (i + 1, d, c, t)
        for i, (d, c, t) in enumerate(zip(data_times, compute_times, step_times))
        if t > 3.0 * statistics.median(step_times)
    ]
    if spikes:
        print("\n=== Potential Spike Steps (>3x median total) ===")
        for s, d, c, t in spikes:
            print(f"step={s:02d} data_ms={d:.1f} compute_ms={c:.1f} total_ms={t:.1f}")
    else:
        print("\nno major spikes detected by 3x-median rule")

    # Torch profiler for operator-level hotspots (short window)
    prof_steps = max(1, min(args.profiler_steps, args.max_steps))
    print(f"\n=== Torch Profiler ({prof_steps} steps) ===")
    model.train()
    it2 = iter(train_loader)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if device.type == "cuda"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(prof_steps):
            try:
                batch = next(it2)
            except StopIteration:
                break
            images = batch["image"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            pseudo = batch["pseudo"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                outputs = model(images, pseudo)
                logits = outputs["logits"]
                sigma_maps = outputs["sigma_maps"]
                loss, _ = dynamic_uncertainty_loss(logits, pseudo, sigma_maps)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            prof.step()

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
