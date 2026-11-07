"""
Compute model complexity metrics for RABC-Net:
  - Total / trainable parameter counts
  - FLOPs for a single 224x224 input
  - Inference speed (FPS) on CPU and GPU
  - ConvNeXt-Tiny backbone comparison
"""

import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ---- make project importable ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rpi_net.model import RPINet

# ------------------------------------------------------------------ #
# 1.  Model kwargs matching RABC-Net config (model: section)
# ------------------------------------------------------------------ #
MODEL_KWARGS = dict(
    hidden_dim=128,
    dropout=0.1,
    backbone="convnext_tiny",
    backbone_pretrained=False,       # no need for pretrained weights here
    use_boundary_head=True,
    use_uncertainty_head=True,
    refine_with_heads=True,
    use_aspp=False,
    use_aux_head=True,
    use_se_context=True,
    use_detail_fusion=True,
    use_coord_attn=True,
    use_edge_prompt=True,
    use_dual_supervised_head=False,
    use_eca_context=True,
    use_ema_context=True,
    use_cbam_decoder=True,
    use_simam_refine=True,
    use_safm_decoder=True,
    use_uncertainty_gated_refine=True,
    use_sk_context=True,
    use_triplet_decoder=True,
    use_msaa_fusion=True,
    use_ega_refine=False,
    use_cot_context=False,
    use_lsk_decoder=False,
    use_rabc=True,
)


def count_parameters(model: nn.Module):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt(n: int) -> str:
    """Format parameter count with commas and M suffix."""
    return f"{n:>14,d}  ({n / 1e6:.2f} M)"


def measure_flops(model, image, pseudo, device):
    """Try multiple backends for FLOPs estimation."""
    model.eval()
    model.to(device)
    image = image.to(device)
    if pseudo is not None:
        pseudo = pseudo.to(device)

    flops_result = None

    # --- attempt 1: torchinfo ---
    try:
        from torchinfo import summary
        info = summary(
            model,
            input_data=[image, pseudo],
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
        )
        flops_result = ("torchinfo", info.total_mult_adds)
    except Exception as e:
        print(f"  [torchinfo] not available or failed: {e}")

    # --- attempt 2: thop ---
    if flops_result is None:
        try:
            from thop import profile, clever_format
            inputs = (image,) if pseudo is None else (image, pseudo)
            macs, params = profile(model, inputs=inputs, verbose=False)
            flops_result = ("thop", macs)  # thop returns MACs (= multiply-accumulate)
        except Exception as e:
            print(f"  [thop] not available or failed: {e}")

    # --- attempt 3: fvcore ---
    if flops_result is None:
        try:
            from fvcore.nn import FlopCountAnalysis
            inputs = (image,) if pseudo is None else (image, pseudo)
            fca = FlopCountAnalysis(model, inputs)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            flops_result = ("fvcore", fca.total())
        except Exception as e:
            print(f"  [fvcore] not available or failed: {e}")

    # --- attempt 4: torch.utils.flop_counter (PyTorch >=2.1) ---
    if flops_result is None:
        try:
            from torch.utils.flop_counter import FlopCounterMode
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                if pseudo is not None:
                    model(image, pseudo)
                else:
                    model(image)
            flops_result = ("torch.utils.flop_counter", flop_counter.get_total_flops())
        except Exception as e:
            print(f"  [torch.utils.flop_counter] not available or failed: {e}")

    return flops_result


@torch.no_grad()
def measure_fps(model, image, pseudo, device, warmup=10, runs=50):
    """Measure inference FPS."""
    model.eval()
    model.to(device)
    image = image.to(device)
    if pseudo is not None:
        pseudo = pseudo.to(device)

    # warmup
    for _ in range(warmup):
        model(image, pseudo)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        model(image, pseudo)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = runs / elapsed
    ms_per_img = elapsed / runs * 1000
    return fps, ms_per_img


def main():
    print("=" * 72)
    print("  RABC-Net Model Complexity Analysis")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model = RPINet(**MODEL_KWARGS)
    model.eval()

    total, trainable = count_parameters(model)
    print(f"\n{'--- Parameter Counts (Full RABC-Net) ---':^72}")
    print(f"  Total params:      {fmt(total)}")
    print(f"  Trainable params:  {fmt(trainable)}")

    # ------------------------------------------------------------------ #
    # Backbone-only comparison
    # ------------------------------------------------------------------ #
    from torchvision.models import convnext_tiny
    backbone = convnext_tiny(weights=None)
    bb_total, bb_trainable = count_parameters(backbone)
    print(f"\n{'--- Parameter Counts (ConvNeXt-Tiny backbone only) ---':^72}")
    print(f"  Total params:      {fmt(bb_total)}")
    print(f"  Trainable params:  {fmt(bb_trainable)}")

    overhead = total - bb_total
    print(f"\n{'--- Decoder / Head Overhead ---':^72}")
    print(f"  Extra params:      {fmt(overhead)}")
    print(f"  Overhead ratio:    {overhead / bb_total * 100:.1f}% over backbone")

    # ------------------------------------------------------------------ #
    # Per-component breakdown
    # ------------------------------------------------------------------ #
    print(f"\n{'--- Per-Component Parameter Breakdown ---':^72}")
    components = {
        "Backbone (encoder)":       model.encoder,
        "Pseudo Encoder":           model.pseudo_encoder,
        "IPC Attention":            model.ipc_attn,
        "PIA Attention":            model.pia_attn,
        "Sigma Heads (x4)":         model.sigma_heads,
        "C2/C3/C4 Projections":     nn.ModuleList([model.c2_proj, model.c2_down,
                                                    model.c3_proj, model.c4_proj]),
        "Fuse Block":               model.fuse,
        "MSAA Fusion":              model.msaa_fusion,
        "SK Context":               model.sk_context,
        "SE Context":               model.se_context,
        "Coord Attention":          model.coord_attn,
        "ECA Context":              model.eca_context,
        "EMA Context":              model.ema_context,
        "Edge Prompt":              nn.ModuleList([m for m in [model.edge_proj, model.edge_gate,
                                                               model.edge_extractor] if m is not None]),
        "Detail Fusion":            nn.ModuleList([m for m in [model.detail_proj, model.detail_gate,
                                                               model.detail_head] if m is not None]),
        "Seg Refine":               model.seg_refine,
        "CBAM Decoder":             model.cbam_decoder,
        "SimAM Refine":             model.simam_refine,
        "SAFM Decoder":             model.safm_decoder,
        "Triplet Decoder":          model.triplet_decoder,
        "Seg Head":                 model.seg_head,
        "Aux Head":                 model.aux_head,
        "Boundary Head":            nn.ModuleList([m for m in [model.boundary_fuse,
                                                               model.boundary_head] if m is not None]),
        "Uncertainty Head":         model.uncertainty_head,
        "Head-Guided Refine":       model.head_guided_refine,
        "Uncertainty-Gated Refine": model.uncertainty_gated_refine,
        "RABC Calibration Head":    model.rabc,
    }

    for name, module in components.items():
        if module is None or (isinstance(module, nn.Identity)):
            continue
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            print(f"  {name:<30s} {fmt(n)}")

    # ------------------------------------------------------------------ #
    # FLOPs
    # ------------------------------------------------------------------ #
    print(f"\n{'--- FLOPs Estimation ---':^72}")
    image = torch.randn(1, 3, 224, 224)
    pseudo = torch.randn(1, 4, 224, 224)

    # Image-only forward (pseudo=None)
    print("\n  [A] Image-only forward (pseudo=None):")
    result_img = measure_flops(model, image, None, torch.device("cpu"))
    if result_img:
        backend, flops = result_img
        print(f"      Backend: {backend}")
        print(f"      FLOPs:   {flops:,.0f}  ({flops / 1e9:.2f} GFLOPs)")
    else:
        print("      Could not compute FLOPs (no backend available)")

    # Full forward (with pseudo labels)
    print("\n  [B] Full forward (image + pseudo):")
    result_full = measure_flops(model, image, pseudo, torch.device("cpu"))
    if result_full:
        backend, flops = result_full
        print(f"      Backend: {backend}")
        print(f"      FLOPs:   {flops:,.0f}  ({flops / 1e9:.2f} GFLOPs)")
    else:
        print("      Could not compute FLOPs (no backend available)")

    # ConvNeXt-Tiny standalone
    print("\n  [C] ConvNeXt-Tiny backbone only:")
    bb_input = torch.randn(1, 3, 224, 224)
    result_bb = measure_flops(backbone, bb_input, None, torch.device("cpu"))
    if result_bb:
        backend, flops = result_bb
        print(f"      Backend: {backend}")
        print(f"      FLOPs:   {flops:,.0f}  ({flops / 1e9:.2f} GFLOPs)")
    else:
        print("      Could not compute FLOPs (no backend available)")

    # ------------------------------------------------------------------ #
    # Inference speed
    # ------------------------------------------------------------------ #
    print(f"\n{'--- Inference Speed ---':^72}")

    # CPU
    cpu_device = torch.device("cpu")
    print("\n  CPU (image-only, pseudo=None):")
    fps_cpu, ms_cpu = measure_fps(model, image, None, cpu_device, warmup=5, runs=20)
    print(f"      {fps_cpu:.2f} FPS  ({ms_cpu:.1f} ms/image)")

    print("  CPU (full forward, image+pseudo):")
    fps_cpu_full, ms_cpu_full = measure_fps(model, image, pseudo, cpu_device, warmup=5, runs=20)
    print(f"      {fps_cpu_full:.2f} FPS  ({ms_cpu_full:.1f} ms/image)")

    # GPU
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  GPU: {gpu_name}")

        model_gpu = RPINet(**MODEL_KWARGS).eval().to(gpu_device)

        print("  GPU (image-only, pseudo=None):")
        fps_gpu, ms_gpu = measure_fps(model_gpu, image, None, gpu_device, warmup=20, runs=100)
        print(f"      {fps_gpu:.2f} FPS  ({ms_gpu:.1f} ms/image)")

        print("  GPU (full forward, image+pseudo):")
        fps_gpu_full, ms_gpu_full = measure_fps(model_gpu, image, pseudo, gpu_device, warmup=20, runs=100)
        print(f"      {fps_gpu_full:.2f} FPS  ({ms_gpu_full:.1f} ms/image)")

        # GPU memory
        torch.cuda.reset_peak_memory_stats()
        _ = model_gpu(image.to(gpu_device), pseudo.to(gpu_device))
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n  Peak GPU memory (full forward): {peak_mem:.1f} MB")
    else:
        print("\n  GPU: not available, skipping GPU benchmarks")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 72}")
    print(f"{'SUMMARY TABLE':^72}")
    print(f"{'=' * 72}")
    print(f"  {'Metric':<40s} {'Value':>28s}")
    print(f"  {'-'*40:<40s} {'-'*28:>28s}")
    print(f"  {'RABC-Net Total Params':<40s} {total / 1e6:>24.2f} M")
    print(f"  {'RABC-Net Trainable Params':<40s} {trainable / 1e6:>24.2f} M")
    print(f"  {'ConvNeXt-Tiny Params':<40s} {bb_total / 1e6:>24.2f} M")
    print(f"  {'Decoder/Head Overhead':<40s} {overhead / 1e6:>24.2f} M")
    if result_img:
        print(f"  {'GFLOPs (image-only)':<40s} {result_img[1] / 1e9:>24.2f}")
    if result_full:
        print(f"  {'GFLOPs (image+pseudo)':<40s} {result_full[1] / 1e9:>24.2f}")
    if result_bb:
        print(f"  {'GFLOPs (ConvNeXt-Tiny only)':<40s} {result_bb[1] / 1e9:>24.2f}")
    print(f"  {'CPU FPS (image-only)':<40s} {fps_cpu:>24.2f}")
    print(f"  {'CPU FPS (image+pseudo)':<40s} {fps_cpu_full:>24.2f}")
    if torch.cuda.is_available():
        print(f"  {'GPU FPS (image-only)':<40s} {fps_gpu:>24.2f}")
        print(f"  {'GPU FPS (image+pseudo)':<40s} {fps_gpu_full:>24.2f}")
        print(f"  {'Peak GPU Mem (full fwd, MB)':<40s} {peak_mem:>24.1f}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
