# RABC-Net

This repository provides the code release for **RABC-Net**, a reliability-aware annotation-free skin lesion segmentation framework for low-resource dermoscopy.

RABC-Net is designed around three practical goals:

- learning from multiple unsupervised pseudo-label sources without manual training masks;
- stabilizing noisy pseudo-supervision through reliability-aware pseudo-label interaction;
- preserving an image-only deployment path while using lightweight Reliability-Adaptive Boundary Calibration (RABC) for local boundary correction.

## Repository Structure

```text
src/rpi_net/        Core package: datasets, pseudo-labels, models, losses, metrics, utilities
scripts/            Training, evaluation, pseudo-label generation, sweeps, and reporting scripts
configs/            YAML experiment configurations
draw/               Lightweight plotting/evaluation utilities used for manuscript figures
requirements.txt    Python dependencies
```

Large local artifacts are intentionally excluded from version control, including datasets, checkpoints, runtime outputs, generated manuscript files, and local tool state.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code expects the package root to be available through `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

## Data Layout

Datasets are not included in this repository. Place the dermoscopy datasets under `datasets/` or update the corresponding paths in the YAML configuration files.

A typical local layout is:

```text
datasets/
  ISIC2017/
  ISIC2018/
  PH2/
```

The exact image and mask paths are controlled by the selected config file.

## Pseudo-Label Generation

Generate offline multi-source pseudo-labels before training:

```bash
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2017.yaml \
  --split train
```

For target-domain adaptation or evaluation protocols, use the corresponding dataset config, for example `configs/rpi_net_isic2018_innovation_rabc.yaml` or `configs/rpi_net_ph2_innovation_rabc.yaml`.

## Training

Run a full training configuration:

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017_innovation_rabc.yaml
```

For a quick smoke run:

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017_smoke.yaml
```

Training outputs are written under `outputs/` by default. This directory is ignored by Git.

## Evaluation

Evaluate a checkpoint on a selected split:

```bash
PYTHONPATH=src python3 scripts/evaluate_rpi_net.py \
  --config configs/rpi_net_isic2017_innovation_rabc.yaml \
  --checkpoint outputs/<run_name>/checkpoints/best.pt \
  --split test
```

Validation-selected operating-point search can be performed with:

```bash
PYTHONPATH=src python3 scripts/select_inference_by_val.py \
  --config configs/rpi_net_isic2017_innovation_rabc.yaml \
  --checkpoint outputs/<run_name>/checkpoints/best.pt
```

## Reproducing ISIC 2018 Results (Table 5)

This section provides step-by-step instructions and automated commands to reproduce the validation-assisted literature benchmark results on **ISIC 2018** (test split, $n=1{,}000$) as reported in Table 5 of the manuscript:

| Method | ACC | DICE | JAC | SEN | SPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RABC-Net (Proposed full system)** | **91.93** | **85.67** | **79.14** | 83.67 | **97.85** |

### Hardware Specifications & Numerical Variations

- **Benchmarked Hardware:** NVIDIA GeForce RTX 4060 Ti (16 GB VRAM), Intel Core i7 / AMD Ryzen CPU, 32 GB RAM, Ubuntu 22.04 LTS Linux.
- **Environment:** PyTorch $\ge$ 2.0 with CUDA 12.x and Brain Floating Point (`amp_dtype: bf16`) automatic mixed precision.
- **Numerical Variations:** Due to variations across GPU architectures (Tensor Core precision differences between Ada Lovelace, Ampere, and Volta), cuDNN non-deterministic algorithm selection, and mixed-precision implementations (BF16 vs. FP16), slight numerical variations of approximately **$\pm0.1\%$ to $\pm0.2\%$** in DICE/JAC are expected across different hardware environments.

### Multi-Stage Adaptation Workflow

The unsupervised adaptation to ISIC-2018 is organized into three sequential stages:

```text
[Stage 1: Source Pre-training]
  Dataset: ISIC-2017 (train, unsupervised)
  Config:  configs/rpi_net_isic2017_unsup_v17_msaa.yaml
       │
       ▼ (outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt)
[Stage 2: Source Distillation]
  Dataset: ISIC-2018 (train, unsupervised) + Stage 1 Teacher
  Config:  configs/rpi_net_isic2018_unsup_v19_distill_short.yaml
       │
       ▼ (outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt)
[Stage 3: Target Reliability Adaptation]
  Dataset: ISIC-2018 (train, unsupervised) with RABC
  Config:  configs/rpi_net_isic2018_innovation_rabc.yaml
       │
       ▼ (outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt)
[Stage 4: Validation-Selected Operating Point Evaluation]
  Test split (n=1,000): --threshold 0.25 --tta flip4 --keep-largest --fill-holes
```

#### Stage 1: ISIC-2017 Source Pre-training
Train the foundational representation on the source domain (ISIC-2017) using multi-source pseudo-labels and iterative refinement:
```bash
# 1. Generate multi-source pseudo-labels
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2017_unsup_v17_msaa.yaml \
  --split train

# 2. Train Stage 1 model from scratch (24 epochs, 1 iterative round)
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017_unsup_v17_msaa.yaml
```
Output checkpoint: `outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt`.

#### Stage 2: ISIC-2018 Source-Output Distillation
Distill the segmentation prior from the frozen Stage 1 source teacher to the ISIC-2018 target domain. We employ the compact distillation schedule (`v19_distill_short.yaml`: 6 epochs, strong teacher retention `source_distill_weight: 0.80`, no self-training feedback rounds) to transfer clean source features without letting noisy target pseudo-labels drift the representation:
```bash
# 1. Generate pseudo-labels for ISIC-2018
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2018_unsup_v19_distill_short.yaml \
  --split train

# 2. Train distillation model on ISIC-2018 (teacher loaded automatically)
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2018_unsup_v19_distill_short.yaml
```
Output checkpoint: `outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt`.

#### Stage 3: Target Domain Adaptation with RABC

> [!WARNING] **CRITICAL NOTICE: Mandatory `--resume` Flag**
> Configuration `configs/rpi_net_isic2018_innovation_rabc.yaml` enforces `freeze_segmentation_path: true`. This intentionally freezes the ConvNeXt-Tiny encoder, decoder, and segmentation heads to prevent noisy target pseudo-labels from corrupting the deployed image-to-mask mapping, restricting gradient updates strictly to the reliability branch and the lightweight RABC calibration head (only 3.50% of model parameters).
> 
> **You MUST provide `--resume outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt`** (or ensure `resume_checkpoint` in the YAML points to it). If `--resume` is omitted, the segmentation backbone remains randomly initialized and frozen throughout training, leading to complete training failure (near 0% DICE).

```bash
# 1. Generate pseudo-labels for adaptation
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2018_innovation_rabc.yaml \
  --split train

# 2. Adapt reliability branch and RABC head (MUST pass --resume)
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2018_innovation_rabc.yaml \
  --resume outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt
```
Output checkpoint: `outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt`.

#### Stage 4: Test Evaluation at Validation-Selected Operating Point

Evaluate the adapted model on the ISIC-2018 test split ($n=1{,}000$) using the optimal validation-selected operating point:

```bash
PYTHONPATH=src python3 scripts/evaluate_rpi_net.py \
  --config configs/rpi_net_isic2018_innovation_rabc.yaml \
  --checkpoint outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt \
  --split test \
  --threshold 0.25 \
  --tta flip4 \
  --keep-largest \
  --fill-holes \
  --compute-distance
```

**Operating Point Components:**
- `--threshold 0.25`: Optimal probability decision threshold discovered on the validation split via `scripts/select_inference_by_val.py`.
- `--tta flip4`: Four-fold test-time augmentation (original, horizontal flip, vertical flip, and dual flip) to stabilize edge predictions.
- `--keep-largest`: Retains the dominant connected foreground component, eliminating isolated spurious artifacts.
- `--fill-holes`: Binary morphological hole filling for contiguous lesion delineation.

Expected test performance: **DICE $\sim 85.67\%$, JAC $\sim 79.14\%$, ACC $\sim 91.93\%$**.

---

### Automated Reproduction Script

For convenience, the automated script [`scripts/reproduce_isic2018.sh`](scripts/reproduce_isic2018.sh) encapsulates the entire pipeline with error checking and stage logging:

```bash
# Run the entire pipeline end-to-end:
bash scripts/reproduce_isic2018.sh --all

# Or run individual stages modularly:
bash scripts/reproduce_isic2018.sh stage1
bash scripts/reproduce_isic2018.sh stage2
bash scripts/reproduce_isic2018.sh stage3
bash scripts/reproduce_isic2018.sh eval

# Checkpoints can be overridden via environment variables if needed:
STAGE2_CKPT=/path/to/stage2_distill.pt bash scripts/reproduce_isic2018.sh stage3
```


## Reproducing Main Experiment Families

The main RABC experiment configs are:

```text
configs/rpi_net_isic2017_innovation_rabc.yaml
configs/rpi_net_isic2018_innovation_rabc.yaml
configs/rpi_net_ph2_innovation_rabc.yaml
```

Additional generated seed configs are stored in:

```text
configs/generated/
```

Decoder ablation configs are stored in:

```text
configs/ablation_decoder/
```

## Notes

- Manual segmentation masks are not used for training supervision in the annotation-free pipeline.
- Validation labels, when used, are only for operating-point selection and final evaluation.
- The repository excludes datasets and checkpoints. Users should download public datasets from their official sources and set paths in the YAML configs.
- Some scripts support long-running experiments and assume sufficient GPU memory and disk space.

## Citation

If you use this code, please cite the associated manuscript once it becomes available.
