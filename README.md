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
