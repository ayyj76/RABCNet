# RABC-Net: Reliability-Adaptive Boundary Calibration for Annotation-Free Skin Lesion Segmentation

Official PyTorch implementation of **RABC-Net**, a reliability-aware annotation-free skin lesion segmentation framework for dermoscopy images.

RABC-Net addresses the challenges of low-resource medical image segmentation where manual pixel-level annotations are unavailable or costly to obtain. By integrating multi-source unsupervised pseudo-labeling, reliability-guided interaction, and lightweight boundary calibration, RABC-Net achieves state-of-the-art segmentation performance while maintaining a pure image-only deployment path during clinical inference.

---

## Key Highlights & Core Contributions

1. **Multi-Source Unsupervised Pseudo-Labeling**:
   Constructs a 4-channel spatial prior tensor from complementary foundation representations (self-supervised and contrastive vision features) without requiring manual training masks.
2. **Reliability-Guided Interaction & Uncertainty Weighting**:
   Builds upon Inter-Pixel Cross-correlation (IPC) and Pixel-Interaction Aggregation (PIA) with Dual Uncertainty Learning (DUL) to dynamically discount noisy pseudo-label regions and form a robust consensus.
3. **Reliability-Adaptive Boundary Calibration (RABC)**:
   Introduces a lightweight local logit calibration module (constituting only ~3.50% of the total network parameters) that adaptively sharpens and calibrates uncertain boundary transitions.
4. **Image-Only Inference Deployment**:
   During inference, the model strictly requires only RGB dermoscopy images as input. All multi-source pseudo-label pipelines and auxiliary reliability branches are decoupled, ensuring fast, practical clinical deployment.

---

## Benchmark Results

Evaluation on standard public dermoscopy benchmarks across test splits under validation-assisted operating points:

| Benchmark | Test Split | Method | ACC (%) | DICE (%) | JAC (%) | SEN (%) | SPE (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **ISIC-2018** | $n=1{,}000$ | USL-Net | 88.45 | 78.79 | 68.32 | **90.91** | 87.76 |
| | | SLED | 86.93 | 77.68 | 69.35 | 80.12 | 91.84 |
| | | RPI-Net anchor | 90.21 | 82.88 | 74.14 | 87.17 | 94.05 |
| | | **RABC-Net (Ours)** | **91.93** | **85.67** | **79.14** | 83.67 | **97.85** |
| **ISIC-2017** | $n=600$ | USL-Net | 90.47 | 80.45 | 68.45 | **88.59** | 93.68 |
| | | SLED | 88.81 | 73.90 | 64.45 | 77.96 | 94.61 |
| | | RPI-Net anchor | 87.62 | 75.84 | 66.21 | 84.15 | 91.73 |
| | | **RABC-Net (Ours)** | **92.08** | **81.90** | **72.86** | 81.57 | **97.21** |
| **PH2** | $n=30$ | USL-Net | 92.43 | 88.90 | 80.11 | 93.62 | 93.07 |
| | | SLED | 93.00 | 90.34 | 83.50 | 89.97 | 96.37 |
| | | RPI-Net anchor | 93.25 | 88.91 | 81.98 | 93.66 | 94.60 |
| | | **RABC-Net (Ours)** | **94.97** | **92.17** | **86.41** | **94.70** | 94.07 |

*Note: Results correspond to published literature positioning protocols. Manual masks are never utilized during representation training or unsupervised target adaptation.*

---

## Hardware Specifications & Numerical Variations

- **Reference Benchmark Platform:** NVIDIA GeForce RTX 4060 Ti (16 GB VRAM), AMD Ryzen / Intel Core i7 CPU, 32 GB RAM, Ubuntu 22.04 LTS.
- **Software Environment:** PyTorch $\ge$ 2.0 with CUDA 12.x and Brain Floating Point (`amp_dtype: bf16`) automatic mixed precision.
- **Numerical Variations:** Due to variations across GPU architectures (Tensor Core precision differences between Ada Lovelace, Ampere, and Volta), cuDNN non-deterministic algorithm selection, and mixed-precision implementations (BF16 vs. FP16), slight numerical variations of approximately **$\pm0.1\%$ to $\pm0.2\%$** in DICE/JAC are expected across different hardware platforms.

---

## Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/ayyj76/RABCNet.git
cd RABCNet
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## Data Preparation

Datasets are organized under `datasets/` (or customized via YAML configs):

```text
datasets/
├── ISIC2017/
│   ├── ISIC-2017_Training_Data/
│   ├── ISIC-2017_Validation_Data/
│   └── ISIC-2017_Test_v2_Data/
├── ISIC2018/
│   ├── ISIC2018_Task1-2_Training_Input/
│   ├── ISIC2018_Task1-2_Validation_Input/
│   └── ISIC2018_Task1-2_Test_Input/
└── PH2/
    └── PH2Dataset/
```

Images and split definitions are configured in the corresponding YAML files under `configs/`.

---

## Offline Pseudo-Label Generation

Before model training, generate multi-source pseudo-labels from unsupervised feature extractors:

```bash
# Source domain (ISIC-2017)
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2017_unsup_v17_msaa.yaml \
  --split train

# Target domain (ISIC-2018)
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_isic2018_unsup_v19_distill_short.yaml \
  --split train

# Target domain (PH2)
PYTHONPATH=src python3 scripts/generate_pseudo_labels.py \
  --config configs/rpi_net_ph2_innovation_rabc.yaml \
  --split train
```

---

## Training & Multi-Stage Adaptation Pipeline

RABC-Net employs a multi-stage progressive learning and adaptation workflow to transfer clean source-domain representations to target datasets while preventing representation drift from noisy target pseudo-labels:

```text
[Stage 1: Source Domain Representation Learning]
  Dataset: ISIC-2017 (train, unsupervised)
  Config:  configs/rpi_net_isic2017_unsup_v17_msaa.yaml
       │
       ▼ (outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt)
[Stage 2: Cross-Domain Teacher Distillation]
  Dataset: ISIC-2018 (train, unsupervised) + Stage 1 Teacher
  Config:  configs/rpi_net_isic2018_unsup_v19_distill_short.yaml
       │
       ▼ (outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt)
[Stage 3: Target Reliability & Boundary Adaptation (RABC)]
  Dataset: ISIC-2018 (train, unsupervised, backbone frozen)
  Config:  configs/rpi_net_isic2018_innovation_rabc.yaml
       │
       ▼ (outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt)
[Stage 4: Validation-Selected Operating Point Evaluation]
  Test split: --threshold 0.25 --tta flip4 --keep-largest --fill-holes
```

### Stage 1: Source Domain Representation Learning
Train the foundation segmentation model on the source domain (ISIC-2017) using multi-source pseudo-labels and iterative refinement:

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017_unsup_v17_msaa.yaml
```
Output checkpoint: `outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt`.

### Stage 2: Cross-Domain Teacher Distillation
Distill the segmentation prior from the frozen Stage 1 source teacher to the target domain (ISIC-2018). We employ a compact distillation schedule (`v19_distill_short.yaml`: 6 epochs, teacher retention weight `source_distill_weight: 0.80`, no self-training feedback rounds) to transfer clean source features without letting noisy target pseudo-labels drift the representation:

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2018_unsup_v19_distill_short.yaml
```
Output checkpoint: `outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt`.

### Stage 3: Target Reliability & Boundary Adaptation (RABC)

> [!IMPORTANT] **Backbone Freezing & Mandatory `--resume` Flag**
> The target adaptation configuration `configs/rpi_net_isic2018_innovation_rabc.yaml` enforces `freeze_segmentation_path: true`. This intentionally freezes the ConvNeXt-Tiny encoder, decoder, and primary segmentation heads to prevent noisy target pseudo-labels from corrupting the established image-to-mask mapping, restricting gradient updates strictly to the reliability branch and the lightweight RABC calibration head (only ~3.50% of model parameters).
> 
> Consequently, **you MUST provide `--resume outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt`** (or specify it via `resume_checkpoint` in the YAML). If `--resume` is omitted, the segmentation backbone remains randomly initialized and frozen throughout training, which prevents the network from learning valid representations.

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2018_innovation_rabc.yaml \
  --resume outputs/rpi_net_isic2018_unsup_v19_distill_short/checkpoints/best.pt
```
Output checkpoint: `outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt`.

---

## Inference & Post-Processing Optimization

### 1. Operating Point Selection via Validation Search
Optimal decision thresholds and morphological post-processing parameters are discovered through validation search:

```bash
PYTHONPATH=src python3 scripts/select_inference_by_val.py \
  --config configs/rpi_net_isic2018_innovation_rabc.yaml \
  --checkpoint outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt \
  --output outputs/innovation/rpi_net_isic2018_rabc/select_val.json
```

### 2. Test Split Evaluation
Evaluate the adapted model on the test split using the discovered optimal operating point:

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

**Operating Point Configuration:**
- `--threshold 0.25`: Optimal probability decision threshold determined on the validation set.
- `--tta flip4`: Four-fold test-time augmentation (original, horizontal flip, vertical flip, dual flip) to stabilize edge predictions.
- `--keep-largest`: Retains the dominant connected foreground component, eliminating isolated spurious artifacts.
- `--fill-holes`: Binary morphological hole filling for contiguous lesion delineation.
- `--compute-distance`: Computes boundary distance metrics (ASSD, HD95).

---

## End-to-End Automated Pipeline

For convenience, [`scripts/reproduce_isic2018.sh`](scripts/reproduce_isic2018.sh) encapsulates the complete end-to-end multi-stage pipeline with automated error checking and logging:

```bash
# Run the entire pipeline end-to-end:
bash scripts/reproduce_isic2018.sh --all

# Or run individual stages modularly:
bash scripts/reproduce_isic2018.sh stage1   # Stage 1: ISIC-2017 source pre-training
bash scripts/reproduce_isic2018.sh stage2   # Stage 2: ISIC-2018 teacher distillation
bash scripts/reproduce_isic2018.sh stage3   # Stage 3: ISIC-2018 target RABC adaptation
bash scripts/reproduce_isic2018.sh eval     # Evaluation at optimal operating point

# Checkpoints can be overridden via environment variables if desired:
STAGE2_CKPT=path/to/stage2_distill.pt bash scripts/reproduce_isic2018.sh stage3
STAGE3_CKPT=path/to/stage3_rabc.pt bash scripts/reproduce_isic2018.sh eval
```

---

## Repository Structure & Config Families

```text
├── configs/
│   ├── rpi_net_isic2017_unsup_v17_msaa.yaml         # Stage 1 foundation pre-training
│   ├── rpi_net_isic2018_unsup_v19_distill_short.yaml# Stage 2 source-to-target distillation
│   ├── rpi_net_isic2018_innovation_rabc.yaml        # Stage 3 target RABC adaptation
│   ├── rpi_net_isic2017_innovation_rabc.yaml        # Source RABC adaptation
│   ├── rpi_net_ph2_innovation_rabc.yaml             # PH2 RABC adaptation
│   ├── rpi_net_isic2017_smoke.yaml                  # Quick smoke run
│   ├── generated/                                   # Multi-seed experiment configs
│   └── ablation_decoder/                            # Decoder component ablation configs
├── src/rpi_net/
│   ├── __init__.py
│   ├── datasets.py      # Dataset loaders and transforms
│   ├── losses.py        # Uncertainty, boundary, and distillation losses
│   ├── metrics.py       # Segmentation evaluation metrics
│   ├── model.py         # ConvNeXt encoder, decoders, IPC/PIA, and RABC calibration modules
│   ├── preprocess.py    # Preprocessing pipelines and hair artifact removal
│   ├── pseudo_labels.py # Multi-source pseudo-label generators and consensus logic
│   └── utils.py         # Logging, checkpointing, config loading, and utilities
└── scripts/             # Execution workflows, training, evaluation, complexity, sweeps, and plotting tools
```

### Quick Smoke Test
To verify the training and evaluation loop on minimal iterations:

```bash
PYTHONPATH=src python3 scripts/train_rpi_net.py \
  --config configs/rpi_net_isic2017_smoke.yaml
```

---

## License & Usage Notes

- Manual segmentation masks are never used for training supervision in the annotation-free pipeline.
- Validation labels, when used, are strictly for operating-point selection and final evaluation.
- The repository excludes datasets and checkpoints. Users should download public datasets from their official sources and configure paths in the YAML configs.
- For academic and research use.

---

## Citation

If you find this work or code useful in your research, please cite:

```bibtex
@article{rabcnet2025,
  title={RABC-Net: Reliability-Adaptive Boundary Calibration for Annotation-Free Skin Lesion Segmentation},
  author={...},
  journal={...},
  year={2025}
}
```
