#!/usr/bin/env bash
# ==============================================================================
# reproduce_isic2018.sh: End-to-end Reproduction Pipeline for ISIC 2018 (Table 5)
#
# Hardware Specification:
#   Benchmarked on NVIDIA GeForce RTX 4060 Ti (16 GB VRAM) under Ubuntu Linux.
#   Note: Slight numerical variations (±0.1% ~ 0.2% in DICE/JAC) may occur
#   across different GPU architectures, CUDA/cuDNN versions, and mixed-precision
#   implementations (e.g. BF16 vs FP16).
#
# Expected Performance (ISIC 2018 Test split, n=1,000):
#   ACC: ~91.93% | DICE: ~85.67% | JAC: ~79.14% | SEN: ~83.67% | SPE: ~97.85%
#
# Multi-Stage Adaptation Workflow:
#   Stage 1: Source-domain unsupervised pre-training on ISIC-2017.
#   Stage 2: Source-output distillation on ISIC-2018 using Stage 1 teacher.
#   Stage 3: Target-domain reliability adaptation with RABC on ISIC-2018.
#            WARNING: Stage 3 configs specify `freeze_segmentation_path: true`.
#            You MUST provide `--resume <checkpoint>` (the Stage 2 checkpoint),
#            otherwise the segmentation backbone remains randomly initialized
#            and completely frozen!
#   Stage 4: Evaluation at validation-selected operating point:
#            --threshold 0.25 --tta flip4 --keep-largest --fill-holes
# ==============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

# Config paths
CFG_STAGE1="configs/rpi_net_isic2017_unsup_v17_msaa.yaml"
CFG_STAGE2="configs/rpi_net_isic2018_unsup_v19_distill.yaml"
CFG_STAGE3="configs/rpi_net_isic2018_innovation_rabc.yaml"

# Default checkpoint paths
CKPT_STAGE1="${STAGE1_CKPT:-outputs/rpi_net_isic2017_unsup_v17_msaa/checkpoints/best.pt}"
CKPT_STAGE2="${STAGE2_CKPT:-outputs/rpi_net_isic2018_unsup_v19_distill/checkpoints/best.pt}"
CKPT_STAGE3="${STAGE3_CKPT:-outputs/innovation/rpi_net_isic2018_rabc/checkpoints/best.pt}"

log_info() {
    echo -e "\033[1;34m[INFO] $(date '+%Y-%m-%d %H:%M:%S')\033[0m $*"
}

log_warn() {
    echo -e "\033[1;33m[WARN] $(date '+%Y-%m-%d %H:%M:%S')\033[0m $*"
}

log_success() {
    echo -e "\033[1;32m[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S')\033[0m $*"
}

log_error() {
    echo -e "\033[1;31m[ERROR] $(date '+%Y-%m-%d %H:%M:%S')\033[0m $*" >&2
}

print_usage() {
    cat <<EOF
Usage: $0 [STAGE]

Stages:
  --all (default)  Run complete pipeline (Stage 1 -> Stage 2 -> Stage 3 -> Evaluation)
  stage1           Run Stage 1: ISIC-2017 unsupervised pre-training
  stage2           Run Stage 2: ISIC-2018 source-output distillation (requires Stage 1 checkpoint)
  stage3           Run Stage 3: ISIC-2018 target adaptation with RABC (requires Stage 2 checkpoint)
  eval             Run test evaluation at validation-selected operating point
  --help, -h       Show this help message

Environment variables for checkpoint overrides:
  STAGE1_CKPT      Path to Stage 1 checkpoint (default: ${CKPT_STAGE1})
  STAGE2_CKPT      Path to Stage 2 checkpoint (default: ${CKPT_STAGE2})
  STAGE3_CKPT      Path to Stage 3 checkpoint (default: ${CKPT_STAGE3})

Example:
  $0 --all
  STAGE2_CKPT=checkpoints/distill_best.pt $0 stage3
  STAGE3_CKPT=checkpoints/rabc_best.pt $0 eval
EOF
}

# ------------------------------------------------------------------------------
# Stage 1: Source Pre-training on ISIC-2017
# ------------------------------------------------------------------------------
run_stage1() {
    log_info "================================================================="
    log_info "Stage 1: Unsupervised Source Pre-training on ISIC-2017"
    log_info "================================================================="

    log_info "Generating offline multi-source pseudo-labels for ISIC-2017 train split..."
    python3 scripts/generate_pseudo_labels.py \
        --config "$CFG_STAGE1" \
        --split train

    log_info "Training RABC-Net foundation model on ISIC-2017..."
    python3 scripts/train_rpi_net.py \
        --config "$CFG_STAGE1"

    if [[ ! -f "$CKPT_STAGE1" ]]; then
        log_error "Stage 1 checkpoint not found at: $CKPT_STAGE1"
        exit 1
    fi
    log_success "Stage 1 completed successfully. Checkpoint saved at: $CKPT_STAGE1"
}

# ------------------------------------------------------------------------------
# Stage 2: Knowledge Distillation on ISIC-2018
# ------------------------------------------------------------------------------
run_stage2() {
    log_info "================================================================="
    log_info "Stage 2: Source-Output Distillation on ISIC-2018"
    log_info "================================================================="

    if [[ ! -f "$CKPT_STAGE1" ]]; then
        log_error "Stage 1 teacher checkpoint not found: $CKPT_STAGE1"
        log_error "Please run Stage 1 first, or set STAGE1_CKPT to an existing checkpoint."
        exit 1
    fi

    log_info "Generating offline multi-source pseudo-labels for ISIC-2018 train split..."
    python3 scripts/generate_pseudo_labels.py \
        --config "$CFG_STAGE2" \
        --split train

    log_info "Running distillation training on ISIC-2018 with Stage 1 teacher: $CKPT_STAGE1..."
    python3 scripts/train_rpi_net.py \
        --config "$CFG_STAGE2"

    if [[ ! -f "$CKPT_STAGE2" ]]; then
        log_error "Stage 2 checkpoint not found at: $CKPT_STAGE2"
        exit 1
    fi
    log_success "Stage 2 completed successfully. Checkpoint saved at: $CKPT_STAGE2"
}

# ------------------------------------------------------------------------------
# Stage 3: Target Domain Adaptation (RABC) on ISIC-2018
# ------------------------------------------------------------------------------
run_stage3() {
    log_info "================================================================="
    log_info "Stage 3: Target Domain Adaptation with RABC on ISIC-2018"
    log_info "================================================================="

    if [[ ! -f "$CKPT_STAGE2" ]]; then
        log_error "Stage 2 initialization checkpoint not found: $CKPT_STAGE2"
        log_error "Please run Stage 2 first, or set STAGE2_CKPT to an existing checkpoint."
        exit 1
    fi

    log_warn "CRITICAL NOTE: $CFG_STAGE3 sets 'freeze_segmentation_path: true'."
    log_warn "Resuming from Stage 2 checkpoint ($CKPT_STAGE2) is MANDATORY to transfer learned weights."
    log_warn "Without --resume, the backbone would remain randomly initialized and frozen!"

    log_info "Generating offline pseudo-labels for target adaptation..."
    python3 scripts/generate_pseudo_labels.py \
        --config "$CFG_STAGE3" \
        --split train

    log_info "Adapting reliability branch and RABC head on ISIC-2018..."
    python3 scripts/train_rpi_net.py \
        --config "$CFG_STAGE3" \
        --resume "$CKPT_STAGE2"

    if [[ ! -f "$CKPT_STAGE3" ]]; then
        log_error "Stage 3 checkpoint not found at: $CKPT_STAGE3"
        exit 1
    fi
    log_success "Stage 3 completed successfully. Checkpoint saved at: $CKPT_STAGE3"
}

# ------------------------------------------------------------------------------
# Stage 4: Test Evaluation at Validation-Selected Operating Point
# ------------------------------------------------------------------------------
run_eval() {
    log_info "================================================================="
    log_info "Evaluation: Testing ISIC-2018 Checkpoint at Validated Operating Point"
    log_info "================================================================="

    if [[ ! -f "$CKPT_STAGE3" ]]; then
        log_error "Target checkpoint not found: $CKPT_STAGE3"
        log_error "Please run Stage 3 first, or set STAGE3_CKPT to an existing checkpoint."
        exit 1
    fi

    log_info "Evaluating on ISIC-2018 test split (n=1,000)..."
    log_info "Operating Point: --threshold 0.25 --tta flip4 --keep-largest --fill-holes"
    log_info "Expected Metrics (Table 5): DICE ~85.67%, JAC ~79.14%, ACC ~91.93%"

    python3 scripts/evaluate_rpi_net.py \
        --config "$CFG_STAGE3" \
        --checkpoint "$CKPT_STAGE3" \
        --split test \
        --threshold 0.25 \
        --tta flip4 \
        --keep-largest \
        --fill-holes \
        --compute-distance

    log_success "Evaluation finished. Check the output above for final metrics."
}

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
TARGET="${1:---all}"

case "$TARGET" in
    stage1)
        run_stage1
        ;;
    stage2)
        run_stage2
        ;;
    stage3)
        run_stage3
        ;;
    eval)
        run_eval
        ;;
    --all)
        log_info "Starting full ISIC-2018 reproduction pipeline..."
        run_stage1
        run_stage2
        run_stage3
        run_eval
        log_success "All stages finished successfully!"
        ;;
    -h|--help|help)
        print_usage
        exit 0
        ;;
    *)
        log_error "Unknown stage or option: $TARGET"
        print_usage
        exit 2
        ;;
esac
