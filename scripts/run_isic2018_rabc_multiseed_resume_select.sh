#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src

for seed in 41 42 43; do
  python3 scripts/select_inference_by_val.py \
    --config "configs/generated/rpi_net_isic2018_unsup_v20_sigma_only_s${seed}.yaml" \
    --checkpoint "outputs/innovation/isic2018_norabc_multiseed_resume/rpi_net_isic2018_norabc_resume_s${seed}/checkpoints/best.pt" \
    --threshold-min 0.09 \
    --threshold-max 0.31 \
    --threshold-step 0.02 \
    --tta-modes none,flip4 \
    --min-areas 0 \
    --keep-largest-options false,true \
    --fill-holes-options false,true \
    --dilation-iters-list 0 \
    --adaptive-scales 0.0 \
    --gaussian-sigmas 0.0 \
    --objective jac \
    --output "outputs/innovation/isic2018_norabc_multiseed_resume/rpi_net_isic2018_norabc_resume_s${seed}/select_jac.json"

  python3 scripts/select_inference_by_val.py \
    --config "configs/generated/rpi_net_isic2018_innovation_rabc_s${seed}.yaml" \
    --checkpoint "outputs/innovation/isic2018_rabc_multiseed_resume/rpi_net_isic2018_rabc_resume_s${seed}/checkpoints/best.pt" \
    --threshold-min 0.09 \
    --threshold-max 0.31 \
    --threshold-step 0.02 \
    --tta-modes none,flip4 \
    --min-areas 0 \
    --keep-largest-options false,true \
    --fill-holes-options false,true \
    --dilation-iters-list 0 \
    --adaptive-scales 0.0 \
    --gaussian-sigmas 0.0 \
    --objective jac \
    --output "outputs/innovation/isic2018_rabc_multiseed_resume/rpi_net_isic2018_rabc_resume_s${seed}/select_jac.json"
done
