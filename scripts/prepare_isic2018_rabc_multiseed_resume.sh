#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/make_seed_variants.py \
  --config configs/rpi_net_isic2018_unsup_v20_sigma_only.yaml \
  --seeds 41 42 43 \
  --out-dir configs/generated \
  --output-prefix outputs/innovation/isic2018_norabc_multiseed_resume/rpi_net_isic2018_norabc_resume

python3 scripts/make_seed_variants.py \
  --config configs/rpi_net_isic2018_innovation_rabc.yaml \
  --seeds 41 42 43 \
  --out-dir configs/generated \
  --output-prefix outputs/innovation/isic2018_rabc_multiseed_resume/rpi_net_isic2018_rabc_resume

