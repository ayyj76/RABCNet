#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="outputs/innovation/logs/isic2018_rabc_multiseed"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/generated/rpi_net_isic2018_unsup_v20_sigma_only_s41.yaml"
  "configs/generated/rpi_net_isic2018_unsup_v20_sigma_only_s42.yaml"
  "configs/generated/rpi_net_isic2018_unsup_v20_sigma_only_s43.yaml"
  "configs/generated/rpi_net_isic2018_innovation_rabc_s41.yaml"
  "configs/generated/rpi_net_isic2018_innovation_rabc_s42.yaml"
  "configs/generated/rpi_net_isic2018_innovation_rabc_s43.yaml"
)

export PYTHONPATH=src
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "${cfg%.yaml}")"
  log_path="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] start ${cfg}" | tee "$log_path"
  python3 scripts/train_rpi_net.py --config "$cfg" 2>&1 | tee -a "$log_path"
  echo "[$(date '+%F %T')] done ${cfg}" | tee -a "$log_path"
done
