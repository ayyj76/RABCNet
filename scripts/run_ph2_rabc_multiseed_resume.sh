#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="outputs/innovation/logs/ph2_rabc_multiseed_resume"
mkdir -p "$LOG_DIR"

export PYTHONPATH=src
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

run_one() {
  local cfg="$1"
  local resume="$2"
  local name
  name="$(basename "${cfg%.yaml}")"
  local log_path="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] start ${cfg}" | tee "$log_path"
  echo "[$(date '+%F %T')] resume ${resume}" | tee -a "$log_path"
  python3 scripts/train_rpi_net.py --config "$cfg" --resume "$resume" 2>&1 | tee -a "$log_path"
  echo "[$(date '+%F %T')] done ${cfg}" | tee -a "$log_path"
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    norabc_s41)
      run_one "configs/generated/rpi_net_ph2_unsup_v20_sigma_only_s41.yaml" \
        "outputs/rpi_net_ph2_unsup_v20_sigma_only/checkpoints/best.pt"
      ;;
    norabc_s42)
      run_one "configs/generated/rpi_net_ph2_unsup_v20_sigma_only_s42.yaml" \
        "outputs/rpi_net_ph2_unsup_v20_sigma_only/checkpoints/best.pt"
      ;;
    norabc_s43)
      run_one "configs/generated/rpi_net_ph2_unsup_v20_sigma_only_s43.yaml" \
        "outputs/rpi_net_ph2_unsup_v20_sigma_only/checkpoints/best.pt"
      ;;
    rabc_s41)
      run_one "configs/generated/rpi_net_ph2_innovation_rabc_s41.yaml" \
        "outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt"
      ;;
    rabc_s42)
      run_one "configs/generated/rpi_net_ph2_innovation_rabc_s42.yaml" \
        "outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt"
      ;;
    rabc_s43)
      run_one "configs/generated/rpi_net_ph2_innovation_rabc_s43.yaml" \
        "outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt"
      ;;
    *)
      echo "unknown target: $1" >&2
      exit 2
      ;;
  esac
  exit 0
fi

run_one "configs/generated/rpi_net_ph2_unsup_v20_sigma_only_s41.yaml" \
  "outputs/rpi_net_ph2_unsup_v20_sigma_only/checkpoints/best.pt"
run_one "configs/generated/rpi_net_ph2_unsup_v20_sigma_only_s42.yaml" \
  "outputs/rpi_net_ph2_unsup_v20_sigma_only/checkpoints/best.pt"
run_one "configs/generated/rpi_net_ph2_unsup_v20_sigma_only_s43.yaml" \
  "outputs/rpi_net_ph2_unsup_v20_sigma_only/checkpoints/best.pt"
run_one "configs/generated/rpi_net_ph2_innovation_rabc_s41.yaml" \
  "outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt"
run_one "configs/generated/rpi_net_ph2_innovation_rabc_s42.yaml" \
  "outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt"
run_one "configs/generated/rpi_net_ph2_innovation_rabc_s43.yaml" \
  "outputs/innovation/rpi_net_ph2_rabc/checkpoints/best.pt"
