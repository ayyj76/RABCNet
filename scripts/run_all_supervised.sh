#!/usr/bin/env bash
# run_all_supervised.sh
# 串行训练 6 模型 × 3 数据集，共 18 组
# batch size 按显存合理最大化（RTX 4060 Ti 16GB）
set -e
cd "$(dirname "$0")/.."

LOG_DIR="outputs/supervised/logs"
mkdir -p "$LOG_DIR"

run() {
    local model=$1
    local dataset=$2
    local data_root=$3
    local bs=$4
    local out="outputs/supervised/${model}_${dataset}"
    local log="$LOG_DIR/${model}_${dataset}.log"

    echo "════════════════════════════════════════"
    echo "▶  $model / $dataset  bs=$bs"
    echo "   输出: $out"
    echo "   日志: $log"
    echo "════════════════════════════════════════"

    python scripts/train_supervised.py \
        --model "$model" \
        --data-root "$data_root" \
        --output-dir "$out" \
        --epochs 50 \
        --batch-size "$bs" \
        --lr 1e-4 \
        --num-workers 4 \
        --patience 10 \
        2>&1 | tee "$log"

    echo "✔  $model / $dataset 完成"
    echo ""
}

# ── ISIC-2017 ────────────────────────────────────────────────────────
run unet             isic2017 datasets/ISIC2017 32
run unetplusplus     isic2017 datasets/ISIC2017 24
run attdeeplabv3plus isic2017 datasets/ISIC2017 24
run transfuse        isic2017 datasets/ISIC2017 24
run fatnet           isic2017 datasets/ISIC2017 16
run malunet          isic2017 datasets/ISIC2017 48

# ── ISIC-2018 ────────────────────────────────────────────────────────
run unet             isic2018 datasets/ISIC2018 32
run unetplusplus     isic2018 datasets/ISIC2018 24
run attdeeplabv3plus isic2018 datasets/ISIC2018 24
run transfuse        isic2018 datasets/ISIC2018 24
run fatnet           isic2018 datasets/ISIC2018 16
run malunet          isic2018 datasets/ISIC2018 48

# ── PH2 ─────────────────────────────────────────────────────────────
run unet             ph2 datasets/PH2 16
run unetplusplus     ph2 datasets/PH2 12
run attdeeplabv3plus ph2 datasets/PH2 12
run transfuse        ph2 datasets/PH2 12
run fatnet           ph2 datasets/PH2  8
run malunet          ph2 datasets/PH2 24

echo "════════════════════════════════════════"
echo "✅ 全部 18 组训练完成！"
echo "   运行以下命令汇总结果："
echo "   python scripts/collect_supervised_results.py"
echo "════════════════════════════════════════"
