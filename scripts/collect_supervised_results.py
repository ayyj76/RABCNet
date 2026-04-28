#!/usr/bin/env python3
"""
collect_supervised_results.py
汇总 18 个监督方法训练结果（6 模型 × 3 数据集），
生成论文 Table B 的 Markdown 表格，打印到终端并写入文件。

用法：
  python scripts/collect_supervised_results.py
  python scripts/collect_supervised_results.py --output-dir outputs/supervised --out paper/tables/table_b_supervised.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("unet",              "U-Net"),
    ("unetplusplus",      "U-Net++"),
    ("attdeeplabv3plus",  "Att-DeepLabv3+"),
    ("transfuse",         "TransFuse"),
    ("fatnet",            "FAT-Net"),
    ("malunet",           "MALUNet"),
]

DATASETS = [
    ("isic2017", "ISIC-2017"),
    ("isic2018", "ISIC-2018"),
    ("ph2",      "PH2"),
]

METRICS = ["acc", "dice", "jac", "sen", "spe"]
METRIC_LABELS = ["ACC", "DICE", "JAC", "SEN", "SPE"]


def load_result(sup_dir: Path, model_key: str, dataset_key: str) -> dict | None:
    path = sup_dir / f"{model_key}_{dataset_key}" / "test_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("test", None)


def fmt(v: float | None) -> str:
    if v is None:
        return "  —  "
    return f"{v:.2f}"


def build_table(sup_dir: Path) -> str:
    lines = []

    # 按数据集分组，每个数据集一张子表
    for ds_key, ds_name in DATASETS:
        lines.append(f"### {ds_name}\n")

        # 表头
        header = "| Method | " + " | ".join(METRIC_LABELS) + " |"
        sep    = "| :--- | " + " | ".join([":---:"] * len(METRICS)) + " |"
        lines.append(header)
        lines.append(sep)

        for model_key, model_name in MODELS:
            res = load_result(sup_dir, model_key, ds_key)
            if res is None:
                vals = ["  —  "] * len(METRICS)
            else:
                vals = [fmt(res.get(m)) for m in METRICS]
            lines.append(f"| {model_name} | " + " | ".join(vals) + " |")

        lines.append("")  # 空行分隔

    return "\n".join(lines)


def build_unified_table(sup_dir: Path) -> str:
    """所有数据集放在一张大表，列为 数据集×指标"""
    lines = []

    # 表头（每个数据集 5 列）
    ds_header = "| Method |"
    for _, ds_name in DATASETS:
        ds_header += f" {ds_name} ACC | {ds_name} DICE | {ds_name} JAC | {ds_name} SEN | {ds_name} SPE |"
    lines.append(ds_header)

    sep = "| :--- |" + " :---: |" * (len(DATASETS) * len(METRICS))
    lines.append(sep)

    for model_key, model_name in MODELS:
        row = f"| {model_name} |"
        for ds_key, _ in DATASETS:
            res = load_result(sup_dir, model_key, ds_key)
            if res is None:
                row += " — | — | — | — | — |"
            else:
                row += " " + " | ".join(fmt(res.get(m)) for m in METRICS) + " |"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "supervised"),
                        help="监督模型输出根目录（含各子目录）")
    parser.add_argument("--out", default=str(ROOT / "paper" / "tables" / "table_b_supervised.md"),
                        help="输出 Markdown 文件路径")
    parser.add_argument("--unified", action="store_true",
                        help="输出宽表（所有数据集在一张表），默认为按数据集分表")
    args = parser.parse_args()

    sup_dir = Path(args.output_dir)
    out_path = Path(args.out)

    print(f"扫描目录：{sup_dir}\n")

    # 打印缺失情况
    missing = []
    for model_key, model_name in MODELS:
        for ds_key, ds_name in DATASETS:
            p = sup_dir / f"{model_key}_{ds_key}" / "test_results.json"
            if not p.exists():
                missing.append(f"  ✗ {model_name} / {ds_name}  ({p})")
    if missing:
        print("以下结果尚未生成：")
        for m in missing:
            print(m)
        print()

    if args.unified:
        table = build_unified_table(sup_dir)
    else:
        table = build_table(sup_dir)

    # 添加说明头部
    header_note = (
        "# Table B: Supervised Baseline Results\n\n"
        "> Threshold = 0.5, no TTA, no postprocessing.\n"
        "> All models trained from ImageNet-pretrained weights on each dataset's official train split.\n"
        "> Metrics reported on test split (%).\n\n"
    )
    full_content = header_note + table + "\n"

    print(full_content)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_content, encoding="utf-8")
    print(f"[done] 已保存: {out_path}")


if __name__ == "__main__":
    main()
