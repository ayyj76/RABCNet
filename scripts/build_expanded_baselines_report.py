#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ["acc", "dice", "jac", "sen", "spe"]
METRIC_LABELS = {"acc": "ACC", "dice": "DIC", "jac": "JAC", "sen": "SEN", "spe": "SPE"}
DATASETS = ["ISIC-2018", "ISIC-2017", "PH2"]
PAPER_METRICS = {
    "ISIC-2018": {"acc": 90.21, "dice": 82.88, "jac": 74.14, "sen": 87.17, "spe": 94.05},
    "ISIC-2017": {"acc": 87.62, "dice": 75.84, "jac": 66.21, "sen": 84.15, "spe": 91.73},
    "PH2": {"acc": 93.25, "dice": 88.91, "jac": 81.98, "sen": 93.66, "spe": 94.60},
}


@dataclass
class Row:
    dataset: str
    method: str
    source: str
    protocol: str
    acc: float
    dice: float
    jac: float
    sen: float
    spe: float
    std: dict[str, float] | None
    op: dict[str, object] | None
    val_jac: float | None
    path: str | None

    def score(self) -> float:
        return (self.acc + self.dice + self.jac + self.sen + self.spe) / 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build expanded baseline report and visualizations")
    p.add_argument("--outputs-root", type=str, default="outputs")
    p.add_argument("--stats-root", type=str, default="outputs/stats")
    p.add_argument("--out-dir", type=str, default="outputs/stats/expanded_baselines")
    p.add_argument("--topk-heatmap", type=int, default=12)
    return p.parse_args()


def dataset_from_path(path: str) -> str | None:
    lp = path.lower()
    if "isic2018" in lp:
        return "ISIC-2018"
    if "isic2017" in lp:
        return "ISIC-2017"
    if "/ph2" in lp or "_ph2" in lp:
        return "PH2"
    return None


def method_from_parent_dir(parent_name: str) -> str:
    m = re.search(r"unsup_(.+)$", parent_name)
    if m:
        return m.group(1)
    return parent_name


def read_sweep_rows(outputs_root: Path) -> list[Row]:
    patterns = [
        "rpi_net_*/**/inference_optimizer_sweep*.json",
        "rpi_net_*/**/sweep_inference_v*_best.json",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(outputs_root.glob(pat))

    best_by_model: dict[tuple[str, str], Row] = {}
    for p in sorted(set(files)):
        if p.name.endswith("_stdout.txt"):
            continue
        dataset = dataset_from_path(str(p))
        if dataset is None:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = data.get("summary", {})
        best = summary.get("best_by_val_jac")
        if not isinstance(best, dict):
            continue
        test = best.get("test", {})
        val = best.get("val", {})
        if not all(k in test for k in METRICS):
            continue
        parent = p.parent.name
        grand = p.parent.parent.name if p.parent.parent != p.parent else p.parent.name
        model_dir = grand if grand.startswith("rpi_net_") else parent
        method = method_from_parent_dir(model_dir)
        row = Row(
            dataset=dataset,
            method=method,
            source="val-selected sweep",
            protocol="single-seed val-selected",
            acc=float(test["acc"]) * 100.0,
            dice=float(test["dice"]) * 100.0,
            jac=float(test["jac"]) * 100.0,
            sen=float(test["sen"]) * 100.0,
            spe=float(test["spe"]) * 100.0,
            std=None,
            op={
                "tta": best.get("tta"),
                "threshold": best.get("threshold"),
                "min_area": best.get("min_area"),
                "keep_largest": best.get("keep_largest"),
                "fill_holes": best.get("fill_holes"),
            },
            val_jac=(float(val["jac"]) * 100.0) if isinstance(val, dict) and ("jac" in val) else None,
            path=str(p),
        )
        key = (dataset, method)
        prev = best_by_model.get(key)
        if prev is None:
            best_by_model[key] = row
        else:
            prev_val = -1e9 if prev.val_jac is None else prev.val_jac
            cur_val = -1e9 if row.val_jac is None else row.val_jac
            if cur_val > prev_val:
                best_by_model[key] = row
    return list(best_by_model.values())


def read_multiseed_rows(stats_root: Path) -> list[Row]:
    specs = [
        ("ISIC-2018", "ours_main_3seed", stats_root / "isic2018_v20_fixedop_distance_multiseed.json"),
        ("ISIC-2017", "ours_main_3seed", stats_root / "isic2017_v18_ega_full_resume_thr012_multiseed.json"),
        ("PH2", "ours_main_3seed", stats_root / "ph2_v20_fixedop_distance_multiseed.json"),
    ]
    out: list[Row] = []
    for dataset, method, path in specs:
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            continue
        metrics = {m: np.array([float(r[m]) for r in rows], dtype=np.float64) for m in METRICS}
        op = {
            "tta": rows[0].get("tta"),
            "threshold": rows[0].get("threshold"),
            "min_area": rows[0].get("min_area"),
            "keep_largest": rows[0].get("keep_largest"),
            "fill_holes": rows[0].get("fill_holes"),
        }
        out.append(
            Row(
                dataset=dataset,
                method=method,
                source="fixed-op multiseed",
                protocol="3-seed fixed-op",
                acc=float(metrics["acc"].mean()),
                dice=float(metrics["dice"].mean()),
                jac=float(metrics["jac"].mean()),
                sen=float(metrics["sen"].mean()),
                spe=float(metrics["spe"].mean()),
                std={m: float(metrics[m].std(ddof=1)) for m in METRICS},
                op=op,
                val_jac=None,
                path=str(path),
            )
        )
    return out


def paper_rows() -> list[Row]:
    out = []
    for dataset, m in PAPER_METRICS.items():
        out.append(
            Row(
                dataset=dataset,
                method="paper",
                source="paper table",
                protocol="paper",
                acc=float(m["acc"]),
                dice=float(m["dice"]),
                jac=float(m["jac"]),
                sen=float(m["sen"]),
                spe=float(m["spe"]),
                std=None,
                op=None,
                val_jac=None,
                path=None,
            )
        )
    return out


def rank_dataset(rows: list[Row]) -> list[Row]:
    return sorted(rows, key=lambda x: (x.score(), x.jac, x.dice), reverse=True)


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "method",
                "source",
                "protocol",
                "score",
                "acc",
                "dice",
                "jac",
                "sen",
                "spe",
                "std_acc",
                "std_dice",
                "std_jac",
                "std_sen",
                "std_spe",
                "val_jac",
                "threshold",
                "tta",
                "min_area",
                "keep_largest",
                "fill_holes",
                "path",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.dataset,
                    r.method,
                    r.source,
                    r.protocol,
                    f"{r.score():.4f}",
                    f"{r.acc:.4f}",
                    f"{r.dice:.4f}",
                    f"{r.jac:.4f}",
                    f"{r.sen:.4f}",
                    f"{r.spe:.4f}",
                    "" if not r.std else f"{r.std.get('acc', 0.0):.4f}",
                    "" if not r.std else f"{r.std.get('dice', 0.0):.4f}",
                    "" if not r.std else f"{r.std.get('jac', 0.0):.4f}",
                    "" if not r.std else f"{r.std.get('sen', 0.0):.4f}",
                    "" if not r.std else f"{r.std.get('spe', 0.0):.4f}",
                    "" if r.val_jac is None else f"{r.val_jac:.4f}",
                    "" if not r.op else r.op.get("threshold", ""),
                    "" if not r.op else r.op.get("tta", ""),
                    "" if not r.op else r.op.get("min_area", ""),
                    "" if not r.op else r.op.get("keep_largest", ""),
                    "" if not r.op else r.op.get("fill_holes", ""),
                    r.path or "",
                ]
            )


def _bar_colors(methods: list[str]) -> list[str]:
    colors = []
    for m in methods:
        if m == "paper":
            colors.append("#8c8c8c")
        elif m == "ours_main_3seed":
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")
    return colors


def plot_jac_rank(rows: list[Row], out_png: Path, dataset: str) -> None:
    ranked = rank_dataset(rows)
    labels = [r.method for r in ranked]
    vals = [r.jac for r in ranked]
    colors = _bar_colors(labels)

    fig_h = max(4.0, 0.42 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("JAC (%)")
    ax.set_title(f"{dataset} Expanded Baselines (JAC)")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_heatmap(rows: list[Row], out_png: Path, dataset: str, topk: int) -> None:
    ranked = rank_dataset(rows)[:topk]
    labels = [r.method for r in ranked]
    arr = np.array([[getattr(r, m) for m in METRICS] for r in ranked], dtype=np.float64)
    vmin = np.floor(arr.min())
    vmax = np.ceil(arr.max())

    fig_h = max(4.0, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"{dataset} Top-{len(labels)} Baseline Heatmap")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Metric (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def write_markdown(path: Path, grouped: dict[str, list[Row]], stats: dict[str, dict]) -> None:
    lines: list[str] = []
    lines.append("# Expanded Baseline Comparison")
    lines.append("")
    lines.append("## Leaderboard (score = mean of ACC/DIC/JAC/SEN/SPE)")
    lines.append("")
    for ds in DATASETS:
        rows = rank_dataset(grouped.get(ds, []))
        if not rows:
            continue
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Rank | Method | Source | Score | ACC | DIC | JAC | SEN | SPE |")
        lines.append("| ---: | ------ | ------ | ----: | --: | --: | --: | --: | --: |")
        for i, r in enumerate(rows, start=1):
            lines.append(
                f"| {i} | {r.method} | {r.source} | {r.score():.2f} | {r.acc:.2f} | {r.dice:.2f} | {r.jac:.2f} | {r.sen:.2f} | {r.spe:.2f} |"
            )
        best = stats[ds]["best"]
        paper_rank = stats[ds]["paper_rank"]
        lines.append("")
        lines.append(
            f"- Best by score: `{best['method']}` (score {best['score']:.2f}, JAC {best['jac']:.2f}); paper rank: {paper_rank}/{len(rows)}"
        )
        pass_row = stats[ds].get("best_all_metrics_pass")
        if pass_row is None:
            lines.append("- Best all-metrics-pass vs paper: none")
        else:
            lines.append(
                f"- Best all-metrics-pass vs paper: `{pass_row['method']}` (score {pass_row['score']:.2f})"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    stats_root = Path(args.stats_root)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"

    rows = []
    rows.extend(read_sweep_rows(outputs_root))
    rows.extend(read_multiseed_rows(stats_root))
    rows.extend(paper_rows())

    grouped: dict[str, list[Row]] = {d: [] for d in DATASETS}
    for r in rows:
        grouped.setdefault(r.dataset, []).append(r)

    stats: dict[str, dict] = {}
    serializable_rows: list[dict] = []

    for ds in DATASETS:
        ds_rows = rank_dataset(grouped.get(ds, []))
        if not ds_rows:
            continue
        for r in ds_rows:
            serializable_rows.append(
                {
                    "dataset": r.dataset,
                    "method": r.method,
                    "source": r.source,
                    "protocol": r.protocol,
                    "score": r.score(),
                    "metrics": {m: getattr(r, m) for m in METRICS},
                    "std": r.std,
                    "val_jac": r.val_jac,
                    "operating_point": r.op,
                    "path": r.path,
                }
            )

        paper_rank = -1
        for i, r in enumerate(ds_rows, start=1):
            if r.method == "paper":
                paper_rank = i
                break
        best = ds_rows[0]
        paper = next((r for r in ds_rows if r.method == "paper"), None)
        delta_vs_paper = {}
        best_all_metrics_pass = None
        if paper is not None:
            for m in METRICS:
                delta_vs_paper[m] = float(getattr(best, m) - getattr(paper, m))
            pass_rows = []
            for r in ds_rows:
                if r.method == "paper":
                    continue
                if all(getattr(r, m) > getattr(paper, m) for m in METRICS):
                    pass_rows.append(r)
            if pass_rows:
                pass_rows = rank_dataset(pass_rows)
                p = pass_rows[0]
                best_all_metrics_pass = {
                    "method": p.method,
                    "source": p.source,
                    "score": p.score(),
                    "metrics": {m: getattr(p, m) for m in METRICS},
                }
        stats[ds] = {
            "n_methods": len(ds_rows),
            "best": {
                "method": best.method,
                "source": best.source,
                "score": best.score(),
                "jac": best.jac,
            },
            "paper_rank": paper_rank,
            "delta_best_vs_paper": delta_vs_paper,
            "best_all_metrics_pass": best_all_metrics_pass,
            "top3_by_score": [
                {"method": r.method, "source": r.source, "score": r.score(), "jac": r.jac}
                for r in ds_rows[:3]
            ],
        }

        write_csv(out_dir / f"{ds.lower().replace('-', '').replace(' ', '')}_expanded.csv", ds_rows)
        plot_jac_rank(ds_rows, fig_dir / f"{ds.lower().replace('-', '').replace(' ', '')}_jac_rank.png", ds)
        plot_heatmap(
            ds_rows,
            fig_dir / f"{ds.lower().replace('-', '').replace(' ', '')}_heatmap_top{min(args.topk_heatmap, len(ds_rows))}.png",
            ds,
            topk=args.topk_heatmap,
        )

    write_csv(out_dir / "all_datasets_expanded.csv", serializable_to_rows(serializable_rows))

    report = {
        "datasets": DATASETS,
        "n_total_rows": len(serializable_rows),
        "leaderboard": serializable_rows,
        "summary": stats,
        "artifacts": {
            "csv_all": str(out_dir / "all_datasets_expanded.csv"),
            "markdown": str(out_dir / "expanded_baselines.md"),
            "figures_dir": str(fig_dir),
        },
    }
    (out_dir / "expanded_baselines_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(out_dir / "expanded_baselines.md", grouped, stats)
    print(f"saved: {out_dir / 'expanded_baselines_summary.json'}")
    print(f"saved: {out_dir / 'expanded_baselines.md'}")


def serializable_to_rows(items: list[dict]) -> list[Row]:
    rows: list[Row] = []
    for x in items:
        m = x["metrics"]
        rows.append(
            Row(
                dataset=x["dataset"],
                method=x["method"],
                source=x["source"],
                protocol=x["protocol"],
                acc=float(m["acc"]),
                dice=float(m["dice"]),
                jac=float(m["jac"]),
                sen=float(m["sen"]),
                spe=float(m["spe"]),
                std=x.get("std"),
                op=x.get("operating_point"),
                val_jac=x.get("val_jac"),
                path=x.get("path"),
            )
        )
    return rows


if __name__ == "__main__":
    main()
