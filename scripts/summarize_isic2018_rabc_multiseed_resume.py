#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "no-RABC": ROOT / "outputs/innovation/isic2018_norabc_multiseed_resume",
    "RABC": ROOT / "outputs/innovation/isic2018_rabc_multiseed_resume",
}

SUBDIRS = {
    "no-RABC": "rpi_net_isic2018_norabc_resume_s{seed}",
    "RABC": "rpi_net_isic2018_rabc_resume_s{seed}",
}

SEEDS = [41, 42, 43]
METRICS = ["acc", "dice", "jac", "sen", "spe"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_mean_std(values: list[float]) -> str:
    mean = stats.mean(values)
    std = stats.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.6f} ± {std:.6f}"


def main() -> None:
    paired_by_seed = {
        seed: load_json(ROOT / f"outputs/innovation/isic2018_rabc_multiseed_resume/paired_s{seed}.json")
        for seed in SEEDS
    }
    out_rows: dict[str, dict] = {}

    for label in RUNS:
        rows = []
        for seed in SEEDS:
            run_dir = RUNS[label] / SUBDIRS[label].format(seed=seed)
            summary = load_json(run_dir / "summary.json")
            select = load_json(run_dir / "select_jac.json")
            best = select["summary"]["best"]
            paired = paired_by_seed[seed]
            pair_metrics = paired["metrics"]
            if label == "no-RABC":
                selected_test = {m: pair_metrics[m]["a_mean"] for m in METRICS}
            else:
                selected_test = {m: pair_metrics[m]["b_mean"] for m in METRICS}
            row = {
                "seed": seed,
                "best_val": summary["best_val"],
                "best_test": summary["best_test"],
                "selected_test": selected_test,
                "selected_operating_point": {
                    "threshold": best["threshold"],
                    "tta": best["tta"],
                    "min_area": best["min_area"],
                    "keep_largest": best["keep_largest"],
                    "fill_holes": best["fill_holes"],
                    "gaussian_sigma": best.get("gaussian_sigma", 0.0),
                    "adaptive_scale": best.get("adaptive_scale", 0.0),
                    "dilation_iters": best.get("dilation_iters", 0),
                },
            }
            rows.append(row)
        out_rows[label] = {"rows": rows}

    for label, payload in out_rows.items():
        selected = payload["rows"]
        payload["selected_test_mean_std"] = {
            m: fmt_mean_std([row["selected_test"][m] for row in selected]) for m in METRICS
        }
        payload["best_test_mean_std"] = {
            m: fmt_mean_std([row["best_test"][m] for row in selected]) for m in METRICS
        }

    out_json = ROOT / "outputs/innovation/isic2018_rabc_multiseed_resume_summary.json"
    out_md = ROOT / "outputs/innovation/isic2018_rabc_multiseed_resume_summary.md"
    out_json.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")

    lines = ["# ISIC-2018 Resumed Multiseed Summary", ""]
    for label, payload in out_rows.items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append("### Selected test mean±std")
        for m in METRICS:
            lines.append(f"- {m}: {payload['selected_test_mean_std'][m]}")
        lines.append("")
        lines.append("### Per-seed selected operating points and test metrics")
        for row in payload["rows"]:
            op = row["selected_operating_point"]
            tst = row["selected_test"]
            lines.append(
                f"- seed={row['seed']}: thr={op['threshold']}, tta={op['tta']}, "
                f"min_area={op['min_area']}, keep_largest={op['keep_largest']}, "
                f"fill_holes={op['fill_holes']}, sigma={op['gaussian_sigma']}, "
                f"dice={tst['dice']:.6f}, jac={tst['jac']:.6f}, sen={tst['sen']:.6f}, spe={tst['spe']:.6f}"
            )
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
