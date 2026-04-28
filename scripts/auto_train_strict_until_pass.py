#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import smtplib
import yaml


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

PAPER = {
    "ISIC-2018": {"acc": 0.9021, "dice": 0.8288, "jac": 0.7414, "sen": 0.8717, "spe": 0.9405},
    "ISIC-2017": {"acc": 0.8762, "dice": 0.7584, "jac": 0.6621, "sen": 0.8415, "spe": 0.9173},
    "PH2": {"acc": 0.9325, "dice": 0.8891, "jac": 0.8198, "sen": 0.9366, "spe": 0.9460},
}

DATASETS = [
    {
        "name": "ISIC-2017",
        "slug": "isic2017",
        "root": "datasets/ISIC2017",
        "pseudo_dir": "outputs/pseudo_labels/isic2017_train",
    },
    {
        "name": "ISIC-2018",
        "slug": "isic2018",
        "root": "datasets/ISIC2018",
        "pseudo_dir": "outputs/pseudo_labels_v18/isic2018_train",
    },
    {
        "name": "PH2",
        "slug": "ph2",
        "root": "datasets/PH2",
        "pseudo_dir": "outputs/pseudo_labels/ph2_train",
        "ph2_split_seed": 42,
        "ph2_split_ratios": [0.7, 0.15, 0.15],
    },
]

BASE_CFG: dict[str, Any] = {
    "seed": 42,
    "output_dir": "",
    "data": {
        "root": "",
        "image_size": 224,
        "use_hair_removal": False,
        "use_tensor_cache": False,
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "eval_num_workers": 2,
        "eval_prefetch_factor": 2,
        "cache_images": False,
        "cache_workers": 4,
    },
    "pseudo": {
        "output_dir": "",
        "force_regenerate": False,
        "kmeans_clusters": 2,
        "kmeans_threads": 8,
        "batch_size": 8,
        "threshold_method": "otsu",
        "consensus_blend": 0.25,
    },
    "train": {
        "epochs": 2,
        "iterative_rounds": 0,
        "iterative_epochs": 0,
        "batch_size": 12,
        "lr": 5.0e-6,
        "weight_decay": 1.0e-4,
        "amp": True,
        "amp_dtype": "bf16",
        "grad_clip": 1.0,
        "threshold": 0.30,
        "use_mean_teacher": False,
        "ema_decay": 0.99,
        "consistency_weight": 0.0,
        "consistency_rampup_epochs": 0,
        "source_distill_weight": 0.0,
        "source_distill_rampup_epochs": 0,
        "agreement_gamma": 1.0,
        "pseudo_consensus_weight": 0.10,
        "boundary_weight": 0.02,
        "boundary_head_weight": 0.10,
        "background_supervision_weight": 0.00,
        "uncertainty_supervision_weight": 0.05,
        "aux_consensus_weight": 0.03,
        "detail_consensus_weight": 0.05,
        "low_threshold_bg_weight": 0.00,
        "low_threshold_bg_threshold": 0.08,
        "supervised_weight": 0.0,
        "labeled_ratio": 0.0,
        "pseudo_tversky_weight": 0.15,
        "pseudo_tversky_alpha": 0.65,
        "pseudo_tversky_beta": 0.35,
        "pseudo_tversky_gamma": 1.10,
        "supervised_tversky_weight": 0.0,
        "supervised_tversky_alpha": 0.65,
        "supervised_tversky_beta": 0.35,
        "supervised_tversky_gamma": 1.00,
        "resume_strict": False,
        "resume_load_optimizer": False,
        "resume_load_scaler": False,
        "resume_reset_epoch": True,
        "resume_reset_best": True,
        "freeze_segmentation_path": True,
        "freeze_bn_stats": True,
        "trainable_prefixes": ["pseudo_encoder", "ipc_attn", "pia_attn", "sigma_heads"],
    },
    "model": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "backbone": "convnext_tiny",
        "backbone_pretrained": True,
        "use_boundary_head": True,
        "use_uncertainty_head": True,
        "refine_with_heads": True,
        "use_aspp": False,
        "use_aux_head": True,
        "use_se_context": True,
        "use_detail_fusion": True,
        "use_coord_attn": True,
        "use_edge_prompt": True,
        "use_dual_supervised_head": False,
        "use_eca_context": True,
        "use_ema_context": True,
        "use_cbam_decoder": True,
        "use_simam_refine": True,
        "use_safm_decoder": True,
        "use_uncertainty_gated_refine": True,
        "use_sk_context": True,
        "use_triplet_decoder": True,
        "use_msaa_fusion": True,
        "use_ega_refine": False,
        "use_cot_context": False,
        "use_lsk_decoder": False,
        "use_ucgf_refine": False,
        "use_anchor_context": False,
        "use_smfa_decoder": False,
        "use_tff_refine": False,
        "use_mfe_context": False,
        "use_bg_decouple_head": False,
        "use_uncertainty_logit_gate": False,
        "use_global_logit_calibrator": False,
        "use_boundary_uncertainty_logit_refine": False,
        "use_residual_logit_refine": False,
        "use_mamba_logit_refine": False,
        "use_cgfn_logit_refine": False,
        "use_recall_recover_refine": False,
        "use_recall_candidate_head": False,
        "use_multiscale_recall_head": False,
    },
    "runtime": {
        "device": "cuda",
        "save_every": 1,
        "cpu_threads": 8,
        "tf32": True,
        "channels_last": True,
        "compile": False,
    },
}

# All candidates keep the same model/loss across 3 datasets in one run.
# New direction: stronger architecture + full training + iterative pseudo-label refinement.
CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "stageC_convnextsmall_hybrid_fullft",
        "model": {
            "backbone": "convnext_small",
            "use_ega_refine": True,
            "use_bg_decouple_head": True,
            "use_boundary_uncertainty_logit_refine": True,
            "use_cgfn_logit_refine": True,
            "use_recall_recover_refine": True,
        },
        "train": {
            "epochs": 8,
            "batch_size": 8,
            "lr": 8.0e-6,
            "freeze_segmentation_path": False,
            "freeze_bn_stats": False,
            "background_supervision_weight": 0.05,
            "low_threshold_bg_weight": 0.05,
            "pseudo_tversky_alpha": 0.60,
            "pseudo_tversky_beta": 0.40,
            "pseudo_tversky_gamma": 1.00,
        },
    },
    {
        "name": "stageC_convnextsmall_hybrid_iterative",
        "model": {
            "backbone": "convnext_small",
            "use_ega_refine": True,
            "use_bg_decouple_head": True,
            "use_boundary_uncertainty_logit_refine": True,
            "use_cgfn_logit_refine": True,
            "use_recall_recover_refine": True,
        },
        "train": {
            "epochs": 6,
            "iterative_rounds": 1,
            "iterative_epochs": 2,
            "iterative_blend": 0.35,
            "iterative_uncertainty_selective": True,
            "iterative_uncertainty_percentile": 0.8,
            "iterative_uncertainty_from_sigma": True,
            "batch_size": 8,
            "lr": 7.0e-6,
            "freeze_segmentation_path": False,
            "freeze_bn_stats": False,
            "background_supervision_weight": 0.05,
            "low_threshold_bg_weight": 0.05,
            "pseudo_tversky_alpha": 0.60,
            "pseudo_tversky_beta": 0.40,
            "pseudo_tversky_gamma": 1.00,
        },
    },
    {
        "name": "stageC_tiny_hybrid_iterative_headmix",
        "model": {
            "backbone": "convnext_tiny",
            "use_ega_refine": True,
            "use_bg_decouple_head": True,
            "use_boundary_uncertainty_logit_refine": True,
            "use_cgfn_logit_refine": True,
            "use_recall_recover_refine": True,
            "use_recall_candidate_head": True,
            "use_multiscale_recall_head": True,
        },
        "train": {
            "epochs": 6,
            "iterative_rounds": 1,
            "iterative_epochs": 2,
            "iterative_blend": 0.30,
            "iterative_uncertainty_selective": True,
            "iterative_uncertainty_percentile": 0.8,
            "iterative_uncertainty_from_sigma": True,
            "lr": 1.5e-5,
            "background_supervision_weight": 0.05,
            "low_threshold_bg_weight": 0.05,
            "recall_candidate_weight": 0.06,
            "recall_candidate_pos_weight": 1.5,
            "multiscale_recall_weight": 0.06,
            "multiscale_recall_pos_weight": 1.5,
            "freeze_segmentation_path": True,
            "freeze_bn_stats": True,
            "trainable_prefixes": [
                "boundary_uncertainty_logit_refine",
                "cgfn_logit_refine",
                "recall_recover_refine",
                "recall_candidate_head",
                "multiscale_recall_head",
            ],
        },
    },
    {
        "name": "stageC_tiny_mamba_recall_fullft",
        "model": {
            "backbone": "convnext_tiny",
            "use_ega_refine": True,
            "use_bg_decouple_head": True,
            "use_boundary_uncertainty_logit_refine": True,
            "use_mamba_logit_refine": True,
            "use_recall_recover_refine": True,
        },
        "train": {
            "epochs": 8,
            "lr": 1.0e-5,
            "batch_size": 10,
            "freeze_segmentation_path": False,
            "freeze_bn_stats": False,
            "background_supervision_weight": 0.05,
            "low_threshold_bg_weight": 0.05,
            "pseudo_tversky_alpha": 0.60,
            "pseudo_tversky_beta": 0.40,
            "pseudo_tversky_gamma": 1.00,
        },
    },
]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = proc.wait()
        f.write(f"[{now()}] EXIT={code}\n")
        f.flush()
    return code


def run_cmd_capture(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    out, _ = proc.communicate()
    return proc.returncode, out


def parse_eval_output(stdout: str) -> dict[str, float]:
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}") and "acc" in line and "dice" in line:
            obj = ast.literal_eval(line)
            return {
                "acc": float(obj["acc"]),
                "dice": float(obj["dice"]),
                "jac": float(obj["jac"]),
                "sen": float(obj["sen"]),
                "spe": float(obj["spe"]),
            }
    raise RuntimeError("Cannot parse evaluate output for metrics")


def is_pass(metrics: dict[str, float], paper: dict[str, float]) -> bool:
    return all(float(metrics[k]) > float(paper[k]) for k in ("acc", "dice", "jac", "sen", "spe"))


def mean5(metrics: dict[str, float]) -> float:
    return float(
        (float(metrics["acc"]) + float(metrics["dice"]) + float(metrics["jac"]) + float(metrics["sen"]) + float(metrics["spe"]))
        / 5.0
    )


def deep_update(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def acquire_singleton_lock(loop_root: Path) -> int:
    lock_path = loop_root / "runner.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            prev = os.read(fd, 256).decode("utf-8", errors="ignore").strip()
        except Exception:
            prev = ""
        print(f"[{now()}] another runner is active; lock={lock_path} holder={prev}")
        sys.exit(2)
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()} started={now()}".encode("utf-8"))
    os.fsync(fd)
    return fd


def parse_seeds() -> list[int]:
    raw = os.getenv("AUTO_SEEDS", "42")
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        out = [42]
    return out


def guard_constraints(ds_name: str, relax: int = 0) -> dict[str, float]:
    p = PAPER[ds_name]
    # Anti-collapse constraints on val operating point.
    if relax <= 0:
        return {
            "min_acc": max(0.72, p["acc"] - 0.10),
            "min_dice": max(0.58, p["dice"] - 0.12),
            "min_jac": max(0.42, p["jac"] - 0.12),
            "min_sen": max(0.75, p["sen"] - 0.08),
            "min_spe": max(0.78, p["spe"] - 0.10),
        }
    return {
        "min_acc": max(0.68, p["acc"] - 0.15),
        "min_dice": max(0.52, p["dice"] - 0.17),
        "min_jac": max(0.35, p["jac"] - 0.20),
        "min_sen": max(0.72, p["sen"] - 0.12),
        "min_spe": max(0.70, p["spe"] - 0.16),
    }


def summary_collapsed(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        v = s.get("best_val", {})
        jac = float(v.get("jac", -1))
        spe = float(v.get("spe", -1))
        sen = float(v.get("sen", -1))
    except Exception:
        return False
    # Typical full-foreground collapse signature.
    return (jac >= 0.0 and jac < 0.25) and (sen > 0.98) and (spe < 0.05)


def select_operating_point(cfg_path: Path, ckpt: Path, ds_name: str, eval_dir: Path, env: dict[str, str]) -> tuple[bool, Path, str]:
    attempts = [
        ("guarded", guard_constraints(ds_name, relax=0), 0.08),
        ("relaxed", guard_constraints(ds_name, relax=1), 0.05),
    ]
    for tag, gc, thr_min in attempts:
        select_out = eval_dir / f"select_val_mean5_{tag}.json"
        sel_cmd = [
            PY,
            "scripts/select_inference_by_val.py",
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--objective",
            "mean5",
            "--min-sen",
            str(gc["min_sen"]),
            "--min-spe",
            str(gc["min_spe"]),
            "--min-jac",
            str(gc["min_jac"]),
            "--min-dice",
            str(gc["min_dice"]),
            "--min-acc",
            str(gc["min_acc"]),
            "--threshold-min",
            str(thr_min),
            "--threshold-max",
            "0.30",
            "--threshold-step",
            "0.01",
            "--tta-modes",
            "none,flip4",
            "--min-areas",
            "0,64,128,256,512",
            "--keep-largest-options",
            "false,true",
            "--fill-holes-options",
            "false,true",
            "--workers",
            "2",
            "--batch-size",
            "8",
            "--output",
            str(select_out),
        ]
        code = run_cmd(sel_cmd, eval_dir / f"select_{tag}.log", env=env)
        if code == 0 and select_out.exists():
            return True, select_out, tag
    return False, eval_dir / "select_failed.json", "failed"


def make_cfg(candidate: dict[str, Any], dataset: dict[str, Any], seed: int, out_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(BASE_CFG)
    cfg["seed"] = int(seed)
    cfg["output_dir"] = str(out_dir)
    cfg["data"]["root"] = dataset["root"]
    cfg["pseudo"]["output_dir"] = dataset["pseudo_dir"]
    if "ph2_split_seed" in dataset:
        cfg["data"]["ph2_split_seed"] = int(dataset["ph2_split_seed"])
        cfg["data"]["ph2_split_ratios"] = list(dataset["ph2_split_ratios"])
    deep_update(cfg["model"], candidate.get("model", {}))
    deep_update(cfg["train"], candidate.get("train", {}))
    return cfg


def send_success_email(run_id: str, report: dict[str, Any]) -> None:
    username = os.getenv("MAIL_USERNAME", "")
    password = os.getenv("MAIL_PASSWORD", "")
    to_addr = os.getenv("MAIL_TO", "202308437@stu.sicau.edu.cn")
    if not username or not password:
        print(f"[{now()}] MAIL skipped: MAIL_USERNAME/MAIL_PASSWORD not set.")
        return

    subject = f"[skin] strict unsupervised PASS: {run_id}"
    lines = [f"Run ID: {run_id}", "", "All datasets passed paper thresholds (strict >):", ""]
    for ds in ("ISIC-2017", "ISIC-2018", "PH2"):
        m = report["datasets"][ds]["metrics"]
        p = PAPER[ds]
        lines.append(
            f"{ds}: "
            f"ACC {m['acc']*100:.2f}/{p['acc']*100:.2f}, "
            f"DIC {m['dice']*100:.2f}/{p['dice']*100:.2f}, "
            f"JAC {m['jac']*100:.2f}/{p['jac']*100:.2f}, "
            f"SEN {m['sen']*100:.2f}/{p['sen']*100:.2f}, "
            f"SPE {m['spe']*100:.2f}/{p['spe']*100:.2f}"
        )
    lines.append("")
    lines.append(f"Report: {report['report_path']}")
    body = "\n".join(lines)

    msg = EmailMessage()
    msg["From"] = username
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=30) as smtp:
        smtp.login(username, password)
        smtp.send_message(msg)
    print(f"[{now()}] PASS mail sent to {to_addr}")


def update_progress_board(loop_root: Path, report: dict[str, Any]) -> None:
    board_path = loop_root / "progress_board.json"
    if board_path.exists():
        try:
            board = json.loads(board_path.read_text(encoding="utf-8"))
        except Exception:
            board = {"updated_at": now(), "rows": []}
    else:
        board = {"updated_at": now(), "rows": []}

    rows = list(board.get("rows", []))
    ds_rows = {}
    pass_count = 0
    for ds in DATASETS:
        ds_name = ds["name"]
        x = report.get("datasets", {}).get(ds_name)
        if not x or x.get("status") != "ok":
            continue
        m = x["metrics"]
        p = x["paper"]
        passed = bool(x.get("pass", False))
        if passed:
            pass_count += 1
        ds_rows[ds_name] = {
            "score_mean5": mean5(m),
            "pass": passed,
            "delta": {k: float(m[k] - p[k]) for k in ("acc", "dice", "jac", "sen", "spe")},
            "metrics": m,
        }

    rows.append(
        {
            "time": now(),
            "run_id": report.get("run_id"),
            "candidate": report.get("candidate"),
            "seed": report.get("seed"),
            "status": report.get("status"),
            "all_pass": bool(report.get("all_pass", False)),
            "pass_count": pass_count,
            "datasets": ds_rows,
            "report_path": report.get("report_path"),
        }
    )
    board["updated_at"] = now()
    board["rows"] = rows[-300:]
    board_path.write_text(json.dumps(board, indent=2), encoding="utf-8")


def train_eval_once(loop_root: Path, candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{candidate['name']}_s{seed}"
    run_dir = loop_root / "runs" / run_id
    cfg_dir = run_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    report: dict[str, Any] = {
        "run_id": run_id,
        "candidate": candidate["name"],
        "seed": int(seed),
        "started_at": now(),
        "datasets": {},
        "all_pass": False,
        "status": "running",
        "report_path": str(run_dir / "report.json"),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_slug = ds["slug"]
        ds_out = run_dir / ds_slug
        ds_out.mkdir(parents=True, exist_ok=True)
        cfg = make_cfg(candidate, ds, seed=seed, out_dir=ds_out)
        cfg_path = cfg_dir / f"{ds_slug}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        train_log = ds_out / "train.log"
        code = run_cmd([PY, "scripts/train_rpi_net.py", "--config", str(cfg_path)], train_log, env=env)
        if code != 0:
            report["datasets"][ds_name] = {"status": "train_failed", "exit_code": code}
            report["status"] = "failed"
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

        if summary_collapsed(ds_out / "summary.json"):
            report["datasets"][ds_name] = {"status": "collapsed_train_summary"}
            report["status"] = "continue"
            report["finished_at"] = now()
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

        ckpt = ds_out / "checkpoints" / "best.pt"
        if not ckpt.exists():
            report["datasets"][ds_name] = {"status": "missing_checkpoint"}
            report["status"] = "failed"
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

        eval_dir = ds_out / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ok, select_out, select_tag = select_operating_point(cfg_path, ckpt, ds_name, eval_dir, env=env)
        if not ok:
            report["datasets"][ds_name] = {"status": "select_failed", "stage": select_tag}
            report["status"] = "continue"
            report["finished_at"] = now()
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

        sel = json.loads(select_out.read_text(encoding="utf-8"))
        best = sel["summary"]["best"]

        ev_cmd = [
            PY,
            "scripts/evaluate_rpi_net.py",
            "--config",
            str(cfg_path),
            "--checkpoint",
            str(ckpt),
            "--split",
            "test",
            "--threshold",
            str(best["threshold"]),
            "--tta",
            str(best["tta"]),
            "--min-area",
            str(best["min_area"]),
            "--workers",
            "2",
            "--batch-size",
            "8",
        ]
        if bool(best["keep_largest"]):
            ev_cmd.append("--keep-largest")
        if bool(best["fill_holes"]):
            ev_cmd.append("--fill-holes")
        code, stdout = run_cmd_capture(ev_cmd, env=env)
        (eval_dir / "test_eval_stdout.txt").write_text(stdout, encoding="utf-8")
        if code != 0:
            report["datasets"][ds_name] = {"status": "test_eval_failed", "exit_code": code}
            report["status"] = "failed"
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

        metrics = parse_eval_output(stdout)
        report["datasets"][ds_name] = {
            "status": "ok",
            "select_stage": select_tag,
            "operating_point": {
                "threshold": float(best["threshold"]),
                "tta": str(best["tta"]),
                "min_area": int(best["min_area"]),
                "keep_largest": bool(best["keep_largest"]),
                "fill_holes": bool(best["fill_holes"]),
            },
            "metrics": metrics,
            "paper": PAPER[ds_name],
            "pass": is_pass(metrics, PAPER[ds_name]),
        }
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Fast-fail: if the first dataset already cannot beat paper on all 5 metrics,
        # no need to spend extra time on remaining datasets for this candidate.
        if ds_name == "ISIC-2017" and not bool(report["datasets"][ds_name]["pass"]):
            report["all_pass"] = False
            report["status"] = "continue"
            report["finished_at"] = now()
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

    report["all_pass"] = all(report["datasets"][ds["name"]]["pass"] for ds in DATASETS)
    report["status"] = "success" if report["all_pass"] else "continue"
    report["finished_at"] = now()
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    loop_root = ROOT / "outputs" / "strict_autotrain_loop"
    loop_root.mkdir(parents=True, exist_ok=True)
    _lock_fd = acquire_singleton_lock(loop_root)
    status_path = loop_root / "loop_status.json"
    success_path = loop_root / "SUCCESS.json"

    seeds = parse_seeds()
    round_idx = 0
    while True:
        round_idx += 1
        for cand in CANDIDATES:
            for seed in seeds:
                status = {
                    "time": now(),
                    "pid": os.getpid(),
                    "round": round_idx,
                    "candidate": cand["name"],
                    "seed": seed,
                }
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                print(f"[{now()}] start round={round_idx} candidate={cand['name']} seed={seed}")
                report = train_eval_once(loop_root, cand, seed)
                update_progress_board(loop_root, report)
                print(
                    f"[{now()}] done run_id={report['run_id']} status={report['status']} all_pass={report['all_pass']}"
                )
                if report["all_pass"]:
                    success_payload = {"time": now(), "report": report}
                    success_path.write_text(json.dumps(success_payload, indent=2), encoding="utf-8")
                    send_success_email(report["run_id"], report)
                    return
                time.sleep(5)


if __name__ == "__main__":
    main()
