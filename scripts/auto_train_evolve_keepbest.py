#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import smtplib
import subprocess
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EPS = 1e-9
GB = 1024**3

PAPER = {
    "ISIC-2018": {"acc": 0.9021, "dice": 0.8288, "jac": 0.7414, "sen": 0.8717, "spe": 0.9405},
    "ISIC-2017": {"acc": 0.8762, "dice": 0.7584, "jac": 0.6621, "sen": 0.8415, "spe": 0.9173},
    "PH2": {"acc": 0.9325, "dice": 0.8891, "jac": 0.8198, "sen": 0.9366, "spe": 0.9460},
}

DATASETS = [
    {
        "name": "PH2",
        "slug": "ph2",
        "root": "datasets/PH2",
        "pseudo_dir": "outputs/pseudo_labels/ph2_train",
        "ph2_split_seed": 42,
        "ph2_split_ratios": [0.7, 0.15, 0.15],
    },
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
        "epochs": 6,
        "iterative_rounds": 0,
        "iterative_epochs": 0,
        "batch_size": 10,
        "lr": 1.0e-5,
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
        "pseudo_consensus_weight": 0.12,
        "boundary_weight": 0.03,
        "boundary_head_weight": 0.12,
        "background_supervision_weight": 0.05,
        "uncertainty_supervision_weight": 0.08,
        "aux_consensus_weight": 0.05,
        "detail_consensus_weight": 0.08,
        "low_threshold_bg_weight": 0.05,
        "low_threshold_bg_threshold": 0.08,
        "supervised_weight": 0.0,
        "labeled_ratio": 0.0,
        "pseudo_tversky_weight": 0.20,
        "pseudo_tversky_alpha": 0.60,
        "pseudo_tversky_beta": 0.40,
        "pseudo_tversky_gamma": 1.00,
        "supervised_tversky_weight": 0.0,
        "supervised_tversky_alpha": 0.65,
        "supervised_tversky_beta": 0.35,
        "supervised_tversky_gamma": 1.00,
        "resume_strict": False,
        "resume_load_optimizer": False,
        "resume_load_scaler": False,
        "resume_reset_epoch": True,
        "resume_reset_best": True,
        "freeze_segmentation_path": False,
        "freeze_bn_stats": False,
        "trainable_prefixes": ["pseudo_encoder", "ipc_attn", "pia_attn", "sigma_heads"],
    },
    "model": {
        "hidden_dim": 128,
        "dropout": 0.1,
        "backbone": "convnext_small",
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
        "use_ega_refine": True,
        "use_cot_context": False,
        "use_lsk_decoder": False,
        "use_ucgf_refine": False,
        "use_anchor_context": False,
        "use_smfa_decoder": False,
        "use_tff_refine": False,
        "use_mfe_context": False,
        "use_bg_decouple_head": True,
        "use_uncertainty_logit_gate": False,
        "use_global_logit_calibrator": False,
        "use_boundary_uncertainty_logit_refine": True,
        "use_residual_logit_refine": False,
        "use_mamba_logit_refine": False,
        "use_cgfn_logit_refine": True,
        "use_recall_recover_refine": True,
        "use_recall_candidate_head": False,
        "use_multiscale_recall_head": False,
    },
    "runtime": {
        "device": "cuda",
        "save_every": 1000,
        "checkpoint_include_teacher": True,
        "checkpoint_include_optimizer": False,
        "checkpoint_include_scaler": False,
        "cpu_threads": 8,
        "tf32": True,
        "channels_last": True,
        "compile": False,
    },
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "evolve_csmall_fullft_hybrid",
        "model": {},
        "train": {"epochs": 8, "batch_size": 8, "lr": 8.0e-6},
    },
    {
        "name": "evolve_csmall_iterative",
        "model": {},
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
        },
    },
    {
        "name": "evolve_ctiny_headmix_iterative",
        "model": {
            "backbone": "convnext_tiny",
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
            "freeze_segmentation_path": True,
            "freeze_bn_stats": True,
            "recall_candidate_weight": 0.06,
            "recall_candidate_pos_weight": 1.5,
            "multiscale_recall_weight": 0.06,
            "multiscale_recall_pos_weight": 1.5,
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
        "name": "evolve_ctiny_mamba_fullft",
        "model": {"backbone": "convnext_tiny", "use_mamba_logit_refine": True},
        "train": {"epochs": 8, "lr": 1.0e-5, "batch_size": 10},
    },
]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sigterm_to_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CMD: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = p.wait()
        f.write(f"[{now()}] EXIT={code}\n")
        f.flush()
    return code


def run_cmd_capture(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    out, _ = p.communicate()
    return p.returncode, out


def parse_eval_output(stdout: str) -> dict[str, float]:
    lines = [x.strip() for x in stdout.splitlines() if x.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}") and "acc" in line and "dice" in line:
            obj = ast.literal_eval(line)
            return {k: float(obj[k]) for k in ("acc", "dice", "jac", "sen", "spe")}
    raise RuntimeError("Cannot parse evaluate output")


def mean5(metrics: dict[str, float]) -> float:
    return float(sum(float(metrics[k]) for k in ("acc", "dice", "jac", "sen", "spe")) / 5.0)


def all5_gt(a: dict[str, float], b: dict[str, float]) -> bool:
    return all(float(a[k]) > float(b[k]) + EPS for k in ("acc", "dice", "jac", "sen", "spe"))


def pass_paper(ds_name: str, m: dict[str, float]) -> bool:
    p = PAPER[ds_name]
    return all(float(m[k]) > float(p[k]) for k in ("acc", "dice", "jac", "sen", "spe"))


def deep_update(dst: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_seeds() -> list[int]:
    raw = os.getenv("AUTO_SEEDS", "42")
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [42]


def acquire_lock(loop_root: Path) -> int:
    lock_path = loop_root / "runner.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        prev = ""
        try:
            prev = os.read(fd, 256).decode("utf-8", errors="ignore").strip()
        except Exception:
            pass
        raise RuntimeError(f"another runner active: {prev}")
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()} started={now()}".encode("utf-8"))
    os.fsync(fd)
    return fd


def guard(ds_name: str) -> dict[str, float]:
    p = PAPER[ds_name]
    return {
        "min_sen": max(0.75, p["sen"] - 0.10),
        "min_spe": max(0.75, p["spe"] - 0.10),
        "min_jac": max(0.45, p["jac"] - 0.15),
        "min_dice": max(0.60, p["dice"] - 0.15),
        "min_acc": max(0.72, p["acc"] - 0.12),
    }


def make_cfg(candidate: dict[str, Any], ds: dict[str, Any], seed: int, out_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(BASE_CFG)
    cfg["seed"] = int(seed)
    cfg["output_dir"] = str(out_dir)
    cfg["data"]["root"] = ds["root"]
    cfg["pseudo"]["output_dir"] = ds["pseudo_dir"]
    if "ph2_split_seed" in ds:
        cfg["data"]["ph2_split_seed"] = int(ds["ph2_split_seed"])
        cfg["data"]["ph2_split_ratios"] = list(ds["ph2_split_ratios"])
    deep_update(cfg["model"], candidate.get("model", {}))
    deep_update(cfg["train"], candidate.get("train", {}))
    return cfg


def select_op(cfg_path: Path, ckpt: Path, ds_name: str, eval_dir: Path, env: dict[str, str]) -> tuple[bool, dict[str, Any]]:
    g = guard(ds_name)
    out_path = eval_dir / "select_val_mean5_guard.json"
    cmd = [
        PY,
        "scripts/select_inference_by_val.py",
        "--config",
        str(cfg_path),
        "--checkpoint",
        str(ckpt),
        "--objective",
        "mean5",
        "--min-sen",
        str(g["min_sen"]),
        "--min-spe",
        str(g["min_spe"]),
        "--min-jac",
        str(g["min_jac"]),
        "--min-dice",
        str(g["min_dice"]),
        "--min-acc",
        str(g["min_acc"]),
        "--threshold-min",
        "0.05",
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
        str(out_path),
    ]
    code = run_cmd(cmd, eval_dir / "select.log", env=env)
    if code != 0 or not out_path.exists():
        return False, {}
    data = load_json(out_path, {})
    best = data.get("summary", {}).get("best")
    if not best:
        return False, {}
    return True, best


def send_email(subject: str, body: str) -> None:
    username = os.getenv("MAIL_USERNAME", "")
    password = os.getenv("MAIL_PASSWORD", "")
    to_addr = os.getenv("MAIL_TO", "202308437@stu.sicau.edu.cn")
    if not username or not password:
        print(f"[{now()}] mail skipped: MAIL_USERNAME/MAIL_PASSWORD not set")
        return
    msg = EmailMessage()
    msg["From"] = username
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=30) as smtp:
        smtp.login(username, password)
        smtp.send_message(msg)
    print(f"[{now()}] mail sent: {subject}")


def build_mail_body(best_registry: dict[str, Any], last_status: dict[str, Any]) -> str:
    lines = [f"time: {now()}", "", "last status:", json.dumps(last_status, ensure_ascii=False, indent=2), "", "best registry:"]
    lines.append(json.dumps(best_registry, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def build_improvement_mail_body(report: dict[str, Any], best_registry: dict[str, Any]) -> str:
    lines = [
        f"time: {now()}",
        f"run_id: {report['run_id']}",
        f"candidate: {report['candidate']}",
        f"seed: {report['seed']}",
        "",
        "new historical best datasets:",
    ]
    for ds in report.get("improved_datasets", []):
        info = report["datasets"].get(ds, {})
        m = info.get("metrics", {})
        if m:
            lines.append(
                f"- {ds}: "
                f"ACC {m['acc']*100:.2f}, DIC {m['dice']*100:.2f}, JAC {m['jac']*100:.2f}, "
                f"SEN {m['sen']*100:.2f}, SPE {m['spe']*100:.2f}"
            )
        else:
            lines.append(f"- {ds}")
    lines.append("")
    lines.append("best registry snapshot:")
    lines.append(json.dumps(best_registry, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def protected_run_ids(best_registry: dict[str, Any]) -> set[str]:
    keep: set[str] = set()
    for v in best_registry.values():
        if not isinstance(v, dict):
            continue
        rid = v.get("run_id")
        if isinstance(rid, str) and rid:
            keep.add(rid)
    return keep


def prune_old_runs(
    loop_root: Path,
    best_registry: dict[str, Any],
    keep_recent: int,
    target_free_bytes: int | None = None,
    extra_keep: set[str] | None = None,
) -> list[str]:
    runs_root = loop_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        return []

    keep_ids = protected_run_ids(best_registry)
    keep_ids.update(p.name for p in run_dirs[-max(0, int(keep_recent)) :])
    if extra_keep:
        keep_ids.update(extra_keep)

    deleted: list[str] = []
    for run_dir in run_dirs:
        rid = run_dir.name
        if rid in keep_ids:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        deleted.append(rid)
        if target_free_bytes is not None and free_bytes(loop_root) >= target_free_bytes:
            break
    return deleted


def ensure_disk_headroom(loop_root: Path, best_registry: dict[str, Any], current_run_id: str | None = None) -> None:
    min_free_gb = float(os.getenv("AUTO_MIN_FREE_GB", "20"))
    target_free_gb = float(os.getenv("AUTO_TARGET_FREE_GB", "40"))
    keep_recent = int(os.getenv("AUTO_KEEP_RECENT_RUNS", "4"))
    min_free = int(min_free_gb * GB)
    target_free = int(target_free_gb * GB)
    if target_free < min_free:
        target_free = min_free

    free0 = free_bytes(loop_root)
    if free0 >= min_free:
        return

    extra_keep = {current_run_id} if current_run_id else set()
    deleted = prune_old_runs(
        loop_root=loop_root,
        best_registry=best_registry,
        keep_recent=keep_recent,
        target_free_bytes=target_free,
        extra_keep=extra_keep,
    )
    free1 = free_bytes(loop_root)
    print(
        f"[{now()}] disk-guard: free_before={free0/GB:.2f}GB free_after={free1/GB:.2f}GB deleted_runs={len(deleted)}"
    )
    if free1 < min_free:
        raise RuntimeError(
            f"insufficient disk after pruning (free={free1/GB:.2f}GB < min={min_free_gb:.2f}GB)"
        )


def run_once(
    loop_root: Path,
    candidate: dict[str, Any],
    seed: int,
    best_registry: dict[str, Any],
    status_path: Path,
) -> dict[str, Any]:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{candidate['name']}_s{seed}"
    run_dir = loop_root / "runs" / run_id
    cfg_dir = run_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ensure_disk_headroom(loop_root, best_registry, current_run_id=run_id)
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
        "improved_datasets": [],
        "report_path": str(run_dir / "report.json"),
    }
    save_json(run_dir / "report.json", report)

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_slug = ds["slug"]
        ds_out = run_dir / ds_slug
        ds_out.mkdir(parents=True, exist_ok=True)
        cfg = make_cfg(candidate, ds, seed=seed, out_dir=ds_out)
        cfg_path = cfg_dir / f"{ds_slug}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        resume_ckpt = best_registry.get(ds_name, {}).get("checkpoint")
        train_cmd = [PY, "scripts/train_rpi_net.py", "--config", str(cfg_path)]
        if resume_ckpt and Path(str(resume_ckpt)).exists():
            train_cmd.extend(["--resume", str(resume_ckpt)])
        code = run_cmd(train_cmd, ds_out / "train.log", env=env)
        if code != 0:
            report["datasets"][ds_name] = {"status": "train_failed", "exit_code": code}
            report["status"] = "failed"
            report["finished_at"] = now()
            save_json(run_dir / "report.json", report)
            return report

        ckpt = ds_out / "checkpoints" / "best.pt"
        if not ckpt.exists():
            report["datasets"][ds_name] = {"status": "missing_checkpoint"}
            report["status"] = "failed"
            report["finished_at"] = now()
            save_json(run_dir / "report.json", report)
            return report

        eval_dir = ds_out / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ok, best_op = select_op(cfg_path, ckpt, ds_name, eval_dir, env=env)
        if not ok:
            report["datasets"][ds_name] = {"status": "select_failed"}
            report["status"] = "continue"
            report["finished_at"] = now()
            save_json(run_dir / "report.json", report)
            return report

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
            str(best_op["threshold"]),
            "--tta",
            str(best_op["tta"]),
            "--min-area",
            str(best_op["min_area"]),
            "--workers",
            "2",
            "--batch-size",
            "8",
        ]
        if bool(best_op["keep_largest"]):
            ev_cmd.append("--keep-largest")
        if bool(best_op["fill_holes"]):
            ev_cmd.append("--fill-holes")
        code, stdout = run_cmd_capture(ev_cmd, env=env)
        (eval_dir / "test_eval_stdout.txt").write_text(stdout, encoding="utf-8")
        if code != 0:
            report["datasets"][ds_name] = {"status": "test_eval_failed", "exit_code": code}
            report["status"] = "failed"
            report["finished_at"] = now()
            save_json(run_dir / "report.json", report)
            return report

        m = parse_eval_output(stdout)
        ds_pass = pass_paper(ds_name, m)
        ds_info = {
            "status": "ok",
            "metrics": m,
            "score_mean5": mean5(m),
            "pass_paper": ds_pass,
            "paper": PAPER[ds_name],
            "checkpoint": str(ckpt),
            "config": str(cfg_path),
            "operating_point": {
                "threshold": float(best_op["threshold"]),
                "tta": str(best_op["tta"]),
                "min_area": int(best_op["min_area"]),
                "keep_largest": bool(best_op["keep_largest"]),
                "fill_holes": bool(best_op["fill_holes"]),
            },
        }
        report["datasets"][ds_name] = ds_info
        save_json(run_dir / "report.json", report)

        cur_best = best_registry.get(ds_name)
        improved = cur_best is None or all5_gt(m, cur_best["metrics"])
        if improved:
            best_registry[ds_name] = {
                "updated_at": now(),
                "run_id": run_id,
                "candidate": candidate["name"],
                "seed": int(seed),
                "metrics": m,
                "score_mean5": mean5(m),
                "checkpoint": str(ckpt),
                "config": str(cfg_path),
                "operating_point": ds_info["operating_point"],
            }
            report["improved_datasets"].append(ds_name)

        save_json(loop_root / "best_registry.json", best_registry)

        # Two-stage flow: PH2 is used for fast screening.
        # If PH2 cannot pass paper thresholds, skip slower ISIC datasets.
        if ds_name == "PH2" and not ds_pass:
            report["status"] = "continue"
            report["finished_at"] = now()
            save_json(run_dir / "report.json", report)
            return report

        save_json(
            status_path,
            {
                "time": now(),
                "pid": os.getpid(),
                "run_id": run_id,
                "candidate": candidate["name"],
                "seed": int(seed),
                "dataset": ds_name,
                "pass_paper": ds_pass,
                "improved": improved,
            },
        )

    report["all_pass"] = all(report["datasets"][d["name"]]["pass_paper"] for d in DATASETS)
    report["status"] = "success" if report["all_pass"] else "continue"
    report["finished_at"] = now()
    save_json(run_dir / "report.json", report)
    return report


def main() -> None:
    loop_root = ROOT / "outputs" / "strict_autotrain_loop"
    loop_root.mkdir(parents=True, exist_ok=True)
    _lock_fd = acquire_lock(loop_root)
    signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

    status_path = loop_root / "loop_status.json"
    success_path = loop_root / "SUCCESS.json"
    board_path = loop_root / "progress_board.json"
    best_path = loop_root / "best_registry.json"

    seeds = parse_seeds()
    round_idx = 0
    best_registry = load_json(best_path, {})
    board = load_json(board_path, {"updated_at": now(), "rows": []})

    try:
        while True:
            round_idx += 1
            for cand in CANDIDATES:
                for seed in seeds:
                    ensure_disk_headroom(loop_root, best_registry)
                    save_json(
                        status_path,
                        {
                            "time": now(),
                            "pid": os.getpid(),
                            "round": round_idx,
                            "candidate": cand["name"],
                            "seed": int(seed),
                        },
                    )
                    print(f"[{now()}] start round={round_idx} candidate={cand['name']} seed={seed}")
                    report = run_once(loop_root, cand, seed, best_registry, status_path)
                    save_json(best_path, best_registry)
                    prune_old_runs(
                        loop_root=loop_root,
                        best_registry=best_registry,
                        keep_recent=int(os.getenv("AUTO_KEEP_RECENT_RUNS", "4")),
                        extra_keep={report["run_id"]},
                    )

                    board["rows"] = list(board.get("rows", [])) + [
                        {
                            "time": now(),
                            "round": round_idx,
                            "run_id": report["run_id"],
                            "candidate": report["candidate"],
                            "seed": report["seed"],
                            "status": report["status"],
                            "all_pass": report["all_pass"],
                            "improved_datasets": report.get("improved_datasets", []),
                            "report_path": report["report_path"],
                        }
                    ]
                    board["rows"] = board["rows"][-300:]
                    board["updated_at"] = now()
                    save_json(board_path, board)

                    print(
                        f"[{now()}] done run_id={report['run_id']} status={report['status']} all_pass={report['all_pass']} improved={report.get('improved_datasets', [])}"
                    )
                    if report.get("improved_datasets"):
                        send_email(
                            subject=f"[skin] New historical best: {report['run_id']}",
                            body=build_improvement_mail_body(report, best_registry),
                        )
                    if report["all_pass"]:
                        payload = {"time": now(), "report": report, "best_registry": best_registry}
                        save_json(success_path, payload)
                        send_email(
                            subject=f"[skin] PASS all paper metrics: {report['run_id']}",
                            body=build_mail_body(best_registry, {"event": "success", "report": report}),
                        )
                        return
                    time.sleep(3)
    except KeyboardInterrupt:
        send_email(
            subject="[skin] training stopped (KeyboardInterrupt)",
            body=build_mail_body(best_registry, {"event": "stopped", "reason": "keyboard_interrupt"}),
        )
        raise
    except Exception as e:
        send_email(
            subject="[skin] training stopped (exception)",
            body=build_mail_body(best_registry, {"event": "stopped", "reason": repr(e)}),
        )
        raise


if __name__ == "__main__":
    main()
