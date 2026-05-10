#!/usr/bin/env python3
"""Launch VOSR inference for multiple experiments across multiple GPUs.

This is a lightweight scheduler around the existing single-GPU inference
entrypoints. Each experiment/checkpoint is executed as one subprocess with a
single visible GPU.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOTS = (
    "exp_vosr",
    "exp_vosr_text",
    "exp_vosr_text_distill",
    "exp_vosr_text_distill_ablation",
)


@dataclass(frozen=True)
class Experiment:
    root: Path
    args_path: Path
    args: dict

    @property
    def name(self) -> str:
        return self.args.get("exp_name") or self.root.name


@dataclass(frozen=True)
class Job:
    experiment: Experiment
    checkpoint: Path
    script: str
    output_dir: Path
    command: list[str]
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VOSR inference for many experiment checkpoints across GPUs."
    )
    parser.add_argument(
        "--exp-roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Experiment root directories to scan.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Only run experiments whose path or exp_name matches one of these glob patterns.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Skip experiments whose path or exp_name matches one of these glob patterns.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help=(
            "Checkpoint selector: latest, all, a comma-separated step list "
            "(e.g. 10000,20000), or a checkpoint directory path."
        ),
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent inference subprocesses per GPU.",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default=None,
        help="Input image file/folder. Defaults to each experiment's test_lq_dir.",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        default="preset/results/infer_experiments",
        help="Root directory for all inference outputs and logs.",
    )
    parser.add_argument("--upscale", "-u", type=int, default=None)
    parser.add_argument("--align-method", choices=("wavelet", "adain", "nofix"), default="nofix")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--infer-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--weak-cond-strength-aelq", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def matches_any(value: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(value, pat) for pat in patterns)


def discover_experiments(exp_roots: Iterable[str]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for root in exp_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for args_path in sorted(root_path.glob("*/args.json")):
            with args_path.open("r", encoding="utf-8") as f:
                exp_args = json.load(f)
            experiments.append(Experiment(args_path.parent, args_path, exp_args))
    return experiments


def filter_experiments(
    experiments: Iterable[Experiment], include: list[str], exclude: list[str]
) -> list[Experiment]:
    selected = []
    for exp in experiments:
        keys = (exp.name, exp.root.as_posix())
        if include and not any(matches_any(k, include) for k in keys):
            continue
        if exclude and any(matches_any(k, exclude) for k in keys):
            continue
        selected.append(exp)
    return selected


def checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", checkpoint.name)
    if not match:
        return -1
    return int(match.group(1))


def list_checkpoints(exp: Experiment) -> list[Path]:
    ckpt_root = exp.root / "checkpoints"
    if not ckpt_root.is_dir():
        return []
    checkpoints = [p for p in ckpt_root.glob("checkpoint-*") if p.is_dir()]
    return sorted(checkpoints, key=checkpoint_step)


def select_checkpoints(exp: Experiment, selector: str) -> list[Path]:
    selector_path = Path(selector)
    if selector_path.is_dir():
        return [selector_path]

    checkpoints = list_checkpoints(exp)
    if not checkpoints:
        return []

    if selector == "latest":
        return [checkpoints[-1]]
    if selector == "all":
        return checkpoints

    requested_steps = {
        int(item.strip()) for item in selector.split(",") if item.strip().isdigit()
    }
    if requested_steps:
        by_step = {checkpoint_step(p): p for p in checkpoints}
        return [by_step[step] for step in sorted(requested_steps) if step in by_step]

    raise ValueError(f"Unsupported checkpoint selector: {selector}")


def is_onestep(exp: Experiment) -> bool:
    if "distill_type" in exp.args:
        return True
    return int(exp.args.get("infer_steps", 25)) == 1


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_job(exp: Experiment, checkpoint: Path, args: argparse.Namespace) -> Job:
    script = "inference_vosr_onestep.py" if is_onestep(exp) else "inference_vosr.py"
    step_name = checkpoint.name
    output_dir = Path(args.output_root) / safe_name(exp.name) / step_name
    log_dir = Path(args.output_root) / "_logs"
    log_path = log_dir / f"{safe_name(exp.name)}__{step_name}.log"

    input_dir = args.input_dir or exp.args.get("test_lq_dir")
    if not input_dir:
        raise ValueError(f"{exp.root} has no test_lq_dir; pass --input-dir.")

    cmd = [
        sys.executable,
        script,
        "-c",
        checkpoint.as_posix(),
        "-i",
        str(input_dir),
        "-o",
        output_dir.as_posix(),
        "--align_method",
        args.align_method,
    ]

    if args.upscale is not None:
        cmd += ["-u", str(args.upscale)]
    elif exp.args.get("test_upscale") is not None:
        cmd += ["-u", str(exp.args["test_upscale"])]

    optional_flags = (
        ("--tile_size", args.tile_size),
        ("--tile_overlap", args.tile_overlap),
        ("--infer_steps", args.infer_steps),
        ("--seed", args.seed),
    )
    for flag, value in optional_flags:
        if value is not None:
            cmd += [flag, str(value)]

    if not is_onestep(exp):
        if args.cfg_scale is not None:
            cmd += ["--cfg_scale", str(args.cfg_scale)]
        if args.weak_cond_strength_aelq is not None:
            cmd += ["--weak_cond_strength_aelq", str(args.weak_cond_strength_aelq)]

    if args.force_rerun:
        cmd.append("--force_rerun")

    return Job(exp, checkpoint, script, output_dir, cmd, log_path)


def run_job(job: Job, gpu: str) -> tuple[Job, int, float]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start = time.time()
    with job.log_path.open("w", encoding="utf-8") as log:
        log.write(f"GPU: {gpu}\n")
        log.write("Command: " + shlex.join(job.command) + "\n\n")
        log.flush()
        proc = subprocess.run(
            job.command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return job, proc.returncode, time.time() - start


def run_slot(slot_name: str, gpu: str, jobs: list[Job]) -> list[tuple[Job, int, float, str]]:
    results = []
    for job in jobs:
        job_result = run_job(job, gpu)
        results.append((*job_result, slot_name))
    return results


def main() -> int:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id.")
    if args.jobs_per_gpu < 1:
        raise SystemExit("--jobs-per-gpu must be >= 1.")

    experiments = discover_experiments(args.exp_roots)
    experiments = filter_experiments(experiments, args.include, args.exclude)
    jobs: list[Job] = []
    for exp in experiments:
        for checkpoint in select_checkpoints(exp, args.checkpoint):
            jobs.append(build_job(exp, checkpoint, args))

    if not jobs:
        print("No inference jobs discovered.")
        return 1

    print(f"Discovered {len(experiments)} experiments, launching {len(jobs)} jobs.")
    slots = [(f"{gpu}:{slot}", gpu) for gpu in gpus for slot in range(args.jobs_per_gpu)]
    slot_jobs: list[list[Job]] = [[] for _ in slots]

    for index, job in enumerate(jobs):
        slot_index = index % len(slots)
        gpu = slots[slot_index][1]
        slot_jobs[slot_index].append(job)
        print(f"[GPU {gpu}] {shlex.join(job.command)}")

    if args.dry_run:
        return 0

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = {}
        for (slot_name, gpu), assigned_jobs in zip(slots, slot_jobs):
            if assigned_jobs:
                futures[pool.submit(run_slot, slot_name, gpu, assigned_jobs)] = slot_name

        for future in concurrent.futures.as_completed(futures):
            for job, returncode, elapsed, slot_name in future.result():
                status = "OK" if returncode == 0 else f"FAIL({returncode})"
                print(
                    f"[{status}] slot={slot_name} {job.experiment.name} {job.checkpoint.name} "
                    f"{elapsed / 60:.1f} min log={job.log_path}"
                )
                if returncode != 0:
                    failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
