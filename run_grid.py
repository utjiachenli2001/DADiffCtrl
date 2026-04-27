#!/usr/bin/env python3
"""
Grid runner for Trajectory Influence Functions experiments.

Orchestrates runs across multiple (env, dataset, seed) cells.
Supports sequential single-GPU and parallel multi-GPU execution.

Usage:
    # Sequential (single GPU):
    python run_grid.py

    # Parallel (multi-GPU, 4 workers):
    python run_grid.py --mode parallel --n-workers 4

    # Specific cells only:
    python run_grid.py --cells halfcheetah:medium hopper:medium --seeds 0 1

    # Smoke test (single cell, all experiments, minimal sizes):
    python run_grid.py --smoke-test

    # Dry run (print commands without executing):
    python run_grid.py --dry-run

    # Resume: skip cells whose result JSON already exists:
    python run_grid.py --resume
"""

import argparse
import glob
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Allow importing from the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from configs import GRID_CELLS, GRID_SEEDS

logger = logging.getLogger("run_grid")

RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_experiments.py")
CHECKPOINT_BASE = os.environ.get("DADIFFCTRL_CHECKPOINT_DIR", "./checkpoints")
RESULTS_DIR = os.environ.get("DADIFFCTRL_RESULTS_DIR", "./analysis")
FAILED_CELLS_FILE = os.path.join(RESULTS_DIR, "failed_cells.json")


def parse_args():
    p = argparse.ArgumentParser(description="TIF Grid Runner")
    p.add_argument(
        "--cells", nargs="+", default=None,
        help="Cells as env:dataset (e.g., halfcheetah:medium). Default: all 9.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to run. Default: 0 1 2.",
    )
    p.add_argument(
        "--experiments", nargs="+", default=None,
        choices=["lds", "safety", "curation", "intervention", "all"],
        help="Experiments to run. Default: all.",
    )
    p.add_argument(
        "--hessian-approx", default="ekfac",
        choices=["ekfac", "kfac", "diagonal", "plain_dot"],
    )
    p.add_argument(
        "--mode", default="sequential",
        choices=["sequential", "parallel"],
    )
    p.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel GPU workers (for parallel mode).",
    )
    p.add_argument(
        "--gpu-ids", nargs="+", type=int, default=None,
        help="GPU IDs to use (cycles through for parallel mode).",
    )
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip cells whose result JSON already exists in analysis/.",
    )
    p.add_argument(
        "--retry-failed", action="store_true",
        help="Retry cells previously recorded in failed_cells.json.",
    )
    return p.parse_args()


def _result_exists(env: str, dataset: str, seed: int, exp: str) -> bool:
    """Check if a result JSON already exists for this cell."""
    pattern = os.path.join(
        RESULTS_DIR,
        f"{env}_{dataset}_seed{seed}_{exp}_*.json",
    )
    matches = glob.glob(pattern)
    return len(matches) > 0


def _load_failed_cells() -> List[Dict]:
    """Load previously failed cells from JSON."""
    if os.path.exists(FAILED_CELLS_FILE):
        with open(FAILED_CELLS_FILE) as f:
            return json.load(f)
    return []


def _save_failed_cells(failed: List[Dict]) -> None:
    """Persist failed cells to JSON for later retry."""
    os.makedirs(os.path.dirname(FAILED_CELLS_FILE) or ".", exist_ok=True)
    with open(FAILED_CELLS_FILE, "w") as f:
        json.dump(failed, f, indent=2)
    logger.info("Saved %d failed cells to %s", len(failed), FAILED_CELLS_FILE)


def build_commands(args) -> List[Tuple[List[str], Dict]]:
    """Build list of (command_parts, metadata_dict) for each grid cell."""
    cells = GRID_CELLS
    if args.cells:
        cells = [tuple(c.split(":")) for c in args.cells]
    seeds = args.seeds or GRID_SEEDS
    experiments = args.experiments or ["all"]

    # Smoke test overrides
    if args.smoke_test:
        cells = [("halfcheetah", "medium")]
        seeds = [0]
        experiments = ["all"]

    commands = []
    skipped = 0
    for (env, dataset), seed in itertools.product(cells, seeds):
        for exp in experiments:
            # Resume: skip if result already exists
            if args.resume and _result_exists(env, dataset, seed, exp):
                logger.info(
                    "SKIP (result exists): %s_%s_seed%d_%s",
                    env, dataset, seed, exp,
                )
                skipped += 1
                continue

            cmd_parts = [
                sys.executable, RUN_SCRIPT,
                "--env", env,
                "--dataset", dataset,
                "--seed", str(seed),
                "--experiment", exp,
                "--hessian-approx", args.hessian_approx,
            ]
            if args.smoke_test:
                cmd_parts.append("--smoke-test")
            elif args.debug:
                cmd_parts.append("--debug")

            # Check for existing checkpoint to skip retraining
            ckpt_dir = os.path.join(
                CHECKPOINT_BASE,
                f"{env}_{dataset}_seed{seed}",
            )
            ckpt_path = os.path.join(ckpt_dir, "diffuser_final.pt")
            if os.path.exists(ckpt_path):
                cmd_parts.extend(["--checkpoint", ckpt_path])

            meta = {
                "env": env, "dataset": dataset,
                "seed": seed, "experiment": exp,
            }
            commands.append((cmd_parts, meta))

    if skipped > 0:
        logger.info("Skipped %d cells due to existing results (--resume).", skipped)

    return commands


def _cleanup_between_runs():
    """Sleep briefly and hint CUDA to release memory between runs."""
    import torch
    time.sleep(10)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential(
    commands: List[Tuple[List[str], Dict]],
    dry_run: bool = False,
    retry_once: bool = True,
):
    """Run commands one at a time with optional retry on failure."""
    total = len(commands)
    failed = []
    for i, (cmd_parts, meta) in enumerate(commands):
        cmd_str = " ".join(cmd_parts)
        logger.info("[%d/%d] %s", i + 1, total, meta)
        logger.info("  CMD: %s", cmd_str)
        if dry_run:
            continue

        result = subprocess.run(cmd_parts)
        if result.returncode != 0:
            logger.error("FAILED (rc=%d): %s", result.returncode, meta)
            if retry_once:
                logger.info("RETRYING once after cleanup...")
                _cleanup_between_runs()
                result2 = subprocess.run(cmd_parts)
                if result2.returncode != 0:
                    logger.error("RETRY FAILED (rc=%d): %s", result2.returncode, meta)
                    failed.append(meta)
                else:
                    logger.info("RETRY SUCCEEDED: %s", meta)
            else:
                failed.append(meta)
        else:
            logger.info("DONE: %s", meta)

        # Cleanup between runs to reduce memory fragmentation
        _cleanup_between_runs()

    if failed:
        logger.error("%d/%d runs failed:", len(failed), total)
        for f in failed:
            logger.error("  %s", f)
        _save_failed_cells(failed)
    else:
        logger.info("All %d runs completed successfully.", total)


def run_parallel(
    commands: List[Tuple[List[str], Dict]],
    n_workers: int,
    gpu_ids: Optional[List[int]] = None,
    dry_run: bool = False,
):
    """Run commands in parallel across GPUs with retry on failure."""
    if gpu_ids is None:
        gpu_ids = list(range(n_workers))

    total = len(commands)
    processes: Dict[int, Tuple] = {}  # pid -> (proc, meta, gpu_id, cmd_parts, retried)
    cmd_queue = [(cmd, meta, False) for cmd, meta in commands]  # (cmd, meta, retried)
    completed = 0
    failed = []

    while cmd_queue or processes:
        # Launch up to n_workers
        while cmd_queue and len(processes) < n_workers:
            cmd_parts, meta, retried = cmd_queue.pop(0)
            gpu_id = gpu_ids[len(processes) % len(gpu_ids)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info("Launching on GPU %d: %s", gpu_id, meta)
            if dry_run:
                completed += 1
                continue
            proc = subprocess.Popen(cmd_parts, env=env)
            processes[proc.pid] = (proc, meta, gpu_id, cmd_parts, retried)

        if not processes:
            break

        # Poll for completion
        for pid in list(processes.keys()):
            proc, meta, gpu, cmd_parts, retried = processes[pid]
            ret = proc.poll()
            if ret is not None:
                if ret != 0:
                    if not retried:
                        # Retry once
                        logger.warning(
                            "FAILED (GPU %d, rc=%d), will retry: %s",
                            gpu, ret, meta,
                        )
                        cmd_queue.append((cmd_parts, meta, True))
                    else:
                        logger.error(
                            "RETRY FAILED [%d/%d] (GPU %d, rc=%d): %s",
                            completed + 1, total, gpu, ret, meta,
                        )
                        completed += 1
                        failed.append(meta)
                else:
                    completed += 1
                    logger.info(
                        "DONE [%d/%d] (GPU %d): %s",
                        completed, total, gpu, meta,
                    )
                del processes[pid]

        if processes:
            time.sleep(5)

    if failed:
        logger.error("%d/%d runs failed.", len(failed), total)
        _save_failed_cells(failed)
    else:
        logger.info("All %d runs completed successfully.", total)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    commands = build_commands(args)
    logger.info("Grid: %d total runs", len(commands))

    if args.mode == "sequential":
        run_sequential(commands, args.dry_run)
    elif args.mode == "parallel":
        run_parallel(commands, args.n_workers, args.gpu_ids, args.dry_run)


if __name__ == "__main__":
    main()
