#!/usr/bin/env python3
"""
Ablation study: compare Hessian approximation methods on LDS.

Runs LDS experiment with each of:
  - ekfac     (EK-FAC, default)
  - kfac      (K-FAC, no corrected eigenvalues)
  - diagonal  (diagonal Fisher)
  - plain_dot (raw gradient inner product, no Hessian)

For each cell (env, dataset, seed), trains the model ONCE, then computes
influence scores + LDS under each approximation.

Usage:
    python run_ablation.py
    python run_ablation.py --cells halfcheetah:medium --seeds 0 1 2
    python run_ablation.py --smoke-test
"""

import argparse
import itertools
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from configs import (
    ExperimentConfig, InfluenceConfig, DiffuserConfig, DTRAKConfig,
    EvaluationConfig, GRID_CELLS, GRID_SEEDS,
)
from diffuser_minimal import (
    GaussianDiffusion, TemporalUNet, TrajectoryDataset, train, plan,
)
from influence_functions import TrajectoryInfluenceComputer
from baselines import RandomAttribution
from evaluation import save_results

logger = logging.getLogger("run_ablation")

HESSIAN_MODES = ["ekfac", "kfac", "diagonal", "plain_dot"]
RESULTS_DIR = "/mnt/sdb/ljc/DADiffCtrl/analysis"
CHECKPOINT_BASE = "/mnt/sdb/ljc/DADiffCtrl/checkpoints"


def build_config(
    env: str, dataset: str, seed: int,
    hessian_approx: str = "ekfac",
    smoke_test: bool = False, debug: bool = False,
) -> ExperimentConfig:
    """Build an ExperimentConfig for one ablation cell."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffuser_cfg = DiffuserConfig(device=device)
    influence_cfg = InfluenceConfig(
        hessian_approx=hessian_approx, device=device,
    )
    dtrak_cfg = DTRAKConfig(device=device)
    eval_cfg = EvaluationConfig(base_seed=seed)

    if smoke_test:
        diffuser_cfg.dim = 32
        diffuser_cfg.dim_mults = (1, 2)
        diffuser_cfg.n_residual_blocks = 1
        diffuser_cfg.n_diffusion_steps = 20
        diffuser_cfg.n_train_steps = 2000
        diffuser_cfg.batch_size = 32
        diffuser_cfg.n_plan_samples = 4
        diffuser_cfg.n_planning_steps = 20
        diffuser_cfg.log_interval = 500
        diffuser_cfg.save_interval = 1000
        influence_cfg.n_eigenvectors = 10
        influence_cfg.gradient_batch_size = 8
        dtrak_cfg.projection_dim = 256
        eval_cfg.n_subsets = 3
        eval_cfg.retrain_steps = 1000
    elif debug:
        diffuser_cfg.dim = 32
        diffuser_cfg.dim_mults = (1, 2)
        diffuser_cfg.n_residual_blocks = 1
        diffuser_cfg.n_diffusion_steps = 20
        diffuser_cfg.n_train_steps = 500
        diffuser_cfg.batch_size = 32
        diffuser_cfg.n_plan_samples = 4
        diffuser_cfg.n_planning_steps = 20
        influence_cfg.n_eigenvectors = 10
        influence_cfg.gradient_batch_size = 8
        dtrak_cfg.projection_dim = 256
        eval_cfg.n_subsets = 3
        eval_cfg.retrain_steps = 200

    config = ExperimentConfig(
        env_name=env, dataset=dataset, experiment="lds",
        diffuser=diffuser_cfg, influence=influence_cfg,
        dtrak=dtrak_cfg, evaluation=eval_cfg,
        results_dir=RESULTS_DIR,
    )
    config.checkpoint_dir = os.path.join(
        CHECKPOINT_BASE, f"{env}_{dataset}_seed{seed}",
    )
    return config


def run_ablation_cell(
    env: str, dataset: str, seed: int,
    debug: bool = False, smoke_test: bool = False,
) -> Dict:
    """Run ablation for one (env, dataset, seed) cell."""
    # Build base config (ekfac; we override per mode below)
    config = build_config(env, dataset, seed, "ekfac", smoke_test, debug)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(config.diffuser.device)
    max_traj = 50 if (debug or smoke_test) else None

    # Load dataset
    ds = TrajectoryDataset(
        env_name=env, dataset_variant=dataset,
        horizon=config.diffuser.horizon,
        max_trajectories=max_traj,
    )

    # Train model ONCE (or load checkpoint)
    ckpt_path = os.path.join(config.checkpoint_dir, "diffuser_final.pt")
    if os.path.exists(ckpt_path):
        logger.info("Loading checkpoint: %s", ckpt_path)
        unet = TemporalUNet(
            transition_dim=config.transition_dim,
            dim=config.diffuser.dim,
            dim_mults=config.diffuser.dim_mults,
            n_residual_blocks=config.diffuser.n_residual_blocks,
            dropout=config.diffuser.dropout,
        )
        diffusion = GaussianDiffusion(unet, config.diffuser).to(device)
        diffusion.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
    else:
        logger.info("Training Diffuser for %s_%s_seed%d...", env, dataset, seed)
        diffusion, ds = train(config, dataset=ds, verbose=True)
    diffusion.eval()

    # Generate reference plan ONCE
    plans = plan(diffusion, ds, config, reward_guidance=False)
    ref_plan = plans[0]
    ref_plan_tensor = torch.from_numpy(
        (ref_plan - ds.data_mean) / ds.data_std
    ).float()

    max_inf = 100 if (debug or smoke_test) else None
    n_effective = min(max_inf or len(ds), len(ds))

    # Import LDS evaluator
    from run_experiments import run_lds_experiment

    ablation_results: Dict = {}

    for mode in HESSIAN_MODES:
        logger.info("=" * 50)
        logger.info("Ablation: hessian_approx = %s", mode)
        logger.info("=" * 50)

        # Build influence config for this mode
        mode_config = build_config(env, dataset, seed, mode, smoke_test, debug)

        # Compute influence scores
        tic = TrajectoryInfluenceComputer(
            model=diffusion, dataset=ds,
            config=mode_config.influence,
            diffusion_steps=config.diffuser.n_diffusion_steps,
        )
        t0 = time.time()
        tic.compute_hessian_approximation(n_samples=n_effective)
        tif_scores = tic.compute_all_influences_batched(
            plan_tau=ref_plan_tensor,
            proxy_type=mode_config.influence.proxy_type,
            max_samples=n_effective,
        )
        elapsed = time.time() - t0
        logger.info("  TIF-%s computed in %.1fs", mode, elapsed)
        del tic
        torch.cuda.empty_cache()

        # Random baseline for reference
        rand = RandomAttribution(ds, seed=seed)
        rand_scores = rand.compute_scores()[:n_effective]

        influence_scores = {
            f"TIF-{mode}": tif_scores,
            "Random": rand_scores,
        }

        # Run LDS (reuses shared cache — retrained models are mode-independent)
        lds_cache = os.path.join(
            RESULTS_DIR, "lds_cache", f"{env}_{dataset}_seed{seed}",
        )
        lds_result = run_lds_experiment(
            mode_config, diffusion, ds, influence_scores,
            ref_plan=ref_plan, cache_dir=lds_cache,
        )

        tif_key = f"TIF-{mode}"
        tif_lds = lds_result.get("methods", {}).get(tif_key, {})
        ablation_results[mode] = {
            "lds_spearman": tif_lds.get("lds_spearman", None),
            "lds_pearson": tif_lds.get("lds_pearson", None),
            "compute_time_s": elapsed,
        }

    return ablation_results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="TIF Hessian Ablation")
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cells = GRID_CELLS if args.cells is None else [
        tuple(c.split(":")) for c in args.cells
    ]
    seeds = GRID_SEEDS if args.seeds is None else args.seeds

    if args.smoke_test:
        cells = [("halfcheetah", "medium")]
        seeds = [0]

    all_results: Dict = {}
    for (env, ds_name), seed in itertools.product(cells, seeds):
        key = f"{env}_{ds_name}_seed{seed}"
        logger.info("Running ablation for %s", key)
        result = run_ablation_cell(
            env, ds_name, seed, args.debug, args.smoke_test,
        )
        all_results[key] = result

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = save_results(
        all_results, f"ablation_hessian_{timestamp}", RESULTS_DIR,
    )
    logger.info("Ablation results saved to: %s", filepath)

    # Print summary table
    print("\n" + "=" * 70)
    print("Ablation Table: LDS Spearman rho")
    print("=" * 70)

    # Header
    cell_keys = [
        f"{env}_{ds}_seed{seed}"
        for (env, ds), seed in itertools.product(cells, seeds)
    ]
    header = f"{'Mode':<12}"
    for k in cell_keys:
        header += f"  {k:<24}"
    print(header)
    print("-" * len(header))

    for mode in HESSIAN_MODES:
        row = f"{mode:<12}"
        for k in cell_keys:
            val = all_results.get(k, {}).get(mode, {}).get("lds_spearman")
            if val is not None:
                row += f"  {val:+.4f}                   "
            else:
                row += f"  {'N/A':<24}"
        print(row)

    print()


if __name__ == "__main__":
    main()
