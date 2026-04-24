#!/usr/bin/env python3
"""
Main experiment runner for Trajectory Influence Functions.

Wires together the Diffuser model, influence computation, baselines,
and evaluation protocols.

Usage
-----
    # LDS experiment on HalfCheetah-medium
    python run_experiments.py --env halfcheetah --dataset medium --experiment lds

    # Safety attribution on Hopper-medium-expert
    python run_experiments.py --env hopper --dataset medium-expert --experiment safety

    # Data curation on Walker2d-medium-replay
    python run_experiments.py --env walker2d --dataset medium-replay --experiment curation

    # Run all experiments on a given env/dataset
    python run_experiments.py --env halfcheetah --dataset medium --experiment all

    # Quick smoke test (tiny model, few steps)
    python run_experiments.py --env halfcheetah --dataset medium --experiment lds --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure the core_code directory is on the path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from configs import (
    DTRAKConfig,
    DiffuserConfig,
    EvaluationConfig,
    ExperimentConfig,
    InfluenceConfig,
)
from diffuser_minimal import (
    GaussianDiffusion,
    TemporalUNet,
    TrajectoryDataset,
    plan,
    train,
)
from influence_functions import TrajectoryInfluenceComputer
from baselines import (
    NearestNeighborAttribution,
    RandomAttribution,
    RewardRanking,
    TrajectoryDTRAK,
)
from evaluation import (
    DataCurationEvaluator,
    SafetyAttributionAUC,
    TrajectoryLDS,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiments")

RESULTS_DIR = "/mnt/sdb/ljc/DADiffCtrl/analysis"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trajectory Influence Functions — Experiment Runner"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="halfcheetah",
        choices=["halfcheetah", "hopper", "walker2d"],
        help="D4RL environment name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medium",
        choices=["medium", "medium-replay", "medium-expert", "expert"],
        help="D4RL dataset variant.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="lds",
        choices=["lds", "safety", "curation", "intervention", "all"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--hessian-approx",
        type=str,
        default="ekfac",
        choices=["ekfac", "kfac", "diagonal", "plain_dot"],
        help="Hessian approximation for influence functions.",
    )
    parser.add_argument(
        "--proxy-type",
        type=str,
        default="likelihood",
        choices=["likelihood", "reward_conditioned", "constraint_satisfaction", "conditioning_gap"],
        help="Proxy measurement type for influence computation.",
    )
    parser.add_argument(
        "--n-train-steps",
        type=int,
        default=None,
        help="Override number of training steps.",
    )
    parser.add_argument(
        "--n-subsets",
        type=int,
        default=None,
        help="Override number of LDS subsets.",
    )
    parser.add_argument(
        "--max-influence-samples",
        type=int,
        default=None,
        help="Limit number of training samples for influence computation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: tiny model, few steps, small dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained model checkpoint to load.",
    )
    parser.add_argument(
        "--lds-worker",
        action="store_true",
        help=(
            "Worker mode: skip influence/baseline computation. "
            "Only retrain LDS subsets (in reverse) and populate the "
            "shared cache. Pair with a main process for parallel LDS."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Smoke test: run all experiments end-to-end with minimal sizes. "
            "Implies --debug model architecture but uses slightly longer "
            "training for meaningful (if noisy) results."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from CLI arguments."""
    device = args.device if torch.cuda.is_available() else "cpu"

    diffuser_cfg = DiffuserConfig(device=device)
    influence_cfg = InfluenceConfig(
        hessian_approx=args.hessian_approx,
        device=device,
    )
    dtrak_cfg = DTRAKConfig(device=device)
    eval_cfg = EvaluationConfig(base_seed=args.seed)

    if args.n_train_steps is not None:
        diffuser_cfg.n_train_steps = args.n_train_steps
    if args.n_subsets is not None:
        eval_cfg.n_subsets = args.n_subsets

    if getattr(args, "smoke_test", False):
        logger.info("SMOKE TEST mode: minimal end-to-end validation.")
        # Tiny model architecture (same as debug)
        diffuser_cfg.dim = 32
        diffuser_cfg.dim_mults = (1, 2)
        diffuser_cfg.n_residual_blocks = 1
        diffuser_cfg.n_diffusion_steps = 20
        diffuser_cfg.batch_size = 32
        diffuser_cfg.n_plan_samples = 4
        diffuser_cfg.n_planning_steps = 20
        diffuser_cfg.log_interval = 500
        diffuser_cfg.save_interval = 1000
        # Slightly longer training than debug for meaningful results
        diffuser_cfg.n_train_steps = 2000
        influence_cfg.n_eigenvectors = 10
        influence_cfg.gradient_batch_size = 8
        dtrak_cfg.projection_dim = 256
        eval_cfg.n_subsets = 3
        eval_cfg.retrain_steps = 1000
        eval_cfg.prune_fractions = [0.3]
        eval_cfg.intervention_prune_fractions = [0.1]
        eval_cfg.n_rollout_episodes = 3
        args.debug = True  # also set debug for max_traj limit
    elif args.debug:
        logger.info("DEBUG mode: using tiny configuration.")
        diffuser_cfg.dim = 32
        diffuser_cfg.dim_mults = (1, 2)
        diffuser_cfg.n_residual_blocks = 1
        diffuser_cfg.n_diffusion_steps = 20
        diffuser_cfg.n_train_steps = 500
        diffuser_cfg.batch_size = 32
        diffuser_cfg.n_plan_samples = 4
        diffuser_cfg.n_planning_steps = 20
        diffuser_cfg.log_interval = 100
        diffuser_cfg.save_interval = 250
        influence_cfg.n_eigenvectors = 10
        influence_cfg.gradient_batch_size = 8
        dtrak_cfg.projection_dim = 256
        eval_cfg.n_subsets = 3
        eval_cfg.retrain_steps = 200
        eval_cfg.prune_fractions = [0.2, 0.5]

    influence_cfg.proxy_type = args.proxy_type

    config = ExperimentConfig(
        env_name=args.env,
        dataset=args.dataset,
        experiment=args.experiment,
        diffuser=diffuser_cfg,
        influence=influence_cfg,
        dtrak=dtrak_cfg,
        evaluation=eval_cfg,
        results_dir=RESULTS_DIR,
    )

    # Per-cell checkpoint directory to avoid collisions in grid runs
    config.checkpoint_dir = os.path.join(
        config.checkpoint_dir,
        f"{config.env_name}_{config.dataset}_seed{args.seed}",
    )

    return config


# ---------------------------------------------------------------------------
# Proxy function factory
# ---------------------------------------------------------------------------


def make_proxy_fn(config: ExperimentConfig, dataset, diffusion_model=None):
    """Return a callable that evaluates the SAME proxy used for influence attribution.

    The proxy must match `config.influence.proxy_type` exactly, so that the
    LDS experiment measures whether influence scores correctly predict the
    change in the attributed quantity.

    For model-dependent proxies (likelihood, reward_conditioned, constraint,
    conditioning_gap), this requires the diffusion model.  For these we
    re-evaluate the diffusion loss on the generated plan.

    Args:
        config: ExperimentConfig.
        dataset: TrajectoryDataset (for normalization).
        diffusion_model: trained GaussianDiffusion model (needed for
                         model-dependent proxies).
    """
    proxy_type = config.influence.proxy_type
    state_dim = config.state_dim
    device = torch.device(config.diffuser.device if torch.cuda.is_available() else "cpu")

    def proxy_fn(plan_np: np.ndarray) -> float:
        """Evaluate the proxy on a generated plan.

        Args:
            plan_np: (n_samples, H, D) or (H, D) in ORIGINAL (unnormalized) space.
        """
        if plan_np.ndim == 3:
            plan_np = plan_np[0]  # Take first sample — matches reference plan

        # Normalize
        plan_norm = (plan_np - dataset.data_mean) / dataset.data_std
        plan_t = torch.from_numpy(plan_norm).float().unsqueeze(0).to(device)

        if diffusion_model is None or proxy_type == "velocity":
            # Fallback: simple velocity proxy (no model needed)
            return float(plan_np[:, 0].sum())

        diffusion_model.eval()
        n_steps = config.diffuser.n_diffusion_steps

        if proxy_type == "likelihood":
            # f_lik = E_t[||eps - eps_theta(tau_t, t)||^2]
            with torch.no_grad():
                total_loss = 0.0
                n_t = 20
                for _ in range(n_t):
                    t = torch.randint(0, n_steps, (1,), device=device)
                    noise = torch.randn_like(plan_t)
                    x_noisy = diffusion_model.q_sample(plan_t, t, noise)
                    pred = diffusion_model.model(x_noisy, t)
                    total_loss += F.mse_loss(pred, noise).item()
                return total_loss / n_t

        elif proxy_type == "reward_conditioned":
            with torch.no_grad():
                total_loss = 0.0
                n_t = 20
                for _ in range(n_t):
                    t = torch.randint(0, max(1, n_steps // 5), (1,), device=device)
                    noise = torch.randn_like(plan_t)
                    x_noisy = diffusion_model.q_sample(plan_t, t, noise)
                    pred = diffusion_model.model(x_noisy, t)
                    total_loss += F.mse_loss(pred, noise).item()
                return total_loss / n_t

        elif proxy_type == "constraint_satisfaction":
            with torch.no_grad():
                states = plan_t[:, :, :state_dim]
                bound = 3.0
                violation = torch.clamp(states.abs() - bound, min=0.0)
                lambda_con = 1.0
                w_h = torch.exp(-lambda_con * violation.sum(dim=-1))  # (1, H)
                total_loss = 0.0
                n_t = 20
                for _ in range(n_t):
                    t = torch.randint(0, n_steps, (1,), device=device)
                    noise = torch.randn_like(plan_t)
                    x_noisy = diffusion_model.q_sample(plan_t, t, noise)
                    pred = diffusion_model.model(x_noisy, t)
                    per_step = ((pred - noise) ** 2).mean(dim=-1)
                    total_loss += (w_h * per_step).sum().item()
                return total_loss / n_t

        elif proxy_type == "conditioning_gap":
            with torch.no_grad():
                n_t = 20
                loss_low = 0.0
                for _ in range(n_t):
                    t = torch.randint(0, max(1, n_steps // 5), (1,), device=device)
                    noise = torch.randn_like(plan_t)
                    x_noisy = diffusion_model.q_sample(plan_t, t, noise)
                    pred = diffusion_model.model(x_noisy, t)
                    loss_low += F.mse_loss(pred, noise).item()
                loss_low /= n_t

                loss_high = 0.0
                for _ in range(n_t):
                    t = torch.randint(n_steps * 4 // 5, n_steps, (1,), device=device)
                    noise = torch.randn_like(plan_t)
                    x_noisy = diffusion_model.q_sample(plan_t, t, noise)
                    pred = diffusion_model.model(x_noisy, t)
                    loss_high += F.mse_loss(pred, noise).item()
                loss_high /= n_t

                return loss_low - loss_high

        else:
            return float(plan_np[:, 0].sum())

    return proxy_fn


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def run_lds_experiment(
    config: ExperimentConfig,
    diffusion: GaussianDiffusion,
    dataset: TrajectoryDataset,
    influence_scores: Dict[str, np.ndarray],
    ref_plan: Optional[np.ndarray] = None,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Run the Linear Datamodeling Score experiment.

    Compares LDS for our method (TIF) vs. all baselines.

    Args:
        ref_plan: (H, D) the SAME reference plan used to compute influence
                  scores in main(). Must be provided so that full_proxy is
                  evaluated on the exact plan that TIF attributed.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Linear Datamodeling Score (LDS)")
    logger.info("=" * 60)

    proxy_fn = make_proxy_fn(config, dataset, diffusion_model=diffusion)

    # Factory: creates proxy evaluator bound to a specific model
    def proxy_fn_factory(model):
        return make_proxy_fn(config, dataset, diffusion_model=model)

    # Compute full-model proxy on the SAME plan used for influence attribution
    if ref_plan is None:
        logger.warning(
            "ref_plan not provided to run_lds_experiment; generating a new plan. "
            "This may mismatch the plan used for influence scores!"
        )
        plans = plan(diffusion, dataset, config, reward_guidance=False)
        ref_plan = plans[0]
    full_proxy = proxy_fn(ref_plan)
    logger.info("Full-model proxy value: %.4f", full_proxy)

    lds_evaluator = TrajectoryLDS(
        model_class=None,  # not needed — train_fn handles creation
        train_fn=train,
        plan_fn=lambda m, d, c: plan(m, d, c, reward_guidance=False),
        proxy_fn=proxy_fn,
        dataset=dataset,
        config=config,
        eval_config=config.evaluation,
        proxy_fn_factory=proxy_fn_factory,
        cache_dir=cache_dir,
    )

    results = {"full_model_proxy": full_proxy, "methods": {}}

    for method_name, scores in influence_scores.items():
        logger.info("Computing LDS for method: %s", method_name)
        lds_result = lds_evaluator.compute_lds(
            influence_scores=scores,
            full_model_proxy=full_proxy,
        )
        results["methods"][method_name] = lds_result

    # Summary table
    logger.info("\n--- LDS Results ---")
    logger.info("%-25s  Spearman  Pearson", "Method")
    logger.info("-" * 55)
    for name, r in results["methods"].items():
        logger.info(
            "%-25s  %+.4f    %+.4f",
            name,
            r["lds_spearman"],
            r["lds_pearson"],
        )

    return results


def run_safety_experiment(
    config: ExperimentConfig,
    diffusion: GaussianDiffusion,
    dataset: TrajectoryDataset,
    influence_scores: Dict[str, np.ndarray],
) -> Dict:
    """Run the safety attribution experiment.

    Recomputes TIF scores with the constraint_satisfaction proxy specifically,
    to measure how training data affects constraint violations. Plan-dependent
    baselines (NearestNeighbor, TRAK) are also recomputed on the selected unsafe
    plan to ensure a fair comparison — the scores from main() were computed on
    the original ref_plan which may differ from the constraint-violating plan.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Safety Attribution AUC")
    logger.info("=" * 60)

    safety_eval = SafetyAttributionAUC(dataset, config.evaluation)
    safety_labels = safety_eval.label_training_safety()

    # Get the reference plan (should already exist from main)
    n_scores = len(next(iter(influence_scores.values())))
    safety_labels = safety_labels[:n_scores]

    # Recompute TIF scores with constraint_satisfaction proxy
    # (This is the paper's claimed protocol: attribute constraint violations)
    # Step 1: Generate plans and select one that violates constraints
    logger.info("Generating plans and selecting constraint-violating plan...")
    plans = plan(diffusion, dataset, config, reward_guidance=False)

    # Score each plan for safety violations
    state_dim = config.state_dim
    best_plan_idx = 0
    best_violation = 0.0
    for i in range(plans.shape[0]):
        p = plans[i]
        # Normalize to check in normalized space
        p_norm = (p - dataset.data_mean) / dataset.data_std
        states = p_norm[:, :state_dim]
        violation = np.maximum(np.abs(states) - 3.0, 0.0).sum()
        if violation > best_violation:
            best_violation = violation
            best_plan_idx = i
    ref_plan = plans[best_plan_idx]
    logger.info(
        "Selected plan %d with violation score %.4f",
        best_plan_idx, best_violation,
    )
    if best_violation == 0.0:
        logger.warning(
            "No constraint-violating plan found in batch. "
            "Safety AUC may be less meaningful. Using plan 0."
        )
    ref_plan_tensor = torch.from_numpy(
        (ref_plan - dataset.data_mean) / dataset.data_std
    ).float()

    safety_influence_cfg = InfluenceConfig(
        hessian_approx=config.influence.hessian_approx,
        damping=config.influence.damping,
        n_eigenvectors=config.influence.n_eigenvectors,
        proxy_type="constraint_satisfaction",
        gradient_batch_size=config.influence.gradient_batch_size,
        device=config.influence.device,
    )
    tic = TrajectoryInfluenceComputer(
        model=diffusion,
        dataset=dataset,
        config=safety_influence_cfg,
        diffusion_steps=config.diffuser.n_diffusion_steps,
    )
    tic.compute_hessian_approximation(n_samples=n_scores)
    safety_tif_scores = tic.compute_all_influences_batched(
        plan_tau=ref_plan_tensor,
        proxy_type="constraint_satisfaction",
        max_samples=n_scores,
    )

    # Recompute plan-dependent baselines on the selected unsafe plan
    # (NN and TRAK scores from main() were computed on the original ref_plan,
    # not on this selected unsafe plan -- they must match for a fair comparison)
    logger.info("Recomputing plan-dependent baselines on unsafe plan...")
    nn_baseline = NearestNeighborAttribution(dataset)
    nn_scores_safety = nn_baseline.compute_scores(ref_plan_tensor)[:n_scores]

    # Free safety TIF compute (Hessian factors) before D-TRAK allocates 18GB+
    del tic
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    dtrak_baseline = TrajectoryDTRAK(
        model=diffusion,
        dataset=dataset,
        config=config.dtrak,
        diffusion_steps=config.diffuser.n_diffusion_steps,
    )
    dtrak_baseline.precompute_training_features(max_samples=n_scores)
    dtrak_scores_safety = dtrak_baseline.compute_scores(ref_plan_tensor)[:n_scores]
    del dtrak_baseline
    torch.cuda.empty_cache()

    # Build scores dict: only include methods evaluated on the same unsafe plan.
    # Drop "TIF (ours)" from main() since it was computed on the original ref_plan,
    # not on this selected unsafe plan. Keep proxy-agnostic baselines (Random,
    # RewardRanking) which don't depend on the plan.
    safety_scores = {}
    for k, v in influence_scores.items():
        if k in ("TIF (ours)",):
            continue  # stale — computed on different plan
        safety_scores[k] = v
    # Override plan-dependent methods with recomputed scores
    safety_scores["TIF-safety (ours)"] = safety_tif_scores
    safety_scores["NearestNeighbor"] = nn_scores_safety
    safety_scores["TRAK (1-ckpt)"] = dtrak_scores_safety

    results = {"methods": {}}

    for method_name, scores in safety_scores.items():
        logger.info("Computing Safety AUC for method: %s", method_name)
        auc_result = safety_eval.compute_auc(scores, safety_labels)
        results["methods"][method_name] = auc_result

    # Summary table
    logger.info("\n--- Safety Attribution AUC ---")
    logger.info("%-25s  AUC(signed)  AUC(abs)", "Method")
    logger.info("-" * 50)
    for name, r in results["methods"].items():
        logger.info("%-25s  %.4f       %.4f", name, r["auc"], r["auc_abs"])

    return results


def run_curation_experiment(
    config: ExperimentConfig,
    diffusion: GaussianDiffusion,
    dataset: TrajectoryDataset,
    influence_scores: Dict[str, np.ndarray],
    ref_plan: Optional[np.ndarray] = None,
) -> Dict:
    """Run the data curation experiment.

    Args:
        ref_plan: (H, D) the SAME reference plan used to compute influence
                  scores in main(). Must be provided so that full_proxy is
                  evaluated on the exact plan that TIF attributed.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Data Curation")
    logger.info("=" * 60)

    proxy_fn = make_proxy_fn(config, dataset, diffusion_model=diffusion)

    if ref_plan is None:
        logger.warning(
            "ref_plan not provided to run_curation_experiment; generating a new plan. "
            "This may mismatch the plan used for influence scores!"
        )
        plans = plan(diffusion, dataset, config, reward_guidance=False)
        ref_plan = plans[0]
    full_proxy = proxy_fn(ref_plan)

    def proxy_fn_factory(model):
        return make_proxy_fn(config, dataset, diffusion_model=model)

    curation_eval = DataCurationEvaluator(
        train_fn=train,
        plan_fn=lambda m, d, c: plan(m, d, c, reward_guidance=False),
        proxy_fn=proxy_fn,
        dataset=dataset,
        config=config,
        eval_config=config.evaluation,
        proxy_fn_factory=proxy_fn_factory,
    )

    results = {"full_model_proxy": full_proxy, "methods": {}}

    for method_name, scores in influence_scores.items():
        logger.info("Running curation for method: %s", method_name)
        curation_result = curation_eval.evaluate_pruning(
            scores, full_model_proxy=full_proxy
        )
        results["methods"][method_name] = curation_result

    return results


def run_intervention_experiment(
    config: ExperimentConfig,
    diffusion: "GaussianDiffusion",
    dataset: "TrajectoryDataset",
    influence_scores: Dict[str, np.ndarray],
) -> Dict:
    """Run downstream intervention experiment.

    Removes top-attributed data, retrains, rolls out in actual Gym env,
    and measures return + constraint violations.
    """
    from evaluation import DownstreamInterventionEvaluator

    logger.info("=" * 60)
    logger.info("EXPERIMENT: Downstream Intervention")
    logger.info("=" * 60)

    intervention_eval = DownstreamInterventionEvaluator(
        train_fn=train,
        plan_fn=lambda m, d, c: plan(m, d, c, reward_guidance=False),
        dataset=dataset,
        config=config,
        model=diffusion,
    )

    # Baseline: full-data model rollout
    logger.info("Computing baseline (full-data model) rollout...")
    baseline_result = intervention_eval.rollout_in_env(
        diffusion, config.evaluation.n_rollout_episodes
    )
    logger.info(
        "Baseline: return=%.1f +/- %.1f, violation_rate=%.2f",
        baseline_result["mean_return"],
        baseline_result["std_return"],
        baseline_result["violation_rate"],
    )

    results: Dict = {"baseline": baseline_result, "methods": {}}

    # Run intervention for TIF and NN (the two most informative methods)
    methods_to_test = ["TIF", "NearestNeighbor"]
    for method_name in methods_to_test:
        if method_name not in influence_scores:
            continue
        logger.info("Running intervention for method: %s", method_name)
        intervention_result = intervention_eval.evaluate_intervention(
            influence_scores[method_name], method_name=method_name
        )
        results["methods"][method_name] = intervention_result

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = build_config(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(config.diffuser.device)
    logger.info("Config: env=%s, dataset=%s, experiment=%s", config.env_name, config.dataset, config.experiment)
    logger.info("Device: %s", device)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------
    logger.info("Loading dataset: %s", config.gym_id)
    max_traj = 50 if args.debug else None
    dataset = TrajectoryDataset(
        env_name=config.env_name,
        dataset_variant=config.dataset,
        horizon=config.diffuser.horizon,
        max_trajectories=max_traj,
    )
    logger.info(
        "Dataset: %d segments, transition_dim=%d",
        len(dataset),
        dataset.transition_dim,
    )

    # ---------------------------------------------------------------
    # 2. Train or load the Diffuser
    # ---------------------------------------------------------------
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info("Loading checkpoint: %s", args.checkpoint)
        unet = TemporalUNet(
            transition_dim=config.transition_dim,
            dim=config.diffuser.dim,
            dim_mults=config.diffuser.dim_mults,
            n_residual_blocks=config.diffuser.n_residual_blocks,
            dropout=config.diffuser.dropout,
        )
        diffusion = GaussianDiffusion(unet, config.diffuser).to(device)
        state_dict = torch.load(args.checkpoint, map_location=device)
        diffusion.load_state_dict(state_dict)
    else:
        logger.info("Training Diffuser from scratch...")
        t0 = time.time()
        diffusion, dataset = train(config, dataset=dataset, verbose=True)
        logger.info("Training took %.1f seconds.", time.time() - t0)

    diffusion.eval()

    # ---------------------------------------------------------------
    # 3. Generate a reference plan
    # ---------------------------------------------------------------
    logger.info("Generating reference plan...")
    plans = plan(diffusion, dataset, config, reward_guidance=False)
    ref_plan = plans[0]  # (H, D) — take the first sample
    ref_plan_tensor = torch.from_numpy(
        (ref_plan - dataset.data_mean) / dataset.data_std
    ).float()
    logger.info("Reference plan shape: %s", ref_plan.shape)

    # ---------------------------------------------------------------
    # LDS worker mode: only populate the subset cache (in reverse) and exit.
    # Influence/baselines are skipped — the main process computes those.
    # ---------------------------------------------------------------
    if args.lds_worker:
        from evaluation import TrajectoryLDS
        lds_cache = os.path.join(
            RESULTS_DIR, "lds_cache",
            f"{args.env}_{args.dataset}_seed{args.seed}",
        )
        logger.info("LDS WORKER mode — cache dir: %s", lds_cache)
        proxy_fn = make_proxy_fn(config, dataset, diffusion_model=diffusion)
        proxy_fn_factory = lambda m: make_proxy_fn(config, dataset, diffusion_model=m)
        full_proxy = proxy_fn(ref_plan)
        worker_evaluator = TrajectoryLDS(
            model_class=None,
            train_fn=train,
            plan_fn=lambda m, d, c: plan(m, d, c, reward_guidance=False),
            proxy_fn=proxy_fn,
            dataset=dataset,
            config=config,
            eval_config=config.evaluation,
            proxy_fn_factory=proxy_fn_factory,
            cache_dir=lds_cache,
        )
        # Dummy zero scores — worker only populates the proxy cache.
        dummy_scores = np.zeros(len(dataset), dtype=np.float32)
        worker_evaluator.compute_lds(
            influence_scores=dummy_scores,
            full_model_proxy=full_proxy,
            reverse=True,
        )
        logger.info("LDS WORKER finished populating cache. Exiting.")
        return

    # ---------------------------------------------------------------
    # 4. Compute influence scores (our method + baselines)
    # ---------------------------------------------------------------
    max_inf_samples = args.max_influence_samples
    if args.debug and max_inf_samples is None:
        max_inf_samples = min(100, len(dataset))

    influence_scores: Dict[str, np.ndarray] = {}

    # --- Our method: Trajectory Influence Functions ---
    logger.info("Computing Trajectory Influence Functions (TIF)...")
    t0 = time.time()
    tic = TrajectoryInfluenceComputer(
        model=diffusion,
        dataset=dataset,
        config=config.influence,
        diffusion_steps=config.diffuser.n_diffusion_steps,
    )
    tic.compute_hessian_approximation(
        n_samples=max_inf_samples,
    )
    tif_scores = tic.compute_all_influences_batched(
        plan_tau=ref_plan_tensor,
        proxy_type=config.influence.proxy_type,
        max_samples=max_inf_samples,
    )
    influence_scores["TIF (ours)"] = tif_scores
    logger.info("TIF computed in %.1f seconds.", time.time() - t0)

    # Determine effective number of samples for all methods
    n_effective = max_inf_samples if max_inf_samples else len(dataset)
    n_effective = min(n_effective, len(dataset))

    # --- Baseline: Random ---
    logger.info("Computing Random baseline...")
    random_baseline = RandomAttribution(dataset, seed=args.seed)
    rand_scores = random_baseline.compute_scores()[:n_effective]
    influence_scores["Random"] = rand_scores

    # --- Baseline: Reward Ranking ---
    logger.info("Computing Reward Ranking baseline...")
    reward_baseline = RewardRanking(dataset)
    rew_scores = reward_baseline.compute_scores()[:n_effective]
    influence_scores["RewardRanking"] = rew_scores

    # --- Baseline: Nearest Neighbor ---
    logger.info("Computing Nearest Neighbor baseline...")
    nn_baseline = NearestNeighborAttribution(dataset)
    nn_scores = nn_baseline.compute_scores(ref_plan_tensor)[:n_effective]
    influence_scores["NearestNeighbor"] = nn_scores

    # --- Baseline: D-TRAK ---
    logger.info("Computing D-TRAK baseline...")
    t0 = time.time()
    dtrak = TrajectoryDTRAK(
        model=diffusion,
        dataset=dataset,
        config=config.dtrak,
        diffusion_steps=config.diffuser.n_diffusion_steps,
    )
    dtrak.precompute_training_features(max_samples=n_effective)
    dtrak_scores = dtrak.compute_scores(ref_plan_tensor)[:n_effective]
    influence_scores["TRAK (1-ckpt)"] = dtrak_scores
    logger.info("TRAK (1-ckpt) computed in %.1f seconds.", time.time() - t0)
    # Free the 37GB GPU projection matrix before downstream stages
    del dtrak
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # 5. Run experiments
    # ---------------------------------------------------------------
    all_results: Dict[str, Dict] = {
        "config": {
            "env": config.env_name,
            "dataset": config.dataset,
            "hessian_approx": config.influence.hessian_approx,
            "proxy_type": config.influence.proxy_type,
            "n_train_segments": len(dataset),
            "n_influence_samples": max_inf_samples or len(dataset),
            "seed": args.seed,
            "debug": args.debug,
        },
    }

    experiments_to_run = (
        ["lds", "safety", "curation", "intervention"]
        if config.experiment == "all"
        else [config.experiment]
    )

    # If scores are truncated, create a matching subset dataset for evaluation
    from evaluation import _SubsetDataset
    eval_dataset = dataset
    if n_effective < len(dataset):
        eval_indices = np.arange(n_effective)
        eval_dataset = _SubsetDataset(dataset, eval_indices)
        logger.info(
            "Using subset of %d/%d samples for evaluation (matching score vectors).",
            n_effective, len(dataset),
        )

    for exp_name in experiments_to_run:
        logger.info("\n" + "=" * 60)
        logger.info("Running experiment: %s", exp_name)
        logger.info("=" * 60)

        if exp_name == "lds":
            lds_cache = os.path.join(
                RESULTS_DIR, "lds_cache",
                f"{args.env}_{args.dataset}_seed{args.seed}",
            )
            result = run_lds_experiment(
                config, diffusion, eval_dataset, influence_scores,
                ref_plan=ref_plan, cache_dir=lds_cache,
            )
        elif exp_name == "safety":
            result = run_safety_experiment(
                config, diffusion, eval_dataset, influence_scores,
            )
        elif exp_name == "curation":
            result = run_curation_experiment(
                config, diffusion, eval_dataset, influence_scores,
                ref_plan=ref_plan,
            )
        elif exp_name == "intervention":
            result = run_intervention_experiment(
                config, diffusion, eval_dataset, influence_scores,
            )
        else:
            logger.error("Unknown experiment: %s", exp_name)
            continue

        all_results[exp_name] = result

    # ---------------------------------------------------------------
    # 6. Save results
    # ---------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_name = (
        f"{config.env_name}_{config.dataset}_seed{args.seed}"
        f"_{config.experiment}_{config.influence.hessian_approx}_{timestamp}"
    )
    filepath = save_results(all_results, result_name, config.results_dir)
    logger.info("All results saved to: %s", filepath)

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info("Environment:  %s-%s", config.env_name, config.dataset)
    logger.info("Experiments:  %s", ", ".join(experiments_to_run))
    logger.info("Results file: %s", filepath)


if __name__ == "__main__":
    main()
