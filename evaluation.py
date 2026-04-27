"""
Evaluation metrics for Trajectory Influence Functions.

Implements four evaluation protocols:

1. TrajectoryLDS                  -- Linear Datamodeling Score: Spearman
                                     correlation between predicted and actual
                                     influence via subset retraining.
2. SafetyAttributionAUC           -- Area under ROC curve measuring whether
                                     influence scores correctly identify unsafe
                                     training data.
3. DataCurationEvaluator          -- Evaluate attribution-guided data pruning
                                     via proxy delta measurement.
4. DownstreamInterventionEvaluator -- Validate influence scores via actual
                                     downstream impact: remove data, retrain,
                                     rollout in env, measure return and
                                     constraint violations.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from configs import EvaluationConfig, ExperimentConfig

logger = logging.getLogger(__name__)


def _try_acquire_lock(lock_path: str) -> bool:
    """Atomically create lock file. Returns True if we got it."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


# ---------------------------------------------------------------------------
# 1. Linear Datamodeling Score (LDS)
# ---------------------------------------------------------------------------


class TrajectoryLDS:
    """Trajectory-level Linear Datamodeling Score.

    The LDS (Park et al., 2023) measures how well influence scores predict
    the actual effect of removing training data.  Concretely:

    1. Sample K random subsets S_k of the training data (each keeping a
       fraction alpha of the data).
    2. For each S_k: retrain the diffusion model on S_k, generate a plan
       tau*_{S_k}, and evaluate the proxy f(tau*_{S_k}).
    3. Compute the actual change: delta_f_k = f(tau*_{S_k}) - f(tau*_full).
    4. Compute the predicted change from influence scores:
           delta_hat_k = sum_{i not in S_k} I(z_i, tau*)
    5. Report the Spearman rank correlation rho(delta_f, delta_hat) over
       the K subsets.

    A high LDS means the influence scores faithfully linearise the
    leave-subset-out effect.
    """

    def __init__(
        self,
        model_class,
        train_fn,
        plan_fn,
        proxy_fn,
        dataset,
        config: ExperimentConfig,
        eval_config: EvaluationConfig,
        proxy_fn_factory=None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_class: callable that returns a new (untrained) model.
            train_fn:    callable(config, dataset) -> trained model.
            plan_fn:     callable(model, dataset, config) -> plan_tau.
            proxy_fn:    callable(plan_tau) -> scalar proxy value.
                         Used for the full-model proxy evaluation.
            dataset:     full TrajectoryDataset.
            config:      ExperimentConfig (for model training).
            eval_config: EvaluationConfig (for LDS parameters).
            proxy_fn_factory: callable(model) -> proxy_fn. If provided,
                         creates a model-specific proxy evaluator for each
                         retrained model. This ensures the proxy is evaluated
                         with the retrained model, not the original.
        """
        self.model_class = model_class
        self.train_fn = train_fn
        self.plan_fn = plan_fn
        self.proxy_fn = proxy_fn
        self.dataset = dataset
        self.config = config
        self.eval_config = eval_config
        self.proxy_fn_factory = proxy_fn_factory
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_paths(self, k: int) -> Tuple[Optional[str], Optional[str]]:
        if self.cache_dir is None:
            return None, None
        steps = self.eval_config.retrain_steps
        seed = self.eval_config.base_seed
        base = os.path.join(self.cache_dir, f"subset_{k:04d}_seed{seed}_steps{steps}")
        return base + ".json", base + ".lock"

    def _load_cached_proxy(self, k: int) -> Optional[float]:
        cache_path, _ = self._cache_paths(k)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path) as f:
                return float(json.load(f)["proxy_val"])
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _claim_subset(self, k: int) -> bool:
        _, lock_path = self._cache_paths(k)
        if lock_path is None:
            return True
        return _try_acquire_lock(lock_path)

    def _release_lock(self, k: int) -> None:
        _, lock_path = self._cache_paths(k)
        if lock_path is None:
            return
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

    def _save_cached_proxy(self, k: int, proxy_val: float) -> None:
        cache_path, _ = self._cache_paths(k)
        if cache_path is None:
            return
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"proxy_val": proxy_val}, f)
        os.replace(tmp, cache_path)

    def _wait_for_cache(self, k: int, timeout: float = 7200.0) -> Optional[float]:
        cache_path, lock_path = self._cache_paths(k)
        if cache_path is None:
            return None
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(cache_path):
                return self._load_cached_proxy(k)
            if not os.path.exists(lock_path):
                # Worker died without writing; we should retry
                return None
            time.sleep(5.0)
        logger.warning("Timeout waiting for subset %d cache.", k)
        return None

    def compute_lds(
        self,
        influence_scores: np.ndarray,
        full_model_proxy: Optional[float] = None,
        n_subsets: Optional[int] = None,
        subset_fraction: Optional[float] = None,
        reverse: bool = False,
    ) -> Dict[str, float]:
        """Compute the Linear Datamodeling Score.

        Args:
            influence_scores: (N_train,) influence scores from the method
                              being evaluated.
            full_model_proxy: f(tau*_full).  If None, will be computed.
            n_subsets: override for eval_config.n_subsets.
            subset_fraction: override for eval_config.subset_fraction.

        Returns:
            Dict with keys:
                'lds_spearman': Spearman rank correlation.
                'lds_spearman_pvalue': p-value for the correlation.
                'lds_pearson': Pearson correlation (for reference).
                'actual_deltas': list of actual proxy changes.
                'predicted_deltas': list of predicted proxy changes.
        """
        n_sub = n_subsets or self.eval_config.n_subsets
        alpha = subset_fraction or self.eval_config.subset_fraction
        N = len(self.dataset)

        logger.info(
            "Computing LDS: %d subsets, alpha=%.2f, N=%d", n_sub, alpha, N
        )

        # Compute full-model proxy if not provided
        if full_model_proxy is None:
            logger.info("Computing full-model proxy value...")
            # We assume the caller has already trained and can provide
            # the proxy.  As a fallback, return a dummy.
            full_model_proxy = 0.0
            logger.warning(
                "full_model_proxy not provided; using 0.0. "
                "Pass the actual value for meaningful LDS."
            )

        # Pre-generate all subsets deterministically so worker/main agree
        # regardless of iteration direction.
        rng = np.random.RandomState(self.eval_config.base_seed)
        subset_size = int(alpha * N)
        all_subsets = [
            rng.choice(N, size=subset_size, replace=False) for _ in range(n_sub)
        ]

        # proxy_vals indexed by k; None = not yet computed/loaded
        proxy_vals: List[Optional[float]] = [None] * n_sub

        order = list(range(n_sub))
        if reverse:
            order = order[::-1]

        for k in tqdm(order, desc="LDS subsets"):
            subset_indices = all_subsets[k]

            # Cache hit? (other process may have computed it)
            cached = self._load_cached_proxy(k)
            if cached is not None:
                proxy_vals[k] = cached
                continue

            # Try to claim this subset via lock
            if not self._claim_subset(k):
                # Another process is computing; busy-wait for the cache file
                proxy_vals[k] = self._wait_for_cache(k)
                continue

            # We hold the lock — retrain
            subset_dataset = _SubsetDataset(self.dataset, subset_indices)
            retrain_config = _shallow_copy_config(self.config)
            retrain_config.diffuser.n_train_steps = self.eval_config.retrain_steps

            try:
                retrained_model, _ = self.train_fn(
                    retrain_config, dataset=subset_dataset, verbose=False
                )
                plan = self.plan_fn(retrained_model, self.dataset, retrain_config)
                if self.proxy_fn_factory is not None:
                    retrained_proxy_fn = self.proxy_fn_factory(retrained_model)
                    proxy_val = retrained_proxy_fn(plan)
                else:
                    proxy_val = self.proxy_fn(plan)
                self._save_cached_proxy(k, float(proxy_val))
                proxy_vals[k] = float(proxy_val)
            except Exception as e:
                logger.warning("Subset %d failed: %s. Skipping.", k, e)
            finally:
                self._release_lock(k)
                # Free GPU memory after each retrain iteration
                del retrained_model
                torch.cuda.empty_cache()
                gc.collect()

        actual_deltas = []
        predicted_deltas = []
        for k in range(n_sub):
            proxy_val = proxy_vals[k]
            if proxy_val is None:
                continue
            subset_mask = np.zeros(N, dtype=bool)
            subset_mask[all_subsets[k]] = True
            actual_deltas.append(proxy_val - full_model_proxy)
            removed_mask = ~subset_mask
            predicted_deltas.append(
                -influence_scores[removed_mask].sum() / len(influence_scores)
            )

        actual_deltas = np.array(actual_deltas)
        predicted_deltas = np.array(predicted_deltas)

        if len(actual_deltas) < 3:
            logger.warning("Too few successful subsets for correlation.")
            return {
                "lds_spearman": 0.0,
                "lds_spearman_pvalue": 1.0,
                "lds_pearson": 0.0,
                "actual_deltas": actual_deltas.tolist(),
                "predicted_deltas": predicted_deltas.tolist(),
            }

        sp_corr, sp_pval = stats.spearmanr(actual_deltas, predicted_deltas)
        pe_corr, _ = stats.pearsonr(actual_deltas, predicted_deltas)

        results = {
            "lds_spearman": float(sp_corr),
            "lds_spearman_pvalue": float(sp_pval),
            "lds_pearson": float(pe_corr),
            "actual_deltas": actual_deltas.tolist(),
            "predicted_deltas": predicted_deltas.tolist(),
        }

        logger.info(
            "LDS result: Spearman=%.4f (p=%.4f), Pearson=%.4f",
            sp_corr,
            sp_pval,
            pe_corr,
        )
        return results


# ---------------------------------------------------------------------------
# 2. Safety Attribution AUC
# ---------------------------------------------------------------------------


class SafetyAttributionAUC:
    """Measure whether influence scores correctly attribute constraint
    violations to unsafe training data.

    Setup:
    - Label each training trajectory as safe/unsafe based on whether it
      contains constraint violations (e.g., joint limits, torque limits).
    - Generate plans and check which ones violate constraints.
    - For constraint-violating plans, compute influence scores.
    - Compute AUC: do high-influence training samples coincide with
      unsafe training data?

    A high AUC means the attribution method successfully traces plan
    unsafety back to unsafe training trajectories.
    """

    def __init__(
        self,
        dataset,
        eval_config: EvaluationConfig,
    ) -> None:
        self.dataset = dataset
        self.eval_config = eval_config

    def label_training_safety(
        self,
        state_bounds: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Label each training segment as safe (0) or unsafe (1).

        A segment is unsafe if it contains constraint violations — specifically,
        if any state dimension exceeds safety bounds at any timestep.  In
        normalized space, we use the 95th percentile of per-dimension max
        absolute values as an adaptive threshold, unless one is provided.

        Args:
            state_bounds: (state_dim,) maximum absolute value per state dim.
            threshold: scalar threshold. If None, uses adaptive (95th pctile).

        Returns:
            (N_train,) binary labels.
        """
        segments = self.dataset.segments  # (N, H, D)
        state_dim = self.dataset.state_dim
        states = segments[:, :, :state_dim]  # (N, H, S)

        if threshold is not None:
            thresh = threshold
        elif state_bounds is not None:
            # Per-dimension thresholds
            max_per_dim = np.abs(states).max(axis=1)  # (N, S)
            labels = np.any(max_per_dim > state_bounds[None, :], axis=1).astype(np.float64)
            n_unsafe = labels.sum()
            logger.info(
                "Safety labels: %d unsafe / %d total (%.1f%%)",
                int(n_unsafe), len(labels), 100 * n_unsafe / len(labels),
            )
            return labels
        else:
            # Adaptive threshold: 95th percentile of max absolute state values
            max_abs_all = np.abs(states).max(axis=(1, 2))  # (N,)
            thresh = float(np.percentile(max_abs_all, 95))
            logger.info("Adaptive safety threshold: %.3f (95th pctile)", thresh)

        max_abs = np.abs(states).max(axis=(1, 2))  # (N,)
        labels = (max_abs > thresh).astype(np.float64)

        n_unsafe = labels.sum()
        logger.info(
            "Safety labels: %d unsafe / %d total (%.1f%%)",
            int(n_unsafe), len(labels), 100 * n_unsafe / len(labels),
        )
        return labels

    def compute_plan_safety_score(
        self,
        plan_tau: np.ndarray,
        state_dim: int,
    ) -> float:
        """Compute a scalar safety score for a generated plan.

        Higher = more unsafe (more constraint violations).

        Args:
            plan_tau: (horizon, transition_dim) plan.
            state_dim: number of state dimensions.

        Returns:
            Scalar safety violation score.
        """
        states = plan_tau[:, :state_dim]
        # Sum of squared violations beyond threshold
        threshold = 3.0
        violations = np.maximum(np.abs(states) - threshold, 0.0)
        return float(violations.sum())

    def compute_auc(
        self,
        influence_scores: np.ndarray,
        safety_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute AUC for safety attribution.

        For constraint-violating plans, do training trajectories with high
        influence scores correspond to unsafe training data?

        Args:
            influence_scores: (N_train,) influence scores (higher = more
                              influential on the constraint-violating plan).
            safety_labels: (N_train,) binary labels (1 = unsafe).
                          If None, will be computed automatically.

        Returns:
            Dict with 'auc', 'n_unsafe', 'n_safe'.
        """
        if safety_labels is None:
            safety_labels = self.label_training_safety()

        n_unsafe = int(safety_labels.sum())
        n_safe = len(safety_labels) - n_unsafe

        if n_unsafe == 0 or n_safe == 0:
            logger.warning(
                "Cannot compute AUC: %d unsafe, %d safe. "
                "Need at least 1 of each.",
                n_unsafe,
                n_safe,
            )
            return {"auc": 0.5, "n_unsafe": n_unsafe, "n_safe": n_safe}

        # AUC: does higher influence score predict unsafe label?
        # Under the constraint proxy, positive I(z_i) means z_i pushes
        # the plan toward constraint violation.  Use signed scores so that
        # the direction of influence is tested, not just magnitude.
        # Also compute AUC with absolute scores for comparison.
        auc_signed = roc_auc_score(safety_labels, influence_scores)
        auc_abs = roc_auc_score(safety_labels, np.abs(influence_scores))

        results = {
            "auc": float(auc_signed),
            "auc_abs": float(auc_abs),
            "n_unsafe": n_unsafe,
            "n_safe": n_safe,
        }
        logger.info(
            "Safety AUC: signed=%.4f, abs=%.4f (%d unsafe, %d safe)",
            auc_signed, auc_abs, n_unsafe, n_safe,
        )
        return results


# ---------------------------------------------------------------------------
# 3. Data Curation Evaluator
# ---------------------------------------------------------------------------


class DataCurationEvaluator:
    """Evaluate attribution-guided data pruning.

    Idea: if influence scores are meaningful, then:
    - Removing the most *negatively* influential data (data that hurts plan
      quality) should *improve* plan quality.
    - Removing the most *positively* influential data (helpful data) should
      *degrade* plan quality.

    We test this by:
    1. Rank training data by influence score.
    2. For several prune fractions p in {10%, 20%, 30%, 50%}:
       a. Remove the bottom-p% (most negative influence) -> retrain -> plan.
       b. Remove the top-p% (most positive influence) -> retrain -> plan.
       c. Remove a random p% -> retrain -> plan.
    3. Compare proxy values.
    """

    def __init__(
        self,
        train_fn,
        plan_fn,
        proxy_fn,
        dataset,
        config: ExperimentConfig,
        eval_config: EvaluationConfig,
        proxy_fn_factory=None,
    ) -> None:
        self.train_fn = train_fn
        self.plan_fn = plan_fn
        self.proxy_fn = proxy_fn
        self.dataset = dataset
        self.config = config
        self.eval_config = eval_config
        self.proxy_fn_factory = proxy_fn_factory

    def evaluate_pruning(
        self,
        influence_scores: np.ndarray,
        prune_fractions: Optional[List[float]] = None,
        full_model_proxy: Optional[float] = None,
    ) -> Dict[str, object]:
        """Run the data curation evaluation.

        Args:
            influence_scores: (N_train,) influence scores.
            prune_fractions: list of fractions to prune.
            full_model_proxy: proxy value from full-data model.

        Returns:
            Dict with per-fraction results.
        """
        fractions = prune_fractions or self.eval_config.prune_fractions
        N = len(self.dataset)

        if full_model_proxy is None:
            full_model_proxy = 0.0
            logger.warning("full_model_proxy not provided; using 0.0.")

        sorted_indices = np.argsort(influence_scores)  # ascending
        rng = np.random.RandomState(self.eval_config.base_seed)

        results: Dict[str, object] = {
            "full_model_proxy": full_model_proxy,
            "fractions": {},
        }

        for frac in fractions:
            n_remove = int(frac * N)
            logger.info("Pruning %.0f%% (%d samples)...", frac * 100, n_remove)

            frac_results = {}

            # Sign convention:
            # I(z_i) = -g_test^T H^{-1} g_train estimates f(θ*) - f(θ_{-i}*)
            # Positive I → z_i INCREASES the proxy value (harmful if proxy=loss)
            # Negative I → z_i DECREASES the proxy value (helpful if proxy=loss)
            #
            # For quality improvement: remove most POSITIVE (harmful) data
            # For quality degradation: remove most NEGATIVE (helpful) data

            # --- Remove most harmful (highest influence = most positive) ---
            # sorted_indices is ascending, so highest are at the end
            keep_indices = sorted_indices[:-n_remove] if n_remove > 0 else sorted_indices
            proxy_val = self._retrain_and_evaluate(keep_indices)
            frac_results["remove_harmful"] = {
                "proxy": proxy_val,
                "delta": proxy_val - full_model_proxy,
            }

            # --- Remove most helpful (lowest influence = most negative) ---
            keep_indices = sorted_indices[n_remove:]
            proxy_val = self._retrain_and_evaluate(keep_indices)
            frac_results["remove_helpful"] = {
                "proxy": proxy_val,
                "delta": proxy_val - full_model_proxy,
            }

            # --- Remove random ---
            random_remove = rng.choice(N, size=n_remove, replace=False)
            random_keep = np.setdiff1d(np.arange(N), random_remove)
            proxy_val = self._retrain_and_evaluate(random_keep)
            frac_results["remove_random"] = {
                "proxy": proxy_val,
                "delta": proxy_val - full_model_proxy,
            }

            results["fractions"][str(frac)] = frac_results
            logger.info(
                "  frac=%.0f%%: remove_harmful=%.4f, remove_helpful=%.4f, random=%.4f",
                frac * 100,
                frac_results["remove_harmful"]["delta"],
                frac_results["remove_helpful"]["delta"],
                frac_results["remove_random"]["delta"],
            )

        return results

    def _retrain_and_evaluate(self, keep_indices: np.ndarray) -> float:
        """Retrain on a subset and evaluate the plan proxy with the retrained model."""
        subset_dataset = _SubsetDataset(self.dataset, keep_indices)
        retrain_config = _shallow_copy_config(self.config)
        retrain_config.diffuser.n_train_steps = self.eval_config.retrain_steps

        model = None
        try:
            model, _ = self.train_fn(
                retrain_config, dataset=subset_dataset, verbose=False
            )
            plan = self.plan_fn(model, self.dataset, retrain_config)
            # Evaluate with retrained model
            if self.proxy_fn_factory is not None:
                retrained_proxy_fn = self.proxy_fn_factory(model)
                return retrained_proxy_fn(plan)
            else:
                return self.proxy_fn(plan)
        except Exception as e:
            logger.warning("Retrain failed: %s", e)
            return 0.0
        finally:
            # Free GPU memory after each retrain iteration
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SubsetDataset(torch.utils.data.Dataset):
    """Wraps a TrajectoryDataset to expose only a subset of indices."""

    def __init__(self, dataset, indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = indices
        # Expose attributes needed by TrajectoryDataset consumers
        self.segments = dataset.segments[indices]
        self.segment_rewards = dataset.segment_rewards[indices]
        self.segment_episode_idx = dataset.segment_episode_idx[indices]
        self.state_dim = dataset.state_dim
        self.action_dim = dataset.action_dim
        self.transition_dim = dataset.transition_dim
        self.data_mean = dataset.data_mean
        self.data_std = dataset.data_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[int(self.indices[idx])]

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return self.dataset.unnormalize(x)


def _shallow_copy_config(config: ExperimentConfig) -> ExperimentConfig:
    """Create a shallow copy of ExperimentConfig suitable for retraining
    with modified hyperparameters."""
    import copy
    return copy.deepcopy(config)


# ---------------------------------------------------------------------------
# Downstream Intervention Evaluator
# ---------------------------------------------------------------------------


class DownstreamInterventionEvaluator:
    """Validate influence scores via actual downstream impact.

    Protocol for each prune fraction K%:
      1. Remove top-K% most influential (harmful) training data.
      2. Retrain the diffusion model on the remaining data.
      3. Rollout the retrained model in the actual Gym environment.
      4. Measure: mean episodic return + constraint violation rate.
      5. Repeat for bottom-K% (helpful) and random-K% removals.

    This closes the gap between proxy-based evaluation and real-world
    downstream impact, as requested by reviewers.
    """

    def __init__(
        self,
        train_fn,
        plan_fn,
        dataset,
        config: ExperimentConfig,
        model,
    ) -> None:
        """
        Args:
            train_fn: callable(config, dataset, verbose) -> (model, dataset).
            plan_fn: callable(model, dataset, config) -> plans (N, H, D) np.
            dataset: full TrajectoryDataset.
            config: ExperimentConfig.
            model: trained diffusion model (for baseline rollout).
        """
        self.train_fn = train_fn
        self.plan_fn = plan_fn
        self.dataset = dataset
        self.config = config
        self.model = model
        self.eval_config = config.evaluation

    def rollout_in_env(
        self,
        model,
        n_episodes: int,
    ) -> Dict:
        """Execute MPC-style rollouts in the actual Gym environment.

        Replans every ``horizon`` steps using the diffusion model.

        Returns:
            Dict with mean_return, std_return, violation_rate,
            mean_violations_per_episode, all_returns.
        """
        import gym
        try:
            import d4rl  # noqa: F401  — registers D4RL envs
        except ImportError:
            pass

        from configs import get_gym_id

        gym_id = get_gym_id(self.config.env_name, self.config.dataset)
        env = gym.make(gym_id)
        state_dim = self.config.state_dim
        action_dim = self.config.action_dim
        horizon = self.config.diffuser.horizon
        bound = self.eval_config.constraint_bound
        max_steps = self.eval_config.max_episode_steps

        returns: List[float] = []
        violation_counts: List[int] = []

        for ep in range(n_episodes):
            obs = env.reset()
            episode_return = 0.0
            violations = 0
            step = 0

            while step < max_steps:
                # Generate a batch of plans and pick the best one
                plans = self.plan_fn(model, self.dataset, self.config)
                best_plan = plans[0]  # (H, D) unnormalized

                for h in range(min(horizon, max_steps - step)):
                    action = best_plan[h, state_dim:state_dim + action_dim]
                    action = np.clip(
                        action, env.action_space.low, env.action_space.high
                    )
                    obs, reward, done, info = env.step(action)
                    episode_return += reward
                    if np.any(np.abs(obs) > bound):
                        violations += 1
                    step += 1
                    if done:
                        break
                if done:
                    break

            returns.append(episode_return)
            violation_counts.append(violations)

        env.close()

        returns_arr = np.array(returns)
        violations_arr = np.array(violation_counts)

        return {
            "mean_return": float(returns_arr.mean()),
            "std_return": float(returns_arr.std()),
            "violation_rate": float((violations_arr > 0).mean()),
            "mean_violations_per_episode": float(violations_arr.mean()),
            "all_returns": returns_arr.tolist(),
        }

    def evaluate_intervention(
        self,
        influence_scores: np.ndarray,
        method_name: str = "TIF",
    ) -> Dict:
        """Run the full intervention evaluation for one attribution method.

        Args:
            influence_scores: (N_train,) influence scores.
            method_name: label for logging.

        Returns:
            Dict with per-fraction results for each removal strategy.
        """
        fractions = self.eval_config.intervention_prune_fractions
        N = len(self.dataset)
        n_episodes = self.eval_config.n_rollout_episodes
        sorted_indices = np.argsort(influence_scores)  # ascending
        rng = np.random.RandomState(self.eval_config.base_seed + 999)

        results: Dict = {"fractions": {}}

        for frac in fractions:
            n_remove = int(frac * N)
            if n_remove == 0:
                continue
            logger.info(
                "Intervention [%s]: removing %d/%d (%.0f%%)",
                method_name, n_remove, N, frac * 100,
            )
            frac_results: Dict = {}

            strategies = [
                ("remove_most_harmful", sorted_indices[:-n_remove]),
                ("remove_most_helpful", sorted_indices[n_remove:]),
                (
                    "remove_random",
                    np.setdiff1d(
                        np.arange(N),
                        rng.choice(N, n_remove, replace=False),
                    ),
                ),
            ]

            for removal_type, keep_indices in strategies:
                logger.info("  Strategy: %s", removal_type)
                subset_dataset = _SubsetDataset(self.dataset, keep_indices)
                retrain_config = _shallow_copy_config(self.config)
                retrain_config.diffuser.n_train_steps = (
                    self.eval_config.retrain_steps
                )

                try:
                    retrained_model, _ = self.train_fn(
                        retrain_config, dataset=subset_dataset, verbose=False
                    )
                    rollout_result = self.rollout_in_env(
                        retrained_model, n_episodes
                    )
                    frac_results[removal_type] = rollout_result
                    logger.info(
                        "    return=%.1f +/- %.1f, violation_rate=%.2f",
                        rollout_result["mean_return"],
                        rollout_result["std_return"],
                        rollout_result["violation_rate"],
                    )
                    del retrained_model
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(
                        "    Intervention %s failed: %s", removal_type, e
                    )
                    frac_results[removal_type] = {"error": str(e)}

            results["fractions"][str(frac)] = frac_results

        return results


def save_results(
    results: Dict,
    experiment_name: str,
    results_dir: str,
) -> str:
    """Save experiment results to JSON.

    Args:
        results: dict of results.
        experiment_name: descriptive name for the file.
        results_dir: directory to save to.

    Returns:
        Path to the saved file.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{experiment_name}.json")

    # Convert numpy types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            converted = _convert(o)
            if converted is not o:
                return converted
            return super().default(o)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info("Results saved to %s", filepath)
    return filepath
