"""
Attribution baselines for comparison with Trajectory Influence Functions.

Implements four baseline methods:

1. RandomAttribution      -- random scores (negative control).
2. RewardRanking          -- rank by cumulative reward (task-aware heuristic).
3. NearestNeighborAttrib  -- L2 distance in state-action space.
4. TrajectoryDTRAK        -- D-TRAK (Park et al., 2023) adapted for
                             trajectory diffusion gradients.

All baselines expose the same interface:
    scores = baseline.compute_scores(plan_tau)
returning an (N_train,) array of attribution scores.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import DTRAKConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Random attribution (negative control)
# ---------------------------------------------------------------------------


class RandomAttribution:
    """Assign uniformly random influence scores.

    Serves as a sanity-check baseline: any meaningful attribution method
    should substantially outperform random on LDS and other metrics.
    """

    def __init__(self, dataset, seed: int = 0) -> None:
        self.n = len(dataset)
        self.rng = np.random.RandomState(seed)

    def compute_scores(
        self, plan_tau: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Return random scores in [-1, 1].

        Args:
            plan_tau: ignored (kept for interface compatibility).

        Returns:
            (N_train,) random attribution scores.
        """
        return self.rng.uniform(-1.0, 1.0, size=self.n).astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Reward ranking
# ---------------------------------------------------------------------------


class RewardRanking:
    """Rank training trajectories by cumulative reward.

    The intuition is that high-reward training data should have the most
    positive influence on plan quality.  This is a simple but surprisingly
    competitive baseline for reward-related proxies.
    """

    def __init__(self, dataset) -> None:
        """
        Args:
            dataset: TrajectoryDataset with .segment_rewards attribute.
        """
        self.segment_rewards = dataset.segment_rewards  # (N, H)

    def compute_scores(
        self, plan_tau: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Return cumulative reward per training segment as the score.

        Args:
            plan_tau: ignored.

        Returns:
            (N_train,) reward-based scores.
        """
        return self.segment_rewards.sum(axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Nearest-neighbor attribution
# ---------------------------------------------------------------------------


class NearestNeighborAttribution:
    """Attribute influence based on trajectory-space L2 distance.

    Scores = -||tau_train - tau_plan||_2  (negated so closer = higher score).

    This baseline captures the intuition that a diffusion model interpolates
    most strongly from nearby training examples.
    """

    def __init__(self, dataset) -> None:
        """
        Args:
            dataset: TrajectoryDataset with .segments attribute.
        """
        # (N, H, D) flattened to (N, H*D)
        self.train_flat = dataset.segments.reshape(len(dataset), -1)

    def compute_scores(self, plan_tau: torch.Tensor) -> np.ndarray:
        """Compute negative L2 distance from each training segment to the plan.

        Args:
            plan_tau: (horizon, transition_dim) or (1, horizon, transition_dim).

        Returns:
            (N_train,) negative-distance scores (higher = closer).
        """
        if isinstance(plan_tau, torch.Tensor):
            plan_np = plan_tau.detach().cpu().numpy()
        else:
            plan_np = plan_tau

        plan_flat = plan_np.reshape(1, -1)  # (1, H*D)
        # Efficient vectorised L2
        diff = self.train_flat - plan_flat  # (N, H*D)
        distances = np.linalg.norm(diff, axis=1)  # (N,)
        return -distances.astype(np.float64)


# ---------------------------------------------------------------------------
# 4. Trajectory D-TRAK
# ---------------------------------------------------------------------------


class TrajectoryDTRAK:
    """TRAK-style attribution (Park et al., 2023) adapted for trajectory
    diffusion models.

    This is a single-checkpoint approximation of TRAK/D-TRAK.  It uses random
    projections of per-sample gradients and solves a ridge regression in the
    projected space.  The full TRAK estimator would ensemble over multiple
    checkpoints, but for computational feasibility we use one checkpoint.

    Steps:
    1. For each training sample i, compute gradient g_i of the diffusion loss.
    2. Project: phi_i = P g_i  where P is a (projection_dim x param_dim)
       random Gaussian matrix.
    3. For the test plan, compute gradient g_test and project: phi_test = P g_test.
    4. Solve ridge regression: score = Φ (Φ^T Φ + λI)^{-1} φ_test.

    The random projection avoids materialising the full kernel while preserving
    inner-product structure (Johnson-Lindenstrauss).

    Note: n_checkpoints > 1 would require saving intermediate checkpoints during
    training and averaging the resulting score vectors.  Not yet implemented.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: DTRAKConfig,
        diffusion_steps: int = 200,
    ) -> None:
        """
        Args:
            model: trained GaussianDiffusion model.
            dataset: TrajectoryDataset.
            config: DTRAKConfig with projection parameters.
            diffusion_steps: number of diffusion steps T.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.diffusion_steps = diffusion_steps
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Count parameters
        self.param_dim = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            "D-TRAK: %d parameters, projecting to %d dims.",
            self.param_dim,
            config.projection_dim,
        )

        # Generate random projection matrix (lazy, stored as a seed for memory)
        self._proj_seed = config.proj_seed
        self._projection_dim = config.projection_dim

        # Pre-computed projected training gradients: (N, projection_dim)
        self._train_features: Optional[np.ndarray] = None

    def _get_projection_matrix(self) -> torch.Tensor:
        """Generate the random projection matrix P ~ N(0, 1/d).

        Shape: (projection_dim, param_dim). Cached on the instance and
        materialized in fp16 directly on the GPU so it fits for large
        param_dim (e.g. 1024 x 18M params ~= 37 GB).
        """
        if getattr(self, "_P_cache", None) is None:
            rng = torch.Generator(device=self.device).manual_seed(self._proj_seed)
            P = torch.randn(
                self._projection_dim,
                self.param_dim,
                generator=rng,
                device=self.device,
                dtype=torch.float16,
            )
            P /= np.sqrt(self._projection_dim)
            self._P_cache = P
        return self._P_cache

    def _compute_flat_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute flattened gradient of diffusion loss for a single sample.

        Args:
            x: (1, horizon, transition_dim).

        Returns:
            (param_dim,) gradient vector.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Average over a few timesteps
        loss = torch.tensor(0.0, device=self.device)
        n_t = 3
        for _ in range(n_t):
            t = torch.randint(0, self.diffusion_steps, (1,), device=self.device)
            noise = torch.randn_like(x)
            x_noisy = self.model.q_sample(x, t, noise)
            pred = self.model.model(x_noisy, t)
            loss = loss + F.mse_loss(pred, noise)
        loss = loss / n_t

        self.model.zero_grad()
        loss.backward()

        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.detach().flatten())
            elif p.requires_grad:
                grads.append(torch.zeros(p.numel(), device=self.device))
        return torch.cat(grads)

    def precompute_training_features(
        self, max_samples: Optional[int] = None
    ) -> None:
        """Project all training gradients and cache them.

        phi_i = P @ g_i   for each training sample i.
        """
        P = self._get_projection_matrix()
        n = max_samples or len(self.dataset)
        n = min(n, len(self.dataset))

        features = np.zeros((n, self._projection_dim), dtype=np.float32)

        for i in tqdm(range(n), desc="D-TRAK: projecting training gradients"):
            x = self.dataset[i]["trajectories"].unsqueeze(0).to(self.device)
            g = self._compute_flat_gradient(x)
            phi = (P @ g.to(P.dtype)).float().cpu().numpy()
            features[i] = phi

        self._train_features = features
        logger.info("D-TRAK: cached %d projected training features.", n)

    def compute_scores(self, plan_tau: torch.Tensor) -> np.ndarray:
        """Compute D-TRAK attribution scores with kernel solve.

        D-TRAK/TRAK (Park et al., 2023) scores are:
            score = Φ_train (Φ_train^T Φ_train + λI)^{-1} φ_test

        where Φ_train is (N, d), φ_test is (d,), and λ is a regularizer.
        This is equivalent to solving a ridge regression in the projected space.

        Args:
            plan_tau: (horizon, transition_dim) generated plan.

        Returns:
            (N_train,) attribution scores.
        """
        if self._train_features is None:
            self.precompute_training_features()

        P = self._get_projection_matrix()
        if plan_tau.dim() == 2:
            plan_tau = plan_tau.unsqueeze(0)
        plan_tau = plan_tau.to(self.device)

        g_test = self._compute_flat_gradient(plan_tau)
        phi_test = (P @ g_test.to(P.dtype)).float().cpu().numpy()  # (projection_dim,)

        # Kernel solve: (Φ^T Φ + λI)^{-1} φ_test
        # Φ is (N, d), so Φ^T Φ is (d, d)
        Phi = self._train_features  # (N, d)
        d = Phi.shape[1]
        lambda_reg = 1e-3  # regularization
        gram = Phi.T @ Phi + lambda_reg * np.eye(d, dtype=np.float32)  # (d, d)

        # Solve (Φ^T Φ + λI) w = φ_test for w, then scores = Φ w
        try:
            w = np.linalg.solve(gram, phi_test)  # (d,)
            scores = Phi @ w  # (N,)
        except np.linalg.LinAlgError:
            logger.warning("D-TRAK kernel solve failed; falling back to inner product.")
            scores = Phi @ phi_test

        return scores.astype(np.float64)
