"""
Trajectory Influence Functions for Diffusion-Based Control Planners.

Core contribution: adapt influence functions from the supervised-learning
setting to trajectory diffusion planners, handling the unique structure of
temporal U-Nets and diffusion training objectives.

The classical influence function (Koh & Liang, ICML 2017) estimates the
effect of removing a training point z_i on the model parameters:

    I(z_i, z_test) = - nabla_theta L(z_test; theta^*)^T
                       H_{theta^*}^{-1}
                       nabla_theta L(z_i; theta^*)

where H is the Hessian of the empirical risk.

For diffusion models the training loss per sample is:

    L(tau; theta) = E_{t, eps} || eps - eps_theta(tau^t, t) ||^2

We approximate H^{-1} via EK-FAC (George et al., 2018), which decomposes
the Fisher/GGN into Kronecker products per layer and then eigendecomposes
for cheap inverse-vector products.

This module provides:
  - TrajectoryInfluenceComputer: main class that computes per-training-
    trajectory influence scores for a given plan.
  - _EKFACState: internal bookkeeping for accumulated Kronecker factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import ExperimentConfig, InfluenceConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal: per-layer EK-FAC state
# ---------------------------------------------------------------------------


@dataclass
class _LayerKroneckerFactors:
    """Accumulated Kronecker factors for one Conv1d or Linear layer."""

    # A = E[a a^T]  — input activation covariance  (d_in x d_in)
    A: Optional[torch.Tensor] = None
    # G = E[g g^T]  — output gradient covariance   (d_out x d_out)
    G: Optional[torch.Tensor] = None
    # Eigendecomposition of A
    Q_A: Optional[torch.Tensor] = None
    Lambda_A: Optional[torch.Tensor] = None
    # Eigendecomposition of G
    Q_G: Optional[torch.Tensor] = None
    Lambda_G: Optional[torch.Tensor] = None
    n_samples: int = 0


@dataclass
class _EKFACState:
    """Collection of per-layer Kronecker factors for the whole model."""

    layers: Dict[str, _LayerKroneckerFactors] = field(default_factory=dict)
    damping: float = 1e-4


# ---------------------------------------------------------------------------
# Hook helpers to capture activations and output gradients
# ---------------------------------------------------------------------------


def _register_hooks(
    model: nn.Module,
    target_types: Tuple = (nn.Linear, nn.Conv1d, nn.ConvTranspose1d),
) -> Tuple[Dict[str, List], Dict[str, List], List]:
    """Register forward and backward hooks on target layers.

    Returns:
        (activations_dict, gradients_dict, hook_handles)
    """
    activations: Dict[str, List[torch.Tensor]] = {}
    gradients: Dict[str, List[torch.Tensor]] = {}
    handles: List = []

    for name, module in model.named_modules():
        if isinstance(module, target_types):
            activations[name] = []
            gradients[name] = []

            def _fwd_hook(mod, inp, out, n=name):
                # inp[0]: input activation
                activations[n].append(inp[0].detach())

            def _bwd_hook(mod, grad_in, grad_out, n=name):
                # grad_out[0]: gradient w.r.t. output
                gradients[n].append(grad_out[0].detach())

            handles.append(module.register_forward_hook(_fwd_hook))
            handles.append(module.register_full_backward_hook(_bwd_hook))

    return activations, gradients, handles


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TrajectoryInfluenceComputer:
    """Compute influence of training trajectories on generated plans.

    Usage:
        tic = TrajectoryInfluenceComputer(diffusion_model, dataset, config)
        tic.compute_hessian_approximation()
        scores = tic.compute_all_influences(plan_tau, proxy_type='likelihood')
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: InfluenceConfig,
        diffusion_steps: int = 200,
    ) -> None:
        """
        Args:
            model: trained GaussianDiffusion module.
            dataset: TrajectoryDataset instance.
            config: InfluenceConfig hyperparameters.
            diffusion_steps: number of diffusion timesteps T.
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.diffusion_steps = diffusion_steps
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.ekfac_state: Optional[_EKFACState] = None

        # Identify target layers
        self._target_layers: Dict[str, nn.Module] = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                self._target_layers[name] = mod

    # ------------------------------------------------------------------
    # 1. Hessian approximation (EK-FAC / K-FAC / diagonal)
    # ------------------------------------------------------------------

    def compute_hessian_approximation(
        self, n_samples: Optional[int] = None
    ) -> None:
        """Accumulate and eigendecompose Kronecker factors.

        For each target layer l with weight W_l, the Fisher/GGN block is
        approximated as:

            F_l  approx  A_l  otimes  G_l

        where
            A_l = E[a_{l-1} a_{l-1}^T]   (input activation covariance)
            G_l = E[g_l g_l^T]            (pre-activation gradient covariance)

        EK-FAC further eigendecomposes A_l and G_l for efficient inversion:
            A_l = Q_A Lambda_A Q_A^T
            G_l = Q_G Lambda_G Q_G^T

        Then (A_l otimes G_l)^{-1} vec(v) can be computed in O(d_in^2 + d_out^2)
        instead of O((d_in * d_out)^2).

        Layer handling:
        - Linear layers: standard KFAC (exact up to Kronecker assumption)
        - Conv1d (kernel_size > 1): uses im2col (F.unfold) with correct
          stride/dilation, yielding A of shape (C_in*K, C_in*K)
        - ConvTranspose1d: uses a spatial-average approximation where
          G = input covariance (C_in, C_in) and A = output-grad covariance
          (C_out, C_out), applied independently per kernel position.
          This is an approximation that drops cross-kernel correlations
          but remains correct in expectation for stationary activations.

        Args:
            n_samples: number of data points to use.  None -> full dataset.
        """
        if self.config.hessian_approx == "diagonal":
            self._compute_diagonal_hessian(n_samples)
            return

        logger.info("Computing %s Hessian approximation...", self.config.hessian_approx)

        self.model.eval()
        state = _EKFACState(damping=self.config.damping)

        # Initialise per-layer accumulators
        for name in self._target_layers:
            state.layers[name] = _LayerKroneckerFactors()

        # Register hooks
        acts_dict, grads_dict, handles = _register_hooks(self.model)

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.gradient_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        total = n_samples or len(self.dataset)
        seen = 0

        for batch in tqdm(loader, desc="Accumulating Kronecker factors"):
            if seen >= total:
                break
            x = batch["trajectories"].to(self.device)
            bs = x.shape[0]

            # Sample random diffusion timestep
            t = torch.randint(0, self.diffusion_steps, (bs,), device=self.device)
            noise = torch.randn_like(x)

            # Forward + backward to get activations and gradients
            x_noisy = self.model.q_sample(x, t, noise)
            pred = self.model.model(x_noisy, t)
            loss = F.mse_loss(pred, noise, reduction="sum") / bs

            self.model.zero_grad()
            loss.backward()

            # Accumulate factors
            for name, layer_factors in state.layers.items():
                if name not in acts_dict or len(acts_dict[name]) == 0:
                    continue
                a = acts_dict[name][-1]  # (batch, ...)
                g = grads_dict[name][-1]  # (batch, ...)

                # Flatten spatial dims for Conv1d/ConvTranspose1d with proper unfolding
                if a.dim() == 3:
                    B, C_a, L_a = a.shape
                    B2, C_g, L_g = g.shape
                    layer_mod = self._target_layers.get(name)
                    if layer_mod is not None and isinstance(layer_mod, nn.ConvTranspose1d):
                        # ConvTranspose1d: weight (C_in, C_out, K)
                        # The transpose conv maps input (B, C_in, L_in) to output (B, C_out, L_out)
                        # For KFAC:
                        #   G = input covariance: (B*L_in, C_in) -> (C_in, C_in)
                        #   A = output-grad "patches": need to gather output grad
                        #       patches that each input position contributes to
                        # For simplicity with strided transposed convs, use the
                        # spatial-average approximation (valid when L is large):
                        g_final = a.permute(0, 2, 1).reshape(-1, C_a)  # "G" side: (B*L_in, C_in)
                        # For A: the transposed conv's backward w.r.t. weight is
                        # equivalent to a regular conv of the output grad on the input.
                        # Use spatial averaging: A ≈ E[g_out g_out^T] per spatial position
                        # Reshape to (B*L_g, C_g) and tile kernel_size times
                        kernel_size = layer_mod.kernel_size[0] if isinstance(layer_mod.kernel_size, tuple) else layer_mod.kernel_size
                        g_spatial = g.permute(0, 2, 1).reshape(-1, C_g)  # (B*L_out, C_out)
                        # Approximate: A ≈ (g_spatial^T g_spatial / N) ⊗ I_K
                        # This is the spatial-averaged KFAC for transposed convs
                        # We store A as (C_g, C_g) and handle the kernel in IHVP
                        a_final = g_spatial
                        a = a_final
                        g = g_final
                    elif layer_mod is not None and hasattr(layer_mod, 'kernel_size'):
                        kernel_size = layer_mod.kernel_size[0] if isinstance(layer_mod.kernel_size, tuple) else layer_mod.kernel_size
                        padding = layer_mod.padding[0] if isinstance(layer_mod.padding, tuple) else layer_mod.padding
                        stride = layer_mod.stride[0] if isinstance(layer_mod.stride, tuple) else layer_mod.stride
                        dilation = layer_mod.dilation[0] if isinstance(layer_mod.dilation, tuple) else layer_mod.dilation
                        # Conv1d: weight (C_out, C_in, K)
                        # A from input unfolded: (B, C_in, L) -> (B*L_out, C_in*K)
                        # Must pass stride and dilation to match actual conv output length
                        a_unfolded = F.unfold(
                            a.unsqueeze(2),  # (B, C_in, 1, L)
                            kernel_size=(1, kernel_size),
                            padding=(0, padding),
                            stride=(1, stride),
                            dilation=(1, dilation),
                        ).reshape(B, C_a * kernel_size, -1)  # (B, C_in*K, L_out)
                        L_out = a_unfolded.shape[2]
                        a = a_unfolded.permute(0, 2, 1).reshape(-1, C_a * kernel_size)
                        # G from output gradient: use only the first L_out positions
                        # (output grad should already have L_out = L_g)
                        g = g.permute(0, 2, 1).reshape(-1, C_g)
                    else:
                        # Fallback: treat as pointwise (kernel=1)
                        a = a.permute(0, 2, 1).reshape(-1, C_a)
                        g = g.permute(0, 2, 1).reshape(-1, C_g)
                elif a.dim() == 2:
                    pass  # Linear: already (B, d_in)
                else:
                    continue

                # Outer products
                A_batch = (a.T @ a) / a.shape[0]
                G_batch = (g.T @ g) / g.shape[0]

                if layer_factors.A is None:
                    layer_factors.A = A_batch
                    layer_factors.G = G_batch
                else:
                    layer_factors.A += A_batch
                    layer_factors.G += G_batch
                layer_factors.n_samples += 1

            # Clear hook buffers
            for v in acts_dict.values():
                v.clear()
            for v in grads_dict.values():
                v.clear()

            seen += bs

        # Remove hooks
        for h in handles:
            h.remove()

        # Average and eigendecompose
        for name, lf in state.layers.items():
            if lf.A is None or lf.n_samples == 0:
                continue
            lf.A /= lf.n_samples
            lf.G /= lf.n_samples

            # Add damping to diagonals
            lf.A += self.config.damping * torch.eye(
                lf.A.shape[0], device=lf.A.device
            )
            lf.G += self.config.damping * torch.eye(
                lf.G.shape[0], device=lf.G.device
            )

            # Eigendecomposition
            lf.Lambda_A, lf.Q_A = torch.linalg.eigh(lf.A)
            lf.Lambda_G, lf.Q_G = torch.linalg.eigh(lf.G)

            # Optionally truncate to top-k eigenvectors
            k = self.config.n_eigenvectors
            if k > 0:
                lf.Lambda_A = lf.Lambda_A[-k:]
                lf.Q_A = lf.Q_A[:, -k:]
                lf.Lambda_G = lf.Lambda_G[-k:]
                lf.Q_G = lf.Q_G[:, -k:]

        self.ekfac_state = state
        logger.info("Hessian approximation complete (%d layers).", len(state.layers))

    def _compute_diagonal_hessian(self, n_samples: Optional[int] = None) -> None:
        """Diagonal Fisher approximation: F_ii = E[g_i^2]."""
        logger.info("Computing diagonal Hessian approximation...")
        self.model.eval()
        diag_acc: Dict[str, torch.Tensor] = {}

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.gradient_batch_size,
            shuffle=True,
            num_workers=0,
        )
        total = n_samples or len(self.dataset)
        seen = 0

        for batch in tqdm(loader, desc="Diagonal Hessian"):
            if seen >= total:
                break
            x = batch["trajectories"].to(self.device)
            bs = x.shape[0]
            t = torch.randint(0, self.diffusion_steps, (bs,), device=self.device)
            noise = torch.randn_like(x)
            x_noisy = self.model.q_sample(x, t, noise)
            pred = self.model.model(x_noisy, t)
            loss = F.mse_loss(pred, noise, reduction="sum") / bs
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                g2 = param.grad.detach() ** 2
                if name not in diag_acc:
                    diag_acc[name] = g2
                else:
                    diag_acc[name] += g2
            seen += bs

        n_batches = max(1, seen // self.config.gradient_batch_size)
        for name in diag_acc:
            diag_acc[name] /= n_batches
            diag_acc[name] += self.config.damping

        self._diagonal_hessian = diag_acc
        logger.info("Diagonal Hessian complete.")

    # ------------------------------------------------------------------
    # 2. Gradient computations
    # ------------------------------------------------------------------

    def compute_proxy_gradient(
        self, plan_tau: torch.Tensor, proxy_type: str = "likelihood"
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of a proxy measurement w.r.t. model parameters.

        The proxy defines *what property* of the generated plan we want to
        attribute.  Different proxies answer different questions:

        - 'likelihood': Which training data most influenced this specific
          plan?  Uses L_diff(tau*; theta).
        - 'reward_conditioned': Which data drives high-reward planning?
          Uses the diffusion loss under high-reward conditioning.
        - 'constraint_satisfaction': Which data causes constraint violations?
          Sums constraint values along the planned trajectory.
        - 'return': Which data causes high/low return plans?
          Uses discounted cumulative reward.

        Args:
            plan_tau: (horizon, transition_dim) or (1, horizon, transition_dim)
                      the generated plan to attribute.
            proxy_type: one of the proxy types above.

        Returns:
            Dict mapping parameter name -> gradient tensor.
        """
        if plan_tau.dim() == 2:
            plan_tau = plan_tau.unsqueeze(0)
        plan_tau = plan_tau.to(self.device)

        self.model.eval()
        # Ensure parameters require grad
        for p in self.model.parameters():
            p.requires_grad_(True)

        if proxy_type == "likelihood":
            # f_lik: negative diffusion training loss on the plan
            # f_lik(tau*; theta) = -E_{t,eps} || eps - eps_theta(tau*_t, t) ||^2
            # Negative so that higher = model fits the plan better
            loss = torch.tensor(0.0, device=self.device)
            n_t_samples = 10
            for _ in range(n_t_samples):
                t = torch.randint(
                    0, self.diffusion_steps, (plan_tau.shape[0],), device=self.device
                )
                noise = torch.randn_like(plan_tau)
                x_noisy = self.model.q_sample(plan_tau, t, noise)
                pred = self.model.model(x_noisy, t)
                loss = loss + F.mse_loss(pred, noise)
            loss = loss / n_t_samples
            # Use positive loss here; sign handled in influence formula

        elif proxy_type == "reward_conditioned":
            # f_rew: diffusion loss with low-noise timesteps (near-clean)
            # Approximates conditioning on high-reward by evaluating model's
            # denoising quality at low noise levels where the plan structure
            # (and thus its reward properties) is most preserved
            loss = torch.tensor(0.0, device=self.device)
            n_t_samples = 10
            for _ in range(n_t_samples):
                # Use only the first 20% of timesteps (low noise, plan structure preserved)
                t = torch.randint(
                    0,
                    max(1, self.diffusion_steps // 5),
                    (plan_tau.shape[0],),
                    device=self.device,
                )
                noise = torch.randn_like(plan_tau)
                x_noisy = self.model.q_sample(plan_tau, t, noise)
                pred = self.model.model(x_noisy, t)
                loss = loss + F.mse_loss(pred, noise)
            loss = loss / n_t_samples

        elif proxy_type == "constraint_satisfaction":
            # f_con: constraint-weighted denoising loss
            # f_con(tau*; theta) = -sum_h w_h * E_{t,eps}[||eps_h - eps_theta_h||^2]
            # where w_h = exp(-lambda * sum_j c_j(s_h*))
            # c_j(s) = max(|s_j| - bound, 0) for state bounds
            # This IS theta-differentiable through eps_theta
            state_dim = self.dataset.state_dim
            states = plan_tau[:, :, :state_dim]  # (1, H, S)
            # Compute constraint weights per timestep
            bound = 3.0  # normalised space bound
            constraint_violation = torch.clamp(states.abs() - bound, min=0.0)
            # w_h = exp(-lambda * sum_j c_j(s_h))
            lambda_con = 1.0
            w_h = torch.exp(-lambda_con * constraint_violation.sum(dim=-1))  # (1, H)

            loss = torch.tensor(0.0, device=self.device)
            n_t_samples = 10
            for _ in range(n_t_samples):
                t = torch.randint(
                    0, self.diffusion_steps, (plan_tau.shape[0],), device=self.device
                )
                noise = torch.randn_like(plan_tau)
                x_noisy = self.model.q_sample(plan_tau, t, noise)
                pred = self.model.model(x_noisy, t)
                # Per-timestep MSE weighted by constraint weights
                per_step_mse = ((pred - noise) ** 2).mean(dim=-1)  # (1, H)
                weighted_loss = (w_h * per_step_mse).sum()
                loss = loss + weighted_loss
            loss = loss / n_t_samples

        elif proxy_type == "conditioning_gap":
            # f_val: conditioning gap proxy
            # Compares denoising loss under "high-reward" vs "median-reward" conditions
            # Approximated by: loss at low noise (plan is informative) minus
            # loss at high noise (plan structure destroyed)
            # High noise ≈ conditioning on median (unconditional), low noise ≈ conditioning on R_max
            loss = torch.tensor(0.0, device=self.device)
            n_t_samples = 10

            # Low-noise loss (plan structure preserved = "high reward conditioning")
            loss_low = torch.tensor(0.0, device=self.device)
            for _ in range(n_t_samples):
                t = torch.randint(
                    0, max(1, self.diffusion_steps // 5),
                    (plan_tau.shape[0],), device=self.device
                )
                noise = torch.randn_like(plan_tau)
                x_noisy = self.model.q_sample(plan_tau, t, noise)
                pred = self.model.model(x_noisy, t)
                loss_low = loss_low + F.mse_loss(pred, noise)
            loss_low = loss_low / n_t_samples

            # High-noise loss (plan structure destroyed = "median conditioning")
            loss_high = torch.tensor(0.0, device=self.device)
            for _ in range(n_t_samples):
                t = torch.randint(
                    self.diffusion_steps * 4 // 5, self.diffusion_steps,
                    (plan_tau.shape[0],), device=self.device
                )
                noise = torch.randn_like(plan_tau)
                x_noisy = self.model.q_sample(plan_tau, t, noise)
                pred = self.model.model(x_noisy, t)
                loss_high = loss_high + F.mse_loss(pred, noise)
            loss_high = loss_high / n_t_samples

            # Conditioning gap: how much better model is at low noise vs high noise
            # Negative of low-noise loss + positive of high-noise loss
            # (higher gap = model relies more on plan structure)
            loss = loss_low - loss_high

        else:
            raise ValueError(f"Unknown proxy_type: {proxy_type}")

        self.model.zero_grad()
        loss.backward()

        grads: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
        return grads

    def compute_training_gradient(
        self, train_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of training loss for a single trajectory segment.

        L_train(tau_i; theta) = E_t || eps - eps_theta(tau_i^t, t) ||^2

        Args:
            train_idx: index into the dataset.

        Returns:
            Dict mapping parameter name -> gradient tensor.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        sample = self.dataset[train_idx]
        x = sample["trajectories"].unsqueeze(0).to(self.device)

        # Average over a few timesteps for a more stable gradient
        loss = torch.tensor(0.0, device=self.device)
        n_t_samples = 5
        for _ in range(n_t_samples):
            t = torch.randint(0, self.diffusion_steps, (1,), device=self.device)
            noise = torch.randn_like(x)
            x_noisy = self.model.q_sample(x, t, noise)
            pred = self.model.model(x_noisy, t)
            loss = loss + F.mse_loss(pred, noise)
        loss = loss / n_t_samples

        self.model.zero_grad()
        loss.backward()

        grads: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
        return grads

    # ------------------------------------------------------------------
    # 3. H^{-1} v products
    # ------------------------------------------------------------------

    def _ihvp_ekfac(
        self, v: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute H^{-1} v using EK-FAC.

        For layer l with Kronecker factorisation F_l = A_l otimes G_l:

            F_l^{-1} vec(V_l) = vec( Q_G  diag(1 / (lambda_G otimes lambda_A + damping))
                                      (Q_G^T V_l Q_A) Q_A^T )

        where V_l is the gradient reshaped to (d_out, d_in).

        Args:
            v: per-parameter gradient dict.

        Returns:
            per-parameter dict of H^{-1} v.
        """
        assert self.ekfac_state is not None, (
            "Must call compute_hessian_approximation() first."
        )

        result: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if name not in v:
                continue
            grad = v[name]

            # Find the corresponding layer in ekfac_state
            # Parameters are named like "model.down_blocks.0.0.conv1.weight"
            # We need to match to the module name
            layer_name = self._param_to_layer_name(name)
            lf = self.ekfac_state.layers.get(layer_name)

            if lf is None or lf.Q_A is None or lf.Q_G is None:
                # Fall back to simple damped inverse for layers without factors
                result[name] = grad / self.config.damping
                continue

            # Reshape grad to 2D: (d_out, d_in)
            original_shape = grad.shape
            if grad.dim() == 1:
                # bias — just use diagonal
                result[name] = grad / self.config.damping
                continue
            elif grad.dim() == 3:
                layer_mod = self._target_layers.get(layer_name)
                if isinstance(layer_mod, nn.ConvTranspose1d):
                    # ConvTranspose1d weight: (C_in, C_out, K)
                    # With spatial-averaged KFAC: G is (C_in, C_in), A is (C_out, C_out)
                    # We use per-kernel-position approximation:
                    # F^{-1} vec(W) ≈ G^{-1} W_k A^{-1} for each kernel position k
                    C_in, C_out, K = grad.shape
                    # Apply KFAC per kernel position
                    if lf is not None and lf.Q_A is not None and lf.Q_G is not None:
                        Q_A = lf.Q_A; Q_G = lf.Q_G
                        L_A = lf.Lambda_A; L_G = lf.Lambda_G
                        inv_result = torch.zeros_like(grad)
                        for k_idx in range(K):
                            V_k = grad[:, :, k_idx]  # (C_in, C_out)
                            try:
                                proj = Q_G.T @ V_k @ Q_A
                                eig_mat = L_G[:, None] * L_A[None, :] + self.config.damping
                                proj = proj / eig_mat
                                inv_result[:, :, k_idx] = Q_G @ proj @ Q_A.T
                            except RuntimeError:
                                inv_result[:, :, k_idx] = V_k / self.config.damping
                        result[name] = inv_result
                        continue
                    else:
                        result[name] = grad / self.config.damping
                        continue
                else:
                    # Conv1d weight: (C_out, C_in, kernel_size)
                    d_out = grad.shape[0]
                    d_in = grad.shape[1] * grad.shape[2]
                    V = grad.reshape(d_out, d_in)
            else:
                # Linear weight: (d_out, d_in)
                V = grad

            # Ensure dimension compatibility (truncated eigenvectors)
            Q_A = lf.Q_A  # (d_in, k_A)
            Q_G = lf.Q_G  # (d_out, k_G)
            L_A = lf.Lambda_A  # (k_A,)
            L_G = lf.Lambda_G  # (k_G,)

            # Project: Q_G^T V Q_A  ->  (k_G, k_A)
            try:
                projected = Q_G.T @ V @ Q_A
            except RuntimeError:
                # Dimension mismatch — fall back
                result[name] = grad / self.config.damping
                continue

            # Eigenvalue matrix: lambda_G_i * lambda_A_j + damping
            eig_matrix = L_G[:, None] * L_A[None, :] + self.config.damping
            projected = projected / eig_matrix

            # Un-project: Q_G (projected) Q_A^T  ->  (d_out, d_in)
            V_inv = Q_G @ projected @ Q_A.T
            result[name] = V_inv.reshape(original_shape)

        return result

    def _ihvp_diagonal(
        self, v: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute H^{-1} v using diagonal Fisher: (H^{-1} v)_i = v_i / F_ii."""
        result: Dict[str, torch.Tensor] = {}
        for name, grad in v.items():
            if name in self._diagonal_hessian:
                result[name] = grad / self._diagonal_hessian[name]
            else:
                result[name] = grad / self.config.damping
        return result

    def _param_to_layer_name(self, param_name: str) -> str:
        """Map a parameter name (e.g., 'model.conv1.weight') to module name
        (e.g., 'model.conv1')."""
        parts = param_name.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else param_name

    # ------------------------------------------------------------------
    # 4. Influence score computation
    # ------------------------------------------------------------------

    def compute_influence(
        self,
        train_idx: int,
        plan_tau: torch.Tensor,
        proxy_type: str = "likelihood",
    ) -> float:
        """Compute influence of training sample train_idx on the plan.

        I(z_i, z_test) = - g_test^T H^{-1} g_train

        A positive score means removing z_i would *decrease* the proxy
        (i.e., z_i is helpful for the proxy).  A negative score means
        removing z_i would *increase* the proxy (z_i is harmful).

        Args:
            train_idx: index of training trajectory segment.
            plan_tau: the generated plan (horizon, transition_dim).
            proxy_type: which proxy measurement to use.

        Returns:
            Scalar influence score.
        """
        g_test = self.compute_proxy_gradient(plan_tau, proxy_type)
        g_train = self.compute_training_gradient(train_idx)

        # H^{-1} g_train
        if self.config.hessian_approx in ("ekfac", "kfac"):
            h_inv_g = self._ihvp_ekfac(g_train)
        elif self.config.hessian_approx == "diagonal":
            h_inv_g = self._ihvp_diagonal(g_train)
        else:
            raise ValueError(
                f"Unknown hessian_approx: {self.config.hessian_approx}"
            )

        # Dot product: -g_test^T H^{-1} g_train
        score = 0.0
        for name in g_test:
            if name in h_inv_g:
                score -= (g_test[name] * h_inv_g[name]).sum().item()

        return score

    def compute_all_influences(
        self,
        plan_tau: torch.Tensor,
        proxy_type: str = "likelihood",
        max_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute influence scores for all (or a subset of) training trajectories.

        This is the main entry point for influence computation.  It pre-
        computes the test gradient and H^{-1} factorisation, then iterates
        over training samples.

        Args:
            plan_tau: (horizon, transition_dim) generated plan.
            proxy_type: proxy measurement type.
            max_samples: if set, only compute for the first N samples.

        Returns:
            (N,) array of influence scores, one per training segment.
        """
        n = max_samples or len(self.dataset)
        n = min(n, len(self.dataset))

        # Pre-compute test gradient
        g_test = self.compute_proxy_gradient(plan_tau, proxy_type)

        # Pre-compute H^{-1} g_test for efficiency
        # (then I(z_i) = - (H^{-1} g_test)^T g_train_i,
        #  which is equivalent by symmetry of H)
        if self.config.hessian_approx in ("ekfac", "kfac"):
            h_inv_g_test = self._ihvp_ekfac(g_test)
        elif self.config.hessian_approx == "diagonal":
            h_inv_g_test = self._ihvp_diagonal(g_test)
        else:
            raise ValueError(
                f"Unknown hessian_approx: {self.config.hessian_approx}"
            )

        scores = np.zeros(n, dtype=np.float64)

        for i in tqdm(range(n), desc="Computing influences"):
            g_train = self.compute_training_gradient(i)
            score = 0.0
            for name in h_inv_g_test:
                if name in g_train:
                    score -= (h_inv_g_test[name] * g_train[name]).sum().item()
            scores[i] = score

        return scores

    # ------------------------------------------------------------------
    # Batched influence (more efficient for large datasets)
    # ------------------------------------------------------------------

    def compute_all_influences_batched(
        self,
        plan_tau: torch.Tensor,
        proxy_type: str = "likelihood",
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Batched version of compute_all_influences.

        Instead of computing per-sample gradients one by one, this uses
        the per-sample gradient trick with vmap-style accumulation.

        Args:
            plan_tau: (horizon, transition_dim) generated plan.
            proxy_type: proxy measurement type.
            batch_size: number of training samples per batch.
            max_samples: if set, only compute for the first N samples.

        Returns:
            (N,) array of influence scores.
        """
        n = max_samples or len(self.dataset)
        n = min(n, len(self.dataset))

        # Pre-compute test gradient and H^{-1} g_test
        g_test = self.compute_proxy_gradient(plan_tau, proxy_type)
        if self.config.hessian_approx in ("ekfac", "kfac"):
            h_inv_g_test = self._ihvp_ekfac(g_test)
        elif self.config.hessian_approx == "diagonal":
            h_inv_g_test = self._ihvp_diagonal(g_test)
        else:
            raise ValueError(
                f"Unknown hessian_approx: {self.config.hessian_approx}"
            )

        # Flatten H^{-1} g_test into a single vector for dot products
        flat_h_inv = []
        param_names = []
        for name, param in self.model.named_parameters():
            if name in h_inv_g_test:
                flat_h_inv.append(h_inv_g_test[name].flatten())
                param_names.append(name)
        flat_h_inv = torch.cat(flat_h_inv)  # (P,)

        scores = np.zeros(n, dtype=np.float64)

        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        idx = 0
        for batch in tqdm(loader, desc="Batched influence"):
            if idx >= n:
                break
            x = batch["trajectories"].to(self.device)
            bs = x.shape[0]

            # Compute per-sample gradients using sum trick
            for j in range(bs):
                if idx >= n:
                    break
                xi = x[j : j + 1]
                loss = torch.tensor(0.0, device=self.device)
                for _ in range(3):
                    t = torch.randint(
                        0, self.diffusion_steps, (1,), device=self.device
                    )
                    noise = torch.randn_like(xi)
                    x_noisy = self.model.q_sample(xi, t, noise)
                    pred = self.model.model(x_noisy, t)
                    loss = loss + F.mse_loss(pred, noise)
                loss = loss / 3.0

                self.model.zero_grad()
                loss.backward()

                flat_grad = []
                for name in param_names:
                    p = dict(self.model.named_parameters())[name]
                    if p.grad is not None:
                        flat_grad.append(p.grad.detach().flatten())
                    else:
                        flat_grad.append(
                            torch.zeros(p.numel(), device=self.device)
                        )
                flat_grad = torch.cat(flat_grad)

                scores[idx] = -(flat_h_inv * flat_grad).sum().item()
                idx += 1

        return scores[:n]
