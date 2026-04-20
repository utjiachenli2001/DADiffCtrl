"""
Minimal Diffuser implementation for trajectory planning.

Reference: Janner et al., "Planning with Diffusion for Flexible Behavior
Synthesis", ICML 2022.

This is a self-contained, research-oriented implementation of the Diffuser
architecture.  It includes:
  - TemporalUNet: 1-D temporal U-Net over trajectory segments
    (batch, horizon, transition_dim).
  - GaussianDiffusion: DDPM wrapper (forward / reverse / loss).
  - TrajectoryDataset: D4RL dataset loader that chunks episodes into
    fixed-horizon segments.
  - train(): training loop.
  - plan(): reward-conditioned trajectory generation.

Approximately 500 lines.  Prioritizes clarity over speed.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs import DiffuserConfig, ExperimentConfig

# ---------------------------------------------------------------------------
# Helper: sinusoidal timestep embeddings
# ---------------------------------------------------------------------------


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timestep t.

    Args:
        timesteps: (batch,) integer timesteps.
        dim: embedding dimension.

    Returns:
        (batch, dim) embedding vectors.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResidualTemporalBlock(nn.Module):
    """Residual block with 1-D convolutions along the time axis.

    Architecture per block:
        GroupNorm -> Mish -> Conv1d -> GroupNorm -> Mish -> Dropout -> Conv1d
        + skip connection (with optional 1x1 conv if dims differ)
        + timestep conditioning via linear projection added after first conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, horizon)
            t_emb: (batch, embed_dim) diffusion-step embedding
        """
        h = self.conv1(F.mish(self.norm1(x)))
        # Add timestep embedding (broadcast over horizon)
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.conv2(self.dropout(F.mish(self.norm2(h))))
        return h + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Temporal U-Net
# ---------------------------------------------------------------------------


class TemporalUNet(nn.Module):
    """1-D temporal U-Net that processes trajectory segments.

    Input shape:  (batch, horizon, transition_dim)
    Output shape: (batch, horizon, transition_dim)

    Internally works in channel-first layout (batch, C, horizon).
    """

    def __init__(
        self,
        transition_dim: int,
        dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        n_residual_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.transition_dim = transition_dim

        # Timestep embedding MLP
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(transition_dim, dim, 1)

        # Encoder (downsampling path)
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.down_blocks = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            blocks = nn.ModuleList()
            for _ in range(n_residual_blocks):
                blocks.append(ResidualTemporalBlock(d_in, d_out, time_dim, dropout))
                d_in = d_out
            if not is_last:
                blocks.append(Downsample1d(d_out))
            self.down_blocks.append(blocks)

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim, dropout)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, time_dim, dropout)

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            blocks = nn.ModuleList()
            for j in range(n_residual_blocks):
                # First block in each level receives skip connection (double channels)
                res_in = d_out * 2 if j == 0 else d_in
                blocks.append(ResidualTemporalBlock(res_in, d_in, time_dim, dropout))
            if not is_last:
                blocks.append(Upsample1d(d_in))
            self.up_blocks.append(blocks)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.Mish(),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, horizon, transition_dim) noisy trajectory
            t: (batch,) integer diffusion timestep

        Returns:
            (batch, horizon, transition_dim) predicted noise (or x0)
        """
        # (batch, horizon, C) -> (batch, C, horizon)
        x = x.permute(0, 2, 1)
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))

        x = self.input_proj(x)

        # Encoder
        skips: List[torch.Tensor] = []
        for blocks in self.down_blocks:
            for block in blocks:
                if isinstance(block, ResidualTemporalBlock):
                    x = block(x, t_emb)
                else:
                    skips.append(x)
                    x = block(x)
            if not isinstance(blocks[-1], Downsample1d):
                skips.append(x)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        # Decoder
        for blocks in self.up_blocks:
            for block in blocks:
                if isinstance(block, ResidualTemporalBlock):
                    skip = skips.pop()
                    # Pad if needed (horizon may not be exactly divisible)
                    if x.shape[-1] != skip.shape[-1]:
                        x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
                    x = block(torch.cat([x, skip], dim=1), t_emb)
                else:
                    x = block(x)

        x = self.output_proj(x)
        # (batch, C, horizon) -> (batch, horizon, C)
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Gaussian Diffusion (DDPM)
# ---------------------------------------------------------------------------


class GaussianDiffusion(nn.Module):
    """Denoising Diffusion Probabilistic Model wrapper.

    Implements the forward process q(x_t | x_0) and the reverse process
    p_theta(x_{t-1} | x_t) with the standard DDPM loss:

        L = E_{t, eps} || eps - eps_theta(x_t, t) ||^2

    where x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
    """

    def __init__(
        self,
        model: TemporalUNet,
        config: DiffuserConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.n_steps = config.n_diffusion_steps
        self.predict_epsilon = config.predict_epsilon

        # Build noise schedule
        if config.beta_schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, self.n_steps)
        elif config.beta_schedule == "cosine":
            steps = torch.arange(self.n_steps + 1) / self.n_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers (moved to device automatically)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - F.pad(alpha_bar[:-1], (1, 0), value=1.0))
            / (1.0 - alpha_bar),
        )

    # -- Forward process (add noise) ----------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0).

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_ab * x_start + sqrt_one_minus_ab * noise

    # -- Loss ---------------------------------------------------------------

    def compute_loss(
        self, x_start: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard DDPM training loss.

        L = E_{t, eps} || eps - eps_theta(x_t, t) ||^2
        """
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred = self.model(x_noisy, t)

        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        return F.mse_loss(pred, target)

    # -- Reverse process (denoise) ------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        guidance_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """Single reverse step: sample x_{t-1} ~ p_theta(x_{t-1} | x_t)."""
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

        pred = self.model(x, t_tensor)

        if self.predict_epsilon:
            # Reconstruct x_0 estimate
            sqrt_ab = self.sqrt_alpha_bar[t]
            sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[t]
            x0_pred = (x - sqrt_one_minus_ab * pred) / sqrt_ab
        else:
            x0_pred = pred

        # Posterior mean: mu_theta(x_t, x_0_pred)
        alpha = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
        beta = self.betas[t]

        coef1 = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / (1.0 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x

        # Optional guidance (reward-conditioned planning)
        if guidance_fn is not None:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                guide_val = guidance_fn(x_in)
                grad = torch.autograd.grad(guide_val.sum(), x_in)[0]
            mean = mean + self.config.guidance_scale * self.posterior_variance[t] * grad

        if t > 0:
            noise = torch.randn_like(x)
            std = torch.sqrt(self.posterior_variance[t])
            return mean + std * noise
        return mean

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        guidance_fn: Optional[callable] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Full reverse process: generate trajectories from noise."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            x = self.p_sample(x, t, guidance_fn)
        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TrajectoryDataset(Dataset):
    """Load a D4RL dataset and chunk it into fixed-horizon trajectory segments.

    Each item is a tensor of shape (horizon, state_dim + action_dim).
    Rewards are stored separately for conditioning.
    """

    def __init__(
        self,
        env_name: str,
        dataset_variant: str,
        horizon: int = 32,
        max_trajectories: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.horizon = horizon

        # Load D4RL dataset (lazy imports to avoid top-level dependency)
        import gym  # noqa: E402
        import d4rl  # noqa: F401,E402 — registers environments
        gym_id = f"{env_name}-{dataset_variant}-v2"
        env = gym.make(gym_id)
        raw = env.get_dataset()

        observations = raw["observations"]  # (N, state_dim)
        actions = raw["actions"]  # (N, action_dim)
        rewards = raw["rewards"]  # (N,)
        terminals = raw["terminals"]  # (N,)
        timeouts = raw.get("timeouts", np.zeros_like(terminals))

        self.state_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.transition_dim = self.state_dim + self.action_dim

        # Split into episodes
        episodes_obs: List[np.ndarray] = []
        episodes_act: List[np.ndarray] = []
        episodes_rew: List[np.ndarray] = []

        ep_start = 0
        for i in range(len(terminals)):
            if terminals[i] or timeouts[i] or i == len(terminals) - 1:
                episodes_obs.append(observations[ep_start : i + 1])
                episodes_act.append(actions[ep_start : i + 1])
                episodes_rew.append(rewards[ep_start : i + 1])
                ep_start = i + 1
                if max_trajectories and len(episodes_obs) >= max_trajectories:
                    break

        # Compute per-episode returns and safety labels
        self.episode_returns = np.array(
            [r.sum() for r in episodes_rew], dtype=np.float32
        )

        # Normalisation statistics (computed over ALL transitions)
        all_obs = np.concatenate(episodes_obs, axis=0)
        all_act = np.concatenate(episodes_act, axis=0)
        all_transitions = np.concatenate([all_obs, all_act], axis=1)
        self.data_mean = all_transitions.mean(axis=0).astype(np.float32)
        self.data_std = all_transitions.std(axis=0).astype(np.float32) + 1e-6

        # Chunk episodes into segments of length `horizon`
        self.segments: List[np.ndarray] = []
        self.segment_rewards: List[np.ndarray] = []
        self.segment_episode_idx: List[int] = []  # which episode each segment came from

        for ep_idx, (obs, act, rew) in enumerate(
            zip(episodes_obs, episodes_act, episodes_rew)
        ):
            transitions = np.concatenate([obs, act], axis=1)  # (T, D)
            T = transitions.shape[0]
            for start in range(0, T - horizon + 1, horizon // 2):  # 50% overlap
                end = start + horizon
                if end > T:
                    break
                seg = (transitions[start:end] - self.data_mean) / self.data_std
                self.segments.append(seg.astype(np.float32))
                self.segment_rewards.append(rew[start:end].astype(np.float32))
                self.segment_episode_idx.append(ep_idx)

        self.segments = np.stack(self.segments)  # (N_seg, H, D)
        self.segment_rewards = np.stack(self.segment_rewards)  # (N_seg, H)
        self.segment_episode_idx = np.array(self.segment_episode_idx)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "trajectories": torch.from_numpy(self.segments[idx]),
            "rewards": torch.from_numpy(self.segment_rewards[idx]),
        }

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """Map normalised trajectory back to original scale."""
        return x * self.data_std + self.data_mean


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = decay
        self.shadow = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    config: ExperimentConfig,
    dataset: Optional[TrajectoryDataset] = None,
    verbose: bool = True,
) -> Tuple[GaussianDiffusion, TrajectoryDataset]:
    """Train the Diffuser model.

    Args:
        config: full experiment configuration.
        dataset: optional pre-built dataset (avoids reloading).
        verbose: whether to print progress.

    Returns:
        (diffusion_model, dataset)
    """
    dc = config.diffuser
    device = torch.device(dc.device if torch.cuda.is_available() else "cpu")

    # Dataset
    if dataset is None:
        dataset = TrajectoryDataset(
            env_name=config.env_name,
            dataset_variant=config.dataset,
            horizon=dc.horizon,
        )
    loader = DataLoader(
        dataset, batch_size=dc.batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    # Model
    unet = TemporalUNet(
        transition_dim=config.transition_dim,
        dim=dc.dim,
        dim_mults=dc.dim_mults,
        n_residual_blocks=dc.n_residual_blocks,
        dropout=dc.dropout,
    )
    diffusion = GaussianDiffusion(unet, dc).to(device)
    ema = EMA(diffusion, decay=dc.ema_decay)

    optimizer = torch.optim.Adam(
        diffusion.parameters(), lr=dc.learning_rate, weight_decay=dc.weight_decay
    )

    # Training loop
    step = 0
    data_iter = iter(loader)
    pbar = tqdm(total=dc.n_train_steps, desc="Training Diffuser", disable=not verbose)

    while step < dc.n_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["trajectories"].to(device)
        loss = diffusion.compute_loss(x)

        optimizer.zero_grad()
        loss.backward()
        if dc.grad_clip > 0:
            nn.utils.clip_grad_norm_(diffusion.parameters(), dc.grad_clip)
        optimizer.step()
        ema.update(diffusion)

        step += 1
        if step % dc.log_interval == 0 and verbose:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)

        if step % dc.save_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(
                diffusion.state_dict(),
                os.path.join(config.checkpoint_dir, f"diffuser_{step}.pt"),
            )

    pbar.close()

    # Save final checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(
        diffusion.state_dict(),
        os.path.join(config.checkpoint_dir, "diffuser_final.pt"),
    )

    # Apply EMA weights for inference
    ema.apply(diffusion)

    return diffusion, dataset


# ---------------------------------------------------------------------------
# Planning (inference)
# ---------------------------------------------------------------------------


def plan(
    diffusion: GaussianDiffusion,
    dataset: TrajectoryDataset,
    config: ExperimentConfig,
    reward_guidance: bool = True,
) -> np.ndarray:
    """Generate trajectory plans via iterative denoising.

    If reward_guidance is True, applies classifier-free reward-conditioned
    guidance using cumulative reward as the guidance signal.

    Args:
        diffusion: trained diffusion model.
        dataset: dataset (used for unnormalization).
        config: experiment configuration.
        reward_guidance: whether to apply reward guidance.

    Returns:
        (n_samples, horizon, transition_dim) planned trajectories in
        original (unnormalized) space.
    """
    dc = config.diffuser
    device = next(diffusion.parameters()).device
    shape = (dc.n_plan_samples, dc.horizon, config.transition_dim)

    guidance_fn = None
    if reward_guidance:
        # Simple guidance: encourage high cumulative reward.
        # We approximate reward as a linear function of state features
        # (first state_dim dimensions of each timestep).
        def guidance_fn(x: torch.Tensor) -> torch.Tensor:
            """Sum over horizon of the first state dimension as a rough
            reward proxy.  In practice this would be replaced by a
            learned reward model.
            """
            states = x[:, :, : config.state_dim]
            # Use velocity (often correlated with reward in locomotion)
            return states[:, :, 0].sum(dim=1)

    plans = diffusion.p_sample_loop(shape, guidance_fn=guidance_fn, device=device)
    plans_np = plans.cpu().numpy()

    # Unnormalize
    plans_np = dataset.unnormalize(plans_np)
    return plans_np
