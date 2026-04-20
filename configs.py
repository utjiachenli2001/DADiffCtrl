"""
Configuration defaults for Trajectory Influence Functions experiments.

Centralizes hyperparameters for the Diffuser model, influence computation,
evaluation protocols, and D4RL environment specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# D4RL environment registry
# ---------------------------------------------------------------------------

ENV_SPECS: Dict[str, Dict] = {
    "halfcheetah": {
        "full_name": "halfcheetah",
        "gym_id_template": "halfcheetah-{dataset}-v2",
        "state_dim": 17,
        "action_dim": 6,
        "max_episode_steps": 1000,
        "reward_scale": 1.0,
    },
    "hopper": {
        "full_name": "hopper",
        "gym_id_template": "hopper-{dataset}-v2",
        "state_dim": 11,
        "action_dim": 3,
        "max_episode_steps": 1000,
        "reward_scale": 1.0,
    },
    "walker2d": {
        "full_name": "walker2d",
        "gym_id_template": "walker2d-{dataset}-v2",
        "state_dim": 17,
        "action_dim": 6,
        "max_episode_steps": 1000,
        "reward_scale": 1.0,
    },
}

DATASET_VARIANTS: List[str] = [
    "medium",
    "medium-replay",
    "medium-expert",
    "expert",
]


def get_gym_id(env_name: str, dataset: str) -> str:
    """Return the full Gym environment ID for a given env/dataset pair."""
    spec = ENV_SPECS[env_name]
    return spec["gym_id_template"].format(dataset=dataset)


# ---------------------------------------------------------------------------
# Diffuser model config
# ---------------------------------------------------------------------------

@dataclass
class DiffuserConfig:
    """Hyperparameters for the minimal Diffuser (Temporal U-Net + DDPM)."""

    # Trajectory horizon (number of timesteps per planning segment)
    horizon: int = 32

    # Architecture -----------------------------------------------------------
    # Hidden dimension of the U-Net
    dim: int = 128
    # Channel multipliers at each U-Net resolution level
    dim_mults: Tuple[int, ...] = (1, 2, 4)
    # Number of residual blocks per resolution level
    n_residual_blocks: int = 2
    # Dropout probability inside residual blocks
    dropout: float = 0.0

    # Diffusion process ------------------------------------------------------
    n_diffusion_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # "linear" | "cosine"
    # Whether to predict noise (eps) or the clean sample (x0)
    predict_epsilon: bool = True
    # Gradient scale for reward-conditioned guidance during planning
    guidance_scale: float = 1.2

    # Training ---------------------------------------------------------------
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    n_train_steps: int = 200_000
    # EMA decay for model weights
    ema_decay: float = 0.995
    # How often to log training metrics
    log_interval: int = 1000
    # How often to save checkpoints
    save_interval: int = 20_000
    # Gradient clipping (max norm); 0 disables
    grad_clip: float = 1.0

    # Planning / inference ---------------------------------------------------
    n_planning_steps: int = 200  # typically == n_diffusion_steps
    # Number of parallel plans to sample, then pick best
    n_plan_samples: int = 64

    # Device -----------------------------------------------------------------
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Influence function config
# ---------------------------------------------------------------------------

@dataclass
class InfluenceConfig:
    """Hyperparameters for trajectory influence computation."""

    # Hessian approximation: "ekfac" | "kfac" | "diagonal"
    hessian_approx: str = "ekfac"

    # Number of data points to use when accumulating Kronecker factors.
    # None means use the full training set.
    n_hessian_samples: Optional[int] = None

    # Damping term lambda added to Hessian diagonal for numerical stability
    damping: float = 1e-4

    # For EK-FAC: number of top eigenvectors to keep per layer (0 = all)
    n_eigenvectors: int = 100

    # Proxy measurement type: "likelihood" | "reward_conditioned" |
    #                         "constraint_satisfaction" | "return"
    proxy_type: str = "likelihood"

    # Batch size when computing per-sample gradients
    gradient_batch_size: int = 32

    # Device (follows DiffuserConfig.device by default)
    device: str = "cuda"


# ---------------------------------------------------------------------------
# D-TRAK baseline config
# ---------------------------------------------------------------------------

@dataclass
class DTRAKConfig:
    """Hyperparameters for the Trajectory D-TRAK baseline."""

    # Dimension of random projection
    projection_dim: int = 4096

    # Number of model checkpoints to ensemble over
    n_checkpoints: int = 1

    # Random seed for the projection matrix
    proj_seed: int = 42

    # Device
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Hyperparameters for evaluation protocols."""

    # --- Linear Datamodeling Score (LDS) ---
    # Number of random training subsets to sample
    n_subsets: int = 50
    # Fraction of training data kept in each subset
    subset_fraction: float = 0.5
    # Number of training steps for each retrained model (can be fewer than full)
    retrain_steps: int = 50_000

    # --- Data curation ---
    # Fractions of training data to prune
    prune_fractions: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )

    # --- Safety attribution AUC ---
    # Threshold on per-step constraint value to mark a trajectory as unsafe
    safety_threshold: float = 0.0

    # --- General ---
    # Number of independent seeds for repeated experiments
    n_seeds: int = 3
    # Base random seed
    base_seed: int = 0


# ---------------------------------------------------------------------------
# Convenience: build a full experiment config from CLI-style arguments
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level configuration aggregating all sub-configs."""

    env_name: str = "halfcheetah"
    dataset: str = "medium"
    experiment: str = "lds"  # "lds" | "safety" | "curation" | "all"

    diffuser: DiffuserConfig = field(default_factory=DiffuserConfig)
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)
    dtrak: DTRAKConfig = field(default_factory=DTRAKConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Where to save results
    results_dir: str = "/home/ljc37/dr-claw/DiffusionControl/Experiment/analysis"

    # Checkpoint directory
    checkpoint_dir: str = (
        "/home/ljc37/dr-claw/DiffusionControl/Experiment/core_code/checkpoints"
    )

    def __post_init__(self) -> None:
        """Propagate environment dimensions into diffuser config."""
        spec = ENV_SPECS.get(self.env_name)
        if spec is None:
            raise ValueError(
                f"Unknown environment '{self.env_name}'. "
                f"Choose from {list(ENV_SPECS.keys())}."
            )
        # transition_dim used by the U-Net = state_dim + action_dim
        self.transition_dim: int = spec["state_dim"] + spec["action_dim"]
        self.state_dim: int = spec["state_dim"]
        self.action_dim: int = spec["action_dim"]
        self.gym_id: str = get_gym_id(self.env_name, self.dataset)
