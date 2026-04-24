# Trajectory Influence Functions for Diffusion-Based Control

Code artifact for the paper:

> **Trajectory Influence Functions for Diffusion-Based Control: Data Attribution for Safe Planning**
> Anonymous Author(s). NeurIPS 2026 Submission.

This repository implements Trajectory Influence Functions (Trajectory-IF), the first framework for data attribution in diffusion-based trajectory planners. It includes the core influence computation method, baselines, evaluation protocols, and experiment runners needed to reproduce the paper's results.

---

## Installation

**Requirements:** Python >= 3.9, CUDA-capable GPU (recommended).

```bash
# Clone the repository
git clone <repo-url>
cd DiffusionControl/Experiment/core_code

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note on D4RL:** The D4RL package requires MuJoCo. Follow the [MuJoCo installation guide](https://github.com/openai/mujoco-py#install-mujoco) and ensure `mujoco-py` is properly configured before running experiments.

---

## File Descriptions

| File | Description |
|------|-------------|
| `configs.py` | Centralized configuration dataclasses for all hyperparameters: Diffuser architecture (`DiffuserConfig`), influence computation (`InfluenceConfig`), D-TRAK baseline (`DTRAKConfig`), evaluation protocols (`EvaluationConfig`), and the top-level `ExperimentConfig`. Also contains the D4RL environment registry with state/action dimensions. |
| `diffuser_minimal.py` | Self-contained Diffuser implementation (~680 lines). Includes the `TemporalUNet` (1D temporal U-Net with residual blocks, sinusoidal timestep embeddings, and skip connections), `GaussianDiffusion` (DDPM forward/reverse process and loss), `TrajectoryDataset` (D4RL loader that chunks episodes into fixed-horizon segments with normalization), `EMA` helper, `train()` loop, and `plan()` inference with optional reward-conditioned guidance. |
| `influence_functions.py` | Core contribution: `TrajectoryInfluenceComputer` class (~790 lines). Implements EK-FAC Hessian approximation adapted for 1D temporal convolutions, four proxy measurement gradients (likelihood, reward-conditioned, constraint satisfaction, return), inverse-Hessian-vector products via Kronecker-factored eigenbasis, and both sequential and batched influence score computation. |
| `baselines.py` | Four attribution baselines (~300 lines): `RandomAttribution` (uniform random scores), `RewardRanking` (cumulative reward), `NearestNeighborAttribution` (negative L2 distance in trajectory space), and `TrajectoryDTRAK` (D-TRAK with random-projected gradients adapted for trajectory diffusion). |
| `evaluation.py` | Three evaluation protocols (~550 lines): `TrajectoryLDS` (Linear Datamodeling Score via subset retraining and Spearman correlation), `SafetyAttributionAUC` (ROC AUC for identifying unsafe training data), and `DataCurationEvaluator` (attribution-guided data pruning with retrain-and-evaluate). Also includes `save_results()` for JSON serialization. |
| `run_experiments.py` | Main experiment runner (~585 lines). Parses CLI arguments, builds configs, trains or loads the Diffuser, computes influence scores for all methods, and runs the selected experiment (LDS, safety, curation, or all). Outputs results as JSON to the analysis directory. |
| `__init__.py` | Package marker. |
| `requirements.txt` | Python package dependencies. |

---

## Quick Start: Computing Influence Scores

```python
import torch
from configs import ExperimentConfig, InfluenceConfig
from diffuser_minimal import GaussianDiffusion, TemporalUNet, TrajectoryDataset, train, plan
from influence_functions import TrajectoryInfluenceComputer

# 1. Set up configuration
config = ExperimentConfig(env_name="halfcheetah", dataset="medium")

# 2. Load dataset
dataset = TrajectoryDataset(
    env_name="halfcheetah",
    dataset_variant="medium",
    horizon=32,
)

# 3. Train the Diffuser (or load a checkpoint)
diffusion, dataset = train(config, dataset=dataset)

# 4. Generate a reference plan
plans = plan(diffusion, dataset, config, reward_guidance=False)
ref_plan = torch.from_numpy(
    (plans[0] - dataset.data_mean) / dataset.data_std
).float()

# 5. Compute influence scores
tic = TrajectoryInfluenceComputer(
    model=diffusion,
    dataset=dataset,
    config=config.influence,
    diffusion_steps=config.diffuser.n_diffusion_steps,
)
tic.compute_hessian_approximation()  # Phase 1: EK-FAC factors (one-time)
scores = tic.compute_all_influences(ref_plan, proxy_type="likelihood")

# scores is an (N,) array: positive = helpful, negative = harmful
print(f"Top 5 most influential training indices: {scores.argsort()[-5:][::-1]}")
print(f"Top 5 most harmful training indices: {scores.argsort()[:5]}")
```

---

## Reproducing Paper Results

All paper experiments can be reproduced using `run_experiments.py`:

### T-LDS Experiment (Table 1)

```bash
# Run across all 9 environments (3 envs x 3 datasets)
for env in halfcheetah hopper walker2d; do
  for dataset in medium medium-replay medium-expert; do
    python run_experiments.py \
      --env $env \
      --dataset $dataset \
      --experiment lds \
      --hessian-approx ekfac \
      --seed 0
  done
done
```

### Safety Attribution (Table 2)

```bash
for env in halfcheetah hopper walker2d; do
  for dataset in medium medium-replay medium-expert; do
    python run_experiments.py \
      --env $env \
      --dataset $dataset \
      --experiment safety \
      --proxy-type constraint_satisfaction \
      --seed 0
  done
done
```

### Data Curation (Table 3)

```bash
for env in halfcheetah hopper walker2d; do
  for dataset in medium medium-replay medium-expert; do
    python run_experiments.py \
      --env $env \
      --dataset $dataset \
      --experiment curation \
      --seed 0
  done
done
```

### Ablation: Hessian Approximation (Table 4)

```bash
for approx in diagonal kfac ekfac; do
  python run_experiments.py \
    --env halfcheetah \
    --dataset medium \
    --experiment lds \
    --hessian-approx $approx \
    --seed 0
done
```

### Quick Smoke Test

```bash
# Debug mode: tiny model, few steps, small dataset (~2 min on GPU)
python run_experiments.py --env halfcheetah --dataset medium --experiment lds --debug
```

### Multi-Seed Runs

To reproduce the mean +/- std results reported in the paper, run each experiment with seeds 0, 1, 2:

```bash
for seed in 0 1 2; do
  python run_experiments.py \
    --env halfcheetah \
    --dataset medium \
    --experiment all \
    --seed $seed
done
```

Results are saved as JSON files in `Experiment/analysis/`.

---

## Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--env` | `halfcheetah` | D4RL environment: halfcheetah, hopper, walker2d |
| `--dataset` | `medium` | Dataset variant: medium, medium-replay, medium-expert |
| `--experiment` | `lds` | Experiment: lds, safety, curation, all |
| `--hessian-approx` | `ekfac` | Hessian approximation: ekfac, kfac, diagonal |
| `--proxy-type` | `likelihood` | Proxy measurement: likelihood, reward_conditioned, constraint_satisfaction, return |
| `--seed` | `0` | Random seed |
| `--debug` | `false` | Tiny config for quick testing |
| `--checkpoint` | `None` | Path to pre-trained model checkpoint |

See `configs.py` for the full set of configurable hyperparameters.

---

## License

MIT License (placeholder -- to be finalized upon publication).
