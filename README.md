# Trajectory Influence Functions for Diffusion-Based Control

Code artifact for the paper:

> **Trajectory Influence Functions for Diffusion-Based Control: Data Attribution for Safe Planning**
> Anonymous Author(s). NeurIPS 2026 Submission.

This repository implements Trajectory Influence Functions (Trajectory-IF), the first framework for data attribution in diffusion-based trajectory planners. It includes the core influence computation method, baselines, evaluation protocols, and experiment runners needed to reproduce the paper's results.

---

## Repository Layout

| File | Description |
|------|-------------|
| `configs.py` | Configuration dataclasses for all hyperparameters (Diffuser, influence, D-TRAK, evaluation) |
| `diffuser_minimal.py` | Minimal Diffuser implementation (Temporal U-Net + DDPM) |
| `influence_functions.py` | Core TIF implementation with EK-FAC/K-FAC/diagonal approximations |
| `baselines.py` | Attribution baselines: Random, RewardRanking, NearestNeighbor, D-TRAK |
| `evaluation.py` | Evaluation protocols: LDS, SafetyAUC, DataCuration, Intervention |
| `run_experiments.py` | Main experiment runner for a single (env, dataset, seed) cell |
| `run_grid.py` | Grid orchestrator for multi-cell experiments |
| `run_ablation.py` | Hessian approximation ablation study |
| `aggregate_results.py` | Aggregate results across seeds, output LaTeX tables |
| `run_all.sh` | Top-level script to run full pipeline |
| `debug/` | Debug and validation scripts |

---

## Installation

**Requirements:** Python 3.10+, CUDA-capable GPU with 48GB+ VRAM recommended.

```bash
# Create conda environment
conda create -n dadiffctrl python=3.10
conda activate dadiffctrl

# Install cython<3 BEFORE mujoco_py builds (required for older mujoco)
pip install "cython<3"

# Install setuptools with pkg_resources (required by d4rl entry points)
pip install "setuptools<81"

# Install mjrl (required by d4rl locomotion environments)
pip install git+https://github.com/aravindr93/mjrl.git

# Install core dependencies
pip install torch>=2.0 numpy scipy scikit-learn tqdm

# Install D4RL (after mjrl)
pip install d4rl

# Install gym version compatible with D4RL
pip install "gym==0.21.0"

# Set library path for NVIDIA drivers (WSL/Linux)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Avoid fragmentation OOM in D-TRAK (recommended)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Quick Start

### Smoke Test (~15 minutes, 1 GPU)

```bash
python run_grid.py --smoke-test
```

### Single Cell Run

```bash
python run_experiments.py \
    --env halfcheetah \
    --dataset medium \
    --experiment all \
    --seed 0
```

### Full Grid (9 cells x 3 seeds = 27 runs)

```bash
# Sequential (1 GPU):
python run_grid.py --experiments all

# Parallel (4 GPUs):
python run_grid.py --experiments all --mode parallel --n-workers 4 --gpu-ids 0 1 2 3

# With resume (skip completed cells):
python run_grid.py --experiments all --mode parallel --n-workers 4 --gpu-ids 0 1 2 3 --resume
```

---

## Reproducing the Paper

### Full Experiment Pipeline

```bash
# Set environment variables (optional, defaults to ./analysis and ./checkpoints)
export DADIFFCTRL_RESULTS_DIR=/path/to/analysis
export DADIFFCTRL_CHECKPOINT_DIR=/path/to/checkpoints

# Run all experiments
bash run_all.sh
```

Or step by step:

```bash
# 1. Main grid: all 9 cells x 3 seeds, all experiments
python run_grid.py --experiments all --mode parallel --n-workers 12 --gpu-ids 0 1 2 3 4 5 6 7 8 9 10 11

# 2. Hessian ablation (EK-FAC vs K-FAC vs diagonal vs plain-dot)
python run_ablation.py

# 3. Aggregate results and output LaTeX tables
python aggregate_results.py --latex --output analysis/aggregated.json
```

### Expected Output

After running, you'll have:
- `analysis/{env}_{dataset}_seed{seed}_all_ekfac_{timestamp}.json` — per-cell results
- `analysis/ablation_hessian_{timestamp}.json` — ablation results
- `analysis/aggregated.json` — mean +/- std across seeds
- `analysis/failed_cells.json` — any failed runs (for retry)

---

## Computing Influence Scores

```python
import torch
from configs import ExperimentConfig
from diffuser_minimal import TrajectoryDataset, train, plan
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

## Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--env` | `halfcheetah` | D4RL environment: halfcheetah, hopper, walker2d |
| `--dataset` | `medium` | Dataset variant: medium, medium-replay, medium-expert |
| `--experiment` | `lds` | Experiment: lds, safety, curation, intervention, all |
| `--hessian-approx` | `ekfac` | Hessian approximation: ekfac, kfac, diagonal, plain_dot |
| `--proxy-type` | `likelihood` | Proxy: likelihood, reward_conditioned, constraint_satisfaction, conditioning_gap |
| `--seed` | `0` | Random seed |
| `--smoke-test` | `false` | Minimal end-to-end validation |
| `--debug` | `false` | Tiny config for quick testing |
| `--checkpoint` | `None` | Path to pre-trained model checkpoint |

See `configs.py` for the full set of configurable hyperparameters.

---

## Known Caveats

- **D-TRAK GPU memory**: The random projection matrix for D-TRAK with `projection_dim=1024` consumes ~37GB in fp16. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
- **LDS subset_fraction**: We use `subset_fraction=0.5` following the TRAK convention.
- **Safety threshold**: Unsafe trajectories are labeled at the 95th percentile of state constraint values (top 5% marked unsafe).
- **Intervention rollouts**: MPC-style replanning every `horizon=32` steps.

## Hardware Requirements

- **GPU memory**: ~47GB per process recommended (A6000, A100)
- **System RAM**: ~40GB per concurrent worker
- **Wall-clock time**: ~3 days for full 9-env x 3-seed grid on 2 GPUs

---

## Citation

```bibtex
@inproceedings{tif2026,
  title={Trajectory Influence Functions for Diffusion-Based Control},
  author={[Authors]},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

---

## License

MIT License (placeholder — to be finalized upon publication).
