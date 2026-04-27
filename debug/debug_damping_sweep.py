"""Damping sweep — finds a damping value that produces sensible iHVP norms."""
import random
import numpy as np
import torch
from scipy.stats import spearmanr

from configs import (
    ExperimentConfig, DiffuserConfig, InfluenceConfig,
    EvaluationConfig, DTRAKConfig,
)
from diffuser_minimal import (
    TemporalUNet, GaussianDiffusion, TrajectoryDataset, plan,
)
from influence_functions import TrajectoryInfluenceComputer

device = "cuda"
N_HESS = 200
N_EVAL = 100
CKPT = "/mnt/sdb/ljc/DADiffCtrl/checkpoints/diffuser_final.pt"

diffuser_cfg = DiffuserConfig(device=device)
config = ExperimentConfig(
    env_name="halfcheetah", dataset="medium", experiment="all",
    diffuser=diffuser_cfg,
    influence=InfluenceConfig(device=device, hessian_approx="ekfac",
                               damping=1e-4, n_eigenvectors=100),
    dtrak=DTRAKConfig(device=device), evaluation=EvaluationConfig(),
)

print("Loading dataset + model...")
dataset = TrajectoryDataset("halfcheetah", "medium", horizon=config.diffuser.horizon)
unet = TemporalUNet(transition_dim=config.transition_dim, dim=128,
                     dim_mults=(1,2,4), n_residual_blocks=2)
diffusion = GaussianDiffusion(unet, config.diffuser).to(device)
diffusion.load_state_dict(torch.load(CKPT, map_location=device))
diffusion.eval()

plans = plan(diffusion, dataset, config, reward_guidance=False)
ref_plan_tensor = torch.from_numpy(
    (plans[0] - dataset.data_mean) / dataset.data_std
).float()

# Pre-compute Hessian factors once (damping is applied AFTER, in IHVP)
# But damping is also added into A and G before eigendecomposition... so
# we need to recompute. To keep this fast, compute factors with tiny initial
# damping then just override the eig_matrix damping.
print(f"Computing Hessian factors over {N_HESS} samples...")
tic = TrajectoryInfluenceComputer(model=diffusion, dataset=dataset,
                                   config=config.influence, diffusion_steps=200)
tic.compute_hessian_approximation(n_samples=N_HESS)

# Pre-compute proxy + training grads ONCE, then sweep damping
print("Computing proxy + training gradients...")
g_test = tic.compute_proxy_gradient(ref_plan_tensor, "likelihood")
total_norm_test = (sum(g.norm().item() ** 2 for g in g_test.values())) ** 0.5

random.seed(0)
indices = random.sample(range(len(dataset)), N_EVAL)
g_trains = []
train_norms = []
for i in indices:
    gt = tic.compute_training_gradient(i)
    norm = (sum(g.norm().item() ** 2 for g in gt.values())) ** 0.5
    g_trains.append(gt)
    train_norms.append(norm)
train_norms = np.array(train_norms)

print(f"\n{'damping':>10s} | {'||iHVP||/||g||':>14s} | {'score std':>10s} | "
      f"{'Sp(|score|,||g||)':>18s}")
print("-" * 72)

for damping in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    # Override damping in TIC
    tic.config.damping = damping
    h_inv_g = tic._ihvp_ekfac(g_test)
    ihvp_norm = (sum(g.norm().item() ** 2 for g in h_inv_g.values())) ** 0.5
    ratio = ihvp_norm / total_norm_test

    scores = []
    for gt in g_trains:
        s = sum((h_inv_g[name] * gt[name]).sum().item()
                for name in g_test if name in gt)
        scores.append(s)
    scores = np.array(scores)
    sp, _ = spearmanr(np.abs(scores), train_norms)
    print(f"{damping:>10.0e} | {ratio:>14.2f} | {scores.std():>10.3e} | {sp:>+18.3f}")

print("\nWant: ratio ~10-100, Spearman(|score|,||g||) close to 0.")
