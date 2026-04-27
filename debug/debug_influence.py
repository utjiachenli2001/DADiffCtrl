"""Quick diagnostic: inspect what the influence pipeline actually produces."""
import os
import random
import numpy as np
import torch

from configs import (
    ExperimentConfig, DiffuserConfig, InfluenceConfig,
    EvaluationConfig, DTRAKConfig,
)
from diffuser_minimal import (
    TemporalUNet, GaussianDiffusion, TrajectoryDataset, plan,
)
from influence_functions import TrajectoryInfluenceComputer

device = "cuda"
N_HESS_SAMPLES = 200       # quick Hessian estimation
N_EVAL_SAMPLES = 200       # how many trajectories to score

CKPT = "/mnt/sdb/ljc/DADiffCtrl/checkpoints/diffuser_final.pt"

# --- Build config exactly like main run ---
diffuser_cfg = DiffuserConfig(device=device)
influence_cfg = InfluenceConfig(
    device=device, hessian_approx="ekfac",
    damping=1e-4, n_eigenvectors=100,
    proxy_type="likelihood",
)
config = ExperimentConfig(
    env_name="halfcheetah", dataset="medium", experiment="all",
    diffuser=diffuser_cfg, influence=influence_cfg,
    dtrak=DTRAKConfig(device=device), evaluation=EvaluationConfig(),
)

# --- Load dataset + model ---
print("Loading dataset...")
dataset = TrajectoryDataset(env_name="halfcheetah", dataset_variant="medium",
                            horizon=config.diffuser.horizon)
print(f"  {len(dataset)} segments, transition_dim={config.transition_dim}")

print("Loading checkpoint...")
unet = TemporalUNet(
    transition_dim=config.transition_dim,
    dim=config.diffuser.dim, dim_mults=config.diffuser.dim_mults,
    n_residual_blocks=config.diffuser.n_residual_blocks,
)
diffusion = GaussianDiffusion(unet, config.diffuser).to(device)
diffusion.load_state_dict(torch.load(CKPT, map_location=device))
diffusion.eval()

# --- Generate reference plan ---
print("Generating reference plan...")
plans = plan(diffusion, dataset, config, reward_guidance=False)
ref_plan = plans[0]
ref_plan_tensor = torch.from_numpy(
    (ref_plan - dataset.data_mean) / dataset.data_std
).float()

# --- TIC ---
print(f"Computing Hessian factors over {N_HESS_SAMPLES} samples...")
tic = TrajectoryInfluenceComputer(
    model=diffusion, dataset=dataset, config=influence_cfg, diffusion_steps=200,
)
tic.compute_hessian_approximation(n_samples=N_HESS_SAMPLES)

# --- Inspect proxy gradient ---
print("\n=== PROXY GRADIENT ===")
g_test = tic.compute_proxy_gradient(ref_plan_tensor, "likelihood")
norms = {n: g.norm().item() for n, g in g_test.items()}
total_norm = (sum(v ** 2 for v in norms.values())) ** 0.5
print(f"  total ||g_test||         = {total_norm:.4e}")
print(f"  layers in g_test         = {len(g_test)}")
print(f"  layers with nonzero grad = {sum(1 for v in norms.values() if v > 1e-8)}")
top5 = sorted(norms.items(), key=lambda kv: -kv[1])[:5]
print(f"  top-5 layer norms:")
for name, v in top5:
    print(f"    {v:.4e}  {name}")

# --- iHVP ---
print("\n=== iHVP ===")
h_inv_g = tic._ihvp_ekfac(g_test)
total_norm_ihvp = (sum(g.norm().item() ** 2 for g in h_inv_g.values())) ** 0.5
print(f"  total ||H^-1 g_test||    = {total_norm_ihvp:.4e}")
print(f"  ratio ||iHVP||/||g||     = {total_norm_ihvp / total_norm:.4f}")

# --- Score distribution over random samples ---
print(f"\n=== SCORES over {N_EVAL_SAMPLES} random training samples ===")
random.seed(0)
indices = random.sample(range(len(dataset)), N_EVAL_SAMPLES)

scores = []
train_norms = []
cosines = []
for i in indices:
    g_train = tic.compute_training_gradient(i)
    norm = (sum(g.norm().item() ** 2 for g in g_train.values())) ** 0.5
    train_norms.append(norm)
    # Score (sign-corrected: I = +g_test · iHVP · g_train)
    s = 0.0
    dot_test_train = 0.0
    for name in g_test:
        if name in g_train:
            s += (h_inv_g[name] * g_train[name]).sum().item()
            dot_test_train += (g_test[name] * g_train[name]).sum().item()
    scores.append(s)
    cos = dot_test_train / (total_norm * norm + 1e-20)
    cosines.append(cos)

scores = np.array(scores)
train_norms = np.array(train_norms)
cosines = np.array(cosines)

def stats(name, arr):
    print(f"  {name:25s}  mean={arr.mean():+.4e}  std={arr.std():.4e}  "
          f"min={arr.min():+.4e}  max={arr.max():+.4e}")

stats("scores", scores)
stats("|scores|", np.abs(scores))
stats("train grad norms", train_norms)
stats("cos(g_test, g_train)", cosines)

# Diagnostic checks
print("\n=== DIAGNOSTICS ===")
if scores.std() < 1e-10:
    print("  ⚠️  Scores are essentially constant — pipeline is broken.")
elif np.abs(scores).max() / (np.abs(scores).mean() + 1e-30) > 1e4:
    print("  ⚠️  A few extreme scores dominate; rest are tiny — likely outlier samples.")
else:
    print("  Score distribution looks well-spread.")

if abs(cosines.mean()) < 0.01:
    print("  ⚠️  Mean cos(g_test, g_train) ≈ 0 — proxy & training gradients orthogonal.")
    print("     This often means the proxy gradient is dominated by random noise.")

# Correlation: do scores correlate with grad-norms? (Should not — that'd be a bias.)
from scipy.stats import spearmanr
sp, p = spearmanr(np.abs(scores), train_norms)
print(f"  Spearman(|score|, ||g_train||) = {sp:+.3f} (p={p:.3f})")
if abs(sp) > 0.5:
    print("  ⚠️  Score magnitudes strongly tracking training-gradient norms.")
    print("     This means scores are dominated by ||g_train|| not by alignment.")
