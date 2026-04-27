"""Compare: TIF (with iHVP) vs plain dot product (no Hessian).

If plain dot-product gives MORE varied scores and LESS magnitude-bias than
TIF, then the EK-FAC iHVP is destroying signal rather than adding it.
"""
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
N_HESS = 1000      # more samples for better Hessian estimate
N_EVAL = 200
CKPT = "/mnt/sdb/ljc/DADiffCtrl/checkpoints/diffuser_final.pt"

diffuser_cfg = DiffuserConfig(device=device)
config = ExperimentConfig(
    env_name="halfcheetah", dataset="medium", experiment="all",
    diffuser=diffuser_cfg,
    influence=InfluenceConfig(device=device, hessian_approx="ekfac",
                               damping=1e-2, n_eigenvectors=100),
    dtrak=DTRAKConfig(device=device), evaluation=EvaluationConfig(),
)

print("Loading...")
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

print(f"Computing Hessian over {N_HESS} samples...")
tic = TrajectoryInfluenceComputer(model=diffusion, dataset=dataset,
                                   config=config.influence, diffusion_steps=200)
tic.compute_hessian_approximation(n_samples=N_HESS)

print("Computing gradients...")
g_test = tic.compute_proxy_gradient(ref_plan_tensor, "likelihood")
total_norm_test = (sum(g.norm().item() ** 2 for g in g_test.values())) ** 0.5
h_inv_g = tic._ihvp_ekfac(g_test)

random.seed(0)
indices = random.sample(range(len(dataset)), N_EVAL)

scores_tif = []   # with iHVP
scores_dot = []   # plain dot product
scores_norm = []  # ||g_train|| only (baseline: pure norm ranker)
train_norms = []
return_per_sample = []

for i in indices:
    gt = tic.compute_training_gradient(i)
    norm = (sum(g.norm().item() ** 2 for g in gt.values())) ** 0.5
    train_norms.append(norm)
    s_tif = sum((h_inv_g[name] * gt[name]).sum().item()
                for name in g_test if name in gt)
    s_dot = sum((g_test[name] * gt[name]).sum().item()
                for name in g_test if name in gt)
    scores_tif.append(s_tif)
    scores_dot.append(s_dot)
    scores_norm.append(norm)
    # Also grab the trajectory return for cross-correlation
    seg = dataset[i]
    if "rewards" in seg:
        return_per_sample.append(float(seg["rewards"].sum()))
    else:
        return_per_sample.append(0.0)

scores_tif = np.array(scores_tif)
scores_dot = np.array(scores_dot)
scores_norm = np.array(scores_norm)
train_norms = np.array(train_norms)
returns = np.array(return_per_sample)

def report(name, s):
    print(f"\n[{name}]")
    print(f"  range:  [{s.min():+.3e}, {s.max():+.3e}]   std: {s.std():.3e}")
    sp_norm, _ = spearmanr(np.abs(s), train_norms)
    sp_ret, _ = spearmanr(s, returns)
    print(f"  Sp(|score|, ||g_train||) = {sp_norm:+.3f}   "
          f"(ideal: ~0; high = magnitude bias)")
    print(f"  Sp(score, return)        = {sp_ret:+.3f}   "
          f"(if signal: should be nonzero for likelihood proxy)")

report("TIF (iHVP)", scores_tif)
report("Plain dot product", scores_dot)
report("Pure norm baseline", scores_norm)

# Cross-method agreement
sp_tif_dot, _ = spearmanr(scores_tif, scores_dot)
sp_tif_norm, _ = spearmanr(scores_tif, scores_norm)
sp_dot_norm, _ = spearmanr(scores_dot, scores_norm)
print(f"\n=== AGREEMENT ===")
print(f"  Spearman(TIF, dot) = {sp_tif_dot:+.3f}")
print(f"  Spearman(TIF, norm) = {sp_tif_norm:+.3f}   "
      f"(if ~1.0, TIF reduces to norm ranker)")
print(f"  Spearman(dot, norm) = {sp_dot_norm:+.3f}")
