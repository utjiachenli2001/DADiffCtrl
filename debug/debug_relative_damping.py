"""Test relative per-layer damping: δ_l = α · λ_max(l).

This is the standard regularization used in modern influence-function pipelines
(Mlodozeniec et al. 2025): damping scales with each layer's top eigenvalue
rather than being a fixed absolute value. This preserves directional structure
in well-conditioned subspaces while damping out noisy near-zero components.
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
N_HESS = 1000
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
).float().to(device)

print(f"Computing Hessian over {N_HESS} samples...")
tic = TrajectoryInfluenceComputer(model=diffusion, dataset=dataset,
                                   config=config.influence, diffusion_steps=200)
tic.compute_hessian_approximation(n_samples=N_HESS)


# --- Relative-damping iHVP (replaces _ihvp_ekfac for this test) ---
def ihvp_with_relative_damping(g_test_dict, alpha):
    """iHVP using per-layer relative damping δ_l = α·λ_max(l)."""
    import torch.nn as nn
    result = {}
    for name, param in tic.model.named_parameters():
        if name not in g_test_dict:
            continue
        grad = g_test_dict[name]
        layer_name = tic._param_to_layer_name(name)
        lf = tic.ekfac_state.layers.get(layer_name)
        if lf is None or lf.Q_A is None or lf.Q_G is None:
            result[name] = grad / (alpha + 1e-12)
            continue
        if grad.dim() == 1:  # bias
            result[name] = grad / (alpha + 1e-12)
            continue
        if grad.dim() == 3:
            layer_mod = tic._target_layers.get(layer_name)
            if isinstance(layer_mod, nn.ConvTranspose1d):
                # skip for simplicity
                result[name] = grad / (alpha + 1e-12)
                continue
            d_out = grad.shape[0]
            V = grad.reshape(d_out, -1)
        else:
            V = grad

        Q_A, Q_G = lf.Q_A, lf.Q_G
        if lf.Lambda_hat is not None:
            eig = lf.Lambda_hat
        else:
            eig = lf.Lambda_G[:, None] * lf.Lambda_A[None, :]
        # PER-LAYER relative damping
        delta = alpha * eig.max().clamp(min=1e-30)
        try:
            projected = Q_G.T @ V @ Q_A
            projected = projected / (eig + delta)
            V_inv = Q_G @ projected @ Q_A.T
            result[name] = V_inv.reshape(grad.shape)
        except RuntimeError:
            result[name] = grad / (alpha + 1e-12)
    return result


print("Computing gradients...")
g_test = tic.compute_proxy_gradient(ref_plan_tensor, "likelihood")

random.seed(0)
indices = random.sample(range(len(dataset)), N_EVAL)
g_trains = []
train_norms = []
returns = []
for i in indices:
    gt = tic.compute_training_gradient(i)
    g_trains.append(gt)
    train_norms.append((sum(g.norm().item() ** 2 for g in gt.values())) ** 0.5)
    seg = dataset[i]
    returns.append(float(seg.get("rewards", torch.tensor([0.0])).sum())
                   if "rewards" in seg else 0.0)
train_norms = np.array(train_norms)
returns = np.array(returns)


def score_with_iHVP(h_inv_g):
    scores = []
    for gt in g_trains:
        s = sum((h_inv_g[name] * gt[name]).sum().item()
                for name in g_test if name in gt)
        scores.append(s)
    return np.array(scores)


print(f"\n{'method':>30s} | {'Sp(score, return)':>18s} | {'Sp(|score|, ||g||)':>20s}")
print("-" * 76)
# Baseline: original (absolute damping 1e-2)
h_orig = tic._ihvp_ekfac(g_test)
s_orig = score_with_iHVP(h_orig)
sp_ret, _ = spearmanr(s_orig, returns)
sp_norm, _ = spearmanr(np.abs(s_orig), train_norms)
print(f"{'EK-FAC, abs damp 1e-2':>30s} | {sp_ret:>+18.3f} | {sp_norm:>+20.3f}")

# Plain dot product baseline
s_dot = []
total_norm_test = (sum(g.norm().item()**2 for g in g_test.values()))**0.5
for gt in g_trains:
    s = sum((g_test[name] * gt[name]).sum().item()
            for name in g_test if name in gt)
    s_dot.append(s)
s_dot = np.array(s_dot)
sp_ret, _ = spearmanr(s_dot, returns)
sp_norm, _ = spearmanr(np.abs(s_dot), train_norms)
print(f"{'Plain dot product':>30s} | {sp_ret:>+18.3f} | {sp_norm:>+20.3f}")

# Sweep alpha
for alpha in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    h = ihvp_with_relative_damping(g_test, alpha)
    s = score_with_iHVP(h)
    sp_ret, _ = spearmanr(s, returns)
    sp_norm, _ = spearmanr(np.abs(s), train_norms)
    sp_dot, _ = spearmanr(s, s_dot)
    print(f"{f'rel-damp α={alpha:.0e}':>30s} | {sp_ret:>+18.3f} | "
          f"{sp_norm:>+20.3f}   (Sp(score,dot)={sp_dot:+.3f})")
