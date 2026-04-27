"""Sweep MC samples M for proxy gradient + inspect eigenvalue spectrum.

Hypothesis: if proxy gradient is dominated by MC noise, increasing M
should produce a sharper, more meaningful gradient direction.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
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


# Custom proxy gradient with arbitrary M
def compute_proxy_gradient_M(plan_tau, M):
    """Likelihood proxy gradient with M MC samples."""
    if plan_tau.dim() == 2:
        plan_tau = plan_tau.unsqueeze(0)
    diffusion.eval()
    for p in diffusion.parameters():
        p.requires_grad_(True)
    loss = torch.tensor(0.0, device=device)
    for _ in range(M):
        t = torch.randint(0, 200, (1,), device=device)
        noise = torch.randn_like(plan_tau)
        x_noisy = diffusion.q_sample(plan_tau, t, noise)
        pred = diffusion.model(x_noisy, t)
        loss = loss + F.mse_loss(pred, noise)
    loss = loss / M
    diffusion.zero_grad()
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in diffusion.named_parameters() if p.grad is not None}


# Pre-compute training gradients ONCE
print("Computing training gradients...")
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


# Sweep M
print(f"\n{'M':>6s} | {'||g_test||':>12s} | "
      f"{'Sp(TIF,return)':>16s} | {'Sp(|TIF|,||g||)':>17s} | "
      f"{'Sp(M,M_prev)':>14s}")
print("-" * 80)

prev_scores = None
for M in [10, 100, 500, 2000]:
    # Fresh seed for fair comparison
    torch.manual_seed(0)
    g_test = compute_proxy_gradient_M(ref_plan_tensor, M)
    norm_test = (sum(g.norm().item() ** 2 for g in g_test.values())) ** 0.5
    h_inv_g = tic._ihvp_ekfac(g_test)

    scores = []
    for gt in g_trains:
        s = sum((h_inv_g[name] * gt[name]).sum().item()
                for name in g_test if name in gt)
        scores.append(s)
    scores = np.array(scores)

    sp_ret, _ = spearmanr(scores, returns)
    sp_norm, _ = spearmanr(np.abs(scores), train_norms)
    sp_prev = "—"
    if prev_scores is not None:
        sp_prev_v, _ = spearmanr(scores, prev_scores)
        sp_prev = f"{sp_prev_v:+.3f}"
    print(f"{M:>6d} | {norm_test:>12.3e} | {sp_ret:>+16.3f} | "
          f"{sp_norm:>+17.3f} | {sp_prev:>14s}")
    prev_scores = scores

# Eigenvalue spectrum inspection
print("\n=== EIGENVALUE SPECTRUM (a few representative layers) ===")
shown = 0
for name, lf in tic.ekfac_state.layers.items():
    if lf.Lambda_hat is None or shown >= 6:
        continue
    L = lf.Lambda_hat.flatten().cpu().numpy()
    L_sorted = np.sort(L)[::-1]
    ratio = L_sorted[0] / max(L_sorted[-1], 1e-30)
    median = L_sorted[len(L_sorted) // 2]
    print(f"  {name[:50]:50s} | shape={tuple(lf.Lambda_hat.shape)} | "
          f"max={L_sorted[0]:.3e} | median={median:.3e} | "
          f"min={L_sorted[-1]:.3e} | max/min={ratio:.1e}")
    shown += 1
