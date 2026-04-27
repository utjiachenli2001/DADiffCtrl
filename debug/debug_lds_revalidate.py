"""Re-evaluate LDS using NEW TIF scores against CACHED subset proxies.

The cached proxy_val for each subset is independent of attribution method,
so we can re-score TIF (with the relative-damping fix) and recompute
Spearman correlations cheaply — no retraining required.
"""
import json
import os
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
N_HESS = 1000  # smaller than full but enough for sanity
N_SCORE = 5000  # subsample training data for speed
CKPT = "/mnt/sdb/ljc/DADiffCtrl/checkpoints/diffuser_final.pt"
CACHE_DIR = "/mnt/sdb/ljc/DADiffCtrl/analysis/lds_cache/halfcheetah_medium_seed0"

# Try a few damping values to see effect
DAMPING_VALUES = [1e-4, 1e-3, 1e-2]

# --- Setup ---
diffuser_cfg = DiffuserConfig(device=device)
config = ExperimentConfig(
    env_name="halfcheetah", dataset="medium", experiment="all",
    diffuser=diffuser_cfg,
    influence=InfluenceConfig(device=device, hessian_approx="ekfac",
                               damping=1e-3, n_eigenvectors=100),
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
ref_plan = plans[0]
ref_plan_tensor = torch.from_numpy(
    (ref_plan - dataset.data_mean) / dataset.data_std
).float().to(device)

# Compute full-model proxy (negative loss on the plan, single MC sample for speed)
import torch.nn.functional as F
def likelihood_proxy(model, plan_arr):
    pt = torch.from_numpy(
        (plan_arr - dataset.data_mean) / dataset.data_std
    ).float().to(device).unsqueeze(0)
    losses = []
    torch.manual_seed(42)
    for _ in range(20):
        t = torch.randint(0, 200, (1,), device=device)
        noise = torch.randn_like(pt)
        x_noisy = model.q_sample(pt, t, noise)
        with torch.no_grad():
            pred = model.model(x_noisy, t)
        losses.append(F.mse_loss(pred, noise).item())
    return -np.mean(losses)  # negative loss = likelihood proxy

full_proxy = likelihood_proxy(diffusion, ref_plan)
print(f"Full proxy: {full_proxy:.4f}")

# Reproduce the same subset RNG
N_total = len(dataset)
subset_size = int(0.5 * N_total)  # subset_fraction = 0.5
rng = np.random.RandomState(0)  # base_seed = 0
all_subsets = [
    rng.choice(N_total, size=subset_size, replace=False) for _ in range(50)
]

# Load cached proxies
cached_proxies = []
for k in range(50):
    path = os.path.join(CACHE_DIR, f"subset_{k:04d}_seed0_steps50000.json")
    if not os.path.exists(path):
        cached_proxies.append(None)
    else:
        with open(path) as f:
            cached_proxies.append(json.load(f)["proxy_val"])

n_valid = sum(1 for p in cached_proxies if p is not None)
print(f"Cached subset proxies: {n_valid}/50 available")

actual_deltas = np.array([p - full_proxy if p is not None else np.nan
                          for p in cached_proxies])
print(f"Actual delta stats: mean={np.nanmean(actual_deltas):+.4f} "
      f"std={np.nanstd(actual_deltas):.4f}")

# Compute Hessian
print(f"\nComputing Hessian over {N_HESS} samples...")
tic = TrajectoryInfluenceComputer(model=diffusion, dataset=dataset,
                                   config=config.influence, diffusion_steps=200)
tic.compute_hessian_approximation(n_samples=N_HESS)

# Sweep damping values + score subsample
print(f"\nScoring {N_SCORE} training samples per damping value...")
random.seed(123)
score_indices = random.sample(range(N_total), N_SCORE)

g_test = tic.compute_proxy_gradient(ref_plan_tensor, "likelihood")

# Pre-compute iHVP for each damping value (small, fits in memory)
h_inv_dict = {}
for damping in DAMPING_VALUES:
    tic.config.damping = damping
    h_inv_dict[damping] = {n: v.clone() for n, v in tic._ihvp_ekfac(g_test).items()}

returns = np.array([float(dataset[i].get("rewards", torch.tensor([0.0])).sum())
                    if "rewards" in dataset[i] else 0.0
                    for i in score_indices])

# Stream training gradients: compute, score for all dampings, free
print(f"Streaming {N_SCORE} training gradient projections...")
scores_per_damping = {d: np.zeros(N_SCORE) for d in DAMPING_VALUES}
for j, i in enumerate(score_indices):
    gt = tic.compute_training_gradient(i)
    for damping in DAMPING_VALUES:
        h_inv_g = h_inv_dict[damping]
        scores_per_damping[damping][j] = sum(
            (h_inv_g[name] * gt[name]).sum().item()
            for name in g_test if name in gt
        )
    del gt
    if (j + 1) % 500 == 0:
        torch.cuda.empty_cache()

print(f"\n{'damping':>10s} | {'Sp(LDS)':>10s} | {'Pearson(LDS)':>13s} | "
      f"{'Sp(score, return)':>18s}")
print("-" * 70)

for damping in DAMPING_VALUES:
    scores_sub = scores_per_damping[damping]

    # Build full-N score vector (zero where not scored — biases the sum slightly)
    scores_full = np.zeros(N_total, dtype=np.float64)
    for idx, s in zip(score_indices, scores_sub):
        scores_full[idx] = s

    # Predicted deltas: -sum(removed)/N for each subset
    predicted = []
    actual = []
    for k, sub in enumerate(all_subsets):
        if cached_proxies[k] is None:
            continue
        mask_kept = np.zeros(N_total, dtype=bool)
        mask_kept[sub] = True
        removed = ~mask_kept
        pred_d = -scores_full[removed].sum() / N_total
        predicted.append(pred_d)
        actual.append(actual_deltas[k])

    actual = np.array(actual)
    predicted = np.array(predicted)
    sp, _ = spearmanr(actual, predicted)
    pe = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0.0
    sp_ret, _ = spearmanr(scores_sub, returns)
    print(f"{damping:>10.0e} | {sp:>+10.3f} | {pe:>+13.3f} | {sp_ret:>+18.3f}")

print("\n(Note: scores use a 5k-sample subset of training data; full-N would tighten estimates)")
