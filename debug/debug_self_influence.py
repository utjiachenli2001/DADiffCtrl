"""Self-influence sanity check.

For each query trajectory τ_k drawn from the training set, the influence
ranking I(τ_i, τ_k) should place τ_k itself at or near the top — that
trajectory is by construction maximally influential on itself.

This is the standard sanity test for IF pipelines. If TIF can't pass it,
no downstream metric will work; if it does pass it, we have evidence the
pipeline produces meaningful rankings (independent of LDS noise).

We measure:
  - rank of τ_k in I(*, τ_k), normalized to [0, 1] (lower = better)
  - top-1% recall: fraction of queries where τ_k is in the top 1% of scores
  - same for plain dot product (no Hessian) and pure norm baseline

If TIF beats plain-dot meaningfully on this test, EK-FAC is doing useful work.
If both succeed, the pipeline works (IF in general is valid here).
If both fail, the proxy gradient itself doesn't carry enough info.
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
N_QUERIES = 100       # how many query trajectories
N_CANDIDATES = 5000   # candidate pool size (subset of training data) for ranking
CKPT = "/mnt/sdb/ljc/DADiffCtrl/checkpoints/diffuser_final.pt"

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

print(f"Computing Hessian over {N_HESS} samples...")
tic = TrajectoryInfluenceComputer(model=diffusion, dataset=dataset,
                                   config=config.influence, diffusion_steps=200)
tic.compute_hessian_approximation(n_samples=N_HESS)

# Pick candidates and queries
random.seed(0)
N_total = len(dataset)
candidate_idx = random.sample(range(N_total), N_CANDIDATES)
query_idx = random.sample(candidate_idx, N_QUERIES)
candidate_pos = {idx: pos for pos, idx in enumerate(candidate_idx)}

# Pre-compute candidate training gradient *projections* — store on CPU as
# small flat tensors via projection through Q matrices is heavy; instead,
# we cache the raw gradient sums into per-layer matrices on CPU.
print(f"Pre-computing {N_CANDIDATES} candidate training gradients (streamed)...")
# Strategy: stream queries, compute per-query iHVP, then stream candidates,
# scoring each. Flush memory between.

results = {
    "TIF": [],
    "Plain dot": [],
    "Pure norm": [],
}

# Get returns for sanity correlation
returns = np.array([
    float(dataset[i].get("rewards", torch.tensor([0.0])).sum())
    if "rewards" in dataset[i] else 0.0
    for i in candidate_idx
])

print(f"\nRunning {N_QUERIES} queries × {N_CANDIDATES} candidates...")
for q_n, q_idx in enumerate(query_idx):
    q_position = candidate_pos[q_idx]

    # Compute proxy gradient on this query trajectory
    seg = dataset[q_idx]
    plan_tensor = seg["trajectories"].unsqueeze(0).to(device).float()
    g_test = tic.compute_proxy_gradient(plan_tensor, "likelihood")
    h_inv_g = tic._ihvp_ekfac(g_test)
    norm_test = (sum(g.norm().item() ** 2 for g in g_test.values())) ** 0.5

    # Score all candidates
    scores_tif = np.zeros(N_CANDIDATES)
    scores_dot = np.zeros(N_CANDIDATES)
    scores_norm = np.zeros(N_CANDIDATES)
    for j, c_idx in enumerate(candidate_idx):
        gt = tic.compute_training_gradient(c_idx)
        scores_tif[j] = sum((h_inv_g[name] * gt[name]).sum().item()
                            for name in g_test if name in gt)
        scores_dot[j] = sum((g_test[name] * gt[name]).sum().item()
                            for name in g_test if name in gt)
        scores_norm[j] = (sum(g.norm().item() ** 2 for g in gt.values())) ** 0.5
        del gt
        if (j + 1) % 1000 == 0:
            torch.cuda.empty_cache()

    # Rank of self (lower = better; 0 = top, N-1 = bottom)
    def rank_of_self(scores, q_pos):
        # We want HIGH scores to mean "more influential"; rank by descending
        order = np.argsort(-scores)
        return int(np.where(order == q_pos)[0][0])

    rank_tif = rank_of_self(scores_tif, q_position)
    rank_dot = rank_of_self(scores_dot, q_position)
    rank_norm = rank_of_self(scores_norm, q_position)
    results["TIF"].append(rank_tif)
    results["Plain dot"].append(rank_dot)
    results["Pure norm"].append(rank_norm)

    if q_n < 5 or (q_n + 1) % 10 == 0:
        print(f"  query {q_n+1:3d}/{N_QUERIES}  "
              f"rank_tif={rank_tif:5d}  "
              f"rank_dot={rank_dot:5d}  "
              f"rank_norm={rank_norm:5d}  "
              f"(of {N_CANDIDATES})")
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(f"Self-Influence Sanity Test ({N_QUERIES} queries, {N_CANDIDATES} candidates)")
print("=" * 70)
print(f"{'Method':>15s} | {'median rank':>12s} | {'mean rank %ile':>15s} | "
      f"{'top-1% recall':>13s} | {'top-10% recall':>14s}")
print("-" * 80)
summary = {}
for name in ["TIF", "Plain dot", "Pure norm"]:
    ranks = np.array(results[name])
    median_rank = int(np.median(ranks))
    mean_pctile = float(ranks.mean() / N_CANDIDATES * 100)
    recall_1 = float((ranks < N_CANDIDATES * 0.01).mean())
    recall_10 = float((ranks < N_CANDIDATES * 0.10).mean())
    print(f"{name:>15s} | {median_rank:>12d} | {mean_pctile:>14.2f}% | "
          f"{recall_1:>13.2%} | {recall_10:>14.2%}")
    summary[name] = {
        "ranks": ranks.tolist(),
        "median_rank": median_rank,
        "mean_percentile": mean_pctile,
        "recall_at_1pct": recall_1,
        "recall_at_10pct": recall_10,
    }
print("\nIdeal: median rank ≈ 0, mean %ile ≈ 0%, recall ≈ 100%.")
print("Random baseline: median rank ≈ N/2, mean %ile ≈ 50%, "
      "recall@1% ≈ 1%, recall@10% ≈ 10%.")

import json as _json
out = {
    "n_queries": N_QUERIES,
    "n_candidates": N_CANDIDATES,
    "n_hess": N_HESS,
    "summary": summary,
}
with open("/mnt/sdb/ljc/DADiffCtrl/analysis/self_influence_results.json", "w") as f:
    _json.dump(out, f, indent=2)
print("\nSaved full results to analysis/self_influence_results.json")
