# DADiffCtrl Reproduction — Seed 0 Grid Results

_Generated: 2026-04-30 14:00:46 on `dcwipphrtx0004` (8× RTX PRO 6000 Blackwell)_

## Run configuration

- **Repo**: [utjiachenli2001/DADiffCtrl](https://github.com/utjiachenli2001/DADiffCtrl) @ commit `9eb93ce`
- **Branch**: `zt_0501`
- **Command** (from README, adapted to 8 GPUs and 1 seed):
  ```bash
  python run_grid.py --experiments all --mode parallel \
      --n-workers 12 --gpu-ids 0 1 2 3 4 5 6 7 --seeds 0
  ```
- **Hessian approx**: EK-FAC (default)
- **Grid scope**: 9 cells (3 envs × 3 datasets) × **1 seed** (seed 0 only)
- **Wall clock**: 2026-04-28 00:27 → 2026-04-30 06:04 EDT (≈53.6 h, all 9 cells finished, 0 failures)

> The README's full reproduction calls for 3 seeds × 12 GPUs (~6 h). We ran a single-seed pass on 8 GPUs; per-cell numbers below are point estimates rather than mean ± std across seeds.

## 1. LDS — Linear Datamodeling Score (Spearman ρ ↑)

Higher is better. Measures how well the influence-score-based predicted Δ-loss correlates with actual Δ-loss across 50 LDS subsets.

| Env / Dataset | TIF (ours) | DiagonalIF | RewardRanking | NearestNeighbor | TRAK (1-ckpt) | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `halfcheetah` / `medium` | -0.1733 | +0.1615 | +0.0300 | +0.0431 | +0.0196 | +0.0364 |
| `halfcheetah` / `medium-replay` | +0.0671 | -0.1045 | +0.0629 | -0.0250 | +0.1000 | -0.1345 |
| `halfcheetah` / `medium-expert` | +0.0391 | -0.0061 | +0.2542 | -0.0544 | -0.1111 | +0.1582 |
| `hopper` / `medium` | +0.2406 | +0.2512 | +0.1086 | -0.0776 | -0.0350 | -0.0489 |
| `hopper` / `medium-replay` | +0.1230 | +0.2356 | -0.2282 | +0.1253 | +0.1310 | +0.0645 |
| `hopper` / `medium-expert` | -0.0756 | +0.0599 | -0.0147 | -0.0124 | -0.0659 | -0.1782 |
| `walker2d` / `medium` | +0.1996 | -0.1452 | +0.0429 | -0.0844 | +0.1002 | +0.0081 |
| `walker2d` / `medium-replay` | -0.0356 | +0.3995 | +0.0556 | -0.0902 | -0.0707 | -0.1153 |
| `walker2d` / `medium-expert` | -0.0449 | -0.3658 | -0.0583 | +0.0899 | -0.0576 | -0.1166 |

## 2. Safety AUC (↑)

ROC-AUC of method-assigned scores against ground-truth unsafe-trajectory labels (constraint violations).

| Env / Dataset | TIF-safety (ours) | DiagonalIF | RewardRanking | NearestNeighbor | TRAK (1-ckpt) | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `halfcheetah` / `medium` | 0.4890 | 0.4680 | 0.2636 | 0.1769 | 0.4840 | 0.5053 |
| `halfcheetah` / `medium-replay` | 0.4994 | 0.4954 | 0.2295 | 0.2324 | 0.4999 | 0.4897 |
| `halfcheetah` / `medium-expert` | 0.4841 | 0.5578 | 0.1458 | 0.3987 | 0.5008 | 0.4991 |
| `hopper` / `medium` | 0.5540 | 0.4706 | 0.7941 | 0.3881 | 0.5253 | 0.5053 |
| `hopper` / `medium-replay` | 0.4580 | 0.4453 | 0.6307 | 0.0911 | 0.4838 | 0.4889 |
| `hopper` / `medium-expert` | 0.5004 | 0.4337 | 0.7581 | 0.2610 | 0.4737 | 0.5010 |
| `walker2d` / `medium` | 0.4999 | 0.4876 | 0.1212 | 0.0398 | 0.4861 | 0.5056 |
| `walker2d` / `medium-replay` | 0.4687 | 0.4794 | 0.2476 | 0.0192 | 0.5287 | 0.4852 |
| `walker2d` / `medium-expert` | 0.5162 | 0.4968 | 0.1209 | 0.0669 | 0.5130 | 0.4983 |

## 3. Curation — Δ proxy-loss after pruning by influence (averaged over fractions)

Per cell we average `remove_harmful.delta` (proxy after removing fraction f of most-harmful − full-data proxy) across all available prune fractions {0.1, 0.2, 0.3, 0.5}. Higher (less negative) = pruning the most-harmful samples preserves proxy quality better.

| Env / Dataset | TIF (ours) | DiagonalIF | RewardRanking | NearestNeighbor | TRAK (1-ckpt) | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `halfcheetah` / `medium` | +0.0285 | +0.0212 | +0.0155 | +0.0097 | +0.0301 | +0.0349 |
| `halfcheetah` / `medium-replay` | +0.0642 | +0.0091 | +0.0049 | +0.0084 | +0.0300 | +0.0386 |
| `halfcheetah` / `medium-expert` | -0.0480 | -0.0402 | -0.0542 | -0.0674 | -0.0720 | -0.0426 |
| `hopper` / `medium` | +0.0200 | +0.0260 | +0.0185 | +0.0281 | +0.0009 | +0.0234 |
| `hopper` / `medium-replay` | +0.0055 | +0.0117 | +0.0014 | +0.0165 | +0.0007 | -0.0017 |
| `hopper` / `medium-expert` | +0.0142 | +0.0112 | +0.0047 | +0.0066 | -0.0072 | +0.0050 |
| `walker2d` / `medium` | +0.0131 | +0.0537 | +0.0388 | +0.0377 | +0.0370 | +0.0629 |
| `walker2d` / `medium-replay` | +0.0096 | -0.0041 | -0.0020 | +0.0358 | +0.0314 | -0.0075 |
| `walker2d` / `medium-expert` | +0.0259 | +0.0176 | +0.0246 | +0.0489 | +0.0216 | +0.0299 |

## 4. Intervention — policy return after data removal

Baseline (no removal) return is reported per cell. The `intervention` block is fully populated only for `NearestNeighbor` in this run.

| Env / Dataset | Baseline mean return | Baseline violation rate |
|---|---:|---:|
| `halfcheetah` / `medium` | -120.84 ± 228.90 | 1.00 |
| `halfcheetah` / `medium-replay` | -115.57 ± 279.36 | 1.00 |
| `halfcheetah` / `medium-expert` | -154.82 ± 135.18 | 1.00 |
| `hopper` / `medium` | +49.64 ± 51.88 | 1.00 |
| `hopper` / `medium-replay` | +49.41 ± 80.47 | 0.40 |
| `hopper` / `medium-expert` | +27.47 ± 26.83 | 0.90 |
| `walker2d` / `medium` | -3.48 ± 15.78 | 1.00 |
| `walker2d` / `medium-replay` | +42.30 ± 72.36 | 1.00 |
| `walker2d` / `medium-expert` | -4.96 ± 21.68 | 1.00 |

## 5. Files in this branch

- `analysis/*_seed0_all_ekfac_*.json` — 9 per-cell raw result JSONs from `run_experiments.py`
- `analysis/lds_cache/<cell>/subset_*_seed0_steps50000.json` — 50 LDS subsets per cell (cached counterfactual evaluations)

## 6. Status of remaining pipeline steps

- [x] `run_grid.py --experiments all --seeds 0` (this commit)
- [ ] `run_ablation.py --seeds 0` — running on `dcwipphrtx0004` at time of push, 8-way GPU shard
- [ ] `aggregate_results.py --latex` — to be run after ablation completes

