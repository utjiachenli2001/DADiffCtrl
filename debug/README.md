# Debug and Validation Scripts

This directory contains scripts used to validate and debug the TIF implementation during development.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `debug_self_influence.py` | Validate self-influence scores (diagonal of influence matrix should be positive) | `python debug/debug_self_influence.py` |
| `debug_compare_methods.py` | Compare TIF vs baselines on a single plan | `python debug/debug_compare_methods.py` |
| `debug_damping_sweep.py` | Sweep over damping values to find optimal regularization | `python debug/debug_damping_sweep.py` |
| `debug_relative_damping.py` | Test per-layer relative damping (alpha * lambda_max) | `python debug/debug_relative_damping.py` |
| `debug_m_sweep.py` | Sweep over number of eigenvectors (m) for EK-FAC | `python debug/debug_m_sweep.py` |
| `debug_lds_revalidate.py` | Re-run LDS with cached subsets to verify reproducibility | `python debug/debug_lds_revalidate.py` |
| `debug_influence.py` | Basic influence computation sanity checks | `python debug/debug_influence.py` |

## Key Validation Results

### Self-Influence Test

The self-influence of a training point (how much removing it affects its own loss) should be positive. This validates that the gradient/Hessian math is correct:

```
I(z_i, z_i) = g_i^T H^{-1} g_i > 0  (for all i)
```

Results are saved to `analysis/self_influence_results.json`.

### Damping Sweep

Tests damping values from 1e-6 to 1e-1. Too low causes numerical instability (NaN), too high over-regularizes (flat scores). Optimal is typically 1e-4 to 1e-3.

### Relative Damping

Tests the per-layer relative damping formula:
```
delta_l = alpha * lambda_max(A_l otimes G_l)
```

This adapts regularization to each layer's curvature scale, improving conditioning without over-damping small layers.
