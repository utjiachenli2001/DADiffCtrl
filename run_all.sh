#!/bin/bash
# run_all.sh — Top-level script for TIF NeurIPS experiments.
#
# Runs the full experiment pipeline:
#   1. Main grid: 4 cells × 3 seeds, all experiments (LDS, safety, curation, intervention)
#   2. Ablation: EK-FAC vs K-FAC vs diagonal vs plain-dot on LDS
#   3. Aggregate: Collect results, compute mean±std, output LaTeX tables
#
# Usage:
#   bash run_all.sh                  # Full run
#   bash run_all.sh --smoke-test     # Quick validation (~15 min on 1 GPU)
#   bash run_all.sh --debug          # Debug-size models
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="/mnt/sdb/ljc/DADiffCtrl/analysis"

echo "=============================================="
echo "TIF Experiment Pipeline"
echo "=============================================="
echo "Script dir: $SCRIPT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Args: $*"
echo "Start time: $(date)"
echo "=============================================="

# Step 1: Main grid (4 cells × 3 seeds, all experiments)
echo ""
echo "[Step 1/3] Running main experiment grid..."
python run_grid.py --experiments all "$@"

# Step 2: Ablation (hessian approximation comparison on LDS)
echo ""
echo "[Step 2/3] Running hessian ablation..."
python run_ablation.py "$@"

# Step 3: Aggregate results
echo ""
echo "[Step 3/3] Aggregating results..."
python aggregate_results.py \
    --results-dir "$RESULTS_DIR" \
    --latex \
    --output "$RESULTS_DIR/aggregated.json"

echo ""
echo "=============================================="
echo "Pipeline complete."
echo "End time: $(date)"
echo "Aggregated results: $RESULTS_DIR/aggregated.json"
echo "=============================================="
