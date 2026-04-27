#!/usr/bin/env python3
"""
Aggregate experiment results across seeds and environments.

Collects all result JSON files from the results directory, parses
structured filenames, groups by (env, dataset, experiment), computes
mean +/- std across seeds, and outputs summary tables.

Supports --latex for paper-ready LaTeX table fragments.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir /path/to/results
    python aggregate_results.py --latex --output analysis/aggregated.json
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aggregate_results")

RESULTS_DIR = os.environ.get("DADIFFCTRL_RESULTS_DIR", "./analysis")

# ── Filename parsing ────────────────────────────────────────────────

# Structured name: {env}_{dataset}_seed{seed}_{experiment}_{hessian}_{timestamp}
# Example: halfcheetah_medium_seed0_all_ekfac_20260424_120000.json
# Ablation: ablation_hessian_{timestamp}.json
STRUCTURED_RE = re.compile(
    r"^(?P<env>[a-z0-9]+)_(?P<dataset>[a-z\-]+)_seed(?P<seed>\d+)"
    r"_(?P<experiment>[a-z]+)_(?P<hessian>[a-z_]+)"
    r"_\d{8}_\d{6}\.json$"
)
ABLATION_RE = re.compile(r"^ablation_hessian_\d{8}_\d{6}\.json$")


def parse_filename(fname: str) -> Optional[Dict]:
    """Parse a structured result filename into metadata dict."""
    m = STRUCTURED_RE.match(fname)
    if m:
        return {
            "env": m.group("env"),
            "dataset": m.group("dataset"),
            "seed": int(m.group("seed")),
            "experiment": m.group("experiment"),
            "hessian": m.group("hessian"),
            "type": "experiment",
        }
    if ABLATION_RE.match(fname):
        return {"type": "ablation"}
    return None


# ── Metric extraction ──────────────────────────────────────────────

def extract_metrics(data: Dict, experiment: str) -> Dict[str, Dict[str, float]]:
    """Extract per-method metrics from a result JSON.

    Returns:
        {method_name: {metric_name: value, ...}, ...}
    """
    metrics: Dict[str, Dict[str, float]] = {}

    if experiment in ("all", "lds"):
        # LDS results: data["lds"]["methods"][method]["lds_spearman"]
        lds = data.get("lds", data)
        methods = lds.get("methods", {})
        for method, vals in methods.items():
            metrics.setdefault(method, {})
            if "lds_spearman" in vals:
                metrics[method]["lds_spearman"] = vals["lds_spearman"]
            if "lds_pearson" in vals:
                metrics[method]["lds_pearson"] = vals["lds_pearson"]

    if experiment in ("all", "safety"):
        safety = data.get("safety", data)
        methods = safety.get("methods", {})
        for method, vals in methods.items():
            metrics.setdefault(method, {})
            if "auc" in vals:
                metrics[method]["safety_auc"] = vals["auc"]
            elif "safety_auc" in vals:
                metrics[method]["safety_auc"] = vals["safety_auc"]

    if experiment in ("all", "curation"):
        curation = data.get("curation", data)
        methods = curation.get("methods", {})
        for method, vals in methods.items():
            metrics.setdefault(method, {})
            if "curation_delta" in vals:
                metrics[method]["curation_delta"] = vals["curation_delta"]
            elif isinstance(vals, dict):
                # Nested by fraction
                for frac_key, frac_vals in vals.items():
                    if isinstance(frac_vals, dict) and "proxy_delta" in frac_vals:
                        metrics[method][f"curation_{frac_key}"] = frac_vals["proxy_delta"]

    if experiment in ("all", "intervention"):
        interv = data.get("intervention", data)
        methods = interv.get("methods", {})
        for method, vals in methods.items():
            metrics.setdefault(method, {})
            if isinstance(vals, dict):
                for frac_key, frac_vals in vals.items():
                    if isinstance(frac_vals, dict):
                        for k, v in frac_vals.items():
                            if isinstance(v, (int, float)):
                                metrics[method][f"interv_{frac_key}_{k}"] = v

    return metrics


# ── Aggregation ─────────────────────────────────────────────────────

def aggregate_across_seeds(
    grouped: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """Compute mean +/- std for each metric across seeds.

    Args:
        grouped: {group_key: [metrics_dict_per_seed, ...]}

    Returns:
        {group_key: {method: {metric: {"mean": float, "std": float, "n": int}}}}
    """
    aggregated = {}
    for key, seed_metrics_list in grouped.items():
        agg: Dict[str, Dict[str, Dict]] = {}
        # Collect all (method, metric) values
        method_metric_vals: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for seed_metrics in seed_metrics_list:
            for method, mdict in seed_metrics.items():
                for metric, val in mdict.items():
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        method_metric_vals[method][metric].append(val)

        for method, mdict in method_metric_vals.items():
            agg[method] = {}
            for metric, vals in mdict.items():
                arr = np.array(vals)
                agg[method][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n": len(arr),
                }
        aggregated[key] = agg
    return aggregated


# ── Display ─────────────────────────────────────────────────────────

def print_tables(aggregated: Dict, latex: bool = False):
    """Print summary tables to stdout."""
    if not aggregated:
        print("No results to display.")
        return

    # Group by experiment type
    exp_groups: Dict[str, Dict] = defaultdict(dict)
    for key, methods in aggregated.items():
        # key format: env_dataset_experiment
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            cell, exp = parts
        else:
            cell = key
            exp = "all"
        exp_groups[exp][cell] = methods

    for exp, cells in sorted(exp_groups.items()):
        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp}")
        print(f"{'=' * 70}")

        # Gather all methods and metrics
        all_methods = set()
        all_metrics = set()
        for cell_methods in cells.values():
            for method, mdict in cell_methods.items():
                all_methods.add(method)
                all_metrics.update(mdict.keys())

        all_methods = sorted(all_methods)
        all_metrics = sorted(all_metrics)
        cell_names = sorted(cells.keys())

        for metric in all_metrics:
            print(f"\n  Metric: {metric}")
            if latex:
                _print_latex_table(cells, cell_names, all_methods, metric)
            else:
                _print_text_table(cells, cell_names, all_methods, metric)


def _print_text_table(
    cells: Dict, cell_names: List[str],
    methods: List[str], metric: str,
):
    """Print a text-formatted table for one metric."""
    col_w = max(22, max(len(c) for c in cell_names) + 2)
    header = f"  {'Method':<20}"
    for c in cell_names:
        header += f"  {c:<{col_w}}"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for method in methods:
        row = f"  {method:<20}"
        for cell in cell_names:
            val = cells.get(cell, {}).get(method, {}).get(metric)
            if val is not None:
                m, s, n = val["mean"], val["std"], val["n"]
                if n > 1:
                    row += f"  {m:+.4f}±{s:.4f} (n={n})"
                    row += " " * max(0, col_w - len(f"{m:+.4f}±{s:.4f} (n={n})"))
                else:
                    row += f"  {m:+.4f}           "
                    row += " " * max(0, col_w - len(f"{m:+.4f}           "))
            else:
                row += f"  {'--':<{col_w}}"
        print(row)


def _print_latex_table(
    cells: Dict, cell_names: List[str],
    methods: List[str], metric: str,
):
    """Print a LaTeX table fragment for one metric."""
    n_cols = len(cell_names)
    # Clean cell names for LaTeX
    clean_names = [c.replace("_", r"\_") for c in cell_names]

    print(f"  % LaTeX table for metric: {metric}")
    print(f"  \\begin{{tabular}}{{l{'c' * n_cols}}}")
    print(f"  \\toprule")
    print(f"  Method & {' & '.join(clean_names)} \\\\")
    print(f"  \\midrule")

    # Find best per column for bolding
    best_per_col: Dict[str, float] = {}
    for cell in cell_names:
        best = -float("inf")
        for method in methods:
            val = cells.get(cell, {}).get(method, {}).get(metric)
            if val is not None and val["mean"] > best:
                best = val["mean"]
        best_per_col[cell] = best

    for method in methods:
        clean_method = method.replace("_", r"\_")
        parts = [clean_method]
        for cell in cell_names:
            val = cells.get(cell, {}).get(method, {}).get(metric)
            if val is not None:
                m, s, n = val["mean"], val["std"], val["n"]
                if n > 1:
                    entry = f"${m:.3f}_{{\\pm {s:.3f}}}$"
                else:
                    entry = f"${m:.3f}$"
                # Bold the best
                if abs(m - best_per_col.get(cell, float("inf"))) < 1e-8:
                    entry = f"\\textbf{{{entry}}}"
                parts.append(entry)
            else:
                parts.append("--")
        print(f"  {' & '.join(parts)} \\\\")

    print(f"  \\bottomrule")
    print(f"  \\end{{tabular}}")


def print_ablation_table(ablation_data: Dict, latex: bool = False):
    """Print ablation table from ablation result files."""
    print(f"\n{'=' * 70}")
    print("Ablation: Hessian Approximation (LDS Spearman)")
    print(f"{'=' * 70}")

    modes = ["ekfac", "kfac", "diagonal", "plain_dot"]

    # Collect across cells
    cell_keys = sorted(ablation_data.keys())

    if latex:
        n_cols = len(cell_keys)
        clean_keys = [k.replace("_", r"\_") for k in cell_keys]
        print(f"\n  \\begin{{tabular}}{{l{'c' * n_cols}}}")
        print(f"  \\toprule")
        print(f"  Approx. & {' & '.join(clean_keys)} \\\\")
        print(f"  \\midrule")

        for mode in modes:
            clean_mode = mode.replace("_", r"\_")
            parts = [clean_mode]
            for k in cell_keys:
                val = ablation_data.get(k, {}).get(mode, {}).get("lds_spearman")
                if val is not None:
                    parts.append(f"${val:.3f}$")
                else:
                    parts.append("--")
            print(f"  {' & '.join(parts)} \\\\")

        print(f"  \\bottomrule")
        print(f"  \\end{{tabular}}")
    else:
        col_w = max(20, max((len(k) for k in cell_keys), default=20) + 2)
        header = f"  {'Mode':<12}"
        for k in cell_keys:
            header += f"  {k:<{col_w}}"
        print(header)
        print(f"  {'-' * len(header)}")

        for mode in modes:
            row = f"  {mode:<12}"
            for k in cell_keys:
                val = ablation_data.get(k, {}).get(mode, {}).get("lds_spearman")
                if val is not None:
                    t_str = f"{val:+.4f}"
                    elapsed = ablation_data.get(k, {}).get(mode, {}).get("compute_time_s")
                    if elapsed is not None:
                        t_str += f" ({elapsed:.0f}s)"
                    row += f"  {t_str:<{col_w}}"
                else:
                    row += f"  {'--':<{col_w}}"
            print(row)


# ── Main ────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Aggregate TIF experiment results")
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR,
        help="Directory containing result JSON files.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save aggregated JSON (default: results_dir/aggregated.json).",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Output LaTeX table fragments.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        logger.error("Results directory does not exist: %s", results_dir)
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not json_files:
        logger.error("No JSON files found in %s", results_dir)
        sys.exit(1)

    logger.info("Found %d JSON files in %s", len(json_files), results_dir)

    # ── Parse and group ──
    # group key = env_dataset_experiment
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    ablation_all: Dict = {}

    for fpath in json_files:
        fname = os.path.basename(fpath)
        meta = parse_filename(fname)
        if meta is None:
            logger.debug("Skipping unrecognized file: %s", fname)
            continue

        with open(fpath) as f:
            data = json.load(f)

        if meta["type"] == "ablation":
            # Merge ablation data
            ablation_all.update(data)
            continue

        group_key = f"{meta['env']}_{meta['dataset']}_{meta['experiment']}"
        metrics = extract_metrics(data, meta["experiment"])
        grouped[group_key].append(metrics)

    # ── Aggregate ──
    aggregated = aggregate_across_seeds(grouped)

    # ── Display ──
    print_tables(aggregated, latex=args.latex)
    if ablation_all:
        print_ablation_table(ablation_all, latex=args.latex)

    # ── Save ──
    output_path = args.output or os.path.join(results_dir, "aggregated.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_data = {
        "aggregated": aggregated,
        "ablation": ablation_all,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info("Aggregated results saved to %s", output_path)


if __name__ == "__main__":
    main()
