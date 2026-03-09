#!/usr/bin/env python3
"""
Multi-Model Multi-Run Analysis
===============================
Comprehensive analysis of 5 runs × 30 scenarios × 3 models.

1. Loads all run results
2. Recomputes Risk Manager scores where tool call failed (deterministic recomputation)
3. Evaluates each agent against ground truth using the evaluation harness
4. Computes per-scenario mean ± std across 5 runs
5. Computes across-scenario mean ± std
6. Generates multi-model comparison figure with error bars
7. Generates summary table data for thesis

Output:
  - multi_model_comparison.pdf (figure with error bars)
  - ws5_final_multimodel_summary.json (table data)
"""

import json
import logging
import os
import re
import statistics
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Import evaluation functions from the harness
sys.path.insert(0, str(ROOT / "evaluation"))
from evaluation_harness import (
    eval_disruption_monitoring, eval_kg_query, eval_risk_manager, eval_csco,
    safe_json_parse, normalize, AgentPerf
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WS5_OUTPUT = ROOT / "evaluation" / "results"
GT_DIR = ROOT / "evaluation" / "ground_truth"
SCENARIOS_FILE = ROOT / "evaluation" / "scenarios" / "multi_company_scenarios.json"
FIGURE_DIR = ROOT / "evaluation" / "results" / "analysis"

MODELS = {
    "gpt-4o":     {"dir_name": "baseline",    "label": "GPT-4o (baseline)", "color": "#2196F3"},
    "gpt-4.1":    {"dir_name": "gpt-4.1",     "label": "gpt-4.1",          "color": "#4CAF50"},
    "gpt-5-mini": {"dir_name": "gpt-5-mini",  "label": "gpt-5-mini",       "color": "#FF9800"},
}
TOTAL_RUNS = 5


# ---------------------------------------------------------------------------
# Load and process run data
# ---------------------------------------------------------------------------
def load_all_runs() -> Dict[str, Dict[str, List[dict]]]:
    """Load all run results, organized as model -> scenario -> [runs].
    
    Returns: {model_id: {scenario_id: [run_data, ...]}}
    """
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    scenario_map = {sc["scenario_id"]: sc for sc in scenarios}
    
    all_data = {}
    for model_id, info in MODELS.items():
        model_dir = WS5_OUTPUT / info["dir_name"]
        if not model_dir.exists():
            print(f"  WARNING: {model_dir} not found for {model_id}")
            continue
        
        model_data = {}
        for f in sorted(model_dir.glob("*_run*.json")):
            try:
                data = json.load(open(f))
            except:
                continue
            
            sid = data.get("scenario_id", "")
            if not sid or sid not in scenario_map:
                continue
            if not data.get("success"):
                continue
            
            model_data.setdefault(sid, []).append(data)
        
        all_data[model_id] = model_data
    
    return all_data


def validate_runs(all_data: Dict) -> Dict:
    """Validate all runs: check Risk Manager output quality.
    
    With the file-based handoff architecture, the Risk Manager tool loads
    data from disk and computes deterministically. Runs with 0 risk scores
    indicate a genuine pipeline failure.
    """
    stats = {"total": 0, "valid": 0, "zero_scores": 0}
    
    for model_id, model_data in all_data.items():
        for sid, runs in model_data.items():
            for run in runs:
                stats["total"] += 1
                ao = run.get("agent_outputs", {})
                risk = ao.get("risk_assessment", {})
                
                if isinstance(risk, str):
                    try:
                        risk = json.loads(risk)
                    except:
                        risk = {}
                
                has_scores = (isinstance(risk, dict) and 
                            len(risk.get("supplier_risk_scores", {})) > 0)
                
                if has_scores:
                    stats["valid"] += 1
                else:
                    stats["zero_scores"] += 1
    
    print(f"\n  Risk Manager Output Validation:")
    print(f"    Total runs:     {stats['total']}")
    print(f"    Valid (>0 scores): {stats['valid']} ({100*stats['valid']/max(stats['total'],1):.0f}%)")
    print(f"    Zero scores:    {stats['zero_scores']} ({100*stats['zero_scores']/max(stats['total'],1):.0f}%)")
    
    return all_data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_run(ao: dict, gt: dict) -> dict:
    """Evaluate a single run against ground truth. Returns per-agent metrics."""
    dm = eval_disruption_monitoring(ao, gt)
    kg = eval_kg_query(ao, gt)
    risk = eval_risk_manager(ao, gt)
    csco = eval_csco(ao, gt)
    
    return {
        "dm_f1": dm.f1,
        "dm_strict": 1.0 if dm.strict_success else 0.0,
        "kg_strict": 1.0 if kg.strict_success else 0.0,
        "risk_f1": risk.f1,
        "risk_strict": 1.0 if risk.strict_success else 0.0,
        "csco_f1": csco.f1,
        "csco_strict": 1.0 if csco.strict_success else 0.0,
        "constrained": 1.0 if (dm.strict_success and kg.strict_success and 
                               risk.strict_success and csco.strict_success) else 0.0,
    }


def evaluate_all(all_data: Dict, gt_data: Dict) -> Dict:
    """Evaluate all runs for all models.
    
    Returns: {model_id: {scenario_id: [run_metrics, ...]}}
    """
    results = {}
    for model_id, model_data in all_data.items():
        model_results = {}
        for sid, runs in model_data.items():
            gt = gt_data.get(sid)
            if not gt:
                continue
            
            run_metrics = []
            for run in runs:
                ao = run.get("agent_outputs", {})
                metrics = evaluate_run(ao, gt)
                metrics["runtime_seconds"] = run.get("runtime_seconds", 0)
                run_metrics.append(metrics)
            
            model_results[sid] = run_metrics
        results[model_id] = model_results
    
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_model_summary(model_results: Dict[str, List[dict]]) -> dict:
    """Compute aggregate metrics for one model across all scenarios.
    
    For each metric, computes:
    - Per-scenario mean across runs
    - Across-scenario mean and std of per-scenario means
    """
    metrics_keys = ["dm_f1", "dm_strict", "kg_strict", "risk_f1", "risk_strict",
                    "csco_f1", "csco_strict", "constrained"]
    
    scenario_means = {k: [] for k in metrics_keys}
    scenario_stds = {k: [] for k in metrics_keys}
    runtime_means = []
    runtime_stds = []
    n_runs_per_scenario = []
    
    for sid, runs in sorted(model_results.items()):
        n_runs_per_scenario.append(len(runs))
        
        runtimes = [r["runtime_seconds"] for r in runs]
        runtime_means.append(np.mean(runtimes))
        runtime_stds.append(np.std(runtimes, ddof=1) if len(runtimes) > 1 else 0.0)
        
        for key in metrics_keys:
            values = [r[key] for r in runs]
            scenario_means[key].append(np.mean(values))
            scenario_stds[key].append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
    
    summary = {
        "n_scenarios": len(model_results),
        "n_runs_total": sum(n_runs_per_scenario),
        "n_runs_per_scenario_mean": np.mean(n_runs_per_scenario),
    }
    
    for key in metrics_keys:
        means = scenario_means[key]
        summary[f"{key}_mean"] = float(np.mean(means))
        summary[f"{key}_std"] = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        summary[f"{key}_within_std"] = float(np.mean(scenario_stds[key]))
    
    summary["runtime_mean"] = float(np.mean(runtime_means))
    summary["runtime_std"] = float(np.std(runtime_means, ddof=1)) if len(runtime_means) > 1 else 0.0
    summary["runtime_within_std"] = float(np.mean(runtime_stds))
    
    for key in ["dm_strict", "kg_strict", "risk_strict", "csco_strict", "constrained"]:
        means = scenario_means[key]
        n_pass = sum(1 for m in means if m >= 0.5)
        summary[f"{key}_pass_count"] = n_pass
    
    return summary


# ---------------------------------------------------------------------------
# Figure Generation
# ---------------------------------------------------------------------------
def generate_figure(summaries: Dict[str, dict]):
    """Generate multi-model comparison figure.
    
    Style: Arial, blue-green palette, no grids, no title,
    left+bottom spines only, 600 dpi.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    B1 = '#003EAA'
    B3 = '#2D8CFF'
    G2 = '#00994C'
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica',
                            'DejaVu Sans', 'sans-serif'],
        'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'axes.edgecolor': '#333333',
        'axes.labelcolor': '#222222',
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'axes.grid': False, 'legend.frameon': False,
    })
    
    agents = [
        ('dm_strict',   'Disruption\nMonitor'),
        ('kg_strict',   'KG Query'),
        ('risk_strict', 'Risk\nManager'),
        ('csco_strict', 'CSCO'),
        ('constrained', 'End-to-end'),
    ]
    
    model_order = ["gpt-4o", "gpt-4.1", "gpt-5-mini"]
    palette = {"gpt-4o": B1, "gpt-4.1": B3, "gpt-5-mini": G2}
    
    series = [(MODELS[m]["label"], summaries.get(m, {}), palette[m])
              for m in model_order if summaries.get(m)]
    nm = len(series)
    
    W, H = 5.8, 3.5
    fig, ax = plt.subplots(figsize=(W + 0.3, H))
    x = np.arange(len(agents))
    bw = 0.72 / nm
    
    for i, (label, md, col) in enumerate(series):
        vals = [md.get(f'{key}_mean', 0) for key, _ in agents]
        off = (i - nm / 2 + 0.5) * bw
        bars = ax.bar(x + off, vals, bw, color=col, edgecolor='white',
                      linewidth=0.4, label=label, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(v + 0.025, 1.06),
                    f'{v:.0%}', ha='center', va='bottom',
                    fontsize=7.2, color='#1f1f1f')
    
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in agents], linespacing=1.05)
    ax.set_ylabel('Strict success rate')
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc='upper right', fontsize=7.5)
    
    fig.tight_layout()
    
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGURE_DIR / f"multi_model_comparison.{ext}", dpi=600,
                    facecolor='white', edgecolor='none')
    print(f"\n  Figure saved: {FIGURE_DIR / 'multi_model_comparison.pdf'}")
    
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("MULTI-MODEL MULTI-RUN ANALYSIS")
    print("=" * 70)
    
    print("\n[1] Loading scenarios and ground truth...")
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    
    gt_data = {}
    for f in sorted(GT_DIR.glob("*.json")):
        gt = json.load(open(f))
        sid = f.stem
        if sid not in {sc["scenario_id"] for sc in scenarios}:
            continue
        gt_data[sid] = gt
    print(f"  Loaded {len(scenarios)} scenarios, {len(gt_data)} ground truth files")
    
    print("\n[2] Loading all multi-run data...")
    all_data = load_all_runs()
    for model_id, model_data in all_data.items():
        n_scenarios = len(model_data)
        n_runs = sum(len(runs) for runs in model_data.values())
        print(f"  {model_id:15s}: {n_scenarios} scenarios, {n_runs} successful runs")
    
    print("\n[3] Validating Risk Manager outputs...")
    all_data = validate_runs(all_data)
    
    print("\n[4] Evaluating against ground truth...")
    eval_results = evaluate_all(all_data, gt_data)
    
    print("\n[5] Computing model summaries...")
    summaries = {}
    for model_id, model_results in eval_results.items():
        summary = compute_model_summary(model_results)
        summaries[model_id] = summary
        
        print(f"\n  --- {MODELS[model_id]['label']} ---")
        print(f"    Scenarios: {summary['n_scenarios']}, "
              f"Total runs: {summary['n_runs_total']}")
        print(f"    DM composite F1:    {summary['dm_f1_mean']:.3f} +/- {summary['dm_f1_std']:.3f}")
        print(f"    DM strict success:  {summary['dm_strict_mean']:.2f} "
              f"({summary['dm_strict_pass_count']}/{summary['n_scenarios']})")
        print(f"    KG strict success:  {summary['kg_strict_mean']:.2f} "
              f"({summary['kg_strict_pass_count']}/{summary['n_scenarios']})")
        print(f"    Risk strict success:{summary['risk_strict_mean']:.2f} "
              f"({summary['risk_strict_pass_count']}/{summary['n_scenarios']})")
        print(f"    CSCO strict success:{summary['csco_strict_mean']:.2f} "
              f"({summary['csco_strict_pass_count']}/{summary['n_scenarios']})")
        print(f"    Constrained success:{summary['constrained_mean']:.2f} "
              f"({summary['constrained_pass_count']}/{summary['n_scenarios']})")
        print(f"    Runtime:            {summary['runtime_mean']:.1f}s")
    
    print("\n[6] Generating multi-model comparison figure...")
    generate_figure(summaries)
    
    summary_file = FIGURE_DIR / "ws5_final_multimodel_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_file}")
    
    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
