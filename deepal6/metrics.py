"""
deepal.metrics
--------------
Metric helpers used by DataModules and the ActiveLearner.
All functions operate on numpy arrays and are framework-agnostic.
"""

import numpy as np
from typing import Dict, List


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs  = np.asarray(probs)
    labels = np.asarray(labels)
    preds  = (probs > 0.5).astype(int)
    correct    = (preds == labels).astype(float)
    confidence = np.maximum(probs, 1 - probs)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(labels)) * abs(
            correct[mask].mean() - confidence[mask].mean()
        )
    return float(ece)


def aggregate_seeds(runs: List[Dict[str, List[float]]]) -> Dict[str, Dict]:
    """
    Aggregate per-seed results into mean ± std curves.
    Input run keys: labeled_counts, aucs, accuracies, bal_accs, recalls, ece_scores
    Output keys:    labeled_counts, auc, accuracy, bal_acc, recall, ece
    """
    if not runs:
        return {}

    # Map raw run keys -> canonical metric names
    key_map = {
        "aucs":       "auc",
        "accuracies": "accuracy",
        "bal_accs":   "bal_acc",
        "recalls":    "recall",
        "ece_scores": "ece",
    }

    out = {"labeled_counts": np.array(runs[0]["labeled_counts"])}

    for raw_key, canonical in key_map.items():
        if raw_key not in runs[0]:
            continue
        matrix = np.array([r[raw_key] for r in runs])   # (n_seeds, n_rounds)
        out[canonical] = {
            "mean": matrix.mean(axis=0),
            "std":  matrix.std(axis=0),
            "all":  matrix,
        }
    return out


def area_under_learning_curve(labeled_counts: np.ndarray, metric: np.ndarray) -> float:
    # Compatible with NumPy < 2.0 (trapz) and >= 2.0 (trapezoid)
    try:
        return float(np.trapezoid(metric, labeled_counts))
    except AttributeError:
        return float(np.trapz(metric, labeled_counts))


def strategy_vs_random_gap(strategy_curve, random_curve) -> np.ndarray:
    return np.asarray(strategy_curve) - np.asarray(random_curve)


def print_summary_table(results: Dict, metric: str = "auc") -> None:
    """Print formatted table of final-round metrics across strategies."""
    header = f"{'Strategy':<12} {'Final ' + metric.upper():>16} {'±Std':>8} {'AULC':>12}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, agg in results.items():
        if metric not in agg or not isinstance(agg[metric], dict):
            continue
        final_mean = agg[metric]["mean"][-1]
        final_std  = agg[metric]["std"][-1]
        lc   = agg.get("labeled_counts", np.arange(len(agg[metric]["mean"])))
        aulc = area_under_learning_curve(lc, agg[metric]["mean"])
        print(f"{name:<12} {final_mean:>16.4f} {final_std:>8.4f} {aulc:>12.2f}")

    print("=" * len(header))
