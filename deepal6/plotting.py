"""
deepal.plotting
---------------
All visualisation functions for active learning results.
Mirrors thesis Chapter 5 plots: learning curves, strategy gap, ECE.
"""

import numpy as np
from typing import Dict, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


STRATEGY_COLORS = {
    "Random":  "#888888",
    "Entropy": "#E07B39",
    "Margin":  "#4C72B0",
    "BALD":    "#55A868",
    "CoreSet": "#C44E52",
    "BADGE":   "#8172B2",
}

METRIC_LABELS = {
    "auc":      "AUC-ROC",
    "accuracy": "Accuracy",
    "bal_acc":  "Balanced Accuracy",
    "recall":   "Recall (Minority Class)",
    "ece":      "Expected Calibration Error (ECE)",
}


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError(
            "Plotting requires matplotlib.\n"
            "Install: pip install matplotlib"
        )


def plot_learning_curves(
    results: Dict,
    metric: str = "auc",
    show_std: bool = True,
    figsize=(12, 5),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot learning curves (metric vs labeled budget) for all strategies.

    Parameters
    ----------
    results   : output of ActiveLearner.run()
    metric    : 'auc', 'accuracy', 'bal_acc', 'recall', 'ece'
    show_std  : shade ±1 std band across seeds
    save_path : path to save figure, e.g. 'results/auc_curves.png'
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)

    for strategy_name, agg in results.items():
        color = STRATEGY_COLORS.get(strategy_name)
        x     = agg.get("labeled_counts", None)

        if metric not in agg or not isinstance(agg[metric], dict):
            print(f"  [plot] '{metric}' not found for {strategy_name} — skipping.")
            continue

        mean_curve = agg[metric]["mean"]
        std_curve  = agg[metric]["std"]
        if x is None:
            x = np.arange(len(mean_curve))

        lw = 2.5 if strategy_name == "Random" else 1.8
        ls = "--" if strategy_name == "Random" else "-"
        ax.plot(x, mean_curve, label=strategy_name, color=color,
                linewidth=lw, linestyle=ls)

        if show_std and std_curve is not None and std_curve.any():
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.12, color=color)

    ylabel = METRIC_LABELS.get(metric, metric.upper())
    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        title or f"Active Learning — {ylabel} vs. Labelling Budget",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_strategy_gap(
    results: Dict,
    metric: str = "auc",
    figsize=(12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot (strategy − Random) metric gap per round.
    Positive = strategy beats random; negative = not yet justified.
    """
    _require_mpl()
    if "Random" not in results:
        print("No 'Random' baseline found — skipping gap plot.")
        return

    random_agg = results["Random"]
    if metric not in random_agg or not isinstance(random_agg[metric], dict):
        print(f"Random baseline missing '{metric}' — skipping gap plot.")
        return

    random_curve = random_agg[metric]["mean"]
    x = random_agg.get("labeled_counts", np.arange(len(random_curve)))

    fig, ax = plt.subplots(figsize=figsize)
    for strategy_name, agg in results.items():
        if strategy_name == "Random":
            continue
        if metric not in agg or not isinstance(agg[metric], dict):
            continue
        color = STRATEGY_COLORS.get(strategy_name)
        gap   = agg[metric]["mean"] - random_curve
        ax.plot(x, gap, label=strategy_name, color=color, linewidth=1.8)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6,
               label="Random (0 gap)")
    ylabel = METRIC_LABELS.get(metric, metric.upper())
    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel(f"Δ {ylabel} vs. Random", fontsize=12)
    ax.set_title(f"Strategy vs. Random — {ylabel} Gap", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration(
    results: Dict,
    save_path: Optional[str] = None,
    figsize=(10, 4),
) -> None:
    """Plot ECE over rounds for all strategies."""
    plot_learning_curves(
        results, metric="ece", show_std=True, figsize=figsize,
        save_path=save_path,
        title="Calibration (ECE) over Labelling Budget — lower is better",
    )


def plot_batch_size_ablation(
    ablation_results: Dict,
    metric: str = "auc",
    figsize=(10, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot learning curves for a batch-size ablation study.

    Parameters
    ----------
    ablation_results : dict keyed by batch size (int or str),
        values are aggregated results for one strategy.
        E.g. {10: agg_b10, 20: agg_b20, 50: agg_b50}

    Example
    -------
    ablation = {}
    for b in [10, 20, 50]:
        cfg = ALConfig(strategy='BALD', batch_size=b)
        r   = ActiveLearner(data, cfg).run()
        ablation[b] = r['BALD']
    plot_batch_size_ablation(ablation)
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    colors  = cm.viridis(np.linspace(0, 0.85, len(ablation_results)))

    for (b, agg), color in zip(ablation_results.items(), colors):
        if metric not in agg or not isinstance(agg[metric], dict):
            continue
        mean_curve = agg[metric]["mean"]
        std_curve  = agg[metric]["std"]
        x = agg.get("labeled_counts", np.arange(len(mean_curve)))
        ax.plot(x, mean_curve, label=f"batch={b}", color=color, linewidth=1.8)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.1, color=color)

    ylabel = METRIC_LABELS.get(metric, metric.upper())
    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Batch Size Ablation — {ylabel}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
