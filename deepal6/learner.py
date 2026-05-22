"""
deepal.learner
--------------
ActiveLearner is the main entry point. It orchestrates:
  1. Stratified initial draw
  2. Per-round: build model → train → evaluate → query → update pool
  3. Multi-seed averaging
  4. Results aggregation and plotting
"""

import time
import os
import numpy as np
from typing import Dict, List, Optional, Union

from deepal6.config import ALConfig
from deepal6.data.base import BaseDataModule
from deepal6.strategies import STRATEGIES
from deepal6.metrics import aggregate_seeds, print_summary_table
from deepal6.exceptions import ConfigurationError, DataError, StrategyError, ModelError
from deepal6.plotting import plot_learning_curves, plot_strategy_gap, plot_calibration


class ActiveLearner:
    """
    Pool-based deep active learning experiment runner.

    Parameters
    ----------
    data : BaseDataModule
        A TabularDataModule or ImageDataModule wrapping your dataset.
    config : ALConfig or None
        Experiment configuration. If None, uses ALConfig() defaults.
    strategy : str or list of str or None
        Shortcut to set config.strategy without a full ALConfig.
        Ignored if config is provided.

    Examples
    --------
    # Minimal
    learner = ActiveLearner(data)
    results = learner.run()

    # Single strategy
    learner = ActiveLearner(data, strategy='BALD')

    # Full control
    cfg = ALConfig(strategy=['BALD', 'CoreSet', 'Random'],
                   initial_size=30, batch_size=10, n_rounds=25, n_seeds=3)
    learner = ActiveLearner(data, cfg)
    results = learner.run()
    learner.plot(results, metric='auc')
    """

    def __init__(
        self,
        data: BaseDataModule,
        config: Optional[ALConfig] = None,
        strategy: Optional[Union[str, List[str]]] = None,
    ):
        if not isinstance(data, BaseDataModule):
            raise DataError(
                f"data must be a TabularDataModule or ImageDataModule, "
                f"got {type(data).__name__}.\n"
                f"Tip: wrap your arrays with TabularDataModule(X_train, y_train, X_test, y_test)."
            )
        self.data = data

        if config is not None and strategy is not None:
            raise ConfigurationError(
                "Provide either 'config' or 'strategy', not both.\n"
                "Tip: pass strategy inside ALConfig(strategy=...) and give that as config."
            )

        if config is None:
            config = ALConfig(strategy=strategy) if strategy else ALConfig()

        self.config = config

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Run the full active learning experiment for all configured strategies
        and seeds.

        Returns
        -------
        dict : strategy name → aggregated results (from aggregate_seeds).
            Each value has keys: labeled_counts, auc, accuracy, bal_acc,
            recall, ece — each a dict with mean/std/all arrays.
        """
        cfg = self.config
        self._validate_config_vs_data()

        if cfg.verbose:
            print(cfg.summary())

        all_results: Dict[str, List[dict]] = {}
        total_start = time.time()

        for strategy_name in cfg.strategy:
            print(f"\n{'='*60}")
            print(f"  Strategy: {strategy_name}")
            print(f"{'='*60}")

            strategy_fn = STRATEGIES[strategy_name]
            runs = []

            for seed_idx in range(cfg.n_seeds):
                seed = cfg.seed + seed_idx
                if cfg.verbose:
                    print(f"  Seed {seed_idx + 1}/{cfg.n_seeds} (seed={seed})")

                run = self._run_single(strategy_name, strategy_fn, seed)
                runs.append(run)

            all_results[strategy_name] = runs

        total_time = time.time() - total_start
        print(f"\nAll strategies completed in {total_time:.1f}s")

        # Aggregate across seeds
        aggregated = {
            name: aggregate_seeds(runs)
            for name, runs in all_results.items()
        }
        return aggregated

    def run_strategy(self, strategy_name: str, seed: Optional[int] = None) -> dict:
        """
        Run a single strategy with a single seed (raw, unaggregated).

        Useful for debugging or custom analysis.

        Parameters
        ----------
        strategy_name : str
            One of: 'Random', 'Entropy', 'Margin', 'BALD', 'CoreSet', 'BADGE'.
        seed : int or None
            Random seed. Defaults to config.seed.

        Returns
        -------
        dict: labeled_counts, accuracies, aucs, bal_accs, recalls, ece_scores
        """
        if strategy_name not in STRATEGIES:
            raise ConfigurationError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {list(STRATEGIES.keys())}."
            )
        seed = seed if seed is not None else self.config.seed
        return self._run_single(strategy_name, STRATEGIES[strategy_name], seed)

    def plot(
        self,
        results: Dict,
        metric: str = "auc",
        show_std: bool = True,
        figsize=(12, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot learning curves and strategy-vs-random gap.

        Parameters
        ----------
        results : dict  — output of ActiveLearner.run()
        metric : str    — 'auc', 'accuracy', 'bal_acc', 'recall', 'ece'
        show_std : bool — shade ±1 std band
        figsize : tuple
        save_path : str or None — save figure to this path
        """
        plot_learning_curves(
            results, metric=metric, show_std=show_std,
            figsize=figsize, save_path=save_path,
        )
        if "Random" in results:
            plot_strategy_gap(results, metric=metric, figsize=figsize)

    def plot_calibration(self, results: Dict, save_path: Optional[str] = None) -> None:
        """Plot ECE calibration curves for all strategies."""
        plot_calibration(results, save_path=save_path)

    def summary_table(self, results: Dict, metric: str = "auc") -> None:
        """Print a formatted summary table of final-round metrics."""
        print_summary_table(results, metric=metric)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_config_vs_data(self):
        n   = self.data.n_train
        cfg = self.config

        if cfg.initial_size >= n:
            raise ConfigurationError(
                f"initial_size ({cfg.initial_size}) >= n_train ({n}). "
                f"Reduce initial_size or use a larger dataset."
            )

        max_budget = cfg.initial_size + cfg.n_rounds * cfg.batch_size
        if max_budget > n:
            import warnings
            warnings.warn(
                f"Total budget ({max_budget}) exceeds pool size ({n}). "
                f"The experiment will stop early when the pool is exhausted.",
                stacklevel=3,
            )

        labels    = self.data.labels
        n_classes = len(np.unique(labels))
        if n_classes < 2:
            raise DataError(
                f"Training labels contain only {n_classes} class(es). "
                f"Active learning requires at least 2 classes."
            )

    def _stratified_initial_draw(self, seed: int):
        """
        Stratified initial labeled set draw — equal class proportions in L0.
        Critical: prevents random baseline from winning by lucky class balance.
        """
        rng    = np.random.RandomState(seed)
        labels = self.data.labels
        n_init = self.config.initial_size

        unique_classes = np.unique(labels)
        n_per_class    = n_init // len(unique_classes)
        remainder      = n_init % len(unique_classes)

        labeled_idx = []
        for i, cls in enumerate(unique_classes):
            cls_idx = np.where(labels == cls)[0]
            n_draw  = n_per_class + (1 if i < remainder else 0)
            n_draw  = min(n_draw, len(cls_idx))
            chosen  = rng.choice(cls_idx, size=n_draw, replace=False)
            labeled_idx.extend(chosen.tolist())

        all_idx       = set(range(len(labels)))
        unlabeled_idx = sorted(all_idx - set(labeled_idx))
        return labeled_idx, unlabeled_idx

    def _run_single(self, strategy_name: str, strategy_fn, seed: int) -> dict:
        """Execute one complete active learning run (one seed)."""
        cfg = self.config
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

        labeled_idx, unlabeled_idx = self._stratified_initial_draw(seed)

        labeled_counts = []
        accuracies     = []
        aucs           = []
        bal_accs       = []
        recalls        = []
        ece_scores     = []

        for round_t in range(cfg.n_rounds):
            if len(unlabeled_idx) == 0:
                if cfg.verbose:
                    print(f"  [{strategy_name}] Pool exhausted at round {round_t}.")
                break

            # Build fresh model each round (no representation drift)
            try:
                model = self.data.build_model(cfg)
            except Exception as e:
                raise ModelError(
                    f"[{strategy_name}] Round {round_t+1}: build_model failed.\n{e}"
                ) from e

            # Train on current labeled set
            try:
                self.data.train_model(model, labeled_idx, cfg)
            except Exception as e:
                raise ModelError(
                    f"[{strategy_name}] Round {round_t+1}: training failed "
                    f"(n_labeled={len(labeled_idx)}).\n{e}"
                ) from e

            # Evaluate on held-out test set
            metrics = self.data.evaluate(model)
            labeled_counts.append(len(labeled_idx))
            accuracies.append(metrics["accuracy"])
            aucs.append(metrics["auc"])
            bal_accs.append(metrics["bal_acc"])
            recalls.append(metrics["recall"])
            ece_scores.append(metrics["ece"])

            if cfg.verbose:
                print(
                    f"  [{strategy_name}] Round {round_t+1:2d} | "
                    f"n={len(labeled_idx):4d} | "
                    f"AUC={metrics['auc']:.4f} | "
                    f"BalAcc={metrics['bal_acc']:.4f} | "
                    f"Recall={metrics['recall']:.4f} | "
                    f"ECE={metrics['ece']:.4f}"
                )

            # Query strategy
            n_query = min(cfg.batch_size, len(unlabeled_idx))
            extra   = {**cfg.extra_strategy_kwargs, "mc_passes": cfg.mc_passes}

            try:
                local_indices = strategy_fn(
                    model       = model,
                    data        = self.data,
                    pool_idx    = unlabeled_idx,
                    n_query     = n_query,
                    labeled_idx = labeled_idx,
                    **extra,
                )
            except StrategyError:
                raise
            except Exception as e:
                raise StrategyError(
                    f"[{strategy_name}] Round {round_t+1}: strategy query failed.\n"
                    f"pool size={len(unlabeled_idx)}, n_query={n_query}\nError: {e}"
                ) from e

            # Update labeled / unlabeled sets
            queried_global = [unlabeled_idx[i] for i in local_indices]
            labeled_idx.extend(queried_global)
            for i in sorted(local_indices, reverse=True):
                unlabeled_idx.pop(i)

            # Optional checkpoint
            if cfg.save_checkpoints:
                self._save_checkpoint(model, strategy_name, seed, round_t)

        return {
            "labeled_counts": labeled_counts,
            "accuracies":     accuracies,
            "aucs":           aucs,
            "bal_accs":       bal_accs,
            "recalls":        recalls,
            "ece_scores":     ece_scores,
        }

    def _save_checkpoint(self, model, strategy_name, seed, round_t):
        try:
            import torch
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            fname = os.path.join(
                self.config.checkpoint_dir,
                f"{strategy_name}_seed{seed}_round{round_t+1}.pt",
            )
            torch.save(model.state_dict(), fname)
        except Exception as e:
            import warnings
            warnings.warn(f"Checkpoint save failed: {e}")
