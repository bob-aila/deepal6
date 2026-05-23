"""
deepal6.config
--------------
ALConfig — single source of truth for all experiment parameters.
Validated on construction so errors surface before any training begins.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepal6.exceptions import ConfigurationError
from deepal6.strategies.registry import STRATEGIES


@dataclass
class ALConfig:
    """
    Configuration for an active learning experiment.

    Parameters
    ----------
    strategy : str or list of str
        Query strategy name(s). One of:
        'Random', 'Entropy', 'Margin', 'BALD', 'CoreSet', 'BADGE'.
        Defaults to all six strategies.
    budget : int or None
        Total labelling budget (initial labels + queried labels).
        When set, n_rounds is computed as floor((budget - initial_size) / batch_size).
        Remainder samples are ignored to keep all rounds uniform.
        Provide either budget or n_rounds, not both.
    initial_size : int
        Stratified initial labeled set size. Default: 50.
    batch_size : int
        Samples queried per AL round. Default: 20.
    n_rounds : int or None
        Number of AL rounds. Computed from budget if budget is set.
        Default: 20 (when budget is not provided).
    n_seeds : int
        Independent runs per strategy. Mean ± std reported. Default: 5.
    train_epochs : int
        Training epochs per round. Default: 50.
    lr : float
        Adam learning rate. Default: 1e-3.
    weight_decay : float
        L2 regularisation. Default: 1e-4.
    dropout_rate : float
        Dropout probability (also governs BALD MC stochasticity). Default: 0.3.
    mc_passes : int
        MC Dropout forward passes for BALD. Default: 20.
    train_batch_size : int
        Mini-batch size during model training. Default: 32.
    device : str or None
        'cuda', 'cpu', or None (auto-detect). Default: None.
    seed : int
        Base random seed; run i uses seed + i. Default: 42.
    verbose : bool
        Print per-round metrics. Default: True.
    save_checkpoints : bool
        Save model state dict each round. Default: False.
    checkpoint_dir : str
        Checkpoint output directory. Default: './checkpoints'.
    extra_strategy_kwargs : dict
        Extra kwargs forwarded to strategy functions. Default: {}.

    Examples
    --------
    # Budget-driven (recommended for ablations)
    cfg = ALConfig(strategy='BALD', budget=450, batch_size=20)

    # Fixed rounds (original API — still supported)
    cfg = ALConfig(strategy='BALD', n_rounds=20, batch_size=20)

    # Batch size ablation — same budget, different batch sizes
    for b in [10, 20, 50]:
        cfg = ALConfig(strategy='BALD', budget=450, batch_size=b)
    """

    strategy: Any           = field(default_factory=lambda: list(STRATEGIES.keys()))
    budget: Optional[int]   = None
    initial_size: int       = 50
    batch_size: int         = 20
    n_rounds: Optional[int] = None
    n_seeds: int            = 5
    train_epochs: int       = 50
    lr: float               = 1e-3
    weight_decay: float     = 1e-4
    dropout_rate: float     = 0.3
    mc_passes: int          = 20
    train_batch_size: int   = 32
    device: Optional[str]   = None
    seed: int               = 42
    verbose: bool           = True
    save_checkpoints: bool  = False
    checkpoint_dir: str     = "./checkpoints"
    extra_strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._resolve_rounds()
        self._validate()

    def _resolve_rounds(self):
        """Resolve n_rounds from budget, or fall back to default."""
        if self.budget is not None and self.n_rounds is not None:
            raise ConfigurationError(
                "Provide either 'budget' or 'n_rounds', not both.\n"
                "budget drives n_rounds automatically: "
                "n_rounds = floor((budget - initial_size) / batch_size)."
            )

        if self.budget is not None:
            queryable = self.budget - self.initial_size
            if queryable <= 0:
                raise ConfigurationError(
                    f"budget ({self.budget}) must be greater than "
                    f"initial_size ({self.initial_size})."
                )
            self.n_rounds = math.floor(queryable / self.batch_size)
            remainder = queryable % self.batch_size
            if remainder > 0:
                actual = self.initial_size + self.n_rounds * self.batch_size
                warnings.warn(
                    f"Budget {self.budget} is not evenly divisible: "
                    f"{remainder} sample(s) will be unused. "
                    f"Effective budget: {actual}. "
                    f"Set budget={actual} to suppress this warning.",
                    stacklevel=4,
                )

        if self.n_rounds is None:
            self.n_rounds = 20

    def _validate(self):
        if isinstance(self.strategy, str):
            self.strategy = [self.strategy]

        unknown = [s for s in self.strategy if s not in STRATEGIES]
        if unknown:
            raise ConfigurationError(
                f"Unknown strategy name(s): {unknown}.\n"
                f"Available: {list(STRATEGIES.keys())}.\n"
                "Strategy names are case-sensitive."
            )

        if self.initial_size < 2:
            raise ConfigurationError(
                f"initial_size must be >= 2 (got {self.initial_size})."
            )

        if self.batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be >= 1 (got {self.batch_size})."
            )

        if self.n_rounds < 1:
            raise ConfigurationError(
                f"n_rounds must be >= 1 (got {self.n_rounds}). "
                "Increase budget or reduce batch_size."
            )

        if self.n_seeds < 1:
            raise ConfigurationError(
                f"n_seeds must be >= 1 (got {self.n_seeds})."
            )

        if self.train_epochs < 1:
            raise ConfigurationError(
                f"train_epochs must be >= 1 (got {self.train_epochs})."
            )

        if not (0.0 < self.lr < 1.0):
            raise ConfigurationError(
                f"lr={self.lr} is outside the expected range (1e-5, 1e-2)."
            )

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ConfigurationError(
                f"dropout_rate must be in [0, 1) (got {self.dropout_rate})."
            )

        if self.mc_passes < 1:
            raise ConfigurationError(
                f"mc_passes must be >= 1 (got {self.mc_passes})."
            )

        if self.device not in (None, "cpu", "cuda"):
            raise ConfigurationError(
                f"device must be None, 'cpu', or 'cuda' (got '{self.device}')."
            )

    @property
    def total_budget(self) -> int:
        """Total labels used: initial_size + n_rounds * batch_size."""
        return self.initial_size + self.n_rounds * self.batch_size

    def summary(self) -> str:
        """Formatted parameter summary printed at experiment start."""
        remainder_note = ""
        if self.budget is not None and self.budget != self.total_budget:
            remainder_note = f" ({self.budget - self.total_budget} unused)"

        lines = [
            "=" * 55,
            "  ALConfig — Experiment Parameters",
            "=" * 55,
            f"  Strategies     : {self.strategy}",
            f"  Initial size   : {self.initial_size}",
            f"  Batch size     : {self.batch_size}",
            f"  Rounds         : {self.n_rounds}",
            f"  Seeds          : {self.n_seeds}",
            f"  Total budget   : {self.total_budget} labels{remainder_note}",
            f"  Train epochs   : {self.train_epochs}",
            f"  LR / dropout   : {self.lr} / {self.dropout_rate}",
            f"  MC passes      : {self.mc_passes} (BALD)",
            f"  Device         : {self.device or 'auto'}",
            "=" * 55,
        ]
        return "\n".join(lines)
