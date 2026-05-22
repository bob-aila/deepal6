"""
deepal.config
-------------
ALConfig is the single source of truth for all experiment hyper-parameters.
Every parameter has a sensible default drawn from the thesis experiments.
All values are validated on construction so errors surface early.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
        Pass a list to run multiple strategies in one experiment.
        Default: all 6 strategies.
    initial_size : int
        Number of samples in the initial labeled set L0.
        The draw is always stratified (equal class proportions) to
        prevent the random baseline from winning by luck.
        Default: 50.
    batch_size : int
        Number of samples queried from the pool per AL round.
        Smaller batches (10–20) favour uncertainty strategies;
        larger batches (50+) favour diversity strategies (CoreSet, BADGE).
        Default: 20.
    n_rounds : int
        Maximum number of active learning rounds.
        Total labelling budget = initial_size + n_rounds * batch_size.
        Default: 20.
    n_seeds : int
        Number of independent runs per strategy (different random seeds).
        Mean ± std across seeds is reported.  ≥3 recommended for publication.
        Default: 5.
    train_epochs : int
        Epochs to train at each AL round.
        Default: 50 (tabular) — ImageDataModule overrides to 10.
    lr : float
        Adam learning rate.  Default: 1e-3 (tabular), 1e-4 (image).
    weight_decay : float
        L2 regularisation strength.  Default: 1e-4.
    dropout_rate : float
        Dropout probability used in both CreditNet and ResNet-18 head.
        Also controls MC Dropout stochasticity for BALD.
        Default: 0.3.
    mc_passes : int
        Number of stochastic forward passes for BALD.
        Higher = lower variance estimate, higher cost.
        Default: 20.
    train_batch_size : int
        Mini-batch size used during model training (not AL batch size).
        Default: 32.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).  Default: None.
    seed : int
        Base random seed.  Each of the n_seeds runs uses seed+i.
        Default: 42.
    verbose : bool
        Print per-round metrics during the experiment.  Default: True.
    save_checkpoints : bool
        Save the best model checkpoint per strategy per seed.
        Default: False.
    checkpoint_dir : str
        Directory for checkpoints (only used if save_checkpoints=True).
        Default: './checkpoints'.
    extra_strategy_kwargs : dict
        Extra keyword arguments forwarded to specific strategy functions.
        E.g., {'mc_passes': 30} overrides the top-level mc_passes for BALD.
        Default: {}.

    Examples
    --------
    # Single strategy, quick experiment
    cfg = ALConfig(strategy='BALD', initial_size=30, batch_size=10, n_rounds=15)

    # Compare all strategies with publication-quality settings
    cfg = ALConfig(n_seeds=5, train_epochs=50)

    # Image domain — fewer epochs, lower LR, larger batches
    cfg = ALConfig(strategy=['BALD', 'CoreSet', 'Random'],
                   train_epochs=10, lr=1e-4, batch_size=20)
    """

    strategy: Any = field(default_factory=lambda: list(STRATEGIES.keys()))
    initial_size: int = 50
    batch_size: int = 20
    n_rounds: int = 20
    n_seeds: int = 5
    train_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout_rate: float = 0.3
    mc_passes: int = 20
    train_batch_size: int = 32
    device: Optional[str] = None
    seed: int = 42
    verbose: bool = True
    save_checkpoints: bool = False
    checkpoint_dir: str = "./checkpoints"
    extra_strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """Validate all parameters; raise ConfigurationError with clear message."""
        # Normalise strategy to list
        if isinstance(self.strategy, str):
            self.strategy = [self.strategy]

        unknown = [s for s in self.strategy if s not in STRATEGIES]
        if unknown:
            raise ConfigurationError(
                f"Unknown strategy name(s): {unknown}.\n"
                f"Available strategies: {list(STRATEGIES.keys())}.\n"
                f"Tip: strategy names are case-sensitive. "
                f"Use deepal.list_strategies() to see all options."
            )

        if self.initial_size < 2:
            raise ConfigurationError(
                f"initial_size must be at least 2 (got {self.initial_size}). "
                f"You need at least one sample per class for a meaningful start."
            )

        if self.batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be >= 1 (got {self.batch_size}). "
                f"Typical values: 10–50."
            )

        if self.n_rounds < 1:
            raise ConfigurationError(
                f"n_rounds must be >= 1 (got {self.n_rounds})."
            )

        if self.n_seeds < 1:
            raise ConfigurationError(
                f"n_seeds must be >= 1 (got {self.n_seeds}). "
                f"Use n_seeds >= 3 for statistically meaningful results."
            )

        if self.train_epochs < 1:
            raise ConfigurationError(
                f"train_epochs must be >= 1 (got {self.train_epochs})."
            )

        if not (0.0 < self.lr < 1.0):
            raise ConfigurationError(
                f"lr={self.lr} looks suspicious. Typical range: 1e-5 to 1e-2."
            )

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ConfigurationError(
                f"dropout_rate must be in [0, 1) (got {self.dropout_rate})."
            )

        if self.mc_passes < 1:
            raise ConfigurationError(
                f"mc_passes must be >= 1 (got {self.mc_passes}). "
                f"BALD typically uses 20–50 passes."
            )

        if self.device not in (None, "cpu", "cuda"):
            raise ConfigurationError(
                f"device must be None, 'cpu', or 'cuda' (got '{self.device}')."
            )

    @property
    def total_budget(self) -> int:
        """Maximum number of labeled samples at end of experiment."""
        return self.initial_size + self.n_rounds * self.batch_size

    def summary(self) -> str:
        """Human-readable parameter summary."""
        lines = [
            "=" * 55,
            "  ALConfig — Experiment Parameters",
            "=" * 55,
            f"  Strategies     : {self.strategy}",
            f"  Initial size   : {self.initial_size}",
            f"  Batch size     : {self.batch_size}",
            f"  Rounds         : {self.n_rounds}",
            f"  Seeds          : {self.n_seeds}",
            f"  Total budget   : {self.total_budget} labels",
            f"  Train epochs   : {self.train_epochs}",
            f"  LR / dropout   : {self.lr} / {self.dropout_rate}",
            f"  MC passes      : {self.mc_passes} (BALD)",
            f"  Device         : {self.device or 'auto'}",
            "=" * 55,
        ]
        return "\n".join(lines)
