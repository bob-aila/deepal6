"""
deepal.strategies.registry
--------------------------
Central registry of all query strategies.
New strategies can be added here or via register_strategy().
"""

from typing import Callable, Dict


# Will be populated by the strategy modules below
STRATEGIES: Dict[str, Callable] = {}


def register_strategy(name: str, fn: Callable) -> None:
    """
    Register a custom query strategy.

    Parameters
    ----------
    name : str
        Strategy identifier (used in ALConfig and results dict).
    fn : callable
        A function with signature:
            fn(model, data, pool_idx, n_query, **kwargs) -> np.ndarray
        where the return value is an array of integer indices into pool_idx.

    Examples
    --------
    def my_strategy(model, data, pool_idx, n_query, **kwargs):
        # ... your logic ...
        return selected_indices

    from deepal6.strategies.registry import register_strategy
    register_strategy('MyStrategy', my_strategy)
    """
    STRATEGIES[name] = fn


def list_strategies() -> list:
    """Return names of all registered strategies."""
    return list(STRATEGIES.keys())
