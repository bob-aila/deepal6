"""
deepal.exceptions
-----------------
All package-specific exceptions with clear, actionable messages.
"""


class DeepALError(Exception):
    """Base exception for all deepal errors."""
    pass


class ConfigurationError(DeepALError):
    """
    Raised when ALConfig parameters are invalid.

    Examples
    --------
    - initial_size is larger than the training pool
    - batch_size is 0 or negative
    - n_rounds is 0 or negative
    - unknown strategy name is passed
    """
    pass


class DataError(DeepALError):
    """
    Raised when DataModule inputs are malformed or incompatible.

    Examples
    --------
    - X and y have different lengths
    - X_test contains NaN values
    - Image file paths do not exist
    - Label column missing from DataFrame
    """
    pass


class StrategyError(DeepALError):
    """
    Raised when a query strategy fails internally.

    Examples
    --------
    - Pool is empty and strategy requests samples
    - MC Dropout requested but model has no Dropout layers
    - Embedding extraction hook fails (unsupported architecture)
    """
    pass


class ModelError(DeepALError):
    """
    Raised when model construction or training fails.

    Examples
    --------
    - input_dim mismatch between model and data
    - CUDA out of memory (with helpful suggestion)
    - Pretrained weights cannot be downloaded
    """
    pass
