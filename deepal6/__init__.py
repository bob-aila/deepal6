"""
DeepAL6 — Deep Active Learning Library
======================================
A flexible, research-grade active learning framework supporting 6 query
strategies compared against a random baseline, for both tabular and image
domains.

Strategies
----------
- Random      : Uniform random draw (baseline)
- Entropy     : Highest predictive entropy H[y|x]
- Margin      : Smallest top-2 class probability gap
- BALD        : Mutual information I[y;θ|x,D] via MC Dropout
- CoreSet     : Greedy k-center covering in embedding space
- BADGE       : Gradient embeddings + k-means++ diversity

Quickstart
----------
    from deepal6 import ActiveLearner, TabularDataModule

    data = TabularDataModule(X_train, y_train, X_test, y_test)
    learner = ActiveLearner(data, strategy="BALD")
    results = learner.run(initial_size=50, batch_size=20, n_rounds=20)
    learner.plot(results)
"""

import deepal6.strategies.query  # registers all 6 strategies into STRATEGIES dict
from deepal6.learner import ActiveLearner
from deepal6.data.tabular import TabularDataModule
from deepal6.data.image import ImageDataModule
from deepal6.strategies.registry import STRATEGIES, list_strategies
from deepal6.config import ALConfig
from deepal6.exceptions import (
    DeepALError,
    ConfigurationError,
    DataError,
    StrategyError,
    ModelError,
)

__version__ = "1.0.1"
__author__ = "Bob Philip Aila — AIMS Rwanda"

__all__ = [
    "ActiveLearner",
    "TabularDataModule",
    "ImageDataModule",
    "STRATEGIES",
    "list_strategies",
    "ALConfig",
    "DeepALError",
    "ConfigurationError",
    "DataError",
    "StrategyError",
    "ModelError",
]
