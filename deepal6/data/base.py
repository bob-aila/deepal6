"""
deepal.data.base
----------------
Abstract base class for all DataModules.

A DataModule wraps your dataset and exposes a standard interface that all
query strategies use:
    - predict_proba(model, indices, mc_passes=1)
    - get_embeddings(model, indices)
    - get_gradient_embeddings(model, indices)
    - labels          : np.ndarray of all training labels
    - n_train         : total pool size
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseDataModule(ABC):
    """
    Abstract interface every DataModule must implement.

    Subclass this to add support for new data types (e.g. text, graphs).
    """

    @property
    @abstractmethod
    def labels(self) -> np.ndarray:
        """Full array of training labels (length = n_train)."""
        ...

    @property
    @abstractmethod
    def n_train(self) -> int:
        """Total number of training samples in the pool."""
        ...

    @abstractmethod
    def predict_proba(
        self,
        model,
        indices,
        mc_passes: int = 1,
    ) -> np.ndarray:
        """
        Predict class probabilities for the given pool indices.

        Parameters
        ----------
        model : nn.Module
            Trained PyTorch model.
        indices : array-like of int
            Global indices into the training pool.
        mc_passes : int
            1  → standard deterministic inference, shape (N,).
            >1 → MC Dropout passes, shape (mc_passes, N).

        Returns
        -------
        np.ndarray of float in (0, 1).
        """
        ...

    @abstractmethod
    def get_embeddings(self, model, indices) -> np.ndarray:
        """
        Extract penultimate-layer embeddings for the given indices.

        Returns
        -------
        np.ndarray, shape (len(indices), embedding_dim).
        """
        ...

    @abstractmethod
    def get_gradient_embeddings(self, model, indices) -> np.ndarray:
        """
        Compute gradient embeddings (∇_{θ_last} BCE) for BADGE.

        Returns
        -------
        np.ndarray, shape (len(indices), n_last_weights).
        """
        ...

    @abstractmethod
    def train_model(self, model, labeled_idx, config) -> None:
        """
        Train model in-place on the current labeled set.

        Parameters
        ----------
        model : nn.Module
        labeled_idx : list of int
        config : ALConfig
        """
        ...

    @abstractmethod
    def evaluate(self, model) -> dict:
        """
        Evaluate model on the held-out test set.

        Returns
        -------
        dict with keys: 'accuracy', 'auc', 'bal_acc', 'recall', 'ece'.
        """
        ...

    @abstractmethod
    def build_model(self, config) -> "nn.Module":
        """
        Construct and return a fresh model (reset weights each AL round).

        Parameters
        ----------
        config : ALConfig
        """
        ...
