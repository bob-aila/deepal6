"""
deepal.data.tabular
-------------------
TabularDataModule wraps numpy arrays (X_train, y_train, X_test, y_test)
and provides all the hooks that the active learning loop and query strategies
need.

Model: CreditNet — a fully-connected network with BatchNorm, ReLU, and
Dropout support, mirroring the thesis Chapter 5 (finance domain) design.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Union

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
)

from deepal6.data.base import BaseDataModule
from deepal6.exceptions import DataError, ModelError
from deepal6.metrics import compute_ece


class CreditNet(nn.Module):
    """
    Fully-connected network for binary tabular classification.

    Architecture (from thesis Chapter 5):
        Input → [Linear → BatchNorm → ReLU → Dropout] x3 → Linear → Sigmoid

    Hidden layer widths scale with input_dim to avoid over-parameterising
    small feature sets. BatchNorm1d stabilises training on small labeled sets
    (as few as 50 samples). Dropout stays active at inference for BALD's MC
    Dropout via enable_dropout().

    Parameters
    ----------
    input_dim : int
        Number of input features.
    dropout_rate : float
        Dropout probability. Default: 0.3.
    hidden_multipliers : tuple of float
        Multipliers applied to input_dim for each hidden layer.
        Default: (6, 3, 1) → widths max(64, 6*d), max(32, 3*d), max(16, d).
    """

    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.3,
        hidden_multipliers=(6, 3, 1),
    ):
        super().__init__()
        floors = (64, 32, 16)
        widths = [max(f, int(input_dim * m)) for f, m in zip(floors, hidden_multipliers)]

        layers = []
        prev = input_dim
        for h in widths:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def enable_dropout(self):
        """Activate dropout at inference time for MC Dropout (BALD)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class TabularDataModule(BaseDataModule):
    """
    DataModule for numpy-array tabular datasets.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Training feature matrix (already scaled).
    y_train : np.ndarray, shape (n_train,)
        Binary training labels (0 / 1).
    X_test : np.ndarray, shape (n_test, n_features)
        Test feature matrix (scaled with training statistics).
    y_test : np.ndarray, shape (n_test,)
        Binary test labels.
    pos_label : int
        Which integer value is the "positive" (minority) class.
        Used for recall computation.  Default: 0 (Bad credit in thesis).
    device : str or None
        'cpu', 'cuda', or None (auto-detect).  Default: None.

    Examples
    --------
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw_train)
    X_test  = scaler.transform(X_raw_test)

    data = TabularDataModule(X_train, y_train, X_test, y_test)
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        pos_label: int = 0,
        device: Optional[str] = None,
    ):
        self._validate_inputs(X_train, y_train, X_test, y_test)
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.int64)
        self.X_test  = X_test.astype(np.float32)
        self.y_test  = y_test.astype(np.int64)
        self.pos_label = pos_label

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(X_train, y_train, X_test, y_test):
        for name, arr in [("X_train", X_train), ("y_train", y_train),
                          ("X_test", X_test),  ("y_test",  y_test)]:
            if not isinstance(arr, np.ndarray):
                raise DataError(
                    f"{name} must be a numpy ndarray, got {type(arr).__name__}.\n"
                    f"Tip: call .values on a pandas DataFrame."
                )
        if X_train.ndim != 2:
            raise DataError(
                f"X_train must be 2-D (n_samples, n_features), "
                f"got shape {X_train.shape}."
            )
        if X_test.ndim != 2:
            raise DataError(
                f"X_test must be 2-D (n_samples, n_features), "
                f"got shape {X_test.shape}."
            )
        if len(X_train) != len(y_train):
            raise DataError(
                f"X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}."
            )
        if len(X_test) != len(y_test):
            raise DataError(
                f"X_test and y_test length mismatch: "
                f"{len(X_test)} vs {len(y_test)}."
            )
        if X_train.shape[1] != X_test.shape[1]:
            raise DataError(
                f"X_train and X_test have different feature counts: "
                f"{X_train.shape[1]} vs {X_test.shape[1]}.\n"
                f"Tip: apply the same scaler/encoder to both."
            )
        if np.isnan(X_train).any():
            raise DataError("X_train contains NaN values. Impute or drop them first.")
        if np.isnan(X_test).any():
            raise DataError("X_test contains NaN values. Impute or drop them first.")
        unique_labels = set(np.unique(y_train))
        if not unique_labels.issubset({0, 1}):
            raise DataError(
                f"y_train must contain only 0 and 1 (got {unique_labels}).\n"
                f"Tip: map your labels to binary integers before passing to TabularDataModule."
            )

    # ── BaseDataModule interface ───────────────────────────────────────────────

    @property
    def labels(self) -> np.ndarray:
        return self.y_train

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    def build_model(self, config) -> nn.Module:
        """
        Build a fresh CreditNet. Called at the start of each AL round so
        each round trains from scratch (no representation drift).
        """
        try:
            model = CreditNet(
                input_dim=self.X_train.shape[1],
                dropout_rate=config.dropout_rate,
            ).to(self._device)
        except Exception as e:
            raise ModelError(
                f"Failed to build CreditNet: {e}\n"
                f"input_dim={self.X_train.shape[1]}, "
                f"dropout_rate={config.dropout_rate}"
            ) from e
        return model

    def train_model(self, model: nn.Module, labeled_idx, config) -> None:
        """
        Train model in-place on the current labeled set with class-weighted
        BCE loss (prevents majority-class collapse on small labeled sets).

        Uses Adam + StepLR scheduler (halves LR every 20 epochs).
        """
        if len(labeled_idx) == 0:
            raise DataError(
                "train_model received an empty labeled_idx. "
                "initial_size must be >= 2."
            )

        model.train()
        X = self.X_train[labeled_idx]
        y = self.y_train[labeled_idx]

        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self._device)

        # Class-weighted BCE — critical for imbalanced early labeled sets
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        ).to(self._device)
        criterion = nn.BCELoss()

        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        bs = min(config.train_batch_size, len(X))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True)

        for epoch in range(config.train_epochs):
            for bX, by in loader:
                optimizer.zero_grad()
                out = model(bX)
                weight_mask = torch.where(
                    by == 1, pos_weight, torch.ones_like(by)
                )
                loss = (criterion(out, by) * weight_mask).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

    def predict_proba(
        self,
        model: nn.Module,
        indices,
        mc_passes: int = 1,
    ) -> np.ndarray:
        """
        Predict probabilities for given pool indices.

        mc_passes=1  → deterministic eval, shape (N,).
        mc_passes>1  → MC Dropout, shape (mc_passes, N).
        """
        X = self.X_train[list(indices)]
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)

        if mc_passes == 1:
            model.eval()
            with torch.no_grad():
                probs = model(X_t).squeeze().cpu().numpy()
            return np.atleast_1d(probs)
        else:
            model.eval()
            model.enable_dropout()
            all_p = []
            with torch.no_grad():
                for _ in range(mc_passes):
                    p = model(X_t).squeeze().cpu().numpy()
                    all_p.append(np.atleast_1d(p))
            return np.stack(all_p, axis=0)  # (mc_passes, N)

    def get_embeddings(self, model: nn.Module, indices) -> np.ndarray:
        """
        Extract penultimate-layer embeddings via a forward hook.
        For CreditNet the penultimate layer is the last Dropout before the
        output linear layer (index -3 of model.net children).
        """
        model.eval()
        X = self.X_train[list(indices)]
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)

        activation = {}
        layers = list(model.net.children())

        # Hook the activation just before the final Linear+Sigmoid
        target = layers[-3]  # last Dropout
        hook = target.register_forward_hook(
            lambda m, inp, out: activation.update({"embed": out.detach().cpu().numpy()})
        )
        with torch.no_grad():
            model(X_t)
        hook.remove()

        if "embed" not in activation:
            raise ModelError(
                "Embedding hook did not fire. "
                "CreditNet architecture may have changed — "
                "check that model.net has at least 3 children."
            )
        return activation["embed"]

    def get_gradient_embeddings(self, model: nn.Module, indices) -> np.ndarray:
        """
        Compute BADGE gradient embeddings for each pool sample:
            g_x = ∇_{θ_last} BCE(f_θ(x), ŷ_x)

        ŷ_x is the predicted pseudo-label (argmax over p).
        """
        model.eval()
        X = self.X_train[list(indices)]
        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)

        # Find last linear layer
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ModelError(
                "get_gradient_embeddings: no nn.Linear found in model. "
                "BADGE requires at least one linear layer."
            )

        grad_embeddings = []
        for i in range(len(X)):
            x = X_t[i].unsqueeze(0)
            out = model(x)
            p = out.item()
            y_hat = torch.tensor(
                [[1.0 if p > 0.5 else 0.0]], dtype=torch.float32
            ).to(self._device)
            model.zero_grad()
            nn.BCELoss()(model(x), y_hat).backward()
            if last_linear.weight.grad is not None:
                g = last_linear.weight.grad.detach().cpu().numpy().flatten()
            else:
                g = np.zeros(last_linear.weight.numel())
            grad_embeddings.append(g)

        return np.stack(grad_embeddings)  # (N, d_last_weights)

    def evaluate(self, model: nn.Module) -> dict:
        """
        Evaluate on the held-out test set.

        Returns
        -------
        dict: accuracy, auc, bal_acc, recall (minority class), ece
        """
        X_t = torch.tensor(self.X_test, dtype=torch.float32).to(self._device)
        model.eval()
        with torch.no_grad():
            probs = model(X_t).squeeze().cpu().numpy()
        probs = np.atleast_1d(probs)
        preds = (probs > 0.5).astype(int)

        acc     = accuracy_score(self.y_test, preds)
        bal_acc = balanced_accuracy_score(self.y_test, preds)
        recall  = recall_score(
            self.y_test, preds, pos_label=self.pos_label, zero_division=0
        )
        try:
            auc = roc_auc_score(self.y_test, probs)
        except ValueError:
            auc = 0.5  # only one class in test set (degenerate)

        ece = compute_ece(probs, self.y_test)

        return {
            "accuracy": acc,
            "auc":      auc,
            "bal_acc":  bal_acc,
            "recall":   recall,
            "ece":      ece,
        }
