"""
deepal.data.image
-----------------
ImageDataModule wraps a PyTorch Dataset (or a pandas DataFrame with
filepath + label columns) and provides all the hooks the active learning
loop and query strategies need.

Model: ResNet-18 pretrained on ImageNet with a custom Dropout head,
mirroring the thesis Chapter 5 (healthcare / NIH Chest X-ray) design.

Re-initialisation policy: model is reset to ImageNet pretrained weights
at each AL round — prevents representation drift while preserving feature
extractor quality.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional, List, Union

try:
    from torchvision import transforms, models
    from torchvision.models import ResNet18_Weights
    from PIL import Image
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
)

from deepal6.data.base import BaseDataModule
from deepal6.exceptions import DataError, ModelError
from deepal6.metrics import compute_ece

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _check_torchvision():
    if not _TORCHVISION_AVAILABLE:
        raise ImportError(
            "ImageDataModule requires torchvision and Pillow.\n"
            "Install with: pip install torchvision Pillow"
        )


# ── Default transforms ─────────────────────────────────────────────────────────

def default_train_transform(img_size: int = 256):
    """Standard training augmentation used in thesis (horizontal flip + ±5° rotation)."""
    _check_torchvision()
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),   # works for RGB too
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_test_transform(img_size: int = 256):
    """Deterministic test transform — no augmentation."""
    _check_torchvision()
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Built-in Dataset wrapper ───────────────────────────────────────────────────

class _DataFrameDataset(Dataset):
    """
    Minimal PyTorch Dataset over a pandas DataFrame with
    'filepath' and 'label' columns.
    """
    def __init__(self, df, transform=None):
        _check_torchvision()
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self._labels   = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])

    def get_labels(self):
        return self._labels


# ── ResNet-18 with MC Dropout head ────────────────────────────────────────────

def _build_resnet18(dropout_rate: float = 0.4, device=None) -> nn.Module:
    """
    ResNet-18 (ImageNet pretrained) with a custom dropout head.

    Architecture:
        Frozen: conv1, bn1, layer1, layer2  ← general visual features
        Trainable: layer3, layer4, avgpool, custom FC head
        FC head: Dropout → Linear(512,256) → ReLU → Dropout → Linear(256,1) → Sigmoid

    Reset to ImageNet pretrained weights each AL round.
    """
    _check_torchvision()
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers to prevent overfitting on small labeled sets
    for name, param in model.named_parameters():
        if any(k in name for k in ["layer1", "layer2", "conv1", "bn1"]):
            param.requires_grad = False

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    if device is not None:
        model = model.to(device)
    return model


class ImageDataModule(BaseDataModule):
    """
    DataModule for image datasets using ResNet-18.

    Accepts either:
    A) A pandas DataFrame with 'filepath' and 'label' columns
       (same format as the thesis NIH Chest X-ray notebook), or
    B) Any PyTorch Dataset that has a .get_labels() method returning
       a numpy array of integer labels.

    Parameters
    ----------
    train_data : pd.DataFrame or torch.utils.data.Dataset
        Training data. If DataFrame, must have 'filepath' and 'label' columns.
    test_data : pd.DataFrame or torch.utils.data.Dataset
        Test data. Same format requirements as train_data.
    train_transform : callable or None
        torchvision transform pipeline for training. If None, uses
        default_train_transform(img_size).
    test_transform : callable or None
        torchvision transform pipeline for test. If None, uses
        default_test_transform(img_size).
    img_size : int
        Image resize target (both dimensions). Default: 256.
    pos_label : int
        Positive (minority) class label for recall computation. Default: 1.
    num_workers : int
        DataLoader worker count. Default: 2.
    device : str or None
        'cpu', 'cuda', or None (auto-detect). Default: None.
    dropout_rate : float
        Dropout probability in ResNet-18 head. Default: 0.4.

    Examples
    --------
    # From a DataFrame (filepath + label columns)
    data = ImageDataModule(train_df, test_df)

    # From an existing PyTorch Dataset with get_labels()
    data = ImageDataModule(my_train_ds, my_test_ds, img_size=224)

    # With custom augmentations
    from torchvision import transforms
    aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    data = ImageDataModule(train_df, test_df, train_transform=aug)
    """

    def __init__(
        self,
        train_data,
        test_data,
        train_transform=None,
        test_transform=None,
        img_size: int = 256,
        pos_label: int = 1,
        num_workers: int = 2,
        device: Optional[str] = None,
        dropout_rate: float = 0.4,
    ):
        _check_torchvision()
        self.img_size     = img_size
        self.pos_label    = pos_label
        self.num_workers  = num_workers
        self.dropout_rate = dropout_rate

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        train_tf = train_transform or default_train_transform(img_size)
        test_tf  = test_transform  or default_test_transform(img_size)

        self.train_dataset = self._wrap(train_data, train_tf, "train_data")
        self.test_dataset  = self._wrap(test_data,  test_tf,  "test_data")

    @staticmethod
    def _wrap(data, transform, name):
        """Wrap a DataFrame or pass through an existing Dataset."""
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                if "filepath" not in data.columns or "label" not in data.columns:
                    raise DataError(
                        f"{name} DataFrame must have 'filepath' and 'label' columns. "
                        f"Found: {list(data.columns)}."
                    )
                return _DataFrameDataset(data, transform)
        except ImportError:
            pass  # pandas not installed — must be a Dataset

        if isinstance(data, Dataset):
            if not hasattr(data, "get_labels"):
                raise DataError(
                    f"{name} is a PyTorch Dataset but is missing a "
                    f".get_labels() method that returns a numpy array of labels.\n"
                    f"Add: def get_labels(self): return self._labels"
                )
            return data

        raise DataError(
            f"{name} must be a pandas DataFrame or a PyTorch Dataset. "
            f"Got {type(data).__name__}."
        )

    # ── BaseDataModule interface ───────────────────────────────────────────────

    @property
    def labels(self) -> np.ndarray:
        return self.train_dataset.get_labels()

    @property
    def n_train(self) -> int:
        return len(self.train_dataset)

    def build_model(self, config) -> nn.Module:
        """
        Build a fresh ResNet-18 (reset to ImageNet weights each AL round).
        Respects config.dropout_rate.
        """
        try:
            return _build_resnet18(
                dropout_rate=config.dropout_rate,
                device=self._device,
            )
        except Exception as e:
            raise ModelError(
                f"Failed to build ResNet-18: {e}\n"
                f"Tip: check your internet connection — pretrained weights "
                f"are downloaded on first use."
            ) from e

    def train_model(self, model: nn.Module, labeled_idx, config) -> None:
        """
        Fine-tune on the current labeled subset with class-weighted BCE.
        Uses Adam + CosineAnnealingLR (smoother than StepLR for fine-tuning).
        """
        if len(labeled_idx) == 0:
            raise DataError("train_model: labeled_idx is empty.")

        model.train()
        subset = Subset(self.train_dataset, list(labeled_idx))
        loader = DataLoader(
            subset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self._device.type == "cuda",
            drop_last=len(labeled_idx) > config.train_batch_size,
        )

        labels = self.train_dataset.get_labels()[list(labeled_idx)]
        n_neg  = (labels == 0).sum()
        n_pos  = (labels == 1).sum()
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        ).to(self._device)
        criterion = nn.BCELoss()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train_epochs
        )

        for _ in range(config.train_epochs):
            for imgs, lbls in loader:
                imgs = imgs.to(self._device)
                lbls = lbls.float().unsqueeze(1).to(self._device)
                optimizer.zero_grad()
                out = model(imgs)
                weight_mask = torch.where(lbls == 1, pos_weight, torch.ones_like(lbls))
                loss = (criterion(out, lbls) * weight_mask).mean()
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
        Predict probabilities via a DataLoader over indexed pool samples.

        mc_passes=1  → shape (N,).
        mc_passes>1  → MC Dropout, shape (mc_passes, N).
        """
        subset = Subset(self.train_dataset, list(indices))
        loader = DataLoader(
            subset,
            batch_size=64,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._device.type == "cuda",
        )

        def _single_pass():
            model.eval()
            all_p = []
            with torch.no_grad():
                for imgs, _ in loader:
                    p = model(imgs.to(self._device)).squeeze().cpu().numpy()
                    all_p.append(np.atleast_1d(p))
            return np.concatenate(all_p)

        if mc_passes == 1:
            return _single_pass()
        else:
            model.eval()
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            return np.stack([_single_pass() for _ in range(mc_passes)], axis=0)

    def get_embeddings(self, model: nn.Module, indices) -> np.ndarray:
        """
        Extract ResNet-18 avgpool embeddings (512-dim) via a forward hook.
        """
        model.eval()
        subset = Subset(self.train_dataset, list(indices))
        loader = DataLoader(
            subset,
            batch_size=64,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._device.type == "cuda",
        )

        embeddings = []
        activation = {}

        hook = model.avgpool.register_forward_hook(
            lambda m, inp, out: activation.update(
                {"embed": out.squeeze(-1).squeeze(-1).detach().cpu().numpy()}
            )
        )
        with torch.no_grad():
            for imgs, _ in loader:
                model(imgs.to(self._device))
                embeddings.append(activation["embed"].copy())
        hook.remove()

        return np.vstack(embeddings)

    def get_gradient_embeddings(self, model: nn.Module, indices) -> np.ndarray:
        """
        BADGE gradient embeddings over the last linear layer weights.
        """
        model.eval()
        subset = Subset(self.train_dataset, list(indices))
        loader = DataLoader(subset, batch_size=1, shuffle=False,
                            num_workers=0)  # batch=1 for per-sample gradients

        # Find last linear layer
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ModelError("get_gradient_embeddings: no nn.Linear found.")

        grad_embeddings = []
        for imgs, _ in loader:
            imgs = imgs.to(self._device)
            out  = model(imgs)
            p    = out.item()
            y_hat = torch.tensor(
                [[1.0 if p > 0.5 else 0.0]], dtype=torch.float32
            ).to(self._device)
            model.zero_grad()
            nn.BCELoss()(model(imgs), y_hat).backward()
            if last_linear.weight.grad is not None:
                g = last_linear.weight.grad.detach().cpu().numpy().flatten()
            else:
                g = np.zeros(last_linear.weight.numel())
            grad_embeddings.append(g)

        return np.stack(grad_embeddings)

    def evaluate(self, model: nn.Module) -> dict:
        """Evaluate on the full test dataset."""
        loader = DataLoader(
            self.test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._device.type == "cuda",
        )
        model.eval()
        all_probs = []
        with torch.no_grad():
            for imgs, _ in loader:
                p = model(imgs.to(self._device)).squeeze().cpu().numpy()
                all_probs.append(np.atleast_1d(p))
        probs  = np.concatenate(all_probs)
        preds  = (probs > 0.5).astype(int)
        labels = self.test_dataset.get_labels()

        acc     = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        recall  = recall_score(labels, preds, pos_label=self.pos_label, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5
        ece = compute_ece(probs, labels)

        return {
            "accuracy": acc,
            "auc":      auc,
            "bal_acc":  bal_acc,
            "recall":   recall,
            "ece":      ece,
        }
