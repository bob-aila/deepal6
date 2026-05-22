# DeepAL6 — Deep Active Learning Library

> **Bob Philip Aila | AIMS Rwanda**
> Pool-based deep active learning: 6 query strategies vs. random baseline.

---

## Overview

`deepal6` is a flexible, research-grade active learning framework that lets you:

- Run **6 query strategies** (Random, Entropy, Margin, BALD, CoreSet, BADGE) under identical budget protocols
- Work with **tabular data** (CreditNet) or **image data** (ResNet-18 + Dropout head)
- Control every parameter: initial size, batch size, rounds, seeds, augmentations, and more
- Get publication-quality plots out of the box
- Extend with **custom strategies** via a one-line registration API

---

## Installation

```bash
# Tabular support (default)
pip install -e .

# Image support (adds torchvision + Pillow)
pip install -e ".[image]"
```

---

## Quick Start

### Tabular (German Credit / any binary classification)

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deepal6 import ActiveLearner, TabularDataModule, ALConfig

# 1. Prepare data (already encoded, binary labels 0/1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2. Wrap in DataModule
data = TabularDataModule(X_train, y_train, X_test, y_test, pos_label=0)

# 3. Configure experiment
config = ALConfig(
    strategy    = ['Random', 'Entropy', 'BALD', 'CoreSet', 'BADGE'],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 20,
    n_seeds      = 5,
    train_epochs = 50,
)

# 4. Run
learner = ActiveLearner(data, config)
results = learner.run()

# 5. Visualise
learner.plot(results, metric='auc')
learner.summary_table(results)
```

---

### Image (NIH Chest X-ray / any binary image dataset)

```python
import pandas as pd
from deepal6 import ActiveLearner, ImageDataModule, ALConfig

# DataFrame with 'filepath' and 'label' columns
train_df = pd.DataFrame({'filepath': [...], 'label': [0, 1, ...]})
test_df  = pd.DataFrame({'filepath': [...], 'label': [0, 1, ...]})

# Optional: custom augmentations
from torchvision import transforms
my_aug = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data = ImageDataModule(
    train_df,
    test_df,
    train_transform = my_aug,   # your augmentations
    img_size        = 256,
    pos_label       = 1,        # Pneumonia is positive class
)

config = ALConfig(
    strategy     = ['Random', 'BALD', 'CoreSet', 'BADGE'],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 20,
    n_seeds      = 5,
    train_epochs = 10,          # fewer epochs for fine-tuning
    lr           = 1e-4,        # lower LR for pretrained model
    dropout_rate = 0.4,
)

learner = ActiveLearner(data, config)
results = learner.run()
learner.plot(results, metric='auc', show_std=True)
```

---

## ALConfig Parameters

| Parameter | Default | Description |
|---|---|---|
| `strategy` | all 6 | Strategy name(s): 'Random', 'Entropy', 'Margin', 'BALD', 'CoreSet', 'BADGE' |
| `initial_size` | 50 | Size of stratified initial labeled set |
| `batch_size` | 20 | Samples queried per AL round |
| `n_rounds` | 20 | Maximum AL rounds |
| `n_seeds` | 5 | Independent runs per strategy (for mean±std) |
| `train_epochs` | 50 | Training epochs per round |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-4 | L2 regularisation |
| `dropout_rate` | 0.3 | Dropout probability (also controls BALD MC stochasticity) |
| `mc_passes` | 20 | MC Dropout forward passes for BALD |
| `train_batch_size` | 32 | Mini-batch size during training |
| `device` | auto | 'cpu', 'cuda', or None (auto-detect) |
| `seed` | 42 | Base seed; each of n_seeds runs uses seed+i |
| `verbose` | True | Print per-round metrics |
| `save_checkpoints` | False | Save model checkpoints each round |

---

## Strategies

| Strategy | Type | Selection Criterion |
|---|---|---|
| **Random** | Baseline | Uniform draw — no model information |
| **Entropy** | Uncertainty | Highest Shannon entropy H[y\|x] |
| **Margin** | Uncertainty | Smallest gap between top-2 class probabilities |
| **BALD** | Bayesian | Mutual information I[y;θ\|x,D] via MC Dropout |
| **CoreSet** | Diversity | Greedy k-center covering in embedding space |
| **BADGE** | Hybrid | Gradient embeddings + k-means++ diversity |

---

## Custom Strategies

```python
from deepal6.strategies import register_strategy

def my_strategy(model, data, pool_idx, n_query, **kwargs):
    """
    Custom query strategy.
    Must return np.ndarray of LOCAL indices into pool_idx.
    """
    import numpy as np
    probs = data.predict_proba(model, pool_idx)
    # ... your logic ...
    return np.argsort(my_scores)[-n_query:]

register_strategy('MyStrategy', my_strategy)

# Now use it like any built-in strategy:
config = ALConfig(strategy=['Random', 'BALD', 'MyStrategy'])
```

---

## Batch Size Ablation

```python
from deepal6.plotting import plot_batch_size_ablation

ablation = {}
for b in [10, 20, 50]:
    cfg = ALConfig(strategy='BALD', batch_size=b, n_seeds=3)
    r = ActiveLearner(data, cfg).run()
    ablation[b] = r['BALD']

plot_batch_size_ablation(ablation, metric='auc')
```

---

## Available Plots

```python
learner.plot(results, metric='auc')          # learning curves
learner.plot(results, metric='bal_acc')      # balanced accuracy
learner.plot(results, metric='recall')       # recall (minority class)
learner.plot(results, metric='ece')          # calibration
learner.plot_calibration(results)            # ECE detail
learner.summary_table(results, metric='auc') # printed table
```

---

## Project Structure

```
deepal/
├── __init__.py          # Public API
├── config.py            # ALConfig dataclass
├── learner.py           # ActiveLearner (main loop)
├── metrics.py           # ECE, AULC, aggregate_seeds
├── plotting.py          # All visualisation functions
├── exceptions.py        # Package-specific errors
├── data/
│   ├── base.py          # BaseDataModule abstract interface
│   ├── tabular.py       # TabularDataModule + CreditNet
│   └── image.py         # ImageDataModule + ResNet-18
├── strategies/
│   ├── __init__.py      # STRATEGIES dict + register_strategy()
│   └── query.py         # All 6 strategy implementations
└── models/
    └── __init__.py      # CreditNet, build_resnet18 re-exports
```

---

## Design Principles

1. **Stratified initial draw** — L₀ is always class-balanced to prevent the random baseline winning by luck
2. **Class-weighted BCE** — handles imbalanced datasets; prevents majority-class collapse on small labeled sets
3. **Fresh model each round** — no representation drift; tabular uses random init, image resets to ImageNet weights
4. **Unified strategy interface** — `fn(model, data, pool_idx, n_query, **kwargs) → np.ndarray`
5. **Fail-fast with clear errors** — all validation happens before the experiment starts
