# DeepAL6 — Deep Active Learning Library

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-1.0.0-purple" alt="Version">
  <img src="https://img.shields.io/badge/strategies-6-red" alt="Strategies">
</p>

<p align="center">
  <b>Bob Philip Aila · AIMS Rwanda</b><br>
  Pool-based deep active learning: 6 query strategies benchmarked against a random baseline,<br>
  for tabular and image domains.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Tabular Data](#tabular-german-credit--any-binary-classification)
  - [Image Data](#image-nih-chest-x-ray--any-binary-image-dataset)
- [Strategies](#strategies)
- [ALConfig Parameters](#alconfig-parameters)
- [Available Plots](#available-plots)
- [Custom Strategies](#custom-strategies)
- [Batch Size Ablation](#batch-size-ablation)
- [Loading Any Data Type](#loading-any-data-type)
- [Design Principles](#design-principles)
- [Project Structure](#project-structure)
- [Common Errors](#common-errors)
- [Citation](#citation)
- [License](#license)

---

## Overview

`deepal6` is a flexible, research-grade active learning framework built for reproducible experimentation. It lets you:

- Run **6 query strategies** — Random, Entropy, Margin, BALD, CoreSet, BADGE — under identical budget protocols
- Work with **tabular data** (CreditNet) or **image data** (ResNet-18 + Dropout head)
- Control every parameter: initial size, batch size, rounds, seeds, augmentations, and more
- Get **publication-quality plots** — learning curves, strategy-vs-random gap, calibration (ECE)
- Extend with **custom strategies** via a one-line registration API
- Compare strategies fairly with **stratified initial draws** and **class-weighted loss**

---

## Installation

### Standard install (recommended)

```bash
pip install deepal6
```

### With image support (ResNet-18 / NIH Chest X-ray)

```bash
pip install "deepal6[image]"
```

### Development install (editable, from source)

```bash
git clone https://github.com/bob-aila/deepal6.git
cd deepal6
pip install -e .

# With image support
pip install -e ".[image]"
```

### Verify the install

```python
from deepal6 import list_strategies
print(list_strategies())
# ['Random', 'Entropy', 'Margin', 'BALD', 'CoreSet', 'BADGE']
```

### Requirements

| Package | Minimum version |
|---|---|
| Python | 3.8+ |
| PyTorch | 1.12+ |
| scikit-learn | 1.0+ |
| numpy | 1.21+ |
| matplotlib | 3.4+ |
| torchvision *(image only)* | 0.13+ |
| Pillow *(image only)* | 9.0+ |

---

## Quick Start

### Tabular (German Credit / any binary classification)

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deepal6 import ActiveLearner, TabularDataModule, ALConfig

# 1. Prepare data — scale your features, binary labels 0/1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2. Wrap in DataModule
data = TabularDataModule(X_train, y_train, X_test, y_test, pos_label=0)

# 3. Configure experiment
config = ALConfig(
    strategy     = ['Random', 'Entropy', 'BALD', 'CoreSet', 'BADGE'],
    initial_size = 50,    # stratified initial labeled set
    batch_size   = 20,    # samples queried per round
    n_rounds     = 20,    # AL rounds
    n_seeds      = 5,     # independent runs for mean ± std
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
    train_df, test_df,
    train_transform = my_aug,
    img_size        = 256,
    pos_label       = 1,        # minority / positive class
)

config = ALConfig(
    strategy     = ['Random', 'BALD', 'CoreSet', 'BADGE'],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 20,
    n_seeds      = 5,
    train_epochs = 10,          # fewer epochs — fine-tuning pretrained weights
    lr           = 1e-4,        # lower LR for ResNet-18
    dropout_rate = 0.4,
)

learner = ActiveLearner(data, config)
results = learner.run()
learner.plot(results, metric='auc', show_std=True)
```

---

## Strategies

| Strategy | Type | Selection Criterion | When it shines |
|---|---|---|---|
| **Random** | Baseline | Uniform draw — no model used | Always run as lower bound |
| **Entropy** | Uncertainty | Highest Shannon entropy H[y\|x] | Well-calibrated models |
| **Margin** | Uncertainty | Smallest gap \|p − 0.5\| | Fast, low compute |
| **BALD** | Bayesian | Mutual information I[y;θ\|x,D] via MC Dropout | Best epistemic uncertainty estimate |
| **CoreSet** | Diversity | Greedy k-center in embedding space | Rich embeddings (image > tabular) |
| **BADGE** | Hybrid | Gradient embeddings + k-means++ | Combines uncertainty + diversity |

> **Note:** BALD requires `dropout_rate > 0` (default 0.3). All strategies use the same
> budget, training procedure, and evaluation protocol so results are directly comparable.

---

## ALConfig Parameters

```python
from deepal6 import ALConfig

config = ALConfig(
    strategy         = ['Random', 'BALD'],  # or a single string: 'BALD'
    initial_size     = 50,
    batch_size       = 20,
    n_rounds         = 20,
    n_seeds          = 5,
    train_epochs     = 50,
    lr               = 1e-3,
    weight_decay     = 1e-4,
    dropout_rate     = 0.3,
    mc_passes        = 20,
    train_batch_size = 32,
    device           = None,    # auto-detect GPU/CPU
    seed             = 42,
    verbose          = True,
    save_checkpoints = False,
    checkpoint_dir   = './checkpoints',
)
print(config.summary())        # prints a formatted parameter table
print(config.total_budget)     # initial_size + n_rounds * batch_size
```

| Parameter | Default | Description |
|---|---|---|
| `strategy` | all 6 | Strategy name(s) to run |
| `initial_size` | 50 | Stratified initial labeled set size |
| `batch_size` | 20 | Samples queried per AL round |
| `n_rounds` | 20 | Maximum AL rounds |
| `n_seeds` | 5 | Independent runs per strategy (for mean ± std) |
| `train_epochs` | 50 | Training epochs per round |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-4 | L2 regularisation |
| `dropout_rate` | 0.3 | Dropout probability (also controls BALD MC stochasticity) |
| `mc_passes` | 20 | MC Dropout forward passes for BALD |
| `train_batch_size` | 32 | Mini-batch size during model training |
| `device` | auto | `'cpu'`, `'cuda'`, or `None` (auto-detect) |
| `seed` | 42 | Base seed; each of `n_seeds` runs uses `seed + i` |
| `verbose` | True | Print per-round metrics during experiment |
| `save_checkpoints` | False | Save best model checkpoint per strategy per seed |
| `checkpoint_dir` | `'./checkpoints'` | Directory for saved checkpoints |
| `extra_strategy_kwargs` | `{}` | Extra kwargs forwarded to strategy functions |

---

## Available Plots

```python
# Learning curves — metric vs labelling budget
learner.plot(results, metric='auc')          # AUC-ROC
learner.plot(results, metric='bal_acc')      # Balanced accuracy
learner.plot(results, metric='recall')       # Recall (minority class)
learner.plot(results, metric='ece')          # Calibration error

# Strategy vs random gap (automatically shown when 'Random' is in results)
# — positive gap means the strategy beats random at that budget point

# ECE calibration detail
learner.plot_calibration(results)

# Printed summary table
learner.summary_table(results, metric='auc')

# Save any plot to file
learner.plot(results, metric='auc', save_path='results/auc_curves.png')
```

All plots support `show_std=True` (default) to shade ± 1 std band across seeds.

---

## Custom Strategies

Register your own query strategy with one line and use it alongside the built-ins:

```python
import numpy as np
from deepal6.strategies import register_strategy
from deepal6 import ActiveLearner, ALConfig

def my_strategy(model, data, pool_idx, n_query, **kwargs):
    """
    Custom query strategy.

    Parameters
    ----------
    model    : trained PyTorch model
    data     : TabularDataModule or ImageDataModule
    pool_idx : list of global indices into the unlabeled pool
    n_query  : number of samples to select
    **kwargs : extra kwargs from ALConfig.extra_strategy_kwargs

    Returns
    -------
    np.ndarray of LOCAL indices into pool_idx (not global indices)
    """
    probs = data.predict_proba(model, pool_idx)
    # your selection logic here ...
    scores = np.abs(probs - 0.5)
    return np.argsort(scores)[:n_query]

register_strategy('MyStrategy', my_strategy)

# Use exactly like any built-in strategy
config = ALConfig(strategy=['Random', 'BALD', 'MyStrategy'])
results = ActiveLearner(data, config).run()
```

---

## Batch Size Ablation

Study how query batch size affects learning efficiency for your best strategy:

```python
from deepal6 import ActiveLearner, ALConfig
from deepal6.plotting import plot_batch_size_ablation

ablation = {}
for b in [10, 20, 50]:
    cfg = ALConfig(strategy='BALD', batch_size=b, n_seeds=3)
    r   = ActiveLearner(data, cfg).run()
    ablation[b] = r['BALD']

plot_batch_size_ablation(ablation, metric='auc')
```

---

## Loading Any Data Type

### Tabular

`TabularDataModule` requires **numpy arrays**. Here is how to convert from every common source:

```python
# From a CSV file
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from deepal6 import TabularDataModule

df = pd.read_csv('your_data.csv')
df['label'] = df['label'].map({'Good': 1, 'Bad': 0})   # encode target to 0/1
for col in df.select_dtypes(include='object').columns:
    if col != 'label':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('label', axis=1).values   # .values → numpy
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train only — no leakage
X_test  = scaler.transform(X_test)

data = TabularDataModule(X_train, y_train, X_test, y_test)
```

| Source | One-liner |
|---|---|
| pandas DataFrame | `X_train_df.values` |
| PyTorch tensor | `X_train_tensor.numpy()` |
| sklearn dataset | `bunch.data`, `bunch.target` directly |
| `.npy` / `.npz` file | `np.load('file.npz')['X_train']` |

### Image

`ImageDataModule` accepts two formats:

**A) DataFrame with `filepath` + `label` columns** (images on disk):

```python
import pandas as pd
from deepal6 import ImageDataModule

train_df = pd.DataFrame({
    'filepath': ['/data/train/img001.png', ...],
    'label':    [0, 1, ...]
})
data = ImageDataModule(train_df, test_df)
```

**B) Any PyTorch `Dataset` with a `.get_labels()` method** (in-memory / MedMNIST):

```python
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class NumpyImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images  = images.astype(np.uint8)
        self._labels = labels.astype(int)
        self.tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.images[i])), int(self._labels[i])
    def get_labels(self):   # ← required by deepal6
        return self._labels

data = ImageDataModule(
    NumpyImageDataset(X_train_imgs, y_train),
    NumpyImageDataset(X_test_imgs,  y_test),
)
```

---

## Design Principles

| Decision | Reason |
|---|---|
| **Stratified initial draw** | Prevents random baseline winning by lucky class balance in L₀ |
| **Class-weighted BCE loss** | Prevents majority-class collapse on small labeled sets |
| **Fresh model each round** | No representation drift — tabular: random init, image: ImageNet weights |
| **Unified strategy interface** | `fn(model, data, pool_idx, n_query, **kwargs) → np.ndarray` |
| **DataModule pattern** | Decouples data I/O from strategy logic — easy to extend |
| **Fail-fast validation** | All config/data errors surface before the experiment starts |

---

## Project Structure

```
deepal6/
├── __init__.py          ← Public API — import everything from here
├── config.py            ← ALConfig — all experiment parameters + validation
├── learner.py           ← ActiveLearner — main experiment loop
├── metrics.py           ← ECE, AULC, aggregate_seeds, summary table
├── plotting.py          ← Learning curves, gap plot, calibration plots
├── exceptions.py        ← DeepAL6Error, ConfigurationError, DataError, ...
├── data/
│   ├── base.py          ← BaseDataModule (abstract interface)
│   ├── tabular.py       ← TabularDataModule + CreditNet architecture
│   └── image.py         ← ImageDataModule + ResNet-18 with Dropout head
├── strategies/
│   ├── __init__.py      ← STRATEGIES registry + register_strategy()
│   ├── registry.py      ← Re-export for config
│   └── query.py         ← All 6 strategy implementations
└── models/
    └── __init__.py      ← CreditNet, build_resnet18 re-exports
```

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `ConfigurationError: Unknown strategy` | Typo in strategy name | Names are case-sensitive: `'BALD'` not `'bald'` |
| `DataError: NaN values` | Missing values in features | Impute or drop NaNs before passing to `TabularDataModule` |
| `DataError: length mismatch` | X and y have different lengths | Check your train/test split code |
| `DataError: feature counts` | X_train and X_test have different columns | Apply the same scaler/encoder to both |
| `ModelError: build_model failed` | CUDA out of memory | Set `device='cpu'` or reduce `train_batch_size` |
| `StrategyError: no Dropout layers` | BALD with `dropout_rate=0` | Set `dropout_rate > 0` (default: 0.3) |
| `ImportError: torchvision` | Image support not installed | `pip install "deepal6[image]"` |
| `DataError: missing get_labels()` | Custom Dataset missing method | Add `def get_labels(self): return self._labels` |

---

## Citation

If you use DeepAL6 in your research, please cite:

```bibtex
@misc{aila2025deepal6,
  authors      = {Bob Philip Aila, Yae Gaba},
  title        = {DeepAL6: A Deep Active Learning Library for Tabular and Image Domains},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/bob-aila/deepal6}},
  note         = {AIMS Rwanda Master's Thesis}
}
```

> The full citation will be updated once the thesis is published. Check back here or contact the author.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
<p align="center">
  Lest save the budgets by Labeling data informatively.
</p>
