# deepal — Complete Setup, Testing & Publishing Guide

> Bob Philip Aila | AIMS Rwanda Thesis

---

## Table of Contents

1. [Local Installation & Quick Test](#1-local-installation--quick-test)
2. [Running All Tests](#2-running-all-tests)
3. [Running the Example Scripts](#3-running-the-example-scripts)
4. [Publishing to GitHub](#4-publishing-to-github)
5. [Publishing to PyPI (pip install deepal6)](#5-publishing-to-pypi)
6. [Using on Kaggle (Notebook)](#6-using-on-kaggle)
7. [Package Structure Reference](#7-package-structure-reference)

---

## 1. Local Installation & Quick Test

### Step 1 — Install

```bash
# Navigate to the package folder
cd deepal6/

# Install in editable mode (changes to source take effect immediately)
pip install -e .

# For image support (NIH Chest X-ray / ResNet-18):
pip install -e ".[image]"
```

### Step 2 — Verify the install

```python
from deepal6 import list_strategies
print(list_strategies())
# Expected: ['Random', 'Entropy', 'Margin', 'BALD', 'CoreSet', 'BADGE']
```

### Step 3 — Run a mini experiment (copy-paste into Python / Jupyter)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepal6 import ActiveLearner, TabularDataModule, ALConfig

# Synthetic data (replace with your real dataset)
X, y = make_classification(n_samples=800, n_features=15,
                            weights=[0.7, 0.3], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# Wrap data
data = TabularDataModule(X_tr, y_tr, X_te, y_te)

# Configure (fast settings for a quick check)
cfg = ALConfig(
    strategy     = ['Random', 'Entropy', 'BALD'],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 10,
    n_seeds      = 3,
    train_epochs = 30,
)

# Run
learner  = ActiveLearner(data, cfg)
results  = learner.run()

# View results
learner.summary_table(results, metric='auc')
learner.plot(results, metric='auc')
```

---

## 2. Running All Tests

```bash
cd deepal6/

# Install test dependencies
pip install pytest pytest-cov

# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=deepal6 --cov-report=term-missing

# Run a specific test
pytest tests/test_tabular.py::test_all_strategies -v
```

Expected output: **14 passed** in ~5 seconds.

---

## 3. Running the Example Scripts

### Tabular (German Credit / synthetic)

```bash
cd deepal6/
python examples/tabular_credit_risk.py
```

This generates synthetic data matching the thesis setup (70/30 class imbalance,
20 features) and runs all 6 strategies for 3 seeds × 20 rounds.

### Image (NIH Chest X-ray)

1. Edit `examples/image_chest_xray.py` and set `DATA_DIR` to your local
   path containing the NIH Chest X-ray images and `Data_Entry_2017.csv`.
2. Install image dependencies: `pip install -e ".[image]"`
3. Run: `python examples/image_chest_xray.py`

### Custom Strategy

```bash
python examples/custom_strategy.py
```

Demonstrates how to register your own query strategy and include it in
the experiment alongside built-in ones.

---

## 4. Publishing to GitHub

### Step 1 — Create a GitHub repository

Go to https://github.com/new and create a repository named `deepal6`.

### Step 2 — Push the code

```bash
cd deepal6/

git init
git add .
git commit -m "Initial release — deepal6 v1.0.0"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/deepal6.git
git push -u origin main
```

### Step 3 — Add a GitHub Actions CI (optional but recommended)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

Push this file and GitHub will automatically run your tests on every push.

### Step 4 — Allow others to install directly from GitHub

```bash
pip install git+https://github.com/YOUR_USERNAME/deepal6.git
```

---

## 5. Publishing to PyPI

This makes `pip install deepal6` work for anyone in the world.

### Step 1 — Create a PyPI account

Go to https://pypi.org/account/register/ and verify your email.

### Step 2 — Install build tools

```bash
pip install build twine
```

### Step 3 — Build the distribution

```bash
cd deepal6/
python -m build
```

This creates a `dist/` folder with:
- `deepal6-1.0.0.tar.gz`  (source distribution)
- `deepal6-1.0.0-py3-none-any.whl`  (wheel)

### Step 4 — Test on TestPyPI first (strongly recommended)

```bash
# Upload to test server
twine upload --repository testpypi dist/*

# Install from test server to verify
pip install --index-url https://test.pypi.org/simple/ deepal
```

### Step 5 — Upload to real PyPI

```bash
twine upload dist/*
# Enter your PyPI username and password when prompted
```

After this, anyone can install with:
```bash
pip install deepal6
pip install "deepal6[image]"   # with image support
```

### Step 6 — Version updates

When you make changes, bump the version in:
- `deepal/__init__.py` → `__version__ = "1.0.1"`
- `pyproject.toml` → `version = "1.0.1"`
- `setup.py` → `version="1.0.1"`

Then rebuild and re-upload:
```bash
python -m build
twine upload dist/*
```

---

## 6. Using on Kaggle

### Option A — Upload as a Kaggle Dataset (recommended for thesis)

1. Zip the `deepal/` folder:
   ```bash
   zip -r deepal_package.zip deepal/
   ```
2. Go to https://www.kaggle.com/datasets → "New Dataset"
3. Upload `deepal_package.zip`
4. In your Kaggle notebook:
   ```python
   import subprocess
   subprocess.run(['pip', 'install', '/kaggle/input/deepal-package/deepal/', '-e', '-q'])

   from deepal6 import ActiveLearner, TabularDataModule, ALConfig
   ```

### Option B — Install directly from GitHub in a Kaggle notebook

```python
# Cell 1 — Install
import subprocess
subprocess.run(['pip', 'install', 'git+https://github.com/YOUR_USERNAME/deepal6.git', '-q'])

# Cell 2 — Use (same as local)
from deepal6 import ActiveLearner, TabularDataModule, ALConfig
```

### Option C — Inline install from PyPI (after Step 5 above)

```python
!pip install deepal6 -q
from deepal6 import ActiveLearner, TabularDataModule, ALConfig
```

### Kaggle notebook example (tabular)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Load German Credit (adjust path to your Kaggle dataset) ──
df = pd.read_csv('/kaggle/input/your-dataset/german_credit_data_clean.csv')

# Encode
df['credit_risk'] = df['credit_risk'].map({'Good': 1, 'Bad': 0})
for col in df.select_dtypes(include='object').columns:
    if col != 'credit_risk':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('credit_risk', axis=1).values
y = df['credit_risk'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── deepal ──
from deepal6 import ActiveLearner, TabularDataModule, ALConfig

data = TabularDataModule(X_train, y_train, X_test, y_test, pos_label=0)

config = ALConfig(
    strategy     = ['Random', 'Entropy', 'Margin', 'BALD', 'CoreSet', 'BADGE'],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 20,
    n_seeds      = 5,
    train_epochs = 50,
)

learner = ActiveLearner(data, config)
results = learner.run()

learner.summary_table(results, metric='auc')
learner.plot(results, metric='auc',     save_path='auc_curves.png')
learner.plot(results, metric='bal_acc', save_path='bal_acc_curves.png')
learner.plot(results, metric='recall',  save_path='recall_curves.png')
learner.plot_calibration(results,       save_path='ece_curves.png')
```

---

## 7. Package Structure Reference

```
deepal/
├── deepal/
│   ├── __init__.py          ← Public API (import everything from here)
│   ├── config.py            ← ALConfig — all experiment parameters
│   ├── learner.py           ← ActiveLearner — main experiment runner
│   ├── metrics.py           ← ECE, AULC, aggregate_seeds, summary table
│   ├── plotting.py          ← Learning curves, gap plot, calibration
│   ├── exceptions.py        ← DeepALError, ConfigurationError, DataError, etc.
│   ├── data/
│   │   ├── base.py          ← BaseDataModule (abstract interface)
│   │   ├── tabular.py       ← TabularDataModule + CreditNet
│   │   └── image.py         ← ImageDataModule + ResNet-18
│   ├── strategies/
│   │   ├── __init__.py      ← STRATEGIES dict + register_strategy()
│   │   ├── registry.py      ← Re-export for config
│   │   └── query.py         ← All 6 strategy implementations
│   └── models/
│       └── __init__.py      ← CreditNet, build_resnet18 re-exports
├── tests/
│   └── test_tabular.py      ← 14 unit/integration tests
├── examples/
│   ├── tabular_credit_risk.py
│   ├── image_chest_xray.py
│   └── custom_strategy.py
├── README.md                ← Full usage documentation
├── INSTRUCTIONS.md          ← This file
├── setup.py
├── pyproject.toml
└── LICENSE
```

### Key design decisions (from thesis)

| Decision | Why |
|---|---|
| Stratified initial draw | Prevents random winning by lucky class balance in L0 |
| Class-weighted BCE | Prevents majority-class collapse on small labeled sets |
| Fresh model each round | No representation drift; tabular = random init, image = ImageNet weights |
| Unified strategy interface | `fn(model, data, pool_idx, n_query, **kwargs) → np.ndarray` |
| DataModule pattern | Decouples data I/O from strategy logic; easy to extend |
| register_strategy() | One-line extension for custom strategies |

---

## Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ConfigurationError: Unknown strategy` | Typo in strategy name | Names are case-sensitive: 'BALD' not 'bald' |
| `DataError: NaN values` | Missing data in X_train | Impute before passing to TabularDataModule |
| `DataError: length mismatch` | X and y different lengths | Check your split code |
| `ModelError: build_model failed` | CUDA OOM | Set `device='cpu'` in ALConfig or reduce train_batch_size |
| `ImportError: torchvision` | Using ImageDataModule without image extras | `pip install "deepal6[image]"` |
| `StrategyError: no Dropout layers` | BALD with dropout_rate=0 | Set dropout_rate > 0 (default 0.3) |

---

## 8. Loading Any Data Type

This section shows exactly how to feed every common data format into `deepal6`.

---

### Tabular Data

**TabularDataModule requires numpy arrays.** Here is how to convert from every common source:

#### From a CSV file (pandas → numpy)

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from deepal6 import TabularDataModule

df = pd.read_csv('your_data.csv')

# 1. Encode target to binary 0/1
df['label'] = df['label'].map({'Good': 1, 'Bad': 0})   # adjust for your labels

# 2. Encode any categorical columns
for col in df.select_dtypes(include='object').columns:
    if col != 'label':
        df[col] = LabelEncoder().fit_transform(df[col])

# 3. Split
X = df.drop('label', axis=1).values   # .values converts DataFrame -> numpy
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Scale (fit on train only — no leakage)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 5. Wrap — ready to go
data = TabularDataModule(X_train, y_train, X_test, y_test)
```

#### From a pandas DataFrame directly

```python
# TabularDataModule needs numpy. Convert with .values
data = TabularDataModule(
    X_train_df.values,   # pandas DataFrame → numpy
    y_train_series.values,
    X_test_df.values,
    y_test_series.values,
)
```

#### From scikit-learn datasets

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepal6 import TabularDataModule

bunch = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    bunch.data, bunch.target, test_size=0.2, stratify=bunch.target)

sc = StandardScaler()
data = TabularDataModule(
    sc.fit_transform(X_train), y_train,
    sc.transform(X_test),      y_test,
)
```

#### From PyTorch tensors

```python
import torch
from deepal6 import TabularDataModule

# If you have tensors, convert to numpy first
data = TabularDataModule(
    X_train_tensor.numpy(), y_train_tensor.numpy(),
    X_test_tensor.numpy(),  y_test_tensor.numpy(),
)
```

#### From a numpy .npy / .npz file

```python
import numpy as np
from deepal6 import TabularDataModule

d = np.load('dataset.npz')
data = TabularDataModule(d['X_train'], d['y_train'], d['X_test'], d['y_test'])
```

---

### Image Data

**ImageDataModule accepts two formats:**

| Format | When to use |
|---|---|
| `pd.DataFrame` with `filepath` + `label` columns | You have image files on disk (JPG/PNG) |
| Any `torch.utils.data.Dataset` with `.get_labels()` | You have a pre-built Dataset (MedMNIST, custom, etc.) |

#### From image files on disk (DataFrame format)

```python
import pandas as pd
from deepal6 import ImageDataModule

# Build a DataFrame pointing to your image files
train_df = pd.DataFrame({
    'filepath': ['/data/train/img001.png', '/data/train/img002.png', ...],
    'label':    [0, 1, ...]   # binary: 0 or 1
})
test_df = pd.DataFrame({
    'filepath': ['/data/test/img100.png', ...],
    'label':    [1, ...]
})

data = ImageDataModule(train_df, test_df)
```

#### From a folder of images (using glob)

```python
import glob, os
import pandas as pd
from sklearn.model_selection import train_test_split
from deepal6 import ImageDataModule

# Folder structure: data/class0/*.png and data/class1/*.png
rows = []
for cls in [0, 1]:
    for path in glob.glob(f'data/class{cls}/*.png'):
        rows.append({'filepath': path, 'label': cls})

df = pd.DataFrame(rows)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

data = ImageDataModule(train_df.reset_index(drop=True),
                       test_df.reset_index(drop=True))
```

#### From MedMNIST (or any in-memory dataset)

```python
# Any PyTorch Dataset works — just add get_labels() to it
import medmnist
from medmnist import INFO
import numpy as np
from deepal6 import ImageDataModule

info  = INFO['bloodmnist']
DClass = getattr(medmnist, info['python_class'])

class MedMNISTBinary(DClass):
    def __init__(self, split):
        super().__init__(split=split, download=True)
        # Binarise labels: classes 4-7 vs 0-3
        self._labels = (self.labels.squeeze() >= 4).astype(int)

    def get_labels(self):          # ← required by deepal
        return self._labels

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, int(self._labels[idx])

data = ImageDataModule(MedMNISTBinary('train'), MedMNISTBinary('test'))
```

#### From a custom in-memory numpy image array

```python
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from deepal6 import ImageDataModule

class NumpyImageDataset(Dataset):
    """Wraps numpy arrays of shape (N, H, W, C) as a deepal-compatible Dataset."""
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images    = images.astype(np.uint8)
        self._labels   = labels.astype(int)
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self._labels[idx])

    def get_labels(self):          # ← required by deepal
        return self._labels

# Usage
X_img  = np.random.randint(0, 255, (500, 32, 32, 3), dtype=np.uint8)
y      = np.random.randint(0, 2, 500)
X_img_test = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
y_test     = np.random.randint(0, 2, 100)

data = ImageDataModule(
    NumpyImageDataset(X_img, y),
    NumpyImageDataset(X_img_test, y_test),
)
```

#### With custom augmentations

```python
from torchvision import transforms
from deepal6 import ImageDataModule

train_aug = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_aug = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data = ImageDataModule(
    train_df, test_df,
    train_transform = train_aug,
    test_transform  = test_aug,
    img_size        = 256,
)
```

---

### Input Format Summary

| Your data | Use | How to load |
|---|---|---|
| CSV file | `TabularDataModule` | `pd.read_csv()` → `.values` → pass numpy |
| pandas DataFrame | `TabularDataModule` | `.values` on X and y columns |
| numpy arrays | `TabularDataModule` | Pass directly |
| PyTorch tensors | `TabularDataModule` | `.numpy()` first |
| sklearn dataset | `TabularDataModule` | Use `bunch.data`, `bunch.target` directly |
| Image files on disk | `ImageDataModule` | Build DataFrame with `filepath` + `label` columns |
| MedMNIST / any Dataset | `ImageDataModule` | Add `get_labels()` method to your Dataset |
| In-memory numpy images | `ImageDataModule` | Wrap in `NumpyImageDataset` (see example above) |
