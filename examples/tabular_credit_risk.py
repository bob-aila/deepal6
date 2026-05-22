"""
examples/tabular_credit_risk.py
--------------------------------
Reproduces the thesis Chapter 5 (Finance domain) experiment on the German
Credit dataset (or any binary tabular dataset).

Usage
-----
    python examples/tabular_credit_risk.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Import deepal (install from root: pip install -e .) ──────────────────────
from deepal6 import ActiveLearner, TabularDataModule, ALConfig
from deepal6.plotting import plot_batch_size_ablation


# ── 1. Load & preprocess data ────────────────────────────────────────────────
# Replace this block with your own CSV or data source.
# Generating synthetic data here so the example runs without external files.
from sklearn.datasets import make_classification

SEED = 42
np.random.seed(SEED)

X_raw, y_raw = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    weights=[0.70, 0.30],   # 70% class 0 ("Good"), 30% class 1 ("Bad")
    class_sep=0.8,
    flip_y=0.02,
    random_state=SEED,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Class balance — 0: {(y_train==0).sum()} | 1: {(y_train==1).sum()}")


# ── 2. Wrap in DataModule ────────────────────────────────────────────────────
data = TabularDataModule(
    X_train, y_train,
    X_test,  y_test,
    pos_label=0,   # class 0 is the minority class (Bad credit in thesis)
)


# ── 3. Configure experiment ──────────────────────────────────────────────────
config = ALConfig(
    strategy     = ["Random", "Entropy", "Margin", "BALD", "CoreSet", "BADGE"],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 20,
    n_seeds      = 3,        # use 5 for publication quality
    train_epochs = 50,
    lr           = 1e-3,
    dropout_rate = 0.3,
    mc_passes    = 20,
    verbose      = True,
)

print(config.summary())


# ── 4. Run experiment ────────────────────────────────────────────────────────
learner = ActiveLearner(data, config)
results = learner.run()


# ── 5. Print summary table ───────────────────────────────────────────────────
learner.summary_table(results, metric="auc")
learner.summary_table(results, metric="bal_acc")


# ── 6. Plot learning curves ──────────────────────────────────────────────────
learner.plot(results, metric="auc",     save_path="results_auc.png")
learner.plot(results, metric="bal_acc", save_path="results_bal_acc.png")
learner.plot(results, metric="recall",  save_path="results_recall.png")
learner.plot_calibration(results,       save_path="results_ece.png")


# ── 7. Batch size ablation (top-3 strategies) ────────────────────────────────
ablation = {}
for b in [10, 20, 50]:
    cfg = ALConfig(
        strategy=["BALD", "CoreSet", "BADGE"],
        batch_size=b, n_seeds=3, train_epochs=50, verbose=False,
    )
    r = ActiveLearner(data, cfg).run()
    for name, agg in r.items():
        ablation[f"{name}_b{b}"] = agg

print("\nBatch size ablation complete.")
