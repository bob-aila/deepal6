"""
examples/custom_strategy.py
----------------------------
Shows how to register a custom query strategy and run it alongside
the built-in strategies.

Custom strategy: PowerMargin — raises the margin score to the power k
to sharpen the selection boundary focus.
"""

import numpy as np
from deepal6 import ActiveLearner, TabularDataModule, ALConfig
from deepal6.strategies import register_strategy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Define your custom strategy ──────────────────────────────────────────────
def power_margin_sampling(model, data, pool_idx, n_query, power=2.0, **kwargs):
    """
    Powered margin sampling: |p - 0.5|^k.

    Lower power → closer to standard margin sampling.
    Higher power → focuses only on the most uncertain samples near the boundary.
    """
    probs  = data.predict_proba(model, pool_idx)
    margin = np.abs(probs - 0.5) ** power
    n = min(n_query, len(pool_idx))
    return np.argsort(margin)[:n]


# ── Register it under a custom name ─────────────────────────────────────────
register_strategy("PowerMargin", power_margin_sampling)

# Verify it's now available
from deepal6 import list_strategies
print("Registered strategies:", list_strategies())


# ── Run experiment including the custom strategy ─────────────────────────────
X, y = make_classification(
    n_samples=800, n_features=15, weights=[0.7, 0.3], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

data = TabularDataModule(X_train, y_train, X_test, y_test)

config = ALConfig(
    strategy     = ["Random", "Margin", "PowerMargin"],
    initial_size = 30,
    batch_size   = 10,
    n_rounds     = 15,
    n_seeds      = 3,
    train_epochs = 30,
    verbose      = True,
    extra_strategy_kwargs = {"power": 3.0},  # passed to PowerMargin
)

learner = ActiveLearner(data, config)
results = learner.run()
learner.summary_table(results)
learner.plot(results, metric="auc")
