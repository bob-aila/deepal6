"""
tests/test_tabular.py
Run: pytest tests/ -v
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepal6 import ActiveLearner, TabularDataModule, ALConfig
from deepal6.exceptions import ConfigurationError, DataError


@pytest.fixture
def data():
    X, y = make_classification(n_samples=200, n_features=10, weights=[0.7,0.3], random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    return TabularDataModule(sc.fit_transform(X_tr), y_tr, sc.transform(X_te), y_te)

def _fast_cfg(**kwargs):
    defaults = dict(initial_size=20, batch_size=10, n_rounds=2, n_seeds=1, train_epochs=3, verbose=False)
    defaults.update(kwargs)
    return ALConfig(**defaults)

def test_data_properties(data):
    assert data.n_train == 160
    assert len(data.labels) == 160

def test_build_model(data):
    assert data.build_model(_fast_cfg(strategy='Random')) is not None

def test_random_runs(data):
    results = ActiveLearner(data, _fast_cfg(strategy='Random')).run()
    assert 'Random' in results
    assert len(results['Random']['auc']['mean']) == 2

def test_all_strategies(data):
    results = ActiveLearner(data, _fast_cfg()).run()
    for s in ['Random','Entropy','Margin','BALD','CoreSet','BADGE']:
        assert s in results, f"{s} missing"

def test_summary_table(data, capsys):
    results = ActiveLearner(data, _fast_cfg(strategy=['Random','BALD'])).run()
    ActiveLearner(data, _fast_cfg(strategy=['Random','BALD'])).summary_table(results)
    captured = capsys.readouterr()
    assert 'Random' in captured.out
    assert 'BALD' in captured.out

def test_unknown_strategy():
    with pytest.raises(ConfigurationError, match="Unknown strategy"):
        ALConfig(strategy='GhostStrategy')

def test_bad_initial_size():
    with pytest.raises(ConfigurationError, match="initial_size"):
        ALConfig(initial_size=1)

def test_nan_raises():
    X = np.array([[1.0, float('nan')],[2.0,3.0]])
    with pytest.raises(DataError, match="NaN"):
        TabularDataModule(X, np.array([0,1]), X, np.array([0,1]))

def test_length_mismatch():
    with pytest.raises(DataError, match="length mismatch"):
        TabularDataModule(np.ones((10,5)), np.zeros(8), np.ones((5,5)), np.zeros(5))

def test_feature_mismatch():
    with pytest.raises(DataError, match="feature counts"):
        TabularDataModule(np.ones((10,5)), np.zeros(10), np.ones((5,3)), np.zeros(5))

def test_custom_strategy(data):
    from deepal6.strategies import register_strategy
    def dummy(model, data, pool_idx, n_query, **kw):
        return np.arange(min(n_query, len(pool_idx)))
    register_strategy('Dummy', dummy)
    results = ActiveLearner(data, _fast_cfg(strategy='Dummy')).run()
    assert 'Dummy' in results

def test_run_strategy_single(data):
    cfg = _fast_cfg(strategy=['Random'])
    raw = ActiveLearner(data, cfg).run_strategy('Random', seed=0)
    assert 'aucs' in raw
    assert len(raw['aucs']) == 2

def test_config_total_budget():
    cfg = ALConfig(strategy='Random', initial_size=50, batch_size=20, n_rounds=10)
    assert cfg.total_budget == 250

def test_initial_size_too_large(data):
    from deepal6.exceptions import ConfigurationError
    cfg = _fast_cfg(strategy='Random', initial_size=170)  # > 160 train samples
    with pytest.raises(ConfigurationError, match="initial_size"):
        ActiveLearner(data, cfg).run()
