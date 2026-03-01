"""
tests/test_classical_models.py
================================
Unit tests for models/classical_models.py

Coverage
--------
- LogisticRegressionBaseline  : fit, predict, predict_proba, evaluate
- ClassicalSVMBaseline        : fit, predict, predict_proba, evaluate
- NaiveBayesBaseline          : fit, predict, predict_proba, evaluate
- _compute_metrics            : output keys, value ranges
- train_all_baselines         : integration test
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classical_models import (
    ClassicalSVMBaseline,
    LogisticRegressionBaseline,
    NaiveBayesBaseline,
    train_all_baselines,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """80-sample classification dataset with 20 features."""
    X, y = make_classification(
        n_samples=80, n_features=20, n_informative=10,
        n_redundant=4, random_state=42
    )
    return train_test_split(X, y, test_size=0.25, random_state=42)


def _assert_evaluate_dict(result: dict) -> None:
    """Assert the evaluate() return dict has correct keys and value ranges."""
    required_keys = {"accuracy", "f1", "precision", "recall", "confusion_matrix"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"
    for metric in ("accuracy", "f1", "precision", "recall"):
        val = result[metric]
        assert 0.0 <= val <= 1.0, f"{metric}={val} not in [0,1]"
    cm = result["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


# ---------------------------------------------------------------------------
# LogisticRegressionBaseline
# ---------------------------------------------------------------------------


class TestLogisticRegressionBaseline:
    def test_name_attribute(self):
        model = LogisticRegressionBaseline()
        assert isinstance(model.name, str) and len(model.name) > 0

    def test_fit_returns_self(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        result = model.fit(X_tr, y_tr)
        assert result is model

    def test_predict_shape(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == (len(y_te),)

    def test_predict_binary_values(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.shape == (len(y_te), 2)

    def test_predict_proba_sums_to_one(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y_te)), atol=1e-6)

    def test_evaluate_keys_and_range(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        model.fit(X_tr, y_tr)
        result = model.evaluate(X_te, y_te)
        _assert_evaluate_dict(result)

    def test_above_chance_accuracy(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = LogisticRegressionBaseline()
        model.fit(X_tr, y_tr)
        result = model.evaluate(X_te, y_te)
        assert result["accuracy"] > 0.5, "LR should beat random chance on informative features"


# ---------------------------------------------------------------------------
# ClassicalSVMBaseline
# ---------------------------------------------------------------------------


class TestClassicalSVMBaseline:
    def test_name_attribute(self):
        assert isinstance(ClassicalSVMBaseline.name, str)

    def test_fit_returns_self(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline()
        assert model.fit(X_tr, y_tr) is model

    def test_predict_shape(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline()
        model.fit(X_tr, y_tr)
        assert model.predict(X_te).shape == (len(y_te),)

    def test_predict_binary_values(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline()
        model.fit(X_tr, y_tr)
        assert set(model.predict(X_te)).issubset({0, 1})

    def test_predict_proba_shape(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert proba.shape == (len(y_te), 2)

    def test_predict_proba_sums_to_one(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y_te)), atol=1e-4)

    def test_evaluate_keys_and_range(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline()
        model.fit(X_tr, y_tr)
        _assert_evaluate_dict(model.evaluate(X_te, y_te))

    def test_custom_C_parameter(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = ClassicalSVMBaseline(C=10.0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)


# ---------------------------------------------------------------------------
# NaiveBayesBaseline
# ---------------------------------------------------------------------------


class TestNaiveBayesBaseline:
    def test_name_attribute(self):
        assert isinstance(NaiveBayesBaseline.name, str)

    def test_fit_returns_self(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = NaiveBayesBaseline()
        assert model.fit(X_tr, y_tr) is model

    def test_predict_shape(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = NaiveBayesBaseline()
        model.fit(X_tr, y_tr)
        assert model.predict(X_te).shape == (len(y_te),)

    def test_predict_binary_values(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = NaiveBayesBaseline()
        model.fit(X_tr, y_tr)
        assert set(model.predict(X_te)).issubset({0, 1})

    def test_predict_proba_shape(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = NaiveBayesBaseline()
        model.fit(X_tr, y_tr)
        assert model.predict_proba(X_te).shape == (len(y_te), 2)

    def test_predict_proba_sums_to_one(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = NaiveBayesBaseline()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y_te)), atol=1e-6)

    def test_evaluate_keys_and_range(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        model = NaiveBayesBaseline()
        model.fit(X_tr, y_tr)
        _assert_evaluate_dict(model.evaluate(X_te, y_te))


# ---------------------------------------------------------------------------
# train_all_baselines (integration)
# ---------------------------------------------------------------------------


class TestTrainAllBaselines:
    def test_returns_dict(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        results = train_all_baselines(X_tr, y_tr, X_te, y_te)
        assert isinstance(results, dict)

    def test_has_three_models(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        results = train_all_baselines(X_tr, y_tr, X_te, y_te)
        assert len(results) == 3

    def test_each_result_has_accuracy(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        results = train_all_baselines(X_tr, y_tr, X_te, y_te)
        for name, res in results.items():
            assert "accuracy" in res, f"{name} missing 'accuracy'"
            assert 0.0 <= res["accuracy"] <= 1.0

    def test_train_time_recorded(self, synthetic_data):
        X_tr, X_te, y_tr, y_te = synthetic_data
        results = train_all_baselines(X_tr, y_tr, X_te, y_te)
        for name, res in results.items():
            assert "train_time_sec" in res, f"{name} missing 'train_time_sec'"
            assert res["train_time_sec"] >= 0
