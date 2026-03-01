"""
tests/test_vqc_model.py
========================
Unit + smoke tests for models/vqc_model.py

Coverage
--------
- VQCClassifier init          : attribute existence and types
- fit                         : runs without error, loss history populated
- predict                     : binary output, correct shape
- predict_proba               : shape, values in [0,1], sums to 1
- evaluate                    : returns required metric keys in valid range
- draw_circuit                : returns non-empty string
- _expval_to_prob helper      : boundary values
- vqc_circuit (QNode)         : returns scalar in [-1, 1]

All quantum circuit tests are marked ``slow`` and skipped by default
unless explicitly requested with: pytest -m slow
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SMOKE = 12  # small dataset for fast CPU simulation

@pytest.fixture(scope="module")
def smoke_data():
    """12 samples in [0,π]^8 with balanced binary labels."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, np.pi, (N_SMOKE, 8))
    y = np.array([0, 1] * (N_SMOKE // 2))
    return X, y


@pytest.fixture(scope="module")
def trained_vqc(smoke_data):
    """VQCClassifier trained for 3 epochs on smoke data."""
    from models.vqc_model import VQCClassifier
    X, y = smoke_data
    model = VQCClassifier(n_qubits=8, n_layers=2, max_iter=3, batch_size=4, lr=0.05)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestVQCInit:
    def test_default_init(self):
        from models.vqc_model import VQCClassifier
        model = VQCClassifier()
        assert model.n_qubits == 8
        assert model.n_layers == 2
        assert model.max_iter > 0
        assert model.batch_size > 0
        assert model.lr > 0

    def test_loss_history_empty_before_fit(self):
        from models.vqc_model import VQCClassifier
        model = VQCClassifier()
        assert model.loss_history == []

    def test_weights_correct_shape(self):
        from models.vqc_model import VQCClassifier, _WEIGHTS_SHAPE
        import pennylane.numpy as pnp
        model = VQCClassifier()
        assert model.weights.shape == _WEIGHTS_SHAPE

    def test_name_is_string(self):
        from models.vqc_model import VQCClassifier
        assert isinstance(VQCClassifier.name, str) and len(VQCClassifier.name) > 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVQCFit:
    def test_fit_returns_self(self, smoke_data):
        from models.vqc_model import VQCClassifier
        X, y = smoke_data
        model = VQCClassifier(max_iter=2, batch_size=4)
        assert model.fit(X, y) is model

    def test_loss_history_populated(self, trained_vqc):
        assert len(trained_vqc.loss_history) > 0

    def test_loss_history_values_are_finite(self, trained_vqc):
        for loss in trained_vqc.loss_history:
            assert np.isfinite(loss), f"Non-finite loss: {loss}"

    def test_fit_with_validation_data(self, smoke_data):
        from models.vqc_model import VQCClassifier
        X, y = smoke_data
        model = VQCClassifier(max_iter=2, batch_size=4)
        model.fit(X[:8], y[:8], X_val=X[8:], y_val=y[8:])
        assert len(model.loss_history) > 0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVQCPredict:
    def test_predict_shape(self, trained_vqc, smoke_data):
        X, y = smoke_data
        preds = trained_vqc.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_binary_values(self, trained_vqc, smoke_data):
        X, y = smoke_data
        preds = trained_vqc.predict(X)
        assert set(preds).issubset({0, 1}), f"Non-binary predictions: {set(preds)}"

    def test_predict_proba_shape(self, trained_vqc, smoke_data):
        X, _ = smoke_data
        proba = trained_vqc.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_in_zero_one(self, trained_vqc, smoke_data):
        X, _ = smoke_data
        proba = trained_vqc.predict_proba(X)
        assert proba.min() >= 0.0, f"proba min {proba.min()} < 0"
        assert proba.max() <= 1.0 + 1e-6, f"proba max {proba.max()} > 1"

    def test_predict_proba_sums_to_one(self, trained_vqc, smoke_data):
        X, _ = smoke_data
        proba = trained_vqc.predict_proba(X)
        np.testing.assert_allclose(
            proba.sum(axis=1), np.ones(len(X)), atol=1e-5
        )

    def test_predict_single_sample(self, trained_vqc, smoke_data):
        X, _ = smoke_data
        preds = trained_vqc.predict(X[:1])
        assert preds.shape == (1,)
        assert preds[0] in {0, 1}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVQCEvaluate:
    def test_evaluate_returns_dict(self, trained_vqc, smoke_data):
        X, y = smoke_data
        result = trained_vqc.evaluate(X, y)
        assert isinstance(result, dict)

    def test_evaluate_has_required_keys(self, trained_vqc, smoke_data):
        X, y = smoke_data
        result = trained_vqc.evaluate(X, y)
        for key in ("accuracy", "f1", "precision", "recall"):
            assert key in result, f"Missing key: {key}"

    def test_evaluate_values_in_range(self, trained_vqc, smoke_data):
        X, y = smoke_data
        result = trained_vqc.evaluate(X, y)
        for key in ("accuracy", "f1", "precision", "recall"):
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} not in [0,1]"


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVQCCircuit:
    def test_draw_circuit_returns_nonempty_string(self):
        from models.vqc_model import VQCClassifier
        model = VQCClassifier()
        diagram = model.draw_circuit()
        assert isinstance(diagram, str) and len(diagram) > 10

    def test_vqc_circuit_output_in_range(self):
        """vqc_circuit should return a value in [-1, 1]."""
        import pennylane.numpy as pnp
        from models.vqc_model import vqc_circuit, N_QUBITS, _WEIGHTS_SHAPE

        dummy_inputs  = pnp.array(np.zeros(N_QUBITS), requires_grad=False)
        dummy_weights = pnp.array(np.zeros(_WEIGHTS_SHAPE), requires_grad=False)
        result = float(vqc_circuit(dummy_inputs, dummy_weights))
        assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6, f"Circuit output {result} out of [-1,1]"

    def test_vqc_circuit_random_input_in_range(self):
        """Random input should still produce output in [-1, 1]."""
        import pennylane.numpy as pnp
        from models.vqc_model import vqc_circuit, N_QUBITS, _WEIGHTS_SHAPE

        rng = np.random.RandomState(7)
        inputs  = pnp.array(rng.uniform(0, np.pi, N_QUBITS), requires_grad=False)
        weights = pnp.array(rng.normal(0, 0.1, _WEIGHTS_SHAPE), requires_grad=True)
        result = float(vqc_circuit(inputs, weights))
        assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestExpvalToProb:
    def test_minus_one_to_zero(self):
        from models.vqc_model import _expval_to_prob
        assert _expval_to_prob(-1.0) == pytest.approx(0.0)

    def test_plus_one_to_one(self):
        from models.vqc_model import _expval_to_prob
        assert _expval_to_prob(1.0) == pytest.approx(1.0)

    def test_zero_to_half(self):
        from models.vqc_model import _expval_to_prob
        assert _expval_to_prob(0.0) == pytest.approx(0.5)

    def test_range(self):
        from models.vqc_model import _expval_to_prob
        for v in np.linspace(-1, 1, 21):
            p = _expval_to_prob(v)
            assert 0.0 <= p <= 1.0
