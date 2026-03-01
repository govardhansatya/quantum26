"""
tests/test_quantum_kernel_svm.py
=================================
Unit + smoke tests for models/quantum_kernel_svm.py

Coverage
--------
- quantum_kernel              : value in [0,1], self-overlap ≈ 1
- compute_kernel_matrix       : shape, diagonal values, symmetry
- QuantumKernelSVM init/fit/predict/evaluate

All tests that invoke PennyLane circuits are marked ``slow``.
Run them with:
    pytest -m slow tests/test_quantum_kernel_svm.py
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


@pytest.fixture(scope="module")
def smoke_vectors():
    """A few 8-dimensional vectors in [0, π] for kernel tests."""
    rng = np.random.RandomState(99)
    return rng.uniform(0, np.pi, (6, 8))


@pytest.fixture(scope="module")
def trained_qksvm(smoke_vectors):
    """QuantumKernelSVM fitted on 4 smoke vectors."""
    from models.quantum_kernel_svm import QuantumKernelSVM

    X_tr = smoke_vectors[:4]
    y_tr = np.array([0, 1, 0, 1])
    model = QuantumKernelSVM(C=1.0)
    model.fit(X_tr, y_tr)
    return model, smoke_vectors


# ---------------------------------------------------------------------------
# quantum_kernel (single pair)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQuantumKernel:
    def test_returns_float(self, smoke_vectors):
        from models.quantum_kernel_svm import quantum_kernel

        k = quantum_kernel(smoke_vectors[0], smoke_vectors[1])
        assert isinstance(k, float)

    def test_value_in_zero_one(self, smoke_vectors):
        from models.quantum_kernel_svm import quantum_kernel

        for i in range(len(smoke_vectors)):
            for j in range(len(smoke_vectors)):
                k = quantum_kernel(smoke_vectors[i], smoke_vectors[j])
                assert 0.0 - 1e-6 <= k <= 1.0 + 1e-6, (
                    f"K({i},{j}) = {k} is outside [0,1]"
                )

    def test_self_overlap_near_one(self, smoke_vectors):
        """K(x, x) should be very close to 1.0 (perfect self-overlap)."""
        from models.quantum_kernel_svm import quantum_kernel

        for i in range(len(smoke_vectors)):
            k_self = quantum_kernel(smoke_vectors[i], smoke_vectors[i])
            assert k_self >= 0.95, f"Self-kernel K({i},{i}) = {k_self} < 0.95"

    def test_symmetry(self, smoke_vectors):
        """K(x1, x2) ≈ K(x2, x1) for the ZZ kernel."""
        from models.quantum_kernel_svm import quantum_kernel

        k12 = quantum_kernel(smoke_vectors[0], smoke_vectors[1])
        k21 = quantum_kernel(smoke_vectors[1], smoke_vectors[0])
        assert abs(k12 - k21) < 0.05, f"Kernel not symmetric: K(0,1)={k12}, K(1,0)={k21}"


# ---------------------------------------------------------------------------
# compute_kernel_matrix
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestComputeKernelMatrix:
    def test_shape(self, smoke_vectors):
        from models.quantum_kernel_svm import compute_kernel_matrix

        K = compute_kernel_matrix(smoke_vectors[:3], smoke_vectors[:4])
        assert K.shape == (3, 4)

    def test_square_shape(self, smoke_vectors):
        from models.quantum_kernel_svm import compute_kernel_matrix

        K = compute_kernel_matrix(smoke_vectors[:4], smoke_vectors[:4])
        assert K.shape == (4, 4)

    def test_diagonal_near_one(self, smoke_vectors):
        from models.quantum_kernel_svm import compute_kernel_matrix

        K = compute_kernel_matrix(smoke_vectors[:4], smoke_vectors[:4])
        for i in range(4):
            assert K[i, i] >= 0.9, f"K[{i},{i}] = {K[i,i]} < 0.9"

    def test_values_in_range(self, smoke_vectors):
        from models.quantum_kernel_svm import compute_kernel_matrix

        K = compute_kernel_matrix(smoke_vectors[:3], smoke_vectors[:3])
        assert K.min() >= 0.0 - 1e-6
        assert K.max() <= 1.0 + 1e-6

    def test_dtype_is_float(self, smoke_vectors):
        from models.quantum_kernel_svm import compute_kernel_matrix

        K = compute_kernel_matrix(smoke_vectors[:2], smoke_vectors[:2])
        assert K.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# QuantumKernelSVM
# ---------------------------------------------------------------------------


class TestQuantumKernelSVMInit:
    def test_default_C(self):
        from models.quantum_kernel_svm import QuantumKernelSVM

        model = QuantumKernelSVM()
        assert model.C == 1.0

    def test_name_attribute(self):
        from models.quantum_kernel_svm import QuantumKernelSVM

        assert isinstance(QuantumKernelSVM.name, str) and len(QuantumKernelSVM.name) > 0

    def test_x_train_none_before_fit(self):
        from models.quantum_kernel_svm import QuantumKernelSVM

        model = QuantumKernelSVM()
        assert model.X_train_ is None

    def test_predict_raises_before_fit(self, smoke_vectors):
        from models.quantum_kernel_svm import QuantumKernelSVM

        model = QuantumKernelSVM()
        with pytest.raises(RuntimeError):
            model.predict(smoke_vectors[:2])

    def test_predict_proba_raises_before_fit(self, smoke_vectors):
        from models.quantum_kernel_svm import QuantumKernelSVM

        model = QuantumKernelSVM()
        with pytest.raises(RuntimeError):
            model.predict_proba(smoke_vectors[:2])


@pytest.mark.slow
class TestQuantumKernelSVMFitPredict:
    def test_fit_returns_self(self, smoke_vectors):
        from models.quantum_kernel_svm import QuantumKernelSVM

        X_tr, y_tr = smoke_vectors[:4], np.array([0, 1, 0, 1])
        model = QuantumKernelSVM(C=1.0)
        assert model.fit(X_tr, y_tr) is model

    def test_x_train_set_after_fit(self, trained_qksvm):
        model, _ = trained_qksvm
        assert model.X_train_ is not None

    def test_predict_shape(self, trained_qksvm, smoke_vectors):
        model, vecs = trained_qksvm
        preds = model.predict(vecs[4:])
        assert preds.shape == (2,)

    def test_predict_binary_values(self, trained_qksvm, smoke_vectors):
        model, vecs = trained_qksvm
        preds = model.predict(vecs[4:])
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, trained_qksvm, smoke_vectors):
        model, vecs = trained_qksvm
        proba = model.predict_proba(vecs[4:])
        assert proba.shape == (2, 2)

    def test_predict_proba_sums_to_one(self, trained_qksvm, smoke_vectors):
        model, vecs = trained_qksvm
        proba = model.predict_proba(vecs[4:])
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(2), atol=1e-4)

    def test_evaluate_returns_required_keys(self, trained_qksvm, smoke_vectors):
        model, vecs = trained_qksvm
        X_te = vecs[4:]
        y_te = np.array([0, 1])
        result = model.evaluate(X_te, y_te)
        for key in ("accuracy", "f1", "precision", "recall"):
            assert key in result
