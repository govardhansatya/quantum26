"""
quantum_kernel_svm.py
=====================
Quantum Kernel SVM (QKSVM) for binary sarcasm detection.

The quantum kernel is defined as:

    K(x1, x2) = |⟨φ(x1) | φ(x2)⟩|² = P(|0…0⟩)

where ``|φ(x)⟩`` is the state prepared by the ZZ feature map and the kernel
circuit computes the overlap between two feature-mapped states.

Using the full 2^8 = 256-dimensional Hilbert space as implicit feature space,
this kernel can capture higher-order interactions that RBF cannot represent.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as _np
import pennylane as qml
from pennylane import numpy as np  # differentiable numpy
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

N_QUBITS = 8
dev_kernel = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Quantum kernel circuit
# ---------------------------------------------------------------------------


@qml.qnode(dev_kernel)
def kernel_circuit(x1, x2):
    """Compute the quantum kernel between two feature vectors.

    The circuit applies the ZZ feature map for ``x1``, then its inverse for
    ``x2``.  Measuring the probability of the all-zeros state gives
    ``|⟨φ(x1)|φ(x2)⟩|²``.

    Parameters
    ----------
    x1, x2 : array-like, shape (8,)
        Feature vectors with values in ``[0, π]``.

    Returns
    -------
    np.ndarray
        Probabilities of all 2^8 = 256 computational basis states.
    """
    # Forward pass — encode x1
    qml.AngleEmbedding(x1, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2 * (np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])

    # Inverse pass — uncompute x2 (adjoint of x2 encoding)
    for i in range(N_QUBITS - 2, -1, -1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(-2 * (np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS), rotation="Y")

    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2) -> float:
    """Return the quantum kernel value K(x1, x2) = P(|0…0⟩).

    Parameters
    ----------
    x1, x2 : array-like, shape (8,)

    Returns
    -------
    float
        Kernel value in [0, 1].
    """
    x1_nd = np.array(x1, requires_grad=False)
    x2_nd = np.array(x2, requires_grad=False)
    return float(kernel_circuit(x1_nd, x2_nd)[0])


# ---------------------------------------------------------------------------
# Kernel matrix computation
# ---------------------------------------------------------------------------


def compute_kernel_matrix(
    X1: _np.ndarray,
    X2: _np.ndarray,
    desc: str = "Computing kernel",
) -> _np.ndarray:
    """Compute the full kernel matrix K where K[i, j] = K(X1[i], X2[j]).

    Parameters
    ----------
    X1 : np.ndarray, shape (N1, 8)
    X2 : np.ndarray, shape (N2, 8)
    desc : str
        Label for the tqdm progress bar.

    Returns
    -------
    np.ndarray, shape (N1, N2)
    """
    N1, N2 = len(X1), len(X2)
    K = _np.zeros((N1, N2))

    t0 = time.time()
    for i in tqdm(range(N1), desc=desc):
        for j in range(N2):
            K[i, j] = quantum_kernel(X1[i], X2[j])

    elapsed = time.time() - t0
    print(f"Kernel matrix shape: {K.shape} | computed in {elapsed:.1f}s")
    return K


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------


class QuantumKernelSVM:
    """Support Vector Classifier that uses the quantum ZZ kernel.

    Parameters
    ----------
    C : float
        SVM regularisation parameter.
    """

    name = "Quantum Kernel SVM (ZZ Kernel)"

    def __init__(self, C: float = 1.0) -> None:
        self.svm = SVC(
            kernel="precomputed", C=C, probability=True, random_state=42
        )
        self.X_train: Optional[_np.ndarray] = None

    def fit(self, X_train: _np.ndarray, y_train: _np.ndarray) -> "QuantumKernelSVM":
        """Compute the training kernel matrix and fit the SVM.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, 8)
        y_train : np.ndarray, shape (N,)

        Returns
        -------
        QuantumKernelSVM
            self.
        """
        self.X_train = X_train
        K_train = compute_kernel_matrix(X_train, X_train, desc="Training kernel")
        self.svm.fit(K_train, y_train)
        return self

    def predict(self, X_test: _np.ndarray) -> _np.ndarray:
        """Predict labels for *X_test*.

        Parameters
        ----------
        X_test : np.ndarray, shape (M, 8)

        Returns
        -------
        np.ndarray, shape (M,)
        """
        K_test = compute_kernel_matrix(X_test, self.X_train, desc="Test kernel")
        return self.svm.predict(K_test)

    def predict_proba(self, X_test: _np.ndarray) -> _np.ndarray:
        """Return class probabilities, shape ``(M, 2)``.

        Parameters
        ----------
        X_test : np.ndarray, shape (M, 8)

        Returns
        -------
        np.ndarray, shape (M, 2)
        """
        K_test = compute_kernel_matrix(X_test, self.X_train, desc="Test kernel")
        return self.svm.predict_proba(K_test)

    def evaluate(self, X_test: _np.ndarray, y_test: _np.ndarray) -> dict:
        """Compute and return standard binary classification metrics.

        Parameters
        ----------
        X_test, y_test : np.ndarray

        Returns
        -------
        dict
            Keys: accuracy, f1, precision, recall, confusion_matrix.
        """
        y_pred = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }


# ---------------------------------------------------------------------------
# Self-test (10 samples)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running QKSVM smoke test (10 samples)…")
    rng = _np.random.default_rng(0)
    X_smoke = rng.uniform(0, _np.pi, (10, N_QUBITS))
    y_smoke = rng.integers(0, 2, 10)

    model = QuantumKernelSVM(C=1.0)
    model.fit(X_smoke, y_smoke)
    preds = model.predict(X_smoke)
    print(f"Predictions : {preds}")
    print(f"Accuracy    : {accuracy_score(y_smoke, preds):.4f}")
    print("✅ QKSVM smoke test passed.")
