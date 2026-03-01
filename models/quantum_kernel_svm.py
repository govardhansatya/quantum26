"""
models/quantum_kernel_svm.py
=============================
Quantum Kernel SVM (QKSVM) for sarcasm detection.

The quantum kernel implicitly computes inner products in a 2^N_QUBITS = 256-
dimensional Hilbert space (the quantum feature space), providing a richer
similarity measure than a classical RBF kernel.

Kernel definition
-----------------
K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²

where |φ(x)⟩ is the quantum state prepared by applying the ZZ feature map
to the computational zero state |0⟩^⊗N.

Implementation: compute K(x₁, x₂) as P(|0…0⟩) after applying
the feature map for x₁ followed by the *adjoint* feature map for x₂.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as _np               # plain numpy for non-differentiable ops
import pennylane as qml
import pennylane.numpy as pnp    # pennylane-aware numpy (not needed for gradients here)
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
# Quantum device & circuit constants
# ---------------------------------------------------------------------------

N_QUBITS = 8
dev_kernel = qml.device("default.qubit", wires=N_QUBITS)


# ---------------------------------------------------------------------------
# Quantum kernel circuit
# ---------------------------------------------------------------------------

@qml.qnode(dev_kernel)
def kernel_circuit(x1: _np.ndarray, x2: _np.ndarray) -> _np.ndarray:
    """Compute the quantum kernel overlap between two feature vectors.

    The circuit is structured as:
      Φ(x₁) |0⟩  ·  Φ(x₂)†|0⟩

    so that the probability of measuring |0…0⟩ equals |⟨φ(x₁)|φ(x₂)⟩|².

    Parameters
    ----------
    x1 : np.ndarray
        First feature vector, shape (N_QUBITS,), values in [0, π].
    x2 : np.ndarray
        Second feature vector, same shape.

    Returns
    -------
    np.ndarray
        Probability distribution over all 2^N_QUBITS basis states.
        ``result[0]`` = P(|0…0⟩) = K(x1, x2).
    """
    # --- Forward pass: encode x1 ---
    qml.AngleEmbedding(x1, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(
            2.0 * (_np.pi - x1[i]) * (_np.pi - x1[i + 1]),
            wires=i + 1,
        )
        qml.CNOT(wires=[i, i + 1])

    # --- Inverse pass: adjoint encoding of x2 ---
    # Apply the ZZ-entanglement adjoint (reversed order + negated angles)
    for i in range(N_QUBITS - 2, -1, -1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(
            -2.0 * (_np.pi - x2[i]) * (_np.pi - x2[i + 1]),
            wires=i + 1,
        )
        qml.CNOT(wires=[i, i + 1])
    # Adjoint of AngleEmbedding: negate angles
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS), rotation="Y")

    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1: _np.ndarray, x2: _np.ndarray) -> float:
    """Compute K(x1, x2) = P(|0…0⟩) from the kernel circuit.

    Parameters
    ----------
    x1, x2 : np.ndarray
        Feature vectors, shape (N_QUBITS,).

    Returns
    -------
    float
        Kernel value in [0, 1].  K(x, x) ≈ 1.0 (self-overlap).
    """
    probs = kernel_circuit(
        _np.asarray(x1, dtype=float),
        _np.asarray(x2, dtype=float),
    )
    return float(probs[0])


# ---------------------------------------------------------------------------
# Kernel matrix computation
# ---------------------------------------------------------------------------

def compute_kernel_matrix(
    X1: _np.ndarray,
    X2: _np.ndarray,
    desc: str = "Computing kernel",
) -> _np.ndarray:
    """Build the full kernel matrix K where K[i,j] = quantum_kernel(X1[i], X2[j]).

    Time complexity: O(N1 × N2) circuit evaluations.
    For N training samples, the training kernel is O(N²).

    Parameters
    ----------
    X1 : np.ndarray
        First set of feature vectors, shape (N1, N_QUBITS).
    X2 : np.ndarray
        Second set of feature vectors, shape (N2, N_QUBITS).
    desc : str
        Progress bar label. Default ``"Computing kernel"``.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (N1, N2).
    """
    N1, N2 = len(X1), len(X2)
    K = _np.zeros((N1, N2), dtype=float)
    t0 = time.time()

    for i in tqdm(range(N1), desc=desc, unit="row"):
        for j in range(N2):
            K[i, j] = quantum_kernel(X1[i], X2[j])

    elapsed = time.time() - t0
    print(
        f"[Kernel] {desc} → shape={K.shape} | time={elapsed:.1f}s | "
        f"diagonal mean={_np.diag(K).mean():.4f}"
        if X1 is X2 or (N1 == N2 and _np.allclose(X1, X2))
        else f"[Kernel] {desc} → shape={K.shape} | time={elapsed:.1f}s"
    )
    return K


# ---------------------------------------------------------------------------
# Quantum Kernel SVM class
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: _np.ndarray, y_pred: _np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


class QuantumKernelSVM:
    """Support Vector Machine with a quantum kernel (precomputed).

    The SVM is trained on the kernel matrix K_train = Φ(X_train)^T Φ(X_train)
    in the 256-dimensional Hilbert space.

    Parameters
    ----------
    C : float
        SVM regularisation parameter. Default 1.0.

    Attributes
    ----------
    svm : sklearn.svm.SVC
        Underlying SVM with ``kernel='precomputed'``.
    X_train_ : np.ndarray
        Stored training features needed to compute test kernel matrices.
    name : str
        Human-readable model name.

    Examples
    --------
    >>> model = QuantumKernelSVM(C=1.0)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    name = "Quantum Kernel SVM (ZZ Kernel)"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.svm = SVC(
            kernel="precomputed",
            C=C,
            probability=True,
            random_state=42,
        )
        self.X_train_: Optional[_np.ndarray] = None

    def fit(
        self,
        X_train: _np.ndarray,
        y_train: _np.ndarray,
    ) -> "QuantumKernelSVM":
        """Compute training kernel matrix and fit the SVM.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (N, N_QUBITS).
        y_train : np.ndarray
            Binary training labels.

        Returns
        -------
        QuantumKernelSVM
            self (for chaining).
        """
        self.X_train_ = _np.asarray(X_train, dtype=float)
        K_train = compute_kernel_matrix(
            self.X_train_, self.X_train_, desc="Training kernel"
        )
        self.svm.fit(K_train, y_train)
        print(f"[QKSVM] Fitted on {len(X_train)} samples.")
        return self

    def predict(self, X_test: _np.ndarray) -> _np.ndarray:
        """Predict class labels for X_test.

        Parameters
        ----------
        X_test : np.ndarray
            Test features, shape (M, N_QUBITS).

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1), shape (M,).
        """
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_test = _np.asarray(X_test, dtype=float)
        K_test = compute_kernel_matrix(
            X_test, self.X_train_, desc="Test kernel"
        )
        return self.svm.predict(K_test)

    def predict_proba(self, X_test: _np.ndarray) -> _np.ndarray:
        """Return class probabilities for X_test.

        Parameters
        ----------
        X_test : np.ndarray
            Test features, shape (M, N_QUBITS).

        Returns
        -------
        np.ndarray
            Probability matrix of shape (M, 2).
        """
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X_test = _np.asarray(X_test, dtype=float)
        K_test = compute_kernel_matrix(
            X_test, self.X_train_, desc="Test kernel (proba)"
        )
        return self.svm.predict_proba(K_test)

    def evaluate(
        self,
        X_test: _np.ndarray,
        y_test: _np.ndarray,
    ) -> dict:
        """Evaluate on the test set.

        Returns
        -------
        dict
            Keys: accuracy, f1, precision, recall, confusion_matrix.
        """
        y_pred = self.predict(X_test)
        return _compute_metrics(_np.asarray(y_test), y_pred)


# ---------------------------------------------------------------------------
# Standalone smoke-test (10 samples)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _np.random.seed(0)
    N = 10
    X_smoke = _np.random.uniform(0, _np.pi, (N, N_QUBITS))
    y_smoke = _np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    print(f"=== QKSVM Smoke Test ({N} samples) ===")

    # Quick self-kernel test: K(x, x) should be ≈ 1
    k_self = quantum_kernel(X_smoke[0], X_smoke[0])
    print(f"Self-kernel K(x0, x0) = {k_self:.4f}  (expected ≈ 1.0)")

    model = QuantumKernelSVM(C=1.0)
    model.fit(X_smoke[:8], y_smoke[:8])
    preds = model.predict(X_smoke[8:])
    print(f"Predictions on 2 test samples: {preds}")
    print("✅ QKSVM smoke test complete")
