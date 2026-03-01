"""
vqc_model.py
============
Variational Quantum Classifier (VQC) using PennyLane.

Architecture
------------
* Feature map : AngleEmbedding (Y) + ZZ entanglement layer + AngleEmbedding (Z)
  — implements a data re-uploading / IQP-inspired encoding.
* Ansatz      : StronglyEntanglingLayers (2 layers × 8 qubits × 3 params = 48 params).
* Measurement : Expectation value of PauliZ on qubit 0, mapped to [0,1] via sigmoid.

Optimisation
------------
Gradient computation uses PennyLane's built-in parameter-shift rule, triggered
automatically by ``AdamOptimizer.step_and_cost``.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np
import pennylane as qml
from pennylane import numpy as np  # differentiable numpy
from pennylane.optimize import AdamOptimizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global device / circuit configuration
# ---------------------------------------------------------------------------

N_QUBITS = 8
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)
WEIGHTS_SHAPE = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
# WEIGHTS_SHAPE == (2, 8, 3) — 48 trainable parameters


@qml.qnode(dev)
def vqc_circuit(inputs, weights):
    """VQC quantum circuit: ZZ feature map + StronglyEntangling ansatz.

    Parameters
    ----------
    inputs : array-like, shape (8,)
        Angle-encoded classical features (values in [0, π]).
    weights : array-like, shape (N_LAYERS, N_QUBITS, 3)
        Variational parameters.

    Returns
    -------
    float
        Expectation value of PauliZ on qubit 0 — range [-1, 1].
    """
    # ---- Feature map ----
    # First angle embedding (Y rotations)
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    # ZZ entanglement: pairwise feature interactions
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2 * (np.pi - inputs[i]) * (np.pi - inputs[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    # Second angle embedding — data re-uploading (Z rotations)
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Z")

    # ---- Variational ansatz ----
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # ---- Measurement ----
    return qml.expval(qml.PauliZ(0))


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------


class VQCClassifier:
    """Variational Quantum Classifier for binary sarcasm detection.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must equal ``N_QUBITS = 8``).
    n_layers : int
        Number of StronglyEntanglingLayers repetitions.
    lr : float
        Adam learning rate.
    max_iter : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size for gradient updates.
    """

    name = "VQC (ZZ Feature Map + StronglyEntangling)"

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        lr: float = 0.05,
        max_iter: int = 60,
        batch_size: int = 16,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size

        # Initialise weights in pennylane.numpy so gradients flow
        rng = _np.random.default_rng(42)
        init = rng.uniform(-_np.pi, _np.pi, WEIGHTS_SHAPE)
        self.weights = np.array(init, requires_grad=True)

        self.optimizer = AdamOptimizer(lr)
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cost(self, weights, X_batch: _np.ndarray, y_batch: _np.ndarray) -> float:
        """Binary cross-entropy loss over a mini-batch.

        Parameters
        ----------
        weights : pennylane.numpy.ndarray
            Variational parameters (differentiable).
        X_batch : np.ndarray, shape (B, 8)
            Feature batch.
        y_batch : np.ndarray, shape (B,)
            Binary labels.

        Returns
        -------
        float
            Scalar cross-entropy loss.
        """
        loss = np.array(0.0)
        eps = 1e-7
        for x, y_true in zip(X_batch, y_batch):
            x_nd = np.array(x, requires_grad=False)
            expval = vqc_circuit(x_nd, weights)
            p_pos = (expval + 1.0) / 2.0  # map [-1,1] → [0,1]
            p_pos = np.clip(p_pos, eps, 1 - eps)
            if y_true == 1:
                loss = loss - np.log(p_pos)
            else:
                loss = loss - np.log(1.0 - p_pos)
        return loss / len(X_batch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: _np.ndarray,
        y_train: _np.ndarray,
        X_val: Optional[_np.ndarray] = None,
        y_val: Optional[_np.ndarray] = None,
    ) -> "VQCClassifier":
        """Train the VQC using mini-batch gradient descent (parameter-shift).

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data.
        X_val, y_val : np.ndarray, optional
            Validation data for early stopping and progress reporting.

        Returns
        -------
        VQCClassifier
            self (for method chaining).
        """
        n = len(X_train)
        best_val_acc = 0.0
        no_improve = 0
        patience = 15

        pbar = tqdm(range(self.max_iter), desc=f"Training {self.name}")
        for epoch in pbar:
            # Shuffle
            perm = _np.random.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            epoch_losses = []
            for start in range(0, n, self.batch_size):
                X_b = X_shuf[start : start + self.batch_size]
                y_b = y_shuf[start : start + self.batch_size]

                def cost_fn(w, _X=X_b, _y=y_b):
                    return self._cost(w, _X, _y)

                self.weights, batch_loss = self.optimizer.step_and_cost(
                    cost_fn, self.weights
                )
                epoch_losses.append(float(batch_loss))

            epoch_loss = float(_np.mean(epoch_losses))
            self.loss_history.append(epoch_loss)
            pbar.set_postfix(loss=f"{epoch_loss:.4f}")

            if X_val is not None and (epoch + 1) % 10 == 0:
                val_acc = accuracy_score(y_val, self.predict(X_val))
                print(f"  Epoch {epoch+1:3d} | loss={epoch_loss:.4f} | val_acc={val_acc:.4f}")
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience // 10:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        return self

    def predict_proba(self, X: _np.ndarray) -> _np.ndarray:
        """Return class probabilities, shape ``(N, 2)``.

        Parameters
        ----------
        X : np.ndarray, shape (N, 8)

        Returns
        -------
        np.ndarray
            ``[[p_not_sarcastic, p_sarcastic], ...]``
        """
        probs = []
        for x in X:
            x_nd = np.array(x, requires_grad=False)
            expval = float(vqc_circuit(x_nd, self.weights))
            p_pos = (expval + 1.0) / 2.0
            p_pos = float(_np.clip(p_pos, 0.0, 1.0))
            probs.append([1.0 - p_pos, p_pos])
        return _np.array(probs)

    def predict(self, X: _np.ndarray) -> _np.ndarray:
        """Return binary predictions (threshold = 0.5).

        Parameters
        ----------
        X : np.ndarray, shape (N, 8)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

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

    def draw_circuit(self) -> str:
        """Print and return an ASCII circuit diagram of the VQC.

        Returns
        -------
        str
            ASCII representation of the circuit.
        """
        sample_inputs = _np.ones(N_QUBITS) * 1.5
        sample_weights = _np.zeros(WEIGHTS_SHAPE)
        diagram = qml.draw(vqc_circuit)(sample_inputs, sample_weights)
        print(diagram)
        return diagram

    def plot_loss(self, save_path: str = "results/vqc_loss_curve.png") -> None:
        """Plot and save the training loss curve.

        Parameters
        ----------
        save_path : str
            File path for the output PNG.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.loss_history, color="crimson", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("VQC Training Loss Curve")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {save_path}")


# ---------------------------------------------------------------------------
# Self-test (smoke test with 20 samples)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running VQC smoke test (20 samples, 5 epochs)…")
    rng = _np.random.default_rng(0)
    X_smoke = rng.uniform(0, _np.pi, (20, N_QUBITS))
    y_smoke = rng.integers(0, 2, 20)

    clf = VQCClassifier(max_iter=5, batch_size=4)
    clf.fit(X_smoke, y_smoke)
    preds = clf.predict(X_smoke)
    print(f"Predictions: {preds}")
    print(f"Accuracy: {accuracy_score(y_smoke, preds):.4f}")
    clf.draw_circuit()
    print("✅ VQC smoke test passed.")
