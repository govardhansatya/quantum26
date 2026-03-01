"""
models/vqc_model.py
====================
Variational Quantum Classifier (VQC) for binary sarcasm detection.

Architecture
------------
Feature map  : AngleEmbedding (Y-rotation) + ZZ entanglement + AngleEmbedding (Z, re-upload)
Ansatz       : StronglyEntanglingLayers  (N_LAYERS=2, N_QUBITS=8 → 48 parameters)
Measurement  : ⟨Z₀⟩ ∈ [−1, 1]  →  sigmoid to probability
Optimiser    : PennyLane AdamOptimizer with parameter-shift gradient

Important
---------
All differentiable tensors must use ``pennylane.numpy`` (pnp), NOT regular numpy.
Input data should be wrapped as ``pnp.array(..., requires_grad=False)`` to avoid
differentiating through the data encoder.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np                          # plain numpy — non-differentiable ops
import pennylane as qml
import pennylane.numpy as pnp               # differentiable numpy
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
# Quantum device & circuit constants
# ---------------------------------------------------------------------------

N_QUBITS = 8
N_LAYERS = 2

# Use lightning.qubit with adjoint diff for ~10-20x speedup over default.qubit+parameter-shift.
# Falls back gracefully if lightning is not available.
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    _DIFF_METHOD: str = "adjoint"
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    _DIFF_METHOD = "parameter-shift"

# Shape is (n_layers, n_wires, 3) = (2, 8, 3) = 48 trainable parameters
_WEIGHTS_SHAPE = qml.StronglyEntanglingLayers.shape(
    n_layers=N_LAYERS, n_wires=N_QUBITS
)

# Observable: mean of all qubit PauliZ — gathers signal from every qubit, not just qubit 0.
_MEAN_Z = qml.dot([1.0 / N_QUBITS] * N_QUBITS, [qml.PauliZ(i) for i in range(N_QUBITS)])


# ---------------------------------------------------------------------------
# Quantum circuit definition
# ---------------------------------------------------------------------------

@qml.qnode(dev, diff_method=_DIFF_METHOD)
def vqc_circuit(inputs: pnp.ndarray, weights: pnp.ndarray) -> float:
    """Evaluate the VQC circuit for one input sample.

    Parameters
    ----------
    inputs : pnp.ndarray
        Feature vector of length N_QUBITS with values in [0, π].
        Must be wrapped as ``requires_grad=False``.
    weights : pnp.ndarray
        Trainable ansatz weights of shape ``(N_LAYERS, N_QUBITS, 3)``.
        Must be wrapped as ``requires_grad=True``.

    Returns
    -------
    float
        Expectation value ⟨Z₀⟩ ∈ [−1, 1].

    Circuit structure
    -----------------
    1. AngleEmbedding (Y)  — encode features as Ry rotations
    2. ZZ entanglement  — nearest-neighbour CNOT + RZ(2(π−xᵢ)(π−xⱼ)) + CNOT
    3. AngleEmbedding (Z)  — data re-uploading (second pass)
    4. StronglyEntanglingLayers — parameterised ansatz
    5. Measure ⟨Z₀⟩
    """
    # --- Step 1: First angle embedding (Y rotations) ---
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    # --- Step 2: ZZ-feature map (nearest-neighbour entanglement) ---
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(
            2.0 * (pnp.pi - inputs[i]) * (pnp.pi - inputs[i + 1]),
            wires=i + 1,
        )
        qml.CNOT(wires=[i, i + 1])

    # --- Step 3: Second angle embedding (Z rotations — data re-uploading) ---
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Z")

    # --- Step 4: Variational ansatz ---
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    # --- Step 5: Measurement — mean ⟨Z⟩ over all qubits (more signal than single qubit) ---
    return qml.expval(_MEAN_Z)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + _np.exp(-x))


def _expval_to_prob(expval):
    """Map ⟨Z⟩ ∈ [−1, 1] to probability ∈ [0, 1]."""
    return (expval + 1.0) / 2.0


def _bce_loss(probs: pnp.ndarray, labels: _np.ndarray) -> pnp.ndarray:
    """Binary cross-entropy loss (numerically stable).

    Parameters
    ----------
    probs : array-like
        Predicted positive-class probabilities in (0, 1).
    labels : array-like
        Ground-truth binary labels (0 or 1).

    Returns
    -------
    scalar
        Mean BCE loss over the batch.
    """
    eps = 1e-7
    probs = pnp.clip(probs, eps, 1.0 - eps)
    loss = -(labels * pnp.log(probs) + (1 - labels) * pnp.log(1 - probs))
    return pnp.mean(loss)


def _compute_metrics(y_true: _np.ndarray, y_pred: _np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# VQC Classifier class
# ---------------------------------------------------------------------------

class VQCClassifier:
    """Variational Quantum Classifier using ZZ Feature Map + StronglyEntangling ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Default 8.
    n_layers : int
        Number of StronglyEntanglingLayers repetitions. Default 2.
    lr : float
        Adam learning rate. Default 0.05.
    max_iter : int
        Maximum training epochs. Default 60.
    batch_size : int
        Mini-batch size for gradient estimation. Default 16.
    patience : int
        Early stopping patience (epochs without improvement). Default 15.

    Attributes
    ----------
    weights : pnp.ndarray
        Trainable circuit parameters (requires_grad=True).
    loss_history : list of float
        Training loss recorded each epoch.
    name : str
        Human-readable model name for logging and plotting.

    Examples
    --------
    >>> model = VQCClassifier(max_iter=10)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    name = "VQC (ZZ Feature Map + StronglyEntangling)"

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        lr: float = 0.1,
        max_iter: int = 60,
        batch_size: int = 32,
        patience: int = 15,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patience = patience

        # Small normal init avoids barren plateaus (vanishing gradients from wide uniform init).
        _init = _np.random.normal(0.0, 0.01, _WEIGHTS_SHAPE)
        self.weights: pnp.ndarray = pnp.array(_init, requires_grad=True)
        self.optimizer = AdamOptimizer(lr)
        self.loss_history: List[float] = []

    # ------------------------------------------------------------------
    # Internal cost function
    # ------------------------------------------------------------------

    def _cost(
        self,
        weights: pnp.ndarray,
        X_batch: _np.ndarray,
        y_batch: _np.ndarray,
    ) -> pnp.ndarray:
        """Compute mean BCE loss over a mini-batch.

        Parameters
        ----------
        weights : pnp.ndarray
            Current circuit weights (differentiable).
        X_batch : np.ndarray
            Batch of feature vectors (non-differentiable).
        y_batch : np.ndarray
            Batch of labels.

        Returns
        -------
        pnp.ndarray
            Scalar loss value.
        """
        probs = []
        for x in X_batch:
            x_tensor = pnp.array(x, requires_grad=False)
            expval = vqc_circuit(x_tensor, weights)
            p = _expval_to_prob(expval)
            probs.append(p)
        probs_arr = pnp.stack(probs)
        return _bce_loss(probs_arr, y_batch)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: _np.ndarray,
        y_train: _np.ndarray,
        X_val: Optional[_np.ndarray] = None,
        y_val: Optional[_np.ndarray] = None,
    ) -> "VQCClassifier":
        """Train the VQC on (X_train, y_train).

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (N, n_qubits).
        y_train : np.ndarray
            Binary training labels.
        X_val : np.ndarray, optional
            Validation features for early stopping / progress monitoring.
        y_val : np.ndarray, optional
            Validation labels.

        Returns
        -------
        VQCClassifier
            self (for chaining).
        """
        X_train = _np.asarray(X_train, dtype=float)
        y_train = _np.asarray(y_train, dtype=float)
        n = len(X_train)
        best_val_acc = 0.0
        patience_counter = 0

        pbar = tqdm(range(self.max_iter), desc="VQC Training", unit="epoch")
        for epoch in pbar:
            # Shuffle mini-batches
            perm = _np.random.permutation(n)
            epoch_losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                X_b = X_train[idx]
                y_b = y_train[idx]

                def cost_fn(w):
                    return self._cost(w, X_b, y_b)

                self.weights, loss_val = self.optimizer.step_and_cost(
                    cost_fn, self.weights
                )
                epoch_losses.append(float(loss_val))

            mean_loss = float(_np.mean(epoch_losses))
            self.loss_history.append(mean_loss)
            pbar.set_postfix(loss=f"{mean_loss:.4f}")

            # Validation monitoring
            if X_val is not None and epoch % 10 == 0:
                val_preds = self.predict(X_val)
                val_acc = accuracy_score(y_val, val_preds)
                pbar.write(
                    f"  Epoch {epoch:>3d} | loss={mean_loss:.4f} | "
                    f"val_acc={val_acc:.4f}"
                )
                if val_acc > best_val_acc + 1e-4:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience // 10:
                    pbar.write(f"  ⛔ Early stopping at epoch {epoch}")
                    break

        pbar.close()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: _np.ndarray) -> _np.ndarray:
        """Return class probabilities for each sample.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (N, n_qubits).

        Returns
        -------
        np.ndarray
            Probability matrix of shape (N, 2): ``[:, 0]`` = P(not sarcastic),
            ``[:, 1]`` = P(sarcastic).
        """
        X = _np.asarray(X, dtype=float)
        probs_pos = []
        for x in X:
            x_tensor = pnp.array(x, requires_grad=False)
            expval = float(vqc_circuit(x_tensor, self.weights))
            p_pos = _expval_to_prob(expval)
            probs_pos.append(p_pos)
        p_pos_arr = _np.array(probs_pos)
        return _np.column_stack([1.0 - p_pos_arr, p_pos_arr])

    def predict(self, X: _np.ndarray) -> _np.ndarray:
        """Return binary class predictions (threshold = 0.5).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (N, n_qubits).

        Returns
        -------
        np.ndarray
            Integer labels 0 or 1.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def evaluate(self, X_test: _np.ndarray, y_test: _np.ndarray) -> dict:
        """Evaluate model performance on the test set.

        Returns
        -------
        dict
            Keys: accuracy, f1, precision, recall, confusion_matrix.
        """
        y_pred = self.predict(X_test)
        return _compute_metrics(_np.asarray(y_test), y_pred)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def draw_circuit(self) -> str:
        """Print and return an ASCII diagram of the VQC circuit.

        Uses dummy inputs and random weights to draw the circuit structure.

        Returns
        -------
        str
            ASCII circuit diagram as a string.
        """
        dummy_inputs = pnp.array(_np.zeros(N_QUBITS), requires_grad=False)
        dummy_weights = pnp.array(
            _np.zeros(_WEIGHTS_SHAPE), requires_grad=False
        )
        diagram = qml.draw(vqc_circuit)(dummy_inputs, dummy_weights)
        print(diagram)
        return diagram

    def plot_loss(self, save_dir: str = "results") -> None:
        """Plot and save the training loss curve.

        Parameters
        ----------
        save_dir : str
            Directory where ``vqc_loss_curve.png`` will be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, color="crimson", linewidth=2, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("VQC Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, "vqc_loss_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Loss curve saved → {save_path}")


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _np.random.seed(42)
    N_SMOKE = 20
    X_smoke = _np.random.uniform(0, _np.pi, (N_SMOKE, N_QUBITS))
    y_smoke = _np.random.randint(0, 2, N_SMOKE)

    print(f"=== VQC Smoke Test ({N_SMOKE} samples, 5 epochs) ===")
    model = VQCClassifier(max_iter=5, batch_size=4, lr=0.05)
    model.fit(X_smoke, y_smoke)
    preds = model.predict(X_smoke)
    acc = accuracy_score(y_smoke, preds)
    print(f"Smoke-test accuracy: {acc:.4f}")
    print("Circuit diagram:")
    model.draw_circuit()
    model.plot_loss()
    print("✅ VQC smoke test complete")
