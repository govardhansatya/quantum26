"""
models/classical_models.py
===========================
Classical baseline models for sarcasm detection.

Provides three baselines — Logistic Regression, Classical SVM (RBF), and
Naive Bayes — all sharing a common interface so they can be swapped
transparently against the quantum alternatives.

Interface
---------
All model classes expose:
  fit(X_train, y_train)              → self
  predict(X_test)                    → np.ndarray (int labels)
  predict_proba(X_test)              → np.ndarray (N, 2)
  evaluate(X_test, y_test)           → dict
  name                               → str
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a standard set of binary-classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict
        Keys: accuracy, f1, precision, recall, confusion_matrix.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Model 1 — Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionBaseline:
    """Logistic Regression baseline for binary sarcasm classification.

    Wraps ``sklearn.linear_model.LogisticRegression`` with a clean interface
    that mirrors the quantum models for easy benchmarking.

    Parameters
    ----------
    C : float
        Inverse regularisation strength. Larger → less regularisation.
    max_iter : int
        Maximum solver iterations.

    Examples
    --------
    >>> model = LogisticRegressionBaseline()
    >>> model.fit(X_train, y_train)
    >>> results = model.evaluate(X_test, y_test)
    >>> print(results["accuracy"])
    """

    name = "Logistic Regression"

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=42, n_jobs=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        """Train on (X_train, y_train)."""
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary class labels."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, 2)."""
        return self._model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute accuracy, F1, precision, recall and confusion matrix."""
        y_pred = self.predict(X_test)
        return _compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# Model 2 — Classical SVM (RBF)
# ---------------------------------------------------------------------------

class ClassicalSVMBaseline:
    """Classical Support Vector Machine with RBF kernel — the direct
    classical counterpart to the Quantum Kernel SVM.

    Uses ``sklearn.svm.SVC`` with ``kernel='rbf'`` and ``probability=True``
    to allow probability estimates via Platt scaling.

    Parameters
    ----------
    C : float
        Regularisation parameter. Default 1.0.
    gamma : str or float
        Kernel coefficient. Default ``'scale'``.

    Examples
    --------
    >>> model = ClassicalSVMBaseline()
    >>> model.fit(X_train, y_train)
    >>> model.predict_proba(X_test)[:3]
    """

    name = "Classical SVM (RBF)"

    def __init__(self, C: float = 1.0, gamma: str | float = "scale") -> None:
        self._model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClassicalSVMBaseline":
        """Train on (X_train, y_train)."""
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary class labels."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, 2)."""
        return self._model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute accuracy, F1, precision, recall and confusion matrix."""
        y_pred = self.predict(X_test)
        return _compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# Model 3 — Naive Bayes
# ---------------------------------------------------------------------------

class NaiveBayesBaseline:
    """Gaussian Naive Bayes baseline.

    Gaussian NB is particularly fast and works well with dense continuous
    features like those produced by the quantum feature pipeline.

    Examples
    --------
    >>> model = NaiveBayesBaseline()
    >>> model.fit(X_train, y_train)
    >>> model.evaluate(X_test, y_test)
    """

    name = "Naive Bayes"

    def __init__(self) -> None:
        self._model = GaussianNB()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NaiveBayesBaseline":
        """Train on (X_train, y_train)."""
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary class labels."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (N, 2)."""
        return self._model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute accuracy, F1, precision, recall and confusion matrix."""
        y_pred = self.predict(X_test)
        return _compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, dict]:
    """Train all three classical baselines and return their results.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    dict
        ``{model_name: results_dict}`` where each results dict has keys
        accuracy, f1, precision, recall, confusion_matrix.
    """
    models = [
        LogisticRegressionBaseline(),
        ClassicalSVMBaseline(),
        NaiveBayesBaseline(),
    ]
    all_results: Dict[str, dict] = {}

    for model in tqdm(models, desc="Training classical baselines", unit="model"):
        t0 = time.time()
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        results["train_time_sec"] = round(time.time() - t0, 4)
        all_results[model.name] = results

    # Pretty-print summary table
    print("\n" + "=" * 65)
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>8} {'Time (s)':>10}")
    print("-" * 65)
    for name, res in all_results.items():
        print(
            f"{name:<30} {res['accuracy']:>10.4f} "
            f"{res['f1']:>8.4f} {res['train_time_sec']:>10.4f}"
        )
    print("=" * 65 + "\n")

    return all_results


def train_at_sample_sizes(
    X: np.ndarray,
    y: np.ndarray,
    sizes: Optional[List[int]] = None,
    n_trials: int = 3,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Low-data regime experiment: train all baselines at varying training sizes.

    For each sample size, each model is trained ``n_trials`` times with
    different random seeds and the mean ± std accuracy is recorded.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Full label array.
    sizes : list of int
        Training set sizes to sweep. Default ``[50, 100, 200, 300, 500, 800]``.
    n_trials : int
        Number of repetitions per size. Default 3.
    test_size : float
        Fraction of *sizes* samples reserved for testing. Default 0.2.

    Returns
    -------
    pd.DataFrame
        Columns: n_samples, model_name, mean_accuracy, std_accuracy.
    """
    if sizes is None:
        sizes = [50, 100, 200, 300, 500, 800]

    models_cls = [
        LogisticRegressionBaseline,
        ClassicalSVMBaseline,
        NaiveBayesBaseline,
    ]

    records = []
    pbar = tqdm(total=len(sizes) * n_trials, desc="Low-data regime sweep")

    for size in sizes:
        # Collect per-trial accuracies for all models
        trial_accs: Dict[str, List[float]] = {
            m().name: [] for m in models_cls
        }
        for trial in range(n_trials):
            seed = trial * 7 + size  # reproducible but varied seeds
            idx = np.random.RandomState(seed).choice(
                len(X), min(size, len(X)), replace=False
            )
            X_sub, y_sub = X[idx], y[idx]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=test_size, random_state=seed, stratify=y_sub
            )
            for m_cls in models_cls:
                m = m_cls()
                m.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, m.predict(X_te))
                trial_accs[m.name].append(acc)
            pbar.update(1)

        for m_cls in models_cls:
            m_name = m_cls().name
            accs = trial_accs[m_name]
            records.append(
                {
                    "n_samples": size,
                    "model_name": m_name,
                    "mean_accuracy": float(np.mean(accs)),
                    "std_accuracy": float(np.std(accs)),
                }
            )

    pbar.close()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples=300, n_features=8, random_state=0
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_demo, y_demo, test_size=0.2, random_state=42
    )
    print("Training all baselines on synthetic data …")
    results = train_all_baselines(X_tr, y_tr, X_te, y_te)
    print("Low-data regime sweep …")
    df = train_at_sample_sizes(X_demo, y_demo, sizes=[50, 100, 150], n_trials=2)
    print(df.to_string(index=False))
