"""
classical_models.py
===================
Three classical baseline classifiers for comparison against quantum models:

* :class:`LogisticRegressionBaseline`
* :class:`ClassicalSVMBaseline`  (RBF kernel — direct counterpart to Quantum SVM)
* :class:`NaiveBayesBaseline`

All follow the same interface: ``fit``, ``predict``, ``predict_proba``,
``evaluate``.
"""

from __future__ import annotations

import time
from typing import Dict, List

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
# Shared helper
# ---------------------------------------------------------------------------


def _evaluate_generic(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict
        Keys: accuracy, f1, precision, recall, confusion_matrix.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Baseline classes
# ---------------------------------------------------------------------------


class LogisticRegressionBaseline:
    """Logistic Regression classifier baseline.

    Parameters
    ----------
    C : float
        Inverse of regularisation strength.
    max_iter : int
        Maximum number of solver iterations.
    random_state : int
        Random seed for reproducibility.
    """

    name = "Logistic Regression"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        """Fit the model on training data."""
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary class predictions."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(N, 2)``."""
        return self._model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the model and return a metrics dictionary."""
        return _evaluate_generic(self, X_test, y_test)


class ClassicalSVMBaseline:
    """Classical SVM with RBF kernel — direct classical counterpart to Quantum SVM.

    Parameters
    ----------
    C : float
        Regularisation parameter.
    random_state : int
        Random seed for reproducibility.
    """

    name = "Classical SVM (RBF)"

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self._model = SVC(
            kernel="rbf", C=C, probability=True, random_state=random_state
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ClassicalSVMBaseline":
        """Fit the SVM on training data."""
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary class predictions."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(N, 2)``."""
        return self._model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the model and return a metrics dictionary."""
        return _evaluate_generic(self, X_test, y_test)


class NaiveBayesBaseline:
    """Gaussian Naïve Bayes baseline classifier.

    Parameters
    ----------
    None — GaussianNB has no critical hyper-parameters.
    """

    name = "Naive Bayes"

    def __init__(self):
        self._model = GaussianNB()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NaiveBayesBaseline":
        """Fit the model on training data."""
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return binary class predictions."""
        return self._model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(N, 2)``."""
        return self._model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the model and return a metrics dictionary."""
        return _evaluate_generic(self, X_test, y_test)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict]:
    """Train all three baselines and return their evaluation results.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and labels.
    X_test, y_test : np.ndarray
        Test features and labels.

    Returns
    -------
    dict
        ``{model_name: results_dict}`` where each *results_dict* contains
        accuracy, f1, precision, recall, confusion_matrix.
    """
    baselines = [
        LogisticRegressionBaseline(),
        ClassicalSVMBaseline(),
        NaiveBayesBaseline(),
    ]
    results: Dict[str, Dict] = {}

    for model in tqdm(baselines, desc="Training baselines"):
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[model.name] = metrics

    # Pretty-print table
    header = f"{'Model':<30} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>8}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<30} {m['accuracy']:>9.4f} {m['f1']:>9.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f}"
        )

    return results


def train_at_sample_sizes(
    X: np.ndarray,
    y: np.ndarray,
    sizes: List[int] = None,
    n_trials: int = 3,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Evaluate all three baselines across multiple training set sizes.

    For each size, *n_trials* independent train/test splits are run and
    accuracy statistics are averaged.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Labels.
    sizes : list of int, optional
        Training set sizes to sweep. Defaults to ``[50,100,200,300,500,800]``.
    n_trials : int
        Number of independent trials per size.
    test_size : float
        Fraction of each sampled set held out for testing.

    Returns
    -------
    pd.DataFrame
        Columns: ``n_samples``, ``model_name``, ``mean_accuracy``, ``std_accuracy``.
    """
    if sizes is None:
        sizes = [50, 100, 200, 300, 500, 800]

    rows = []
    for n in tqdm(sizes, desc="Sample sizes"):
        model_accs: Dict[str, List[float]] = {
            LogisticRegressionBaseline.name: [],
            ClassicalSVMBaseline.name: [],
            NaiveBayesBaseline.name: [],
        }
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            # Subsample without replacement up to min(n, len(X))
            idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
            X_sub, y_sub = X[idx], y[idx]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=test_size, random_state=seed, stratify=None
            )

            for model_cls in [
                LogisticRegressionBaseline,
                ClassicalSVMBaseline,
                NaiveBayesBaseline,
            ]:
                m = model_cls()
                m.fit(X_tr, y_tr)
                acc = float(accuracy_score(y_te, m.predict(X_te)))
                model_accs[m.name].append(acc)

        for model_name, accs in model_accs.items():
            rows.append(
                {
                    "n_samples": n,
                    "model_name": model_name,
                    "mean_accuracy": float(np.mean(accs)),
                    "std_accuracy": float(np.std(accs)),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples=400, n_features=8, n_informative=6, random_state=0
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.25, random_state=0)

    results = train_all_baselines(X_tr, y_tr, X_te, y_te)
    for name, m in results.items():
        print(f"{name}: accuracy={m['accuracy']:.4f}")
