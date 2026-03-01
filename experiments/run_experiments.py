"""
run_experiments.py
==================
Main experiment runner: trains and evaluates all five models (3 classical + 2 quantum)
across multiple training set sizes, producing a CSV results file for plotting.

Usage
-----
Quick demo (N=200, 1 trial, ~5 min):
    python experiments/run_experiments.py

Full experiment:
    python experiments/run_experiments.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as _np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.classical_models import (  # noqa: E402
    ClassicalSVMBaseline,
    LogisticRegressionBaseline,
    NaiveBayesBaseline,
)
from models.quantum_kernel_svm import QuantumKernelSVM  # noqa: E402
from models.vqc_model import VQCClassifier  # noqa: E402
from utils.data_pipeline import load_and_clean_dataset  # noqa: E402
from utils.feature_engineering import QuantumFeaturePipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SAMPLE_SIZES = [50, 100, 150, 200, 300, 500]
N_TRIALS = 3
TEST_SIZE = 0.25
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_results.csv")

# Quantum kernel is O(N²): cap training set size for QKSVM
QKSVM_MAX_TRAIN = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stratified_sample(
    X: _np.ndarray, y: _np.ndarray, n: int, rng: _np.random.Generator
) -> tuple:
    """Return a stratified subsample of size *n* (or len(X) if smaller)."""
    classes, counts = _np.unique(y, return_counts=True)
    indices: List[int] = []
    for cls, cnt in zip(classes, counts):
        cls_idx = _np.where(y == cls)[0]
        take = max(1, int(round(n * cnt / len(y))))
        take = min(take, len(cls_idx))
        chosen = rng.choice(cls_idx, size=take, replace=False)
        indices.extend(chosen.tolist())
    indices = indices[:n]
    rng.shuffle(indices)
    return X[indices], y[indices]


def _train_test_split_manual(
    X: _np.ndarray, y: _np.ndarray, test_size: float, seed: int
) -> tuple:
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=seed)


def _cap_quantum_training_set(
    X_tr: _np.ndarray,
    y_tr: _np.ndarray,
    model,
    max_size: int,
    rng: _np.random.Generator,
) -> tuple:
    """Cap the training set for quantum models whose kernel is O(N²).

    Parameters
    ----------
    X_tr, y_tr : np.ndarray
        Full training arrays.
    model : object
        Fitted or unfitted model instance.
    max_size : int
        Maximum training set size for :class:`~models.quantum_kernel_svm.QuantumKernelSVM`.
    rng : np.random.Generator
        Random number generator for reproducible subsampling.

    Returns
    -------
    tuple
        ``(X_use, y_use)`` — possibly capped arrays.
    """
    if isinstance(model, QuantumKernelSVM) and len(X_tr) > max_size:
        cap_idx = rng.choice(len(X_tr), size=max_size, replace=False)
        return X_tr[cap_idx], y_tr[cap_idx]
    return X_tr, y_tr


def _sarcasm_accuracy(y_true: _np.ndarray, y_pred: _np.ndarray) -> float:
    """Accuracy restricted to samples where the true label is 1 (sarcastic)."""
    mask = y_true == 1
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == y_true[mask]).mean())


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_full_experiment(
    data_path: str = "data/Sarcasm_Headlines_Dataset_v2.json",
    sizes: List[int] = None,
    n_trials: int = N_TRIALS,
) -> pd.DataFrame:
    """Run the full quantum-vs-classical sarcasm detection experiment.

    For each training size and each trial, all five models are trained and
    evaluated.  Results are saved to ``results/experiment_results.csv``.

    Parameters
    ----------
    data_path : str
        Path to the JSON dataset.
    sizes : list of int, optional
        Training set sizes.  Defaults to :data:`SAMPLE_SIZES`.
    n_trials : int
        Number of independent repetitions per size.

    Returns
    -------
    pd.DataFrame
        Full results table.
    """
    if sizes is None:
        sizes = SAMPLE_SIZES

    # 1. Load dataset
    print("=" * 60)
    print("Loading and cleaning dataset…")
    df = load_and_clean_dataset(data_path)
    texts = df["clean_text"].tolist()
    labels = df["label"].values

    # 2. Fit feature pipeline on the full corpus
    print("\nFitting QuantumFeaturePipeline on full dataset…")
    pipeline = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
    X_full = pipeline.fit_transform(texts)
    y_full = labels.astype(int)
    print(f"Full feature matrix: {X_full.shape}")

    rows = []
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    for n in sizes:
        for trial in range(n_trials):
            rng = _np.random.default_rng(trial * 100 + n)

            # Subsample
            X_sub, y_sub = _stratified_sample(X_full, y_full, n, rng)
            X_tr, X_te, y_tr, y_te = _train_test_split_manual(
                X_sub, y_sub, TEST_SIZE, seed=trial
            )

            models = [
                LogisticRegressionBaseline(),
                ClassicalSVMBaseline(),
                NaiveBayesBaseline(),
                VQCClassifier(n_layers=2, max_iter=40, batch_size=16),
                QuantumKernelSVM(C=1.0),
            ]

            for model in models:
                model_name = model.name
                print(f"  N={n}, Trial {trial+1}/{n_trials}, Model: {model_name}…")

                # QKSVM: cap training set (kernel is O(N²))
                X_tr_use, y_tr_use = _cap_quantum_training_set(
                    X_tr, y_tr, model, QKSVM_MAX_TRAIN, rng
                )

                try:
                    t0 = time.time()
                    model.fit(X_tr_use, y_tr_use)
                    train_time = time.time() - t0

                    metrics = model.evaluate(X_te, y_te)
                    y_pred = model.predict(X_te)
                    sarc_acc = _sarcasm_accuracy(y_te, y_pred)

                    rows.append(
                        {
                            "n_samples": n,
                            "trial": trial,
                            "model_name": model_name,
                            "accuracy": metrics["accuracy"],
                            "f1_score": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "sarcasm_accuracy": sarc_acc,
                            "train_time_sec": round(train_time, 2),
                        }
                    )
                except Exception as exc:
                    print(f"  ⚠️  {model_name} failed: {exc}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Summary table at N=200
    _print_summary(results_df, pivot_n=200)
    return results_df


def run_quick_demo(
    data_path: str = "data/Sarcasm_Headlines_Dataset_v2.json",
    n: int = 200,
) -> pd.DataFrame:
    """Run a single trial at N=*n* for all models (~5 minutes).

    Parameters
    ----------
    data_path : str
        Path to the JSON dataset.
    n : int
        Training + test set size.

    Returns
    -------
    pd.DataFrame
        Results table.
    """
    return run_full_experiment(data_path=data_path, sizes=[n], n_trials=1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_summary(df: pd.DataFrame, pivot_n: int = 200) -> None:
    """Print a formatted accuracy summary table."""
    sub = df[df["n_samples"] == pivot_n]
    if sub.empty:
        # Fallback: use any available N
        pivot_n = df["n_samples"].iloc[0]
        sub = df[df["n_samples"] == pivot_n]

    summary = (
        sub.groupby("model_name")["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print(f"\n--- Results at N={pivot_n} ---")
    header = f"{'Model':<40} {'Mean Acc':>10} {'Std':>8}"
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        print(f"{row['model_name']:<40} {row['mean']:>10.4f} {row['std']:>8.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum sarcasm detection experiments")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full experiment across all sample sizes and trials.",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "..", "data",
                             "Sarcasm_Headlines_Dataset_v2.json"),
        help="Path to the sarcasm dataset JSON file.",
    )
    args = parser.parse_args()

    if args.full:
        run_full_experiment(data_path=args.data)
    else:
        run_quick_demo(data_path=args.data)
