"""
experiments/run_experiments.py
================================
Main experimental comparison script: Quantum vs Classical sarcasm detection
across multiple training sizes.

Scientific goal
---------------
Demonstrate "quantum advantage in the low-data regime" by showing that
Quantum Kernel SVM and VQC achieve higher accuracy than classical baselines
when training data is scarce (N ≤ 200 samples).

Usage
-----
# Quick demo — 1 trial at N=200 (≈5 min)
python experiments/run_experiments.py

# Full experiment — multiple trials at all sample sizes (≈40 min)
python experiments/run_experiments.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Dict, List

import numpy as _np
import pandas as pd
from sklearn.model_selection import train_test_split

# Make sure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_pipeline import load_and_clean_dataset
from utils.feature_engineering import QuantumFeaturePipeline
from models.classical_models import (
    ClassicalSVMBaseline,
    LogisticRegressionBaseline,
    NaiveBayesBaseline,
)
from models.vqc_model import VQCClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_SIZES = [50, 100, 150, 200, 300, 500]
N_TRIALS = 3
TEST_SIZE = 0.25
RESULTS_PATH = "results/experiment_results.csv"
SAVED_MODELS_DIR = "results/saved_models"

# VQC iteration cap for experiment speed
VQC_MAX_ITER = 40
# VQC iteration cap for quick demo (keeps runtime to ~5 min)
VQC_DEMO_MAX_ITER = 50

os.makedirs("results", exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stratified_sample(
    X: _np.ndarray,
    y: _np.ndarray,
    n: int,
    seed: int,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Draw a stratified random subsample of size *n* from (X, y).

    If n ≥ len(X) the full dataset is returned.
    """
    if n >= len(X):
        return X, y
    idx_0 = _np.where(y == 0)[0]
    idx_1 = _np.where(y == 1)[0]
    rng = _np.random.RandomState(seed)
    n0 = max(1, n // 2)
    n1 = n - n0
    chosen_0 = rng.choice(idx_0, min(n0, len(idx_0)), replace=False)
    chosen_1 = rng.choice(idx_1, min(n1, len(idx_1)), replace=False)
    idx = _np.concatenate([chosen_0, chosen_1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _sarcasm_accuracy(
    model,
    X_test: _np.ndarray,
    y_test: _np.ndarray,
) -> float:
    """Accuracy restricted to the sarcastic samples (label == 1)."""
    mask = y_test == 1
    if mask.sum() == 0:
        return float("nan")
    preds = model.predict(X_test[mask])
    return float((preds == y_test[mask]).mean())


def _print_table(df: pd.DataFrame, n_focus: int = 200) -> None:
    """Print a formatted summary table for a specific training size."""
    sub = df[df["n_samples"] == n_focus]
    if sub.empty:
        print(f"No results at N={n_focus}")
        return
    pivot = (
        sub.groupby("model_name")[["accuracy", "f1_score"]]
        .mean()
        .sort_values("accuracy", ascending=False)
        .reset_index()
    )
    print(f"\n{'='*65}")
    print(f"  Results at N={n_focus} training samples")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Accuracy':>10} {'F1':>8}")
    print(f"  {'-'*55}")
    for _, row in pivot.iterrows():
        print(
            f"  {row['model_name']:<35} {row['accuracy']:>10.4f} "
            f"{row['f1_score']:>8.4f}"
        )
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_full_experiment(
    data_path: str = "data/Sarcasm_Headlines_Dataset_v2.json",
    sample_sizes: List[int] = SAMPLE_SIZES,
    n_trials: int = N_TRIALS,
    vqc_max_iter: int = VQC_MAX_ITER,
) -> pd.DataFrame:
    """Run ALL models across ALL sample sizes.

    Parameters
    ----------
    data_path : str
        Path to the dataset JSON file.
    sample_sizes : list of int
        Training sizes to sweep.
    n_trials : int
        Number of repetitions per (model, size) pair.

    Returns
    -------
    pd.DataFrame
        Full results table with columns:
        n_samples, trial, model_name, accuracy, f1_score,
        sarcasm_accuracy, train_time_sec.
    """
    # Step 1 — Load and clean dataset
    df_raw = load_and_clean_dataset(data_path)
    texts = df_raw["clean_text"].tolist()
    labels = df_raw["label"].values

    # Step 2a — Classical feature pipeline: rich TF-IDF (2000 features) — fitted once
    print("\n🔧 Fitting classical TF-IDF pipeline on full dataset …")
    _clf_tfidf = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_clf_raw = _clf_tfidf.fit_transform(texts)
    _clf_scaler = MaxAbsScaler()
    X_clf_full = _clf_scaler.fit_transform(X_clf_raw).toarray()
    print(f"   Classical features: {X_clf_full.shape}")

    # Step 2b — Quantum feature pipeline: 4 PCA + 4 hand-crafted — fitted once
    print("\n🔧 Fitting QuantumFeaturePipeline on full dataset …")
    pipeline = QuantumFeaturePipeline(n_qubits=8, n_tfidf=2000)
    X_vqc_full = pipeline.fit_transform(texts)
    y_full = labels

    records: List[dict] = []

    total_runs = len(sample_sizes) * n_trials
    run_idx = 0

    for size in sample_sizes:
        for trial in range(n_trials):
            run_idx += 1
            seed = trial * 13 + size
            print(
                f"\n[{run_idx}/{total_runs}] N={size}, Trial={trial + 1}/{n_trials} "
                f"(seed={seed})"
            )

            # Subsample (stratified) — classical and quantum indices stay aligned
            _, y_sub = _stratified_sample(X_vqc_full, y_full, size, seed)
            # Get the same indices by re-sampling both feature matrices
            rng = _np.random.RandomState(seed)
            idx_0 = _np.where(y_full == 0)[0]
            idx_1 = _np.where(y_full == 1)[0]
            n0 = max(1, size // 2)
            n1 = size - n0
            chosen_0 = rng.choice(idx_0, min(n0, len(idx_0)), replace=False)
            chosen_1 = rng.choice(idx_1, min(n1, len(idx_1)), replace=False)
            sub_idx = _np.concatenate([chosen_0, chosen_1])
            rng.shuffle(sub_idx)

            X_clf_sub  = X_clf_full[sub_idx]
            X_vqc_sub  = X_vqc_full[sub_idx]
            y_sub      = y_full[sub_idx]

            # Train / test split — same split for both feature sets
            try:
                split = train_test_split(
                    X_clf_sub, X_vqc_sub, y_sub,
                    test_size=TEST_SIZE, random_state=seed, stratify=y_sub,
                )
            except ValueError:
                split = train_test_split(
                    X_clf_sub, X_vqc_sub, y_sub,
                    test_size=TEST_SIZE, random_state=seed,
                )
            X_clf_tr, X_clf_te, X_vqc_tr, X_vqc_te, y_tr, y_te = split

            # ---- Define models for this run ----
            # Classical models get full TF-IDF features; VQC gets 8-qubit quantum features
            classical_models = [
                ("Logistic Regression", LogisticRegressionBaseline(), X_clf_tr, X_clf_te),
                ("Classical SVM (RBF)", ClassicalSVMBaseline(),       X_clf_tr, X_clf_te),
                ("Naive Bayes",         NaiveBayesBaseline(),          X_clf_tr, X_clf_te),
            ]
            quantum_models = [
                ("VQC (ZZ + SEL)", VQCClassifier(n_layers=2, max_iter=vqc_max_iter, lr=0.1, batch_size=32), X_vqc_tr, X_vqc_te),
            ]
            all_models = classical_models + quantum_models

            for m_name, model, X_tr, X_te in all_models:
                print(f"  → {m_name} …", end=" ", flush=True)
                t0 = time.time()
                try:
                    model.fit(X_tr, y_tr)
                    result = model.evaluate(X_te, y_te)
                    sar_acc = _sarcasm_accuracy(model, X_te, y_te)
                    elapsed = time.time() - t0
                    print(f"acc={result['accuracy']:.4f}  ({elapsed:.1f}s)")
                    records.append(
                        {
                            "n_samples": size,
                            "trial": trial,
                            "model_name": m_name,
                            "accuracy": result["accuracy"],
                            "f1_score": result["f1"],
                            "precision": result["precision"],
                            "recall": result["recall"],
                            "sarcasm_accuracy": sar_acc,
                            "train_time_sec": round(elapsed, 3),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.time() - t0
                    print(f"FAILED ({elapsed:.1f}s) — {exc}")
                    traceback.print_exc()
                    records.append(
                        {
                            "n_samples": size,
                            "trial": trial,
                            "model_name": m_name,
                            "accuracy": float("nan"),
                            "f1_score": float("nan"),
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "sarcasm_accuracy": float("nan"),
                            "train_time_sec": round(elapsed, 3),
                        }
                    )

    results_df = pd.DataFrame(records)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Results saved → {RESULTS_PATH}")
    _print_table(results_df, n_focus=200)
    return results_df


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def run_quick_demo(
    n: int = 200,
    data_path: str = "data/Sarcasm_Headlines_Dataset_v2.json",
) -> pd.DataFrame:
    """Run a single trial at N=*n* for all models.

    This completes in roughly 5 minutes and is ideal for validating the
    pipeline before committing to a full experiment.

    Parameters
    ----------
    n : int
        Number of training + test samples (before split). Default 200.
    data_path : str
        Path to the dataset JSON.

    Returns
    -------
    pd.DataFrame
        Results table for this demo run.
    """
    print(f"\n{'='*65}")
    print(f"  ⚡ Quick Demo — N={n}, 1 trial")
    print(f"{'='*65}")
    return run_full_experiment(
        data_path=data_path,
        sample_sizes=[n],
        n_trials=1,
        vqc_max_iter=VQC_DEMO_MAX_ITER,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantum vs Classical Sarcasm Detection Experiments"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full experiment across all sample sizes and trials (~40 min).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Sample size for quick demo mode. Default 200.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/Sarcasm_Headlines_Dataset_v2.json",
        help="Path to the dataset JSON file.",
    )
    args = parser.parse_args()

    if args.full:
        results = run_full_experiment(data_path=args.data)
    else:
        results = run_quick_demo(n=args.n, data_path=args.data)

    print(results.to_string(index=False))
