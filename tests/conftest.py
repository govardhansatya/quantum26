"""
tests/conftest.py
=================
Shared pytest fixtures for the quantum sarcasm detector test suite.

Fixtures
--------
sample_headlines     – 10 representative headline strings (raw)
sample_clean_texts   – manually cleaned versions of the above
binary_labels        – corresponding 0/1 labels (5 sarcastic, 5 not)
small_feature_matrix – (10, 8) float array in [0, π]  (quantum features)
small_classical_matrix – (20, 50) float TF-IDF matrix (classical features)
fitted_quantum_pipeline – QuantumFeaturePipeline fitted on sample_headlines
fitted_classical_bundle – dict with tfidf, scaler, and fitted LR/SVM/NB
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Raw headlines
# ---------------------------------------------------------------------------

_RAW_HEADLINES = [
    # sarcastic (label=1)
    "scientists confirm water is still wet, world shocked",
    "area man wins lottery, immediately loses ticket yeah right",
    "government solves all problems overnight, totally fine",
    "oh great another monday just perfect what could go wrong",
    "shocking: sun rises in east for millionth consecutive day",
    # not sarcastic (label=0)
    "local election results announced for third district",
    "new study shows regular exercise improves cardiovascular health",
    "city council approves new public library construction project",
    "researchers discover new species of deep-sea fish in pacific",
    "annual rainfall below average across northwest region",
]

_LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_headlines() -> list[str]:
    return list(_RAW_HEADLINES)


@pytest.fixture(scope="session")
def binary_labels() -> list[int]:
    return list(_LABELS)


@pytest.fixture(scope="session")
def sample_clean_texts(sample_headlines) -> list[str]:
    """Return the cleaning-pipeline output for each headline."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_pipeline import clean_text, negation_binding, remove_stopwords

    return [
        remove_stopwords(negation_binding(clean_text(h)))
        for h in sample_headlines
    ]


@pytest.fixture(scope="session")
def small_feature_matrix() -> np.ndarray:
    """10 × 8 random float matrix in [0, π] — mimics quantum features."""
    rng = np.random.RandomState(42)
    return rng.uniform(0, np.pi, (10, 8)).astype(float)


@pytest.fixture(scope="session")
def small_classical_matrix() -> np.ndarray:
    """20 × 50 random float matrix in [0, 1] — mimics scaled TF-IDF."""
    rng = np.random.RandomState(7)
    return rng.uniform(0, 1, (20, 50)).astype(float)


@pytest.fixture(scope="session")
def classical_labels() -> np.ndarray:
    return np.array([0, 1] * 10)


@pytest.fixture(scope="session")
def fitted_quantum_pipeline(sample_clean_texts):
    """QuantumFeaturePipeline fitted on the 10 sample headlines."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.feature_engineering import QuantumFeaturePipeline

    pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
    pipe.fit_transform(sample_clean_texts)
    return pipe


@pytest.fixture(scope="session")
def fitted_classical_bundle(sample_clean_texts, binary_labels):
    """Dict with fitted TF-IDF, scaler, LR, SVM, NB trained on 10 headlines."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MaxAbsScaler
    from models.classical_models import (
        LogisticRegressionBaseline,
        ClassicalSVMBaseline,
        NaiveBayesBaseline,
    )

    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
    X_raw = tfidf.fit_transform(sample_clean_texts)
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X_raw).toarray()
    y = np.array(binary_labels)

    lr  = LogisticRegressionBaseline(max_iter=200)
    svm = ClassicalSVMBaseline()
    nb  = NaiveBayesBaseline()
    lr.fit(X, y)
    svm.fit(X, y)
    nb.fit(X, y)

    return {"tfidf": tfidf, "scaler": scaler, "lr": lr, "svm": svm, "nb": nb, "X": X, "y": y}
