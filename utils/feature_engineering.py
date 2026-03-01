"""
feature_engineering.py
=======================
Transforms cleaned text into 8 quantum-ready angle features per sample.

Pipeline:
    Text → TF-IDF (200) → PCA (32) → Haar DWT (→8 coefficients) → [0, π] scaling

The 8 output features are used as rotation angles for quantum angle encoding
on an 8-qubit circuit.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def haar_1d(x: np.ndarray) -> np.ndarray:
    """Apply a 1-D Haar Discrete Wavelet Transform (DWT) in-place.

    The length of *x* must be a power of 2.  The transform iteratively
    computes pairwise averages (approximation coefficients) and differences
    (detail coefficients), storing them in the first and second halves of
    each sub-array respectively.

    Parameters
    ----------
    x : np.ndarray
        1-D signal of length 2^k.

    Returns
    -------
    np.ndarray
        Haar-transformed array of the same length.
    """
    n = len(x)
    out = x.copy().astype(float)
    while n > 1:
        half = n // 2
        avg = (out[:n:2] + out[1:n:2]) / np.sqrt(2)
        diff = (out[:n:2] - out[1:n:2]) / np.sqrt(2)
        out[:half] = avg
        out[half:n] = diff
        n = half
    return out


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is >= *n*."""
    p = 1
    while p < n:
        p <<= 1
    return p


class QuantumFeaturePipeline:
    """End-to-end feature extraction pipeline for quantum angle encoding.

    Transforms a list of (cleaned) text strings into an array of shape
    ``(N, n_qubits)`` with values in ``[0, π]``.

    Parameters
    ----------
    n_qubits : int
        Number of output features (= number of qubits), default 8.
    n_tfidf : int
        Maximum number of TF-IDF vocabulary features, default 200.
    """

    def __init__(self, n_qubits: int = 8, n_tfidf: int = 200) -> None:
        self.n_qubits = n_qubits
        self.n_tfidf = n_tfidf

        self._tfidf = TfidfVectorizer(
            max_features=n_tfidf,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._pca = PCA(n_components=32, random_state=42)
        self._scaler = MinMaxScaler(feature_range=(0, np.pi))
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the pipeline on *texts* and return transformed features.

        Steps
        -----
        1. TF-IDF vectorisation (200 features, bigrams, sublinear TF).
        2. PCA dimensionality reduction to 32 components.
        3. Haar DWT, keeping the first ``n_qubits`` approximation coefficients.
        4. MinMax scaling to ``[0, π]``.

        Parameters
        ----------
        texts : list of str
            Cleaned, negation-bound headlines.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, n_qubits)`` with values in ``[0, π]``.
        """
        # Step 1 — TF-IDF
        X_tfidf = self._tfidf.fit_transform(texts).toarray()  # (N, 200)

        # Step 2 — PCA
        n_components = min(32, X_tfidf.shape[0] - 1, X_tfidf.shape[1])
        self._pca = PCA(n_components=n_components, random_state=42)
        X_pca = self._pca.fit_transform(X_tfidf)  # (N, 32)
        print(
            f"PCA explained variance: "
            f"{self._pca.explained_variance_ratio_.sum():.3f}"
        )

        # Step 3 — Haar DWT
        X_haar = self._apply_haar(X_pca)  # (N, n_qubits)

        # Step 4 — Scale to [0, π]
        X_scaled = self._scaler.fit_transform(X_haar)  # (N, n_qubits)

        self._fitted = True
        return X_scaled

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform *texts* using the already-fitted pipeline (no re-fitting).

        Parameters
        ----------
        texts : list of str
            Cleaned headlines.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, n_qubits)`` with values in ``[0, π]``.

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        X_tfidf = self._tfidf.transform(texts).toarray()
        X_pca = self._pca.transform(X_tfidf)
        X_haar = self._apply_haar(X_pca)
        X_scaled = self._scaler.transform(X_haar)
        return X_scaled

    def get_feature_names(self) -> List[str]:
        """Return human-readable feature names for each qubit angle.

        Returns
        -------
        list of str
            ``["qubit_0_feature", ..., "qubit_7_feature"]``
        """
        return [f"qubit_{i}_feature" for i in range(self.n_qubits)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_haar(self, X_pca: np.ndarray) -> np.ndarray:
        """Apply Haar DWT row-wise and return first *n_qubits* coefficients.

        Parameters
        ----------
        X_pca : np.ndarray
            Array of shape ``(N, D)`` where D >= n_qubits.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, n_qubits)``.
        """
        N, D = X_pca.shape
        pad_to = _next_power_of_two(D)
        if pad_to > D:
            X_padded = np.pad(X_pca, ((0, 0), (0, pad_to - D)))
        else:
            X_padded = X_pca

        X_haar = np.array([haar_1d(row) for row in X_padded])
        return X_haar[:, : self.n_qubits]


# ---------------------------------------------------------------------------
# Standalone heuristic
# ---------------------------------------------------------------------------

_SARCASM_MARKERS = [
    "yeah right",
    "oh great",
    "just perfect",
    "totally fine",
    "so helpful",
    "clearly",
    "obviously",
    "best ever",
    "love how",
    "wow thanks",
    "great idea",
    "sure",
    "what could go wrong",
    "shocking",
    "surprise surprise",
]


def detect_sarcasm_heuristic(text: str) -> float:
    """Return a heuristic sarcasm score based on common marker phrases.

    The score is computed as::

        min(1.0, count_of_matches / len(markers) * 3)

    It can be used as supplementary context in a UI without any ML model.

    Parameters
    ----------
    text : str
        Raw or cleaned headline.

    Returns
    -------
    float
        Score in ``[0.0, 1.0]``.
    """
    text_lower = text.lower()
    matches = sum(1 for m in _SARCASM_MARKERS if m in text_lower)
    score = min(1.0, matches / len(_SARCASM_MARKERS) * 3)
    return score


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy_texts = [
        "oh great another meeting that could have been an email",
        "scientists discover water still wet shocking revelation",
        "local man thrilled about monday morning",
        "new study finds exercise is good for health",
        "area man passionate defender of what he imagines constitution to say",
        "yeah right this is totally fine nothing to see here",
        "researchers publish important findings on climate change",
        "politicians agree on something for once in a surprise move",
        "dog bites man story covered by local newspaper",
        "obviously the best solution to this obvious problem",
    ]

    pipeline = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
    X = pipeline.fit_transform(dummy_texts)

    assert X.shape == (len(dummy_texts), 8), f"Unexpected shape: {X.shape}"
    assert X.min() >= 0.0, "Values below 0"
    assert X.max() <= 3.15, f"Values above π: {X.max()}"
    print("✅ Feature pipeline verified")
    print(f"Output shape: {X.shape}")
    print(f"Value range:  [{X.min():.4f}, {X.max():.4f}]")
    print("Feature names:", pipeline.get_feature_names())

    score = detect_sarcasm_heuristic("oh great another tax hike yeah right")
    print(f"Heuristic sarcasm score: {score:.3f}")
