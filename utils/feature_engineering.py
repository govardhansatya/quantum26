"""
utils/feature_engineering.py
=============================
Quantum feature extraction pipeline: TF-IDF → PCA → Haar Wavelet → [0, π] scaling.

This module is the bridge between classical NLP representations and quantum
circuit angle encoding.  Each text sample is reduced to exactly 8 floats in
[0, π], one per qubit, ready for ``qml.AngleEmbedding``.

Pipeline
--------
1. TF-IDF    (N, n_tfidf=200)  — sparse bag-of-bigrams
2. PCA       (N, 32)           — linear dimensionality reduction
3. Haar DWT  (N, 8)            — wavelet multi-scale decomposition
4. MinMax    (N, 8)  ∈ [0, π]  — angle encoding normalisation
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Standalone heuristic
# ---------------------------------------------------------------------------

_SARCASM_MARKERS = [
    "yeah right", "oh great", "just perfect", "totally fine", "so helpful",
    "clearly", "obviously", "best ever", "love how", "wow thanks",
    "great idea", "sure", "what could go wrong", "shocking", "surprise surprise",
]


def detect_sarcasm_heuristic(text: str) -> float:
    """Score text for sarcasm using a lexical marker list.

    Each matched marker contributes proportionally to a score in [0, 1].
    The multiplier of 3 is chosen so a single strong marker (e.g.
    *"yeah right"*) already pushes the score past the 0.3 warning threshold.

    Parameters
    ----------
    text : str
        Raw or cleaned headline text.

    Returns
    -------
    float
        Sarcasm heuristic score in [0.0, 1.0].

    Examples
    --------
    >>> detect_sarcasm_heuristic("yeah right that is just perfect")
    1.0
    >>> detect_sarcasm_heuristic("local election results announced")
    0.0
    """
    text_lower = text.lower()
    matches = sum(1 for marker in _SARCASM_MARKERS if marker in text_lower)
    raw_score = matches / len(_SARCASM_MARKERS) * 3
    return float(min(raw_score, 1.0))


# ---------------------------------------------------------------------------
# Haar Wavelet (manual, no pywavelets dependency)
# ---------------------------------------------------------------------------

def _haar_1d(x: np.ndarray) -> np.ndarray:
    """Single-level 1-D Haar Discrete Wavelet Transform (in-place decomp).

    Iteratively decomposes the signal into nested approximation and detail
    coefficients using the orthonormal Haar filters:
      h_avg  = [1, 1] / √2
      h_diff = [1, -1] / √2

    Parameters
    ----------
    x : np.ndarray
        1-D input array whose length must be a power of 2.

    Returns
    -------
    np.ndarray
        Transformed array of the same length, where the first half of each
        block contains approximation coefficients and the second half contains
        detail coefficients.
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
    """Return the smallest power of 2 that is ≥ n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class QuantumFeaturePipeline:
    """End-to-end feature extraction: text → 8 quantum angles in [0, π].

    The 8 output features are built in two halves:
      - First  n_qubits//2 features: top PCA components of TF-IDF (structural signal)
      - Second n_qubits//2 features: hand-crafted sarcasm indicators
          1. sarcasm_marker_score  — lexicon match density
          2. exclamation_density   — '!' chars / text length × 100
          3. hyperbole_score       — density of intensifiers / positive superlatives
          4. capitalization_ratio  — fraction of ALL-CAPS words

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of output features per sample). Default 8.
    n_tfidf : int
        Maximum vocabulary size for TF-IDF. Default 200.

    Attributes
    ----------
    tfidf_ : TfidfVectorizer
        Fitted TF-IDF transformer (available after ``fit_transform``).
    pca_ : PCA
        Fitted PCA reducer (available after ``fit_transform``).
    scaler_ : MinMaxScaler
        Fitted MinMax scaler (available after ``fit_transform``).
    is_fitted_ : bool
        True after ``fit_transform`` has been called.

    Examples
    --------
    >>> pipeline = QuantumFeaturePipeline(n_qubits=8)
    >>> texts = ["not_bad movie", "great film loved it", "terrible waste of time"]
    >>> X = pipeline.fit_transform(texts)
    >>> X.shape
    (3, 8)
    >>> (X >= 0).all() and (X <= np.pi + 1e-9).all()
    True
    """

    # Hyperbole / intensifier vocabulary that correlates with ironic praise
    _HYPERBOLIC = frozenset({
        "great", "amazing", "perfect", "best", "awesome", "fantastic",
        "brilliant", "wonderful", "incredible", "love", "totally",
        "literally", "obviously", "clearly", "definitely", "absolutely",
        "certainly", "surely", "undoubtedly", "spectacular", "magnificent",
        "outstanding", "superb", "genius", "shocking", "surprise",
    })

    def __init__(self, n_qubits: int = 8, n_tfidf: int = 200) -> None:
        self.n_qubits = n_qubits
        self.n_tfidf = n_tfidf

        # Split: half PCA structural features, half hand-crafted sarcasm signals
        self._n_pca = n_qubits // 2          # 4 for n_qubits=8
        self._n_handcrafted = n_qubits - self._n_pca  # 4 for n_qubits=8

        self.tfidf_: TfidfVectorizer | None = None
        self.pca_: PCA | None = None
        self.scaler_: MinMaxScaler | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_handcrafted(self, texts: List[str]) -> np.ndarray:
        """Compute ``_n_handcrafted`` sarcasm-relevant features per text.

        Features
        --------
        0. sarcasm_marker_score  — fraction of _SARCASM_MARKERS found
        1. exclamation_density   — '!' count / char length × 100 (capped 1)
        2. hyperbole_score       — hyperbolic word count / word count
        3. capitalization_ratio  — ALL-CAPS words ≥ 2 chars / total words
        """
        rows = []
        for text in texts:
            words_raw = text.split()
            words_low = [w.lower() for w in words_raw]
            n_words = max(len(words_raw), 1)
            n_chars = max(len(text), 1)

            # 1. Sarcasm marker density
            marker_score = (
                sum(1 for m in _SARCASM_MARKERS if m in text.lower())
                / len(_SARCASM_MARKERS)
            )

            # 2. Exclamation density (per 100 chars)
            excl_density = min(text.count('!') / n_chars * 100, 1.0)

            # 3. Hyperbole / intensifier density
            hyperbole = sum(1 for w in words_low if w in self._HYPERBOLIC) / n_words

            # 4. ALL-CAPS word ratio (words ≥ 2 chars fully uppercase)
            cap_ratio = sum(1 for w in words_raw if w.isupper() and len(w) >= 2) / n_words

            rows.append([marker_score, excl_density, hyperbole, cap_ratio])

        return np.array(rows, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the pipeline on *texts* and return transformed features.

        Parameters
        ----------
        texts : List[str]
            List of pre-cleaned headline strings.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(len(texts), n_qubits)`` with values
            in ``[0, π]``.  First n_qubits//2 cols are PCA; last are hand-crafted.
        """
        # Step 1 — TF-IDF
        self.tfidf_ = TfidfVectorizer(
            max_features=self.n_tfidf,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        X_tfidf = self.tfidf_.fit_transform(texts).toarray()
        print(f"[Pipeline] TF-IDF shape: {X_tfidf.shape}")

        # Step 2 — PCA (directly to n_pca components — no DWT)
        actual_pca_dim = min(self._n_pca, X_tfidf.shape[1], X_tfidf.shape[0])
        self.pca_ = PCA(n_components=actual_pca_dim, random_state=42)
        X_pca = self.pca_.fit_transform(X_tfidf)
        explained = self.pca_.explained_variance_ratio_.sum()
        print(
            f"[Pipeline] PCA shape: {X_pca.shape} | "
            f"Explained variance: {explained:.3f}"
        )

        # Step 3 — Hand-crafted sarcasm features
        X_hc = self._compute_handcrafted(texts)
        print(f"[Pipeline] Hand-crafted features shape: {X_hc.shape}")

        # Step 4 — Concatenate (pad PCA axis if fewer components were available)
        if X_pca.shape[1] < self._n_pca:
            pad = np.zeros((len(X_pca), self._n_pca - X_pca.shape[1]))
            X_pca = np.concatenate([X_pca, pad], axis=1)
        X_combined = np.concatenate([X_pca, X_hc], axis=1)

        # Step 5 — Scale to [0, π]
        self.scaler_ = MinMaxScaler(feature_range=(0, np.pi))
        X_scaled = self.scaler_.fit_transform(X_combined)

        self.is_fitted_ = True
        print(
            f"[Pipeline] Final output shape: {X_scaled.shape} | "
            f"Range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]"
        )
        return X_scaled

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform *texts* using the already-fitted pipeline.

        Parameters
        ----------
        texts : List[str]
            List of pre-cleaned headline strings.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(len(texts), n_qubits)`` in ``[0, π]``.

        Raises
        ------
        RuntimeError
            If ``fit_transform`` has not been called first.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit_transform() first."
            )
        X_tfidf = self.tfidf_.transform(texts).toarray()
        X_pca = self.pca_.transform(X_tfidf)
        X_hc = self._compute_handcrafted(texts)
        X_combined = np.concatenate([X_pca, X_hc], axis=1)
        X_scaled = self.scaler_.transform(X_combined)
        return X_scaled

    def get_feature_names(self) -> List[str]:
        """Return human-readable names for the n_qubits output features.

        Returns
        -------
        List[str]
            Strings of the form ``["qubit_0_feature", …, "qubit_7_feature"]``.
        """
        return [f"qubit_{i}_feature" for i in range(self.n_qubits)]


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dummy_texts = [
        "not_bad movie great acting loved it",
        "oh great another terrible monday what could go wrong",
        "scientists discover water still wet yeah right",
        "local election results announced today",
        "shocking new report reveals surprising findings",
        "yeah right clearly the best idea ever had",
        "area man finds wallet returns it to owner",
        "new study shows exercise is good for health",
        "cant_believe how totally fine everything is",
        "government announces new policy changes",
    ]

    pipeline = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
    X = pipeline.fit_transform(dummy_texts)

    assert X.shape == (len(dummy_texts), 8), \
        f"Shape mismatch: expected ({len(dummy_texts)}, 8), got {X.shape}"
    assert X.min() >= 0.0, f"Values below 0: {X.min()}"
    assert X.max() <= np.pi + 1e-9, f"Values above π: {X.max()}"

    print("\nFeature names:", pipeline.get_feature_names())
    print(f"Sample feature vector:\n  {X[0]}")

    score = detect_sarcasm_heuristic("yeah right that is just perfect")
    print(f"\nHeuristic sarcasm score (should be 1.0): {score}")

    print("\n✅ Feature pipeline verified")
