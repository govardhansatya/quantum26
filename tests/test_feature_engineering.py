"""
tests/test_feature_engineering.py
===================================
Unit tests for utils/feature_engineering.py

Coverage
--------
- detect_sarcasm_heuristic       : score range, positive/negative cases
- QuantumFeaturePipeline         : shape, value range, transform/fit_transform,
                                   unfitted error, feature names, hand-crafted features
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import (
    QuantumFeaturePipeline,
    detect_sarcasm_heuristic,
)


# ---------------------------------------------------------------------------
# detect_sarcasm_heuristic
# ---------------------------------------------------------------------------


class TestDetectSarcasmHeuristic:
    def test_returns_float(self):
        score = detect_sarcasm_heuristic("test text")
        assert isinstance(score, float)

    def test_score_in_range(self):
        for text in [
            "yeah right that is just perfect",
            "local election results announced",
            "oh great another monday what could go wrong",
            "",
        ]:
            score = detect_sarcasm_heuristic(text)
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for: {text}"

    def test_strong_sarcasm_high_score(self):
        score = detect_sarcasm_heuristic("yeah right that is just perfect")
        assert score >= 0.3, f"Expected high sarcasm score, got {score}"

    def test_double_marker_higher_than_single(self):
        single = detect_sarcasm_heuristic("oh great")
        double = detect_sarcasm_heuristic("oh great yeah right")
        assert double >= single

    def test_neutral_text_low_score(self):
        score = detect_sarcasm_heuristic("local election results announced today")
        assert score < 0.3, f"Expected low sarcasm score for neutral text, got {score}"

    def test_empty_string_returns_zero(self):
        assert detect_sarcasm_heuristic("") == 0.0

    def test_max_capped_at_one(self):
        # Text saturated with every marker
        many_markers = " ".join(
            [
                "yeah right oh great just perfect totally fine so helpful",
                "clearly obviously best ever love how wow thanks great idea",
                "sure what could go wrong shocking surprise surprise",
            ]
        )
        score = detect_sarcasm_heuristic(many_markers)
        assert score == 1.0

    def test_case_insensitive(self):
        lower = detect_sarcasm_heuristic("yeah right")
        upper = detect_sarcasm_heuristic("YEAH RIGHT")
        # Both should produce the same score (text is lower-cased internally)
        assert lower == upper


# ---------------------------------------------------------------------------
# QuantumFeaturePipeline
# ---------------------------------------------------------------------------


class TestQuantumFeaturePipeline:
    @pytest.fixture
    def texts(self):
        return [
            "not_bad movie great acting loved it",
            "oh great another monday what could go wrong",
            "scientists discover water still wet yeah right",
            "local election results announced today",
            "shocking report reveals surprising findings",
            "yeah right clearly best idea ever had",
            "area man finds wallet returns it",
            "exercise good health study shows",
            "cant_believe totally fine everything",
            "government announces new policy changes",
        ]

    def test_fit_transform_returns_ndarray(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        X = pipe.fit_transform(texts)
        assert isinstance(X, np.ndarray)

    def test_fit_transform_shape(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        X = pipe.fit_transform(texts)
        assert X.shape == (len(texts), 8), f"Expected ({len(texts)}, 8), got {X.shape}"

    def test_values_in_zero_pi_range(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        X = pipe.fit_transform(texts)
        assert X.min() >= 0.0, f"Min value {X.min()} below 0"
        assert X.max() <= np.pi + 1e-9, f"Max value {X.max()} above π"

    def test_is_fitted_flag(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        assert not pipe.is_fitted_
        pipe.fit_transform(texts)
        assert pipe.is_fitted_

    def test_transform_after_fit(self, fitted_quantum_pipeline, texts):
        X = fitted_quantum_pipeline.transform(texts)
        assert X.shape == (len(texts), 8)
        assert X.min() >= 0.0
        assert X.max() <= np.pi + 1e-9

    def test_transform_raises_if_not_fitted(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        with pytest.raises(RuntimeError):
            pipe.transform(texts)

    def test_transform_consistent_with_fit_transform(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        X_fit = pipe.fit_transform(texts)
        X_transform = pipe.transform(texts)
        np.testing.assert_allclose(X_fit, X_transform, atol=1e-8)

    def test_feature_names_length(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        pipe.fit_transform(texts)
        names = pipe.get_feature_names()
        assert len(names) == 8

    def test_feature_names_are_strings(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        pipe.fit_transform(texts)
        names = pipe.get_feature_names()
        assert all(isinstance(n, str) for n in names)

    def test_n_qubits_4_produces_4_features(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=4, n_tfidf=30)
        X = pipe.fit_transform(texts)
        assert X.shape[1] == 4

    def test_different_texts_produce_different_features(self, texts):
        pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=50)
        X = pipe.fit_transform(texts)
        # Not all rows should be identical
        assert not np.allclose(X[0], X[1]), "Different texts produced identical features"

    def test_single_text_also_works(self, fitted_quantum_pipeline):
        X = fitted_quantum_pipeline.transform(["oh great another terrible day"])
        assert X.shape == (1, 8)

    @pytest.mark.parametrize("text,expected_hi", [
        ("oh great yeah right just perfect", True),   # lots of markers → high h-crafted
        ("annual rainfall below average region", False),  # neutral
    ])
    def test_handcrafted_features_direction(self, text, expected_hi, fitted_quantum_pipeline):
        X = fitted_quantum_pipeline.transform([text])
        # Last 4 features are hand-crafted; sarcastic text should have at least one > 0
        hc = X[0, 4:]
        if expected_hi:
            assert hc.max() > 0.0
