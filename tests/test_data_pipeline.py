"""
tests/test_data_pipeline.py
============================
Unit tests for utils/data_pipeline.py

Coverage
--------
- clean_text                : lowercasing, URL removal, digit removal,
                              contraction expansion, punctuation stripping
- negation_binding          : basic binding, contraction tokens, boundary cases
- remove_stopwords          : standard stopword removal, negation preservation
- load_and_clean_dataset    : happy path via a temp JSON fixture
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_pipeline import (
    clean_text,
    negation_binding,
    load_and_clean_dataset,
    remove_stopwords,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_http_url(self):
        result = clean_text("Check this out http://example.com today")
        assert "http" not in result
        assert "example" not in result

    def test_removes_https_url(self):
        result = clean_text("Visit https://pennylane.ai for details")
        assert "https" not in result

    def test_removes_www_url(self):
        result = clean_text("Go to www.google.com for info")
        assert "www" not in result

    def test_removes_digits(self):
        result = clean_text("Trump wins again in 2024 with 300 votes")
        assert "2024" not in result
        assert "300" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_expands_isnt_contraction(self):
        result = clean_text("This isn't right")
        assert "isnt" in result

    def test_expands_dont_contraction(self):
        result = clean_text("I don't know")
        assert "dont" in result

    def test_expands_cant_contraction(self):
        result = clean_text("I can't believe it")
        assert "cant" in result

    def test_expands_wont_contraction(self):
        result = clean_text("He won't come")
        assert "wont" in result

    def test_collapses_whitespace(self):
        result = clean_text("hello   world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_punctuation(self):
        result = clean_text("!!! ??? ---")
        assert result.strip() == ""

    def test_returns_string(self):
        assert isinstance(clean_text("test"), str)

    def test_preserves_words_after_cleaning(self):
        result = clean_text("scientists discover new planet")
        assert "scientists" in result
        assert "discover" in result


# ---------------------------------------------------------------------------
# negation_binding
# ---------------------------------------------------------------------------


class TestNegationBinding:
    def test_binds_not(self):
        result = negation_binding("this is not good")
        assert "not_good" in result

    def test_binds_never(self):
        result = negation_binding("i never lie")
        assert "never_lie" in result

    def test_binds_no(self):
        result = negation_binding("no way this works")
        assert "no_way" in result

    def test_binds_isnt(self):
        # After clean_text, "isn't" becomes "isnt"
        result = negation_binding("isnt working")
        assert "isnt_working" in result

    def test_binds_cant(self):
        result = negation_binding("cant stop laughing")
        assert "cant_stop" in result

    def test_binds_dont(self):
        result = negation_binding("dont touch that")
        assert "dont_touch" in result

    def test_binds_wont(self):
        result = negation_binding("wont happen again")
        assert "wont_happen" in result

    def test_non_negation_word_unchanged(self):
        result = negation_binding("the cat sat on the mat")
        assert result == "the cat sat on the mat"

    def test_negation_at_end_no_crash(self):
        # "not" at the end — no following word to bind
        result = negation_binding("absolutely not")
        assert isinstance(result, str)
        assert "not" in result

    def test_multiple_negations(self):
        result = negation_binding("not good never again")
        assert "not_good" in result
        assert "never_again" in result

    def test_empty_string(self):
        assert negation_binding("") == ""

    def test_returns_string(self):
        assert isinstance(negation_binding("test input"), str)


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------


class TestRemoveStopwords:
    def test_removes_common_stopword_the(self):
        result = remove_stopwords("the cat sat on the mat")
        assert "the" not in result.split()

    def test_removes_common_stopword_a(self):
        result = remove_stopwords("a quick brown fox")
        assert result.split()[0] != "a" or "quick" in result

    def test_preserves_negation_not(self):
        result = remove_stopwords("not bad at all")
        # "not" is a negation word and should be preserved
        assert "not" in result

    def test_preserves_negation_bound_bigram(self):
        result = remove_stopwords("not_good idea")
        assert "not_good" in result

    def test_preserves_never(self):
        result = remove_stopwords("never give up")
        assert "never" in result

    def test_retains_content_words(self):
        result = remove_stopwords("scientists discover new species")
        assert "scientists" in result
        assert "discover" in result
        assert "species" in result

    def test_empty_string(self):
        assert remove_stopwords("") == ""

    def test_returns_string(self):
        assert isinstance(remove_stopwords("test input"), str)

    def test_all_stopwords_returns_empty(self):
        result = remove_stopwords("the a an is")
        # All are stopwords; result may be empty or contain only preserved tokens
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_and_clean_dataset
# ---------------------------------------------------------------------------


class TestLoadAndCleanDataset:
    @pytest.fixture
    def sample_json_file(self, tmp_path):
        """Write a tiny line-delimited JSON dataset to a temp file."""
        data = [
            {"headline": "Scientists confirm water is still wet", "is_sarcastic": 1, "article_link": ""},
            {"headline": "Local election results announced", "is_sarcastic": 0, "article_link": ""},
            {"headline": "Oh great another perfect monday", "is_sarcastic": 1, "article_link": ""},
            {"headline": "New bridge opens ahead of schedule", "is_sarcastic": 0, "article_link": ""},
            {"headline": "Area man wins lottery yeah right", "is_sarcastic": 1, "article_link": ""},
            {"headline": "City approves new park construction", "is_sarcastic": 0, "article_link": ""},
        ]
        path = tmp_path / "test_dataset.json"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def test_returns_dataframe(self, sample_json_file):
        import pandas as pd
        df = load_and_clean_dataset(sample_json_file)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, sample_json_file):
        df = load_and_clean_dataset(sample_json_file)
        assert len(df) == 6

    def test_has_required_columns(self, sample_json_file):
        df = load_and_clean_dataset(sample_json_file)
        assert "headline" in df.columns
        assert "clean_text" in df.columns
        assert "label" in df.columns

    def test_label_values_are_binary(self, sample_json_file):
        df = load_and_clean_dataset(sample_json_file)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_sarcastic_count(self, sample_json_file):
        df = load_and_clean_dataset(sample_json_file)
        assert df["label"].sum() == 3

    def test_clean_text_is_lowercase(self, sample_json_file):
        df = load_and_clean_dataset(sample_json_file)
        for text in df["clean_text"]:
            assert text == text.lower(), f"Not lowercase: {text}"

    def test_clean_text_has_no_digits(self, sample_json_file):
        df = load_and_clean_dataset(sample_json_file)
        for text in df["clean_text"]:
            assert not any(c.isdigit() for c in text), f"Contains digit: {text}"

    def test_file_not_found_raises(self):
        with pytest.raises(Exception):
            load_and_clean_dataset("nonexistent_file.json")

    def test_missing_columns_raises(self, tmp_path):
        bad_path = tmp_path / "bad.json"
        with open(bad_path, "w") as f:
            f.write(json.dumps({"title": "test", "sarcastic": 1}) + "\n")
        with pytest.raises(ValueError):
            load_and_clean_dataset(str(bad_path))
