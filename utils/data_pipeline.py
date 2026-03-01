"""
data_pipeline.py
================
Loads, cleans, and prepares the Sarcasm Headlines Dataset for downstream NLP
and quantum machine learning.

Pipeline:
    JSON → clean_text → negation_binding → remove_stopwords → DataFrame
"""

import re
import string
import pandas as pd
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

# Negation words that must survive stopword removal so TF-IDF can capture them.
NEGATION_WORDS = [
    "not", "never", "no", "neither", "nor",
    "isn't", "wasn't", "don't", "didn't", "won't",
    "can't", "couldn't", "shouldn't", "hardly", "barely",
]


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/digits/punctuation, strip extra whitespace.

    Negation words are intentionally preserved so that negation_binding can
    later join them with the following token.

    Parameters
    ----------
    text : str
        Raw headline string.

    Returns
    -------
    str
        Cleaned text with normalised whitespace.
    """
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Remove punctuation (keep apostrophes temporarily for contractions like "can't")
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def negation_binding(text: str) -> str:
    """Join each negation word with the immediately following word using '_'.

    Example
    -------
    "not good"  → "not_good"
    "can't win" → "cant_win"

    This ensures TF-IDF treats "not_good" as a single discriminative token
    rather than discarding "not" as a stopword.

    Parameters
    ----------
    text : str
        Cleaned headline string.

    Returns
    -------
    str
        Text with negation-bound bigrams.
    """
    tokens = text.split()
    result = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        # Strip apostrophes for matching (e.g. "can't" → "cant")
        normalised = token.replace("'", "")
        if normalised in NEGATION_WORDS or token in NEGATION_WORDS:
            if i + 1 < len(tokens):
                result.append(normalised + "_" + tokens[i + 1])
                skip_next = True
            else:
                result.append(normalised)
        else:
            result.append(token)
    return " ".join(result)


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords while preserving negation tokens.

    Negation compound tokens (e.g. "not_good") and standalone negation words
    are always kept because they carry sentiment signal.

    Parameters
    ----------
    text : str
        Text after negation binding.

    Returns
    -------
    str
        Text with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    # Ensure none of the negation words are accidentally removed
    negation_set = set(NEGATION_WORDS) | {w.replace("'", "") for w in NEGATION_WORDS}
    stop_words -= negation_set

    tokens = text.split()
    filtered = [t for t in tokens if t not in stop_words or "_" in t]
    return " ".join(filtered)


def load_and_clean_dataset(path: str) -> pd.DataFrame:
    """Load the Sarcasm Headlines Dataset and apply the full cleaning pipeline.

    The JSON file is expected to be line-delimited (one JSON object per line)
    with the columns: ``is_sarcastic``, ``headline``, ``article_link``.

    Pipeline applied to each headline:
        1. :func:`clean_text`
        2. :func:`negation_binding`
        3. :func:`remove_stopwords`

    Parameters
    ----------
    path : str
        Path to ``Sarcasm_Headlines_Dataset_v2.json``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['headline', 'clean_text', 'label']``.
    """
    df = pd.read_json(path, lines=True)
    df = df[["headline", "is_sarcastic"]].copy()
    df.rename(columns={"is_sarcastic": "label"}, inplace=True)
    df.dropna(subset=["headline"], inplace=True)

    df["clean_text"] = (
        df["headline"]
        .apply(clean_text)
        .apply(negation_binding)
        .apply(remove_stopwords)
    )

    print(f"Dataset loaded: {len(df):,} samples")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    return df[["headline", "clean_text", "label"]]


if __name__ == "__main__":
    import os

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "Sarcasm_Headlines_Dataset_v2.json"
    )
    dataset = load_and_clean_dataset(data_path)
    print("\n--- 5 sample rows ---")
    print(dataset.head(5).to_string(index=False))
    print("\nLabel distribution:")
    print(dataset["label"].value_counts())
