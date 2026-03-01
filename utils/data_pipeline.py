"""
utils/data_pipeline.py
======================
NLP data pipeline for quantum sarcasm detection.

Loads the Sarcasm Headlines Dataset, applies cleaning, negation binding,
and stopword removal to prepare text for downstream feature engineering.

Pipeline order: raw text → clean_text → negation_binding → remove_stopwords
"""

import re
import string
import pandas as pd
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEGATION_WORDS = [
    "not", "never", "no", "neither", "nor",
    "isn't", "wasn't", "don't", "didn't", "won't",
    "can't", "couldn't", "shouldn't", "hardly", "barely",
]

# Normalised negation tokens that appear after contraction stripping
_NEGATION_TOKENS = {
    "isnt", "wasnt", "dont", "didnt", "wont",
    "cant", "couldnt", "shouldnt",
    "not", "never", "no", "neither", "nor",
    "hardly", "barely",
}


def clean_text(text: str) -> str:
    """Lower-case, strip URLs / digits / punctuation, collapse whitespace.

    Deliberately does **not** remove negation words — those are handled
    separately by :func:`negation_binding`.

    Parameters
    ----------
    text : str
        Raw headline string.

    Returns
    -------
    str
        Cleaned text ready for negation binding.

    Examples
    --------
    >>> clean_text("Trump Wins! Again?! https://t.co/abc 123")
    'trump wins again'
    """
    text = text.lower()
    # Remove URLs (http/https/www)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Expand common contractions before punctuation removal so we can bind
    # negation tokens ("isn't" → "isnt" preserves the negation signal)
    contractions = {
        "isn't": "isnt", "wasn't": "wasnt", "aren't": "arent",
        "weren't": "werent", "don't": "dont", "doesn't": "doesnt",
        "didn't": "didnt", "won't": "wont", "can't": "cant",
        "couldn't": "couldnt", "shouldn't": "shouldnt",
        "wouldn't": "wouldnt", "haven't": "havent", "hasn't": "hasnt",
        "hadn't": "hadnt", "mightn't": "mightnt", "mustn't": "mustnt",
    }
    for contracted, expanded in contractions.items():
        text = text.replace(contracted, expanded)
    # Remove remaining punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def negation_binding(text: str) -> str:
    """Join each negation word with its immediately following word.

    This ensures TF-IDF treats *not_good* as a single feature token instead
    of two independent tokens, preserving directional sentiment.

    Parameters
    ----------
    text : str
        Cleaned text (output of :func:`clean_text`).

    Returns
    -------
    str
        Text with negation-bound bigrams, e.g. "not_good", "cant_win".

    Examples
    --------
    >>> negation_binding("this is not good at all")
    'this is not_good at all'
    >>> negation_binding("cant stop wont stop")
    'cant_stop wont_stop'
    """
    tokens = text.split()
    bound = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in _NEGATION_TOKENS and i + 1 < len(tokens):
            # bind with following word
            bound.append(f"{token}_{tokens[i + 1]}")
            skip_next = True
        else:
            bound.append(token)
    return " ".join(bound)


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords while **preserving** negation words.

    Negation words and their bound variants (e.g. *not_good*) carry critical
    sentiment signal and must survive stopword removal.

    Parameters
    ----------
    text : str
        Text after negation binding.

    Returns
    -------
    str
        Text with stopwords removed (negation words retained).

    Examples
    --------
    >>> remove_stopwords("the cat sat on the mat")
    'cat sat mat'
    >>> remove_stopwords("not_bad at the moment")
    'not_bad moment'
    """
    sw = set(stopwords.words("english")) - _NEGATION_TOKENS
    tokens = text.split()
    # Keep token if it is NOT a stopword, OR if it contains an underscore
    # (negation-bound bigram — always preserve)
    filtered = [t for t in tokens if t not in sw or "_" in t]
    return " ".join(filtered)


def load_and_clean_dataset(path: str) -> pd.DataFrame:
    """Load, clean, and preprocess the Sarcasm Headlines Dataset.

    Applies the full cleaning pipeline in sequence:
    ``raw → clean_text → negation_binding → remove_stopwords``

    Parameters
    ----------
    path : str
        Path to the line-delimited JSON file
        (``data/Sarcasm_Headlines_Dataset_v2.json``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``headline``  : original headline text
        - ``clean_text``: fully pre-processed text
        - ``label``     : 1 = sarcastic, 0 = not sarcastic

    Raises
    ------
    FileNotFoundError
        If the dataset file cannot be located.
    """
    # Load line-delimited JSON (each line is a JSON object)
    df = pd.read_json(path, lines=True)

    # Validate expected columns
    required_cols = {"is_sarcastic", "headline"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset missing expected columns. Found: {list(df.columns)}"
        )

    # Drop rows with missing headlines
    df = df.dropna(subset=["headline"])
    df = df.reset_index(drop=True)

    print(f"📦 Loaded {len(df):,} samples from '{path}'")
    print(f"   Class distribution:\n{df['is_sarcastic'].value_counts().to_string()}")

    # Apply pipeline
    print("🔧 Applying cleaning pipeline …")
    df["clean_text"] = (
        df["headline"]
        .apply(clean_text)
        .apply(negation_binding)
        .apply(remove_stopwords)
    )

    # Rename label column
    df = df.rename(columns={"is_sarcastic": "label"})

    # Keep only the three relevant columns
    df = df[["headline", "clean_text", "label"]]

    print(
        f"✅ Cleaning done. "
        f"Sarcastic: {df['label'].sum():,} | "
        f"Not sarcastic: {(df['label'] == 0).sum():,}"
    )
    return df


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    data_path = os.path.join("data", "Sarcasm_Headlines_Dataset_v2.json")
    df = load_and_clean_dataset(data_path)

    print("\n" + "=" * 60)
    print("SAMPLE HEADLINES (5 sarcastic, 5 not)")
    print("=" * 60)
    for _, row in df[df["label"] == 1].head(5).iterrows():
        print(f"  [SARCASTIC] {row['headline']}")
        print(f"              → {row['clean_text']}\n")
    for _, row in df[df["label"] == 0].head(5).iterrows():
        print(f"  [NORMAL]    {row['headline']}")
        print(f"              → {row['clean_text']}\n")

    print(f"Total samples: {len(df):,}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
