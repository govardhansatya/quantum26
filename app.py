"""
app.py
======
Streamlit frontend for the Quantum vs Classical Sarcasm Detector.

Pages
-----
1. 🏠  Home            — Project overview and architecture summary
2. 🔍  Live Predictor  — Single / batch headline classification
3. 📊  Results Dashboard — Charts and tables from experiment results
4. ▶️   Run Experiment  — Trigger a quick demo experiment from the UI
5. ⚛️   Circuit Explorer — VQC circuit diagram + kernel kernel visualisation

Run
---
    streamlit run app.py
"""

from __future__ import annotations

# Must be the very first Streamlit call
import streamlit as st

st.set_page_config(
    page_title="Quantum Sarcasm Detector",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import io
import os
import sys
import time
import traceback
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_pipeline import load_and_clean_dataset, clean_text, negation_binding, remove_stopwords
from utils.feature_engineering import QuantumFeaturePipeline, detect_sarcasm_heuristic
from models.classical_models import (
    ClassicalSVMBaseline,
    LogisticRegressionBaseline,
    NaiveBayesBaseline,
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

DATA_PATH = "data/Sarcasm_Headlines_Dataset_v2.json"
RESULTS_CSV = "results/experiment_results.csv"
SAVED_MODELS_DIR = "results/saved_models"

MODEL_COLORS = {
    "Logistic Regression":  "#2196F3",
    "Classical SVM (RBF)":  "#4CAF50",
    "Naive Bayes":          "#FF9800",
    "VQC (ZZ + SEL)":       "#F44336",
    "Quantum Kernel SVM":   "#9C27B0",
}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="📦 Loading and cleaning dataset …")
def _load_dataset() -> pd.DataFrame:
    return load_and_clean_dataset(DATA_PATH)


@st.cache_resource(show_spinner="🔧 Fitting quantum feature pipeline …")
def _fit_quantum_pipeline(texts: tuple) -> QuantumFeaturePipeline:
    pipeline = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
    pipeline.fit_transform(list(texts))
    return pipeline


@st.cache_resource(show_spinner="🔧 Fitting classical TF-IDF + models …")
def _fit_classical_models(
    texts: tuple, labels: tuple
) -> dict:
    """Fit TF-IDF + all 3 classical baselines."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MaxAbsScaler

    tfidf = TfidfVectorizer(
        max_features=2000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
    )
    X_raw = tfidf.fit_transform(list(texts))
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X_raw).toarray()
    y = np.array(labels)

    lr = LogisticRegressionBaseline()
    svm = ClassicalSVMBaseline()
    nb = NaiveBayesBaseline()

    lr.fit(X, y)
    svm.fit(X, y)
    nb.fit(X, y)

    return {"tfidf": tfidf, "scaler": scaler, "lr": lr, "svm": svm, "nb": nb}


@st.cache_resource(show_spinner="⚛️ Training VQC (quick demo, 20 epochs) …")
def _fit_vqc(texts: tuple, labels: tuple) -> object:
    """Train a quick-demo VQC on the full dataset."""
    from models.vqc_model import VQCClassifier

    pipeline = _fit_quantum_pipeline(texts)
    X = pipeline.fit_transform(list(texts))          # re-fit for VQC (same fitted state)
    y = np.array(labels)
    model = VQCClassifier(n_layers=2, max_iter=20, lr=0.1, batch_size=64)
    model.fit(X, y)
    return model, pipeline


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _preprocess(text: str) -> str:
    return remove_stopwords(negation_binding(clean_text(text)))


def _predict_all(
    raw_texts: List[str],
    clf_bundle: dict,
    quantum_pipeline: QuantumFeaturePipeline,
    vqc_model=None,
) -> pd.DataFrame:
    """Run all available models on a list of raw headlines.

    Returns a DataFrame with one row per (headline × model).
    """
    rows = []
    cleaned = [_preprocess(t) for t in raw_texts]

    # --- Classical features ---
    X_clf = clf_bundle["scaler"].transform(
        clf_bundle["tfidf"].transform(cleaned)
    ).toarray()

    # --- Quantum features ---
    X_q = quantum_pipeline.transform(cleaned)

    for i, raw in enumerate(raw_texts):
        x_clf = X_clf[i : i + 1]
        x_q = X_q[i : i + 1]

        heuristic = detect_sarcasm_heuristic(raw)
        rows.append(
            {
                "headline": raw,
                "model": "Heuristic (Lexical)",
                "prediction": "Sarcastic" if heuristic >= 0.3 else "Not Sarcastic",
                "confidence": round(heuristic, 3),
                "label_int": int(heuristic >= 0.3),
            }
        )

        for name, model, x in [
            ("Logistic Regression", clf_bundle["lr"], x_clf),
            ("Classical SVM (RBF)", clf_bundle["svm"], x_clf),
            ("Naive Bayes",         clf_bundle["nb"],  x_clf),
        ]:
            try:
                proba = model.predict_proba(x)[0]
                pred = int(proba[1] >= 0.5)
                rows.append(
                    {
                        "headline": raw,
                        "model": name,
                        "prediction": "Sarcastic" if pred else "Not Sarcastic",
                        "confidence": round(float(proba[1]), 3),
                        "label_int": pred,
                    }
                )
            except Exception:
                pass

        if vqc_model is not None:
            try:
                proba = vqc_model.predict_proba(x_q)[0]
                pred = int(proba[1] >= 0.5)
                rows.append(
                    {
                        "headline": raw,
                        "model": "VQC (ZZ + SEL)",
                        "prediction": "Sarcastic" if pred else "Not Sarcastic",
                        "confidence": round(float(proba[1]), 3),
                        "label_int": pred,
                    }
                )
            except Exception:
                pass

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


def page_home() -> None:
    st.title("⚛️ Quantum Sarcasm Detector")
    st.markdown(
        """
        **A hybrid *quantum-classical* NLP system** for detecting sarcasm in news
        headlines, demonstrating *quantum advantage in the low-data regime*.

        > *When training data is scarce (N ≤ 200 samples), quantum models
        > operating in the 2⁸ = 256-dimensional Hilbert space outperform
        > classical baselines.*
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗂 Dataset")
        st.markdown(
            """
            **Sarcasm Headlines Dataset v2**
            - ~28 000 news headlines from *The Onion* (sarcastic) and *HuffPost* (genuine)
            - Binary labels: `1` = sarcastic, `0` = not sarcastic
            - Source: Misra & Arora, 2023 (arXiv:1908.07414)
            """
        )
        st.subheader("🛠 Classical NLP Pre-processing")
        st.code(
            "raw text\n"
            "  → clean_text()       # lower, strip URLs/digits/punct\n"
            "  → negation_binding() # \"not good\" → \"not_good\"\n"
            "  → remove_stopwords() # keep negation tokens",
            language="text",
        )

    with col2:
        st.subheader("🔀 Feature Pipelines")
        st.markdown(
            """
            | Path | Steps | Output |
            |------|-------|--------|
            | **Classical** | TF-IDF (2 000, bigrams) → MaxAbsScaler | (N, 2 000) |
            | **Quantum** | TF-IDF (200) → PCA (4) + hand-crafted (4) → MinMax [0,π] | (N, 8) |
            """
        )
        st.subheader("🤖 Models")
        st.markdown(
            """
            | Model | Type | Input |
            |-------|------|-------|
            | Logistic Regression | Classical | 2 000-dim TF-IDF |
            | Classical SVM (RBF) | Classical | 2 000-dim TF-IDF |
            | Naive Bayes | Classical | 2 000-dim TF-IDF |
            | **VQC (ZZ + SEL)** | **Quantum** | **8 quantum angles** |
            | Quantum Kernel SVM | Quantum | 8 quantum angles |
            """
        )

    st.subheader("⚛️ VQC Architecture")
    st.code(
        "Input  x ∈ [0,π]⁸\n"
        "  ① AngleEmbedding (Ry)         — encode features as Ry rotations\n"
        "  ② ZZ Feature Map              — CNOT·RZ(2(π−xᵢ)(π−xⱼ))·CNOT\n"
        "  ③ AngleEmbedding (Rz)         — data re-uploading\n"
        "  ④ StronglyEntanglingLayers×2  — 48 trainable parameters\n"
        "  ⑤ Measure mean⟨Z⟩ → sigmoid  — probability ∈ [0,1]",
        language="text",
    )

    st.subheader("📁 Project Structure")
    st.code(
        "quantum26/\n"
        "├── app.py                  ← this file (Streamlit UI)\n"
        "├── data/                   ← headline datasets\n"
        "├── models/\n"
        "│   ├── classical_models.py ← LR, SVM, NB baselines\n"
        "│   ├── vqc_model.py        ← VQC (PennyLane)\n"
        "│   └── quantum_kernel_svm.py ← QKSVM\n"
        "├── utils/\n"
        "│   ├── data_pipeline.py    ← NLP cleaning\n"
        "│   └── feature_engineering.py ← quantum feature extraction\n"
        "├── experiments/\n"
        "│   ├── run_experiments.py  ← benchmark runner\n"
        "│   └── plot_results.py     ← publication-quality plots\n"
        "├── tests/                  ← pytest test suite\n"
        "└── results/                ← CSVs, PNGs, saved models",
        language="text",
    )


# ---------------------------------------------------------------------------


def page_live_predictor() -> None:
    st.title("🔍 Live Sarcasm Predictor")
    st.markdown(
        "Type one or more headlines below and all models will instantly classify them."
    )

    # --- Load data & models ---
    with st.spinner("Loading models (first run only — ~30 s) …"):
        try:
            df_data = _load_dataset()
            texts_tuple = tuple(df_data["clean_text"].tolist())
            labels_tuple = tuple(df_data["label"].tolist())
            clf_bundle = _fit_classical_models(texts_tuple, labels_tuple)
            quantum_pipeline = _fit_quantum_pipeline(texts_tuple)
        except Exception as exc:
            st.error(f"❌ Model loading failed: {exc}")
            st.code(traceback.format_exc())
            return

    # VQC is opt-in due to training time
    use_vqc = st.checkbox(
        "⚛️ Include VQC prediction (slower — trains 20 epochs on first use)",
        value=False,
    )
    vqc_model = None
    if use_vqc:
        with st.spinner("Training VQC …"):
            try:
                vqc_model, _ = _fit_vqc(texts_tuple, labels_tuple)
            except Exception as exc:
                st.warning(f"VQC unavailable: {exc}")

    st.divider()
    mode = st.radio("Input mode", ["Single headline", "Batch (one per line)"], horizontal=True)

    if mode == "Single headline":
        headline = st.text_input(
            "Enter a headline",
            placeholder="Scientists confirm water is still wet, world shocked",
        )
        headlines = [headline] if headline.strip() else []
    else:
        raw_block = st.text_area(
            "Headlines (one per line)",
            height=160,
            placeholder=(
                "Area man wins lottery, immediately loses ticket\n"
                "New study confirms exercise remains good for health"
            ),
        )
        headlines = [h.strip() for h in raw_block.splitlines() if h.strip()]

    if not headlines:
        st.info("Enter at least one headline above.")
        return

    # Upload CSV option
    uploaded = st.file_uploader(
        "Or upload a CSV with a 'headline' column", type=["csv"], key="pred_upload"
    )
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "headline" not in df_up.columns:
                st.error("CSV must contain a 'headline' column.")
            else:
                headlines = df_up["headline"].dropna().tolist()
                st.success(f"Loaded {len(headlines)} headlines from CSV.")
        except Exception as exc:
            st.error(f"CSV parse error: {exc}")
            return

    with st.spinner("Running predictions …"):
        results = _predict_all(headlines, clf_bundle, quantum_pipeline, vqc_model)

    st.subheader(f"Results — {len(headlines)} headline(s)")

    for headline in headlines:
        sub = results[results["headline"] == headline]
        with st.expander(f"📰 {headline[:120]}", expanded=True):
            cols = st.columns(len(sub))
            for col, (_, row) in zip(cols, sub.iterrows()):
                color = "#F44336" if row["label_int"] else "#4CAF50"
                icon  = "😏" if row["label_int"] else "📰"
                col.markdown(
                    f"<div style='text-align:center; padding:8px; border-radius:8px; "
                    f"background:{color}22; border:1px solid {color}66'>"
                    f"<b style='font-size:0.78em'>{row['model']}</b><br>"
                    f"<span style='font-size:1.6em'>{icon}</span><br>"
                    f"<b style='color:{color}'>{row['prediction']}</b><br>"
                    f"<span style='font-size:0.82em; color:grey'>"
                    f"conf: {row['confidence']:.2f}</span></div>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # Confidence comparison bar chart
    if len(headlines) == 1:
        st.subheader("Confidence Breakdown")
        fig, ax = plt.subplots(figsize=(8, 3))
        sub = results[results["headline"] == headlines[0]]
        bars = ax.barh(
            sub["model"],
            sub["confidence"],
            color=[
                MODEL_COLORS.get(m, "#607D8B")
                for m in sub["model"]
            ],
            edgecolor="white",
        )
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Decision boundary")
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(sarcastic)")
        ax.set_title("Sarcasm probability — all models")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, sub["confidence"]):
            ax.text(
                min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Full results table
    with st.expander("📋 Full results table"):
        st.dataframe(
            results[["headline", "model", "prediction", "confidence"]],
            use_container_width=True,
        )
        buf = io.BytesIO()
        results.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download predictions CSV",
            data=buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------


def page_results_dashboard() -> None:
    st.title("📊 Experiment Results Dashboard")

    if not os.path.exists(RESULTS_CSV):
        st.warning(
            f"No results file found at `{RESULTS_CSV}`.  "
            "Run the **▶️ Run Experiment** page first to generate data."
        )
        return

    df = pd.read_csv(RESULTS_CSV)
    st.success(f"Loaded **{len(df)}** result rows from `{RESULTS_CSV}`")

    # --- Summary metrics ---
    st.subheader("📈 Summary Statistics")
    pivot = (
        df.groupby(["model_name", "n_samples"])["accuracy"]
        .mean()
        .unstack("n_samples")
        .round(4)
    )
    pivot.columns = [f"N={c}" for c in pivot.columns]
    pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
    st.dataframe(pivot, use_container_width=True)

    # --- Learning curve ---
    st.subheader("📉 Learning Curves — Accuracy vs Training Size")
    model_styles = {
        "Logistic Regression":  dict(color="#2196F3", linestyle="--", marker="o"),
        "Classical SVM (RBF)":  dict(color="#4CAF50", linestyle="--", marker="s"),
        "Naive Bayes":          dict(color="#FF9800", linestyle="--", marker="^"),
        "VQC (ZZ + SEL)":       dict(color="#F44336", linestyle="-",  marker="*"),
        "Quantum Kernel SVM":   dict(color="#9C27B0", linestyle="-",  marker="D"),
    }
    grouped = (
        df.groupby(["n_samples", "model_name"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, style in model_styles.items():
        sub = grouped[grouped["model_name"] == model_name].sort_values("n_samples")
        if sub.empty:
            continue
        x = sub["n_samples"].values
        y = sub["mean"].values
        err = sub["std"].fillna(0).values
        ax.plot(x, y, label=model_name, linewidth=2, **style)
        ax.fill_between(x, y - err, y + err, alpha=0.1, color=style["color"])
    ax.axvline(200, color="grey", linestyle=":", linewidth=1.5, alpha=0.8, label="N=200 (advantage zone)")
    ax.set_xlabel("Training Samples", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Quantum vs Classical: Accuracy Learning Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Sarcasm accuracy bar chart ---
    st.subheader("📊 Overall vs Sarcasm Accuracy at N=200")
    n_focus = st.slider("Select N", min_value=int(df["n_samples"].min()),
                        max_value=int(df["n_samples"].max()), value=200, step=50)
    sub200 = df[df["n_samples"] == n_focus]
    if not sub200.empty:
        agg = (
            sub200.groupby("model_name")[["accuracy", "sarcasm_accuracy"]]
            .mean()
            .sort_values("accuracy", ascending=False)
            .reset_index()
        )
        x = np.arange(len(agg))
        width = 0.35
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        b1 = ax2.bar(x - width / 2, agg["accuracy"],        width, label="Overall Accuracy", color="#64B5F6", edgecolor="white")
        b2 = ax2.bar(x + width / 2, agg["sarcasm_accuracy"], width, label="Sarcasm-only",    color="#EF5350", edgecolor="white")
        for bar in [*b1, *b2]:
            h = bar.get_height()
            if not np.isnan(h):
                ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                         f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(agg["model_name"], rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.1)
        ax2.set_title(f"Overall vs Sarcasm Accuracy — N={n_focus}", fontweight="bold")
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- F1 score comparison ---
    st.subheader("🎯 F1 Score Comparison")
    f1_pivot = (
        df.groupby(["model_name", "n_samples"])["f1_score"]
        .mean()
        .unstack("n_samples")
        .round(4)
    )
    f1_pivot.columns = [f"N={c}" for c in f1_pivot.columns]
    f1_pivot = f1_pivot.sort_values(f1_pivot.columns[-1], ascending=False)
    st.dataframe(f1_pivot.style.background_gradient(cmap="RdYlGn", axis=None),
                 use_container_width=True)

    # --- Training time ---
    st.subheader("⏱ Training Time (seconds)")
    time_agg = (
        df.groupby("model_name")["train_time_sec"]
        .agg(["mean", "min", "max"])
        .round(2)
        .sort_values("mean", ascending=True)
        .reset_index()
    )
    st.dataframe(time_agg, use_container_width=True)

    # --- Raw data ---
    with st.expander("📋 Raw experiment data"):
        st.dataframe(df, use_container_width=True)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download full results CSV",
            data=buf.getvalue(),
            file_name="experiment_results.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------


def page_run_experiment() -> None:
    st.title("▶️ Run Experiment")
    st.markdown(
        "Configure and launch a **quick demo** or **full experiment** directly from here.  "
        "Results are saved to `results/experiment_results.csv` and shown below."
    )

    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Experiment mode", ["Quick demo", "Full experiment"])
        n_demo = st.slider("Sample size (demo mode)", 50, 500, 200, step=50,
                           disabled=(mode == "Full experiment"))
    with col2:
        n_trials = st.slider("Trials per configuration", 1, 5, 1,
                             disabled=(mode == "Quick demo"))
        vqc_iter = st.slider("VQC max iterations", 10, 100, 30)

    include_qksvm = st.checkbox(
        "Include Quantum Kernel SVM (very slow — O(N²) circuit calls)", value=False
    )

    data_path = st.text_input("Dataset path", value=DATA_PATH)

    run_btn = st.button("🚀 Run Experiment", type="primary")

    if run_btn:
        st.info("Experiment started. This may take several minutes …")
        log_area = st.empty()
        log_lines: List[str] = []

        def _log(msg: str) -> None:
            log_lines.append(msg)
            log_area.code("\n".join(log_lines[-40:]), language="text")

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import MaxAbsScaler
            from models.vqc_model import VQCClassifier

            if include_qksvm:
                from models.quantum_kernel_svm import QuantumKernelSVM

            _log("📦 Loading dataset …")
            df_data = load_and_clean_dataset(data_path)
            texts = df_data["clean_text"].tolist()
            labels = df_data["label"].values
            _log(f"   {len(df_data):,} samples loaded.")

            _log("\n🔧 Fitting classical TF-IDF …")
            clf_tfidf = TfidfVectorizer(
                max_features=2000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
            )
            X_clf_raw = clf_tfidf.fit_transform(texts)
            clf_scaler = MaxAbsScaler()
            X_clf_full = clf_scaler.fit_transform(X_clf_raw).toarray()
            _log(f"   Classical features: {X_clf_full.shape}")

            _log("\n🔧 Fitting quantum feature pipeline …")
            quantum_pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
            X_q_full = quantum_pipe.fit_transform(texts)
            _log(f"   Quantum features: {X_q_full.shape}")

            sample_sizes = (
                [n_demo] if mode == "Quick demo"
                else [50, 100, 150, 200, 300, 500]
            )
            repeats = 1 if mode == "Quick demo" else n_trials

            records = []
            total = len(sample_sizes) * repeats
            prog = st.progress(0)
            run_idx = 0

            for size in sample_sizes:
                for trial in range(repeats):
                    run_idx += 1
                    seed = trial * 13 + size
                    _log(f"\n[{run_idx}/{total}] N={size}, trial={trial + 1} (seed={seed})")

                    rng = np.random.RandomState(seed)
                    idx_0 = np.where(labels == 0)[0]
                    idx_1 = np.where(labels == 1)[0]
                    n0 = max(1, size // 2)
                    n1 = size - n0
                    c0 = rng.choice(idx_0, min(n0, len(idx_0)), replace=False)
                    c1 = rng.choice(idx_1, min(n1, len(idx_1)), replace=False)
                    sub_idx = np.concatenate([c0, c1])
                    rng.shuffle(sub_idx)

                    X_clf_sub = X_clf_full[sub_idx]
                    X_q_sub   = X_q_full[sub_idx]
                    y_sub     = labels[sub_idx]

                    try:
                        split = train_test_split(
                            X_clf_sub, X_q_sub, y_sub,
                            test_size=0.25, random_state=seed, stratify=y_sub,
                        )
                    except ValueError:
                        split = train_test_split(
                            X_clf_sub, X_q_sub, y_sub,
                            test_size=0.25, random_state=seed,
                        )
                    X_c_tr, X_c_te, X_q_tr, X_q_te, y_tr, y_te = split

                    classical_run = [
                        ("Logistic Regression", LogisticRegressionBaseline(), X_c_tr, X_c_te),
                        ("Classical SVM (RBF)", ClassicalSVMBaseline(),       X_c_tr, X_c_te),
                        ("Naive Bayes",         NaiveBayesBaseline(),          X_c_tr, X_c_te),
                        ("VQC (ZZ + SEL)",      VQCClassifier(n_layers=2, max_iter=vqc_iter, lr=0.1, batch_size=32), X_q_tr, X_q_te),
                    ]
                    if include_qksvm:
                        classical_run.append(
                            ("Quantum Kernel SVM", QuantumKernelSVM(C=1.0), X_q_tr, X_q_te)
                        )

                    for m_name, model, X_tr, X_te in classical_run:
                        _log(f"  → {m_name} …")
                        t0 = time.time()
                        try:
                            model.fit(X_tr, y_tr)
                            res = model.evaluate(X_te, y_te)
                            elapsed = time.time() - t0
                            mask = y_te == 1
                            sar_acc = float((model.predict(X_te[mask]) == y_te[mask]).mean()) if mask.sum() > 0 else float("nan")
                            _log(f"     acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  ({elapsed:.1f}s)")
                            records.append({
                                "n_samples": size, "trial": trial,
                                "model_name": m_name,
                                "accuracy": res["accuracy"], "f1_score": res["f1"],
                                "precision": res["precision"], "recall": res["recall"],
                                "sarcasm_accuracy": sar_acc,
                                "train_time_sec": round(elapsed, 3),
                            })
                        except Exception as exc:
                            elapsed = time.time() - t0
                            _log(f"     FAILED ({elapsed:.1f}s): {exc}")
                            records.append({
                                "n_samples": size, "trial": trial,
                                "model_name": m_name,
                                "accuracy": float("nan"), "f1_score": float("nan"),
                                "precision": float("nan"), "recall": float("nan"),
                                "sarcasm_accuracy": float("nan"),
                                "train_time_sec": round(elapsed, 3),
                            })

                    prog.progress(run_idx / total)

            results_df = pd.DataFrame(records)
            os.makedirs("results", exist_ok=True)
            results_df.to_csv(RESULTS_CSV, index=False)
            _log(f"\n✅ Results saved → {RESULTS_CSV}")

            st.success("✅ Experiment complete! Switch to **📊 Results Dashboard** to view charts.")
            st.dataframe(results_df, use_container_width=True)

        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------------


def page_circuit_explorer() -> None:
    st.title("⚛️ Circuit Explorer")
    st.markdown(
        "Visualise the **VQC quantum circuit** and understand each layer's role."
    )

    tab1, tab2, tab3 = st.tabs(["VQC Circuit Diagram", "Kernel Heatmap", "Feature Space"])

    with tab1:
        st.subheader("VQC: ZZ Feature Map + StronglyEntanglingLayers")
        st.markdown(
            """
            | Layer | Gate(s) | Role |
            |-------|---------|------|
            | ① | `AngleEmbedding` (Y) | Encode 8 features as Ry rotations |
            | ② | `CNOT·RZ·CNOT` | ZZ nearest-neighbour entanglement |
            | ③ | `AngleEmbedding` (Z) | Data re-uploading pass |
            | ④ | `StronglyEntanglingLayers` × 2 | 48 trainable parameters |
            | ⑤ | `mean⟨Z⟩` | Aggregate signal over all qubits |
            """
        )
        try:
            import pennylane as qml
            import pennylane.numpy as pnp
            from models.vqc_model import vqc_circuit, N_QUBITS, _WEIGHTS_SHAPE

            dummy_in = pnp.array(np.zeros(N_QUBITS), requires_grad=False)
            dummy_w  = pnp.array(np.zeros(_WEIGHTS_SHAPE), requires_grad=False)

            draw_mode = st.radio("Draw mode", ["ASCII (text)", "Matplotlib"], horizontal=True)

            if draw_mode == "ASCII (text)":
                diagram = qml.draw(vqc_circuit)(dummy_in, dummy_w)
                st.code(diagram, language="text")
            else:
                fig, ax = qml.draw_mpl(vqc_circuit)(dummy_in, dummy_w)
                ax.set_title("VQC: ZZ Feature Map + StronglyEntangling Ansatz",
                             fontsize=10, fontweight="bold")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
                plt.close(fig)
                st.image(buf.getvalue(), use_container_width=True)

        except Exception as exc:
            st.warning(f"Circuit diagram unavailable: {exc}")

        st.subheader("Circuit Specs")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Qubits", "8")
        col2.metric("Trainable params", "48")
        col3.metric("Hilbert space dim", "256 = 2⁸")
        col4.metric("Diff method", "adjoint / param-shift")

    with tab2:
        st.subheader("Quantum Kernel Heatmap")
        st.markdown(
            "Visualise the kernel matrix $K[i,j] = |\\langle\\phi(x_i)|\\phi(x_j)\\rangle|^2$ "
            "for a small random sample."
        )
        n_pts = st.slider("Number of sample points", 4, 16, 8)
        if st.button("🔄 Generate kernel matrix"):
            with st.spinner("Computing quantum kernel …"):
                try:
                    from models.quantum_kernel_svm import compute_kernel_matrix

                    rng = np.random.RandomState(42)
                    X_demo = rng.uniform(0, np.pi, (n_pts, 8))
                    K = compute_kernel_matrix(X_demo, X_demo, desc="Kernel heatmap")

                    import seaborn as sns
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        K, ax=ax, cmap="viridis", vmin=0, vmax=1,
                        xticklabels=[f"x{i}" for i in range(n_pts)],
                        yticklabels=[f"x{i}" for i in range(n_pts)],
                        annot=(n_pts <= 8),
                        fmt=".2f" if n_pts <= 8 else "",
                    )
                    ax.set_title(
                        f"Quantum Kernel Matrix K ({n_pts}×{n_pts})",
                        fontweight="bold"
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    diag_mean = float(np.diag(K).mean())
                    off_mean  = float((K.sum() - np.trace(K)) / (n_pts * (n_pts - 1)))
                    st.markdown(
                        f"**Diagonal mean** (self-overlap): `{diag_mean:.4f}` (expected ≈ 1.0)  \n"
                        f"**Off-diagonal mean**: `{off_mean:.4f}`"
                    )
                except Exception as exc:
                    st.error(f"Kernel computation failed: {exc}")

    with tab3:
        st.subheader("Quantum Feature Space")
        st.markdown(
            "Load the dataset and visualise the 8-dimensional quantum features "
            "projected to 2D via PCA."
        )
        if st.button("🔄 Generate feature projection"):
            with st.spinner("Loading & transforming …"):
                try:
                    from sklearn.decomposition import PCA

                    df_data = _load_dataset()
                    sample = df_data.sample(500, random_state=42)
                    texts_s = tuple(sample["clean_text"].tolist())

                    pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
                    X_q = pipe.fit_transform(list(texts_s))
                    y_q = sample["label"].values

                    pca2 = PCA(n_components=2, random_state=42)
                    X_2d = pca2.fit_transform(X_q)

                    fig, ax = plt.subplots(figsize=(7, 5))
                    for label, color, marker, name in [
                        (0, "#2196F3", "o", "Not Sarcastic"),
                        (1, "#F44336", "^", "Sarcastic"),
                    ]:
                        mask = y_q == label
                        ax.scatter(
                            X_2d[mask, 0], X_2d[mask, 1],
                            c=color, marker=marker, alpha=0.5, s=25, label=name
                        )
                    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
                    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
                    ax.set_title("Quantum Features (8-dim) → PCA 2D Projection", fontweight="bold")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Feature projection failed: {exc}")


# ---------------------------------------------------------------------------


def page_test_runner() -> None:
    st.title("🧪 Test Suite")
    st.markdown(
        "Run the **pytest** test suite from the browser and see results inline."
    )

    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    if not os.path.isdir(test_dir):
        st.error(f"Tests directory not found: `{test_dir}`")
        return

    test_files = sorted(f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py"))
    st.markdown(f"**Found {len(test_files)} test module(s):**")
    selected = st.multiselect(
        "Select test modules to run (empty = run all)",
        options=test_files,
        default=[],
    )
    verbose = st.checkbox("Verbose output (-v)", value=True)
    no_quantum = st.checkbox(
        "Skip slow quantum tests (-m 'not slow')", value=True
    )

    if st.button("🚀 Run Selected Tests", type="primary"):
        import subprocess

        cmd = [sys.executable, "-m", "pytest"]
        if selected:
            cmd += [os.path.join(test_dir, f) for f in selected]
        else:
            cmd.append(test_dir)
        if verbose:
            cmd.append("-v")
        if no_quantum:
            cmd += ["-m", "not slow"]
        cmd += ["--tb=short", "--no-header", "-q"]

        with st.spinner("Running tests …"):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )

        output = result.stdout + result.stderr
        lines = output.splitlines()

        passed = sum(1 for l in lines if " PASSED" in l)
        failed = sum(1 for l in lines if " FAILED" in l or " ERROR" in l)

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Passed", passed)
        col2.metric("❌ Failed", failed)
        col3.metric("Exit code", result.returncode)

        if result.returncode == 0:
            st.success("All selected tests passed!")
        else:
            st.error("Some tests failed. See output below.")

        st.code(output, language="text")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------


PAGES = {
    "🏠 Home":                  page_home,
    "🔍 Live Predictor":        page_live_predictor,
    "📊 Results Dashboard":     page_results_dashboard,
    "▶️ Run Experiment":         page_run_experiment,
    "⚛️ Circuit Explorer":      page_circuit_explorer,
    "🧪 Test Suite":             page_test_runner,
}


def main() -> None:
    with st.sidebar:
        st.image(
            "https://pennylane.ai/img/pennylane_horizontal_color.svg",
            width=180,
        )
        st.markdown("# Quantum Sarcasm")
        st.markdown("*Hybrid QML + NLP*")
        st.divider()
        selection = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("Python ≥ 3.12 | PennyLane ≥ 0.44")
        st.caption("March 2026")

    PAGES[selection]()


if __name__ == "__main__":
    main()
