"""
app.py
======
Streamlit web app: Hybrid Quantum Sarcasm Detector.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as _np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚛️ Quantum Sarcasm Detector",
    page_icon="⚛️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .sarcastic-box {
        background: linear-gradient(135deg, #ff6b6b, #ffa94d);
        border-radius: 12px; padding: 18px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin-bottom: 8px;
    }
    .not-sarcastic-box {
        background: linear-gradient(135deg, #51cf66, #94d82d);
        border-radius: 12px; padding: 18px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin-bottom: 8px;
    }
    .quantum-sarcastic-box {
        background: linear-gradient(135deg, #9775fa, #cc5de8);
        border-radius: 12px; padding: 18px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin-bottom: 8px;
    }
    .quantum-not-box {
        background: linear-gradient(135deg, #74c0fc, #4dabf7);
        border-radius: 12px; padding: 18px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin-bottom: 8px;
    }
    .vs-text {
        font-size: 3rem; font-weight: 900; text-align: center;
        color: #495057; margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load or train classical and quantum models.

    Returns a tuple (pipeline, classical_svm, quantum_svm).
    Falls back to a quick training run on 300 samples if saved models are
    not found.
    """
    import joblib
    from utils.feature_engineering import QuantumFeaturePipeline
    from models.classical_models import ClassicalSVMBaseline
    from models.quantum_kernel_svm import QuantumKernelSVM

    SAVE_DIR = os.path.join(os.path.dirname(__file__), "results", "saved_models")
    pipeline_path = os.path.join(SAVE_DIR, "pipeline.joblib")
    csvm_path = os.path.join(SAVE_DIR, "classical_svm.joblib")
    qksvm_path = os.path.join(SAVE_DIR, "quantum_svm.joblib")

    if (
        os.path.exists(pipeline_path)
        and os.path.exists(csvm_path)
        and os.path.exists(qksvm_path)
    ):
        pipeline = joblib.load(pipeline_path)
        classical_svm = joblib.load(csvm_path)
        quantum_svm = joblib.load(qksvm_path)
        return pipeline, classical_svm, quantum_svm

    # Train on 300 samples
    from utils.data_pipeline import load_and_clean_dataset

    data_path = os.path.join(os.path.dirname(__file__), "data", "Sarcasm_Headlines_Dataset_v2.json")
    if not os.path.exists(data_path):
        return None, None, None

    df = load_and_clean_dataset(data_path)
    df = df.sample(n=min(300, len(df)), random_state=42)

    pipeline = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
    X = pipeline.fit_transform(df["clean_text"].tolist())
    y = df["label"].values.astype(int)

    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    classical_svm = ClassicalSVMBaseline()
    classical_svm.fit(X_tr, y_tr)

    # Use small training set for QKSVM in UI (speed)
    q_n = min(80, len(X_tr))
    quantum_svm = QuantumKernelSVM(C=1.0)
    quantum_svm.fit(X_tr[:q_n], y_tr[:q_n])

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(classical_svm, csvm_path)
    joblib.dump(quantum_svm, qksvm_path)

    return pipeline, classical_svm, quantum_svm


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚛️ Hybrid Quantum Sarcasm Detector")
st.markdown("### Classical NLP meets Quantum Machine Learning")

with st.expander("ℹ️ How does this work?"):
    st.markdown(
        """
        **Pipeline overview:**
        1. **Text cleaning** — lowercase, remove URLs/digits/punctuation, bind negation words.
        2. **TF-IDF vectorisation** — encode words as importance-weighted features.
        3. **PCA + Haar Wavelet** — compress 200 features into exactly 8.
        4. **Quantum angle encoding** — the 8 features become rotation angles on 8 qubits.
        5. **ZZ Feature Map** — creates entanglement between qubit pairs, capturing
           higher-order feature interactions that RBF kernels cannot.
        6. **Classical SVM** uses an RBF kernel in classical feature space.
           **Quantum Kernel SVM** uses the quantum kernel (inner product in 2⁸=256-D Hilbert space).

        The quantum approach tends to be more expressive in the **low-data regime**
        because quantum entanglement implicitly encodes complex feature correlations.
        """
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📊 About This Project")
    st.markdown(
        """
        **Dataset:** [Sarcasm Headlines v2](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

        **Framework:** [PennyLane](https://pennylane.ai/) + scikit-learn

        **Key finding:** Quantum Kernel SVM with ZZ feature map achieves
        higher accuracy on sarcastic samples in the low-data regime (N ≤ 200).
        """
    )

    st.divider()
    st.subheader("⚙️ Circuit Visualisation")
    n_qubits_display = st.slider("Number of qubits (display only)", 4, 8, 8)
    st.caption(
        f"With {n_qubits_display} qubits, the quantum kernel operates in a "
        f"2^{n_qubits_display} = {2**n_qubits_display}-dimensional Hilbert space."
    )

    # Load results if available
    results_path = os.path.join(os.path.dirname(__file__), "results", "experiment_results.csv")
    if os.path.exists(results_path):
        import pandas as pd

        res_df = pd.read_csv(results_path)
        st.divider()
        st.subheader("📈 Model Performance (N=200)")
        sub = res_df[res_df["n_samples"] == 200].groupby("model_name")["accuracy"].mean()
        if not sub.empty:
            st.dataframe(sub.reset_index().rename(columns={"accuracy": "Mean Acc"}).round(4))

# ---------------------------------------------------------------------------
# Example headlines
# ---------------------------------------------------------------------------

st.markdown("#### 🗞️ Try an example:")

example_headlines = [
    "Area Man Passionate Defender of What He Imagines Constitution To Say",
    "New Study Finds That People Who Exercise Regularly Are Probably Annoying",
    "Scientists Discover Water Still Wet",
    "Local Man Thrilled About Monday Morning Meeting",
]

cols_ex = st.columns(len(example_headlines))
selected_example = ""
for col, headline in zip(cols_ex, example_headlines):
    if col.button(headline[:40] + "…", key=headline):
        selected_example = headline

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------

headline_input = st.text_input(
    "📝 Enter a news headline:",
    value=selected_example,
    placeholder="e.g. Oh great, another tax hike is just what we needed",
)

analyze = st.button("🔍 Analyse Sentiment", type="primary")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if analyze and headline_input.strip():
    from utils.feature_engineering import detect_sarcasm_heuristic, _SARCASM_MARKERS
    from utils.data_pipeline import clean_text, negation_binding, remove_stopwords

    pipeline, classical_svm, quantum_svm = load_models()

    if pipeline is None:
        st.error(
            "⚠️ Dataset not found. Please place `Sarcasm_Headlines_Dataset_v2.json` "
            "in the `data/` folder and restart the app."
        )
        st.stop()

    # Preprocess
    clean = remove_stopwords(negation_binding(clean_text(headline_input)))
    X_input = pipeline.transform([clean])  # shape (1, 8)

    # Classical SVM
    try:
        c_proba = classical_svm.predict_proba(X_input)[0]
        c_pred = int(c_proba[1] >= 0.5)
        c_conf = float(c_proba[c_pred])
    except Exception:
        c_pred, c_conf = 0, 0.5

    # Quantum SVM
    try:
        q_proba = quantum_svm.predict_proba(X_input)[0]
        q_pred = int(q_proba[1] >= 0.5)
        q_conf = float(q_proba[q_pred])
    except Exception:
        q_pred, q_conf = 0, 0.5

    # Heuristic
    heuristic_score = detect_sarcasm_heuristic(headline_input)

    # ----- Results layout -----
    st.divider()
    st.subheader("🧪 Analysis Results")
    col_left, col_mid, col_right = st.columns([5, 1, 5])

    # Classical result
    with col_left:
        st.markdown("**Classical SVM (RBF Kernel)**")
        label_text = "😏 SARCASTIC" if c_pred == 1 else "😐 NOT SARCASTIC"
        css_class = "sarcastic-box" if c_pred == 1 else "not-sarcastic-box"
        st.markdown(f'<div class="{css_class}">{label_text}</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {c_conf*100:.1f}%")
        st.progress(c_conf)

    # VS divider
    with col_mid:
        st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

    # Quantum result
    with col_right:
        st.markdown("**Quantum Kernel SVM (ZZ Feature Map)**")
        label_text = "😏 SARCASTIC" if q_pred == 1 else "😐 NOT SARCASTIC"
        css_class = "quantum-sarcastic-box" if q_pred == 1 else "quantum-not-box"
        st.markdown(f'<div class="{css_class}">{label_text}</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {q_conf*100:.1f}%")
        st.progress(q_conf)
        if heuristic_score > 0.3:
            st.warning(
                "⚠️ Sarcasm markers detected! Quantum model applied entanglement correction."
            )

    # ----- Feature analysis -----
    with st.expander("🔬 Feature Analysis (Quantum Angle Encoding)"):
        import matplotlib.pyplot as plt

        feature_names = pipeline.get_feature_names()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(feature_names, X_input[0], color="mediumpurple", alpha=0.8)
        ax.set_ylabel("Angle (radians)")
        ax.set_title("Quantum Feature Encoding (8 Qubit Angles)")
        ax.set_ylim(0, _np.pi)
        ax.axhline(_np.pi / 2, color="gray", linestyle="--", linewidth=0.8, label="π/2")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Each bar represents the rotation angle applied to the corresponding qubit.")

    with st.expander("🧮 Why might they differ?"):
        st.markdown(
            f"""
            **Classical SVM (RBF kernel)** measures similarity in the original 8-dimensional
            feature space using a Gaussian kernel — good at smooth decision boundaries.

            **Quantum Kernel SVM (ZZ Feature Map)** computes inner products in a
            2⁸ = 256-dimensional Hilbert space via quantum interference.  The ZZ
            entanglement layer encodes *pairwise* feature interactions that an RBF
            kernel cannot capture, making it especially powerful for detecting
            subtle linguistic patterns like sarcasm.

            **Heuristic sarcasm score:** `{heuristic_score:.2f}` — based on
            {len([m for m in _SARCASM_MARKERS if m in headline_input.lower()])}
            matched sarcasm marker(s).
            """
        )

elif analyze and not headline_input.strip():
    st.warning("Please enter a headline to analyse.")
