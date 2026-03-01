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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3-D projection)

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
    "Heuristic (Lexical)":  "#607D8B",
    "Logistic Regression":  "#2196F3",
    "Classical SVM (RBF)":  "#4CAF50",
    "Naive Bayes":          "#FF9800",
    "VQC (ZZ + SEL)":       "#F44336",
    "Quantum Kernel SVM":   "#9C27B0",
}

FEATURE_NAMES = [
    "PCA Component 1", "PCA Component 2", "PCA Component 3", "PCA Component 4",
    "Sarcasm Marker", "Exclamation Density", "Hyperbole Score", "CAPS Ratio",
]

_PALETTE = [
    "#F44336", "#E91E63", "#9C27B0", "#673AB7",
    "#3F51B5", "#2196F3", "#009688", "#4CAF50",
]

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


# ===========================================================================
# Rich visualisation helpers
# ===========================================================================


# ---------------------------------------------------------------------------
# 1.  Bloch Sphere
# ---------------------------------------------------------------------------

def _draw_bloch_sphere(
    ax: "Axes3D",
    theta: float,
    phi: float = 0.0,
    qubit_idx: int = 0,
    color: str = "#F44336",
) -> None:
    """Draw a single qubit Bloch sphere on a 3-D axes.

    After AngleEmbedding with Ry(theta) the state is:
        |psi> = cos(theta/2)|0> + sin(theta/2)|1>
    Bloch vector: (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    """
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 28)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="#CCCCCC", alpha=0.10, linewidth=0.4)

    # Equator
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(np.cos(th), np.sin(th), np.zeros_like(th),
            color="#AAAAAA", linewidth=0.7, alpha=0.5)
    # Prime meridian (XZ-plane)
    ph = np.linspace(0, np.pi, 60)
    ax.plot(np.sin(ph), np.zeros_like(ph), np.cos(ph),
            color="#AAAAAA", linewidth=0.7, alpha=0.4)

    # Axis stubs
    for vec, lbl in [([1.3, 0, 0], "x"), ([0, 1.3, 0], "y"), ([0, 0, 1.4], "|0>")]:
        ax.quiver(0, 0, 0, *vec, color="#888888", arrow_length_ratio=0.15,
                  linewidth=0.8, alpha=0.55)
        ax.text(*[v * 1.12 for v in vec], lbl, fontsize=6, color="#888888")
    ax.text(0, 0, -1.6, "|1>", fontsize=7, color="#555555", fontweight="bold",
            ha="center")

    # State vector
    bx = np.sin(theta) * np.cos(phi)
    by = np.sin(theta) * np.sin(phi)
    bz = np.cos(theta)
    ax.quiver(0, 0, 0, bx, by, bz, color=color,
              arrow_length_ratio=0.22, linewidth=2.5, zorder=10)
    ax.scatter([bx], [by], [bz], s=50, c=color, zorder=11)

    # Dashed projection onto XZ-plane
    ax.plot([0, bx], [0, 0], [0, bz],
            color=color, linestyle="--", linewidth=0.9, alpha=0.45)

    feat_lbl = FEATURE_NAMES[qubit_idx] if qubit_idx < len(FEATURE_NAMES) else f"Qubit {qubit_idx}"
    ax.set_title(
        f"Q{qubit_idx}  theta={theta:.2f}\n{feat_lbl[:18]}",
        fontsize=7, pad=2, color=color, fontweight="bold",
    )
    for lim_fn in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        lim_fn(-1.6, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def plot_bloch_spheres(angles: np.ndarray) -> plt.Figure:
    """Plot up to 8 Bloch spheres in a 2x4 grid for the given angle vector."""
    n = min(len(angles), 8)
    fig = plt.figure(figsize=(14, 6.8))
    fig.patch.set_facecolor("#0F0F1A")
    for i in range(n):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.set_facecolor("#0F0F1A")
        _draw_bloch_sphere(ax, theta=float(angles[i]), phi=0.0,
                           qubit_idx=i, color=_PALETTE[i])
    fig.suptitle(
        "Qubit States after AngleEmbedding  (Ry rotations, phi=0)",
        color="white", fontsize=12, fontweight="bold", y=1.00,
    )
    plt.tight_layout(pad=0.4)
    return fig


# ---------------------------------------------------------------------------
# 2.  Confidence Gauge
# ---------------------------------------------------------------------------

def plot_confidence_gauge(confidence: float, is_sarcastic: bool) -> plt.Figure:
    """Half-circle speedometer gauge for a single confidence value."""
    fig, ax = plt.subplots(figsize=(5, 3.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    bg_th = np.linspace(np.pi, 0, 300)
    ax.plot(bg_th, [0.9] * 300, color="#333344", linewidth=18,
            solid_capstyle="round")

    needle_theta = np.pi - confidence * np.pi
    fill_th = np.linspace(np.pi, needle_theta, 300)
    zone_color = "#F44336" if is_sarcastic else "#4CAF50"
    ax.plot(fill_th, [0.9] * len(fill_th), color=zone_color,
            linewidth=18, solid_capstyle="round")

    ax.annotate(
        "",
        xy=(needle_theta, 0.82),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="white", lw=2.5),
    )

    for val, lbl in [(0, "0.0"), (0.25, "0.25"), (0.5, "0.5"),
                     (0.75, "0.75"), (1.0, "1.0")]:
        ax.text(np.pi - val * np.pi, 1.2, lbl, ha="center", va="center",
                fontsize=8, color="#AAAAAA")

    ax.text(np.pi / 2, 0.30, f"{confidence:.3f}", ha="center", va="center",
            fontsize=22, color="white", fontweight="bold",
            transform=ax.transData)
    verdict = "SARCASTIC" if is_sarcastic else "NOT SARCASTIC"
    icon = "😏" if is_sarcastic else "📰"
    ax.text(np.pi / 2, 0.05, f"{icon}  {verdict}", ha="center", va="center",
            fontsize=9, color=zone_color, fontweight="bold",
            transform=ax.transData)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_title("Confidence Gauge", color="white", fontsize=11, pad=10)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3.  Radar / Spider chart
# ---------------------------------------------------------------------------

def plot_radar_comparison(results_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Spider / radar chart comparing model sarcasm confidence scores."""
    models = results_df["model"].tolist()
    confs  = results_df["confidence"].tolist()
    n = len(models)
    if n < 2:
        return None

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    c_closed = confs  + [confs[0]]
    a_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    for r in [0.25, 0.5, 0.75, 1.0]:
        ring = np.linspace(0, 2 * np.pi, 200)
        ax.plot(ring, [r] * 200, color="#333344", linewidth=0.7)

    ax.fill(a_closed, c_closed, color="#9C27B0", alpha=0.28)
    ax.plot(a_closed, c_closed, color="#CE93D8", linewidth=2.0)
    ax.plot(np.linspace(0, 2 * np.pi, 200), [0.5] * 200,
            color="#FF9800", linewidth=1.2, linestyle="--", alpha=0.7)

    for angle, conf, model in zip(angles, confs, models):
        col = MODEL_COLORS.get(model, "#607D8B")
        ax.scatter([angle], [conf], s=80, c=col, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.set_thetagrids(np.degrees(angles), labels=models,
                      fontsize=8, color="white")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                       fontsize=7, color="#AAAAAA")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#444455")
    ax.set_title("Model Confidence — Radar", color="white",
                 fontsize=11, pad=18)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4.  Quantum feature heatmap
# ---------------------------------------------------------------------------

def plot_feature_heatmap(angles_list: List[np.ndarray],
                         headlines: List[str]) -> plt.Figure:
    """Heatmap of 8 quantum angles across multiple headlines."""
    mat = np.array(angles_list)           # (n_heads, 8)
    n_heads = len(angles_list)

    fig, ax = plt.subplots(figsize=(max(9, n_heads * 1.5), 4.2))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    im = ax.imshow(mat.T, aspect="auto", cmap="plasma",
                   vmin=0, vmax=np.pi, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    for tick in cbar.ax.yaxis.get_ticklabels():
        tick.set_color("white")
    cbar.set_ticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    cbar.set_ticklabels(["0", "pi/4", "pi/2", "3pi/4", "pi"])
    cbar.set_label("Angle (radians)", color="white")

    short = [h[:28] + "..." if len(h) > 28 else h for h in headlines]
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8, color="white")
    ax.set_yticks(range(8))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8, color="white")
    ax.set_title("Quantum Feature Angles per Qubit  (theta in [0, pi])",
                 color="white", fontsize=11, pad=10)

    for row in range(8):
        for col in range(n_heads):
            val = mat[col, row]
            tc = "black" if val > np.pi * 0.55 else "white"
            sz = 7 if n_heads <= 6 else 5
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=sz, color=tc)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5.  Probability distribution bar chart
# ---------------------------------------------------------------------------

def plot_probability_bars(results_df: pd.DataFrame) -> plt.Figure:
    """Horizontal confidence bar chart."""
    fig, ax = plt.subplots(figsize=(9, max(3.2, len(results_df) * 0.55)))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#141428")

    colors = [MODEL_COLORS.get(m, "#607D8B") for m in results_df["model"]]
    bars = ax.barh(results_df["model"], results_df["confidence"],
                   color=colors, edgecolor="#1A1A2E", height=0.6)
    ax.axvline(0.5, color="#FF9800", linestyle="--", linewidth=1.5,
               alpha=0.8, label="Decision boundary (0.5)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(sarcastic)", color="white", fontsize=10)
    ax.set_title("Sarcasm Probability — All Models", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#444455")
    ax.legend(fontsize=8, facecolor="#1A1A2E", labelcolor="white")

    for bar, val in zip(bars, results_df["confidence"]):
        ax.text(min(val + 0.02, 0.93), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="white")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6.  Quantum state amplitude chart
# ---------------------------------------------------------------------------

def plot_amplitude_chart(angles: np.ndarray) -> plt.Figure:
    """Bar chart of |<0|psi>|^2 and |<1|psi>|^2 (|0> / |1> amplitudes) per qubit."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 3.8))
    fig.patch.set_facecolor("#0F0F1A")
    fig.suptitle("|0> and |1> Probability Amplitudes for Each Qubit",
                 color="white", fontsize=11, fontweight="bold")

    for idx, (ax, theta) in enumerate(zip(axes.flat, angles[:8])):
        ax.set_facecolor("#141428")
        p0 = np.cos(theta / 2) ** 2
        p1 = np.sin(theta / 2) ** 2
        col = _PALETTE[idx]
        bars = ax.bar(["  |0>", "  |1>"], [p0, p1],
                      color=[col + "99", col], edgecolor="white",
                      linewidth=0.8, width=0.55)
        ax.set_ylim(0, 1.18)
        ax.set_title(f"Q{idx}: theta={theta:.2f}", fontsize=8,
                     color=col, fontweight="bold")
        for bar, val in zip(bars, [p0, p1]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, color="white")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#444455")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# 7.  Prediction donut chart
# ---------------------------------------------------------------------------

def plot_prediction_donut(results_df: pd.DataFrame) -> plt.Figure:
    """Donut chart — consensus of models predicting sarcastic vs not."""
    n_s = int((results_df["label_int"] == 1).sum())
    n_n = int((results_df["label_int"] == 0).sum())
    total = n_s + n_n

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=12, color="white", transform=ax.transAxes)
        return fig

    wedges, _, autotexts = ax.pie(
        [n_s, n_n],
        colors=["#F44336", "#4CAF50"],
        explode=[0.05, 0.0],
        autopct=lambda pct: f"{pct:.0f}%\n({int(round(pct/100*total))})",
        startangle=90,
        pctdistance=0.72,
        wedgeprops=dict(width=0.48, edgecolor="#0F0F1A", linewidth=2),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(10)

    ax.text(0, 0, f"{total}\nmodels", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")

    patches = [
        mpatches.Patch(color="#F44336", label=f"Sarcastic ({n_s})"),
        mpatches.Patch(color="#4CAF50", label=f"Not Sarcastic ({n_n})"),
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=9, facecolor="#1A1A2E", labelcolor="white",
              framealpha=0.8)
    ax.set_title("Model Consensus", color="white", fontsize=11, pad=12)
    return fig


# ---------------------------------------------------------------------------
# 8.  Feature importance bar
# ---------------------------------------------------------------------------

def plot_feature_importance(angles: np.ndarray) -> plt.Figure:
    """Horizontal bar chart of normalised quantum feature angles."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#141428")

    norm = angles / np.pi
    bars = ax.barh(FEATURE_NAMES, norm, color=_PALETTE,
                   edgecolor="#0F0F1A", height=0.6)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#FF9800", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Mid (pi/2)")
    ax.set_xlabel("Normalised angle  (theta / pi)", color="white", fontsize=9)
    ax.set_title("Quantum Feature Activations per Qubit",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#444455")
    ax.legend(fontsize=8, facecolor="#1A1A2E", labelcolor="white")

    for bar, nv, raw in zip(bars, norm, angles):
        ax.text(min(nv + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                f"theta={raw:.3f}", va="center", fontsize=8, color="white")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


def page_live_predictor() -> None:
    st.title("🔍 Live Sarcasm Predictor")
    st.markdown(
        "Type one or more headlines.  All models classify them and the results are "
        "shown with **Bloch spheres, confidence gauges, radar charts, and more**."
    )

    # ── Load models ──────────────────────────────────────────────────────────
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

    if not headlines:
        st.info("Enter at least one headline above.")
        return

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("Running predictions …"):
        results = _predict_all(headlines, clf_bundle, quantum_pipeline, vqc_model)
        cleaned = [_preprocess(h) for h in headlines]
        angles_matrix = quantum_pipeline.transform(cleaned)   # (n_heads, 8)

    st.subheader(f"Results — {len(headlines)} headline(s)")

    # ── Per-headline model cards ───────────────────────────────────────────────
    for headline in headlines:
        sub = results[results["headline"] == headline]
        with st.expander(f"📰 {headline[:120]}", expanded=True):
            cols = st.columns(max(len(sub), 1))
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

    # ── Rich visualisation tabs ───────────────────────────────────────────────
    (tab_prob, tab_donut, tab_radar, tab_gauge,
     tab_bloch, tab_feat, tab_amp, tab_heat, tab_table) = st.tabs([
        "📊 Probabilities",
        "🍩 Consensus",
        "🕸 Radar",
        "⏱ Gauge",
        "🌐 Bloch Spheres",
        "🔬 Features",
        "🔢 Amplitudes",
        "🌡 Heatmap",
        "📋 Table",
    ])

    # Reference index for single-headline tabs (default = first headline)
    ref_hl = headlines[0]
    ref_sub = results[results["headline"] == ref_hl]
    ref_angles = angles_matrix[0]

    # --- Probabilities ---
    with tab_prob:
        st.subheader("P(sarcastic) — All Models")
        chosen = ref_hl
        if len(headlines) > 1:
            chosen = st.selectbox("Headline", headlines, key="prob_sel")
        sub_p = results[results["headline"] == chosen]
        fig = plot_probability_bars(sub_p)
        st.pyplot(fig)
        plt.close(fig)

    # --- Donut ---
    with tab_donut:
        st.subheader("Model Consensus — Donut Chart")
        chosen_d = ref_hl
        if len(headlines) > 1:
            chosen_d = st.selectbox("Headline", headlines, key="donut_sel")
        sub_d = results[results["headline"] == chosen_d]
        fig = plot_prediction_donut(sub_d)
        st.pyplot(fig)
        plt.close(fig)

    # --- Radar ---
    with tab_radar:
        st.subheader("Spider / Radar — Model Confidences")
        chosen_r = ref_hl
        if len(headlines) > 1:
            chosen_r = st.selectbox("Headline", headlines, key="radar_sel")
        sub_r = results[results["headline"] == chosen_r]
        fig = plot_radar_comparison(sub_r)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Need ≥ 2 models to draw a radar chart.")

    # --- Gauge ---
    with tab_gauge:
        st.subheader("Confidence Gauge")
        chosen_g = ref_hl
        if len(headlines) > 1:
            chosen_g = st.selectbox("Headline", headlines, key="gauge_sel")
        sub_g = results[results["headline"] == chosen_g]
        # Prefer VQC → LR → first row
        row_g = sub_g[sub_g["model"].str.startswith("VQC")]
        if row_g.empty:
            row_g = sub_g[sub_g["model"] == "Logistic Regression"]
        if row_g.empty:
            row_g = sub_g.head(1)
        row_g = row_g.iloc[0]
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            fig = plot_confidence_gauge(float(row_g["confidence"]),
                                        bool(row_g["label_int"]))
            st.pyplot(fig)
            plt.close(fig)
        with gc2:
            st.markdown(f"**Headline:** {row_g['headline']}")
            st.markdown(f"**Model:** `{row_g['model']}`")
            st.markdown(f"**Verdict:** `{row_g['prediction']}`")
            st.metric("Sarcasm probability", f"{row_g['confidence']:.3f}")

    # --- Bloch Spheres ---
    with tab_bloch:
        st.subheader("🌐 Qubit States after AngleEmbedding (Ry rotations)")
        st.markdown(
            "After **Ry(θᵢ)** the qubit state is  "
            "cos(θ/2)|0⟩ + sin(θ/2)|1⟩.  "
            "The coloured arrow is its Bloch vector."
        )
        chosen_b = ref_hl
        if len(headlines) > 1:
            chosen_b = st.selectbox("Headline", headlines, key="bloch_sel")
        bi = headlines.index(chosen_b)
        bloch_angles = angles_matrix[bi]
        with st.spinner("Rendering Bloch spheres …"):
            try:
                fig = plot_bloch_spheres(bloch_angles)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Bloch sphere render failed: {exc}")
        # Angle table
        angle_df = pd.DataFrame({
            "Qubit":     [f"Q{i}" for i in range(8)],
            "Feature":   FEATURE_NAMES,
            "θ (rad)":   [f"{a:.4f}" for a in bloch_angles],
            "θ / π":     [f"{a/np.pi:.3f}" for a in bloch_angles],
            "Bloch x":   [f"{np.sin(a):.3f}" for a in bloch_angles],
            "Bloch z":   [f"{np.cos(a):.3f}" for a in bloch_angles],
        })
        st.dataframe(angle_df, width='stretch', hide_index=True)

    # --- Feature importance ---
    with tab_feat:
        st.subheader("🔬 Quantum Feature Activations")
        chosen_f = ref_hl
        if len(headlines) > 1:
            chosen_f = st.selectbox("Headline", headlines, key="feat_sel")
        fi = headlines.index(chosen_f)
        fig = plot_feature_importance(angles_matrix[fi])
        st.pyplot(fig)
        plt.close(fig)

    # --- Amplitude chart ---
    with tab_amp:
        st.subheader("🔢 Qubit Probability Amplitudes")
        st.markdown(
            "P(|0⟩) = cos²(θ/2) and P(|1⟩) = sin²(θ/2) after Ry(θ)."
        )
        chosen_a = ref_hl
        if len(headlines) > 1:
            chosen_a = st.selectbox("Headline", headlines, key="amp_sel")
        ai = headlines.index(chosen_a)
        fig = plot_amplitude_chart(angles_matrix[ai])
        st.pyplot(fig)
        plt.close(fig)

    # --- Feature heatmap (all headlines) ---
    with tab_heat:
        st.subheader("🌡 Feature Angle Heatmap — All Headlines")
        fig = plot_feature_heatmap(
            [angles_matrix[i] for i in range(len(headlines))],
            headlines,
        )
        st.pyplot(fig)
        plt.close(fig)

    # --- Full table ---
    with tab_table:
        st.subheader("📋 Full Predictions Table")
        st.dataframe(
            results[["headline", "model", "prediction", "confidence"]],
            width='stretch',
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
    st.dataframe(pivot, width='stretch')

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
    _n_vals = sorted(df["n_samples"].unique().tolist())
    _n_min, _n_max = int(_n_vals[0]), int(_n_vals[-1])
    if _n_min == _n_max:
        n_focus = _n_min
        st.write(f"Single training size in results: **N={n_focus}**")
    else:
        _default = int(min(200, _n_max))
        n_focus = st.slider("Select N", min_value=_n_min,
                            max_value=_n_max, value=_default, step=max(1, (_n_max - _n_min) // 20))
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
                 width='stretch')

    # --- Training time ---
    st.subheader("⏱ Training Time (seconds)")
    time_agg = (
        df.groupby("model_name")["train_time_sec"]
        .agg(["mean", "min", "max"])
        .round(2)
        .sort_values("mean", ascending=True)
        .reset_index()
    )
    st.dataframe(time_agg, width='stretch')

    # --- Raw data ---
    with st.expander("📋 Raw experiment data"):
        st.dataframe(df, width='stretch')
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
            st.dataframe(results_df, width='stretch')

        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
            st.code(traceback.format_exc())


# ---------------------------------------------------------------------------


def page_circuit_explorer() -> None:
    st.title("⚛️ Circuit Explorer")
    st.markdown(
        "Visualise the **VQC quantum circuit**, explore **Bloch spheres**, "
        "inspect the **quantum kernel**, and project the **feature space**."
    )

    tab_circ, tab_bloch, tab_kernel, tab_feat = st.tabs([
        "📐 Circuit Diagram",
        "🌐 Bloch Sphere Demo",
        "🌡 Kernel Heatmap",
        "🔭 Feature Space",
    ])

    # ── Circuit Diagram ───────────────────────────────────────────────────────
    with tab_circ:
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
                st.image(buf.getvalue(), width="stretch")

        except Exception as exc:
            st.warning(f"Circuit diagram unavailable: {exc}")

        st.subheader("Circuit Specs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Qubits", "8")
        c2.metric("Trainable params", "48")
        c3.metric("Hilbert space dim", "256 = 2⁸")
        c4.metric("Diff method", "adjoint / param-shift")

    # ── Interactive Bloch Sphere Demo ─────────────────────────────────────────
    with tab_bloch:
        st.subheader("🌐 Interactive Bloch Sphere Explorer")
        st.markdown(
            "Adjust the **angle sliders** (one per qubit / feature) to see how the "
            "qubit states move on the Bloch sphere.  These values are exactly what "
            "is passed to `qml.AngleEmbedding` when classifying a headline."
        )

        preset = st.selectbox(
            "Quick preset",
            ["Custom (use sliders)", "All |0> (theta=0)", "Equator (theta=pi/2)",
             "All |1> (theta=pi)", "Gradient 0→pi"],
        )
        _preset_vals: dict = {
            "All |0> (theta=0)":    [0.0] * 8,
            "Equator (theta=pi/2)": [np.pi / 2] * 8,
            "All |1> (theta=pi)":   [np.pi] * 8,
            "Gradient 0→pi":        np.linspace(0, np.pi, 8).tolist(),
        }

        slider_angles: List[float] = []
        scols = st.columns(4)
        for i in range(8):
            default = float(_preset_vals.get(preset, [np.pi / 4] * 8)[i])
            with scols[i % 4]:
                val = st.slider(
                    f"Q{i}: {FEATURE_NAMES[i][:14]}",
                    min_value=0.0, max_value=float(np.pi),
                    value=default, step=0.05,
                    key=f"ce_bloch_q{i}",
                )
            slider_angles.append(val)

        angles_arr = np.array(slider_angles)

        with st.spinner("Rendering Bloch spheres …"):
            try:
                fig = plot_bloch_spheres(angles_arr)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Render error: {exc}")

        st.subheader("Probability Amplitudes for Current Angles")
        fig_amp = plot_amplitude_chart(angles_arr)
        st.pyplot(fig_amp)
        plt.close(fig_amp)

        st.subheader("ZZ Entanglement Phase  φᵢⱼ = 2(π − θᵢ)(π − θⱼ)")
        with st.spinner("Building ZZ phase matrix …"):
            try:
                import seaborn as sns
                zz = np.array([
                    [2 * (np.pi - slider_angles[i]) * (np.pi - slider_angles[j])
                     for j in range(8)]
                    for i in range(8)
                ])
                fig_zz, ax_zz = plt.subplots(figsize=(7, 5.5))
                sns.heatmap(
                    zz, ax=ax_zz, cmap="coolwarm", annot=True, fmt=".2f",
                    annot_kws={"size": 7},
                    xticklabels=[f"Q{i}" for i in range(8)],
                    yticklabels=[f"Q{i}" for i in range(8)],
                )
                ax_zz.set_title("ZZ Feature-Map Entanglement Phases (radians)",
                                fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_zz)
                plt.close(fig_zz)
            except Exception as exc:
                st.error(f"ZZ phase matrix failed: {exc}")

    # ── Kernel Heatmap ────────────────────────────────────────────────────────
    with tab_kernel:
        st.subheader("Quantum Kernel Matrix  K[i,j] = |⟨φ(xᵢ)|φ(xⱼ)⟩|²")
        st.markdown(
            "Diagonal ≈ 1.0 (self-overlap). "
            "Off-diagonal entries measure feature-space similarity."
        )
        n_pts = st.slider("Number of sample points", 4, 16, 8, key="ke_pts")
        if st.button("🔄 Generate kernel matrix"):
            with st.spinner("Computing quantum kernel …"):
                try:
                    from models.quantum_kernel_svm import compute_kernel_matrix
                    import seaborn as sns

                    rng = np.random.RandomState(42)
                    X_demo = rng.uniform(0, np.pi, (n_pts, 8))
                    K = compute_kernel_matrix(X_demo, X_demo, desc="Kernel heatmap")

                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(
                        K, ax=ax, cmap="viridis", vmin=0, vmax=1,
                        xticklabels=[f"x{i}" for i in range(n_pts)],
                        yticklabels=[f"x{i}" for i in range(n_pts)],
                        annot=(n_pts <= 8),
                        fmt=".2f" if n_pts <= 8 else "",
                    )
                    ax.set_title(f"Quantum Kernel Matrix K ({n_pts}×{n_pts})",
                                 fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    diag_mean = float(np.diag(K).mean())
                    off_mean  = float(
                        (K.sum() - np.trace(K)) / max(n_pts * (n_pts - 1), 1)
                    )
                    st.markdown(
                        f"**Diagonal mean**: `{diag_mean:.4f}` (expect ≈ 1.0)  \n"
                        f"**Off-diagonal mean**: `{off_mean:.4f}`"
                    )

                    # Eigenvalue spectrum
                    st.subheader("Kernel Eigenvalue Spectrum")
                    eigvals = np.linalg.eigvalsh(K)[::-1]
                    fig_eig, ax_eig = plt.subplots(figsize=(7, 3))
                    ax_eig.bar(range(len(eigvals)), eigvals,
                               color="#9C27B0", edgecolor="white")
                    ax_eig.set_xlabel("Index")
                    ax_eig.set_ylabel("Eigenvalue")
                    ax_eig.set_title("Eigenvalues of K (sorted descending)")
                    ax_eig.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_eig)
                    plt.close(fig_eig)

                except Exception as exc:
                    st.error(f"Kernel computation failed: {exc}")

    # ── Feature Space Projection ──────────────────────────────────────────────
    with tab_feat:
        st.subheader("Quantum Feature Space Projection")
        proj_type = st.radio("Projection", ["PCA 2D", "PCA 3D"], horizontal=True)
        n_sample = st.slider("Sample size", 100, 1000, 500, step=100)
        if st.button("🔄 Generate projection"):
            with st.spinner("Loading & transforming …"):
                try:
                    from sklearn.decomposition import PCA

                    df_data = _load_dataset()
                    sample = df_data.sample(n_sample, random_state=42)
                    pipe = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
                    X_q = pipe.fit_transform(sample["clean_text"].tolist())
                    y_q = sample["label"].values

                    if proj_type == "PCA 2D":
                        pca = PCA(n_components=2, random_state=42)
                        X_p = pca.fit_transform(X_q)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        for lbl, col, mk, nm in [
                            (0, "#2196F3", "o", "Not Sarcastic"),
                            (1, "#F44336", "^", "Sarcastic"),
                        ]:
                            m = y_q == lbl
                            ax.scatter(X_p[m, 0], X_p[m, 1], c=col, marker=mk,
                                       alpha=0.55, s=28, label=nm, edgecolors="none")
                        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                        ax.set_title("Quantum Features (8-dim) → PCA 2D",
                                     fontweight="bold")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        pca = PCA(n_components=3, random_state=42)
                        X_p = pca.fit_transform(X_q)
                        fig = plt.figure(figsize=(8, 6))
                        ax3 = fig.add_subplot(111, projection="3d")
                        for lbl, col, mk, nm in [
                            (0, "#2196F3", "o", "Not Sarcastic"),
                            (1, "#F44336", "^", "Sarcastic"),
                        ]:
                            m = y_q == lbl
                            ax3.scatter(X_p[m, 0], X_p[m, 1], X_p[m, 2],
                                        c=col, marker=mk, alpha=0.55, s=28,
                                        label=nm, edgecolors="none")
                        ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                        ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                        ax3.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
                        ax3.set_title("Quantum Features (8-dim) → PCA 3D",
                                      fontweight="bold")
                        ax3.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                except Exception as exc:
                    st.error(f"Projection failed: {exc}")

        st.subheader("Pairwise Scatter — Hand-crafted Sarcasm Features")
        if st.button("🔄 Pairwise scatter", key="pair_btn"):
            with st.spinner("Computing …"):
                try:
                    df_data2 = _load_dataset()
                    samp2 = df_data2.sample(400, random_state=7)
                    pipe2 = QuantumFeaturePipeline(n_qubits=8, n_tfidf=200)
                    X_p2 = pipe2.fit_transform(samp2["clean_text"].tolist())
                    y_p2 = samp2["label"].values

                    hc = X_p2[:, 4:]
                    hn = FEATURE_NAMES[4:]
                    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
                    for r in range(4):
                        for c in range(4):
                            ax = axes[r][c]
                            if r == c:
                                ax.hist(hc[y_p2 == 0, r], bins=18,
                                        color="#2196F3", alpha=0.55, density=True)
                                ax.hist(hc[y_p2 == 1, r], bins=18,
                                        color="#F44336", alpha=0.55, density=True)
                                ax.set_title(hn[r], fontsize=7)
                            else:
                                for lbl2, col2 in [(0, "#2196F3"), (1, "#F44336")]:
                                    m2 = y_p2 == lbl2
                                    ax.scatter(hc[m2, c], hc[m2, r], s=5,
                                               c=col2, alpha=0.4, edgecolors="none")
                            if r == 3:
                                ax.set_xlabel(hn[c], fontsize=6)
                            if c == 0:
                                ax.set_ylabel(hn[r], fontsize=6)
                            ax.tick_params(labelsize=5)
                    fig.suptitle("Pairwise Feature Scatter — Hand-crafted Sarcasm Signals",
                                 fontsize=11, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Pairwise scatter failed: {exc}")


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
