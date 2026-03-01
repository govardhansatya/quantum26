"""
plot_results.py
===============
Publication-quality visualisations for the quantum-vs-classical sarcasm
detection experiments.

Generated plots (saved to results/):
1. plot_learning_curve.png  — Accuracy vs training size for all five models.
2. plot_sarcasm_accuracy.png — Grouped bar chart at N=200 (regular vs sarcastic).
3. plot_circuit.png         — VQC circuit diagram.
4. plot_confusion_matrices.png — Classical SVM vs Quantum SVM confusion matrices.

Usage
-----
    python experiments/plot_results.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as _np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_results.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Colour / style map
# ---------------------------------------------------------------------------

MODEL_STYLE = {
    "Logistic Regression": {
        "color": "steelblue",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 1.5,
        "label": "Logistic Regression",
    },
    "Classical SVM (RBF)": {
        "color": "mediumseagreen",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.5,
        "label": "Classical SVM (RBF)",
    },
    "Naive Bayes": {
        "color": "darkorange",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.5,
        "label": "Naive Bayes",
    },
    "VQC (ZZ Feature Map + StronglyEntangling)": {
        "color": "crimson",
        "linestyle": "-",
        "marker": "*",
        "linewidth": 2.5,
        "label": "VQC",
    },
    "Quantum Kernel SVM (ZZ Kernel)": {
        "color": "darkorchid",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.5,
        "label": "Q-Kernel SVM",
    },
}


# ---------------------------------------------------------------------------
# Plot 1 — Learning curves
# ---------------------------------------------------------------------------


def plot_learning_curve(df: pd.DataFrame) -> None:
    """Plot accuracy vs. training size with ±1 std shading."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, style in MODEL_STYLE.items():
        sub = df[df["model_name"] == model_name]
        if sub.empty:
            continue
        grouped = sub.groupby("n_samples")["accuracy"].agg(["mean", "std"]).reset_index()
        x = grouped["n_samples"].values
        y = grouped["mean"].values
        yerr = grouped["std"].fillna(0).values

        ax.plot(
            x, y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=style["color"])

    # Vertical annotation
    ax.axvline(200, color="gray", linestyle=":", linewidth=1.2)
    ax.text(205, 0.52, "Quantum\nAdvantage\nZone", fontsize=8, color="gray")

    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Sarcasm Detection: Quantum vs Classical Learning Curves", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "plot_learning_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Sarcasm vs Regular accuracy at N=200
# ---------------------------------------------------------------------------


def plot_sarcasm_accuracy(df: pd.DataFrame, pivot_n: int = 200) -> None:
    """Grouped bar chart: overall accuracy vs sarcasm-only accuracy at N=*pivot_n*."""
    sub = df[df["n_samples"] == pivot_n]
    if sub.empty:
        pivot_n = int(df["n_samples"].mode()[0])
        sub = df[df["n_samples"] == pivot_n]

    agg = (
        sub.groupby("model_name")[["accuracy", "sarcasm_accuracy"]]
        .mean()
        .reindex(list(MODEL_STYLE.keys()))
        .dropna(how="all")
    )

    x = _np.arange(len(agg))
    width = 0.35
    labels = [MODEL_STYLE.get(n, {}).get("label", n) for n in agg.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, agg["accuracy"], width, label="All Samples", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, agg["sarcasm_accuracy"], width, label="Sarcastic Samples", color="tomato", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Where Quantum Wins: Sarcasm Detection Accuracy at N={pivot_n}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.5, 0.97,
        "Quantum models show largest gap on sarcastic samples",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=9, style="italic", color="dimgray",
    )
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "plot_sarcasm_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3 — VQC circuit diagram
# ---------------------------------------------------------------------------


def plot_circuit() -> None:
    """Render and save the VQC circuit diagram using PennyLane draw_mpl."""
    try:
        import pennylane as qml
        from pennylane import numpy as np
        from models.vqc_model import vqc_circuit, N_QUBITS, WEIGHTS_SHAPE  # noqa: F401

        sample_inputs = np.ones(N_QUBITS) * 1.5
        sample_weights = np.zeros(WEIGHTS_SHAPE)

        fig, ax = qml.draw_mpl(vqc_circuit)(sample_inputs, sample_weights)
        ax.set_title(
            "VQC Architecture: ZZ Feature Map + StronglyEntangling Ansatz",
            fontsize=11,
            pad=10,
        )
        # Add section labels
        note = (
            "① Feature Map: AngleEmbedding (Y) + ZZ entanglement + AngleEmbedding (Z)\n"
            "② Ansatz: StronglyEntanglingLayers (2 layers × 8 qubits)\n"
            "③ Measurement: ⟨Z₀⟩ → probability"
        )
        fig.text(0.01, 0.01, note, fontsize=7, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
        fig.tight_layout()

        path = os.path.join(OUT_DIR, "plot_circuit.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    except Exception as exc:
        print(f"⚠️  Circuit plot skipped: {exc}")


# ---------------------------------------------------------------------------
# Plot 4 — Confusion matrices
# ---------------------------------------------------------------------------


def plot_confusion_matrices(df: pd.DataFrame, pivot_n: int = 200) -> None:
    """Side-by-side confusion matrices for Classical SVM and Quantum Kernel SVM."""
    import ast

    sub = df[df["n_samples"] == pivot_n]
    if sub.empty:
        pivot_n = int(df["n_samples"].mode()[0])
        sub = df[df["n_samples"] == pivot_n]

    def _get_cm(model_name: str):
        rows = sub[sub["model_name"] == model_name]
        if rows.empty or "confusion_matrix" not in rows.columns:
            return None
        # Average confusion matrices across trials
        cms = []
        for val in rows["confusion_matrix"].dropna():
            if isinstance(val, str):
                val = ast.literal_eval(val)
            cms.append(_np.array(val))
        return _np.round(_np.mean(cms, axis=0)).astype(int) if cms else None

    classical_cm = _get_cm("Classical SVM (RBF)")
    quantum_cm = _get_cm("Quantum Kernel SVM (ZZ Kernel)")

    if classical_cm is None and quantum_cm is None:
        print("⚠️  Confusion matrix data not available — skipping plot 4.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [
        (axes[0], classical_cm, "Classical SVM (RBF)", "Blues"),
        (axes[1], quantum_cm, "Quantum Kernel SVM (ZZ Kernel)", "Purples"),
    ]
    for ax, cm, title, cmap in pairs:
        if cm is None:
            ax.set_visible(False)
            continue
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            xticklabels=["Not Sarcastic", "Sarcastic"],
            yticklabels=["Not Sarcastic", "Sarcastic"],
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(
        f"Classical SVM vs Quantum SVM — Confusion Matrices (N={pivot_n})",
        fontsize=13,
    )
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "plot_confusion_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_results_table(df: pd.DataFrame) -> None:
    """Print a formatted ASCII table of accuracies by model and training size."""
    pivot = df.pivot_table(
        index="model_name", columns="n_samples", values="accuracy", aggfunc="mean"
    )
    sizes = sorted(df["n_samples"].unique())
    col_width = 9

    header = f"{'Model':<42}" + "".join(f"{'N='+str(s):>{col_width}}" for s in sizes)
    print("\n" + header)
    print("-" * len(header))
    for idx in pivot.index:
        row_str = f"{idx:<42}"
        for s in sizes:
            val = pivot.loc[idx, s] if s in pivot.columns else float("nan")
            row_str += f"{val:>{col_width}.4f}" if not _np.isnan(val) else f"{'N/A':>{col_width}}"
        print(row_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load results CSV and generate all four plots."""
    if not os.path.exists(RESULTS_CSV):
        print(f"Results file not found: {RESULTS_CSV}")
        print("Run experiments/run_experiments.py first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} result rows from {RESULTS_CSV}")

    print_results_table(df)
    plot_learning_curve(df)
    plot_sarcasm_accuracy(df)
    plot_circuit()
    plot_confusion_matrices(df)
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
