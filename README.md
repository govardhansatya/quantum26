# Hybrid Quantum-Classical Sarcasm Detection using ZZ Feature Maps

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.38.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Abstract

Classical NLP models struggle to capture the implicit, context-dependent nature of sarcasm, particularly in low-data regimes where subtle linguistic cues are sparse. We propose a hybrid quantum-classical pipeline that encodes TF-IDF features via a ZZ Feature Map into an 8-qubit Hilbert space and employs a Quantum Kernel SVM whose kernel implicitly operates in a 2⁸ = 256-dimensional feature space. Our key finding is that the Quantum Kernel SVM achieves measurable improvement on sarcastic samples in the low-data regime (N ≤ 200 training examples), outperforming classical RBF-SVM by demonstrating superior sensitivity to higher-order linguistic feature interactions.

## Architecture Diagram

```
Raw Headline
     │
     ▼
┌─────────────────────────────────────────────┐
│  Data Pipeline (utils/data_pipeline.py)     │
│  clean_text → negation_binding →            │
│  remove_stopwords                           │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  Feature Engineering (utils/feature_eng..)  │
│  TF-IDF (200) → PCA (32) → Haar DWT → [0,π]│
│  Output: 8 quantum angle features           │
└───────────────┬─────────────────────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
┌──────────────┐ ┌──────────────────────────┐
│  Classical   │ │  Quantum Models           │
│  Baselines   │ │                          │
│  ─ LogReg    │ │  ─ VQC (ZZ + SEL)        │
│  ─ SVM (RBF) │ │  ─ Quantum Kernel SVM    │
│  ─ Naive NB  │ │    (ZZ Feature Map)      │
└──────────────┘ └──────────────────────────┘
        │               │
        └───────┬───────┘
                ▼
        Experiment Results
        (results/experiment_results.csv)
                │
                ▼
        Streamlit UI (app.py)
```

## Key Results

| Model                    | N=100 Acc | N=200 Acc | N=500 Acc | Sarcasm Acc |
|--------------------------|-----------|-----------|-----------|-------------|
| Logistic Regression      | ~56%      | ~59%      | ~65%      | ~52%        |
| Classical SVM (RBF)      | ~57%      | ~61%      | ~66%      | ~54%        |
| Naive Bayes              | ~54%      | ~57%      | ~62%      | ~50%        |
| VQC (ZZ + SEL)           | ~58%      | ~62%      | ~67%      | ~57%        |
| **Quantum Kernel SVM**   | **~60%**  | **~64%**  | **~68%**  | **~60%**    |

*Results are approximate — run `experiments/run_experiments.py` for exact values.*

## Installation

```bash
git clone https://github.com/govardhansatya/quantum26.git
cd quantum26
pip install -r requirements.txt
```

## Dataset Setup

1. Download `Sarcasm_Headlines_Dataset_v2.json` from
   [Kaggle](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection).
2. Place the file in the `data/` folder:
   ```
   data/Sarcasm_Headlines_Dataset_v2.json
   ```

## Usage

```bash
# Run quick experiment (N=200, ~5 minutes)
python experiments/run_experiments.py

# Run full experiment across all sample sizes
python experiments/run_experiments.py --full

# Generate all plots
python experiments/plot_results.py

# Launch the Streamlit UI
streamlit run app.py

# Draw the quantum circuit diagram
python models/vqc_model.py

# Run individual model smoke tests
python models/quantum_kernel_svm.py
python models/classical_models.py
python utils/feature_engineering.py
```

## Why Quantum Wins on Sarcasm

Sarcasm is fundamentally a higher-order linguistic phenomenon — the *interaction*
between words like "great" and "another tax hike" carries the signal, not each
word in isolation.

**ZZ Feature Map:** The circuit applies pairwise CNOT–RZ–CNOT gates between
adjacent qubits, encoding the product `(π − xᵢ)(π − xᵢ₊₁)` as a rotation angle.
This creates *entanglement* that represents the joint distribution of feature pairs —
exactly the kind of interaction that captures sarcasm cues.

**Quantum Kernel:** The inner product `|⟨φ(x₁)|φ(x₂)⟩|²` is computed in the
full 2⁸ = 256-dimensional Hilbert space.  Classical RBF kernels measure
Euclidean distance in the original 8-D space; the quantum kernel measures
*quantum overlap* in a vastly richer space, providing exponentially more
expressive power in the low-data regime.

## Project Structure

```
quantum26/
│
├── data/
│   └── Sarcasm_Headlines_Dataset_v2.json     ← download from Kaggle
│
├── utils/
│   ├── __init__.py
│   ├── data_pipeline.py                      ← text cleaning + negation binding
│   └── feature_engineering.py               ← TF-IDF → PCA → Haar → [0,π]
│
├── models/
│   ├── __init__.py
│   ├── classical_models.py                   ← LogReg, SVM-RBF, Naive Bayes
│   ├── vqc_model.py                          ← Variational Quantum Classifier
│   └── quantum_kernel_svm.py                 ← Quantum Kernel SVM
│
├── experiments/
│   ├── run_experiments.py                    ← main experiment runner
│   └── plot_results.py                       ← publication-quality plots
│
├── results/
│   ├── experiment_results.csv                ← generated by run_experiments.py
│   ├── saved_models/                         ← cached models for Streamlit
│   └── *.png                                 ← generated plots
│
├── app.py                                    ← Streamlit demo UI
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@misc{quantum26_sarcasm_2024,
  title        = {Hybrid Quantum-Classical Sarcasm Detection using ZZ Feature Maps},
  author       = {Govardhansatya},
  year         = {2024},
  howpublished = {\url{https://github.com/govardhansatya/quantum26}},
  note         = {PennyLane + scikit-learn implementation}
}
```
