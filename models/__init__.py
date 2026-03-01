"""
models package — classical baselines and quantum models for sarcasm detection.
"""
from .classical_models import (
    LogisticRegressionBaseline,
    ClassicalSVMBaseline,
    NaiveBayesBaseline,
    train_all_baselines,
    train_at_sample_sizes,
)
from .vqc_model import VQCClassifier
from .quantum_kernel_svm import QuantumKernelSVM

__all__ = [
    "LogisticRegressionBaseline",
    "ClassicalSVMBaseline",
    "NaiveBayesBaseline",
    "train_all_baselines",
    "train_at_sample_sizes",
    "VQCClassifier",
    "QuantumKernelSVM",
]
