"""Configuração central do pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Raiz do repositório (dois níveis acima de src/breast_cancer_ml)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = int(os.getenv("BCM_RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("BCM_TEST_SIZE", "0.2"))
USE_MLFLOW = os.getenv("BCM_USE_MLFLOW", "0") == "1"

# Piso mínimo de métricas no conjunto de teste (semente fixa) — sanity check
METRIC_FLOORS = {
    "accuracy": 0.90,
    "precision": 0.90,
    "recall": 0.90,
    "f1": 0.90,
    "roc_auc": 0.95,
}

# Colunas de feature no dataset sklearn / UCI (sem id)
FEATURE_NAMES: list[str] = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

TARGET_COL = "target"
TARGET_LABELS = {0: "malignant", 1: "benign"}  # sklearn encoding
# No Kaggle/UCI original: M = malignant, B = benign

# URL estável UCI (espelho OpenML / UCI archive via raw CSV conhecido)
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
)

ID_COLUMN_CANDIDATES = ("id", "ID", "Id")
