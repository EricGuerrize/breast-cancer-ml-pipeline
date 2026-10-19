"""Testes de treino e piso de métricas (semente fixa)."""

from __future__ import annotations

from breast_cancer_ml.config import METRIC_FLOORS, RANDOM_STATE
from breast_cancer_ml.ingestion.load import load_from_sklearn
from breast_cancer_ml.training.pipeline import assert_metrics_above_floor, train
from breast_cancer_ml.validation.schema import validate_dataframe


def test_train_metrics_above_floor() -> None:
    df = validate_dataframe(load_from_sklearn())
    results = train(df, random_state=RANDOM_STATE, persist=False)
    best = results["best_model"]
    assert best is not None
    metrics = results["models"][best]
    assert_metrics_above_floor(metrics, METRIC_FLOORS)
    for model_metrics in results["models"].values():
        assert "confusion_matrix" in model_metrics
        assert model_metrics["roc_auc"] >= 0.90


def test_stratified_split_sizes() -> None:
    df = validate_dataframe(load_from_sklearn())
    results = train(df, random_state=42, test_size=0.2, persist=False)
    assert results["n_train"] + results["n_test"] == 569
    assert results["n_test"] == 114  # 20% estratificado de 569
