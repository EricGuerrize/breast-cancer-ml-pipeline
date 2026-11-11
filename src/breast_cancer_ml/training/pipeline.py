"""Treino com Pipeline sklearn (scaler + modelo) — sem vazamento de dados."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from breast_cancer_ml.config import (
    FEATURE_NAMES,
    FIGURES_DIR,
    METRIC_FLOORS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TARGET_LABELS,
    TEST_SIZE,
    USE_MLFLOW,
)
from breast_cancer_ml.utils.io import ensure_dirs, save_json, save_model

logger = logging.getLogger(__name__)


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Pipelines com preprocessing dentro do Pipeline (evita leakage)."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm,
        "target_labels": TARGET_LABELS,
    }


def _calibration_note(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Nota de calibração (frações vs. confiança média por bin)."""
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=5, strategy="quantile")
        return {
            "note": (
                "Curva de calibração (quantile, 5 bins). "
                "Probabilidades bem calibradas aproximam a diagonal. "
                "Este projeto é educacional — não use como dispositivo médico."
            ),
            "fraction_of_positives": [float(x) for x in frac_pos],
            "mean_predicted_value": [float(x) for x in mean_pred],
        }
    except ValueError:
        return {
            "note": "Calibração indisponível (bins insuficientes).",
            "fraction_of_positives": [],
            "mean_predicted_value": [],
        }


def _maybe_mlflow(model_name: str, metrics: dict[str, Any], model: Pipeline) -> None:
    if not USE_MLFLOW:
        return
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("breast-cancer-ml")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params({"model": model_name, "test_size": TEST_SIZE})
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                mlflow.log_metric(key, metrics[key])
            mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Métricas registradas no MLflow para %s", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow opcional falhou: %s", exc)


def _feature_importance(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_).ravel()
    else:
        return {}
    pairs = sorted(zip(feature_names, values, strict=True), key=lambda x: -x[1])
    return {name: float(val) for name, val in pairs}


def train(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    persist: bool = True,
) -> dict[str, Any]:
    """Split estratificado, treino de baselines, persistência de artefatos."""
    ensure_dirs(MODELS_DIR, METRICS_DIR, FIGURES_DIR)

    x = df[FEATURE_NAMES]
    y = df[TARGET_COL].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results: dict[str, Any] = {
        "random_state": random_state,
        "test_size": test_size,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "models": {},
        "best_model": None,
        "metric_floors": METRIC_FLOORS,
        "calibration": {},
    }

    best_name = None
    best_f1 = -1.0
    best_pipeline: Pipeline | None = None

    for name, pipe in build_models(random_state).items():
        logger.info("Treinando %s ...", name)
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        y_proba = pipe.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test.to_numpy(), y_pred, y_proba)
        metrics["feature_importance"] = _feature_importance(pipe, FEATURE_NAMES)
        results["models"][name] = metrics
        results["calibration"][name] = _calibration_note(y_test.to_numpy(), y_proba)
        _maybe_mlflow(name, metrics, pipe)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_pipeline = pipe

        if persist:
            save_model(pipe, MODELS_DIR / f"{name}.joblib")

    results["best_model"] = best_name
    if persist and best_pipeline is not None and best_name is not None:
        save_model(best_pipeline, MODELS_DIR / "best_model.joblib")
        save_json(results, METRICS_DIR / "metrics.json")
        _save_confusion_figure(
            results["models"][best_name]["confusion_matrix"],
            best_name,
        )
        logger.info("Melhor modelo: %s (F1=%.4f)", best_name, best_f1)

    return results


def _save_confusion_figure(cm: list[list[int]], model_name: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["maligno (0)", "benigno (1)"],
            yticklabels=["maligno (0)", "benigno (1)"],
            ax=ax,
        )
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusão — {model_name}")
        fig.tight_layout()
        path = FIGURES_DIR / "confusion_matrix.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Figura salva em %s", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível salvar figura: %s", exc)


def assert_metrics_above_floor(
    metrics: dict[str, Any], floors: dict[str, float] | None = None
) -> None:
    floors = floors or METRIC_FLOORS
    failures = []
    for key, floor in floors.items():
        value = metrics.get(key)
        if value is None or value < floor:
            failures.append(f"{key}={value} < {floor}")
    if failures:
        raise AssertionError("Métricas abaixo do piso: " + "; ".join(failures))
