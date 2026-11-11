"""Testes de ingestão."""

from __future__ import annotations

from breast_cancer_ml.config import FEATURE_NAMES, TARGET_COL
from breast_cancer_ml.ingestion.load import load_dataset, load_from_sklearn


def test_sklearn_shape_and_columns() -> None:
    df = load_from_sklearn()
    assert df.shape == (569, 31)
    assert list(df.columns) == FEATURE_NAMES + [TARGET_COL]


def test_load_dataset_persists(tmp_path, monkeypatch) -> None:
    import breast_cancer_ml.ingestion.load as load_mod

    monkeypatch.setattr(load_mod, "DATA_PROCESSED_DIR", tmp_path)
    df = load_dataset(source="sklearn")
    assert len(df) == 569
    assert (tmp_path / "breast_cancer.csv").exists()
