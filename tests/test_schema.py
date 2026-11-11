"""Testes de schema e anti-vazamento de ID."""

from __future__ import annotations

import pandas as pd
import pytest

from breast_cancer_ml.ingestion.load import load_from_sklearn
from breast_cancer_ml.validation.schema import assert_no_id_leakage, validate_dataframe


def test_validate_sklearn_dataset() -> None:
    df = load_from_sklearn()
    validated = validate_dataframe(df)
    assert len(validated) == 569
    assert validated["target"].isin([0, 1]).all()


def test_no_id_column_in_sklearn_load() -> None:
    df = load_from_sklearn()
    assert "id" not in df.columns
    assert_no_id_leakage(df)


def test_id_leakage_raises() -> None:
    df = load_from_sklearn().copy()
    df["id"] = range(len(df))
    with pytest.raises(ValueError, match="Vazamento"):
        validate_dataframe(df)


def test_bad_target_raises() -> None:
    from pandera.errors import SchemaError, SchemaErrors

    df = load_from_sklearn().copy()
    df.loc[0, "target"] = 2
    with pytest.raises((SchemaError, SchemaErrors, ValueError)):
        validate_dataframe(df)


def test_assert_no_id_leakage_patient_id() -> None:
    df = pd.DataFrame({"patient_id": [1], "mean radius": [1.0]})
    with pytest.raises(ValueError, match="Vazamento"):
        assert_no_id_leakage(df)
