"""Validação de schema (pandera) — dtypes, ranges, anti-vazamento de ID."""

from __future__ import annotations

import logging

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from breast_cancer_ml.config import FEATURE_NAMES, ID_COLUMN_CANDIDATES, TARGET_COL

logger = logging.getLogger(__name__)


def build_schema() -> DataFrameSchema:
    """Schema das 30 features numéricas + target binário (0/1)."""
    feature_cols = {
        name: Column(
            float,
            checks=[
                Check.ge(0, error=f"{name} deve ser >= 0"),
                Check(lambda s: s.notna().all(), error=f"{name} não pode ter NaN"),
            ],
            nullable=False,
            coerce=True,
        )
        for name in FEATURE_NAMES
    }
    cols = {
        **feature_cols,
        TARGET_COL: Column(
            int,
            checks=[Check.isin([0, 1], error="target deve ser 0 (malignant) ou 1 (benign)")],
            nullable=False,
            coerce=True,
        ),
    }
    return DataFrameSchema(
        cols,
        strict=True,
        coerce=True,
        name="BreastCancerWisconsinDiagnostic",
    )


BREAST_CANCER_SCHEMA = build_schema()


def assert_no_id_leakage(df: pd.DataFrame) -> None:
    """Garante que colunas de identificação não estejam presentes."""
    leaked = [c for c in df.columns if c in ID_COLUMN_CANDIDATES]
    # Também rejeita colunas que claramente são IDs
    leaked += [c for c in df.columns if str(c).lower() in {"patient_id", "sample_id"}]
    leaked = sorted(set(leaked))
    if leaked:
        raise ValueError(
            f"Vazamento de identificação detectado: colunas {leaked}. Remova IDs antes do treino."
        )


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida dtypes, target, ranges e ausência de ID."""
    assert_no_id_leakage(df)

    missing = set(FEATURE_NAMES + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    extra = set(df.columns) - set(FEATURE_NAMES + [TARGET_COL])
    if extra:
        logger.warning("Colunas extras serão ignoradas na validação: %s", sorted(extra))
        df = df[FEATURE_NAMES + [TARGET_COL]].copy()

    # Coagir features para float
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    validated = BREAST_CANCER_SCHEMA.validate(df, lazy=True)
    logger.info(
        "Validação OK: %d linhas, %d features, target balance=%s",
        len(validated),
        len(FEATURE_NAMES),
        validated[TARGET_COL].value_counts().to_dict(),
    )
    return validated
