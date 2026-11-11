"""Ingestão do Breast Cancer Wisconsin (Diagnostic).

Fonte preferida: ``sklearn.datasets.load_breast_cancer`` (offline, sem credenciais).
Alternativas: download UCI via URL HTTP estável, ou Kaggle se credenciais existirem.

O conjunto é o mesmo disponibilizado no Kaggle (uciml/breast-cancer-wisconsin-data)
e na UCI ML Repository (Wolberg et al.).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from breast_cancer_ml.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FEATURE_NAMES,
    ID_COLUMN_CANDIDATES,
    TARGET_COL,
    UCI_DATA_URL,
)
from breast_cancer_ml.utils.io import ensure_dirs

logger = logging.getLogger(__name__)

# Colunas UCI wdbc.data: id, diagnosis (M/B), depois 30 features numéricas
UCI_COLUMNS = ["id", "diagnosis"] + [f"f{i}" for i in range(30)]


def _drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c in ID_COLUMN_CANDIDATES]
    if drop:
        logger.info("Removendo colunas de identificação (anti-vazamento): %s", drop)
        df = df.drop(columns=drop)
    return df


def load_from_sklearn() -> pd.DataFrame:
    """Carrega via scikit-learn (equivalente ao UCI/Kaggle Diagnostic)."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # sklearn: target 0=malignant, 1=benign; features já nomeadas
    df = df.rename(columns={"target": TARGET_COL})
    return df


def load_from_uci(dest: Path | None = None) -> pd.DataFrame:
    """Baixa o CSV WDBC da UCI e normaliza para o schema sklearn."""
    ensure_dirs(DATA_RAW_DIR)
    dest = dest or (DATA_RAW_DIR / "wdbc.data")
    if not dest.exists():
        try:
            import httpx

            logger.info("Baixando dataset UCI de %s", UCI_DATA_URL)
            response = httpx.get(UCI_DATA_URL, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            dest.write_bytes(response.content)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Falha ao baixar UCI ({UCI_DATA_URL}). Use source='sklearn' (padrão, offline)."
            ) from exc

    raw = pd.read_csv(dest, header=None, names=UCI_COLUMNS)
    # Mapear diagnosis M/B -> 0/1 (sklearn: malignant=0, benign=1)
    diagnosis = raw["diagnosis"].astype(str).str.upper()
    target = diagnosis.map({"M": 0, "B": 1})
    if target.isna().any():
        raise ValueError("Valores inesperados em diagnosis; esperado M/B.")

    feature_cols = [c for c in raw.columns if c.startswith("f")]
    features = raw[feature_cols].copy()
    features.columns = FEATURE_NAMES
    df = features.copy()
    df[TARGET_COL] = target.astype(int)
    df = _drop_id_columns(pd.concat([raw[["id"]], df], axis=1))
    # Reordenar: features + target
    return df[FEATURE_NAMES + [TARGET_COL]]


def load_from_kaggle(dest_dir: Path | None = None) -> pd.DataFrame:
    """Opcional: ``kaggle datasets download`` se KAGGLE_USERNAME/KEY estiverem setados."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise OSError("KAGGLE_USERNAME e KAGGLE_KEY são necessários para source='kaggle'.")

    dest_dir = dest_dir or DATA_RAW_DIR
    ensure_dirs(dest_dir)
    zip_path = dest_dir / "breast-cancer-wisconsin-data.zip"
    csv_candidates = list(dest_dir.glob("**/data.csv")) + list(dest_dir.glob("**/*.csv"))

    if not csv_candidates:
        logger.info("Baixando dataset Kaggle uciml/breast-cancer-wisconsin-data")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "uciml/breast-cancer-wisconsin-data",
                "-p",
                str(dest_dir),
                "--unzip",
            ],
            check=True,
        )
        csv_candidates = list(dest_dir.glob("**/data.csv")) + list(dest_dir.glob("**/*.csv"))

    if not csv_candidates:
        raise FileNotFoundError(f"CSV Kaggle não encontrado em {dest_dir} (zip={zip_path})")

    raw = pd.read_csv(csv_candidates[0])
    # Kaggle típico: id, diagnosis, features..., Unnamed: 32
    raw = raw.loc[:, ~raw.columns.str.contains("^Unnamed")]
    raw = _drop_id_columns(raw)

    if "diagnosis" in raw.columns:
        diagnosis = raw["diagnosis"].astype(str).str.upper()
        target = diagnosis.map({"M": 0, "B": 1})
        features = raw.drop(columns=["diagnosis"])
    elif TARGET_COL in raw.columns:
        target = raw[TARGET_COL]
        features = raw.drop(columns=[TARGET_COL])
    else:
        raise ValueError("Coluna de alvo (diagnosis/target) não encontrada no CSV Kaggle.")

    # Normalizar nomes se possível
    if list(features.columns) != FEATURE_NAMES and len(features.columns) == len(FEATURE_NAMES):
        features.columns = FEATURE_NAMES

    df = features.copy()
    df[TARGET_COL] = target.astype(int)
    return df


def load_dataset(source: str = "sklearn") -> pd.DataFrame:
    """Carrega o dataset a partir da fonte solicitada.

    Parameters
    ----------
    source:
        ``sklearn`` (padrão, offline), ``uci`` (HTTP) ou ``kaggle`` (credenciais).
    """
    source = source.lower().strip()
    if source == "sklearn":
        df = load_from_sklearn()
    elif source == "uci":
        df = load_from_uci()
    elif source == "kaggle":
        df = load_from_kaggle()
    else:
        raise ValueError(f"Fonte desconhecida: {source!r}. Use sklearn|uci|kaggle.")

    ensure_dirs(DATA_PROCESSED_DIR)
    csv_out = DATA_PROCESSED_DIR / "breast_cancer.csv"
    df.to_csv(csv_out, index=False)
    logger.info("Dataset processado salvo em %s", csv_out)
    return df
