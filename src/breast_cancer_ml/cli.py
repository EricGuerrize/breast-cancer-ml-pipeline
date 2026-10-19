"""CLI Typer: ingest, validate, train, predict, demo."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from breast_cancer_ml import __version__
from breast_cancer_ml.config import FEATURE_NAMES, MODELS_DIR, TARGET_LABELS
from breast_cancer_ml.ingestion.load import load_dataset
from breast_cancer_ml.training.pipeline import assert_metrics_above_floor, train
from breast_cancer_ml.utils.io import load_model
from breast_cancer_ml.validation.schema import validate_dataframe

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"

app = typer.Typer(
    name="bcm",
    help="Breast Cancer ML Pipeline — CLI de portfólio (Eric Guerrize).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log detalhado."),
) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Mostra a versão do pacote."""
    console.print(f"breast-cancer-ml {__version__}")


@app.command("ingest")
def ingest_cmd(
    source: str = typer.Option(
        "sklearn",
        "--source",
        "-s",
        help="Fonte: sklearn (offline), uci (HTTP) ou kaggle (credenciais).",
    ),
) -> None:
    """Carrega e persiste o dataset processado."""
    df = load_dataset(source=source)
    console.print(f"[green]OK[/green] {len(df)} linhas × {df.shape[1]} colunas (source={source})")


@app.command("validate")
def validate_cmd(
    source: str = typer.Option("sklearn", "--source", "-s"),
) -> None:
    """Valida schema, dtypes, ranges e ausência de ID."""
    df = load_dataset(source=source)
    validated = validate_dataframe(df)
    console.print(f"[green]Validação OK[/green] — {len(validated)} linhas")


@app.command("train")
def train_cmd(
    source: str = typer.Option("sklearn", "--source", "-s"),
    check_floors: bool = typer.Option(
        True,
        "--check-floors/--no-check-floors",
        help="Falha se métricas do melhor modelo ficarem abaixo do piso.",
    ),
) -> None:
    """Treina modelos, salva artefatos e imprime métricas."""
    df = validate_dataframe(load_dataset(source=source))
    results = train(df)
    best = results["best_model"]
    metrics = results["models"][best]

    table = Table(title=f"Métricas — melhor modelo: {best}")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        table.add_row(key, f"{metrics[key]:.4f}")
    console.print(table)
    console.print("Calibração:", results["calibration"][best]["note"])

    if check_floors:
        assert_metrics_above_floor(metrics)
        console.print("[green]Pisos de métrica OK[/green]")


@app.command("predict")
def predict_cmd(
    model_path: Path = typer.Option(
        DEFAULT_MODEL_PATH,
        "--model",
        "-m",
        help="Caminho do modelo joblib.",
    ),
    features_json: str | None = typer.Option(
        None,
        "--features",
        "-f",
        help="JSON com as 30 features, ex: '{\"mean radius\": 17.9, ...}'",
    ),
) -> None:
    """Predição pontual a partir de JSON de features."""
    if not model_path.exists():
        console.print("[red]Modelo não encontrado.[/red] Rode: bcm train")
        raise typer.Exit(code=1)

    if not features_json:
        # Demo: usa a média do dataset sklearn
        from sklearn.datasets import load_breast_cancer

        bunch = load_breast_cancer(as_frame=True)
        sample = bunch.frame.drop(columns=["target"]).iloc[0]
        features = {name: float(sample[name]) for name in FEATURE_NAMES}
        console.print("[yellow]Sem --features: usando a 1ª amostra do sklearn.[/yellow]")
    else:
        features = json.loads(features_json)

    missing = [n for n in FEATURE_NAMES if n not in features]
    if missing:
        console.print(f"[red]Features ausentes:[/red] {missing}")
        raise typer.Exit(code=1)

    import pandas as pd

    model = load_model(model_path)
    row = pd.DataFrame([{n: features[n] for n in FEATURE_NAMES}])
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0][pred])
    label = TARGET_LABELS.get(pred, str(pred))
    console.print(f"Predição: [bold]{label}[/bold] (classe={pred}, confiança={proba:.3f})")


@app.command("demo")
def demo_cmd(
    source: str = typer.Option("sklearn", "--source", "-s"),
) -> None:
    """Pipeline completo offline: ingest → validate → train → predict."""
    console.rule("[bold]Demo Breast Cancer ML[/bold]")
    df = validate_dataframe(load_dataset(source=source))
    results = train(df)
    best = results["best_model"]
    metrics = results["models"][best]
    assert_metrics_above_floor(metrics)
    console.print(
        f"[green]Demo OK[/green] — {best} F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}"
    )
    # Smoke predict
    predict_cmd(model_path=DEFAULT_MODEL_PATH, features_json=None)


if __name__ == "__main__":
    app()
