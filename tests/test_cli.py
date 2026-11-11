"""Smoke tests da CLI Typer."""

from __future__ import annotations

from typer.testing import CliRunner

from breast_cancer_ml.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "breast-cancer-ml" in result.stdout


def test_cli_validate() -> None:
    result = runner.invoke(app, ["validate", "--source", "sklearn"])
    assert result.exit_code == 0
    assert "Validação OK" in result.stdout


def test_cli_demo() -> None:
    result = runner.invoke(app, ["demo", "--source", "sklearn"])
    assert result.exit_code == 0
    assert "Demo OK" in result.stdout
