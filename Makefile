.PHONY: setup lint test train demo dashboard docker-build docker-up clean

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

setup:
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"
	@command -v pre-commit >/dev/null && pre-commit install || true

lint:
	ruff check src tests dashboard
	ruff format --check src tests dashboard

fmt:
	ruff check --fix src tests dashboard
	ruff format src tests dashboard

test:
	pytest

train:
	bcm train --source sklearn

demo:
	bcm demo --source sklearn

dashboard:
	streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port $${PORT:-8501}

docker-build:
	docker build -t breast-cancer-ml:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
