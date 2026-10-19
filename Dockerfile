# Pipeline + dashboard Breast Cancer ML (Python 3.11+)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY dashboard ./dashboard
COPY Makefile ./

RUN pip install -U pip && pip install -e ".[dev]"

# Treina artefato no build para demo offline (sklearn, sem rede de dados)
RUN bcm train --source sklearn --check-floors

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
