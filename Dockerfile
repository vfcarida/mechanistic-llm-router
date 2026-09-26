# Multi-stage optimized Dockerfile for Mechanistic LLM Router Gateway
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .[dev]

FROM python:3.12-slim AS runner

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY configs/ /app/configs/

EXPOSE 8000 9090

ENV PYTHONPATH=/app/src
ENV ROUTER_SEED=42

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" || exit 1

CMD ["uvicorn", "mechanistic_router.gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]
