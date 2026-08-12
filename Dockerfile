# Multi-stage optimized Dockerfile for Mechanistic LLM Router Gateway
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .[dev]

FROM python:3.10-slim AS runner

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY configs/ /app/configs/

EXPOSE 8000 9090

ENV PYTHONPATH=/app/src
ENV ROUTER_SEED=42

CMD ["uvicorn", "mechanistic_router.gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]
