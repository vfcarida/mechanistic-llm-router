# Installation Guide

This guide details how to install and configure `mechanistic-llm-router` for local development, research experiments, and production gateway deployment.

---

## Requirements

- **Python**: `>= 3.10` (tested with Python 3.12)
- **PyTorch**: `>= 2.0.0`
- **Operating System**: Linux, macOS, or Windows

---

## 1. Standard Installation

Clone the repository and install dependencies using `pip` or modern package managers (`uv`, `poetry`):

```bash
# Clone the repository
git clone https://github.com/vfcarida/mechanistic-llm-router.git
cd mechanistic-llm-router

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install runtime package
pip install -e .
```

---

## 2. Development & Research Setup

To run test suites, static analysis, benchmarks, and experimental activation probing on transformer models:

```bash
# Install with all development extras
pip install -e ".[dev]"
```

Verify the environment installation:

```bash
# Run test suite
pytest

# Run linter and formatting checks
ruff check .
ruff format --check .

# Run static type checking
mypy src/
```

---

## 3. Environment Variables Configuration

The gateway and routing policies are configured using environment variables with the `ROUTER_` prefix:

| Variable | Type | Default | Description |
|---|---|---|---|
| `ROUTER_API_KEY` | `string` | *(required)* | Secret key used for authenticating incoming API requests. |
| `ROUTER_MAX_PROMPT_CHARS` | `integer` | `10000` | Maximum character length permitted per prompt payload. |
| `ROUTER_EFFECTIVE_LAMBDA` | `float` | `0.5` | Balances cost optimization ($0.0$) versus accuracy/capability ($1.0$). |
| `ROUTER_METRICS_ENABLED` | `boolean` | `true` | Enables OpenTelemetry and Prometheus metric exports. |
| `PROMETHEUS_METRICS_PORT` | `integer` | `9090` | HTTP port for Prometheus metrics scraping. |

Create a `.env` file in the project root:

```bash
ROUTER_API_KEY="your-secure-secret-key"
ROUTER_MAX_PROMPT_CHARS=10000
ROUTER_EFFECTIVE_LAMBDA=0.5
```

---

## 4. Docker Deployment

A multi-stage Docker build is provided for containerized deployment:

```bash
# Build the Docker image
docker build -t mechanistic-llm-router:latest .

# Run the container with environment variables
docker run -d \
  -p 8000:8000 \
  -p 9090:9090 \
  -e ROUTER_API_KEY="your-secure-secret-key" \
  --name llm-router \
  mechanistic-llm-router:latest
```

You can also use Docker Compose to spin up the router along with Prometheus and Grafana:

```bash
docker compose up -d
```
