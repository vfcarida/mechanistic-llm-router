# Contributing to Mechanistic LLM Router

Thank you for your interest in contributing to the **Mechanistic LLM Router**! We welcome contributions from researchers, software engineers, and machine learning practitioners.

This document provides guidelines and instructions for setting up your development environment, running tests and code quality tools, and submitting pull requests.

---

## Code of Conduct

By participating in this project, you agree to abide by our standards of respectful, collaborative, and inclusive communication.

---

## Development Setup

### Prerequisites

- **Python 3.10+** (Python 3.12 is recommended and tested in CI)
- **Git**
- Virtual environment tool (such as `venv` or `uv`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vfcarida/mechanistic-llm-router.git
   cd mechanistic-llm-router
   ```

2. Create and activate a virtual environment:
   ```bash
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install development dependencies and the package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks (optional but highly recommended):
   ```bash
   pre-commit install
   ```

---

## Quality Gates and Standards

Before submitting any code, verify that all quality checks pass. Our CI strictly enforces these gates:

### 1. Linting & Formatting (Ruff)

We use [Ruff](https://github.com/astral-sh/ruff) for fast linting and code formatting:

```bash
# Check for lint violations
ruff check .

# Automatically fix lint violations where possible
ruff check --fix .

# Verify formatting without modifying files
ruff format --check .

# Auto-format all code
ruff format .
```

### 2. Static Type Checking (MyPy)

We enforce strict type annotations across the codebase:

```bash
mypy src/
```

### 3. Testing & Coverage (Pytest)

All tests must pass, and new features must include unit or integration tests:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=mechanistic_router --cov-report=term-missing
```

---

## Scientific & Architectural Principles

When contributing to the core routing engine or evaluation framework, please uphold these key principles:

1. **Strict Leakage Isolation**:
   The runtime `RoutingRequest` must *never* contain ground-truth benchmark labels or task tier indicators. Routing decisions must be derived solely from prompt inputs, model configurations, and activation/interpretability signals.

2. **Causal Validity**:
   Interpretability signals (e.g. SVD entropy $d_{\text{eff}}$, Fisher discriminant $J$, SAE circuit activations) should directly and causally inform the routing policy. Avoid ornamental metric calculations that do not affect the decision path.

3. **Statistical Rigor**:
   When introducing or evaluating new routing policies, use paired bootstrap confidence intervals (e.g., 95% CIs) and evaluate on disjoint prompt splits to prevent optimistic evaluation bias.

---

## Pull Request Guidelines

1. **Branch Naming**:
   Use descriptive branch names:
   - `feat/feature-name`
   - `fix/bug-description`
   - `docs/documentation-update`
   - `refactor/component-name`

2. **Atomic Commits**:
   Write clear, concise commit messages adhering to standard conventions (e.g., `feat: implement causal probe router`).

3. **Checklist Before Submitting**:
   - [ ] All tests pass (`pytest`).
   - [ ] Ruff check and formatting pass (`ruff check . && ruff format --check .`).
   - [ ] MyPy type checking passes (`mypy src/`).
   - [ ] New code is covered by automated tests.
   - [ ] Documentation has been updated (docstrings, `README.md`, or `docs/` where applicable).

---

## Reporting Issues

If you find a bug or have a feature proposal, please check the existing issues before opening a new one.

- **Bug Reports**: Include Python version, OS, reproducible code snippet, and complete traceback.
- **Feature Requests**: Describe the use case, expected interface, and architectural implications.
- **Security Disclosures**: Please see our [SECURITY.md](SECURITY.md) for vulnerability reporting procedures.
