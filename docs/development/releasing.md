# Release Process

This document defines the release workflow and versioning standards for the **Mechanistic LLM Router** package.

---

## Versioning Standards

We adhere to [Semantic Versioning 2.0.0](https://semver.org/):

$$\text{MAJOR}.\text{MINOR}.\text{PATCH}$$

- **MAJOR**: Incompatible API breaking changes or schema alterations.
- **MINOR**: Backward-compatible new routing strategies, algorithms, or features.
- **PATCH**: Backward-compatible bug fixes, performance optimizations, or documentation revisions.

---

## Pre-Release Quality Checklist

Before tagging any release:

- [ ] All unit, integration, and mathematical tests pass: `pytest` (100% pass rate).
- [ ] Static type analysis reports zero errors: `mypy src/`.
- [ ] Linter checks pass with zero warnings: `ruff check .`.
- [ ] Code formatting verified: `ruff format --check .`.
- [ ] Ast source-tree label leakage scan passes: `pytest tests/test_no_leakage.py`.
- [ ] `CHANGELOG.md` is updated with all notable additions, fixes, and security improvements under the tagged release header.
- [ ] Version number bumped in `pyproject.toml` and `src/mechanistic_router/__init__.py`.

---

## Building Distribution Artifacts

```bash
# Clean previous build artifacts
rm -rf dist/ build/ *.egg-info

# Build source distribution and binary wheel
python -m pip install --upgrade build
python -m build

# Inspect archive contents
twine check dist/*
```

---

## Publishing Releases

1. **Tag the Release Commit**:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0 - Production Causal Probe Router & Gateway"
   git push origin v0.1.0
   ```

2. **Publish to PyPI**:
   ```bash
   twine upload dist/*
   ```

3. **Deploy Container Image**:
   ```bash
   docker build -t mechanistic-llm-router:latest -t mechanistic-llm-router:v0.1.0 .
   docker push ghcr.io/vfcarida/mechanistic-llm-router:v0.1.0
   ```
