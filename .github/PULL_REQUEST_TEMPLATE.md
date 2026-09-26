## Description

Brief summary of changes proposed in this pull request and the rationale behind them.

Fixes #(issue) / Related to #(issue)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance / refactoring / technical debt

## Scientific & Architectural Integrity

- [ ] **No Label Leakage**: Ensures `RoutingRequest` remains strictly independent of ground-truth evaluation labels (`reference_tier`, `task_complexity`).
- [ ] **Empirical / Benchmark Integrity**: Any claimed routing improvement includes statistical validation (e.g. bootstrap confidence intervals, Pareto curves).

## Quality Gates Checklist

- [ ] Ran `ruff check .` with zero errors.
- [ ] Ran `ruff format --check .` and verified code formatting.
- [ ] Ran `mypy src/` and verified static type checking passes.
- [ ] Ran `pytest` and all unit / integration tests pass.
- [ ] Added or updated unit tests covering the new functionality or bug fix.
- [ ] Updated documentation and `CHANGELOG.md` where appropriate.
