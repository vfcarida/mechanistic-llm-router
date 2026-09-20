# Mechanistic LLM Router — Baseline & Preflight Record

- **Task ID**: MLR-T01
- **Repository**: [mechanistic-llm-router](https://github.com/vfcarida/mechanistic-llm-router)
- **Baseline Commit SHA**: `fdfc05c0c75ef8c6b50eda28fd5db2dadeb47bfb` (branch `main`)
- **Execution Date**: 2026-09-18
- **Operating System**: Windows 11 (NT 10.0.26200 amd64 / win32)
- **Host Python**: Python 3.12.10 (`C:\Users\vinicius\AppData\Local\Programs\Python\Python312\python.exe`)
- **Virtual Environment**: `.venv` (`c:\Users\vinicius\Documents\GeminiCodes\mechanistic-llm-router\.venv`)
- **Companion File**: [SHARED-CONVENTIONS.md](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/SHARED-CONVENTIONS.md)
- **Execution Boundary Compliance**: Local `.venv` only, zero paid model API network calls, zero modifications to `src/`, no unapproved wheel downloads, no remote git pushing.

---

## 1. Environment & Dependency Inventory

### Host & Virtual Environment Provisioning
To adhere to execution boundary 8 (*"Torch/large downloads require explicit approval; otherwise document as blocked"*), the disposable virtual environment `.venv` was initialized using `--system-site-packages` on top of Python 3.12.10:
```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\pip install --no-build-isolation --no-deps -e .
```
This linked the existing local CPU PyTorch stack without multi-gigabyte PyPI downloads.

### Installed Toolchain & Package Versions
- **Python**: `3.12.10`
- **PyTorch**: `2.13.0+cpu`
- **LiteLLM**: `1.96.2`
- **FastAPI**: `0.129.0`
- **Pydantic**: `2.13.4`
- **pytest**: `9.1.1`
- **pytest-cov**: `7.1.0`
- **pytest-asyncio**: `1.4.0`
- **pytest-mock**: `3.15.1`
- **ruff**: `0.16.3`
- **mypy**: `2.3.1`
- **OpenTelemetry API**: `1.44.0`
- **OpenTelemetry SDK**: `1.44.0`
- **Prometheus Client**: `0.25.0`

---

## 2. Executed Quality Gates Summary

| Gate | Exact Command | Exit Code | Summary | Status |
|---|---|:---:|---|:---:|
| **Lint & Style** | `.\.venv\Scripts\python -m ruff check .` | `1` | 152 errors found (81 fixable) across line length, whitespace, imports, and unused symbols | **RECORDED** |
| **Type Check** | `.\.venv\Scripts\python -m mypy src` | `1` | Blocked by syntax version mismatch in numpy stub (Python 3.10 pin in `pyproject.toml` vs Python 3.12 PEP 695 `type` statement) | **RECORDED** |
| **Pytest Suite** | `.\.venv\Scripts\python -m pytest -q --cov=mechanistic_router --cov-report=term-missing` | `0` | **27 passed, 3 warnings in 26.73s**; **94% coverage** (675 stmts, 38 missed) | **PASS** |

---

## 3. Detailed Command Execution Logs

### A. Static Linting (`ruff check .`)
```text
Command: .\.venv\Scripts\python -m ruff check --statistics .
Exit Code: 1

57	E501	[ ] line-too-long
33	W293	[-] blank-line-with-whitespace
32	I001	[*] unsorted-imports
10	W291	[-] trailing-whitespace
 9	F401	[*] unused-import
 8	E402	[ ] module-import-not-at-top-of-file
 2	F841	[ ] unused-variable
 1	B007	[ ] unused-loop-control-variable
Found 152 errors.
[*] 81 fixable with the `--fix` option (6 hidden fixes can be enabled with the `--unsafe-fixes` option).
```

### B. Static Type Checking (`mypy src`)
```text
Command: .\.venv\Scripts\python -m mypy src
Exit Code: 1

C:\Users\vinicius\AppData\Local\Programs\Python\Python312\Lib\site-packages\numpy\__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater  [syntax]
Found 1 error in 1 file (errors prevented further checking)
```
*Root Cause*: `pyproject.toml` defines `[tool.mypy] python_version = "3.10"`. The host environment's `numpy 2.5.2` type stubs utilize PEP 695 `type` alias statements, which causes MyPy's Python 3.10 parser to abort during initial AST construction of dependencies.

### C. Test Suite & Coverage (`pytest -q --cov=mechanistic_router --cov-report=term-missing`)
```text
Command: .\.venv\Scripts\python -m pytest -q --cov=mechanistic_router --cov-report=term-missing
Exit Code: 0

...........................                                              [100%]
Running teardown with pytest sessionfinish...

============================== warnings summary ===============================
src\mechanistic_router\gateway\server.py:30: DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
..\..\..\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\applications.py:4681: DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
..\..\..\AppData\Local\Programs\Python\Python312\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.

=============================== tests coverage ================================
______________ coverage: platform win32, python 3.12.10-final-0 _______________

Name                                                      Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------------------
src\mechanistic_router\__init__.py                            9      0   100%
src\mechanistic_router\config.py                             15      0   100%
src\mechanistic_router\core\__init__.py                       0      0   100%
src\mechanistic_router\core\encoder.py                       19      0   100%
src\mechanistic_router\core\router.py                        98      8    92%   45, 47, 87, 239, 257-260
src\mechanistic_router\data\__init__.py                       0      0   100%
src\mechanistic_router\data\mock_dataset.py                  21      0   100%
src\mechanistic_router\gateway\__init__.py                    3      0   100%
src\mechanistic_router\gateway\dispatcher.py                 20      2    90%   50, 95
src\mechanistic_router\gateway\server.py                     53      4    92%   33, 59, 80, 82
src\mechanistic_router\models\__init__.py                     0      0   100%
src\mechanistic_router\models\pool.py                        11      0   100%
src\mechanistic_router\models\types.py                       12      0   100%
src\mechanistic_router\observability\__init__.py              2      0   100%
src\mechanistic_router\observability\metrics.py              30      2    93%   57-58
src\mechanistic_router\probing\__init__.py                    4      0   100%
src\mechanistic_router\probing\nnsight_probe.py              11      0   100%
src\mechanistic_router\probing\sae_engine.py                 36      4    89%   56, 58, 73-74
src\mechanistic_router\probing\transformer_lens_hook.py      28      0   100%
src\mechanistic_router\routers\__init__.py                    5      0   100%
src\mechanistic_router\routers\base.py                       14      1    93%   37
src\mechanistic_router\routers\cost_performance.py           29      1    97%   58
src\mechanistic_router\routers\mechanistic.py                86      7    92%   45, 50, 71-72, 159, 167-168
src\mechanistic_router\routers\semantic.py                   46      1    98%   81
src\mechanistic_router\schemas\__init__.py                    3      0   100%
src\mechanistic_router\schemas\openai.py                     34      0   100%
src\mechanistic_router\schemas\routing.py                    26      0   100%
src\mechanistic_router\signals\__init__.py                    0      0   100%
src\mechanistic_router\signals\math_utils.py                 43      6    86%   28, 32-33, 38, 88, 106
src\mechanistic_router\utils\__init__.py                      0      0   100%
src\mechanistic_router\utils\metrics.py                      17      2    88%   26, 45
---------------------------------------------------------------------------------------
TOTAL                                                       675     38    94%
27 passed, 3 warnings in 26.73s
```

---

## 4. Coverage Discrepancy Analysis (Finding MLR-F12)

Prior to this run, the repository contained a checked-in `.coverage` SQLite database (size 53,248 bytes) and README documentation claiming 98% coverage across 45 unit tests.
Our executed baseline reveals:
1. **Total Tests**: Exactly 27 unit and integration tests exist in `tests/`, not 45 tests.
2. **Actual Coverage**: **94%** (675 statements, 38 missed), not 98%.
3. **Execution Freshness**: The committed `.coverage` file was an artifact from a prior state or external run and did not match the checked-in test suite.

---

## 5. Label Leakage Inventory (Finding MLR-F9)

The repository's routing architecture suffers from pervasive label leakage: `TaskComplexity` (ground truth label) is passed directly into the router's input parameters or used to synthetically fabricate competence activations and ceiling evaluations.

### Leakage Mechanisms in Source (`src/`)
- **`src/mechanistic_router/core/router.py:117-128, 156-170, 194-196`**: `route(prompt, complexity: TaskComplexity)` takes the ground truth complexity as a required argument, which is then fed into `_simulate_fisher_activations(..., complexity)`. If `task_idx <= ceiling_idx`, it sets high synthetic competence `0.85 + ...`, artificially forcing the Fisher J score to pass.
- **`src/mechanistic_router/routers/cost_performance.py:28-37`**: `route(request: RoutingRequest)` reads `request.task_complexity` and directly checks `req_idx <= ceiling_idx` to assign competence and route the request.
- **`src/mechanistic_router/routers/mechanistic.py:64-73, 116-140`**: Reads `request.task_complexity` to simulate Fisher competence and select hardcoded SAE feature vectors (`[42, 108]` if `TaskComplexity.COMPLEX` else `[7]`).

### Inventory of Existing Tests Encoding Leakage

| Test Function | File Path & Lines | Injected Label | Leakage Assertion Marker | Root Leakage Cause |
|---|---|---|---|---|
| `test_cost_performance_router` (Routine) | [tests/test_routers.py:21-25](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_routers.py#L21-L25) | `TaskComplexity.ROUTINE` | `assert decision_routine.selected_model == "SLM-BERTau-Local"` | Passes ground-truth label via `RoutingRequest(..., task_complexity=...)` |
| `test_cost_performance_router` (Complex) | [tests/test_routers.py:27-30](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_routers.py#L27-L30) | `TaskComplexity.COMPLEX` | `assert decision_complex.selected_model == "LLM-Frontier-Oracle"` | Passes ground-truth label via `RoutingRequest(..., task_complexity=...)` |
| `test_mechanistic_router` | [tests/test_routers.py:55-63](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_routers.py#L55-L63) | `TaskComplexity.ROUTINE` | `assert decision_routine.selected_model == "SLM-BERTau-Local"`, `assert signals["SLM-BERTau-Local"].is_competent is True` | Injects routine complexity to force synthetic Fisher competence gate |
| `test_router_routine_task` | [tests/test_router.py:17-23](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_router.py#L17-L23) | `TaskComplexity.ROUTINE` | `assert selected == "SLM-BERTau-Local"`, `assert details["SLM-BERTau-Local"]["is_competent"] == 1.0` | Directly passes `TaskComplexity.ROUTINE` into legacy `route(prompt, complexity)` |
| `test_router_moderate_task` | [tests/test_router.py:25-32](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_router.py#L25-L32) | `TaskComplexity.MODERATE` | `assert selected == "LLM-Mid-Tier"`, `assert details["LLM-Mid-Tier"]["is_competent"] == 1.0` | Directly passes `TaskComplexity.MODERATE` into legacy `route()` |
| `test_router_complex_task` | [tests/test_router.py:34-42](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_router.py#L34-L42) | `TaskComplexity.COMPLEX` | `assert selected == "LLM-Frontier-Oracle"`, `assert details["LLM-Frontier-Oracle"]["is_competent"] == 1.0` | Directly passes `TaskComplexity.COMPLEX` into legacy `route()` |
| `test_lambda_elasticity` | [tests/test_router.py:44-53](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_router.py#L44-L53) | `TaskComplexity.ROUTINE` | `selected, _ = router_rich.route(prompt, TaskComplexity.ROUTINE)` | Requires complexity label even when evaluating lambda sensitivity |
| `test_empty_prompt_handling` | [tests/test_router.py:55-60](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_router.py#L55-L60) | `TaskComplexity.ROUTINE` | `legacy_router.route(prompt, TaskComplexity.ROUTINE)` | Requires ground-truth complexity label even on empty prompt fallback |
| `test_effective_dimensionality_is_dynamic` | [tests/test_router.py:74-90](file:///c:/Users/vinicius/Documents/GeminiCodes/mechanistic-llm-router/tests/test_router.py#L74-L90) | `TaskComplexity.ROUTINE` (short) vs `TaskComplexity.COMPLEX` (long) | `assert d_eff_long > d_eff_short` | Injects ground truth complexity into `details_short` and `details_long` to assert dynamic dimensional divergence |

---

## 6. Verification of Source Preservation

- `git status --porcelain src/` returns **empty**.
- No file under `src/` has been altered.
- All baseline tests, routers, and gateway endpoints retain their exact original behavior.
