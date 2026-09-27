<div align="center">
  <h1>⚡ Mechanistic LLM Router</h1>
  <p><em>Research Prototype: Exploratory LLM Routing via Encoder-Target Decoupling & Prefill Probing Simulation</em></p>

  [![CI](https://github.com/vfcarida/mechanistic-llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/vfcarida/mechanistic-llm-router/actions/workflows/ci.yml)
  ![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
  ![Pydantic](https://img.shields.io/badge/Pydantic-v2-green.svg)
  ![LiteLLM](https://img.shields.io/badge/Dispatcher-LiteLLM-purple.svg)
  ![OpenTelemetry](https://img.shields.io/badge/Observability-OpenTelemetry-orange.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
</div>

---

> [!WARNING]
> **Status: Research & Production Evolution.** The repository provides two routing paradigms: (1) `CausalProbeRouter`, an empirical router driven by real prefill hidden states (e.g., from `SmolLM-135M`) and trained `LinearActivationProbe` hyperplanes, and (2) `SimulatedRouter` (`MechanisticRouter`), an exploratory simulation illustrating SVD Effective Dimensionality and Gaussian Fisher gating. See [`docs/BASELINE.md`](docs/BASELINE.md) and [`docs/MECHANISTIC_SPIKE_REPORT.md`](docs/MECHANISTIC_SPIKE_REPORT.md) for detailed empirical findings.

---

## 📖 Project Scope & Conceptual Exploration

The economic feasibility and latency scaling of multi-model AI architectures depend on intelligent request orchestration. Standard semantic routers rely on shallow text-embedding heuristics or static cost thresholds, often introducing routing latency without directly inspecting internal model dynamics.

The **Mechanistic LLM Router** investigates **Encoder-Target Decoupling** and mechanistic interpretability concepts for LLM routing:
1. **Empirical Causal Probing (`CausalProbeRouter`)**: Extracts unpooled prefill activations from lightweight open-weight local models, classifying routing difficulty via learned linear hyperplanes in latent activation space with zero ground-truth label leakage.
2. **Topological Complexity Probing (`SimulatedRouter`)**: Explores whether inspecting unpooled sequence activation matrices $A \in \mathbb{R}^{S \times D}$, computing Effective Dimensionality ($d_{eff}$) spectrum entropy, evaluating Fisher Discriminant separability ($J$), and extracting Sparse Autoencoder (SAE) circuits can provide topological signals of prompt complexity before full autoregressive generation.

---

## 🧠 Architecture Overview

```mermaid
graph TD
    A[User Prompt Query] --> B{Routing Strategy}

    subgraph Empirical Causal Probing (Production Strategy)
        B -->|causal-probe| C1[PrefillActivationExtractor / SmolLM-135M]
        C1 --> C2[LinearActivationProbe Hyperplane w·x + b]
        C2 --> C3[Difficulty Probability & Margin]
    end

    subgraph Topological Probing (Simulated Strategy)
        B -->|mechanistic| D1[SharedTrunkEncoder]
        D1 --> D2[Effective Dimensionality d_eff via SVD Entropy]
        D1 --> D3[Fisher Gating & Top-K SAE Circuits]
    end

    C3 --> E[Target Model Selection]
    D2 --> E
    D3 --> E

    E --> F[LiteLLM Dispatcher / SSE Streaming]
    F --> G((SLM Local - $0.02))
    F --> H((Mid-Tier LLM - $0.25))
    F --> I((Frontier Oracle - $1.50))
```

---

## 🧮 Mathematical Formulations (Theoretical Basis)

The mathematical implementations in [`src/mechanistic_router/signals/math_utils.py`](src/mechanistic_router/signals/math_utils.py) provide formal geometric signals:

### 1. Effective Dimensionality ($d_{eff}$)
Calculates spectrum entropy over singular values of sequence activation matrices $A \in \mathbb{R}^{S \times D}$:
$$\sigma = \text{SVD}(A)$$
$$E_i = \sigma_i^2 \quad \text{and} \quad p_i = \frac{E_i}{\sum_j E_j}$$
$$H = -\sum_{i} p_i \ln(p_i) \implies d_{eff} = \exp(H)$$

### 2. Fisher Separability ($J$)
Measures structural separability between success and failure clusters in latent representation space:
$$J = \frac{1}{D} \sum_{d=1}^{D} \frac{(\mu_{\text{success}, d} - \mu_{\text{failure}, d})^2}{\sigma^2_{\text{success}, d} + \sigma^2_{\text{failure}, d} + \epsilon}$$

### 3. Non-Decreasing Convex Hull (Pareto Optimization)
Constructs a cost-accuracy Pareto frontier mapping average query cost (x-axis) to response accuracy (y-axis) to identify strictly dominated routing strategies.

---

## ⚙️ Environment Configuration Reference

The router is configured via environment variables prefixed with `ROUTER_` or programmatically via `RouterConfig`:

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ROUTER_API_KEY` | `str` | `None` | Authentication secret key for gateway endpoints. |
| `ROUTER_PORT` | `int` | `8000` | Gateway HTTP server binding port. |
| `ROUTER_HOST` | `str` | `"0.0.0.0"` | Gateway HTTP server network host binding. |
| `ROUTER_MAX_PROMPT_CHARS` | `int` | `10000` | Maximum character length allowed for incoming user prompts before HTTP 413. |
| `ROUTER_RATE_LIMIT_CAPACITY`| `float`| `100.0` | Token bucket maximum burst capacity per API key. |
| `ROUTER_RATE_LIMIT_REFILL_RATE`| `float`| `10.0` | Token refill rate per second per API key. |
| `ROUTER_COST_WEIGHT` | `float` | `0.5` | Weight ($\lambda$) balancing normalized cost vs. accuracy. |
| `ROUTER_FISHER_ALPHA` | `float` | `0.1` | Penalty multiplier applied to models deemed non-competent by Fisher gating. |

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/vfcarida/mechanistic-llm-router.git
cd mechanistic-llm-router

# Create and activate virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Running Causal Probe Routing

```python
import asyncio
from pathlib import Path
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.linear_probe import LinearActivationProbe
from mechanistic_router.routers.causal_probe import CausalProbeRouter
from mechanistic_router.schemas.routing import RoutingRequest


async def main():
    # Load trained probe weights securely from compressed NumPy archive (.npz)
    probe = LinearActivationProbe.load(".cache/checkpoints/causal_probe_demo.npz")
    # Alternatively use CausalProbeRouter.from_saved_probe(...)

    # Run demonstration script
    # python examples/causal_probe_routing.py


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🧪 Testing & Quality Assurance

Every pull request and commit is validated against rigorous offline testing and quality gates:

```bash
# Run complete test suite (103 passed, 100% pass rate)
pytest

# Static type analysis across all 42 source files
mypy src/

# Code quality and formatting checks
ruff check .
ruff format --check .
```

- **Pytest Suite**: **103 passed, 0 failures (100% pass rate)**.
- **Label Leakage Protection**: Source-tree AST scans enforce zero caller-supplied ground-truth label leakage into `RoutingRequest`.
- **Concurrency Isolation**: Verified per-key locking under simultaneous `asyncio.gather` execution.
- **Ruff & MyPy**: 0 lint errors, 0 type errors across all modules.

---

## 🔮 How to Reproduce Real Numbers (Eval Harness — Future Work)

To graduate from synthetic label-driven simulation to real, empirical mechanistic routing:
1. **Unsupervised Complexity Probing**: Replacing `TaskComplexity` ground truth inputs with genuine intrinsic activation metrics (e.g. true unpooled singular value entropy or learned probing classifiers on real open-weight models).
2. **Empirical Evaluation Harness (MLR-T05)**: Testing router decisions against real API endpoints (or deterministic offline cache matrices) across open benchmarks (e.g., MMLU, GSM8K, LMSYS Chatbot Arena) to measure genuine cost reduction, latency trade-offs, and downstream response quality.

---

<div align="center">
  <small>Research and Demonstration Prototype for Mechanistic Interpretability Routing Concepts.</small>
</div>
