<div align="center">
  <h1>⚡ Mechanistic LLM Router</h1>
  <p><em>Research Prototype: Exploratory LLM Routing via Encoder-Target Decoupling & Prefill Probing Simulation</em></p>

  ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
  ![Pydantic](https://img.shields.io/badge/Pydantic-v2-green.svg)
  ![LiteLLM](https://img.shields.io/badge/Dispatcher-LiteLLM-purple.svg)
  ![OpenTelemetry](https://img.shields.io/badge/Observability-OpenTelemetry-orange.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
</div>

---

> [!WARNING]
> **Status: Research/demo artifact.** The router is currently heuristic and simulated: activation quantities (`d_eff`, Fisher J, SAE features) are computed for illustration and do not causally drive routing (see findings in [`docs/BASELINE.md`](docs/BASELINE.md)). Performance numbers below are modeled artifacts of a hand-assigned label distribution and a fictional price table, not measured outcomes.

---

## 📖 Project Scope & Conceptual Exploration

The economic feasibility and latency scaling of multi-model AI architectures depend on intelligent request orchestration. Standard semantic routers rely on shallow text-embedding heuristics or static cost thresholds, often introducing routing latency without directly inspecting internal model dynamics.

The **Mechanistic LLM Router** is an exploratory research prototype investigating **Encoder-Target Decoupling** and mechanistic interpretability concepts for LLM routing. The conceptual vision explores whether inspecting unpooled prefill activation matrices $A \in \mathbb{R}^{S \times D}$, computing Effective Dimensionality ($d_{eff}$) spectrum entropy, evaluating Fisher Discriminant separability ($J$), and extracting Sparse Autoencoder (SAE) circuits can provide topological signals of prompt complexity before full autoregressive generation.

> [!NOTE]
> **Current Implementation Reality**: In this prototype stage, latent activation metrics ($d_{eff}$, Fisher $J$, SAE feature indices) are calculated for mathematical and structural demonstration, but do **not** causally decide the model route. Routing decisions currently branch on hand-assigned task complexity labels (`TaskComplexity`) and synthetic Gaussian distributions. Real, non-circular empirical evaluation is planned for future iterations (see [Real Evaluation Harness](#-how-to-reproduce-real-numbers-eval-harness-future-work)).

---

## 📊 Modeled Financial Illustration (Not Measured Outcomes)

Previous project materials referenced a headline claim of *"78.75% cost reduction while maintaining Oracle-level accuracy (>91%)"*. To ensure scientific honesty and transparency, this section details exactly how this modeled figure is derived and why it represents a synthetic arithmetic model rather than an empirical benchmark:

1. **Synthetic Dataset**: Shipped under [`src/mechanistic_router/data/mock_dataset.py`](src/mechanistic_router/data/mock_dataset.py), the dataset consists of **24 synthetic Portuguese prompt templates** across 6 banking categories (labeled "BERTaú" as an illustrative domain theme). It is **not** a real enterprise or financial banking dataset.
2. **Hand-Assigned Mixture**: Prompts are hand-tagged with ground-truth complexity labels (`ROUTINE`: 55%, `MODERATE`: 30%, `COMPLEX`: 15%).
3. **Fictional Model Pool**: Costs and baseline capabilities are fixed synthetic constants defined in [`src/mechanistic_router/models/pool.py`](src/mechanistic_router/models/pool.py):
   - `SLM-BERTau-Local`: $0.02 / query (ceiling: `ROUTINE`, base accuracy: 0.91)
   - `LLM-Mid-Tier`: $0.25 / query (ceiling: `MODERATE`, base accuracy: 0.88)
   - `LLM-Frontier-Oracle`: $1.50 / query (ceiling: `COMPLEX`, base accuracy: 0.97)
4. **Circular Routing & Grading**: The router reads the hand-assigned `TaskComplexity` ground-truth label directly from `RoutingRequest`, routing routine prompts to the SLM ($0.02), moderate to Mid-Tier ($0.25), and complex to Oracle ($1.50). The benchmark evaluator (`scripts/benchmark_evaluator.py`) then evaluates accuracy against that exact same ceiling threshold.
5. **Arithmetic Derivation**:
   $$\text{Expected Cost} = 0.55 \times \$0.02 + 0.30 \times \$0.25 + 0.15 \times \$1.50 = \$0.011 + \$0.075 + \$0.225 = \$0.311$$
   $$\text{Modeled Cost Delta vs Oracle (\$1.50)} \approx \frac{1.50 - 0.31875}{1.50} = \mathbf{78.75\%}$$

This number illustrates theoretical savings under an idealized routing scenario where every query is classified with 100% precision into predetermined cost tiers; it does **not** demonstrate real-world cost savings on unlabelled production workloads.

---

## 🧠 Conceptual Architecture

```mermaid
graph TD
    A[User Prompt Query] --> B[SharedTrunk Encoder / Prefill Stage]
    B --> C[TransformerLens Hook Manager]

    subgraph Probing Engine (Illustrative / Simulated)
        C --> D["Effective Dimensionality (d_eff)<br><i>SVD & Shannon Entropy over [seq_len, hidden_dim]</i>"]
        C --> E["Fisher Separability (J)<br><i>Intra/Inter-Class Variance (Simulated)</i>"]
        C --> F["SAE Engine (SAELens)<br><i>Sparse Autoencoder Circuit Extraction</i>"]
    end

    D -.-> G{Routing Strategy Pipeline}
    E -.-> G
    F -.-> G
    LBL["TaskComplexity Ground Truth (Cheating Input)"] ==> G

    subgraph Strategy Pattern Pool
        G --> H[CostPerformanceRouter]
        G --> I[SemanticRouter]
        G --> J[MechanisticRouter]
    end

    J --> K[LiteLLM Universal Dispatcher]

    subgraph Target Endpoint Pool (Fictional Costs)
        K --> L((SLM Local - $0.02))
        K --> M((Mid-Tier LLM - $0.25))
        K --> N((Frontier Oracle - $1.50))
    end
```

---

## 🧮 Mathematical Formulations (Theoretical Basis)

The mathematical implementations in [`src/mechanistic_router/signals/math_utils.py`](src/mechanistic_router/signals/math_utils.py) are mathematically sound in isolation, though currently non-causal to the router's final route selection:

### 1. Effective Dimensionality ($d_{eff}$)
Calculates spectrum entropy over singular values of sequence activation matrices $A \in \mathbb{R}^{S \times D}$:
$$\sigma = \text{SVD}(A)$$
$$E_i = \sigma_i^2 \quad \text{and} \quad p_i = \frac{E_i}{\sum_j E_j}$$
$$H = -\sum_{i} p_i \ln(p_i) \implies d_{eff} = \exp(H)$$

### 2. Fisher Separability ($J$)
Measures structural separability between success and failure clusters in latent representation space:
$$J = \frac{1}{D} \sum_{d=1}^{D} \frac{(\mu_{\text{success}, d} - \mu_{\text{failure}, d})^2}{\sigma^2_{\text{success}, d} + \sigma^2_{\text{failure}, d} + \epsilon}$$
*(Note: In the current prototype, success/failure clusters are generated via synthetic Gaussian perturbation conditioned on the prompt's `TaskComplexity` label rather than measured from empirical target model checkpoints).*

### 3. Non-Decreasing Convex Hull (Pareto Optimization)
Constructs a cost-accuracy Pareto frontier mapping average query cost (x-axis) to response accuracy (y-axis) to identify strictly dominated routing strategies.

---

## ⚙️ Environment Configuration Reference

The router prototype is configured via YAML files (`configs/default_config.yaml`) or environment variables prefixed with `ROUTER_`:

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ROUTER_SEED` | `int` | `42` | Random seed for statistical reproducibility. |
| `ROUTER_HIDDEN_DIM` | `int` | `128` | Latent space activation dimension size. |
| `ROUTER_NUM_PREFILL_LAYERS` | `int` | `6` | Number of simulated prefill layers. |
| `ROUTER_LAMBDA_BUDGET` | `float` | `0.68` | Cost-optimality balance weight (0.0 to 1.0). |
| `ROUTER_FISHER_J_THRESHOLD` | `float` | `0.30` | Minimum Fisher J score required to pass Competence Gate. |
| `ROUTER_EMBEDDING_DIM` | `int` | `384` | Vector dimension size for semantic router. |

---

## 🚀 Quickstart & Installation

### Local Python Package Installation

```bash
# Clone the repository
git clone https://github.com/vfcarida/mechanistic-llm-router.git
cd mechanistic-llm-router

# Create and activate virtual environment (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # On Linux/macOS: source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Docker Compose Observability Stack (Optional)

To test the containerized REST gateway alongside Prometheus and Grafana:

```bash
docker-compose up --build -d
```

- **OpenAI REST Gateway Endpoint:** `http://localhost:8000/v1/chat/completions`
- **Prometheus Metrics:** `http://localhost:9090/metrics`
- **Grafana Dashboard:** `http://localhost:3000` (User: `admin`, Pass: `admin`)

---

## 💻 Python Code Snippet Usage

```python
import asyncio
from mechanistic_router.config import DEFAULT_CONFIG
from mechanistic_router.core.encoder import SharedTrunkEncoder
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.models.types import TaskComplexity
from mechanistic_router.routers.mechanistic import MechanisticRouter
from mechanistic_router.schemas.routing import RoutingRequest

async def run_routing_example():
    # Initialize prefill encoder and router
    encoder = SharedTrunkEncoder(DEFAULT_CONFIG)
    router = MechanisticRouter(encoder, MODEL_POOL, DEFAULT_CONFIG)

    # Note: task_complexity is currently passed as an illustrative ground-truth tag
    request = RoutingRequest(
        prompt="Analyze my debt-to-income ratio and project credit score impact.",
        task_complexity=TaskComplexity.COMPLEX
    )

    decision = await router.route(request)
    print(f"Selected Target Route: {decision.selected_model}")
    print(f"Modeled Cost: ${decision.estimated_cost_usd:.4f}")
    print(f"Decision Latency: {decision.latency_ms:.2f} ms")

if __name__ == "__main__":
    asyncio.run(run_routing_example())
```

---

## 🧪 Testing, Quality Assurance, & Baseline Status

Actual executed baseline measurements from [docs/BASELINE.md](docs/BASELINE.md):

```bash
# Run Ruff linter and formatter check
ruff check .

# Run Mypy static type analysis
mypy src/

# Run Pytest suite with line coverage
pytest -q --cov=mechanistic_router --cov-report=term-missing
```

- **Pytest Suite**: **27 passed** in 26.7s (the checked-in `.coverage` and badge claiming 45 tests at 98% was stale; see finding MLR-F12).
- **Line Coverage**: **94%** across 675 statements.
- **Ruff**: 152 diagnostics identified across imports, formatting, and line length.
- **MyPy**: Python 3.10 pin in `pyproject.toml` conflicts with PEP 695 type alias statements in modern numpy stubs.

---

## 🔮 How to Reproduce Real Numbers (Eval Harness — Future Work)

To graduate from synthetic label-driven simulation to real, empirical mechanistic routing:
1. **Unsupervised Complexity Probing**: Replacing `TaskComplexity` ground truth inputs with genuine intrinsic activation metrics (e.g. true unpooled singular value entropy or learned probing classifiers on real open-weight models).
2. **Empirical Evaluation Harness (MLR-T05)**: Testing router decisions against real API endpoints (or deterministic offline cache matrices) across open benchmarks (e.g., MMLU, GSM8K, LMSYS Chatbot Arena) to measure genuine cost reduction, latency trade-offs, and downstream response quality.

---

<div align="center">
  <small>Research and Demonstration Prototype for Mechanistic Interpretability Routing Concepts.</small>
</div>
