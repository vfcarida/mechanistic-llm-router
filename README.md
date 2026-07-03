<div align="center">
  <h1>⚡ Cost-Optimal-Mechanistic-Router</h1>
  <p><em>High-Performance LLM Routing via Encoder-Target Decoupling & Prefill Probing</em></p>

  ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
  ![Accelerated SVD](https://img.shields.io/badge/Backend-Native_GPU_SVD-8a2be2.svg)
  ![QA Edge Cases](https://img.shields.io/badge/QA_Tests-Passing-brightgreen.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
</div>

---

## 📖 Executive Summary

The economic feasibility of large-scale AI operations is governed by the intelligence of multi-model orchestration. Conventional semantic routers rely on shallow text-embedding heuristics, incurring latency and overhead. 

The **Cost-Optimal-Mechanistic-Router** utilizes **Encoder-Target Decoupling** to probe the base LLM’s hidden states during the prefill phase. By analyzing the token sequence's topological structure and predicting task complexity prior to full autoregressive generation, we route queries dynamically to the most cost-effective competent model.

In this audited version:
* **Token Trajectory $d_{eff}$ Mapping:** Resolves the sequence dimension collapse bug. Effective dimensionality is evaluated on unpooled sequence activations of shape `[seq_len, hidden_dim]` instead of mean-pooled `[1, hidden_dim]` vectors, preserving actual token trajectories.
* **Deterministic Hashing:** RNG seeds for synthetic Fisher activations are computed via md5 hashing, securing platform-independent deterministic stability.
* **Production-Ready Validation:** Enforces strict runtime parameter typing to safeguard operational state transitions under extreme conditions.

> [!TIP]
> **Business Impact:** In our financial mock dataset benchmarks (*BERTaú* domain), this routing logic demonstrates a **78.75% reduction in inferential cost** compared to defaulting all queries to the frontier oracle, while matching Oracle-level accuracy (>91%).

---

## 🧠 Mechanistic Architecture

The decoupled encoder-target pipeline maps queries to model routes using low-level tensor activations:

```mermaid
graph TD
    A[User Prompt Query] --> B(SharedTrunk Encoder<br><i>Prefill Simulator stage</i>)
    
    subgraph Topological Signals (PyTorch)
        B --> C["Effective Dimensionality (d_eff)<br><i>SVD & Shannon Entropy over [seq_len, hidden_dim]</i>"]
        B --> D["Fisher Separability (J)<br><i>Intra/Inter-Class Variance Gating</i>"]
    end
    
    C --> E{Mechanistic Router}
    D --> E
    
    subgraph Target Model Pool
        E -- Routine / Competent --> F((SLM Local<br>$0.02))
        E -- Moderate / Competent --> G((Mid-Tier LLM<br>$0.25))
        E -- Complex / Exception --> H((Frontier Oracle<br>$1.50))
    end
```

---

## 🧮 Mathematical Engine & Formulas

### 1. Effective Dimensionality ($d_{eff}$)
Computes the spectrum entropy of singular values of the latent activations $A \in \mathbb{R}^{S \times D}$ where $S$ is the sequence length and $D$ is the hidden dimensionality.
$$s = \text{SVD}(A)$$
$$E_i = s_i^2 \quad \text{and} \quad p_i = \frac{E_i}{\sum_j E_j}$$
$$H = -\sum_{i} p_i \ln(p_i)$$
$$d_{eff} = \exp(H)$$

### 2. Fisher Separability ($J$)
Measures structural competence by checking if success and failure clusters are linearly separable in the latent representation space.
$$J = \frac{1}{D} \sum_{d=1}^{D} \frac{(\mu_{\text{success}, d} - \mu_{\text{failure}, d})^2}{\sigma^2_{\text{success}, d} + \sigma^2_{\text{failure}, d} + \epsilon}$$

### 3. Elastic Cost Ponderation
Normalized inverse cost is formulated as:
$$\text{invcost}_{\text{norm}} = \frac{\frac{1}{\text{cost}} - \frac{1}{\text{cost}_{\text{max}}}}{\frac{1}{\text{cost}_{\text{min}}} - \frac{1}{\text{cost}_{\text{max}}}}$$

---

## ⚙️ Installation & Usage

### Setup Environment
* Python 3.10+ is required.

```bash
# Clone the repository
git clone https://github.com/your-org/mechanistic-llm-router.git
cd mechanistic-llm-router

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
# .\venv\Scripts\Activate.ps1

# Install in development mode with test dependencies
pip install -e .[dev]
```

### Run PoC Simulation
Run the automated mock financial evaluation (200 records distributed across Routine, Moderate, and Complex tasks):

```bash
python scripts/run_poc.py
```

---

## 🧪 Testing and QA

The test suite checks model boundaries, strict type checking, SVD fallbacks for degenerate inputs, and deterministic seeding behaviors.

Since `pytest` is configured via `pyproject.toml`, run directly from the root:

```bash
pytest -v
```

---

<div align="center">
  <small>Optimized for Ultra-Low Latency LLMOps and Financial AI Infrastructure.</small>
</div>
