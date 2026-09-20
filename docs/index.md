# Mechanistic LLM Router

> [!WARNING]
> **Status: Research/demo artifact.** The router is currently heuristic and simulated: activation quantities (`d_eff`, Fisher J, SAE features) are computed for illustration and do not causally drive routing (see findings in [`BASELINE.md`](BASELINE.md)). Performance numbers in associated materials are modeled artifacts of a hand-assigned label distribution and a fictional price table, not measured outcomes.

Welcome to the documentation for **Mechanistic LLM Router**, a research prototype and exploratory testbed investigating multi-model orchestration via **Encoder-Target Decoupling**, prefill activation inspection, and Sparse Autoencoder (SAE) cognitive circuit analysis concepts.

## Core Architectural Concepts

- **Encoder-Target Decoupling**: Exploring whether unpooled prefill activations $A \in \mathbb{R}^{S \times D}$ from a lightweight shared encoder can predict prompt difficulty prior to downstream model invocation.
- **Topological Signals (Theoretical & Simulated)**:
  - **Effective Dimensionality ($d_{eff}$)**: Spectrum entropy of SVD singular values evaluating representation dispersion across sequence positions.
  - **Fisher Separability ($J$)**: Intra/inter-class variance gating intended to assess candidate model competence (currently simulated via Gaussian perturbations conditioned on ground-truth task labels).
  - **SAE Cognitive Circuits**: Structural integration with Sparse Autoencoder feature extractors.
- **Universal Provider Dispatcher**: LiteLLM wrapper abstraction for provider failover and mock routing execution.
- **Drop-in OpenAI REST Gateway**: FastAPI `/v1/chat/completions` server compatible with standard OpenAI client libraries.
- **FinOps Observability**: OpenTelemetry metrics instrumentation exporting router latency deltas and modeled financial savings to Prometheus and Grafana.

For details on baseline quality gates and label leakage inventory, see the [Baseline Record](BASELINE.md). For deep system specification, see [Architecture](architecture.md).
