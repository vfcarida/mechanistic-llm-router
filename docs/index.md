# Mechanistic LLM Router

Welcome to the documentation for **Mechanistic LLM Router**, an enterprise-grade AI control plane and data plane that routes, orchestrates, and secures Large Language Model (LLM) traffic using deep prefill activation probing, Sparse Autoencoder (SAE) cognitive circuit analysis, and Encoder-Target Decoupling.

## Core Highlights

- **Encoder-Target Decoupling**: Probe prefill hidden activations $A \in \mathbb{R}^{S \times D}$ before full autoregressive generation.
- **Topological Signals**:
  - **Effective Dimensionality ($d_{eff}$)**: Spectrum entropy of SVD singular values evaluating token trajectory spread.
  - **Fisher Separability ($J$)**: Intra/inter-class cluster variance gating to eliminate incompetent models.
  - **SAE Cognitive Circuits**: Sparse Autoencoder (SAELens) feature extraction to detect mathematical reasoning vs trivial factual retrieval.
- **Universal Provider Dispatch**: LiteLLM integration for OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, Ollama, and vLLM.
- **Drop-in OpenAI REST Gateway**: FastAPI `/v1/chat/completions` server.
- **FinOps Observability**: OpenTelemetry metrics exporter with Grafana & Prometheus monitoring dashboard.
