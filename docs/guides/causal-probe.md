# Causal Activation Probing Guide

This guide describes how to extract prefill activations from real transformer backends and train linear probes to guide routing decisions causally.

---

## 1. Overview

Traditional LLM routers rely on superficial surface heuristics (such as keyword matching, character length, or embedding distances) to guess task difficulty.

The **Mechanistic LLM Router** explores extracting internal latent states from a lightweight encoder during prefill:
- **Prefill Extraction**: Capture internal residual-stream representations $h \in \mathbb{R}^{S \times D}$ from intermediate transformer layers (e.g. `SmolLM-135M` or `Qwen2.5-0.5B`).
- **Mean / Last-Token Pooling**: Condense token representations into a fixed-width vector $z \in \mathbb{R}^D$.
- **Linear Probe**: Map $z$ to difficulty logits or model suitability probabilities.
- **Causal Validation**: Verify that the probe's decision direction causally affects the downstream model routing via activation ablation (projecting out the probe direction) vs. random direction controls.

---

## 2. Using `CausalProbeRouter`

The `CausalProbeRouter` is part of the library (`src/mechanistic_router/routers/causal_probe.py`) and is interoperable with the standard `AbstractRouter` interface.

```python
import asyncio
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing.activation_extractor import (
    LinearActivationProbe,
    PrefillActivationExtractor,
)
from mechanistic_router.routers.causal_probe import CausalProbeRouter
from mechanistic_router.schemas.routing import RoutingRequest


async def main():
    # Initialize real prefill activation extractor on SmolLM-135M
    extractor = PrefillActivationExtractor(
        model_name="HuggingFaceTB/SmolLM-135M",
        target_layer=15,  # mid-to-late transformer layer
        device="cpu",
    )

    # Initialize or load pre-trained linear probe
    probe = LinearActivationProbe(feature_dim=576)

    # Instantiate causal router
    router = CausalProbeRouter(
        extractor=extractor,
        probe=probe,
        model_pool=MODEL_POOL,
        cost_weight=0.5,
    )

    # Route real prompt
    request = RoutingRequest(prompt="Prove that the square root of 2 is irrational.")
    decision = await router.route(request)

    print(f"Selected: {decision.selected_model}")
    print(f"Confidence: {decision.probing_signals.probe_confidence:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3. Gateway Integration

The `CausalProbeRouter` can be activated dynamically in the production OpenAI-compatible gateway either per-request or globally:

### Via Request Header
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ROUTER_API_KEY" \
  -H "X-Router-Strategy: causal-probe" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]
  }'
```

### Via Virtual Model Identifier
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "causal-probe-auto",
    "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]
  }'
```

---

## 4. Running the Empirical Spike Experiment

To reproduce the experimental findings reported in `docs/MECHANISTIC_SPIKE_REPORT.md`:

```bash
# Run the mechanistic probe test suite
pytest experiments/mechanistic_probe/tests/ -v

# Run the full training and causal ablation pipeline
python experiments/mechanistic_probe/scripts/run_causal_spike.py
```

The script performs:
1. Prefill activation extraction across train and test prompt splits.
2. Training of a `LinearActivationProbe` predicting task complexity.
3. Activation ablation along the probe vector $w$.
4. Paired statistical bootstrap comparison against $K=50$ random subspace controls.
