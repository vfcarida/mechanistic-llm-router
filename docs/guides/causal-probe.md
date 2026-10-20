# Causal Activation Probing & Conformal Cascade Guide

This guide describes how to extract prefill activations from transformer backends, train linear probes, validate directional necessity via counterfactual ablation, and calibrate multi-tier cascades using Conformal Risk Control.

---

## 1. Overview

Traditional LLM routers rely on surface heuristics (such as keyword matching, character length, or black-box text embeddings) to estimate task difficulty.

The **Mechanistic LLM Router** extracts internal latent states from a lightweight encoder during prefill:
- **Prefill Extraction (`PrefillActivationExtractor`)**: Capture internal residual-stream representations $h \in \mathbb{R}^{S \times D}$ from intermediate transformer layers (e.g., `SmolLM-135M`).
- **Mean / Last-Token Pooling**: Condense token representations into a fixed-width vector $z \in \mathbb{R}^D$.
- **Linear Probe (`LinearActivationProbe`)**: Map $z$ to difficulty logits and complexity probabilities $p(y=1 \mid z)$ using pure-NumPy inference and `.npz` persistence.
- **Causal Validation (`CausalAblationHarness`)**: Verify that the probe's decision direction $\hat{w}$ causally drives routing accuracy via directional projection ablation ($z' = z - (z^\top \hat{w})\hat{w}$) compared against $K=50$ random orthogonal subspace controls.
- **Conformal Risk Control (`ConformalRiskController`)**: Calibrate multi-tier thresholds ($\hat{\lambda}_{\text{low}}, \hat{\lambda}_{\text{high}}$) with finite-sample risk guarantees ($\mathbb{E}[\ell] \le \alpha$) and binary Shannon entropy ($H_2(p)$) abstention escalation.

---

## 2. Using `CausalProbeRouter` with Multi-Tier Conformal Calibration

`CausalProbeRouter` (`src/mechanistic_router/routers/causal_probe.py`) implements the `AbstractRouter` interface and supports both binary routing and 3-tier conformal cascades.

```python
import asyncio
from mechanistic_router.models.pool import MODEL_POOL
from mechanistic_router.probing import (
    ConformalRiskController,
    LinearActivationProbe,
    PrefillActivationExtractor,
)
from mechanistic_router.routers.causal_probe import CausalProbeRouter
from mechanistic_router.schemas.routing import RoutingRequest


async def main() -> None:
    # 1. Initialize prefill activation extractor on SmolLM-135M
    extractor = PrefillActivationExtractor(
        model_name="HuggingFaceTB/SmolLM-135M",
        target_layer=-1,
        device="cpu",
    )

    # 2. Initialize or load pre-trained linear probe
    probe = LinearActivationProbe(feature_dim=576)

    # 3. Instantiate multi-tier causal router with uncertainty abstention
    router = CausalProbeRouter(
        extractor=extractor,
        probe=probe,
        model_pool=MODEL_POOL,
        threshold=0.70,
        mid_threshold=0.30,
        max_uncertainty=0.95,
        cost_weight=0.5,
    )

    # 4. Route prompt and inspect probing signals
    request = RoutingRequest(prompt="Prove that the square root of 2 is irrational.")
    decision = await router.route(request)
    sig = decision.signals[decision.selected_model]

    print(f"Selected:    {decision.selected_model}")
    print(f"P(Strong):   {sig.extra_metadata['prob_strong']:.4f}")
    print(f"Uncertainty: {sig.extra_metadata['epistemic_uncertainty']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3. Gateway Integration

The `CausalProbeRouter` can be activated dynamically in the OpenAI-compatible gateway either per-request or globally:

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

## 4. Running the Empirical Spike & Causal Ablation Pipeline

To reproduce the experimental findings reported in `docs/MECHANISTIC_SPIKE_REPORT.md`:

```bash
# Run the mechanistic probe unit and causal ablation test suite
pytest experiments/mechanistic_probe/tests/ -v

# Run the full training and causal ablation pipeline
python experiments/mechanistic_probe/run_spike.py
```

The script performs:
1. Prompt-disjoint partitioning across train, dev, and test splits.
2. Prefill activation extraction across splits using `SmolLM-135M`.
3. Training of a `LinearActivationProbe` predicting task complexity.
4. Counterfactual directional ablation along the probe vector $\hat{w}$ vs. $K=50$ random subspace controls.
5. Paired statistical bootstrap comparison (`1,000` resamples) against `LearnedLogistic` and `AlwaysCheap` baselines.
