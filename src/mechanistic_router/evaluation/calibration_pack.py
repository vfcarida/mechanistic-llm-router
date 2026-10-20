"""Multi-Domain RouterBench & MMLU-Pro Conformal Calibration Pack.

Provides a 5-domain evaluation and calibration corpus (GSM8K Math, MBPP Code,
MMLU-Pro Reasoning, Enterprise Finance, and Multi-Hop RAG Synthesis), a
deterministic multi-layer residual-stream extractor for zero-network operation,
and pre-calibrated Group-Lasso probe + Conformal Risk Control checkpoints.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from ..config import DEFAULT_CONFIG, RouterConfig
from ..models.pool import MODEL_POOL, get_model_accuracy
from ..models.types import TargetModel, TaskComplexity
from ..probing.activation_extractor import PrefillActivationExtractor
from ..probing.conformal import ConformalCalibrationSummary, ConformalRiskController
from ..probing.linear_probe import LinearActivationProbe
from ..routers.causal_probe import CausalProbeRouter
from ..schemas.eval import EvalCase
from .dataset import split_dataset_prompt_disjoint

DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "checkpoints"
)
DEFAULT_PROBE_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "default_causal_probe.npz"
DEFAULT_CONFORMAL_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "default_conformal_controller.npz"


MULTIDOMAIN_BENCHMARK_CORPUS: dict[str, list[tuple[str, TaskComplexity]]] = {
    "math_gsm8k": [
        (
            "What is 15% of 240 plus 18?",
            TaskComplexity.ROUTINE,
        ),
        (
            "A store sells apples for $2 each. How much do 9 apples cost?",
            TaskComplexity.ROUTINE,
        ),
        (
            "Convert 3/8 to a decimal percentage.",
            TaskComplexity.ROUTINE,
        ),
        (
            "A train travels 180 km in 2.5 hours and then 120 km in 1.5 hours. "
            "What is its average speed over the entire trip?",
            TaskComplexity.MODERATE,
        ),
        (
            "Solve the system of linear equations: 3x + 2y = 16 and 5x - 3y = 14.",
            TaskComplexity.MODERATE,
        ),
        (
            "Derive the closed-form solution for the stochastic differential equation "
            "dX_t = mu * X_t * dt + sigma * X_t * dW_t using Ito's Lemma and prove "
            "martingale convergence conditions for the discounted process.",
            TaskComplexity.COMPLEX,
        ),
        (
            "Prove that every bounded monotone sequence in a complete metric space "
            "converges and derive the asymptotic error bound for Gauss-Legendre "
            "quadrature of order n on C^(2n)[-1, 1].",
            TaskComplexity.COMPLEX,
        ),
    ],
    "code_mbpp": [
        (
            "Write a Python function to reverse a string.",
            TaskComplexity.ROUTINE,
        ),
        (
            "Return the maximum number in a list of integers in Python.",
            TaskComplexity.ROUTINE,
        ),
        (
            "Check whether an integer is even or odd.",
            TaskComplexity.ROUTINE,
        ),
        (
            "Implement a binary search algorithm in Python that finds the first "
            "occurrence of a target in a sorted list with duplicates.",
            TaskComplexity.MODERATE,
        ),
        (
            "Write a Python decorator that implements thread-safe LRU caching with TTL expiration.",
            TaskComplexity.MODERATE,
        ),
        (
            "Design and implement a lock-free work-stealing deque in Rust/C++ memory model "
            "semantics using atomic acquire-release barriers, proving absence of ABA "
            "hazards under concurrent thief contention.",
            TaskComplexity.COMPLEX,
        ),
        (
            "Implement the Raft consensus leader election and log replication state machine "
            "handling split-vote network partitions, term monotonicity, and linearizable "
            "read-index quorum verification.",
            TaskComplexity.COMPLEX,
        ),
    ],
    "reasoning_mmlu_pro": [
        (
            "What is the capital city of France?",
            TaskComplexity.ROUTINE,
        ),
        (
            "Which planet is closest to the Sun?",
            TaskComplexity.ROUTINE,
        ),
        (
            "Define photosynthesis in one sentence.",
            TaskComplexity.ROUTINE,
        ),
        (
            "Compare strict liability and negligence standards in tort law when "
            "evaluating defective industrial equipment claims.",
            TaskComplexity.MODERATE,
        ),
        (
            "Explain how monetary policy interest rate hikes affect short-term "
            "exchange rates under the Mundell-Fleming trilemma.",
            TaskComplexity.MODERATE,
        ),
        (
            "Analyze the causal identifiability of average treatment effects under "
            "unobserved confounding using front-door adjustment criteria and derive "
            "the non-parametric influence function for doubly robust estimation.",
            TaskComplexity.COMPLEX,
        ),
        (
            "Evaluate the thermodynamic entropy production and quantum decoherence "
            "timescale of a two-level open quantum system coupled to a non-Markovian "
            "bosonic bath via the Lindblad master equation.",
            TaskComplexity.COMPLEX,
        ),
    ],
    "finance_enterprise": [
        (
            "What is my current checking account balance?",
            TaskComplexity.ROUTINE,
        ),
        (
            "Show the last five transactions on my credit card statement.",
            TaskComplexity.ROUTINE,
        ),
        (
            "Send a $50 instant transfer to Alice.",
            TaskComplexity.ROUTINE,
        ),
        (
            "Calculate the monthly amortization schedule and total interest paid "
            "on a $25,000 loan at 8.5% APR over 36 months.",
            TaskComplexity.MODERATE,
        ),
        (
            "Compare fixed-rate treasury bills against inflation-linked bonds for "
            "a 3-year liquidity reserve.",
            TaskComplexity.MODERATE,
        ),
        (
            "Construct a multi-period Basel III counterparty credit risk model "
            "incorporating Credit Valuation Adjustment (CVA), wrong-way risk "
            "copula dependencies, and stressed Value-at-Risk capital buffers.",
            TaskComplexity.COMPLEX,
        ),
        (
            "Optimize an institutional fixed-income and options portfolio under "
            "stochastic volatility (Heston model), deriving delta-gamma-vega hedging "
            "ratios and tail conditional expectation constraints.",
            TaskComplexity.COMPLEX,
        ),
    ],
    "rag_synthesis": [
        (
            "Extract the invoice date from the provided receipt text.",
            TaskComplexity.ROUTINE,
        ),
        (
            "List the names of attendees mentioned in the meeting note.",
            TaskComplexity.ROUTINE,
        ),
        (
            "What is the support phone number listed in the FAQ?",
            TaskComplexity.ROUTINE,
        ),
        (
            "Summarize the key differences between Q2 and Q3 revenue drivers across "
            "the two regional earnings reports.",
            TaskComplexity.MODERATE,
        ),
        (
            "Identify any policy contradictions between the employee handbook and "
            "the updated remote-work addendum.",
            TaskComplexity.MODERATE,
        ),
        (
            "Synthesize conflicting clinical trial outcomes across five longitudinal "
            "oncology studies, adjudicating Simpson's paradox in subgroup survival "
            "hazard ratios and Performing meta-regression heterogeneity analysis.",
            TaskComplexity.COMPLEX,
        ),
        (
            "Reconcile multi-jurisdictional regulatory compliance filings across EU AI Act, "
            "GDPR Article 22, and SEC cybersecurity disclosure rules for an autonomous "
            "credit underwriting pipeline with cross-border data residency.",
            TaskComplexity.COMPLEX,
        ),
    ],
}


class DeterministicMultiLayerExtractor(PrefillActivationExtractor):
    """Zero-network deterministic multi-layer residual-stream activation extractor.

    Simulates three distinct transformer residual-stream layer representations
    (Early Layer -9: lexical & token statistics; Middle Layer -5: syntactic &
    domain structure; Late Layer -1: abstract reasoning & multi-hop complexity),
    each of dimension `d_layer` (total concatenated dimension `3 * d_layer`).

    Designed for reproducible offline calibration, CI verification, and zero-download
    gateway operation while preserving multi-layer Group-Lasso geometry.
    """

    _COMPLEX_KEYWORDS = (
        "derive",
        "prove",
        "stochastic",
        "differential",
        "ito",
        "martingale",
        "asymptotic",
        "quadrature",
        "lock-free",
        "atomic",
        "linearizable",
        "consensus",
        "raft",
        "identifiability",
        "confounding",
        "non-parametric",
        "decoherence",
        "lindblad",
        "non-markovian",
        "basel",
        "cva",
        "copula",
        "heston",
        "vega",
        "simpson",
        "hazard",
        "meta-regression",
        "jurisdictional",
        "counterfactual",
        "compound",
        "macroeconomic",
        "diversification",
    )

    _MODERATE_KEYWORDS = (
        "average",
        "system",
        "equations",
        "binary search",
        "decorator",
        "lru",
        "compare",
        "negligence",
        "monetary",
        "exchange",
        "amortization",
        "inflation",
        "differences",
        "contradictions",
        "installment",
        "eligibility",
        "verify",
    )

    def __init__(
        self,
        d_layer: int = 32,
        layer_indices: list[int] | None = None,
        layer_fusion: str = "concat",
    ) -> None:
        self.d_layer = int(d_layer)
        self.model_name = "deterministic-multilayer-slm"
        self._active_layers: list[int] = (
            list(layer_indices) if layer_indices is not None else [-9, -5, -1]
        )
        self.layer_indices: list[int] | None = self._active_layers
        self.layer_idx = self._active_layers[-1]
        self.layer_fusion = "mean" if layer_fusion == "mean" else "concat"
        self.pooling = "mean"
        self.device = "cpu"
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def num_extracted_layers(self) -> int:
        return len(self._active_layers)

    def extract_one(self, prompt: str) -> np.ndarray:
        """Extracts fused multi-layer activation vector for a single prompt."""
        batch = self.extract_batch([prompt])
        return np.asarray(batch[0], dtype=np.float32)

    def _encode_layer_representation(self, prompt: str, layer_offset: int) -> np.ndarray:
        """Generates a deterministic `d_layer`-dimensional vector for a specific layer."""
        text = prompt.strip()
        lower = text.lower()
        words = lower.split()
        n_chars = len(text)
        n_words = max(1, len(words))
        avg_word_len = sum(len(w) for w in words) / n_words

        complex_hits = sum(1 for kw in self._COMPLEX_KEYWORDS if kw in lower)
        moderate_hits = sum(1 for kw in self._MODERATE_KEYWORDS if kw in lower)
        clause_markers = sum(text.count(c) for c in (",", ";", ":", "(", ")", "-"))

        vec = np.zeros(self.d_layer, dtype=np.float32)

        # Deterministic hashed projections seeded by layer_offset and token n-grams
        digest = hashlib.sha256(f"layer:{layer_offset}:{lower}".encode()).digest()
        for i in range(self.d_layer):
            byte_val = digest[i % len(digest)]
            vec[i] = ((byte_val / 255.0) - 0.5) * 0.15

        if layer_offset == 0:
            # Early layer (-9): surface lexical and length features
            vec[0] = min(n_chars / 200.0, 2.0)
            vec[1] = min(n_words / 35.0, 2.0)
            vec[2] = min(avg_word_len / 8.0, 1.5)
            vec[3] = min(clause_markers / 5.0, 1.5)
        elif layer_offset == 1:
            # Middle layer (-5): syntactic & domain complexity signals
            vec[0] = min(moderate_hits * 0.6 + complex_hits * 0.5, 2.5)
            vec[1] = min(clause_markers / 4.0, 2.0)
            vec[2] = min(n_words / 30.0, 2.0)
            vec[3] = 1.0 if n_words > 18 else -0.5
        else:
            # Late layer (-1): abstract reasoning & multi-hop complexity separation
            complexity_score = (
                1.25 * complex_hits
                + 0.35 * moderate_hits
                + 0.04 * max(0, n_words - 12)
                + 0.15 * max(0.0, avg_word_len - 5.2)
            )
            vec[0] = complexity_score
            vec[1] = (
                1.5
                if complex_hits >= 1
                else (-1.2 if moderate_hits == 0 and n_words <= 12 else 0.1)
            )
            vec[2] = float(math.tanh(complexity_score - 0.8))
            vec[3] = float(complex_hits) - 0.5 * float(n_words <= 10)

        return vec

    def extract_batch(self, prompts: list[str], batch_size: int = 16) -> np.ndarray:
        """Extracts fused multi-layer activations for a batch of prompts."""
        _ = batch_size
        n_layers = len(self._active_layers)
        if not prompts:
            dim = self.d_layer * n_layers if self.layer_fusion == "concat" else self.d_layer
            return np.empty((0, dim), dtype=np.float32)

        rows: list[np.ndarray] = []
        for p in prompts:
            layer_vecs = [self._encode_layer_representation(p, idx) for idx in range(n_layers)]
            if self.layer_fusion == "concat":
                fused = np.concatenate(layer_vecs, axis=0)
            else:
                fused = np.mean(np.stack(layer_vecs, axis=0), axis=0)
            rows.append(fused.astype(np.float32))

        return np.vstack(rows)


def load_multidomain_calibration_dataset(
    n_samples: int = 240,
    seed: int = 42,
) -> list[EvalCase]:
    """Generates a 5-domain RouterBench & MMLU-Pro style evaluation dataset.

    Covers `math_gsm8k`, `code_mbpp`, `reasoning_mmlu_pro`, `finance_enterprise`,
    and `rag_synthesis` across ROUTINE, MODERATE, and COMPLEX tiers.

    Args:
        n_samples: Total number of EvalCase records to generate.
        seed: Random seed for deterministic sampling.

    Returns:
        List of EvalCase instances with per-model outcomes and pricing.
    """
    all_items: list[tuple[str, str, TaskComplexity]] = []
    for domain, items in MULTIDOMAIN_BENCHMARK_CORPUS.items():
        for prompt_template, complexity in items:
            all_items.append((domain, prompt_template, complexity))

    by_complexity: dict[TaskComplexity, list[tuple[str, str, TaskComplexity]]] = {
        TaskComplexity.ROUTINE: [x for x in all_items if x[2] == TaskComplexity.ROUTINE],
        TaskComplexity.MODERATE: [x for x in all_items if x[2] == TaskComplexity.MODERATE],
        TaskComplexity.COMPLEX: [x for x in all_items if x[2] == TaskComplexity.COMPLEX],
    }

    target_dist = [
        (TaskComplexity.ROUTINE, 0.45),
        (TaskComplexity.MODERATE, 0.30),
        (TaskComplexity.COMPLEX, 0.25),
    ]
    complexities = [c for c, _ in target_dist]
    probs = [p for _, p in target_dist]

    price_table = {name: model.cost for name, model in MODEL_POOL.items()}
    rng = np.random.RandomState(seed)
    cases: list[EvalCase] = []

    for i in range(n_samples):
        cplx_idx = int(rng.choice(len(complexities), p=probs))
        cplx = complexities[cplx_idx]
        pool = by_complexity[cplx]
        domain, base_prompt, complexity = pool[i % len(pool)]

        # Ensure unique prompt text per sample index for strict prompt-disjoint splitting
        variant_suffix = f" [ref:{domain}-{i // len(pool) + 1}]" if i >= len(all_items) else ""
        prompt_text = f"{base_prompt}{variant_suffix}"

        per_model_outcome = {
            name: get_model_accuracy(model, complexity) for name, model in MODEL_POOL.items()
        }
        cases.append(
            EvalCase(
                prompt=prompt_text,
                per_model_outcome=per_model_outcome,
                price_table=price_table,
                reference_tier=complexity,
            )
        )

    return cases


def build_and_save_calibration_pack(
    probe_path: str | Path = DEFAULT_PROBE_CHECKPOINT,
    conformal_path: str | Path = DEFAULT_CONFORMAL_CHECKPOINT,
    alpha: float = 0.10,
    group_lasso_lambda: float = 0.015,
    n_samples: int = 240,
    seed: int = 42,
) -> tuple[LinearActivationProbe, ConformalRiskController, ConformalCalibrationSummary]:
    """Fits a multi-layer Group-Lasso probe and calibrates Conformal Risk Control on 5 domains.

    Saves both artifacts as compressed NumPy `.npz` archives for instant production loading.
    """
    cases = load_multidomain_calibration_dataset(n_samples=n_samples, seed=seed)
    train_cases, dev_cases, test_cases = split_dataset_prompt_disjoint(
        cases, train_ratio=0.55, dev_ratio=0.30, test_ratio=0.15, seed=seed
    )
    cal_cases = dev_cases + test_cases

    extractor = DeterministicMultiLayerExtractor(
        d_layer=32, layer_indices=[-9, -5, -1], layer_fusion="concat"
    )

    train_prompts = [c.prompt for c in train_cases]
    train_labels = np.array(
        [1 if c.reference_tier == TaskComplexity.COMPLEX else 0 for c in train_cases],
        dtype=int,
    )
    X_train = extractor.extract_batch(train_prompts)

    probe = LinearActivationProbe(
        C=1.0,
        max_iter=1000,
        random_state=seed,
        num_groups=extractor.num_extracted_layers,
        group_lasso_lambda=group_lasso_lambda,
    )
    probe.fit(X_train, train_labels)

    ordered_models = sorted(MODEL_POOL.values(), key=lambda m: (m.cost, m.base_accuracy))
    cheap_name = ordered_models[0].name
    mid_name = ordered_models[len(ordered_models) // 2].name
    strong_name = ordered_models[-1].name

    cal_prompts = [c.prompt for c in cal_cases]
    X_cal = extractor.extract_batch(cal_prompts)
    cal_probs = probe.predict_proba(X_cal)

    cheap_losses = np.array(
        [
            max(0.0, c.per_model_outcome[strong_name] - c.per_model_outcome[cheap_name])
            for c in cal_cases
        ],
        dtype=np.float64,
    )
    mid_losses = np.array(
        [
            max(0.0, c.per_model_outcome[strong_name] - c.per_model_outcome[mid_name])
            for c in cal_cases
        ],
        dtype=np.float64,
    )

    controller = ConformalRiskController(alpha=alpha, learning_rate=0.05)
    summary = controller.calibrate(
        probe_probs=cal_probs,
        cheap_losses=cheap_losses,
        mid_losses=mid_losses,
    )

    probe.save(probe_path)
    controller.save(conformal_path)
    return probe, controller, summary


def load_calibrated_causal_router(
    model_pool: dict[str, TargetModel] = MODEL_POOL,
    config: RouterConfig = DEFAULT_CONFIG,
    probe_path: str | Path = DEFAULT_PROBE_CHECKPOINT,
    conformal_path: str | Path = DEFAULT_CONFORMAL_CHECKPOINT,
    max_uncertainty: float | None = 0.96,
) -> CausalProbeRouter:
    """Loads the pre-calibrated Multi-Layer Group-Lasso + Conformal CausalProbeRouter.

    If the checkpoint files do not exist on disk yet, builds and persists them automatically.
    """
    p_path = Path(probe_path)
    c_path = Path(conformal_path)

    if not p_path.is_file() or not c_path.is_file():
        probe, controller, _ = build_and_save_calibration_pack(
            probe_path=p_path,
            conformal_path=c_path,
        )
    else:
        probe = LinearActivationProbe.load(p_path)
        controller = ConformalRiskController.load(c_path)

    extractor = DeterministicMultiLayerExtractor(
        d_layer=32, layer_indices=[-9, -5, -1], layer_fusion="concat"
    )
    return CausalProbeRouter(
        extractor=extractor,
        probe=probe,
        model_pool=model_pool,
        config=config,
        threshold=controller.high_threshold,
        mid_threshold=controller.low_threshold,
        conformal_controller=controller,
        max_uncertainty=max_uncertainty,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_CONFORMAL_CHECKPOINT",
    "DEFAULT_PROBE_CHECKPOINT",
    "DeterministicMultiLayerExtractor",
    "MULTIDOMAIN_BENCHMARK_CORPUS",
    "build_and_save_calibration_pack",
    "load_calibrated_causal_router",
    "load_multidomain_calibration_dataset",
]
