# RouterBench Evaluation Guide

This guide describes how to benchmark routing strategies against the RouterBench evaluation dataset using `mechanistic-llm-router`'s benchmark harness.

---

## 1. What is RouterBench?

[RouterBench](https://arxiv.org/abs/2403.12031) is an open benchmark for evaluating LLM routing systems, containing over 405,000 inference outcomes across 11 language models and 64 distinct evaluation tasks.

It provides realistic task-level accuracy scores and cost measurements, enabling fair empirical comparisons between routing policies.

---

## 2. Downloading RouterBench

We provide a dedicated download script to fetch and prepare RouterBench data:

```bash
# Download and cache RouterBench data into data/routerbench/
python scripts/download_routerbench.py
```

Optional arguments:
- `--output-dir PATH`: Target output directory (default: `data/routerbench`)
- `--force`: Force re-download even if files already exist locally.

---

## 3. Running the Benchmark

Execute the evaluation benchmark harness comparing all available baseline policies:

```bash
# Run benchmark against RouterBench with default baselines
python scripts/run_benchmark.py --dataset routerbench

# Run comprehensive benchmark across all 9 policies (including CostPerformance, Semantic, CausalProbe)
python scripts/run_benchmark.py --dataset routerbench --all-policies

# Output results in Markdown format
python scripts/run_benchmark.py --dataset routerbench --all-policies --output docs/ROUTERBENCH_REPORT.md
```

### Evaluated Policies

The benchmark harness supports 9 distinct routing policies:
1. **AlwaysCheapPolicy**: Routes every query to the lowest-cost model.
2. **AlwaysStrongPolicy**: Routes every query to the most capable/expensive frontier model.
3. **RandomPolicy**: Randomly selects a model from the pool.
4. **LengthThresholdPolicy**: Routes based on input token length.
5. **LearnedLogisticPolicy**: Supervised logistic regression on TF-IDF representation.
6. **MechanisticPolicy**: Mechanistic router incorporating prefill dimensionality and separability.
7. **CostPerformancePolicy**: Heuristic router optimizing cost-performance tradeoffs and budget caps.
8. **SemanticPolicy**: Embedding centroid-based domain classifier and router.
9. **CausalProbePolicy**: Prefill activation linear probing router using mechanistic representation.

---

## 4. Evaluation Metrics & Statistical Gates

The harness generates:
- **Cost vs. Quality Pareto Front**: Identifies non-dominated routing policies.
- **Pareto Area Under Curve (AUC)**: Holistic measure of routing efficiency.
- **Paired Bootstrap Confidence Intervals (95% CI)**: Vectorized 2D bootstrap testing whether quality gains and cost savings are statistically significant.
- **Go/No-Go Decision Criteria**: Ensures candidate policies strictly outperform cheap baselines without exceeding budget bounds.
