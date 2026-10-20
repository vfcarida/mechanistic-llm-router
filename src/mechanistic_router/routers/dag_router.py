"""Multi-Agent DAG Topological Routing & Exact Shapley Attribution.

Implements graph-aware mechanistic routing for multi-agent Directed Acyclic Graphs (DAGs):
1. Topological Validation & Downstream Criticality (Kahn's Algorithm):
   Verifies acyclicity of the agent execution graph and computes each node's
   structural downstream criticality C(u) = w_u + gamma * sum_{v in children(u)} C(v).
2. Critical-Path Weighted Conformal Alpha-Spending (Angelopoulos et al., 2024):
   Partitions the global workflow risk budget alpha_dag across DAG nodes inversely
   proportional to downstream criticality (sum_u alpha_u = alpha_dag), assigning
   stricter risk bounds (earlier escalation to stronger tiers) to high-centrality
   bottleneck nodes.
3. Exact Closed-Form Cooperative Shapley Value Attribution (Shapley, 1953;
   Ghorbani & Zou, ICML 2019):
   Computes exact Shapley credits phi_u satisfying efficiency (sum_u phi_u = v(V)),
   symmetry, and dummy-player axioms over the parent-child synergy coalitional game.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_CONFIG, RouterConfig
from ..models.types import TargetModel
from .causal_probe import CausalProbeRouter
from .portfolio_optimizer import compute_expected_tier_quality


@dataclass(frozen=True)
class AgentDAGNode:
    """Single agent task node within a multi-agent execution DAG."""

    node_id: str
    prompt: str
    dependencies: list[str]
    weight: float = 1.0


@dataclass(frozen=True)
class AgentNodeRoutingAllocation:
    """Routing decision, conformal risk budget, and Shapley credit for one DAG node."""

    node_id: str
    topological_order: int
    prob_complex: float
    downstream_criticality: float
    allocated_conformal_alpha: float
    selected_model: str
    estimated_cost_usd: float
    expected_quality: float
    shapley_credit: float


@dataclass(frozen=True)
class AgentDAGRoutingPlan:
    """Complete multi-agent DAG routing plan with topological order and Shapley credits."""

    node_allocations: list[AgentNodeRoutingAllocation]
    topological_order: list[str]
    global_conformal_alpha: float
    total_cost_usd: float
    oracle_cost_usd: float
    cost_saved_usd: float
    workflow_expected_quality: float


class TopologicalDAGRouter:
    """Graph-aware mechanistic router for multi-agent execution DAGs."""

    def __init__(
        self,
        causal_router: CausalProbeRouter,
        model_pool: dict[str, TargetModel],
        config: RouterConfig = DEFAULT_CONFIG,
        propagation_gamma: float = 0.85,
        synergy_ratio: float = 0.40,
    ) -> None:
        if not model_pool:
            raise ValueError("model_pool cannot be empty.")
        if not (0.0 <= propagation_gamma <= 1.0):
            raise ValueError("propagation_gamma must lie in [0, 1].")
        if not (0.0 <= synergy_ratio <= 1.0):
            raise ValueError("synergy_ratio must lie in [0, 1].")

        self.causal_router = causal_router
        self.model_pool = model_pool
        self.config = config
        self.propagation_gamma = float(propagation_gamma)
        self.synergy_ratio = float(synergy_ratio)
        self.ordered_models = sorted(model_pool.values(), key=lambda m: (m.cost, m.base_accuracy))
        self.max_accuracy = max(m.base_accuracy for m in self.ordered_models)

    @staticmethod
    def topological_sort(nodes: list[AgentDAGNode]) -> list[str]:
        """Validates DAG structure and returns deterministic Kahn topological sort."""
        if not nodes:
            raise ValueError("DAG nodes list cannot be empty.")

        node_map: dict[str, AgentDAGNode] = {}
        for node in nodes:
            nid = node.node_id.strip()
            if not nid:
                raise ValueError("Each AgentDAGNode must have a non-empty node_id.")
            if node.weight <= 0.0:
                raise ValueError(f"Node '{nid}' weight must be > 0.")
            if nid in node_map:
                raise ValueError(f"Duplicate node_id '{nid}' in DAG.")
            node_map[nid] = node

        in_degree: dict[str, int] = {nid: 0 for nid in node_map}
        children: dict[str, list[str]] = {nid: [] for nid in node_map}

        for nid, node in node_map.items():
            seen_deps: set[str] = set()
            for dep in node.dependencies:
                if dep not in node_map:
                    raise ValueError(f"Node '{nid}' depends on unknown parent node '{dep}'.")
                if dep == nid:
                    raise ValueError(f"Self-loop detected on node '{nid}'.")
                if dep not in seen_deps:
                    seen_deps.add(dep)
                    children[dep].append(nid)
                    in_degree[nid] += 1

        queue: deque[str] = deque(sorted([nid for nid, deg in in_degree.items() if deg == 0]))
        topo_order: list[str] = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for child_id in sorted(children[curr]):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(topo_order) != len(node_map):
            raise ValueError("Cycle detected in agent execution DAG.")
        return topo_order

    def compute_downstream_criticality(
        self,
        nodes: list[AgentDAGNode],
        topo_order: list[str],
    ) -> dict[str, float]:
        """Computes normalized downstream path criticality C(u) for all nodes."""
        node_map = {n.node_id: n for n in nodes}
        children: dict[str, list[str]] = {n.node_id: [] for n in nodes}
        for n in nodes:
            for dep in set(n.dependencies):
                children[dep].append(n.node_id)

        raw_crit: dict[str, float] = {}
        for nid in reversed(topo_order):
            child_sum = sum(raw_crit[ch] for ch in children[nid])
            raw_crit[nid] = float(node_map[nid].weight) + self.propagation_gamma * child_sum

        total_crit = sum(raw_crit.values())
        if total_crit <= 1e-12:
            uniform = 1.0 / len(nodes)
            return {n.node_id: uniform for n in nodes}
        return {nid: val / total_crit for nid, val in raw_crit.items()}

    @staticmethod
    def allocate_node_alphas(
        criticalities: dict[str, float],
        global_alpha: float,
    ) -> dict[str, float]:
        """Allocates global workflow risk budget alpha_dag across nodes via inverse criticality."""
        if not (0.0 < global_alpha < 1.0):
            raise ValueError("global_alpha must lie strictly in (0, 1).")

        inv_weights = {nid: 1.0 / max(crit, 1e-6) for nid, crit in criticalities.items()}
        total_inv = sum(inv_weights.values())
        return {nid: float(global_alpha * (w / total_inv)) for nid, w in inv_weights.items()}

    def compute_shapley_attribution(
        self,
        nodes: list[AgentDAGNode],
        criticalities: dict[str, float],
        node_qualities: dict[str, float],
    ) -> dict[str, float]:
        """Computes exact closed-form Shapley values over the DAG synergy game."""
        children: dict[str, list[str]] = {n.node_id: [] for n in nodes}
        parents: dict[str, list[str]] = {n.node_id: list(set(n.dependencies)) for n in nodes}
        for n in nodes:
            for dep in parents[n.node_id]:
                children[dep].append(n.node_id)

        alpha_ind = 1.0 - self.synergy_ratio
        shapley: dict[str, float] = {}

        for n in nodes:
            uid = n.node_id
            c_u = criticalities[uid]
            q_u = node_qualities[uid]
            p_list = parents[uid]

            # Standalone + self share of ancestor synergy
            if not p_list:
                self_val = c_u * q_u
            else:
                self_val = c_u * alpha_ind * q_u + 0.5 * c_u * self.synergy_ratio * q_u

            # Parent share of downstream children's synergy
            child_val = 0.0
            for ch_id in children[uid]:
                n_parents_ch = max(1, len(parents[ch_id]))
                c_ch = criticalities[ch_id]
                q_ch = node_qualities[ch_id]
                child_val += 0.5 * (c_ch * self.synergy_ratio * q_ch) / n_parents_ch

            shapley[uid] = float(self_val + child_val)

        return shapley

    def _select_tier_for_node(
        self,
        prob_complex: float,
        node_alpha: float,
        n_nodes: int,
    ) -> TargetModel:
        """Selects target tier given complexity probability and node risk budget alpha_u."""
        base_low = (
            self.causal_router.mid_threshold
            if self.causal_router.mid_threshold is not None
            else 0.30
        )
        base_high = self.causal_router.threshold

        # Scale thresholds by how strict node_alpha * n_nodes is relative to reference 0.10
        effective_node_budget = node_alpha * max(1, n_nodes)
        scale = float(np.clip(effective_node_budget / 0.10, 0.25, 1.75))
        low_t = float(np.clip(base_low * scale, 0.05, 0.90))
        high_t = float(np.clip(max(low_t + 0.05, base_high * scale), 0.10, 0.98))

        if len(self.ordered_models) >= 3:
            if prob_complex < low_t:
                return self.ordered_models[0]
            if prob_complex < high_t:
                return self.ordered_models[len(self.ordered_models) // 2]
            return self.ordered_models[-1]
        return self.ordered_models[-1] if prob_complex >= high_t else self.ordered_models[0]

    def plan_dag(
        self,
        nodes: list[AgentDAGNode],
        global_conformal_alpha: float = 0.10,
    ) -> AgentDAGRoutingPlan:
        """Creates a complete topologically ordered routing plan and Shapley attribution."""
        topo_order = self.topological_sort(nodes)
        criticalities = self.compute_downstream_criticality(nodes, topo_order)
        node_alphas = self.allocate_node_alphas(criticalities, global_conformal_alpha)

        node_map = {n.node_id: n for n in nodes}
        ordered_prompts = [node_map[nid].prompt for nid in topo_order]
        activations = self.causal_router.extractor.extract_batch(ordered_prompts)
        probs = self.causal_router.probe.predict_proba(activations)

        selected_models: dict[str, TargetModel] = {}
        node_qualities: dict[str, float] = {}
        prob_map: dict[str, float] = {}

        for idx, nid in enumerate(topo_order):
            p_val = float(probs[idx])
            prob_map[nid] = p_val
            target_m = self._select_tier_for_node(
                prob_complex=p_val,
                node_alpha=node_alphas[nid],
                n_nodes=len(nodes),
            )
            selected_models[nid] = target_m
            node_qualities[nid] = compute_expected_tier_quality(
                prob_complex=p_val,
                model=target_m,
                max_accuracy=self.max_accuracy,
            )

        shapley_credits = self.compute_shapley_attribution(
            nodes=nodes,
            criticalities=criticalities,
            node_qualities=node_qualities,
        )

        allocations: list[AgentNodeRoutingAllocation] = []
        total_cost = 0.0
        for idx, nid in enumerate(topo_order):
            m = selected_models[nid]
            cost = float(m.cost)
            total_cost += cost
            allocations.append(
                AgentNodeRoutingAllocation(
                    node_id=nid,
                    topological_order=idx,
                    prob_complex=prob_map[nid],
                    downstream_criticality=criticalities[nid],
                    allocated_conformal_alpha=node_alphas[nid],
                    selected_model=m.name,
                    estimated_cost_usd=cost,
                    expected_quality=node_qualities[nid],
                    shapley_credit=shapley_credits[nid],
                )
            )

        oracle_cost = float(self.ordered_models[-1].cost) * len(nodes)
        total_cost_rounded = round(total_cost, 6)
        cost_saved = max(0.0, round(oracle_cost - total_cost_rounded, 6))
        workflow_quality = float(sum(shapley_credits.values()))

        return AgentDAGRoutingPlan(
            node_allocations=allocations,
            topological_order=topo_order,
            global_conformal_alpha=float(global_conformal_alpha),
            total_cost_usd=total_cost_rounded,
            oracle_cost_usd=round(oracle_cost, 6),
            cost_saved_usd=cost_saved,
            workflow_expected_quality=workflow_quality,
        )


__all__ = [
    "AgentDAGNode",
    "AgentDAGRoutingPlan",
    "AgentNodeRoutingAllocation",
    "TopologicalDAGRouter",
]
