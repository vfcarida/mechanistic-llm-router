"""OpenTelemetry FinOps, GenAI, and Mechanistic Observability Metrics Module."""

import logging
from typing import Any

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import start_http_server

logger = logging.getLogger(__name__)


class RouterMetrics:
    """OpenTelemetry instrumentation manager for FinOps cost tracking and mechanistic observability.

    Exposes quantitative metrics:
    - Incremental router inference latency vs raw endpoint execution latency.
    - Delta financial cost tracking (calculated USD saved per query).
    - Statistical distribution of routing traffic pathways over time.
    - Mechanistic white-box telemetry (`gen_ai.mechanistic.d_eff`, `gen_ai.mechanistic.fisher_j`,
      `gen_ai.mechanistic.epistemic_uncertainty`, and `gen_ai.mechanistic.abstention_escalations`).
    """

    def __init__(self, service_name: str = "mechanistic-llm-router"):
        resource = Resource.create({SERVICE_NAME: service_name})
        self.reader = PrometheusMetricReader()
        self.provider = MeterProvider(resource=resource, metric_readers=[self.reader])
        metrics.set_meter_provider(self.provider)

        self.meter = metrics.get_meter("mechanistic_router.observability")

        # Legacy metric instruments
        self.router_latency = self.meter.create_histogram(
            name="router.inference.latency_ms",
            description="Incremental router decision-making latency in milliseconds.",
            unit="ms",
        )
        self.endpoint_latency = self.meter.create_histogram(
            name="llm.endpoint.latency_ms",
            description="Raw upstream LLM endpoint execution latency in milliseconds.",
            unit="ms",
        )
        self.cost_saved_usd = self.meter.create_counter(
            name="finops.cost_saved_usd",
            description="Cumulative financial USD saved by diverting traffic from frontier oracle.",
            unit="USD",
        )
        self.traffic_distribution = self.meter.create_counter(
            name="router.traffic.distribution",
            description="Total requests routed per target model candidate.",
            unit="1",
        )

        # OpenTelemetry Standard GenAI Semantic Convention Instruments
        self.genai_routing_duration = self.meter.create_histogram(
            name="gen_ai.routing.duration",
            description="Duration of router decision-making in milliseconds.",
            unit="ms",
        )
        self.genai_client_operation_duration = self.meter.create_histogram(
            name="gen_ai.client.operation.duration",
            description="Raw upstream LLM generation latency in milliseconds.",
            unit="ms",
        )
        self.genai_cost_saved = self.meter.create_counter(
            name="gen_ai.cost.saved",
            description="Cumulative financial USD saved versus frontier oracle baseline.",
            unit="USD",
        )
        self.genai_routing_requests = self.meter.create_counter(
            name="gen_ai.routing.requests",
            description="Total number of routing requests partitioned by candidate model.",
            unit="1",
        )

        # OpenTelemetry Mechanistic White-Box Semantic Convention Instruments
        self.genai_mechanistic_d_eff = self.meter.create_histogram(
            name="gen_ai.mechanistic.d_eff",
            description="Effective dimensionality (d_eff) of residual-stream representation.",
            unit="1",
        )
        self.genai_mechanistic_fisher_j = self.meter.create_histogram(
            name="gen_ai.mechanistic.fisher_j",
            description="Normalized Fisher Information sensitivity / complexity score.",
            unit="1",
        )
        self.genai_mechanistic_epistemic_uncertainty = self.meter.create_histogram(
            name="gen_ai.mechanistic.epistemic_uncertainty",
            description="Normalized binary Shannon entropy H_2(p) of routing boundary.",
            unit="1",
        )
        self.genai_mechanistic_abstention_escalations = self.meter.create_counter(
            name="gen_ai.mechanistic.abstention_escalations",
            description="Total uncertainty-triggered conformal abstention escalations.",
            unit="1",
        )

        self._exporter_started = False

    def start_exporter(self, port: int = 9090) -> None:
        """Starts Prometheus HTTP metrics server on specified port."""
        if not self._exporter_started:
            try:
                start_http_server(port=port)
                self._exporter_started = True
            except Exception as exc:
                logger.warning(
                    "Prometheus metrics exporter failed to bind on port %d: %s. "
                    "Skipping exporter startup (normal if already running or in test).",
                    port,
                    exc,
                )

    def record_route(
        self,
        route_name: str,
        strategy: str,
        router_latency_ms: float,
        endpoint_latency_ms: float,
        cost_saved_usd: float,
        signals: dict[str, Any] | None = None,
    ) -> None:
        """Records telemetry data points for a single request execution route.

        Args:
            route_name: Target model selected.
            strategy: Router strategy class name applied.
            router_latency_ms: Router decision latency in ms.
            endpoint_latency_ms: Upstream execution latency in ms.
            cost_saved_usd: Financial savings delta in USD.
            signals: Optional dictionary of per-model ProbingSignals for mechanistic telemetry.
        """
        # Legacy attributes
        attributes = {"target_model": route_name, "strategy": strategy}

        self.router_latency.record(router_latency_ms, attributes=attributes)
        self.endpoint_latency.record(endpoint_latency_ms, attributes=attributes)
        self.cost_saved_usd.add(cost_saved_usd, attributes=attributes)
        self.traffic_distribution.add(1, attributes=attributes)

        # OpenTelemetry GenAI Semantic Convention attributes
        genai_attributes = {
            "gen_ai.response.model": route_name,
            "gen_ai.routing.strategy": strategy,
        }
        self.genai_routing_duration.record(router_latency_ms, attributes=genai_attributes)
        self.genai_client_operation_duration.record(
            endpoint_latency_ms, attributes=genai_attributes
        )
        self.genai_cost_saved.add(cost_saved_usd, attributes=genai_attributes)
        self.genai_routing_requests.add(1, attributes=genai_attributes)

        # Record Mechanistic White-Box Telemetry when probing signals are available
        if signals:
            selected_sig = signals.get(route_name) or next(iter(signals.values()), None)
            if selected_sig is not None:
                d_eff_val = float(getattr(selected_sig, "d_eff_mean", 0.0))
                fisher_val = float(getattr(selected_sig, "fisher_j_norm", 0.0))
                self.genai_mechanistic_d_eff.record(d_eff_val, attributes=genai_attributes)
                self.genai_mechanistic_fisher_j.record(fisher_val, attributes=genai_attributes)

                extra = getattr(selected_sig, "extra_metadata", None) or {}
                if "epistemic_uncertainty" in extra and extra["epistemic_uncertainty"] is not None:
                    self.genai_mechanistic_epistemic_uncertainty.record(
                        float(extra["epistemic_uncertainty"]),
                        attributes=genai_attributes,
                    )
                if extra.get("abstention_escalated"):
                    self.genai_mechanistic_abstention_escalations.add(
                        1, attributes=genai_attributes
                    )
