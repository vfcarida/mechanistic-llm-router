"""OpenTelemetry FinOps and Observability Metrics Module."""

import logging

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import start_http_server

logger = logging.getLogger(__name__)


class RouterMetrics:
    """OpenTelemetry instrumentation manager for FinOps cost tracking and performance observability.

    Exposes quantitative metrics:
    - Incremental router inference latency vs raw endpoint execution latency.
    - Delta financial cost tracking (calculated USD saved per query).
    - Statistical distribution of routing traffic pathways over time.
    """

    def __init__(self, service_name: str = "mechanistic-llm-router"):
        resource = Resource.create({SERVICE_NAME: service_name})
        self.reader = PrometheusMetricReader()
        self.provider = MeterProvider(resource=resource, metric_readers=[self.reader])
        metrics.set_meter_provider(self.provider)

        self.meter = metrics.get_meter("mechanistic_router.observability")

        # Metric instruments
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
    ) -> None:
        """Records telemetry data points for a single request execution route.

        Args:
            route_name: Target model selected.
            strategy: Router strategy class name applied.
            router_latency_ms: Router decision latency in ms.
            endpoint_latency_ms: Upstream execution latency in ms.
            cost_saved_usd: Financial savings delta in USD.
        """
        attributes = {"target_model": route_name, "strategy": strategy}

        self.router_latency.record(router_latency_ms, attributes=attributes)
        self.endpoint_latency.record(endpoint_latency_ms, attributes=attributes)
        self.cost_saved_usd.add(cost_saved_usd, attributes=attributes)
        self.traffic_distribution.add(1, attributes=attributes)
