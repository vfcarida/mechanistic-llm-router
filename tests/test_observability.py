"""Unit Tests for OpenTelemetry Observability Metrics."""

import pytest
from mechanistic_router.observability.metrics import RouterMetrics


def test_router_metrics_record() -> None:
    """Test recording metric measurements."""
    metrics_mgr = RouterMetrics(service_name="test-router-service")

    # Record telemetry event
    metrics_mgr.record_route(
        route_name="SLM-BERTau-Local",
        strategy="MechanisticRouter",
        router_latency_ms=1.2,
        endpoint_latency_ms=45.0,
        cost_saved_usd=1.48,
    )

    assert metrics_mgr._exporter_started is False
    metrics_mgr.start_exporter(port=9099)
    assert metrics_mgr._exporter_started is True
