"""Unit tests for RouterConfig settings, aliases, and immutability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mechanistic_router.config import DEFAULT_CONFIG, RouterConfig


def test_router_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default configuration values and documented attributes."""
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    cfg = RouterConfig()
    assert cfg.seed == 42
    assert cfg.hidden_dim == 128
    assert cfg.vocab_size == 10000
    assert cfg.num_prefill_layers == 6
    assert cfg.num_layers == 6
    assert cfg.lambda_budget == pytest.approx(0.68)
    assert cfg.cost_weight == pytest.approx(0.68)
    assert cfg.conformal_alpha == pytest.approx(0.10)
    assert cfg.max_prompt_chars == 10000
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.api_key is None
    assert DEFAULT_CONFIG.seed == 42


def test_router_config_aliases_and_immutability() -> None:
    """Verify cost_weight and num_layers constructor aliases and frozen immutability."""
    cfg = RouterConfig(
        cost_weight=0.75,
        num_layers=4,
        fisher_alpha=0.05,
        max_prompt_chars=4096,
        conformal_alpha=0.05,
    )
    assert cfg.lambda_budget == pytest.approx(0.75)
    assert cfg.cost_weight == pytest.approx(0.75)
    assert cfg.num_prefill_layers == 4
    assert cfg.num_layers == 4
    assert cfg.max_prompt_chars == 4096
    assert cfg.conformal_alpha == pytest.approx(0.05)

    dumped = cfg.model_dump()
    assert dumped["lambda_budget"] == pytest.approx(0.75)
    assert dumped["num_prefill_layers"] == 4

    with pytest.raises(ValidationError):
        cfg.lambda_budget = 0.5  # type: ignore[misc]
