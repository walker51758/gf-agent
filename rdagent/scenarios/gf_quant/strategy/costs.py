"""Trusted execution-cost contract for every GF portfolio backtest.

Transaction costs are a system fact, not a strategy parameter. Neither an LLM
proposal, a persisted candidate, settings, environment variables nor the CLI
can override these values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

BUY_COST_BPS = 10.0
SELL_COST_BPS = 10.0
ROUND_TRIP_COST_BPS = BUY_COST_BPS + SELL_COST_BPS

BUY_COST_RATE = BUY_COST_BPS * 1e-4
SELL_COST_RATE = SELL_COST_BPS * 1e-4

COST_TOLERANCE_BPS = 1e-9


def execution_costs_request() -> dict[str, float]:
    """Top-level remote request contract verified by the standalone worker."""

    return {"buy_bps": BUY_COST_BPS, "sell_bps": SELL_COST_BPS}


def execution_costs_metrics() -> dict[str, float]:
    """Auditable cost facts stored with metrics, registries and reports."""

    return {
        "buy_cost_bps": BUY_COST_BPS,
        "sell_cost_bps": SELL_COST_BPS,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    }


def has_fixed_execution_costs(metrics: Mapping[str, Any]) -> bool:
    """Return whether an evaluation reports the immutable 10/10/20 contract."""

    expected = execution_costs_metrics()
    for key, required in expected.items():
        try:
            actual = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(actual) or abs(actual - required) > COST_TOLERANCE_BPS:
            return False
    return True
