"""Strategy candidate specification for the GF portfolio layer.

A strategy candidate is a **specification**, not generated code: it names the
accepted factors or the selection rule it relies on, and how the resulting
cross-sectional score is turned into a portfolio.  Keeping the candidate
declarative is what makes the strategy loop auditable -- the LLM chooses rules
and parameters while the portfolio construction and the backtest stay
deterministic framework code that can be unit tested once.

This dataclass is deliberately independent of
:class:`rdagent.scenarios.gf_quant.service.FactorCandidate`.  Factor fields
(expression, family, benchmark percentiles) do not transfer to a portfolio, and
mixing them would invite reusing single-factor IC thresholds on strategies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

WeightingScheme = Literal["equal", "rank", "proportional"]

WEIGHTING_SCHEMES: tuple[str, ...] = ("equal", "rank", "proportional")


@dataclass(frozen=True)
class CompositionSpec:
    """Turn a cross-sectional score panel into target portfolio weights.

    ``top_n`` is the number of holdings selected per rebalance date,
    ``weighting`` picks the allocation scheme and ``weight_cap`` optionally
    limits a single holding (the CSI free-cash-flow style indices use this).
    """

    top_n: int
    weighting: WeightingScheme = "equal"
    weight_cap: float | None = None

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.weighting not in WEIGHTING_SCHEMES:
            raise ValueError(f"weighting must be one of {WEIGHTING_SCHEMES}, got {self.weighting!r}")
        if self.weight_cap is not None and not 0 < self.weight_cap <= 1:
            raise ValueError("weight_cap must be in (0, 1] when provided")

    def as_payload(self) -> dict[str, Any]:
        """Canonical JSON-serialisable form; also feeds ``strategy_id_for``."""

        return {"top_n": self.top_n, "weighting": self.weighting, "weight_cap": self.weight_cap}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CompositionSpec":
        return cls(
            top_n=int(payload["top_n"]),
            weighting=str(payload.get("weighting", "equal")),
            weight_cap=None if payload.get("weight_cap") is None else float(payload["weight_cap"]),
        )


@dataclass(frozen=True)
class BacktestSpec:
    """Execution assumptions of a strategy backtest.

    ``signal_lag`` is the number of trading days between a decision date and
    the first day the resulting weights can be traded.  ``1`` is the minimum on
    purpose: ``0`` would mean trading on a signal computed with the same day's
    prices, i.e. look-ahead.  ``cost_bps`` is charged per unit of one-way
    turnover (a round trip therefore costs twice as much).
    """

    signal_lag: int = 1
    cost_bps: float = 0.0
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.signal_lag < 1:
            raise ValueError("signal_lag must be at least 1; 0 would be look-ahead")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")

    def as_payload(self) -> dict[str, Any]:
        return {
            "signal_lag": self.signal_lag,
            "cost_bps": self.cost_bps,
            "trading_days_per_year": self.trading_days_per_year,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BacktestSpec":
        return cls(
            signal_lag=int(payload.get("signal_lag", 1)),
            cost_bps=float(payload.get("cost_bps", 0.0)),
            trading_days_per_year=int(payload.get("trading_days_per_year", 252)),
        )


@dataclass(frozen=True)
class StrategyCandidate:
    """A portfolio strategy to be evaluated and registered.

    Either ``factor_names`` (references into the factor registry) or ``rule``
    (a declarative selection rule such as the free-cash-flow-rate ranking) must
    be given; both may be present when a rule is applied on top of factor
    scores.  Neither is interpreted here -- the composition layer receives the
    resulting score panel, so this class never touches market data.
    """

    name: str
    universe: str
    composition: CompositionSpec
    factor_names: tuple[str, ...] = ()
    rule: str = ""
    hypothesis: str = ""
    backtest: BacktestSpec = field(default_factory=BacktestSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Strategy candidate requires a name")
        if not self.universe:
            raise ValueError("Strategy candidate requires a universe")
        if not self.factor_names and not self.rule:
            raise ValueError("Strategy candidate needs factor_names or a rule")
        if any(not name for name in self.factor_names):
            raise ValueError("factor_names must not contain empty names")

    def as_payload(self) -> dict[str, Any]:
        """Canonical payload used for identity hashing and registry storage."""

        return {
            "name": self.name,
            "universe": self.universe,
            "composition": self.composition.as_payload(),
            "factor_names": list(self.factor_names),
            "rule": self.rule,
            "backtest": self.backtest.as_payload(),
        }

    def canonical_payload(self) -> str:
        return json.dumps(self.as_payload(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StrategyCandidate":
        return cls(
            name=str(payload["name"]),
            universe=str(payload["universe"]),
            composition=CompositionSpec.from_payload(dict(payload["composition"])),
            factor_names=tuple(str(value) for value in payload.get("factor_names", [])),
            rule=str(payload.get("rule", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            backtest=BacktestSpec.from_payload(dict(payload.get("backtest", {}))),
            metadata=dict(payload.get("metadata", {})),
        )
