"""Evaluate, gate, de-duplicate and register GF portfolio strategies.

Mirrors :mod:`rdagent.scenarios.gf_quant.service` but for the strategy layer:

* the gate is a **portfolio** gate (cost, lag, holdings, turnover, risk) and is
  fail-closed -- a missing yearly breakdown or a zero signal lag is a failure,
  not a skipped check;
* strategy redundancy reuses :class:`~rdagent.scenarios.gf_quant.dedup.FactorDeduplicator`
  with the strategy's daily return profile in the ``return_profile`` slot, so
  the correlation threshold and minimum-overlap semantics stay identical to the
  factor layer and no second dedup implementation can drift;
* the calibrated-standards layer is intentionally absent: strategy thresholds
  are only calibrated once a benchmark strategy pool exists (design doc,
  batch 3), so the explicit policy below is the single source of truth.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rdagent.scenarios.gf_quant.dedup import DedupResult, FactorDeduplicator, FactorSignature
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.strategy.backtest import PortfolioResult
from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRecord, StrategyRegistry, strategy_id_for

PROFILE_KEY = "return_profile"
"""Registry metadata key holding the compact daily return trace used for dedup."""


@dataclass(frozen=True)
class StrategyAcceptancePolicy:
    """Explicit strategy gate.

    The defaults are development-grade, not calibrated; they exist so a missing
    threshold never silently means "accept anything".
    """

    min_signal_lag: int = 1
    require_cost: bool = True
    min_observations: int = 100
    min_avg_holdings: float = 5.0
    min_sharpe: float = 0.30
    min_annualized_return: float = 0.0
    min_excess_annualized_return: float | None = None
    max_drawdown_abs: float = 0.50
    max_annual_turnover: float = 12.0
    min_valid_years: int = 2
    min_positive_return_years: int = 0

    @staticmethod
    def _finite(metrics: Mapping[str, Any], name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def annual_requirements(self, expected_years: int | None = None) -> dict[str, int]:
        """Resolve stability requirements for the bound evaluation window.

        A rolling outer epoch may intentionally evaluate only one calendar
        year. Such an epoch cannot provide two valid or positive years, so the
        configured multi-year requirements are capped by the number of years
        the window can actually contain. Missing years still fail closed:
        ``expected_years`` comes from the requested window, not from the worker
        payload.
        """

        if expected_years is not None and expected_years < 1:
            raise ValueError("expected_years must be positive")
        return {
            "min_valid_years": min(self.min_valid_years, expected_years)
            if expected_years is not None
            else self.min_valid_years,
            "min_positive_return_years": min(self.min_positive_return_years, expected_years)
            if expected_years is not None
            else self.min_positive_return_years,
        }

    def _annual_failures(
        self,
        metrics: Mapping[str, Any],
        *,
        expected_years: int | None = None,
    ) -> list[str]:
        by_year = metrics.get("metrics_by_year")
        if not isinstance(by_year, Mapping) or not by_year:
            # Fail closed: without a yearly breakdown the stability of the
            # strategy cannot be judged at all.
            return ["strategy_metrics_by_year_missing"]
        valid = 0
        positive = 0
        for values in by_year.values():
            if not isinstance(values, Mapping):
                continue
            if self._finite(values, "sharpe") is not None and self._finite(values, "annualized_return") is not None:
                valid += 1
                if self._finite(values, "annualized_return") > 0:
                    positive += 1
        requirements = self.annual_requirements(expected_years)
        if valid < requirements["min_valid_years"]:
            return ["strategy_valid_years"]
        if positive < requirements["min_positive_return_years"]:
            return ["strategy_positive_return_years"]
        return []

    def check(
        self,
        metrics: Mapping[str, Any],
        candidate: StrategyCandidate,
        *,
        expected_years: int | None = None,
    ) -> tuple[bool, list[str]]:
        checks: list[tuple[str, float | None, Callable[[float], bool]]] = [
            (
                "strategy_signal_lag",
                None if candidate.backtest.signal_lag < self.min_signal_lag else candidate.backtest.signal_lag,
                lambda value: value >= self.min_signal_lag,
            ),
            (
                "strategy_observations",
                self._finite(metrics, "observations"),
                lambda value: value >= self.min_observations,
            ),
            (
                "strategy_holdings",
                self._finite(metrics, "avg_holdings"),
                lambda value: value >= self.min_avg_holdings,
            ),
            (
                "annualized_return",
                self._finite(metrics, "annualized_return"),
                lambda value: value >= self.min_annualized_return,
            ),
            ("sharpe", self._finite(metrics, "sharpe"), lambda value: value >= self.min_sharpe),
            ("max_drawdown", self._finite(metrics, "max_drawdown"), lambda value: abs(value) <= self.max_drawdown_abs),
            (
                "annual_turnover",
                self._finite(metrics, "annual_turnover"),
                lambda value: value <= self.max_annual_turnover,
            ),
        ]
        if self.min_excess_annualized_return is not None:
            checks.append(
                (
                    "excess_annualized_return",
                    self._finite(metrics, "excess_annualized_return"),
                    lambda value: value >= self.min_excess_annualized_return,
                )
            )
        if self.require_cost:
            # cost_bps == 0 means "no cost model at all", which is a modelling
            # omission rather than a cheap strategy.
            checks.append(
                (
                    "strategy_cost_missing",
                    None if candidate.backtest.cost_bps <= 0 else candidate.backtest.cost_bps,
                    lambda value: value > 0,
                )
            )
        failures = [name for name, value, predicate in checks if value is None or not predicate(value)]
        failures.extend(self._annual_failures(metrics, expected_years=expected_years))
        return not failures, failures


@dataclass(frozen=True)
class StrategyEvaluation:
    """What a strategy evaluator returns for one candidate.

    ``metrics`` must embed ``metrics_by_year`` (the gate reads it from there);
    ``returns`` is the daily net portfolio return series used for redundancy
    checks and for the compact trace stored in the registry.
    """

    metrics: Mapping[str, Any]
    returns: pd.Series | None = None
    weights: pd.DataFrame | None = None
    artifact_path: str | None = None
    artifact_checksum: str | None = None
    factor_contributions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: PortfolioResult, **overrides: Any) -> StrategyEvaluation:
        # The yearly breakdown is merged into ``metrics`` because the gate reads
        # it from there, mirroring how the factor worker nests ``metrics_by_year``.
        metrics = {**result.metrics, "metrics_by_year": dict(result.metrics_by_year)}
        payload: dict[str, Any] = {
            "metrics": metrics,
            "returns": result.daily_returns,
            "weights": result.weights,
        }
        payload.update(overrides)
        return cls(**payload)


@dataclass(frozen=True)
class StrategyProcessingResult:
    record: StrategyRecord
    gate_failures: tuple[str, ...] = ()
    dedup: DedupResult | None = None
    evaluation: StrategyEvaluation | None = None


def compact_return_profile(returns: pd.Series) -> dict[str, list[Any]]:
    """Round-trippable daily return trace, following the benchmark profile shape."""

    return {
        "dates": [pd.Timestamp(index).strftime("%Y-%m-%d") for index in returns.index],
        "returns": [round(float(value), 8) for value in returns],
    }


def profile_from_payload(payload: Mapping[str, Any] | None) -> pd.Series | None:
    if not isinstance(payload, Mapping):
        return None
    dates = payload.get("dates")
    values = payload.get("returns")
    if not isinstance(dates, list) or not isinstance(values, list) or len(dates) != len(values):
        return None
    series = pd.Series(
        [value for value in values if value is not None],
        index=[date for date, value in zip(dates, values) if value is not None],
        dtype=float,
    )
    return series.rename("strategy_returns") if len(series) else None


def _iso_indexed(series: pd.Series) -> pd.Series:
    """Reindex a daily series on ISO date strings.

    Stored profiles use ISO date keys (they live in JSON), so the candidate side
    must be keyed the same way or the correlation would see zero overlap.
    """

    return pd.Series(
        series.to_numpy(dtype=float),
        index=[pd.Timestamp(index).strftime("%Y-%m-%d") for index in series.index],
        dtype=float,
    )


class StrategyAcceptanceService:
    """Gate strategy candidates and register the outcome."""

    def __init__(
        self,
        registry: StrategyRegistry,
        evaluator: Callable[[StrategyCandidate], StrategyEvaluation],
        *,
        policy: StrategyAcceptancePolicy | None = None,
        deduplicator: FactorDeduplicator | None = None,
        factor_registry: FactorRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator
        self.policy = policy or StrategyAcceptancePolicy()
        self.deduplicator = deduplicator or FactorDeduplicator()
        # Read-only reference: lets a strategy be audited against the factors it
        # claims to use.  Nothing is ever written to the factor registry here.
        self.factor_registry = factor_registry
        self._expected_evaluation_years: int | None = None

    def bind_evaluation_window(self, evaluation_start: str, evaluation_end: str) -> None:
        """Bind the calendar-year capacity used by the annual stability gate."""

        try:
            start = pd.Timestamp(evaluation_start)
            end = pd.Timestamp(evaluation_end)
        except (TypeError, ValueError) as exc:
            raise ValueError("strategy evaluation window must contain ISO dates") from exc
        if start > end:
            raise ValueError("strategy evaluation start must not be after end")
        self._expected_evaluation_years = int(end.year - start.year + 1)

    def _referenced_factors(self, candidate: StrategyCandidate) -> dict[str, dict[str, Any]]:
        """Resolve referenced factors for auditability (missing ones are kept)."""

        if self.factor_registry is None or not candidate.factor_names:
            return {}
        resolved: dict[str, dict[str, Any]] = {}
        for name in candidate.factor_names:
            record = self.factor_registry.get_by_name(name)
            if record is None:
                resolved[name] = {"status": "unknown"}
            else:
                resolved[name] = {
                    "factor_id": record.factor_id,
                    "status": record.status,
                    "family": record.family,
                }
        return resolved

    def _stored_signatures(self, exclude: str) -> list[FactorSignature]:
        signatures: list[FactorSignature] = []
        for record in self.registry.list(status="accepted"):
            if record.strategy_id == exclude:
                continue
            signatures.append(
                FactorSignature(
                    record.strategy_id,
                    record.spec_json,
                    return_profile=profile_from_payload(record.metadata.get(PROFILE_KEY)),
                )
            )
        return signatures

    def _dedup(
        self,
        candidate: StrategyCandidate,
        strategy_id: str,
        evaluation: StrategyEvaluation,
        *,
        existing: Iterable[FactorSignature] | None = None,
    ) -> DedupResult:
        if existing is None:
            existing = self._stored_signatures(strategy_id)
        signature = FactorSignature(
            strategy_id,
            candidate.canonical_payload(),
            values=None,
            return_profile=_iso_indexed(evaluation.returns),
        )
        return self.deduplicator.compare(signature, existing)

    def process(
        self,
        candidate: StrategyCandidate,
        *,
        existing_signatures: Iterable[FactorSignature] | None = None,
    ) -> StrategyProcessingResult:
        evaluation = self.evaluator(candidate)
        passed, failures = self.policy.check(
            dict(evaluation.metrics),
            candidate,
            expected_years=self._expected_evaluation_years,
        )
        if evaluation.returns is None or evaluation.returns.empty:
            # Fail closed: without a daily return series neither the redundancy
            # check nor a reproducible audit is possible.
            failures = [*failures, "strategy_return_profile_missing"]
            passed = False
        strategy_id = strategy_id_for(candidate.name, candidate.canonical_payload())
        base_metadata = {
            **candidate.metadata,
            "hypothesis": candidate.hypothesis,
            "gate_failures": failures,
            "referenced_factors": self._referenced_factors(candidate),
            "annual_gate_requirements": self.policy.annual_requirements(
                self._expected_evaluation_years
            ),
        }
        if not passed:
            record = self.registry.upsert(
                StrategyRecord(
                    strategy_id=strategy_id,
                    name=candidate.name,
                    universe=candidate.universe,
                    rule=candidate.rule,
                    spec_json=candidate.canonical_payload(),
                    factor_names=list(candidate.factor_names),
                    status="rejected",
                    metrics=dict(evaluation.metrics),
                    artifact_path=evaluation.artifact_path,
                    artifact_checksum=evaluation.artifact_checksum,
                    metadata=base_metadata,
                )
            )
            return StrategyProcessingResult(record, tuple(failures), evaluation=evaluation)

        dedup = self._dedup(candidate, strategy_id, evaluation, existing=existing_signatures)
        metadata = {
            **base_metadata,
            "dedup_reason": dedup.reason if dedup.is_duplicate else "no_duplicate_signal",
        }
        if evaluation.returns is not None:
            # Stored so a later candidate can be compared against this one; the
            # trace is compact (dates + rounded returns), never a holdings panel.
            metadata[PROFILE_KEY] = compact_return_profile(evaluation.returns)
        record = self.registry.upsert(
            StrategyRecord(
                strategy_id=strategy_id,
                name=candidate.name,
                universe=candidate.universe,
                rule=candidate.rule,
                spec_json=candidate.canonical_payload(),
                factor_names=list(candidate.factor_names),
                status="duplicate" if dedup.is_duplicate else "accepted",
                metrics=dict(evaluation.metrics),
                artifact_path=evaluation.artifact_path,
                artifact_checksum=evaluation.artifact_checksum,
                dedup_of=dedup.duplicate_of,
                metadata=metadata,
            )
        )
        return StrategyProcessingResult(record, (), dedup, evaluation)
