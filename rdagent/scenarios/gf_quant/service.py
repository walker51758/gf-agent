"""Evaluate, gate, de-duplicate and register GF factor candidates."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from rdagent.scenarios.gf_quant.dedup import DedupResult, FactorDeduplicator, FactorSignature
from rdagent.scenarios.gf_quant.registry import FactorFamily, FactorRecord, FactorRegistry, factor_id_for


@dataclass(frozen=True)
class FactorCandidate:
    name: str
    family: FactorFamily
    expression: str
    hypothesis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, Any]
    artifact_path: str | None = None
    artifact_checksum: str | None = None
    values: pd.Series | None = None
    return_profile: pd.Series | None = None


@dataclass(frozen=True)
class AcceptancePolicy:
    min_abs_rank_ic: float = 0.03
    min_annualized_return: float = 0.0
    min_sharpe: float = 0.30
    max_drawdown_abs: float = 0.35
    max_turnover: float = 1.0
    min_coverage: float = 0.50

    @staticmethod
    def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
        for name in names:
            if name in metrics and metrics[name] is not None:
                try:
                    return float(metrics[name])
                except (TypeError, ValueError):
                    return None
        return None

    def check(self, metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
        checks = (
            ("rank_ic", self._metric(metrics, "Rank_IC", "Rank IC", "rank_ic"), lambda x: abs(x) >= self.min_abs_rank_ic),
            ("annualized_return", self._metric(metrics, "annualized_return", "arr"), lambda x: x >= self.min_annualized_return),
            ("sharpe", self._metric(metrics, "sharpe", "information_ratio"), lambda x: x >= self.min_sharpe),
            ("max_drawdown", self._metric(metrics, "max_drawdown", "mdd"), lambda x: abs(x) <= self.max_drawdown_abs),
            ("turnover", self._metric(metrics, "turnover"), lambda x: x <= self.max_turnover),
            ("coverage", self._metric(metrics, "coverage"), lambda x: x >= self.min_coverage),
        )
        failures = [name for name, value, predicate in checks if value is None or not predicate(value)]
        return not failures, failures


@dataclass(frozen=True)
class ProcessingResult:
    record: FactorRecord
    gate_failures: tuple[str, ...] = ()
    dedup: DedupResult | None = None


class FactorAcceptanceService:
    def __init__(
        self,
        registry: FactorRegistry,
        evaluator: Callable[[FactorCandidate], EvaluationResult],
        *,
        policy: AcceptancePolicy | None = None,
        deduplicator: FactorDeduplicator | None = None,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator
        self.policy = policy or AcceptancePolicy()
        self.deduplicator = deduplicator or FactorDeduplicator()

    def process(
        self,
        candidate: FactorCandidate,
        *,
        existing_signatures: Iterable[FactorSignature] | None = None,
    ) -> ProcessingResult:
        if candidate.family not in ("price_volume", "fundamental"):
            raise ValueError(f"Unsupported family: {candidate.family}")
        evaluation = self.evaluator(candidate)
        passed, failures = self.policy.check(evaluation.metrics)
        factor_id = factor_id_for(candidate.name, candidate.expression)
        base_metadata = {**candidate.metadata, "hypothesis": candidate.hypothesis, "gate_failures": failures}
        if not passed:
            record = self.registry.upsert(
                FactorRecord(
                    factor_id=factor_id,
                    name=candidate.name,
                    family=candidate.family,
                    expression=candidate.expression,
                    status="rejected",
                    metrics=dict(evaluation.metrics),
                    artifact_path=evaluation.artifact_path,
                    artifact_checksum=evaluation.artifact_checksum,
                    metadata=base_metadata,
                )
            )
            return ProcessingResult(record, tuple(failures))

        if existing_signatures is None:
            existing_signatures = [
                FactorSignature(record.factor_id, record.expression)
                for record in self.registry.list(status="accepted")
                if record.factor_id != factor_id
            ]
        signature = FactorSignature(factor_id, candidate.expression, evaluation.values, evaluation.return_profile)
        dedup = self.deduplicator.compare(signature, existing_signatures)
        status = "duplicate" if dedup.is_duplicate else "accepted"
        record = self.registry.upsert(
            FactorRecord(
                factor_id=factor_id,
                name=candidate.name,
                family=candidate.family,
                expression=candidate.expression,
                status=status,
                metrics=dict(evaluation.metrics),
                artifact_path=evaluation.artifact_path,
                artifact_checksum=evaluation.artifact_checksum,
                dedup_of=dedup.duplicate_of,
                metadata={**base_metadata, "dedup_reason": dedup.reason},
            )
        )
        return ProcessingResult(record, (), dedup)
