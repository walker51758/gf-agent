"""Evaluate, gate, de-duplicate and register GF factor candidates."""

from __future__ import annotations

import dataclasses
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.scenarios.gf_quant.benchmark import (
    SUPPORTED_FAMILIES,
    BenchmarkFactorResult,
    FamilyCalibration,
    GateReport,
    StandardsValidationError,
    evaluate_against_calibration,
    load_family_calibrations,
    load_family_signatures,
    profile_correlation,
)
from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.dedup import DedupResult, FactorDeduplicator, FactorSignature
from rdagent.scenarios.gf_quant import gate_policy
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
    """Explicit development fallback for tests and offline bootstrapping.

    These constants are **not** calibrated: they predate the benchmark
    calibration system and are only used when a family has no calibrated
    standards (see :class:`FactorAcceptanceService`).  Production calibration
    mode replaces them entirely.
    """

    min_abs_rank_ic: float = 0.03
    min_annualized_return: float = 0.0
    min_sharpe: float = 0.30
    max_drawdown_abs: float = 0.35
    max_turnover: float = 1.0
    min_coverage: float = 0.50
    # TEMPORARY annual gate.  The screening window now spans 2020-2023, so a
    # candidate must also hold up inside individual calendar years instead of
    # only in aggregate.  Defaults come from the GF_SCENARIO_SETTINGS knobs
    # (annual_gate_*) so the gate stays configurable without code changes; the
    # whole section is expected to be retuned or retired once the calibrated
    # family standards are in place.
    annual_min_years_with_rank_ic: int = field(
        default_factory=lambda: GF_SCENARIO_SETTINGS.annual_gate_min_years_with_rank_ic
    )
    annual_min_matching_sign_years: int = field(
        default_factory=lambda: GF_SCENARIO_SETTINGS.annual_gate_min_matching_sign_years
    )

    @staticmethod
    def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
        for name in names:
            if name in metrics and metrics[name] is not None:
                try:
                    return float(metrics[name])
                except (TypeError, ValueError):
                    return None
        return None

    @classmethod
    def _finite_metric(cls, metrics: Mapping[str, Any], *names: str) -> float | None:
        value = cls._metric(metrics, *names)
        return value if value is not None and math.isfinite(value) else None

    def _annual_failures(self, metrics: Mapping[str, Any]) -> list[str]:
        """TEMPORARY gate over ``metrics_by_year``; see the field comments above."""

        by_year = metrics.get("metrics_by_year")
        if not isinstance(by_year, Mapping) or not by_year:
            # Fail closed: without per-year metrics the annual checks cannot run.
            return ["annual_metrics_by_year_missing"]

        aggregate_rank_ic = self._finite_metric(metrics, "Rank_IC", "Rank IC", "rank_ic")
        failures: list[str] = []
        if aggregate_rank_ic is None or aggregate_rank_ic == 0.0:
            # No aggregate sign, so no year can match it.
            failures.append("annual_rank_ic_sign_alignment")

        yearly_rank_ic = [
            self._finite_metric(by_year[year], "Rank_IC", "Rank IC", "rank_ic") for year in sorted(by_year, key=str)
        ]
        finite = [value for value in yearly_rank_ic if value is not None]
        if len(finite) < self.annual_min_years_with_rank_ic:
            failures.append("annual_rank_ic_years")
        if aggregate_rank_ic is not None and aggregate_rank_ic != 0.0:
            aligned = sum(1 for value in finite if value != 0.0 and (value > 0) == (aggregate_rank_ic > 0))
            if aligned < self.annual_min_matching_sign_years:
                failures.append("annual_rank_ic_sign_alignment")
        return failures

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
        failures.extend(self._annual_failures(metrics))
        return not failures, failures


@dataclass(frozen=True)
class ProcessingResult:
    record: FactorRecord
    gate_failures: tuple[str, ...] = ()
    dedup: DedupResult | None = None
    benchmark_gate: GateReport | None = None
    benchmark_redundancy: Mapping[str, Any] | None = None


DEVELOPMENT_FALLBACK_SOURCE = "development_fallback"


class FactorAcceptanceService:
    """Gate candidates with the calibrated standards of their own family.

    Standards are selected by ``candidate.family``:

    * ``price_volume`` candidates are held to the merged artifacts + alpha158
      benchmark distribution;
    * ``fundamental`` candidates are held to the basic_fundamental distribution.

    When no calibration exists the service falls back to the explicit
    development policy and says so in the registry metadata.  With
    ``GF_BENCHMARK_CALIBRATION_REQUIRED=true`` a missing or invalid calibration
    raises instead of falling back.
    """

    def __init__(
        self,
        registry: FactorRegistry,
        evaluator: Callable[[FactorCandidate], EvaluationResult],
        *,
        policy: AcceptancePolicy | None = None,
        deduplicator: FactorDeduplicator | None = None,
        calibrations: Mapping[str, FamilyCalibration] | None = None,
        standards_path: str | Path | None = None,
        benchmark_signatures: Sequence[BenchmarkFactorResult] | None = None,
        signature_dir: str | Path | None = None,
        settings: Any = GF_SCENARIO_SETTINGS,
        active_gates: Sequence[str] | None = None,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator
        # ``None`` means every calibrated gate is enforced.  A non-None tuple
        # narrows which gate verdicts may fail a candidate; see
        # :mod:`rdagent.scenarios.gf_quant.gate_policy` for the temporary regime.
        self.active_gates = active_gates
        self.policy = policy or AcceptancePolicy()
        self.deduplicator = deduplicator or FactorDeduplicator()
        self.settings = settings
        self.benchmark_signatures: Mapping[str, list[BenchmarkFactorResult]] = {}
        self.calibrations = (
            dict(calibrations) if calibrations is not None else self._load_calibrations(standards_path)
        )
        self.standards_source = (
            "calibrated" if self.calibrations else DEVELOPMENT_FALLBACK_SOURCE
        )
        self.benchmark_signatures = (
            self._index_signatures(benchmark_signatures)
            if benchmark_signatures is not None
            else self._load_signatures(signature_dir)
        )

    # ---- standards loading -------------------------------------------------

    @staticmethod
    def _project_path(path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return Path(__file__).parents[3] / candidate

    def _load_calibrations(self, standards_path: str | Path | None) -> dict[str, FamilyCalibration]:
        path = self._project_path(standards_path or self.settings.benchmark_standards_path)
        required = bool(getattr(self.settings, "benchmark_calibration_required", False))
        try:
            return load_family_calibrations(path)
        except StandardsValidationError as exc:
            if required:
                raise RuntimeError(f"Calibrated benchmark standards are required but unusable: {exc}") from exc
            print(f"[gf-quant] development fallback for acceptance standards: {exc}", file=sys.stderr)
            return {}

    @staticmethod
    def _index_signatures(
        signatures: Sequence[BenchmarkFactorResult] | Mapping[str, list[BenchmarkFactorResult]],
    ) -> Mapping[str, list[BenchmarkFactorResult]]:
        """Index signatures by family, accepting either a flat list or a mapping."""
        if isinstance(signatures, Mapping):
            return signatures
        indexed: dict[str, list[BenchmarkFactorResult]] = {}
        for signature in signatures:
            indexed.setdefault(signature.family, []).append(signature)
        return indexed

    def _load_signatures(self, signature_dir: str | Path | None) -> dict[str, list[BenchmarkFactorResult]]:
        directory = self._project_path(signature_dir or self.settings.benchmark_signature_dir)
        by_family: dict[str, list[BenchmarkFactorResult]] = {}
        for family in SUPPORTED_FAMILIES:
            try:
                by_family[family] = load_family_signatures(directory, family)
            except StandardsValidationError as exc:
                print(f"[gf-quant] no {family} benchmark signatures for redundancy checks: {exc}", file=sys.stderr)
        return by_family

    # ---- gating ------------------------------------------------------------

    def _gate(self, candidate: FactorCandidate, evaluation: EvaluationResult) -> tuple[bool, list[str], GateReport | None, dict[str, Any]]:
        calibration = self.calibrations.get(candidate.family)
        if calibration is None:
            if self.calibrations:
                raise RuntimeError(f"No calibrated benchmark standards for family {candidate.family!r}")
            passed, failures = self.policy.check(evaluation.metrics)
            metadata = {"standards_source": DEVELOPMENT_FALLBACK_SOURCE}
            return passed, failures, None, metadata
        report = evaluate_against_calibration(
            evaluation.metrics,
            calibration,
            return_profile=evaluation.return_profile,
            require_profile=bool(getattr(self.settings, "benchmark_profile_required", False)),
        )
        report = self._apply_gate_selection(report)
        metadata = {
            "standards_source": "calibrated",
            "benchmark_percentiles": dict(report.percentiles),
            "benchmark_skipped_gates": list(report.skipped_gates),
            "benchmark_profile_available": report.profile_available,
            "standards": calibration.standards.to_payload(),
        }
        return report.passed, list(report.failures), report, metadata


    def _apply_gate_selection(self, report: GateReport) -> GateReport:
        """Narrow which gate verdicts may fail a candidate.

        The gates themselves are all evaluated, and their verdicts are all still
        reported: a gate the current selection suppresses is moved into
        ``skipped_gates`` rather than disappearing, so a report can always show what
        was measured and what was set aside.  With ``active_gates`` unset this is the
        identity function.
        """

        if self.active_gates is None:
            return report
        kept, suppressed = gate_policy.partition(tuple(report.failures), self.active_gates)
        if not suppressed:
            return report
        return dataclasses.replace(
            report,
            passed=not kept,
            failures=kept,
            skipped_gates=tuple(dict.fromkeys((*report.skipped_gates, *suppressed))),
        )


    def _benchmark_redundancy(self, candidate: FactorCandidate, evaluation: EvaluationResult) -> dict[str, Any] | None:
        """Family-aware redundancy against the benchmark pool.

        Compares the candidate's compact daily long-short return profile with
        the benchmark profiles of the same family using the same correlation
        definition as the calibration clustering.  This is kept separate from
        the global registry deduplication, which still runs afterwards.
        """
        if evaluation.return_profile is None:
            return None
        candidate_profile = {
            pd.Timestamp(index).strftime("%Y-%m-%d"): float(value)
            for index, value in evaluation.return_profile.items()
        }
        for benchmark in self.benchmark_signatures.get(candidate.family, ()):
            correlation = profile_correlation(
                candidate_profile,
                benchmark.spread_profile,
                min_overlap=self.deduplicator.min_overlap,
            )
            if correlation is None or abs(correlation) < self.deduplicator.threshold:
                continue
            return {
                "benchmark_redundant_of": benchmark.name,
                "benchmark_redundancy_family": candidate.family,
                "benchmark_redundancy_reason": "benchmark_family_profile_correlation",
                "benchmark_redundancy_correlation": correlation,
                "benchmark_redundancy_source": benchmark.source_dir,
            }
        return None

    def process(
        self,
        candidate: FactorCandidate,
        *,
        existing_signatures: Iterable[FactorSignature] | None = None,
    ) -> ProcessingResult:
        if candidate.family not in ("price_volume", "fundamental"):
            raise ValueError(f"Unsupported family: {candidate.family}")
        evaluation = self.evaluator(candidate)
        passed, failures, gate, gate_metadata = self._gate(candidate, evaluation)
        factor_id = factor_id_for(candidate.name, candidate.expression)
        base_metadata = {
            **candidate.metadata,
            "hypothesis": candidate.hypothesis,
            "gate_failures": failures,
            **gate_metadata,
        }
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
            return ProcessingResult(record, tuple(failures), benchmark_gate=gate)

        redundancy = self._benchmark_redundancy(candidate, evaluation)

        if existing_signatures is None:
            existing_signatures = [
                FactorSignature(record.factor_id, record.expression)
                for record in self.registry.list(status="accepted")
                if record.factor_id != factor_id
            ]
        signature = FactorSignature(factor_id, candidate.expression, evaluation.values, evaluation.return_profile)
        # Secondary safety check: the global registry dedup is deliberately
        # separate from the family-aware benchmark redundancy above.
        dedup = self.deduplicator.compare(signature, existing_signatures)
        status = "duplicate" if (redundancy or dedup.is_duplicate) else "accepted"
        metadata = {
            **base_metadata,
            "dedup_reason": dedup.reason if dedup.is_duplicate else "no_duplicate_signal",
            **(redundancy or {}),
        }
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
                metadata=metadata,
            )
        )
        return ProcessingResult(record, (), dedup, benchmark_gate=gate, benchmark_redundancy=redundancy)
