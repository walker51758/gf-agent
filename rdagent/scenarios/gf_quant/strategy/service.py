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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rdagent.scenarios.gf_quant.dedup import DedupResult, FactorDeduplicator, FactorSignature
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.strategy.backtest import PortfolioResult
from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.costs import (
    BUY_COST_BPS,
    COST_TOLERANCE_BPS,
    ROUND_TRIP_COST_BPS,
    SELL_COST_BPS,
    execution_costs_metrics,
)
from rdagent.scenarios.gf_quant.strategy.deployment import DeploymentPolicy, deployment_verdict
from rdagent.scenarios.gf_quant.strategy.manifest import build_strategy_manifest
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRecord, StrategyRegistry, strategy_id_for
from rdagent.scenarios.gf_quant.strategy.reproducibility import reproducibility_report

PROFILE_KEY = "return_profile"
"""Registry metadata key holding the compact daily return trace used for dedup."""

@dataclass(frozen=True)
class StrategyAcceptancePolicy:
    """Explicit strategy gate.

    The defaults are development-grade, not calibrated; they exist so a missing
    threshold never silently means "accept anything".

    The execution-cost contract is not a policy knob. Every evaluation must
    prove that it used the system-owned 10 bps buy / 10 bps sell rates.
    """

    min_signal_lag: int = 1
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
    def execution_costs() -> dict[str, float]:
        """Describe the immutable cost facts every evaluation must report."""

        return execution_costs_metrics()

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
        for failure, key, required in (
            ("strategy_buy_cost_not_fixed", "buy_cost_bps", BUY_COST_BPS),
            ("strategy_sell_cost_not_fixed", "sell_cost_bps", SELL_COST_BPS),
            ("strategy_round_trip_cost_not_fixed", "round_trip_cost_bps", ROUND_TRIP_COST_BPS),
        ):
            checks.append(
                (
                    failure,
                    self._finite(metrics, key),
                    lambda value, expected=required: math.isclose(
                        value, expected, rel_tol=0.0, abs_tol=COST_TOLERANCE_BPS
                    ),
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
    # Request-level result-report audits (roadmap 9.4/10.2/10.3/10.4).  The
    # worker emits ``audit_report`` only when the request carried an ``audit``
    # object, so absent means "no audit was requested"; the audits are
    # read-only report output and never part of the candidate payload.
    audit_report: Mapping[str, Mapping[str, Any]] | None = None

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


def _provenance_text(value: Any) -> str:
    """A provenance string, with ``None`` read as *absent*, not as ``"None"``.

    ``facts.get(key, "")`` substitutes its default only when the key is missing,
    so a key that is present and holds ``None`` would be stringified into the
    literal ``"None"`` -- a value ``reproducibility._is_carried`` counts as
    carried, which would let two manifests agree on a field neither of them
    actually carries.  ``None`` is normalised to the same empty string the
    absent key produces.
    """

    return "" if value is None else str(value)


def _provenance_mapping(value: Any) -> dict[str, Any]:
    """A provenance mapping, with ``None`` read as absent.

    Same reading as :func:`_provenance_text`: an evaluation window that is not
    carried is an empty mapping, not a reason to fail building the manifest.
    """

    return {} if value is None else dict(value)


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
        emit_manifest: bool = False,
        reproducibility_check: bool = False,
        deployment_policy: DeploymentPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.evaluator = evaluator
        self.policy = policy or StrategyAcceptancePolicy()
        self.deduplicator = deduplicator or FactorDeduplicator()
        # Read-only reference: lets a strategy be audited against the factors it
        # claims to use.  Nothing is ever written to the factor registry here.
        self.factor_registry = factor_registry
        # Governance switches (roadmap 10.8/10.9/10.10).  All three default off,
        # and the registry record is byte-for-byte what it was before this batch
        # while they are off -- none of them touches the candidate payload, so
        # none of them can move a strategy identity hash.
        #
        # The three travel together, and the reason is that they are one
        # evidence chain rather than three independent knobs: the manifest is
        # what gets recorded, the reproducibility verdict is a comparison
        # *against that recorded manifest*, and the deployment gate reads the
        # verdict.  Enabling the check without recording the manifest leaves
        # nothing to compare against, so it would compare each manifest with
        # itself and report agreement forever -- an enabled check that can never
        # fail.  That combination is refused here instead.
        self.emit_manifest = bool(emit_manifest)
        self.reproducibility_check = bool(reproducibility_check)
        if self.reproducibility_check and not self.emit_manifest:
            raise ValueError(
                "reproducibility_check requires emit_manifest: the check compares against the manifest "
                "on record, so nothing can be verified unless that manifest is written"
            )
        self.deployment_policy = deployment_policy
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

    def _manifest_facts(self) -> dict[str, Any]:
        """Provenance facts for the manifest, read from the bound evaluator.

        The evaluator holds the frozen snapshot, working set, artifact set and
        evaluation window, so it is the only place these facts exist -- and an
        evaluator that cannot supply them is a hard error, not a quiet zero.
        Returning ``{}`` here would produce a manifest whose provenance fields
        are all empty strings while still reading as a healthy record, which is
        exactly the failure mode that lets an undocumented provenance gap pass
        for a fact.  Absent provenance is reported as absent.
        """

        facts = getattr(self.evaluator, "manifest_facts", None)
        if not callable(facts):
            raise ValueError("strategy evaluator does not provide manifest facts")
        resolved = facts()
        if not isinstance(resolved, Mapping):
            raise ValueError("evaluator manifest facts must be an object")
        return dict(resolved)

    def _build_manifest(
        self,
        candidate: StrategyCandidate,
        strategy_id: str,
        evaluation: StrategyEvaluation,
        failures: Sequence[str],
    ) -> dict[str, Any]:
        """Assemble the roadmap 10.8 manifest from facts already on hand.

        ``provenance_available`` is written explicitly.  It is ``False`` when the
        evaluator could not supply provenance facts at all (no ``manifest_facts``,
        or the call raised): the manifest then says so rather than carrying a row
        of empty strings that a reader would take for real values, and
        :mod:`reproducibility` turns that into ``not_reproducible``.  That is one
        of the ways this flag is set, not the only one; the others are below.

        ``None`` is read as *absent* throughout.  ``facts.get(key, default)``
        substitutes its default only when the key is missing, so a key present
        with ``None`` would otherwise be stringified into ``"None"`` -- a value
        :func:`reproducibility._is_carried` counts as carried -- or, for the two
        collection-valued facts, crash the builder outright.  Every fact goes
        through the same normalisation, so a ``None`` lands on the empty shape of
        its own type: one of ``_is_carried``'s four, not all four.  A fact that
        arrives *already* empty (``""``, ``{}``, ``[]``) is a different case and
        is not reported by this normalisation.

        Absence is then reported by one of two mechanisms, never by neither:

        * the facts that are members of ``REPRODUCIBILITY_IDENTITY_FIELDS`` --
          ``snapshot_checksum``, ``working_set_snapshot_checksum``,
          ``artifact_set_id``, ``evaluation`` -- are reported *per field*: once
          normalised, ``_is_carried`` names each absent one as
          ``field_unknown:<field>``.
        * the remaining provenance facts -- ``snapshot_id``, ``working_set_id``,
          ``data_cutoff``, ``factor_ids`` -- are not in that set, so nothing
          downstream would ever notice their absence; when one of them arrives
          as ``None`` it is recorded as the empty shape of its type *and*
          reported through the manifest-level flag above, which forces
          ``not_reproducible``.  (``execution_costs`` and ``worker_attestation``
          are not facts of this kind: the first comes from the policy, the second
          is carried whole or not at all.)

        The flag reports the ``None`` shape, so a fact that reaches this builder
        as ``None`` can never read as healthy on any of the four.  It does *not*
        report the other three empty shapes ``_is_carried`` rejects, nor a fact
        whose value is of some other type: each is recorded as whatever
        ``_provenance_text`` / ``_provenance_mapping`` make of it -- ``{}`` in a
        text fact is stringified into ``"{}"`` -- and since nothing downstream
        polices these four, such a value would pass for a real one.  What keeps
        that from happening sits upstream, not here.  ``snapshot_id`` and
        ``working_set_id`` are ids derived from the frozen contents, and the
        frozen dataclasses refuse a value that disagrees with them, so the
        ``str`` a production evaluator hands over is already the derived,
        non-empty id.  That is an equality check on a value that reached the
        dataclass as a ``str``: it constrains which string a well-behaved
        producer supplies, and it does not constrain the type of whatever a
        facts mapping carries, which this builder does not police either.
        ``data_cutoff`` is refused empty by the snapshot builder (one level
        weaker: the dataclass itself does not check it).  ``factor_ids`` needs no
        such guard, because an empty list is a value rather than an absence -- a
        rule-only strategy has no factors -- and that is exactly why ``None``,
        the shape that would be a lie, is the one the flag reports.  Either way
        the flag cannot say *which* fact is absent, and it stays silent about
        facts the evaluator never claimed to carry in the first place.
        """

        worker_attestation: Any = None
        if evaluation.audit_report is not None:
            section = evaluation.audit_report.get("manifest")
            if isinstance(section, Mapping):
                worker_attestation = dict(section)

        provenance_available = True
        try:
            facts = self._manifest_facts()
        except ValueError:
            facts = {}
            provenance_available = False

        if provenance_available and not facts:
            # An empty but successfully returned facts object carries no
            # information either; treat it the same as an absent one.
            provenance_available = False

        # Provenance facts that no downstream check polices (see the docstring).
        # Each absent one is recorded as its empty shape and flagged here.
        unpoliced_absent = [
            name for name in ("snapshot_id", "working_set_id", "data_cutoff") if facts.get(name) is None
        ]
        factor_ids = facts.get("factor_ids")
        if factor_ids is None:
            factor_ids = []
            unpoliced_absent.append("factor_ids")
        if unpoliced_absent:
            provenance_available = False

        return build_strategy_manifest(
            strategy_id=strategy_id,
            strategy_name=candidate.name,
            canonical_payload=candidate.canonical_payload(),
            evaluation=_provenance_mapping(facts.get("evaluation")),
            execution_costs=self.policy.execution_costs(),
            artifact_set_id=_provenance_text(facts.get("artifact_set_id")),
            factor_ids=factor_ids,
            snapshot_id=_provenance_text(facts.get("snapshot_id")),
            snapshot_checksum=_provenance_text(facts.get("snapshot_checksum")),
            working_set_id=_provenance_text(facts.get("working_set_id")),
            working_set_snapshot_checksum=_provenance_text(facts.get("working_set_snapshot_checksum")),
            data_cutoff=_provenance_text(facts.get("data_cutoff")),
            acceptance={
                "status": "rejected" if failures else "accepted",
                "gate_failures": list(failures),
            },
            worker_attestation=worker_attestation,
            provenance_available=provenance_available,
        )

    def _recorded_manifest(self, strategy_id: str) -> dict[str, Any] | None:
        """The manifest already on record for this strategy id, if any.

        Reproducibility is a comparison against what was stored, so the second
        emission of the same strategy id is what actually proves the identity
        still holds.  A registry without a manifest for this id yields ``None``,
        in which case :meth:`_governance_metadata` reports ``first_emission``.
        """

        record = self.registry.get(strategy_id)
        if record is None:
            return None
        manifest = record.metadata.get("manifest")
        return dict(manifest) if isinstance(manifest, Mapping) else None

    def _governance_metadata(
        self,
        candidate: StrategyCandidate,
        strategy_id: str,
        evaluation: StrategyEvaluation,
        failures: Sequence[str],
        dedup: DedupResult | None,
    ) -> dict[str, Any]:
        """The three governance reports, in the order each consumes the previous.

        Shared by the accepted and rejected paths: a rejected candidate is still
        a candidate the governance layer has an opinion about, and roadmap 10.10
        lists ``reject`` as one of its three outputs -- an acceptance gate wired
        so that governance never runs on a failure would make that output
        unreachable and the gate decorative.  ``dedup`` is ``None`` on the
        rejected path (the redundancy comparison never runs there), which reads
        as "not a duplicate", the permissive value.

        All three live in ``metadata`` only -- no column, no status value and no
        payload key is added, so the record shape is unchanged while the switches
        are off.
        """

        metadata: dict[str, Any] = {}
        manifest: dict[str, Any] | None = None
        if self.emit_manifest and self.reproducibility_check:
            manifest = self._build_manifest(candidate, strategy_id, evaluation, failures)
            metadata["manifest"] = manifest
            reference = self._recorded_manifest(strategy_id)
            metadata["reproducibility"] = (
                {**reproducibility_report(manifest, manifest), "mode": "first_emission"}
                if reference is None
                else {**reproducibility_report(reference, manifest), "mode": "comparison"}
            )
        if self.deployment_policy is not None:
            metadata["deployment"] = deployment_verdict(
                gate_failures=list(failures),
                audit_report=evaluation.audit_report,
                dedup_is_duplicate=bool(dedup.is_duplicate) if dedup is not None else False,
                reproducibility=metadata.get("reproducibility"),
                policy=self.deployment_policy,
            )
        return metadata

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
            "execution_costs": self.policy.execution_costs(),
        }
        if not passed:
            # Governance runs on the rejected path too: roadmap 10.10 lists
            # ``reject`` among its three outputs, and the deployment verdict is
            # only reachable when the acceptance failures are handed to it.
            # Leaving the record untraced here would make that whole band dead
            # code.  ``dedup`` is absent because the redundancy comparison never
            # runs for a candidate that failed acceptance.
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
                    metadata={
                        **base_metadata,
                        **self._governance_metadata(
                            candidate, strategy_id, evaluation, failures, dedup=None
                        ),
                    },
                )
            )
            return StrategyProcessingResult(record, tuple(failures), evaluation=evaluation)

        dedup = self._dedup(candidate, strategy_id, evaluation, existing=existing_signatures)
        metadata = {
            **base_metadata,
            "dedup_reason": dedup.reason if dedup.is_duplicate else "no_duplicate_signal",
            **self._governance_metadata(candidate, strategy_id, evaluation, failures, dedup=dedup),
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
