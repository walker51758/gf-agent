"""Turn a portfolio-derived :class:`FactorDemand` into real factor research.

The strategy orchestrator emits :class:`FactorDemand` objects after its inner
trials finish but does not know how to run the factor side.  This module is the
missing adapter: it renders a demand as factor-proposal feedback, calls the
existing ``RDAGentDSLProposer`` / ``FactorAcceptanceService`` pair, and returns a
bounded serializable summary instead of raw registry rows.

Two invariants matter for the outer loop to stay sound:

* **Names already present in the factor registry are skipped.**  ``factor_id``
  is derived from ``name`` *and* expression, so a re-proposal under a known name
  with a tweaked expression would violate the registry's unique-name constraint
  rather than update the original factor -- and it would also defeat the
  "new factors stay out of the frozen snapshot" rule by rewriting history.
* **The returned outcome is data, not side effects.**  The orchestrator stores
  it in :class:`~rdagent.scenarios.gf_quant.strategy.orchestrator.OuterEpochResult`,
  so everything must survive a JSON round trip.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.strategy.orchestrator import FactorResearchRequest

SUPPORTED_FAMILIES: tuple[str, ...] = ("price_volume", "fundamental")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class FactorAttemptSummary:
    """One proposed factor and what the acceptance gate decided about it."""

    name: str
    family: str
    expression: str
    factor_id: str
    status: str
    gate_failures: tuple[str, ...] = ()
    rank_ic: float | None = None
    error: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "expression": self.expression,
            "factor_id": self.factor_id,
            "status": self.status,
            "gate_failures": list(self.gate_failures),
            "rank_ic": self.rank_ic,
            "error": self.error,
        }


@dataclass(frozen=True)
class FactorResearchOutcome:
    """Compact result of one demand-driven factor research round."""

    epoch_id: str
    round_index: int
    mode: str
    demand_id: str | None
    intended_role: str
    target_family: str
    attempted_family: str
    feedback_digest: str
    attempts: tuple[FactorAttemptSummary, ...] = ()
    accepted_factor_ids: tuple[str, ...] = ()
    skipped_known_names: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    demand_evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "round_index": self.round_index,
            "mode": self.mode,
            "demand_id": self.demand_id,
            "intended_role": self.intended_role,
            "target_family": self.target_family,
            "attempted_family": self.attempted_family,
            "feedback_digest": self.feedback_digest,
            "attempts": [attempt.as_payload() for attempt in self.attempts],
            "accepted_factor_ids": list(self.accepted_factor_ids),
            "skipped_known_names": list(self.skipped_known_names),
            "notes": list(self.notes),
            "demand_evidence": dict(self.demand_evidence),
        }


def render_demand_feedback(request: FactorResearchRequest, *, avoid_names: Sequence[str] = ()) -> str:
    """Render a demand (or its absence) as text the factor proposer can act on."""

    demand = request.demand
    if demand is None:
        return (
            "Exploratory round: no portfolio failure is being targeted.  Propose a "
            "factor that is economically distinct from the ones already registered."
        )
    evidence = json.dumps(dict(demand.evidence), ensure_ascii=False, sort_keys=True)
    lines = [
        f"Portfolio research demand {demand.demand_id} (priority {demand.priority:.2f}).",
        f"Intended role in a portfolio: {demand.intended_role}.",
        f"Preferred economic theme: {demand.economic_theme}.",
        f"Reason: {demand.rationale}.",
        f"Evidence: {evidence}.",
    ]
    if avoid_names:
        lines.append(
            "Do not re-propose these factors or a close variant of them: "
            + ", ".join(avoid_names)
            + "."
        )
    lines.append(
        "The factor will be composed into a long-only top-N portfolio, so it must "
        "have a usable cross-sectional ranking, not just a high absolute return."
    )
    return "\n".join(lines)


def _feedback_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class DemandDrivenFactorResearchEngine:
    """Implement the orchestrator's ``FactorResearchEngine`` protocol.

    The proposer must expose ``propose(family, feedback)`` (the
    :class:`~rdagent.scenarios.gf_quant.mining.RDAGentDSLProposer` shape) and the
    service must expose ``process(candidate)`` (the
    :class:`~rdagent.scenarios.gf_quant.service.FactorAcceptanceService` shape).
    Both are duck-typed so tests can substitute fakes.
    """

    def __init__(
        self,
        *,
        proposer: Any,
        factor_service: Any,
        factor_registry: FactorRegistry | None = None,
        default_family: str = "fundamental",
        max_attempts: int = 3,
        stop_on_acceptance: bool = True,
    ) -> None:
        if default_family not in SUPPORTED_FAMILIES:
            raise ValueError(f"default_family must be one of {SUPPORTED_FAMILIES}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.proposer = proposer
        self.factor_service = factor_service
        self.factor_registry = factor_registry
        self.default_family = default_family
        self.max_attempts = max_attempts
        self.stop_on_acceptance = stop_on_acceptance

    # ---- helpers -----------------------------------------------------------

    def _resolve_family(self, target_family: str) -> str:
        return target_family if target_family in SUPPORTED_FAMILIES else self.default_family

    def _known_names(self) -> set[str]:
        if self.factor_registry is None:
            return set()
        return {record.name for record in self.factor_registry.list()}

    def _avoid_names(self, avoid_factor_ids: Sequence[str]) -> list[str]:
        """Resolve avoided factor ids to names so the prompt can name them.

        An id that no longer resolves is still reported: the demand recorded it
        for a reason, and dropping it silently would let the proposer walk back
        into the factor that just failed.
        """

        if self.factor_registry is None:
            return []
        names: list[str] = []
        for factor_id in avoid_factor_ids:
            record = self.factor_registry.get(str(factor_id))
            names.append(record.name if record is not None else str(factor_id))
        return names

    @staticmethod
    def _summary(candidate: Any, result: Any) -> FactorAttemptSummary:
        record = result.record
        metrics = record.metrics or {}
        return FactorAttemptSummary(
            name=str(candidate.name),
            family=str(candidate.family),
            expression=str(candidate.expression),
            factor_id=str(record.factor_id),
            status=str(record.status),
            gate_failures=tuple(result.gate_failures),
            rank_ic=_finite(metrics.get("Rank_IC")),
        )

    @staticmethod
    def _failed_summary(candidate: Any, error: Exception) -> FactorAttemptSummary:
        return FactorAttemptSummary(
            name=str(getattr(candidate, "name", "")),
            family=str(getattr(candidate, "family", "")),
            expression=str(getattr(candidate, "expression", "")),
            factor_id="",
            status="error",
            error=f"{type(error).__name__}: {error}",
        )

    # ---- protocol ----------------------------------------------------------

    def __call__(self, request: FactorResearchRequest) -> FactorResearchOutcome:
        demand = request.demand
        target_family = demand.target_family if demand is not None else "any"
        family = self._resolve_family(target_family)
        avoid_names = self._avoid_names(demand.avoid_factor_ids) if demand is not None else []
        known = self._known_names()
        feedback = render_demand_feedback(request, avoid_names=avoid_names)

        notes: list[str] = []
        attempts: list[FactorAttemptSummary] = []
        accepted: list[str] = []
        skipped: list[str] = []

        proposals: list[Any] = []
        proposal_error: str | None = None
        try:
            proposals = list(self.proposer.propose(family, feedback) or [])
        except Exception as error:  # a failed proposal round must not abort the epoch
            proposal_error = f"{type(error).__name__}: {error}"

        if proposal_error is not None:
            notes.append(f"proposal_failed: {proposal_error}")
        elif not proposals:
            notes.append("no_parseable_proposal")

        for candidate in proposals[: self.max_attempts]:
            name = str(getattr(candidate, "name", ""))
            if name in known:
                # Re-proposing a known name with a new expression cannot be
                # registered (unique name) and would rewrite a frozen factor.
                skipped.append(name)
                continue
            try:
                result = self.factor_service.process(candidate)
            except Exception as error:  # one bad candidate must not abort the round
                attempts.append(self._failed_summary(candidate, error))
                continue
            attempts.append(self._summary(candidate, result))
            known.add(name)
            if result.record.status == "accepted":
                accepted.append(str(result.record.factor_id))
                if self.stop_on_acceptance:
                    break

        if demand is not None and demand.avoid_factor_ids and not self.factor_registry:
            notes.append("avoid_factor_ids_unresolved_no_registry")
        if len(proposals) > self.max_attempts:
            notes.append(f"truncated_proposals_{len(proposals)}_to_{self.max_attempts}")

        return FactorResearchOutcome(
            epoch_id=request.epoch_id,
            round_index=request.round_index,
            mode=request.mode,
            demand_id=demand.demand_id if demand is not None else None,
            intended_role=demand.intended_role if demand is not None else "exploration",
            target_family=target_family,
            attempted_family=family,
            feedback_digest=_feedback_digest(feedback),
            attempts=tuple(attempts),
            accepted_factor_ids=tuple(accepted),
            skipped_known_names=tuple(skipped),
            notes=tuple(notes),
            demand_evidence=dict(demand.evidence) if demand is not None else {},
        )
