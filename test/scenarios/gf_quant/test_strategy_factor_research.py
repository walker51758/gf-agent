from __future__ import annotations

import json

import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.service import FactorCandidate, ProcessingResult
from rdagent.scenarios.gf_quant.strategy.factor_research import (
    DemandDrivenFactorResearchEngine,
    render_demand_feedback,
)
from rdagent.scenarios.gf_quant.strategy.orchestrator import FactorDemand, FactorResearchRequest


def _record(factor_id: str, name: str, *, status: str = "accepted", family: str = "fundamental") -> FactorRecord:
    return FactorRecord(
        factor_id=factor_id,
        name=name,
        family=family,
        expression=f"cs_rank({name})",
        status=status,
        metrics={"Rank_IC": 0.05},
    )


class _FakeService:
    """Records what it was asked to register and returns canned results."""

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.calls: list[FactorCandidate] = []

    def process(self, candidate: FactorCandidate, **_: object) -> ProcessingResult:
        self.calls.append(candidate)
        status = self.statuses[min(len(self.calls) - 1, len(self.statuses) - 1)]
        factor_id = f"gf_{candidate.name}"
        return ProcessingResult(
            _record(factor_id, candidate.name, status=status, family=candidate.family),
            gate_failures=() if status == "accepted" else ("rank_ic",),
        )


class _FakeProposer:
    def __init__(self, proposals: list[FactorCandidate] | Exception) -> None:
        self.proposals = proposals
        self.seen_feedback: list[str] = []

    def propose(self, family: str, feedback: str | None = None) -> list[FactorCandidate]:
        self.seen_feedback.append(feedback or "")
        if isinstance(self.proposals, Exception):
            raise self.proposals
        return [item.model_copy() if hasattr(item, "model_copy") else item for item in self.proposals]


def _request(demand: FactorDemand | None, *, round_index: int = 0, mode: str = "targeted") -> FactorResearchRequest:
    return FactorResearchRequest(
        epoch_id="epoch_1",
        round_index=round_index,
        mode=mode,
        demand=demand,
        frozen_snapshot_id="gf_snapshot_x",
    )


def _demand(**overrides: object) -> FactorDemand:
    values: dict[str, object] = {
        "intended_role": "alpha",
        "priority": 2.0,
        "rationale": "return_or_sharpe_gate_failed",
        "evidence": {"failed_trials": 3},
    }
    values.update(overrides)
    return FactorDemand.create(epoch_id="epoch_1", **values)


def _candidate(name: str, family: str = "fundamental") -> FactorCandidate:
    return FactorCandidate(name=name, family=family, expression=f"cs_rank({name})")


def test_demand_targets_requested_family_and_registers_accepted_factor(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    service = _FakeService(["accepted"])
    proposer = _FakeProposer([_candidate("cash_quality")])

    outcome = DemandDrivenFactorResearchEngine(
        proposer=proposer, factor_service=service, factor_registry=registry
    )(_request(_demand(target_family="price_volume")))

    assert outcome.attempted_family == "price_volume"
    assert outcome.accepted_factor_ids == ("gf_cash_quality",)
    assert outcome.attempts[0].status == "accepted"
    assert outcome.demand_id.startswith("gf_factor_demand_")
    assert not outcome.notes


def test_unknown_target_family_falls_back_to_default() -> None:
    proposer = _FakeProposer([])
    service = _FakeService([])

    outcome = DemandDrivenFactorResearchEngine(
        proposer=proposer, factor_service=service, default_family="price_volume"
    )(_request(_demand(target_family="any")))

    assert outcome.attempted_family == "price_volume"
    assert "no_parseable_proposal" in outcome.notes


def test_known_factor_names_are_skipped_not_reprocessed(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    registry.upsert(_record("gf_existing", "existing"))
    service = _FakeService(["accepted", "accepted"])
    proposer = _FakeProposer([_candidate("existing"), _candidate("fresh")])

    outcome = DemandDrivenFactorResearchEngine(
        proposer=proposer, factor_service=service, factor_registry=registry
    )(_request(_demand()))

    # The known name never reaches the service: a different expression under the
    # same name could not be registered at all.
    assert [call.name for call in service.calls] == ["fresh"]
    assert outcome.skipped_known_names == ("existing",)
    assert outcome.accepted_factor_ids == ("gf_fresh",)


def test_stop_on_acceptance_bounds_the_round(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    service = _FakeService(["rejected", "accepted", "accepted"])
    proposer = _FakeProposer([_candidate("a"), _candidate("b"), _candidate("c")])

    outcome = DemandDrivenFactorResearchEngine(
        proposer=proposer, factor_service=service, factor_registry=registry
    )(_request(_demand()))

    assert [call.name for call in service.calls] == ["a", "b"]
    assert outcome.accepted_factor_ids == ("gf_b",)
    assert [attempt.status for attempt in outcome.attempts] == ["rejected", "accepted"]


def test_all_attempts_run_when_stop_on_acceptance_is_disabled(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    service = _FakeService(["accepted", "accepted"])
    proposer = _FakeProposer([_candidate("a"), _candidate("b")])

    outcome = DemandDrivenFactorResearchEngine(
        proposer=proposer,
        factor_service=service,
        factor_registry=registry,
        stop_on_acceptance=False,
    )(_request(_demand()))

    assert len(outcome.accepted_factor_ids) == 2


def test_avoid_factor_ids_are_rendered_into_feedback(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    registry.upsert(_record("gf_dead", "dead_factor"))
    service = _FakeService(["accepted"])
    proposer = _FakeProposer([_candidate("replacement")])

    DemandDrivenFactorResearchEngine(
        proposer=proposer, factor_service=service, factor_registry=registry
    )(_request(_demand(avoid_factor_ids=("gf_dead", "gf_gone"))))

    feedback = proposer.seen_feedback[0]
    assert "dead_factor" in feedback  # resolved to the registry name
    assert "gf_gone" in feedback  # unresolvable id is still reported


def test_exploratory_round_has_no_demand_narrative() -> None:
    proposer = _FakeProposer([_candidate("exploration")])
    service = _FakeService(["accepted"])

    outcome = DemandDrivenFactorResearchEngine(proposer=proposer, factor_service=service)(
        _request(None, round_index=3, mode="exploratory")
    )

    assert outcome.demand_id is None
    assert outcome.intended_role == "exploration"
    assert outcome.mode == "exploratory"
    assert "dead_factor" not in proposer.seen_feedback[0]
    assert "Exploratory round" in proposer.seen_feedback[0]


def test_proposer_failure_does_not_abort_the_round() -> None:
    outcome = DemandDrivenFactorResearchEngine(
        proposer=_FakeProposer(RuntimeError("llm down")),
        factor_service=_FakeService([]),
    )(_request(_demand()))

    assert outcome.attempts == ()
    assert any(note.startswith("proposal_failed:") for note in outcome.notes)


def test_service_failure_becomes_a_recorded_error_not_an_exception(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")

    class _ExplodingService:
        def process(self, candidate: FactorCandidate, **_: object) -> ProcessingResult:
            raise ValueError("bad expression")

    outcome = DemandDrivenFactorResearchEngine(
        proposer=_FakeProposer([_candidate("broken")]),
        factor_service=_ExplodingService(),
        factor_registry=registry,
    )(_request(_demand()))

    assert outcome.attempts[0].status == "error"
    assert "bad expression" in (outcome.attempts[0].error or "")


def test_max_attempts_truncates_long_proposal_lists(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    service = _FakeService(["rejected"])
    proposer = _FakeProposer([_candidate(name) for name in ("a", "b", "c", "d", "e")])

    outcome = DemandDrivenFactorResearchEngine(
        proposer=proposer,
        factor_service=service,
        factor_registry=registry,
        max_attempts=2,
        stop_on_acceptance=False,
    )(_request(_demand()))

    assert len(outcome.attempts) == 2
    assert "truncated_proposals_5_to_2" in outcome.notes


def test_outcome_is_json_round_trippable(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    outcome = DemandDrivenFactorResearchEngine(
        proposer=_FakeProposer([_candidate("json_safe")]),
        factor_service=_FakeService(["accepted"]),
        factor_registry=registry,
    )(_request(_demand(avoid_factor_ids=("gf_dead",))))

    payload = outcome.as_payload()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert payload["accepted_factor_ids"] == ["gf_json_safe"]


def test_feedback_digest_is_stable_for_identical_requests() -> None:
    first = render_demand_feedback(_request(_demand()))
    second = render_demand_feedback(_request(_demand()))
    assert first == second
    assert "return_or_sharpe_gate_failed" in first


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        DemandDrivenFactorResearchEngine(
            proposer=_FakeProposer([]), factor_service=_FakeService([]), default_family="momentum"
        )
    with pytest.raises(ValueError):
        DemandDrivenFactorResearchEngine(
            proposer=_FakeProposer([]), factor_service=_FakeService([]), max_attempts=0
        )
