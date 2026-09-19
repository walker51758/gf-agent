from __future__ import annotations

import json
import inspect

import pandas as pd
import pytest

from rdagent.app.gf_rd_loop.strategy import (
    EpochBoundStrategyService,
    MaterializingRemoteStrategyEvaluator,
    RemotePortfolioEvaluator,
    build_strategy_orchestrator,
    epoch_summary,
    load_epoch_windows,
    run_strategy_epochs,
)
from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.costs import execution_costs_metrics
from rdagent.scenarios.gf_quant.strategy.deployment import DeploymentPolicy
from rdagent.scenarios.gf_quant.strategy.orchestrator import OuterEpochConfig
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry
from rdagent.scenarios.gf_quant.strategy.reproducibility import REPRODUCIBILITY_IDENTITY_FIELDS
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
)


def _factor(factor_id: str, name: str) -> FactorRecord:
    return FactorRecord(
        factor_id=factor_id,
        name=name,
        family="fundamental",
        expression=f"cs_rank({name})",
        status="accepted",
        metrics={"Rank_IC": 0.04, "Rank_ICIR": 0.3, "coverage": 0.99},
        artifact_path=f"G:/artifacts/{name}.parquet",
        artifact_checksum=(factor_id * 64)[:64],
        metadata={"gate_failures": [], "search_direction": name},
    )


def _strategy_evaluation(candidate: StrategyCandidate) -> StrategyEvaluation:
    dates = pd.bdate_range("2022-01-03", periods=120)
    return StrategyEvaluation(
        metrics={
            **execution_costs_metrics(),
            "observations": 120,
            "avg_holdings": 20,
            "annualized_return": 0.02,
            "sharpe": 0.6,
            "max_drawdown": -0.1,
            "annual_turnover": 2.0,
            "metrics_by_year": {
                "2022": {"annualized_return": 0.02, "sharpe": 0.6},
                "2023": {"annualized_return": 0.02, "sharpe": 0.6},
            },
        },
        returns=pd.Series(0.0003, index=dates),
        factor_contributions={name: {"marginal_annualized_return": 0.01} for name in candidate.factor_names},
    )


class _ScriptedProposer:
    """Returns one fixed candidate specification per trial."""

    def __init__(self, factor_names: tuple[str, ...]) -> None:
        self.factor_names = factor_names
        self.proposed: list[StrategyCandidate] = []

    def __call__(self, context) -> StrategyCandidate:
        candidate = StrategyCandidate(
            name=f"strategy_{context.trial_index:02d}",
            universe="csi_all",
            composition=CompositionSpec(top_n=20),
            factor_names=self.factor_names,
            backtest=BacktestSpec(signal_lag=1),
            metadata={"epoch_id": context.epoch_id, "trial_index": context.trial_index},
        )
        self.proposed.append(candidate)
        return candidate


class _RecordingFactorResearcher:
    """Stands in for the demand-driven researcher; records what it is asked."""

    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.requests: list[object] = []

    def __call__(self, request) -> dict:
        self.requests.append(request)
        return self.results[min(request.round_index, len(self.results) - 1)]


def _build(tmp_path, *, factor_names: tuple[str, ...] = ("value", "quality"), weak_first_trial: bool = False):
    """Build the loop with every LLM/SSH collaborator replaced by a fake.

    ``weak_first_trial`` makes trial 0 miss the Sharpe gate, which is what makes
    the orchestrator derive a targeted factor demand from the failed trials.
    """

    factor_registry = FactorRegistry(tmp_path / "factors.sqlite")
    factor_registry.upsert(_factor("f1", "value"))
    factor_registry.upsert(_factor("f2", "quality"))

    def evaluate(candidate: StrategyCandidate) -> StrategyEvaluation:
        if weak_first_trial and candidate.metadata.get("trial_index") == 0:
            weak = _strategy_evaluation(candidate)
            metrics = dict(weak.metrics)
            metrics["sharpe"] = 0.0
            metrics["metrics_by_year"] = {
                year: {"annualized_return": 0.0, "sharpe": 0.0} for year in ("2022", "2023")
            }
            return StrategyEvaluation(**{**weak.__dict__, "metrics": metrics})
        return _strategy_evaluation(candidate)

    strategy_registry = StrategyRegistry(tmp_path / "strategies.sqlite")
    service = EpochBoundStrategyService(
        StrategyAcceptanceService(strategy_registry, None, factor_registry=factor_registry),
        lambda **_: evaluate,
    )
    researcher = _RecordingFactorResearcher(
        [{"mode": "targeted", "accepted_factor_ids": ["gf_new"], "notes": []}]
    )

    orchestrator = build_strategy_orchestrator(
        data_cutoff="2023-12-31",
        evaluation_start="2020-01-01",
        evaluation_end="2023-12-31",
        config=OuterEpochConfig(
            strategy_trials_per_epoch=2,
            working_set_core=2,
            working_set_conditional=0,
            min_factors_per_strategy=2,
            targeted_factor_rounds_max=1,
            exploratory_factor_rounds=0,
        ),
        factor_registry=factor_registry,
        strategy_registry=strategy_registry,
        proposal_engine=_ScriptedProposer(factor_names),
    )
    # Swap in the fake strategy service / researcher without rebuilding the loop.
    orchestrator.strategy_service = service
    orchestrator.factor_research_engine = researcher
    orchestrator.snapshot_dir = tmp_path / "snapshots"
    return orchestrator, factor_registry, strategy_registry, service, researcher


def test_epoch_binding_gives_the_service_the_frozen_snapshot(tmp_path) -> None:
    orchestrator, factor_registry, _registry, service, _researcher = _build(tmp_path)

    epoch = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_1")

    assert service._bound is True
    # The evaluator was re-bound for this epoch and can see the frozen snapshot.
    assert service._service.evaluator is not None
    assert epoch.snapshot.snapshot_id


def test_builder_uses_the_unconfigurable_system_execution_costs(tmp_path) -> None:
    factor_registry = FactorRegistry(tmp_path / "factors.sqlite")
    factor_registry.upsert(_factor("f1", "value"))
    factor_registry.upsert(_factor("f2", "quality"))
    registry = StrategyRegistry(tmp_path / "strategies.sqlite")
    config = OuterEpochConfig(
        strategy_trials_per_epoch=1,
        working_set_core=2,
        working_set_conditional=0,
        min_factors_per_strategy=2,
        targeted_factor_rounds_max=0,
        exploratory_factor_rounds=0,
    )

    orchestrator = build_strategy_orchestrator(
        data_cutoff="2023-12-31",
        evaluation_start="2020-01-01",
        evaluation_end="2023-12-31",
        config=config,
        factor_registry=factor_registry,
        strategy_registry=registry,
        proposal_engine=_ScriptedProposer(("value", "quality")),
    )

    assert orchestrator.strategy_service.policy.execution_costs() == execution_costs_metrics()
    assert "fixed_cost_bps" not in inspect.signature(build_strategy_orchestrator).parameters
    assert "strategy_fixed_cost_bps" not in GF_SCENARIO_SETTINGS.__class__.model_fields
    assert not hasattr(config, "fixed_cost_bps")


def _loop_kwargs(factor_registry: FactorRegistry, strategy_registry_path, **overrides) -> dict:
    kwargs = {
        "data_cutoff": "2023-12-31",
        "evaluation_start": "2020-01-01",
        "evaluation_end": "2023-12-31",
        "config": OuterEpochConfig(
            strategy_trials_per_epoch=1,
            working_set_core=2,
            working_set_conditional=0,
            min_factors_per_strategy=2,
            targeted_factor_rounds_max=0,
            exploratory_factor_rounds=0,
        ),
        "factor_registry": factor_registry,
        "strategy_registry": StrategyRegistry(strategy_registry_path),
        "proposal_engine": _ScriptedProposer(("value", "quality")),
    }
    kwargs.update(overrides)
    return kwargs


def test_builder_rejects_the_obsolete_cost_override_keyword(tmp_path) -> None:
    factor_registry = FactorRegistry(tmp_path / "factors.sqlite")
    factor_registry.upsert(_factor("f1", "value"))
    factor_registry.upsert(_factor("f2", "quality"))

    with pytest.raises(TypeError, match="fixed_cost_bps"):
        build_strategy_orchestrator(
            **_loop_kwargs(factor_registry, tmp_path / "strategies.sqlite"),
            fixed_cost_bps=1.0,
        )


def test_builder_needs_no_cost_configuration(tmp_path) -> None:
    factor_registry = FactorRegistry(tmp_path / "factors.sqlite")
    factor_registry.upsert(_factor("f1", "value"))
    factor_registry.upsert(_factor("f2", "quality"))
    orchestrator = build_strategy_orchestrator(
        **_loop_kwargs(factor_registry, tmp_path / "strategies.sqlite")
    )
    assert orchestrator.strategy_service.policy.execution_costs() == execution_costs_metrics()


def test_epoch_binding_adapts_annual_gate_to_single_year_window(tmp_path) -> None:
    orchestrator, _factors, _registry, service, _researcher = _build(tmp_path)

    orchestrator.run_epoch(
        data_cutoff="2023-12-31",
        epoch_id="single_year",
        evaluation_start="2024-01-02",
        evaluation_end="2024-12-31",
    )

    assert service._service.policy.annual_requirements(
        service._service._expected_evaluation_years
    ) == {"min_valid_years": 1, "min_positive_return_years": 0}


def test_service_refuses_to_process_before_an_epoch_is_bound(tmp_path) -> None:
    _orchestrator, factor_registry, _s, service, _r = _build(tmp_path)
    candidate = StrategyCandidate(
        name="unbound",
        universe="csi_all",
        composition=CompositionSpec(top_n=20),
        factor_names=("value", "quality"),
        backtest=BacktestSpec(signal_lag=1),
    )

    with pytest.raises(RuntimeError, match="no epoch is bound"):
        service.process(candidate)


def test_run_strategy_epochs_persists_one_report_per_epoch(tmp_path) -> None:
    orchestrator, _f, _s, _sv, _r = _build(tmp_path, weak_first_trial=True)
    report_dir = tmp_path / "reports"

    reports = run_strategy_epochs(
        orchestrator, epochs=2, data_cutoff="2023-12-31", report_dir=report_dir
    )

    assert len(reports) == 2
    assert all(report["status"] == "ok" for report in reports)
    assert all(report["strategy_trials"] == 2 for report in reports)
    written = sorted(report_dir.glob("*.json"))
    assert len(written) == 2
    for path in written:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["summary"]["strategy_trials"] == 2
        assert payload["summary"]["strategy_statuses"] == {"accepted": 1, "rejected": 1}
        assert payload["summary"]["factor_research_rounds"] == 1
        assert payload["summary"]["factor_demands"] == 1


def test_successive_epochs_see_factors_registered_by_the_previous_round(tmp_path) -> None:
    orchestrator, factor_registry, _s, _sv, _r = _build(tmp_path)

    run_strategy_epochs(orchestrator, epochs=1, data_cutoff="2023-12-31", report_dir=tmp_path / "r1")
    # Nothing new was registered by the fake researcher; add one and confirm the
    # next epoch's snapshot is refrozen rather than reused.
    factor_registry.upsert(_factor("gf_new", "new_factor"))

    epoch = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_2")
    assert "new_factor" in {factor.name for factor in epoch.snapshot.factors}


def test_failed_epoch_is_reported_without_aborting_the_run(tmp_path) -> None:
    orchestrator, _f, _s, _sv, _r = _build(tmp_path)

    def explode(**_: object) -> None:
        raise RuntimeError("remote unreachable")

    orchestrator.run_epoch = explode  # type: ignore[method-assign]
    report_dir = tmp_path / "reports"

    reports = run_strategy_epochs(
        orchestrator, epochs=2, data_cutoff="2023-12-31", report_dir=report_dir
    )

    assert [report["status"] for report in reports] == ["error", "error"]
    assert all("remote unreachable" in report["error"] for report in reports)
    assert list(report_dir.glob("*.json")) == []


def test_rejects_non_positive_epoch_count(tmp_path) -> None:
    orchestrator, _f, _s, _sv, _r = _build(tmp_path)
    with pytest.raises(ValueError):
        run_strategy_epochs(orchestrator, epochs=0, data_cutoff="2023-12-31", report_dir=tmp_path / "r")


def test_epoch_summary_shape(tmp_path) -> None:
    orchestrator, _f, _s, _sv, _r = _build(tmp_path, weak_first_trial=True)
    epoch = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_1")

    summary = epoch_summary(epoch)
    assert summary["epoch_id"] == "epoch_1"
    assert summary["strategy_statuses"] == {"accepted": 1, "rejected": 1}
    assert summary["factor_research_rounds"] == 1


def test_epoch_window_factory_advances_evaluation_provenance(tmp_path) -> None:
    orchestrator, _f, _s, service, _r = _build(tmp_path)
    seen: list[dict] = []

    def factory(**kwargs):
        seen.append(kwargs)
        return _strategy_evaluation

    service._evaluator_factory = factory
    windows = [
        {"data_cutoff": "2023-12-31", "evaluation_start": "2020-01-01", "evaluation_end": "2021-12-31"},
        {"data_cutoff": "2024-12-31", "evaluation_start": "2022-01-01", "evaluation_end": "2023-12-31"},
    ]

    reports = run_strategy_epochs(
        orchestrator,
        epochs=2,
        data_cutoff="unused",
        report_dir=tmp_path / "reports",
        epoch_window_factory=lambda index: windows[index],
    )

    assert [entry["evaluation_start"] for entry in seen] == ["2020-01-01", "2022-01-01"]
    assert [entry["evaluation_end"] for entry in seen] == ["2021-12-31", "2023-12-31"]
    assert [report["evaluation_start"] for report in reports] == ["2020-01-01", "2022-01-01"]
    assert [report["evaluation_end"] for report in reports] == ["2021-12-31", "2023-12-31"]


def test_orchestrator_resolves_builder_default_window_for_report_and_service(tmp_path) -> None:
    orchestrator, _f, _s, service, _r = _build(tmp_path)
    seen: list[tuple[str | None, str | None]] = []

    def factory(**kwargs):
        seen.append((kwargs.get("evaluation_start"), kwargs.get("evaluation_end")))
        return _strategy_evaluation

    service._evaluator_factory = factory
    epoch = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="defaults")

    assert (epoch.evaluation_start, epoch.evaluation_end) == ("2020-01-01", "2023-12-31")
    assert seen == [("2020-01-01", "2023-12-31")]


def test_load_epoch_windows_rejects_bad_dates_backwards_and_overlap(tmp_path) -> None:
    path = tmp_path / "windows.json"

    path.write_text(
        json.dumps(
            [
                {
                    "data_cutoff": "2023-12-31",
                    "evaluation_start": "2024-01-02",
                    "evaluation_end": "2024-12-31",
                },
                {
                    "data_cutoff": "2023-12-30",
                    "evaluation_start": "2025-01-02",
                    "evaluation_end": "2025-12-31",
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="moves data_cutoff backwards"):
        load_epoch_windows(path)

    path.write_text(
        json.dumps(
            [
                {
                    "data_cutoff": "2023-12-31",
                    "evaluation_start": "2024-01-02",
                    "evaluation_end": "2024-12-31",
                },
                {
                    "data_cutoff": "2024-12-31",
                    "evaluation_start": "2024-06-01",
                    "evaluation_end": "2025-12-31",
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlaps"):
        load_epoch_windows(path)

    path.write_text(
        json.dumps(
            [
                {
                    "data_cutoff": "not-a-date",
                    "evaluation_start": "2024-01-02",
                    "evaluation_end": "2024-12-31",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ISO date"):
        load_epoch_windows(path)


def test_load_epoch_windows_validates_explicit_schedule(tmp_path) -> None:
    path = tmp_path / "windows.json"
    path.write_text(
        json.dumps(
            [
                {
                    "data_cutoff": "2023-12-31",
                    "evaluation_start": "2020-01-01",
                    "evaluation_end": "2021-12-31",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_epoch_windows(path)[0]["data_cutoff"] == "2023-12-31"

    path.write_text(json.dumps([{"data_cutoff": "2023-12-31"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_epoch_windows(path)


def test_portfolio_feedback_demotes_harmful_factor_in_next_working_set(tmp_path) -> None:
    orchestrator, factor_registry, _s, _service, _r = _build(tmp_path)
    factor_registry.upsert(_factor("f3", "stable"))
    value = factor_registry.get_by_name("value")
    assert value is not None
    factor_registry.upsert(
        FactorRecord(
            **{
                **value.__dict__,
                "metrics": {"Rank_IC": 0.08, "Rank_ICIR": 0.8, "coverage": 0.99},
            }
        )
    )

    def evaluate(candidate: StrategyCandidate) -> StrategyEvaluation:
        result = _strategy_evaluation(candidate)
        contributions = {
            name: {"marginal_annualized_return": -0.2 if name == "value" else 0.1}
            for name in candidate.factor_names
        }
        return StrategyEvaluation(**{**result.__dict__, "factor_contributions": contributions})

    _service = EpochBoundStrategyService(
        StrategyAcceptanceService(StrategyRegistry(tmp_path / "feedback_strategies.sqlite"), None, factor_registry=factor_registry),
        lambda **_: evaluate,
    )
    orchestrator.strategy_service = _service
    orchestrator.factor_research_engine = None
    first = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="feedback_1")
    second = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="feedback_2")

    assert "value" in {factor.name for factor in first.working_set.factors}
    assert "value" not in {factor.name for factor in second.working_set.factors}
    assert second.working_set.portfolio_feedback


def test_remote_portfolio_evaluator_rejects_unknown_frequency(tmp_path) -> None:
    _o, factor_registry, _s, _sv, _r = _build(tmp_path)
    with pytest.raises(ValueError, match="rebalance_frequency"):
        RemotePortfolioEvaluator(
            snapshot=None,
            working_set=None,
            evaluation_start="2020-01-01",
            evaluation_end="2023-12-31",
            rebalance_frequency="hourly",
        )


def test_failed_trials_produce_a_targeted_research_request(tmp_path) -> None:
    """A gate failure in the inner loop must reach the factor researcher."""

    orchestrator, _f, _s, _sv, researcher = _build(tmp_path, weak_first_trial=True)
    orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_1")

    assert len(researcher.requests) == 1
    request = researcher.requests[0]
    assert request.mode == "targeted"
    assert request.demand is not None
    assert request.demand.intended_role == "alpha"
    assert request.demand.evidence["failed_trials"] == 1


def test_clean_trials_produce_no_targeted_request(tmp_path) -> None:
    orchestrator, _f, _s, _sv, researcher = _build(tmp_path)
    orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_1")

    assert researcher.requests == []


# ---------------------------------------------------------------------------
# Batch 9 production wiring: the governance switches and the request sections
# ---------------------------------------------------------------------------


def _governance_kwargs(tmp_path, **overrides) -> dict:
    return {
        "data_cutoff": "2023-12-31",
        "evaluation_start": "2020-01-01",
        "evaluation_end": "2023-12-31",
        "snapshot_dir": tmp_path / "snapshots",
        **overrides,
    }


def _governance_service(tmp_path, **overrides):
    factory = build_strategy_orchestrator(**_governance_kwargs(tmp_path, **overrides))
    return factory.strategy_service._service  # noqa: SLF001


def test_governance_switches_are_off_by_default(tmp_path) -> None:
    """With no manifest requested the record shape must be exactly today's."""

    service = _governance_service(tmp_path)
    assert service.emit_manifest is False
    assert service.reproducibility_check is False
    assert service.deployment_policy is None


def test_requesting_a_manifest_turns_the_governance_switches_on(tmp_path) -> None:
    service = _governance_service(tmp_path, manifest={})
    assert service.emit_manifest is True
    assert service.reproducibility_check is True
    assert service.deployment_policy is not None
    assert service.deployment_policy.thresholds_defined is False


def test_reproducibility_check_without_a_manifest_is_refused(tmp_path) -> None:
    """The check compares against the recorded manifest, so it needs one recorded."""

    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    with pytest.raises(ValueError) as excinfo:
        StrategyAcceptanceService(
            StrategyRegistry(tmp_path / "guard_probe.sqlite"),
            lambda _c: None,
            emit_manifest=False,
            reproducibility_check=True,
        )
    assert "reproducibility_check requires emit_manifest" in str(excinfo.value)


class _StubSnapshot:
    snapshot_id = "s"
    checksum = "c"
    data_cutoff = "2023-12-31"


class _StubWorkingSet:
    working_set_id = "w"
    snapshot_checksum = "c"
    snapshot_id = "s"
    factors = ()


def _bound_evaluator(orchestrator):
    """Bind one epoch so the evaluator factory runs, then hand back the evaluator."""

    orchestrator.strategy_service.bind_epoch(
        snapshot=_StubSnapshot(),
        working_set=_StubWorkingSet(),
        epoch_id="probe_epoch",
        evaluation_start="2020-01-01",
        evaluation_end="2023-12-31",
    )
    return orchestrator.strategy_service._service.evaluator  # noqa: SLF001


def test_evaluator_receives_the_request_level_sections(tmp_path) -> None:
    """The builder forwards audit / walk_forward / manifest onto the evaluator."""

    audit = {"cost_sensitivity": {"scenarios": [[10.0, 10.0]]}}
    walk_forward = {"train_size": 20, "test_size": 5}
    orchestrator = build_strategy_orchestrator(
        **_governance_kwargs(tmp_path, audit=audit, walk_forward=walk_forward, manifest={})
    )
    evaluator = _bound_evaluator(orchestrator)
    assert evaluator.audit == audit
    assert evaluator.walk_forward == walk_forward
    assert evaluator.manifest == {}


def test_evaluator_sections_default_to_not_requested(tmp_path) -> None:
    orchestrator = build_strategy_orchestrator(**_governance_kwargs(tmp_path))
    evaluator = _bound_evaluator(orchestrator)
    assert evaluator.audit is None
    assert evaluator.walk_forward is None
    assert evaluator.manifest is None


def test_default_run_adds_no_metadata_keys(tmp_path) -> None:
    """The three governance keys must be absent while the switches are off."""

    from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec

    record = _process_one_candidate(tmp_path)
    assert "manifest" not in record.metadata
    assert "reproducibility" not in record.metadata
    assert "deployment" not in record.metadata


def test_manifest_enabled_run_adds_the_three_metadata_keys(tmp_path) -> None:
    record = _process_one_candidate(tmp_path, manifest_enabled=True)
    assert isinstance(record.metadata["manifest"], dict)
    assert isinstance(record.metadata["reproducibility"], dict)
    assert isinstance(record.metadata["deployment"], dict)
    assert record.metadata["manifest"]["strategy_name"] == "governance_probe"
    assert record.metadata["deployment"]["thresholds_defined"] is False


def _process_one_candidate(tmp_path, *, manifest_enabled: bool = False, shared=None, attestation=None):
    """Drive one candidate through a service, with the governance switches set.

    ``shared`` lets two calls reuse one registry, which is what makes the second
    emission compare against the first instead of against itself.
    """

    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry, strategy_id_for

    registry = shared if shared is not None else StrategyRegistry(tmp_path / "governance.sqlite")
    candidate = StrategyCandidate(
        name="governance_probe",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="governance wiring probe",
    )
    if manifest_enabled:
        evaluator = _ManifestEvaluator(candidate, attestation=attestation)
        service = StrategyAcceptanceService(
            registry,
            evaluator,
            emit_manifest=True,
            reproducibility_check=True,
            deployment_policy=DeploymentPolicy(),
        )
    else:
        service = StrategyAcceptanceService(registry, lambda _c: _strategy_evaluation(candidate))
    result = service.process(candidate)
    assert result.record.strategy_id == strategy_id_for(candidate.name, candidate.canonical_payload())
    return result.record


class _ManifestEvaluator:
    """Minimal evaluator that also exposes the provenance facts a manifest needs."""

    def __init__(self, candidate: StrategyCandidate, *, attestation: dict | None = None, facts: dict | None = None) -> None:
        self.attestation = attestation or {
            "attestation_schema_version": 1,
            "worker_file_sha256": "deadbeef",
            "worker_runtime": {"python": "3.13.0", "numpy": "2.5.3", "pandas": "3.0.5", "pyarrow": "25.0.1"},
        }
        self._facts = facts

    def __call__(self, candidate: StrategyCandidate) -> StrategyEvaluation:
        base = _strategy_evaluation(candidate)
        return StrategyEvaluation(**{**base.__dict__, "audit_report": {"manifest": dict(self.attestation)}})

    def manifest_facts(self) -> dict:
        if self._facts is not None:
            return self._facts
        return {
            "snapshot_id": "gf_snapshot_probe",
            "snapshot_checksum": "snapshot_checksum_probe",
            "working_set_id": "working_set_probe",
            "working_set_snapshot_checksum": "snapshot_checksum_probe",
            "data_cutoff": "2023-12-31",
            "artifact_set_id": "artifact_set_probe",
            "factor_ids": [{"factor_id": "f1", "name": "value", "artifact_checksum": "abc"}],
            "evaluation": {
                "start": "2020-01-01",
                "end": "2023-12-31",
                "rebalance_frequency": "monthly",
            },
        }


def test_the_first_emission_reports_coverage_not_agreement(tmp_path) -> None:
    """With no record on hand there is nothing to agree with, and the mode says so."""

    record = _process_one_candidate(tmp_path, manifest_enabled=True)
    reproducibility = record.metadata["reproducibility"]
    assert reproducibility["mode"] == "first_emission"
    assert reproducibility["verdict"] == "reproducible"
    assert len(reproducibility["checked_fields"]) == len(REPRODUCIBILITY_IDENTITY_FIELDS)
    assert reproducibility["missing_fields"] == []


def test_the_second_emission_is_compared_against_the_recorded_manifest(tmp_path) -> None:
    """The real comparison branch: same registry, two emissions."""

    registry_dir = tmp_path / "shared"
    registry_dir.mkdir()
    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    shared = StrategyRegistry(registry_dir / "governance.sqlite")
    first = _process_one_candidate(registry_dir, manifest_enabled=True, shared=shared)
    assert first.metadata["reproducibility"]["mode"] == "first_emission"

    second = _process_one_candidate(registry_dir, manifest_enabled=True, shared=shared)
    reproducibility = second.metadata["reproducibility"]
    assert reproducibility["mode"] == "comparison"
    assert reproducibility["verdict"] == "reproducible"
    assert reproducibility["missing_fields"] == []
    assert "worker_attestation.worker_file_sha256" in reproducibility["checked_fields"]


def test_a_changed_worker_digest_makes_the_second_emission_not_reproducible(tmp_path) -> None:
    """The comparison branch is live: a moved identity field really does fail it."""

    registry_dir = tmp_path / "shared_moved"
    registry_dir.mkdir()
    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    shared = StrategyRegistry(registry_dir / "governance.sqlite")
    _process_one_candidate(registry_dir, manifest_enabled=True, shared=shared)

    moved = {
        "attestation_schema_version": 1,
        "worker_file_sha256": "cafebabe",
        "worker_runtime": {"python": "3.13.0", "numpy": "2.5.3", "pandas": "3.0.5", "pyarrow": "25.0.1"},
    }
    second = _process_one_candidate(registry_dir, manifest_enabled=True, shared=shared, attestation=moved)
    reproducibility = second.metadata["reproducibility"]
    assert reproducibility["mode"] == "comparison"
    assert reproducibility["verdict"] == "not_reproducible"
    assert "field_mismatch:worker_attestation.worker_file_sha256" in reproducibility["reasons"]


def test_absent_provenance_is_marked_not_silently_empty(tmp_path) -> None:
    """A manifest whose facts are missing must say so rather than look healthy."""

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY
    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    candidate = StrategyCandidate(
        name="governance_no_provenance",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="governance wiring probe",
    )
    registry = StrategyRegistry(tmp_path / "no_provenance.sqlite")
    service = StrategyAcceptanceService(
        registry,
        _ManifestEvaluator(candidate, facts={}),
        emit_manifest=True,
        reproducibility_check=True,
        deployment_policy=DeploymentPolicy(),
    )
    record = service.process(candidate).record
    manifest = record.metadata["manifest"]
    assert manifest[PROVENANCE_AVAILABLE_KEY] is False
    reproducibility = record.metadata["reproducibility"]
    assert reproducibility["verdict"] == "not_reproducible"
    assert "provenance_unavailable" in reproducibility["reasons"]
    # The provenance-backed fields are reported unknown, not compared empty-to-empty.
    for field in ("snapshot_checksum", "working_set_snapshot_checksum", "artifact_set_id", "evaluation"):
        assert field in reproducibility["missing_fields"]
    # Only the fields that genuinely carry a value are counted as checked.
    assert set(reproducibility["checked_fields"]) == {
        "strategy_id",
        "execution_costs",
        "worker_attestation.worker_file_sha256",
        "worker_attestation.worker_runtime",
    }


def test_an_evaluator_without_manifest_facts_is_marked_not_silently_empty(tmp_path) -> None:
    """The wrapper that cannot answer the provenance question reports it as absent."""

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY
    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    candidate = StrategyCandidate(
        name="governance_no_hook",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="governance wiring probe",
    )
    registry = StrategyRegistry(tmp_path / "no_hook.sqlite")
    service = StrategyAcceptanceService(
        registry,
        lambda c: _strategy_evaluation(c),
        emit_manifest=True,
        reproducibility_check=True,
        deployment_policy=DeploymentPolicy(),
    )
    record = service.process(candidate).record
    assert record.metadata["manifest"][PROVENANCE_AVAILABLE_KEY] is False


def _complete_facts(**overrides) -> dict:
    """A provenance fact set with every field carried, minus the overrides."""

    return {
        "snapshot_id": "gf_snapshot_probe",
        "snapshot_checksum": "snapshot_checksum_probe",
        "working_set_id": "working_set_probe",
        "working_set_snapshot_checksum": "snapshot_checksum_probe",
        "data_cutoff": "2023-12-31",
        "artifact_set_id": "artifact_set_probe",
        "factor_ids": [{"factor_id": "f1", "name": "value", "artifact_checksum": "abc"}],
        "evaluation": {"start": "2020-01-01", "end": "2023-12-31", "rebalance_frequency": "monthly"},
        **overrides,
    }


def _facts_service(tmp_path, facts: dict, *, name: str, shared=None):
    """A governance-enabled service whose evaluator reports ``facts`` verbatim."""

    candidate = StrategyCandidate(
        name=name,
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="governance null-provenance probe",
    )
    registry = shared if shared is not None else StrategyRegistry(tmp_path / f"{name}.sqlite")
    service = StrategyAcceptanceService(
        registry,
        _ManifestEvaluator(candidate, facts=facts),
        emit_manifest=True,
        reproducibility_check=True,
        deployment_policy=DeploymentPolicy(),
    )
    return service, candidate


def test_a_complete_fact_set_keeps_the_manifest_unchanged(tmp_path) -> None:
    """The null normalisation must not touch a manifest that carries every fact."""

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY

    service, candidate = _facts_service(tmp_path, _complete_facts(), name="governance_complete_facts")
    record = service.process(candidate).record
    manifest = record.metadata["manifest"]
    assert manifest["snapshot_checksum"] == "snapshot_checksum_probe"
    assert manifest["working_set_snapshot_checksum"] == "snapshot_checksum_probe"
    assert manifest["artifact_set_id"] == "artifact_set_probe"
    assert manifest["factor_ids"] == [{"factor_id": "f1", "name": "value", "artifact_checksum": "abc"}]
    assert manifest["evaluation"]["rebalance_frequency"] == "monthly"
    assert manifest[PROVENANCE_AVAILABLE_KEY] is True
    assert record.metadata["reproducibility"]["verdict"] == "reproducible"


@pytest.mark.parametrize(
    ("field", "empty"),
    [
        ("snapshot_checksum", ""),
        ("working_set_snapshot_checksum", ""),
        ("artifact_set_id", ""),
        ("evaluation", {}),
    ],
)
def test_a_null_provenance_fact_is_read_as_absent_not_as_a_value(tmp_path, field, empty) -> None:
    """``None`` in a fact is an absent fact, never the string ``"None"``.

    ``facts.get(key, default)`` substitutes its default only when the key is
    missing, so a key present with ``None`` used to be recorded as the literal
    ``"None"`` (or crash the builder outright).  ``"None"`` is a value
    ``_is_carried`` counts as carried, which let two emissions agree on a field
    neither of them carries -- hence the second emission asserted here.

    These four are members of ``REPRODUCIBILITY_IDENTITY_FIELDS``, so absence is
    reported *per field* and the manifest-level flag stays ``True``; the
    unpoliced facts are covered by the test below.
    """

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY

    registry_dir = tmp_path / f"null_{field}"
    registry_dir.mkdir()
    shared = StrategyRegistry(registry_dir / "governance.sqlite")
    name = f"governance_null_{field}"
    service, candidate = _facts_service(registry_dir, _complete_facts(**{field: None}), name=name, shared=shared)

    first = service.process(candidate).record
    assert first.metadata["manifest"][field] == empty
    assert first.metadata["manifest"][PROVENANCE_AVAILABLE_KEY] is True
    reproducibility = first.metadata["reproducibility"]
    assert reproducibility["verdict"] == "not_reproducible"
    assert f"field_unknown:{field}" in reproducibility["reasons"]
    assert field in reproducibility["missing_fields"]

    second = service.process(candidate).record
    comparison = second.metadata["reproducibility"]
    assert comparison["mode"] == "comparison"
    assert comparison["verdict"] == "not_reproducible"
    assert f"field_unknown:{field}" in comparison["reasons"]


def test_an_absent_factor_list_degrades_instead_of_raising(tmp_path) -> None:
    """``factor_ids=None`` is a provenance gap, not a crash.

    ``factor_ids`` is not part of the reproducibility identity set, so an empty
    table would never be reported downstream; the manifest-level
    ``provenance_available`` flag is the remaining carrier, and it forces
    ``not_reproducible`` rather than letting the record read as "this strategy
    references no factors".
    """

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY

    service, candidate = _facts_service(
        tmp_path, _complete_facts(factor_ids=None), name="governance_null_factors"
    )
    record = service.process(candidate).record
    manifest = record.metadata["manifest"]
    assert manifest["factor_ids"] == []
    assert manifest[PROVENANCE_AVAILABLE_KEY] is False
    reproducibility = record.metadata["reproducibility"]
    assert reproducibility["verdict"] == "not_reproducible"
    assert "provenance_unavailable" in reproducibility["reasons"]
    # The other facts are still recorded: the gap is scoped, not a wipe.
    assert manifest["snapshot_checksum"] == "snapshot_checksum_probe"


@pytest.mark.parametrize(
    ("field", "empty"),
    [
        ("snapshot_id", ""),
        ("working_set_id", ""),
        ("data_cutoff", ""),
        ("factor_ids", []),
    ],
)
def test_a_null_unpoliced_fact_degrades_instead_of_vanishing(tmp_path, field, empty) -> None:
    """Facts no downstream check polices are reported by the manifest-level flag.

    ``snapshot_id`` / ``working_set_id`` / ``data_cutoff`` / ``factor_ids`` are not
    members of ``REPRODUCIBILITY_IDENTITY_FIELDS``, so ``_is_carried`` never names
    them: a ``None`` here would be recorded as the empty shape and read as a real
    -- if empty -- value, and nothing would object.  ``provenance_available`` is
    the remaining carrier, and it forces ``not_reproducible``.  The second
    emission is asserted for the same reason as in the per-field test: the flag
    has to survive the comparison, not only the first emission.

    The pin that keeps ``None`` off the production path lives in the dataclass
    guards (``snapshot_id``, ``working_set_id``) and in the builder that rejects a
    falsy cutoff (``data_cutoff``); this test exercises the fail-closed reading
    that backs those guards up.
    """

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY

    registry_dir = tmp_path / f"unpoliced_{field}"
    registry_dir.mkdir()
    shared = StrategyRegistry(registry_dir / "governance.sqlite")
    name = f"governance_unpoliced_{field}"
    service, candidate = _facts_service(registry_dir, _complete_facts(**{field: None}), name=name, shared=shared)

    first = service.process(candidate).record
    manifest = first.metadata["manifest"]
    assert manifest[field] == empty
    assert manifest[PROVENANCE_AVAILABLE_KEY] is False
    reproducibility = first.metadata["reproducibility"]
    assert reproducibility["verdict"] == "not_reproducible"
    assert "provenance_unavailable" in reproducibility["reasons"]
    # Nothing names the field: that is exactly why the flag has to carry it.
    assert field not in reproducibility["missing_fields"]
    assert not any(str(reason).endswith(f":{field}") for reason in reproducibility["reasons"])

    second = service.process(candidate).record
    comparison = second.metadata["reproducibility"]
    assert comparison["mode"] == "comparison"
    assert comparison["verdict"] == "not_reproducible"


def test_the_wrapper_keeps_the_inner_evaluator_for_provenance() -> None:
    """F1: the wrapper must be able to answer the provenance question.

    ``MaterializingRemoteStrategyEvaluator`` builds a fresh inner evaluator per
    candidate; keeping it is what lets the manifest carry real snapshot /
    working-set / artifact-set facts instead of empty strings.
    """

    snapshot = _StubSnapshot()
    working_set = _StubWorkingSet()
    wrapper = _wrapper_with_stubbed_artifacts(
        snapshot=snapshot,
        working_set=working_set,
        remote_evaluator_factory=_ProvenanceInnerEvaluator,
    )
    assert callable(getattr(wrapper, "manifest_facts", None))
    # Before any evaluation there is nothing to report, and that must be an error
    # rather than an empty facts object.
    with pytest.raises(ValueError, match="no provenance yet"):
        wrapper.manifest_facts()

    candidate = StrategyCandidate(
        name="wrapper_provenance",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="wrapper provenance probe",
    )
    wrapper(candidate)
    facts = wrapper.manifest_facts()
    assert facts["snapshot_id"] == "s"
    assert facts["working_set_id"] == "w"
    assert facts["evaluation"]["rebalance_frequency"] == "monthly"
    assert facts["factor_ids"]


def test_the_wrapper_reports_an_inner_evaluator_without_facts_as_absent() -> None:
    """A substitute evaluator that cannot supply facts must not be cached as a fact source."""

    class _NoFacts:
        def __init__(self, _artifact_set):
            pass

        def __call__(self, candidate, audit=None, walk_forward=None, manifest=None):
            return _strategy_evaluation(candidate)

    wrapper = _wrapper_with_stubbed_artifacts(
        snapshot=_StubSnapshot(),
        working_set=_StubWorkingSet(),
        remote_evaluator_factory=_NoFacts,
    )
    candidate = StrategyCandidate(
        name="wrapper_no_facts",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="wrapper provenance probe",
    )
    wrapper(candidate)
    # The wrapper reports *its own* message here.  ``__call__`` only caches an
    # inner evaluator that can answer the provenance question, so a substitute
    # that cannot is not remembered and the wrapper sees "nothing evaluated
    # yet" -- ``does not provide manifest facts`` is the service layer's
    # message (``StrategyAcceptanceService._manifest_facts``), and a disjunctive
    # match would accept it, hiding which layer actually fired.
    with pytest.raises(ValueError, match="no provenance yet") as excinfo:
        wrapper.manifest_facts()
    assert str(excinfo.value).startswith("strategy evaluator has no provenance yet")
    assert "does not provide manifest facts" not in str(excinfo.value)


class _StubArtifact:
    name = "value"
    factor_id = "f1"
    artifact_checksum = "abc"


class _StubArtifactSet:
    artifact_set_id = "artifact_set_probe"


class _StubArtifactCache:
    """Replaces SSH-backed materialization for the provenance tests."""

    def artifact_set_for(self, _factor_names):
        return _StubArtifactSet()


def _wrapper_with_stubbed_artifacts(**kwargs):
    """A real ``MaterializingRemoteStrategyEvaluator`` with materialization stubbed."""

    wrapper = MaterializingRemoteStrategyEvaluator(evaluation_start="2020-01-01", evaluation_end="2023-12-31", **kwargs)
    wrapper.artifact_cache = _StubArtifactCache()
    return wrapper


class _ProvenanceInnerEvaluator:
    """Stands in for ``RemoteStrategyEvaluator``: it can report its provenance."""
    def __init__(self, artifact_set) -> None:
        self.artifact_set = artifact_set

    def manifest_facts(self) -> dict:
        return {
            "snapshot_id": "s",
            "snapshot_checksum": "c",
            "working_set_id": "w",
            "working_set_snapshot_checksum": "c",
            "data_cutoff": "2023-12-31",
            "artifact_set_id": getattr(self.artifact_set, "artifact_set_id", "artifact_set_probe"),
            "factor_ids": [{"factor_id": "f1", "name": "value", "artifact_checksum": "abc"}],
            "evaluation": {"start": "2020-01-01", "end": "2023-12-31", "rebalance_frequency": "monthly"},
        }

    def __call__(self, candidate, audit=None, walk_forward=None, manifest=None):
        return _strategy_evaluation(candidate)


def test_rejected_candidates_are_traced_by_governance(tmp_path) -> None:
    """Roadmap 10.10 lists reject among its outputs; it must be reachable."""

    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    candidate = StrategyCandidate(
        name="governance_rejected",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="governance wiring probe",
    )
    registry = StrategyRegistry(tmp_path / "rejected.sqlite")

    def weak_evaluation(_candidate: StrategyCandidate) -> StrategyEvaluation:
        base = _strategy_evaluation(_candidate)
        # Strip the metrics the acceptance policy requires, so the gate fails.
        return StrategyEvaluation(**{**base.__dict__, "metrics": {"execution_costs": {}}})

    service = StrategyAcceptanceService(
        registry,
        weak_evaluation,
        emit_manifest=True,
        reproducibility_check=True,
        deployment_policy=DeploymentPolicy(),
    )
    record = service.process(candidate).record
    assert record.status == "rejected"
    assert record.metadata["gate_failures"]
    # The three governance keys are present on the rejected path too.
    assert isinstance(record.metadata["manifest"], dict)
    assert isinstance(record.metadata["reproducibility"], dict)
    deployment = record.metadata["deployment"]
    assert deployment["verdict"] == "reject"
    assert "acceptance_gate_failed" in deployment["reasons"]
    assert deployment["gate_failures"] == list(record.metadata["gate_failures"])
    assert record.metadata["manifest"]["acceptance"]["status"] == "rejected"
    # "Traced" means durable: the stored record must carry them too, not just the
    # object handed back to the caller.
    stored = registry.get(record.strategy_id)
    assert stored is not None
    for key in ("manifest", "reproducibility", "deployment"):
        assert key in stored.metadata, f"{key} was not persisted"


def test_a_clean_audit_with_a_failed_gate_still_rejects(tmp_path) -> None:
    """Acceptance failure is a hard defect on its own.

    The candidate here has complete evidence and a healthy return profile, so
    the *only* reason to withhold deployment is the acceptance gate.  Without
    this case, a mutation that ignored ``gate_failures`` would be masked by the
    other failures a stripped evaluation produces.
    """

    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    candidate = StrategyCandidate(
        name="governance_gate_only",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=("value", "quality"),
        rule="governance wiring probe",
    )
    registry = StrategyRegistry(tmp_path / "gate_only.sqlite")

    def audit_only_failure(_candidate: StrategyCandidate) -> StrategyEvaluation:
        base = _strategy_evaluation(_candidate)
        return StrategyEvaluation(
            **{
                **base.__dict__,
                "audit_report": {
                    "walk_forward": {"segments": [1]},
                    "feasibility": {"violations": []},
                    "cost_sensitivity": {"scenarios": [1]},
                    "data_leakage": {"findings": []},
                    "manifest": {"worker_file_sha256": "a" * 64, "worker_runtime": {"python": "3.13.0"}},
                },
            }
        )

    # A policy whose first requirement cannot be met by the probe metrics.
    policy = StrategyAcceptancePolicy(min_annualized_return=999.0)
    service = StrategyAcceptanceService(
        registry,
        audit_only_failure,
        policy=policy,
        emit_manifest=True,
        reproducibility_check=True,
        deployment_policy=DeploymentPolicy(),
    )
    record = service.process(candidate).record
    assert record.status == "rejected"
    assert record.metadata["gate_failures"] == ["annualized_return"]
    deployment = record.metadata["deployment"]
    assert deployment["verdict"] == "reject"
    assert "acceptance_gate_failed" in deployment["reasons"]
    # The evidence was complete, so nothing else could have caused the reject.
    assert deployment["missing_evidence"] == []
    assert deployment["leakage_breaches"] == []
    assert deployment["feasibility_violations"] == []


def test_governance_reports_do_not_change_the_record_shape(tmp_path) -> None:
    """The reports live in metadata only: no new column and no new status value."""

    plain = _process_one_candidate(tmp_path / "plain")
    governed = _process_one_candidate(tmp_path / "governed", manifest_enabled=True)
    governed_metadata = {k: v for k, v in governed.metadata.items() if k not in ("manifest", "reproducibility", "deployment")}
    plain_metadata = {k: v for k, v in plain.metadata.items()}
    # The dedup reason differs only because the two runs used different
    # registries; every shared key must otherwise be identical.
    assert set(governed_metadata) == set(plain_metadata)
    assert governed.strategy_id == plain.strategy_id
    assert governed.status == plain.status == "accepted"


def test_load_json_object_rejects_a_non_object(tmp_path) -> None:
    from rdagent.app.gf_rd_loop.strategy import _load_json_object

    assert _load_json_object(None, "audit_config") is None
    path = tmp_path / "audit.json"
    path.write_text('{"cost_sensitivity": {}}', encoding="utf-8")
    assert _load_json_object(path, "audit_config") == {"cost_sensitivity": {}}
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="audit_config file must contain a JSON object"):
        _load_json_object(bad, "audit_config")
