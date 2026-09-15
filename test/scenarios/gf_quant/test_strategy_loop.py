from __future__ import annotations

import json
import inspect

import pandas as pd
import pytest

from rdagent.app.gf_rd_loop.strategy import (
    EpochBoundStrategyService,
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
from rdagent.scenarios.gf_quant.strategy.orchestrator import OuterEpochConfig
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry
from rdagent.scenarios.gf_quant.strategy.service import StrategyAcceptanceService, StrategyEvaluation


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
