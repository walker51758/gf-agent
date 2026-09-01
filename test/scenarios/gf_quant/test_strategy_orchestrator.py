from __future__ import annotations

import pandas as pd

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.orchestrator import GFQuantOrchestrator, OuterEpochConfig
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
)
from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot


def _factor(factor_id: str, name: str, *, status: str = "accepted") -> FactorRecord:
    return FactorRecord(
        factor_id=factor_id,
        name=name,
        family="fundamental",
        expression=f"cs_rank({name})",
        status=status,
        metrics={"Rank_IC": 0.03, "Rank_ICIR": 0.3, "coverage": 0.99},
        artifact_path=f"G:/artifacts/{name}.parquet",
        artifact_checksum=(factor_id * 64)[:64],
        metadata={"gate_failures": [] if status == "accepted" else ["rank_ic"], "search_direction": name},
    )


def test_outer_epoch_freezes_factors_then_emits_demand_and_researches(tmp_path) -> None:
    factors = FactorRegistry(tmp_path / "factors.sqlite")
    factors.upsert(_factor("f1", "value"))
    factors.upsert(_factor("f2", "quality"))
    factors.upsert(_factor("f3", "cashflow", status="rejected"))

    def evaluate(candidate: StrategyCandidate) -> StrategyEvaluation:
        dates = pd.bdate_range("2022-01-03", periods=120)
        returns = pd.Series(0.0002, index=dates)
        return StrategyEvaluation(
            metrics={
                "observations": 120,
                "avg_holdings": 20,
                "annualized_return": -0.01,
                "sharpe": 0.5,
                "max_drawdown": -0.1,
                "annual_turnover": 2.0,
                "metrics_by_year": {
                    "2022": {"annualized_return": -0.01, "sharpe": 0.5},
                    "2023": {"annualized_return": -0.01, "sharpe": 0.5},
                },
            },
            returns=returns,
            factor_contributions={
                candidate.factor_names[0]: {"marginal_annualized_return": -0.02, "marginal_sharpe": -0.1},
                candidate.factor_names[1]: {"marginal_annualized_return": 0.01, "marginal_sharpe": 0.1},
            },
        )

    service = StrategyAcceptanceService(
        StrategyRegistry(tmp_path / "strategies.sqlite"),
        evaluate,
        policy=StrategyAcceptancePolicy(
            min_observations=100,
            min_avg_holdings=5,
            min_sharpe=0.3,
            min_annualized_return=0.0,
            min_valid_years=2,
        ),
    )

    def propose(context):
        names = tuple(factor.name for factor in context.working_set.factors[:2])
        return StrategyCandidate(
            name=f"strategy_{context.trial_index}",
            universe="csi_all",
            composition=CompositionSpec(top_n=20),
            factor_names=names,
            backtest=BacktestSpec(signal_lag=1, cost_bps=10),
        )

    requests = []

    def research(request):
        requests.append(request)
        new_id = f"new{request.round_index}"
        factors.upsert(_factor(new_id, new_id))
        return {"factor_id": new_id, "mode": request.mode}

    orchestrator = GFQuantOrchestrator(
        factor_registry=factors,
        strategy_service=service,
        proposal_engine=propose,
        factor_research_engine=research,
        config=OuterEpochConfig(
            strategy_trials_per_epoch=2,
            targeted_factor_rounds_max=1,
            exploratory_factor_rounds=1,
            working_set_core=2,
            working_set_conditional=1,
            min_factors_per_strategy=2,
        ),
        snapshot_dir=tmp_path / "snapshots",
    )

    epoch = orchestrator.run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_1")

    assert len(epoch.strategy_trials) == 2
    assert all(trial.result.record.status == "rejected" for trial in epoch.strategy_trials)
    assert epoch.factor_demands
    assert {request.mode for request in requests} == {"targeted", "exploratory"}
    assert all(factor.factor_id not in {"new0", "new1"} for factor in epoch.snapshot.factors)
    assert (tmp_path / "snapshots" / f"{epoch.snapshot.snapshot_id}.json").exists()

    next_snapshot = build_factor_universe_snapshot(factors, data_cutoff="2024-01-31")
    assert {"new0", "new1"}.issubset({factor.factor_id for factor in next_snapshot.factors})


def test_out_of_snapshot_factor_is_rejected_before_evaluation(tmp_path) -> None:
    factors = FactorRegistry(tmp_path / "factors.sqlite")
    factors.upsert(_factor("f1", "value"))
    factors.upsert(_factor("f2", "quality"))

    evaluated = []

    def evaluate(candidate):
        evaluated.append(candidate)
        raise AssertionError("must not evaluate an invalid proposal")

    service = StrategyAcceptanceService(StrategyRegistry(tmp_path / "strategies.sqlite"), evaluate)

    def propose(context):
        return StrategyCandidate(
            name="invalid",
            universe="csi_all",
            composition=CompositionSpec(top_n=10),
            factor_names=(context.working_set.factors[0].name, "future_factor"),
        )

    epoch = GFQuantOrchestrator(
        factor_registry=factors,
        strategy_service=service,
        proposal_engine=propose,
        config=OuterEpochConfig(strategy_trials_per_epoch=1, working_set_core=2, working_set_conditional=0),
    ).run_epoch(data_cutoff="2023-12-31")

    assert not evaluated
    assert epoch.strategy_trials[0].result is None
    assert "outside the frozen working set" in epoch.strategy_trials[0].error
