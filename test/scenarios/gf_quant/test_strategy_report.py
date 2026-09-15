from __future__ import annotations

import json
import math

import pandas as pd

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.costs import execution_costs_metrics
from rdagent.scenarios.gf_quant.strategy.orchestrator import (
    GFQuantOrchestrator,
    OuterEpochConfig,
    OuterEpochResult,
)
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry
from rdagent.scenarios.gf_quant.strategy.report import epoch_report, write_epoch_report
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
)


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


def _evaluation(candidate: StrategyCandidate) -> StrategyEvaluation:
    dates = pd.bdate_range("2022-01-03", periods=120)
    return StrategyEvaluation(
        metrics={
            **execution_costs_metrics(),
            "observations": 120,
            "avg_holdings": 20,
            "annualized_return": -0.01,
            "sharpe": 0.5,
            "max_drawdown": -0.1,
            "annual_turnover": 2.0,
            "signal_lag": candidate.backtest.signal_lag,
            "degenerate": float("nan"),
            "infinite": float("inf"),
            "metrics_by_year": {
                "2022": {"annualized_return": -0.01, "sharpe": float("nan"), "noise": float("inf")},
                "2023": {"annualized_return": -0.01, "sharpe": 0.5},
            },
        },
        returns=pd.Series(0.0002, index=dates),
        factor_contributions={
            candidate.factor_names[0]: {"marginal_annualized_return": -0.02, "marginal_sharpe": -0.1},
            candidate.factor_names[1]: {"marginal_annualized_return": 0.01, "marginal_sharpe": 0.1},
        },
    )


def _run_epoch(
    tmp_path, *, research_results: tuple[object, ...] = (), policy: StrategyAcceptancePolicy | None = None
) -> OuterEpochResult:
    factors = FactorRegistry(tmp_path / "factors.sqlite")
    factors.upsert(_factor("f1", "value"))
    factors.upsert(_factor("f2", "quality"))

    service = StrategyAcceptanceService(
        StrategyRegistry(tmp_path / "strategies.sqlite"), _evaluation, policy=policy
    )

    def propose(context):
        names = tuple(factor.name for factor in context.working_set.factors[:2])
        return StrategyCandidate(
            name=f"strategy_{context.trial_index}",
            universe="csi_all",
            composition=CompositionSpec(top_n=20),
            factor_names=names,
            backtest=BacktestSpec(signal_lag=1),
        )

    rounds = iter(research_results * 4)

    def research(request):
        return next(rounds)

    return GFQuantOrchestrator(
        factor_registry=factors,
        strategy_service=service,
        proposal_engine=propose,
        factor_research_engine=research if research_results else None,
        config=OuterEpochConfig(
            strategy_trials_per_epoch=2,
            working_set_core=2,
            working_set_conditional=0,
            min_factors_per_strategy=2,
            targeted_factor_rounds_max=1 if research_results else 0,
            exploratory_factor_rounds=0,
        ),
        snapshot_dir=tmp_path / "snapshots",
    ).run_epoch(data_cutoff="2023-12-31", epoch_id="epoch_1")


def test_report_is_json_serializable_with_non_finite_metrics(tmp_path) -> None:
    epoch = _run_epoch(tmp_path)
    report = epoch_report(epoch)

    # NaN/inf would make json.dumps raise; they must be coerced, not dropped.
    encoded = json.dumps(report, ensure_ascii=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert report["strategy_trials"][0]["metrics_by_year"]["2022"]["sharpe"] is None
    assert report["strategy_trials"][0]["metrics_by_year"]["2023"]["sharpe"] == 0.5


def test_report_covers_every_layer(tmp_path) -> None:
    epoch = _run_epoch(tmp_path)
    report = epoch_report(epoch)

    assert report["schema_version"] == 1
    assert report["epoch_id"] == "epoch_1"
    assert report["snapshot"]["snapshot_id"] == epoch.snapshot.snapshot_id
    assert report["snapshot"]["data_cutoff"] == "2023-12-31"
    assert report["working_set"]["working_set_id"] == epoch.working_set.working_set_id
    assert report["summary"]["strategy_trials"] == 2
    assert report["summary"]["strategy_statuses"] == {"rejected": 2}
    assert report["summary"]["factor_demands"] == len(epoch.factor_demands)

    trial = report["strategy_trials"][0]
    assert sorted(trial["factor_names"]) == ["quality", "value"]
    assert trial["signal_lag"] == 1
    assert trial["execution_costs"] == execution_costs_metrics()
    assert report["execution_costs"] == execution_costs_metrics()
    assert trial["annual_gate_requirements"] == {
        "min_valid_years": 2,
        "min_positive_return_years": 0,
    }
    assert set(trial["metrics_by_year"]) == {"2022", "2023"}
    assert "observations" in trial["metrics"]
    assert "marginal_annualized_return" not in trial["metrics"]  # headline only

    assert sorted(item["factor_name"] for item in report["factor_contributions"]) == ["quality", "value"]
    assert report["factor_demands"]


def test_report_states_the_fixed_cost_contract(tmp_path) -> None:
    epoch = _run_epoch(tmp_path, policy=StrategyAcceptancePolicy())
    trial = epoch_report(epoch)["strategy_trials"][0]

    assert trial["execution_costs"] == {
        "buy_cost_bps": 10.0,
        "sell_cost_bps": 10.0,
        "round_trip_cost_bps": 20.0,
    }


def test_report_headline_metrics_are_bounded(tmp_path) -> None:
    epoch = _run_epoch(tmp_path)
    trial = epoch_report(epoch)["strategy_trials"][0]

    allowed = {
        "annualized_return",
        "excess_annualized_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "avg_holdings",
        "observations",
        "signal_lag",
        "buy_cost_bps",
        "sell_cost_bps",
        "round_trip_cost_bps",
        "buy_turnover",
        "sell_turnover",
        "total_transaction_cost",
    }
    assert set(trial["metrics"]) <= allowed
    # The full metrics dict is never embedded: the registry is the detailed record.
    assert "win_rate" not in trial["metrics"]
    assert "total_return" not in trial["metrics"]


def test_research_outcomes_are_projected_through_as_payload(tmp_path) -> None:
    class _Outcome:
        def as_payload(self) -> dict:
            return {"mode": "targeted", "accepted_factor_ids": ["gf_new"], "notes": []}

    epoch = _run_epoch(tmp_path, research_results=(_Outcome(),))
    report = epoch_report(epoch)

    assert report["factor_research"] == [{"mode": "targeted", "accepted_factor_ids": ["gf_new"], "notes": []}]
    assert report["summary"]["factor_research_rounds"] == 1
    assert report["summary"]["accepted_factors_from_research"] == 1


def test_research_outcomes_without_payload_fall_back_to_text(tmp_path) -> None:
    epoch = _run_epoch(tmp_path, research_results=("raw object",))
    assert epoch_report(epoch)["factor_research"] == [{"representation": "raw object"}]


def test_write_epoch_report_persists_readable_json(tmp_path) -> None:
    epoch = _run_epoch(tmp_path)
    path = write_epoch_report(epoch, tmp_path / "reports" / "000_epoch_1.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["epoch_id"] == "epoch_1"
    assert loaded["summary"]["strategy_trials"] == 2
    assert math.isfinite(loaded["strategy_trials"][0]["metrics"]["sharpe"])
