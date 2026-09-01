"""Tests for the GF strategy layer (composition, backtest, registry, gate).

The whole suite runs on synthetic panels -- no remote host, no ``G:/base`` and
no factor registry fixture is required.  Arithmetic expectations are hand
computed so a regression in the engine shows up as a wrong number, not just as
a changed one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdagent.scenarios.gf_quant.dedup import FactorDeduplicator
from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.backtest import BacktestSpec, run_portfolio_backtest
from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.composition import build_target_weights
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRecord, StrategyRegistry, strategy_id_for
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
    compact_return_profile,
    profile_from_payload,
)


def _returns() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {"A": [0.10, 0.20, 0.02, 0.02], "B": [0.02, 0.02, 0.30, 0.02]},
        index=index,
    )


# ---- composition -----------------------------------------------------------


def test_composition_selects_top_scores_and_weights_equally() -> None:
    score = pd.DataFrame(
        {"A": [3.0], "B": [1.0], "C": [2.0], "D": [np.nan]}, index=pd.date_range("2024-01-01", periods=1)
    )
    weights = build_target_weights(CompositionSpec(top_n=2), score)
    assert weights.loc["2024-01-01", "A"] == pytest.approx(0.5)
    assert weights.loc["2024-01-01", "C"] == pytest.approx(0.5)
    # Unselected and NaN names stay at exactly zero so turnover is measurable.
    assert weights.loc["2024-01-01", "B"] == 0.0
    assert weights.loc["2024-01-01", "D"] == 0.0


def test_composition_applies_eligibility_mask() -> None:
    dates = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]}, index=dates)
    eligible = pd.DataFrame({"A": [False], "B": [True], "C": [True]}, index=dates)
    weights = build_target_weights(CompositionSpec(top_n=2), score, eligible=eligible)
    assert weights.loc["2024-01-01", "A"] == 0.0  # highest score, but screened out
    assert weights.loc["2024-01-01", ["B", "C"]].sum() == pytest.approx(1.0)

    # A missing eligibility entry is treated as not eligible: fail closed on
    # unknown data instead of silently investing in it.
    partial = pd.DataFrame({"B": [True]}, index=dates)
    weights = build_target_weights(CompositionSpec(top_n=2), score, eligible=partial)
    assert weights.loc["2024-01-01"].sum() == pytest.approx(1.0)
    assert weights.loc["2024-01-01", "B"] == pytest.approx(1.0)


def test_composition_rank_and_proportional_weighting() -> None:
    dates = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [30.0], "B": [20.0], "C": [10.0]}, index=dates)

    rank = build_target_weights(CompositionSpec(top_n=3, weighting="rank"), score)
    assert rank.loc["2024-01-01"].to_list() == pytest.approx([3 / 6, 2 / 6, 1 / 6])

    proportional = build_target_weights(CompositionSpec(top_n=3, weighting="proportional"), score)
    assert proportional.loc["2024-01-01"].to_list() == pytest.approx([0.5, 1 / 3, 1 / 6])

    with pytest.raises(ValueError, match="proportional"):
        build_target_weights(
            CompositionSpec(top_n=3, weighting="proportional"),
            pd.DataFrame({"A": [1.0], "B": [-1.0], "C": [1.0]}, index=dates),
        )


def test_composition_weight_cap_is_solved_exactly() -> None:
    dates = pd.date_range("2024-01-01", periods=1)
    score = pd.DataFrame({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]}, index=dates)

    # Proportional weights [0.444, 0.333, 0.222]: only A is capped and the other
    # two are scaled up proportionally, keeping the book fully invested.
    weights = build_target_weights(
        CompositionSpec(top_n=3, weighting="proportional", weight_cap=0.4), score[["A", "B", "C"]]
    )
    assert weights.loc["2024-01-01", "A"] == pytest.approx(0.4)
    assert weights.loc["2024-01-01", ["B", "C"]].sum() == pytest.approx(0.6)
    assert weights.loc["2024-01-01"].sum() == pytest.approx(1.0)

    # Four names with a 20% cap cannot invest the whole book: everything sits at
    # the cap and the residual stays in cash instead of exceeding the cap.
    cash = build_target_weights(CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2), score)
    assert cash.loc["2024-01-01"].to_list() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert cash.loc["2024-01-01"].sum() == pytest.approx(0.8)


def test_composition_spec_validation() -> None:
    with pytest.raises(ValueError, match="top_n"):
        CompositionSpec(top_n=0)
    with pytest.raises(ValueError, match="weighting"):
        CompositionSpec(top_n=5, weighting="market_cap")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight_cap"):
        CompositionSpec(top_n=5, weight_cap=1.5)


def test_composition_rejects_mismatched_panels() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    score = pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 0.0]}, index=dates)
    with pytest.raises(ValueError, match="eligible panel is missing"):
        build_target_weights(CompositionSpec(top_n=1), score, eligible=score.iloc[:1])
    with pytest.raises(ValueError, match="unknown symbols"):
        build_target_weights(
            CompositionSpec(top_n=1),
            score,
            eligible=pd.DataFrame({"A": [True, True], "Z": [True, True]}, index=dates),
        )
    with pytest.raises(ValueError, match="empty"):
        build_target_weights(CompositionSpec(top_n=1), score.iloc[:0])


# ---- backtest --------------------------------------------------------------


def test_backtest_honours_signal_lag() -> None:
    returns = _returns()
    dates = returns.index
    targets = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    targets.loc[dates[0], "A"] = 1.0
    targets.loc[dates[1], "B"] = 1.0  # dates 2 and 3 stay zero: go to cash

    one_day = run_portfolio_backtest(targets, returns, spec=BacktestSpec(signal_lag=1))
    # Day 0 cannot trade on itself; day 1 earns A's day-1 return, day 2 earns
    # B's day-2 return, day 3 is in cash.  A full switch counts as one-way
    # turnover of 1.0 under the half-L1 convention.
    assert one_day.daily_returns.to_list() == pytest.approx([0.0, 0.20, 0.30, 0.0])
    assert one_day.turnover.to_list() == pytest.approx([0.0, 1.0, 1.0, 1.0])

    two_days = run_portfolio_backtest(targets, returns, spec=BacktestSpec(signal_lag=2))
    assert two_days.daily_returns.to_list() == pytest.approx([0.0, 0.0, 0.02, 0.02])


def test_backtest_drifts_holdings_between_rebalances() -> None:
    dates = _returns().index
    targets = pd.DataFrame(0.0, index=[dates[0]], columns=["A", "B"])
    targets.loc[dates[0]] = [0.5, 0.5]
    returns = pd.DataFrame({"A": [0.0, 0.20, 0.0, 0.0], "B": [0.0, 0.0, 0.10, 0.0]}, index=dates)
    result = run_portfolio_backtest(targets, returns, spec=BacktestSpec(signal_lag=1))

    # Only dates present in the target panel are decisions; day 2 must therefore
    # be earned on drifted weights, not on a re-applied or liquidated book.
    drifted_a = 0.5 * 1.20 / 1.10
    assert result.daily_returns.loc[dates[2]] == pytest.approx((1.0 - drifted_a) * 0.10)
    assert result.weights.loc[dates[2], "A"] == pytest.approx(drifted_a)
    assert result.turnover.loc[dates[2:]].abs().sum() == pytest.approx(0.0)


def test_backtest_charges_cost_on_one_way_turnover() -> None:
    dates = _returns().index
    targets = pd.DataFrame(0.0, index=[dates[0]], columns=["A", "B"])
    targets.loc[dates[0], "A"] = 1.0

    result = run_portfolio_backtest(targets, _returns(), spec=BacktestSpec(signal_lag=1, cost_bps=100))
    assert result.metrics["cost_bps"] == 100.0
    assert result.daily_returns.loc[dates[1]] == pytest.approx(0.20 - 1.0 * 0.01)
    gross = run_portfolio_backtest(targets, _returns(), spec=BacktestSpec(signal_lag=1))
    assert result.daily_gross_returns.loc[dates[1]] == pytest.approx(gross.daily_returns.loc[dates[1]])


def test_backtest_reports_yearly_and_benchmark_metrics() -> None:
    index = pd.bdate_range("2022-01-03", periods=500)
    symbols = [f"S{position}" for position in range(10)]
    rng = np.random.default_rng(7)
    drift = np.array([0.001 * (position - 4.5) for position in range(10)])
    returns = pd.DataFrame(rng.normal(0.0, 0.001, size=(len(index), 10)) + drift, index=index, columns=symbols)

    score = pd.DataFrame({symbol: float(position) for position, symbol in enumerate(symbols)}, index=index[::21])
    weights = build_target_weights(CompositionSpec(top_n=5), score)

    result = run_portfolio_backtest(
        weights,
        returns,
        spec=BacktestSpec(signal_lag=1, cost_bps=5.0),
        benchmark_returns=pd.Series(0.0002, index=index),
    )
    assert set(result.metrics_by_year) == {"2022", "2023"}
    for year_metrics in result.metrics_by_year.values():
        assert set(year_metrics) == set(result.metrics) - {"cost_bps", "signal_lag"}
    assert result.metrics["avg_holdings"] == pytest.approx(5.0, abs=0.1)
    assert result.metrics["excess_annualized_return"] > 0.0
    assert result.metrics["tracking_error"] > 0.0
    assert result.metrics["observations"] == len(index)


def test_backtest_goes_to_cash_on_an_empty_selection() -> None:
    dates = _returns().index
    targets = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    result = run_portfolio_backtest(targets, _returns(), spec=BacktestSpec(signal_lag=1))
    assert result.daily_returns.abs().sum() == pytest.approx(0.0)
    assert result.metrics["avg_holdings"] == 0.0


def test_backtest_drawdown_includes_the_initial_nav() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    targets = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
    returns = pd.DataFrame({"A": [0.0, -0.20, 0.0]}, index=dates)

    result = run_portfolio_backtest(targets, returns, spec=BacktestSpec(signal_lag=1))

    assert result.metrics["max_drawdown"] == pytest.approx(-0.20)


def test_backtest_validates_inputs_instead_of_dropping_them() -> None:
    dates = _returns().index
    spec = BacktestSpec(signal_lag=1)
    with pytest.raises(ValueError, match="absent from the returns panel"):
        run_portfolio_backtest(
            pd.DataFrame({"A": [1.0]}, index=[dates[-1] + pd.Timedelta(days=1)]), _returns(), spec=spec
        )
    with pytest.raises(ValueError, match="unknown symbols"):
        run_portfolio_backtest(pd.DataFrame({"Z": [1.0] * 4}, index=dates), _returns(), spec=spec)
    with pytest.raises(ValueError, match="duplicate dates"):
        run_portfolio_backtest(pd.DataFrame({"A": [1.0, 1.0]}, index=[dates[0], dates[0]]), _returns(), spec=spec)
    with pytest.raises(ValueError, match="row sums"):
        run_portfolio_backtest(pd.DataFrame({"A": [0.8], "B": [0.4]}, index=[dates[0]]), _returns(), spec=spec)
    with pytest.raises(ValueError, match="non-negative"):
        run_portfolio_backtest(pd.DataFrame({"A": [-0.1]}, index=[dates[0]]), _returns(), spec=spec)
    with pytest.raises(ValueError, match="sorted"):
        run_portfolio_backtest(
            pd.DataFrame({"A": [1.0]}, index=[dates[0]]), _returns().sort_index(ascending=False), spec=spec
        )
    with pytest.raises(ValueError, match="target_weights is empty"):
        run_portfolio_backtest(pd.DataFrame(), _returns(), spec=spec)
    with pytest.raises(ValueError, match="benchmark"):
        run_portfolio_backtest(
            pd.DataFrame({"A": [1.0] * 4}, index=dates),
            _returns(),
            spec=spec,
            benchmark_returns=pd.Series(0.0, index=dates[:2]),
        )


def test_backtest_spec_rejects_look_ahead_execution() -> None:
    with pytest.raises(ValueError, match="look-ahead"):
        BacktestSpec(signal_lag=0)
    with pytest.raises(ValueError, match="cost_bps"):
        BacktestSpec(cost_bps=-1.0)


# ---- registry --------------------------------------------------------------


def test_strategy_identity_covers_the_whole_spec() -> None:
    base = StrategyCandidate(
        name="fcf_top50",
        universe="csi_all",
        composition=CompositionSpec(top_n=50, weighting="proportional", weight_cap=0.1),
        rule="free cash flow rate ranking",
        backtest=BacktestSpec(signal_lag=1, cost_bps=15.0),
    )
    other = StrategyCandidate(
        name="fcf_top50",
        universe="csi_all",
        composition=CompositionSpec(top_n=50, weighting="equal"),
        rule="free cash flow rate ranking",
    )
    base_id = strategy_id_for(base.name, base.canonical_payload())
    assert base_id == strategy_id_for(base.name, base.as_payload())
    assert base_id != strategy_id_for(other.name, other.canonical_payload())

    candidate = StrategyCandidate(
        name="solo",
        universe="csi_all",
        composition=CompositionSpec(top_n=1),
        factor_names=(),
        rule="rank by score",
    )
    with pytest.raises(ValueError, match="factor_names or a rule"):
        StrategyCandidate(
            name="empty", universe="csi_all", composition=CompositionSpec(top_n=1), rule="", factor_names=()
        )
    with pytest.raises(ValueError, match="universe"):
        StrategyCandidate(name="no_universe", universe="", composition=CompositionSpec(top_n=1), rule="rank by score")
    del candidate


def test_strategy_candidate_payload_round_trip() -> None:
    candidate = StrategyCandidate(
        name="round_trip",
        universe="csi_all",
        composition=CompositionSpec(top_n=30, weighting="rank", weight_cap=0.05),
        factor_names=("value", "quality"),
        rule="rank blend",
        hypothesis="complementary factors",
        backtest=BacktestSpec(signal_lag=2, cost_bps=8.0),
        metadata={"epoch_id": "e1"},
    )
    payload = {**candidate.as_payload(), "hypothesis": candidate.hypothesis, "metadata": candidate.metadata}

    restored = StrategyCandidate.from_payload(payload)

    assert restored == candidate


def test_strategy_registry_upsert_and_query(tmp_path) -> None:
    registry = StrategyRegistry(tmp_path / "strategy_registry.sqlite")
    candidate = StrategyCandidate(
        name="equal_weight_top5",
        universe="csi_all",
        composition=CompositionSpec(top_n=5),
        factor_names=("pv_reversal_20",),
    )
    strategy_id = strategy_id_for(candidate.name, candidate.canonical_payload())
    stored = registry.upsert(
        StrategyRecord(
            strategy_id=strategy_id,
            name=candidate.name,
            universe=candidate.universe,
            rule=candidate.rule,
            spec_json=candidate.canonical_payload(),
            factor_names=list(candidate.factor_names),
            status="accepted",
        )
    )
    assert stored.factor_names == ["pv_reversal_20"]
    assert stored.created_at is not None

    updated = registry.upsert(
        StrategyRecord(
            strategy_id=stored.strategy_id,
            name=stored.name,
            universe=stored.universe,
            rule=stored.rule,
            spec_json=stored.spec_json,
            factor_names=stored.factor_names,
            status="retired",
        )
    )
    assert updated.status == "retired"
    assert updated.created_at == stored.created_at  # creation time is preserved
    assert registry.get(stored.strategy_id).status == "retired"
    assert registry.get_by_name(candidate.name).strategy_id == strategy_id
    assert [record.name for record in registry.list(status="retired")] == [candidate.name]
    assert registry.list(status="accepted") == []

    with pytest.raises(Exception):
        # Names are unique, mirroring the factor registry contract.
        registry.upsert(
            StrategyRecord(
                strategy_id="gf_strategy_other",
                name=candidate.name,
                universe=stored.universe,
                rule=stored.rule,
                spec_json=stored.spec_json,
                factor_names=stored.factor_names,
                status="accepted",
            )
        )


# ---- acceptance service ----------------------------------------------------


def _skilled_backtest() -> StrategyEvaluation:
    """A three-year strategy with a persistent edge, built through the real engine."""

    index = pd.bdate_range("2020-01-02", periods=560)
    symbols = [f"S{position}" for position in range(10)]
    rng = np.random.default_rng(11)
    drift = np.array([0.001 * (position - 4.5) for position in range(10)])
    returns = pd.DataFrame(rng.normal(0.0, 0.001, size=(len(index), 10)) + drift, index=index, columns=symbols)
    score = pd.DataFrame({symbol: float(position) for position, symbol in enumerate(symbols)}, index=index[::21])
    weights = build_target_weights(CompositionSpec(top_n=5), score)
    result = run_portfolio_backtest(weights, returns, spec=BacktestSpec(signal_lag=1, cost_bps=10.0))
    return StrategyEvaluation.from_result(result)


def _candidate(name: str = "fcf_top5", **overrides) -> StrategyCandidate:
    fields: dict = {
        "name": name,
        "universe": "csi_all",
        "composition": CompositionSpec(top_n=5),
        "rule": "free cash flow rate ranking",
        "backtest": BacktestSpec(signal_lag=1, cost_bps=10.0),
    }
    fields.update(overrides)
    return StrategyCandidate(**fields)


def _service(tmp_path, evaluation: StrategyEvaluation, factor_registry: FactorRegistry | None = None):
    return StrategyAcceptanceService(
        StrategyRegistry(tmp_path / "strategy_registry.sqlite"),
        lambda candidate: evaluation,
        policy=StrategyAcceptancePolicy(min_observations=100, min_avg_holdings=4.0, min_sharpe=0.3),
        deduplicator=FactorDeduplicator(threshold=0.88, min_overlap=100),
        factor_registry=factor_registry,
    )


def test_strategy_service_accepts_a_valid_candidate(tmp_path) -> None:
    evaluation = _skilled_backtest()
    result = _service(tmp_path, evaluation).process(_candidate())

    assert result.gate_failures == ()
    assert result.record.status == "accepted"
    assert result.record.metrics["signal_lag"] == 1
    assert set(result.record.metrics["metrics_by_year"]) == {"2020", "2021", "2022"}
    # The stored trace is what later candidates are compared against.
    assert len(result.record.metadata["return_profile"]["dates"]) == len(evaluation.returns)


def test_strategy_service_rejects_gate_failures_fail_closed(tmp_path) -> None:
    evaluation = _skilled_backtest()
    result = _service(tmp_path, evaluation).process(
        _candidate(name="no_cost", backtest=BacktestSpec(signal_lag=1, cost_bps=0.0))
    )
    assert result.record.status == "rejected"
    assert "strategy_cost_missing" in result.gate_failures

    # A policy can raise the execution lag; a lag-1 candidate then fails too.
    strict = StrategyAcceptanceService(
        StrategyRegistry(tmp_path / "strict.sqlite"),
        lambda candidate: evaluation,
        policy=StrategyAcceptancePolicy(min_signal_lag=2, min_observations=100, min_avg_holdings=4.0),
    )
    rejected = strict.process(_candidate(name="slow_lag"))
    assert "strategy_signal_lag" in rejected.gate_failures
    assert rejected.record.status == "rejected"

    # Missing daily returns disable both the redundancy check and the audit.
    empty = StrategyEvaluation(metrics=evaluation.metrics, returns=None)
    failed = _service(tmp_path, empty).process(_candidate(name="no_returns"))
    assert "strategy_return_profile_missing" in failed.gate_failures
    assert failed.record.status == "rejected"

    # A metrics payload without a yearly breakdown is a failure, not a skip.
    flat = StrategyEvaluation(metrics={"sharpe": 2.0, "annualized_return": 0.5}, returns=evaluation.returns)
    missing_years = _service(tmp_path, flat).process(_candidate(name="no_years"))
    assert "strategy_metrics_by_year_missing" in missing_years.gate_failures


def test_strategy_policy_can_enforce_validation_target_and_yearly_stability() -> None:
    policy = StrategyAcceptancePolicy(
        min_annualized_return=0.08,
        min_excess_annualized_return=0.0,
        min_positive_return_years=2,
    )
    candidate = _candidate(name="strict_target")
    metrics = {
        "observations": 250,
        "avg_holdings": 20,
        "annualized_return": 0.07,
        "excess_annualized_return": -0.01,
        "sharpe": 0.8,
        "max_drawdown": -0.2,
        "annual_turnover": 3.0,
        "metrics_by_year": {
            "2022": {"annualized_return": 0.10, "sharpe": 0.8},
            "2023": {"annualized_return": -0.02, "sharpe": -0.1},
        },
    }

    passed, failures = policy.check(metrics, candidate)

    assert passed is False
    assert {
        "annualized_return",
        "excess_annualized_return",
        "strategy_positive_return_years",
    }.issubset(failures)


def test_strategy_policy_caps_stability_gate_to_bound_single_year_window() -> None:
    policy = StrategyAcceptancePolicy(
        min_annualized_return=0.08,
        min_excess_annualized_return=0.0,
        min_valid_years=2,
        min_positive_return_years=2,
    )
    candidate = _candidate(name="single_year_validation")
    metrics = {
        "observations": 242,
        "avg_holdings": 50,
        "annualized_return": 0.20,
        "excess_annualized_return": 0.05,
        "sharpe": 0.8,
        "max_drawdown": -0.2,
        "annual_turnover": 3.0,
        "metrics_by_year": {
            "2024": {"annualized_return": 0.20, "sharpe": 0.8},
        },
    }

    passed_single, failures_single = policy.check(metrics, candidate, expected_years=1)
    passed_multi, failures_multi = policy.check(metrics, candidate, expected_years=2)

    assert passed_single is True
    assert failures_single == []
    assert passed_multi is False
    assert "strategy_valid_years" in failures_multi


def test_strategy_service_marks_overlapping_strategies_as_duplicates(tmp_path) -> None:
    service = _service(tmp_path, _skilled_backtest())
    first = service.process(_candidate())
    assert first.record.status == "accepted"

    # Same portfolio, different declared rule: still the same bet, so it must
    # not enter the accepted pool twice.
    second = service.process(_candidate(name="fcf_top5_copy", rule="another phrasing"))
    assert second.record.status == "duplicate"
    assert second.dedup is not None
    assert second.dedup.is_duplicate
    assert second.dedup.duplicate_of == first.record.strategy_id
    assert second.dedup.reason == "return_profile_correlation"

    # Re-processing the identical candidate is idempotent: the same strategy_id
    # is excluded from its own comparison pool and simply refreshed.
    again = service.process(_candidate())
    assert again.record.status == "accepted"
    assert not again.dedup.is_duplicate


def test_strategy_service_records_referenced_factors(tmp_path) -> None:
    factor_registry = FactorRegistry(tmp_path / "factors.sqlite")
    factor_registry.upsert(
        FactorRecord(
            factor_id="gf_factor_1",
            name="pv_reversal_20",
            family="price_volume",
            expression="-pct_change(adjclose, 20)",
            status="accepted",
        )
    )
    evaluation = _skilled_backtest()
    result = _service(tmp_path, evaluation, factor_registry=factor_registry).process(
        _candidate(factor_names=("pv_reversal_20", "not_yet_mined"))
    )
    assert result.record.status == "accepted"
    references = result.record.metadata["referenced_factors"]
    assert references["pv_reversal_20"]["status"] == "accepted"
    assert references["not_yet_mined"] == {"status": "unknown"}


def test_return_profile_payload_round_trip() -> None:
    returns = pd.Series([0.1, 0.0, -0.2], index=pd.date_range("2024-01-01", periods=3))
    restored = profile_from_payload(compact_return_profile(returns))
    # Profiles are stored with ISO date keys, matching the benchmark signature
    # format, so the round trip is through strings rather than timestamps.
    assert list(restored.index) == [index.strftime("%Y-%m-%d") for index in returns.index]
    assert restored.to_list() == pytest.approx(returns.to_list())
    assert profile_from_payload({"dates": ["2024-01-01"]}) is None
