"""Unit tests for the execution-layer operators DriftBandRebalance (8.3)
and TurnoverShrinkage (8.5).

Both operators live inside the two mirrored backtest main loops; the loop
itself is guarded by the end-to-end parity suite, so this file pins the
arithmetic of the extracted helpers (identical functions on the local and the
worker side, compared directly), the boundary behaviour, the carried-state
initialisation / carry-forward and the spec validation.
"""

from __future__ import annotations

import pandas as pd
import pytest
from rdagent.scenarios.gf_quant.strategy.backtest import (
    BacktestSpec,
    _drift_band_triggers,
    _shrink_turnover,
    run_portfolio_backtest,
)
from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.registry import strategy_id_for
from scripts.gf_quant_strategy_worker import (
    _drift_band_triggers as worker_drift_band_triggers,
    _shrink_turnover as worker_shrink_turnover,
)


def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---- drift band trigger -----------------------------------------------------


@pytest.mark.parametrize("use_worker", [False, True])
def test_drift_band_boundary_is_inclusive(use_worker: bool) -> None:
    """A deviation exactly on the band stays inside the tolerance.

    All weights below are exact binary fractions, so the 0.125 boundary is
    hit without float noise on either side of the comparison.
    """

    carried = _series({"A": 0.25, "B": 0.25, "C": 0.0})
    target = _series({"A": 0.375, "B": 0.125, "C": 0.0})
    trigger = worker_drift_band_triggers if use_worker else _drift_band_triggers

    # Per-name band: max |delta| is exactly 0.125.
    assert trigger(carried, target, 0.125, None) is False
    assert trigger(carried, target, 0.124, None) is True

    # Portfolio band: half-L1 incl. cash is |0.125| + |0.125| = 0.25 / 2.
    assert trigger(carried, target, None, 0.125) is False
    assert trigger(carried, target, None, 0.124) is True


def test_drift_band_trigger_agrees_between_local_and_worker() -> None:
    cases = [
        (_series({"A": 0.4, "B": 0.6}), _series({"A": 0.0, "B": 0.0}), 0.3, None),
        (_series({"A": 0.4, "B": 0.6}), _series({"A": 0.0, "B": 0.0}), None, 0.4),
        (_series({"A": 0.4, "B": 0.6}), _series({"A": 0.0, "B": 0.0}), None, 0.6),
        (_series({"A": 0.5, "B": 0.5}), _series({"A": 0.5, "B": 0.5}), 0.01, 0.01),
        (_series({"A": 0.3, "B": 0.3, "C": 0.0}), _series({"A": 0.5, "B": 0.0, "C": 0.5}), None, 0.5),
    ]
    for carried, target, band_name, band_portfolio in cases:
        assert _drift_band_triggers(carried, target, band_name, band_portfolio) is (
            worker_drift_band_triggers(carried, target, band_name, band_portfolio)
        )


# ---- turnover shrinkage ------------------------------------------------------


def test_shrink_turnover_soft_scales_the_whole_delta() -> None:
    carried = _series({"A": 0.0, "B": 0.0})
    target = _series({"A": 0.5, "B": 0.5})
    executed = _shrink_turnover(carried, target, 0.5, None)
    assert executed.to_list() == pytest.approx([0.25, 0.25])

    # Liquidation shrinks symmetrically: the book only sells half way.
    executed = _shrink_turnover(target, carried, 0.5, None)
    assert executed.to_list() == pytest.approx([0.25, 0.25])

    # Shrinkage 0 is a no-op towards the target.
    assert _shrink_turnover(carried, target, 0.0, None).to_list() == pytest.approx([0.5, 0.5])


def test_shrink_turnover_hard_clip_caps_the_step_exactly() -> None:
    carried = _series({"A": 0.0, "B": 0.0})
    target = _series({"A": 0.5, "B": 0.5})
    # Full investment is a half-L1 step of 1.0; a 0.25 cap scales it to 1/4.
    executed = _shrink_turnover(carried, target, None, 0.25)
    assert executed.to_list() == pytest.approx([0.125, 0.125])

    # A step exactly on the cap is unchanged (strict comparison).
    on_band = _series({"A": 0.375, "B": 0.125})
    carried_on = _series({"A": 0.25, "B": 0.25})
    assert _shrink_turnover(carried_on, on_band, None, 0.125).to_list() == pytest.approx(
        on_band.to_list()
    )


def test_shrink_turnover_soft_then_hard_compose() -> None:
    carried = _series({"A": 0.0, "B": 0.0})
    target = _series({"A": 0.5, "B": 0.5})
    # Soft shrink to half, then the clip halves the remaining step again.
    executed = _shrink_turnover(carried, target, 0.5, 0.25)
    assert executed.to_list() == pytest.approx([0.125, 0.125])


def test_shrink_turnover_zero_weight_exits_participate_without_special_casing() -> None:
    """A name the target exits at zero shrinks like any other leg (A4 note).

    The truncated weighting of batch A3 leaves zero-weight target names whose
    exit shows up in the turnover deltas; the shrink acts on the weight
    vector, so such an exit is simply scaled -- never frozen, never revived.
    """

    carried = _series({"A": 0.4, "B": 0.3, "C": 0.3})
    target = _series({"A": 0.5, "B": 0.5, "C": 0.0})
    executed = _shrink_turnover(carried, target, 0.5, None)
    assert executed.to_list() == pytest.approx([0.45, 0.4, 0.15])
    assert 0.0 < executed["C"] < carried["C"]


def test_shrink_turnover_agrees_between_local_and_worker() -> None:
    cases = [
        (_series({"A": 0.0, "B": 0.0}), _series({"A": 0.5, "B": 0.5}), 0.5, None),
        (_series({"A": 0.0, "B": 0.0}), _series({"A": 0.5, "B": 0.5}), None, 0.1),
        (_series({"A": 0.0, "B": 0.0}), _series({"A": 0.5, "B": 0.5}), 0.7, 0.05),
        (_series({"A": 0.4, "B": 0.3, "C": 0.3}), _series({"A": 0.5, "B": 0.5, "C": 0.0}), 0.5, 0.2),
        (_series({"A": 0.25, "B": 0.25}), _series({"A": 0.375, "B": 0.125}), None, 0.125),
    ]
    for carried, target, shrinkage, max_step in cases:
        local = _shrink_turnover(carried, target, shrinkage, max_step)
        remote = worker_shrink_turnover(carried, target, shrinkage, max_step)
        assert local.to_list() == pytest.approx(remote.to_list())


# ---- engine-level behaviour ---------------------------------------------------


def _zero_returns(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=dates, columns=["A", "B"])


def test_shrinkage_applies_on_the_first_day_starting_from_cash() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]])
    returns = _zero_returns(dates)

    result = run_portfolio_backtest(
        targets, returns, spec=BacktestSpec(signal_lag=1, turnover_shrinkage=0.5)
    )

    # Day 0 cannot trade; day 1 executes half of the move from cash, so the
    # book starts at 0.25/0.25 and the half-L1 turnover (incl. cash) is 0.5.
    assert result.weights.loc[dates[0]].to_list() == pytest.approx([0.0, 0.0])
    assert result.weights.loc[dates[1]].to_list() == pytest.approx([0.25, 0.25])
    assert result.turnover.loc[dates[1]] == pytest.approx(0.5)
    assert result.buy_turnover.loc[dates[1]] == pytest.approx(0.5)
    assert result.sell_turnover.loc[dates[1]] == pytest.approx(0.0)
    # The immutable cost formula simply sees the smaller executed delta.
    assert result.transaction_costs.loc[dates[1]] == pytest.approx(0.5 * 10e-4)


def test_carried_state_carries_the_shrunk_book_forward() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]])
    returns = pd.DataFrame(
        {"A": [0.0, 0.10, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=dates
    )

    result = run_portfolio_backtest(
        targets, returns, spec=BacktestSpec(signal_lag=1, turnover_shrinkage=0.5)
    )

    # Day 1 executes 0.25/0.25; A's day-1 return drifts that book to
    # 0.25 * 1.10 / 1.025 and 0.25 / 1.025 on day 2 -- the shrunk weights, not
    # the targets, are what the carried state continues from.
    drifted_a = 0.25 * 1.10 / 1.025
    drifted_b = 0.25 / 1.025
    assert result.weights.loc[dates[2]].to_list() == pytest.approx([drifted_a, drifted_b])
    assert result.turnover.loc[dates[2:]].abs().sum() == pytest.approx(0.0)


def test_drift_band_skips_rebalances_and_forced_days_force_them() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    decision_dates = [dates[0], dates[2], dates[4]]
    targets = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=decision_dates)
    returns = _zero_returns(dates)

    # The 0.9 name band can never be crossed by a 0.5 target move, so every
    # decision is skipped and the book stays flat.
    skipped = run_portfolio_backtest(
        targets, returns, spec=BacktestSpec(signal_lag=1, drift_band_name=0.9)
    )
    assert skipped.weights.to_numpy() == pytest.approx(0.0)
    assert skipped.turnover.abs().sum() == pytest.approx(0.0)

    # With a forced cadence of 2 trading days the counter (which also ticks on
    # decision-free days) forces the day-3 decision: 0 (trade-free start) ->
    # 1 -> 2 -> 3 >= 2 on the day the dates[2] target arrives.
    forced = run_portfolio_backtest(
        targets,
        returns,
        spec=BacktestSpec(signal_lag=1, drift_band_name=0.9, drift_band_forced_days=2),
    )
    assert forced.weights.loc[dates[1]].to_list() == pytest.approx([0.0, 0.0])  # skipped
    assert forced.weights.loc[dates[3]].to_list() == pytest.approx([0.5, 0.5])  # forced
    assert forced.weights.loc[dates[5]].to_list() == pytest.approx([0.5, 0.5])  # skipped again
    assert forced.turnover.loc[dates[3]] == pytest.approx(1.0)
    assert forced.turnover.drop(dates[3]).abs().sum() == pytest.approx(0.0)


def test_drift_band_treats_a_go_to_cash_decision_as_a_trade() -> None:
    """An empty selection means go to cash and the band judges it like any
    other trade: a full investment or liquidation (step 1.0) trips even a
    wide 0.9 band instead of being silently held."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    targets = pd.DataFrame({"A": [0.5, 0.0, 0.0], "B": [0.5, 0.0, 0.0]}, index=dates)
    returns = _zero_returns(dates)

    result = run_portfolio_backtest(
        targets, returns, spec=BacktestSpec(signal_lag=1, drift_band_portfolio=0.9)
    )
    assert result.weights.loc[dates[1]].to_list() == pytest.approx([0.5, 0.5])
    assert result.weights.loc[dates[2]].to_list() == pytest.approx([0.0, 0.0])
    assert result.turnover.loc[dates[2]] == pytest.approx(1.0)


# ---- spec validation and payload contract -------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("drift_band_name", 0.0, "drift_band_name must be in (0, 1] when provided"),
        ("drift_band_name", 1.5, "drift_band_name must be in (0, 1] when provided"),
        ("drift_band_portfolio", -0.1, "drift_band_portfolio must be in (0, 1] when provided"),
        ("drift_band_forced_days", 0, "drift_band_forced_days must be at least 1 when provided"),
        ("drift_band_forced_days", 2.5, "drift_band_forced_days must be a whole number when provided"),
        (
            "drift_band_forced_days",
            5,
            "drift_band_forced_days requires drift_band_name or drift_band_portfolio",
        ),
        ("turnover_shrinkage", 1.0, "turnover_shrinkage must be in [0, 1) when provided"),
        ("turnover_shrinkage", -0.2, "turnover_shrinkage must be in [0, 1) when provided"),
        ("turnover_max_step", 0.0, "turnover_max_step must be in (0, 1] when provided"),
        ("turnover_max_step", 1.2, "turnover_max_step must be in (0, 1] when provided"),
    ],
)
def test_backtest_spec_rejects_invalid_execution_parameters(field: str, value, message: str) -> None:
    with pytest.raises(ValueError) as error:
        BacktestSpec(signal_lag=1, **{field: value})
    assert str(error.value) == message


def test_backtest_spec_accepts_the_documented_edge_values() -> None:
    spec = BacktestSpec(
        signal_lag=1,
        drift_band_name=1.0,
        drift_band_portfolio=1.0,
        drift_band_forced_days=1,
        turnover_shrinkage=0.0,
        turnover_max_step=1.0,
    )
    assert spec.drift_band_forced_days == 1


def test_execution_fields_stay_out_of_the_payload_until_enabled() -> None:
    disabled = BacktestSpec(signal_lag=1)
    assert set(disabled.as_payload()) == {"signal_lag", "trading_days_per_year"}

    enabled = BacktestSpec(
        signal_lag=2,
        drift_band_name=0.1,
        drift_band_forced_days=20,
        turnover_shrinkage=0.5,
        turnover_max_step=0.3,
    )
    payload = enabled.as_payload()
    assert set(payload) == {
        "signal_lag",
        "trading_days_per_year",
        "drift_band_name",
        "drift_band_forced_days",
        "turnover_shrinkage",
        "turnover_max_step",
    }
    assert BacktestSpec.from_payload(payload) == enabled
    # A field left at its None default is simply absent from the payload.
    assert "drift_band_portfolio" not in payload


def test_enabled_execution_fields_change_the_strategy_identity() -> None:
    def candidate(backtest: BacktestSpec) -> StrategyCandidate:
        return StrategyCandidate(
            name="identity",
            universe="csi_all",
            composition=CompositionSpec(top_n=5),
            factor_names=("value", "quality"),
            rule="rank blend",
            backtest=backtest,
        )

    plain = candidate(BacktestSpec(signal_lag=1))
    banded = candidate(BacktestSpec(signal_lag=1, turnover_shrinkage=0.5))
    assert strategy_id_for(plain.name, plain.canonical_payload()) != strategy_id_for(
        banded.name, banded.canonical_payload()
    )
