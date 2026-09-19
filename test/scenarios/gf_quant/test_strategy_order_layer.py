"""Directed tests for the order layer and the two portfolio-risk scalings.

``strategy/execution.py`` holds the pure functions the main loop calls for
roadmap 8.6-8.9 and 9.1/9.2.  The worker carries a value-for-value mirror of
each one, so every semantic decision pinned here is also pinned on the wire by
``test_strategy_worker_parity.py``; these tests exist to fix the *local*
reading first, where a failure points at one function instead of at a
two-sided diff.

The cases below are deliberately asymmetric where the operator is: a blocked
buy and a blocked sell must resolve differently, ``down`` rounding must never
overspend while ``nearest`` may, and the drawdown machine must be able to walk
back down the ladder it climbed.
"""

from __future__ import annotations

import math
import re

import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.backtest import BacktestSpec
from rdagent.scenarios.gf_quant.strategy.candidate import SignalSpec
from rdagent.scenarios.gf_quant.strategy.execution import (
    _order_names,
    apply_execution_layer,
    apply_lot_rounding,
    apply_participation_cap,
    apply_side_aware_tradability,
    drawdown_state_machine,
    factor_decay_scales,
    factor_decay_state,
    factor_ic_series,
    plan_order,
    stage_orders,
    volatility_scale,
)


def _series(**weights: float) -> pd.Series:
    return pd.Series(weights, dtype=float)


# ---------------------------------------------------------------------------
# 8.8 SideAwareTradability
# ---------------------------------------------------------------------------


def test_side_aware_block_both_blocks_by_state_and_direction() -> None:
    """A halt stops both sides; a limit only stops the side it strands."""

    carried = _series(A=0.2, B=0.2, C=0.2, D=0.2)
    target = _series(A=0.5, B=0.0, C=0.5, D=0.5)
    halt = {"2024-01-02": {"A": "suspended", "B": "suspended"}}
    limits = {"2024-01-02": {"C": {"upper": 10.0}, "D": {"lower": 1.0}}}
    price = {"2024-01-02": {"C": 10.0, "D": 1.0}}

    out = apply_side_aware_tradability(carried, target, halt, limits, price, mode="block_both", day="2024-01-02")

    # A is halted and wants to buy, B is halted and wants to sell: both pinned.
    assert float(out["A"]) == pytest.approx(0.2)
    assert float(out["B"]) == pytest.approx(0.2)
    # C is at its upper limit and wants to buy: pinned.  D is at its lower
    # limit and wants to buy: allowed, because a limit-down does not strand a
    # buyer.  This is what makes the mask directional rather than a blanket.
    assert float(out["C"]) == pytest.approx(0.2)
    assert float(out["D"]) == pytest.approx(0.5)


def test_side_aware_holds_are_never_touched() -> None:
    """A name that cannot be bought must not turn into a forced seller."""

    carried = _series(A=0.3, B=0.3)
    target = _series(A=0.3, B=0.6)
    halt = {"2024-01-02": {"A": "suspended", "B": "suspended"}}

    out = apply_side_aware_tradability(carried, target, halt, None, None, mode="block_both", day="2024-01-02")

    assert float(out["A"]) == pytest.approx(0.3)
    assert float(out["B"]) == pytest.approx(0.3)


def test_side_aware_single_sided_modes_trade_the_other_side_freely() -> None:
    """``block_buy`` stops the buy side on every abnormal state, and only it."""

    carried = _series(A=0.2, B=0.5)
    target = _series(A=0.5, B=0.2)
    limits = {"2024-01-02": {"A": {"lower": 1.0}, "B": {"upper": 10.0}}}
    price = {"2024-01-02": {"A": 1.0, "B": 10.0}}

    out = apply_side_aware_tradability(carried, target, None, limits, price, mode="block_buy", day="2024-01-02")

    # A is at its lower limit and wants to buy: blocked (stricter than
    # block_both, which lets a limit-down buy through).
    assert float(out["A"]) == pytest.approx(0.2)
    # B is at its upper limit and wants to sell: allowed (looser than
    # block_both, which pins a limit-up sale).
    assert float(out["B"]) == pytest.approx(0.2)


def test_side_aware_halt_cell_semantics() -> None:
    """The halt panel is a sparse string panel: an entry means abnormal."""

    carried = _series(A=0.2, B=0.2, C=0.2)
    target = _series(A=0.5, B=0.5, C=0.5)
    # A carries a label, B carries the empty string, C is absent entirely.
    halt = {"2024-01-02": {"A": "suspended", "B": ""}}

    out = apply_side_aware_tradability(carried, target, halt, None, None, mode="block_both", day="2024-01-02")

    assert float(out["A"]) == pytest.approx(0.2)
    assert float(out["B"]) == pytest.approx(0.5)
    assert float(out["C"]) == pytest.approx(0.5)


def test_side_aware_day_absent_from_the_panels_blocks_nothing() -> None:
    carried = _series(A=0.2)
    target = _series(A=0.5)
    halt = {"2024-01-02": {"A": "suspended"}}

    out = apply_side_aware_tradability(carried, target, halt, None, None, mode="block_both", day="2024-01-03")

    assert float(out["A"]) == pytest.approx(0.5)


def test_side_aware_missing_inputs_fail_closed() -> None:
    carried = _series(A=0.2)
    target = _series(A=0.5)
    limits = {"2024-01-02": {"A": {"upper": 10.0}}}

    with pytest.raises(ValueError) as error:
        apply_side_aware_tradability(carried, target, None, limits, None, mode="block_both", day="2024-01-02")
    assert str(error.value) == "side_aware_tradability requires the execution_data price panel"

    with pytest.raises(ValueError) as error:
        apply_side_aware_tradability(carried, target, None, limits, {"2024-01-02": {}}, mode="block_both", day="2024-01-02")
    assert str(error.value) == "side_aware_tradability is missing a price for A on 2024-01-02"

    with pytest.raises(ValueError) as error:
        apply_side_aware_tradability(carried, target, None, None, None, mode="block_both", day="2024-01-02")
    assert str(error.value) == "side_aware_tradability requires execution_data.halt or execution_data.price_limits"

    with pytest.raises(ValueError) as error:
        apply_side_aware_tradability(carried, target, {"2024-01-02": {"A": "x"}}, None, None, mode="block_neither", day="2024-01-02")
    assert str(error.value) == "side_aware_tradability mode must be one of ('block_buy', 'block_sell', 'block_both') when provided"


# ---------------------------------------------------------------------------
# 8.7 LotRounding
# ---------------------------------------------------------------------------


def test_lot_rounding_down_never_overspends() -> None:
    """``down`` keeps every traded weight at or below its target."""

    carried = _series(A=0.0, B=0.0)
    target = _series(A=0.3125, B=0.5)
    price = {"2024-01-02": {"A": 0.125, "B": 0.125}}
    lot_sizes = {"A": 1, "B": 1}

    out = apply_lot_rounding(target, carried, price, lot_sizes, min_notional=None, mode="down", day="2024-01-02")

    # 0.3125 / 0.125 is exactly 2.5 lots, so the grid is exercised at its
    # tightest: down lands on 2 lots, which the +LOT_SNAP guard must not lift.
    assert float(out["A"]) == pytest.approx(0.25)
    assert float(out["B"]) == pytest.approx(0.5)
    for symbol in out.index:
        assert float(out[symbol]) <= float(target[symbol])


def test_lot_rounding_nearest_may_overshoot_by_half_a_lot() -> None:
    carried = _series(A=0.0, B=0.0)
    target = _series(A=0.3125, B=0.5)
    price = {"2024-01-02": {"A": 0.125, "B": 0.125}}
    lot_sizes = {"A": 1, "B": 1}

    out = apply_lot_rounding(target, carried, price, lot_sizes, min_notional=None, mode="nearest", day="2024-01-02")

    assert float(out["A"]) == pytest.approx(0.375)
    assert float(out["A"]) > float(target["A"])


def test_lot_rounding_min_notional_is_judged_after_rounding() -> None:
    carried = _series(A=0.0)
    target = _series(A=0.3125)
    price = {"2024-01-02": {"A": 0.125}}
    lot_sizes = {"A": 1}

    kept = apply_lot_rounding(target, carried, price, lot_sizes, min_notional=0.2, mode="nearest", day="2024-01-02")
    dropped = apply_lot_rounding(target, carried, price, lot_sizes, min_notional=0.3, mode="down", day="2024-01-02")

    assert float(kept["A"]) == pytest.approx(0.375)  # 0.375 >= 0.2
    assert float(dropped["A"]) == 0.0  # 0.25 < 0.3, so the order is cancelled


def test_lot_rounding_leaves_idle_names_alone_and_needs_no_data() -> None:
    """A book that trades nothing must pass through without any injected data."""

    flat = _series(A=0.25, B=0.5)

    assert apply_lot_rounding(flat, flat, None, None, min_notional=None, mode="down", day="2024-01-02").equals(flat)

    moved = _series(A=0.5, B=0.5)
    carried = _series(A=0.25, B=0.5)
    price = {"2024-01-02": {"A": 0.125}}
    lot_sizes = {"A": 1}
    out = apply_lot_rounding(moved, carried, price, lot_sizes, min_notional=None, mode="down", day="2024-01-02")

    assert float(out["A"]) == pytest.approx(0.5)
    assert float(out["B"]) == pytest.approx(0.5)


def test_lot_rounding_missing_inputs_fail_closed() -> None:
    carried = _series(A=0.0)
    target = _series(A=0.5)
    price = {"2024-01-02": {"A": 0.125}}

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, None, {"A": 1}, min_notional=None, mode="down", day="2024-01-02")
    assert str(error.value) == "lot rounding requires the execution_data price panel"

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, price, None, min_notional=None, mode="down", day="2024-01-02")
    assert str(error.value) == "lot rounding requires execution_data.lot_sizes"

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, {"2024-01-02": {}}, {"A": 1}, min_notional=None, mode="down", day="2024-01-02")
    assert str(error.value) == "lot rounding is missing a price for A on 2024-01-02"

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, {"2024-01-02": {"A": 0.0}}, {"A": 1}, min_notional=None, mode="down", day="2024-01-02")
    assert str(error.value) == "lot rounding requires a positive price for A on 2024-01-02"

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, price, {}, min_notional=None, mode="down", day="2024-01-02")
    assert str(error.value) == "lot rounding is missing a lot size for A"

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, price, {"A": 0.5}, min_notional=None, mode="down", day="2024-01-02")
    assert str(error.value) == "lot rounding is missing a lot size for A"

    with pytest.raises(ValueError) as error:
        apply_lot_rounding(target, carried, price, {"A": 1}, min_notional=None, mode="sideways", day="2024-01-02")
    assert str(error.value) == "lot_rounding_mode must be one of ('down', 'nearest') when provided"


# ---------------------------------------------------------------------------
# 8.6 ParticipationCap
# ---------------------------------------------------------------------------

_AMOUNT = {f"d{day}": {"A": 0.1, "B": 0.1} for day in range(6)}


def _cap(carried: pd.Series, pending: dict[str, list[float]], target: pd.Series | None, *, position: int, day: str, **overrides: object):
    kwargs = {"window_keys": ["d0"], "rate": 1.0, "window": 1, "unfilled": "carry", "max_days": 3}
    kwargs.update(overrides)
    return apply_participation_cap(carried, pending, target, _AMOUNT, position=position, day=day, **kwargs)


def test_participation_cap_keeps_the_residual_as_a_carry_plan() -> None:
    """The regression this pins: the residual must survive the call.

    An earlier draft returned a reaped copy of a synthetic pending table, which
    erased the very plan it had just registered -- the cap silently became a
    one-day cancel.  The plan must still be there, and it must keep trading.
    """

    target = _series(A=0.5, B=0.5)
    pending: dict[str, list[float]] = {}

    row, pending = _cap(_series(A=0.0, B=0.0), pending, target, position=0, day="d0")

    assert float(row["A"]) == pytest.approx(0.1)
    assert float(row["B"]) == pytest.approx(0.1)
    assert sorted(pending) == ["A", "B"]
    assert pending["A"][0] == pytest.approx(0.4)
    assert pending["A"][1] == 0.0

    row, pending = _cap(row, pending, None, position=1, day="d1")
    assert float(row["A"]) == pytest.approx(0.2)
    assert pending["A"][0] == pytest.approx(0.3)

    row, pending = _cap(row, pending, None, position=2, day="d2")
    assert float(row["A"]) == pytest.approx(0.3)
    assert pending["A"][0] == pytest.approx(0.2)

    # max_days = 3 counts completed trading days since creation, so the order
    # is cancelled before it can trade on the day the age reaches it.
    row, pending = _cap(row, pending, None, position=3, day="d3")
    assert float(row["A"]) == pytest.approx(0.3)
    assert pending == {}


def test_participation_cap_cancel_drops_the_residual_immediately() -> None:
    pending: dict[str, list[float]] = {}

    row, pending = _cap(_series(A=0.0), pending, _series(A=0.5), position=0, day="d0", unfilled="cancel", max_days=None)

    assert float(row["A"]) == pytest.approx(0.1)
    assert pending == {}


def test_participation_cap_replaces_an_outstanding_plan() -> None:
    pending: dict[str, list[float]] = {"A": [0.4, 0.0]}

    _, pending = _cap(_series(A=0.1), pending, _series(A=0.5), position=1, day="d1")

    # The residual is rebuilt from the new decision (0.5 - 0.1 - 0.1), not
    # carried over from the half-filled old one (0.4 - 0.1).
    assert pending["A"][0] == pytest.approx(0.3)


def test_participation_cap_windows_the_amount_backwards() -> None:
    """``participation_amount_window`` averages the trailing run of days."""

    amount = {"d0": {"A": 1.0}, "d1": {"A": 3.0}}
    target = _series(A=0.5)

    single, _ = apply_participation_cap(
        _series(A=0.0), {}, target, amount, position=1, day="d1", window_keys=["d0", "d1"], rate=0.1, window=1,
        unfilled="cancel", max_days=None,
    )
    paired, _ = apply_participation_cap(
        _series(A=0.0), {}, target, amount, position=1, day="d1", window_keys=["d0", "d1"], rate=0.1, window=2,
        unfilled="cancel", max_days=None,
    )

    assert float(single["A"]) == pytest.approx(0.3)  # rate * 3.0
    assert float(paired["A"]) == pytest.approx(0.2)  # rate * mean(1.0, 3.0)


def test_participation_cap_validation_and_missing_amount() -> None:
    target = _series(A=0.5)
    carried = _series(A=0.0)

    with pytest.raises(ValueError) as error:
        _cap(carried, {}, target, position=0, day="d0", unfilled="carry", max_days=None)
    assert str(error.value) == "participation_max_days must be at least 1 when participation_unfilled is carry"

    with pytest.raises(ValueError) as error:
        _cap(carried, {}, target, position=0, day="d0", rate=0.0)
    assert str(error.value) == "participation_max_rate must be in (0, 1] when provided"

    with pytest.raises(ValueError) as error:
        _cap(carried, {}, target, position=0, day="d0", unfilled="keep")
    assert str(error.value) == "participation_unfilled must be one of ('cancel', 'carry') when provided"

    with pytest.raises(ValueError) as error:
        apply_participation_cap(carried, {}, target, None, position=0, day="d0", window_keys=["d0"], rate=1.0, window=1, unfilled="cancel", max_days=None)
    assert str(error.value) == "participation_cap requires the execution_data amount panel"

    # A panel that answers for the day but not for the name: the cap cannot be
    # evaluated, so the order fails closed instead of being sized by a guess.
    with pytest.raises(ValueError) as error:
        apply_participation_cap(
            carried, {}, target, {"d0": {"B": 1.0}}, position=0, day="d0",
            window_keys=["d0"], rate=1.0, window=1, unfilled="cancel", max_days=None,
        )
    assert str(error.value) == "participation cap is missing an amount for A on d0"

    with pytest.raises(ValueError) as error:
        _cap(carried, {}, target, position=0, day="d0", window=0)
    assert str(error.value) == "participation_amount_window must be at least 1 when provided"


def test_participation_cap_bounds_every_slice_by_the_traded_amount() -> None:
    """The cap is a per-name budget, and the split conserves the requirement.

    Neither half may go missing: no slice is larger than
    ``rate * mean(amount)``, and the slice plus the residual it leaves behind
    reproduces the intended move exactly.
    """

    amount = {"d0": {"A": 0.1, "B": 0.4}, "d1": {"A": 0.1, "B": 0.4}}
    target = _series(A=0.5, B=0.5)
    carried = _series(A=0.0, B=0.0)

    row, pending = apply_participation_cap(
        carried, {}, target, amount, position=0, day="d0", window_keys=["d0", "d1"], rate=0.5, window=1,
        unfilled="carry", max_days=4,
    )

    # rate * amount: 0.05 for A, 0.2 for B.  Both are far below the move the
    # target asks for, so both legs are capped and both leave a residual.
    assert float(row["A"]) == pytest.approx(0.05)
    assert float(row["B"]) == pytest.approx(0.2)
    for symbol in target.index:
        moved = float(row[symbol]) - float(carried[symbol])
        intended = float(target[symbol]) - float(carried[symbol])
        assert abs(moved) <= 0.5 * float(amount["d0"][symbol]) + 1e-12
        assert moved + pending[symbol][0] == pytest.approx(intended)


def test_participation_cap_never_resurrects_an_expired_slice() -> None:
    """A residual dies at ``max_days`` and the next decision starts from the book.

    The trap this pins: rebuilding the residual as ``target - carried`` from
    the *original* target would revive the cancelled 0.3 at the next decision,
    so the plan would outlive its own horizon.  A later target equal to the
    carried book is then a genuine no-op -- which is the observable that keeps
    the expiry honest.
    """

    row, pending = _cap(_series(A=0.0), {}, _series(A=0.5), position=0, day="d0", max_days=2)
    assert float(row["A"]) == pytest.approx(0.1)
    row, pending = _cap(row, pending, None, position=1, day="d1", max_days=2)
    assert float(row["A"]) == pytest.approx(0.2)

    # Two completed trading days are spent: the remaining 0.3 is cancelled
    # without trading, and nothing of it survives into the next decision.
    row, pending = _cap(row, pending, None, position=2, day="d2", max_days=2)
    assert float(row["A"]) == pytest.approx(0.2)
    assert pending == {}

    row, pending = _cap(row, pending, _series(A=0.2), position=3, day="d3", max_days=2)
    assert float(row["A"]) == pytest.approx(0.2)
    assert pending == {}


# ---------------------------------------------------------------------------
# 8.9 StagedExecution
# ---------------------------------------------------------------------------


def test_stage_orders_completes_the_plan_in_exactly_max_days() -> None:
    carried = _series(A=0.0, B=0.0)
    target = _series(A=0.3, B=0.1)
    pending: dict[str, list[float]] = {}

    row, pending = stage_orders(carried, pending, target, position=0, max_days=2, order=["A", "B"])
    # The day's budget is 0.4 / 2 = 0.2 and it is spent in priority order, so
    # the first name is filled to the budget and the second waits for tomorrow.
    assert float(row["A"]) == pytest.approx(0.2)
    assert float(row["B"]) == pytest.approx(0.0)

    row, pending = stage_orders(row, pending, None, position=1, max_days=2, order=["A", "B"])
    assert float(row["A"]) == pytest.approx(0.3)
    assert float(row["B"]) == pytest.approx(0.1)
    assert pending == {}


def test_stage_orders_drops_a_plan_that_ran_out_of_days() -> None:
    carried = _series(A=0.0)
    target = _series(A=1.0)
    pending: dict[str, list[float]] = {}

    row, pending = stage_orders(carried, pending, target, position=0, max_days=1, order=["A"])
    assert float(row["A"]) == pytest.approx(1.0)

    stale = {"A": [0.5, 0.0]}
    row, pending = stage_orders(_series(A=0.2), stale, None, position=1, max_days=1, order=["A"])
    assert float(row["A"]) == pytest.approx(0.2)
    assert pending == {}


def test_stage_orders_replaces_an_outstanding_plan() -> None:
    carried = _series(A=0.15, B=0.05)
    pending = {"A": [0.15, 0.0], "B": [0.05, 0.0]}

    row, pending = stage_orders(carried, pending, _series(A=0.5, B=0.0), position=1, max_days=2, order=["A", "B"])

    # The new decision is 0.35 / -0.05 from the carried book; its first slice
    # is taken today.  Keeping the old plan would have produced 0.3 / 0.1.
    # The slice goes to A first (priority order), so B does not move at all.
    assert float(row["A"]) == pytest.approx(0.35)
    assert float(row["B"]) == pytest.approx(0.05)
    assert pending["A"][0] == pytest.approx(0.15)
    assert pending["B"][0] == pytest.approx(-0.05)
    assert pending["A"][1] == pytest.approx(1.0)
    assert pending["B"][1] == pytest.approx(1.0)


def test_stage_orders_requires_a_positive_horizon() -> None:
    with pytest.raises(ValueError) as error:
        stage_orders(_series(A=0.0), {}, _series(A=0.5), position=0, max_days=0, order=["A"])
    assert str(error.value) == "staged_max_days must be at least 1 when provided"


def test_staged_priority_moves_the_path_not_the_budget() -> None:
    """``staged_priority`` has to be observable, so pin what it changes.

    The day's budget belongs to the outstanding requirement, not to the order
    it is spent in: reversing the service order must leave the per-day traded
    total and the completion day untouched, and move only *who* trades first
    and who is carried.  Under the proportional alternative the budget would be
    split evenly across names, so the first assertion below -- the leading name
    absorbs the whole budget before the next one starts -- could not hold.
    """

    target = _series(A=0.3, B=0.2, C=0.1)

    def walk(order: list[str]) -> tuple[list[pd.Series], list[float], dict[str, list[float]]]:
        carried = _series(A=0.0, B=0.0, C=0.0)
        pending: dict[str, list[float]] = {}
        rows: list[pd.Series] = []
        traded: list[float] = []
        for position in range(3):
            decision = target if position == 0 else None
            previous = carried
            carried, pending = stage_orders(carried, pending, decision, position=position, max_days=3, order=order)
            rows.append(carried)
            traded.append(float(sum(abs(float(carried[name]) - float(previous[name])) for name in target.index)))
        return rows, traded, pending

    forward_rows, forward_traded, forward_pending = walk(["A", "B", "C"])
    reverse_rows, reverse_traded, reverse_pending = walk(["C", "B", "A"])

    # Same budget every day, same completion: only the path differs.
    assert forward_traded == pytest.approx([0.2, 0.2, 0.2])
    assert reverse_traded == pytest.approx([0.2, 0.2, 0.2])
    assert not forward_rows[0].equals(reverse_rows[0])
    assert not forward_rows[1].equals(reverse_rows[1])
    assert forward_rows[-1].equals(reverse_rows[-1])
    assert forward_pending == {}
    assert reverse_pending == {}

    # Priority order, made explicit: A is filled to the budget alone, while the
    # reverse order fills C outright (0.1) and then gives B the remainder.
    assert float(forward_rows[0]["A"]) == pytest.approx(0.2)
    assert float(forward_rows[0]["B"]) == pytest.approx(0.0)
    assert float(reverse_rows[0]["C"]) == pytest.approx(0.1)
    assert float(reverse_rows[0]["B"]) == pytest.approx(0.1)
    assert float(reverse_rows[0]["A"]) == pytest.approx(0.0)


def test_order_names_reads_each_priority_the_way_it_is_documented() -> None:
    """``weight`` ranks by the size of the intended move, ``score`` by score."""

    carried = _series(A=0.1, B=0.2, C=0.3)
    executed = _series(A=0.4, B=0.0, C=0.31)
    score = _series(A=0.1, B=0.9, C=0.5)

    # |delta| = 0.3 / 0.2 / 0.01, largest adjustment first.
    assert _order_names(executed, carried, "weight", None) == ["A", "B", "C"]
    # Best score first, and the score panel is what it is ranked on.
    assert _order_names(executed, carried, "score", score) == ["B", "C", "A"]

    with pytest.raises(ValueError) as error:
        _order_names(executed, carried, "score", None)
    assert str(error.value) == "staged_execution with score priority requires the score panel"
    with pytest.raises(ValueError) as error:
        _order_names(executed, carried, "volatility", None)
    assert str(error.value) == "unknown staged_priority: volatility"


def test_plan_order_resolves_the_priority_default() -> None:
    """``staged_priority`` is optional and the default is ``weight``.

    An unset knob must behave exactly like the documented default rather than
    falling through to the other reading: a score panel supplied alongside
    ``priority=None`` is ignored, so the default cannot drift into ``score``
    just because the caller happened to have a panel around.
    """

    carried = _series(A=0.1, B=0.2)
    executed = _series(A=0.4, B=0.0)
    score = _series(A=0.1, B=0.9)

    # |delta| = 0.3 / 0.2 vs score 0.1 / 0.9: the two readings disagree, so the
    # assertions below can tell which one was taken.
    assert plan_order(executed, carried, priority=None, score_row=None) == ["A", "B"]
    assert plan_order(executed, carried, priority=None, score_row=score) == ["A", "B"]
    assert plan_order(executed, carried, priority="weight", score_row=None) == ["A", "B"]
    assert plan_order(executed, carried, priority="score", score_row=score) == ["B", "A"]

    with pytest.raises(ValueError) as error:
        plan_order(executed, carried, priority="score", score_row=None)
    assert str(error.value) == "staged_execution with score priority requires the score panel"


# ---------------------------------------------------------------------------
# order-layer entry point: 8.9 allocation, then 8.8 masking, then 8.7 rounding
# ---------------------------------------------------------------------------


def _layer(
    carried: pd.Series,
    pending: dict[str, list[float]],
    target: pd.Series | None,
    **overrides: object,
) -> tuple[pd.Series, dict[str, list[float]]]:
    kwargs: dict[str, object] = {
        "position": 0,
        "day": "d0",
        "calendar": ["d0", "d1"],
        "order": ["A", "B"],
        "participation_rate": None,
        "participation_window": None,
        "participation_unfilled": None,
        "participation_max_days": None,
        "staged_max_days": None,
        "lot_rounding": False,
        "lot_min_notional": None,
        "lot_rounding_mode": None,
        "side_aware_tradability": None,
        "amount": None,
        "price": None,
        "lot_sizes": None,
        "halt": None,
        "limits": None,
    }
    kwargs.update(overrides)
    return apply_execution_layer(carried, pending, target, **kwargs)  # type: ignore[arg-type]


def test_execution_layer_masks_the_slice_and_not_the_target() -> None:
    """The mask sees the order placed today, not the decision behind it.

    Masking the target row first would let the allocator re-inflate a
    direction the day cannot trade, and it would erase that direction from the
    plan altogether.  What the layer must do instead is allocate, then clamp
    the slice: the blocked slice is cancelled -- the residual the allocator
    already took is not put back -- and the name keeps only the part that was
    never taken.
    """

    carried = _series(A=0.0, B=0.0)
    target = _series(A=0.3, B=0.1)
    pending: dict[str, list[float]] = {}

    executed, pending = _layer(
        carried, pending, target, staged_max_days=2, side_aware_tradability="block_both", halt={"d0": {"A": "suspended"}}
    )

    # Day one's budget is 0.4 / 2 = 0.2 and A is served first, so the slice the
    # halt blocks is A's 0.2.  Nothing moves for either name.
    assert float(executed["A"]) == pytest.approx(0.0)
    assert float(executed["B"]) == pytest.approx(0.0)
    # A's plan keeps 0.3 - 0.2 = 0.1: the blocked 0.2 is cancelled, not
    # re-queued, and B's untouched 0.1 is still waiting for tomorrow.
    assert pending["A"][0] == pytest.approx(0.1)
    assert pending["B"][0] == pytest.approx(0.1)

    # Serve B tomorrow: the interrupted plan still finishes.
    executed, pending = _layer(
        executed,
        pending,
        None,
        position=1,
        day="d1",
        staged_max_days=2,
        side_aware_tradability="block_both",
        halt={"d1": {"A": "suspended"}},
    )
    assert float(executed["B"]) == pytest.approx(0.1)
    assert float(executed["A"]) == pytest.approx(0.0)
    assert pending == {}


def test_execution_layer_rounds_the_slice_and_not_the_target() -> None:
    """The lot grid shaves the order placed today, and the shave is cancelled.

    Rounding the target row first would hand the allocator a whole-lot plan and
    hide the shave from the residual.  The layer rounds last instead, so the
    shaved part leaves the plan without coming back: the target asked for 0.27
    and the two-day plan lands on 0.2 against an empty pending table.
    """

    price = {"d0": {"A": 0.05}, "d1": {"A": 0.05}}
    lot_sizes = {"A": 1}
    pending: dict[str, list[float]] = {}

    executed, pending = _layer(
        _series(A=0.0),
        pending,
        _series(A=0.27),
        order=["A"],
        staged_max_days=2,
        lot_rounding=True,
        price=price,
        lot_sizes=lot_sizes,
    )

    # Day one's budget is 0.27 / 2 = 0.135, which is 2.7 lots of 0.05: ``down``
    # keeps two (0.1) and cancels the 0.035 it had to shave.
    assert float(executed["A"]) == pytest.approx(0.1)
    assert pending["A"][0] == pytest.approx(0.135)

    executed, pending = _layer(
        executed,
        pending,
        None,
        position=1,
        day="d1",
        order=["A"],
        staged_max_days=2,
        lot_rounding=True,
        price=price,
        lot_sizes=lot_sizes,
    )

    # 0.1 + 0.135 = 0.235 is 4.7 lots, so the position lands on 0.2 and the
    # last shave is cancelled with the plan instead of being re-queued.
    assert float(executed["A"]) == pytest.approx(0.2)
    assert pending == {}


def test_execution_layer_prefers_the_participation_cap_when_given_both() -> None:
    """The two allocators share one branch, in a fixed order.

    ``participation_cap`` and ``staged_execution`` are mutually exclusive at the
    spec level, so this is the shape the branch has to have for the guard to be
    the only thing keeping a caller out: the cap is consulted first and the
    staged plan is never entered.  Pinning it keeps a future reordering from
    silently making the spec-level guard load-bearing in a different way.
    """

    amount = {"d0": {"A": 0.1}}
    target = _series(A=0.5)

    capped, capped_pending = _layer(
        _series(A=0.0),
        {},
        target,
        order=["A"],
        calendar=["d0"],
        amount=amount,
        participation_rate=0.5,
        participation_window=1,
        participation_unfilled="carry",
        participation_max_days=3,
    )
    both, both_pending = _layer(
        _series(A=0.0),
        {},
        target,
        order=["A"],
        calendar=["d0"],
        amount=amount,
        participation_rate=0.5,
        participation_window=1,
        participation_unfilled="carry",
        participation_max_days=3,
        staged_max_days=2,
    )

    assert float(capped["A"]) == pytest.approx(0.05)
    assert both.equals(capped)
    assert both_pending == capped_pending


# ---------------------------------------------------------------------------
# 9.1 PortfolioVolatilityTarget
# ---------------------------------------------------------------------------


def _scale(values: list[float], **overrides: float | int) -> float:
    kwargs: dict[str, float | int] = {"target_annual": 0.1, "window": 8, "min_scale": 0.5, "max_scale": 1.5, "periods_per_year": 252}
    kwargs.update(overrides)
    return volatility_scale(values, **kwargs)  # type: ignore[arg-type]


def test_volatility_scale_declared_fallbacks() -> None:
    """Short history does nothing; a flat book is not a reason to de-risk."""

    assert _scale([]) == 1.0
    assert _scale([0.01]) == 1.0
    assert _scale([0.5, 0.5, 0.5, 0.5]) == 1.5


def test_volatility_scale_clips_both_ends() -> None:
    volatile = _scale([0.1, -0.1, 0.1, -0.1])
    calm = _scale([0.0001, -0.0001])

    assert volatile == 0.5
    assert calm == 1.5


def test_volatility_scale_tracks_the_target_inside_the_band() -> None:
    values = [0.008, -0.008, 0.008, -0.008, 0.008, -0.008, 0.008, -0.008]
    scale = _scale(values, target_annual=0.2, window=8, min_scale=0.1, max_scale=10.0)

    std = pd.Series(values).std(ddof=1)
    assert scale == pytest.approx(0.2 / (std * math.sqrt(252)))
    assert 0.1 < scale < 10.0


def test_volatility_scale_reads_only_the_requested_window() -> None:
    """A wild run before the window must not leak into today's estimate."""

    quiet = [0.5, -0.5] * 4
    wild = [0.001, -0.001] * 4
    windowed = _scale(quiet + wild, window=8, min_scale=0.01, max_scale=100.0)
    quiet_only = _scale(quiet, window=8, min_scale=0.01, max_scale=100.0)

    # Only the trailing eight observations are read: a wild run inside the
    # window moves the estimate, the same run pushed out of it does not.
    assert windowed != quiet_only
    assert _scale(wild + quiet, window=8, min_scale=0.01, max_scale=100.0) == pytest.approx(quiet_only)


# ---------------------------------------------------------------------------
# 9.2 DrawdownStateMachine
# ---------------------------------------------------------------------------

_THRESHOLDS = {"warning": 0.03, "reduce": 0.06, "halt": 0.10}
_SCALES = {"normal": 1.0, "warning": 0.8, "reduce": 0.5, "halt": 0.0}


def _nav(machine: dict[str, object], nav: float, **overrides: object) -> float:
    kwargs: dict[str, object] = {"thresholds": _THRESHOLDS, "scales": _SCALES, "confirm_days": 2, "recovery": 0.02}
    kwargs.update(overrides)
    return drawdown_state_machine([nav], machine=machine, **kwargs)  # type: ignore[arg-type]


def test_drawdown_state_machine_needs_consecutive_days() -> None:
    """One bad day is not a halt."""

    machine: dict[str, object] = {}
    assert _nav(machine, 1.0) == 1.0
    assert _nav(machine, 0.95) == 1.0  # warning band, day 1 of 2
    assert machine["state"] == "normal"
    assert _nav(machine, 0.95) == 0.8  # day 2 of 2 confirms
    assert machine["state"] == "warning"


def test_drawdown_state_machine_oscillation_never_confirms() -> None:
    """Alternating bands give no consecutive run, so nothing is adopted."""

    machine: dict[str, object] = {}
    _nav(machine, 1.0)
    for nav in (0.95, 0.93, 0.95, 0.93, 0.95):
        _nav(machine, nav)

    assert machine["state"] == "normal"


def test_drawdown_state_machine_climbs_one_tier_at_a_time() -> None:
    machine: dict[str, object] = {}

    _nav(machine, 1.0)
    for _ in range(2):
        _nav(machine, 0.95)
    assert machine["state"] == "warning"

    for _ in range(2):
        _nav(machine, 0.93)
    assert machine["state"] == "reduce"

    for _ in range(2):
        _nav(machine, 0.89)
    assert machine["state"] == "halt"
    assert _nav(machine, 0.89) == 0.0


def test_drawdown_state_machine_recovers_from_its_deepest_point() -> None:
    """The regression this pins: recovery must be reachable at all.

    ``deepest`` is a running minimum, so a recovery measure of
    ``(deepest - nav)`` is non-positive by construction and the de-escalation
    branch can never fire.  The bounce off the low is what narrows the
    drawdown, and it must take the full confirmation window.
    """

    machine: dict[str, object] = {}
    _nav(machine, 1.0)
    for _ in range(2):
        _nav(machine, 0.89)
    assert machine["state"] == "halt"

    # 0.91 is a 0.02 bounce off the 0.89 low: enough for one day only.
    assert _nav(machine, 0.91) == 0.0
    assert machine["state"] == "halt"
    # A second confirmed day steps down a single tier, not to normal.
    assert _nav(machine, 0.91) == 0.5
    assert machine["state"] == "reduce"


def test_drawdown_state_machine_recovery_needs_a_real_bounce() -> None:
    machine: dict[str, object] = {}
    _nav(machine, 1.0)
    for _ in range(2):
        _nav(machine, 0.89)

    # 0.895 is only a 0.005 bounce, well short of the 0.02 recovery.
    for _ in range(4):
        _nav(machine, 0.895)

    assert machine["state"] == "halt"


def test_drawdown_state_machine_declares_the_flat_halt_absorbing_state() -> None:
    """``drawdown_scales['halt'] = 0.0`` freezes the book; pin the consequence.

    A halt scale of zero moves the book to cash, so NAV stops moving and the
    severity the state is judged on stops moving with it: the observation
    equals the state, no de-escalation branch is entered, and the book stays in
    the halt band for good.  The setting is declared, not patched -- a reader
    who wants the ladder to be able to come back has to give ``halt`` a
    positive scale -- so what this test fixes is the *consequence*, not a fix.
    """

    machine: dict[str, object] = {}
    _nav(machine, 1.0)
    for _ in range(2):
        _nav(machine, 0.89)
    assert machine["state"] == "halt"

    for _ in range(6):
        assert _nav(machine, 0.89) == 0.0
    assert machine["state"] == "halt"
    # A frozen book never even starts a de-escalation run.
    assert int(machine["recovery_days"]) == 0


def test_drawdown_state_machine_new_high_closes_the_segment() -> None:
    """A new high must not strand the ladder in the state it climbed to.

    ``deepest`` is the trough of the *open* drawdown segment, so a new
    high-water mark moves ``high_water`` and leaves ``deepest`` alone; the
    recovery gate reads the bounce off that trough, which is what lets a fully
    recovered book walk back down.  The segment closes -- and the yardstick is
    reset -- only when the ladder reaches ``normal``.
    """

    machine: dict[str, object] = {}
    _nav(machine, 1.0)
    for _ in range(2):
        _nav(machine, 0.89)
    assert machine["state"] == "halt"

    assert _nav(machine, 1.05) == 0.0
    assert float(machine["high_water"]) == pytest.approx(1.05)
    # Still the low of the open segment: this is the number the recovery gate
    # measures the bounce against.
    assert float(machine["deepest"]) == pytest.approx(0.89)

    # Two confirmed days per tier, one tier at a time, all the way down.
    for expected_state, expected_scale in (("reduce", 0.5), ("warning", 0.8), ("normal", 1.0)):
        for _ in range(2):
            scale = _nav(machine, 1.05)
        assert machine["state"] == expected_state
        assert scale == pytest.approx(expected_scale)

    # The return to normal is where the old trough stops being the yardstick.
    assert float(machine["deepest"]) == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# spec level: 8.6-8.9 / 9.1 / 9.2 / 9.5 keys, companions and payload presence
# ---------------------------------------------------------------------------

#: The spec-level drawdown scales cover exactly the three escalatable states;
#: ``normal`` is absent on purpose (it always scales by 1.0), which is why this
#: map is not ``_SCALES`` -- the pure function tolerates the extra key, the spec
#: does not.
_SPEC_SCALES = {"warning": 0.8, "reduce": 0.5, "halt": 0.0}


_ORDER_LAYER_SPEC_ERRORS = [
    # 8.6 ParticipationCap
    ({"participation_max_rate": 0.0}, "participation_max_rate must be in (0, 1] when provided"),
    ({"participation_max_rate": 1.5}, "participation_max_rate must be in (0, 1] when provided"),
    ({"participation_max_rate": True}, "participation_max_rate must be in (0, 1] when provided"),
    ({"participation_max_rate": 0.5, "participation_unfilled": "hold"}, "participation_unfilled must be one of ('cancel', 'carry') when provided"),
    ({"participation_max_rate": 0.5, "participation_unfilled": "carry"}, "participation_max_days must be at least 1 when participation_unfilled is carry"),
    ({"participation_max_rate": 0.5, "participation_max_days": 0}, "participation_max_days must be at least 1 when provided"),
    ({"participation_max_rate": 0.5, "participation_max_days": 1.5}, "participation_max_days must be a whole number when provided"),
    ({"participation_max_rate": 0.5, "participation_amount_window": 0}, "participation_amount_window must be at least 1 when provided"),
    ({"participation_max_days": 3}, "participation_max_days requires participation_max_rate"),
    ({"participation_unfilled": "cancel"}, "participation_unfilled requires participation_max_rate"),
    ({"participation_amount_window": 2}, "participation_amount_window requires participation_max_rate"),
    # 8.7 LotRounding
    ({"lot_rounding": "yes"}, "lot_rounding must be a boolean"),
    ({"lot_min_notional": 0.0}, "lot_min_notional must be positive when provided"),
    ({"lot_rounding": True, "lot_rounding_mode": "up"}, "lot_rounding_mode must be one of ('down', 'nearest') when provided"),
    ({"lot_min_notional": 1.0}, "lot_min_notional requires lot_rounding"),
    ({"lot_rounding_mode": "nearest"}, "lot_rounding_mode requires lot_rounding"),
    # 8.8 SideAwareTradability
    ({"side_aware_tradability": "block"}, "side_aware_tradability must be one of ('block_buy', 'block_sell', 'block_both') when provided"),
    # 8.9 StagedExecution
    ({"staged_max_days": 0}, "staged_max_days must be at least 1 when provided"),
    ({"staged_max_days": 2.5}, "staged_max_days must be a whole number when provided"),
    ({"staged_max_days": 2, "staged_priority": "momentum"}, "staged_priority must be one of ('score', 'weight') when provided"),
    ({"staged_priority": "score"}, "staged_priority requires staged_max_days"),
    # 8.6 x 8.9
    ({"participation_max_rate": 0.5, "staged_max_days": 2}, "participation_cap and staged_execution are mutually exclusive"),
    # 9.1 PortfolioVolatilityTarget
    ({"vol_target_annual": 0.0}, "vol_target_annual must be positive when provided"),
    ({"vol_target_window": 1}, "vol_target_window must be at least 2 when provided"),
    ({"vol_target_window": 5}, "vol_target_window requires vol_target_annual"),
    ({"vol_target_annual": 0.2}, "vol_target_window is required when vol_target_annual is provided"),
    (
        {"vol_target_annual": 0.2, "vol_target_window": 5, "vol_target_min_scale": 0.5, "vol_target_max_scale": 0.9},
        "vol_target_max_scale must be at least 1 when provided",
    ),
    (
        {"vol_target_annual": 0.2, "vol_target_window": 5, "vol_target_min_scale": 1.5, "vol_target_max_scale": 2.0},
        "vol_target_min_scale must be in (0, 1] when provided",
    ),
    (
        {"vol_target_annual": 0.2, "vol_target_window": 5, "vol_target_min_scale": 1.0, "vol_target_max_scale": 2.0},
        "vol_target_min_scale must be below 1 and vol_target_max_scale at least 1",
    ),
    # 9.2 DrawdownStateMachine
    (
        {"drawdown_thresholds": {"warning": 0.0, "reduce": 0.06, "halt": 0.10}},
        "drawdown_thresholds must be in (0, 1] when provided",
    ),
    (
        {"drawdown_thresholds": {"warning": 0.06, "reduce": 0.03, "halt": 0.10}},
        "drawdown_thresholds must be strictly increasing",
    ),
    ({"drawdown_thresholds": {"warning": 0.03, "reduce": 0.06}}, "drawdown_thresholds must cover exactly the drawdown states"),
    ({"drawdown_confirm_days": 2}, "drawdown_confirm_days requires drawdown_thresholds"),
    ({"drawdown_thresholds": _THRESHOLDS}, "drawdown_confirm_days is required when drawdown_thresholds is provided"),
    (
        {"drawdown_thresholds": _THRESHOLDS, "drawdown_confirm_days": 0},
        "drawdown_confirm_days must be at least 1 when provided",
    ),
    (
        {"drawdown_thresholds": _THRESHOLDS, "drawdown_confirm_days": 2, "drawdown_recovery": 2.0},
        "drawdown_recovery must be in (0, 1] when provided",
    ),
    (
        {
            "drawdown_thresholds": _THRESHOLDS,
            "drawdown_confirm_days": 2,
            "drawdown_recovery": 0.02,
            "drawdown_scales": {"warning": 0.8, "reduce": 0.5, "halt": 2.0},
        },
        "drawdown_scales must be in [0, 1] when provided",
    ),
]


@pytest.mark.parametrize(("overrides", "message"), _ORDER_LAYER_SPEC_ERRORS)
def test_order_layer_spec_rejects_malformed_keys(overrides: dict[str, object], message: str) -> None:
    """Every knob of the batch fails closed with its own message.

    These strings are the wire contract: the worker mirror re-raises the same
    text for the same payload, so a reworded message here is a parity break
    wherever the request is built by hand.
    """

    with pytest.raises(ValueError, match=re.escape(message)):
        BacktestSpec(**overrides)  # type: ignore[arg-type]


def test_signal_spec_rejects_decay_companions_without_the_enabler() -> None:
    """9.5 follows the same ``x requires y`` convention as 8.6-9.2."""

    for companion, value in (("decay_fail_periods", 2), ("decay_ic_threshold", 0.01), ("decay_scale", 0.5)):
        with pytest.raises(ValueError, match=re.escape(f"{companion} requires decay_window")):
            SignalSpec(**{companion: value})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=re.escape("decay_window must be at least 2")):
        SignalSpec(decay_window=1)
    with pytest.raises(ValueError, match=re.escape("decay_window must be a whole number")):
        SignalSpec(decay_window=2.5)
    with pytest.raises(ValueError, match=re.escape("decay_fail_periods must be at least 1")):
        SignalSpec(decay_window=4, decay_fail_periods=0)
    with pytest.raises(ValueError, match=re.escape("decay_ic_threshold must be finite")):
        SignalSpec(decay_window=4, decay_ic_threshold=float("nan"))
    with pytest.raises(ValueError, match=re.escape("decay_scale must be in [0, 1]")):
        SignalSpec(decay_window=4, decay_scale=1.5)


def test_order_layer_spec_accepts_the_documented_boundaries() -> None:
    """The boundaries the messages above are drawn around are admissible."""

    spec = BacktestSpec(
        participation_max_rate=1.0,
        participation_max_days=1,
        participation_unfilled="cancel",
        participation_amount_window=1,
    )
    assert spec.participation_max_rate == 1.0

    staged = BacktestSpec(staged_max_days=1)
    assert staged.staged_priority == "weight"

    rounding = BacktestSpec(lot_rounding=True)
    assert rounding.lot_rounding_mode == "down"

    risk = BacktestSpec(
        vol_target_annual=1e-6,
        vol_target_window=2,
        vol_target_min_scale=0.999,
        vol_target_max_scale=1.0,
    )
    assert risk.vol_target_min_scale < 1.0 <= risk.vol_target_max_scale

    drawdown = BacktestSpec(
        drawdown_thresholds=_THRESHOLDS,
        drawdown_confirm_days=1,
        drawdown_recovery=1.0,
        drawdown_scales=_SPEC_SCALES,
    )
    assert drawdown.drawdown_scales == _SPEC_SCALES


def test_disabled_order_layer_keys_stay_out_of_the_canonical_payload() -> None:
    """A disabled operator must not move the identity hash, key by key.

    Every new field travels only while its enabler is set, so the canonical
    payload is the enforcement point: this asserts the absence of *all* of them
    at once, which is what keeps a pre-batch candidate's ``strategy_id`` stable.
    """

    keys = {
        "participation_max_rate",
        "participation_max_days",
        "participation_unfilled",
        "participation_amount_window",
        "lot_rounding",
        "lot_min_notional",
        "lot_rounding_mode",
        "side_aware_tradability",
        "staged_max_days",
        "staged_priority",
        "vol_target_annual",
        "vol_target_window",
        "vol_target_min_scale",
        "vol_target_max_scale",
        "drawdown_thresholds",
        "drawdown_confirm_days",
        "drawdown_recovery",
        "drawdown_scales",
    }
    assert not (keys & set(BacktestSpec().as_payload()))
    assert not ({"decay_window", "decay_fail_periods", "decay_ic_threshold", "decay_scale"} & set(SignalSpec().as_payload()))

    # And an enabled group lands as a resolved group, not as a lone key: the
    # executors read the companions, so they are what the payload has to carry.
    enabled = BacktestSpec(participation_max_rate=0.5, participation_max_days=3).as_payload()
    assert enabled["participation_max_rate"] == 0.5
    assert enabled["participation_max_days"] == 3
    assert enabled["participation_unfilled"] == "carry"
    assert enabled["participation_amount_window"] == 1
    assert not ({"lot_rounding", "staged_max_days", "vol_target_annual", "drawdown_thresholds"} & set(enabled))


# ---------------------------------------------------------------------------
# 9.5 FactorDecayMonitor
# ---------------------------------------------------------------------------


def _decay_returns() -> pd.DataFrame:
    """Six trading days whose ranking is contradicted first and confirmed next."""

    rows = [
        (0.01, 0.01, 0.01),
        (-0.10, -0.05, 0.10),
        (-0.10, -0.05, 0.10),
        (0.10, 0.05, -0.10),
        (0.10, 0.05, -0.10),
        (0.01, 0.01, 0.01),
    ]
    return pd.DataFrame(rows, index=pd.bdate_range("2024-01-01", periods=6), columns=["A", "B", "C"])


def _decay_ranks() -> pd.DataFrame:
    """Decisions on rows 0, 2 and 4 of ``_decay_returns``; the last one is open."""

    return pd.DataFrame(
        [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 1.0]],
        index=_decay_returns().index[[0, 2, 4]],
        columns=["A", "B", "C"],
    )


def _decay_scales(ranks: dict[str, pd.DataFrame], returns: pd.DataFrame, **overrides: object) -> dict[str, pd.Series]:
    params: dict[str, object] = {"lag": 1, "window": 4, "fail_periods": 1, "threshold": 0.0, "scale": 0.25}
    params.update(overrides)
    return factor_decay_scales(ranks, returns, **params)  # type: ignore[arg-type]


def test_decay_state_needs_an_unbroken_run_strictly_below_the_threshold() -> None:
    """9.5 reads the trailing run of consecutive failures, not a count."""

    # An IC exactly on the threshold is a borderline observation, not a failure.
    assert factor_decay_state([0.05, 0.05], window=4, fail_periods=2, threshold=0.05, scale=0.5) == 1.0
    # One observation is one period short of the run the monitor waits for.
    assert factor_decay_state([0.04], window=4, fail_periods=2, threshold=0.05, scale=0.5) == 1.0
    assert factor_decay_state([0.04, 0.04], window=4, fail_periods=2, threshold=0.05, scale=0.5) == 0.5
    # A recovery in between breaks the run: only the tail counts.
    assert factor_decay_state([0.04, 0.04, 0.9], window=4, fail_periods=2, threshold=0.05, scale=0.5) == 1.0
    assert factor_decay_state([0.9, 0.04, 0.04], window=4, fail_periods=2, threshold=0.05, scale=0.5) == 0.5


def test_decay_state_ignores_unobservable_periods_and_reads_only_the_window() -> None:
    """``None`` means "not observable yet", which is not a failure."""

    assert factor_decay_state([None, 0.04, None, 0.04], window=4, fail_periods=2, threshold=0.05, scale=0.5) == 0.5
    # The window is an upper bound on the history the state reads.
    assert factor_decay_state([0.04, 0.04], window=1, fail_periods=2, threshold=0.05, scale=0.5) == 1.0
    non_finite = factor_decay_state(
        [0.04, float("nan"), float("inf")], window=4, fail_periods=1, threshold=0.05, scale=0.5
    )
    assert non_finite == 0.5
    # An empty history cannot fail, the way an empty sample cannot scale.
    assert factor_decay_state([], window=4, fail_periods=1, threshold=0.05, scale=0.5) == 1.0


def test_decay_ic_series_reads_only_completed_windows() -> None:
    """The IC of a decision is the rank correlation over the window it held."""

    values = factor_ic_series(_decay_ranks(), _decay_returns(), positions=[0, 2, 4], lag=1)
    assert values[0] == pytest.approx(-1.0)
    assert values[1] == pytest.approx(1.0)
    # The trailing decision has no completed window yet.
    assert values[2] is None

    # Neither has one whose window runs past the end of the return panel.
    truncated = factor_ic_series(_decay_ranks(), _decay_returns().iloc[:4], positions=[0, 2, 4], lag=1)
    assert truncated[0] == pytest.approx(-1.0)
    assert truncated[1] is None and truncated[2] is None

    # One name is not a cross-section.
    single = _decay_ranks()
    single.loc[:, ["A", "B"]] = float("nan")
    assert factor_ic_series(single, _decay_returns(), positions=[0, 2, 4], lag=1)[0] is None


def test_decay_scales_never_read_an_ic_whose_window_is_still_open() -> None:
    """A decision only sees ICs whose holding window closed before it.

    The first IC belongs to the window that opens after decision 0 and closes
    before decision 2, so decisions 0 and 1 both see an empty history -- the
    monitor stays silent until the evidence it would read actually exists.
    """

    scales = _decay_scales({"value": _decay_ranks()}, _decay_returns())["value"]
    assert list(scales) == [1.0, 1.0, 0.25]
    assert list(scales.index) == list(_decay_ranks().index)
    # The one failure is not enough for a two-period run.
    assert list(_decay_scales({"value": _decay_ranks()}, _decay_returns(), fail_periods=2)["value"]) == [1.0, 1.0, 1.0]


def test_decay_scales_need_their_decision_dates_on_the_returns_panel() -> None:
    assert factor_decay_scales({}, _decay_returns(), lag=1, window=4, fail_periods=1, threshold=0.0, scale=0.25) == {}

    elsewhere = _decay_ranks().set_axis(pd.bdate_range("2025-01-01", periods=3))
    with pytest.raises(ValueError) as error:
        _decay_scales({"value": elsewhere}, _decay_returns())
    assert str(error.value) == "factor decay monitor requires the decision dates to be present in the returns panel"
