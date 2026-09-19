"""Order layer, portfolio-risk scaling and the factor-decay monitor.

Roadmap 8.6-8.9 (order layer), 9.1-9.2 (portfolio risk scaling) and 9.5
(factor decay).

The engine is a pure weight engine: ``start_weights`` is a weight row, the cost
formula reads ``weights - carried`` and drift is applied per name.  The four
execution operators (8.6 ParticipationCap, 8.7 LotRounding, 8.8
SideAwareTradability, 8.9 StagedExecution) all need prices, lot sizes, traded
amounts or halt/limit states, none of which a weight vector can express, so
they live here as explicit functions over weights plus explicitly injected
data.  The two portfolio-risk scalings (9.1 PortfolioVolatilityTarget, 9.2
DrawdownStateMachine) reduce the target row to a multiple of itself; they need
the book's own realised return path, which only the per-trading-day main loop
knows, so they are called from the loop as well, ahead of the drift-band gate.

Every public function here mirrors ``scripts/gf_quant_strategy_worker.py`` value
for value (the ``apply_*`` / ``stage_orders`` / ``volatility_scale`` /
``drawdown_state_machine`` copies live under the same names with a leading
underscore); ``test/scenarios/gf_quant/test_strategy_worker_parity.py`` runs both
sides on identical injected panels and fails on any drift.

Unit convention: the book is denominated in a capital of :data:`CAPITAL` (one
unit of account, never NAV-normalised -- the engine's cash leg is
``1 - sum(weights)`` and ``carried`` drifts in the same units).  The injected
``amount`` panel must therefore be expressed in that same unit of account, so a
weight and an amount are directly comparable and the participation cap
``rate * amount`` bounds a weight delta without a second scale factor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd

from rdagent.scenarios.gf_quant.strategy.universe import PRICE_LIMIT_TOLERANCE

#: Unit of account of one book.  ``capital = 1.0`` is pinned rather than
#: derived from a NAV path: it is the same convention the cost formulas and
#: the drift recursion already use.
CAPITAL = 1.0

#: Snap tolerance used when a lot count lands an ulp below a whole number.
#: The subsequent ``amount > target`` guard keeps `down` conservative.
LOT_SNAP = 1e-12


def day_key(value: object) -> str:
    """Normalised ISO date key used by every explicitly injected panel."""

    return pd.Timestamp(str(value)).normalize().date().isoformat()


def _row(panel: Mapping[str, object] | None, day: str) -> Mapping[str, object] | None:
    if panel is None:
        return None
    row = panel.get(day)
    return row if isinstance(row, Mapping) else None


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# 8.8 SideAwareTradability
# ---------------------------------------------------------------------------


def apply_side_aware_tradability(
    carried: pd.Series,
    target: pd.Series,
    halt: Mapping[str, object] | None,
    limits: Mapping[str, object] | None,
    price: Mapping[str, object] | None,
    *,
    mode: str,
    day: str,
) -> pd.Series:
    """Apply the side-aware tradability mask to the *target* side only.

    The mask answers two independent questions per name -- may it be bought
    today, may it be sold today -- from the injected halt/limit state, and then
    clamps the executed weight so a blocked direction cannot move: a blocked
    buy caps the weight at the carried weight, a blocked sell floors it there.
    Holdings themselves are never touched, so "not allowed to buy" can never
    turn into "forced to sell" (roadmap 8.8 explicitly allows holding a name
    that cannot be bought).  A blocked slice is cancelled rather than being
    re-queued, matching the handoff the roadmap 8.4 lock uses.

    ``mode`` maps a state to the direction it blocks rather than selecting
    which directions to inspect -- both sides are always queried:

    * ``block_both`` (default): halt blocks both sides, limit-up blocks buys,
      limit-down blocks sells; the only reading that avoids fake fills.
    * ``block_buy`` / ``block_sell``: any abnormal state blocks that one side
      and leaves the other side fully open.  These are single-sided variants,
      not uniform tightenings of ``block_both`` -- ``block_buy`` is stricter on
      the buy side (a limit-down now blocks too) and looser on the sell side
      (a halt or a limit-up no longer blocks a sale).  Whichever way a given
      state resolves, the engine is making a fill assumption a real book would
      not, so the two side variants are the settings to reach for when
      studying how much of a result depends on the tradability assumption.
    """

    out = target.copy()
    if mode not in {"block_buy", "block_sell", "block_both"}:
        raise ValueError(f"side_aware_tradability mode must be one of ('block_buy', 'block_sell', 'block_both') when provided")
    if halt is None and limits is None:
        raise ValueError("side_aware_tradability requires execution_data.halt or execution_data.price_limits")
    halt_row = _row(halt, day)
    limit_row = _row(limits, day)
    price_row = _row(price, day)
    if limit_row and price is None:
        raise ValueError("side_aware_tradability requires the execution_data price panel")
    for symbol in out.index:
        state = None if halt_row is None else halt_row.get(symbol)
        halted = state is not None and str(state) != ""
        at_upper = False
        at_lower = False
        bounds = None if limit_row is None else limit_row.get(symbol)
        if isinstance(bounds, Mapping) and (bounds.get("upper") is not None or bounds.get("lower") is not None):
            close = None if price_row is None else _finite(price_row.get(symbol))
            if close is None:
                raise ValueError(f"side_aware_tradability is missing a price for {symbol} on {day}")
            upper = _finite(bounds.get("upper"))
            lower = _finite(bounds.get("lower"))
            at_upper = upper is not None and close >= upper - PRICE_LIMIT_TOLERANCE
            at_lower = lower is not None and close <= lower + PRICE_LIMIT_TOLERANCE
        untradable = halted or at_upper or at_lower
        if mode == "block_both":
            block_buy = halted or at_upper
            block_sell = halted or at_lower
        elif mode == "block_buy":
            block_buy = untradable
            block_sell = False
        else:
            block_buy = False
            block_sell = untradable
        delta = float(out[symbol]) - float(carried[symbol])
        if (block_buy and delta > 0.0) or (block_sell and delta < 0.0):
            out[symbol] = float(carried[symbol])
    return out


# ---------------------------------------------------------------------------
# 8.7 LotRounding
# ---------------------------------------------------------------------------


def apply_lot_rounding(
    executed: pd.Series,
    carried: pd.Series,
    price: Mapping[str, object] | None,
    lot_sizes: Mapping[str, object] | None,
    *,
    min_notional: float | None,
    mode: str,
    day: str,
) -> pd.Series:
    """Round every traded name onto its lot grid using the execution-day price.

    The raw close of the *execution* day is used (never an adjusted price and
    never the decision day's), because the lot grid is a property of the
    order actually placed.  ``down`` -- the default -- keeps
    ``rounded <= target`` per name, so the cash residual stays non-negative
    and the operator can never overspend; ``nearest`` may overshoot by up to
    half a lot.  ``min_notional`` is judged on the *rounded* amount: a grid
    amount below it is not placed at all and the name falls to zero weight
    (the order, and its residual, are cancelled).

    Every name that trades that day needs both a price and a lot size; a
    missing or non-positive input fails closed rather than inventing an
    execution assumption.  Names whose executed weight equals the carried
    weight are untouched and need no data at all.
    """

    if mode not in {"down", "nearest"}:
        raise ValueError("lot_rounding_mode must be one of ('down', 'nearest') when provided")
    needs_data = any(float(executed[symbol]) != float(carried[symbol]) for symbol in executed.index)
    if not needs_data:
        return executed
    if price is None:
        raise ValueError("lot rounding requires the execution_data price panel")
    if lot_sizes is None:
        raise ValueError("lot rounding requires execution_data.lot_sizes")
    price_row = _row(price, day)
    out = executed.copy()
    for symbol in out.index:
        if float(out[symbol]) == float(carried[symbol]):
            continue
        close = None if price_row is None else _finite(price_row.get(symbol))
        if close is None:
            raise ValueError(f"lot rounding is missing a price for {symbol} on {day}")
        if close <= 0.0:
            raise ValueError(f"lot rounding requires a positive price for {symbol} on {day}")
        raw_lot = lot_sizes.get(symbol)
        lot = _finite(raw_lot)
        if lot is None or lot < 1.0 or lot != float(int(lot)):
            raise ValueError(f"lot rounding is missing a lot size for {symbol}")
        lot = float(int(lot))
        target_amount = float(out[symbol]) * CAPITAL
        lots = target_amount / (close * lot)
        if mode == "down":
            count = math.floor(lots + LOT_SNAP)
            amount = count * lot * close
            while count > 0 and amount > target_amount:
                count -= 1
                amount = count * lot * close
        else:
            count = math.floor(lots + 0.5)
            amount = count * lot * close
        if min_notional is not None and amount < float(min_notional):
            amount = 0.0
        out[symbol] = amount / CAPITAL
    return out


# ---------------------------------------------------------------------------
# pending orders: shared state for 8.6 and 8.9
# ---------------------------------------------------------------------------


def _empty_pending() -> dict[str, list[float]]:
    return {}


def _order_names(executed: pd.Series, carried: pd.Series, priority: str, score_row: pd.Series | None) -> list[str]:
    """Deterministic service order for the day's orders.

    ``weight`` orders by the size of the intended move (largest first), so the
    biggest adjustments are attempted before the budget runs out; ``score``
    serves the best-scoring names first and therefore needs the score panel.
    Name is the final tie-break so the plan is reproducible.
    """

    names = list(executed.index)
    if priority == "score":
        if score_row is None:
            raise ValueError("staged_execution with score priority requires the score panel")

        def key(symbol: str) -> tuple[float, str]:
            raw = score_row.get(symbol) if hasattr(score_row, "get") else None
            value = _finite(raw)
            return (-value if value is not None else math.inf, symbol)

        return sorted(names, key=key)
    if priority != "weight":
        raise ValueError(f"unknown staged_priority: {priority}")
    return sorted(names, key=lambda symbol: (-abs(float(executed[symbol]) - float(carried[symbol])), symbol))


def _apply_pending(carried: pd.Series, slices: Mapping[str, float]) -> pd.Series:
    """Move ``carried`` by the day's slice of each remaining order."""

    out = carried.copy()
    for symbol, amount in slices.items():
        if amount:
            out[symbol] = float(out[symbol]) + float(amount)
    return out


def _reap(pending: Mapping[str, list[float]], slices: Mapping[str, float]) -> dict[str, list[float]]:
    """Reduce every order by the slice just attempted, dropping zeroes.

    Slices are removed from the residual whatever the mask or the lock then
    does with them: an order blocked on the day it would have traded is
    cancelled, not silently re-queued (roadmap 8.4 explicitly declines to let
    a lock turn into a queued sell, and the same reasoning holds for a halt or
    a limit state).
    """

    remaining = {symbol: list(entry) for symbol, entry in pending.items()}
    for symbol in list(remaining):
        entry = remaining[symbol]
        entry[0] = float(entry[0]) - float(slices.get(symbol, 0.0))
        if abs(entry[0]) <= 1e-15:
            del remaining[symbol]
    return remaining


def _mean_amount(
    amount: Mapping[str, object] | None,
    window_keys: Sequence[str],
    symbol: str,
    day: str,
    rate: float,
    window: int,
) -> float:
    """Trailing mean traded amount for one name, over the injected window.

    ``window_keys`` is the ordered run of trading dates the panel can answer
    for; ``window`` keeps only its tail, so a window of ``1`` reads the
    execution day alone and a longer one averages the same name backwards over
    the requested number of days.  Days the panel does not carry are skipped
    rather than poisoning the mean, but a name with no answer inside the window
    at all fails closed -- a missing traded amount means the cap cannot be
    evaluated, and inventing one would silently over-trade.
    """

    if amount is None:
        raise ValueError("participation_cap requires the execution_data amount panel")
    keys = list(window_keys)
    span = int(window)
    if span < 1:
        raise ValueError("participation_amount_window must be at least 1 when provided")
    keys = keys[-span:] if span else keys
    values = []
    for key in keys:
        row = _row(amount, key)
        value = None if row is None else _finite(row.get(symbol))
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError(f"participation cap is missing an amount for {symbol} on {day}")
    return sum(values) / len(values) * float(rate)


def apply_participation_cap(
    carried: pd.Series,
    pending: dict[str, list[float]],
    target: pd.Series | None,
    amount: Mapping[str, object] | None,
    *,
    position: int,
    day: str,
    window_keys: Sequence[str],
    rate: float,
    window: int,
    unfilled: str,
    max_days: int | None,
    score_row: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """ParticipationCap (8.6): split a fresh target into orderable slices.

    Every name whose target differs from the carried weight can trade at most
    ``rate * mean(traded amount)`` of the book today; the rest becomes a
    pending order.  ``unfilled='carry'`` keeps that residual alive for at most
    ``max_days`` completed trading days (``'cancel'`` drops it immediately), so
    an unfilled slice is never quietly filled later -- it is either executed on
    a subsequent day or cancelled on expiry.  The daily slice is always a
    fraction of the intended move, so the executed row stays between the
    carried and target rows (and therefore inside the budget).

    A ``target`` row means the day carries a fresh decision: it replaces the
    outstanding plan outright, and the residual this call registers is
    ``target - carried - today`` in full, so the returned state is the new plan
    and never a blend of the new decision with the old one.  The part already
    executed on an earlier day is not re-created, because ``carried`` has
    already moved by it.  ``target is None`` is a non-decision trading day: the
    carried orders advance on their own and nothing new is created.
    """

    if rate <= 0.0 or rate > 1.0:
        raise ValueError("participation_max_rate must be in (0, 1] when provided")
    if unfilled not in {"cancel", "carry"}:
        raise ValueError("participation_unfilled must be one of ('cancel', 'carry') when provided")
    if unfilled == "carry" and (max_days is None or int(max_days) < 1):
        raise ValueError("participation_max_days must be at least 1 when participation_unfilled is carry")
    if target is None:
        return _advance_participation(
            carried,
            pending,
            position=position,
            day=day,
            window_keys=window_keys,
            amount=amount,
            rate=rate,
            window=window,
            max_days=max_days,
        )
    pending.clear()
    names = _order_names(target, carried, "weight", score_row)
    slices: dict[str, float] = {}
    for symbol in names:
        delta = float(target[symbol]) - float(carried[symbol])
        if delta == 0.0:
            continue
        capacity = _mean_amount(amount, window_keys, symbol, day, rate, window)
        allowed = math.copysign(min(abs(delta), capacity), delta)
        slices[symbol] = allowed
        remainder = delta - allowed
        if remainder != 0.0 and unfilled == "carry":
            pending[symbol] = [remainder, float(position)]
    return _apply_pending(carried, slices), pending


def _advance_participation(
    carried: pd.Series,
    pending: Mapping[str, list[float]],
    *,
    position: int,
    day: str,
    window_keys: Sequence[str],
    amount: Mapping[str, object] | None,
    rate: float,
    window: int,
    max_days: int | None,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """Advance the carried 8.6 pending orders on a non-decision trading day.

    Orders that have reached their maximum age are cancelled before they
    trade, so ``max_days`` bounds how long a residual can survive; the rest
    trade up to today's participation capacity.  Whatever the day does not
    fill stays pending for the next day.

    A residual is cancelled on expiry without trading, and the returned state
    carries the *post-trade* residual only.  Every order this call slices is
    therefore reaped here, in one place, and never twice.
    """

    remaining = {symbol: list(entry) for symbol, entry in pending.items()}
    slices: dict[str, float] = {}
    for symbol in list(remaining):
        residual, created = remaining[symbol]
        if residual == 0.0 or (max_days is not None and position - int(created) >= int(max_days)):
            del remaining[symbol]
            continue
        capacity = _mean_amount(amount, window_keys, symbol, day, rate, window)
        slices[symbol] = math.copysign(min(abs(residual), capacity), residual)
    return _apply_pending(carried, slices), _reap(remaining, slices)


def stage_orders(
    carried: pd.Series,
    pending: dict[str, list[float]],
    target: pd.Series | None,
    *,
    position: int,
    max_days: int,
    order: Sequence[str],
) -> tuple[pd.Series, dict[str, list[float]]]:
    """StagedExecution (8.9): a fresh target becomes a multi-day order plan.

    A ``target`` row rebuilds the plan outright from ``target - carried`` --
    the new decision's own requirement, not a silently dropped order -- and the
    day's first slice is taken immediately.  ``target is None`` is a
    non-decision trading day: the carried plan advances by one day.

    ``order`` -- the deterministic service order, from ``staged_priority`` --
    decides who is served first when the day's budget runs out; a plan
    cancelled for having spent ``max_days`` is dropped rather than leaking into
    the next one.
    """

    if max_days < 1:
        raise ValueError("staged_max_days must be at least 1 when provided")
    if target is None:
        return _advance_staged(carried, pending, position=position, max_days=max_days, order=order)
    pending.clear()
    for symbol in target.index:
        residual = float(target[symbol]) - float(carried[symbol])
        if residual != 0.0:
            pending[symbol] = [residual, float(position)]
    return _advance_staged(carried, pending, position=position, max_days=max_days, order=order)


def _advance_staged(
    carried: pd.Series,
    pending: Mapping[str, list[float]],
    *,
    position: int,
    max_days: int,
    order: Sequence[str],
) -> tuple[pd.Series, dict[str, list[float]]]:
    """Advance the 8.9 plan by one trading day, in priority order.

    The day's budget is the outstanding requirement spread evenly over what is
    left of ``max_days`` after the oldest live order's age, and it is spent in
    ``order``: each order is filled completely before the next one is served, so
    the priority really decides who trades first and the tail of the plan is the
    part that gets spread out.  The budget itself never changes with the order.
    Orders still alive once ``max_days`` is spent are cancelled -- a plan that
    ran out of days does not leak into the next one -- and a plan whose days are
    already spent trades nothing at all.
    """

    remaining = {symbol: list(entry) for symbol, entry in pending.items()}
    total = sum(abs(entry[0]) for entry in remaining.values())
    if total <= 0.0:
        return carried, {}
    oldest = max((position - int(entry[1]) for entry in remaining.values()), default=0)
    remaining_days = int(max_days) - int(oldest)
    if remaining_days < 1:
        return carried, {}
    budget = total / remaining_days
    slices: dict[str, float] = {}
    used = 0.0
    for symbol in order:
        residual = float(remaining[symbol][0]) if symbol in remaining else 0.0
        if residual == 0.0:
            continue
        room = budget - used
        if room <= 1e-15:
            break
        take = math.copysign(min(abs(residual), room), residual)
        slices[symbol] = take
        used += abs(take)
    return _apply_pending(carried, slices), _reap(remaining, slices)


# ---------------------------------------------------------------------------
# 9.1 PortfolioVolatilityTarget
# ---------------------------------------------------------------------------


def volatility_scale(realised: Sequence[float], *, target_annual: float, window: int, min_scale: float, max_scale: float, periods_per_year: int) -> float:
    """Scale the whole target row so realised volatility tracks a target.

    The estimate reads only returns already realised before the decision day.
    Two declared fallbacks keep the very first rebalance from being an error:
    fewer than two observations returns ``1.0`` (nothing to estimate, so the
    operator does nothing -- the same declared-degradation convention 7.5 uses
    for an empty panel), and a strictly non-positive sample standard deviation
    returns ``max_scale`` (no measured risk is no reason to de-risk).  The
    result is clipped to ``[min_scale, max_scale]`` so the scaler can neither
    lever the book up without a declared bound nor silently move it to cash.

    The 7.5 contrast is deliberate: there a *single name* failing to produce a
    volatility estimate is a data defect and fails closed; here the *whole
    book* having a flat history is a sample property, so it degrades instead.
    """

    values = [float(value) for value in realised if value is not None and math.isfinite(float(value))]
    if len(values) < 2:
        return 1.0
    sample = values[-int(window) :] if window else values
    if len(sample) < 2:
        return 1.0
    mean = sum(sample) / len(sample)
    variance = sum((value - mean) ** 2 for value in sample) / (len(sample) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std <= 0.0:
        return float(max_scale)
    realised_annual = std * math.sqrt(float(periods_per_year))
    if realised_annual <= 0.0:
        return float(max_scale)
    scale = float(target_annual) / realised_annual
    return float(min(max(scale, float(min_scale)), float(max_scale)))


# ---------------------------------------------------------------------------
# 9.2 DrawdownStateMachine
# ---------------------------------------------------------------------------


def drawdown_state_machine(realised_navs: Sequence[float], *, machine: dict[str, object], thresholds: Mapping[str, float], scales: Mapping[str, float], confirm_days: int, recovery: float) -> float:
    """One frozen transition of the drawdown state machine (roadmap 9.2).

    ``realised_navs`` is the book's net-of-cost NAV path up to (excluding) the
    decision day, built inside the loop from the same cost formula the engine
    reports.  The high-water mark and the deepest point are updated first, then
    the observed severity is mapped to one of ``normal``/``warning``/``reduce``/
    ``halt`` and only adopted after ``confirm_days`` consecutive observations
    -- a single bad day cannot halt the book.  De-escalation has its own gate:
    the drawdown must have narrowed by ``drawdown_recovery`` from its deepest
    point for ``confirm_days`` consecutive days, and even then the state steps
    down one level at a time.  Nothing here reads a future value.

    Returns the scale to apply to today's target row (``normal`` is 1.0).
    """

    state = str(machine.get("state", "normal"))
    if not realised_navs:
        machine["state"] = state
        return float(scales.get(state, 1.0))
    nav = float(realised_navs[-1])
    high_water = max(float(machine.get("high_water", nav)), nav)
    deepest = min(float(machine.get("deepest", nav)), nav)
    machine["high_water"] = high_water
    severity = (high_water - nav) / high_water if high_water > 0 else 0.0
    ordered = [("halt", float(thresholds["halt"])), ("reduce", float(thresholds["reduce"])), ("warning", float(thresholds["warning"]))]
    observed = "normal"
    for name, level in ordered:
        if severity >= level:
            observed = name
            break
    rank = {"normal": 0, "warning": 1, "reduce": 2, "halt": 3}
    if rank[observed] > rank[state]:
        pending_state = str(machine.get("pending_state", state))
        pending_days = int(machine.get("pending_days", 0))
        pending_days = pending_days + 1 if pending_state == observed else 1
        machine["pending_state"] = observed
        machine["pending_days"] = pending_days
        machine["recovery_days"] = 0
        if pending_days >= int(confirm_days):
            state = observed
            machine["pending_days"] = 0
    elif rank[observed] < rank[state]:
        machine["pending_state"] = observed
        machine["pending_days"] = 0
        narrowed = (nav - deepest) / high_water if high_water > 0 else 0.0
        if narrowed >= float(recovery):
            days = int(machine.get("recovery_days", 0)) + 1
            machine["recovery_days"] = days
            if days >= int(confirm_days):
                state = next(name for name, level in rank.items() if level == rank[state] - 1)
                machine["recovery_days"] = 0
        else:
            machine["recovery_days"] = 0
    else:
        machine["pending_state"] = observed
        machine["pending_days"] = 0
        machine["recovery_days"] = 0
    if state == "normal":
        # ``deepest`` is the trough of the *current* drawdown segment -- the
        # yardstick the recovery gate reads.  A new high-water mark moves
        # ``high_water`` but leaves ``deepest`` behind, so keeping an old
        # segment's trough here would hold the gate at zero for as long as the
        # book stays at its high and a fully recovered book could never step
        # back down.  The segment is closed here, at the return to normal.
        deepest = nav
    machine["state"] = state
    machine["deepest"] = deepest
    return float(scales.get(state, 1.0))


# ---------------------------------------------------------------------------
# order-layer entry point (8.6 / 8.7 / 8.8 / 8.9)
# ---------------------------------------------------------------------------


def plan_order(
    executed: pd.Series,
    carried: pd.Series,
    *,
    priority: str | None,
    score_row: pd.Series | None,
) -> list[str]:
    """Service order of one plan, with ``staged_priority``'s default resolved.

    ``staged_priority`` is a spec-level knob while :func:`_order_names` reads the
    resolved value, so the engines call this thin wrapper instead of each
    repeating the default.
    """

    return _order_names(executed, carried, "weight" if priority is None else str(priority), score_row)


def apply_execution_layer(
    carried: pd.Series,
    pending: dict[str, list[float]],
    target: pd.Series | None,
    *,
    position: int,
    day: str,
    calendar: Sequence[str],
    order: Sequence[str],
    participation_rate: float | None,
    participation_window: int | None,
    participation_unfilled: str | None,
    participation_max_days: int | None,
    staged_max_days: int | None,
    lot_rounding: bool,
    lot_min_notional: float | None,
    lot_rounding_mode: str | None,
    side_aware_tradability: str | None,
    amount: Mapping[str, object] | None,
    price: Mapping[str, object] | None,
    lot_sizes: Mapping[str, object] | None,
    halt: Mapping[str, object] | None,
    limits: Mapping[str, object] | None,
    trace: dict[str, object] | None = None,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """One trading day of the order layer, in the order the roadmap fixes.

    The allocator runs first -- 8.6 and 8.9 are mutually exclusive, so at most
    one of them is live -- and produces the day's executed slices plus the
    carried plan.  The side-aware mask then clamps those slices and the lot grid
    rounds them.  Masking and rounding *after* the allocator is deliberate: a
    slice is the order actually placed today, so the slice is what has to be
    tradable and whole-lot.  Masking the target first would let the allocator
    re-inflate a direction the day cannot trade, and rounding first would let the
    allocator split a whole-lot order back onto a fractional grid.

    Consequence of that order: the part of a slice the mask blocks or the lot
    grid shaves off is *cancelled*, not re-queued -- the same handoff 8.4 uses
    for a locked sell, and the reason a rounded-down order is never re-offered
    on a later day.

    ``trace`` is an optional observation sink: when given, each stage records
    the book as it stood after that stage under its own key (``allocator``,
    ``tradability``, ``lot_rounding``) plus the name of the allocator that ran.
    Nothing here reads it back, so the executed row and the carried plan are
    bit-identical whether or not a caller asks for a trace.
    """

    executed = carried if target is None else target
    if participation_rate is not None:
        executed, pending = apply_participation_cap(
            carried,
            pending,
            target,
            amount,
            position=position,
            day=day,
            window_keys=calendar,
            rate=float(participation_rate),
            window=1 if participation_window is None else int(participation_window),
            unfilled="carry" if participation_unfilled is None else str(participation_unfilled),
            max_days=participation_max_days,
        )
        if trace is not None:
            trace["__allocator_reason__"] = "participation"
    elif staged_max_days is not None:
        executed, pending = stage_orders(
            carried, pending, target, position=position, max_days=int(staged_max_days), order=order
        )
        if trace is not None:
            trace["__allocator_reason__"] = "staging"
    if trace is not None:
        trace["allocator"] = executed.copy()
    if side_aware_tradability is not None:
        executed = apply_side_aware_tradability(
            carried, executed, halt, limits, price, mode=str(side_aware_tradability), day=day
        )
    if trace is not None:
        trace["tradability"] = executed.copy()
    if lot_rounding:
        executed = apply_lot_rounding(
            executed,
            carried,
            price,
            lot_sizes,
            min_notional=lot_min_notional,
            mode="down" if lot_rounding_mode is None else str(lot_rounding_mode),
            day=day,
        )
    if trace is not None:
        trace["lot_rounding"] = executed.copy()
    return executed, pending


# ---------------------------------------------------------------------------
# 9.5 FactorDecayMonitor
# ---------------------------------------------------------------------------


def factor_decay_state(
    values: Sequence[float | None],
    *,
    window: int,
    fail_periods: int,
    threshold: float,
    scale: float,
) -> float:
    """One frozen state evaluation of the factor-decay monitor (roadmap 9.5).

    ``values`` is the factor's realised rank-IC history in decision-date order
    (``None`` marks a period that is not observable yet).  The monitor reads
    the trailing ``window`` observations and counts the run of *consecutive*
    failures ending at the most recent one: the factor is in its decay state
    when that run reaches ``fail_periods``, and the state then shrinks the
    factor's contribution to ``scale`` instead of replacing the factor -- a
    failure period is not a reason to hand the book to a factor chosen by the
    failing signal itself.

    Two boundary conventions are pinned here.  A period counts as a failure
    only when its IC is *strictly* below ``threshold``, so an IC exactly on the
    threshold is not a failure (the conservative reading: never pause on a
    borderline observation).  And a history shorter than ``fail_periods``
    cannot fail: the monitor is silent until it has enough observations, the
    same declared degradation ``volatility_scale`` uses for an empty sample.
    """

    sample = [float(value) for value in values if value is not None and math.isfinite(float(value))][-int(window) :]
    failing = 0
    for value in reversed(sample):
        if value < float(threshold):
            failing += 1
        else:
            break
    return float(scale) if failing >= int(fail_periods) else 1.0


def factor_ic_series(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    positions: Sequence[int],
    lag: int,
) -> list[float | None]:
    """Rank IC of one factor over each decision's realised holding window.

    ``panel`` is the factor's final percentile-rank panel (rows = decision
    dates, columns = symbols), ``returns`` the engine's daily simple-return
    panel and ``positions`` the row index of each decision date inside
    ``returns``.  The window a decision earns is the run of trading days that
    starts when its target first becomes tradable (``signal_lag`` days later)
    and stops the day before the next decision's target takes over -- exactly
    the slice of the return panel the position held.  The IC is the
    cross-sectional Spearman correlation between that date's ranks and the
    compounded window return, so the estimate only ever reads returns that the
    window itself realised.

    The last decision has no complete window, and a window covering fewer than
    two names is not a cross-section; both report ``None`` ("not observable
    yet"), which the state evaluation ignores rather than counting as a
    failure.
    """

    out: list[float | None] = []
    for index in range(len(positions) - 1):
        start = int(positions[index]) + int(lag)
        stop = int(positions[index + 1]) + int(lag)
        if start >= len(returns.index) or stop > len(returns.index) or stop <= start:
            out.append(None)
            continue
        forward = (1.0 + returns.iloc[start:stop].fillna(0.0)).prod() - 1.0
        joined = pd.DataFrame({"rank": panel.iloc[index], "forward": forward}).dropna()
        if len(joined) < 2:
            out.append(None)
            continue
        value = joined["rank"].corr(joined["forward"].rank())
        out.append(None if pd.isna(value) else float(value))
    out.append(None)
    return out


def factor_decay_scales(
    ranks: Mapping[str, pd.DataFrame],
    returns: pd.DataFrame,
    *,
    lag: int,
    window: int,
    fail_periods: int,
    threshold: float,
    scale: float,
) -> dict[str, pd.Series]:
    """Per-factor decay scale for every decision date (roadmap 9.5).

    A value is only used once the decision date it will steer is *after* the
    window that produced it: an IC whose holding window ends on or after the
    decision day is not yet realised at that day's close, so it is dropped
    rather than read one day early.
    """

    if not ranks:
        return {}
    index = next(iter(ranks.values())).index
    positions = [int(value) for value in returns.index.get_indexer(index)]
    if any(value < 0 for value in positions):
        raise ValueError("factor decay monitor requires the decision dates to be present in the returns panel")
    out: dict[str, pd.Series] = {}
    for name, panel in ranks.items():
        values = factor_ic_series(panel, returns, positions=positions, lag=lag)
        scales = []
        for position in range(len(positions)):
            realised = [
                value
                for offset, value in enumerate(values[:position])
                if value is not None and int(positions[offset + 1]) + int(lag) <= int(positions[position])
            ]
            scales.append(
                factor_decay_state(
                    realised, window=window, fail_periods=fail_periods, threshold=threshold, scale=scale
                )
            )
        out[name] = pd.Series(scales, index=index, dtype=float)
    return out
