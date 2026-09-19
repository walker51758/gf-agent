"""Local portfolio backtest engine with explicit lag, drift and cost.

The engine is intentionally small and fully deterministic: it exists so the
strategy layer can be unit tested locally (and later ported verbatim to a
single-file remote worker) without hiding any execution assumption.  Three
assumptions are enforced here rather than trusted to the caller:

* a decision date's weights are only tradable ``signal_lag`` trading days
  later, so a same-day signal can never earn its own day's return;
* holdings drift with realised returns between rebalances, which is what makes
  reported turnover comparable to a real rebalance schedule;
* costs are charged separately on actual asset buys and sells at the immutable
  system rates; candidates and callers cannot select a cheaper assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, ExecutionDataSpec
from rdagent.scenarios.gf_quant.strategy.composition import build_buffered_target_row
from rdagent.scenarios.gf_quant.strategy.costs import (
    BUY_COST_RATE,
    SELL_COST_RATE,
    execution_costs_metrics,
)
from rdagent.scenarios.gf_quant.strategy.execution import (
    apply_execution_layer,
    day_key,
    drawdown_state_machine,
    plan_order,
    volatility_scale,
)


@dataclass(frozen=True)
class PortfolioResult:
    """Backtest output: headline metrics, yearly breakdown and the traces."""

    metrics: Mapping[str, Any]
    metrics_by_year: Mapping[str, Mapping[str, Any]]
    daily_returns: pd.Series
    daily_gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    buy_turnover: pd.Series
    sell_turnover: pd.Series
    transaction_costs: pd.Series


def _max_drawdown(returns: pd.Series) -> float:
    # Include the initial NAV.  Without it, a loss on the first invested day
    # becomes the starting high-water mark and is incorrectly reported as zero
    # drawdown.
    wealth = pd.Series([1.0, *((1.0 + returns).cumprod().to_list())], dtype=float)
    return float((wealth / wealth.cummax() - 1.0).min())


def _performance(
    daily: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
    spec: BacktestSpec,
) -> dict[str, Any]:
    """Headline metrics for one return series.

    ``annualized_return`` follows the factor worker's convention (mean daily
    return x trading days per year) so factor and strategy numbers stay
    comparable inside one registry; ``cagr`` is the geometric alternative.
    """

    per_year = spec.trading_days_per_year
    mean = float(daily.mean()) if len(daily) else 0.0
    std = float(daily.std()) if len(daily) > 1 else 0.0
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
    days = len(daily)
    terminal_wealth = 1.0 + total
    cagr = float(terminal_wealth ** (per_year / days) - 1.0) if days and terminal_wealth > 0 else -1.0
    mean_turnover = float(turnover.mean()) if len(turnover) else 0.0
    holdings = (weights != 0.0).sum(axis=1)
    return {
        "observations": int(days),
        "total_return": total,
        "annualized_return": mean * per_year,
        "cagr": cagr,
        "annualized_vol": std * per_year**0.5,
        "sharpe": (mean / std * per_year**0.5) if std > 0 else 0.0,
        "max_drawdown": _max_drawdown(daily) if days else 0.0,
        "turnover": mean_turnover,
        "annual_turnover": mean_turnover * per_year,
        "avg_holdings": float(holdings.mean()) if len(holdings) else 0.0,
        "win_rate": float((daily > 0).mean()) if days else 0.0,
    }


def _relative_performance(daily: pd.Series, benchmark: pd.Series, spec: BacktestSpec) -> dict[str, Any]:
    """Excess-return metrics against an explicitly provided benchmark."""

    excess = daily - benchmark
    mean = float(excess.mean()) if len(excess) else 0.0
    std = float(excess.std()) if len(excess) > 1 else 0.0
    per_year = spec.trading_days_per_year
    return {
        "excess_annualized_return": mean * per_year,
        "tracking_error": std * per_year**0.5,
        "information_ratio": (mean / std * per_year**0.5) if std > 0 else 0.0,
    }


def _cost_metrics(
    buy_turnover: pd.Series,
    sell_turnover: pd.Series,
    transaction_costs: pd.Series,
    spec: BacktestSpec,
) -> dict[str, Any]:
    """Project immutable rates plus realised side-specific trading activity."""

    per_year = spec.trading_days_per_year
    mean_buy = float(buy_turnover.mean()) if len(buy_turnover) else 0.0
    mean_sell = float(sell_turnover.mean()) if len(sell_turnover) else 0.0
    mean_cost = float(transaction_costs.mean()) if len(transaction_costs) else 0.0
    return {
        **execution_costs_metrics(),
        "buy_turnover": mean_buy,
        "sell_turnover": mean_sell,
        "annual_buy_turnover": mean_buy * per_year,
        "annual_sell_turnover": mean_sell * per_year,
        "average_transaction_cost": mean_cost,
        "total_transaction_cost": float(transaction_costs.sum()),
    }


def _drift_band_triggers(
    carried: pd.Series,
    target: pd.Series,
    band_name: float | None,
    band_portfolio: float | None,
) -> bool:
    """DriftBandRebalance (roadmap 8.3): does the drifted book need this rebalance?

    The comparison deliberately reads the *drifted* carried weights against
    the fresh target: tolerating drift is the operator's business purpose, so
    this is the one explicit exception to the judge-on-target-weights rule --
    the composition cap and floor keep judging target weights only.  The
    per-name band trips when any single name's absolute weight change exceeds
    it; the portfolio band trips when the half-L1 change *including the cash
    leg* exceeds it, i.e. the same measurement convention the engine reports
    as turnover, so a band can be read straight off the turnover metric.  The
    comparison is strict: a deviation exactly on the band stays inside the
    tolerance.  With both bands ``None`` the rebalance always happens.
    """

    delta = target - carried
    if band_name is not None and bool((delta.abs() > float(band_name)).any()):
        return True
    if band_portfolio is not None:
        cash_delta = float(carried.sum()) - float(target.sum())
        step = 0.5 * (float(delta.abs().sum()) + abs(cash_delta))
        if step > float(band_portfolio):
            return True
    return False


def _shrink_turnover(
    carried: pd.Series,
    target: pd.Series,
    shrinkage: float | None,
    max_step: float | None,
) -> pd.Series:
    """TurnoverShrinkage (roadmap 8.5): move the book only part of the way.

    The soft shrinkage scales the whole weight delta -- a convex combination
    of the carried and target vectors -- so it stays non-negative, within
    budget and treats every leg alike: a name the target exits at zero shrinks
    exactly like any other leg and is never special-cased.  The hard clip then
    caps the executed step at ``max_step`` of half-L1 turnover *including
    cash*, the same convention the engine reports; scaling the delta keeps the
    result between the carried and target vectors, so no constraint of the
    target row can be violated by the shrink itself.  The cost formulas are
    untouched: they simply observe a smaller executed delta.
    """

    executed = target
    if shrinkage is not None:
        executed = carried + (target - carried) * (1.0 - float(shrinkage))
    if max_step is not None:
        delta = executed - carried
        step = 0.5 * (float(delta.abs().sum()) + abs(float(carried.sum()) - float(executed.sum())))
        if step > float(max_step):
            executed = carried + delta * (float(max_step) / step)
    return executed


def _apply_min_holding(
    carried: pd.Series,
    executed: pd.Series,
    entry_index: Mapping[str, int],
    position: int,
    min_holding_days: int,
) -> pd.Series | None:
    """MinimumHoldingPeriod (roadmap 8.4): a recent buy cannot be sold yet.

    A name still inside its lock window (held, with fewer than
    ``min_holding_days`` completed trading days since the trade that entered
    it) keeps its carried weight: its executed weight is raised back up to
    the carried weight if the fresh target tried to sell it.  The budget the
    locked names take back is funded by scaling the remaining names down
    proportionally, so the book never exceeds the executed total and no
    unlocked name is pushed above its planned weight.  The clamp runs *after*
    the drift-band gate and the turnover shrinkage: the band still decides
    whether to trade at all, and the lock is the last veto that makes
    "no sell of a locked name" hold against every other execution-layer
    operator.  Costs are untouched -- they simply observe the smaller
    executed sell delta.

    Returns ``None`` when the locked book cannot be funded: raising every
    locked name back to its carried weight needs more budget than the
    unlocked names' executed weights can give up (including the case of no
    unlocked weight at all).  Partially raising a locked name would mean
    partially selling it, which violates the lock itself, so the *whole*
    rebalance is abandoned and the caller keeps the carried weights -- the
    same skip semantics as a rebalance inside the drift band (including the
    days-since-rebalance counter).  This keeps the executed row sum at or
    below one in every branch.
    """

    locked = [
        name
        for name in executed.index
        if carried[name] > 0.0 and name in entry_index and position - entry_index[name] < min_holding_days
    ]
    if not locked:
        return executed
    raised = [name for name in locked if executed[name] < carried[name]]
    if not raised:
        return executed
    needed = float(sum(carried[name] - executed[name] for name in raised))
    unlocked_mask = ~executed.index.isin(locked)
    available = float(executed[unlocked_mask].sum())
    if needed > available:
        # The unlocked book cannot fund the raise-back: abandon the trade.
        return None
    adjusted = executed.copy()
    for name in raised:
        adjusted[name] = float(carried[name])
    locked_sum = float(adjusted[locked].sum())
    total = float(executed.sum())
    unlocked_sum = available
    if unlocked_sum > 0.0:
        factor = (total - locked_sum) / unlocked_sum
        if factor < 0.0:
            # Guard against float rounding at the exact-feasibility boundary.
            factor = 0.0
        adjusted[unlocked_mask] = executed[unlocked_mask] * factor
    return adjusted


def _execution_day(
    carried: pd.Series,
    pending: dict[str, list[float]],
    target: pd.Series | None,
    *,
    spec: BacktestSpec,
    data: ExecutionDataSpec,
    order: Sequence[str],
    day: str,
    calendar: Sequence[str],
    calendar_index: Mapping[str, int],
    trace: dict[str, object] | None = None,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """One trading day of the order layer, with its calendar resolved.

    ``day`` is the *trade* day -- the day the slices are placed, which is
    ``signal_lag`` days after the decision that produced ``target`` -- so the
    price, halt and limit state read here are the ones that actually apply to
    the trade.  The order layer's own calendar supplies the position a pending
    plan is aged against and the window the participation cap averages the
    traded amount over; the window is truncated at ``day`` so the estimate can
    only read already-realised amounts.  A trade day the injected calendar does
    not carry fails closed: aging a plan against a calendar that skipped the
    day would silently stretch or shorten ``max_days``.

    ``target is None`` on a day with no fresh decision: the outstanding plan
    advances on its own and nothing new is created.
    """

    position = -1
    window_keys: list[str] = [day]
    if calendar:
        if day not in calendar_index:
            raise ValueError(f"execution_data trading_dates do not contain trade date {day}")
        position = calendar_index[day]
        window_keys = list(calendar[: position + 1])

    return apply_execution_layer(
        carried,
        pending,
        target,
        position=position,
        day=day,
        calendar=window_keys,
        order=order,
        participation_rate=spec.participation_max_rate,
        participation_window=spec.participation_amount_window,
        participation_unfilled=spec.participation_unfilled,
        participation_max_days=spec.participation_max_days,
        staged_max_days=spec.staged_max_days,
        lot_rounding=bool(spec.lot_rounding),
        lot_min_notional=spec.lot_min_notional,
        lot_rounding_mode=spec.lot_rounding_mode,
        side_aware_tradability=spec.side_aware_tradability,
        amount=data.amount,
        price=data.price,
        lot_sizes=data.lot_sizes,
        halt=data.halt,
        limits=data.price_limits,
        trace=trace,
    )


def run_portfolio_backtest(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    spec: BacktestSpec,
    benchmark_returns: pd.Series | None = None,
    composition: Any | None = None,
    score: pd.DataFrame | None = None,
    execution_data: ExecutionDataSpec | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
) -> PortfolioResult:
    """Simulate a rebalanced portfolio with lag, drift and fixed side costs.

    Parameters
    ----------
    target_weights:
        Weights decided at each rebalance date (rows) per symbol (columns).
        Every target date must be present in ``returns`` and every symbol must
        exist there -- both are validated instead of silently dropped.
    returns:
        Daily simple returns per symbol, ``index=dt`` / ``columns=symbol``.
        A missing return is treated as a flat day for that holding.
    spec:
        Candidate-independent execution assumptions (lag and annualisation).
    benchmark_returns:
        Optional daily benchmark returns used for excess-return metrics.
    composition:
        Optional composition spec enabling the selection-layer operators
        (HoldingBuffer 5.5 / DiversitySelector 5.6).  When given together
        with ``score``, the target row for each decision date is rebuilt
        inside the loop from that date's score, the names still held after
        the previous executed trade and the realised-return history up to
        the decision date.  Both parameters must be given together.
    score:
        Score panel matching ``composition``; required whenever
        ``composition`` is given.  An infinite score is rejected here too,
        because the rebuilt rows are read straight off this panel.
    execution_data:
        Explicitly injected order-layer data (prices, traded amounts, lot
        sizes, halt and limit states) plus the order layer's own trading
        calendar.  The weight engine has no market data of its own, so the
        order layer (8.6-8.9) reads it here; without it an enabled order-layer
        operator fails closed rather than inventing a price or a lot size.
    execution_trace:
        Optional list the engine appends one entry to per rebalance day that
        actually places an order, carrying the target row, the executed row and
        the book as it stood after each execution stage.  The FeasibilityAudit
        (10.6) reads this to attribute the untouched ``target - executed`` gap to
        the stage that caused it.  It is pure observation: passing it changes no
        weight, no cost and no return, and a run without it walks the exact
        same path.

        Scope is decision days only: a non-decision trading day merely advances
        an already-registered plan, so it has no fresh target to measure the day
        against and is left out rather than reported as a second, targetless
        row.  A run whose order layer is disabled traces nothing at all -- there
        the executed row *is* the target, so every day would read as fully
        executable.

    The execution chain runs on the target row in the order the roadmap fixes:
    the target row is built (or rebuilt by the selection layer), the drawdown
    state machine and the volatility target then scale that row, the drift-band
    gate decides whether to trade at all, turnover shrinkage trims the step,
    the order layer allocates and constrains the day's slices, and the
    minimum-holding lock has the last veto.  The two risk scalings and the four
    order-layer operators are all dormant unless the spec enables them, so a
    candidate that enables none of them walks the pre-batch path bit for bit.
    """

    if target_weights.empty:
        raise ValueError("target_weights is empty")
    if returns.empty:
        raise ValueError("returns panel is empty")
    missing_dates = target_weights.index.difference(returns.index)
    if len(missing_dates):
        raise ValueError(
            f"target weight dates are absent from the returns panel: {[str(date) for date in missing_dates[:3]]}"
        )
    unknown_symbols = target_weights.columns.difference(returns.columns)
    if len(unknown_symbols):
        raise ValueError(f"target weights reference unknown symbols: {list(unknown_symbols)[:3]}")
    if benchmark_returns is not None:
        missing_bench = returns.index.difference(benchmark_returns.index)
        if len(missing_bench):
            raise ValueError(
                f"benchmark returns do not cover the backtest window: {[str(date) for date in missing_bench[:3]]}"
            )

    # Only reindex the *columns*: rows keep their own dates.  Filling the
    # missing rows would turn "no decision on this date" into "decision = go to
    # cash" and silently liquidate a drifting book.
    if target_weights.index.has_duplicates:
        raise ValueError("target_weights has duplicate dates")
    if returns.index.has_duplicates:
        raise ValueError("returns has duplicate dates")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns dates must be sorted in increasing order")

    numeric_targets = target_weights.apply(pd.to_numeric, errors="coerce")
    if numeric_targets.isna().any().any() or not np.isfinite(numeric_targets.to_numpy(dtype=float)).all():
        raise ValueError("target_weights must contain only finite numeric values")
    if (numeric_targets < -1e-12).any().any():
        raise ValueError("target_weights must be non-negative")
    row_sums = numeric_targets.sum(axis=1)
    if (row_sums > 1.0 + 1e-10).any():
        raise ValueError("target_weights row sums must not exceed 1.0")

    aligned = numeric_targets.reindex(columns=returns.columns).fillna(0.0)
    target_by_date = {date: row.to_numpy(dtype=float) for date, row in aligned.iterrows()}
    dates = returns.index
    symbols = list(returns.columns)

    has_selection_operators = composition is not None and score is not None
    if (composition is not None or score is not None) and not has_selection_operators:
        raise ValueError("composition and score must be provided together for the selection-layer operators")
    if score is not None:
        missing_score_dates = target_weights.index.difference(score.index)
        if len(missing_score_dates):
            raise ValueError(
                f"score panel is missing target weight dates: {[str(date) for date in missing_score_dates[:3]]}"
            )
        unknown_score_symbols = score.columns.difference(returns.columns)
        if len(unknown_score_symbols):
            raise ValueError(f"score panel references unknown symbols: {list(unknown_score_symbols)[:3]}")
        # The selection-layer operators rebuild the target row straight from
        # this panel, so the same non-finite guard as
        # ``composition.build_target_weights`` applies here.
        numeric_score = score.apply(pd.to_numeric, errors="coerce")
        if bool(np.isinf(numeric_score.to_numpy(dtype=float)).any()):
            raise ValueError("score panel must not contain infinite values")

    has_min_holding = spec.min_holding_days is not None
    held_names: frozenset[str] = frozenset()
    entry_index: dict[str, int] = {}

    order_data = execution_data if execution_data is not None else ExecutionDataSpec()
    order_calendar = list(execution_data.trading_dates) if execution_data is not None and execution_data.trading_dates else []
    order_calendar_index = {key: index for index, key in enumerate(order_calendar)}
    has_order_layer = (
        spec.participation_max_rate is not None
        or spec.staged_max_days is not None
        or spec.side_aware_tradability is not None
        or spec.lot_rounding
    )
    if (spec.participation_max_rate is not None or spec.staged_max_days is not None) and not order_calendar:
        # A pending plan is aged in the order layer's own calendar; without one
        # it could never expire, so an allocator without a calendar fails here
        # instead of quietly trading a plan forever.
        raise ValueError("participation_cap and staged_execution require execution_data.trading_dates")
    has_risk_scaling = spec.vol_target_annual is not None or spec.drawdown_thresholds is not None

    start_weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    gross = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    buy_turnover = pd.Series(0.0, index=dates, dtype=float)
    sell_turnover = pd.Series(0.0, index=dates, dtype=float)

    carried = pd.Series(0.0, index=symbols, dtype=float)
    has_drift_band = spec.drift_band_name is not None or spec.drift_band_portfolio is not None
    has_shrinkage = spec.turnover_shrinkage is not None or spec.turnover_max_step is not None
    days_since_rebalance = 0
    # Order-layer state.  ``pending`` is the outstanding 8.6 / 8.9 plan and is
    # carried across trading days; ``plan_service_order`` is that plan's fixed
    # service order.  Both live outside the loop because a plan outlives the
    # decision that created it.
    pending: dict[str, list[float]] = {}
    plan_service_order: list[str] = []
    # FeasibilityAudit (10.6) trace sink: ``None`` unless the caller asked for
    # the observation *and* the order layer is live.  A run with no execution
    # operator has nothing to attribute -- the book is the target row -- so the
    # audit reports the day as fully executable from the weights alone.  Being a
    # pure observation the trace is never read back inside this function.
    if execution_trace is not None and not has_order_layer:
        execution_trace = None
    # Portfolio-risk state: the realised NAV path and the daily net returns,
    # both built in-loop from the same formula the engine reports at the end
    # (``net = gross - buy * BUY_COST_RATE - sell * SELL_COST_RATE``).
    drawdown_machine: dict[str, object] = {}
    nav_path: list[float] = [1.0]
    net_path: list[float] = []
    for position, date in enumerate(dates):
        source = position - spec.signal_lag
        if source >= 0 and dates[source] in target_by_date:
            # Target rows are authoritative, including all-zero rows: an empty
            # selection means go to cash, never silently hold stale names.
            weights = pd.Series(target_by_date[dates[source]], index=symbols, dtype=float)
            if has_selection_operators:
                # HoldingBuffer (5.5) / DiversitySelector (5.6): rebuild the
                # target row for the decision date from the still-held names
                # and the return history available up to that date.
                decision_date = dates[source]
                buffered = build_buffered_target_row(
                    composition, score.loc[decision_date], held_names, returns.loc[:decision_date]
                )
                weights = buffered.reindex(symbols).fillna(0.0)
            if has_risk_scaling:
                # DrawdownStateMachine (9.2) first -- the slow, strategic cap --
                # and PortfolioVolatilityTarget (9.1) inside it, composed as
                # ``min(dd_scale, vol_scale)`` so the circuit breaker always
                # outranks the tactical scaler.  Both read only returns already
                # realised before this decision day, and both scale the whole
                # target row: they are position decisions, not execution ones.
                risk_scales = []
                if spec.drawdown_thresholds is not None:
                    risk_scales.append(
                        drawdown_state_machine(
                            nav_path,
                            machine=drawdown_machine,
                            thresholds=spec.drawdown_thresholds,
                            scales=spec.drawdown_scales,
                            confirm_days=int(spec.drawdown_confirm_days),
                            recovery=float(spec.drawdown_recovery),
                        )
                    )
                if spec.vol_target_annual is not None:
                    risk_scales.append(
                        volatility_scale(
                            net_path,
                            target_annual=float(spec.vol_target_annual),
                            window=int(spec.vol_target_window),
                            min_scale=float(spec.vol_target_min_scale),
                            max_scale=float(spec.vol_target_max_scale),
                            periods_per_year=int(spec.trading_days_per_year),
                        )
                    )
                risk_scale = min(risk_scales)
                if risk_scale != 1.0:
                    weights = weights * risk_scale
            skipped = False
            if has_drift_band:
                forced = (
                    spec.drift_band_forced_days is not None
                    and days_since_rebalance >= spec.drift_band_forced_days
                )
                if forced or _drift_band_triggers(carried, weights, spec.drift_band_name, spec.drift_band_portfolio):
                    days_since_rebalance = 0
                else:
                    # DriftBandRebalance (8.3): inside the band the rebalance
                    # is skipped entirely -- the book keeps its drifted
                    # positions and the next target only arrives at the next
                    # rebalance date, so a skipped decision is not re-offered.
                    weights = carried
                    days_since_rebalance += 1
                    skipped = True
            if has_shrinkage:
                weights = _shrink_turnover(carried, weights, spec.turnover_shrinkage, spec.turnover_max_step)
            layer_ran = has_order_layer and not skipped
            if layer_ran:
                # Order layer (8.6-8.9): the plan's service order is fixed when
                # the plan is created and reused while it is alive, so the
                # priority keeps meaning "who this plan serves first" on the
                # days the plan only advances.  A skipped rebalance neither
                # advances nor revokes the outstanding plan.
                if spec.staged_max_days is not None:
                    plan_service_order = plan_order(
                        weights,
                        carried,
                        priority=spec.staged_priority,
                        score_row=score.loc[dates[source]] if score is not None else None,
                    )
                restore_pending = {symbol: list(entry) for symbol, entry in pending.items()}
                restore_order = list(plan_service_order)
                # The feasibility trace observes the order layer only; the
                # surrounding book (target and pre-order weights) is recorded
                # alongside it so the audit can reconstruct the whole chain.
                # ``pre_order_weights`` is the row the order layer is about to
                # receive -- the theoretical intention for this day, after the
                # risk scalings, band gate and shrinkage but before any of the
                # four execution operators.
                day_trace: dict[str, object] = {}
                pre_order_weights = weights.copy() if execution_trace is not None else None
                weights, pending = _execution_day(
                    carried,
                    pending,
                    weights,
                    spec=spec,
                    data=order_data,
                    order=plan_service_order,
                    day=day_key(date),
                    calendar=order_calendar,
                    calendar_index=order_calendar_index,
                    trace=day_trace if execution_trace is not None else None,
                )
                if execution_trace is not None:
                    execution_trace.append(
                        {
                            "date": date,
                            "position": position,
                            "carried": carried.copy(),
                            "target": pre_order_weights,
                            "stages": day_trace,
                        }
                    )
            if has_min_holding:
                adjusted = _apply_min_holding(
                    carried, weights, entry_index, position, int(spec.min_holding_days)
                )
                if adjusted is None:
                    # MinimumHoldingPeriod (8.4): the locked names cannot be
                    # funded, so the whole rebalance is abandoned -- the book
                    # keeps its drifted weights exactly like a band skip,
                    # including the days-since-rebalance counter.  The order
                    # layer is rolled back with it: a plan the engine declined
                    # to trade must not linger and trade itself tomorrow.
                    weights = carried
                    days_since_rebalance += 1
                    if layer_ran:
                        pending = restore_pending
                        plan_service_order = restore_order
                else:
                    weights = adjusted
        else:
            weights = carried
            if has_drift_band:
                days_since_rebalance += 1
            if has_order_layer:
                # No fresh decision: the outstanding plan advances on its own.
                weights, pending = _execution_day(
                    carried,
                    pending,
                    None,
                    spec=spec,
                    data=order_data,
                    order=plan_service_order,
                    day=day_key(date),
                    calendar=order_calendar,
                    calendar_index=order_calendar_index,
                )
        # State maintenance runs on every trading day (weights == carried on
        # non-trade days, so the membership is stable there).  A name enters
        # when its executed weight turns positive from a zero carried weight;
        # it leaves the book only when its executed weight reaches zero.
        if has_selection_operators:
            held_names = frozenset(sym for sym in symbols if weights[sym] > 0.0)
        if has_min_holding:
            for sym in symbols:
                if weights[sym] > 0.0:
                    if carried[sym] <= 0.0:
                        entry_index[sym] = position
                elif sym in entry_index:
                    del entry_index[sym]
        day_return = returns.loc[date]
        day_return = day_return.fillna(0.0)
        portfolio_return = float((weights * day_return).sum())
        delta = weights - carried
        buy_turnover.loc[date] = float(delta.clip(lower=0.0).sum())
        sell_turnover.loc[date] = float((-delta).clip(lower=0.0).sum())
        # Half-L1 turnover must include the implicit cash asset.  Omitting cash
        # would count a full initial investment (or liquidation) as only 0.5
        # and undercharge costs whenever the book is not fully invested.
        cash_before = 1.0 - float(carried.sum())
        cash_after = 1.0 - float(weights.sum())
        turnover.loc[date] = 0.5 * (float((weights - carried).abs().sum()) + abs(cash_after - cash_before))
        gross.loc[date] = portfolio_return
        start_weights.loc[date] = weights
        if has_risk_scaling:
            # Same cost formula the engine applies after the loop, advanced day
            # by day so the risk scalings only ever read realised history.
            day_net = portfolio_return - (
                float(buy_turnover.loc[date]) * BUY_COST_RATE + float(sell_turnover.loc[date]) * SELL_COST_RATE
            )
            net_path.append(day_net)
            nav_path.append(nav_path[-1] * (1.0 + day_net))
        denominator = 1.0 + portfolio_return
        if denominator > 0:
            carried = weights * (1.0 + day_return) / denominator
        # A non-positive denominator means the book was wiped out; the last
        # weights are kept so the remaining days report zero activity instead
        # of inventing a recovery.

    transaction_costs = buy_turnover * BUY_COST_RATE + sell_turnover * SELL_COST_RATE
    net = gross - transaction_costs

    metrics = _performance(net, turnover, start_weights, spec)
    metrics.update(_cost_metrics(buy_turnover, sell_turnover, transaction_costs, spec))
    metrics["signal_lag"] = spec.signal_lag
    benchmark_aligned: pd.Series | None = None
    if benchmark_returns is not None:
        benchmark_aligned = benchmark_returns.reindex(returns.index).fillna(0.0)
        metrics.update(_relative_performance(net, benchmark_aligned, spec))

    by_year: dict[str, dict[str, Any]] = {}
    years = sorted({str(index.year) for index in dates})
    for year in years:
        mask = [str(index.year) == year for index in dates]
        year_net = net[mask]
        if year_net.empty:
            continue
        year_metrics = _performance(year_net, turnover[mask], start_weights[mask], spec)
        year_metrics.update(
            _cost_metrics(buy_turnover[mask], sell_turnover[mask], transaction_costs[mask], spec)
        )
        if benchmark_aligned is not None:
            year_metrics.update(_relative_performance(year_net, benchmark_aligned[mask], spec))
        by_year[year] = year_metrics

    return PortfolioResult(
        metrics=metrics,
        metrics_by_year=by_year,
        daily_returns=net,
        daily_gross_returns=gross,
        weights=start_weights,
        turnover=turnover,
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        transaction_costs=transaction_costs,
    )
