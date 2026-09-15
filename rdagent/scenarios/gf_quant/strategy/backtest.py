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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec
from rdagent.scenarios.gf_quant.strategy.costs import (
    BUY_COST_RATE,
    SELL_COST_RATE,
    execution_costs_metrics,
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


def run_portfolio_backtest(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    spec: BacktestSpec,
    benchmark_returns: pd.Series | None = None,
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

    start_weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    gross = pd.Series(0.0, index=dates, dtype=float)
    turnover = pd.Series(0.0, index=dates, dtype=float)
    buy_turnover = pd.Series(0.0, index=dates, dtype=float)
    sell_turnover = pd.Series(0.0, index=dates, dtype=float)

    carried = pd.Series(0.0, index=symbols, dtype=float)
    for position, date in enumerate(dates):
        source = position - spec.signal_lag
        if source >= 0 and dates[source] in target_by_date:
            # Target rows are authoritative, including all-zero rows: an empty
            # selection means go to cash, never silently hold stale names.
            weights = pd.Series(target_by_date[dates[source]], index=symbols, dtype=float)
        else:
            weights = carried
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
