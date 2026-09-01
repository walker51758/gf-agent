"""Lightweight remote evaluator for the GF Quant RD-Agent scenario.

This module is intentionally independent from RD-Agent and Qlib.  It runs next
to ``G:/base`` on the Windows data host, evaluates a small, safe factor DSL,
and writes JSON metrics plus an optional parquet artifact.  The local RD-Agent
controller remains responsible for proposing candidates, feedback, de-dup and
registration.

The financial loader is point-in-time: ``dt`` (announcement date), rather than
``report`` (accounting period), is used to make values available to the panel.

Every supported operator is registered through :func:`operator_spec`, which
records its arity and literal-argument rules in ``OPERATOR_SPECS``; the local
coder mirrors that contract and the tests assert the two never drift apart.
See ``docs/factor_operator_catalog.md`` for the human-readable catalog.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DIRECT_FIELDS = {
    "adjclose",
    "adjhigh",
    "adjlow",
    "adjopen",
    "amount",
    "close",
    "floatshare",
    "fmv",
    "high",
    "industry",
    "low",
    "mv",
    "open",
    "pb",
    "pe",
    "ps",
    "totalshare",
    "turnover",
    "volume",
    "yclose",
}
FINANCIAL_TABLES = {"balance", "cashflow", "income"}
_FINANCIAL_FIELD = re.compile(r"^(balance|cashflow|income)__[A-Za-z0-9_]+$")
INDUSTRY_FILE = "industry_ci.parquet"
INDUSTRY_LEVEL = "中信一级"
INDUSTRY_HK_PREFIX = "CIHK"
_RAW_FINANCIAL_ARGUMENTS: dict[str, tuple[int, ...]] = {
    "YOY": (0,),
    "QOQ": (0,),
    "YOYOQOQ": (0,),
    "Delta": (0,),
    "SUD": (0,),
    "DeltaProp": (0,),
    "DeltaA2B": (0,),
    "YOYA_YOYB": (0, 1),
    "DeltaRegA2B": (0,),
    "YOYChange": (0,),
    "CAGR": (0,),
}
_INDUSTRY_ARGUMENTS: dict[str, int] = {
    "IndusProp": 1,
    "DeltaProp": 1,
    "cs_industry_rank": 1,
    "cs_industry_demean": 1,
    "cs_industry_zscore": 1,
}


def _as_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


class PanelLoader:
    """Load only fields referenced by one candidate from ``G:/base``."""

    def __init__(self, base_path: str | Path, start: str, end: str, lookback_days: int = 400) -> None:
        self.base_path = Path(base_path)
        self.start = _as_timestamp(start)
        self.end = _as_timestamp(end)
        self.read_start = self.start - pd.Timedelta(days=lookback_days)
        # SUD needs the current YoY change plus eight *previous* YoY changes,
        # i.e. at least twelve accounting quarters. Financial reads therefore
        # keep a bounded five-year history even when price operators request a
        # much shorter trading-day warm-up.
        self.financial_read_start = min(self.read_start, self.start - pd.DateOffset(years=5))
        self._cache: dict[str, pd.DataFrame] = {}
        self._calendar: pd.DatetimeIndex | None = None

    @property
    def calendar(self) -> pd.DatetimeIndex:
        if self._calendar is None:
            table = pq.read_table(self.base_path / "calendar.parquet", columns=["trade_dt"])
            dates = pd.DatetimeIndex(table.column("trade_dt").to_pandas()).normalize()
            dates = dates[(dates >= self.read_start) & (dates <= self.end)]
            self._calendar = pd.DatetimeIndex(dates.unique()).sort_values()
        return self._calendar

    def load(self, name: str) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name]
        if name == "industry":
            frame = self._load_industry()
        elif name in DIRECT_FIELDS:
            frame = self._load_direct(name)
        elif "__" in name:
            table, factor = name.split("__", 1)
            if table not in FINANCIAL_TABLES or not factor:
                raise ValueError(f"Unsupported financial field: {name}")
            frame = self._load_financial(table, factor)
        else:
            raise ValueError(f"Unsupported field: {name}")
        self._cache[name] = frame
        return frame

    def _load_direct(self, name: str) -> pd.DataFrame:
        table = pq.read_table(
            self.base_path / f"{name}.parquet",
            columns=["dt", "symbol", "value"],
            filters=[("dt", ">=", self.read_start.to_pydatetime()), ("dt", "<=", self.end.to_pydatetime())],
        )
        df = table.to_pandas()
        df["dt"] = pd.to_datetime(df["dt"]).dt.normalize()
        # Some raw sources contain duplicate refresh rows.  Keep the last value
        # deterministically rather than letting pivot fail.
        wide = df.pivot_table(index="dt", columns="symbol", values="value", aggfunc="last")
        wide = wide.reindex(self.calendar).sort_index()
        return _finite(wide.astype(float))

    def _load_industry(self) -> pd.DataFrame:
        """Build a daily CITIC level-1 panel from effective-date intervals.

        The verified remote table contains parallel mainland ``CI...`` and
        Hong Kong mirror ``CIHK...`` rows.  The GF universe is A-share, so the
        mirror rows are excluded before interval expansion.  ``indus_code`` is
        used instead of the display name because it is stable and encoding-safe.
        """

        table = pq.read_table(
            self.base_path / INDUSTRY_FILE,
            columns=["symbol", "beg_date", "end_date", "indus_code", "level"],
            filters=[
                ("level", "=", INDUSTRY_LEVEL),
                ("beg_date", "<=", self.end.to_pydatetime()),
                ("end_date", ">=", self.read_start.to_pydatetime()),
            ],
        )
        intervals = table.to_pandas()
        if intervals.empty:
            raise ValueError(f"No {INDUSTRY_LEVEL} rows overlap the requested period in {INDUSTRY_FILE}")
        intervals["beg_date"] = pd.to_datetime(intervals["beg_date"]).dt.normalize()
        intervals["end_date"] = pd.to_datetime(intervals["end_date"]).dt.normalize()
        codes = intervals["indus_code"].astype("string")
        intervals = intervals.loc[~codes.str.startswith(INDUSTRY_HK_PREFIX, na=False)].copy()
        if intervals.empty:
            raise ValueError(f"No mainland {INDUSTRY_LEVEL} rows found in {INDUSTRY_FILE}")
        if not intervals["indus_code"].astype("string").str.startswith("CI", na=False).all():
            raise ValueError(f"Unexpected non-CITIC industry code in {INDUSTRY_FILE}")

        calendar = self.calendar
        panel: dict[object, np.ndarray] = {}
        for symbol, group in intervals.sort_values(["symbol", "beg_date", "end_date"]).groupby(
            "symbol", sort=True
        ):
            values = np.full(len(calendar), None, dtype=object)
            for row in group.itertuples(index=False):
                positions = np.flatnonzero((calendar >= row.beg_date) & (calendar <= row.end_date))
                if not len(positions):
                    continue
                existing = values[positions]
                conflicting = pd.notna(existing) & (existing != row.indus_code)
                if bool(np.any(conflicting)):
                    raise ValueError(f"Overlapping {INDUSTRY_LEVEL} classifications for {symbol}")
                values[positions] = row.indus_code
            panel[symbol] = values

        result = pd.DataFrame(panel, index=calendar, dtype=object)
        result.attrs["gf_industry_source"] = INDUSTRY_FILE
        result.attrs["gf_industry_level"] = INDUSTRY_LEVEL
        result.attrs["gf_industry_value"] = "indus_code"
        return result

    def _load_financial(self, table_name: str, factor: str) -> pd.DataFrame:
        table = pq.read_table(
            self.base_path / f"{table_name}.parquet",
            columns=["dt", "report", "symbol", "value"],
            filters=[
                ("factor", "=", factor),
                ("dt", ">=", self.financial_read_start.to_pydatetime()),
                ("dt", "<=", self.end.to_pydatetime()),
            ],
        )
        df = table.to_pandas()
        if df.empty:
            raise ValueError(f"No rows for {table_name}__{factor}")
        df["dt"] = pd.to_datetime(df["dt"]).dt.normalize()
        df["report"] = pd.to_datetime(df["report"]).dt.normalize()
        # Preserve distinct report-period revisions announced on the same day
        # for quarter-aware lags. The displayed raw panel uses only the newest
        # accounting period for each symbol/day.
        history = df.sort_values(["symbol", "dt", "report"]).drop_duplicates(
            ["dt", "report", "symbol"], keep="last"
        )
        latest = history.drop_duplicates(["dt", "symbol"], keep="last")
        events = latest.pivot(index="dt", columns="symbol", values="value").sort_index()
        # Announcement dates can be weekends or holidays. Fill on the union
        # first so such events become visible on the next trading day instead
        # of disappearing during the calendar reindex.
        expanded_index = events.index.union(self.calendar).sort_values()
        frame = _finite(events.reindex(expanded_index).ffill().reindex(self.calendar).astype(float))
        # Quarterly operators must align by accounting period while observing
        # values only after their announcement date.  Retain the event history
        # as metadata on the raw financial panel; period-aware operators reject
        # transformed/non-financial arguments whose metadata is absent.
        frame.attrs["gf_financial_events"] = history[["dt", "report", "symbol", "value"]].copy()
        frame.attrs["gf_calendar"] = self.calendar
        frame.attrs["gf_financial_field"] = f"{table_name}__{factor}"
        frame.attrs["gf_quarter_lag_cache"] = {}
        return frame


@dataclass(frozen=True)
class ArgRule:
    """Validation for one positional call argument that must be a literal.

    ``minimum``/``maximum`` are exclusive. Windows and shifts use ``kind="int"``;
    fractions such as the winsorize quantile use ``kind="float"``.
    """

    position: int
    kind: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class OperatorSpec:
    """Declarative contract of one DSL operator: name, signature and argument rules."""

    name: str
    signature: str
    min_args: int
    max_args: int
    arguments: tuple[ArgRule, ...] = ()


OPERATOR_SPECS: dict[str, OperatorSpec] = {}
ALLOWED_FUNCTIONS: dict[str, Callable[..., pd.DataFrame]] = {}


def operator_spec(
    signature: str, min_args: int, max_args: int, arguments: tuple[ArgRule, ...] = ()
) -> Callable[[Callable[..., pd.DataFrame]], Callable[..., pd.DataFrame]]:
    """Register the decorated function as a DSL operator under the name in ``signature``."""

    def register(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        name = signature.split("(", 1)[0].strip()
        if name in OPERATOR_SPECS:
            raise ValueError(f"Duplicated GF DSL operator: {name}")
        OPERATOR_SPECS[name] = OperatorSpec(name, signature, min_args, max_args, arguments)
        ALLOWED_FUNCTIONS[name] = func
        return func

    return register


def _aligned_pair(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect the symbols of two panels so rolling two-panel statistics line up."""
    common = x.columns.intersection(y.columns)
    left = x[common]
    return left, y.reindex(index=left.index, columns=common)


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """Align two panels and turn zero denominators/infinities into missing values."""

    left, right = _aligned_pair(numerator, denominator)
    return _finite(left / right.where(right != 0))


def _quarter_lag(x: pd.DataFrame, quarters: int) -> pd.DataFrame:
    """Return the value from an earlier accounting quarter without lookahead.

    ``x`` must be a raw financial panel loaded by :class:`PanelLoader`.  For
    every announcement event we maintain only report values announced up to
    that ``dt``; a later restatement can therefore never leak backward.
    """

    events = x.attrs.get("gf_financial_events")
    calendar = x.attrs.get("gf_calendar")
    cache = x.attrs.get("gf_quarter_lag_cache")
    if not isinstance(events, pd.DataFrame) or calendar is None or not isinstance(cache, dict):
        raise ValueError("Quarterly operators require a raw point-in-time financial field")
    offset = int(quarters)
    if offset < 1:
        raise ValueError("Quarter lag must be positive")
    if offset in cache:
        return cache[offset]

    records: list[tuple[pd.Timestamp, object, float]] = []
    ordered = events.sort_values(["symbol", "dt", "report"])
    for symbol, group in ordered.groupby("symbol", sort=False):
        seen: dict[pd.Period, float] = {}
        for row in group.itertuples(index=False):
            report = pd.Timestamp(row.report).to_period("Q")
            seen[report] = float(row.value)
            records.append((pd.Timestamp(row.dt), symbol, seen.get(report - offset, math.nan)))

    if records:
        lag_events = pd.DataFrame(records, columns=["dt", "symbol", "value"])
        wide = lag_events.pivot_table(index="dt", columns="symbol", values="value", aggfunc="last")
        calendar_index = pd.DatetimeIndex(calendar)
        expanded_index = wide.index.union(calendar_index).sort_values()
        result = (
            wide.reindex(expanded_index)
            .ffill()
            .reindex(calendar_index)
            .reindex(columns=x.columns)
            .astype(float)
        )
    else:
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    result = _finite(result)
    cache[offset] = result
    return result


def _industry_proportion(x: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    """Divide each value by its same-date, same-industry cross-sectional sum."""

    left, codes = _aligned_pair(x, industry)
    result = pd.DataFrame(index=left.index, columns=left.columns, dtype=float)
    for dt in left.index:
        values = left.loc[dt]
        groups = codes.loc[dt]
        valid = values.notna() & groups.notna()
        if not valid.any():
            continue
        totals = values[valid].groupby(groups[valid]).transform("sum")
        result.loc[dt, valid] = values[valid] / totals.where(totals != 0)
    return _finite(result)


def _cross_sectional_residual(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Daily dependency-free OLS residual of ``a`` on an intercept and ``b``."""

    left, right = _aligned_pair(a, b)
    result = pd.DataFrame(index=left.index, columns=left.columns, dtype=float)
    for dt in left.index:
        pair = pd.concat([left.loc[dt].rename("a"), right.loc[dt].rename("b")], axis=1).dropna()
        if len(pair) < 3:
            continue
        centered_b = pair["b"] - pair["b"].mean()
        denominator = float((centered_b**2).sum())
        if not math.isfinite(denominator) or denominator <= 0:
            continue
        centered_a = pair["a"] - pair["a"].mean()
        beta = float((centered_a * centered_b).sum() / denominator)
        alpha = float(pair["a"].mean() - beta * pair["b"].mean())
        result.loc[dt, pair.index] = pair["a"] - (alpha + beta * pair["b"])
    return _finite(result)


def validate_call(spec: OperatorSpec, node: ast.Call) -> None:
    """Check arity and literal argument rules before any data is loaded."""
    if not spec.min_args <= len(node.args) <= spec.max_args:
        raise ValueError(f"{spec.signature} expects between {spec.min_args} and {spec.max_args} arguments")
    for rule in spec.arguments:
        if rule.position >= len(node.args):
            continue
        argument = node.args[rule.position]
        value = _numeric_literal(argument)
        if value is None:
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be a numeric literal")
        if rule.kind == "int" and value != int(value):
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be a whole number")
        if rule.minimum is not None and value <= rule.minimum:
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be greater than {rule.minimum}")
        if rule.maximum is not None and value >= rule.maximum:
            raise ValueError(f"{spec.signature} argument {rule.position + 1} must be less than {rule.maximum}")
    for position in _RAW_FINANCIAL_ARGUMENTS.get(spec.name, ()):
        argument = node.args[position]
        if not isinstance(argument, ast.Name) or not _FINANCIAL_FIELD.fullmatch(argument.id):
            raise ValueError(f"{spec.signature} argument {position + 1} must be a raw point-in-time financial field")
    if spec.name == "clip" and _numeric_literal(node.args[1]) >= _numeric_literal(node.args[2]):
        raise ValueError("clip(x, lo, hi) requires lo < hi")
    industry_position = _INDUSTRY_ARGUMENTS.get(spec.name)
    if industry_position is not None:
        argument = node.args[industry_position]
        if not isinstance(argument, ast.Name) or argument.id != "industry":
            raise ValueError(f"{spec.signature} requires the industry field as argument {industry_position + 1}")


def _numeric_literal(node: ast.AST) -> float | None:
    """Return a signed numeric literal while rejecting names, bools and expressions."""

    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and not isinstance(node.operand.value, bool)
        and isinstance(node.operand.value, (int, float))
    ):
        value = float(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _rolling(x: pd.DataFrame, window: int) -> Any:
    """Rolling window shared by every ``ts_`` operator.

    ``min_periods`` keeps a stability floor so a handful of observations can
    never define a long-window statistic; it is identical across operators so
    that composed expressions degrade to NaN coherently.
    """
    size = int(window)
    return x.rolling(size, min_periods=max(3, size // 2))


@operator_spec("ts_mean(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).mean()


@operator_spec("ts_std(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).std()


@operator_spec("ts_sum(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_sum(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).sum()


@operator_spec("ts_min(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).min()


@operator_spec("ts_max(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).max()


@operator_spec("ts_median(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_median(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).median()


@operator_spec("ts_mad(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=2),))
def ts_mad(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing median absolute deviation from the trailing median."""

    def _mad(values: np.ndarray) -> float:
        valid = values[np.isfinite(values)]
        if len(valid) < 3:
            return math.nan
        median = float(np.median(valid))
        return float(np.median(np.abs(valid - median)))

    return _rolling(x, int(window)).apply(_mad, raw=True)


@operator_spec("ts_skew(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_skew(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).skew()


@operator_spec("ts_kurt(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_kurt(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling(x, window).kurt()


def _rolling_time_regression(x: pd.DataFrame, window: int, *, residual: bool) -> pd.DataFrame:
    """OLS on actual positions in each trailing window; optionally return the endpoint residual."""

    def _fit(values: np.ndarray) -> float:
        if residual and not np.isfinite(values[-1]):
            return math.nan
        valid = np.isfinite(values)
        if valid.sum() < 3:
            return math.nan
        positions = np.arange(len(values), dtype=float)[valid]
        observations = values[valid].astype(float)
        centered_positions = positions - positions.mean()
        denominator = float(np.dot(centered_positions, centered_positions))
        if not math.isfinite(denominator) or denominator <= 0:
            return math.nan
        slope = float(np.dot(centered_positions, observations - observations.mean()) / denominator)
        if not residual:
            return slope
        intercept = float(observations.mean() - slope * positions.mean())
        return float(values[-1] - (intercept + slope * (len(values) - 1)))

    return _finite(_rolling(x, int(window)).apply(_fit, raw=True))


@operator_spec("ts_slope(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=2),))
def ts_slope(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_time_regression(x, window, residual=False)


@operator_spec("ts_residual(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=2),))
def ts_residual(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_time_regression(x, window, residual=True)


def _rolling_signed_std(x: pd.DataFrame, window: int, *, positive: bool) -> pd.DataFrame:
    def _std(values: np.ndarray) -> float:
        selected = values[values > 0] if positive else values[values < 0]
        selected = selected[np.isfinite(selected)]
        return float(np.std(selected, ddof=1)) if len(selected) >= 3 else math.nan

    return _finite(_rolling(x, int(window)).apply(_std, raw=True))


@operator_spec("ts_upside_std(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=2),))
def ts_upside_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_signed_std(x, window, positive=True)


@operator_spec("ts_downside_std(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=2),))
def ts_downside_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _rolling_signed_std(x, window, positive=False)


@operator_spec("ts_above_mean(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=2),))
def ts_above_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Fraction of recent dates where x exceeded that date's own trailing mean."""

    trailing_mean = ts_mean(x, int(window))
    valid = x.notna() & trailing_mean.notna()
    stood_above = (x > trailing_mean).where(valid).astype(float)
    return _finite(_rolling(stood_above, int(window)).mean())


@operator_spec(
    "ts_quantile(x, n, q)",
    min_args=3,
    max_args=3,
    arguments=(ArgRule(1, "int", minimum=1), ArgRule(2, "float", minimum=0.0, maximum=1.0)),
)
def ts_quantile(x: pd.DataFrame, window: int, quantile: float) -> pd.DataFrame:
    """``q``-th quantile of the trailing window; ``q`` must lie strictly inside (0, 1)."""
    return _rolling(x, window).quantile(float(quantile))


@operator_spec("ts_zscore(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    std = ts_std(x, window)
    return (x - ts_mean(x, window)) / std.where(std > 0)


@operator_spec("ts_rank(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Percentile of the latest value inside its own trailing window, centered like ``cs_rank``."""
    size = int(window)
    floor = max(3, size // 2)

    def _percentile(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return math.nan
        valid = values[~np.isnan(values)]
        if len(valid) < floor:
            return math.nan
        current = valid[-1]
        below = float((valid < current).sum())
        ties = float((valid == current).sum())  # includes the current value itself
        rank = below + 0.5 * (ties + 1.0)  # average tie handling, as in cs_rank
        return rank / len(valid) - 0.5

    return _rolling(x, size).apply(_percentile, raw=True)


@operator_spec("ts_decay_linear(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Linearly weighted mean; the newest observation always carries the largest weight."""

    def _weighted(values: np.ndarray) -> float:
        mask = ~np.isnan(values)
        if not mask.any():
            return math.nan
        weights = np.arange(1, len(values) + 1, dtype=float)
        return float(np.dot(values[mask], weights[mask]) / weights[mask].sum())

    return _rolling(x, int(window)).apply(_weighted, raw=True)


@operator_spec("ts_ewm(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_ewm(x: pd.DataFrame, window: int) -> pd.DataFrame:
    size = int(window)
    return x.ewm(span=size, min_periods=max(3, size // 2)).mean()


@operator_spec("ts_argmax(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Days since the trailing-window maximum; 0 means today holds it."""

    def _offset(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return math.nan
        # Search from today backwards so tied extremes resolve to the most
        # recent occurrence, matching the "days since" interpretation.
        return float(np.nanargmax(values[::-1]))

    return _rolling(x, int(window)).apply(_offset, raw=True)


@operator_spec("ts_argmin(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Days since the trailing-window minimum; 0 means today holds it."""

    def _offset(values: np.ndarray) -> float:
        if np.isnan(values).all():
            return math.nan
        return float(np.nanargmin(values[::-1]))

    return _rolling(x, int(window)).apply(_offset, raw=True)


@operator_spec("delay(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def delay(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return x.shift(int(periods))


@operator_spec("delta(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def delta(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return x.diff(int(periods))


@operator_spec("pct_change(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def pct_change(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return _finite(x.pct_change(int(periods), fill_method=None))


# Directed point-in-time financial operators.  Capitalised names intentionally
# distinguish report-period transformations from the lowercase daily ``delta``.
@operator_spec("Self(x)", min_args=1, max_args=1)
def Self(x: pd.DataFrame) -> pd.DataFrame:
    return x.copy()


@operator_spec("YOY(x)", min_args=1, max_args=1)
def YOY(x: pd.DataFrame) -> pd.DataFrame:
    previous = _quarter_lag(x, 4)
    return _safe_divide(x - previous, previous.abs())


@operator_spec("QOQ(x)", min_args=1, max_args=1)
def QOQ(x: pd.DataFrame) -> pd.DataFrame:
    previous = _quarter_lag(x, 1)
    return _safe_divide(x - previous, previous.abs())


@operator_spec("YOYOQOQ(x)", min_args=1, max_args=1)
def YOYOQOQ(x: pd.DataFrame) -> pd.DataFrame:
    current_yoy = YOY(x)
    prior_quarter = _quarter_lag(x, 1)
    prior_year_base = _quarter_lag(x, 5)
    prior_yoy = _safe_divide(prior_quarter - prior_year_base, prior_year_base.abs())
    return _finite(current_yoy - prior_yoy)


@operator_spec("YOYChange(x, k)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=0),))
def YOYChange(x: pd.DataFrame, quarters: int) -> pd.DataFrame:
    """Change in point-in-time YoY growth versus an earlier accounting quarter."""

    offset = int(quarters)
    prior = _quarter_lag(x, offset)
    prior_base = _quarter_lag(x, offset + 4)
    prior_yoy = _safe_divide(prior - prior_base, prior_base.abs())
    return _finite(YOY(x) - prior_yoy)


@operator_spec("CAGR(x, n)", min_args=2, max_args=2, arguments=(ArgRule(1, "int", minimum=0),))
def CAGR(x: pd.DataFrame, quarters: int) -> pd.DataFrame:
    """Annualised compound growth over a number of accounting quarters."""

    offset = int(quarters)
    previous = _quarter_lag(x, offset)
    valid = (x > 0) & (previous > 0)
    ratio = _safe_divide(x.where(valid), previous.where(valid))
    return _finite(ratio.pow(4.0 / offset) - 1.0)


@operator_spec("Delta(x)", min_args=1, max_args=1)
def Delta(x: pd.DataFrame) -> pd.DataFrame:
    return _finite(x - _quarter_lag(x, 4))


@operator_spec("SUD(x, lookback)", min_args=1, max_args=2, arguments=(ArgRule(1, "int", minimum=1),))
def SUD(x: pd.DataFrame, lookback: int = 8) -> pd.DataFrame:
    """Standardised unexpected YoY absolute change using prior quarters only."""

    history_count = int(lookback)
    current_change = Delta(x)
    history = [_quarter_lag(x, k) - _quarter_lag(x, k + 4) for k in range(1, history_count + 1)]
    values = np.stack([frame.to_numpy(dtype=float) for frame in history], axis=0)
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    total = np.where(valid, values, 0.0).sum(axis=0)
    mean = np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0)
    squared = np.where(valid, (values - mean) ** 2, 0.0).sum(axis=0)
    std = np.sqrt(np.divide(squared, counts - 1, out=np.full_like(total, np.nan), where=counts > 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (current_change.to_numpy(dtype=float) - mean) / std
    result[(counts != history_count) | ~np.isfinite(std) | (std <= 0)] = np.nan
    return pd.DataFrame(result, index=x.index, columns=x.columns)


@operator_spec("IndusProp(x, industry)", min_args=2, max_args=2)
def IndusProp(x: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    return _industry_proportion(x, industry)


@operator_spec("DeltaProp(x, industry)", min_args=2, max_args=2)
def DeltaProp(x: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    current = _industry_proportion(x, industry)
    previous = _industry_proportion(_quarter_lag(x, 4), industry)
    return _finite(current - previous)


@operator_spec("A2B(a, b)", min_args=2, max_args=2)
def A2B(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _safe_divide(a, b)


@operator_spec("PercentileA2B(a, b)", min_args=2, max_args=2)
def PercentileA2B(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    left, right = _aligned_pair(a, b)
    return left.rank(axis=1, pct=True, method="average") - right.rank(axis=1, pct=True, method="average")


@operator_spec("RegA2B(a, b)", min_args=2, max_args=2)
def RegA2B(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _cross_sectional_residual(a, b)


@operator_spec("DeltaA2B(a, b)", min_args=2, max_args=2)
def DeltaA2B(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _safe_divide(Delta(a), b)


@operator_spec("YOYA_YOYB(a, b)", min_args=2, max_args=2)
def YOYA_YOYB(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    left, right = _aligned_pair(YOY(a), YOY(b))
    return _finite(left - right)


@operator_spec("DeltaRegA2B(a, b)", min_args=2, max_args=2)
def DeltaRegA2B(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return _cross_sectional_residual(Delta(a), b)


@operator_spec("ts_corr(x, y, n)", min_args=3, max_args=3, arguments=(ArgRule(2, "int", minimum=1),))
def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    left, right = _aligned_pair(x, y)
    return _rolling(left, window).corr(right)


@operator_spec("ts_cov(x, y, n)", min_args=3, max_args=3, arguments=(ArgRule(2, "int", minimum=1),))
def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    left, right = _aligned_pair(x, y)
    return _rolling(left, window).cov(right)


@operator_spec("cs_rank(x)", min_args=1, max_args=1)
def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True, method="average") - 0.5


@operator_spec("cs_zscore(x)", min_args=1, max_args=1)
def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    std = x.std(axis=1, ddof=1)
    return x.sub(x.mean(axis=1), axis=0).div(std.where(std > 0), axis=0)


@operator_spec("cs_demean(x)", min_args=1, max_args=1)
def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


@operator_spec("cs_scale(x)", min_args=1, max_args=1)
def cs_scale(x: pd.DataFrame) -> pd.DataFrame:
    total = x.abs().sum(axis=1)
    return x.div(total.where(total > 0), axis=0)


def _industry_group_transform(x: pd.DataFrame, industry: pd.DataFrame, transform: str) -> pd.DataFrame:
    left, codes = _aligned_pair(x, industry)
    result = pd.DataFrame(index=left.index, columns=left.columns, dtype=float)
    for dt in left.index:
        values = left.loc[dt]
        groups = codes.loc[dt]
        valid = values.notna() & groups.notna()
        if not valid.any():
            continue
        grouped = values[valid].groupby(groups[valid])
        if transform == "rank":
            transformed = grouped.rank(method="average", pct=True) - 0.5
        elif transform == "demean":
            transformed = values[valid] - grouped.transform("mean")
        else:
            means = grouped.transform("mean")
            stds = grouped.transform("std")
            transformed = (values[valid] - means) / stds.where(stds > 0)
        result.loc[dt, transformed.index] = transformed
    return _finite(result)


@operator_spec("cs_industry_rank(x, industry)", min_args=2, max_args=2)
def cs_industry_rank(x: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    return _industry_group_transform(x, industry, "rank")


@operator_spec("cs_industry_demean(x, industry)", min_args=2, max_args=2)
def cs_industry_demean(x: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    return _industry_group_transform(x, industry, "demean")


@operator_spec("cs_industry_zscore(x, industry)", min_args=2, max_args=2)
def cs_industry_zscore(x: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    return _industry_group_transform(x, industry, "zscore")


@operator_spec("cs_zscore_mad(x)", min_args=1, max_args=1)
def cs_zscore_mad(x: pd.DataFrame) -> pd.DataFrame:
    median = x.median(axis=1)
    deviation = x.sub(median, axis=0).abs()
    mad = deviation.median(axis=1)
    return _finite(x.sub(median, axis=0).div(mad.where(mad > 0), axis=0))


@operator_spec("signed_log(x)", min_args=1, max_args=1)
def signed_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x) * np.log1p(np.abs(x))


@operator_spec("winsorize(x, p)", min_args=1, max_args=2, arguments=(ArgRule(1, "float", minimum=0.0, maximum=0.5),))
def winsorize(x: pd.DataFrame, limit: float = 0.01) -> pd.DataFrame:
    lo = x.quantile(float(limit), axis=1)
    hi = x.quantile(1.0 - float(limit), axis=1)
    return x.clip(lo, hi, axis=0)


@operator_spec(
    "clip(x, lo, hi)",
    min_args=3,
    max_args=3,
    arguments=(ArgRule(1, "float"), ArgRule(2, "float")),
)
def clip(x: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    if not float(lower) < float(upper):
        raise ValueError("clip(x, lo, hi) requires lo < hi")
    return _finite(x.clip(lower=float(lower), upper=float(upper)))


@operator_spec("abs(x)", min_args=1, max_args=1)
def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


@operator_spec("log(x)", min_args=1, max_args=1)
def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


@operator_spec("sqrt(x)", min_args=1, max_args=1)
def _sqrt(x: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(x.where(x >= 0))


@operator_spec("sign(x)", min_args=1, max_args=1)
def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


@operator_spec("inverse(x)", min_args=1, max_args=1)
def _inverse(x: pd.DataFrame) -> pd.DataFrame:
    return 1.0 / x.where(x != 0)


class SafeExpression:
    """Evaluate arithmetic expressions without ``eval`` or arbitrary access."""

    _binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    _unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}

    def __init__(self, expression: str) -> None:
        self.expression = expression.strip()
        self.tree = ast.parse(self.expression, mode="eval")
        called = {
            node.func.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            spec = OPERATOR_SPECS.get(node.func.id) if isinstance(node.func, ast.Name) else None
            if spec is None:
                raise ValueError("GF DSL calls must use an allow-listed operator")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            validate_call(spec, node)
        fields: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Name):
                continue
            if node.id in OPERATOR_SPECS:
                if node.id not in called:
                    raise ValueError(f"Operator {node.id} must be called, not referenced")
                continue
            if node.id in DIRECT_FIELDS or _FINANCIAL_FIELD.fullmatch(node.id):
                fields.add(node.id)
                continue
            raise ValueError(f"Unsupported GF data field: {node.id}")
        if not fields:
            raise ValueError("Expression must reference at least one data field")
        self.fields = sorted(fields)

    def evaluate(self, loader: PanelLoader) -> pd.DataFrame:
        values = {field: loader.load(field) for field in self.fields}
        result = self._node(self.tree.body, values)
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Expression must produce a panel")
        return _finite(result)

    def _node(self, node: ast.AST, values: dict[str, pd.DataFrame]) -> Any:
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            return self._binary[type(node.op)](self._node(node.left, values), self._node(node.right, values))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._node(node.operand, values))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            return ALLOWED_FUNCTIONS[node.func.id](*[self._node(arg, values) for arg in node.args])
        raise ValueError(f"Unsupported expression element: {ast.dump(node, include_attributes=False)}")


@dataclass(frozen=True)
class EvaluationConfig:
    start: str
    end: str
    label_horizon: int = 5
    quantile: float = 0.2
    min_cross_section: int = 30
    sample_step: int = 5


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else math.nan


def _daily_correlations(factor: pd.DataFrame, label: pd.DataFrame, method: str, min_n: int) -> pd.Series:
    result: dict[pd.Timestamp, float] = {}
    for dt in factor.index:
        pair = pd.concat([factor.loc[dt].rename("factor"), label.loc[dt].rename("label")], axis=1).dropna()
        if len(pair) >= min_n and pair["factor"].nunique() > 1 and pair["label"].nunique() > 1:
            if method == "spearman":
                # pandas delegates Spearman to scipy, which is deliberately not
                # installed in the lightweight remote environment.  Ranking
                # both vectors and taking Pearson is the same definition.
                left = pair["factor"].rank(method="average")
                right = pair["label"].rank(method="average")
                result[dt] = float(left.corr(right))
            else:
                result[dt] = float(pair["factor"].corr(pair["label"]))
    return pd.Series(result, dtype=float)


@dataclass(frozen=True)
class DailySeries:
    """Per-date evaluation series shared by the aggregate metrics and profiles.

    ``spread`` / ``rank_ic`` are the compact daily profiles emitted for
    benchmark calibration and correlation clustering; keeping them in one place
    guarantees the benchmark worker and the candidate evaluator share one
    definition.
    """

    coverage: pd.Series
    ic: pd.Series
    rank_ic: pd.Series
    spread: pd.Series
    turnover_daily: pd.Series


def daily_series(factor: pd.DataFrame, close: pd.DataFrame, cfg: EvaluationConfig) -> tuple[DailySeries, pd.DataFrame]:
    """Compute the per-date series behind :func:`evaluate_factor`.

    Returns the series plus the factor restricted to the evaluation window and
    the columns that also exist in ``close``.
    """
    factor = factor.loc[_as_timestamp(cfg.start) : _as_timestamp(cfg.end)]
    common_columns = factor.columns.intersection(close.columns)
    factor = factor.loc[:, common_columns]
    close = close.reindex(index=factor.index, columns=common_columns)
    # Coverage is measured only where the forward-return source (close) exists.
    # A factor can have stale/alternate-field values on rows whose close is
    # missing, so counting factor non-nulls alone can incorrectly exceed 100%.
    coverage = (factor.notna() & close.notna()).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
    forward = close.shift(-cfg.label_horizon) / close - 1.0
    sampled_factor = factor.iloc[:: cfg.sample_step]
    sampled_label = forward.reindex(sampled_factor.index)
    ic = _daily_correlations(sampled_factor, sampled_label, "pearson", cfg.min_cross_section)
    rank_ic = _daily_correlations(sampled_factor, sampled_label, "spearman", cfg.min_cross_section)

    # A t factor is observable after close.  Use it for t+1 close-to-close
    # return, preventing same-day lookahead.
    one_day = close.pct_change(fill_method=None)
    signal = factor.shift(1).rank(axis=1, pct=True)
    long_mask = signal >= 1.0 - cfg.quantile
    short_mask = signal <= cfg.quantile
    long_ret = one_day.where(long_mask).mean(axis=1)
    short_ret = one_day.where(short_mask).mean(axis=1)
    spread = (long_ret - short_ret).replace([np.inf, -np.inf], np.nan).dropna()
    active = long_mask.astype(np.int8) - short_mask.astype(np.int8)
    # The daily series keeps the aggregate turnover definition unchanged while
    # still allowing the per-calendar-year breakdown below.
    turnover_daily = active.diff().abs().sum(axis=1).div(active.abs().sum(axis=1).replace(0, np.nan))
    return DailySeries(coverage, ic, rank_ic, spread, turnover_daily), factor


def evaluate_factor(factor: pd.DataFrame, close: pd.DataFrame, cfg: EvaluationConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    daily, factor = daily_series(factor, close, cfg)
    return metrics_from_daily(daily, factor.index), factor


def metrics_from_daily(daily: DailySeries, index: pd.DatetimeIndex) -> dict[str, Any]:
    """Aggregate + per-year metrics from the daily evaluation series.

    Split from :func:`evaluate_factor` so the benchmark worker can evaluate an
    existing wide panel once and reuse the same series for both metrics and the
    compact profile.
    """
    coverage = daily.coverage
    ic = daily.ic
    rank_ic = daily.rank_ic
    spread = daily.spread
    turnover_daily = daily.turnover_daily
    spread_std = float(spread.std(ddof=1)) if len(spread) > 1 else math.nan
    ann_return = float(spread.mean() * 252.0) if not spread.empty else math.nan
    sharpe = float(spread.mean() / spread_std * math.sqrt(252.0)) if spread_std and spread_std > 0 else math.nan
    turnover = float(turnover_daily.mean())

    def mean_std_ir(series: pd.Series) -> tuple[float, float, float]:
        mean = float(series.mean()) if not series.empty else math.nan
        std = float(series.std(ddof=1)) if len(series) > 1 else math.nan
        ir = mean / std if std and std > 0 else math.nan
        return mean, std, ir

    ic_mean, ic_std, icir = mean_std_ir(ic)
    ric_mean, ric_std, ricir = mean_std_ir(rank_ic)
    yearly = spread.groupby(spread.index.year).agg(["mean", "std", "count"])
    stability = {
        str(int(year)): {
            "annualized_return": float(row["mean"] * 252.0),
            "sharpe": float(row["mean"] / row["std"] * math.sqrt(252.0)) if row["std"] > 0 else math.nan,
            "days": int(row["count"]),
        }
        for year, row in yearly.iterrows()
    }

    # Per-calendar-year metrics, attributed to the signal date.  The forward
    # label may cross a December/January boundary; that does not move the
    # observation between years.  Every year present in the signal window is
    # reported so downstream gates and feedback see a stable key set.
    metrics_by_year: dict[str, dict[str, Any]] = {}
    for year in sorted({int(value) for value in index.year}):
        year_dates = index[index.year == year]
        spread_year = spread.reindex(year_dates).dropna()
        ic_year = ic.reindex(year_dates)
        rank_ic_year = rank_ic.reindex(year_dates)
        ic_mean_year, ic_std_year, icir_year = mean_std_ir(ic_year)
        ric_mean_year, ric_std_year, ricir_year = mean_std_ir(rank_ic_year)
        spread_mean_year = float(spread_year.mean()) if not spread_year.empty else math.nan
        spread_std_year = float(spread_year.std(ddof=1)) if len(spread_year) > 1 else math.nan
        metrics_by_year[str(year)] = {
            "IC": ic_mean_year,
            "IC_std": ic_std_year,
            "ICIR": icir_year,
            "Rank_IC": ric_mean_year,
            "Rank_IC_std": ric_std_year,
            "Rank_ICIR": ricir_year,
            "annualized_return": spread_mean_year * 252.0 if not spread_year.empty else math.nan,
            "sharpe": (
                spread_mean_year / spread_std_year * math.sqrt(252.0)
                if spread_std_year and spread_std_year > 0
                else math.nan
            ),
            "max_drawdown": _max_drawdown(spread_year),
            "turnover": float(turnover_daily.reindex(year_dates).mean()),
            "coverage": float(coverage.reindex(year_dates).mean()),
            "long_short_spread_daily": spread_mean_year,
            "win_rate": float((spread_year > 0).mean()) if not spread_year.empty else math.nan,
            "observations": int(len(spread_year)),
            "ic_observations": int(ic_year.notna().sum()),
        }

    metrics: dict[str, Any] = {
        "IC": ic_mean,
        "IC_std": ic_std,
        "ICIR": icir,
        "Rank_IC": ric_mean,
        "Rank_IC_std": ric_std,
        "Rank_ICIR": ricir,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(spread),
        "turnover": turnover,
        "coverage": float(coverage.mean()),
        "long_short_spread_daily": float(spread.mean()) if not spread.empty else math.nan,
        "win_rate": float((spread > 0).mean()) if not spread.empty else math.nan,
        "observations": int(len(spread)),
        "ic_observations": int(len(ic)),
        "metrics_by_year": metrics_by_year,
        "stability_by_year": stability,
    }
    return metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


PROFILE_ROUND_DIGITS = 6
"""Compact profile precision.  Daily spreads are ~1e-3 and RankIC ~1e-2, so six
decimals keeps clustering correlations stable while keeping the JSON payload a
few tens of kilobytes per factor instead of a full panel."""


def compact_profile(daily: DailySeries) -> dict[str, Any]:
    """Return the compact daily long-short return and RankIC profiles.

    Dates and values are parallel lists on the evaluation calendar; null marks
    a date without an observation.  No factor panel is emitted.
    """

    index = daily.coverage.index
    return {
        "dates": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in index],
        "spread": [round(float(v), PROFILE_ROUND_DIGITS) if pd.notna(v) else None for v in daily.spread.reindex(index)],
        "rank_ic": [
            round(float(v), PROFILE_ROUND_DIGITS) if pd.notna(v) else None for v in daily.rank_ic.reindex(index)
        ],
    }


def run_request(request_path: Path, result_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expression = SafeExpression(str(request["expression"]))
    cfg = EvaluationConfig(**request["evaluation"])
    loader = PanelLoader(request.get("base_path", "G:/base"), cfg.start, cfg.end, int(request.get("lookback_days", 400)))
    factor = expression.evaluate(loader)
    close = loader.load(str(request.get("close_field", "adjclose")))
    daily, factor_window = daily_series(factor, close, cfg)
    metrics = metrics_from_daily(daily, factor_window.index)

    artifact_path = request.get("artifact_path")
    artifact_checksum = None
    if artifact_path:
        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        # Financial operators attach non-tabular PIT event caches to
        # ``DataFrame.attrs``.  Newer pandas/pyarrow versions try to JSON-encode
        # attrs into parquet metadata and fail on the nested DataFrame.  The
        # cache is execution-only provenance, not part of the factor panel.
        artifact_frame = factor_window.loc[_as_timestamp(cfg.start) : _as_timestamp(cfg.end)].copy()
        artifact_frame.attrs.clear()
        artifact_frame.to_parquet(artifact, engine="pyarrow")
        artifact_checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()

    include_profile = bool(request.get("include_profile", True))
    result = _json_safe(
        {
            "name": request["name"],
            "family": request["family"],
            "expression": request["expression"],
            "fields": expression.fields,
            "metrics": metrics,
            "profile": compact_profile(daily) if include_profile else None,
            "artifact_path": str(artifact_path) if artifact_path else None,
            "artifact_checksum": artifact_checksum,
            "evaluation": request["evaluation"],
        }
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a safe GF Quant factor expression on G:/base")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    result = run_request(args.request, args.result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
