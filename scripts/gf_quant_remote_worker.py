"""Lightweight remote evaluator for the GF Quant RD-Agent scenario.

This module is intentionally independent from RD-Agent and Qlib.  It runs next
to ``G:/base`` on the Windows data host, evaluates a small, safe factor DSL,
and writes JSON metrics plus an optional parquet artifact.  The local RD-Agent
controller remains responsible for proposing candidates, feedback, de-dup and
registration.

The financial loader is point-in-time: ``dt`` (announcement date), rather than
``report`` (accounting period), is used to make values available to the panel.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
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
        if name in DIRECT_FIELDS:
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

    def _load_financial(self, table_name: str, factor: str) -> pd.DataFrame:
        table = pq.read_table(
            self.base_path / f"{table_name}.parquet",
            columns=["dt", "report", "symbol", "value"],
            filters=[
                ("factor", "=", factor),
                ("dt", ">=", self.read_start.to_pydatetime()),
                ("dt", "<=", self.end.to_pydatetime()),
            ],
        )
        df = table.to_pandas()
        if df.empty:
            raise ValueError(f"No rows for {table_name}__{factor}")
        df["dt"] = pd.to_datetime(df["dt"]).dt.normalize()
        df["report"] = pd.to_datetime(df["report"]).dt.normalize()
        # On one announcement date retain the newest accounting period.  Then
        # forward-fill from the announcement date only (never from report date).
        df = df.sort_values(["symbol", "dt", "report"]).drop_duplicates(["dt", "symbol"], keep="last")
        events = df.pivot(index="dt", columns="symbol", values="value").sort_index()
        return _finite(events.reindex(self.calendar).ffill().astype(float))


def ts_mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(int(window), min_periods=max(3, int(window) // 2)).mean()


def ts_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(int(window), min_periods=max(3, int(window) // 2)).std()


def delay(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return x.shift(int(periods))


def delta(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return x.diff(int(periods))


def pct_change(x: pd.DataFrame, periods: int) -> pd.DataFrame:
    return x.pct_change(int(periods), fill_method=None)


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True, method="average") - 0.5


def signed_log(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x) * np.log1p(np.abs(x))


def winsorize(x: pd.DataFrame, limit: float = 0.01) -> pd.DataFrame:
    lo = x.quantile(float(limit), axis=1)
    hi = x.quantile(1.0 - float(limit), axis=1)
    return x.clip(lo, hi, axis=0)


ALLOWED_FUNCTIONS: dict[str, Callable[..., pd.DataFrame]] = {
    "cs_rank": cs_rank,
    "delay": delay,
    "delta": delta,
    "pct_change": pct_change,
    "signed_log": signed_log,
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "winsorize": winsorize,
}


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
        self.fields = sorted({node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)} - set(ALLOWED_FUNCTIONS))
        if not self.fields:
            raise ValueError("Expression must reference at least one data field")

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
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
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


def evaluate_factor(factor: pd.DataFrame, close: pd.DataFrame, cfg: EvaluationConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    factor = factor.loc[_as_timestamp(cfg.start) : _as_timestamp(cfg.end)]
    common_columns = factor.columns.intersection(close.columns)
    factor = factor.loc[:, common_columns]
    close = close.reindex(index=factor.index, columns=common_columns)
    coverage = factor.notna().sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
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
    spread_std = float(spread.std(ddof=1)) if len(spread) > 1 else math.nan
    ann_return = float(spread.mean() * 252.0) if not spread.empty else math.nan
    sharpe = float(spread.mean() / spread_std * math.sqrt(252.0)) if spread_std and spread_std > 0 else math.nan
    active = long_mask.astype(np.int8) - short_mask.astype(np.int8)
    turnover = float(active.diff().abs().sum(axis=1).div(active.abs().sum(axis=1).replace(0, np.nan)).mean())

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
        "stability_by_year": stability,
    }
    return metrics, factor


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


def run_request(request_path: Path, result_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expression = SafeExpression(str(request["expression"]))
    cfg = EvaluationConfig(**request["evaluation"])
    loader = PanelLoader(request.get("base_path", "G:/base"), cfg.start, cfg.end, int(request.get("lookback_days", 400)))
    factor = expression.evaluate(loader)
    close = loader.load(str(request.get("close_field", "adjclose")))
    metrics, factor = evaluate_factor(factor, close, cfg)

    artifact_path = request.get("artifact_path")
    artifact_checksum = None
    if artifact_path:
        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        factor.loc[_as_timestamp(cfg.start) : _as_timestamp(cfg.end)].to_parquet(artifact, engine="pyarrow")
        artifact_checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()

    result = _json_safe(
        {
            "name": request["name"],
            "family": request["family"],
            "expression": request["expression"],
            "fields": expression.fields,
            "metrics": metrics,
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
