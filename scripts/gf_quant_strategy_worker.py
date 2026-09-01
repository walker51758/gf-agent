"""Standalone remote portfolio worker for frozen GF factor artifacts.

This file intentionally imports no project package.  It verifies the complete
data-only request, reads only referenced factor panels next to G:/base, builds a
deterministic long-only portfolio and returns a compact checksummed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_set_id(payload: Mapping[str, Any]) -> str:
    factors = sorted(payload["factors"], key=lambda item: str(item["factor_id"]))
    body = {
        "snapshot_id": payload["snapshot_id"],
        "working_set_id": payload["working_set_id"],
        "evaluation_start": payload["evaluation_start"],
        "evaluation_end": payload["evaluation_end"],
        "factors": factors,
    }
    return f"gf_artifact_set_{hashlib.sha256(_canonical_json(body).encode()).hexdigest()[:16]}"


def _validate_request(request: Mapping[str, Any]) -> None:
    if int(request.get("schema_version", 0)) != 1:
        raise ValueError("unsupported strategy request schema_version")
    snapshot_checksum = str(request.get("snapshot_checksum", ""))
    if len(snapshot_checksum) != 64:
        raise ValueError("snapshot_checksum must be a SHA-256 hex digest")
    if request.get("snapshot_id") != f"gf_snapshot_{snapshot_checksum[:16]}":
        raise ValueError("snapshot_id does not match snapshot_checksum")
    if request.get("working_set_snapshot_checksum") != snapshot_checksum:
        raise ValueError("working set provenance does not match snapshot checksum")

    candidate = request.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be an object")
    factor_names = candidate.get("factor_names")
    if not isinstance(factor_names, list) or not 2 <= len(factor_names) <= 8:
        raise ValueError("strategy must reference 2-8 factors")
    if len(factor_names) != len(set(factor_names)):
        raise ValueError("strategy factor names must be distinct")
    composition = candidate.get("composition")
    backtest = candidate.get("backtest")
    if not isinstance(composition, Mapping) or not isinstance(backtest, Mapping):
        raise ValueError("candidate composition and backtest must be objects")
    if int(composition.get("top_n", 0)) < 1:
        raise ValueError("top_n must be positive")
    if composition.get("weighting") not in ("equal", "rank", "proportional"):
        raise ValueError("unsupported weighting")
    cap = composition.get("weight_cap")
    if cap is not None and not 0 < float(cap) <= 1:
        raise ValueError("weight_cap must be in (0, 1]")
    if int(backtest.get("signal_lag", 0)) < 1:
        raise ValueError("signal_lag must be at least 1")
    if float(backtest.get("cost_bps", 0.0)) <= 0:
        raise ValueError("positive transaction cost is required")

    artifact_set = request.get("artifact_set")
    if not isinstance(artifact_set, Mapping):
        raise ValueError("artifact_set must be an object")
    if artifact_set.get("artifact_set_id") != _artifact_set_id(artifact_set):
        raise ValueError("artifact_set_id does not match its contents")
    for key in ("snapshot_id", "working_set_id"):
        if artifact_set.get(key) != request.get(key):
            raise ValueError(f"artifact set {key} does not match request")
    evaluation = request.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation must be an object")
    start, end = str(evaluation.get("start", "")), str(evaluation.get("end", ""))
    if not start or not end or start > end:
        raise ValueError("evaluation window is invalid")
    if artifact_set.get("evaluation_start") > start or artifact_set.get("evaluation_end") < end:
        raise ValueError("artifact set does not cover evaluation window")
    factors = artifact_set.get("factors")
    if not isinstance(factors, list):
        raise ValueError("artifact set factors must be a list")
    artifact_names = [str(item.get("name", "")) for item in factors]
    if set(artifact_names) != set(str(name) for name in factor_names) or len(artifact_names) != len(factor_names):
        raise ValueError("artifact set must contain exactly the candidate factors")
    directions = request.get("factor_directions")
    if not isinstance(directions, Mapping) or set(directions) != set(factor_names):
        raise ValueError("factor directions must match candidate factors exactly")
    if any(int(value) not in (-1, 1) for value in directions.values()):
        raise ValueError("factor directions must be -1 or 1")


def _load_factor(item: Mapping[str, Any], start: str, end: str) -> pd.DataFrame:
    path = Path(str(item["artifact_path"]))
    if not path.exists():
        raise FileNotFoundError(str(path))
    if _sha256(path) != item.get("artifact_checksum"):
        raise ValueError(f"artifact checksum mismatch for {item.get('name')}")
    frame = pd.read_parquet(path)
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"malformed factor panel for {item.get('name')}")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    if frame.index.min() > pd.Timestamp(start) or frame.index.max() < pd.Timestamp(end):
        raise ValueError(f"factor {item.get('name')} does not cover evaluation window")
    result = frame.loc[pd.Timestamp(start) : pd.Timestamp(end)].apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _load_close(base_path: str, field: str, start: str, end: str) -> pd.DataFrame:
    read_start = pd.Timestamp(start) - pd.Timedelta(days=10)
    table = pq.read_table(
        Path(base_path) / f"{field}.parquet",
        columns=["dt", "symbol", "value"],
        filters=[("dt", ">=", read_start.to_pydatetime()), ("dt", "<=", pd.Timestamp(end).to_pydatetime())],
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError("close panel is empty")
    frame["dt"] = pd.to_datetime(frame["dt"]).dt.normalize()
    wide = frame.pivot_table(index="dt", columns="symbol", values="value", aggfunc="last").sort_index()
    return wide.astype(float).replace([np.inf, -np.inf], np.nan)


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    if frequency == "daily":
        return index
    if frequency == "weekly":
        periods = index.to_period("W")
    elif frequency == "monthly":
        periods = index.to_period("M")
    elif frequency == "quarterly":
        periods = index.to_period("Q")
    else:
        raise ValueError("unsupported rebalance_frequency")
    selected = pd.Series(index, index=index).groupby(periods).last().to_list()
    return pd.DatetimeIndex(sorted(set([index[0], *selected])))


def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    result = weights.astype(float).copy()
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        over = ~frozen & (result > cap)
        if not over.any():
            break
        frozen |= over
        excess = float((result - cap)[over].sum())
        result[over] = cap
        free = ~frozen
        room = float(result[free].sum())
        if not free.any() or room <= 0:
            break
        result[free] += excess * result[free] / room
    return result.clip(upper=cap)


def _target_weights(score: pd.DataFrame, composition: Mapping[str, Any]) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    top_n = int(composition["top_n"])
    for date, row in score.iterrows():
        candidates = row.dropna()
        if candidates.empty:
            continue
        ordered = (
            pd.DataFrame({"score": candidates})
            .rename_axis("symbol")
            .reset_index()
            .sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
        )
        selected = ordered.head(top_n).set_index("symbol")["score"]
        scheme = composition["weighting"]
        if scheme == "equal":
            row_weights = pd.Series(1.0 / len(selected), index=selected.index)
        elif scheme == "rank":
            ranks = pd.Series(range(len(selected), 0, -1), index=selected.index, dtype=float)
            row_weights = ranks / ranks.sum()
        else:
            if (selected <= 0).any():
                raise ValueError("proportional weighting requires positive scores")
            row_weights = selected / selected.sum()
        if composition.get("weight_cap") is not None:
            row_weights = _apply_cap(row_weights, float(composition["weight_cap"]))
        weights.loc[date, selected.index] = row_weights.to_numpy()
    return weights


def _max_drawdown(returns: pd.Series) -> float:
    wealth = pd.Series([1.0, *((1.0 + returns).cumprod().to_list())], dtype=float)
    return float((wealth / wealth.cummax() - 1.0).min())


def _metrics(daily: pd.Series, turnover: pd.Series, weights: pd.DataFrame, per_year: int = 252) -> dict[str, Any]:
    mean = float(daily.mean()) if len(daily) else 0.0
    std = float(daily.std()) if len(daily) > 1 else 0.0
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
    terminal = 1.0 + total
    cagr = float(terminal ** (per_year / len(daily)) - 1.0) if len(daily) and terminal > 0 else -1.0
    return {
        "observations": int(len(daily)),
        "total_return": total,
        "annualized_return": mean * per_year,
        "cagr": cagr,
        "annualized_vol": std * per_year**0.5,
        "sharpe": mean / std * per_year**0.5 if std > 0 else 0.0,
        "max_drawdown": _max_drawdown(daily) if len(daily) else 0.0,
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "annual_turnover": float(turnover.mean()) * per_year if len(turnover) else 0.0,
        "avg_holdings": float((weights != 0).sum(axis=1).mean()) if len(weights) else 0.0,
        "win_rate": float((daily > 0).mean()) if len(daily) else 0.0,
    }


def _backtest(targets: pd.DataFrame, returns: pd.DataFrame, backtest: Mapping[str, Any], benchmark: pd.Series):
    lag = int(backtest["signal_lag"])
    cost_rate = float(backtest["cost_bps"]) * 1e-4
    aligned = targets.reindex(columns=returns.columns).fillna(0.0)
    target_by_date = {date: row.to_numpy(float) for date, row in aligned.iterrows()}
    symbols = list(returns.columns)
    carried = pd.Series(0.0, index=symbols)
    start_weights = pd.DataFrame(0.0, index=returns.index, columns=symbols)
    gross = pd.Series(0.0, index=returns.index)
    turnover = pd.Series(0.0, index=returns.index)
    for position, date in enumerate(returns.index):
        source = position - lag
        weights = pd.Series(target_by_date[returns.index[source]], index=symbols) if source >= 0 and returns.index[source] in target_by_date else carried
        day = returns.loc[date].fillna(0.0)
        portfolio_return = float((weights * day).sum())
        cash_before, cash_after = 1.0 - float(carried.sum()), 1.0 - float(weights.sum())
        turnover.loc[date] = 0.5 * (float((weights - carried).abs().sum()) + abs(cash_after - cash_before))
        gross.loc[date] = portfolio_return
        start_weights.loc[date] = weights
        denominator = 1.0 + portfolio_return
        if denominator > 0:
            carried = weights * (1.0 + day) / denominator
    net = gross - turnover * cost_rate
    metrics = _metrics(net, turnover, start_weights)
    excess = net - benchmark
    excess_std = float(excess.std())
    metrics.update(
        {
            "cost_bps": float(backtest["cost_bps"]),
            "signal_lag": lag,
            "excess_annualized_return": float(excess.mean()) * 252,
            "tracking_error": excess_std * 252**0.5,
            "information_ratio": float(excess.mean()) / excess_std * 252**0.5 if excess_std > 0 else 0.0,
        }
    )
    by_year = {}
    for year in sorted(set(net.index.year)):
        mask = net.index.year == year
        yearly = _metrics(net.loc[mask], turnover.loc[mask], start_weights.loc[mask])
        year_excess = net.loc[mask] - benchmark.loc[mask]
        year_std = float(year_excess.std())
        yearly.update(
            {
                "excess_annualized_return": float(year_excess.mean()) * 252,
                "tracking_error": year_std * 252**0.5,
                "information_ratio": float(year_excess.mean()) / year_std * 252**0.5 if year_std > 0 else 0.0,
            }
        )
        by_year[str(year)] = yearly
    return metrics, by_year, net, gross, turnover, start_weights


def _score(ranks: Mapping[str, pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    panels = [ranks[name] for name in names]
    total = panels[0].copy()
    valid = panels[0].notna()
    for panel in panels[1:]:
        total = total.add(panel, fill_value=0.0)
        valid &= panel.notna()
    return (total / len(panels)).where(valid)


def run_request(request_path: Path, result_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _validate_request(request)
    candidate = request["candidate"]
    evaluation = request["evaluation"]
    start, end = str(evaluation["start"]), str(evaluation["end"])
    close = _load_close(str(request.get("base_path", "G:/base")), str(request.get("close_field", "adjclose")), start, end)
    calendar = close.loc[pd.Timestamp(start) : pd.Timestamp(end)].index
    if calendar.empty or calendar.min() > pd.Timestamp(start) or calendar.max() < pd.Timestamp(end):
        raise ValueError("close/benchmark data do not cover evaluation window")
    rebalance_dates = _rebalance_dates(calendar, str(evaluation.get("rebalance_frequency", "monthly")))
    artifact_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    directions = request["factor_directions"]
    ranks: dict[str, pd.DataFrame] = {}
    common_symbols = set(close.columns)
    loaded: dict[str, pd.DataFrame] = {}
    for name in candidate["factor_names"]:
        frame = _load_factor(artifact_items[name], start, end)
        loaded[name] = frame
        common_symbols &= set(frame.columns)
    if not common_symbols:
        raise ValueError("factor and return panels have no common symbols")
    symbols = sorted(common_symbols)
    for name, frame in loaded.items():
        aligned = frame.reindex(index=rebalance_dates, columns=symbols)
        ranks[name] = (int(directions[name]) * aligned).rank(axis=1, pct=True, method="average")

    daily_returns = close.reindex(columns=symbols).pct_change(fill_method=None).loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if daily_returns.index.difference(calendar).size or daily_returns.empty:
        raise ValueError("return panel does not cover evaluation calendar")
    benchmark = daily_returns.mean(axis=1, skipna=True)
    if benchmark.isna().any():
        raise ValueError("benchmark does not cover every evaluation day")

    full_score = _score(ranks, candidate["factor_names"])
    targets = _target_weights(full_score, candidate["composition"])
    full = _backtest(targets, daily_returns, candidate["backtest"], benchmark)
    metrics, metrics_by_year, net, gross, turnover, weights = full

    contributions = {}
    for removed in candidate["factor_names"]:
        kept = [name for name in candidate["factor_names"] if name != removed]
        ablation_targets = _target_weights(_score(ranks, kept), candidate["composition"])
        ablation_metrics = _backtest(ablation_targets, daily_returns, candidate["backtest"], benchmark)[0]
        contributions[removed] = {
            "marginal_annualized_return": metrics["annualized_return"] - ablation_metrics["annualized_return"],
            "marginal_sharpe": metrics["sharpe"] - ablation_metrics["sharpe"],
            "marginal_drawdown_improvement": abs(ablation_metrics["max_drawdown"]) - abs(metrics["max_drawdown"]),
            "marginal_turnover_reduction": ablation_metrics["annual_turnover"] - metrics["annual_turnover"],
        }

    artifact_path = Path(str(request["portfolio_artifact_path"]))
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_frame = weights.copy()
    artifact_frame.attrs.clear()
    artifact_frame.to_parquet(artifact_path, engine="pyarrow")
    artifact_checksum = _sha256(artifact_path)
    request_checksum = hashlib.sha256(_canonical_json(request).encode()).hexdigest()
    result = _json_safe(
        {
            "status": "success",
            "schema_version": 1,
            "name": candidate["name"],
            "snapshot_id": request["snapshot_id"],
            "snapshot_checksum": request["snapshot_checksum"],
            "working_set_id": request["working_set_id"],
            "artifact_set_id": request["artifact_set"]["artifact_set_id"],
            "request_checksum": request_checksum,
            "metrics": metrics,
            "metrics_by_year": metrics_by_year,
            "return_profile": {
                "dates": [date.strftime("%Y-%m-%d") for date in net.index],
                "returns": [round(float(value), 10) for value in net],
            },
            "factor_contributions": contributions,
            "artifact_path": str(artifact_path),
            "artifact_checksum": artifact_checksum,
            "audit": {
                "evaluation": dict(evaluation),
                "score_combination": "equal_weight_cross_sectional_percentile_rank",
                "factor_score_weights_supported": False,
                "benchmark": "daily_equal_weight_investable_universe",
                "factor_count": len(candidate["factor_names"]),
                "symbol_count": len(symbols),
                "rebalance_count": len(rebalance_dates),
                "full_weights_remote_only": True,
            },
        }
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_request(args.request, args.result), ensure_ascii=False))


if __name__ == "__main__":
    main()
