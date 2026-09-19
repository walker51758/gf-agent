"""Standalone remote portfolio worker for frozen GF factor artifacts.

This file intentionally imports no project package.  It verifies the complete
data-only request, reads only referenced factor panels next to G:/base, builds a
deterministic long-only portfolio and returns a compact checksummed result.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

BUY_COST_BPS = 10.0
SELL_COST_BPS = 10.0
ROUND_TRIP_COST_BPS = BUY_COST_BPS + SELL_COST_BPS
BUY_COST_RATE = BUY_COST_BPS * 1e-4
SELL_COST_RATE = SELL_COST_BPS * 1e-4


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


# Key whitelist of candidate["backtest"]; mirrors BacktestSpec.as_payload on
# the local side (only enabled execution-adjustment keys may appear).
BACKTEST_KEYS = frozenset(
    {
        "signal_lag",
        "trading_days_per_year",
        "drift_band_name",
        "drift_band_portfolio",
        "drift_band_forced_days",
        "turnover_shrinkage",
        "turnover_max_step",
        "min_holding_days",
        # Batch 6 order layer / portfolio-risk scalings (roadmap 8.6-8.9,
        # 9.1, 9.2); each key travels only while its operator is enabled.
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
)

#: The three escalatable drawdown states; ``normal`` is the resting state and
#: always scales by 1.0.  Mirrors ``candidate.DRAWDOWN_STATES``.
DRAWDOWN_STATES = ("warning", "reduce", "halt")

EXECUTION_DATA_KEYS = frozenset({"trading_dates", "price", "amount", "lot_sizes", "halt", "price_limits"})

PRICE_LIMIT_SIDES = ("upper", "lower")


def _validate_exposure_map(value: Any, message: str) -> None:
    """Shape check for a ``{name: {symbol: number}}`` injected exposure mapping.

    Mirror of ``candidate._validate_exposure_map``: the neutralization and
    risk-model data panels are explicit configuration inputs (roadmap
    4.6/4.7/7.7); malformed shapes fail closed with the same message the local
    side raises instead of crashing deeper in the pipeline.
    """

    if not isinstance(value, dict) or not value:
        raise ValueError(message)
    for name, row in value.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict) or not row:
            raise ValueError(message)
        for symbol, number in row.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(message)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                raise ValueError(message)


def _backtest_bounded(
    value: Any,
    message: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_open: bool = True,
    upper_open: bool = False,
) -> float:
    """Mirror of ``candidate._bounded_backtest_number`` (batch 6 order layer).

    Booleans are rejected on purpose: ``True`` is not a rate.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(message)
    number = float(value)
    if lower is not None and (number <= lower if lower_open else number < lower):
        raise ValueError(message)
    if upper is not None and (number >= upper if upper_open else number > upper):
        raise ValueError(message)
    return number


def _backtest_drawdown_map(
    value: Any, name: str, *, message: str, upper: float, positive: bool = False
) -> dict[str, float]:
    """Mirror of ``candidate._drawdown_map``: a map over exactly three states.

    ``positive`` selects the ``(0, upper]`` threshold range over the
    ``[0, upper]`` scale range.
    """

    if not isinstance(value, Mapping) or set(value) != set(DRAWDOWN_STATES):
        raise ValueError(f"{name} must cover exactly the drawdown states")
    out: dict[str, float] = {}
    for state in DRAWDOWN_STATES:
        entry = value[state]
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(float(entry)):
            raise ValueError(message)
        number = float(entry)
        lower = 0.0 < number if positive else 0.0 <= number
        if not lower or number > upper:
            raise ValueError(message)
        out[state] = number
    return out


def _validate_execution_data(execution_data: Mapping[str, Any]) -> None:
    """Mirror of ``candidate.ExecutionDataSpec`` shape validation (batch 6).

    The order-layer panels are explicit configuration inputs exactly as
    ``universe_rules`` is, so a malformed shape fails closed here with the
    local side's message instead of crashing deeper in the loop.
    """

    unknown = set(execution_data) - EXECUTION_DATA_KEYS
    if unknown:
        raise ValueError(f"candidate execution_data has unknown keys: {sorted(unknown)[:3]}")
    trading_dates = execution_data.get("trading_dates")
    if trading_dates is not None:
        shape = "execution_data.trading_dates must be a non-empty list of date strings"
        if isinstance(trading_dates, str) or not isinstance(trading_dates, (list, tuple)) or not trading_dates:
            raise ValueError(shape)
        parsed = [_universe_date(day, shape) for day in trading_dates]
        if any(b <= a for a, b in zip(parsed, parsed[1:])):
            raise ValueError("execution_data.trading_dates must be strictly increasing")
    price = execution_data.get("price")
    if price is not None:
        _universe_number_panel(price, "execution_data.price")
    amount = execution_data.get("amount")
    if amount is not None:
        _universe_number_panel(amount, "execution_data.amount")
    lot_sizes = execution_data.get("lot_sizes")
    if lot_sizes is not None:
        shape = "execution_data.lot_sizes must map non-empty symbols to whole positive lot sizes"
        if not isinstance(lot_sizes, dict) or not lot_sizes:
            raise ValueError(shape)
        for symbol, size in lot_sizes.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(shape)
            if (
                isinstance(size, bool)
                or not isinstance(size, (int, float))
                or not math.isfinite(float(size))
                or float(size) != int(size)
                or int(size) < 1
            ):
                raise ValueError(shape)
    halt = execution_data.get("halt")
    if halt is not None:
        _universe_string_panel(halt, "execution_data.halt")
    price_limits = execution_data.get("price_limits")
    if price_limits is not None:
        shape = "execution_data.price_limits must map date strings to objects of symbol -> {upper, lower}"
        if not isinstance(price_limits, dict) or not price_limits:
            raise ValueError(shape)
        for date_key, row in price_limits.items():
            _universe_date(date_key, f"execution_data.price_limits contains an invalid date: {date_key!r}")
            if not isinstance(row, dict) or not row:
                raise ValueError(shape)
            for symbol, bounds in row.items():
                if not isinstance(symbol, str) or not symbol or not isinstance(bounds, Mapping):
                    raise ValueError(shape)
                unknown_sides = {str(key) for key in set(bounds) - set(PRICE_LIMIT_SIDES)}
                if unknown_sides:
                    raise ValueError(f"execution_data.price_limits has unknown keys: {sorted(unknown_sides)[:3]}")
                clean: dict[str, float | None] = {}
                for side in PRICE_LIMIT_SIDES:
                    raw = bounds.get(side)
                    clean[side] = None if raw is None else _universe_number(raw, shape)
                if clean["upper"] is None and clean["lower"] is None:
                    raise ValueError(shape)
                if clean["upper"] is not None and clean["lower"] is not None and clean["lower"] > clean["upper"]:
                    raise ValueError(
                        f"execution_data.price_limits lower must not exceed upper for {symbol} on {date_key}"
                    )


def _validate_order_layer(backtest: Mapping[str, Any]) -> None:
    """Mirror of ``candidate.BacktestSpec._validate_order_layer`` (batch 6).

    Same messages, same acceptance semantics -- whole-valued floats are admitted
    and normalised by the local spec, so they must be admitted here too.  The
    defaults the local side resolves (``participation_unfilled`` carry,
    ``participation_amount_window`` 1, ``lot_rounding_mode`` down,
    ``staged_priority`` weight) are already baked into the payload it ships, so
    this mirror only has to accept what the executors actually read.
    """

    # 8.6 ParticipationCap; mutually exclusive with 8.9 -- one allocator.
    if backtest.get("participation_max_rate") is not None and backtest.get("staged_max_days") is not None:
        raise ValueError("participation_cap and staged_execution are mutually exclusive")
    if backtest.get("participation_max_rate") is not None:
        _backtest_bounded(
            backtest["participation_max_rate"],
            "participation_max_rate must be in (0, 1] when provided",
            lower=0.0,
            upper=1.0,
        )
    if backtest.get("participation_max_days") is not None:
        whole = _universe_whole(
            backtest["participation_max_days"], "participation_max_days must be a whole number when provided"
        )
        if whole < 1:
            raise ValueError("participation_max_days must be at least 1 when provided")
    if backtest.get("participation_amount_window") is not None:
        whole = _universe_whole(
            backtest["participation_amount_window"],
            "participation_amount_window must be a whole number when provided",
        )
        if whole < 1:
            raise ValueError("participation_amount_window must be at least 1 when provided")
    unfilled = backtest.get("participation_unfilled")
    if unfilled is not None and unfilled not in {"cancel", "carry"}:
        raise ValueError("participation_unfilled must be one of ('cancel', 'carry') when provided")
    if backtest.get("participation_max_rate") is None:
        for name in ("participation_max_days", "participation_unfilled", "participation_amount_window"):
            if backtest.get(name) is not None:
                raise ValueError(f"{name} requires participation_max_rate")
    else:
        effective_unfilled = "carry" if unfilled is None else unfilled
        if effective_unfilled == "carry" and backtest.get("participation_max_days") is None:
            raise ValueError("participation_max_days must be at least 1 when participation_unfilled is carry")

    # 8.7 LotRounding.
    lot_rounding = backtest.get("lot_rounding", False)
    if not isinstance(lot_rounding, bool):
        raise ValueError("lot_rounding must be a boolean")
    if backtest.get("lot_min_notional") is not None:
        _backtest_bounded(backtest["lot_min_notional"], "lot_min_notional must be positive when provided", lower=0.0)
    lot_mode = backtest.get("lot_rounding_mode")
    if lot_mode is not None and lot_mode not in {"down", "nearest"}:
        raise ValueError("lot_rounding_mode must be one of ('down', 'nearest') when provided")
    if not lot_rounding:
        for name in ("lot_min_notional", "lot_rounding_mode"):
            if backtest.get(name) is not None:
                raise ValueError(f"{name} requires lot_rounding")

    # 8.8 SideAwareTradability.
    side_mode = backtest.get("side_aware_tradability")
    if side_mode is not None and side_mode not in {"block_buy", "block_sell", "block_both"}:
        raise ValueError("side_aware_tradability must be one of ('block_buy', 'block_sell', 'block_both') when provided")

    # 8.9 StagedExecution.
    if backtest.get("staged_max_days") is not None:
        whole = _universe_whole(backtest["staged_max_days"], "staged_max_days must be a whole number when provided")
        if whole < 1:
            raise ValueError("staged_max_days must be at least 1 when provided")
    staged_priority = backtest.get("staged_priority")
    if staged_priority is not None and staged_priority not in {"score", "weight"}:
        raise ValueError("staged_priority must be one of ('score', 'weight') when provided")
    if backtest.get("staged_max_days") is None and staged_priority is not None:
        raise ValueError("staged_priority requires staged_max_days")

    # 9.1 PortfolioVolatilityTarget.
    if backtest.get("vol_target_annual") is not None:
        _backtest_bounded(backtest["vol_target_annual"], "vol_target_annual must be positive when provided", lower=0.0)
    if backtest.get("vol_target_window") is not None:
        whole = _universe_whole(backtest["vol_target_window"], "vol_target_window must be a whole number when provided")
        if whole < 2:
            raise ValueError("vol_target_window must be at least 2 when provided")
    if backtest.get("vol_target_min_scale") is not None:
        _backtest_bounded(
            backtest["vol_target_min_scale"],
            "vol_target_min_scale must be in (0, 1] when provided",
            lower=0.0,
            upper=1.0,
        )
    if backtest.get("vol_target_max_scale") is not None:
        _backtest_bounded(
            backtest["vol_target_max_scale"],
            "vol_target_max_scale must be at least 1 when provided",
            lower=1.0,
            lower_open=False,
        )
    vol_companions = ("vol_target_window", "vol_target_min_scale", "vol_target_max_scale")
    if backtest.get("vol_target_annual") is None:
        for name in vol_companions:
            if backtest.get(name) is not None:
                raise ValueError(f"{name} requires vol_target_annual")
    else:
        for name in vol_companions:
            if backtest.get(name) is None:
                raise ValueError(f"{name} is required when vol_target_annual is provided")
        if not float(backtest["vol_target_min_scale"]) < 1 <= float(backtest["vol_target_max_scale"]):
            raise ValueError("vol_target_min_scale must be below 1 and vol_target_max_scale at least 1")

    # 9.2 DrawdownStateMachine.
    thresholds = backtest.get("drawdown_thresholds")
    if thresholds is not None:
        clean = _backtest_drawdown_map(
            thresholds,
            "drawdown_thresholds",
            message="drawdown_thresholds must be in (0, 1] when provided",
            upper=1.0,
            positive=True,
        )
        ordered = [clean[state] for state in DRAWDOWN_STATES]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise ValueError("drawdown_thresholds must be strictly increasing")
    if backtest.get("drawdown_scales") is not None:
        _backtest_drawdown_map(
            backtest["drawdown_scales"],
            "drawdown_scales",
            message="drawdown_scales must be in [0, 1] when provided",
            upper=1.0,
        )
    if backtest.get("drawdown_confirm_days") is not None:
        whole = _universe_whole(
            backtest["drawdown_confirm_days"], "drawdown_confirm_days must be a whole number when provided"
        )
        if whole < 1:
            raise ValueError("drawdown_confirm_days must be at least 1 when provided")
    if backtest.get("drawdown_recovery") is not None:
        _backtest_bounded(
            backtest["drawdown_recovery"],
            "drawdown_recovery must be in (0, 1] when provided",
            lower=0.0,
            upper=1.0,
        )
    drawdown_companions = ("drawdown_confirm_days", "drawdown_recovery", "drawdown_scales")
    if thresholds is None:
        for name in drawdown_companions:
            if backtest.get(name) is not None:
                raise ValueError(f"{name} requires drawdown_thresholds")
    else:
        for name in drawdown_companions:
            if backtest.get(name) is None:
                raise ValueError(f"{name} is required when drawdown_thresholds is provided")


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
    if composition.get("weighting") not in (
        "equal",
        "rank",
        "proportional",
        "score_transform",
        "inverse_volatility",
        "hierarchical",
        "risk_parity",
        "factor_risk_budget",
    ):
        raise ValueError("unsupported weighting")
    cap = composition.get("weight_cap")
    if cap is not None and not 0 < float(cap) <= 1:
        raise ValueError("weight_cap must be in (0, 1] when provided")
    floor = composition.get("weight_floor")
    if floor is not None:
        if not 0 < float(floor) <= 1:
            raise ValueError("weight_floor must be in (0, 1] when provided")
        if cap is not None and float(floor) > float(cap):
            raise ValueError("weight_floor must not exceed weight_cap")
        allow_cash = composition.get("weight_floor_allow_cash", True)
        if not isinstance(allow_cash, bool):
            raise ValueError("weight_floor_allow_cash must be a boolean")
    transform = composition.get("score_transform_kind")
    if composition.get("weighting") == "score_transform":
        if transform is None:
            raise ValueError("score_transform_kind is required when weighting is score_transform")
        if transform not in ("linear", "truncated", "sigmoid"):
            raise ValueError(f"score_transform_kind must be one of ('linear', 'truncated', 'sigmoid'), got {transform!r}")
        slope = float(composition.get("score_transform_slope", 1.0))
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError("score_transform_slope must be a finite positive number")
        cut = float(composition.get("score_transform_cut", 0.0))
        if not math.isfinite(cut):
            raise ValueError("score_transform_cut must be finite")
    elif transform is not None:
        raise ValueError("score_transform_kind is only valid with the score_transform weighting")
    buffer_entry = composition.get("holding_buffer_entry")
    if buffer_entry is not None:
        buffer_entry = float(buffer_entry)
        if not math.isfinite(buffer_entry) or not 0 < buffer_entry <= 1:
            raise ValueError("holding_buffer_entry must be in (0, 1] when provided")
    buffer_retain = composition.get("holding_buffer_retain")
    if buffer_retain is not None:
        buffer_retain = float(buffer_retain)
        if not math.isfinite(buffer_retain) or not 0 < buffer_retain <= 1:
            raise ValueError("holding_buffer_retain must be in (0, 1] when provided")
    buffer_extra = composition.get("holding_buffer_max_extra", 0)
    try:
        whole_extra = int(buffer_extra)
    except (TypeError, ValueError) as exc:
        raise ValueError("holding_buffer_max_extra must be a whole number") from exc
    if float(buffer_extra) != whole_extra:
        raise ValueError("holding_buffer_max_extra must be a whole number")
    if whole_extra < 0:
        raise ValueError("holding_buffer_max_extra must not be negative")
    if buffer_entry is None:
        if buffer_retain is not None or whole_extra != 0:
            raise ValueError("holding_buffer_retain and holding_buffer_max_extra require holding_buffer_entry")
    elif buffer_retain is None:
        # Staying is never harder than entering; an unset retention threshold
        # collapses onto the entry threshold (mirrors CompositionSpec).
        buffer_retain = buffer_entry
    elif buffer_retain > buffer_entry:
        raise ValueError("holding_buffer_retain must not exceed holding_buffer_entry")
    max_correlation = composition.get("diversity_max_correlation")
    if max_correlation is not None:
        max_correlation = float(max_correlation)
        if not math.isfinite(max_correlation) or not -1 <= max_correlation <= 1:
            raise ValueError("diversity_max_correlation must be in [-1, 1] when provided")
    industry_cap = composition.get("industry_weight_cap")
    if industry_cap is not None:
        industry_cap = float(industry_cap)
        if not math.isfinite(industry_cap) or not 0 < industry_cap <= 1:
            raise ValueError("industry_weight_cap must be in (0, 1] when provided")
    industry_map = composition.get("industry_map")
    if industry_map is not None:
        if industry_cap is None:
            raise ValueError("industry_map is only valid with industry_weight_cap")
        if not isinstance(industry_map, dict):
            raise ValueError("industry_map must be an object")
        for symbol, label in industry_map.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                raise ValueError("industry_map must map non-empty symbols to non-empty industry labels")
    floor_raw = composition.get("weight_floor")
    if floor_raw is not None and industry_cap is not None and float(floor_raw) > float(industry_cap):
        raise ValueError("weight_floor must not exceed industry_weight_cap")
    cash_residual_max = composition.get("cash_residual_max")
    if cash_residual_max is not None:
        cash_residual_max = float(cash_residual_max)
        if not math.isfinite(cash_residual_max) or not 0 < cash_residual_max <= 1:
            raise ValueError("cash_residual_max must be in (0, 1] when provided")
    inverse_vol_window = composition.get("inverse_vol_window")
    if inverse_vol_window is not None:
        try:
            whole_window = int(inverse_vol_window)
        except (TypeError, ValueError) as exc:
            raise ValueError("inverse_vol_window must be a whole number") from exc
        if float(inverse_vol_window) != whole_window:
            raise ValueError("inverse_vol_window must be a whole number")
        if whole_window < 2:
            raise ValueError("inverse_vol_window must be at least 2")
    inverse_vol_min = composition.get("inverse_vol_min")
    if inverse_vol_min is not None:
        inverse_vol_min = float(inverse_vol_min)
        if not math.isfinite(inverse_vol_min) or inverse_vol_min <= 0:
            raise ValueError("inverse_vol_min must be a positive finite number when provided")
    inverse_vol_multiply = composition.get("inverse_vol_multiply_score")
    if inverse_vol_multiply is not None and not isinstance(inverse_vol_multiply, bool):
        raise ValueError("inverse_vol_multiply_score must be a boolean")
    weighting = composition.get("weighting")
    hierarchical_groups = composition.get("hierarchical_groups")
    if weighting == "hierarchical" and hierarchical_groups is None:
        raise ValueError("hierarchical weighting requires hierarchical_groups")
    if hierarchical_groups is not None:
        if weighting != "hierarchical":
            raise ValueError("hierarchical_groups is only valid with the hierarchical weighting")
        if not isinstance(hierarchical_groups, dict) or not hierarchical_groups:
            raise ValueError("hierarchical_groups must be a non-empty object when provided")
        seen_groups: dict[str, str] = {}
        for group, members in hierarchical_groups.items():
            if not isinstance(group, str) or not group:
                raise ValueError("hierarchical_groups group names must be non-empty strings")
            if not isinstance(members, list) or not members:
                raise ValueError(f"hierarchical group {group!r} must not be empty")
            for member in members:
                if not isinstance(member, str) or not member:
                    raise ValueError("hierarchical_groups members must be non-empty strings")
                if member in seen_groups and seen_groups[member] != group:
                    raise ValueError("hierarchical_groups members must not appear in two groups")
                seen_groups[member] = group
    hierarchical_weights = composition.get("hierarchical_group_weights")
    if hierarchical_weights is not None:
        if hierarchical_groups is None:
            raise ValueError("hierarchical_group_weights is only valid with hierarchical_groups")
        if not isinstance(hierarchical_weights, dict) or set(hierarchical_weights) != set(hierarchical_groups):
            raise ValueError("hierarchical_group_weights must cover exactly the hierarchical_groups keys")
        for value in hierarchical_weights.values():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("hierarchical_group_weights values must be finite positive numbers")
    # Roadmap 7.6 RiskParityWeight / 7.7 FactorRiskBudgetWeight (batch C2).
    if weighting != "risk_parity" and (
        composition.get("risk_parity_budgets") is not None
        or composition.get("risk_parity_window", 60) != 60
        or composition.get("risk_parity_shrinkage_kind", "diagonal") != "diagonal"
        or composition.get("risk_parity_shrinkage", 0.1) != 0.1
        or composition.get("risk_parity_tolerance", 1e-8) != 1e-8
    ):
        raise ValueError("risk_parity parameters require the risk_parity weighting")
    if composition.get("risk_parity_window") is not None:
        risk_window = composition["risk_parity_window"]
        if isinstance(risk_window, bool) or not isinstance(risk_window, (int, float)):
            raise ValueError("risk_parity_window must be a whole number")
        if not math.isfinite(float(risk_window)) or float(risk_window) != int(risk_window):
            raise ValueError("risk_parity_window must be a whole number")
        if int(risk_window) < 2:
            raise ValueError("risk_parity_window must be at least 2")
    risk_budgets = composition.get("risk_parity_budgets")
    if risk_budgets is not None:
        if not isinstance(risk_budgets, dict) or not risk_budgets:
            raise ValueError("risk_parity_budgets must map non-empty symbols to finite positive numbers")
        for symbol, budget in risk_budgets.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("risk_parity_budgets must map non-empty symbols to finite positive numbers")
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) <= 0
            ):
                raise ValueError("risk_parity_budgets must map non-empty symbols to finite positive numbers")
    risk_shrinkage_kind = composition.get("risk_parity_shrinkage_kind", "diagonal")
    if risk_shrinkage_kind not in ("none", "diagonal"):
        raise ValueError(f"risk_parity_shrinkage_kind must be one of ('none', 'diagonal'), got {risk_shrinkage_kind!r}")
    risk_shrinkage = composition.get("risk_parity_shrinkage", 0.1)
    if (
        isinstance(risk_shrinkage, bool)
        or not isinstance(risk_shrinkage, (int, float))
        or not math.isfinite(float(risk_shrinkage))
        or not 0 <= float(risk_shrinkage) < 1
    ):
        raise ValueError("risk_parity_shrinkage must be in [0, 1) when provided")
    risk_tolerance = composition.get("risk_parity_tolerance", 1e-8)
    if (
        isinstance(risk_tolerance, bool)
        or not isinstance(risk_tolerance, (int, float))
        or not math.isfinite(float(risk_tolerance))
        or float(risk_tolerance) <= 0
    ):
        raise ValueError("risk_parity_tolerance must be a finite positive number")
    if weighting == "factor_risk_budget":
        for field, message in (
            ("frb_budgets", "factor_risk_budget weighting requires frb_budgets"),
            ("frb_exposures", "factor_risk_budget weighting requires frb_exposures"),
            ("frb_factor_covariance", "factor_risk_budget weighting requires frb_factor_covariance"),
            ("frb_idiosyncratic", "factor_risk_budget weighting requires frb_idiosyncratic"),
        ):
            if composition.get(field) is None:
                raise ValueError(message)
    elif any(
        composition.get(field) is not None
        for field in (
            "frb_budgets",
            "frb_exposures",
            "frb_factor_covariance",
            "frb_idiosyncratic",
            "frb_idio_max",
        )
    ) or composition.get("frb_ridge", 0.0) != 0.0:
        raise ValueError("factor_risk_budget parameters require the factor_risk_budget weighting")
    frb_budgets = composition.get("frb_budgets")
    if frb_budgets is not None:
        if not isinstance(frb_budgets, dict) or not frb_budgets:
            raise ValueError("frb_budgets must map non-empty factor names to finite positive numbers")
        for factor_name, budget in frb_budgets.items():
            if not isinstance(factor_name, str) or not factor_name:
                raise ValueError("frb_budgets must map non-empty factor names to finite positive numbers")
            if (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) <= 0
            ):
                raise ValueError("frb_budgets must map non-empty factor names to finite positive numbers")
        if set(frb_budgets) != set(str(name) for name in factor_names):
            raise ValueError("frb_budgets must cover exactly the candidate factors")
    frb_exposures = composition.get("frb_exposures")
    if frb_exposures is not None:
        _validate_exposure_map(
            frb_exposures,
            "frb_exposures must map non-empty factor names to non-empty symbol -> finite number objects",
        )
    frb_covariance = composition.get("frb_factor_covariance")
    if frb_covariance is not None:
        _validate_exposure_map(
            frb_covariance,
            "frb_factor_covariance must map non-empty factor names to non-empty factor -> finite number objects",
        )
    frb_idiosyncratic = composition.get("frb_idiosyncratic")
    if frb_idiosyncratic is not None:
        if not isinstance(frb_idiosyncratic, dict) or not frb_idiosyncratic:
            raise ValueError("frb_idiosyncratic must map non-empty symbols to finite non-negative numbers")
        for symbol, variance in frb_idiosyncratic.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("frb_idiosyncratic must map non-empty symbols to finite non-negative numbers")
            if (
                isinstance(variance, bool)
                or not isinstance(variance, (int, float))
                or not math.isfinite(float(variance))
                or float(variance) < 0
            ):
                raise ValueError("frb_idiosyncratic must map non-empty symbols to finite non-negative numbers")
    frb_idio_max = composition.get("frb_idio_max")
    if frb_idio_max is not None and (
        isinstance(frb_idio_max, bool)
        or not isinstance(frb_idio_max, (int, float))
        or not math.isfinite(float(frb_idio_max))
        or float(frb_idio_max) <= 0
    ):
        raise ValueError("frb_idio_max must be a positive finite number when provided")
    frb_ridge = composition.get("frb_ridge", 0.0)
    if (
        isinstance(frb_ridge, bool)
        or not isinstance(frb_ridge, (int, float))
        or not math.isfinite(float(frb_ridge))
        or float(frb_ridge) < 0
    ):
        raise ValueError("frb_ridge must be a finite non-negative number")
    # Roadmap 6.4 IndustryDeviationConstraint (batch C3).
    any_constraint = composition.get("industry_deviation_max") is not None
    if composition.get("industry_deviation_max") is not None:
        deviation_map = composition.get("industry_deviation_max")
        if not isinstance(deviation_map, dict) or not deviation_map:
            raise ValueError("industry_deviation_max must map non-empty industry names to finite positive deviations")
        for industry, deviation in deviation_map.items():
            if not isinstance(industry, str) or not industry:
                raise ValueError("industry_deviation_max must map non-empty industry names to finite positive deviations")
            if (
                isinstance(deviation, bool)
                or not isinstance(deviation, (int, float))
                or not math.isfinite(float(deviation))
                or float(deviation) <= 0
            ):
                raise ValueError("industry_deviation_max must map non-empty industry names to finite positive deviations")
        industry_map = composition.get("industry_deviation_industry_map")
        if industry_map is None:
            raise ValueError("industry deviation requires industry_deviation_industry_map")
        if not isinstance(industry_map, dict) or not industry_map:
            raise ValueError("industry_deviation_industry_map must map non-empty symbols to non-empty industry labels")
        for symbol, label in industry_map.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                raise ValueError("industry_deviation_industry_map must map non-empty symbols to non-empty industry labels")
        exempt = composition.get("industry_deviation_exempt") or ()
        if not isinstance(exempt, (tuple, list)):
            raise ValueError("industry_deviation_exempt must be a list of non-empty industry names")
        for name in exempt:
            if not isinstance(name, str) or not name:
                raise ValueError("industry_deviation_exempt must be a list of non-empty industry names")
            if name not in deviation_map:
                raise ValueError("industry_deviation_exempt industries must appear in industry_deviation_max")
        benchmark = composition.get("industry_deviation_benchmark")
        if benchmark is None:
            raise ValueError("industry deviation requires industry_deviation_benchmark")
        if not isinstance(benchmark, dict) or not benchmark:
            raise ValueError("industry_deviation_benchmark must map non-empty industry names to finite non-negative weights")
        for industry, weight in benchmark.items():
            if not isinstance(industry, str) or not industry:
                raise ValueError("industry_deviation_benchmark must map non-empty industry names to finite non-negative weights")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) < 0
            ):
                raise ValueError("industry_deviation_benchmark must map non-empty industry names to finite non-negative weights")
    elif (
        composition.get("industry_deviation_industry_map") is not None
        or composition.get("industry_deviation_exempt")
        or composition.get("industry_deviation_benchmark") is not None
    ):
        raise ValueError("industry deviation parameters require industry_deviation_max")
    # Roadmap 6.5 StyleExposureConstraint (batch C3).
    any_constraint = any_constraint or composition.get("style_exposure_bounds") is not None
    if composition.get("style_exposure_bounds") is not None:
        style_bounds = composition.get("style_exposure_bounds")
        if not isinstance(style_bounds, dict) or not style_bounds:
            raise ValueError("style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper")
        for style, bounds in style_bounds.items():
            if not isinstance(style, str) or not style:
                raise ValueError("style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError("style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper")
            lower, upper = bounds
            if (
                isinstance(lower, bool)
                or isinstance(upper, bool)
                or not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) > float(upper)
            ):
                raise ValueError("style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper")
        style_matrix = composition.get("style_exposure_matrix")
        if style_matrix is None:
            raise ValueError("style exposure constraint requires style_exposure_matrix")
        _validate_exposure_map(
            style_matrix,
            "style_exposure_matrix must map non-empty style names to non-empty symbol -> finite number objects",
        )
        if not set(style_bounds) <= set(style_matrix):
            raise ValueError("style_exposure_matrix must cover every bounded style")
        style_benchmark = composition.get("style_exposure_benchmark")
        if style_benchmark is None:
            raise ValueError("style exposure constraint requires style_exposure_benchmark")
        if not isinstance(style_benchmark, dict) or not style_benchmark:
            raise ValueError("style_exposure_benchmark must map non-empty style names to finite numbers")
        for style, value in style_benchmark.items():
            if not isinstance(style, str) or not style:
                raise ValueError("style_exposure_benchmark must map non-empty style names to finite numbers")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("style_exposure_benchmark must map non-empty style names to finite numbers")
        if not set(style_bounds) <= set(style_benchmark):
            raise ValueError("style_exposure_benchmark must cover every bounded style")
    elif (
        composition.get("style_exposure_matrix") is not None
        or composition.get("style_exposure_benchmark") is not None
    ):
        raise ValueError("style exposure parameters require style_exposure_bounds")
    # Roadmap 6.6 BetaConstraint (batch C3).
    beta_min = composition.get("beta_constraint_min")
    beta_max = composition.get("beta_constraint_max")
    beta_enabled = beta_min is not None or beta_max is not None
    any_constraint = any_constraint or beta_enabled
    if beta_enabled:
        for field, value, message in (
            ("beta_constraint_min", beta_min, "beta_constraint_min must be a finite number when provided"),
            ("beta_constraint_max", beta_max, "beta_constraint_max must be a finite number when provided"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise ValueError(message)
        if beta_min is not None and beta_max is not None and float(beta_min) > float(beta_max):
            raise ValueError("beta_constraint_min must not exceed beta_constraint_max")
        beta_by_symbol = composition.get("beta_by_symbol")
        if beta_by_symbol is None:
            raise ValueError("beta constraint requires beta_by_symbol")
        if not isinstance(beta_by_symbol, dict) or not beta_by_symbol:
            raise ValueError("beta_by_symbol must map non-empty symbols to finite numbers")
        for symbol, value in beta_by_symbol.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("beta_by_symbol must map non-empty symbols to finite numbers")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("beta_by_symbol must map non-empty symbols to finite numbers")
    elif composition.get("beta_by_symbol") is not None:
        raise ValueError("beta constraint parameters require beta_constraint_min or beta_constraint_max")
    # Roadmap 6.7 FactorExposureConstraint (batch C3).
    any_constraint = any_constraint or composition.get("factor_exposure_bounds") is not None
    if composition.get("factor_exposure_bounds") is not None:
        factor_bounds = composition.get("factor_exposure_bounds")
        if not isinstance(factor_bounds, dict) or not factor_bounds:
            raise ValueError("factor_exposure_bounds must map non-empty factor names to [lower, upper] pairs with lower <= upper")
        for factor, bounds in factor_bounds.items():
            if not isinstance(factor, str) or not factor:
                raise ValueError("factor_exposure_bounds must map non-empty factor names to [lower, upper] pairs with lower <= upper")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError("factor_exposure_bounds must map non-empty factor names to [lower, upper] pairs with lower <= upper")
            lower, upper = bounds
            if (
                isinstance(lower, bool)
                or isinstance(upper, bool)
                or not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) > float(upper)
            ):
                raise ValueError("factor_exposure_bounds must map non-empty factor names to [lower, upper] pairs with lower <= upper")
        factor_matrix = composition.get("factor_exposure_matrix")
        if factor_matrix is None:
            raise ValueError("factor exposure constraint requires factor_exposure_matrix")
        _validate_exposure_map(
            factor_matrix,
            "factor_exposure_matrix must map non-empty factor names to non-empty symbol -> finite number objects",
        )
        if not set(factor_bounds) <= set(factor_matrix):
            raise ValueError("factor_exposure_matrix must cover every bounded factor")
    elif composition.get("factor_exposure_matrix") is not None:
        raise ValueError("factor exposure parameters require factor_exposure_bounds")
    # Roadmap 6.9 ConstrainedWeightProjection knobs (batch C3).
    if not any_constraint and (
        composition.get("projection_max_iter", 512) != 512
        or composition.get("projection_tolerance", 1e-9) != 1e-9
    ):
        raise ValueError("projection parameters require an exposure constraint")
    projection_max_iter = composition.get("projection_max_iter", 512)
    if isinstance(projection_max_iter, bool) or not isinstance(projection_max_iter, (int, float)):
        raise ValueError("projection_max_iter must be a whole number")
    if not math.isfinite(float(projection_max_iter)) or float(projection_max_iter) != int(projection_max_iter):
        raise ValueError("projection_max_iter must be a whole number")
    if int(projection_max_iter) < 1:
        raise ValueError("projection_max_iter must be at least 1")
    projection_tolerance = composition.get("projection_tolerance", 1e-9)
    if (
        isinstance(projection_tolerance, bool)
        or not isinstance(projection_tolerance, (int, float))
        or not math.isfinite(float(projection_tolerance))
        or float(projection_tolerance) <= 0
    ):
        raise ValueError("projection_tolerance must be a finite positive number")
    if int(backtest.get("signal_lag", 0)) < 1:
        raise ValueError("signal_lag must be at least 1")
    if any(key in backtest for key in ("cost_bps", "buy_cost_bps", "sell_cost_bps", "round_trip_cost_bps")):
        raise ValueError("candidate backtest must not contain transaction-cost fields")
    unknown_backtest = set(backtest) - BACKTEST_KEYS
    if unknown_backtest:
        raise ValueError(f"candidate backtest has unknown keys: {sorted(unknown_backtest)[:3]}")
    band_name = backtest.get("drift_band_name")
    if band_name is not None:
        band_name = float(band_name)
        if not math.isfinite(band_name) or not 0 < band_name <= 1:
            raise ValueError("drift_band_name must be in (0, 1] when provided")
    band_portfolio = backtest.get("drift_band_portfolio")
    if band_portfolio is not None:
        band_portfolio = float(band_portfolio)
        if not math.isfinite(band_portfolio) or not 0 < band_portfolio <= 1:
            raise ValueError("drift_band_portfolio must be in (0, 1] when provided")
    forced_days = backtest.get("drift_band_forced_days")
    if forced_days is not None:
        try:
            whole = int(forced_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("drift_band_forced_days must be a whole number when provided") from exc
        if float(forced_days) != whole:
            raise ValueError("drift_band_forced_days must be a whole number when provided")
        if whole < 1:
            raise ValueError("drift_band_forced_days must be at least 1 when provided")
        if band_name is None and band_portfolio is None:
            raise ValueError("drift_band_forced_days requires drift_band_name or drift_band_portfolio")
    shrinkage = backtest.get("turnover_shrinkage")
    if shrinkage is not None:
        shrinkage = float(shrinkage)
        if not math.isfinite(shrinkage) or not 0 <= shrinkage < 1:
            raise ValueError("turnover_shrinkage must be in [0, 1) when provided")
    max_step = backtest.get("turnover_max_step")
    if max_step is not None:
        max_step = float(max_step)
        if not math.isfinite(max_step) or not 0 < max_step <= 1:
            raise ValueError("turnover_max_step must be in (0, 1] when provided")
    min_holding = backtest.get("min_holding_days")
    if min_holding is not None:
        try:
            whole_holding = int(min_holding)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_holding_days must be a whole number when provided") from exc
        if float(min_holding) != whole_holding:
            raise ValueError("min_holding_days must be a whole number when provided")
        if whole_holding < 1:
            raise ValueError("min_holding_days must be at least 1 when provided")
    # Batch 6 order layer and portfolio-risk scalings (roadmap 8.6-8.9, 9.1,
    # 9.2); same messages as the local ``BacktestSpec``.
    _validate_order_layer(backtest)
    signal = candidate.get("signal")
    if signal is not None:
        if not isinstance(signal, Mapping):
            raise ValueError("candidate signal must be an object when provided")
        _validate_signal(signal, [str(name) for name in factor_names])
    gating = candidate.get("gating")
    if gating is not None:
        if not isinstance(gating, Mapping):
            raise ValueError("candidate gating must be an object when provided")
        _validate_gating(gating, [str(name) for name in factor_names])
    # Candidate-pool filters (roadmap 3.1-3.8): the injected panels travel in
    # the canonical payload under ``universe_rules``; the section is absent
    # while every filter is disabled, so the pre-batch identity is unchanged.
    universe_rules = candidate.get("universe_rules")
    if universe_rules is not None:
        if not isinstance(universe_rules, Mapping):
            raise ValueError("candidate universe_rules must be an object when provided")
        _validate_universe_rules(universe_rules)
    # Order-layer data panels (roadmap 8.6-8.9): the same explicit-injection
    # contract as ``universe_rules``.  The section is absent while every
    # order-layer operator is disabled, so pre-batch identity is unchanged.
    execution_data = candidate.get("execution_data")
    if execution_data is not None:
        if not isinstance(execution_data, Mapping):
            raise ValueError("candidate execution_data must be an object when provided")
        _validate_execution_data(execution_data)
    # Result-report audits (roadmap 9.4/10.2/10.3/10.4) travel at the request
    # level, outside the candidate payload: enabling one never changes the
    # strategy identity hash, and audit cost scenarios can never collide with
    # the candidate backtest's cost-key rejection above.
    audit = request.get("audit")
    if audit is not None:
        _validate_audit(audit)
    # WalkForwardEvaluation (roadmap 10.1) rides at the request level *next to*
    # ``audit``: it re-runs the engine once per segment, which is not what the
    # report-only audit object promises.  Either way it stays out of the
    # candidate payload, so enabling it cannot move a strategy identity hash.
    walk_forward = request.get("walk_forward")
    if walk_forward is not None:
        _validate_walk_forward(walk_forward)
    # StrategyManifestEmitter (roadmap 10.8) is likewise request-level and
    # likewise outside the candidate payload: the object carries no settings at
    # all (MANIFEST_KEYS is empty), it only asks the worker to attest which
    # code, environment and data window produced this result.
    manifest = request.get("manifest")
    if manifest is not None:
        _validate_manifest(manifest)
    execution_costs = request.get("execution_costs")
    if not isinstance(execution_costs, Mapping) or set(execution_costs) != {"buy_bps", "sell_bps"}:
        raise ValueError("execution_costs must contain exactly buy_bps and sell_bps")
    try:
        buy_bps = float(execution_costs["buy_bps"])
        sell_bps = float(execution_costs["sell_bps"])
    except (TypeError, ValueError) as exc:
        raise ValueError("execution_costs must be numeric") from exc
    if buy_bps != BUY_COST_BPS or sell_bps != SELL_COST_BPS:
        raise ValueError("execution_costs must match the fixed 10/10 bps system contract")

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


def _apply_weight_floor(weights: pd.Series, floor: float) -> pd.Series:
    """SingleNameFloor (roadmap 6.2): raise held names below ``floor`` up to it.

    Applied *after* the cap: the floor never exceeds the cap (request-level
    validation), so the result stays inside the cap.  Only held names
    (weight > 0) participate -- unselected names stay at exactly zero and are
    never pulled into the book.  The raise is funded proportionally by the
    excess above the floor of the names that can spare it; donors are scaled
    down towards (never below) the floor, so the floor is never violated and
    a cap is never re-breached.  When even the frozen floors do not fit the
    budget the specification is infeasible and fails closed instead of
    shrinking anyone below the floor.  A remaining under-investment is left
    as cash (the same residual convention as the cap), which
    ``weight_floor_allow_cash=False`` rejects at the call site.
    """

    result = weights.astype(float).copy()
    held = result > 0.0
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        below = held & ~frozen & (result < floor)
        if not below.any():
            break
        frozen |= below
        deficit = float((floor - result[below]).sum())
        result[below] = floor
        donors = held & ~frozen
        excess = float((result[donors] - floor).clip(lower=0.0).sum())
        if not donors.any() or excess <= 0.0:
            break
        ratio = min(1.0, deficit / excess)
        result[donors] = floor + (result[donors] - floor) * (1.0 - ratio)
        if deficit <= excess:
            break
    if float(result.sum()) > 1.0 + 1e-9:
        raise ValueError("weight_floor is infeasible: the floors of the held names exceed the budget")
    return result


_UNKNOWN_INDUSTRY = "__no_industry__"


def _inverse_volatility_weights(
    scores: pd.Series, returns_history: pd.DataFrame | None, composition: Mapping[str, Any]
) -> pd.Series:
    """InverseVolatilityWeight (roadmap 7.5): weights proportional to 1/vol.

    Mirror image of ``composition._inverse_volatility_weights``.  The
    volatility of every selected name is the standard deviation of its
    realised daily returns over the last ``inverse_vol_window`` rows of the
    history passed in (realised returns up to and including the decision
    date).  A panel with fewer than two rows falls back to equal weighting
    (declared conservative fallback for the first decision dates); with a
    usable panel, a selected name with a single valid observation or a
    non-positive volatility fails closed with the same message as the local
    side; ``inverse_vol_min`` floors positive-but-tiny estimates.
    """

    if returns_history is None:
        raise ValueError("inverse_volatility weighting requires a returns panel")
    if len(returns_history) < 2:
        return pd.Series(1.0 / len(scores), index=scores.index)
    window = returns_history.tail(int(composition.get("inverse_vol_window", 20)))
    vols: dict[str, float] = {}
    for symbol in scores.index:
        series = window[symbol].dropna() if symbol in window.columns else pd.Series(dtype=float)
        vol = float(series.std()) if len(series) >= 2 else float("nan")
        if not math.isfinite(vol) or vol <= 0.0:
            raise ValueError(
                "inverse_volatility weighting requires a positive volatility estimate for every selected name"
            )
        floor_value = composition.get("inverse_vol_min")
        if floor_value is not None:
            vol = max(vol, float(floor_value))
        vols[symbol] = vol
    if composition.get("inverse_vol_multiply_score", False):
        raw = pd.Series({symbol: float(scores[symbol]) / vols[symbol] for symbol in scores.index})
    else:
        raw = pd.Series({symbol: 1.0 / vols[symbol] for symbol in scores.index})
    if float(raw.sum()) <= 0.0:
        raise ValueError("inverse_volatility weighting requires a positive combined weight")
    return raw / raw.sum()


def _hierarchical_weights(scores: pd.Series, composition: Mapping[str, Any]) -> pd.Series:
    """HierarchicalWeighting (roadmap 7.8): group weight first, then inside.

    Mirror image of ``composition._hierarchical_weights``.  Groups without
    selected names receive nothing; inter-group weights are equal by default
    or the explicit normalised ``hierarchical_group_weights``; inside a group
    the weight is split equally.  A selected name outside every group fails
    closed with the same message as the local side.
    """

    membership: dict[str, str] = {}
    for group, members in (composition.get("hierarchical_groups") or {}).items():
        for symbol in members:
            membership[symbol] = group
    missing = sorted(set(scores.index).difference(membership))
    if missing:
        raise ValueError(f"hierarchical weighting requires every selected name to belong to a group: {missing[:3]}")
    active = sorted({membership[symbol] for symbol in scores.index})
    group_weights = composition.get("hierarchical_group_weights")
    if group_weights is None:
        share = {group: 1.0 / len(active) for group in active}
    else:
        total = sum(float(group_weights[group]) for group in active)
        share = {group: float(group_weights[group]) / total for group in active}
    result = pd.Series(0.0, index=scores.index, dtype=float)
    for group in active:
        members = [symbol for symbol in scores.index if membership[symbol] == group]
        result[members] = share[group] / len(members)
    return result


RISK_PARITY_MAX_ITER = 512


def _risk_parity_ccd(covariance: np.ndarray, budgets: np.ndarray, tolerance: float, label: str = "risk_parity") -> np.ndarray:
    """Cyclical coordinate descent for the risk-budgeting problem.

    Solves ``min 0.5 x' S x - sum(b_i * ln(x_i))`` over ``x > 0``; at the
    optimum ``x_i * (S x)_i`` equals the budget ``b_i`` (up to the sweep
    tolerance), which is exactly the risk-contribution condition.  The
    iteration is deterministic (fixed sweep order, fixed start) so both engine
    sides produce bit-identical weights; an unconverged solve fails closed.
    """

    count = len(budgets)
    x = (budgets / budgets.sum()).astype(float)
    for _ in range(RISK_PARITY_MAX_ITER):
        max_step = 0.0
        for position in range(count):
            others = float(covariance[position] @ x) - float(covariance[position, position]) * float(x[position])
            diagonal = float(covariance[position, position])
            updated = (-others + math.sqrt(others * others + 4.0 * diagonal * float(budgets[position]))) / (
                2.0 * diagonal
            )
            max_step = max(max_step, abs(updated - float(x[position])))
            x[position] = updated
        if max_step <= tolerance:
            return x
    raise ValueError(f"{label} weighting did not converge within {RISK_PARITY_MAX_ITER} iterations")


def _risk_parity_weights(scores: pd.Series, returns_history: pd.DataFrame | None, composition: Mapping[str, Any]) -> pd.Series:
    """RiskParityWeight (roadmap 7.6): equal risk contributions, or the
    explicit ``risk_parity_budgets`` when provided.

    Mirror image of ``composition._risk_parity_weights``.  The covariance of
    the selected names is the sample covariance (ddof=1) of the realised
    daily returns up to and including the decision date over the last
    ``risk_parity_window`` rows (the window is an upper bound, mirroring
    ``inverse_volatility``), shrunk towards its own diagonal by
    ``risk_parity_shrinkage`` when ``risk_parity_shrinkage_kind="diagonal"``.
    A history with fewer than two rows cannot define any covariance, so the
    row falls back to equal weighting (the declared first-decision-date
    fallback).  A selected name without two valid observations, a
    non-positive variance, or a still non-positive-definite covariance fails
    closed with the same message as the local side.
    """

    if returns_history is None:
        raise ValueError("risk_parity weighting requires a returns panel")
    if len(returns_history) < 2:
        return pd.Series(1.0 / len(scores), index=scores.index)
    window = returns_history.tail(int(composition.get("risk_parity_window", 60)))
    selected = list(scores.index)
    # Per-name variance gates on the pandas std (bit-for-bit the
    # inverse-volatility convention); np.cov alone leaves a tiny positive
    # residue for a constant column, so the diagonal check cannot catch it.
    for symbol in selected:
        series = window[symbol].dropna() if symbol in window.columns else pd.Series(dtype=float)
        variance = float(series.var()) if len(series) >= 2 else float("nan")
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                "risk_parity weighting requires a positive variance estimate for every selected name"
            )
    observations = window.reindex(columns=selected).dropna()
    if len(observations) < 2:
        raise ValueError(
            "risk_parity weighting requires a positive variance estimate for every selected name"
        )
    covariance = np.cov(observations.to_numpy(dtype=float), rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    if not bool(np.all(np.isfinite(covariance))):
        raise ValueError("risk_parity weighting requires a positive variance estimate for every selected name")
    shrinkage_kind = composition.get("risk_parity_shrinkage_kind", "diagonal")
    shrinkage = float(composition.get("risk_parity_shrinkage", 0.1))
    if shrinkage_kind == "diagonal" and shrinkage > 0.0:
        covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("risk_parity weighting requires a positive definite covariance estimate") from exc
    budgets_config = composition.get("risk_parity_budgets")
    if budgets_config is not None:
        missing = [symbol for symbol in selected if symbol not in budgets_config]
        if missing:
            raise ValueError(f"risk_parity_budgets must cover every selected name: {missing[:3]}")
        budgets = np.array([float(budgets_config[symbol]) for symbol in selected], dtype=float)
    else:
        budgets = np.ones(len(selected), dtype=float)
    tolerance = float(composition.get("risk_parity_tolerance", 1e-8))
    solved = _risk_parity_ccd(covariance, budgets, tolerance)
    weights = pd.Series(solved / solved.sum(), index=selected)
    if not bool(np.all(np.isfinite(weights.to_numpy(dtype=float)))) or float(weights.sum()) <= 0.0:
        raise ValueError("risk_parity weighting requires a positive combined weight")
    return weights


def _factor_risk_budget_weights(scores: pd.Series, composition: Mapping[str, Any]) -> pd.Series:
    """FactorRiskBudgetWeight (roadmap 7.7): factor risk budgets mapped onto
    the selected names through the injected exposure matrix.

    Mirror image of ``composition._factor_risk_budget_weights``.  Stage one
    solves the risk-budgeting problem in *factor* space: the factor covariance
    (ridged by ``frb_ridge``) and the ``frb_budgets`` feed the same
    deterministic CCD solver, yielding target factor exposures whose risk
    contributions match the budgets.  Stage two maps the target exposures onto
    the selected names: every name's raw weight is its non-negative projection
    ``relu(B @ e)`` onto the target factor mix; if every selected name
    projects non-positive the date fails closed.  Declared simplifications
    versus a full risk model: idiosyncratic risk does not enter the budgeting
    objective (it is bounded only by the ``frb_idio_max`` exclusion cap) and
    the exposure mapping replaces a constrained quadratic program.  Selected
    names without an exposure or an idiosyncratic variance fail closed, as
    does a non-positive-definite (ridged) factor covariance.
    """

    frb_budgets = composition.get("frb_budgets") or {}
    frb_exposures = composition.get("frb_exposures") or {}
    frb_covariance = composition.get("frb_factor_covariance") or {}
    frb_idiosyncratic = composition.get("frb_idiosyncratic") or {}
    frb_idio_max = composition.get("frb_idio_max")
    frb_ridge = float(composition.get("frb_ridge", 0.0))
    factor_names = list(frb_budgets)
    selected = list(scores.index)
    missing_budgets = [name for name in factor_names if name not in frb_exposures]
    if missing_budgets:
        raise ValueError("frb_exposures must cover every budgeted factor")
    exposure_matrix = np.column_stack(
        [
            [float(frb_exposures[name].get(symbol, float("nan"))) for symbol in selected]
            for name in factor_names
        ]
    )
    missing_exposure = [symbol for position, symbol in enumerate(selected) if not np.all(np.isfinite(exposure_matrix[position]))]
    if missing_exposure:
        raise ValueError("factor_risk_budget weighting requires an exposure for every selected name and factor")
    covariance = np.array(
        [[float(frb_covariance[row][column]) for column in factor_names] for row in factor_names],
        dtype=float,
    )
    if frb_ridge > 0.0:
        covariance = covariance + frb_ridge * np.eye(len(factor_names))
    if not bool(np.all(np.isfinite(covariance))):
        raise ValueError("factor_risk_budget weighting requires a finite factor covariance")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("factor_risk_budget weighting requires a positive definite factor covariance") from exc
    budgets = np.array([float(frb_budgets[name]) for name in factor_names], dtype=float)
    target = _risk_parity_ccd(covariance, budgets, 1e-8, "factor_risk_budget")
    # Non-negative projection (the declared relu): negative exposures are a
    # legal input and a negative projected exposure cannot fund a long-only
    # book, so it is floored to zero weight before normalisation.
    projected = np.maximum(exposure_matrix @ target, 0.0)
    for symbol in selected:
        if symbol not in frb_idiosyncratic:
            raise ValueError("factor_risk_budget weighting requires an idiosyncratic variance for every selected name")
    if frb_idio_max is not None:
        for position, symbol in enumerate(selected):
            if float(frb_idiosyncratic[symbol]) > float(frb_idio_max):
                projected[position] = 0.0
    if not bool(np.all(np.isfinite(projected))):
        raise ValueError("factor_risk_budget weighting requires finite projected exposures")
    if not bool((projected > 0.0).any()):
        raise ValueError("factor_risk_budget weighting requires at least one positive projected exposure")
    weights = pd.Series(projected / projected.sum(), index=selected)
    return weights


_NO_INDUSTRY_PROJECTION = "__no_industry__"

#: Composition payload keys consumed by the exposure projection stage
#: (roadmap 6.4/6.5/6.6/6.7/6.9) -- mirror of ``composition._PROJECTION_PARAM_KEYS``.
_PROJECTION_PARAM_KEYS = (
    "industry_deviation_max",
    "industry_deviation_industry_map",
    "industry_deviation_exempt",
    "industry_deviation_benchmark",
    "style_exposure_bounds",
    "style_exposure_matrix",
    "style_exposure_benchmark",
    "beta_constraint_min",
    "beta_constraint_max",
    "beta_by_symbol",
    "factor_exposure_bounds",
    "factor_exposure_matrix",
    "projection_max_iter",
    "projection_tolerance",
)

#: Ceiling on the float-noise allowance of the feasibility check (roadmap 6.9
#: "无解处理"); mirror of ``composition._PROJECTION_FEASIBILITY_CEILING``.
#: ``projection_tolerance`` is only a stopping-accuracy knob, so it must not
#: widen the feasibility check: an early-stopped solve that misses the
#: feasible set fails closed instead of silently returning a book that breaks
#: a box constraint.
_PROJECTION_FEASIBILITY_CEILING = 1e-7


def _has_projection_constraints(composition: Mapping[str, Any]) -> bool:
    return any(
        composition.get(key) is not None
        for key in (
            "industry_deviation_max",
            "style_exposure_bounds",
            "beta_constraint_min",
            "beta_constraint_max",
            "factor_exposure_bounds",
        )
    )


def _projection_params_from_payload(composition: Mapping[str, Any]) -> dict:
    params: dict = {}
    for key in _PROJECTION_PARAM_KEYS:
        value = composition.get(key)
        if value is not None:
            params[key] = value
    # The single-name cap and the held-name floor are pre-C3 chain stages, but
    # they enter the projection as box constraints (mirror of
    # ``composition._projection_params``): without them the projector could
    # push a name back over the cap or under the floor it just enforced.
    for key in ("weight_cap", "weight_floor"):
        value = composition.get(key)
        if value is not None:
            params[key] = value
    params.setdefault("industry_deviation_exempt", ())
    params.setdefault("projection_max_iter", 512)
    params.setdefault("projection_tolerance", 1e-9)
    return params


def _projection_halfspace_slack(x: np.ndarray, a: np.ndarray, b: float) -> float:
    return float(a @ x) - float(b)


def _apply_constrained_projection(values: np.ndarray, symbols: Sequence[str], params: Mapping) -> np.ndarray:
    """Project the post-chain target weights onto the enabled linear
    constraint set (roadmap 6.4/6.5/6.6/6.7 jointly; the projector itself is
    roadmap 6.9 ConstrainedWeightProjection).

    Mirror image of ``composition._apply_constrained_projection``.  Declared
    and locked: distance = equal-weight L2 in weight space; every constraint
    is hard (the roadmap's priority / soft-penalty knobs are deliberately
    absent -- they require a penalised objective, which would break the
    "closest feasible weights" semantics); the solver is Dykstra's
    alternating projection over the fixed constraint order below, which is
    deterministic, dependency-free and version-stable; infeasibility and
    non-convergence fail closed (the roadmap 6.6 "降级策略" reading).
    ``projection_tolerance`` only sets the stopping accuracy: the feasibility
    check uses a fixed float-noise ceiling, so a widened tolerance can turn a
    feasible run into a conservative failure but can never let an infeasible
    point through.  The
    single-name cap and the floor of the already-held names participate as
    box constraints, so the projection can never silently break the earlier
    stages of the chain; non-negativity and the full-investment budget
    (sum <= 1, cash keeps the residual) are always part of the set.
    A non-finite input row fails closed: NaN and +/-inf are upstream bugs, not
    weights, and projecting one would hand the caller a NaN book.
    """

    count = len(values)
    x = np.array(values, dtype=float, copy=True)
    if not bool(np.isfinite(x).all()):
        raise ValueError("constrained weight projection input must contain only finite weights")
    constraints: list[tuple] = []
    cap = params.get("weight_cap")
    if cap is not None:
        constraints.append(("upper", np.full(count, float(cap))))
    floor = params.get("weight_floor")
    if floor is not None:
        constraints.append(("lower", np.where(x > 0.0, float(floor), 0.0)))
    constraints.append(("lower", np.zeros(count)))
    constraints.append(("halfspace", np.ones(count), 1.0))
    deviation_map = params.get("industry_deviation_max")
    if deviation_map:
        industry_map = params.get("industry_deviation_industry_map") or {}
        exempt = set(params.get("industry_deviation_exempt") or ())
        benchmark = params.get("industry_deviation_benchmark") or {}
        groups: dict[str, list[int]] = {}
        for position, symbol in enumerate(symbols):
            groups.setdefault(industry_map.get(symbol) or _NO_INDUSTRY_PROJECTION, []).append(position)
        for industry in sorted(groups):
            if industry in exempt or industry not in deviation_map:
                continue
            rows = groups[industry]
            a = np.zeros(count)
            a[rows] = 1.0
            bench = float(benchmark.get(industry, 0.0))
            deviation = float(deviation_map[industry])
            constraints.append(("halfspace", a, bench + deviation))
            constraints.append(("halfspace", -a, deviation - bench))
    style_bounds = params.get("style_exposure_bounds")
    if style_bounds:
        matrix = params.get("style_exposure_matrix") or {}
        benchmark = params.get("style_exposure_benchmark") or {}
        for style in sorted(style_bounds):
            lower, upper = style_bounds[style]
            column = []
            for symbol in symbols:
                number = matrix.get(style, {}).get(symbol)
                if number is None:
                    raise ValueError("style exposure constraint requires an exposure for every selected name and style")
                column.append(float(number))
            a = np.array(column)
            bench = float(benchmark[style])
            constraints.append(("halfspace", a, float(upper) + bench))
            constraints.append(("halfspace", -a, bench - float(lower)))
    if params.get("beta_constraint_min") is not None or params.get("beta_constraint_max") is not None:
        betas = params.get("beta_by_symbol") or {}
        column = []
        for symbol in symbols:
            if symbol not in betas:
                raise ValueError("beta constraint requires a beta for every selected name")
            column.append(float(betas[symbol]))
        a = np.array(column)
        if params.get("beta_constraint_max") is not None:
            constraints.append(("halfspace", a, float(params["beta_constraint_max"])))
        if params.get("beta_constraint_min") is not None:
            constraints.append(("halfspace", -a, -float(params["beta_constraint_min"])))
    factor_bounds = params.get("factor_exposure_bounds")
    if factor_bounds:
        matrix = params.get("factor_exposure_matrix") or {}
        for factor in sorted(factor_bounds):
            lower, upper = factor_bounds[factor]
            column = []
            for symbol in symbols:
                number = matrix.get(factor, {}).get(symbol)
                if number is None:
                    raise ValueError("factor exposure constraint requires an exposure for every selected name and factor")
                column.append(float(number))
            a = np.array(column)
            constraints.append(("halfspace", a, float(upper)))
            constraints.append(("halfspace", -a, -float(lower)))
    max_iter = int(params.get("projection_max_iter", 512))
    tolerance = float(params.get("projection_tolerance", 1e-9))
    corrections = [np.zeros(count) for _ in constraints]
    converged = False
    for _ in range(max_iter):
        previous = x.copy()
        for position, constraint in enumerate(constraints):
            y = x + corrections[position]
            if constraint[0] == "halfspace":
                _, a, b = constraint
                slack = _projection_halfspace_slack(y, a, b)
                if slack > 0.0:
                    y = y - (slack / float(a @ a)) * a
            elif constraint[0] == "upper":
                y = np.minimum(y, constraint[1])
            else:
                y = np.maximum(y, constraint[1])
            corrections[position] = x + corrections[position] - y
            x = y
        if float(np.max(np.abs(x - previous))) <= tolerance:
            converged = True
            break
    # The projection only guarantees ~`tolerance` accuracy, so the feasibility
    # check must tolerate the accumulated float noise of a convergent solve
    # (observed ~1.5e-9 on the budget boundary with tolerance=1e-9) while
    # staying far below any real violation.  The allowance is therefore capped
    # by a fixed ceiling: `projection_tolerance` is only a stopping-accuracy
    # (performance) knob, so widening it must never widen the feasibility
    # check -- an early-stopped solve that misses the feasible set has to fail
    # closed rather than silently return a point that breaks a box constraint.
    feasibility_tolerance = min(max(16.0 * tolerance, 1e-8), _PROJECTION_FEASIBILITY_CEILING)
    feasible = True
    for constraint in constraints:
        if constraint[0] == "halfspace":
            _, a, b = constraint
            if _projection_halfspace_slack(x, a, b) > feasibility_tolerance:
                feasible = False
                break
        elif constraint[0] == "upper":
            if float(np.max(x - constraint[1])) > feasibility_tolerance:
                feasible = False
                break
        else:
            if float(np.max(constraint[1] - x)) > feasibility_tolerance:
                feasible = False
                break
    if not feasible:
        if converged:
            raise ValueError("constrained weight projection did not reach a feasible point")
        raise ValueError(f"constrained weight projection did not converge within {max_iter} iterations")
    return x


def _apply_industry_cap(
    weights: pd.Series,
    cap: float,
    industry_map: Mapping[str, str],
    name_cap: float | None = None,
) -> pd.Series:
    """IndustryWeightCap (roadmap 6.3): cap each industry's weight sum.

    Verbatim mirror of ``composition._apply_industry_cap``; applied after the
    single-name cap and before the floor.  Every round freezes the names of
    each over-cap industry (scaled down proportionally so the group sits
    exactly on the cap) and hands the freed excess to the remaining names,
    proportional to their remaining single-name-cap headroom when
    ``name_cap`` is set (so no name is pushed back above it) and to their
    weights otherwise.  Unmapped names share the reserved ``_UNKNOWN_INDUSTRY``
    group and the cap applies to it like to any other industry.  Whatever
    cannot be placed stays as uninvested cash.
    """

    result = weights.astype(float).copy()
    labels = pd.Series(
        [industry_map.get(symbol) or _UNKNOWN_INDUSTRY for symbol in result.index], index=result.index
    )
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        free = ~frozen
        if not free.any():
            break
        sums = result[free].groupby(labels[free]).sum()
        over = [group for group in sums.index if sums[group] > cap]
        if not over:
            break
        excess = 0.0
        for group in over:  # groupby keys are sorted, so the order is deterministic
            members = free & (labels == group)
            frozen |= members
            group_sum = float(result[members].sum())
            excess += group_sum - cap
            result[members] = result[members] * (cap / group_sum)
        free = ~frozen
        if not free.any():
            break
        if name_cap is not None:
            headroom = (float(name_cap) - result[free]).clip(lower=0.0)
            budget = float(headroom.sum())
            if budget <= 0.0:
                break
            ratio = min(1.0, excess / budget)
            result[free] = result[free] + ratio * headroom
        else:
            room = float(result[free].sum())
            if room <= 0.0:
                break
            result[free] = result[free] + excess * result[free] / room
    return result


def _score_transform_weights(scores: pd.Series, composition: Mapping[str, Any]) -> pd.Series:
    """ScoreTransformWeight (roadmap 7.4): transform scores, allocate
    proportionally.

    ``linear`` scales by the slope, ``truncated`` zeroes every score at or
    below the cut point (those names keep a zero weight and are not held)
    and ``sigmoid`` maps through the logistic with the cut as its midpoint.
    A non-positive transformed score cannot be held long-only, so linear and
    sigmoid fail closed on it; truncated fails closed only when *every*
    selected score is truncated away on that date.
    """

    kind = str(composition.get("score_transform_kind"))
    slope = float(composition.get("score_transform_slope", 1.0))
    cut = float(composition.get("score_transform_cut", 0.0))
    if kind == "linear":
        transformed = slope * scores
        if not bool((transformed > 0).all()):
            raise ValueError("score_transform weighting requires strictly positive transformed scores")
    elif kind == "truncated":
        transformed = (scores - cut).clip(lower=0.0)
        if not bool((transformed > 0).any()):
            raise ValueError("score_transform truncated weighting requires at least one score above the cut")
    else:  # sigmoid; request validation leaves no other kind
        with np.errstate(over="ignore"):
            transformed = 1.0 / (1.0 + np.exp(-slope * (scores - cut)))
        if not bool((transformed > 0).all()):
            raise ValueError("score_transform weighting requires strictly positive transformed scores")
    return transformed / transformed.sum()


def _target_weights(
    score: pd.DataFrame, composition: Mapping[str, Any], returns: pd.DataFrame | None = None
) -> pd.DataFrame:
    # Mirror of the ``composition.build_target_weights`` guard: an infinite
    # score is rejected instead of ranked (NaN still excludes a name).
    numeric_score = score.apply(pd.to_numeric, errors="coerce")
    if bool(np.isinf(numeric_score.to_numpy(dtype=float)).any()):
        raise ValueError("score panel must not contain infinite values")
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
        elif scheme == "score_transform":
            row_weights = _score_transform_weights(selected, composition)
        elif scheme == "inverse_volatility":
            history = None if returns is None else returns.loc[returns.index <= date]
            row_weights = _inverse_volatility_weights(selected, history, composition)
        elif scheme == "hierarchical":
            row_weights = _hierarchical_weights(selected, composition)
        elif scheme == "risk_parity":
            history = None if returns is None else returns.loc[returns.index <= date]
            row_weights = _risk_parity_weights(selected, history, composition)
        elif scheme == "factor_risk_budget":
            row_weights = _factor_risk_budget_weights(selected, composition)
        else:
            if (selected <= 0).any():
                raise ValueError("proportional weighting requires strictly positive scores")
            row_weights = selected / selected.sum()
        if composition.get("weight_cap") is not None:
            row_weights = _apply_cap(row_weights, float(composition["weight_cap"]))
        if composition.get("industry_weight_cap") is not None:
            row_weights = _apply_industry_cap(
                row_weights,
                float(composition["industry_weight_cap"]),
                composition.get("industry_map") or {},
                name_cap=(float(composition["weight_cap"]) if composition.get("weight_cap") is not None else None),
            )
        if composition.get("weight_floor") is not None:
            row_weights = _apply_weight_floor(row_weights, float(composition["weight_floor"]))
            if not composition.get("weight_floor_allow_cash", True) and float(row_weights.sum()) < 1.0 - 1e-9:
                raise ValueError("weight_floor forbids a cash residual but the book is not fully invested")
        if _has_projection_constraints(composition):
            row_weights = pd.Series(
                _apply_constrained_projection(
                    row_weights.to_numpy(dtype=float),
                    list(row_weights.index),
                    _projection_params_from_payload(composition),
                ),
                index=row_weights.index,
            )
        if composition.get("cash_residual_max") is not None and 1.0 - float(row_weights.sum()) > float(
            composition["cash_residual_max"]
        ) + 1e-9:
            raise ValueError("cash residual exceeds cash_residual_max")
        weights.loc[date, selected.index] = row_weights.to_numpy()
    return weights


def _max_drawdown(returns: pd.Series) -> float:
    wealth = pd.Series([1.0, *((1.0 + returns).cumprod().to_list())], dtype=float)
    return float((wealth / wealth.cummax() - 1.0).min())


def _metrics(
    daily: pd.Series,
    turnover: pd.Series,
    buy_turnover: pd.Series,
    sell_turnover: pd.Series,
    transaction_costs: pd.Series,
    weights: pd.DataFrame,
    per_year: int = 252,
) -> dict[str, Any]:
    mean = float(daily.mean()) if len(daily) else 0.0
    std = float(daily.std()) if len(daily) > 1 else 0.0
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
    terminal = 1.0 + total
    cagr = float(terminal ** (per_year / len(daily)) - 1.0) if len(daily) and terminal > 0 else -1.0
    mean_buy = float(buy_turnover.mean()) if len(buy_turnover) else 0.0
    mean_sell = float(sell_turnover.mean()) if len(sell_turnover) else 0.0
    mean_cost = float(transaction_costs.mean()) if len(transaction_costs) else 0.0
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
        "buy_turnover": mean_buy,
        "sell_turnover": mean_sell,
        "annual_buy_turnover": mean_buy * per_year,
        "annual_sell_turnover": mean_sell * per_year,
        "average_transaction_cost": mean_cost,
        "total_transaction_cost": float(transaction_costs.sum()),
        "buy_cost_bps": BUY_COST_BPS,
        "sell_cost_bps": SELL_COST_BPS,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "avg_holdings": float((weights != 0).sum(axis=1).mean()) if len(weights) else 0.0,
        "win_rate": float((daily > 0).mean()) if len(daily) else 0.0,
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


# ---------------------------------------------------------------------------
# Order layer, portfolio-risk scaling and the factor-decay monitor
# (8.6-8.9, 9.1-9.2, 9.5).  Verbatim mirror of
# ``rdagent/scenarios/gf_quant/strategy/execution.py``: the public helpers
# keep the same bodies under a leading underscore, so the two engines can be
# compared value for value by test/scenarios/gf_quant/
# test_strategy_worker_parity.py.
# ---------------------------------------------------------------------------

#: Unit of account of one book.  ``capital = 1.0`` is pinned rather than
#: derived from a NAV path: it is the same convention the cost formulas and
#: the drift recursion already use.
_CAPITAL = 1.0

#: Snap tolerance used when a lot count lands an ulp below a whole number.
#: The subsequent ``amount > target`` guard keeps `down` conservative.
_LOT_SNAP = 1e-12


def _day_key(value: object) -> str:
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


def _apply_side_aware_tradability(
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
            at_upper = upper is not None and close >= upper - _UNIVERSE_PRICE_LIMIT_TOLERANCE
            at_lower = lower is not None and close <= lower + _UNIVERSE_PRICE_LIMIT_TOLERANCE
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


def _apply_lot_rounding(
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
        target_amount = float(out[symbol]) * _CAPITAL
        lots = target_amount / (close * lot)
        if mode == "down":
            count = math.floor(lots + _LOT_SNAP)
            amount = count * lot * close
            while count > 0 and amount > target_amount:
                count -= 1
                amount = count * lot * close
        else:
            count = math.floor(lots + 0.5)
            amount = count * lot * close
        if min_notional is not None and amount < float(min_notional):
            amount = 0.0
        out[symbol] = amount / _CAPITAL
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


def _apply_participation_cap(
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


def _stage_orders(
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


def _volatility_scale(realised: Sequence[float], *, target_annual: float, window: int, min_scale: float, max_scale: float, periods_per_year: int) -> float:
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


def _drawdown_state_machine(realised_navs: Sequence[float], *, machine: dict[str, object], thresholds: Mapping[str, float], scales: Mapping[str, float], confirm_days: int, recovery: float) -> float:
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


def _plan_order(
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


def _apply_execution_layer(
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
        executed, pending = _apply_participation_cap(
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
        executed, pending = _stage_orders(
            carried, pending, target, position=position, max_days=int(staged_max_days), order=order
        )
        if trace is not None:
            trace["__allocator_reason__"] = "staging"
    if trace is not None:
        trace["allocator"] = executed.copy()
    if side_aware_tradability is not None:
        executed = _apply_side_aware_tradability(
            carried, executed, halt, limits, price, mode=str(side_aware_tradability), day=day
        )
    if trace is not None:
        trace["tradability"] = executed.copy()
    if lot_rounding:
        executed = _apply_lot_rounding(
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


def _factor_decay_state(
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
    same declared degradation ``_volatility_scale`` uses for an empty sample.
    """

    sample = [float(value) for value in values if value is not None and math.isfinite(float(value))][-int(window) :]
    failing = 0
    for value in reversed(sample):
        if value < float(threshold):
            failing += 1
        else:
            break
    return float(scale) if failing >= int(fail_periods) else 1.0


def _factor_ic_series(
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


def _factor_decay_scales(
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
        values = _factor_ic_series(panel, returns, positions=positions, lag=lag)
        scales = []
        for position in range(len(positions)):
            realised = [
                value
                for offset, value in enumerate(values[:position])
                if value is not None and int(positions[offset + 1]) + int(lag) <= int(positions[position])
            ]
            scales.append(
                _factor_decay_state(
                    realised, window=window, fail_periods=fail_periods, threshold=threshold, scale=scale
                )
            )
        out[name] = pd.Series(scales, index=index, dtype=float)
    return out


def _build_buffered_target_row(
    score_row: pd.Series,
    composition: Mapping[str, Any],
    held_names: frozenset,
    returns_history: pd.DataFrame | None,
) -> pd.Series:
    """HoldingBuffer (roadmap 5.5) and DiversitySelector (roadmap 5.6) for one
    date -- the mirror image of ``composition.build_buffered_target_row``.

    Called from ``_backtest`` whenever either operator is enabled, because
    both need information the pure ``_target_weights`` panel pass cannot see:
    the buffer consumes the names still held after the previous executed
    trade, and the diversity filter consumes the realised-return correlation
    labels.  Conservative readings of the roadmap (documented in AGENTS.md):
    the percentile thresholds compare against the *full* valid cross-section
    of the date (best name = 1.0); a held name inside the natural top_n is
    always kept and the retention threshold only rescues held names that
    dropped out of it; names expelled by the diversity filter cannot be
    rescued by the buffer, and every admitted name passes through the same
    greedy correlation screen; the roadmap's per-industry counts are not
    implemented (no industry label data on either side).
    """

    candidates = score_row.dropna()
    if candidates.empty:
        return pd.Series(0.0, index=score_row.index, dtype=float)
    ordered = (
        pd.DataFrame({"score": candidates})
        .rename_axis("symbol")
        .reset_index()
        .sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
    )
    names_ordered = ordered["symbol"].tolist()
    count = len(names_ordered)
    # Percentile from the top: best name 1.0, worst 1/count.
    percentile = {sym: (count - position + 1) / count for position, sym in enumerate(names_ordered, start=1)}

    max_correlation = composition.get("diversity_max_correlation")
    if max_correlation is not None and returns_history is not None and len(returns_history) >= 2:
        correlations = returns_history.corr(min_periods=2)
        eligible_order: list[str] = []
        for sym in names_ordered:
            passes = True
            for chosen in eligible_order:
                value = np.nan
                if sym in correlations.index and chosen in correlations.columns:
                    value = correlations.at[sym, chosen]
                if pd.notna(value) and float(value) > float(max_correlation):
                    passes = False
                    break
            if passes:
                eligible_order.append(sym)
    else:
        eligible_order = names_ordered

    top_n = int(composition["top_n"])
    base = eligible_order[:top_n]
    if composition.get("holding_buffer_entry") is None:
        selection = list(base)
    else:
        entry_threshold = float(composition["holding_buffer_entry"])
        retain_raw = composition.get("holding_buffer_retain")
        # An unset retention threshold collapses onto the entry threshold,
        # mirroring the CompositionSpec resolution.
        retain_threshold = entry_threshold if retain_raw is None else float(retain_raw)
        max_extra = int(composition.get("holding_buffer_max_extra", 0))
        base_set = set(base)
        kept = [sym for sym in base if sym in held_names]
        entrants = [sym for sym in base if sym not in held_names and percentile[sym] >= entry_threshold]
        open_seats = len(base) - len(kept) - len(entrants)
        selection = kept + entrants
        for sym in eligible_order:
            if sym not in held_names or sym in base_set or percentile[sym] < retain_threshold:
                continue
            if open_seats > 0:
                selection.append(sym)
                open_seats -= 1
            elif len(selection) < top_n + max_extra:
                selection.append(sym)
            else:
                break

    if not selection:
        return pd.Series(0.0, index=score_row.index, dtype=float)
    selection_set = set(selection)
    selection_ordered = [sym for sym in names_ordered if sym in selection_set]
    selected = candidates[selection_ordered]
    scheme = composition["weighting"]
    if scheme == "equal":
        row_weights = pd.Series(1.0 / len(selected), index=selected.index)
    elif scheme == "rank":
        ranks = pd.Series(range(len(selected), 0, -1), index=selected.index, dtype=float)
        row_weights = ranks / ranks.sum()
    elif scheme == "score_transform":
        row_weights = _score_transform_weights(selected, composition)
    elif scheme == "inverse_volatility":
        row_weights = _inverse_volatility_weights(selected, returns_history, composition)
    elif scheme == "hierarchical":
        row_weights = _hierarchical_weights(selected, composition)
    elif scheme == "risk_parity":
        row_weights = _risk_parity_weights(selected, returns_history, composition)
    elif scheme == "factor_risk_budget":
        row_weights = _factor_risk_budget_weights(selected, composition)
    else:
        if (selected <= 0).any():
            raise ValueError("proportional weighting requires strictly positive scores")
        row_weights = selected / selected.sum()
    if composition.get("weight_cap") is not None:
        row_weights = _apply_cap(row_weights, float(composition["weight_cap"]))
    if composition.get("industry_weight_cap") is not None:
        row_weights = _apply_industry_cap(
            row_weights,
            float(composition["industry_weight_cap"]),
            composition.get("industry_map") or {},
            name_cap=(float(composition["weight_cap"]) if composition.get("weight_cap") is not None else None),
        )
    if composition.get("weight_floor") is not None:
        row_weights = _apply_weight_floor(row_weights, float(composition["weight_floor"]))
        if not composition.get("weight_floor_allow_cash", True) and float(row_weights.sum()) < 1.0 - 1e-9:
            raise ValueError("weight_floor forbids a cash residual but the book is not fully invested")
    if _has_projection_constraints(composition):
        row_weights = pd.Series(
            _apply_constrained_projection(
                row_weights.to_numpy(dtype=float),
                list(row_weights.index),
                _projection_params_from_payload(composition),
            ),
            index=row_weights.index,
        )
    if composition.get("cash_residual_max") is not None and 1.0 - float(row_weights.sum()) > float(
        composition["cash_residual_max"]
    ) + 1e-9:
        raise ValueError("cash residual exceeds cash_residual_max")
    result = pd.Series(0.0, index=score_row.index, dtype=float)
    result.loc[row_weights.index] = row_weights.to_numpy()
    return result


def _execution_day(
    carried: pd.Series,
    pending: dict[str, list[float]],
    target: pd.Series | None,
    *,
    backtest: Mapping[str, Any],
    data: Mapping[str, Any],
    order: Sequence[str],
    day: str,
    calendar: Sequence[str],
    calendar_index: Mapping[str, int],
    trace: dict[str, object] | None = None,
) -> tuple[pd.Series, dict[str, list[float]]]:
    """Mirror of ``backtest._execution_day``: one trade day of the order layer.

    ``day`` is the *trade* day -- ``signal_lag`` days after the decision that
    produced ``target`` -- so the price, halt and limit state read here are the
    ones that actually apply to the trade.  The injected order-layer calendar
    supplies the position a pending plan is aged against and the window the
    participation cap averages the traded amount over; the window is truncated
    at ``day`` so the estimate can only read already-realised amounts.  A trade
    day the injected calendar does not carry fails closed with the local
    engine's message: aging a plan against a calendar that skipped the day
    would silently stretch or shorten ``max_days``.

    ``trace`` is the optional observation sink the FeasibilityAudit reads; it is
    never read back here, so the executed row is identical either way.
    """

    position = -1
    window_keys: list[str] = [day]
    if calendar:
        if day not in calendar_index:
            raise ValueError(f"execution_data trading_dates do not contain trade date {day}")
        position = calendar_index[day]
        window_keys = list(calendar[: position + 1])

    return _apply_execution_layer(
        carried,
        pending,
        target,
        position=position,
        day=day,
        calendar=window_keys,
        order=order,
        participation_rate=backtest.get("participation_max_rate"),
        participation_window=backtest.get("participation_amount_window"),
        participation_unfilled=backtest.get("participation_unfilled"),
        participation_max_days=backtest.get("participation_max_days"),
        staged_max_days=backtest.get("staged_max_days"),
        lot_rounding=bool(backtest.get("lot_rounding", False)),
        lot_min_notional=backtest.get("lot_min_notional"),
        lot_rounding_mode=backtest.get("lot_rounding_mode"),
        side_aware_tradability=backtest.get("side_aware_tradability"),
        amount=data.get("amount"),
        price=data.get("price"),
        lot_sizes=data.get("lot_sizes"),
        halt=data.get("halt"),
        limits=data.get("price_limits"),
        trace=trace,
    )


def _backtest(
    targets: pd.DataFrame,
    returns: pd.DataFrame,
    backtest: Mapping[str, Any],
    benchmark: pd.Series,
    composition: Mapping[str, Any] | None = None,
    score: pd.DataFrame | None = None,
    execution_data: Mapping[str, Any] | None = None,
    execution_trace: list[dict[str, Any]] | None = None,
):
    """Worker mirror of ``backtest.run_portfolio_backtest``.

    ``execution_data`` is the injected order-layer section (batch 6): prices,
    traded amounts, lot sizes, halt and limit states plus the order layer's own
    trading calendar.  The order layer reads it here; without it an enabled
    order-layer operator fails closed rather than inventing a price or a lot
    size, exactly like the local engine.
    """

    lag = int(backtest["signal_lag"])
    band_name = backtest.get("drift_band_name")
    band_portfolio = backtest.get("drift_band_portfolio")
    forced_days = backtest.get("drift_band_forced_days")
    shrinkage = backtest.get("turnover_shrinkage")
    max_step = backtest.get("turnover_max_step")
    has_drift_band = band_name is not None or band_portfolio is not None
    has_shrinkage = shrinkage is not None or max_step is not None
    has_min_holding = backtest.get("min_holding_days") is not None
    has_selection_operators = composition is not None and score is not None
    aligned = targets.reindex(columns=returns.columns).fillna(0.0)
    target_by_date = {date: row.to_numpy(float) for date, row in aligned.iterrows()}
    symbols = list(returns.columns)
    carried = pd.Series(0.0, index=symbols)
    held_names: frozenset = frozenset()
    entry_index: dict[str, int] = {}
    start_weights = pd.DataFrame(0.0, index=returns.index, columns=symbols)
    gross = pd.Series(0.0, index=returns.index)
    turnover = pd.Series(0.0, index=returns.index)
    buy_turnover = pd.Series(0.0, index=returns.index)
    sell_turnover = pd.Series(0.0, index=returns.index)
    days_since_rebalance = 0
    order_data = execution_data if execution_data is not None else {}
    order_calendar = (
        list(execution_data["trading_dates"])
        if execution_data is not None and execution_data.get("trading_dates")
        else []
    )
    order_calendar_index = {key: index for index, key in enumerate(order_calendar)}
    has_order_layer = (
        backtest.get("participation_max_rate") is not None
        or backtest.get("staged_max_days") is not None
        or backtest.get("side_aware_tradability") is not None
        or backtest.get("lot_rounding", False)
    )
    if (
        backtest.get("participation_max_rate") is not None or backtest.get("staged_max_days") is not None
    ) and not order_calendar:
        # A pending plan is aged in the order layer's own calendar; without one
        # it could never expire, so an allocator without a calendar fails here
        # instead of quietly trading a plan forever.
        raise ValueError("participation_cap and staged_execution require execution_data.trading_dates")
    has_risk_scaling = backtest.get("vol_target_annual") is not None or backtest.get("drawdown_thresholds") is not None
    # Order-layer state.  ``pending`` is the outstanding 8.6 / 8.9 plan and is
    # carried across trading days; ``plan_service_order`` is that plan's fixed
    # service order.  Both live outside the loop because a plan outlives the
    # decision that created it.
    pending: dict[str, list[float]] = {}
    plan_service_order: list[str] = []
    # FeasibilityAudit (10.6) trace sink: ``None`` unless the caller asked for
    # the observation *and* the order layer is live.  Pure observation, never
    # read back inside this function.
    if execution_trace is not None and not has_order_layer:
        execution_trace = None
    # Portfolio-risk state: the realised NAV path and the daily net returns,
    # both built in-loop from the same formula the engine reports at the end
    # (``net = gross - buy * BUY_COST_RATE - sell * SELL_COST_RATE``).
    drawdown_machine: dict[str, object] = {}
    nav_path: list[float] = [1.0]
    net_path: list[float] = []
    for position, date in enumerate(returns.index):
        source = position - lag
        if source >= 0 and returns.index[source] in target_by_date:
            weights = pd.Series(target_by_date[returns.index[source]], index=symbols)
            if has_selection_operators:
                # HoldingBuffer (5.5) / DiversitySelector (5.6): rebuild the
                # target row for the decision date from the still-held names
                # and the return history available up to that date.
                decision_date = returns.index[source]
                buffered = _build_buffered_target_row(
                    score.loc[decision_date], composition, held_names, returns.loc[:decision_date]
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
                if backtest.get("drawdown_thresholds") is not None:
                    risk_scales.append(
                        _drawdown_state_machine(
                            nav_path,
                            machine=drawdown_machine,
                            thresholds=backtest["drawdown_thresholds"],
                            scales=backtest["drawdown_scales"],
                            confirm_days=int(backtest["drawdown_confirm_days"]),
                            recovery=float(backtest["drawdown_recovery"]),
                        )
                    )
                if backtest.get("vol_target_annual") is not None:
                    risk_scales.append(
                        _volatility_scale(
                            net_path,
                            target_annual=float(backtest["vol_target_annual"]),
                            window=int(backtest["vol_target_window"]),
                            min_scale=float(backtest["vol_target_min_scale"]),
                            max_scale=float(backtest["vol_target_max_scale"]),
                            periods_per_year=int(backtest.get("trading_days_per_year", 252)),
                        )
                    )
                risk_scale = min(risk_scales)
                if risk_scale != 1.0:
                    weights = weights * risk_scale
            skipped = False
            if has_drift_band:
                forced = forced_days is not None and days_since_rebalance >= int(forced_days)
                if forced or _drift_band_triggers(carried, weights, band_name, band_portfolio):
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
                weights = _shrink_turnover(carried, weights, shrinkage, max_step)
            layer_ran = has_order_layer and not skipped
            if layer_ran:
                # Order layer (8.6-8.9): the plan's service order is fixed when
                # the plan is created and reused while it is alive, so the
                # priority keeps meaning "who this plan serves first" on the
                # days the plan only advances.  A skipped rebalance neither
                # advances nor revokes the outstanding plan.
                if backtest.get("staged_max_days") is not None:
                    plan_service_order = _plan_order(
                        weights,
                        carried,
                        priority=backtest.get("staged_priority"),
                        score_row=score.loc[returns.index[source]] if score is not None else None,
                    )
                restore_pending = {symbol: list(entry) for symbol, entry in pending.items()}
                restore_order = list(plan_service_order)
                # The feasibility trace observes the order layer only; the
                # surrounding book is recorded alongside it so the audit can
                # reconstruct the whole chain.  ``pre_order_weights`` is the row
                # the order layer is about to receive -- the day's theoretical
                # intention, after the risk scalings, band gate and shrinkage.
                day_trace: dict[str, object] = {}
                pre_order_weights = weights.copy() if execution_trace is not None else None
                weights, pending = _execution_day(
                    carried,
                    pending,
                    weights,
                    backtest=backtest,
                    data=order_data,
                    order=plan_service_order,
                    day=_day_key(date),
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
                    carried, weights, entry_index, position, int(backtest["min_holding_days"])
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
                    backtest=backtest,
                    data=order_data,
                    order=plan_service_order,
                    day=_day_key(date),
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
        day = returns.loc[date].fillna(0.0)
        portfolio_return = float((weights * day).sum())
        delta = weights - carried
        buy_turnover.loc[date] = float(delta.clip(lower=0.0).sum())
        sell_turnover.loc[date] = float((-delta).clip(lower=0.0).sum())
        cash_before, cash_after = 1.0 - float(carried.sum()), 1.0 - float(weights.sum())
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
            carried = weights * (1.0 + day) / denominator
    transaction_costs = buy_turnover * BUY_COST_RATE + sell_turnover * SELL_COST_RATE
    net = gross - transaction_costs
    metrics = _metrics(net, turnover, buy_turnover, sell_turnover, transaction_costs, start_weights)
    excess = net - benchmark
    # Mirror of the local engine's ``_relative_performance`` guard: a
    # one-observation window has an undefined sample standard deviation, and
    # pandas returns NaN for it rather than raising.  Left unguarded the worker
    # would report null where the local engine reports 0.0 -- reachable as soon
    # as a window can be a single day (WalkForwardEvaluation, 10.1, builds
    # one-day segments from a small test_size).
    excess_std = float(excess.std()) if len(excess) > 1 else 0.0
    metrics.update(
        {
            "signal_lag": lag,
            "excess_annualized_return": float(excess.mean()) * 252,
            "tracking_error": excess_std * 252**0.5,
            "information_ratio": float(excess.mean()) / excess_std * 252**0.5 if excess_std > 0 else 0.0,
        }
    )
    by_year = {}
    for year in sorted(set(net.index.year)):
        mask = net.index.year == year
        yearly = _metrics(
            net.loc[mask],
            turnover.loc[mask],
            buy_turnover.loc[mask],
            sell_turnover.loc[mask],
            transaction_costs.loc[mask],
            start_weights.loc[mask],
        )
        year_excess = net.loc[mask] - benchmark.loc[mask]
        # Same guard the whole-window branch above applies, for the same
        # reason: the local engine reaches every year through
        # ``_relative_performance``, which reports 0.0 rather than NaN for a
        # one-observation window (pandas returns NaN for ddof=1).  Reachable
        # whenever a calendar year contributes exactly one trading day of a
        # multi-year window -- WalkForwardEvaluation (10.1) makes short
        # windows ordinary, and a window that straddles a year boundary puts
        # a one-day year in the report.
        year_std = float(year_excess.std()) if len(year_excess) > 1 else 0.0
        yearly.update(
            {
                "excess_annualized_return": float(year_excess.mean()) * 252,
                "tracking_error": year_std * 252**0.5,
                "information_ratio": float(year_excess.mean()) / year_std * 252**0.5 if year_std > 0 else 0.0,
            }
        )
        by_year[str(year)] = yearly
    return metrics, by_year, net, gross, turnover, start_weights, buy_turnover, sell_turnover


def _score(
    ranks: Mapping[str, pd.DataFrame],
    names: Sequence[str],
    signal: Mapping[str, Any] | None = None,
    *,
    decay: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Blend factor and interaction rank panels per the candidate signal.

    Default (no blend configuration): the untouched equal-weight mean with the
    strict missing contract -- a symbol/date is valid only when *every* member
    panel has a value there.  ``blend_weights`` (roadmap 4.9) allocates fixed
    frozen weights, normalised to sum one, over exactly the same members;
    ``blend_groups`` (roadmap 4.10) averages within groups first
    (``blend_within_weights``, default equal) and then across the group scores
    plus every interaction term as a singleton member (``blend_group_weights``,
    default equal).  All three keep the any-missing-kills semantics.

    ``decay`` (roadmap 9.5 FactorDecayMonitor) maps a factor name to its
    per-decision-date scale; that factor's rank panel is multiplied by its
    scale before the blend so a decaying factor contributes less, while every
    other member keeps its defined weight.  ``None`` -- the disabled default,
    and the two-argument ablation call -- leaves every panel untouched.
    """

    signal = signal or {}
    interactions = _interaction_rank_panels(ranks, signal)
    merged = dict(ranks)
    merged.update(interactions)
    members = list(names) + [f"{first}*{second}" for first, second, _ in _interaction_pairs(signal)]
    if decay:
        for name in names:
            scale = decay.get(name)
            if scale is not None:
                merged[name] = merged[name].mul(scale.reindex(merged[name].index).astype(float), axis=0)
    if signal.get("regime_factor_groups") is not None:
        # RegimeConditionalSignal (roadmap 4.12) replaces the static blend
        # outright; request validation keeps it mutually exclusive with every
        # blend/interaction configuration, so nothing else can be set here.
        # The two-argument ablation call (no signal) never reaches this and
        # keeps the untouched equal-weight path.
        return _blend_regime_panels({name: merged[name] for name in ranks}, signal)
    weights = signal.get("blend_weights")
    groups = signal.get("blend_groups")
    if weights is None and groups is None:
        panels = [merged[name] for name in members]
        total = panels[0].copy()
        valid = panels[0].notna()
        for panel in panels[1:]:
            total = total.add(panel, fill_value=0.0)
            valid &= panel.notna()
        return (total / len(panels)).where(valid)
    if weights is not None:
        return _weighted_blend([(merged[name], weights[name]) for name in members])
    top: list[tuple[pd.DataFrame, float]] = []
    for group, factors in groups.items():
        within = (signal.get("blend_within_weights") or {}).get(group)
        weighted = [(merged[factor], (within[factor] if within else 1.0)) for factor in factors]
        top.append((_weighted_blend(weighted), (signal.get("blend_group_weights") or {}).get(group, 1.0)))
    for panel in interactions.values():
        top.append((panel, 1.0))
    return _weighted_blend(top)


def _weighted_blend(weighted: Sequence[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """Blend (panel, weight) pairs; any member missing on a cell kills it.

    The weights are normalised by their sum.  With all weights 1.0 this is
    numerically identical to the default equal-weight path of ``_score``.
    """

    total: pd.DataFrame | None = None
    valid: pd.DataFrame | None = None
    for panel, weight in weighted:
        term = panel * float(weight)
        if total is None:
            total, valid = term, panel.notna()
        else:
            total = total.add(term, fill_value=0.0)
            valid = valid & panel.notna()
    assert total is not None and valid is not None
    normalizer = float(sum(weight for _, weight in weighted))
    return (total / normalizer).where(valid)


def _interaction_pairs(signal: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    # Every pair must be an exact [first, second, kind] triple; a longer or
    # shorter pair fails closed on both ends with the same message instead of
    # being silently truncated (or crashing with an IndexError).
    pairs: list[tuple[str, str, str]] = []
    for pair in signal.get("interaction_pairs") or ():
        if not isinstance(pair, (list, tuple)) or len(pair) != 3:
            raise ValueError("each interaction pair must be a [first, second, kind] triple")
        pairs.append((str(pair[0]), str(pair[1]), str(pair[2])))
    return pairs


def _interaction_rank_panels(ranks: Mapping[str, pd.DataFrame], signal: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """FactorInteraction (roadmap 4.11): one extra blend member per term.

    Each term combines the *final* post-direction percentile ranks of its two
    factors (product, elementwise minimum or geometric mean) and is itself
    re-ranked to a percentile, so it enters the blend on exactly the same
    scale as every factor.  Inputs stay in rank space (the conservative
    reading of the roadmap's "rank first" parameter).  A NaN in either input
    keeps the combined cell NaN, which the blend's any-missing-kills mask
    consumes.
    """

    panels: dict[str, pd.DataFrame] = {}
    for first, second, kind in _interaction_pairs(signal):
        left, right = ranks[first], ranks[second]
        if kind == "product":
            combined = left * right
        elif kind == "min":
            # np.minimum propagates NaN elementwise; pandas min would skip it.
            combined = pd.DataFrame(np.minimum(left, right), index=left.index, columns=left.columns)
        else:  # geometric_mean; request validation leaves no other kind
            combined = np.sqrt(left * right)
        panels[f"{first}*{second}"] = combined.rank(axis=1, pct=True, method="average")
    return panels


def _apply_missing_value(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    """MissingValuePolicy (roadmap 4.5), value-space half.

    ``drop_stock`` removes every symbol that has at least one missing value in
    the panel; ``cross_section_median`` fills each missing cell with the
    median of that date's valid cross-section -- after the A1 signal stages
    and before the direction map, mirroring the local front-end.  The
    neutral-rank policy fills in rank space and is applied by
    ``_apply_neutral_rank_fill`` after ranking.  When
    ``missing_max_fill_ratio`` is set, a date that would need more than that
    share of the cross-section filled fails closed.
    """

    policy = signal.get("missing_value")
    if policy is None:
        return values
    if policy == "drop_stock":
        keep = [column for column in values.columns if bool(values[column].notna().all())]
        if not keep:
            raise ValueError("drop_stock removed every symbol from a factor panel")
        return values[keep]
    if signal.get("missing_max_fill_ratio") is not None:
        ratio = values.isna().sum(axis=1) / len(values.columns)
        if bool((ratio > float(signal["missing_max_fill_ratio"])).any()):
            raise ValueError(
                f"missing_value fill exceeds {float(signal['missing_max_fill_ratio']):g} of the cross-section on some date"
            )
    if policy == "cross_section_median":
        # A fully-missing date keeps its NaN cells (median of nothing), which
        # the blend's any-missing-kills mask consumes unchanged.
        return values.where(values.notna(), values.median(axis=1), axis=0)
    return values


def _apply_neutral_rank_fill(ranks: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    """MissingValuePolicy (roadmap 4.5), rank-space half.

    With ``missing_value="neutral_rank"`` every missing percentile rank is
    replaced by the neutral 0.5, so a filled name neither benefits nor suffers
    from the missing value regardless of the cross-section's shape.
    """

    if signal.get("missing_value") != "neutral_rank":
        return ranks
    if signal.get("missing_max_fill_ratio") is not None:
        ratio = ranks.isna().sum(axis=1) / len(ranks.columns)
        if bool((ratio > float(signal["missing_max_fill_ratio"])).any()):
            raise ValueError(
                f"missing_value fill exceeds {float(signal['missing_max_fill_ratio']):g} of the cross-section on some date"
            )
    return ranks.where(ranks.notna(), 0.5)


_NO_INDUSTRY_SIGNAL = "__no_industry__"

INDUSTRY_NEUTRALIZE_METHODS = ("demean", "regression")


def _industry_demean(values: pd.DataFrame, labels: Mapping[str, str], min_group: int) -> pd.DataFrame:
    """De-mean ``values`` within the injected industry groups, per date.

    Mirror image of ``signal._industry_demean``.  Group means are computed
    over the valid cells only; missing cells stay missing.  A group with
    fewer than ``min_group`` valid values on a date is passed through
    unchanged -- the declared fallback for thin industries (the roadmap
    explicitly demands a fallback, so a thin group never fails the whole
    date).
    """

    result = values.copy()
    groups: dict[str, list[str]] = {}
    for symbol in values.columns:
        groups.setdefault(labels.get(symbol) or _NO_INDUSTRY_SIGNAL, []).append(symbol)
    for label in sorted(groups):
        members = groups[label]
        block = values[members]
        ok = block.notna().sum(axis=1) >= min_group
        result[members] = block.sub(block.mean(axis=1), axis=0).where(ok, block)
    return result


def _cross_section_demean(series: pd.Series, labels: Mapping[str, str], min_group: int) -> pd.Series:
    """One-shot (no time dimension) industry de-mean of an exposure column.

    Mirror image of ``signal._cross_section_demean``.
    """

    groups: dict[str, list[str]] = {}
    for symbol in series.index:
        groups.setdefault(labels.get(symbol) or _NO_INDUSTRY_SIGNAL, []).append(symbol)
    result = series.copy()
    for label in sorted(groups):
        members = groups[label]
        block = series[members]
        if int(block.notna().sum()) < min_group:
            continue
        result[members] = block - block.mean()
    return result


def _cross_sectional_residual(values: pd.DataFrame, exposure_matrix: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """Per-date least-squares residual of ``values`` on the exposure columns.

    Mirror image of ``signal._cross_sectional_residual``.  Rows missing any
    exposure keep their value (listwise deletion); a date with fewer
    joint-valid rows than ``max(min_rows, n_exposures + 1)`` keeps the whole
    row unchanged (the declared thin-sample fallback).  Both the values and
    the exposures are de-meaned across the regression sample first, so the
    residual has zero cross-sectional mean by construction (the implicit
    intercept).  ``numpy.linalg.lstsq`` keeps the solve deterministic and
    identical on both engine sides.
    """

    result = values.copy()
    if exposure_matrix is None or exposure_matrix.empty:
        return result
    needed = max(int(min_rows), exposure_matrix.shape[1] + 1)
    valid_rows = exposure_matrix.notna().all(axis=1)
    for date in values.index:
        row = values.loc[date]
        joint = row.index[row.notna() & valid_rows]
        if len(joint) < needed:
            continue
        y = row.loc[joint].to_numpy(dtype=float)
        x = exposure_matrix.loc[joint].to_numpy(dtype=float)
        y_demeaned = y - float(y.mean())
        x_demeaned = x - x.mean(axis=0)
        beta, _, _, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
        fitted = x_demeaned @ beta
        if not bool(np.all(np.isfinite(fitted))):
            raise ValueError("neutralization regression produced a non-finite fit")
        result.loc[date, joint] = y_demeaned - fitted
    return result


def _exposure_matrix(exposures: Mapping[str, Mapping[str, float]], values: pd.DataFrame) -> pd.DataFrame:
    """Wide (symbol x exposure) frame of the injected per-symbol exposures."""

    frame = pd.DataFrame(
        {name: pd.Series({symbol: float(number) for symbol, number in mapping.items()}) for name, mapping in exposures.items()}
    )
    return frame.reindex(values.columns)


def _apply_industry_neutralize(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    """IndustryNeutralize (roadmap 4.6), value space.

    Mirror image of ``signal.apply_industry_neutralize``.  ``demean``
    de-means the panel within the injected industry groups; ``regression``
    first de-means within industries and then takes the residual of the
    industry-demeaned values on the industry-demeaned exposures
    (Frisch-Waugh: the residual of a regression on industry dummies plus
    exposures, so with no exposures configured it is exactly the de-mean).
    Symbols missing from the mapping share the reserved group, which is
    neutralized like any other industry.  Disabled, the panel passes through
    untouched bit for bit.
    """

    if not signal.get("industry_neutralize"):
        return values
    industry_map = signal.get("industry_neutralize_industry_map") or {}
    min_group = int(signal.get("industry_neutralize_min_group", 3))
    labels = {symbol: industry_map.get(symbol) or _NO_INDUSTRY_SIGNAL for symbol in values.columns}
    demeaned = _industry_demean(values, labels, min_group)
    if signal.get("industry_neutralize_method") == "regression" and signal.get("industry_neutralize_exposures"):
        exposure_matrix = _exposure_matrix(signal["industry_neutralize_exposures"], values)
        for name in exposure_matrix.columns:
            exposure_matrix[name] = _cross_section_demean(exposure_matrix[name], labels, min_group)
        return _cross_sectional_residual(demeaned, exposure_matrix, min_group)
    return demeaned


def _apply_style_neutralize(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    """StyleNeutralize (roadmap 4.7), value space.

    Mirror image of ``signal.apply_style_neutralize``.  Residual of every
    date's cross-section on the injected per-symbol style exposures; applied
    after the industry stage when both are enabled, so the style regression
    neutralizes the industry-neutralized values.  Disabled, the panel passes
    through untouched bit for bit.
    """

    if not signal.get("style_neutralize"):
        return values
    exposure_matrix = _exposure_matrix(signal.get("style_neutralize_exposures") or {}, values)
    return _cross_sectional_residual(values, exposure_matrix, int(signal.get("style_neutralize_min_samples", 5)))


def _regime_state_series(states: Mapping[str, str], lag: int) -> tuple[list[pd.Timestamp], list[str]]:
    """The lag-shifted state series: the state observed at row ``i`` of the
    injected (date-sorted) series becomes effective ``lag`` rows later, the
    same convention as the membership panel's ``lag_days`` shift."""

    parsed = sorted((pd.Timestamp(date_key), str(label)) for date_key, label in states.items())
    dates = [date for date, _ in parsed]
    labels = [label for _, label in parsed]
    if lag <= 0:
        return dates, labels
    if lag >= len(dates):
        return [], []
    return dates[lag:], labels[:-lag]


def _regime_label_for(date: pd.Timestamp, effective_dates: list[pd.Timestamp], effective_labels: list[str]) -> str:
    position = bisect.bisect_right(effective_dates, date) - 1
    if position < 0:
        raise ValueError(f"regime state is missing for decision date {date:%Y-%m-%d}")
    return effective_labels[position]


def _blend_regime_panels(ranks: Mapping[str, pd.DataFrame], signal: Mapping[str, Any]) -> pd.DataFrame:
    """RegimeConditionalSignal (roadmap 4.12): per-date state-conditional blend.

    Mirror image of ``signal.blend_regime_panels``.  On every decision date
    the injected PIT state (read as-of the date, after the ``regime_lag`` row
    shift) selects one factor group; the score row is the equal-weight blend
    of that group's rank panels with the strict missing contract -- a cell is
    valid only when every member of the date's group has a value there.  A
    date without a resolvable state fails closed (the state is a required
    operator input; defaulting one would fabricate information), and so does
    a state label the group mapping does not define.
    """

    groups = signal.get("regime_factor_groups") or {}
    effective_dates, effective_labels = _regime_state_series(
        signal.get("regime_states") or {}, int(signal.get("regime_lag", 1))
    )
    first = next(iter(ranks.values()))
    result = pd.DataFrame(np.nan, index=first.index, columns=first.columns)
    for date in first.index:
        label = _regime_label_for(date, effective_dates, effective_labels)
        members = groups.get(label)
        if members is None:
            raise ValueError(f"regime_factor_groups does not define state {label!r}")
        panels = [ranks[name].loc[date] for name in members]
        total = panels[0].copy()
        valid = panels[0].notna()
        for panel in panels[1:]:
            total = total.add(panel, fill_value=0.0)
            valid &= panel.notna()
        result.loc[date] = (total / len(panels)).where(valid)
    return result


def _build_rank_panels(
    loaded: Mapping[str, pd.DataFrame],
    directions: Mapping[str, Any],
    signal: Mapping[str, Any],
    rebalance_dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    eligible: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Direction map + A1 signal stages + missing-value policy + percentile
    rank + neutral-rank fill for every panel (the mirror image of the local
    ``build_factor_rank_panels``).

    ``eligible`` (candidate-pool filters, roadmap section 3) is applied twice,
    exactly like the local front-end: after the value-space missing-value
    stage (the percentile rank is computed over the post-filter universe) and
    after the rank-space neutral fill (which would otherwise resurrect an
    excluded name at the neutral 0.5).
    """

    ranks: dict[str, pd.DataFrame] = {}
    for name, frame in loaded.items():
        aligned = frame.reindex(index=rebalance_dates, columns=list(symbols))
        values = _apply_missing_value(_apply_signal(aligned, signal), signal)
        if eligible is not None:
            values = values.where(eligible)
        # Batch C2 value-space stage: neutralize the post-filter cross-section
        # (the same universe the percentile rank sees), before the direction
        # map.  Both helpers are no-ops while disabled, so the disabled
        # pipeline is bit-for-bit the pre-change one.
        values = _apply_industry_neutralize(values, signal)
        values = _apply_style_neutralize(values, signal)
        panel = (int(directions[name]) * values).rank(axis=1, pct=True, method="average")
        panel = _apply_neutral_rank_fill(panel, signal)
        if eligible is not None:
            panel = panel.where(eligible)
        ranks[name] = panel
    return ranks


SIGNAL_KEYS = frozenset(
    {
        "transform",
        "transform_scale",
        "transform_shift",
        "winsorize_lower",
        "winsorize_upper",
        "winsorize_min_samples",
        "standardize",
        "standardize_min_std",
        "missing_value",
        "missing_max_fill_ratio",
        "blend_weights",
        "blend_groups",
        "blend_group_weights",
        "blend_within_weights",
        "interaction_pairs",
        "industry_neutralize",
        "industry_neutralize_industry_map",
        "industry_neutralize_method",
        "industry_neutralize_min_group",
        "industry_neutralize_version",
        "industry_neutralize_exposures",
        "style_neutralize",
        "style_neutralize_exposures",
        "style_neutralize_min_samples",
        "regime_states",
        "regime_factor_groups",
        "regime_lag",
        "decay_window",
        "decay_fail_periods",
        "decay_ic_threshold",
        "decay_scale",
    }
)

MISSING_VALUE_POLICIES = ("drop_stock", "cross_section_median", "neutral_rank")

INTERACTION_KINDS = ("product", "min", "geometric_mean")

MAX_INTERACTION_TERMS = 8


def _validate_signal(signal: Mapping[str, Any], factor_names: Sequence[str]) -> None:
    unknown = set(signal) - SIGNAL_KEYS
    if unknown:
        raise ValueError(f"candidate signal has unknown keys: {sorted(unknown)[:3]}")
    transform = signal.get("transform")
    if transform not in (None, "linear", "log", "signed_log", "sqrt"):
        raise ValueError("unsupported signal transform")
    scale = float(signal.get("transform_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("signal transform_scale must be a finite positive number")
    shift = float(signal.get("transform_shift", 0.0))
    if not math.isfinite(shift):
        raise ValueError("signal transform_shift must be finite")
    lower, upper = signal.get("winsorize_lower"), signal.get("winsorize_upper")
    if (lower is None) != (upper is None):
        raise ValueError("candidate signal winsorize quantiles must be provided together")
    if lower is not None and not 0 <= float(lower) < float(upper) <= 1:
        raise ValueError("winsorize quantiles must satisfy 0 <= lower < upper <= 1")
    if int(signal.get("winsorize_min_samples", 1)) < 1:
        raise ValueError("signal winsorize_min_samples must be at least 1")
    standardize = signal.get("standardize")
    if standardize not in (None, "zscore", "rank_z", "robust_z"):
        raise ValueError("unsupported signal standardize")
    min_std = float(signal.get("standardize_min_std", 1e-12))
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("signal standardize_min_std must be a finite positive number")
    missing_value = signal.get("missing_value")
    if missing_value is not None:
        if missing_value not in MISSING_VALUE_POLICIES:
            raise ValueError(
                f"missing_value must be one of {MISSING_VALUE_POLICIES} or None, got {missing_value!r}"
            )
        max_ratio = signal.get("missing_max_fill_ratio")
        if max_ratio is not None:
            max_ratio = float(max_ratio)
            if not math.isfinite(max_ratio) or not 0 < max_ratio <= 1:
                raise ValueError("missing_max_fill_ratio must be in (0, 1] when provided")
            if missing_value == "drop_stock":
                raise ValueError("missing_max_fill_ratio does not apply to the drop_stock policy")
    elif signal.get("missing_max_fill_ratio") is not None:
        raise ValueError("missing_max_fill_ratio requires a missing_value policy")
    blend_weights = signal.get("blend_weights")
    blend_groups = signal.get("blend_groups")
    if blend_weights is not None and blend_groups is not None:
        raise ValueError("blend_weights and blend_groups are mutually exclusive")
    interaction_pairs = _interaction_pairs(signal)
    if len(interaction_pairs) > MAX_INTERACTION_TERMS:
        raise ValueError(f"at most {MAX_INTERACTION_TERMS} interaction terms are allowed")
    seen_members: set[str] = set()
    for first, second, kind in interaction_pairs:
        if kind not in INTERACTION_KINDS:
            raise ValueError(f"interaction kind must be one of {INTERACTION_KINDS}, got {kind!r}")
        if not first or not second or first == second:
            raise ValueError("interaction terms need two distinct non-empty factor names")
        if first not in factor_names or second not in factor_names:
            raise ValueError("interaction terms must reference candidate factors")
        member = f"{first}*{second}"
        if member in seen_members:
            raise ValueError("interaction terms must be distinct")
        seen_members.add(member)
    if blend_weights is not None:
        if not isinstance(blend_weights, Mapping) or not blend_weights:
            raise ValueError("blend_weights must not be empty")
        expected = set(factor_names) | seen_members
        if set(blend_weights) != expected:
            raise ValueError(
                "blend_weights must cover exactly the blend members: "
                "the candidate factors plus every interaction term"
            )
        for weight in blend_weights.values():
            weight = float(weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("blend weights must be finite positive numbers")
    if blend_groups is not None:
        if not isinstance(blend_groups, Mapping) or not blend_groups:
            raise ValueError("blend_groups must not be empty")
        assigned: set[str] = set()
        for group, factors in blend_groups.items():
            if not isinstance(group, str) or not group:
                raise ValueError("blend_groups keys must be non-empty group names")
            if not isinstance(factors, list) or not factors:
                raise ValueError(f"blend group {group!r} must name at least one factor")
            for factor_name in factors:
                if factor_name in assigned:
                    raise ValueError("blend_groups must not assign a factor to two groups")
                assigned.add(factor_name)
        if assigned != set(factor_names):
            raise ValueError("blend_groups must partition the candidate factors exactly")
        group_weights = signal.get("blend_group_weights")
        if group_weights is not None:
            if not isinstance(group_weights, Mapping) or set(group_weights) != set(blend_groups):
                raise ValueError("blend_group_weights keys must match the blend_groups exactly")
            for weight in group_weights.values():
                weight = float(weight)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("blend group weights must be finite positive numbers")
        within_weights = signal.get("blend_within_weights")
        if within_weights is not None:
            if not isinstance(within_weights, Mapping):
                raise ValueError("blend_within_weights must be an object when provided")
            unknown_groups = set(within_weights) - set(blend_groups)
            if unknown_groups:
                raise ValueError(f"blend_within_weights names unknown groups: {sorted(unknown_groups)[:3]}")
            for group, weights in within_weights.items():
                if not isinstance(weights, Mapping) or set(weights) != set(blend_groups[group]):
                    raise ValueError(f"blend_within_weights for group {group!r} must cover its factors exactly")
                for weight in weights.values():
                    weight = float(weight)
                    if not math.isfinite(weight) or weight <= 0:
                        raise ValueError("blend within-group weights must be finite positive numbers")
    # Roadmap 4.6 IndustryNeutralize (batch C2).
    industry_neutralize = signal.get("industry_neutralize", False)
    if not isinstance(industry_neutralize, bool):
        raise ValueError("industry_neutralize must be a boolean")
    if not industry_neutralize:
        lazy_industry = (
            signal.get("industry_neutralize_industry_map") is not None
            or signal.get("industry_neutralize_exposures") is not None
            or signal.get("industry_neutralize_version") is not None
            or signal.get("industry_neutralize_method", "demean") != "demean"
            or signal.get("industry_neutralize_min_group", 3) != 3
        )
        if lazy_industry:
            raise ValueError("industry neutralize parameters require industry_neutralize")
    else:
        industry_map = signal.get("industry_neutralize_industry_map")
        if industry_map is None:
            raise ValueError("industry neutralize requires industry_neutralize_industry_map")
        if not isinstance(industry_map, dict) or not industry_map:
            raise ValueError("industry_neutralize_industry_map must map non-empty symbols to non-empty industry labels")
        for symbol, label in industry_map.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                raise ValueError("industry_neutralize_industry_map must map non-empty symbols to non-empty industry labels")
    industry_method = signal.get("industry_neutralize_method", "demean")
    if industry_method not in INDUSTRY_NEUTRALIZE_METHODS:
        raise ValueError(
            f"industry_neutralize_method must be one of {INDUSTRY_NEUTRALIZE_METHODS}, got {industry_method!r}"
        )
    industry_min_group = signal.get("industry_neutralize_min_group", 3)
    if isinstance(industry_min_group, bool) or not isinstance(industry_min_group, (int, float)):
        raise ValueError("industry_neutralize_min_group must be a whole number")
    if float(industry_min_group) != int(industry_min_group):
        raise ValueError("industry_neutralize_min_group must be a whole number")
    if int(industry_min_group) < 2:
        raise ValueError("industry_neutralize_min_group must be at least 2")
    industry_version = signal.get("industry_neutralize_version")
    if industry_version is not None:
        if not industry_neutralize:
            raise ValueError("industry neutralize parameters require industry_neutralize")
        if not isinstance(industry_version, str) or not industry_version:
            raise ValueError("industry_neutralize_version must be a non-empty string when provided")
    industry_exposures = signal.get("industry_neutralize_exposures")
    if industry_exposures is not None:
        if not industry_neutralize:
            raise ValueError("industry neutralize parameters require industry_neutralize")
        if industry_method != "regression":
            raise ValueError("industry_neutralize_exposures is only valid with the regression method")
        _validate_exposure_map(
            industry_exposures,
            "industry_neutralize_exposures must map non-empty exposure names to non-empty symbol -> finite number objects",
        )
    # Roadmap 4.7 StyleNeutralize (batch C2).
    style_neutralize = signal.get("style_neutralize", False)
    if not isinstance(style_neutralize, bool):
        raise ValueError("style_neutralize must be a boolean")
    if not style_neutralize:
        if (
            signal.get("style_neutralize_exposures") is not None
            or signal.get("style_neutralize_min_samples", 5) != 5
        ):
            raise ValueError("style neutralize parameters require style_neutralize")
    else:
        style_exposures = signal.get("style_neutralize_exposures")
        if style_exposures is None:
            raise ValueError("style neutralize requires style_neutralize_exposures")
        _validate_exposure_map(
            style_exposures,
            "style_neutralize_exposures must map non-empty style names to non-empty symbol -> finite number objects",
        )
    style_min_samples = signal.get("style_neutralize_min_samples", 5)
    if isinstance(style_min_samples, bool) or not isinstance(style_min_samples, (int, float)):
        raise ValueError("style_neutralize_min_samples must be a whole number")
    if float(style_min_samples) != int(style_min_samples):
        raise ValueError("style_neutralize_min_samples must be a whole number")
    if int(style_min_samples) < 2:
        raise ValueError("style_neutralize_min_samples must be at least 2")
    # Roadmap 4.12 RegimeConditionalSignal (batch C2).
    regime_states = signal.get("regime_states")
    regime_groups = signal.get("regime_factor_groups")
    if (regime_states is None) != (regime_groups is None):
        raise ValueError("regime_states and regime_factor_groups must be provided together")
    if regime_states is not None:
        if blend_weights is not None:
            raise ValueError("regime_factor_groups and blend_weights are mutually exclusive")
        if blend_groups is not None:
            raise ValueError("regime_factor_groups and blend_groups are mutually exclusive")
        if interaction_pairs:
            raise ValueError("regime_factor_groups and interaction_pairs are mutually exclusive")
        if not isinstance(regime_states, dict) or not regime_states:
            raise ValueError("regime_states must map date strings to non-empty state labels")
        for date_key, label in regime_states.items():
            try:
                parsed_date = date.fromisoformat(str(date_key))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"regime_states contains an invalid date: {date_key!r}") from exc
            del parsed_date
            if not isinstance(label, str) or not label:
                raise ValueError("regime_states must map date strings to non-empty state labels")
        if not isinstance(regime_groups, dict) or not regime_groups:
            raise ValueError("regime_factor_groups must map non-empty state labels to non-empty factor lists")
        for label, factors in regime_groups.items():
            if not isinstance(label, str) or not label:
                raise ValueError("regime_factor_groups must map non-empty state labels to non-empty factor lists")
            if not isinstance(factors, list) or not factors:
                raise ValueError(f"regime state {label!r} must name at least one factor")
            for factor_name in factors:
                if not isinstance(factor_name, str) or not factor_name:
                    raise ValueError("regime_factor_groups members must be non-empty factor names")
                if factor_name not in factor_names:
                    raise ValueError("regime_factor_groups members must reference candidate factors")
    regime_lag = signal.get("regime_lag", 1)
    if isinstance(regime_lag, bool) or not isinstance(regime_lag, (int, float)):
        raise ValueError("regime_lag must be a whole number")
    if float(regime_lag) != int(regime_lag):
        raise ValueError("regime_lag must be a whole number")
    if int(regime_lag) < 0:
        raise ValueError("regime_lag must not be negative")
    # Roadmap 9.5 FactorDecayMonitor.  decay_window enables the monitor; the
    # three dependent keys follow the "x requires decay_window" convention and
    # collapse onto their conservative defaults once the monitor is on.  Same
    # messages as SignalSpec.__post_init__.
    decay_window = signal.get("decay_window")
    for dependent in ("decay_fail_periods", "decay_ic_threshold", "decay_scale"):
        if decay_window is None and signal.get(dependent) is not None:
            raise ValueError(f"{dependent} requires decay_window")
    if decay_window is not None:
        window = _universe_whole(decay_window, "decay_window must be a whole number")
        if window < 2:
            raise ValueError("decay_window must be at least 2")
        raw_fail = signal.get("decay_fail_periods")
        fail_periods = (
            1 if raw_fail is None else _universe_whole(raw_fail, "decay_fail_periods must be a whole number")
        )
        if fail_periods < 1:
            raise ValueError("decay_fail_periods must be at least 1")
        raw_threshold = signal.get("decay_ic_threshold")
        if raw_threshold is not None:
            _universe_number(raw_threshold, "decay_ic_threshold must be finite")
        raw_scale = signal.get("decay_scale")
        if raw_scale is not None and not 0.0 <= _universe_number(raw_scale, "decay_scale must be in [0, 1]") <= 1.0:
            raise ValueError("decay_scale must be in [0, 1]")


def _apply_transform(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    kind = signal.get("transform")
    if kind is None:
        return values
    if kind == "linear":
        # Missing keys fall back to the same neutral defaults the local
        # SignalSpec.from_payload applies (scale 1.0, shift 0.0).
        return values * float(signal.get("transform_scale", 1.0)) + float(signal.get("transform_shift", 0.0))
    if kind == "log":
        if bool((values <= 0).any().any()):
            raise ValueError("log transform requires strictly positive factor values")
        return np.log(values)
    if kind == "signed_log":
        return np.sign(values) * np.log1p(np.abs(values))
    if kind == "sqrt":
        if bool((values < 0).any().any()):
            raise ValueError("sqrt transform requires non-negative factor values")
        return np.sqrt(values)
    raise ValueError(f"unsupported signal transform: {kind!r}")


def _apply_winsorize(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    lower = signal.get("winsorize_lower")
    if lower is None:
        return values
    min_samples = int(signal.get("winsorize_min_samples", 1))
    counts = values.notna().sum(axis=1)
    if bool((counts < min_samples).any()):
        raise ValueError(f"winsorize requires at least {min_samples} valid cross-sectional values on every date")
    lower_bounds = values.quantile(float(lower), axis=1)
    upper_bounds = values.quantile(float(signal["winsorize_upper"]), axis=1)
    return values.clip(lower=lower_bounds, upper=upper_bounds, axis=0)


def _apply_standardize(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    kind = signal.get("standardize")
    if kind is None:
        return values
    min_std = float(signal.get("standardize_min_std", 1e-12))
    if kind == "rank_z":
        values = values.rank(axis=1, pct=True, method="average")
    if kind == "robust_z":
        center = values.median(axis=1)
        scale = values.sub(center, axis=0).abs().median(axis=1) * 1.4826
    else:
        center = values.mean(axis=1)
        scale = values.std(axis=1)
    scale = scale.astype(float)
    if bool((~np.isfinite(scale)).any()) or bool((scale < min_std).any()):
        raise ValueError(
            f"cross-sectional standardize requires a dispersion of at least {min_std:g} on every date"
        )
    return values.sub(center, axis=0).div(scale, axis=0)


def _apply_signal(values: pd.DataFrame, signal: Mapping[str, Any]) -> pd.DataFrame:
    return _apply_standardize(_apply_winsorize(_apply_transform(values, signal), signal), signal)


GATING_KEYS = frozenset({"layers"})

GATE_LAYER_KEYS = frozenset({"factor", "direction", "quantile", "min_pass"})


def _validate_gating(gating: Mapping[str, Any], factor_names: Sequence[str]) -> None:
    unknown = set(gating) - GATING_KEYS
    if unknown:
        raise ValueError(f"candidate gating has unknown keys: {sorted(unknown)[:3]}")
    layers = gating.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("candidate gating must contain a non-empty layers list")
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("candidate gating layers must be objects")
        unknown = set(layer) - GATE_LAYER_KEYS
        if unknown:
            raise ValueError(f"gating layer has unknown keys: {sorted(unknown)[:3]}")
        if str(layer.get("factor")) not in factor_names:
            raise ValueError("gating layer factor must be one of the candidate factors")
        if layer.get("direction") not in ("top", "bottom"):
            raise ValueError("unsupported gating layer direction")
        try:
            quantile = float(layer["quantile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("gating layer quantile must be a number") from exc
        if not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("gating layer quantile must be strictly between 0 and 1")
        if int(layer.get("min_pass", 1)) < 1:
            raise ValueError("gating layer min_pass must be at least 1")


def _apply_gating(score: pd.DataFrame, ranks: Mapping[str, pd.DataFrame], gating: Mapping[str, Any]) -> pd.DataFrame:
    """QuantileGate / CascadedFactorGating (roadmap 5.1 / 5.2).

    Every layer keeps, per date, the symbols whose post-direction percentile
    rank of its factor passes the quantile (``top``: rank >= quantile,
    ``bottom``: rank <= quantile; the inclusive comparison on the average rank
    is the tie rule).  Layers combine with *and* semantics and a symbol that
    fails any layer has its score set to NaN, which ``_target_weights``
    consumes through its existing dropna.  A date whose surviving set drops
    below the layer's min_pass fails closed.
    """

    layers = gating.get("layers") or []
    if not layers:
        return score
    survivors = None
    for position, layer in enumerate(layers):
        panel = ranks[str(layer["factor"])]
        mask = panel >= float(layer["quantile"]) if layer.get("direction") == "top" else panel <= float(layer["quantile"])
        survivors = mask if survivors is None else survivors & mask
        min_pass = int(layer.get("min_pass", 1))
        if bool((survivors.sum(axis=1) < min_pass).any()):
            raise ValueError(
                f"gating layer {position} ({layer.get('factor')} {layer.get('direction')} {float(layer['quantile']):g}) "
                f"leaves fewer than {min_pass} surviving names on some date"
            )
    return score.where(survivors)


# ---------------------------------------------------------------------------
# Candidate-pool / tradability filters (roadmap 3.1-3.8).
#
# Every data input (membership, statuses, limit prices, amounts, raw close,
# trading calendar) is an explicit payload panel -- neither engine carries
# such data, so the upper layer injects it (the composition industry_map
# pattern).  Validation mirrors ``UniverseSpec`` in
# ``rdagent/scenarios/gf_quant/strategy/candidate.py`` and the executor
# mirrors ``universe.build_eligible_mask`` verbatim, including every error
# message; the parity tests feed identical payloads to both sides.
# ---------------------------------------------------------------------------

UNIVERSE_TOP_LEVEL_KEYS = frozenset(
    {
        "trading_dates",
        "price",
        "base_universe",
        "index_membership",
        "listing_age",
        "status_exclusion",
        "price_limit",
        "liquidity",
        "price_range",
        "data_quality",
    }
)

_UNIVERSE_SECTION_KEYS = {
    "base_universe": frozenset({"symbols"}),
    "index_membership": frozenset({"membership", "lag_days", "exclude_recent_days"}),
    "listing_age": frozenset({"listing_dates", "min_days"}),
    "status_exclusion": frozenset({"statuses", "excluded_statuses"}),
    "price_limit": frozenset({"limit_up", "limit_down", "exclude_limit_up", "exclude_limit_down"}),
    "liquidity": frozenset({"amount", "window", "min_amount", "quantile", "missing"}),
    "price_range": frozenset({"min_price", "max_price", "consecutive_days"}),
    "data_quality": frozenset({"min_coverage", "max_missing_run", "window", "non_positive_price"}),
}

_UNIVERSE_PRICE_LIMIT_TOLERANCE = 1e-9


def _universe_number(value: Any, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(message)
    return float(value)


def _universe_whole(value: Any, message: str) -> int:
    number = _universe_number(value, message)
    if number != int(number):
        raise ValueError(message)
    return int(number)


def _universe_date(value: Any, message: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(message)
    try:
        return pd.Timestamp(value).normalize()
    except ValueError:
        raise ValueError(message) from None


def _universe_number_panel(value: Any, prefix: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, dict) or (not value and not allow_empty):
        raise ValueError(f"{prefix} must map date strings to objects of symbol -> number")
    for date_key, row in value.items():
        _universe_date(date_key, f"{prefix} contains an invalid date: {date_key!r}")
        if not isinstance(row, dict):
            raise ValueError(f"{prefix} must map date strings to objects of symbol -> number")
        for symbol, number in row.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(f"{prefix} must map date strings to objects of symbol -> number")
            _universe_number(number, f"{prefix} must map date strings to objects of symbol -> number")


def _universe_true_panel(value: Any, prefix: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{prefix} must map date strings to objects of symbol -> true")
    for date_key, row in value.items():
        _universe_date(date_key, f"{prefix} contains an invalid date: {date_key!r}")
        if not isinstance(row, dict) or not row:
            raise ValueError(f"{prefix} must map date strings to objects of symbol -> true")
        for symbol, flag in row.items():
            if not isinstance(symbol, str) or not symbol or flag is not True:
                raise ValueError(f"{prefix} must map date strings to objects of symbol -> true")


def _universe_string_panel(value: Any, prefix: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must map date strings to objects of symbol -> non-empty string")
    for date_key, row in value.items():
        _universe_date(date_key, f"{prefix} contains an invalid date: {date_key!r}")
        if not isinstance(row, dict):
            raise ValueError(f"{prefix} must map date strings to objects of symbol -> non-empty string")
        for symbol, label in row.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                raise ValueError(f"{prefix} must map date strings to objects of symbol -> non-empty string")


def _universe_symbol_dates(value: Any, prefix: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{prefix} must map non-empty symbols to date strings")
    for symbol, day in value.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{prefix} must map non-empty symbols to date strings")
        _universe_date(day, f"{prefix} contains an invalid date for {symbol!r}")


def _validate_universe_rules(universe: Mapping[str, Any]) -> None:
    """Mirror of ``UniverseSpec.from_payload`` + every rule ``__post_init__``."""

    unknown_top = set(universe) - UNIVERSE_TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"candidate universe_rules has unknown keys: {sorted(unknown_top)[:3]}")
    for section in _UNIVERSE_SECTION_KEYS:
        raw = universe.get(section)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"universe_rules.{section} must be an object when provided")
        unknown = set(raw) - _UNIVERSE_SECTION_KEYS[section]
        if unknown:
            raise ValueError(f"universe_rules.{section} has unknown keys: {sorted(unknown)[:3]}")

    trading_dates = universe.get("trading_dates")
    if trading_dates is not None:
        if not isinstance(trading_dates, list) or not trading_dates:
            raise ValueError("universe_rules.trading_dates must be a non-empty list of date strings")
        parsed = [_universe_date(day, "universe_rules.trading_dates must be a non-empty list of date strings") for day in trading_dates]
        if any(b <= a for a, b in zip(parsed, parsed[1:])):
            raise ValueError("universe_rules.trading_dates must be strictly increasing")
    price = universe.get("price")
    if price is not None:
        _universe_number_panel(price, "universe_rules.price")

    base = universe.get("base_universe")
    if base is not None:
        if "symbols" not in base:
            raise ValueError("universe_rules.base_universe is missing the symbols key")
        symbols = base["symbols"]
        if (
            isinstance(symbols, str)
            or not isinstance(symbols, list)
            or not symbols
            or any(not isinstance(name, str) or not name for name in symbols)
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError(
                "universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings"
            )

    membership = universe.get("index_membership")
    if membership is not None:
        if "membership" not in membership:
            raise ValueError("universe_rules.index_membership is missing the membership key")
        _universe_true_panel(membership["membership"], "universe_rules.index_membership.membership")
        if _universe_whole(membership.get("lag_days", 0), "universe_rules.index_membership.lag_days must be a whole number >= 0") < 0:
            raise ValueError("universe_rules.index_membership.lag_days must be a whole number >= 0")
        if (
            _universe_whole(
                membership.get("exclude_recent_days", 0),
                "universe_rules.index_membership.exclude_recent_days must be a whole number >= 0",
            )
            < 0
        ):
            raise ValueError("universe_rules.index_membership.exclude_recent_days must be a whole number >= 0")

    listing = universe.get("listing_age")
    if listing is not None:
        if "listing_dates" not in listing:
            raise ValueError("universe_rules.listing_age is missing the listing_dates key")
        if "min_days" not in listing:
            raise ValueError("universe_rules.listing_age is missing the min_days key")
        _universe_symbol_dates(listing["listing_dates"], "universe_rules.listing_age.listing_dates")
        if _universe_whole(listing["min_days"], "universe_rules.listing_age.min_days must be a whole number >= 1") < 1:
            raise ValueError("universe_rules.listing_age.min_days must be a whole number >= 1")

    status = universe.get("status_exclusion")
    if status is not None:
        if "statuses" not in status:
            raise ValueError("universe_rules.status_exclusion is missing the statuses key")
        if "excluded_statuses" not in status:
            raise ValueError("universe_rules.status_exclusion is missing the excluded_statuses key")
        _universe_string_panel(status["statuses"], "universe_rules.status_exclusion.statuses")
        excluded = status["excluded_statuses"]
        if (
            isinstance(excluded, str)
            or not isinstance(excluded, list)
            or not excluded
            or any(not isinstance(name, str) or not name for name in excluded)
            or len(set(excluded)) != len(excluded)
        ):
            raise ValueError(
                "universe_rules.status_exclusion.excluded_statuses must be a non-empty list of distinct non-empty strings"
            )

    limit = universe.get("price_limit")
    if limit is not None:
        exclude_up = limit.get("exclude_limit_up", True)
        exclude_down = limit.get("exclude_limit_down", True)
        if not isinstance(exclude_up, bool) or not isinstance(exclude_down, bool):
            raise ValueError("universe_rules.price_limit.exclude_limit_up and exclude_limit_down must be booleans")
        if not exclude_up and not exclude_down:
            raise ValueError("universe_rules.price_limit must exclude at least one of limit_up and limit_down")
        if exclude_up and limit.get("limit_up") is None:
            raise ValueError("universe_rules.price_limit.limit_up is required when exclude_limit_up is true")
        if not exclude_up and limit.get("limit_up") is not None:
            raise ValueError("universe_rules.price_limit.limit_up is only valid when exclude_limit_up is true")
        if exclude_down and limit.get("limit_down") is None:
            raise ValueError("universe_rules.price_limit.limit_down is required when exclude_limit_down is true")
        if not exclude_down and limit.get("limit_down") is not None:
            raise ValueError("universe_rules.price_limit.limit_down is only valid when exclude_limit_down is true")
        if limit.get("limit_up") is not None:
            _universe_number_panel(limit["limit_up"], "universe_rules.price_limit.limit_up")
        if limit.get("limit_down") is not None:
            _universe_number_panel(limit["limit_down"], "universe_rules.price_limit.limit_down")

    liquidity = universe.get("liquidity")
    if liquidity is not None:
        if "amount" not in liquidity:
            raise ValueError("universe_rules.liquidity is missing the amount key")
        _universe_number_panel(liquidity["amount"], "universe_rules.liquidity.amount")
        if _universe_whole(liquidity.get("window", 20), "universe_rules.liquidity.window must be a whole number >= 1") < 1:
            raise ValueError("universe_rules.liquidity.window must be a whole number >= 1")
        if liquidity.get("min_amount") is not None:
            if _universe_number(
                liquidity["min_amount"],
                "universe_rules.liquidity.min_amount must be a finite number >= 0 when provided",
            ) < 0:
                raise ValueError("universe_rules.liquidity.min_amount must be a finite number >= 0 when provided")
        if liquidity.get("quantile") is not None:
            quantile = _universe_number(liquidity["quantile"], "universe_rules.liquidity.quantile must be in [0, 1] when provided")
            if not 0 <= quantile <= 1:
                raise ValueError("universe_rules.liquidity.quantile must be in [0, 1] when provided")
        if liquidity.get("missing", "exclude") not in ("exclude", "include"):
            raise ValueError(
                f"universe_rules.liquidity.missing must be one of ('exclude', 'include'), got {liquidity.get('missing', 'exclude')!r}"
            )
        if liquidity.get("min_amount") is None and liquidity.get("quantile") is None:
            raise ValueError("universe_rules.liquidity requires min_amount or quantile")

    price_range = universe.get("price_range")
    if price_range is not None:
        min_price = price_range.get("min_price")
        max_price = price_range.get("max_price")
        if min_price is not None and _universe_number(
            min_price, "universe_rules.price_range.min_price must be a finite positive number when provided"
        ) <= 0:
            raise ValueError("universe_rules.price_range.min_price must be a finite positive number when provided")
        if max_price is not None and _universe_number(
            max_price, "universe_rules.price_range.max_price must be a finite positive number when provided"
        ) <= 0:
            raise ValueError("universe_rules.price_range.max_price must be a finite positive number when provided")
        if min_price is None and max_price is None:
            raise ValueError("universe_rules.price_range requires min_price or max_price")
        if min_price is not None and max_price is not None and float(min_price) > float(max_price):
            raise ValueError("universe_rules.price_range.min_price must not exceed max_price")
        if (
            _universe_whole(
                price_range.get("consecutive_days", 1),
                "universe_rules.price_range.consecutive_days must be a whole number >= 1",
            )
            < 1
        ):
            raise ValueError("universe_rules.price_range.consecutive_days must be a whole number >= 1")

    quality = universe.get("data_quality")
    if quality is not None:
        if quality.get("min_coverage") is not None:
            coverage = _universe_number(
                quality["min_coverage"], "universe_rules.data_quality.min_coverage must be in [0, 1] when provided"
            )
            if not 0 <= coverage <= 1:
                raise ValueError("universe_rules.data_quality.min_coverage must be in [0, 1] when provided")
        if quality.get("max_missing_run") is not None:
            if (
                _universe_whole(
                    quality["max_missing_run"],
                    "universe_rules.data_quality.max_missing_run must be a whole number >= 1 when provided",
                )
                < 1
            ):
                raise ValueError("universe_rules.data_quality.max_missing_run must be a whole number >= 1 when provided")
        if _universe_whole(quality.get("window", 20), "universe_rules.data_quality.window must be a whole number >= 1") < 1:
            raise ValueError("universe_rules.data_quality.window must be a whole number >= 1")
        if not isinstance(quality.get("non_positive_price", False), bool):
            raise ValueError("universe_rules.data_quality.non_positive_price must be a boolean")
        if quality.get("min_coverage") is None and quality.get("max_missing_run") is None and not quality.get(
            "non_positive_price", False
        ):
            raise ValueError("universe_rules.data_quality requires min_coverage, max_missing_run or non_positive_price")

    if universe.get("listing_age") is not None or universe.get("liquidity") is not None or universe.get("price_range") is not None:
        if universe.get("trading_dates") is None:
            raise ValueError("universe_rules.trading_dates is required when listing_age, liquidity or price_range is enabled")
    if universe.get("price_limit") is not None and universe.get("price") is None:
        raise ValueError("universe_rules.price is required when price_limit is enabled")
    if universe.get("price_range") is not None and universe.get("price") is None:
        raise ValueError("universe_rules.price is required when price_range is enabled")
    if universe.get("data_quality") is not None and universe["data_quality"].get("non_positive_price", False) and universe.get(
        "price"
    ) is None:
        raise ValueError("universe_rules.price is required when data_quality requests the non-positive-price check")


def _universe_panel(data: Mapping[str, Any], dtype: str) -> pd.DataFrame:
    """Mirror of ``universe.panel_from_payload``."""

    dates = sorted({pd.Timestamp(key).normalize() for key in data})
    symbols = sorted({str(symbol) for row in data.values() for symbol in row})
    if dtype == "bool":
        frame = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=symbols, dtype=bool)
    elif dtype == "object":
        frame = pd.DataFrame(None, index=pd.DatetimeIndex(dates), columns=symbols, dtype=object)
    else:
        frame = pd.DataFrame(float("nan"), index=pd.DatetimeIndex(dates), columns=symbols, dtype=float)
    for key, row in data.items():
        day = pd.Timestamp(key).normalize()
        for symbol, value in row.items():
            frame.at[day, symbol] = value
    return frame


def _universe_calendar_coverage(calendar: pd.DatetimeIndex, index: pd.DatetimeIndex) -> None:
    missing = index.difference(calendar)
    if len(missing):
        raise ValueError(f"universe trading_dates do not contain decision date {missing[0].date().isoformat()}")


def _universe_membership_mask(rule: Mapping[str, Any], index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    panel = _universe_panel(rule["membership"], "bool")
    if int(rule.get("lag_days", 0)) > 0:
        panel = panel.shift(int(rule["lag_days"])).fillna(False).astype(bool)
    exclude_recent = int(rule.get("exclude_recent_days", 0))
    if exclude_recent > 0:
        excluded = pd.DataFrame(False, index=panel.index, columns=panel.columns)
        for symbol in panel.columns:
            member = panel[symbol]
            inclusion = member & ~member.shift(1, fill_value=False).astype(bool)
            run = inclusion.cumsum()
            since = run.groupby(run).cumcount()
            excluded[symbol] = member & (run > 0) & (since < exclude_recent)
        panel = panel & ~excluded
    combined = panel.index.union(index)
    asof = panel.reindex(combined).ffill().loc[index]
    return asof.reindex(columns=columns).fillna(False).astype(bool)


def _universe_listing_age_mask(
    rule: Mapping[str, Any], calendar: pd.DatetimeIndex, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    position = calendar.searchsorted(index, side="right")
    mask = pd.DataFrame(False, index=index, columns=columns)
    for symbol in columns:
        day = rule["listing_dates"].get(symbol)
        if day is None:
            continue
        listed = pd.Timestamp(day).normalize()
        mask[symbol] = (position - calendar.searchsorted(listed, side="right")) >= int(rule["min_days"])
    return mask


def _universe_status_mask(rule: Mapping[str, Any], index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    panel = _universe_panel(rule["statuses"], "object")
    at = panel.reindex(index=index).reindex(columns=columns)
    return ~at.isin(set(rule["excluded_statuses"]))


def _universe_price_limit_mask(
    rule: Mapping[str, Any], price: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    close = price.reindex(index=index).reindex(columns=columns)
    excluded = pd.DataFrame(False, index=index, columns=columns)
    if rule.get("exclude_limit_up", True):
        limit = _universe_panel(rule["limit_up"], "float").reindex(index=index).reindex(columns=columns)
        excluded |= close.ge(limit - _UNIVERSE_PRICE_LIMIT_TOLERANCE)
        excluded |= close.isna() | limit.isna()
    if rule.get("exclude_limit_down", True):
        limit = _universe_panel(rule["limit_down"], "float").reindex(index=index).reindex(columns=columns)
        excluded |= close.le(limit + _UNIVERSE_PRICE_LIMIT_TOLERANCE)
        excluded |= close.isna() | limit.isna()
    return ~excluded


def _universe_liquidity_mask(
    rule: Mapping[str, Any], calendar: pd.DatetimeIndex, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    amount = _universe_panel(rule["amount"], "float")
    aligned = amount.reindex(index=calendar).reindex(columns=columns)
    rolling = aligned.rolling(int(rule.get("window", 20)), min_periods=1).mean()
    position = calendar.get_indexer(index)
    rows = rolling.iloc[position].set_axis(index, axis=0)
    ok = pd.DataFrame(True, index=index, columns=columns)
    if rule.get("min_amount") is not None:
        ok &= rows.ge(float(rule["min_amount"]))
    if rule.get("quantile") is not None:
        threshold = rows.quantile(float(rule["quantile"]), axis=1)
        leg = rows.ge(threshold, axis=0)
        ok &= leg.where(threshold.notna(), True)
    if rule.get("missing", "exclude") == "exclude":
        ok &= rows.notna()
    else:
        ok |= rows.isna()
    return ok


def _universe_price_range_mask(
    rule: Mapping[str, Any], price: pd.DataFrame, calendar: pd.DatetimeIndex, index: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    close = price.reindex(index=calendar).reindex(columns=columns)
    position = calendar.get_indexer(index)
    low = float("-inf") if rule.get("min_price") is None else float(rule["min_price"])
    high = float("inf") if rule.get("max_price") is None else float(rule["max_price"])
    ok = np.ones((len(index), len(columns)), dtype=bool)
    for offset in range(int(rule.get("consecutive_days", 1))):
        source = position - offset
        has_row = source >= 0
        window = close.iloc[np.clip(source, 0, max(len(close) - 1, 0))].to_numpy(dtype=float)
        ok &= (~np.isnan(window)) & (window >= low) & (window <= high) & has_row[:, None]
    return pd.DataFrame(ok, index=index, columns=columns)


def _universe_data_quality_mask(
    rule: Mapping[str, Any],
    price: pd.DataFrame | None,
    factor_panels: Mapping[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    ok = pd.DataFrame(True, index=index, columns=columns)
    if rule.get("non_positive_price", False):
        close = price.reindex(index=index).reindex(columns=columns)
        ok &= close.gt(0)
    if rule.get("min_coverage") is None and rule.get("max_missing_run") is None:
        return ok
    if not factor_panels:
        raise ValueError("universe data_quality coverage requires factor panels")
    union = pd.DatetimeIndex(index)
    for frame in factor_panels.values():
        union = union.union(pd.DatetimeIndex(frame.index))
    union = union.sort_values()
    missing = [frame.reindex(index=union).reindex(columns=columns).isna() for frame in factor_panels.values()]
    if rule.get("min_coverage") is not None:
        miss_count = missing[0].astype(int)
        for miss in missing[1:]:
            miss_count = miss_count + miss.astype(int)
        windowed = miss_count.rolling(int(rule.get("window", 20)), min_periods=1).sum()
        rows_in_window = np.minimum(int(rule.get("window", 20)), np.arange(1, len(union) + 1))
        coverage = 1 - windowed.div(pd.Series(len(missing) * rows_in_window, index=union), axis=0)
        ok &= coverage.loc[index].ge(float(rule["min_coverage"]))
    if rule.get("max_missing_run") is not None:
        bad = pd.DataFrame(False, index=union, columns=columns)
        for miss in missing:
            for symbol in columns:
                column = miss[symbol].astype(bool)
                group = (~column).cumsum()
                run = column.groupby(group).cumsum()
                bad[symbol] |= run.gt(int(rule["max_missing_run"]))
        ok &= ~bad.loc[index]
    return ok


def _build_universe_eligible(
    universe: Mapping[str, Any],
    loaded: Mapping[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> pd.DataFrame | None:
    """Mirror of ``universe.build_eligible_mask`` on the validated payload."""

    idx = pd.DatetimeIndex(index)
    cols = list(symbols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    calendar = (
        pd.DatetimeIndex([pd.Timestamp(day).normalize() for day in universe["trading_dates"]])
        if universe.get("trading_dates") is not None
        else None
    )
    price = _universe_panel(universe["price"], "float") if universe.get("price") is not None else None
    if universe.get("listing_age") is not None or universe.get("liquidity") is not None or universe.get("price_range") is not None:
        _universe_calendar_coverage(calendar, idx)
    if universe.get("base_universe") is not None:
        members = set(universe["base_universe"]["symbols"])
        mask &= pd.DataFrame([[symbol in members for symbol in cols]] * len(idx), index=idx, columns=cols)
    if universe.get("index_membership") is not None:
        mask &= _universe_membership_mask(universe["index_membership"], idx, cols)
    if universe.get("listing_age") is not None:
        mask &= _universe_listing_age_mask(universe["listing_age"], calendar, idx, cols)
    if universe.get("status_exclusion") is not None:
        mask &= _universe_status_mask(universe["status_exclusion"], idx, cols)
    if universe.get("price_limit") is not None:
        mask &= _universe_price_limit_mask(universe["price_limit"], price, idx, cols)
    if universe.get("liquidity") is not None:
        mask &= _universe_liquidity_mask(universe["liquidity"], calendar, idx, cols)
    if universe.get("price_range") is not None:
        mask &= _universe_price_range_mask(universe["price_range"], price, calendar, idx, cols)
    if universe.get("data_quality") is not None:
        mask &= _universe_data_quality_mask(universe["data_quality"], price, loaded, idx, cols)
    return mask


# ---------------------------------------------------------------------------
# Result-report audit operators (roadmap 9.3 / 9.4 / 10.2 / 10.3 / 10.4).
#
# Read-only consumers of the executed portfolio: they produce report fields in
# the result and never change portfolio behaviour, so the audit configuration
# travels at the request level and the result carries ``audit_report`` only
# when the request asks for one.  The functions below mirror
# ``rdagent/scenarios/gf_quant/strategy/audit.py`` verbatim, including every
# error message; the parity tests feed identical inputs to both sides.
# ---------------------------------------------------------------------------

AUDIT_KEYS = frozenset(
    {
        "concentration",
        "cost_sensitivity",
        "parameter_perturbation",
        "ablation",
        "stress",
        "feasibility",
        "data_leakage",
    }
)

DEFAULT_COST_SENSITIVITY_SCENARIOS = ((10.0, 10.0), (15.0, 15.0), (20.0, 20.0))

DEFAULT_CONCENTRATION_TOP_K = (5, 10)

STRESS_MODES = ("warn", "reject")

DEFAULT_STRESS_MODE = "warn"

_UNKNOWN_INDUSTRY = "__no_industry__"

_PERTURBATION_SECTIONS = ("composition", "backtest", "signal", "gating")

_PERTURBATION_METRIC_KEYS = (
    "total_return",
    "annualized_return",
    "cagr",
    "annualized_vol",
    "sharpe",
    "max_drawdown",
    "annual_turnover",
)

FEASIBILITY_REASONS = ("participation", "staging", "tradability", "lot_rounding")

DEFAULT_ALLOW_CASH_RESIDUAL = True

LEAKAGE_SEVERITIES = ("high", "medium", "low", "unknown")

LEAKAGE_UNKNOWN_SEVERITY = "unknown"

LEAKAGE_BREACH_SEVERITY = "high"

LEAKAGE_UNKNOWN_RULE_SEVERITY = "medium"

LEAKAGE_FIELD_LOCATIONS = (
    ("index_membership", "universe_rules.index_membership.lag_days"),
    ("listing_age", "universe_rules.listing_age.min_days"),
    ("signal_lag", "backtest.signal_lag"),
    ("regime_lag", "signal.regime_lag"),
)


def _validate_audit(audit: Mapping[str, Any]) -> None:
    if not isinstance(audit, Mapping):
        raise ValueError("request audit must be an object")
    unknown = set(audit) - AUDIT_KEYS
    if unknown:
        raise ValueError(f"request audit has unknown keys: {sorted(unknown)[:3]}")
    for section_name in (
        "concentration",
        "cost_sensitivity",
        "parameter_perturbation",
        "ablation",
        "stress",
        "feasibility",
        "data_leakage",
    ):
        if section_name in audit and not isinstance(audit[section_name], Mapping):
            raise ValueError(f"request audit section {section_name!r} must be an object")
    if "concentration" in audit:
        section = audit["concentration"]
        top_k = section.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, (list, tuple)) or not top_k:
                raise ValueError("audit concentration top_k must be a non-empty list of positive integers")
            for value in top_k:
                if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                    raise ValueError("audit concentration top_k must be a non-empty list of positive integers")
        industry_map = section.get("industry_map")
        industry_threshold = section.get("industry_threshold")
        if (industry_map is None) != (industry_threshold is None):
            raise ValueError("audit concentration requires industry_map and industry_threshold together")
        if industry_map is not None:
            if not isinstance(industry_map, Mapping) or not industry_map:
                raise ValueError("audit concentration industry_map must be a non-empty object")
            for symbol, label in industry_map.items():
                if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                    raise ValueError(
                        "audit concentration industry_map must map non-empty symbols to non-empty industry labels"
                    )
            if (
                isinstance(industry_threshold, bool)
                or not isinstance(industry_threshold, (int, float))
                or not math.isfinite(float(industry_threshold))
                or not 0 < float(industry_threshold) <= 1
            ):
                raise ValueError("audit concentration industry_threshold must be in (0, 1]")
    if "cost_sensitivity" in audit:
        scenarios = audit["cost_sensitivity"].get("scenarios")
        if scenarios is None:
            return
        if not isinstance(scenarios, (list, tuple)) or not scenarios:
            raise ValueError(
                "audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs"
            )
        for pair in scenarios:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    "audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs"
                )
            for value in pair:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ValueError(
                        "audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs"
                    )
    if "parameter_perturbation" in audit:
        perturbations = audit["parameter_perturbation"].get("perturbations")
        if not isinstance(perturbations, Mapping) or not perturbations:
            raise ValueError("audit parameter_perturbation requires a non-empty perturbations object")
        for key, values in perturbations.items():
            if not isinstance(key, str) or not key:
                raise ValueError("audit parameter perturbation keys must be non-empty strings")
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"audit parameter perturbation values must be a non-empty list for key {key!r}")
            for value in values:
                if isinstance(value, bool) or isinstance(value, str):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    continue
                raise ValueError("audit parameter perturbation values must be finite scalars (number, boolean or string)")
    if "ablation" in audit:
        mode = audit["ablation"].get("mode", "factors")
        if mode != "factors":
            raise ValueError("audit ablation mode must be 'factors'")
    if "stress" in audit:
        _validate_stress(audit["stress"])
    if "feasibility" in audit:
        _validate_feasibility(audit["feasibility"])
    if "data_leakage" in audit:
        _validate_data_leakage(audit["data_leakage"])


def _validate_lag_table(value: Any, message: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(message)
    for name, days in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(message)
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError(message)


def _validate_feasibility(section: Mapping[str, Any]) -> None:
    ratio = section.get("min_executable_ratio")
    if ratio is not None:
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0 < float(ratio) <= 1
        ):
            raise ValueError("audit feasibility min_executable_ratio must be in (0, 1]")
    allow = section.get("allow_cash_residual", DEFAULT_ALLOW_CASH_RESIDUAL)
    if not isinstance(allow, bool):
        raise ValueError("audit feasibility allow_cash_residual must be a boolean")


def _validate_data_leakage(section: Mapping[str, Any]) -> None:
    _validate_lag_table(
        section.get("field_availability_lag"),
        "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
    )
    _validate_lag_table(
        section.get("announcement_to_effective"),
        "audit data_leakage announcement_to_effective must map non-empty rule names to non-negative whole days",
    )


def _validate_stress(section: Mapping[str, Any]) -> None:
    mode = section.get("mode", DEFAULT_STRESS_MODE)
    if mode not in STRESS_MODES:
        raise ValueError("audit stress mode must be 'warn' or 'reject'")
    version = section.get("scenario_table_version")
    if not isinstance(version, str) or not version:
        raise ValueError("audit stress requires a non-empty scenario_table_version")
    scenarios = section.get("scenarios")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("audit stress requires a non-empty scenarios object")
    needs_industry_map = False
    for scenario_id, shocks in scenarios.items():
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("audit stress scenario identifiers must be non-empty strings")
        if not isinstance(shocks, Mapping):
            raise ValueError(f"audit stress scenario {scenario_id!r} must be an object")
        if not set(shocks) & {"symbol_shocks", "industry_shocks", "default_shock"}:
            raise ValueError(
                f"audit stress scenario {scenario_id!r} must provide symbol_shocks, industry_shocks or default_shock"
            )
        for field in ("symbol_shocks", "industry_shocks"):
            if field not in shocks:
                continue
            table = shocks[field]
            if not isinstance(table, Mapping) or not table:
                raise ValueError(f"audit stress scenario {scenario_id!r} {field} must be a non-empty object")
            for shock_key, shock_value in table.items():
                if not isinstance(shock_key, str) or not shock_key:
                    raise ValueError(f"audit stress scenario {scenario_id!r} {field} keys must be non-empty strings")
                if (
                    isinstance(shock_value, bool)
                    or not isinstance(shock_value, (int, float))
                    or not math.isfinite(float(shock_value))
                ):
                    raise ValueError(f"audit stress scenario {scenario_id!r} {field} values must be finite numbers")
            needs_industry_map = needs_industry_map or field == "industry_shocks"
        if "default_shock" in shocks:
            default_shock = shocks["default_shock"]
            if (
                isinstance(default_shock, bool)
                or not isinstance(default_shock, (int, float))
                or not math.isfinite(float(default_shock))
            ):
                raise ValueError(f"audit stress scenario {scenario_id!r} default_shock must be a finite number")
    threshold = section.get("loss_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0
    ):
        raise ValueError("audit stress loss_threshold must be a finite number >= 0")
    industry_map = section.get("industry_map")
    if needs_industry_map and industry_map is None:
        raise ValueError("audit stress industry_shocks requires industry_map")
    if industry_map is not None:
        if not isinstance(industry_map, Mapping) or not industry_map:
            raise ValueError("audit stress industry_map must be a non-empty object")
        for symbol, label in industry_map.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                raise ValueError("audit stress industry_map must map non-empty symbols to non-empty industry labels")
    scenario_ids = section.get("scenario_ids")
    if scenario_ids is not None:
        if not isinstance(scenario_ids, (list, tuple)) or not scenario_ids:
            raise ValueError("audit stress scenario_ids must be a non-empty list of scenario identifiers")
        for scenario_id in scenario_ids:
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError("audit stress scenario_ids must be a non-empty list of scenario identifiers")
        missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenarios]
        if missing:
            raise ValueError(f"audit stress scenario_ids must all be present in the scenario table: {missing}")


def _resolve_perturbation_target(payload: Mapping[str, Any], dotted_key: str) -> Any:
    """Fail-closed resolution of a dotted perturbation key in the payload.

    A key that is not present (lazy fields such as ``inverse_vol_window``
    under another weighting) must abort the audit instead of perturbing
    nothing and reporting a fake "not sensitive" result.
    """

    if not isinstance(dotted_key, str) or not dotted_key:
        raise ValueError("audit parameter perturbation keys must be non-empty strings")
    segments = dotted_key.split(".")
    if segments[0] not in _PERTURBATION_SECTIONS:
        raise ValueError(
            f"parameter perturbation keys must target one of composition/backtest/signal/gating: {dotted_key!r}"
        )
    node: Any = payload
    for segment in segments:
        if not isinstance(node, Mapping) or segment not in node:
            raise ValueError(f"parameter perturbation key {dotted_key!r} is not present in the candidate payload")
        node = node[segment]
    return node


def _concentration_report(
    weights: pd.DataFrame,
    *,
    top_k: Sequence[int],
    industry_map: Mapping[str, str] | None = None,
    industry_threshold: float | None = None,
) -> dict[str, Any]:
    """ConcentrationMonitor (roadmap 9.4): mirror of ``audit.concentration_report``."""

    frame = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = frame.to_numpy(dtype=float)
    hhi = np.square(values).sum(axis=1)
    effective = np.where(hhi > 0.0, 1.0 / np.where(hhi > 0.0, hhi, 1.0), 0.0)
    top_k_shares = {int(k): np.sort(values, axis=1)[:, -int(k) :].sum(axis=1) for k in top_k}
    report: dict[str, Any] = {
        "hhi": {"mean": float(hhi.mean()), "max": float(hhi.max()), "min": float(hhi.min())},
        "effective_names": {"mean": float(effective.mean()), "max": float(effective.max())},
        "top_k": {str(k): {"mean": float(share.mean()), "max": float(share.max())} for k, share in top_k_shares.items()},
        "by_date": {
            "dates": [date.strftime("%Y-%m-%d") for date in frame.index],
            "hhi": [round(float(value), 10) for value in hhi],
            "top_k": {str(k): [round(float(value), 10) for value in share] for k, share in top_k_shares.items()},
        },
    }
    if industry_map is not None:
        labels = pd.Series(
            [
                str(industry_map[symbol]) if symbol in industry_map and industry_map[symbol] else _UNKNOWN_INDUSTRY
                for symbol in frame.columns
            ],
            index=frame.columns,
        )
        grouped = frame.T.groupby(labels).sum().T
        max_share = grouped.max(axis=1)
        report["industry"] = {
            "threshold": float(industry_threshold),
            "max_share": {"mean": float(max_share.mean()), "max": float(max_share.max())},
            "breach_count": int((max_share > float(industry_threshold)).sum()),
            "by_date": [round(float(value), 10) for value in max_share],
        }
    return report


def _stress_scenario_report(weights: pd.DataFrame, section: Mapping[str, Any]) -> dict[str, Any]:
    """StressScenarioCheck (roadmap 9.3): mirror of ``audit.stress_scenario_report``."""

    version = section.get("scenario_table_version")
    mode = section.get("mode", DEFAULT_STRESS_MODE)
    threshold = float(section.get("loss_threshold"))
    table = section.get("scenarios") or {}
    selected = list(section.get("scenario_ids") or table)
    industry_map = section.get("industry_map") or {}
    frame = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if frame.empty or not len(frame.columns):
        book: dict[str, float] = {}
        as_of: Any = None
    else:
        book = {str(symbol): float(weight) for symbol, weight in frame.iloc[-1].items() if float(weight) != 0.0}
        as_of = frame.index[-1]
    rows: list[dict[str, Any]] = []
    for scenario_id in selected:
        shocks = table[scenario_id]
        symbol_shocks = shocks.get("symbol_shocks") or {}
        industry_shocks = shocks.get("industry_shocks") or {}
        default_shock = shocks.get("default_shock")
        portfolio_return = 0.0
        explicit_coverage = 0.0
        for symbol, weight in book.items():
            if symbol in symbol_shocks:
                shock = float(symbol_shocks[symbol])
                explicit_coverage += weight
            elif symbol in industry_map and industry_map[symbol] in industry_shocks:
                shock = float(industry_shocks[industry_map[symbol]])
                explicit_coverage += weight
            elif default_shock is not None:
                shock = float(default_shock)
            else:
                raise ValueError(f"stress scenario {scenario_id!r} does not cover held symbol {symbol!r}")
            portfolio_return += weight * shock
        loss = -portfolio_return
        rows.append(
            {
                "scenario_id": scenario_id,
                "portfolio_return": portfolio_return,
                "loss": loss,
                "explicit_coverage": explicit_coverage,
                "default_shock": None if default_shock is None else float(default_shock),
                "passed": loss <= threshold,
            }
        )
    worst = max(rows, key=lambda row: row["loss"]) if rows else None
    return {
        "scenario_table_version": version,
        "mode": mode,
        "loss_threshold": threshold,
        "as_of": None if as_of is None else pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        "held_names": len(book),
        "passed": all(row["passed"] for row in rows),
        "max_loss": 0.0 if worst is None else float(worst["loss"]),
        "worst_scenario": None if worst is None else worst["scenario_id"],
        "scenarios": rows,
    }


def _ensure_stress_passed(report: Mapping[str, Any]) -> None:
    """Mirror of ``audit.ensure_stress_passed``."""

    if report.get("mode") != "reject" or report.get("passed"):
        return
    raise ValueError(
        f"stress scenario check failed: loss {report['max_loss']!r} exceeds loss_threshold "
        f"{report['loss_threshold']!r} for scenario {report['worst_scenario']!r} "
        f"(scenario_table_version {report['scenario_table_version']!r})"
    )


def _audit_max_drawdown(returns: pd.Series) -> float:
    wealth = pd.Series([1.0, *((1.0 + returns).cumprod().to_list())], dtype=float)
    return float((wealth / wealth.cummax() - 1.0).min())


def _audit_scenario_metrics(
    daily: pd.Series, turnover: pd.Series, per_year: int, transaction_costs: pd.Series
) -> dict[str, Any]:
    """Headline metrics for one scenario return series (same formulas as
    ``_metrics`` for the reported subset, so the production-rate scenario
    reproduces the production metrics exactly)."""

    mean = float(daily.mean()) if len(daily) else 0.0
    std = float(daily.std()) if len(daily) > 1 else 0.0
    total = float((1.0 + daily).prod() - 1.0) if len(daily) else 0.0
    terminal = 1.0 + total
    cagr = float(terminal ** (per_year / len(daily)) - 1.0) if len(daily) and terminal > 0 else -1.0
    return {
        "total_return": total,
        "annualized_return": mean * per_year,
        "cagr": cagr,
        "annualized_vol": std * per_year**0.5,
        "sharpe": mean / std * per_year**0.5 if std > 0 else 0.0,
        "max_drawdown": _audit_max_drawdown(daily) if len(daily) else 0.0,
        "annual_turnover": float(turnover.mean()) * per_year if len(turnover) else 0.0,
        "total_transaction_cost": float(transaction_costs.sum()),
    }


def _cost_sensitivity_report(
    daily_gross: pd.Series,
    turnover: pd.Series,
    buy_turnover: pd.Series,
    sell_turnover: pd.Series,
    scenarios: Sequence[Sequence[float]],
    per_year: int,
) -> dict[str, Any]:
    """CostSensitivityAudit (roadmap 10.2): mirror of ``audit.cost_sensitivity_report``.

    Re-prices the already-executed trade legs under audit-only scenarios: the
    gross returns and the per-day buy/sell turnover are the strategy's real
    activity, so no strategy computation is re-run.  The production rates are
    reported as provenance and the scenarios never enter the candidate
    payload (cost keys there are rejected outright in ``_validate_request``).
    """

    rows: list[dict[str, Any]] = []
    for buy_bps, sell_bps in scenarios:
        scenario_cost = buy_turnover.astype(float) * (float(buy_bps) * 1e-4) + sell_turnover.astype(float) * (
            float(sell_bps) * 1e-4
        )
        scenario_net = daily_gross.astype(float) - scenario_cost
        row: dict[str, Any] = {"buy_bps": float(buy_bps), "sell_bps": float(sell_bps)}
        row.update(_audit_scenario_metrics(scenario_net, turnover, per_year, scenario_cost))
        rows.append(row)
    return {
        "scenarios": rows,
        "production": {"buy_bps": BUY_COST_BPS, "sell_bps": SELL_COST_BPS},
    }


def _audit_metric_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in _PERTURBATION_METRIC_KEYS}


def _parameter_perturbation_report(
    payload: Mapping[str, Any],
    perturbations: Mapping[str, Sequence[Any]],
    baseline_metrics: Mapping[str, Any],
    evaluate: Any,
) -> dict[str, Any]:
    """ParameterPerturbationAudit (roadmap 10.3): mirror of
    ``audit.parameter_perturbation_report``."""

    if not isinstance(perturbations, Mapping) or not perturbations:
        raise ValueError("audit parameter_perturbation requires a non-empty perturbations object")
    normalized: dict[str, list[Any]] = {}
    for key, values in perturbations.items():
        if not isinstance(key, str) or not key:
            raise ValueError("audit parameter perturbation keys must be non-empty strings")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"audit parameter perturbation values must be a non-empty list for key {key!r}")
        for value in values:
            if isinstance(value, bool) or isinstance(value, str):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                continue
            raise ValueError("audit parameter perturbation values must be finite scalars (number, boolean or string)")
        normalized[str(key)] = list(values)
    # Fail-closed resolution first: one lazy key aborts the whole audit.
    for key in normalized:
        _resolve_perturbation_target(payload, key)
    baseline = _audit_metric_subset(baseline_metrics)
    variants: list[dict[str, Any]] = []
    for key, values in normalized.items():
        for value in values:
            try:
                variant_metrics = _audit_metric_subset(evaluate(key, value))
            except ValueError as exc:
                variants.append({"key": key, "value": value, "status": "error", "error": str(exc)})
                continue
            variants.append(
                {
                    "key": key,
                    "value": value,
                    "status": "ok",
                    "metrics": variant_metrics,
                    "marginal": {
                        metric: variant_metrics[metric] - baseline[metric] for metric in _PERTURBATION_METRIC_KEYS
                    },
                }
            )
    return {"baseline": {"metrics": baseline}, "variants": variants}


def _ablation_report(base_metrics: Mapping[str, Any], ablated_metrics_by_factor: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """AblationAudit (roadmap 10.4): mirror of ``audit.ablation_report``.

    Report assembly over the metrics the existing ungated two-argument
    ablation loop already computes; the loop itself is untouched.
    """

    factors: dict[str, Any] = {}
    for name, ablated in ablated_metrics_by_factor.items():
        factors[str(name)] = {
            "ablated": _audit_metric_subset(ablated),
            "marginal": {
                "marginal_annualized_return": float(base_metrics["annualized_return"])
                - float(ablated["annualized_return"]),
                "marginal_sharpe": float(base_metrics["sharpe"]) - float(ablated["sharpe"]),
                "marginal_drawdown_improvement": abs(float(ablated["max_drawdown"]))
                - abs(float(base_metrics["max_drawdown"])),
                "marginal_turnover_reduction": float(ablated["annual_turnover"]) - float(base_metrics["annual_turnover"]),
            },
        }
    return {"mode": "factors", "factors": factors}


# ---------------------------------------------------------------------------
# WalkForwardEvaluation (roadmap 10.1).
#
# Unlike the sections above this one does re-run the engine -- once per
# segment -- so its configuration rides at the request level *next to*
# ``audit`` and ``evaluation`` rather than inside ``audit``: the audit object's
# documented contract is that it never changes portfolio behaviour, and a
# segmented re-run is a different kind of work.  Either way the configuration
# stays out of the candidate payload, so enabling it cannot move a strategy
# identity hash, and a request without the section walks the pre-batch path
# bit for bit.
#
# The splitter below is a port of
# ``rdagent/scenarios/gf_quant/split.py``'s rolling splitter (the worker
# imports no project package), not a verbatim copy: the date helper becomes
# ``_walk_forward_date``, the result is a plain tuple tree instead of
# ``DateSplit``/``Period`` objects, and ``_ordered_dates``' empty-input guard
# is inlined.  The window arithmetic is character for character the same.
# ---------------------------------------------------------------------------

WALK_FORWARD_KEYS = frozenset(
    {
        "train_size",
        "valid_size",
        "test_size",
        "step_size",
        "expanding_train",
        "min_oos_segments",
        "max_segments",
    }
)

DEFAULT_MAX_SEGMENTS = 8

DEFAULT_MIN_OOS_SEGMENTS = 1

DEFAULT_VALID_SIZE = 1

#: Keys of the request-level ``manifest`` object (roadmap 10.8).  Empty by
#: design: every parameter the roadmap lists is a fact for the worker to
#: collect, never a caller setting, so the object is a bare switch that asks
#: for the worker-side attestation and rejects any key the caller invents.
MANIFEST_KEYS = frozenset()

#: Schema version of the worker attestation layout.  A semantic label: it says
#: which layout produced the record, and is excluded from reproducibility
#: equality on the local side.
MANIFEST_ATTESTATION_SCHEMA_VERSION = 1

WALK_FORWARD_METRIC_KEYS = (
    "total_return",
    "annualized_return",
    "cagr",
    "annualized_vol",
    "sharpe",
    "max_drawdown",
    "annual_turnover",
)


def _build_rolling_splits(
    trading_dates: Sequence[Any],
    *,
    train_size: int,
    valid_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding_train: bool = True,
) -> list[tuple[tuple[date, date], tuple[date, date], tuple[date, date]]]:
    """Mirror of ``split.build_rolling_splits``: returns ``(train, valid, test)``.

    Boundaries land on observed trading dates and are counts of those dates,
    exactly like the local splitter, so the two ends cannot drift apart on a
    window edge.
    """

    dates = sorted({_walk_forward_date(value) for value in trading_dates})
    if not dates:
        raise ValueError("At least one trading date is required")
    sizes = (train_size, valid_size, test_size)
    if any(size <= 0 for size in sizes):
        raise ValueError("All window sizes must be positive")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    total = sum(sizes)
    if len(dates) < total:
        raise ValueError(f"Need at least {total} dates, got {len(dates)}")

    result: list[tuple] = []
    offset = 0
    while offset + total <= len(dates):
        train_start = 0 if expanding_train else offset
        train_end = offset + train_size - 1
        valid_start = train_end + 1
        valid_end = valid_start + valid_size - 1
        test_start = valid_end + 1
        test_end = test_start + test_size - 1
        result.append(
            (
                (dates[train_start], dates[train_end]),
                (dates[valid_start], dates[valid_end]),
                (dates[test_start], dates[test_end]),
            )
        )
        offset += step
    return result


def _walk_forward_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _positive_walk_forward_int(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
        raise ValueError(message)
    return int(value)


def _validate_walk_forward(walk_forward: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror of ``walk_forward.validate_walk_forward_config``."""

    if not isinstance(walk_forward, Mapping):
        raise ValueError("request walk_forward must be an object")
    unknown = set(walk_forward) - WALK_FORWARD_KEYS
    if unknown:
        raise ValueError(f"request walk_forward has unknown keys: {sorted(unknown)[:3]}")

    if "train_size" not in walk_forward:
        raise ValueError("walk_forward train_size must be a positive integer")
    train_size = _positive_walk_forward_int(walk_forward["train_size"], "walk_forward train_size must be a positive integer")

    if "test_size" not in walk_forward:
        raise ValueError("walk_forward test_size must be a positive integer")
    test_size = _positive_walk_forward_int(walk_forward["test_size"], "walk_forward test_size must be a positive integer")

    valid_size = _positive_walk_forward_int(
        walk_forward.get("valid_size", DEFAULT_VALID_SIZE),
        "walk_forward valid_size must be a positive integer",
    )

    step_size = walk_forward.get("step_size")
    if step_size is None:
        step_size = test_size
    else:
        step_size = _positive_walk_forward_int(step_size, "walk_forward step_size must be a positive integer")

    min_oos_segments = _positive_walk_forward_int(
        walk_forward.get("min_oos_segments", DEFAULT_MIN_OOS_SEGMENTS),
        "walk_forward min_oos_segments must be a positive integer",
    )
    max_segments = _positive_walk_forward_int(
        walk_forward.get("max_segments", DEFAULT_MAX_SEGMENTS),
        "walk_forward max_segments must be a positive integer",
    )
    expanding_train = walk_forward.get("expanding_train", True)
    if not isinstance(expanding_train, bool):
        raise ValueError("walk_forward expanding_train must be a boolean")

    return {
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size,
        "step_size": step_size,
        "expanding_train": expanding_train,
        "min_oos_segments": min_oos_segments,
        "max_segments": max_segments,
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("request manifest must be an object")
    unknown = set(manifest) - MANIFEST_KEYS
    if unknown:
        raise ValueError(f"request manifest has unknown keys: {sorted(unknown)[:3]}")


def _worker_file_digest(path: Path) -> str:
    """SHA-256 of the worker source file: the hard identity of the code that ran."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_runtime_snapshot() -> dict[str, str]:
    """Version strings of the stack the engine actually runs on."""

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def _build_manifest_section(
    request: Mapping[str, Any],
    close: pd.DataFrame,
    *,
    worker_path: Path | None = None,
) -> dict[str, Any]:
    """Worker-side attestation: what code, what environment, what data window.

    Every field here is observed by this process -- the file digest is read
    from disk, the runtime versions from the imported modules, the calendar
    from the panel the run actually consumed, and the request checksum from
    the request this run was handed.  Nothing is taken on the caller's word.
    """

    path = worker_path or Path(__file__).resolve()
    evaluation = request.get("evaluation", {})
    start, end = str(evaluation.get("start", "")), str(evaluation.get("end", ""))
    calendar = close.loc[pd.Timestamp(start) : pd.Timestamp(end)].index if start and end else close.index
    return {
        "attestation_schema_version": MANIFEST_ATTESTATION_SCHEMA_VERSION,
        "worker_file_sha256": _worker_file_digest(path),
        "worker_file_name": path.name,
        "worker_runtime": _worker_runtime_snapshot(),
        "trading_calendar": {
            "first_date": calendar[0].strftime("%Y-%m-%d") if len(calendar) else None,
            "last_date": calendar[-1].strftime("%Y-%m-%d") if len(calendar) else None,
            "rows": int(len(calendar)),
        },
        "request_checksum": hashlib.sha256(_canonical_json(request).encode()).hexdigest(),
    }


def _walk_forward_mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row["metrics"][key]) for row in rows) / len(rows))


def _walk_forward_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "segments": len(rows),
        "mean_metrics": {key: _walk_forward_mean_metric(rows, key) for key in WALK_FORWARD_METRIC_KEYS},
        "positive_return_ratio": (
            float(sum(1 for row in rows if float(row["metrics"]["total_return"]) > 0.0) / len(rows)) if rows else 0.0
        ),
        "observations": int(sum(int(row["metrics"]["observations"]) for row in rows)),
    }


def _walk_forward_report(
    config: Mapping[str, Any],
    trading_dates: Sequence[Any],
    evaluate_segment: Any,
) -> dict[str, Any]:
    """Mirror of ``walk_forward.walk_forward_report``."""

    splits = _build_rolling_splits(
        trading_dates,
        train_size=int(config["train_size"]),
        valid_size=int(config["valid_size"]),
        test_size=int(config["test_size"]),
        step_size=int(config["step_size"]),
        expanding_train=bool(config["expanding_train"]),
    )
    if len(splits) > int(config["max_segments"]):
        raise ValueError(
            f"walk_forward would build {len(splits)} segments, more than max_segments {int(config['max_segments'])}"
        )
    if len(splits) < int(config["min_oos_segments"]):
        raise ValueError(
            f"walk_forward produced {len(splits)} OOS segments, fewer than min_oos_segments "
            f"{int(config['min_oos_segments'])}"
        )

    rows: list[dict[str, Any]] = []
    for index, (train, valid, test) in enumerate(splits):
        entry: dict[str, Any] = {
            "segment": int(index),
            "train_start": train[0].isoformat(),
            "train_end": train[1].isoformat(),
            "valid_start": valid[0].isoformat(),
            "valid_end": valid[1].isoformat(),
            "test_start": test[0].isoformat(),
            "test_end": test[1].isoformat(),
            "step_size": int(config["step_size"]),
            "expanding_train": bool(config["expanding_train"]),
        }
        entry["train"] = {
            "window": [train[0].isoformat(), train[1].isoformat()],
            "in_sample": True,
            "metrics": dict(evaluate_segment("train", train[0], train[1])),
        }
        entry["valid"] = {
            "window": [valid[0].isoformat(), valid[1].isoformat()],
            "in_sample": True,
            "metrics": dict(evaluate_segment("valid", valid[0], valid[1])),
        }
        entry["test"] = {
            "window": [test[0].isoformat(), test[1].isoformat()],
            "in_sample": False,
            "metrics": dict(evaluate_segment("test", test[0], test[1])),
        }
        rows.append(entry)

    test_rows = [{"metrics": row["test"]["metrics"]} for row in rows]
    train_rows = [{"metrics": row["train"]["metrics"]} for row in rows]
    return {
        "mode": "rolling",
        "fit_on_train": False,
        "parameter_refit": False,
        "train_size": int(config["train_size"]),
        "valid_size": int(config["valid_size"]),
        "test_size": int(config["test_size"]),
        "step_size": int(config["step_size"]),
        "expanding_train": bool(config["expanding_train"]),
        "min_oos_segments": int(config["min_oos_segments"]),
        "max_segments": int(config["max_segments"]),
        "segments": rows,
        "summary": {
            "oos": _walk_forward_aggregate(test_rows),
            "is": _walk_forward_aggregate(train_rows),
            "min_oos_segments_met": len(rows) >= int(config["min_oos_segments"]),
        },
    }


def _evaluate_candidate(
    request: Mapping[str, Any],
    loaded: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    include_contributions: bool = True,
) -> dict[str, Any]:
    """Pipeline for one candidate over the pre-loaded panels.

    The body of ``run_request`` between validation and artifact writing,
    extracted verbatim so the parameter-perturbation audit can re-run the
    pipeline for perturbed candidate copies against the same data.  The
    ungated two-argument factor-ablation loop lives here unchanged; its
    ablated metrics are collected alongside ``factor_contributions`` so the
    ablation audit can formalise them without re-running anything.
    """

    candidate = request["candidate"]
    evaluation = request["evaluation"]
    start, end = str(evaluation["start"]), str(evaluation["end"])
    calendar = close.loc[pd.Timestamp(start) : pd.Timestamp(end)].index
    if calendar.empty or calendar.min() > pd.Timestamp(start) or calendar.max() < pd.Timestamp(end):
        raise ValueError("close/benchmark data do not cover evaluation window")
    rebalance_dates = _rebalance_dates(calendar, str(evaluation.get("rebalance_frequency", "monthly")))
    directions = request["factor_directions"]
    common_symbols = set(close.columns)
    for name in candidate["factor_names"]:
        common_symbols &= set(loaded[name].columns)
    if not common_symbols:
        raise ValueError("factor and return panels have no common symbols")
    symbols = sorted(common_symbols)
    signal = candidate.get("signal") or {}
    # The candidate-pool mask is baked into the rank panels, so the main
    # path, the gating stage and the ungated ablation loop all see the same
    # post-filter universe (unlike the loop-state selection operators, this
    # is a pure data-domain assumption).
    universe_rules = candidate.get("universe_rules")
    eligible = (
        _build_universe_eligible(universe_rules, loaded, rebalance_dates, symbols)
        if universe_rules is not None
        else None
    )
    ranks = _build_rank_panels(loaded, directions, signal, rebalance_dates, symbols, eligible=eligible)

    daily_returns = close.reindex(columns=symbols).pct_change(fill_method=None).loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if daily_returns.index.difference(calendar).size or daily_returns.empty:
        raise ValueError("return panel does not cover evaluation calendar")
    benchmark = daily_returns.mean(axis=1, skipna=True)
    if benchmark.isna().any():
        raise ValueError("benchmark does not cover every evaluation day")

    # FactorDecayMonitor (roadmap 9.5) rides on the signal section: enabled by
    # decay_window, it reads each factor's trailing rank IC over the holding
    # window its decision actually earned (signal_lag days after the decision)
    # and scales that factor's blend contribution.  The two-argument ablation
    # call below never receives it (the ungated ablation contract).
    decay = None
    if signal.get("decay_window") is not None:
        decay = _factor_decay_scales(
            ranks,
            daily_returns,
            lag=int(candidate["backtest"].get("signal_lag") or 1),
            window=int(signal["decay_window"]),
            fail_periods=int(signal.get("decay_fail_periods") or 1),
            threshold=float(signal.get("decay_ic_threshold") or 0.0),
            scale=0.0 if signal.get("decay_scale") is None else float(signal["decay_scale"]),
        )
    full_score = _score(ranks, candidate["factor_names"], signal, decay=decay)
    gating = candidate.get("gating")
    if gating:
        full_score = _apply_gating(full_score, ranks, gating)
    targets = _target_weights(full_score, candidate["composition"], daily_returns)
    composition = candidate["composition"]
    # HoldingBuffer / DiversitySelector need the score panel and the executed
    # holdings state inside the main loop, so they are passed only when
    # enabled; the disabled path keeps the pre-change call shape exactly.
    has_selection_operators = composition.get("holding_buffer_entry") is not None or (
        composition.get("diversity_max_correlation") is not None
    )
    # FeasibilityAudit (10.6) needs the execution chain's per-stage books, which
    # only the run that actually placed the orders can produce; the trace is
    # therefore collected on the base run and reused by the report assembly
    # rather than re-running the pipeline.
    execution_trace: list[dict[str, Any]] | None = None
    requested_audit = request.get("audit")
    if isinstance(requested_audit, Mapping) and "feasibility" in requested_audit:
        execution_trace = []
    full = _backtest(
        targets,
        daily_returns,
        candidate["backtest"],
        benchmark,
        composition=composition if has_selection_operators else None,
        score=full_score if has_selection_operators else None,
        execution_data=candidate.get("execution_data"),
        execution_trace=execution_trace,
    )
    metrics, metrics_by_year, net, gross, turnover, weights, buy_turnover, sell_turnover = full

    contributions: dict[str, Any] = {}
    ablated_metrics: dict[str, dict[str, Any]] = {}
    if include_contributions:
        for removed in candidate["factor_names"]:
            kept = [name for name in candidate["factor_names"] if name != removed]
            ablation_targets = _target_weights(_score(ranks, kept), candidate["composition"], daily_returns)
            ablation_metrics = _backtest(
                ablation_targets,
                daily_returns,
                candidate["backtest"],
                benchmark,
                execution_data=candidate.get("execution_data"),
            )[0]
            ablated_metrics[removed] = ablation_metrics
            contributions[removed] = {
                "marginal_annualized_return": metrics["annualized_return"] - ablation_metrics["annualized_return"],
                "marginal_sharpe": metrics["sharpe"] - ablation_metrics["sharpe"],
                "marginal_drawdown_improvement": abs(ablation_metrics["max_drawdown"]) - abs(metrics["max_drawdown"]),
                "marginal_turnover_reduction": ablation_metrics["annual_turnover"] - metrics["annual_turnover"],
            }
    return {
        "metrics": metrics,
        "metrics_by_year": metrics_by_year,
        "net": net,
        "gross": gross,
        "turnover": turnover,
        "weights": weights,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "contributions": contributions,
        "ablated_metrics": ablated_metrics,
        "rebalance_dates": rebalance_dates,
        "symbols": symbols,
        "execution_trace": execution_trace,
    }


def _feasibility_report(
    days: Sequence[Mapping[str, Any]],
    *,
    min_executable_ratio: float | None = None,
    allow_cash_residual: bool = DEFAULT_ALLOW_CASH_RESIDUAL,
) -> dict[str, Any]:
    """FeasibilityAudit (10.6): report-only, never rewrites a weight."""

    per_day: list[dict[str, Any]] = []
    ratios: list[float] = []
    reason_totals = {reason: 0.0 for reason in FEASIBILITY_REASONS}
    theoretical_total = 0.0
    executed_total = 0.0
    unfilled_total = 0.0
    violations: list[dict[str, Any]] = []
    for row in days:
        date = str(row["date"])
        theoretical = float(row["theoretical"])
        executed = float(row["executed"])
        reasons = {reason: float(row.get(reason, 0.0)) for reason in FEASIBILITY_REASONS}
        unfilled = theoretical - executed
        residual = float(row.get("cash_residual", 0.0))
        entry: dict[str, Any] = {
            "date": date,
            "theoretical": theoretical,
            "executed": executed,
            "unfilled": unfilled,
            "executable_ratio": (executed / theoretical) if theoretical > 0.0 else None,
            "cash_residual": residual,
            "unfilled_by_reason": reasons,
        }
        for reason, amount in reasons.items():
            reason_totals[reason] += amount
        theoretical_total += theoretical
        executed_total += executed
        unfilled_total += unfilled
        if entry["executable_ratio"] is not None:
            ratios.append(float(entry["executable_ratio"]))
            if min_executable_ratio is not None and entry["executable_ratio"] < float(min_executable_ratio):
                # Report-only: a short day is recorded and nothing else.
                # Rewriting the book here would turn a reporter into an
                # executor, which is exactly what this audit must not be.
                violations.append(
                    {
                        "kind": "min_executable_ratio",
                        "date": date,
                        "executable_ratio": float(entry["executable_ratio"]),
                        "required": float(min_executable_ratio),
                    }
                )
        if not allow_cash_residual and (unfilled > 0.0 or residual > 0.0):
            violations.append({"kind": "cash_residual_not_allowed", "date": date, "unfilled": unfilled, "cash_residual": residual})
        per_day.append(entry)

    return {
        "allow_cash_residual": bool(allow_cash_residual),
        "min_executable_ratio": None if min_executable_ratio is None else float(min_executable_ratio),
        "per_day": per_day,
        "summary": {
            "days": len(per_day),
            "scored_days": len(ratios),
            "theoretical_total": theoretical_total,
            "executed_total": executed_total,
            "unfilled_total": unfilled_total,
            "mean_executable_ratio": (sum(ratios) / len(ratios)) if ratios else None,
            "min_executable_ratio": min(ratios) if ratios else None,
            "unfilled_by_reason": reason_totals,
        },
        "violations": violations,
    }


def _attribute_execution_shortfall(
    *,
    carried: Mapping[str, float],
    target: Mapping[str, float],
    executed: Mapping[str, float],
    stages: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Split ``target - executed`` across the execution chain's own stages."""

    def cash(row: Mapping[str, float]) -> float:
        return 1.0 - float(sum(float(value) for value in row.values()))

    ordered = ("allocator", "tradability", "lot_rounding")
    previous: Mapping[str, float] = target
    out = {reason: 0.0 for reason in FEASIBILITY_REASONS}
    label_to_reason = {
        "allocator": "participation",
        "tradability": "tradability",
        "lot_rounding": "lot_rounding",
    }
    # Which allocator variant is live decides whether the allocator's withdrawal
    # is reported as participation or as staging; the caller passes the book
    # under a single ``allocator`` key either way.
    allocator_reason = str(stages.get("__allocator_reason__", "participation"))
    for label in ordered:
        row = stages[label]
        withdrawal = 0.0
        for name in set(previous) | set(row):
            before = float(previous.get(name, 0.0))
            after = float(row.get(name, 0.0))
            if before > after:
                withdrawal += before - after
        withdrawal += max(0.0, cash(previous) - cash(row))
        reason = allocator_reason if label == "allocator" else label_to_reason[label]
        out[reason] += withdrawal
        previous = row
    # Anything the stage walk did not account for (a floating-point difference
    # in the cash leg, or a stage the caller did not supply) lands in the
    # allocator bucket rather than being silently dropped.
    accounted = sum(out.values())
    total = 0.0
    for name in set(target) | set(executed):
        before = float(target.get(name, 0.0))
        after = float(executed.get(name, 0.0))
        if before > after:
            total += before - after
    total += max(0.0, cash(target) - cash(executed))
    if total - accounted > 1e-12:
        out["participation"] += total - accounted
    return out


def _feasibility_days(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce one run's execution trace to the per-day feasibility table."""

    days: list[dict[str, Any]] = []
    for entry in trace:
        stages = entry["stages"]
        if not stages:
            continue
        labels = [label for label in ("allocator", "tradability", "lot_rounding") if label in stages]
        if not labels:
            continue
        executed_row = stages[labels[-1]]
        carried_row = entry["carried"]
        target_row = entry["target"]
        names = set(target_row.index) | set(executed_row.index) | set(carried_row.index)

        def weight(row: Mapping[str, Any], name: str) -> float:
            return float(row[name])

        def leg(row: Mapping[str, Any]) -> float:
            return float(sum(weight(row, name) for name in names))

        theoretical = 0.0
        for name in names:
            theoretical += abs(weight(target_row, name) - weight(carried_row, name))
        theoretical += abs((1.0 - leg(target_row)) - (1.0 - leg(carried_row)))
        executed = 0.0
        for name in names:
            executed += abs(weight(executed_row, name) - weight(carried_row, name))
        executed += abs((1.0 - leg(executed_row)) - (1.0 - leg(carried_row)))
        reasons = _attribute_execution_shortfall(
            carried={name: weight(carried_row, name) for name in names},
            target={name: weight(target_row, name) for name in names},
            executed={name: weight(executed_row, name) for name in names},
            stages={label: {name: weight(stages[label], name) for name in names} for label in labels}
            | {"__allocator_reason__": str(stages.get("__allocator_reason__", "participation"))},
        )
        days.append(
            {
                "date": str(entry["date"])[:10],
                "theoretical": theoretical,
                "executed": min(executed, theoretical),
                "cash_residual": 1.0 - leg(executed_row),
                **reasons,
            }
        )
    return days


def _leakage_declarations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect the point-in-time declarations a candidate payload makes."""

    declarations: list[dict[str, Any]] = []
    locations = dict(LEAKAGE_FIELD_LOCATIONS)
    universe = payload.get("universe_rules") or {}
    membership = universe.get("index_membership")
    if isinstance(membership, Mapping):
        declarations.append(
            {
                "field": "index_membership",
                "location": locations["index_membership"],
                "declared_lag": int(membership.get("lag_days", 0)),
                "rule": "index_membership",
            }
        )
    listing = universe.get("listing_age")
    if isinstance(listing, Mapping):
        # ``min_days`` is a *minimum* listing age, not a publication lag: the
        # engine reads the listing date itself, and the declaration that
        # matters is how much of the panel it insists on before admitting a
        # name.  Mapping it to a lag would invent a field the engine never
        # declares, so the location is recorded with no lag at all.
        declarations.append(
            {
                "field": "listing_age",
                "location": locations["listing_age"],
                "declared_lag": None,
                "rule": "listing_age",
            }
        )
    backtest = payload.get("backtest") or {}
    declarations.append(
        {
            "field": "signal_lag",
            "location": locations["signal_lag"],
            "declared_lag": int(backtest.get("signal_lag") or 1),
            "rule": "signal_lag",
        }
    )
    signal = payload.get("signal") or {}
    if "regime_lag" in signal:
        declarations.append(
            {
                "field": "regime_lag",
                "location": locations["regime_lag"],
                "declared_lag": int(signal.get("regime_lag") or 0),
                "rule": "regime_lag",
            }
        )
    return declarations


def _data_leakage_report(
    declarations: Sequence[Mapping[str, Any]],
    *,
    field_availability_lag: Mapping[str, int],
    announcement_to_effective: Mapping[str, int],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """DataLeakageAudit (10.7): report-only look-ahead risk list."""

    risks: list[dict[str, Any]] = []
    undeclared: list[dict[str, Any]] = []
    for declaration in declarations:
        field = str(declaration["field"])
        location = str(declaration["location"])
        declared = declaration.get("declared_lag")
        rule = declaration.get("rule")
        if declared is None:
            entry = {"field": field, "location": location, "severity": LEAKAGE_UNKNOWN_SEVERITY, "declared_lag": None}
            undeclared.append(entry)
            risks.append({**entry, "required_lag": field_availability_lag.get(field), "reason": "unknown"})
            continue
        required = field_availability_lag.get(field)
        if required is None:
            entry = {
                "field": field,
                "location": location,
                "severity": LEAKAGE_UNKNOWN_SEVERITY,
                "declared_lag": int(declared),
            }
            undeclared.append(entry)
            risks.append({**entry, "required_lag": None, "reason": "unknown"})
            continue
        if int(declared) < int(required):
            risks.append(
                {
                    "field": field,
                    "location": location,
                    "severity": LEAKAGE_BREACH_SEVERITY,
                    "declared_lag": int(declared),
                    "required_lag": int(required),
                    "reason": "declared_lag_below_available_lag",
                }
            )
        if rule is not None and str(rule) not in announcement_to_effective:
            risks.append(
                {
                    "field": field,
                    "location": location,
                    "severity": LEAKAGE_UNKNOWN_RULE_SEVERITY,
                    "declared_lag": int(declared),
                    "required_lag": int(required),
                    "reason": "unknown_announcement_rule",
                    "rule": str(rule),
                }
            )
    return {
        "declarations": [dict(declaration) for declaration in declarations],
        "risks": risks,
        "undeclared": undeclared,
        "field_availability_lag": {str(key): int(value) for key, value in field_availability_lag.items()},
        "announcement_to_effective": {str(key): int(value) for key, value in announcement_to_effective.items()},
        "provenance": dict(provenance),
    }


def _build_audit_report(
    request: Mapping[str, Any],
    audit: Mapping[str, Any],
    outcome: Mapping[str, Any],
    loaded: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
) -> dict[str, Any]:
    """Compute the requested audit sections from the executed base run.

    The parameter-perturbation evaluate closure deep-copies the request,
    applies the dotted key and re-runs the full pipeline (with revalidation,
    so an invalid perturbed value surfaces as an error variant instead of a
    crash); everything else consumes the already-computed base series.
    """

    candidate = request["candidate"]

    def evaluate(dotted_key: str, value: Any) -> dict[str, Any]:
        variant = copy.deepcopy(dict(request))
        node: Any = variant["candidate"]
        segments = dotted_key.split(".")
        for segment in segments[:-1]:
            node = node[segment]
        node[segments[-1]] = value
        _validate_request(variant)
        return _evaluate_candidate(variant, loaded, close, include_contributions=False)["metrics"]

    report: dict[str, Any] = {}
    if "concentration" in audit:
        section = audit["concentration"]
        report["concentration"] = _concentration_report(
            outcome["weights"],
            top_k=list(section.get("top_k") or DEFAULT_CONCENTRATION_TOP_K),
            industry_map=section.get("industry_map"),
            industry_threshold=section.get("industry_threshold"),
        )
    if "cost_sensitivity" in audit:
        section = audit["cost_sensitivity"]
        scenarios = [tuple(pair) for pair in (section.get("scenarios") or DEFAULT_COST_SENSITIVITY_SCENARIOS)]
        per_year = int(candidate["backtest"].get("trading_days_per_year", 252))
        report["cost_sensitivity"] = _cost_sensitivity_report(
            outcome["gross"],
            outcome["turnover"],
            outcome["buy_turnover"],
            outcome["sell_turnover"],
            scenarios,
            per_year,
        )
    if "parameter_perturbation" in audit:
        report["parameter_perturbation"] = _parameter_perturbation_report(
            candidate,
            audit["parameter_perturbation"].get("perturbations"),
            outcome["metrics"],
            evaluate,
        )
    if "ablation" in audit:
        report["ablation"] = _ablation_report(outcome["metrics"], outcome["ablated_metrics"])
    if "stress" in audit:
        # StressScenarioCheck (roadmap 9.3): the scenario table is audit-level
        # data, so the section is built from the executed book and then, in
        # reject mode only, fails the whole request closed.  A rejection is an
        # audit failure, not a strategy failure: no weight is rewritten and the
        # loss figures live in the exception message.
        report["stress"] = _stress_scenario_report(outcome["weights"], audit["stress"])
        _ensure_stress_passed(report["stress"])
    if "feasibility" in audit:
        # FeasibilityAudit (10.6): the trace was collected by the base run's
        # order layer, so this section only reduces it -- no second backtest.
        section = audit["feasibility"]
        report["feasibility"] = _feasibility_report(
            _feasibility_days(outcome.get("execution_trace") or []),
            min_executable_ratio=section.get("min_executable_ratio"),
            allow_cash_residual=bool(section.get("allow_cash_residual", DEFAULT_ALLOW_CASH_RESIDUAL)),
        )
    if "data_leakage" in audit:
        # DataLeakageAudit (10.7): five declarations, never an execution change.
        section = audit["data_leakage"]
        report["data_leakage"] = _data_leakage_report(
            _leakage_declarations(candidate),
            field_availability_lag=section.get("field_availability_lag"),
            announcement_to_effective=section.get("announcement_to_effective"),
            provenance={
                "snapshot_id": request.get("snapshot_id"),
                "snapshot_checksum": request.get("snapshot_checksum"),
                "artifact_set_id": (request.get("artifact_set") or {}).get("artifact_set_id"),
                "candidate_name": candidate.get("name"),
            },
        )
    return report


def _walk_forward_evaluate_segment(request: Mapping[str, Any], loaded: Mapping[str, pd.DataFrame], close: pd.DataFrame):
    """Build the per-segment evaluator WalkForwardEvaluation calls.

    Each call narrows the *evaluation window* on a deep copy of the request
    and re-runs the ordinary pipeline through :func:`_evaluate_candidate`, so a
    segment is the frozen candidate over that window and nothing else -- no
    parameter is fitted, no result is fed back.  The candidate payload is
    untouched by construction, hence the strategy identity is unchanged.

    ``include_contributions`` is off: the ablation loop would multiply the cost
    of the most expensive audit by the number of segments for numbers the
    per-segment report does not use.
    """

    def evaluate_segment(role: str, start: date, end: date) -> dict[str, Any]:
        variant = copy.deepcopy(dict(request))
        variant["evaluation"] = dict(variant["evaluation"])
        variant["evaluation"]["start"] = start.isoformat()
        variant["evaluation"]["end"] = end.isoformat()
        outcome = _evaluate_candidate(variant, loaded, close, include_contributions=False)
        return dict(outcome["metrics"])

    return evaluate_segment


def _build_walk_forward_section(
    request: Mapping[str, Any],
    walk_forward: Mapping[str, Any],
    loaded: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
) -> dict[str, Any]:
    """WalkForwardEvaluation (roadmap 10.1): the rolling report."""

    config = _validate_walk_forward(walk_forward)
    evaluation = request["evaluation"]
    calendar = close.loc[pd.Timestamp(str(evaluation["start"])) : pd.Timestamp(str(evaluation["end"]))].index
    return _walk_forward_report(
        config,
        list(calendar),
        _walk_forward_evaluate_segment(request, loaded, close),
    )


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
    artifact_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    loaded: dict[str, pd.DataFrame] = {}
    for name in candidate["factor_names"]:
        loaded[name] = _load_factor(artifact_items[name], start, end)

    outcome = _evaluate_candidate(request, loaded, close)
    metrics = outcome["metrics"]
    net = outcome["net"]
    weights = outcome["weights"]
    contributions = outcome["contributions"]
    symbols = outcome["symbols"]
    rebalance_dates = outcome["rebalance_dates"]

    signal = candidate.get("signal") or {}
    if signal.get("blend_weights") is not None:
        score_combination = "fixed_weight_blend_of_cross_sectional_percentile_rank"
        factor_score_weights_supported = True
    elif signal.get("blend_groups") is not None:
        score_combination = "two_level_group_weight_blend_of_cross_sectional_percentile_rank"
        factor_score_weights_supported = True
    elif signal.get("regime_factor_groups") is not None:
        score_combination = "state_conditional_equal_weight_blend_of_cross_sectional_percentile_rank"
        factor_score_weights_supported = False
    else:
        score_combination = "equal_weight_cross_sectional_percentile_rank"
        factor_score_weights_supported = False

    audit = request.get("audit")
    audit_report = _build_audit_report(request, audit, outcome, loaded, close) if isinstance(audit, Mapping) else None
    # WalkForwardEvaluation (10.1) re-runs the frozen strategy once per
    # segment.  It is a report about the *same* candidate, so its sections
    # share the ``audit_report`` object: a request that asks for neither
    # section still writes a result without the key at all.
    walk_forward = request.get("walk_forward")
    if walk_forward is not None:
        if audit_report is None:
            audit_report = {}
        audit_report["walk_forward"] = _build_walk_forward_section(request, walk_forward, loaded, close)
    # StrategyManifestEmitter (10.8) rides beside walk_forward rather than
    # inside ``_build_audit_report``: that helper only sees the executed
    # outcome, while the attestation needs the worker's own file digest,
    # runtime stack and the calendar this process actually read.
    if request.get("manifest") is not None:
        if audit_report is None:
            audit_report = {}
        audit_report["manifest"] = _build_manifest_section(request, close)

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
            "metrics_by_year": outcome["metrics_by_year"],
            "return_profile": {
                "dates": [date.strftime("%Y-%m-%d") for date in net.index],
                "returns": [round(float(value), 10) for value in net],
            },
            "factor_contributions": contributions,
            "artifact_path": str(artifact_path),
            "artifact_checksum": artifact_checksum,
            "audit": {
                "evaluation": dict(evaluation),
                "execution_costs": dict(request["execution_costs"]),
                # Truthful self-description: the frozen-weight blend and the
                # two-level group blend really do take effect, so the audit
                # fields must reflect the configured score combination.
                "score_combination": score_combination,
                "factor_score_weights_supported": factor_score_weights_supported,
                "benchmark": "daily_equal_weight_investable_universe",
                "factor_count": len(candidate["factor_names"]),
                "symbol_count": len(symbols),
                "rebalance_count": len(rebalance_dates),
                "full_weights_remote_only": True,
            },
        }
    )
    if audit_report is not None:
        result["audit_report"] = _json_safe(audit_report)
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
