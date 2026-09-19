"""Result-report audit operators (roadmap 9.3 / 9.4 / 10.2 / 10.3 / 10.4).

The four audit operators are *read-only*: they consume an already-executed
portfolio backtest (start-of-day weights, per-day gross/net returns, per-day
buy/sell turnover) and produce report fields.  None of them changes portfolio
behaviour, so none of them belongs to the candidate payload -- the audit
configuration travels at the request level (next to ``evaluation``) and a
request without an ``audit`` object produces a result without the
``audit_report`` field.  Scenario cost rates and perturbation values can
therefore never leak into a strategy identity hash, and the worker's hard
rejection of cost keys inside ``candidate["backtest"]`` stays intact.

The standalone worker mirrors every function here verbatim
(``scripts/gf_quant_strategy_worker.py``); the parity tests feed identical
inputs to both implementations and compare bit for bit.  Error messages in
this module are part of the two-side contract and must stay character-for-
character identical to the worker's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from rdagent.scenarios.gf_quant.strategy.costs import BUY_COST_BPS, SELL_COST_BPS

#: Audit sections of the request-level ``audit`` object.
AUDIT_SECTIONS: frozenset[str] = frozenset(
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

#: Documented default audit scenarios: 10/15/20 bps each way (roadmap 10.2).
DEFAULT_COST_SENSITIVITY_SCENARIOS: tuple[tuple[float, float], ...] = ((10.0, 10.0), (15.0, 15.0), (20.0, 20.0))

#: Documented default concentration indicators: Top-5 / Top-10 (roadmap 9.4).
DEFAULT_CONCENTRATION_TOP_K: tuple[int, ...] = (5, 10)

#: Stress modes (roadmap 9.3): ``warn`` reports only, ``reject`` fails the
#: whole audit closed.  Neither mode rewrites a single weight.
STRESS_MODES: tuple[str, ...] = ("warn", "reject")

#: Declared default stress mode: report only, never abort an evaluation.
DEFAULT_STRESS_MODE = "warn"

#: Reserved group for symbols the audit industry map does not cover (same
#: convention as the composition-layer industry cap).
_UNKNOWN_INDUSTRY = "__no_industry__"

#: Perturbation keys are dotted paths into the canonical candidate payload and
#: must start with one of these sections.
PERTURBATION_SECTIONS: tuple[str, ...] = ("composition", "backtest", "signal", "gating")

#: Headline metrics reported per perturbation variant (identical on both ends).
PERTURBATION_METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "annualized_return",
    "cagr",
    "annualized_vol",
    "sharpe",
    "max_drawdown",
    "annual_turnover",
)

#: FeasibilityAudit (roadmap 10.6) un-filled reason buckets, in the order the
#: engine's own execution chain applies them: the allocator (participation cap
#: or staged execution) slices first, the side-aware mask clamps the slice, and
#: the lot grid rounds it last.  ``participation`` and ``staging`` are the two
#: mutually exclusive allocator variants (8.6 vs 8.9), so at most one of them is
#: ever non-zero for one run.
FEASIBILITY_REASONS: tuple[str, ...] = ("participation", "staging", "tradability", "lot_rounding")

#: Declared default for ``feasibility.allow_cash_residual``: reporting an
#: unfilled amount never fails an audit unless the caller asks for it.
DEFAULT_ALLOW_CASH_RESIDUAL = True

#: DataLeakageAudit (roadmap 10.7) severity vocabulary.  ``unknown`` is a
#: severity of its own rather than a grade: a field the payload says nothing
#: about cannot be called a breach *or* cleared, and folding it into a numeric
#: tier would let a reader mistake "not checked" for "checked and mild".
LEAKAGE_SEVERITIES: tuple[str, ...] = ("high", "medium", "low", "unknown")

#: Severity recorded for a field with no usable availability declaration --
#: either the payload declares no lag for it, or the injected table does not
#: name it.  Reported, never silently passed and never fail-closed: this audit
#: is a reporter, not a gate.
LEAKAGE_UNKNOWN_SEVERITY = "unknown"

#: Severity recorded when a strategy reads a field sooner than the injected
#: availability lag allows: the field's value at the decision date embeds
#: information the strategy could not have had.
LEAKAGE_BREACH_SEVERITY = "high"

#: Severity recorded when a declaration leans on an announcement-to-effective
#: rule the injected table does not define.  Softer than a breach: the lag
#: itself is still within the standard, only the rule behind it is unverified.
LEAKAGE_UNKNOWN_RULE_SEVERITY = "medium"


def validate_audit_config(audit: Mapping[str, Any]) -> None:
    """Validate the request-level ``audit`` object (structure and ranges).

    Mirrored by the worker's ``_validate_audit``; both sides raise the same
    message for the same malformed input.  Key *existence* in the canonical
    payload is not checked here -- that happens at audit time in
    :func:`parameter_perturbation_report`, where the payload is available.
    """

    if not isinstance(audit, Mapping):
        raise ValueError("request audit must be an object")
    unknown = set(audit) - AUDIT_SECTIONS
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
        _validate_concentration(audit["concentration"])
    if "cost_sensitivity" in audit:
        _validate_cost_sensitivity(audit["cost_sensitivity"])
    if "parameter_perturbation" in audit:
        _validate_perturbation_shape(audit["parameter_perturbation"])
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


def _validate_stress(section: Mapping[str, Any]) -> None:
    """Validate the ``stress`` section shape (roadmap 9.3).

    Shape only: whether a scenario actually covers every *held* name is a
    property of the executed book, so it is checked at report time where the
    weights exist (the same split as ``cash_residual_max``).
    """

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


def _validate_concentration(section: Mapping[str, Any]) -> None:
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
                raise ValueError("audit concentration industry_map must map non-empty symbols to non-empty industry labels")
        if (
            isinstance(industry_threshold, bool)
            or not isinstance(industry_threshold, (int, float))
            or not math.isfinite(float(industry_threshold))
            or not 0 < float(industry_threshold) <= 1
        ):
            raise ValueError("audit concentration industry_threshold must be in (0, 1]")


def _validate_cost_sensitivity(section: Mapping[str, Any]) -> None:
    scenarios = section.get("scenarios")
    if scenarios is None:
        return
    if not isinstance(scenarios, (list, tuple)) or not scenarios:
        raise ValueError("audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs")
    for pair in scenarios:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs")
        for value in pair:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError("audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs")


def _validate_perturbation_shape(section: Mapping[str, Any]) -> None:
    perturbations = section.get("perturbations")
    if not isinstance(perturbations, Mapping) or not perturbations:
        raise ValueError("audit parameter_perturbation requires a non-empty perturbations object")
    for key, values in perturbations.items():
        if not isinstance(key, str) or not key:
            raise ValueError("audit parameter perturbation keys must be non-empty strings")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"audit parameter perturbation values must be a non-empty list for key {key!r}")
        for value in values:
            _validate_perturbation_value(value)


def _validate_perturbation_value(value: Any) -> None:
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return
    raise ValueError("audit parameter perturbation values must be finite scalars (number, boolean or string)")


def _validate_lag_table(value: Any, message: str) -> None:
    """Shape check for a ``{name: whole days}`` availability/rule table.

    Shared by both data-leakage tables so a malformed entry produces the same
    message on both sides regardless of which table it sits in.
    """

    if not isinstance(value, Mapping) or not value:
        raise ValueError(message)
    for name, days in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(message)
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ValueError(message)


def _validate_feasibility(section: Mapping[str, Any]) -> None:
    """Validate the ``feasibility`` section shape (roadmap 10.6).

    Shape only: the ratio is checked against the *executed* day table at report
    time, because whether a day is short is a property of the run rather than of
    the configuration.  ``allow_cash_residual`` deliberately carries no
    fail-closed behaviour -- see :func:`feasibility_report`.
    """

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
    """Validate the ``data_leakage`` section shape (roadmap 10.7).

    Both tables are required and non-empty: the whole point of the audit is to
    compare the run's declared point-in-time conventions against an injected
    availability standard, and a run with no standard has nothing to be compared
    against.  A field the candidate never declares is *not* an error here -- it
    is reported as ``unknown`` by :func:`data_leakage_report`.
    """

    _validate_lag_table(
        section.get("field_availability_lag"),
        "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
    )
    _validate_lag_table(
        section.get("announcement_to_effective"),
        "audit data_leakage announcement_to_effective must map non-empty rule names to non-negative whole days",
    )


def resolve_perturbation_target(payload: Mapping[str, Any], dotted_key: str) -> Any:
    """Resolve a dotted ``section.field`` path inside the canonical payload.

    This is the fail-closed gate against lazy payload fields: a key that is
    not present (e.g. ``composition.inverse_vol_window`` while the weighting
    is not ``inverse_volatility``) raises instead of silently perturbing
    nothing and reporting a fake insensitivity.  Top-level and unknown
    sections are rejected so an audit cannot rewrite identity fields such as
    the candidate name.
    """

    if not isinstance(dotted_key, str) or not dotted_key:
        raise ValueError("audit parameter perturbation keys must be non-empty strings")
    segments = dotted_key.split(".")
    if segments[0] not in PERTURBATION_SECTIONS:
        raise ValueError(
            f"parameter perturbation keys must target one of composition/backtest/signal/gating: {dotted_key!r}"
        )
    node: Any = payload
    for segment in segments:
        if not isinstance(node, Mapping) or segment not in node:
            raise ValueError(f"parameter perturbation key {dotted_key!r} is not present in the candidate payload")
        node = node[segment]
    return node


def concentration_report(
    weights: pd.DataFrame,
    *,
    top_k: Sequence[int],
    industry_map: Mapping[str, str] | None = None,
    industry_threshold: float | None = None,
) -> dict[str, Any]:
    """ConcentrationMonitor (roadmap 9.4): concentration of the executed book.

    Consumes the start-of-day weight panel the engine already produced and
    computes, per day, the Herfindahl-Hirschman index over held names, the
    effective number of holdings (``1 / HHI``, zero on an empty day) and the
    share of the ``k`` largest positions for every configured ``k``.  With an
    ``industry_map`` plus ``industry_threshold`` (both required together) it
    additionally reports the daily largest-industry share and how many days
    breach the threshold strictly.  Everything here is report output: the
    weights are never rewritten.
    """

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
            [str(industry_map[symbol]) if symbol in industry_map and industry_map[symbol] else _UNKNOWN_INDUSTRY for symbol in frame.columns],
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


def stress_scenario_report(weights: pd.DataFrame, section: Mapping[str, Any]) -> dict[str, Any]:
    """StressScenarioCheck (roadmap 9.3): replay the executed book on a scenario table.

    The scenario table is an injected, versioned data input and lives in the
    request-level ``audit`` object, never in the candidate payload: an agent
    cannot pick the scenarios it is judged on (the roadmap calls that out as a
    selection-bias risk) and enabling the check cannot move a strategy identity
    hash.  Every scenario is a total-return shock applied to the book held at
    the last date of the executed weights panel -- per name first, then per
    industry through the explicit ``industry_map``, then the scenario's
    ``default_shock``.

    A held name that none of the three covers fails closed instead of being
    silently counted as zero exposure: a partially specified table would
    otherwise report a small loss for a book it never looked at.  ``loss`` is
    the negative portfolio return, so a positive number is a loss; a scenario
    passes when its loss does not exceed ``loss_threshold``.  Nothing here
    rewrites a weight -- ``mode`` only decides whether
    :func:`ensure_stress_passed` aborts the audit afterwards.
    """

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


def ensure_stress_passed(report: Mapping[str, Any]) -> None:
    """Abort the audit when a ``reject`` scenario breaches its threshold.

    ``warn`` is the documented default and never aborts: the report fields are
    written and nothing else changes.  In ``reject`` mode the loss figures and
    the failing scenario are already in ``report`` and the *evaluation* then
    fails here -- no weight is rewritten, and no result file is written at all
    because the request fails.  The message therefore carries the loss, the
    threshold and ``scenario_table_version``, so a swapped scenario table
    cannot make a strategy fail without leaving a trace.
    """

    if report.get("mode") != "reject" or report.get("passed"):
        return
    raise ValueError(
        f"stress scenario check failed: loss {report['max_loss']!r} exceeds loss_threshold "
        f"{report['loss_threshold']!r} for scenario {report['worst_scenario']!r} "
        f"(scenario_table_version {report['scenario_table_version']!r})"
    )


def _max_drawdown(returns: pd.Series) -> float:
    wealth = pd.Series([1.0, *((1.0 + returns).cumprod().to_list())], dtype=float)
    return float((wealth / wealth.cummax() - 1.0).min())


def _scenario_metrics(
    daily: pd.Series, turnover: pd.Series, per_year: int, transaction_costs: pd.Series
) -> dict[str, float]:
    """Headline metrics for one scenario return series.

    Same formulas as the engine's ``_performance`` / worker ``_metrics`` for
    the reported subset, so a scenario at the production rates reproduces the
    production metrics exactly.
    """

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
        "max_drawdown": _max_drawdown(daily) if len(daily) else 0.0,
        "annual_turnover": float(turnover.mean()) * per_year if len(turnover) else 0.0,
        "total_transaction_cost": float(transaction_costs.sum()),
    }


def cost_sensitivity_report(
    daily_gross: pd.Series,
    turnover: pd.Series,
    buy_turnover: pd.Series,
    sell_turnover: pd.Series,
    scenarios: Sequence[Sequence[float]],
    per_year: int,
) -> dict[str, Any]:
    """CostSensitivityAudit (roadmap 10.2): re-price the executed trade legs.

    The scenario return series re-uses the already-executed backtest: the
    gross (pre-cost) daily returns and the per-day buy/sell turnover are the
    strategy's real activity, so a scenario only swaps the cost rates --
    no strategy computation is re-run and the traded book is identical in
    every scenario.  The production rates are reported as provenance; the
    scenarios live only in the audit configuration and can never enter the
    candidate payload (the worker still hard-rejects any cost key there).
    """

    rows: list[dict[str, Any]] = []
    for buy_bps, sell_bps in scenarios:
        scenario_cost = buy_turnover.astype(float) * (float(buy_bps) * 1e-4) + sell_turnover.astype(float) * (
            float(sell_bps) * 1e-4
        )
        scenario_net = daily_gross.astype(float) - scenario_cost
        row: dict[str, Any] = {"buy_bps": float(buy_bps), "sell_bps": float(sell_bps)}
        row.update(_scenario_metrics(scenario_net, turnover, per_year, scenario_cost))
        rows.append(row)
    return {
        "scenarios": rows,
        "production": {"buy_bps": BUY_COST_BPS, "sell_bps": SELL_COST_BPS},
    }


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in PERTURBATION_METRIC_KEYS}


def parameter_perturbation_report(
    payload: Mapping[str, Any],
    perturbations: Mapping[str, Sequence[Any]],
    baseline_metrics: Mapping[str, Any],
    evaluate: Callable[[str, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    """ParameterPerturbationAudit (roadmap 10.3): sensitivity to frozen params.

    Every pre-registered dotted key is first resolved against the canonical
    payload -- a key that is not present (lazy fields such as
    ``inverse_vol_window`` under another weighting) fails closed with
    ``ValueError`` instead of producing a misleading "not sensitive" report.
    Each ``(key, value)`` variant then re-runs the caller's engine through
    ``evaluate``; a variant that fails validation is recorded with
    ``status="error"`` and the engine's message instead of aborting the whole
    audit.  ``marginal`` is ``variant - baseline`` per reported metric.
    """

    if not isinstance(perturbations, Mapping) or not perturbations:
        raise ValueError("audit parameter_perturbation requires a non-empty perturbations object")
    normalized: dict[str, list[Any]] = {}
    for key, values in perturbations.items():
        if not isinstance(key, str) or not key:
            raise ValueError("audit parameter perturbation keys must be non-empty strings")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"audit parameter perturbation values must be a non-empty list for key {key!r}")
        for value in values:
            _validate_perturbation_value(value)
        normalized[str(key)] = list(values)
    # Fail-closed resolution first: one lazy key aborts the whole audit.
    for key in normalized:
        resolve_perturbation_target(payload, key)
    baseline = _metric_subset(baseline_metrics)
    variants: list[dict[str, Any]] = []
    for key, values in normalized.items():
        for value in values:
            try:
                variant_metrics = _metric_subset(evaluate(key, value))
            except ValueError as exc:
                variants.append({"key": key, "value": value, "status": "error", "error": str(exc)})
                continue
            variants.append(
                {
                    "key": key,
                    "value": value,
                    "status": "ok",
                    "metrics": variant_metrics,
                    "marginal": {metric: variant_metrics[metric] - baseline[metric] for metric in PERTURBATION_METRIC_KEYS},
                }
            )
    return {"baseline": {"metrics": baseline}, "variants": variants}


def ablation_report(
    base_metrics: Mapping[str, Any], ablated_metrics_by_factor: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """AblationAudit (roadmap 10.4): formalise the factor-ablation loop.

    Pure report assembly over metrics the engines already compute in their
    ungated two-argument ablation path (one full backtest per removed
    factor): per factor the ablated headline metrics and the four marginal
    contributions, with the exact formulas of the existing
    ``factor_contributions`` output so the two stay interchangeable.  The
    roadmap's operator-level removal has no disable switches in the engine
    and is deliberately not fabricated here.
    """

    factors: dict[str, Any] = {}
    for name, ablated in ablated_metrics_by_factor.items():
        factors[str(name)] = {
            "ablated": _metric_subset(ablated),
            "marginal": {
                "marginal_annualized_return": float(base_metrics["annualized_return"]) - float(ablated["annualized_return"]),
                "marginal_sharpe": float(base_metrics["sharpe"]) - float(ablated["sharpe"]),
                "marginal_drawdown_improvement": abs(float(ablated["max_drawdown"])) - abs(float(base_metrics["max_drawdown"])),
                "marginal_turnover_reduction": float(ablated["annual_turnover"]) - float(base_metrics["annual_turnover"]),
            },
        }
    return {"mode": "factors", "factors": factors}


def feasibility_report(
    days: Sequence[Mapping[str, Any]],
    *,
    min_executable_ratio: float | None = None,
    allow_cash_residual: bool = DEFAULT_ALLOW_CASH_RESIDUAL,
) -> dict[str, Any]:
    """FeasibilityAudit (roadmap 10.6): how much of the intended trade the book
    could actually place.

    Consumes one row per rebalance day, already reduced to weight space by the
    engine's own execution chain: the theoretical traded amount (the absolute
    change from the carried book to the target row, including the cash leg) and
    the executed traded amount, plus the per-reason split of the shortfall the
    caller attributed with :func:`attribute_execution_shortfall`.

    ``executable_ratio`` is actual over theoretical; a day whose theoretical
    amount is zero is skipped and never enters the denominator, because a day
    that intended no trade has no ratio to report and averaging zeros into the
    mean would drag it toward zero for no reason.

    ``allow_cash_residual`` is *reported*, not enforced: when it is False every
    day carrying an unfilled amount or a non-zero cash residual produces one
    violation record, and nothing else happens -- no weight is rewritten, no
    request fails, no cost formula changes.  This is the deliberate difference
    from ``CompositionSpec.cash_residual_max``, which bounds the *target* row's
    residual inside ``build_target_weights`` and fails the run closed when it is
    exceeded.  The two never share a field: one constrains the intention, the
    other reports the execution.
    """

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


def attribute_execution_shortfall(
    *,
    carried: Mapping[str, float],
    target: Mapping[str, float],
    executed: Mapping[str, float],
    stages: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Split ``target - executed`` across the execution chain's own stages.

    ``stages`` holds the book as it stood after each engine stage, in order:
    ``allocator`` (the participation cap or the staged plan -- the two are
    mutually exclusive), ``tradability`` (the side-aware mask) and
    ``lot_rounding`` (the lot grid).  Every name is measured against the stage
    immediately before it, so the withdrawals are a running decomposition of the
    same gap.

    The buckets do **not** sum to the total shortfall in general.  Each stage
    contributes its own *withdrawals*: per name ``max(0, before - after)``, plus
    the stage's own cash-leg withdrawal ``max(0, cash_before - cash_after)``.  A
    name that *rises* is therefore never a withdrawal by itself, but the stage
    that raises a position is not invisible either: raising it consumes cash, and
    that cash-leg withdrawal lands in the raising stage's own bucket.  A
    four-bucket split that summed to the gap would need a definition of
    "unfilled" that no stage of this chain produces, so no such identity is
    claimed; the residual is only bounded, not closed.  Callers that need the
    exact total should measure ``target - executed`` directly and treat this as
    an attribution of the withdrawals.

    ``carried`` is accepted so the cash leg is measured too, matching the
    engine's own half-L1 turnover convention where the implicit cash asset is
    part of the book.
    """

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


def data_leakage_report(
    declarations: Sequence[Mapping[str, Any]],
    *,
    field_availability_lag: Mapping[str, int],
    announcement_to_effective: Mapping[str, int],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """DataLeakageAudit (roadmap 10.7): compare declared point-in-time
    conventions against an injected availability standard.

    ``declarations`` is what this candidate actually declares about time: each
    entry carries the ``field`` it concerns, the ``declared_lag`` in trading
    days (or ``None`` when the payload says nothing about it), the payload
    ``location`` the statement lives at, and the ``rule`` naming the
    announcement-to-effective convention it relies on (or ``None``).

    Three outcomes per declared field, and no fourth:

    * a declared lag shorter than the injected availability lag is a look-ahead
      risk -- the value used at the decision date embeds information the
      strategy could not have had;
    * a field with no declaration at all is recorded as ``unknown`` and listed
      under ``undeclared``.  It is never silently passed and never fails the
      audit closed: this is a reporter, and a field the engine cannot speak to
      is a finding to surface, not a reason to refuse the run;
    * everything else is clean.

    A ``rule`` whose announcement-to-effective lag is unknown to the injected
    table is left at that field's own verdict rather than double-counted.
    ``provenance`` reuses the request's existing ``snapshot_id`` /
    ``snapshot_checksum`` identity -- no second version numbering is invented
    here.
    """

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


def feasibility_days(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce one run's execution trace to the per-day feasibility table.

    One row per traced rebalance day: the day's date, the theoretical traded
    amount (the absolute move from the carried book to the target row, cash leg
    included), the executed amount, the cash residual the executed book leaves,
    and the shortfall split across the four reason buckets by
    :func:`attribute_execution_shortfall`.

    A day whose executed book ended *above* its target (the documented staged
    overshoot, where a carried plan can lift a drifted book past the target row)
    reports a shortfall of zero rather than a negative one: the audit measures
    what could not be placed, and an over-filled day placed everything it
    intended.
    """

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
        reasons = attribute_execution_shortfall(
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


#: Point-in-time conventions this engine can declare, as ``(field, payload
#: location)``: the universe filters that shift or age a panel (batch E) and the
#: signal / execution lags that decide which day's data a decision may read.
#: Each entry is harvested by :func:`leakage_declarations` only when the run
#: actually carries the operator behind it -- an operator that is off declares
#: nothing, so it cannot produce a spurious finding.  Declaring the table next
#: to the harvester keeps the two from drifting: adding a declaration means
#: adding a row here *and* the branch that reads it.
LEAKAGE_FIELD_LOCATIONS: tuple[tuple[str, str], ...] = (
    ("index_membership", "universe_rules.index_membership.lag_days"),
    ("listing_age", "universe_rules.listing_age.min_days"),
    ("signal_lag", "backtest.signal_lag"),
    ("regime_lag", "signal.regime_lag"),
)


def leakage_declarations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect the point-in-time declarations a candidate payload actually makes.

    Reads the canonical payload -- not the candidate object -- so both engines
    harvest from the same bytes and agree on what was declared.  A field whose
    operator is absent from the payload is simply not harvested: it declared
    nothing, and the injected availability table names only the fields the
    caller cares about, so an undeclared-but-unused operator is not a finding.

    A field the payload *does* carry keeps its location rooted at the payload
    section the value came from, which is what makes a finding traceable back to
    the configuration that caused it.  Every emitted location is a key of
    :data:`LEAKAGE_FIELD_LOCATIONS`; the tests assert the harvested locations
    resolve inside the real payload.
    """

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
