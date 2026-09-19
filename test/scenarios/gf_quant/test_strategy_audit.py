"""Unit tests for the result-report audit operators (roadmap 9.3/9.4/10.2/10.3/10.4).

The audit operators are read-only: they produce report fields from an already
executed portfolio backtest.  These tests pin the local semantics on toy data
and -- where the worker has the verbatim mirror -- feed identical inputs to
both implementations and require identical output.  The end-to-end local-vs-
worker parity lives in ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.registry import strategy_id_for
from rdagent.scenarios.gf_quant.strategy.audit import (
    AUDIT_SECTIONS,
    DEFAULT_ALLOW_CASH_RESIDUAL,
    DEFAULT_CONCENTRATION_TOP_K,
    DEFAULT_COST_SENSITIVITY_SCENARIOS,
    DEFAULT_STRESS_MODE,
    FEASIBILITY_REASONS,
    LEAKAGE_BREACH_SEVERITY,
    LEAKAGE_SEVERITIES,
    LEAKAGE_UNKNOWN_SEVERITY,
    PERTURBATION_METRIC_KEYS,
    STRESS_MODES,
    ablation_report,
    attribute_execution_shortfall,
    concentration_report,
    cost_sensitivity_report,
    data_leakage_report,
    ensure_stress_passed,
    feasibility_days,
    feasibility_report,
    leakage_declarations,
    parameter_perturbation_report,
    resolve_perturbation_target,
    stress_scenario_report,
    validate_audit_config,
)
from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec
from rdagent.scenarios.gf_quant.strategy.composition import (
    _hierarchical_weights as _local_hierarchical_weights,
)
from scripts.gf_quant_strategy_worker import (
    AUDIT_KEYS as _WORKER_AUDIT_KEYS,
    DEFAULT_ALLOW_CASH_RESIDUAL as _WORKER_DEFAULT_ALLOW_CASH_RESIDUAL,
    DEFAULT_STRESS_MODE as _WORKER_DEFAULT_STRESS_MODE,
    FEASIBILITY_REASONS as _WORKER_FEASIBILITY_REASONS,
    LEAKAGE_BREACH_SEVERITY as _WORKER_LEAKAGE_BREACH_SEVERITY,
    LEAKAGE_SEVERITIES as _WORKER_LEAKAGE_SEVERITIES,
    LEAKAGE_UNKNOWN_SEVERITY as _WORKER_LEAKAGE_UNKNOWN_SEVERITY,
    STRESS_MODES as _WORKER_STRESS_MODES,
    _ablation_report,
    _attribute_execution_shortfall,
    _concentration_report,
    _cost_sensitivity_report,
    _data_leakage_report,
    _ensure_stress_passed,
    _feasibility_days,
    _feasibility_report,
    _leakage_declarations,
    _parameter_perturbation_report,
    _resolve_perturbation_target,
    _stress_scenario_report,
    _validate_audit,
)


def _perturbation_metrics(scale: float) -> dict[str, float]:
    """A full headline-metrics stub (all keys the audit subset reports)."""

    return {
        "total_return": 0.10 * scale,
        "annualized_return": 0.12 * scale,
        "cagr": 0.11 * scale,
        "annualized_vol": 0.15 * scale,
        "sharpe": 0.80 * scale,
        "max_drawdown": -0.20 * scale,
        "annual_turnover": 6.0 * scale,
    }


_SAMPLE_PAYLOAD = {
    "name": "sample",
    "composition": {"top_n": 4, "weighting": "equal"},
    "backtest": {"signal_lag": 1, "trading_days_per_year": 252},
    "signal": {"transform": "signed_log"},
}

#: A versioned stress scenario table (roadmap 9.3); ``crash`` covers one name
#: explicitly and everything else through ``default_shock``.
_STRESS_BASE = {
    "scenario_table_version": "2024Q1",
    "scenarios": {"crash": {"symbol_shocks": {"A": -0.3}, "default_shock": -0.05}},
    "loss_threshold": 0.2,
}


@pytest.mark.parametrize(
    ("audit", "message"),
    [
        ([], "request audit must be an object"),
        ({"mystery": {}}, "request audit has unknown keys: ['mystery']"),
        ({"concentration": []}, "request audit section 'concentration' must be an object"),
        ({"ablation": "factors"}, "request audit section 'ablation' must be an object"),
        ({"concentration": {"top_k": []}}, "audit concentration top_k must be a non-empty list of positive integers"),
        ({"concentration": {"top_k": [0]}}, "audit concentration top_k must be a non-empty list of positive integers"),
        ({"concentration": {"top_k": [1.5]}}, "audit concentration top_k must be a non-empty list of positive integers"),
        ({"concentration": {"industry_map": {"A": "X"}}}, "audit concentration requires industry_map and industry_threshold together"),
        ({"concentration": {"industry_threshold": 0.5}}, "audit concentration requires industry_map and industry_threshold together"),
        ({"concentration": {"industry_map": {}, "industry_threshold": 0.5}}, "audit concentration industry_map must be a non-empty object"),
        (
            {"concentration": {"industry_map": {"A": ""}, "industry_threshold": 0.5}},
            "audit concentration industry_map must map non-empty symbols to non-empty industry labels",
        ),
        (
            {"concentration": {"industry_map": {"A": "X"}, "industry_threshold": 1.5}},
            "audit concentration industry_threshold must be in (0, 1]",
        ),
        (
            {"cost_sensitivity": {"scenarios": []}},
            "audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs",
        ),
        (
            {"cost_sensitivity": {"scenarios": [[10.0]]}},
            "audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs",
        ),
        (
            {"cost_sensitivity": {"scenarios": [[10.0, 0.0]]}},
            "audit cost_sensitivity scenarios must be a non-empty list of [buy_bps, sell_bps] pairs",
        ),
        ({"parameter_perturbation": {}}, "audit parameter_perturbation requires a non-empty perturbations object"),
        ({"parameter_perturbation": {"perturbations": {}}}, "audit parameter_perturbation requires a non-empty perturbations object"),
        (
            {"parameter_perturbation": {"perturbations": {"composition.top_n": []}}},
            "audit parameter perturbation values must be a non-empty list for key 'composition.top_n'",
        ),
        (
            {"parameter_perturbation": {"perturbations": {"composition.top_n": [[2]]}}},
            "audit parameter perturbation values must be finite scalars (number, boolean or string)",
        ),
        (
            {"parameter_perturbation": {"perturbations": {"composition.top_n": [float("inf")]}}},
            "audit parameter perturbation values must be finite scalars (number, boolean or string)",
        ),
        ({"ablation": {"mode": "operators"}}, "audit ablation mode must be 'factors'"),
        ({"stress": {**_STRESS_BASE, "mode": "loud"}}, "audit stress mode must be 'warn' or 'reject'"),
        (
            {"stress": {"scenarios": _STRESS_BASE["scenarios"], "loss_threshold": 0.2}},
            "audit stress requires a non-empty scenario_table_version",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenario_table_version": ""}},
            "audit stress requires a non-empty scenario_table_version",
        ),
        ({"stress": {**_STRESS_BASE, "scenarios": {}}}, "audit stress requires a non-empty scenarios object"),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"": {"default_shock": -0.1}}}},
            "audit stress scenario identifiers must be non-empty strings",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": [-0.1]}}},
            "audit stress scenario 'crash' must be an object",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"other": -0.1}}}},
            "audit stress scenario 'crash' must provide symbol_shocks, industry_shocks or default_shock",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"symbol_shocks": {}}}}},
            "audit stress scenario 'crash' symbol_shocks must be a non-empty object",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"symbol_shocks": {"": -0.1}}}}},
            "audit stress scenario 'crash' symbol_shocks keys must be non-empty strings",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"symbol_shocks": {"A": True}}}}},
            "audit stress scenario 'crash' symbol_shocks values must be finite numbers",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"symbol_shocks": {"A": float("inf")}}}}},
            "audit stress scenario 'crash' symbol_shocks values must be finite numbers",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"symbol_shocks": {"A": float("nan")}}}}},
            "audit stress scenario 'crash' symbol_shocks values must be finite numbers",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"default_shock": float("nan")}}}},
            "audit stress scenario 'crash' default_shock must be a finite number",
        ),
        (
            {"stress": {"scenario_table_version": "2024Q1", "scenarios": _STRESS_BASE["scenarios"]}},
            "audit stress loss_threshold must be a finite number >= 0",
        ),
        ({"stress": {**_STRESS_BASE, "loss_threshold": -0.5}}, "audit stress loss_threshold must be a finite number >= 0"),
        (
            {"stress": {**_STRESS_BASE, "scenarios": {"crash": {"industry_shocks": {"X": -0.2}}}}},
            "audit stress industry_shocks requires industry_map",
        ),
        ({"stress": {**_STRESS_BASE, "industry_map": {}}}, "audit stress industry_map must be a non-empty object"),
        (
            {"stress": {**_STRESS_BASE, "industry_map": {"A": ""}}},
            "audit stress industry_map must map non-empty symbols to non-empty industry labels",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenario_ids": []}},
            "audit stress scenario_ids must be a non-empty list of scenario identifiers",
        ),
        (
            {"stress": {**_STRESS_BASE, "scenario_ids": ["crash", "nope"]}},
            "audit stress scenario_ids must all be present in the scenario table: ['nope']",
        ),
    ],
)
def test_audit_config_failures_fail_closed_on_both_sides(audit, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_audit_config(audit)
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate_audit(audit)


@pytest.mark.parametrize(
    "audit",
    [
        {"concentration": {}},
        {"concentration": {"top_k": [2, 5]}},
        {"concentration": {"top_k": [3], "industry_map": {"S0": "X"}, "industry_threshold": 0.5}},
        {"cost_sensitivity": {}},
        {"cost_sensitivity": {"scenarios": [[10.0, 15.0], [25, 25]]}},
        {"parameter_perturbation": {"perturbations": {"composition.top_n": [2, 3], "backtest.signal_lag": [2]}}},
        {"ablation": {}},
        {"ablation": {"mode": "factors"}},
        {"stress": _STRESS_BASE},
        {"stress": {**_STRESS_BASE, "mode": "reject", "scenario_ids": ["crash"]}},
        {
            "stress": {
                "scenario_table_version": "2024Q2",
                "scenarios": {"crash": {"industry_shocks": {"X": -0.3}, "default_shock": -0.02}},
                "industry_map": {"A": "X", "B": "Y"},
                "loss_threshold": 0.25,
            }
        },
    ],
)
def test_valid_audit_configs_pass_on_both_sides(audit) -> None:
    validate_audit_config(audit)
    _validate_audit(audit)


def test_concentration_report_semantics_and_worker_mirror() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    weights = pd.DataFrame(
        {
            "A": [0.5, 0.0, 0.0],
            "B": [0.3, 0.6, 0.0],
            "C": [0.2, 0.4, 0.0],
            "D": [0.0, 0.0, 0.0],
        },
        index=dates,
    )
    industry_map = {"A": "X", "B": "X", "C": "Y"}
    report = concentration_report(
        weights, top_k=[1, 2, 5], industry_map=industry_map, industry_threshold=0.75
    )

    # Day 1: 0.5/0.3/0.2 -> HHI 0.38, top-1 0.5, top-2 0.8, fully invested.
    assert report["hhi"]["mean"] == pytest.approx((0.38 + 0.52 + 0.0) / 3)
    assert report["hhi"]["min"] == pytest.approx(0.0)
    assert report["top_k"]["1"]["mean"] == pytest.approx((0.5 + 0.6 + 0.0) / 3)
    assert report["top_k"]["2"]["mean"] == pytest.approx((0.8 + 1.0 + 0.0) / 3)
    # Top-5 exceeds the book: the whole invested share.
    assert report["top_k"]["5"]["max"] == pytest.approx(1.0)
    assert report["effective_names"]["max"] == pytest.approx(1.0 / 0.38)
    # Empty day collapses to zero effective names instead of dividing by zero.
    assert report["effective_names"]["mean"] == pytest.approx((1.0 / 0.38 + 1.0 / 0.52 + 0.0) / 3)
    # Industry shares: day1 max(X)=0.8 breaches 0.75 strictly, day2 max(X)=0.6 does not.
    assert report["industry"]["breach_count"] == 1
    assert report["industry"]["max_share"]["max"] == pytest.approx(0.8)
    assert report["industry"]["threshold"] == pytest.approx(0.75)

    worker_report = _concentration_report(
        weights, top_k=[1, 2, 5], industry_map=industry_map, industry_threshold=0.75
    )
    assert report == worker_report


def test_concentration_report_defaults_and_unmapped_symbols() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    weights = pd.DataFrame({"A": [0.6, 0.2], "Z": [0.4, 0.2]}, index=dates)
    report = concentration_report(weights, top_k=DEFAULT_CONCENTRATION_TOP_K)
    assert report["top_k"]["5"]["max"] == pytest.approx(1.0)
    assert "industry" not in report
    mapped = concentration_report(weights, top_k=[2], industry_map={"A": "X"}, industry_threshold=0.5)
    # Unmapped Z shares the reserved group; day1 max share = max(0.6, 0.4) = 0.6 -> breach.
    assert mapped["industry"]["breach_count"] == 1
    worker_mapped = _concentration_report(weights, top_k=[2], industry_map={"A": "X"}, industry_threshold=0.5)
    assert mapped == worker_mapped


def test_cost_sensitivity_report_reuses_the_executed_series() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    gross = pd.Series([0.01, -0.005, 0.02], index=index)
    turnover = pd.Series([0.0, 0.35, 0.0], index=index)
    buy = pd.Series([0.0, 0.5, 0.0], index=index)
    sell = pd.Series([0.0, 0.2, 0.0], index=index)
    report = cost_sensitivity_report(gross, turnover, buy, sell, DEFAULT_COST_SENSITIVITY_SCENARIOS, 252)

    assert report["production"] == {"buy_bps": 10.0, "sell_bps": 10.0}
    assert [(row["buy_bps"], row["sell_bps"]) for row in report["scenarios"]] == [(10.0, 10.0), (15.0, 15.0), (20.0, 20.0)]
    # Day 2 trades 0.5 up and 0.2 down: 10/10 bps cost 7e-4 in total.
    assert report["scenarios"][0]["total_transaction_cost"] == pytest.approx(0.7 * 1e-3)
    scenario_net = gross - (buy * 1e-3 + sell * 1e-3)
    assert report["scenarios"][0]["total_return"] == pytest.approx(float((1.0 + scenario_net).prod() - 1.0))
    assert report["scenarios"][0]["annual_turnover"] == pytest.approx(0.35 / 3 * 252)
    # Dearer scenarios only degrade the same traded book: returns strictly
    # worse and costs strictly higher.
    returns = [row["annualized_return"] for row in report["scenarios"]]
    costs = [row["total_transaction_cost"] for row in report["scenarios"]]
    assert returns[0] > returns[1] > returns[2]
    assert costs[0] < costs[1] < costs[2]

    worker_report = _cost_sensitivity_report(
        gross, turnover, buy, sell, DEFAULT_COST_SENSITIVITY_SCENARIOS, 252
    )
    assert report == worker_report


def test_resolve_perturbation_target_fail_closed_paths() -> None:
    assert resolve_perturbation_target(_SAMPLE_PAYLOAD, "composition.top_n") == 4
    assert resolve_perturbation_target(_SAMPLE_PAYLOAD, "backtest.signal_lag") == 1
    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameter perturbation key 'composition.inverse_vol_window' is not present in the candidate payload"
        ),
    ):
        resolve_perturbation_target(_SAMPLE_PAYLOAD, "composition.inverse_vol_window")
    with pytest.raises(
        ValueError,
        match=re.escape("parameter perturbation keys must target one of composition/backtest/signal/gating: 'name'"),
    ):
        resolve_perturbation_target(_SAMPLE_PAYLOAD, "name")
    with pytest.raises(
        ValueError,
        match=re.escape("parameter perturbation key 'composition.weight_cap' is not present in the candidate payload"),
    ):
        resolve_perturbation_target(_SAMPLE_PAYLOAD, "composition.weight_cap")
    with pytest.raises(
        ValueError,
        match=re.escape("parameter perturbation key 'gating.layers' is not present in the candidate payload"),
    ):
        resolve_perturbation_target(_SAMPLE_PAYLOAD, "gating.layers")

    # Worker mirror raises the same messages.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameter perturbation key 'composition.inverse_vol_window' is not present in the candidate payload"
        ),
    ):
        _resolve_perturbation_target(_SAMPLE_PAYLOAD, "composition.inverse_vol_window")
    with pytest.raises(
        ValueError,
        match=re.escape("parameter perturbation keys must target one of composition/backtest/signal/gating: 'name'"),
    ):
        _resolve_perturbation_target(_SAMPLE_PAYLOAD, "name")


def test_parameter_perturbation_report_semantics_and_worker_mirror() -> None:
    baseline = _perturbation_metrics(1.0)

    def evaluate(key: str, value) -> dict[str, float]:
        if key == "composition.top_n" and value == 99:
            raise ValueError("boom: invalid variant")
        return _perturbation_metrics(float(value) / 4.0)

    perturbations = {"composition.top_n": [2, 99], "backtest.signal_lag": [2]}
    report = parameter_perturbation_report(_SAMPLE_PAYLOAD, perturbations, baseline, evaluate)
    worker_report = _parameter_perturbation_report(_SAMPLE_PAYLOAD, perturbations, baseline, evaluate)
    assert report == worker_report

    assert report["baseline"]["metrics"] == {key: pytest.approx(value) for key, value in baseline.items()}
    variants = report["variants"]
    assert [variant["status"] for variant in variants] == ["ok", "error", "ok"]
    assert variants[1]["error"] == "boom: invalid variant"
    assert variants[0]["metrics"]["annualized_return"] == pytest.approx(0.12 * 0.5)
    assert variants[0]["marginal"]["annualized_return"] == pytest.approx(0.12 * 0.5 - 0.12)
    assert variants[2]["marginal"]["annualized_return"] == pytest.approx(0.12 * 0.5 - 0.12)
    assert set(report["baseline"]["metrics"]) == set(PERTURBATION_METRIC_KEYS)

    # The fail-closed lazy-key rule aborts the whole audit before any variant runs.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameter perturbation key 'composition.inverse_vol_window' is not present in the candidate payload"
        ),
    ):
        parameter_perturbation_report(
            _SAMPLE_PAYLOAD, {"composition.inverse_vol_window": [10]}, baseline, evaluate
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "parameter perturbation key 'composition.inverse_vol_window' is not present in the candidate payload"
        ),
    ):
        _parameter_perturbation_report(_SAMPLE_PAYLOAD, {"composition.inverse_vol_window": [10]}, baseline, evaluate)


def test_ablation_report_uses_the_contributions_formulas_and_worker_mirror() -> None:
    base = _perturbation_metrics(1.0)
    ablated = {
        "value": _perturbation_metrics(0.8),
        "quality": _perturbation_metrics(1.05),
    }
    report = ablation_report(base, ablated)
    worker_report = _ablation_report(base, ablated)
    assert report == worker_report
    assert report["mode"] == "factors"

    value = report["factors"]["value"]
    assert value["ablated"]["sharpe"] == pytest.approx(0.80 * 0.8)
    assert value["marginal"]["marginal_annualized_return"] == pytest.approx(0.12 - 0.12 * 0.8)
    assert value["marginal"]["marginal_sharpe"] == pytest.approx(0.80 - 0.80 * 0.8)
    assert value["marginal"]["marginal_drawdown_improvement"] == pytest.approx(abs(-0.20 * 0.8) - abs(-0.20))
    assert value["marginal"]["marginal_turnover_reduction"] == pytest.approx(6.0 * 0.8 - 6.0)


def test_hierarchical_group_member_order_is_canonical() -> None:
    """Equivalent group-member orders must share one canonical payload/hash.

    The stored spec keeps the configured order for readability; only the
    canonical payload sorts members, because the weighting itself never reads
    member order and an unstable hash would fork registry identities.
    """

    spec_ab = CompositionSpec(
        top_n=2, weighting="hierarchical", hierarchical_groups={"g1": ("A", "B"), "g2": ("C", "D")}
    )
    spec_ba = CompositionSpec(
        top_n=2, weighting="hierarchical", hierarchical_groups={"g1": ("B", "A"), "g2": ("D", "C")}
    )
    assert spec_ab.hierarchical_groups == {"g1": ("A", "B"), "g2": ("C", "D")}
    assert spec_ba.hierarchical_groups == {"g1": ("B", "A"), "g2": ("D", "C")}
    assert spec_ab.as_payload() == spec_ba.as_payload()
    assert spec_ab.as_payload()["hierarchical_groups"]["g1"] == ["A", "B"]
    assert strategy_id_for("strategy", spec_ab.as_payload()) == strategy_id_for("strategy", spec_ba.as_payload())
    # Execution is unaffected: the weighting consumes membership, not order.
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0})
    np.testing.assert_array_equal(
        _local_hierarchical_weights(spec_ab, scores).to_numpy(dtype=float),
        _local_hierarchical_weights(spec_ba, scores).to_numpy(dtype=float),
    )


def test_stress_constants_are_shadowed_identically_on_both_sides() -> None:
    """The stress mode vocabulary is a literal written twice; keep it locked."""

    assert AUDIT_SECTIONS == _WORKER_AUDIT_KEYS
    assert STRESS_MODES == _WORKER_STRESS_MODES
    assert DEFAULT_STRESS_MODE == _WORKER_DEFAULT_STRESS_MODE == "warn"
    assert DEFAULT_STRESS_MODE in STRESS_MODES


def _stress_book() -> pd.DataFrame:
    """Three names held on the last row: A 0.4, B 0.3, C 0.3."""

    dates = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame(
        {"A": [0.5, 0.0, 0.4], "B": [0.3, 0.0, 0.3], "C": [0.2, 0.0, 0.3]},
        index=dates,
    )


def test_stress_scenario_report_semantics_and_worker_mirror() -> None:
    weights = _stress_book()
    section = {
        "scenario_table_version": "2024Q1",
        "scenarios": {
            "crash": {"symbol_shocks": {"A": -0.4}, "industry_shocks": {"X": -0.2}, "default_shock": -0.1},
            "mild": {"default_shock": 0.05},
        },
        "industry_map": {"B": "X", "C": "Y"},
        "loss_threshold": 0.2,
    }
    report = stress_scenario_report(weights, section)
    worker_report = _stress_scenario_report(weights, section)
    assert report == worker_report

    # Absent ``mode`` falls back to the declared default, and the report takes
    # the book from the LAST row of the executed panel only.
    assert report["mode"] == DEFAULT_STRESS_MODE
    assert report["scenario_table_version"] == "2024Q1"
    assert report["held_names"] == 3
    assert report["as_of"] == "2024-01-04"
    assert report["loss_threshold"] == pytest.approx(0.2)

    # crash: A -0.4 explicit, B -0.2 via industry X, C falls through to -0.1.
    assert report["scenarios"][0]["scenario_id"] == "crash"
    assert report["scenarios"][0]["portfolio_return"] == pytest.approx(0.4 * -0.4 + 0.3 * -0.2 + 0.3 * -0.1)
    assert report["scenarios"][0]["loss"] == pytest.approx(0.25)
    assert report["scenarios"][0]["explicit_coverage"] == pytest.approx(0.7)
    assert report["scenarios"][0]["passed"] is False
    # mild: everything through default_shock, so no weight is explicitly covered.
    assert report["scenarios"][1]["portfolio_return"] == pytest.approx(0.4 * 0.05 + 0.3 * 0.05 + 0.3 * 0.05)
    assert report["scenarios"][1]["loss"] == pytest.approx(-0.05)
    assert report["scenarios"][1]["explicit_coverage"] == pytest.approx(0.0)
    assert report["scenarios"][1]["passed"] is True
    assert report["passed"] is False
    assert report["max_loss"] == pytest.approx(0.25)
    assert report["worst_scenario"] == "crash"

    # ``scenario_ids`` selects a subset, in the given order.
    subset = stress_scenario_report(weights, {**section, "scenario_ids": ["mild"]})
    assert [row["scenario_id"] for row in subset["scenarios"]] == ["mild"]
    assert subset["passed"] is True
    assert subset["worst_scenario"] == "mild"
    assert subset == _stress_scenario_report(weights, {**section, "scenario_ids": ["mild"]})

    # The threshold comparison includes equality: loss == threshold passes.
    at_threshold = stress_scenario_report(
        weights,
        {
            "scenario_table_version": "v",
            "scenarios": {"flat": {"default_shock": -0.5}},
            "loss_threshold": 0.5,
        },
    )
    assert at_threshold["max_loss"] == pytest.approx(0.5)
    assert at_threshold["scenarios"][0]["passed"] is True

    # NaN weights count as no position (the book is read, not a metric panel).
    holed = weights.copy()
    holed.iloc[-1, holed.columns.get_loc("B")] = np.nan
    holed_report = stress_scenario_report(holed, {**section, "scenario_ids": ["crash"]})
    assert holed_report["held_names"] == 2
    assert holed_report["scenarios"][0]["loss"] == pytest.approx(0.4 * 0.4 + 0.3 * 0.1)
    assert holed_report == _stress_scenario_report(holed, {**section, "scenario_ids": ["crash"]})

    # An empty panel reports an empty book instead of dividing by nothing.
    empty = stress_scenario_report(pd.DataFrame(index=pd.DatetimeIndex([])), {**section, "scenario_ids": ["crash"]})
    assert empty["held_names"] == 0
    assert empty["as_of"] is None
    assert empty["max_loss"] == pytest.approx(0.0)
    assert empty["passed"] is True


def test_stress_scenario_report_fails_closed_on_uncovered_holdings() -> None:
    """A partial table must never be silently scored as zero exposure."""

    weights = _stress_book()
    partial = {
        "scenario_table_version": "2024Q1",
        "scenarios": {"crash": {"symbol_shocks": {"B": -0.4}}},
        "loss_threshold": 0.2,
    }
    # ``validate_audit_config`` accepts the shape; only the held book can tell.
    validate_audit_config({"stress": partial})
    with pytest.raises(
        ValueError,
        match=re.escape("stress scenario 'crash' does not cover held symbol 'A'"),
    ):
        stress_scenario_report(weights, partial)
    with pytest.raises(
        ValueError,
        match=re.escape("stress scenario 'crash' does not cover held symbol 'A'"),
    ):
        _stress_scenario_report(weights, partial)

    # An industry map that misses the held name does not rescue it either.
    unmapped = {
        "scenario_table_version": "2024Q1",
        "scenarios": {"crash": {"industry_shocks": {"X": -0.3}}},
        "industry_map": {"B": "X"},
        "loss_threshold": 0.2,
    }
    with pytest.raises(
        ValueError,
        match=re.escape("stress scenario 'crash' does not cover held symbol 'A'"),
    ):
        stress_scenario_report(weights, unmapped)


def test_ensure_stress_passed_only_rejects_in_reject_mode() -> None:
    # Powers of two keep the book sum exact, so the message can be pinned
    # character for character on both ends.
    book = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.25, 0.25], "C": [0.25, 0.25]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    section = {
        "scenario_table_version": "2024Q1",
        "scenarios": {"crash": {"default_shock": -0.5}},
        "loss_threshold": 0.2,
        "mode": "reject",
    }
    report = stress_scenario_report(book, section)
    assert report["passed"] is False
    # The report is written either way: ``reject`` only aborts afterwards.
    assert report["max_loss"] == pytest.approx(0.5)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "stress scenario check failed: loss 0.5 exceeds loss_threshold 0.2 for scenario 'crash' "
            "(scenario_table_version '2024Q1')"
        ),
    ):
        ensure_stress_passed(report)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "stress scenario check failed: loss 0.5 exceeds loss_threshold 0.2 for scenario 'crash' "
            "(scenario_table_version '2024Q1')"
        ),
    ):
        _ensure_stress_passed(report)

    # ``warn`` is the declared default and never aborts, even on a breach.
    warned = stress_scenario_report(book, {**section, "mode": "warn"})
    assert warned["passed"] is False
    ensure_stress_passed(warned)
    _ensure_stress_passed(warned)

    # A passing ``reject`` audit is not an abort either.
    passing = stress_scenario_report(
        book,
        {
            "scenario_table_version": "2024Q1",
            "scenarios": {"crash": {"default_shock": -0.05}},
            "loss_threshold": 0.2,
            "mode": "reject",
        },
    )
    assert passing["passed"] is True
    ensure_stress_passed(passing)
    _ensure_stress_passed(passing)

# ---------------------------------------------------------------------------
# Batch 7: FeasibilityAudit (10.6) and DataLeakageAudit (10.7).
# ---------------------------------------------------------------------------


def _stage_series(**weights: float) -> pd.Series:
    return pd.Series(dict(weights), dtype=float)


def _feasibility_day(date: str, carried: dict, target: dict, stages: dict, allocator: str = "participation"):
    return {
        "date": date,
        "carried": _stage_series(**carried),
        "target": _stage_series(**target),
        "stages": {name: _stage_series(**row) for name, row in stages.items()}
        | {"__allocator_reason__": allocator},
    }


def test_batch7_constants_are_shadowed_identically_on_both_sides() -> None:
    """The shared vocabularies are mirrored, not merely similar."""

    assert AUDIT_SECTIONS == _WORKER_AUDIT_KEYS
    assert FEASIBILITY_REASONS == _WORKER_FEASIBILITY_REASONS
    assert LEAKAGE_SEVERITIES == _WORKER_LEAKAGE_SEVERITIES
    assert DEFAULT_ALLOW_CASH_RESIDUAL == _WORKER_DEFAULT_ALLOW_CASH_RESIDUAL
    assert LEAKAGE_BREACH_SEVERITY == _WORKER_LEAKAGE_BREACH_SEVERITY
    assert LEAKAGE_UNKNOWN_SEVERITY == _WORKER_LEAKAGE_UNKNOWN_SEVERITY
    assert {"feasibility", "data_leakage"} <= AUDIT_SECTIONS


def test_feasibility_shortfall_attribution_attributes_each_stage_withdrawal() -> None:
    """Each stage's withdrawal is attributed to that stage.

    Two names: the allocator slices 0.3 -> 0.1, the mask blocks the remaining
    0.1, and the lot grid shaves 0.1 -> 0.08.  Every stage in this configuration
    only ever withdraws, so the buckets happen to sum to the gap here -- that is
    a property of this fixture, not an identity of the function (a stage that
    rounds a position *up* gives weight back and is not credited anywhere).
    """

    stages = {
        "allocator": {"A": 0.1, "B": 0.0},
        "tradability": {"A": 0.1, "B": 0.0},
        "lot_rounding": {"A": 0.08, "B": 0.0},
    }
    target = {"A": 0.3, "B": 0.0}
    carried = {"A": 0.0, "B": 0.0}
    attribution = attribute_execution_shortfall(
        carried=carried, target=target, executed=stages["lot_rounding"], stages=stages
    )
    assert attribution["participation"] == pytest.approx(0.2)
    assert attribution["tradability"] == pytest.approx(0.0)
    assert attribution["lot_rounding"] == pytest.approx(0.02)
    assert sum(attribution.values()) == pytest.approx(0.22)
    worker = _attribute_execution_shortfall(
        carried=carried, target=target, executed=stages["lot_rounding"], stages=stages
    )
    assert worker == attribution


def test_feasibility_staging_bucket_is_not_the_participation_bucket() -> None:
    """The two allocator variants report under their own names."""

    stages = {
        "allocator": {"A": 0.1},
        "tradability": {"A": 0.1},
        "lot_rounding": {"A": 0.1},
        "__allocator_reason__": "staging",
    }
    attribution = attribute_execution_shortfall(
        carried={"A": 0.0}, target={"A": 0.3}, executed={"A": 0.1}, stages=stages
    )
    assert attribution["staging"] == pytest.approx(0.2)
    assert attribution["participation"] == pytest.approx(0.0)


def test_feasibility_shortfall_buckets_do_not_always_sum_to_the_gap() -> None:
    """The buckets attribute stage withdrawals; they are not a partition of the gap.

    The lot grid rounds A back *up* from 0.10 to 0.12.  A rise is not a per-name
    withdrawal, but it is not invisible either: raising the position consumes
    cash, and that stage's cash-leg withdrawal of 0.02 lands in the
    ``lot_rounding`` bucket, so the grid does get credited for it.  The buckets
    report 0.22 while the true half-L1 gap is 0.36; the residual 0.14 is the
    cash-leg asymmetry -- the gap counts the cash leg's *absolute* move (0.18)
    while the buckets count only its negative part (0.02) -- net of the 0.02 the
    stock side over-counts, because the allocator's withdrawal was measured
    against the target (0.20) rather than against what was finally executed
    (0.18).  The shortfall is under-attributed, but not by "what the grid gave
    back".
    """

    stages = {
        "allocator": {"A": 0.1},
        "tradability": {"A": 0.1},
        "lot_rounding": {"A": 0.12},
    }
    target, carried, executed = {"A": 0.30}, {"A": 0.0}, {"A": 0.12}
    attribution = attribute_execution_shortfall(
        carried=carried, target=target, executed=executed, stages=stages
    )
    buckets = sum(attribution.values())
    # Gap under the engine's half-L1 convention: the stock leg (0.30 -> 0.12) plus
    # the matching cash leg (0.70 -> 0.88).
    stock_gap = 0.30 - 0.12
    cash_gap = 0.88 - 0.70
    gap = stock_gap + cash_gap
    assert attribution["participation"] == pytest.approx(0.2)
    assert attribution["tradability"] == pytest.approx(0.0)
    assert attribution["staging"] == pytest.approx(0.0)
    assert buckets == pytest.approx(0.22)
    assert gap == pytest.approx(0.36)
    assert buckets < gap
    # The worker mirrors the same (non-)identity.
    assert (
        _attribute_execution_shortfall(carried=carried, target=target, executed=executed, stages=stages)
        == attribution
    )


def test_feasibility_days_reduces_a_trace_and_floors_the_overshoot() -> None:
    """The day table measures the intended move and never goes negative.

    The second day's executed book sits *above* its target -- the documented
    staged overshoot on a drifted book -- and the audit must report a shortfall
    of zero there rather than a negative one.
    """

    trace = [
        _feasibility_day(
            "2024-01-02",
            carried={"A": 0.0, "B": 0.0},
            target={"A": 0.5, "B": 0.5},
            stages={"allocator": {"A": 0.1, "B": 0.1}, "tradability": {"A": 0.1, "B": 0.1}, "lot_rounding": {"A": 0.1, "B": 0.1}},
        ),
        _feasibility_day(
            "2024-01-03",
            carried={"A": 0.52, "B": 0.52},
            target={"A": 0.5, "B": 0.5},
            stages={"allocator": {"A": 0.52, "B": 0.52}, "tradability": {"A": 0.52, "B": 0.52}, "lot_rounding": {"A": 0.52, "B": 0.52}},
        ),
    ]
    days = feasibility_days(trace)
    assert [row["date"] for row in days] == ["2024-01-02", "2024-01-03"]
    # The theoretical move counts both legs: 0.5 + 0.5 of stock out of cash (1.0)
    # and 1.0 of cash into stock, so the intended traded amount is 2.0.
    assert days[0]["theoretical"] == pytest.approx(2.0)
    assert days[0]["executed"] == pytest.approx(0.4)
    # The stage split is name by name on the stock leg (0.5 -> 0.1 each, so 0.8)
    # plus the cash leg.  Here cash *rises* (0.0 -> 0.8) because the book sells
    # down to target, so the cash withdrawal is zero and the bucket is pure
    # stock.  ``theoretical`` (2.0) counts both legs, which is why it sits above
    # the buckets.
    assert days[0]["participation"] == pytest.approx(0.8)
    # Day two: the target (1.0) sits below the carried book (1.04) and the order
    # layer places nothing, so none of the intended move happens.  The reported
    # shortfall floors at zero rather than going negative -- the audit measures
    # what could not be placed.
    assert days[1]["theoretical"] == pytest.approx(0.08)
    assert days[1]["executed"] == pytest.approx(0.0)
    # Same half-L1 convention on the reason split, and the same "withdrawals
    # only" rule: A and B both *rise* here (0.50 -> 0.52), so no name contributes
    # a withdrawal and the whole 0.04 is the allocator stage's cash leg going
    # negative -- cash 0.00 -> -0.04, so max(0, cash(target) - cash(stage)) is
    # 0.04.  The 0.08 ``theoretical`` is the stock move (0.04) plus the cash move
    # it implies (0.04); the buckets hold the cash move and not the stock move,
    # because a rise is not a withdrawal.
    assert sum(days[1][reason] for reason in FEASIBILITY_REASONS) == pytest.approx(0.04)
    assert _feasibility_days(trace) == days


def test_feasibility_report_semantics_and_worker_mirror() -> None:
    """Ratios skip zero-theory days and violations never touch the numbers."""

    days = [
        {"date": "2024-01-02", "theoretical": 1.0, "executed": 0.9, "cash_residual": 0.1},
        {"date": "2024-01-03", "theoretical": 0.0, "executed": 0.0, "cash_residual": 0.0},
        {"date": "2024-01-04", "theoretical": 1.0, "executed": 0.5, "cash_residual": 0.25},
    ]
    report = feasibility_report(days, min_executable_ratio=0.8, allow_cash_residual=False)
    assert report["summary"]["days"] == 3
    assert report["summary"]["scored_days"] == 2
    assert report["per_day"][1]["executable_ratio"] is None
    assert report["summary"]["mean_executable_ratio"] == pytest.approx(0.7)
    assert report["summary"]["min_executable_ratio"] == pytest.approx(0.5)
    kinds = [violation["kind"] for violation in report["violations"]]
    assert kinds.count("min_executable_ratio") == 1
    assert kinds.count("cash_residual_not_allowed") == 2
    # Report-only: the day rows are echoed untouched.
    assert report["per_day"][0]["executed"] == 0.9
    assert report["per_day"][2]["executed"] == 0.5

    assert _feasibility_report(days, min_executable_ratio=0.8, allow_cash_residual=False) == report


def test_feasibility_report_defaults_allow_cash_residual() -> None:
    """The declared default reports nothing about cash unless asked."""

    days = [{"date": "2024-01-02", "theoretical": 1.0, "executed": 0.5, "cash_residual": 0.5}]
    report = feasibility_report(days)
    assert report["allow_cash_residual"] is True
    assert report["min_executable_ratio"] is None
    assert report["violations"] == []
    assert report["summary"]["unfilled_total"] == pytest.approx(0.5)
    assert report["per_day"][0]["unfilled_by_reason"] == {reason: 0.0 for reason in FEASIBILITY_REASONS}


def test_leakage_declarations_read_the_payload_not_the_object() -> None:
    """Only operators present in the payload declare anything."""

    bare = leakage_declarations({"backtest": {"signal_lag": 2}})
    assert bare == [
        {"field": "signal_lag", "location": "backtest.signal_lag", "declared_lag": 2, "rule": "signal_lag"}
    ]
    full = leakage_declarations(
        {
            "backtest": {"signal_lag": 1},
            "signal": {"regime_lag": 3},
            "universe_rules": {
                "index_membership": {"lag_days": 4},
                "listing_age": {"min_days": 10},
            },
        }
    )
    fields = {entry["field"]: entry for entry in full}
    assert fields["index_membership"]["declared_lag"] == 4
    assert fields["regime_lag"]["declared_lag"] == 3
    # ``min_days`` is a minimum age, not a lag: no lag is claimed.
    assert fields["listing_age"]["declared_lag"] is None
    assert _leakage_declarations(
        {
            "backtest": {"signal_lag": 1},
            "signal": {"regime_lag": 3},
            "universe_rules": {"index_membership": {"lag_days": 4}, "listing_age": {"min_days": 10}},
        }
    ) == full


def test_data_leakage_report_matches_and_reports_unknowns() -> None:
    """Three verdicts, and the unknown one is never a failure."""

    declarations = [
        {"field": "signal_lag", "location": "backtest.signal_lag", "declared_lag": 1, "rule": "signal_lag"},
        {"field": "index_membership", "location": "universe_rules.index_membership.lag_days", "declared_lag": 5, "rule": "index_membership"},
        {"field": "listing_age", "location": "universe_rules.listing_age.min_days", "declared_lag": None, "rule": "listing_age"},
        {"field": "unknown_field", "location": "signal.something", "declared_lag": 2, "rule": "signal_lag"},
    ]
    report = data_leakage_report(
        declarations,
        field_availability_lag={"signal_lag": 2, "index_membership": 3},
        announcement_to_effective={"signal_lag": 0},
        provenance={"snapshot_id": "s", "snapshot_checksum": "c"},
    )
    reasons = {(risk["field"], risk["reason"]) for risk in report["risks"]}
    assert ("signal_lag", "declared_lag_below_available_lag") in reasons
    assert ("unknown_field", "unknown") in reasons
    assert ("listing_age", "unknown") in reasons
    assert ("index_membership", "unknown_announcement_rule") in reasons
    # The lagged-but-declared field is clean on its own axis.
    assert not any(
        risk["field"] == "index_membership" and risk["reason"] == "declared_lag_below_available_lag"
        for risk in report["risks"]
    )
    undeclared_fields = {entry["field"] for entry in report["undeclared"]}
    assert undeclared_fields == {"listing_age", "unknown_field"}
    assert all(entry["severity"] == LEAKAGE_UNKNOWN_SEVERITY for entry in report["undeclared"])
    assert report["provenance"] == {"snapshot_id": "s", "snapshot_checksum": "c"}
    assert report["field_availability_lag"] == {"signal_lag": 2, "index_membership": 3}

    worker = _data_leakage_report(
        declarations,
        field_availability_lag={"signal_lag": 2, "index_membership": 3},
        announcement_to_effective={"signal_lag": 0},
        provenance={"snapshot_id": "s", "snapshot_checksum": "c"},
    )
    assert worker == report


def test_data_leakage_clean_run_reports_no_risk() -> None:
    """A run that meets the standard produces an empty risk list, not a flag."""

    report = data_leakage_report(
        [{"field": "signal_lag", "location": "backtest.signal_lag", "declared_lag": 2, "rule": "signal_lag"}],
        field_availability_lag={"signal_lag": 2},
        announcement_to_effective={"signal_lag": 0},
        provenance={},
    )
    assert report["risks"] == []
    assert report["undeclared"] == []


@pytest.mark.parametrize(
    ("audit", "message"),
    [
        ({"feasibility": {"min_executable_ratio": 0.0}}, "audit feasibility min_executable_ratio must be in (0, 1]"),
        ({"feasibility": {"min_executable_ratio": -1}}, "audit feasibility min_executable_ratio must be in (0, 1]"),
        ({"feasibility": {"min_executable_ratio": "0.5"}}, "audit feasibility min_executable_ratio must be in (0, 1]"),
        ({"feasibility": {"allow_cash_residual": 1}}, "audit feasibility allow_cash_residual must be a boolean"),
        ({"feasibility": {}}, None),
        ({"feasibility": {"min_executable_ratio": 1.0, "allow_cash_residual": True}}, None),
        (
            {"data_leakage": {"field_availability_lag": {"a": 1}, "announcement_to_effective": {"b": 0}}},
            None,
        ),
        (
            {"data_leakage": {"field_availability_lag": {"a": -1}, "announcement_to_effective": {"b": 0}}},
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {"data_leakage": {"field_availability_lag": {"": 1}, "announcement_to_effective": {"b": 0}}},
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {"data_leakage": {"field_availability_lag": {"a": "1"}, "announcement_to_effective": {"b": 0}}},
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {"data_leakage": {"field_availability_lag": {"a": 1}, "announcement_to_effective": {"b": -2}}},
            "audit data_leakage announcement_to_effective must map non-empty rule names to non-negative whole days",
        ),
    ],
)
def test_batch7_audit_config_validation_on_both_sides(audit: dict, message: str | None) -> None:
    """Both ends accept or reject the same configurations with the same text."""

    if message is None:
        validate_audit_config(audit)
        _validate_audit(audit)
        return
    with pytest.raises(ValueError) as local_error:
        validate_audit_config(audit)
    assert str(local_error.value) == message
    with pytest.raises(ValueError) as worker_error:
        _validate_audit(audit)
    assert str(worker_error.value) == message
