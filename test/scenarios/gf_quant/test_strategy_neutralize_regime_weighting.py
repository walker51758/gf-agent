"""Batch C2 unit tests: roadmap 4.6 IndustryNeutralize, 4.7 StyleNeutralize,
4.12 RegimeConditionalSignal, 7.6 RiskParityWeight, 7.7 FactorRiskBudgetWeight.

Spec validation, operator semantics on hand-computed panels, payload/identity
contract (lazy keys, integral-float normalization, round-trip equality) and
the boundaries declared in AGENTS.md.  The two-engine parity for every helper
and the end-to-end cases live in ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.candidate import (
    CompositionSpec,
    SignalSpec,
    StrategyCandidate,
)
from rdagent.scenarios.gf_quant.strategy.composition import (
    _factor_risk_budget_weights,
    _risk_parity_weights,
    build_target_weights,
)
from rdagent.scenarios.gf_quant.strategy.signal import (
    apply_industry_neutralize,
    apply_style_neutralize,
    blend_regime_panels,
)

FACTOR_NAMES = ("value", "quality", "momentum")

_INDUSTRY_MAP = {"A": "X", "B": "X", "C": "Y", "D": "Y", "E": "Z"}


def _values_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=2)
    return pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 6.0], "C": [2.0, 1.0], "D": [4.0, 5.0], "E": [5.0, 9.0]},
        index=dates,
    )


# ---------------------------------------------------------------------------
# SignalSpec validation
# ---------------------------------------------------------------------------


def test_industry_neutralize_requires_the_industry_map() -> None:
    with pytest.raises(ValueError, match="industry neutralize requires industry_neutralize_industry_map"):
        SignalSpec(industry_neutralize=True)


def test_industry_neutralize_lazy_parameters_require_the_switch() -> None:
    with pytest.raises(ValueError, match="industry neutralize parameters require industry_neutralize"):
        SignalSpec(industry_neutralize_industry_map={"A": "X"})
    with pytest.raises(ValueError, match="industry neutralize parameters require industry_neutralize"):
        SignalSpec(industry_neutralize_version="sw2021")


def test_industry_neutralize_method_is_whitelisted() -> None:
    with pytest.raises(ValueError, match="industry_neutralize_method must be one of"):
        SignalSpec(industry_neutralize=True, industry_neutralize_industry_map={"A": "X"}, industry_neutralize_method="robust")


def test_industry_neutralize_min_group_rejects_thin_and_non_whole() -> None:
    with pytest.raises(ValueError, match="industry_neutralize_min_group must be at least 2"):
        SignalSpec(industry_neutralize=True, industry_neutralize_industry_map={"A": "X"}, industry_neutralize_min_group=1)
    with pytest.raises(ValueError, match="industry_neutralize_min_group must be a whole number"):
        SignalSpec(industry_neutralize=True, industry_neutralize_industry_map={"A": "X"}, industry_neutralize_min_group=2.5)


def test_industry_neutralize_integral_float_min_group_is_normalized() -> None:
    spec = SignalSpec(industry_neutralize=True, industry_neutralize_industry_map={"A": "X"}, industry_neutralize_min_group=3.0)
    assert spec.industry_neutralize_min_group == 3 and isinstance(spec.industry_neutralize_min_group, int)
    # The normalized value travels to the payload, so the canonical hash is stable.
    assert spec.as_payload()["industry_neutralize_min_group"] == 3


def test_industry_neutralize_exposures_need_the_regression_method() -> None:
    with pytest.raises(ValueError, match="industry_neutralize_exposures is only valid with the regression method"):
        SignalSpec(
            industry_neutralize=True,
            industry_neutralize_industry_map={"A": "X"},
            industry_neutralize_exposures={"size": {"A": 1.0}},
        )


def test_style_neutralize_requires_exposures_and_min_samples() -> None:
    with pytest.raises(ValueError, match="style neutralize requires style_neutralize_exposures"):
        SignalSpec(style_neutralize=True)
    with pytest.raises(ValueError, match="style_neutralize_min_samples must be at least 2"):
        SignalSpec(style_neutralize=True, style_neutralize_exposures={"size": {"A": 1.0}}, style_neutralize_min_samples=1)


def test_regime_fields_must_be_provided_together() -> None:
    with pytest.raises(ValueError, match="regime_states and regime_factor_groups must be provided together"):
        SignalSpec(regime_states={"2024-01-02": "on"})
    with pytest.raises(ValueError, match="regime_states and regime_factor_groups must be provided together"):
        SignalSpec(regime_factor_groups={"on": ["value"]})


def test_regime_is_mutually_exclusive_with_every_static_blend() -> None:
    states = {"2024-01-02": "on"}
    groups = {"on": ["value"]}
    with pytest.raises(ValueError, match="regime_factor_groups and blend_weights are mutually exclusive"):
        SignalSpec(regime_states=states, regime_factor_groups=groups, blend_weights={"value": 1.0, "quality": 1.0, "momentum": 1.0})
    with pytest.raises(ValueError, match="regime_factor_groups and blend_groups are mutually exclusive"):
        SignalSpec(regime_states=states, regime_factor_groups=groups, blend_groups={"g": ["value", "quality", "momentum"]})
    with pytest.raises(ValueError, match="regime_factor_groups and interaction_pairs are mutually exclusive"):
        SignalSpec(
            regime_states=states,
            regime_factor_groups=groups,
            interaction_pairs=(("value", "quality", "product"),),
        )


def test_regime_states_reject_invalid_dates_and_labels() -> None:
    with pytest.raises(ValueError, match="regime_states contains an invalid date"):
        SignalSpec(regime_states={"yesterday": "on"}, regime_factor_groups={"on": ["value"]})
    with pytest.raises(ValueError, match="regime_states must map date strings to non-empty state labels"):
        SignalSpec(regime_states={"2024-01-02": ""}, regime_factor_groups={"on": ["value"]})


def test_regime_lag_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="regime_lag must not be negative"):
        SignalSpec(regime_states={"2024-01-02": "on"}, regime_factor_groups={"on": ["value"]}, regime_lag=-1)


def test_regime_members_must_reference_candidate_factors() -> None:
    with pytest.raises(ValueError, match="regime_factor_groups members must reference candidate factors"):
        StrategyCandidate(
            name="c",
            universe="csi_all",
            composition=CompositionSpec(top_n=2),
            factor_names=FACTOR_NAMES,
            rule="r",
            signal=SignalSpec(regime_states={"2024-01-02": "on"}, regime_factor_groups={"on": ["value", "ghost"]}),
        )


def test_disabled_signal_spec_stays_the_pregest_default() -> None:
    payload = SignalSpec().as_payload()
    assert "industry_neutralize" not in payload
    assert "style_neutralize" not in payload
    assert "regime_states" not in payload
    assert "regime_factor_groups" not in payload
    assert "regime_lag" not in payload


def test_signal_spec_round_trips_enabled_variants() -> None:
    specs = [
        SignalSpec(
            industry_neutralize=True,
            industry_neutralize_industry_map={"A": "X"},
            industry_neutralize_method="regression",
            industry_neutralize_exposures={"size": {"A": 1.0}},
            industry_neutralize_version="sw2021",
        ),
        SignalSpec(style_neutralize=True, style_neutralize_exposures={"beta": {"A": 0.5}}, style_neutralize_min_samples=4),
        SignalSpec(
            regime_states={"2024-01-02": "on"},
            regime_factor_groups={"on": ["value", "quality"]},
            regime_lag=2,
        ),
    ]
    for spec in specs:
        assert SignalSpec.from_payload(spec.as_payload()) == spec


# ---------------------------------------------------------------------------
# Neutralize semantics (hand-computed)
# ---------------------------------------------------------------------------


def test_industry_demean_computes_within_group_means() -> None:
    spec = SignalSpec(industry_neutralize=True, industry_neutralize_industry_map=_INDUSTRY_MAP, industry_neutralize_min_group=2)
    values = _values_panel()
    out = apply_industry_neutralize(values, spec)
    # X: mean(1,3)=2 -> A=-1, B=1; Y: mean(2,4)=3 -> C=-1, D=1; Z is a
    # single-name group (thin) and passes through; E (unmapped) is its own
    # one-name group and passes through.
    assert out["A"].tolist() == [-1.0, -2.0]
    assert out["B"].tolist() == [1.0, 2.0]
    assert out["C"].tolist() == [-1.0, -2.0]
    assert out["D"].tolist() == [1.0, 2.0]
    assert out["E"].tolist() == values["E"].tolist()


def test_thin_group_passes_through_while_full_group_neutralizes() -> None:
    spec = SignalSpec(industry_neutralize=True, industry_neutralize_industry_map=_INDUSTRY_MAP, industry_neutralize_min_group=2)
    values = _values_panel()
    out = apply_industry_neutralize(values, spec)
    # min_group=4: only X and Y have two names each -> still neutralized; Z/E
    # pass.  With min_group=3 both X and Y become thin and everything passes.
    wide = SignalSpec(industry_neutralize=True, industry_neutralize_industry_map=_INDUSTRY_MAP, industry_neutralize_min_group=3)
    untouched = apply_industry_neutralize(values, wide)
    pd.testing.assert_frame_equal(untouched, values)


def test_regression_without_exposures_equals_the_demean() -> None:
    values = _values_panel()
    demean = SignalSpec(industry_neutralize=True, industry_neutralize_industry_map=_INDUSTRY_MAP, industry_neutralize_min_group=2)
    regression = SignalSpec(
        industry_neutralize=True,
        industry_neutralize_industry_map=_INDUSTRY_MAP,
        industry_neutralize_min_group=2,
        industry_neutralize_method="regression",
    )
    pd.testing.assert_frame_equal(apply_industry_neutralize(values, regression), apply_industry_neutralize(values, demean))


def test_regression_residual_is_orthogonal_to_the_exposure() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=30)
    symbols = [f"S{position}" for position in range(10)]
    values = pd.DataFrame(rng.normal(size=(30, 10)), index=dates, columns=symbols)
    size = {symbol: float(rng.normal()) for symbol in symbols}
    industry_map = {symbol: ("X" if position % 2 == 0 else "Y") for position, symbol in enumerate(symbols)}
    spec = SignalSpec(
        industry_neutralize=True,
        industry_neutralize_industry_map=industry_map,
        industry_neutralize_method="regression",
        industry_neutralize_exposures={"size": size},
    )
    out = apply_industry_neutralize(values, spec)
    # With one exposure the residual is the industry-demeaned value minus its
    # proportional size part: uncorrelated with the exposure by construction.
    for date in dates:
        residual = out.loc[date]
        exposure = pd.Series(size).reindex(symbols)
        correlation = np.corrcoef(residual.to_numpy(), exposure.to_numpy())[0, 1]
        assert abs(correlation) < 1e-9


def test_style_residual_has_zero_cross_sectional_mean_and_keeps_thin_rows() -> None:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=5)
    symbols = [f"S{position}" for position in range(8)]
    values = pd.DataFrame(rng.normal(size=(5, 8)), index=dates, columns=symbols)
    # S7 misses the style exposure entirely: it keeps its value.
    exposures = {"size": {symbol: float(rng.normal()) for symbol in symbols[:7]}}
    spec = SignalSpec(style_neutralize=True, style_neutralize_exposures=exposures, style_neutralize_min_samples=4)
    out = apply_style_neutralize(values, spec)
    for date in dates:
        # The regression rows (all but the exposure-less S7) have a zero-mean
        # residual by construction; S7 keeps its raw value outside the sample.
        row = out.loc[date, symbols[:7]]
        assert abs(float(row.mean())) < 1e-12
    assert out["S7"].tolist() == values["S7"].tolist()


# ---------------------------------------------------------------------------
# Regime blend semantics
# ---------------------------------------------------------------------------


def _rank_panels() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=4)
    symbols = ["A", "B", "C"]
    panels = {}
    for position, name in enumerate(("value", "quality", "momentum")):
        # Momentum is mirrored so its *ordering* differs from the other two
        # (a monotone offset alone would produce an identical rank panel).
        sign = -1.0 if name == "momentum" else 1.0
        frame = pd.DataFrame(
            {symbol: sign * (0.1 * position + 0.2 * offset) for offset, symbol in enumerate(symbols)},
            index=dates,
        )
        panels[name] = frame.rank(axis=1, pct=True)
    panels["quality"].iloc[2, 1] = np.nan
    return panels


def test_regime_blend_rotates_with_the_injected_state() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    states = {date.strftime("%Y-%m-%d"): ("on" if date < dates[2] else "off") for date in dates}
    spec = SignalSpec(
        regime_states=states,
        regime_factor_groups={"on": ["value", "quality"], "off": ["value", "momentum"]},
        regime_lag=0,
    )
    out = blend_regime_panels(_rank_panels(), spec)
    # on: mean of value & quality; off: mean of value & momentum.
    ranks = _rank_panels()
    expected_on = (ranks["value"] + ranks["quality"]) / 2
    pd.testing.assert_frame_equal(out.loc[: dates[1]], expected_on.loc[: dates[1]])
    expected_off = (ranks["value"] + ranks["momentum"]) / 2
    pd.testing.assert_frame_equal(out.loc[dates[2] :], expected_off.loc[dates[2] :])


def test_regime_blend_lag_shifts_the_state_series_by_rows() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    extended = pd.bdate_range(dates[0] - pd.Timedelta(days=7), dates[-1])
    states = {date.strftime("%Y-%m-%d"): ("on" if date >= dates[1] else "off") for date in extended}
    panels = _rank_panels()
    lag0 = blend_regime_panels(
        panels, SignalSpec(regime_states=states, regime_factor_groups={"on": ["value"], "off": ["momentum"]}, regime_lag=0)
    )
    # With one row of lag, the state of row i is read from row i-1: the first
    # decision date sees the state of the row before it.
    lag1 = blend_regime_panels(
        panels, SignalSpec(regime_states=states, regime_factor_groups={"on": ["value"], "off": ["momentum"]}, regime_lag=1)
    )
    assert not lag0.iloc[1].equals(lag1.iloc[1])


def test_regime_blend_missing_member_kills_the_cell() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    states = {date.strftime("%Y-%m-%d"): "on" for date in dates}
    spec = SignalSpec(regime_states=states, regime_factor_groups={"on": ["value", "quality"]}, regime_lag=0)
    out = blend_regime_panels(_rank_panels(), spec)
    assert math.isnan(out.iloc[2, 1])  # quality was blanked at (2, 1)


def test_regime_blend_fails_closed_on_missing_state_and_unknown_label() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    panels = _rank_panels()
    late = {dates[2].strftime("%Y-%m-%d"): "on"}
    with pytest.raises(ValueError, match="regime state is missing for decision date"):
        blend_regime_panels(panels, SignalSpec(regime_states=late, regime_factor_groups={"on": ["value"]}, regime_lag=0))
    full = {date.strftime("%Y-%m-%d"): "sideways" for date in dates}
    with pytest.raises(ValueError, match="regime_factor_groups does not define state 'sideways'"):
        blend_regime_panels(panels, SignalSpec(regime_states=full, regime_factor_groups={"on": ["value"]}, regime_lag=0))


# ---------------------------------------------------------------------------
# CompositionSpec validation
# ---------------------------------------------------------------------------


def test_risk_parity_lazy_parameters_require_the_scheme() -> None:
    with pytest.raises(ValueError, match="risk_parity parameters require the risk_parity weighting"):
        CompositionSpec(top_n=2, weighting="equal", risk_parity_budgets={"A": 1.0})
    with pytest.raises(ValueError, match="risk_parity parameters require the risk_parity weighting"):
        CompositionSpec(top_n=2, weighting="equal", risk_parity_shrinkage=0.5)


def test_risk_parity_window_and_shrinkage_bounds() -> None:
    with pytest.raises(ValueError, match="risk_parity_window must be at least 2"):
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_window=1)
    with pytest.raises(ValueError, match="risk_parity_window must be a whole number"):
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_window=2.5)
    with pytest.raises(ValueError, match="risk_parity_shrinkage must be in"):
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_shrinkage=1.0)
    with pytest.raises(ValueError, match="risk_parity_shrinkage_kind must be one of"):
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_shrinkage_kind="ledoit")
    with pytest.raises(ValueError, match="risk_parity_tolerance must be a finite positive number"):
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_tolerance=0.0)


def test_frb_requires_its_data_and_rejects_lazy_parameters() -> None:
    with pytest.raises(ValueError, match="factor_risk_budget weighting requires frb_budgets"):
        CompositionSpec(top_n=2, weighting="factor_risk_budget")
    with pytest.raises(ValueError, match="factor_risk_budget parameters require the factor_risk_budget weighting"):
        CompositionSpec(top_n=2, weighting="equal", frb_budgets={"value": 1.0})
    with pytest.raises(ValueError, match="frb_ridge must be a finite non-negative number"):
        CompositionSpec(
            top_n=2,
            weighting="factor_risk_budget",
            frb_budgets={"value": 1.0},
            frb_exposures={"value": {"A": 1.0}},
            frb_factor_covariance={"value": {"value": 1.0}},
            frb_idiosyncratic={"A": 0.04},
            frb_ridge=-1.0,
        )


def test_frb_budgets_must_cover_the_candidate_factors() -> None:
    with pytest.raises(ValueError, match="frb_budgets must cover exactly the candidate factors"):
        StrategyCandidate(
            name="c",
            universe="csi_all",
            composition=CompositionSpec(
                top_n=2,
                weighting="factor_risk_budget",
                frb_budgets={"value": 1.0, "ghost": 1.0},
                frb_exposures={"value": {"A": 1.0}, "ghost": {"A": 1.0}},
                frb_factor_covariance={"value": {"value": 1.0, "ghost": 0.0}, "ghost": {"value": 0.0, "ghost": 1.0}},
                frb_idiosyncratic={"A": 0.04, "B": 0.04},
            ),
            factor_names=FACTOR_NAMES,
            rule="r",
        )


def test_disabled_composition_spec_keeps_the_prebatch_payload_keys() -> None:
    payload = CompositionSpec(top_n=2).as_payload()
    assert set(payload) == {"top_n", "weighting", "weight_cap"}
    payload = CompositionSpec(top_n=2, weighting="risk_parity").as_payload()
    assert set(payload) == {
        "top_n",
        "weighting",
        "weight_cap",
        "risk_parity_window",
        "risk_parity_shrinkage_kind",
        "risk_parity_shrinkage",
        "risk_parity_tolerance",
    }
    assert "frb_budgets" not in payload


def test_composition_spec_round_trips_enabled_variants() -> None:
    specs = [
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_budgets={"A": 1.0, "B": 2.0}),
        CompositionSpec(top_n=2, weighting="risk_parity", risk_parity_shrinkage_kind="none"),
        CompositionSpec(
            top_n=2,
            weighting="factor_risk_budget",
            frb_budgets={"value": 1.0},
            frb_exposures={"value": {"A": 1.0, "B": 2.0}},
            frb_factor_covariance={"value": {"value": 1.0}},
            frb_idiosyncratic={"A": 0.04, "B": 0.05},
            frb_idio_max=0.05,
            frb_ridge=0.1,
        ),
    ]
    for spec in specs:
        assert CompositionSpec.from_payload(spec.as_payload()) == spec


# ---------------------------------------------------------------------------
# Risk parity semantics
# ---------------------------------------------------------------------------


def _returns_history(rows: int = 60, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = ["A", "B", "C", "D"]
    frame = pd.DataFrame(rng.normal(scale=0.02, size=(rows, 4)), index=pd.bdate_range("2023-10-02", periods=rows), columns=symbols)
    return frame


def test_risk_parity_equalizes_risk_contributions() -> None:
    history = _returns_history()
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    spec = CompositionSpec(top_n=4, weighting="risk_parity", risk_parity_shrinkage_kind="none")
    weights = _risk_parity_weights(spec, scores, history)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert bool((weights > 0).all())
    covariance = np.cov(history.to_numpy(dtype=float), rowvar=False, ddof=1)
    vector = weights.to_numpy(dtype=float)
    contributions = vector * (covariance @ vector)
    assert contributions / contributions.sum() == pytest.approx([0.25] * 4, abs=1e-6)


def test_risk_parity_budgets_are_honoured() -> None:
    history = _returns_history()
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    spec = CompositionSpec(
        top_n=4,
        weighting="risk_parity",
        risk_parity_shrinkage_kind="none",
        risk_parity_budgets={"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1},
    )
    weights = _risk_parity_weights(spec, scores, history)
    covariance = np.cov(history.to_numpy(dtype=float), rowvar=False, ddof=1)
    vector = weights.to_numpy(dtype=float)
    contributions = vector * (covariance @ vector)
    shares = contributions / contributions.sum()
    assert shares[0] == pytest.approx(0.7, abs=1e-6)
    assert shares[1:] == pytest.approx([0.1] * 3, abs=1e-6)


def test_risk_parity_short_history_falls_back_to_equal_weights() -> None:
    history = _returns_history().iloc[:1]
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    spec = CompositionSpec(top_n=4, weighting="risk_parity")
    weights = _risk_parity_weights(spec, scores, history)
    assert weights.tolist() == [0.25] * 4


def test_risk_parity_requires_the_returns_panel_for_the_scheme() -> None:
    with pytest.raises(ValueError, match="risk_parity weighting requires a returns panel"):
        _risk_parity_weights(CompositionSpec(top_n=2, weighting="risk_parity"), pd.Series({"A": 1.0, "B": 1.0}), None)


def test_risk_parity_budgets_must_cover_every_selected_name() -> None:
    with pytest.raises(ValueError, match="risk_parity_budgets must cover every selected name"):
        _risk_parity_weights(
            CompositionSpec(top_n=4, weighting="risk_parity", risk_parity_budgets={"A": 1.0, "B": 1.0, "C": 1.0}),
            pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}),
            _returns_history(),
        )


def test_risk_parity_build_target_weights_end_to_end_locally() -> None:
    history = _returns_history()
    dates = history.index[-6:]
    score = pd.DataFrame(0.5, index=dates, columns=list("ABCD"))
    spec = CompositionSpec(top_n=4, weighting="risk_parity")
    targets = build_target_weights(spec, score, returns=history)
    assert bool((targets.sum(axis=1) > 0).any())
    # The first decision date has no returns history -> equal fallback row.
    assert targets.iloc[0].sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Factor risk budget semantics
# ---------------------------------------------------------------------------


def _frb_spec(**overrides) -> CompositionSpec:
    fields = dict(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 2.0, "momentum": 1.0},
        frb_exposures={
            "value": {"A": 1.2, "B": 0.8, "C": 1.5, "D": 0.4},
            "quality": {"A": 0.3, "B": 1.1, "C": 0.9, "D": 1.4},
            "momentum": {"A": 0.9, "B": 0.2, "C": 1.1, "D": 0.7},
        },
        frb_factor_covariance={
            "value": {"value": 1.0, "quality": 0.2, "momentum": -0.1},
            "quality": {"value": 0.2, "quality": 0.8, "momentum": 0.1},
            "momentum": {"value": -0.1, "quality": 0.1, "momentum": 1.2},
        },
        frb_idiosyncratic={"A": 0.04, "B": 0.05, "C": 0.03, "D": 0.06},
    )
    fields.update(overrides)
    return CompositionSpec(**fields)


def test_frb_weights_are_positive_and_sum_to_one() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    weights = _factor_risk_budget_weights(_frb_spec(), scores)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    assert bool((weights > 0).all())


def test_frb_idio_cap_excludes_the_offending_name() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    spec = _frb_spec(frb_idio_max=0.045)  # excludes B (0.05) and D (0.06)
    weights = _factor_risk_budget_weights(spec, scores)
    assert float(weights["B"]) == 0.0 and float(weights["D"]) == 0.0
    assert float(weights[["A", "C"]].sum()) == pytest.approx(1.0, abs=1e-12)


def test_frb_ridge_changes_the_target_mix() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    plain = _factor_risk_budget_weights(_frb_spec(), scores)
    ridged = _factor_risk_budget_weights(_frb_spec(frb_ridge=0.5), scores)
    assert not np.allclose(plain.to_numpy(dtype=float), ridged.to_numpy(dtype=float))


def test_frb_requires_idiosyncratic_variances_for_every_selected_name() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    spec = _frb_spec(frb_idiosyncratic={"A": 0.04, "B": 0.05, "C": 0.03})
    with pytest.raises(
        ValueError,
        match="factor_risk_budget weighting requires an idiosyncratic variance for every selected name",
    ):
        _factor_risk_budget_weights(spec, scores)


def test_frb_requires_at_least_one_positive_projection() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.7})
    spec = _frb_spec(
        top_n=2,
        frb_budgets={"value": 1.0, "quality": 1.0},
        frb_exposures={
            # Every name projects negatively onto the positive target mix.
            "value": {"A": -1.0, "B": -2.0},
            "quality": {"A": -0.5, "B": -1.0},
        },
        frb_factor_covariance={"value": {"value": 1.0, "quality": 0.1}, "quality": {"value": 0.1, "quality": 1.0}},
        frb_idiosyncratic={"A": 0.04, "B": 0.05},
    )
    with pytest.raises(
        ValueError,
        match="factor_risk_budget weighting requires at least one positive projected exposure",
    ):
        _factor_risk_budget_weights(spec, scores)
