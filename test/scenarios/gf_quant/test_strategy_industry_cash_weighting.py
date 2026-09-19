"""Unit tests for IndustryWeightCap (6.3), CashResidualPolicy (6.8),
InverseVolatilityWeight (7.5) and HierarchicalWeighting (7.8).

Local side only: spec validation, per-operator semantics and boundaries, the
disabled-default payload contract and the canonical payload round trips.
Two-sided agreement lives in ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec
from rdagent.scenarios.gf_quant.strategy.composition import (
    _apply_industry_cap,
    _hierarchical_weights,
    _inverse_volatility_weights,
    build_target_weights,
)

DATES = pd.bdate_range("2024-01-02", periods=1)


def _score_row(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([values], index=DATES)


# ---- CompositionSpec validation ---------------------------------------------


def test_industry_cap_spec_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="industry_weight_cap must be in"):
        CompositionSpec(top_n=5, industry_weight_cap=0.0)
    with pytest.raises(ValueError, match="industry_weight_cap must be in"):
        CompositionSpec(top_n=5, industry_weight_cap=1.5)
    with pytest.raises(ValueError, match="industry_map is only valid with industry_weight_cap"):
        CompositionSpec(top_n=5, industry_map={"A": "X"})
    with pytest.raises(ValueError, match="industry_map must map non-empty symbols"):
        CompositionSpec(top_n=5, industry_weight_cap=0.4, industry_map={"A": ""})
    with pytest.raises(ValueError, match="weight_floor must not exceed industry_weight_cap"):
        CompositionSpec(top_n=5, weight_floor=0.5, industry_weight_cap=0.3)


def test_cash_residual_max_spec_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="cash_residual_max must be in"):
        CompositionSpec(top_n=5, cash_residual_max=0.0)
    with pytest.raises(ValueError, match="cash_residual_max must be in"):
        CompositionSpec(top_n=5, cash_residual_max=1.5)


def test_inverse_volatility_spec_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="inverse_vol_window must be a whole number"):
        CompositionSpec(top_n=5, weighting="inverse_volatility", inverse_vol_window=2.5)
    with pytest.raises(ValueError, match="inverse_vol_window must be at least 2"):
        CompositionSpec(top_n=5, weighting="inverse_volatility", inverse_vol_window=1)
    with pytest.raises(ValueError, match="inverse_vol_min must be a positive finite number"):
        CompositionSpec(top_n=5, weighting="inverse_volatility", inverse_vol_min=0.0)
    with pytest.raises(ValueError, match="inverse_vol_multiply_score must be a boolean"):
        CompositionSpec(top_n=5, weighting="inverse_volatility", inverse_vol_multiply_score="yes")  # type: ignore[arg-type]


def test_hierarchical_spec_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="hierarchical weighting requires hierarchical_groups"):
        CompositionSpec(top_n=5, weighting="hierarchical")
    with pytest.raises(ValueError, match="hierarchical_groups is only valid with the hierarchical weighting"):
        CompositionSpec(top_n=5, hierarchical_groups={"A": ("X",)})
    with pytest.raises(ValueError, match="hierarchical_groups must be a non-empty object"):
        CompositionSpec(top_n=5, weighting="hierarchical", hierarchical_groups={})
    with pytest.raises(ValueError, match="must not appear in two groups"):
        CompositionSpec(
            top_n=5, weighting="hierarchical", hierarchical_groups={"g1": ("A", "B"), "g2": ("B", "C")}
        )
    with pytest.raises(ValueError, match="hierarchical_group_weights must cover exactly the hierarchical_groups keys"):
        CompositionSpec(
            top_n=5,
            weighting="hierarchical",
            hierarchical_groups={"g1": ("A",), "g2": ("B",)},
            hierarchical_group_weights={"g1": 1.0},
        )
    with pytest.raises(ValueError, match="hierarchical_group_weights values must be finite positive numbers"):
        CompositionSpec(
            top_n=5,
            weighting="hierarchical",
            hierarchical_groups={"g1": ("A",), "g2": ("B",)},
            hierarchical_group_weights={"g1": 1.0, "g2": -0.5},
        )


# ---- IndustryWeightCap ------------------------------------------------------


def test_industry_cap_scales_the_over_group_and_redistributes_once() -> None:
    # Rank weights [0.4, 0.3, 0.2, 0.1]; A+B (industry X) sum to 0.7, so they
    # are frozen at 0.55 and the freed 0.15 flows to C and D proportionally.
    # Y then sums to 0.45, inside the cap, so nothing else binds.
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="rank", industry_weight_cap=0.55, industry_map={"A": "X", "B": "X", "C": "Y", "D": "Y"}),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row["A"] == pytest.approx(0.4 * 0.55 / 0.7)
    assert row["B"] == pytest.approx(0.3 * 0.55 / 0.7)
    assert row["C"] == pytest.approx(0.2 + 0.15 * 0.2 / 0.3)
    assert row["D"] == pytest.approx(0.1 + 0.15 * 0.1 / 0.3)
    assert row.sum() == pytest.approx(1.0)


def test_industry_cap_revisits_groups_pushed_over_by_redistribution() -> None:
    # A tight cap makes both industries bind: the freed excess pushes Y over
    # the cap too, the next round freezes it, and whatever cannot be placed
    # anywhere stays as cash (0.10 here).
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="rank", industry_weight_cap=0.45, industry_map={"A": "X", "B": "X", "C": "Y", "D": "Y"}),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row["A"] + row["B"] == pytest.approx(0.45)
    assert row["C"] + row["D"] == pytest.approx(0.45)
    assert row.sum() == pytest.approx(0.9)


def test_industry_cap_leaves_groups_on_the_cap_untouched() -> None:
    # X sits exactly on the cap: nobody is scaled, the row is the plain rank
    # result.
    weights = build_target_weights(
        CompositionSpec(top_n=2, weighting="rank", industry_weight_cap=0.7, industry_map={"A": "X", "B": "Y"}),
        _score_row({"A": 2.0, "B": 1.0}),
    )
    assert weights.loc[DATES[0]].to_list() == pytest.approx([0.6666666667, 0.3333333333])


def test_industry_cap_all_same_industry_leaves_cash_residual() -> None:
    # Every name is in one industry: the whole book is one group and is
    # scaled down proportionally onto the cap, leaving 0.6 in cash.
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="rank", industry_weight_cap=0.4, industry_map={"A": "X", "B": "X", "C": "X", "D": "X"}),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row.sum() == pytest.approx(0.4)
    assert row.to_list() == pytest.approx([0.16, 0.12, 0.08, 0.04])


def test_industry_cap_never_pushes_a_name_back_over_the_single_name_cap() -> None:
    weights = build_target_weights(
        CompositionSpec(
            top_n=4,
            weighting="rank",
            weight_cap=0.3,
            industry_weight_cap=0.45,
            industry_map={"A": "X", "B": "X", "C": "Y", "D": "Y"},
        ),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row.max() <= 0.3 + 1e-12
    assert row["A"] + row["B"] <= 0.45 + 1e-9


def test_industry_cap_groups_unmapped_names_together() -> None:
    # B, C, D are absent from the map: they share the reserved group, which
    # the cap binds like any other industry.  The freed excess first pushes A
    # over its own group cap; the next round brings it back to the cap, so
    # both groups end exactly on 0.45.
    out = _apply_industry_cap(
        pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}),
        0.45,
        {"A": "X"},
    )
    assert out["A"] == pytest.approx(0.45)
    assert out["B"] + out["C"] + out["D"] == pytest.approx(0.45)


def test_industry_cap_runs_before_the_floor() -> None:
    # The industry cap first caps X at 0.45, then the floor raises D; the
    # floor is the last word and no name ends below it.
    weights = build_target_weights(
        CompositionSpec(
            top_n=4,
            weighting="rank",
            weight_floor=0.15,
            industry_weight_cap=0.45,
            industry_map={"A": "X", "B": "X", "C": "Y", "D": "Y"},
        ),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert (row[row > 0] >= 0.15 - 1e-12).all()
    assert row["A"] + row["B"] == pytest.approx(0.45)


# ---- CashResidualPolicy -----------------------------------------------------


def test_cash_residual_max_accepts_the_capped_book_residual() -> None:
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2, cash_residual_max=0.25),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row.to_list() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert row.sum() == pytest.approx(0.8)


def test_cash_residual_max_fails_closed_when_exceeded() -> None:
    # 4 x 0.2 cap leaves 0.2 in cash: above the 0.1 bound the row fails.
    with pytest.raises(ValueError, match="cash residual exceeds cash_residual_max"):
        build_target_weights(
            CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2, cash_residual_max=0.1),
            _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
        )


def test_cash_residual_max_disabled_keeps_the_old_path_bit_for_bit() -> None:
    score = _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    legacy = build_target_weights(CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2), score)
    same = build_target_weights(
        CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2, cash_residual_max=0.5), score
    )
    np.testing.assert_array_equal(legacy.to_numpy(), same.to_numpy())


# ---- InverseVolatilityWeight -------------------------------------------------


def _vol_history() -> pd.DataFrame:
    # A is twice as volatile as B, C sits in between.
    return pd.DataFrame(
        {
            "A": [0.04, -0.04, 0.04, -0.04, 0.04, -0.04],
            "B": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
            "C": [0.02, -0.02, 0.02, -0.02, 0.02, -0.02],
        }
    )


def test_inverse_volatility_weights_are_proportional_to_one_over_vol() -> None:
    spec = CompositionSpec(top_n=3, weighting="inverse_volatility")
    scores = pd.Series({"A": 0.9, "B": 0.9, "C": 0.9})
    row = _inverse_volatility_weights(spec, scores, _vol_history())
    assert row["B"] == pytest.approx(row["A"] * 4)
    assert row["C"] == pytest.approx(row["A"] * 2)
    assert row.sum() == pytest.approx(1.0)


def test_inverse_volatility_window_slices_the_history() -> None:
    spec = CompositionSpec(top_n=2, weighting="inverse_volatility", inverse_vol_window=2)
    scores = pd.Series({"A": 0.9, "B": 0.9})
    history = _vol_history()
    # A is calm (B-like volatility) inside the last two rows only: with the
    # window honoured A and B weigh the same; without it, A's full-history
    # volatility is 4x B's and the weights would differ sharply.
    history.iloc[-2:, 0] = [0.01, -0.01]
    row = _inverse_volatility_weights(spec, scores, history)
    assert row["A"] == pytest.approx(row["B"])


def test_inverse_volatility_min_floors_the_estimates() -> None:
    spec = CompositionSpec(top_n=2, weighting="inverse_volatility", inverse_vol_min=0.05)
    scores = pd.Series({"A": 0.9, "B": 0.9})
    row = _inverse_volatility_weights(spec, scores, _vol_history())
    assert row["A"] == pytest.approx(row["B"])  # both vols floored to 0.05


def test_inverse_volatility_multiply_score_tilts_towards_high_scores() -> None:
    scores = pd.Series({"A": 0.9, "B": 0.3})
    plain = _inverse_volatility_weights(CompositionSpec(top_n=2, weighting="inverse_volatility"), scores, _vol_history())
    tilted = _inverse_volatility_weights(
        CompositionSpec(top_n=2, weighting="inverse_volatility", inverse_vol_multiply_score=True), scores, _vol_history()
    )
    assert tilted["A"] > plain["A"]
    assert tilted["B"] < plain["B"]


def test_inverse_volatility_falls_back_to_equal_weight_on_a_single_row() -> None:
    spec = CompositionSpec(top_n=3, weighting="inverse_volatility")
    scores = pd.Series({"A": 0.9, "B": 0.9, "C": 0.9})
    row = _inverse_volatility_weights(spec, scores, _vol_history().iloc[:1])
    assert row.to_list() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_inverse_volatility_fails_closed_on_degenerate_estimates() -> None:
    spec = CompositionSpec(top_n=2, weighting="inverse_volatility")
    scores = pd.Series({"A": 0.9, "B": 0.9})
    zero_variance = _vol_history()
    zero_variance["B"] = 0.01
    with pytest.raises(ValueError, match="requires a positive volatility estimate"):
        _inverse_volatility_weights(spec, scores, zero_variance)
    single_observation = _vol_history()
    single_observation.loc[:, "B"] = [np.nan, np.nan, np.nan, np.nan, np.nan, 0.01]
    with pytest.raises(ValueError, match="requires a positive volatility estimate"):
        _inverse_volatility_weights(spec, scores, single_observation)
    with pytest.raises(ValueError, match="inverse_volatility weighting requires a returns panel"):
        _inverse_volatility_weights(spec, scores, None)


def test_inverse_volatility_requires_a_returns_panel_in_the_builder() -> None:
    with pytest.raises(ValueError, match="inverse_volatility weighting requires a returns panel"):
        build_target_weights(
            CompositionSpec(top_n=2, weighting="inverse_volatility"),
            _score_row({"A": 4.0, "B": 3.0}),
        )


def test_inverse_volatility_panel_builder_uses_decision_date_history() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    score = pd.DataFrame({"A": 5.0, "B": 4.0, "C": 3.0}, index=dates)
    returns = pd.DataFrame({"A": 0.04, "B": 0.01, "C": 0.02}, index=dates) * 1.0
    returns.iloc[0] = 0.0  # placeholder before any history exists
    weights = build_target_weights(CompositionSpec(top_n=2, weighting="inverse_volatility"), score, returns=returns)
    # First decision date: a single realised row -> equal fallback.
    assert weights.iloc[0].to_list() == pytest.approx([0.5, 0.5, 0.0])
    # Later dates: B (the least volatile) must hold the largest weight.
    assert (weights.iloc[1:, 1] > weights.iloc[1:, 0]).all()


# ---- HierarchicalWeighting ---------------------------------------------------


def test_hierarchical_splits_group_weight_equally_inside() -> None:
    spec = CompositionSpec(
        top_n=3,
        weighting="hierarchical",
        hierarchical_groups={"big": ("A", "B"), "small": ("C", "D")},
        hierarchical_group_weights={"big": 0.7, "small": 0.3},
    )
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0})
    row = _hierarchical_weights(spec, scores)
    assert row.to_list() == pytest.approx([0.35, 0.35, 0.3])


def test_hierarchical_defaults_to_equal_group_weights_and_skips_empty_groups() -> None:
    spec = CompositionSpec(
        top_n=3,
        weighting="hierarchical",
        hierarchical_groups={"g1": ("A", "B"), "g2": ("C", "D"), "g3": ("E", "F")},
    )
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0})
    row = _hierarchical_weights(spec, scores)
    # g3 holds no selected name, so g1 and g2 split the book 50/50.
    assert row.to_list() == pytest.approx([0.25, 0.25, 0.5])


def test_hierarchical_normalises_the_explicit_group_weights_over_active_groups() -> None:
    spec = CompositionSpec(
        top_n=3,
        weighting="hierarchical",
        hierarchical_groups={"g1": ("A", "B"), "g2": ("C", "D")},
        hierarchical_group_weights={"g1": 3.0, "g2": 1.0},
    )
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0})
    row = _hierarchical_weights(spec, scores)
    assert row.to_list() == pytest.approx([0.375, 0.375, 0.25])


def test_hierarchical_rejects_a_selected_name_outside_every_group() -> None:
    spec = CompositionSpec(top_n=2, weighting="hierarchical", hierarchical_groups={"g1": ("A", "B")})
    with pytest.raises(ValueError, match="requires every selected name to belong to a group"):
        _hierarchical_weights(spec, pd.Series({"A": 5.0, "Z": 1.0}))


# ---- Canonical payload contract ----------------------------------------------


def test_new_fields_stay_out_of_the_payload_while_disabled() -> None:
    payload = CompositionSpec(top_n=4, weighting="equal").as_payload()
    assert set(payload) == {"top_n", "weighting", "weight_cap"}
    assert "industry_weight_cap" not in payload
    assert "industry_map" not in payload
    assert "cash_residual_max" not in payload
    assert "inverse_vol_window" not in payload
    assert "hierarchical_groups" not in payload


def test_new_fields_round_trip_through_the_canonical_payload() -> None:
    industry = CompositionSpec(
        top_n=4, industry_weight_cap=0.45, industry_map={"A": "X", "B": "Y"}, cash_residual_max=0.2
    )
    inverse = CompositionSpec(
        top_n=4,
        weighting="inverse_volatility",
        inverse_vol_window=10,
        inverse_vol_min=0.001,
        inverse_vol_multiply_score=True,
    )
    hierarchical = CompositionSpec(
        top_n=4,
        weighting="hierarchical",
        hierarchical_groups={"g1": ("A", "B"), "g2": ("C", "D")},
        hierarchical_group_weights={"g1": 0.6, "g2": 0.4},
    )
    for spec in (industry, inverse, hierarchical):
        payload = spec.as_payload()
        assert CompositionSpec.from_payload(payload) == spec
    # The inverse-volatility defaults survive a payload without the fields.
    minimal = CompositionSpec(top_n=2, weighting="inverse_volatility")
    assert CompositionSpec.from_payload(minimal.as_payload()) == minimal
