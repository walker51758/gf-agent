"""Unit tests for SingleNameFloor (roadmap 6.2) and ScoreTransformWeight (7.4).

Local side only: spec validation, floor arithmetic, cap/floor stacking,
transform semantics and the disabled-default payload contract.  Two-sided
agreement lives in ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec
from rdagent.scenarios.gf_quant.strategy.composition import build_target_weights

DATES = pd.bdate_range("2024-01-02", periods=1)


def _score_row(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([values], index=DATES)


# ---- CompositionSpec validation --------------------------------------------


def test_floor_spec_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="weight_floor must be in"):
        CompositionSpec(top_n=5, weight_floor=0.0)
    with pytest.raises(ValueError, match="weight_floor must be in"):
        CompositionSpec(top_n=5, weight_floor=1.5)
    with pytest.raises(ValueError, match="weight_floor must not exceed weight_cap"):
        CompositionSpec(top_n=5, weight_cap=0.1, weight_floor=0.2)
    with pytest.raises(ValueError, match="weight_floor_allow_cash must be a boolean"):
        CompositionSpec(top_n=5, weight_floor=0.1, weight_floor_allow_cash="yes")  # type: ignore[arg-type]


def test_score_transform_spec_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="score_transform_kind must be one of"):
        CompositionSpec(top_n=5, weighting="score_transform", score_transform_kind="softmax")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="score_transform_kind is required"):
        CompositionSpec(top_n=5, weighting="score_transform")
    with pytest.raises(ValueError, match="only valid with the score_transform weighting"):
        CompositionSpec(top_n=5, weighting="equal", score_transform_kind="sigmoid")
    with pytest.raises(ValueError, match="score_transform_slope must be a finite positive number"):
        CompositionSpec(top_n=5, weighting="score_transform", score_transform_kind="linear", score_transform_slope=0.0)
    with pytest.raises(ValueError, match="score_transform_cut must be finite"):
        CompositionSpec(
            top_n=5,
            weighting="score_transform",
            score_transform_kind="truncated",
            score_transform_cut=float("nan"),
        )


def test_new_composition_fields_stay_out_of_the_disabled_payload() -> None:
    plain = CompositionSpec(top_n=5)
    payload = plain.as_payload()
    assert set(payload) == {"top_n", "weighting", "weight_cap"}
    assert "weight_floor" not in payload
    assert "score_transform_kind" not in payload

    floored = CompositionSpec(top_n=5, weight_floor=0.02)
    floored_payload = floored.as_payload()
    assert floored_payload["weight_floor"] == 0.02
    assert floored_payload["weight_floor_allow_cash"] is True
    assert "score_transform_kind" not in floored_payload

    transformed = CompositionSpec(
        top_n=5,
        weighting="score_transform",
        score_transform_kind="sigmoid",
        score_transform_slope=2.0,
        score_transform_cut=0.5,
    )
    transformed_payload = transformed.as_payload()
    assert transformed_payload["score_transform_kind"] == "sigmoid"
    assert transformed_payload["score_transform_slope"] == 2.0
    assert transformed_payload["score_transform_cut"] == 0.5
    assert "weight_floor" not in transformed_payload

    # Round trips through the canonical payload.
    assert CompositionSpec.from_payload(floored_payload) == floored
    assert CompositionSpec.from_payload(transformed_payload) == transformed
    assert CompositionSpec.from_payload(payload) == plain


# ---- SingleNameFloor --------------------------------------------------------


def test_floor_raises_small_holdings_and_funds_them_from_the_excess() -> None:
    # Rank weights [0.4, 0.3, 0.2, 0.1]; D is raised to 0.15 and the deficit is
    # funded proportionally by the excess of A/B/C above the floor.
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="rank", weight_floor=0.15),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row.to_list() == pytest.approx([0.15 + 0.25 * 8 / 9, 0.15 + 0.15 * 8 / 9, 0.15 + 0.05 * 8 / 9, 0.15])
    assert row.sum() == pytest.approx(1.0)


def test_floor_leaves_names_at_the_floor_boundary_untouched() -> None:
    # D sits exactly on the floor (rank weight 0.1): nobody is below it, so the
    # row is the plain rank result.
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="rank", weight_floor=0.1),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    assert weights.loc[DATES[0]].to_list() == pytest.approx([0.4, 0.3, 0.2, 0.1])

    # Every holding exactly on the floor: untouched as well.
    equal = build_target_weights(
        CompositionSpec(top_n=4, weight_floor=0.25),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    assert equal.loc[DATES[0]].to_list() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_floor_is_applied_after_the_cap_and_never_rebreaches_it() -> None:
    # cap 0.3 water-fills to [0.3, 0.3, 0.2667, 0.1333]; the floor then raises
    # D to 0.15 funded by the others, and nobody ends above the cap.
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="rank", weight_cap=0.3, weight_floor=0.15),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row["A"] == pytest.approx(0.15 + 0.15 * 0.96)
    assert row["B"] == pytest.approx(0.15 + 0.15 * 0.96)
    assert row["C"] == pytest.approx(0.15 + (0.2666666667 - 0.15) * 0.96)
    assert row["D"] == pytest.approx(0.15)
    assert row.sum() == pytest.approx(1.0)
    assert row.max() <= 0.3 + 1e-12


def test_floor_keeps_the_cap_cash_residual_instead_of_filling_it() -> None:
    # 4 x 0.2 cap leaves 0.2 in cash; a floor below every holding changes
    # nothing and the residual stays as cash.
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2, weight_floor=0.1),
        _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
    )
    row = weights.loc[DATES[0]]
    assert row.to_list() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert row.sum() == pytest.approx(0.8)

    with pytest.raises(ValueError, match="forbids a cash residual"):
        build_target_weights(
            CompositionSpec(top_n=4, weighting="proportional", weight_cap=0.2, weight_floor=0.1, weight_floor_allow_cash=False),
            _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
        )


def test_floor_higher_than_the_budget_fails_closed() -> None:
    # Freezing every holding at the floor needs 1.4 of budget: infeasible.
    with pytest.raises(ValueError, match="weight_floor is infeasible"):
        build_target_weights(
            CompositionSpec(top_n=4, weighting="rank", weight_floor=0.35),
            _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}),
        )


# ---- ScoreTransformWeight ---------------------------------------------------


def test_linear_transform_matches_proportional_allocation() -> None:
    score = _score_row({"A": 0.9, "B": 0.6, "C": 0.3})
    linear = build_target_weights(
        CompositionSpec(top_n=3, weighting="score_transform", score_transform_kind="linear", score_transform_slope=2.5),
        score,
    )
    proportional = build_target_weights(CompositionSpec(top_n=3, weighting="proportional"), score)
    np.testing.assert_allclose(linear.to_numpy(), proportional.to_numpy(), rtol=1e-12)


def test_truncated_transform_zeroes_scores_below_the_cut() -> None:
    weights = build_target_weights(
        CompositionSpec(top_n=4, weighting="score_transform", score_transform_kind="truncated", score_transform_cut=0.5),
        _score_row({"A": 0.9, "B": 0.7, "C": 0.3, "D": 0.1}),
    )
    row = weights.loc[DATES[0]]
    # A and B split the book; C and D were selected but truncated to zero.
    assert row.to_list() == pytest.approx([2 / 3, 1 / 3, 0.0, 0.0])

    # The cap water-filling never revives a truncated name.
    capped = build_target_weights(
        CompositionSpec(
            top_n=4,
            weighting="score_transform",
            score_transform_kind="truncated",
            score_transform_cut=0.5,
            weight_cap=0.5,
        ),
        _score_row({"A": 0.9, "B": 0.7, "C": 0.3, "D": 0.1}),
    )
    assert capped.loc[DATES[0]].to_list() == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_sigmoid_transform_midpoint_and_range() -> None:
    # sigmoid(slope) + sigmoid(-slope) == 1, so the weights are the sigmoids.
    weights = build_target_weights(
        CompositionSpec(
            top_n=2,
            weighting="score_transform",
            score_transform_kind="sigmoid",
            score_transform_slope=2.0,
            score_transform_cut=0.5,
        ),
        _score_row({"A": 1.0, "B": 0.0}),
    )
    row = weights.loc[DATES[0]]
    assert row.to_list() == pytest.approx([0.7310585786300049, 0.2689414213699951])
    assert row.sum() == pytest.approx(1.0)


def test_score_transform_fails_closed_on_degenerate_scores() -> None:
    # Linear keeps the proportional no-shorting rule: a non-positive transformed
    # score must raise, not become a negative position.
    with pytest.raises(ValueError, match="strictly positive transformed scores"):
        build_target_weights(
            CompositionSpec(top_n=3, weighting="score_transform", score_transform_kind="linear", score_transform_slope=2.0),
            _score_row({"A": 1.0, "B": 0.0, "C": 0.5}),
        )
    # Truncated fails closed only when every selected score is below the cut.
    with pytest.raises(ValueError, match="at least one score above the cut"):
        build_target_weights(
            CompositionSpec(top_n=2, weighting="score_transform", score_transform_kind="truncated", score_transform_cut=2.0),
            _score_row({"A": 1.0, "B": 0.5}),
        )
    # A sigmoid that numerically underflows to zero fails closed as well.
    with pytest.raises(ValueError, match="strictly positive transformed scores"):
        build_target_weights(
            CompositionSpec(
                top_n=2,
                weighting="score_transform",
                score_transform_kind="sigmoid",
                score_transform_slope=1e6,
                score_transform_cut=0.5,
            ),
            _score_row({"A": 0.0, "B": 1.0}),
        )


def test_disabled_defaults_keep_the_plain_paths() -> None:
    score = _score_row({"A": 3.0, "B": 2.0, "C": 1.0})
    plain = build_target_weights(CompositionSpec(top_n=3), score)
    floored_off = build_target_weights(CompositionSpec(top_n=3, weight_floor=None), score)
    np.testing.assert_array_equal(plain.to_numpy(), floored_off.to_numpy())
    assert plain.loc[DATES[0]].to_list() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
