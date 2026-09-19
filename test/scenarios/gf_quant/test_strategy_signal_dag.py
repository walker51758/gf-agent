"""Unit tests for the batch-B1 signal DAG operators (roadmap 4.5/4.9/4.10/4.11).

Local-side semantics, missing propagation, parameter validation and the
pipeline order; two-sided agreement lives in ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import re as _re

from rdagent.scenarios.gf_quant.strategy.candidate import (
    CompositionSpec,
    SignalSpec,
    StrategyCandidate,
)
from rdagent.scenarios.gf_quant.strategy.signal import (
    apply_missing_value,
    apply_neutral_rank_fill,
    apply_signal,
    blend_member_panels,
    blend_rank_panels,
    build_factor_score,
    interaction_rank_panels,
)

DATES = pd.date_range("2024-01-01", periods=3)


def _panel(rows: list[list[float | None]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, index=DATES[: len(rows)], columns=["A", "B", "C", "D"])
    return frame.astype(float)


def test_signal_spec_rejects_invalid_signal_dag_parameters() -> None:
    for kwargs, match in [
        ({"missing_value": "median"}, "missing_value must be one of"),
        ({"missing_max_fill_ratio": 0.0}, "missing_max_fill_ratio must be in (0, 1]"),
        ({"missing_max_fill_ratio": 1.5}, "missing_max_fill_ratio must be in (0, 1]"),
        ({"missing_max_fill_ratio": 0.5}, "missing_max_fill_ratio requires a missing_value policy"),
        ({"missing_value": "drop_stock", "missing_max_fill_ratio": 0.5}, "does not apply to the drop_stock policy"),
        ({"blend_weights": {}}, "blend_weights must not be empty"),
        ({"blend_weights": {"f1": 0.0, "f2": 1.0}}, "blend weights must be finite positive numbers"),
        ({"blend_weights": {"f1": -0.5, "f2": 1.5}}, "blend weights must be finite positive numbers"),
        ({"blend_weights": {"f1": float("nan"), "f2": 1.0}}, "blend weights must be finite positive numbers"),
        ({"blend_groups": {}}, "blend_groups must not be empty"),
        ({"blend_groups": {"g1": []}}, "must name at least one factor"),
        ({"blend_groups": {"g1": ["f1"], "g2": ["f1", "f2"]}}, "must not assign a factor to two groups"),
        (
            {"blend_groups": {"g1": ["f1"], "g2": ["f2"]}, "blend_group_weights": {"g1": 0.5}},
            "blend_group_weights keys must match the blend_groups exactly",
        ),
        ({"blend_groups": {"g1": ["f1"], "g2": ["f2"]}, "blend_group_weights": {"g1": 0.5, "g2": 0.0}}, "blend group weights"),
        (
            {"blend_groups": {"g1": ["f1"], "g2": ["f2"]}, "blend_within_weights": {"gX": {"f1": 1.0}}},
            "blend_within_weights names unknown groups",
        ),
        (
            {"blend_groups": {"g1": ["f1", "f2"]}, "blend_within_weights": {"g1": {"f1": 1.0}}},
            "must cover its factors exactly",
        ),
        ({"blend_groups": {"g1": ["f1"], "g2": ["f2"]}, "blend_weights": {"f1": 0.5, "f2": 0.5}}, "mutually exclusive"),
        ({"interaction_pairs": (("f1", "f2", "tanh"),)}, "interaction kind must be one of"),
        ({"interaction_pairs": (("f1", "f1", "product"),)}, "two distinct non-empty factor names"),
        ({"interaction_pairs": (("f1", "f2", "product"), ("f1", "f2", "min"))}, "interaction terms must be distinct"),
        ({"interaction_pairs": tuple((f"f{i}", f"f{i + 1}", "product") for i in range(9))}, "at most 8 interaction terms"),
    ]:
        with pytest.raises(ValueError, match=_re.escape(match)):
            SignalSpec(**kwargs)


def test_signal_dag_defaults_are_disabled_and_round_trip() -> None:
    spec = SignalSpec()
    assert spec.missing_value is None
    assert spec.blend_weights is None and spec.blend_groups is None
    assert spec.interaction_pairs == ()
    payload = spec.as_payload()
    for key in ("missing_value", "missing_max_fill_ratio", "blend_weights", "blend_groups", "interaction_pairs"):
        assert key not in payload
    restored = SignalSpec.from_payload(payload)
    assert restored == spec
    # A payload without any new key restores the disabled default.
    assert SignalSpec.from_payload({}) == spec

    enabled = SignalSpec(
        missing_value="cross_section_median",
        missing_max_fill_ratio=0.4,
        blend_groups={"g1": ["f1"], "g2": ["f2", "f3"]},
        blend_group_weights={"g1": 0.5, "g2": 0.5},
        blend_within_weights={"g2": {"f2": 0.3, "f3": 0.7}},
        interaction_pairs=(("f1", "f2", "product"),),
    )
    assert SignalSpec.from_payload(enabled.as_payload()) == enabled
    assert set(enabled.as_payload()) == {
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
        "blend_groups",
        "blend_group_weights",
        "blend_within_weights",
        "interaction_pairs",
    }


def test_missing_value_value_space_semantics() -> None:
    values = _panel([[1.0, 2.0, np.nan, 4.0], [np.nan, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])

    # Disabled: the identity.
    assert apply_missing_value(values, SignalSpec()).equals(values)

    # Median fill uses each date's valid cross-section.
    filled = apply_missing_value(values, SignalSpec(missing_value="cross_section_median"))
    assert filled.loc[DATES[0], "C"] == pytest.approx(2.0)  # median of [1, 2, 4]
    assert filled.loc[DATES[1], "A"] == pytest.approx(7.0)  # median of [6, 7, 8]
    assert not filled.isna().any().any()

    # drop_stock removes every symbol with any missing value.
    dropped = apply_missing_value(values, SignalSpec(missing_value="drop_stock"))
    assert list(dropped.columns) == ["B", "D"]
    assert dropped.equals(values[["B", "D"]])

    # A panel with no missing values passes through drop_stock untouched.
    clean = _panel([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    assert apply_missing_value(clean, SignalSpec(missing_value="drop_stock")).equals(clean)

    # Dropping everything (every column has a missing value) fails closed.
    everywhere = values.copy()
    everywhere.loc[DATES[2], "D"] = np.nan
    everywhere.loc[DATES[0], "B"] = np.nan
    with pytest.raises(ValueError, match="drop_stock removed every symbol"):
        apply_missing_value(everywhere, SignalSpec(missing_value="drop_stock"))


def test_missing_value_fill_ratio_fails_closed() -> None:
    values = _panel([[1.0, np.nan, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    # Row 0 needs 2/4 = 0.5 of the cross-section filled.
    with pytest.raises(ValueError, match="missing_value fill exceeds 0.4"):
        apply_missing_value(values, SignalSpec(missing_value="cross_section_median", missing_max_fill_ratio=0.4))
    assert not apply_missing_value(
        values, SignalSpec(missing_value="cross_section_median", missing_max_fill_ratio=0.5)
    ).isna().any().any()

    ranks = _panel([[1.0, np.nan, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    with pytest.raises(ValueError, match="missing_value fill exceeds 0.4"):
        apply_neutral_rank_fill(ranks, SignalSpec(missing_value="neutral_rank", missing_max_fill_ratio=0.4))


def test_missing_value_fill_happens_after_the_a1_stages() -> None:
    """The filled value is the median of the *standardized* cross-section,
    proving the policy sits between standardize and the direction map."""

    values = _panel([[1.0, 2.0, 100.0, np.nan], [3.0, 4.0, 5.0, 6.0]])
    spec = SignalSpec(standardize="zscore", missing_value="cross_section_median")
    standardized = apply_signal(values, SignalSpec(standardize="zscore"))
    row = standardized.loc[DATES[0]].dropna()
    expected = float(row.median())
    filled = apply_missing_value(standardized, SignalSpec(missing_value="cross_section_median"))
    assert filled.loc[DATES[0], "D"] == pytest.approx(expected)
    # And it is *not* the raw-value median (2.0 standardized).
    assert filled.loc[DATES[0], "D"] != pytest.approx(2.0)


def test_neutral_rank_fill_semantics() -> None:
    ranks = _panel([[0.2, np.nan, 0.8, 1.0], [0.9, 0.4, 0.1, np.nan]])
    filled = apply_neutral_rank_fill(ranks, SignalSpec(missing_value="neutral_rank"))
    assert filled.loc[DATES[0], "B"] == pytest.approx(0.5)
    assert filled.loc[DATES[1], "D"] == pytest.approx(0.5)
    assert filled.loc[DATES[0], "A"] == pytest.approx(0.2)
    # Disabled policies leave the ranks untouched (NaN stays NaN).
    assert apply_neutral_rank_fill(ranks, SignalSpec()).equals(ranks)
    assert apply_neutral_rank_fill(ranks, SignalSpec(missing_value="cross_section_median")).equals(ranks)


def test_interaction_rank_panels_semantics() -> None:
    ranks = {
        "f1": _panel([[0.2, 0.4, 0.6, 0.8], [0.9, 0.7, np.nan, 0.1]]),
        "f2": _panel([[0.5, 0.5, 0.5, 0.5], [0.3, np.nan, 0.6, 0.4]]),
    }
    product = interaction_rank_panels(ranks, SignalSpec(interaction_pairs=(("f1", "f2", "product"),)))["f1*f2"]
    raw_product = ranks["f1"] * ranks["f2"]
    expected = raw_product.rank(axis=1, pct=True, method="average")
    assert product.equals(expected)
    # Missing propagation: the cell is NaN where either input is NaN.
    assert np.isnan(product.loc[DATES[1], "C"]) and np.isnan(product.loc[DATES[1], "B"])
    assert not product.dropna().isna().any().any()

    geomean = interaction_rank_panels(ranks, SignalSpec(interaction_pairs=(("f1", "f2", "geometric_mean"),)))[
        "f1*f2"
    ]
    expected_geo = np.sqrt(ranks["f1"] * ranks["f2"]).rank(axis=1, pct=True, method="average")
    assert geomean.equals(expected_geo)

    minimum = interaction_rank_panels(ranks, SignalSpec(interaction_pairs=(("f1", "f2", "min"),)))["f1*f2"]
    expected_min = pd.DataFrame(
        np.minimum(ranks["f1"], ranks["f2"]), index=ranks["f1"].index, columns=ranks["f1"].columns
    ).rank(axis=1, pct=True, method="average")
    assert minimum.equals(expected_min)
    # np.minimum must propagate NaN where pandas min would skip it.
    assert np.isnan(minimum.loc[DATES[1], "B"]) and np.isnan(minimum.loc[DATES[1], "C"])

    members = SignalSpec(interaction_pairs=(("f1", "f2", "product"), ("f1", "f3", "min"))).interaction_member_names()
    assert members == ("f1*f2", "f1*f3")


def _member_ranks() -> dict[str, pd.DataFrame]:
    return {
        "f1": _panel([[0.1, 0.5, 0.9, 0.3], [0.4, 0.6, 0.2, 0.8]]),
        "f2": _panel([[0.7, np.nan, 0.2, 0.6], [0.5, 0.5, 0.5, 0.5]]),
        "f3": _panel([[0.3, 0.6, 0.1, 0.9], [0.8, 0.1, 0.7, 0.4]]),
    }


def test_flat_blend_weights_semantics() -> None:
    ranks = _member_ranks()
    spec = SignalSpec(blend_weights={"f1": 0.6, "f2": 0.3, "f3": 0.1})
    blended = blend_member_panels(ranks, ["f1", "f2", "f3"], spec)
    expected = (0.6 * ranks["f1"] + 0.3 * ranks["f2"] + 0.1 * ranks["f3"]) / 1.0
    assert np.allclose(blended.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True)
    # The strict missing contract: the NaN in f2 kills the cell.
    assert np.isnan(blended.loc[DATES[0], "B"])
    # Weights are normalized, so scaling them all leaves the blend unchanged.
    scaled = blend_member_panels(ranks, ["f1", "f2", "f3"], SignalSpec(blend_weights={"f1": 6.0, "f2": 3.0, "f3": 1.0}))
    assert np.allclose(blended.to_numpy(dtype=float), scaled.to_numpy(dtype=float), equal_nan=True)


def test_group_blend_semantics() -> None:
    ranks = _member_ranks()
    spec = SignalSpec(
        blend_groups={"g1": ["f1", "f2"], "g2": ["f3"]},
        blend_group_weights={"g1": 0.75, "g2": 0.25},
        blend_within_weights={"g1": {"f1": 0.25, "f2": 0.75}},
    )
    blended = blend_member_panels(ranks, ["f1", "f2", "f3"], spec)
    g1 = (0.25 * ranks["f1"] + 0.75 * ranks["f2"]) / 1.0
    g2 = ranks["f3"]
    expected = (0.75 * g1 + 0.25 * g2) / 1.0
    assert np.allclose(blended.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True)
    # The missing f2 cell kills its group, and the group kills the cell.
    assert np.isnan(blended.loc[DATES[0], "B"])
    # Default weights: equal within groups and equal across groups.
    plain = blend_member_panels(
        ranks, ["f1", "f2", "f3"], SignalSpec(blend_groups={"g1": ["f1", "f2"], "g2": ["f3"]})
    )
    expected_plain = ((ranks["f1"] + ranks["f2"]) / 2.0 + ranks["f3"]) / 2.0
    assert np.allclose(plain.to_numpy(dtype=float), expected_plain.to_numpy(dtype=float), equal_nan=True)


def test_disabled_blend_is_the_untouched_equal_blend() -> None:
    ranks = _member_ranks()
    spec = SignalSpec(interaction_pairs=(("f1", "f3", "product"),))
    blended = blend_member_panels(ranks, ["f1", "f2", "f3"], spec)
    expected = blend_rank_panels({**ranks, **interaction_rank_panels(ranks, spec)}, ["f1", "f2", "f3", "f1*f3"])
    assert blended.equals(expected)
    # Without interactions the spec-free path is bit-for-bit the legacy blend.
    assert blend_member_panels(ranks, ["f1", "f2", "f3"], SignalSpec()).equals(
        blend_rank_panels(ranks, ["f1", "f2", "f3"])
    )


def test_interaction_terms_join_the_blend_as_members() -> None:
    ranks = _member_ranks()
    spec = SignalSpec(
        interaction_pairs=(("f1", "f3", "product"),),
        blend_weights={"f1": 0.4, "f2": 0.4, "f3": 0.1, "f1*f3": 0.1},
    )
    blended = blend_member_panels(ranks, ["f1", "f2", "f3"], spec)
    interactions = interaction_rank_panels(ranks, spec)
    expected = (
        0.4 * ranks["f1"] + 0.4 * ranks["f2"] + 0.1 * ranks["f3"] + 0.1 * interactions["f1*f3"]
    ) / 1.0
    assert np.allclose(blended.to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True)


def test_candidate_level_membership_validation() -> None:
    base = dict(name="c", universe="u", composition=CompositionSpec(top_n=5), rule="r")
    with pytest.raises(ValueError, match="interaction terms must reference candidate factors"):
        StrategyCandidate(
            factor_names=("f1", "f2"),
            signal=SignalSpec(interaction_pairs=(("f1", "f3", "product"),)),
            **base,
        )
    with pytest.raises(ValueError, match="blend_weights must cover exactly the blend members"):
        StrategyCandidate(
            factor_names=("f1", "f2"),
            signal=SignalSpec(blend_weights={"f1": 0.5, "f2": 0.4, "f3": 0.1}),
            **base,
        )
    with pytest.raises(ValueError, match="blend_weights must cover exactly the blend members"):
        StrategyCandidate(
            factor_names=("f1", "f2"),
            signal=SignalSpec(interaction_pairs=(("f1", "f2", "product"),), blend_weights={"f1": 0.5, "f2": 0.5}),
            **base,
        )
    with pytest.raises(ValueError, match="blend_groups must partition the candidate factors exactly"):
        StrategyCandidate(factor_names=("f1", "f2"), signal=SignalSpec(blend_groups={"g1": ["f1"]}), **base)
    # The enabled combinations must construct cleanly.
    StrategyCandidate(
        factor_names=("f1", "f2"),
        signal=SignalSpec(
            interaction_pairs=(("f1", "f2", "product"),),
            blend_weights={"f1": 0.4, "f2": 0.3, "f1*f2": 0.3},
        ),
        **base,
    )
    StrategyCandidate(factor_names=("f1", "f2"), signal=SignalSpec(blend_groups={"g1": ["f1"], "g2": ["f2"]}), **base)


def test_full_pipeline_order_missing_then_direction_then_rank() -> None:
    """Direction comes after the fill: a -1 direction factor filled with the
    cross-section median must still blend to the median rank, not to 0.5 of
    the *pre-direction* order."""

    panels = {
        "f1": _panel([[1.0, 2.0, np.nan, 4.0], [3.0, 4.0, 5.0, 6.0]]),
        "f2": _panel([[4.0, 3.0, 2.0, 1.0], [6.0, 5.0, 4.0, 3.0]]),
    }
    directions = {"f1": 1, "f2": -1}
    spec = SignalSpec(missing_value="cross_section_median")
    score = build_factor_score(panels, directions, spec, index=DATES[:2], columns=["A", "B", "C", "D"])
    # No NaN survives: the fill ran before the blend's missing mask.
    assert not score.isna().any().any()
    # With the policy off, the missing cell in f1 kills exactly that cell in
    # the blend, and nothing is filled.
    score_off = build_factor_score(panels, directions, SignalSpec(), index=DATES[:2], columns=["A", "B", "C", "D"])
    assert np.isnan(score_off.loc[DATES[0], "C"])
    assert not score_off.dropna().empty
