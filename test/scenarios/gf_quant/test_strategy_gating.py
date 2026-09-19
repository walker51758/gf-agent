"""Unit tests for the local gating front-end and GatingSpec validation.

These cover the local side only (spec validation, cascade semantics, tie and
boundary rules, NaN propagation, payload round-trip); the two-sided agreement
is covered by ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.candidate import (
    CompositionSpec,
    GateLayer,
    GatingSpec,
    StrategyCandidate,
)
from rdagent.scenarios.gf_quant.strategy.gating import apply_gating, build_gated_factor_score
from rdagent.scenarios.gf_quant.strategy.signal import build_factor_score, build_factor_rank_panels

DATES = pd.bdate_range("2024-01-02", periods=2)
COLS = ["A", "B", "C", "D", "E"]


def _ranks_f1() -> pd.DataFrame:
    # Row 0 has a tie exactly on the 0.4 threshold; row 1 has a NaN rank that
    # must be excluded in both directions.
    return pd.DataFrame(
        [[0.2, 0.4, 0.4, 0.6, 1.0], [1.0, 0.8, 0.6, np.nan, 0.2]],
        index=DATES,
        columns=COLS,
    )


def _ranks_f2() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.9, 0.7, 0.5, 0.3, 0.1], [0.2, 0.4, 0.5, 0.6, 0.8]],
        index=DATES,
        columns=COLS,
    )


def _score() -> pd.DataFrame:
    return pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        index=DATES,
        columns=COLS,
    )


def test_gate_layer_rejects_invalid_parameters() -> None:
    for kwargs, match in [
        ({"factor": "", "direction": "top", "quantile": 0.5}, "factor"),
        ({"factor": "f1", "direction": "side", "quantile": 0.5}, "direction must be one of"),
        ({"factor": "f1", "direction": "top", "quantile": 0.0}, "quantile"),
        ({"factor": "f1", "direction": "top", "quantile": 1.0}, "quantile"),
        ({"factor": "f1", "direction": "top", "quantile": -0.25}, "quantile"),
        ({"factor": "f1", "direction": "top", "quantile": float("nan")}, "quantile"),
        ({"factor": "f1", "direction": "top", "quantile": "0.5"}, "quantile"),
        ({"factor": "f1", "direction": "top", "quantile": 0.5, "min_pass": 0}, "min_pass"),
    ]:
        with pytest.raises(ValueError, match=match):
            GateLayer(**kwargs)


def test_gating_spec_default_is_disabled_and_round_trips() -> None:
    spec = GatingSpec()
    assert spec.layers == ()
    assert spec.as_payload() is None
    assert GatingSpec.from_payload(None) == spec

    enabled = GatingSpec(
        layers=(
            GateLayer(factor="f1", direction="top", quantile=0.5),
            GateLayer(factor="f2", direction="bottom", quantile=0.25, min_pass=3),
        )
    )
    payload = enabled.as_payload()
    assert payload == {
        "layers": [
            {"factor": "f1", "direction": "top", "quantile": 0.5, "min_pass": 1},
            {"factor": "f2", "direction": "bottom", "quantile": 0.25, "min_pass": 3},
        ]
    }
    assert GatingSpec.from_payload(payload) == enabled


def test_gating_spec_from_payload_fails_closed() -> None:
    layer = {"factor": "f1", "direction": "top", "quantile": 0.5}
    for payload, match in [
        ({}, "layers"),
        ({"layers": []}, "non-empty"),
        ({"layers": "f1"}, "non-empty"),
        ({"unexpected": 1, "layers": [layer]}, "unknown keys"),
    ]:
        with pytest.raises(ValueError, match=match):
            GatingSpec.from_payload(payload)


def test_quantile_gate_top_and_bottom_set_non_survivors_to_nan() -> None:
    ranks = {"f1": _ranks_f1()}
    score = _score()

    top = apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "top", 0.5),)))
    # Row 0 keeps D (0.6) and E (1.0); row 1 keeps A, B, C and excludes the
    # NaN-rank D in both directions.
    assert list(top.loc[DATES[0]].dropna().index) == ["D", "E"]
    assert list(top.loc[DATES[1]].dropna().index) == ["A", "B", "C"]
    # A surviving but already-NaN score cell stays NaN.
    score_with_hole = score.copy()
    score_with_hole.loc[DATES[0], "D"] = np.nan
    assert np.isnan(apply_gating(score_with_hole, ranks, GatingSpec(layers=(GateLayer("f1", "top", 0.5),))).loc[DATES[0], "D"])

    bottom = apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "bottom", 0.5),)))
    assert list(bottom.loc[DATES[0]].dropna().index) == ["A", "B", "C"]
    assert list(bottom.loc[DATES[1]].dropna().index) == ["E"]

    disabled = apply_gating(score, ranks, GatingSpec())
    assert disabled.equals(score)


def test_threshold_ties_are_inclusive_on_both_sides() -> None:
    # The 0.4 threshold falls exactly on a tie (B and C share 0.4): an
    # inclusive >= / <= comparison keeps both tied names in either direction.
    ranks = {"f1": _ranks_f1()}
    score = _score()
    top = apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "top", 0.4),)))
    assert list(top.loc[DATES[0]].dropna().index) == ["B", "C", "D", "E"]
    bottom = apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "bottom", 0.4),)))
    assert list(bottom.loc[DATES[0]].dropna().index) == ["A", "B", "C"]


def test_cascade_intersects_layer_survivors() -> None:
    ranks = {"f1": _ranks_f1(), "f2": _ranks_f2()}
    score = _score()
    spec = GatingSpec(
        layers=(GateLayer("f1", "top", 0.5), GateLayer("f2", "bottom", 0.5))
    )
    gated = apply_gating(score, ranks, spec)
    # Row 0: {D, E} n {C, D, E} = {D, E}; row 1: {A, B, C} n {A, B, C}.
    assert list(gated.loc[DATES[0]].dropna().index) == ["D", "E"]
    assert list(gated.loc[DATES[1]].dropna().index) == ["A", "B", "C"]
    # The intersection is commutative: layer order does not change the result
    # when every layer passes its min_pass.
    reversed_spec = GatingSpec(
        layers=(GateLayer("f2", "bottom", 0.5), GateLayer("f1", "top", 0.5))
    )
    assert apply_gating(score, ranks, reversed_spec).equals(gated)


def test_min_pass_fails_closed_and_names_the_layer() -> None:
    ranks = {"f1": _ranks_f1(), "f2": _ranks_f2()}
    score = _score()
    # Row 0 keeps only 2 names: below min_pass=3 on layer 0.
    with pytest.raises(ValueError, match=r"gating layer 0 \(f1 top 0\.5\)"):
        apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "top", 0.5, min_pass=3),)))
    # Layer 0 passes; the intersection after layer 1 is empty on row 0.
    cascade = GatingSpec(
        layers=(GateLayer("f1", "top", 0.5), GateLayer("f2", "top", 0.9))
    )
    with pytest.raises(ValueError, match=r"gating layer 1 \(f2 top 0\.9\)"):
        apply_gating(score, ranks, cascade)


def test_all_excluded_fails_closed_and_all_included_is_identity() -> None:
    ranks = {"f1": pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5]], index=DATES[:1], columns=COLS)}
    score = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0]], index=DATES[:1], columns=COLS)
    # The default min_pass=1 already fails closed on an empty surviving set.
    with pytest.raises(ValueError, match="fewer than 1"):
        apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "top", 0.9),)))
    kept = apply_gating(score, ranks, GatingSpec(layers=(GateLayer("f1", "bottom", 0.9),)))
    np.testing.assert_array_equal(kept.to_numpy(dtype=float), score.to_numpy(dtype=float))


def test_candidate_gating_validation_and_payload_round_trip() -> None:
    base = dict(
        name="gated",
        universe="csi_all",
        composition=CompositionSpec(top_n=10),
        factor_names=("value", "quality"),
        rule="rank blend",
    )
    with pytest.raises(ValueError, match="gating layers must reference candidate factors"):
        StrategyCandidate(**base, gating=GatingSpec(layers=(GateLayer("momentum", "top", 0.5),)))
    with pytest.raises(ValueError, match="gating layers must reference candidate factors"):
        StrategyCandidate(
            name="rule_only",
            universe="csi_all",
            composition=CompositionSpec(top_n=10),
            rule="rank blend",
            gating=GatingSpec(layers=(GateLayer("value", "top", 0.5),)),
        )

    gated = StrategyCandidate(**base, gating=GatingSpec(layers=(GateLayer("value", "top", 0.5),)))
    assert "gating" in gated.as_payload()
    assert "gating" not in StrategyCandidate(**base).as_payload()
    # Enabling the gate changes the identity hash; the disabled default does not.
    plain = StrategyCandidate(**base)
    assert gated.canonical_payload() != plain.canonical_payload()
    assert StrategyCandidate.from_payload({**gated.as_payload(), "hypothesis": "", "metadata": {}}) == gated
    assert StrategyCandidate.from_payload(plain.as_payload()) == plain


def test_build_gated_factor_score_matches_the_manual_pipeline() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = ["A", "B", "C", "D"]
    panels = {
        "value": pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0], [np.nan, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]],
            index=dates,
            columns=columns,
        ),
        "quality": pd.DataFrame(
            [[3.0, 3.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], [4.0, 1.0, 3.0, 2.0]],
            index=dates,
            columns=columns,
        ),
    }
    directions = {"value": 1, "quality": 1}
    from rdagent.scenarios.gf_quant.strategy.candidate import SignalSpec

    signal = SignalSpec()
    gating = GatingSpec(layers=(GateLayer("value", "top", 0.5),))

    ranks = build_factor_rank_panels(panels, directions, signal, index=dates, columns=columns)
    manual = apply_gating(build_factor_score(panels, directions, signal, index=dates, columns=columns), ranks, gating)
    one_call = build_gated_factor_score(panels, directions, signal, gating, index=dates, columns=columns)
    np.testing.assert_array_equal(one_call.to_numpy(dtype=float), manual.to_numpy(dtype=float))

    disabled = build_gated_factor_score(
        panels, directions, signal, GatingSpec(), index=dates, columns=columns
    )
    np.testing.assert_array_equal(
        disabled.to_numpy(dtype=float),
        build_factor_score(panels, directions, signal, index=dates, columns=columns).to_numpy(dtype=float),
    )
