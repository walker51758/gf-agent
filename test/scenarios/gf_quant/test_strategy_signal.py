"""Unit tests for the local signal front-end and SignalSpec validation.

These cover the local side only (spec validation, per-stage semantics, payload
round-trip); the two-sided agreement is covered by
``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.candidate import SignalSpec
from rdagent.scenarios.gf_quant.strategy.signal import (
    apply_signal,
    apply_standardize,
    apply_transform,
    apply_winsorize,
    blend_rank_panels,
)

DATES = pd.date_range("2024-01-01", periods=2)


def _panel(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=DATES, columns=["A", "B", "C", "D"])


def test_signal_spec_rejects_invalid_parameters() -> None:
    for kwargs, match in [
        ({"transform": "cube"}, "transform must be one of"),
        ({"transform_scale": 0.0}, "transform_scale"),
        ({"transform_scale": float("nan")}, "transform_scale"),
        ({"transform_shift": float("inf")}, "transform_shift"),
        ({"winsorize_lower": 0.1}, "together"),
        ({"winsorize_upper": 0.9}, "together"),
        ({"winsorize_lower": 0.9, "winsorize_upper": 0.1}, "0 <= lower < upper <= 1"),
        ({"winsorize_lower": 0.2, "winsorize_upper": 0.2}, "0 <= lower < upper <= 1"),
        ({"winsorize_min_samples": 0}, "winsorize_min_samples"),
        ({"standardize": "mad"}, "standardize must be one of"),
        ({"standardize_min_std": 0.0}, "standardize_min_std"),
        ({"standardize_min_std": -1.0}, "standardize_min_std"),
    ]:
        with pytest.raises(ValueError, match=match):
            SignalSpec(**kwargs)


def test_signal_spec_defaults_are_disabled_and_round_trip() -> None:
    spec = SignalSpec()
    assert spec.transform is None and spec.winsorize_lower is None and spec.standardize is None
    restored = SignalSpec.from_payload(spec.as_payload())
    assert restored == spec
    # A payload without any signal keys restores the disabled default.
    assert SignalSpec.from_payload({}) == spec


def test_transform_semantics_and_fail_closed_domains() -> None:
    values = _panel([[1.0, 4.0, np.nan, 9.0], [-1.0, 0.0, 1.0, 8.0]])
    linear = apply_transform(values, SignalSpec(transform="linear", transform_scale=2.0, transform_shift=1.0))
    assert linear.loc[DATES[0], "A"] == pytest.approx(3.0)
    assert np.isnan(linear.loc[DATES[0], "C"])

    with pytest.raises(ValueError, match="strictly positive"):
        apply_transform(values, SignalSpec(transform="log"))
    with pytest.raises(ValueError, match="non-negative"):
        apply_transform(values, SignalSpec(transform="sqrt"))
    signed = apply_transform(values, SignalSpec(transform="signed_log"))
    assert signed.loc[DATES[1], "A"] == pytest.approx(-np.log1p(1.0))
    assert signed.loc[DATES[1], "B"] == pytest.approx(0.0)


def test_winsorize_clips_from_the_same_date_cross_section() -> None:
    # Bounds come from the valid values of each row only (linear-interpolated
    # quantiles): row 1 [0, 1, 2, 99] has q75 = 2 + 0.25 * (99 - 2) = 26.25, so
    # 99 clips to 26.25; row 0 [0, 1, 2, NaN] has q25 = 0.5, so 0 clips up to
    # 0.5 and the NaN passes through untouched.
    values = _panel([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 99.0]])
    values.iloc[0, 3] = np.nan
    spec = SignalSpec(winsorize_lower=0.25, winsorize_upper=0.75)
    clipped = apply_winsorize(values, spec)
    assert clipped.loc[DATES[1], "D"] == pytest.approx(26.25)
    assert clipped.loc[DATES[0], "A"] == pytest.approx(0.5)
    assert np.isnan(clipped.loc[DATES[0], "D"])

    thin = SignalSpec(winsorize_lower=0.1, winsorize_upper=0.9, winsorize_min_samples=4)
    with pytest.raises(ValueError, match="valid cross-sectional values"):
        apply_winsorize(values, thin)  # row 0 has only 3 valid values

    disabled = apply_winsorize(values, SignalSpec())
    assert disabled.equals(values)


def test_standardize_methods_and_zero_dispersion_fail_closed() -> None:
    values = _panel([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, np.nan]])
    zscore = apply_standardize(values, SignalSpec(standardize="zscore"))
    row = zscore.loc[DATES[0]]
    assert row.mean() == pytest.approx(0.0, abs=1e-12)
    # Sample std (ddof=1) of [1, 2, 3, 4] is sqrt(5/3) = 1.2909944...
    assert row.iloc[0] == pytest.approx(-1.5 / np.sqrt(5.0 / 3.0), rel=1e-12)

    robust = apply_standardize(values, SignalSpec(standardize="robust_z"))
    # Median of [1, 2, 3, 4] is 2.5, MAD is 1.0, scale is 1.4826 * 1.0.
    assert robust.loc[DATES[0], "B"] == pytest.approx(-0.5 / 1.4826, rel=1e-12)

    rank_z = apply_standardize(values, SignalSpec(standardize="rank_z"))
    assert rank_z.loc[DATES[0]].is_monotonic_increasing

    # A constant cross-section has zero dispersion -> fail closed instead of
    # dividing by zero; a single valid value has non-finite std -> same.
    constant = _panel([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    with pytest.raises(ValueError, match="dispersion"):
        apply_standardize(constant, SignalSpec(standardize="zscore"))
    with pytest.raises(ValueError, match="dispersion"):
        apply_standardize(constant, SignalSpec(standardize="robust_z"))
    # A single valid value has a non-finite sample std -> also fails closed.
    single = _panel([[5.0, np.nan, np.nan, np.nan], [1.0, 2.0, 3.0, 4.0]])
    with pytest.raises(ValueError, match="dispersion"):
        apply_standardize(single, SignalSpec(standardize="zscore"))


def test_disabled_pipeline_is_identity() -> None:
    values = _panel([[1.0, -2.0, 3.0, np.nan], [4.0, 5.0, 6.0, 7.0]])
    assert apply_signal(values, SignalSpec()).equals(values)


def test_blend_semantics_match_the_missing_any_factor_contract() -> None:
    first = _panel([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    second = _panel([[np.nan, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    blended = blend_rank_panels(
        {
            "f1": first.rank(axis=1, pct=True, method="average"),
            "f2": second.rank(axis=1, pct=True, method="average"),
        },
        ["f1", "f2"],
    )
    # A missing value in any factor kills the cell even though the other
    # factor has a rank there.  pct ranks divide by the non-NaN count, so the
    # surviving cell blends (0.5 + 1/3) / 2.
    assert np.isnan(blended.loc[DATES[0], "A"])
    assert blended.loc[DATES[0], "B"] == pytest.approx((0.5 + 1.0 / 3.0) / 2)
