"""WalkForwardEvaluation (roadmap 10.1): config, assembly and two-side parity.

The operator is the one evaluation that re-runs the engine -- once per
segment -- so the tests here split into three groups: pure configuration
validation (identical messages on both sides), report assembly over a stub
evaluator (so the segmentation arithmetic is checked without a real engine),
and the rolling-window parity against the shared splitter the worker mirrors.
"""

from __future__ import annotations

import math
import re

import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.remote import RemoteStrategyEvaluator  # noqa: F401
from rdagent.scenarios.gf_quant.strategy.walk_forward import (
    DEFAULT_MAX_SEGMENTS,
    DEFAULT_MIN_OOS_SEGMENTS,
    DEFAULT_VALID_SIZE,
    WALK_FORWARD_KEYS,
    WALK_FORWARD_METRIC_KEYS,
    _aggregate,
    _build_rolling_splits,
    _mean_metric,
    validate_walk_forward_config,
    walk_forward_report,
)
from scripts.gf_quant_strategy_worker import (
    DEFAULT_MAX_SEGMENTS as _WORKER_DEFAULT_MAX_SEGMENTS,
    DEFAULT_MIN_OOS_SEGMENTS as _WORKER_DEFAULT_MIN_OOS_SEGMENTS,
    DEFAULT_VALID_SIZE as _WORKER_DEFAULT_VALID_SIZE,
    WALK_FORWARD_KEYS as _WORKER_WALK_FORWARD_KEYS,
    WALK_FORWARD_METRIC_KEYS as _WORKER_WALK_FORWARD_METRIC_KEYS,
    _build_rolling_splits as _worker_build_rolling_splits,
    _validate_walk_forward,
    _walk_forward_aggregate as _worker_aggregate,
    _walk_forward_report,
)


def _dates(count: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=count)


def _stub_metrics(role: str, start, end, *, scale: float = 1.0) -> dict[str, float]:
    """Deterministic metrics whose value depends on the window it was asked for."""

    span = float((end - start).days)
    return {
        "observations": float(span),
        "total_return": 0.001 * span * scale,
        "annualized_return": 0.10 * scale,
        "cagr": 0.11 * scale,
        "annualized_vol": 0.15 * scale,
        "sharpe": 0.70 * scale,
        "max_drawdown": -0.20 * scale,
        "annual_turnover": 6.0 * scale,
    }


def _stub_evaluator(calls: list | None = None, *, scale: float = 1.0):
    def evaluate_segment(role: str, start, end) -> dict[str, float]:
        if calls is not None:
            calls.append((role, start, end))
        return _stub_metrics(role, start, end, scale=scale)

    return evaluate_segment


def test_batch8_constants_are_shadowed_identically_on_both_sides() -> None:
    assert WALK_FORWARD_KEYS == _WORKER_WALK_FORWARD_KEYS
    assert WALK_FORWARD_METRIC_KEYS == _WORKER_WALK_FORWARD_METRIC_KEYS
    assert DEFAULT_MAX_SEGMENTS == _WORKER_DEFAULT_MAX_SEGMENTS
    assert DEFAULT_MIN_OOS_SEGMENTS == _WORKER_DEFAULT_MIN_OOS_SEGMENTS
    assert DEFAULT_VALID_SIZE == _WORKER_DEFAULT_VALID_SIZE


@pytest.mark.parametrize(
    ("walk_forward", "message"),
    [
        ([], "request walk_forward must be an object"),
        ({"mystery": 1, "train_size": 5, "test_size": 5}, "request walk_forward has unknown keys: ['mystery']"),
        ({"test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": 5}, "walk_forward test_size must be a positive integer"),
        ({"train_size": 0, "test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": -3, "test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": 5.0, "test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5.0}, "walk_forward test_size must be a positive integer"),
        # bool is an int in Python; it must not sneak through as 1/0.
        ({"train_size": True, "test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "valid_size": 0}, "walk_forward valid_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "valid_size": -1}, "walk_forward valid_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "valid_size": 2.5}, "walk_forward valid_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "step_size": 0}, "walk_forward step_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "step_size": -2}, "walk_forward step_size must be a positive integer"),
        (
            {"train_size": 5, "test_size": 5, "min_oos_segments": 0},
            "walk_forward min_oos_segments must be a positive integer",
        ),
        (
            {"train_size": 5, "test_size": 5, "max_segments": 0},
            "walk_forward max_segments must be a positive integer",
        ),
        (
            {"train_size": 5, "test_size": 5, "max_segments": 1.5},
            "walk_forward max_segments must be a positive integer",
        ),
        # The roadmap's 10.1 parameter list has no warm-up term, and an
        # unread knob is worse than no knob: ``warmup_days`` used to be
        # accepted, echoed and consumed nowhere.  It is now an unknown key on
        # both ends, so a caller that believes it is warming a segment up is
        # refused instead of silently getting nothing.
        (
            {"train_size": 20, "test_size": 5, "warmup_days": 3},
            "request walk_forward has unknown keys: ['warmup_days']",
        ),
        (
            {"train_size": 20, "test_size": 5, "warmup_days": 0},
            "request walk_forward has unknown keys: ['warmup_days']",
        ),
        (
            {"train_size": 5, "test_size": 5, "expanding_train": "yes"},
            "walk_forward expanding_train must be a boolean",
        ),
        (
            {"train_size": 5, "test_size": 5, "expanding_train": 1},
            "walk_forward expanding_train must be a boolean",
        ),
    ],
)
def test_walk_forward_config_failures_fail_closed_on_both_sides(walk_forward, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_walk_forward_config(walk_forward)
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate_walk_forward(walk_forward)


@pytest.mark.parametrize(
    "walk_forward",
    [
        {"train_size": 20, "test_size": 5},
        {"train_size": 20, "valid_size": 5, "test_size": 5},
        {"train_size": 20, "valid_size": 1, "test_size": 5, "step_size": 1},
        {"train_size": 20, "valid_size": 5, "test_size": 5, "expanding_train": False},
        {"train_size": 20, "valid_size": 5, "test_size": 5, "min_oos_segments": 2, "max_segments": 4},
        {"train_size": 20, "valid_size": 5, "test_size": 5, "step_size": 2, "expanding_train": False},
    ],
)
def test_valid_walk_forward_configs_resolve_identically_on_both_sides(walk_forward) -> None:
    local = validate_walk_forward_config(walk_forward)
    assert local == _validate_walk_forward(walk_forward)


def test_walk_forward_defaults_are_declared_not_improvised() -> None:
    resolved = validate_walk_forward_config({"train_size": 10, "test_size": 3})
    assert resolved == {
        "train_size": 10,
        "valid_size": DEFAULT_VALID_SIZE,
        "test_size": 3,
        # step defaults to the test size, so consecutive OOS windows tile.
        "step_size": 3,
        "expanding_train": True,
        "min_oos_segments": DEFAULT_MIN_OOS_SEGMENTS,
        "max_segments": DEFAULT_MAX_SEGMENTS,
    }
    assert _validate_walk_forward({"train_size": 10, "test_size": 3}) == resolved


def test_walk_forward_windows_match_the_shared_splitter_everywhere() -> None:
    """The two splitters must agree window for window, expanding and rolling."""

    dates = _dates(30)
    for case in (
        {"train_size": 12, "valid_size": 4, "test_size": 3, "step_size": 3, "expanding_train": True},
        {"train_size": 12, "valid_size": 4, "test_size": 3, "step_size": 3, "expanding_train": False},
        {"train_size": 10, "valid_size": 1, "test_size": 5, "step_size": 5, "expanding_train": True},
        {"train_size": 10, "valid_size": 2, "test_size": 4, "step_size": 1, "expanding_train": True},
    ):
        assert _build_rolling_splits(list(dates), **case) == _worker_build_rolling_splits(list(dates), **case), case


def test_walk_forward_honours_a_step_size_that_differs_from_the_test_size() -> None:
    """A step shorter than the test window makes the OOS windows overlap.

    Without this the ``step_size`` field would be indistinguishable from
    ``test_size`` -- a mutation collapsing one onto the other would pass every
    other test, so the overlap is asserted directly.
    """

    dates = _dates(30)
    overlapping = _build_rolling_splits(
        list(dates), train_size=10, valid_size=3, test_size=4, step_size=2, expanding_train=True
    )
    tiling = _build_rolling_splits(
        list(dates), train_size=10, valid_size=3, test_size=4, step_size=4, expanding_train=True
    )
    # A shorter step builds strictly more segments than the test-size step.
    assert len(overlapping) > len(tiling)
    # And the windows genuinely overlap: the second OOS window starts inside
    # the first one's span.
    assert overlapping[1][2][0] < overlapping[0][2][1]
    # The tiling case must not overlap.
    assert tiling[1][2][0] > tiling[0][2][1]
    # Both agree with the worker on the same inputs.
    assert overlapping == _worker_build_rolling_splits(
        list(dates), train_size=10, valid_size=3, test_size=4, step_size=2, expanding_train=True
    )


def test_walk_forward_windows_are_chronological_and_step_is_respected() -> None:
    dates = _dates(25)
    splits = _build_rolling_splits(
        list(dates), train_size=8, valid_size=4, test_size=3, step_size=3, expanding_train=True
    )
    assert len(splits) == 4
    for (train, valid, test) in splits:
        # Strictly chronological and non-overlapping: the OOS window never
        # shares a day with the windows before it.
        assert train[1] < valid[0] < valid[1] < test[0]
        assert dates[0].date() == train[0]
    # Expanding: every segment re-reads the same history from the first date.
    assert all(split[0][0] == dates[0].date() for split in splits)
    # Step 3 -> the test windows advance by exactly 3 observed dates.
    tests = [split[2][0] for split in splits]
    positions = [list(dates.date).index(day) for day in tests]
    assert [b - a for a, b in zip(positions, positions[1:])] == [3, 3, 3]


def test_walk_forward_boundary_lands_on_observed_dates_not_calendar_days() -> None:
    """Gaps in the calendar must not shift a window edge onto a missing day."""

    # Deliberately sparse: weekdays only every other week.
    dates = list(pd.bdate_range("2024-01-02", periods=40)[::2])
    splits = _build_rolling_splits(
        list(dates), train_size=5, valid_size=3, test_size=2, step_size=2, expanding_train=True
    )
    observed = {date.date() for date in dates}
    for (train, valid, test) in splits:
        for start, end in (train, valid, test):
            assert start in observed and end in observed


def test_walk_forward_assembly_calls_the_evaluator_once_per_window() -> None:
    calls: list = []
    config = validate_walk_forward_config({"train_size": 8, "valid_size": 4, "test_size": 3, "step_size": 3})
    report = walk_forward_report(config, list(_dates(25)), _stub_evaluator(calls))

    assert len(report["segments"]) == 4
    # train + valid + test per segment, and nothing else.
    assert len(calls) == 12
    assert {role for role, _, _ in calls} == {"train", "valid", "test"}
    for segment in report["segments"]:
        assert segment["train"]["in_sample"] is True
        assert segment["valid"]["in_sample"] is True
        assert segment["test"]["in_sample"] is False
        # Each role's metrics belong to that role's window.
        assert segment["test"]["metrics"]["observations"] == float(
            (pd.Timestamp(segment["test"]["window"][1]) - pd.Timestamp(segment["test"]["window"][0])).days
        )


def test_walk_forward_report_is_declared_ex_post_not_fitted() -> None:
    """The roadmap's stated risk is exactly what these two flags answer."""

    config = validate_walk_forward_config({"train_size": 8, "test_size": 3})
    report = walk_forward_report(config, list(_dates(20)), _stub_evaluator())
    assert report["fit_on_train"] is False
    assert report["parameter_refit"] is False
    assert report["mode"] == "rolling"


def test_walk_forward_summary_is_equal_weighted_across_segments() -> None:
    """Segments of unequal length must not be weighted by length."""

    config = validate_walk_forward_config({"train_size": 8, "test_size": 3, "step_size": 3})
    report = walk_forward_report(config, list(_dates(20)), _stub_evaluator())
    segments = report["segments"]
    assert len(segments) == 3
    # Equal weight: the plain mean of the per-segment metric, even though the
    # segment spans differ.
    for key in WALK_FORWARD_METRIC_KEYS:
        values = [segment["test"]["metrics"][key] for segment in segments]
        assert report["summary"]["oos"]["mean_metrics"][key] == pytest.approx(sum(values) / len(values)), key
    assert report["summary"]["oos"]["segments"] == 3
    assert report["summary"]["is"]["segments"] == 3
    assert report["summary"]["oos"]["observations"] == sum(
        int(segment["test"]["metrics"]["observations"]) for segment in segments
    )
    assert report["summary"]["min_oos_segments_met"] is True


def test_walk_forward_summary_marks_negative_segments() -> None:
    """A discriminating check: a losing strategy must not report 1.0."""

    def evaluator(role, start, end):
        metrics = _stub_metrics(role, start, end)
        metrics["total_return"] = -0.01 if role == "test" else 0.02
        return metrics

    config = validate_walk_forward_config({"train_size": 8, "test_size": 3, "step_size": 3})
    report = walk_forward_report(config, list(_dates(20)), evaluator)
    assert report["summary"]["oos"]["positive_return_ratio"] == pytest.approx(0.0)
    assert report["summary"]["is"]["positive_return_ratio"] == pytest.approx(1.0)


def test_walk_forward_mean_metric_helpers_are_mirrored() -> None:
    rows = [
        {"metrics": {key: 0.5 for key in WALK_FORWARD_METRIC_KEYS} | {"observations": 3}},
        {"metrics": {key: 1.5 for key in WALK_FORWARD_METRIC_KEYS} | {"observations": 4}},
    ]
    assert _mean_metric(rows, "sharpe") == pytest.approx(1.0)
    assert _mean_metric([], "sharpe") == 0.0
    local = _aggregate(rows)
    worker = _worker_aggregate(rows)
    assert local == worker
    assert local["observations"] == 7
    assert local["mean_metrics"]["sharpe"] == pytest.approx(1.0)


def test_walk_forward_fails_closed_when_the_window_is_too_short() -> None:
    config = validate_walk_forward_config({"train_size": 50, "valid_size": 10, "test_size": 10})
    with pytest.raises(ValueError, match=re.escape("Need at least 70 dates, got 25")):
        walk_forward_report(config, list(_dates(25)), _stub_evaluator())
    with pytest.raises(ValueError, match=re.escape("Need at least 70 dates, got 25")):
        _walk_forward_report(config, list(_dates(25)), _stub_evaluator())


def test_walk_forward_fails_closed_above_max_segments() -> None:
    """Cost control is a hard bound, not a silent truncation."""

    config = validate_walk_forward_config({"train_size": 5, "test_size": 2, "step_size": 1, "max_segments": 4})
    message = "walk_forward would build 18 segments, more than max_segments 4"
    with pytest.raises(ValueError, match=re.escape(message)):
        walk_forward_report(config, list(_dates(25)), _stub_evaluator())
    with pytest.raises(ValueError, match=re.escape(message)):
        _walk_forward_report(config, list(_dates(25)), _stub_evaluator())


def test_walk_forward_fails_closed_below_min_oos_segments() -> None:
    """Too few OOS windows is a refusal, not a footnote."""

    config = validate_walk_forward_config({"train_size": 20, "test_size": 2, "min_oos_segments": 5})
    # 25 dates, train 20 + valid 1 + test 2 = 23, so exactly two windows fit.
    message = "walk_forward produced 2 OOS segments, fewer than min_oos_segments 5"
    with pytest.raises(ValueError, match=re.escape(message)):
        walk_forward_report(config, list(_dates(25)), _stub_evaluator())
    with pytest.raises(ValueError, match=re.escape(message)):
        _walk_forward_report(config, list(_dates(25)), _stub_evaluator())


def test_walk_forward_report_matches_the_worker_on_the_same_inputs() -> None:
    """Full report parity, boundaries included, over one non-trivial case."""

    dates = list(_dates(30))
    for payload in (
        {"train_size": 12, "valid_size": 4, "test_size": 3, "step_size": 3},
        {"train_size": 12, "valid_size": 1, "test_size": 3, "step_size": 3, "expanding_train": False},
        {"train_size": 10, "valid_size": 5, "test_size": 5, "min_oos_segments": 2},
    ):
        config = validate_walk_forward_config(payload)
        local = walk_forward_report(config, dates, _stub_evaluator())
        worker = _walk_forward_report(config, dates, _stub_evaluator())
        assert local == worker, payload
        assert local["segments"] and all(segment["test"]["in_sample"] is False for segment in local["segments"])


def test_walk_forward_never_reports_fractional_counts() -> None:
    """Every count in the report is an int, so a float cannot slip into an id."""

    config = validate_walk_forward_config({"train_size": 8, "test_size": 3})
    report = walk_forward_report(config, list(_dates(20)), _stub_evaluator())
    for key in ("train_size", "valid_size", "test_size", "step_size", "min_oos_segments", "max_segments"):
        assert isinstance(report[key], int) and not isinstance(report[key], bool), key
    for segment in report["segments"]:
        assert isinstance(segment["segment"], int)
        assert isinstance(segment["step_size"], int)
    assert math.isfinite(float(report["summary"]["oos"]["mean_metrics"]["sharpe"]))
