"""Walk-forward evaluation (roadmap 10.1): segment-wise IS/OOS reporting.

Unlike the result-report audits in :mod:`audit` (9.4 / 10.2 / 10.3 / 10.4),
which read an already-executed portfolio and never re-run the engine,
WalkForwardEvaluation genuinely *re-runs the frozen strategy* once per
segment.  The configuration therefore travels at the request level next to
``evaluation`` -- never inside the candidate payload -- so enabling it cannot
change a strategy identity hash, and a request without the object walks the
pre-batch path bit for bit.

Two properties of the roadmap entry drive the whole design:

* the strategy schema is **frozen** (:mod:`split` supplies the windows, the
  engine supplies the numbers, nothing is fitted), so a "train" segment is an
  ex-post description of the same frozen strategy over an earlier window --
  not a fitted model.  The report says so explicitly
  (``fit_on_train`` is false) because the roadmap's stated risk is exactly
  that repeatedly reading the OOS segments and re-tuning against them turns
  OOS back into IS;
* N segments means N engine invocations, so the segment count is explicitly
  bounded (:data:`DEFAULT_MAX_SEGMENTS`) and exceeding the bound fails the
  request closed rather than silently reporting fewer segments.

The window construction is delegated to
:func:`rdagent.scenarios.gf_quant.split.build_rolling_splits` -- the
already-tested splitter -- and is never re-implemented here.

The standalone worker mirrors every function here verbatim
(``scripts/gf_quant_strategy_worker.py``); the parity tests feed identical
inputs to both implementations and compare bit for bit.  Error messages in
this module are part of the two-side contract and must stay character-for-
character identical to the worker's.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Sequence

from rdagent.scenarios.gf_quant.split import build_rolling_splits

#: Keys of the request-level ``walk_forward`` object.
WALK_FORWARD_KEYS: frozenset[str] = frozenset(
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

#: Declared default segment cap.  Each segment is a full engine re-run, so the
#: count has to be bounded by configuration rather than by the caller's
#: patience; exceeding it fails closed instead of silently reporting less.
DEFAULT_MAX_SEGMENTS = 8

#: Declared default number of OOS segments a report must contain.  A
#: walk-forward report built on a single short segment is precisely the number
#: the roadmap warns readers against over-trusting, so it is a floor rather
#: than a hint.
DEFAULT_MIN_OOS_SEGMENTS = 1

#: Declared default validation-window length.  The splitter requires every
#: window to be positive, so a validation window has to be asked for
#: explicitly: a walk-forward report without one is train-then-test, and
#: silently inventing a validation window would put a third window in the
#: report that the caller never configured.
DEFAULT_VALID_SIZE = 1

#: Headline metrics aggregated across segments.  Equal weight per segment --
#: under this splitter every OOS window is exactly ``test_size`` long (the
#: while loop only advances on a full window), so the two conventions differ
#: only in the IS summary, and equal weighting is the conservative pick: a
#: segment that merely started earlier carries no extra authority.
WALK_FORWARD_METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "annualized_return",
    "cagr",
    "annualized_vol",
    "sharpe",
    "max_drawdown",
    "annual_turnover",
)


def _positive_int(value: Any, message: str) -> int:
    """Return ``value`` as an ``int`` strict positive integer, else raise."""

    if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
        raise ValueError(message)
    return int(value)


def validate_walk_forward_config(walk_forward: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the request-level ``walk_forward`` object and resolve defaults.

    Mirrored by the worker's ``_validate_walk_forward``; both sides raise the
    same message for the same malformed input.  Returns the resolved
    configuration so callers do not repeat the default logic.

    Window sizes are counts of *observed trading dates*, not calendar days --
    the same convention the splitter uses.  Whether the evaluation window
    actually holds enough dates is a data question, not a configuration one,
    so it is checked at run time by the splitter itself (same message on both
    ends, since both ends call it).
    """

    if not isinstance(walk_forward, Mapping):
        raise ValueError("request walk_forward must be an object")
    unknown = set(walk_forward) - WALK_FORWARD_KEYS
    if unknown:
        raise ValueError(f"request walk_forward has unknown keys: {sorted(unknown)[:3]}")

    if "train_size" not in walk_forward:
        raise ValueError("walk_forward train_size must be a positive integer")
    train_size = _positive_int(walk_forward["train_size"], "walk_forward train_size must be a positive integer")

    if "test_size" not in walk_forward:
        raise ValueError("walk_forward test_size must be a positive integer")
    test_size = _positive_int(walk_forward["test_size"], "walk_forward test_size must be a positive integer")

    valid_size = _positive_int(
        walk_forward.get("valid_size", DEFAULT_VALID_SIZE),
        "walk_forward valid_size must be a positive integer",
    )

    step_size = walk_forward.get("step_size")
    if step_size is None:
        step_size = test_size
    else:
        step_size = _positive_int(step_size, "walk_forward step_size must be a positive integer")

    min_oos_segments = _positive_int(
        walk_forward.get("min_oos_segments", DEFAULT_MIN_OOS_SEGMENTS),
        "walk_forward min_oos_segments must be a positive integer",
    )
    max_segments = _positive_int(
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


def _build_rolling_splits(
    trading_dates: Sequence[Any],
    *,
    train_size: int,
    valid_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding_train: bool = True,
) -> list[tuple[tuple[date, date], tuple[date, date], tuple[date, date]]]:
    """The tested splitter, adapted to plain ``(start, end)`` date pairs.

    The windows themselves come from
    :func:`rdagent.scenarios.gf_quant.split.build_rolling_splits`; this wrapper
    only flattens its ``DateSplit`` objects so the report assembly and the
    worker's mirrored copy read the same shape.  It is deliberately not a
    re-implementation -- the boundary rules stay in the one tested place.
    """

    splits = build_rolling_splits(
        trading_dates,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        step_size=step_size,
        expanding_train=expanding_train,
    )
    return [
        (
            (split.train.start, split.train.end),
            (split.valid.start, split.valid.end),
            (split.test.start, split.test.end),
        )
        for split in splits
    ]


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    """Equal-weighted mean of one metric across segments (0.0 when empty)."""

    if not rows:
        return 0.0
    return float(sum(float(row["metrics"][key]) for row in rows) / len(rows))


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Equal-weighted cross-segment summary of headline metrics."""

    return {
        "segments": len(rows),
        "mean_metrics": {key: _mean_metric(rows, key) for key in WALK_FORWARD_METRIC_KEYS},
        "positive_return_ratio": (
            float(sum(1 for row in rows if float(row["metrics"]["total_return"]) > 0.0) / len(rows)) if rows else 0.0
        ),
        "observations": int(sum(int(row["metrics"]["observations"]) for row in rows)),
    }


def walk_forward_report(
    config: Mapping[str, Any],
    trading_dates: Sequence[Any],
    evaluate_segment: Any,
) -> dict[str, Any]:
    """Build the walk-forward report over the frozen strategy.

    ``trading_dates`` is the evaluation calendar the caller resolved; the
    windows themselves come from the tested splitter.  ``evaluate_segment`` is
    called once per (segment role, start, end) triple and must return that
    window's metrics -- the caller owns the engine, so this module stays free
    of it and the same assembly runs on both ends.

    Three failure modes fail the request closed rather than degrade:

    * a segment count above ``max_segments`` -- N segments is N engine runs
      and the bound is the cost control;
    * fewer OOS segments than ``min_oos_segments`` -- a report built on too
      few out-of-sample windows is exactly what the roadmap warns against
      being read as evidence.

    ``fit_on_train`` / ``parameter_refit`` are reported as false: the schema
    is frozen, so every number here is an ex-post description of one fixed
    strategy over three windows, never a fitted-then-tested result.
    """

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
            "oos": _aggregate(test_rows),
            "is": _aggregate(train_rows),
            "min_oos_segments_met": len(rows) >= int(config["min_oos_segments"]),
        },
    }


__all__ = [
    "DEFAULT_MAX_SEGMENTS",
    "DEFAULT_MIN_OOS_SEGMENTS",
    "WALK_FORWARD_KEYS",
    "WALK_FORWARD_METRIC_KEYS",
    "validate_walk_forward_config",
    "walk_forward_report",
]
