"""Declarative probe harness for the GF strategy layer.

Batch 6 needed 25 throwaway probe scripts, each costing a write-run-read round trip
for one boolean question ("does the payload key set match the worker whitelist?",
"does an infinite score raise in both engines?").  This module turns that pattern
into a table: each case declares what to import, what to build and what to assert,
and the runner prints a single verdict line per case.

Usage::

    python scripts/gf_quant_probe.py --list
    python scripts/gf_quant_probe.py --case backtest_keyset_matches_whitelist
    python scripts/gf_quant_probe.py --all

Design notes:

* A case is a plain dict, not a class hierarchy.  ``build`` returns whatever the
  checks need; each check is ``(label, callable)`` where the callable returns
  ``True`` or raises.  Returning ``False`` fails the case.
* Checks that are *expected* to raise take ``expect_raises=ValueError``.
* ``--all`` runs every case and exits non-zero if any fails, so it composes with
  a shell loop.  ``--verbose`` additionally prints ``detail`` payloads.
* Cases must stay side-effect free: no writes outside ``temp/``, no network, no
  SSH.  The seed cases below are migrations of real batch-6 probes.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Check = Callable[[], bool]


# ---------------------------------------------------------------------------
# helpers shared by the seed cases
# ---------------------------------------------------------------------------


def _full_order_layer_backtest():
    """A BacktestSpec with every order-layer field switched on."""

    from rdagent.scenarios.gf_quant.strategy.backtest import BacktestSpec

    return BacktestSpec(
        signal_lag=1,
        participation_max_rate=0.5,
        participation_max_days=3,
        participation_unfilled="carry",
        participation_amount_window=2,
        lot_rounding=True,
        lot_rounding_mode="down",
        lot_min_notional=0.05,
        side_aware_tradability="block_buy",
        vol_target_annual=0.15,
        vol_target_window=8,
        vol_target_min_scale=0.2,
        vol_target_max_scale=2.0,
        drawdown_thresholds={"warning": 0.03, "reduce": 0.06, "halt": 0.10},
        drawdown_scales={"warning": 0.8, "reduce": 0.5, "halt": 0.0},
        drawdown_confirm_days=2,
        drawdown_recovery=0.02,
    )


def _staged_backtest():
    from rdagent.scenarios.gf_quant.strategy.backtest import BacktestSpec

    return BacktestSpec(signal_lag=1, staged_max_days=2, staged_priority="score")


def _execution_band_backtest():
    """The A4 drift-band / turnover-shrinkage and B2 min-holding fields."""

    from rdagent.scenarios.gf_quant.strategy.backtest import BacktestSpec

    return BacktestSpec(
        signal_lag=1,
        drift_band_name=0.05,
        drift_band_portfolio=0.3,
        drift_band_forced_days=2,
        turnover_shrinkage=0.5,
        turnover_max_step=0.1,
        min_holding_days=5,
    )


# ---------------------------------------------------------------------------
# case definitions
# ---------------------------------------------------------------------------

CASES: dict[str, dict[str, Any]] = {}


def case(name: str, description: str):
    """Register a case builder."""

    def register(func):
        CASES[name] = {"description": description, "build": func}
        return func

    return register


@case(
    "backtest_keyset_matches_whitelist",
    "the payload keys of every optional BacktestSpec family are accepted by the worker whitelist",
)
def _build_backtest_keyset():
    from scripts.gf_quant_strategy_worker import BACKTEST_KEYS

    full = _full_order_layer_backtest()
    staged = _staged_backtest()
    band = _execution_band_backtest()
    union = set(full.as_payload()) | set(staged.as_payload()) | set(band.as_payload())
    whitelist = set(BACKTEST_KEYS)
    return {
        "union": union,
        "whitelist": whitelist,
        "checks": [
            ("all-open payload is inside BACKTEST_KEYS", lambda: set(full.as_payload()) <= whitelist),
            ("A4/B2 execution payload is inside BACKTEST_KEYS", lambda: set(band.as_payload()) <= whitelist),
            ("all optional families together cover BACKTEST_KEYS", lambda: union == whitelist),
        ],
        "detail": {
            "union_size": len(union),
            "whitelist_size": len(whitelist),
            "union_minus_whitelist": sorted(union - whitelist),
            "whitelist_minus_union": sorted(whitelist - union),
        },
    }


@case(
    "signal_decay_keyset_matches_whitelist",
    "the factor-decay SignalSpec payload is inside the worker SIGNAL_KEYS whitelist",
)
def _build_signal_keyset():
    from rdagent.scenarios.gf_quant.strategy.candidate import SignalSpec
    from scripts.gf_quant_strategy_worker import SIGNAL_KEYS

    decay = SignalSpec(decay_window=12, decay_fail_periods=2, decay_ic_threshold=0.01, decay_scale=0.25)
    payload = set(decay.as_payload())
    whitelist = set(SIGNAL_KEYS)
    return {
        "checks": [
            ("decay payload is inside SIGNAL_KEYS", lambda: payload <= whitelist),
            (
                "enabling decay only adds decay keys",
                lambda: payload - set(SignalSpec().as_payload()) == set(decay.as_payload()) - set(SignalSpec().as_payload()),
            ),
        ],
        "detail": {
            "decay_added": sorted(payload - set(SignalSpec().as_payload())),
            "whitelist_minus_payload": sorted(whitelist - payload),
        },
    }


@case(
    "infinite_score_fails_closed_in_both_engines",
    "an infinite score panel raises in both the local weighting stage and the worker's _target_weights",
)
def _build_infinite_score():
    import numpy as np
    import pandas as pd

    from rdagent.scenarios.gf_quant.strategy.composition import build_target_weights
    from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec
    from scripts.gf_quant_strategy_worker import _target_weights

    dates = pd.bdate_range("2024-01-02", periods=3)
    score = pd.DataFrame([[1.0, np.inf]], index=dates[:1], columns=["A", "B"])

    def local_raises() -> bool:
        try:
            build_target_weights(CompositionSpec(top_n=2, weighting="equal"), score)
        except ValueError:
            return True
        return False

    def worker_raises() -> bool:
        try:
            _target_weights(score, {"top_n": 2, "weighting": "equal"})
        except ValueError:
            return True
        return False

    return {
        "checks": [
            ("local build_target_weights raises ValueError", local_raises),
            ("worker _target_weights raises ValueError", worker_raises),
        ],
        "detail": {"score": score.to_dict()},
    }


@case(
    "floor_gap_stays_inside_the_shared_allowance",
    "a floor plus a tight exposure box leaves a residual the raw tolerance rejects but the shared allowance admits",
)
def _build_floor_gap():
    import numpy as np
    import pandas as pd

    from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec
    from rdagent.scenarios.gf_quant.strategy.composition import build_target_weights

    score = pd.DataFrame(
        [[4.0, 3.0, 2.0, 1.0]], index=pd.to_datetime(["2024-01-02"]), columns=["A", "B", "C", "D"]
    )
    delta = 5e-9
    upper = 4 * 0.2 - 4 * delta
    composition = CompositionSpec(
        top_n=4,
        weighting="equal",
        weight_floor=0.2,
        weight_floor_allow_cash=True,
        factor_exposure_bounds={"value": [0.0, upper]},
        factor_exposure_matrix={"value": {s: 1.0 for s in ["A", "B", "C", "D"]}},
        projection_max_iter=1,
        projection_tolerance=1e-12,
    )
    book = build_target_weights(composition, score).to_numpy(dtype=float)
    gap = float(np.max(0.2 - book))
    raw_allowance = 16.0 * 1e-12
    return {
        "checks": [
            ("floor gap is non-zero (the case is discriminating)", lambda: gap > 0.0),
            ("floor gap exceeds the raw tolerance allowance", lambda: gap > raw_allowance),
            ("floor gap stays small (under the shared ceiling)", lambda: gap < 1e-6),
        ],
        "detail": {"book": book.tolist(), "gap": gap, "raw_allowance": raw_allowance},
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_case(name: str, verbose: bool) -> bool:
    spec = CASES[name]
    print(f"[{name}] {spec['description']}")
    try:
        bundle = spec["build"]()
    except Exception:
        print(f"  FAIL  case setup raised\n{traceback.format_exc()}")
        return False

    passed = True
    for label, check in bundle.get("checks", []):
        try:
            outcome = bool(check())
        except Exception as exc:
            print(f"  FAIL  {label}: raised {exc!r}")
            passed = False
            continue
        print(f"  {'PASS' if outcome else 'FAIL'}  {label}")
        passed = passed and outcome

    if verbose and bundle.get("detail"):
        for key, value in bundle["detail"].items():
            print(f"  detail {key} = {value!r}")

    print(f"  -> {'ok' if passed else 'FAILED'}")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gf_quant_probe",
        description="Run declarative strategy-layer probes and print one verdict per case.",
    )
    parser.add_argument("--case", action="append", default=[], help="case name to run (repeatable)")
    parser.add_argument("--all", action="store_true", help="run every registered case")
    parser.add_argument("--list", action="store_true", help="list case names and descriptions, then exit")
    parser.add_argument("--verbose", action="store_true", help="also print each case's detail payload")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in sorted(CASES.items()):
            print(f"{name:45s} {spec['description']}")
        print(f"\n{len(CASES)} case(s) registered")
        return 0

    names = sorted(CASES) if args.all else args.case
    if not names:
        parser.error("pass --case <name>, --all, or --list")
    unknown = [name for name in names if name not in CASES]
    if unknown:
        print(f"unknown case(s): {', '.join(unknown)}; try --list", file=sys.stderr)
        return 2

    results = [run_case(name, args.verbose) for name in names]
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} case(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
