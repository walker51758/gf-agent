"""One-command demo of the MCTS joint search.

Runs the whole first-stage loop end to end: pull (or load) the cached market
panels, assemble the tree and the action space, spend the credit budget through
selection / expansion / simulation / backpropagation, write the sidecar and
print the winning configuration ordered by Q.

    python -m rdagent.scenarios.gf_quant.strategy.search.demo --refresh-data
    python -m rdagent.scenarios.gf_quant.strategy.search.demo --offline

``--refresh-data`` performs the only rqdatac calls in the whole flow (two per
pull, see ``data_cache.fetch_panels``); without it the run reads whatever cache
is already on disk.  ``--offline`` skips the network entirely and writes a
deterministic synthetic cache, which is how the demo is reproducible without a
licence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rdagent.scenarios.gf_quant.strategy.search.budget import DEFAULT_TOTAL_CREDITS
from rdagent.scenarios.gf_quant.strategy.search.data_cache import (
    DEFAULT_SYMBOL_COUNT,
    DEFAULT_YEARS,
    SearchDataCache,
    default_cache_root,
    synthetic_cache,
    update_cache,
)
from rdagent.scenarios.gf_quant.strategy.search.runner import best_configuration, run_search
from rdagent.scenarios.gf_quant.strategy.search.sidecar import SIDECAR_FILENAME


def _unavailable_factor_evaluator(candidate: object) -> object:
    """Refuse to evaluate a freshly proposed factor, with the reason stated.

    The RankIC acceptance chain needs remote factor evaluation, which the data
    machine provides and which is unreachable here.  Returning a fabricated
    EvaluationResult would let an ungated factor into the tree, so this refuses
    instead: with it wired, a proposal is recorded as rejected and contributes
    no action, which is the truthful outcome while the data machine is down.
    """

    raise RuntimeError(
        "new-factor acceptance needs the remote factor evaluator, which is unreachable "
        "while the data machine is down; proposed factors cannot be gated in this run"
    )


def _print(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=None, help="cache directory (default: the repo ignore root)")
    parser.add_argument("--sidecar-dir", type=Path, default=None, help="where the tree sidecar is written")
    parser.add_argument("--credits", type=int, default=DEFAULT_TOTAL_CREDITS, help="total search budget in credits")
    parser.add_argument("--symbols", type=int, default=DEFAULT_SYMBOL_COUNT, help="how many index members to cache")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="years of daily history to cache")
    parser.add_argument("--evaluation-start", default=None)
    parser.add_argument("--evaluation-end", default=None)
    parser.add_argument("--rebalance-frequency", default="monthly", choices=("daily", "weekly", "monthly", "quarterly"))
    parser.add_argument("--refresh-data", action="store_true", help="pull fresh panels from rqdatac (2 API calls)")
    parser.add_argument("--offline", action="store_true", help="use a deterministic synthetic cache instead")
    parser.add_argument(
        "--with-proposers",
        action="store_true",
        help="accepted for compatibility; proposals are wired by default",
    )
    parser.add_argument(
        "--no-proposals",
        action="store_true",
        help="registry-only action space, no LLM proposal rounds",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="where proposal artifacts and the factor registry are written (default: beside the sidecar)",
    )
    parser.add_argument("--fresh-tree", action="store_true", help="ignore an existing sidecar and start a new tree")
    parser.add_argument("--no-continue", action="store_true", help="alias of --fresh-tree")
    args = parser.parse_args(argv)

    cache_root = args.cache_root or default_cache_root()
    sidecar_dir = args.sidecar_dir or (cache_root.parent / "sidecar")

    if args.offline:
        dates = pd.bdate_range("2023-01-02", "2024-12-31")
        symbols = [f"S{index:03d}.XSHE" for index in range(max(args.symbols, 12))]
        cache = synthetic_cache(cache_root, symbols=symbols, dates=dates, seed=11)
        _print("cache (synthetic)", cache.summary())
    else:
        manifest = update_cache(
            cache_root, symbol_count=args.symbols, years=args.years, force=args.refresh_data
        )
        _print("cache", manifest.as_payload())
        cache = SearchDataCache(cache_root)

    # The proposal chain is wired by ``run_search`` itself when no proposer is
    # passed, so the demo has one code path rather than a second assembly of the
    # same collaborators.  ``--no-proposals`` is the registry-only action space.
    run = run_search(
        cache=cache,
        sidecar_dir=sidecar_dir,
        total_credits=args.credits,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        rebalance_frequency=args.rebalance_frequency,
        continue_existing=not (args.fresh_tree or args.no_continue),
        auto_wire_proposals=not args.no_proposals,
        work_dir=args.work_dir or sidecar_dir.parent,
    )

    _print("budget", run.result.ledger.as_payload())
    _print(
        "proposal activity",
        {
            "llm_proposal_rounds": run.llm_proposal_rounds,
            "factors_accepted": list(run.factors_accepted),
            "proposal_notes": list(run.proposal_notes),
        },
    )
    _print("node trace", list(run.result.trace))
    _print("best configuration", best_configuration(run))
    _print(
        "sidecar",
        {
            "path": str(run.sidecar),
            "filename": SIDECAR_FILENAME,
            "bytes": run.sidecar.stat().st_size if run.sidecar.exists() else 0,
            "nodes": len(run.result.nodes),
            "run_id": run.run_id,
        },
    )
    if run.warnings:
        _print("warnings", list(run.warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
