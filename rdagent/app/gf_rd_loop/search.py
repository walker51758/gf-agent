"""Search-mode entry point: the MCTS joint search as a selectable main path.

The existing strategy loop (``rdagent.app.gf_rd_loop.strategy``) runs the
double-timescale epoch collaboration.  This module runs the single joint action
tree instead, and it is deliberately a sibling command rather than a flag on the
existing one: the two loops share no state, so folding them into one entry point
would mean a dispatcher whose branches have nothing in common.

One command, end to end::

    python -m rdagent.app.gf_rd_loop.search --credits 24 --refresh-data

The old collaboration layer is untouched and remains available as the control
path; nothing here imports or modifies it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from rdagent.scenarios.gf_quant.strategy.search.budget import DEFAULT_TOTAL_CREDITS
from rdagent.scenarios.gf_quant.strategy.search.data_cache import (
    DEFAULT_SYMBOL_COUNT,
    DEFAULT_YEARS,
    SearchDataCache,
    update_cache,
)
from rdagent.scenarios.gf_quant.strategy.search.isolation import (
    SearchPaths,
    isolated_paths,
    production_paths,
)
from rdagent.scenarios.gf_quant.strategy.search.runner import best_configuration, run_search


def _emit(title: str, payload: Any) -> None:
    typer.echo(f"\n=== {title} ===")
    if isinstance(payload, str):
        typer.echo(payload)
        return
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def main(
    credits: int = typer.Option(
        DEFAULT_TOTAL_CREDITS, min=1, help="Total search budget in credits for this run."
    ),
    data_root: str = typer.Option(
        "production",
        help="Where the run writes: 'production' or 'test'.  Test roots are isolated "
        "under git_ignore_folder/gf_quant/search_test and never touch production paths.",
    ),
    run_name: str = typer.Option("run", help="Subdirectory name under the chosen root."),
    cache_root: Path | None = typer.Option(None, help="Override the cache directory."),
    refresh_data: bool = typer.Option(False, help="Pull fresh panels from rqdatac (2 API calls)."),
    no_pull: bool = typer.Option(
        False, help="Refuse to pull when the cache is absent instead of pulling it."
    ),
    symbols: int = typer.Option(DEFAULT_SYMBOL_COUNT, min=10, help="Index members to cache."),
    years: int = typer.Option(DEFAULT_YEARS, min=1, help="Years of daily history to cache."),
    rebalance_frequency: str = typer.Option(
        "monthly", help="Rebalance schedule: daily, weekly, monthly or quarterly."
    ),
    fresh_tree: bool = typer.Option(False, help="Ignore an existing sidecar and start a new tree."),
    no_proposals: bool = typer.Option(False, help="Registry-only action space, no LLM proposal rounds."),
    report: Path | None = typer.Option(None, help="Write a JSON run report to this path."),
) -> None:
    """Run one MCTS joint search to budget exhaustion."""

    if data_root not in ("production", "test"):
        raise typer.BadParameter("data_root must be 'production' or 'test'")
    paths: SearchPaths = (
        production_paths() if data_root == "production" else isolated_paths(run_name)
    )
    paths = SearchPaths(
        root=paths.root,
        cache=cache_root or paths.cache,
        sidecar=paths.sidecar,
        work=paths.work,
        log=paths.log,
    ).create()

    if refresh_data:
        manifest = update_cache(
            paths.cache, symbol_count=symbols, years=years, force=True
        )
        _emit("cache (pulled)", manifest.as_payload())
    elif not (paths.cache / "manifest.json").exists():
        # An isolated root starts empty, and the acceptance gate's observation and
        # annual-year requirements are properties of the window rather than of the
        # factor, so a run without a cache in place would fail every proposal for a
        # reason the caller cannot see.  Pulling here keeps the command
        # self-contained; ``--no-pull`` is the explicit refusal.
        if no_pull:
            raise typer.BadParameter(
                f"no cache at {paths.cache} and --no-pull was given; "
                f"run with --refresh-data or point --cache-root at an existing cache"
            )
        manifest = update_cache(paths.cache, symbol_count=symbols, years=years, force=False)
        _emit("cache (pulled, none found)", manifest.as_payload())
    cache = SearchDataCache(paths.cache)

    run = run_search(
        cache=cache,
        sidecar_dir=paths.sidecar,
        total_credits=credits,
        rebalance_frequency=rebalance_frequency,
        continue_existing=not fresh_tree,
        auto_wire_proposals=not no_proposals,
        work_dir=paths.work,
    )
    best = best_configuration(run)
    summary = {
        "run_id": run.run_id,
        "paths": paths.as_payload(),
        "credits": run.result.ledger.as_payload(),
        "nodes": len(run.result.nodes),
        "sidecar_bytes": run.sidecar.stat().st_size if run.sidecar.exists() else 0,
        "llm_proposal_rounds": run.llm_proposal_rounds,
        "strategy_proposal_rounds": run.strategy_proposal_rounds,
        "factors_accepted": list(run.factors_accepted),
        "factors_evaluated": run.factor_candidates_evaluated,
        "cache_hits": run.result.cache_hits,
        "best": best,
        "notes": list(run.warnings),
        "proposal_notes": list(run.proposal_notes),
    }
    _emit("budget", summary["credits"])
    _emit("proposal activity", {
        "llm_proposal_rounds": summary["llm_proposal_rounds"],
        "strategy_proposal_rounds": summary["strategy_proposal_rounds"],
        "factors_accepted": summary["factors_accepted"],
        "factors_evaluated": summary["factors_evaluated"],
        "proposal_notes": summary["proposal_notes"],
    })
    _emit("best configuration", best)
    _emit("run summary", {
        key: summary[key]
        for key in ("run_id", "paths", "nodes", "sidecar_bytes", "cache_hits", "notes")
    })
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        _emit("report", str(report))


if __name__ == "__main__":
    typer.run(main)
