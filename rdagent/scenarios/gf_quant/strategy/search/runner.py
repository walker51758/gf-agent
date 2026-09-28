"""Run assembly: panels, factors, evaluator, action space, loop, sidecar.

This is the only module that knows about *all* the pieces, and it exists so the
demo script and the tests can share one wiring instead of each rebuilding it.
Everything it assembles is an existing component: the proposal engines, the
acceptance services, the local engine, the search loop.

Two construction details are worth naming.

**Panels.** A search needs factor panels.  In the first version they are derived
from the cached close series by the small, explicit set below -- momentum,
volatility, reversal and liquidity.  A factor panel in this engine is any
date x symbol float frame, so these are real inputs to the real rank pipeline,
not placeholders; what they are *not* is a substitute for a registry of accepted
research factors, which is the second source the action generator also reads.

**Directions.** The direction of a declared factor is a property of its
definition here, and stored factors carry theirs from the factor stage.  The
search never re-derives one, because a search free to flip a sign could always
improve its own Q without learning anything.
"""

from __future__ import annotations

import dataclasses
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec
from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
from rdagent.scenarios.gf_quant.strategy.search.budget import DEFAULT_TOTAL_CREDITS
from rdagent.scenarios.gf_quant.strategy.search.data_cache import SearchDataCache
from rdagent.scenarios.gf_quant.strategy.search.engine import (
    BRANCH_FACTOR,
    BRANCH_STRATEGY,
    SearchEngine,
    SearchRunResult,
)
from rdagent.scenarios.gf_quant.strategy.search.evaluator import (
    LOCAL_BACKEND,
    LocalBacktestEvaluator,
    build_evaluator,
)
from rdagent.scenarios.gf_quant.strategy.search.sidecar import (
    load_state,
    sidecar_path,
    write_state,
)
from rdagent.scenarios.gf_quant.strategy.search.tree import TreeNode, root_node
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
)

# The declared factor panels and their frozen directions.  Long-only, so a
# factor's direction is the sign of its expected cross-sectional relation to
# forward return; these four are the standard price/volume family and need
# nothing beyond the cached close series.
DECLARED_FACTORS: Mapping[str, int] = {
    "mom_20": 1,
    "mom_60": 1,
    "vol_20": -1,
    "rev_5": 1,
    "illiq_20": 1,
}


def declared_factor_panels(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the declared factor panels from the cached return panel.

    Every panel is computed from returns up to and including the decision date
    only: the engine's own signal lag then delays the trade, so no panel may
    embed a forward-looking window.  ``min_periods`` is left at the pandas
    default for the rolling windows, which yields ``NaN`` during warm-up rather
    than a fabricated partial estimate; the rank pipeline already treats a
    missing value as "not selectable today".
    """

    if returns.empty:
        raise ValueError("return panel is empty; cannot build factor panels")
    amount = (1.0 + returns).cumprod()
    panels: dict[str, pd.DataFrame] = {
        "mom_20": amount.pct_change(20),
        "mom_60": amount.pct_change(60),
        "vol_20": returns.rolling(20).std() * (252**0.5),
        "rev_5": amount.pct_change(5).mul(-1.0),
        # Illiquidity proxy: a name whose returns move more per unit of turnover
        # is harder to trade, which is the effect the factor is meant to capture.
        "illiq_20": returns.abs().rolling(20).mean(),
    }
    return {name: panel.astype(float) for name, panel in panels.items()}


def declared_directions(names: tuple[str, ...]) -> dict[str, int]:
    missing = [name for name in names if name not in DECLARED_FACTORS]
    if missing:
        raise ValueError(f"unknown declared factor(s): {missing[:3]}")
    return {name: DECLARED_FACTORS[name] for name in names}


@dataclass
class SearchRun:
    """One assembled run, kept as an object so a caller can inspect its parts."""

    result: SearchRunResult
    sidecar: Path
    run_id: str
    cache_summary: Mapping[str, Any]
    evaluator: LocalBacktestEvaluator
    warnings: list[str] = field(default_factory=list)
    # Measured, not estimated: the number of proposal-engine invocations the run
    # made.  A report that states an LLM call count has to be able to point at
    # the counter behind it.
    llm_proposal_rounds: int = 0
    strategy_proposal_rounds: int = 0
    factor_candidates_evaluated: int = 0
    factors_accepted: tuple[str, ...] = ()
    proposal_notes: tuple[str, ...] = ()


def build_action_generator(
    *,
    factor_names: tuple[str, ...],
    strategy_engine: Any | None = None,
    strategy_context_factory: Any | None = None,
) -> ActionGenerator:
    """Action space over the declared factors plus whatever the registries hold.

    The declared panels are exposed as a registry-shaped stub so the action
    generator's "accepted factors" source has something to read without a
    database.  A real registry, when present, takes precedence.
    """

    return ActionGenerator(
        factor_registry=_DeclaredFactorRegistry(factor_names),
        factor_service=None,
        proposer=None,
        strategy_engine=strategy_engine,
        strategy_context_factory=strategy_context_factory,
    )


def build_factor_proposer(factor_registry: Any, factor_service: Any) -> Any | None:
    """Wire the existing DSL proposer, or ``None`` when the LLM is unusable.

    Configuration goes through the repository's own
    :func:`rdagent.app.gf_rd_loop.conf.configure_gf_llm_backend` -- the same
    entry point the two GF loops call at startup -- rather than a new settings
    path.  A missing credential is reported as "no proposer" instead of raised:
    the design's factor actions have two sources, and losing the proposal one
    must leave the registry source working rather than abort the search.
    """

    from rdagent.app.gf_rd_loop.conf import configure_gf_llm_backend

    configure_gf_llm_backend()
    try:
        from rdagent.scenarios.gf_quant.mining import RDAGentDSLProposer

        proposer = RDAGentDSLProposer()
    except Exception as error:
        warnings.warn(f"factor DSL proposer unavailable: {type(error).__name__}: {error}", stacklevel=2)
        return None
    if factor_registry is None or factor_service is None:
        warnings.warn(
            "factor proposer built but no registry/acceptance service was supplied; "
            "proposed factors would never be gated, so the proposer is not wired",
            stacklevel=2,
        )
        return None
    return proposer


def build_strategy_engine() -> Any | None:
    """Wire the existing LLM strategy proposal engine, or ``None`` when unusable.

    Same convention and same degradation rule as the factor proposer.  The
    engine is returned, not called: the search's action space reads it through
    the ``LLMStrategyProposalEngine`` prompt vocabulary that already exists.
    """

    from rdagent.app.gf_rd_loop.conf import configure_gf_llm_backend

    configure_gf_llm_backend()
    try:
        from rdagent.scenarios.gf_quant.strategy.proposal import LLMStrategyProposalEngine

        return LLMStrategyProposalEngine()
    except Exception as error:
        warnings.warn(f"strategy proposal engine unavailable: {type(error).__name__}: {error}", stacklevel=2)
        return None


class _DeclaredFactorRegistry:
    """Minimal read-only stand-in with the two methods the generator uses."""

    class _Record:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = tuple(names)

    def list(self, *, status: str | None = None) -> list[Any]:
        if status not in (None, "accepted"):
            return []
        return [self._Record(name) for name in self._names]


def build_local_proposal_chain(
    *,
    cache: SearchDataCache,
    work_dir: str | Path,
    factor_registry: Any | None = None,
) -> tuple[Any, Any, Any, list[str]]:
    """Assemble the local factor-proposal chain: proposer + RankIC gate + registry.

    Returns ``(proposer, factor_service, registry, notes)``.  Every collaborator
    is the existing one -- ``RDAGentDSLProposer`` for proposals,
    ``FactorAcceptanceService`` for the gate, ``FactorRegistry`` for accepted
    records -- with only the evaluator swapped from remote to local.  A missing
    API credential is reported as a note and yields no proposer rather than an
    exception, because the registry-only action space still runs.
    """

    from rdagent.scenarios.gf_quant import gate_policy
    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.search.factor_eval import (
        LocalRankICFactorEvaluator,
        settings_for_cache,
        verify_cache_layout,
    )

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    verify_cache_layout(cache.root)
    settings = settings_for_cache(cache)
    notes.append(f"local RankIC window {settings.start}..{settings.end}")
    evaluator = LocalRankICFactorEvaluator(
        cache=cache,
        settings=settings,
        artifact_dir=work / "factor_artifacts",
    )
    registry = factor_registry or FactorRegistry(work / "factor_registry.sqlite")
    # The gate selection comes from the temporary regime switch; setting that
    # switch back to empty restores the full calibrated gate set without touching
    # this call.  See rdagent/scenarios/gf_quant/gate_policy.py.
    selection = gate_policy.active_gates()
    service = FactorAcceptanceService(registry, evaluator, active_gates=selection)
    if selection is None:
        notes.append("acceptance gates: all calibrated gates enforced")
    else:
        suppressed = sorted(set(gate_policy.GATE_ORDER) - set(selection))
        notes.append(f"acceptance gates enforced: {len(selection)}/10")
        notes.append(f"acceptance gates suppressed: {', '.join(suppressed)}")

    from rdagent.app.gf_rd_loop.conf import configure_gf_llm_backend

    # The repository's own LLM wiring; calling it here rather than reading a
    # settings object keeps one configuration path for both GF loops and the
    # search.
    configure_gf_llm_backend()
    try:
        from rdagent.scenarios.gf_quant.mining import RDAGentDSLProposer

        proposer = RDAGentDSLProposer()
    except Exception as error:
        notes.append(f"factor proposer unavailable: {type(error).__name__}: {error}")
        proposer = None
    return proposer, service, registry, notes


def build_strategy_context_factory(
    *,
    directions: Mapping[str, int],
    evaluation_start: str,
    evaluation_end: str,
) -> Any:
    """Build the strategy engine's context from the node being expanded.

    ``LLMStrategyProposalEngine`` expects a frozen working set and a trial
    index, which belong to the collaboration layer's epoch machinery.  The joint
    search has no epochs -- it has nodes -- so the context is synthesised here
    from the node: the working set holds the factors the node carries, the trial
    index is its depth, and the snapshot checksum is its configuration hash.
    That keeps the engine's prompt vocabulary intact while giving it a truthful
    description of what the search is currently looking at.

    Directions come from the caller because they are a property of a factor's
    definition, not of the node, and the search never re-derives one.  Artifact
    provenance is the one field with no search-side counterpart: the node's
    factors were evaluated by the local evaluator, so the node's own
    configuration hash stands in as the version stamp.
    """

    from rdagent.scenarios.gf_quant.strategy.orchestrator import StrategyProposalContext
    from rdagent.scenarios.gf_quant.strategy.snapshot import (
        FactorSnapshotItem,
        FactorWorkingSet,
        working_set_id_for,
    )

    def factory(node: Any) -> Any:
        members = tuple(
            FactorSnapshotItem(
                factor_id=f"search_{node.config_hash[:12]}_{name}",
                name=str(name),
                family="price_volume",
                expression=str(name),
                eligibility_tier="core",
                eligibility_reason="held by the search node this context describes",
                artifact_path=f"search://{node.config_hash[:12]}/{name}",
                artifact_checksum=node.config_hash,
                direction=int(directions.get(name, 1)),
                theme="search",
                intended_role="member",
                correlation_cluster="",
                quality_score=0.0,
                stability_score=0.0,
                metrics_json="{}",
                source_updated_at="",
            )
            for name in node.factor_names
        )
        core_limit = max(len(members), 1)
        factor_ids = [member.factor_id for member in members]
        working_set = FactorWorkingSet(
            working_set_id=working_set_id_for(node.config_hash, factor_ids, core_limit, 0),
            snapshot_id=node.config_hash,
            snapshot_checksum=node.config_hash,
            core_limit=core_limit,
            conditional_limit=0,
            factors=members,
        )
        return StrategyProposalContext(
            epoch_id=f"search_{node.config_hash[:12]}",
            trial_index=int(node.depth),
            data_cutoff=evaluation_end,
            snapshot_id=node.config_hash,
            working_set=working_set,
            prior_outcomes=tuple(_outcome_for(failure) for failure in node.gate_failures[:3]),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )

    return factory


def _outcome_for(failure: str) -> Any:
    from rdagent.scenarios.gf_quant.strategy.orchestrator import TrialOutcomeSummary

    return TrialOutcomeSummary(
        trial_index=0,
        strategy_name=failure,
        status="rejected",
        gate_failures=(failure,),
        metrics={},
    )


def run_search(
    *,
    cache: SearchDataCache | None = None,
    cache_root: str | Path | None = None,
    sidecar_dir: str | Path,
    total_credits: int = DEFAULT_TOTAL_CREDITS,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
    rebalance_frequency: str = "monthly",
    backend: str = LOCAL_BACKEND,
    factor_names: tuple[str, ...] | None = None,
    factor_registry: Any | None = None,
    proposer: Any | None = None,
    factor_service: Any | None = None,
    strategy_engine: Any | None = None,
    policy: StrategyAcceptancePolicy | None = None,
    max_nodes: int = 200,
    continue_existing: bool = True,
    auto_wire_proposals: bool = True,
    work_dir: str | Path | None = None,
) -> SearchRun:
    """Assemble and run one search, persisting the tree to the sidecar.

    ``continue_existing`` loads an existing sidecar and continues it; the tree
    statistics accumulate while the credit budget restarts, which is the design's
    separation between what the search has learned and how much it may spend
    this time.
    """

    search_cache = cache if cache is not None else SearchDataCache(cache_root)
    close = search_cache.close_wide()
    start = evaluation_start or close.index.min().strftime("%Y-%m-%d")
    end = evaluation_end or close.index.max().strftime("%Y-%m-%d")

    local = LocalBacktestEvaluator(
        cache=search_cache,
        evaluation_start=start,
        evaluation_end=end,
        backtest_spec=BacktestSpec(),
        rebalance_frequency=rebalance_frequency,
        policy=policy,
    )
    panels = declared_factor_panels(local.returns)
    resolved_names = tuple(factor_names) if factor_names is not None else tuple(sorted(DECLARED_FACTORS))
    directions = declared_directions(resolved_names)
    # Every declared panel is registered on the evaluator, not only the selected
    # subset: the decision calendar is the intersection of every registered
    # factor's valid dates, and restricting it to one node's factor set would
    # give different nodes different evaluation windows.
    all_directions = declared_directions(tuple(sorted(DECLARED_FACTORS)))
    for name in sorted(DECLARED_FACTORS):
        local.register_factor(name, panels[name], all_directions[name])

    # One local evaluator instance, shared by the search and returned on the run
    # object.  Building a second one through ``build_evaluator`` would load the
    # panels twice and register every factor panel twice for no benefit.
    evaluate = build_evaluator(
        backend,
        cache=search_cache,
        evaluation_start=start,
        evaluation_end=end,
        rebalance_frequency=rebalance_frequency,
        policy=policy,
        factor_evaluator=local,
    )

    notes: list[str] = []
    if proposer is None or factor_service is None:
        if auto_wire_proposals:
            proposer, factor_service, factor_registry, chain_notes = build_local_proposal_chain(
                cache=search_cache,
                work_dir=work_dir or Path(sidecar_dir).parent,
                factor_registry=factor_registry,
            )
            notes.extend(chain_notes)
    if strategy_engine is None and auto_wire_proposals:
        strategy_engine = build_strategy_engine()
        if strategy_engine is None:
            notes.append("strategy proposal engine unavailable; strategy actions are ladder-only")
    actions = ActionGenerator(
        factor_registry=factor_registry,
        factor_service=factor_service,
        proposer=proposer,
    )
    if strategy_engine is not None:
        actions.strategy_engine = strategy_engine
        actions.strategy_context_factory = build_strategy_context_factory(
            directions=all_directions,
            evaluation_start=start,
            evaluation_end=end,
        )
        notes.append("strategy proposals wired: LLMStrategyProposalEngine")
    if proposer is None or factor_service is None:
        actions.factor_registry = _DeclaredFactorRegistry(resolved_names)
        notes.append(
            "factor proposal engines not wired: the action space is the declared "
            "panel set only, so no new factor can enter from a proposal"
        )
    else:
        notes.append("factor proposals wired: RDAGentDSLProposer + local RankIC gate")

    target = sidecar_path(sidecar_dir)
    state = load_state(target) if continue_existing else None
    root = root_node()

    engine = SearchEngine(
        evaluate=evaluate,
        actions=actions,
        total_credits=total_credits,
        policy=policy,
        branch_labels=(BRANCH_FACTOR, BRANCH_STRATEGY),
    )
    if state is not None:
        # Every node comes back, not just the root: the root's children map names
        # ids that selection dereferences through ``engine.nodes``, so restoring
        # the root alone would leave the tree pointing at nodes the engine has
        # never heard of.
        root = _restore_tree(state, root)
        engine.nodes.update(state.nodes)
        engine.seed_root(root, seed_branches=False)
        notes.append(f"continued sidecar with {len(state.nodes)} nodes")

    run_id = f"search_{uuid.uuid4().hex[:12]}"
    result = engine.run(root, max_nodes=max_nodes)
    accepted_factor_names = (
        actions.accepted_factor_names() if actions.factor_registry is not None else []
    )

    previous_runs = 0 if state is None else state.runs_completed
    previous_credits = 0 if state is None else state.credits_spent_total
    write_state(
        target,
        run_id=run_id,
        nodes={node.node_id: node for node in result.nodes},
        root_id=root.node_id,
        runs_completed=previous_runs + 1,
        credits_spent_total=previous_credits + result.ledger.spent,
    )
    return SearchRun(
        result=result,
        sidecar=target,
        run_id=run_id,
        cache_summary=search_cache.summary(),
        evaluator=local,
        warnings=notes,
        llm_proposal_rounds=actions.proposal_rounds,
        strategy_proposal_rounds=actions.strategy_proposal_rounds,
        factor_candidates_evaluated=int(getattr(factor_service, "attempts", 0)),
        factors_accepted=tuple(accepted_factor_names),
        proposal_notes=tuple(actions.notes),
    )


def _restore_tree(state: Any, freshness: TreeNode) -> TreeNode:
    """Rebuild the in-memory tree from a sidecar, keeping the loaded statistics.

    The sidecar's own root object is returned when it has one, because its
    children map is the structure the continuation must extend; ``freshness`` is
    only the fallback for a sidecar written by a version that stored nodes
    without a root entry.
    """

    root = state.nodes.get(state.root_id)
    if root is not None:
        return root
    for node in state.nodes.values():
        if node.parent is None:
            return node
    return freshness


def best_configuration(run: SearchRun, *, policy: StrategyAcceptancePolicy | None = None) -> dict[str, Any]:
    """The convergence output: the best configuration, restated readably.

    Ranking is by Q over nodes with a valid gate.  The hard screen is re-run
    here on the same metrics the node was evaluated with -- taken from the run's
    own trace, not from the node payload, which does not carry metrics -- so the
    reported verdict restates a measurement rather than a stored boolean.  A
    verdict that disagreed with the loop would mean the loop's gate is not the
    one this function applies, which is exactly the drift worth surfacing.
    """

    best = run.result.best
    if best is None:
        return {
            "status": "no_best_node",
            "reason": "every evaluated configuration failed the strategy gate",
            "credits_spent": run.result.ledger.spent,
        }
    candidate = run.result.best_candidate
    assert candidate is not None
    metrics = dict(run.result.metrics_by_node.get(best.node_id, {}))
    gate = policy or StrategyAcceptancePolicy()
    if metrics:
        passed, failures = gate.check(metrics, candidate)
    else:
        passed, failures = False, ["strategy_metrics_missing"]
    return {
        "status": "ok",
        "node_id": best.node_id,
        "depth": best.depth,
        "path": list(best.path),
        "factor_names": list(best.factor_names),
        "config_hash": best.config_hash,
        "q_value": best.q_value,
        "visits": best.visits,
        "gate_passed": best.gate_passed,
        "composition": {
            "top_n": best.composition.top_n,
            "weighting": best.composition.weighting,
            "weight_cap": best.composition.weight_cap,
        },
        "signal": {"standardize": best.signal.standardize},
        "gating_layers": [layer.factor for layer in best.gating.layers],
        "hard_screen_passed": passed,
        "hard_screen_failures": list(failures),
        "credits_spent": run.result.ledger.spent,
        "cache_hits": run.result.cache_hits,
    }


def register_best(
    run: SearchRun,
    *,
    registry: Any,
    policy: StrategyAcceptancePolicy | None = None,
    name: str | None = None,
) -> Any:
    """Hand the winning configuration to :class:`StrategyAcceptanceService`.

    This is the convergence wiring the design asks for: the search does not
    write the registry itself, it produces a candidate and the existing
    acceptance service gates and records it.  The evaluator is supplied as the
    run's own metrics for that node rather than by re-running the backtest --
    the run already paid for that evaluation, and charging a second one would
    make registration depend on the budget it just exhausted.
    """

    best = run.result.best
    if best is None:
        raise ValueError("no node passed the strategy gate; nothing to register")
    candidate = run.result.best_candidate
    assert candidate is not None
    if name is not None:
        candidate = dataclasses.replace(candidate, name=name)
    metrics = dict(run.result.metrics_by_node.get(best.node_id, {}))
    if not metrics:
        raise ValueError(f"node {best.node_id} has no metrics on record; cannot register it")
    evaluation = StrategyEvaluation(
        metrics=metrics,
        returns=None,
    )
    service = StrategyAcceptanceService(
        registry,
        evaluator=lambda _candidate: evaluation,
        policy=policy or StrategyAcceptancePolicy(),
    )
    return service.process(candidate)
