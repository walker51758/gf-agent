"""Tests for the MCTS joint search package.

Coverage targets the four behaviours the design fixes and that a reader cannot
verify by inspection: the three φ-dependent quantities, the tree statistics that
survive a sidecar round-trip, the per-configuration evaluation cache, and the
"failed gate" node semantics.  The searches here run on a synthetic cache so the
suite needs no rqdatac licence and no network.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.candidate import (
    BacktestSpec,
    CompositionSpec,
    GatingSpec,
    SignalSpec,
    StrategyCandidate,
)
from rdagent.scenarios.gf_quant.strategy.search.budget import BACKTEST_CREDIT, CreditLedger
from rdagent.scenarios.gf_quant.strategy.search.data_cache import synthetic_cache
from rdagent.scenarios.gf_quant.strategy.search.engine import (
    EXPLORATION_C,
    REFERENCE_DEPTH,
    SearchEngine,
    SearchRunResult,
)
from rdagent.scenarios.gf_quant.strategy.search.evaluator import SearchEvaluation
from rdagent.scenarios.gf_quant.strategy.search.sidecar import (
    load_state,
    sidecar_path,
    write_state,
)
from rdagent.scenarios.gf_quant.strategy.search.tree import (
    Action,
    TreeNode,
    configuration_hash,
    root_node,
    tree_node_from_payload,
)

SYMBOLS = tuple(f"S{index:03d}.XSHE" for index in range(12))


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("mcts_cache")
    dates = pd.bdate_range("2023-01-02", "2024-12-31")
    return synthetic_cache(root, symbols=SYMBOLS, dates=dates, seed=11)


def _leaf(node_id: str, *, factor_names=("a", "b"), depth: int = 2) -> TreeNode:
    signal = SignalSpec()
    gating = GatingSpec()
    composition = CompositionSpec(top_n=8)
    # An empty factor set is the root shape, so its path label cannot be an
    # "add_factor" edge; the caller that needs a branch root overrides ``path``.
    path = () if not factor_names else (f"add_factor:{factor_names[0]}",)
    return TreeNode(
        node_id=node_id,
        path=path,
        factor_names=tuple(factor_names),
        signal=signal,
        gating=gating,
        composition=composition,
        config_hash=configuration_hash(
            tuple(factor_names), signal=signal, gating=gating, composition=composition
        ),
        depth=depth,
    )


class _ScriptedEvaluator:
    """Evaluate by configuration, returning whatever Q the script names.

    Keyed by factor set so a test can say "this configuration scores 0.4" and get
    exactly that back no matter which path reached it -- which is the property
    the cache test needs to observe.
    """

    def __init__(self, q_by_factors: dict[tuple[str, ...], float], *, gate_ok: bool = True) -> None:
        self.q_by_factors = q_by_factors
        self.gate_ok = gate_ok
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, candidate: StrategyCandidate) -> SearchEvaluation:
        key = tuple(candidate.factor_names)
        self.calls.append(key)
        q = self.q_by_factors.get(key, 0.0)
        return SearchEvaluation(
            q_value=q,
            valid_q=self.gate_ok,
            gate_failures=() if self.gate_ok else ("strategy_holdings",),
            metrics={"excess_annualized_return": q, "observations": 100, "avg_holdings": 8.0},
        )


class _StubActions:
    def __init__(self, actions) -> None:
        self._actions = list(actions)

    def actions_for(self, node: TreeNode):
        return list(self._actions)

    def accepted_factor_names(self):
        return []


# ---- budget ------------------------------------------------------------------


def test_phi_falls_linearly_to_zero_as_credits_are_spent() -> None:
    """φ is the budget signal the whole search reads, so its shape is the contract."""

    ledger = CreditLedger(total=12)
    assert ledger.phi == pytest.approx(1.0)
    ledger.spend("backtest", 3)
    assert ledger.phi == pytest.approx(1.0 - 3 / 12)
    ledger.spend("backtest", 9)
    assert ledger.phi == pytest.approx(0.0)
    assert ledger.exhausted


def test_ledger_refuses_to_overdraw() -> None:
    """A clamped ledger would let the loop spend while φ reads zero."""

    ledger = CreditLedger(total=2)
    ledger.spend("backtest", 2)
    with pytest.raises(RuntimeError, match="exhausted"):
        ledger.spend("backtest", 1)


def test_ledger_rejects_a_spend_of_zero_credits() -> None:
    with pytest.raises(ValueError, match="at least one credit"):
        CreditLedger(total=2).spend("backtest", 0)


# ---- the three φ-dependent quantities ---------------------------------------


def _engine(evaluator, *, total=12, actions=None) -> SearchEngine:
    return SearchEngine(
        evaluate=evaluator,
        actions=actions or _StubActions([]),
        total_credits=total,
    )


def test_exploration_bonus_carries_phi() -> None:
    """The bonus is the standard PUCT term times φ, and φ is what cools it."""

    engine = _engine(_ScriptedEvaluator({}))
    parent = _leaf("parent", depth=1)
    parent.visits = 4
    child = _leaf("child", depth=2)

    full = engine.exploration_bonus(parent, child, 1.0)
    expected = EXPLORATION_C * math.sqrt(math.log(4) / (1 + 0))
    assert full == pytest.approx(expected)

    engine.ledger.spend("backtest", 6)  # φ = 0.5
    assert engine.exploration_bonus(parent, child, 1.0) == pytest.approx(full / 2)

    engine.ledger.spend("backtest", 6)  # φ = 0.0
    assert engine.exploration_bonus(parent, child, 1.0) == pytest.approx(0.0)


def test_depth_correction_peaks_at_half_budget_and_vanishes_at_both_ends() -> None:
    """φ(1-φ)·d/d* is zero at both extremes, so it biases neither end of a run."""

    engine = _engine(_ScriptedEvaluator({}))
    deep = _leaf("deep", depth=REFERENCE_DEPTH)
    assert engine.depth_correction(deep) == pytest.approx(0.0)

    engine.ledger.spend("backtest", 6)  # φ = 0.5
    assert engine.depth_correction(deep) == pytest.approx(0.25)

    engine.ledger.spend("backtest", 6)  # φ = 0.0
    assert engine.depth_correction(deep) == pytest.approx(0.0)


def test_expansion_motivation_keeps_the_mean_and_scales_the_spread_by_phi() -> None:
    engine = _engine(_ScriptedEvaluator({}))
    parent = _leaf("parent")
    for value in (0.1, 0.3, 0.5, 0.7):
        parent.q_sum += value
        parent._q_sq_sum += value * value
        parent.q_valid += 1
    mean = parent.q_value
    spread = math.sqrt(parent.q_variance())
    assert spread > 0

    assert engine.expansion_motivation(parent) == pytest.approx(mean + spread)
    engine.ledger.spend("backtest", 6)  # φ = 0.5
    assert engine.expansion_motivation(parent) == pytest.approx(mean + 0.5 * spread)
    engine.ledger.spend("backtest", 6)  # φ = 0.0
    assert engine.expansion_motivation(parent) == pytest.approx(mean)


def test_expansion_motivation_needs_two_valid_qs_to_report_a_spread() -> None:
    """A single draw has no spread; reporting one would invent agitation."""

    engine = _engine(_ScriptedEvaluator({}))
    parent = _leaf("parent")
    parent.q_sum, parent._q_sq_sum, parent.q_valid = 0.5, 0.25, 1
    assert parent.q_variance() == 0.0
    assert engine.expansion_motivation(parent) == pytest.approx(0.5)


# ---- backpropagation and gate-failed semantics -------------------------------


def test_backpropagation_updates_every_ancestor_q() -> None:
    engine = _engine(_ScriptedEvaluator({}))
    root = root_node()
    engine.seed_root(root)
    mid = _leaf("mid", depth=2)
    mid.parent = root.node_id
    engine.nodes[mid.node_id] = mid
    leaf = _leaf("leaf", depth=3)
    leaf.parent = mid.node_id
    engine.nodes[leaf.node_id] = leaf

    engine.backpropagate(leaf, SearchEvaluation(0.4, True, (), {}))

    assert leaf.q_value == pytest.approx(0.4)
    assert mid.q_value == pytest.approx(0.4)
    assert root.q_value == pytest.approx(0.4)
    assert all(node.visits == 1 for node in (root, mid, leaf))


def test_gate_failed_outcome_counts_visits_but_never_enters_a_mean() -> None:
    """The design keeps a failed node counted and expandable, with an invalid Q."""

    engine = _engine(_ScriptedEvaluator({}))
    root = root_node()
    engine.seed_root(root)
    node = _leaf("node", depth=1)
    node.parent = root.node_id
    engine.nodes[node.node_id] = node

    engine.backpropagate(node, SearchEvaluation(0.9, False, ("strategy_holdings",), {}))

    assert node.visits == 1
    assert node.q_valid == 0
    assert node.q_value == pytest.approx(0.0)
    assert not node.has_valid_q
    assert root.visits == 1


def test_best_node_ignores_gate_failed_and_undersized_configurations() -> None:
    """Convergence ranks by Q, then the hard screen still applies."""

    engine = _engine(_ScriptedEvaluator({}))
    root = root_node()
    engine.seed_root(root)
    good = _leaf("good", factor_names=("a", "b"), depth=1)
    good.q_sum, good.q_sq_sum, good.q_valid, good.gate_passed = 0.2, 0.04, 1, True
    failing = _leaf("failing", factor_names=("a", "b"), depth=1)
    failing.q_sum, failing.q_sq_sum, failing.q_valid, failing.gate_passed = 0.9, 0.81, 1, False
    single = _leaf("single", factor_names=("a",), depth=1)
    single.q_sum, single.q_sq_sum, single.q_valid, single.gate_passed = 0.8, 0.64, 1, True
    engine.nodes.update({node.node_id: node for node in (good, failing, single)})

    best = engine.best_node()
    assert best is not None and best.node_id == "good"


# ---- the evaluation cache ----------------------------------------------------


def test_a_configuration_reached_twice_is_evaluated_once(cache) -> None:
    """The cache is keyed by configuration hash, not by path or node id."""

    evaluator = _ScriptedEvaluator({("a", "b"): 0.3})
    engine = _engine(evaluator, actions=_StubActions([Action(
        kind="add_factor",
        label="add_factor:a",
        factor_names=("a", "b"),
        signal=SignalSpec(),
        gating=GatingSpec(),
        composition=CompositionSpec(top_n=8),
    )]))
    root = root_node()
    engine.seed_root(root)
    left = _leaf("left", factor_names=("a", "b"))
    left.parent = root.node_id
    right = _leaf("right", factor_names=("a", "b"))
    right.parent = root.node_id
    engine.nodes.update({left.node_id: left, right.node_id: right})
    assert left.config_hash == right.config_hash

    first = engine.simulate(left)
    second = engine.simulate(right)

    assert first.cached is False
    assert second.cached is True
    assert len(evaluator.calls) == 1
    assert second.q_value == pytest.approx(0.3)


def test_configuration_hash_ignores_name_and_is_stable_across_field_order() -> None:
    """Identity is the configuration, so a rename must not fork the cache."""

    composition = CompositionSpec(top_n=8)
    signal = SignalSpec()
    gating = GatingSpec()
    first = configuration_hash(("a", "b"), signal=signal, gating=gating, composition=composition)
    second = configuration_hash(("a", "b"), signal=SignalSpec(), gating=GatingSpec(), composition=CompositionSpec(top_n=8))
    third = configuration_hash(("b", "a"), signal=signal, gating=gating, composition=composition)

    assert first == second
    assert first != third
    assert len(first) == 64


def test_configuration_hash_changes_when_the_structure_changes() -> None:
    signal = SignalSpec()
    gating = GatingSpec()
    base = configuration_hash(("a", "b"), signal=signal, gating=gating, composition=CompositionSpec(top_n=8))
    wider = configuration_hash(("a", "b"), signal=signal, gating=gating, composition=CompositionSpec(top_n=12))
    standardized = configuration_hash(
        ("a", "b"), signal=SignalSpec(standardize="zscore"), gating=gating, composition=CompositionSpec(top_n=8)
    )

    assert base != wider
    assert base != standardized


# ---- sidecar round-trip ------------------------------------------------------


def test_sidecar_round_trip_preserves_statistics_and_structure(tmp_path: Path) -> None:
    root = root_node()
    child = _leaf("child", factor_names=("a", "b"), depth=1)
    child.parent = root.node_id
    child.visits, child.q_sum, child.q_sq_sum, child.q_valid = 5, 1.5, 0.5, 5
    child.gate_passed = True
    child.gate_failures = ()
    root.children["add_factor:a"] = child.node_id
    root.child_labels["add_factor:a"] = "append accepted factor a"
    nodes = {root.node_id: root, child.node_id: child}

    path = sidecar_path(tmp_path)
    write_state(
        path,
        run_id="run1",
        nodes=nodes,
        root_id=root.node_id,
        runs_completed=2,
        credits_spent_total=24,
    )
    state = load_state(path)

    assert state is not None
    assert state.run_id == "run1"
    assert state.runs_completed == 2
    assert state.credits_spent_total == 24
    restored_root = state.nodes[state.root_id]
    restored_child = state.nodes["child"]
    assert restored_root.children == {"add_factor:a": "child"}
    assert restored_child.q_value == pytest.approx(child.q_value)
    assert restored_child.visits == 5
    assert restored_child.q_variance() == pytest.approx(child.q_variance())
    assert restored_child.has_valid_q
    assert restored_child.factor_names == ("a", "b")


def test_loading_an_absent_sidecar_reports_nothing_to_continue(tmp_path: Path) -> None:
    assert load_state(sidecar_path(tmp_path)) is None


def test_sidecar_refuses_an_unknown_version(tmp_path: Path) -> None:
    """A half-read tree would look like a search that mysteriously forgot everything."""

    path = sidecar_path(tmp_path)
    path.write_text('{"version": 999, "nodes": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported search sidecar version"):
        load_state(path)


def test_node_payload_round_trip_is_equality_preserving() -> None:
    node = _leaf("node", factor_names=("a", "b"))
    node.visits, node.q_sum, node.q_sq_sum, node.q_valid = 3, 0.9, 0.3, 3
    node.gate_passed = True

    restored = tree_node_from_payload(node.as_payload())

    assert restored.node_id == node.node_id
    assert restored.path == node.path
    assert restored.factor_names == node.factor_names
    assert restored.config_hash == node.config_hash
    assert restored.q_value == pytest.approx(node.q_value)
    assert restored.q_variance() == pytest.approx(node.q_variance())
    assert restored.signal == node.signal
    assert restored.gating == node.gating
    assert restored.composition == node.composition


# ---- the loop ----------------------------------------------------------------


def test_a_run_stops_when_the_budget_is_spent() -> None:
    """With actions to spare, the loop exits on credits and every credit buys a Q."""

    evaluator = _ScriptedEvaluator({}, gate_ok=False)
    actions = _StubActions(
        [
            Action(
                kind="add_factor",
                label=f"add_factor:{name}",
                factor_names=(name, "b"),
                signal=SignalSpec(),
                gating=GatingSpec(),
                composition=CompositionSpec(top_n=8),
            )
            for name in ("a", "c", "d", "e", "f", "g", "h", "i")
        ]
    )
    engine = _engine(evaluator, total=4, actions=actions)
    result = engine.run(root_node(), max_nodes=500)

    assert isinstance(result, SearchRunResult)
    assert result.ledger.spent == 4
    assert result.ledger.exhausted
    assert len(evaluator.calls) == 4


def test_a_run_stops_when_no_action_remains_even_with_budget_left() -> None:
    """An exhausted action space is a stop condition in its own right.

    Asserting only the budget here would be wrong: the loop must not spend its
    way to zero by re-evaluating nodes it has already expanded, and a test that
    demanded a fully spent budget would force exactly that behaviour.
    """

    evaluator = _ScriptedEvaluator({}, gate_ok=False)
    engine = _engine(evaluator, total=4, actions=_StubActions([]))
    result = engine.run(root_node(), max_nodes=500)

    assert result.ledger.spent == 0
    assert len(evaluator.calls) == 0


def test_a_continued_run_resolves_every_child_of_the_restored_root() -> None:
    """A restored root points at ids the engine must already hold.

    The failure this guards is subtle and only shows up on the second run: the
    sidecar restores a root whose ``children`` map names node ids, and selection
    dereferences those through the engine's node table.  Restoring the root
    without its nodes leaves a tree that parses cleanly and then fails on the
    first descent.
    """

    nodes: dict[str, TreeNode] = {}
    root = root_node()
    for label in ("branch:factor_first", "branch:strategy_first"):
        branch = _leaf(f"b{len(nodes)}", factor_names=(), depth=1)
        branch.path = (label,)
        branch.parent = root.node_id
        root.children[label] = branch.node_id
        nodes[branch.node_id] = branch
    nodes[root.node_id] = root

    engine = _engine(_ScriptedEvaluator({}))
    engine.nodes.update(nodes)
    engine.seed_root(root, seed_branches=False)

    assert len(engine.nodes) == len(nodes)
    for child_id in root.children.values():
        assert child_id in engine.nodes
    assert engine.select_child(root) is not None


def test_a_zero_factor_node_is_not_charged_a_credit() -> None:
    """Charging for a guaranteed rejection would burn the budget on nothing."""

    evaluator = _ScriptedEvaluator({})
    engine = _engine(evaluator, total=3, actions=_StubActions([]))
    root = root_node()
    engine.seed_root(root)
    branch = engine.nodes[root.children[next(iter(root.children))]]

    outcome = engine.simulate(branch)

    assert outcome.valid_q is False
    assert engine.ledger.spent == 0
    assert evaluator.calls == []


def test_a_factor_bearing_node_is_charged_one_credit() -> None:
    """The complement of the rule above, and the reason a run can end with 0 spent.

    A run whose every node lacks a factor spends nothing at all -- the budget is
    only consumed by evaluations that could have learned something.  That makes
    "spent == 0" a diagnostic of an action space that never admitted a factor,
    not a ledger fault, and this test is what separates the two readings.
    """

    evaluator = _ScriptedEvaluator({})
    engine = _engine(evaluator, total=3, actions=_StubActions([]))
    root = root_node()
    engine.seed_root(root)
    node = _leaf("carrying", factor_names=("mom_20",), depth=1)
    engine.nodes[node.node_id] = node

    engine.simulate(node)

    assert engine.ledger.spent == BACKTEST_CREDIT
    assert engine.ledger.by_kind == {"backtest": BACKTEST_CREDIT}
    assert evaluator.calls == [("mom_20",)]
