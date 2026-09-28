"""The selection / expansion / simulation / backpropagation loop.

The three budget-dependent quantities are all here, and all three are read off
φ rather than off a phase counter, so there is no step in the run where a
threshold flips the search from exploring to exploiting:

* **selection bonus** -- ``Q(s,a) + φ · c · P(s,a) · √(ln N(s) / (1 + N(s,a)))``.
  φ multiplies only the exploration half, so the bonus cools linearly to zero
  as the budget drains while the exploitation half is untouched.
* **depth correction** -- ``φ(1 - φ) · d / d*`` added to a node's Q before the
  parent's choice.  It peaks at half budget, which is the point the design
  wants deeper branches to look attractive, and vanishes at both ends.
* **expansion motivation** -- ``E_gen = μ(p) + φ · √σ²(p)``.  A parent whose
  children have been producing good numbers is worth expanding further while
  budget remains, and stops being worth it when φ goes to zero.

Failures are values, not exceptions.  A candidate the engine refuses, a gate the
configuration misses, and a proposal that never arrives each produce a node with
an invalid Q that stays counted and stays expandable -- one more action can
replace whatever caused the failure, which is the whole reason the design keeps
such nodes alive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rdagent.scenarios.gf_quant.strategy.candidate import (
    BacktestSpec,
    CompositionSpec,
    GatingSpec,
    SignalSpec,
    StrategyCandidate,
)
from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
from rdagent.scenarios.gf_quant.strategy.search.budget import (
    BACKTEST_CREDIT,
    DEFAULT_TOTAL_CREDITS,
    CreditLedger,
)
from rdagent.scenarios.gf_quant.strategy.search.evaluator import SearchEvaluation
from rdagent.scenarios.gf_quant.strategy.search.tree import (
    MAX_FACTORS,
    MIN_FACTORS,
    Action,
    TreeNode,
    configuration_hash,
)

# PUCT constants.  ``EXPLORATION_C`` is the exploration coefficient the design's
# formula scales by φ; the design leaves its value open, and 1.4 is the usual
# starting point for a bounded reward.
EXPLORATION_C = 1.4

# Reference depth d*.  The design leaves it open; 3 is the depth a two-factor
# configuration with one structural change reaches, which is the shape a
# sixteen-credit run can actually explore.
REFERENCE_DEPTH = 3

# A branch is a name for the first action taken from the root.  Two branches are
# opened so the search evaluates a factor-first and a structure-first
# configuration before committing either way.
BRANCH_FACTOR = "branch:factor_first"
BRANCH_STRATEGY = "branch:strategy_first"


@dataclass
class SearchRunResult:
    """Everything one run produced, in the order the report reads it."""

    best: TreeNode | None
    best_candidate: StrategyCandidate | None
    nodes: list[TreeNode]
    ledger: CreditLedger
    trace: list[Mapping[str, Any]] = field(default_factory=list)
    evaluations: int = 0
    cache_hits: int = 0
    # Full metric dictionaries per node, keyed by node id.  Kept beside the tree
    # rather than on the node because the node is the *configuration* record and
    # a metric set is a measurement of it; a caller that wants to re-run the hard
    # screen on the winning node needs the numbers, not a stored verdict.
    metrics_by_node: dict[str, Mapping[str, Any]] = field(default_factory=dict)


def _eligible_score(node: TreeNode) -> bool:
    """Does this node carry a factor set the strategy gate could accept?

    The floor is the same one the strategy worker enforces (2-8 factors,
    ``_validate_request``): a one-factor configuration is not a strategy the
    acceptance chain would ever register, so it must not be reported as the
    search's best output even if its Q happens to be the highest.  The ceiling is
    the action space's own bound.
    """

    return MIN_FACTORS <= len(node.factor_names) <= MAX_FACTORS


class SearchEngine:
    """One search run over a tree, against a fixed budget.

    The engine owns no data and no model: it reads panels through the evaluator,
    asks the action generator for candidates and writes the tree it built.  That
    split is what lets the same loop run offline against a synthetic cache and
    against the data machine later, without a code path difference.
    """

    def __init__(
        self,
        *,
        evaluate: Callable[[StrategyCandidate], SearchEvaluation],
        actions: ActionGenerator,
        total_credits: int = DEFAULT_TOTAL_CREDITS,
        universe: str = "csi_all",
        policy: Any | None = None,
        branch_labels: Sequence[str] = (BRANCH_FACTOR, BRANCH_STRATEGY),
    ) -> None:
        self.evaluate = evaluate
        self.actions = actions
        self.ledger = CreditLedger(total=total_credits)
        self.universe = universe
        self.policy = policy
        self.branch_labels = tuple(branch_labels)
        # Evaluation cache keyed by the *configuration* hash: the design keeps
        # node statistics per path while an evaluation result is a property of
        # the configuration alone, so two paths reaching one configuration share
        # the cached Q without sharing each other's visit counts.
        self._cache: dict[str, SearchEvaluation] = {}
        # PUCT priors, keyed by (parent node id, action label).  In the first
        # version every action carries the same prior, because the two proposal
        # engines return candidates without a calibrated score; the seam is
        # here so a proposer that does emit probabilities can feed them in
        # without touching the selection formula.
        self._priors: dict[tuple[str, str], float] = {}
        self.nodes: dict[str, TreeNode] = {}

    # ---- tree plumbing -----------------------------------------------------

    def _candidate(self, node: TreeNode, *, name: str) -> StrategyCandidate:
        return StrategyCandidate(
            name=name,
            universe=self.universe,
            composition=node.composition,
            factor_names=node.factor_names,
            rule="cross-sectional rank blend",
            hypothesis=f"MCTS node {node.node_id} at depth {node.depth}",
            backtest=BacktestSpec(),
            signal=node.signal,
            gating=node.gating,
        )

    def _child(self, parent: TreeNode, action: Action) -> TreeNode:
        path = tuple([*parent.path, action.label])
        node_id = f"n{len(self.nodes):04d}"
        node = TreeNode(
            node_id=node_id,
            path=path,
            factor_names=action.factor_names,
            signal=action.signal,
            gating=action.gating,
            composition=action.composition,
            config_hash=configuration_hash(
                action.factor_names,
                signal=action.signal,
                gating=action.gating,
                composition=action.composition,
                universe=self.universe,
            ),
            depth=parent.depth + 1,
            parent=parent.node_id,
        )
        self.nodes[node_id] = node
        parent.children[action.label] = node_id
        parent.child_labels[action.label] = action.detail or action.label
        return node

    def seed_root(self, root: TreeNode, *, seed_branches: bool = True) -> None:
        """Register the root, and the branch roots on a fresh tree.

        ``seed_branches=False`` is the continuation path: a restored tree already
        carries its branch roots as children, and re-creating them would both
        duplicate the structure and overwrite the statistics they accumulated in
        the earlier run.
        """

        self.nodes[root.node_id] = root
        if not seed_branches:
            return
        for label in self.branch_labels:
            branch = TreeNode(
                node_id=f"n{len(self.nodes):04d}",
                path=(label,),
                factor_names=(),
                signal=SignalSpec(),
                gating=GatingSpec(),
                composition=CompositionSpec(top_n=root.composition.top_n),
                config_hash="",
                depth=1,
                parent=root.node_id,
            )
            branch.config_hash = configuration_hash(
                (), signal=branch.signal, gating=branch.gating, composition=branch.composition,
                universe=self.universe,
            )
            self.nodes[branch.node_id] = branch
            root.children[label] = branch.node_id
            root.child_labels[label] = label

    # ---- the budget-aware quantities --------------------------------------

    def exploration_bonus(self, parent: TreeNode, child: TreeNode, action_prior: float) -> float:
        """``φ · c · P(s,a) · √(ln N(s) / (1 + N(s,a)))``.

        The design's one substitution against standard PUCT: the exploration
        term carries φ, so the search stops paying for the unknown as the budget
        runs out.  The first visit of a child gets the largest bonus, which is
        what makes an unvisited branch reachable at all.
        """

        visits = max(parent.visits, 1)
        return (
            self.ledger.phi
            * EXPLORATION_C
            * float(action_prior)
            * math.sqrt(math.log(visits) / (1 + child.visits))
        )

    def depth_correction(self, node: TreeNode) -> float:
        """``φ(1 - φ) · d / d*``, added to a node's Q during selection.

        Zero at both ends of the budget and maximal in the middle: early on
        every branch is equally worth opening, and late on the search should
        consolidate what it has rather than start deep new sub-branches.
        """

        phi = self.ledger.phi
        return phi * (1.0 - phi) * (node.depth / float(REFERENCE_DEPTH))

    def expansion_motivation(self, parent: TreeNode) -> float:
        """``μ(p) + φ · √σ²(p)``; the term that decides whether to expand at all.

        ``μ(p)`` is the parent's mean valid Q and stays constant as φ falls, so
        a parent that has produced good numbers remains worth expanding until
        the budget itself forces the second term away.
        """

        spread = math.sqrt(max(parent.q_variance(), 0.0))
        return parent.q_value + self.ledger.phi * spread

    # ---- selection ---------------------------------------------------------

    def select_child(self, parent: TreeNode) -> tuple[TreeNode, str] | None:
        """Pick the child the parent's own statistics point at.

        Returns ``None`` when the parent has no children yet, which is the
        expansion signal for the caller.  Selection reads ``has_valid_q``: a
        gated child's Q never enters the ranking, so a branch that failed its
        gate can only be reached through the exploration term -- which is the
        intended behaviour, not an accident.
        """

        if not parent.children:
            return None
        best_id: str | None = None
        best_label = ""
        best_score = -math.inf
        for label, child_id in parent.children.items():
            child = self.nodes[child_id]
            q = child.q_value if child.has_valid_q else 0.0
            bonus = self.exploration_bonus(parent, child, self._prior_for(parent, label))
            score = q + self.depth_correction(child) + bonus
            if score > best_score:
                best_score = score
                best_id = child_id
                best_label = label
        assert best_id is not None
        return self.nodes[best_id], best_label

    def _prior_for(self, parent: TreeNode, label: str) -> float:
        return float(self._priors.get((parent.node_id, label), 1.0))

    # ---- expansion ---------------------------------------------------------

    def expand(self, node: TreeNode, actions: Sequence[Action]) -> TreeNode | None:
        """Materialise the first action the node has not taken yet.

        Returning ``None`` means the node is fully expanded, which the caller
        reads as "descend further" rather than "give up".
        """

        for action in actions:
            if action.label in node.children:
                continue
            child = self._child(node, action)
            self._priors[(node.node_id, action.label)] = float(action.prior)
            return child
        return None

    # ---- simulation --------------------------------------------------------

    def simulate(self, node: TreeNode) -> SearchEvaluation:
        """Spend one credit on the node's configuration and return its outcome.

        A cache hit returns the previous outcome for free: the configuration was
        already paid for on another path, and charging twice for one answer
        would make the budget depend on how many routes reach a configuration
        rather than on how much was actually learned.
        """

        cached = self._cache.get(node.config_hash)
        if cached is not None:
            # The outcome keeps its original ``cached=False``; the hit is
            # reported on the returned copy so a caller can count hits without
            # the cached evaluation object mutating under other readers.
            return SearchEvaluation(
                q_value=cached.q_value,
                valid_q=cached.valid_q,
                gate_failures=cached.gate_failures,
                metrics=cached.metrics,
                cached=True,
            )
        if not self.ledger.can_afford(BACKTEST_CREDIT):
            raise RuntimeError("search budget exhausted before a simulation could run")
        # A node with no factors cannot be evaluated at all; charging a credit
        # for it would burn budget on a guaranteed rejection.
        if not node.factor_names:
            outcome = SearchEvaluation(
                q_value=0.0,
                valid_q=False,
                gate_failures=("strategy_needs_at_least_one_factor",),
                metrics={},
            )
        else:
            candidate = self._candidate(node, name=f"mcts_{node.node_id}")
            outcome = self.evaluate(candidate)
            self.ledger.spend("backtest", BACKTEST_CREDIT)
        self._cache[node.config_hash] = outcome
        node.evaluated = True
        node.gate_passed = outcome.valid_q
        node.gate_failures = tuple(outcome.gate_failures)
        return outcome

    # ---- backpropagation ---------------------------------------------------

    def backpropagate(self, node: TreeNode, outcome: SearchEvaluation) -> None:
        """Push the outcome up the path, updating every ancestor's Q.

        A valid Q updates every ancestor's mean.  An invalid one still counts a
        visit everywhere, because the design keeps such nodes counted and
        expandable -- but it never enters a mean, so it cannot lift a branch's
        ranking by failing.
        """

        current: TreeNode | None = node
        while current is not None:
            current.visits += 1
            if outcome.valid_q:
                current.q_sum += outcome.q_value
                current._q_sq_sum += outcome.q_value * outcome.q_value
                current.q_valid += 1
            current = self.nodes.get(current.parent) if current.parent else None

    # ---- the loop ----------------------------------------------------------

    def run(self, root: TreeNode, *, max_nodes: int = 200) -> SearchRunResult:
        """Run until the budget is exhausted; node growth is a safety rail.

        ``max_nodes`` bounds pathological growth (a node whose every action is
        already present still costs a traversal) without ever replacing the
        budget test: the loop exits on credits, and the rail only exists so a
        bug cannot spin forever.
        """

        self.seed_root(root)
        trace: list[Mapping[str, Any]] = []
        metrics_by_node: dict[str, Mapping[str, Any]] = {}
        while not self.ledger.exhausted and len(self.nodes) < max_nodes:
            node, freshly_expanded = self._descend_and_expand(root)
            # Only a node created in this iteration is simulated.  Re-entering an
            # existing node just to expand it one more level is a tree-building
            # step, and charging a credit for it would spend the budget on
            # bookkeeping instead of on evaluations -- with a twelve-credit
            # budget that difference is the difference between reaching a
            # two-factor configuration and never leaving the one-factor floor.
            if not freshly_expanded:
                if not node.children:
                    break
                continue
            if not node.factor_names:
                self.backpropagate(node, SearchEvaluation(0.0, False, ("no_factor_yet",), {}))
                continue
            outcome = self.simulate(node)
            self.backpropagate(node, outcome)
            if outcome.metrics:
                metrics_by_node[node.node_id] = dict(outcome.metrics)
            trace.append(
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "factors": list(node.factor_names),
                    "top_n": node.composition.top_n,
                    "weighting": node.composition.weighting,
                    "gate_layers": len(node.gating.layers),
                    "q": outcome.q_value,
                    "valid_q": outcome.valid_q,
                    "cached": outcome.cached,
                    "gate_failures": list(outcome.gate_failures),
                    "credits_spent": self.ledger.spent,
                    "phi": self.ledger.phi,
                }
            )
        best = self.best_node()
        return SearchRunResult(
            best=best,
            best_candidate=None if best is None else self._candidate(best, name="mcts_best"),
            nodes=sorted(self.nodes.values(), key=lambda item: item.node_id),
            ledger=self.ledger,
            trace=trace,
            evaluations=self.ledger.by_kind.get("backtest", 0),
            cache_hits=sum(1 for row in trace if row["cached"]),
            metrics_by_node=metrics_by_node,
        )

    def _descend_and_expand(self, root: TreeNode) -> tuple[TreeNode, bool]:
        """Walk down by PUCT and materialise one new child.

        Returns ``(node, is_freshly_expanded)``.  A freshly expanded node is one
        this iteration created; a node returned with ``False`` was already in the
        tree and is being re-entered for its own expansion next iteration.  The
        distinction matters because only the freshly created node is worth a
        credit -- see :meth:`run`.

        The expansion-motivation term gates the walk: a node whose ``E_gen`` has
        fallen below zero -- possible only once φ is small and the parent's mean
        Q is negative -- stops being expanded, so the search consolidates
        instead of opening new sub-branches with the last credits.
        """

        node = root
        while True:
            actions = self.actions.actions_for(node)
            child = self.expand(node, actions) if actions else None
            if child is not None:
                return child, True
            if not node.children:
                return node, False
            if self.expansion_motivation(node) < 0 and node.visits > 0:
                return node, False
            step = self.select_child(node)
            if step is None:
                return node, False
            node, _label = step

    def best_node(self) -> TreeNode | None:
        """Highest-Q node among those with a valid gate and a registrable factor set.

        Counting only valid-Q nodes is the design's "convergence ranks by Q and
        the hard screen still applies": a node that failed the boolean gate is
        not a candidate no matter how it scored.  ``_eligible_score`` adds the
        second screen -- the factor count the acceptance chain itself requires --
        so the reported winner is always something the convergence step could
        actually hand to ``StrategyAcceptanceService``.
        """

        candidates = [node for node in self.nodes.values() if node.has_valid_q and _eligible_score(node)]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.q_value, -item.depth, item.node_id))
