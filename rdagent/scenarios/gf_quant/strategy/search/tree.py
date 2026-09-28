"""Joint factor/strategy action tree for the MCTS search loop.

A node is a complete *configuration* -- a factor set plus the strategy
combination structure -- and never a backtest parameter: the evaluation window,
the cost rates and the execution assumptions are one runtime-level constant so
every node is comparable by construction (design doc, "回测参数的全局统一").

Two identities travel together and mean different things:

* **node identity** is ``(parent path, action)`` -- where the configuration sits
  in the tree.  The same configuration reached two ways keeps two separate sets
  of statistics, which is what the design asks for.
* **configuration hash** is the canonical payload hash of the configuration
  itself, reusing the existing ``strategy_id`` mechanism.  It keys the
  evaluation cache, so re-reaching a configuration costs no budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rdagent.scenarios.gf_quant.strategy.candidate import (
    BacktestSpec,
    CompositionSpec,
    GatingSpec,
    SignalSpec,
    StrategyCandidate,
)

# Actions that add a factor are bounded so a search cannot walk past the
# worker's own 2-8 factor whitelist; the bound is a property of the action
# space, not of any one candidate.
MIN_FACTORS = 2
MAX_FACTORS = 6

# Strategy actions carried in the root configuration's summary so the demo can
# show what was searched without re-reading every node.
DEFAULT_UNIVERSE = "csi_all"


def configuration_hash(
    factor_names: Sequence[str],
    *,
    signal: SignalSpec,
    gating: GatingSpec,
    composition: CompositionSpec,
    universe: str = DEFAULT_UNIVERSE,
) -> str:
    """Canonical hash of a node configuration.

    Built through :meth:`StrategyCandidate.as_payload` rather than from the raw
    specs so it inherits the established convention: a field that is disabled
    stays out of the payload and cannot move the hash.  This is what lets a
    configuration reached before and after a new operator was added keep one
    cached evaluation.
    """

    # The backtest section is part of the payload but is a runtime constant for
    # every node in one search, so it contributes the same bytes everywhere and
    # cannot discriminate; it is still included because the payload builder
    # owns the shape and hand-stripping it would fork the format.
    candidate = StrategyCandidate(
        name="search_node",
        universe=universe,
        composition=composition,
        factor_names=tuple(factor_names),
        rule="cross-sectional rank blend",
        backtest=BacktestSpec(),
        signal=signal,
        gating=gating,
    )
    return hashlib.sha256(candidate.canonical_payload().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Action:
    """One edge of the tree: a single factor or strategy edit."""

    kind: str  # "add_factor" | "remove_factor" | "replace_factor" | "strategy"
    label: str
    factor_names: tuple[str, ...]
    signal: SignalSpec
    gating: GatingSpec
    composition: CompositionSpec
    prior: float = 1.0
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("add_factor", "remove_factor", "replace_factor", "strategy"):
            raise ValueError(f"unsupported action kind: {self.kind!r}")
        if not self.label:
            raise ValueError("action requires a label")
        if not (self.prior > 0):
            raise ValueError("action prior must be positive")


@dataclass
class TreeNode:
    """One configuration plus its accumulated statistics.

    ``q_value`` is the *mean* of every valid Q that ever flowed back through
    this node, and ``q_valid`` counts those updates.  A node whose gate failed
    has ``gate_passed=False`` and therefore a ``q_value`` that no selection or
    ranking step may read -- but its ``visits`` and ``children`` keep working,
    because the design keeps such a node expandable: one more action can replace
    the factor that caused the failure.
    """

    node_id: str
    path: tuple[str, ...]
    factor_names: tuple[str, ...]
    signal: SignalSpec
    gating: GatingSpec
    composition: CompositionSpec
    config_hash: str
    depth: int = 0
    visits: int = 0
    q_sum: float = 0.0
    # Second moment of the valid Qs, carried alongside the sum so the spread the
    # expansion-motivation term reads survives a sidecar round-trip.  Only valid
    # Qs contribute, exactly like ``q_sum``.
    _q_sq_sum: float = 0.0
    q_valid: int = 0
    gate_passed: bool = False
    gate_failures: tuple[str, ...] = ()
    evaluated: bool = False
    parent: str | None = None
    children: dict[str, str] = field(default_factory=dict)
    child_labels: dict[str, str] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Mean valid Q, or 0.0 when the node has no valid Q yet.

        The zero is a *selection* fallback only: `has_valid_q` gates every place
        that ranks nodes, so an unexpanded or gated node can never win a
        ranking on this default.
        """

        return self.q_sum / self.q_valid if self.q_valid else 0.0

    @property
    def has_valid_q(self) -> bool:
        return self.q_valid > 0 and self.gate_passed

    def q_variance(self) -> float:
        """Running variance proxy used by the expansion-motivation term.

        The tree stores the sum and the count rather than every draw, so the
        spread is estimated from the second moment.  With fewer than two draws
        the spread is unknown and reported as zero, which switches the
        exploration half of ``E_gen`` off for that parent.
        """

        if self.q_valid < 2:
            return 0.0
        mean = self.q_sum / self.q_valid
        # Clamped at zero because the second moment is accumulated incrementally
        # and can drift a hair below the squared mean through rounding; a tiny
        # negative variance would become a NaN under the square root the caller
        # takes.
        return max(self._q_sq_sum / self.q_valid - mean * mean, 0.0)

    def as_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent": self.parent,
            "path": list(self.path),
            "depth": self.depth,
            "factor_names": list(self.factor_names),
            "config_hash": self.config_hash,
            "signal": self.signal.as_payload(),
            "gating": self.gating.as_payload(),
            "composition": self.composition.as_payload(),
            "visits": self.visits,
            "q_sum": self.q_sum,
            "q_sq_sum": self._q_sq_sum,
            "q_valid": self.q_valid,
            "gate_passed": self.gate_passed,
            "gate_failures": list(self.gate_failures),
            "evaluated": self.evaluated,
            "children": dict(self.children),
            "child_labels": dict(self.child_labels),
        }


def tree_node_from_payload(payload: Mapping[str, Any]) -> TreeNode:
    """Rebuild a node from its sidecar record.

    Disabled sections restore to their default objects through the specs' own
    ``from_payload`` contract, so a sidecar written before a new operator
    existed still loads.
    """

    node = TreeNode(
        node_id=str(payload["node_id"]),
        path=tuple(str(value) for value in payload["path"]),
        factor_names=tuple(str(value) for value in payload["factor_names"]),
        signal=SignalSpec.from_payload(dict(payload["signal"])),
        gating=GatingSpec.from_payload(payload.get("gating")),
        composition=CompositionSpec.from_payload(dict(payload["composition"])),
        config_hash=str(payload["config_hash"]),
        depth=int(payload.get("depth", 0)),
        visits=int(payload.get("visits", 0)),
        q_sum=float(payload.get("q_sum", 0.0)),
        q_valid=int(payload.get("q_valid", 0)),
        gate_passed=bool(payload.get("gate_passed", False)),
        gate_failures=tuple(str(value) for value in payload.get("gate_failures", ())),
        evaluated=bool(payload.get("evaluated", False)),
        parent=None if payload.get("parent") is None else str(payload["parent"]),
        children={str(key): str(value) for key, value in (payload.get("children") or {}).items()},
        child_labels={str(key): str(value) for key, value in (payload.get("child_labels") or {}).items()},
    )
    node._q_sq_sum = float(payload.get("q_sq_sum", 0.0))
    return node


# The root's selection size.  The design says "empty factor set + default
# strategy structure", and this is that structure with one value chosen rather
# than inherited: the composition default of 5 holds fewer names than the
# acceptance policy's `min_avg_holdings`, so a search starting there would spend
# its first credits proving a book size is too small -- a fact about the gate,
# not about any factor.  Starting at a size the policy can accept means every
# credit buys information about the factor set, which is what the tree searches.
ROOT_TOP_N = 8


def root_node() -> TreeNode:
    """The empty-factor root the design starts every run from."""

    signal = SignalSpec()
    gating = GatingSpec()
    composition = CompositionSpec(top_n=ROOT_TOP_N)
    return TreeNode(
        node_id="root",
        path=(),
        factor_names=(),
        signal=signal,
        gating=gating,
        composition=composition,
        config_hash=configuration_hash((), signal=signal, gating=gating, composition=composition),
        depth=0,
    )


def branch_root_node(node_id: str) -> TreeNode:
    """The empty-factor root of a branch, at depth 1.

    The search explores two branches from the single empty-factor root: one that
    grows the strategy structure first and one that grows the factor set first.
    Both are "the root" in the sense of holding no factors yet, but only the
    branch root at depth 1 is a real candidate the engine can evaluate -- the
    depth-0 root has no factor and the engine requires at least one.  Keeping
    them as separate nodes is what lets the search spend its first credits on a
    comparable pair instead of a guaranteed rejection.
    """

    signal = SignalSpec()
    gating = GatingSpec()
    composition = CompositionSpec(top_n=ROOT_TOP_N)
    return TreeNode(
        node_id=node_id,
        path=(node_id,),
        factor_names=(),
        signal=signal,
        gating=gating,
        composition=composition,
        config_hash=configuration_hash((), signal=signal, gating=gating, composition=composition),
        depth=1,
    )
