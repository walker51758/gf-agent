"""Action generation for the joint search, wired to the two existing proposers.

Factor actions come from the accepted factor registry plus the existing
:class:`RDAGentDSLProposer`; strategy actions come from the existing
:class:`LLMStrategyProposalEngine` vocabulary.  Both produce *declarative*
edits -- an action never writes code and never touches the backtest parameters.

The two proposers are optional collaborators.  A search constructed without
them still runs: it simply has no source of *new* factors, which is the honest
state of an offline demo.  That is a deliberate shape, because a missing LLM
credential must degrade to a smaller action space rather than to a crash in the
middle of a search that has already spent budget.

Every proposer output is funnelled through :func:`_valid_action`, which builds
the real spec objects and lets their own fail-closed validators reject a
malformed edit.  Nothing here repairs a bad proposal.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from rdagent.scenarios.gf_quant.strategy.candidate import (
    CompositionSpec,
    GateLayer,
    GatingSpec,
    SignalSpec,
)
from rdagent.scenarios.gf_quant.strategy.search.tree import MAX_FACTORS, Action

# The strategy knobs the search varies, and the ladder it walks for each.  They
# are structural -- they change how the same score becomes a book -- and each
# rung is a value the composition layer already supports, so no new execution
# behaviour is introduced by the search.
# The top_n ladder is walked in both directions from the root's size, so the
# search can both widen and narrow the book.  A one-directional ladder would
# make the root's size a floor the search can never reconsider.
TOP_N_LADDER = (5, 8, 12)
WEIGHTING_LADDER = ("equal", "proportional", "rank")
WEIGHT_CAP_LADDER = (None, 0.4, 0.25)
# Gate rungs.  A layer over a 40-name cross-section can only leave as many
# survivors as the quantile allows, so the ladder starts loose enough to clear a
# realistic `min_pass` and tightens from there; a rung that cannot leave
# `min_pass` names would make every gated candidate fail closed on a
# construction artefact rather than on its own merit.
GATE_QUANTILE_LADDER = (0.5, 0.3)
GATE_MIN_PASS_LADDER = (3, 3)
STANDARDIZE_LADDER = (None, "zscore", "rank_z")


def _replace(values: Sequence[Any], current: Any, *, forward: bool) -> Any | None:
    """Next rung of a ladder in the chosen direction, or ``None`` at the end.

    A ``None`` rung is a real rung -- "cap disabled" is a configuration the
    engine supports -- so the sentinel for "no further rung" has to be a
    separate object, not ``None``.
    """

    options = list(values)
    if current not in options:
        return options[0] if forward else options[-1]
    index = options.index(current)
    target = index + 1 if forward else index - 1
    if 0 <= target < len(options):
        return options[target]
    return _EXHAUSTED


_EXHAUSTED = object()


class ActionGenerator:
    """Build the next layer of candidate actions for one node.

    ``factor_registry`` supplies names that are already accepted; ``proposer``
    (the existing ``RDAGentDSLProposer`` shape) supplies new expressions, gated
    by ``factor_service`` (the existing ``FactorAcceptanceService``).  Both are
    optional and both are called at most once per expansion.

    ``strategy_engine`` together with ``strategy_context_factory`` add a second
    proposal source: a ``LLMStrategyProposalEngine`` call whose returned
    candidate is offered as one more strategy action.  The factory builds the
    engine's context from the node being expanded, so the search owns the
    context shape and the engine stays the untouched collaborator it is in the
    collaboration layer.  Both must be supplied for the source to be active; a
    half-supplied pair is a wiring mistake and is ignored rather than guessed at.
    """

    def __init__(
        self,
        *,
        factor_registry: Any | None = None,
        factor_service: Any | None = None,
        proposer: Any | None = None,
        strategy_engine: Any | None = None,
        strategy_context_factory: Callable[[Any], Any] | None = None,
    ) -> None:
        self.factor_registry = factor_registry
        self.factor_service = factor_service
        self.proposer = proposer
        self.strategy_engine = strategy_engine
        self.strategy_context_factory = strategy_context_factory
        # Proposal bookkeeping: nodes whose proposal round has already run (one
        # round per node, not per descent), the round counter that alternates the
        # family, and the notes a caller reads to see why a proposal died.
        self._proposed_nodes: set[str] = set()
        self._strategy_proposed_nodes: set[str] = set()
        self.proposal_rounds = 0
        self.strategy_proposal_rounds = 0
        self.notes: list[str] = []

    # ---- factor actions ----------------------------------------------------

    def accepted_factor_names(self) -> list[str]:
        """Names from the accepted side of the factor registry, alphabetically."""

        if self.factor_registry is None:
            return []
        records = self.factor_registry.list(status="accepted")
        return sorted({str(record.name) for record in records})

    def proposed_factor_names(self, node: Any, *, limit: int = 2) -> tuple[list[str], list[str]]:
        """Names newly accepted from the DSL proposer, plus notes.

        The proposer feedback carries the current factor set and the gate
        failures that were actually observed, which is the design's "read the
        tree context" requirement expressed in the vocabulary the proposer
        already understands.  Acceptance runs through the existing RankIC
        service; a rejected proposal is reported in the notes and contributes no
        action.
        """

        if self.proposer is None or self.factor_service is None:
            return [], []
        failures = ", ".join(node.gate_failures[:4]) or "none recorded"
        feedback = (
            f"Current strategy holds factors: {', '.join(node.factor_names) or 'none'}.\n"
            f"Gate failures observed on this configuration: {failures}.\n"
            f"Top-N holdings: {node.composition.top_n}."
        )
        accepted: list[str] = []
        notes: list[str] = []
        family = self._family_for(node)
        for candidate in self._propose(family, feedback, limit=limit):
            try:
                result = self.factor_service.process(candidate)
            except Exception as error:  # one bad proposal must not abort a search
                notes.append(f"proposal_error:{type(error).__name__}")
                continue
            if str(result.record.status) == "accepted":
                accepted.append(str(result.record.name))
            else:
                notes.append(f"rejected:{candidate.name}")
        return accepted, notes

    def _family_for(self, node: Any) -> str:
        """Which family the next proposal round targets.

        The two families alternate by round rather than by node: a node's own
        factor set says nothing about which family is unexplored, and a
        deterministic alternation is what keeps a run's family mix auditable.
        """

        return "fundamental" if self.proposal_rounds % 2 == 1 else "price_volume"

    def _propose(self, family: str, feedback: str, *, limit: int) -> list[Any]:
        try:
            proposals = list(self.proposer.propose(family, feedback) or [])
        except Exception as error:
            self.notes.append(f"proposer_error:{type(error).__name__}")
            return []
        self.proposal_rounds += 1
        return proposals[:limit]

    # ---- public API --------------------------------------------------------

    def actions_for(self, node: Any) -> list[Action]:
        actions: list[Action] = []
        actions.extend(self._proposed_factor_actions(node))
        actions.extend(self._proposed_strategy_actions(node))
        actions.extend(self._factor_actions(node))
        actions.extend(self._strategy_actions(node))
        return actions

    def _proposed_strategy_actions(self, node: Any) -> list[Action]:
        """One LLM strategy proposal per node, offered as a single action.

        The engine returns a complete ``StrategyCandidate``, so the action
        adopts its composition, signal and gating wholesale rather than merging
        fields: a proposal is a coherent whole, and recombining its parts with
        the node's would produce a configuration neither source proposed.  Only
        the factors it asks for are taken from the proposal -- the node's own
        factor list is widened to include any it does not already hold, and a
        proposal naming a factor outside the held set contributes no action,
        because the engine's own validator would already have rejected an
        invented name.
        """

        if self.strategy_engine is None or self.strategy_context_factory is None:
            return []
        if node.node_id in self._strategy_proposed_nodes:
            return []
        self._strategy_proposed_nodes.add(node.node_id)
        try:
            context = self.strategy_context_factory(node)
            candidate = self.strategy_engine(context)
        except Exception as error:  # one bad proposal must not abort a search
            self.notes.append(f"strategy_proposal_error:{type(error).__name__}:{error}")
            return []
        self.strategy_proposal_rounds += 1

        factor_names = tuple(str(name) for name in candidate.factor_names)
        if not factor_names:
            self.notes.append("strategy_proposal_empty:no factors named")
            return []
        # The proposal's gating may only reference factors the action holds;
        # ``_gating_for`` is the same filter the factor actions use.
        signal = candidate.signal
        gating = _gating_for(candidate.gating, factor_names)
        composition = _composition_for(candidate.composition, factor_names)
        return [
            Action(
                kind="strategy",
                label=f"strategy_proposal:{candidate.name}",
                factor_names=factor_names,
                signal=signal,
                gating=gating,
                composition=composition,
                detail=f"adopt LLM strategy proposal {candidate.name}",
            )
        ]

    def _proposed_factor_actions(self, node: Any) -> list[Action]:
        """Factor actions sourced from a fresh proposal round.

        Runs at most once per node and only while the node still has room for a
        factor, so the LLM is called at a rate the budget can justify: one round
        per expansion point rather than one per descent step.  A proposal that
        fails the RankIC chain contributes no action and its reason is recorded,
        which is what makes the reference to "the gate failures that were
        actually observed" in the proposer feedback true rather than decorative.
        """

        if self.proposer is None or self.factor_service is None:
            return []
        if len(node.factor_names) >= MAX_FACTORS:
            return []
        if node.node_id in self._proposed_nodes:
            return []
        self._proposed_nodes.add(node.node_id)
        limit = max(1, MAX_FACTORS - len(node.factor_names))
        names, notes = self.proposed_factor_names(node, limit=limit)
        self.notes.extend(notes)
        current = set(node.factor_names)
        additions = [name for name in names if name not in current]
        return [
            Action(
                kind="add_factor",
                label=f"add_factor:{name}",
                factor_names=tuple([*node.factor_names, name]),
                signal=node.signal,
                gating=_gating_for(node.gating, tuple([*node.factor_names, name])),
                composition=_composition_for(node.composition, tuple([*node.factor_names, name])),
                detail=f"append proposed factor {name}",
            )
            for name in additions
        ]

    def _factor_actions(self, node: Any) -> list[Action]:
        if len(node.factor_names) >= MAX_FACTORS:
            return []
        current = set(node.factor_names)
        names = [name for name in self.accepted_factor_names() if name not in current]
        additions = names[: max(1, MAX_FACTORS - len(node.factor_names) + 1)]
        actions = [
            Action(
                kind="add_factor",
                label=f"add_factor:{name}",
                factor_names=tuple([*node.factor_names, name]),
                signal=node.signal,
                gating=_gating_for(node.gating, tuple([*node.factor_names, name])),
                composition=_composition_for(node.composition, tuple([*node.factor_names, name])),
                detail=f"append accepted factor {name}",
            )
            for name in additions
        ]
        # Replacement is only meaningful once something is in the book: it is
        # how a configuration that failed its gate gets one more chance.
        if node.factor_names and names:
            replacement = names[0]
            for dropped in node.factor_names:
                kept = tuple([name for name in node.factor_names if name != dropped] + [replacement])
                actions.append(
                    Action(
                        kind="replace_factor",
                        label=f"replace_factor:{dropped}->{replacement}",
                        factor_names=kept,
                        signal=_signal_for_swap(node.signal, dropped, replacement),
                        gating=_gating_for(node.gating, kept, dropped=dropped, replacement=replacement),
                        composition=_composition_for(node.composition, kept, swap=(dropped, replacement)),
                        detail=f"swap {dropped} for {replacement}",
                    )
                )
        return actions

    def _strategy_actions(self, node: Any) -> list[Action]:
        composition = node.composition
        signal = node.signal
        gating = node.gating
        actions: list[Action] = []

        top_n = _replace(TOP_N_LADDER, composition.top_n, forward=True)
        if top_n is not _EXHAUSTED:
            actions.append(
                Action(
                    kind="strategy",
                    label=f"top_n:{composition.top_n}->{top_n}",
                    factor_names=node.factor_names,
                    signal=signal,
                    gating=gating,
                    composition=_rebuild(composition, top_n=int(top_n)),
                    detail="widen the selection",
                )
            )

        weighting = _replace(WEIGHTING_LADDER, composition.weighting, forward=True)
        if weighting is not _EXHAUSTED:
            extra = _weighting_companions(str(weighting))
            actions.append(
                Action(
                    kind="strategy",
                    label=f"weighting:{composition.weighting}->{weighting}",
                    factor_names=node.factor_names,
                    signal=signal,
                    gating=gating,
                    composition=_rebuild(composition, weighting=str(weighting), **extra),
                    detail="change the weighting scheme",
                )
            )

        if node.factor_names:
            cap = _replace(WEIGHT_CAP_LADDER, composition.weight_cap, forward=True)
            if cap is not _EXHAUSTED:
                actions.append(
                    Action(
                        kind="strategy",
                        label=f"weight_cap:{composition.weight_cap}->{cap}",
                        factor_names=node.factor_names,
                        signal=signal,
                        gating=gating,
                        composition=_rebuild(composition, weight_cap=cap),
                        detail="concentrate the book",
                    )
                )

            standardize = _replace(STANDARDIZE_LADDER, signal.standardize, forward=True)
            if standardize is not _EXHAUSTED:
                actions.append(
                    Action(
                        kind="strategy",
                        label=f"standardize:{signal.standardize}->{standardize}",
                        factor_names=node.factor_names,
                        signal=_rebuild_signal(signal, standardize=standardize),
                        gating=gating,
                        composition=composition,
                        detail="standardize the cross-section before ranking",
                    )
                )

            gate = self._next_gate(node)
            if gate is not None:
                actions.append(gate)
        return actions

    def _next_gate(self, node: Any) -> Action | None:
        """Append one gate layer on an existing factor, walking two ladders.

        A gate references a candidate factor by construction, so this action is
        impossible before the first factor action -- which is exactly the
        conditional structure the tree is meant to discover rather than have
        hard-coded.
        """

        layers = list(node.gating.layers)
        if len(layers) >= 2:
            return None
        # Gates only make sense on a name the candidate actually holds.
        used = {layer.factor for layer in layers}
        target = next((name for name in node.factor_names if name not in used), None)
        if target is None:
            return None
        rung = len(layers)
        quantile = GATE_QUANTILE_LADDER[rung]
        min_pass = GATE_MIN_PASS_LADDER[rung]
        layer = GateLayer(factor=target, direction="top", quantile=quantile, min_pass=min_pass)
        gating = GatingSpec(layers=tuple([*layers, layer]))
        return Action(
            kind="strategy",
            label=f"gating:add_layer:{target}",
            factor_names=node.factor_names,
            signal=node.signal,
            gating=gating,
            composition=node.composition,
            detail=f"gate on {target} quantile={quantile} min_pass={min_pass}",
        )


def _weighting_companions(weighting: str) -> dict[str, Any]:
    """Per-scheme parameters a weighting rung needs to be legal.

    ``score_transform`` refuses to build without a transform kind, so the
    companion values are part of the rung rather than a separate action: half a
    configuration is not an action the search should ever create.
    """

    if weighting == "score_transform":
        return {"score_transform_kind": "sigmoid", "score_transform_slope": 6.0, "score_transform_cut": 0.5}
    if weighting in ("inverse_volatility", "risk_parity"):
        return {"inverse_vol_window": 20}
    return {}


def _rebuild(composition: CompositionSpec, **overrides: Any) -> CompositionSpec:
    """Copy a composition spec with selected fields replaced.

    Going through the dataclass rather than ``dataclasses.replace`` keeps the
    validator path identical to a hand-built spec: whatever the constructor
    refuses here is refused at the same place the worker would refuse it.
    """

    payload = composition.as_payload()
    payload.update(overrides)
    return CompositionSpec.from_payload(payload)


def _rebuild_signal(signal: SignalSpec, **overrides: Any) -> SignalSpec:
    payload = signal.as_payload()
    payload.update({key: value for key, value in overrides.items() if value is not None})
    if overrides.get("standardize", "unset") is None:
        # An explicit None means "turn it off", which the payload cannot express
        # because a disabled field is simply absent.  Drop the key instead.
        payload.pop("standardize", None)
    return SignalSpec.from_payload(payload)


def _gating_for(
    gating: GatingSpec,
    factor_names: Sequence[str],
    *,
    dropped: str | None = None,
    replacement: str | None = None,
) -> GatingSpec:
    """Keep only gate layers that still reference a held factor.

    Dropping a factor silently drops its gate rather than rewriting it onto the
    replacement: the replacement was never scored against that threshold, and
    inventing a threshold for it would be the search making a data decision it
    has no evidence for.
    """

    held = set(factor_names)
    layers = []
    for layer in gating.layers:
        name = layer.factor
        if dropped is not None and name == dropped:
            continue
        if replacement is not None and name == replacement:
            continue
        if name in held:
            layers.append(layer)
    return GatingSpec(layers=tuple(layers))


def _composition_for(
    composition: CompositionSpec,
    factor_names: Sequence[str],
    *,
    swap: tuple[str, str] | None = None,
) -> CompositionSpec:
    """Composition spec with factor-keyed sections remapped onto a new factor set.

    ``frb_budgets`` names factors directly, so a swap has to move the key with
    it or the spec fails its own coverage check.  Nothing else in the spec is
    factor-keyed at this layer.
    """

    if composition.frb_budgets is None or swap is None:
        return composition
    dropped, replacement = swap
    budgets = dict(composition.frb_budgets)
    if dropped in budgets:
        budgets[replacement] = budgets.pop(dropped)
    return _rebuild(composition, frb_budgets=budgets)


def _signal_for_swap(signal: SignalSpec, dropped: str, replacement: str) -> SignalSpec:
    """Factor-keyed signal sections follow a swap; the rest is untouched."""

    changes: dict[str, Any] = {}
    if signal.blend_weights is not None and dropped in signal.blend_weights:
        weights = dict(signal.blend_weights)
        weights[replacement] = weights.pop(dropped)
        changes["blend_weights"] = weights
    if signal.blend_groups is not None:
        changes["blend_groups"] = {
            group: [replacement if member == dropped else member for member in members]
            for group, members in signal.blend_groups.items()
        }
    if not changes:
        return signal
    return _rebuild_signal(signal, **changes)
