"""Switchable candidate evaluator for the search loop.

Two backends sit behind one protocol:

* **local** runs the real backtest engine against the cached panels and returns
  Q plus the gate verdict.  It is the only path the first version takes, because
  the data machine is unreachable and a search that cannot be evaluated cannot
  be demonstrated.
* **remote** is a stub.  It exists so the switch point is explicit and the call
  site does not have to be rewritten when the data machine returns; it refuses
  with a clear message rather than pretending to evaluate.

The local path reads Q from the engine's own excess-return block
(``backtest._relative_performance`` -> ``excess_annualized_return``) and checks
the existing boolean strategy gate through the unchanged
``StrategyAcceptancePolicy``.  That split is the design's: Q measures *how much
better*, the gate decides *whether it may be wanted at all*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from rdagent.scenarios.gf_quant.strategy.backtest import run_portfolio_backtest
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.composition import build_target_weights
from rdagent.scenarios.gf_quant.strategy.gating import build_gated_factor_score
from rdagent.scenarios.gf_quant.strategy.search.data_cache import SearchDataCache
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyEvaluation,
)

LOCAL_BACKEND = "local"
REMOTE_BACKEND = "remote"


@dataclass(frozen=True)
class SearchEvaluation:
    """One candidate's outcome: a scalar Q, a gate verdict, and the trace.

    ``valid_q`` is ``False`` when the node failed the boolean gate.  The node's
    Q must then not enter any mean or ranking, which is enforced by the tree's
    ``has_valid_q`` rather than by leaving the value out -- a missing value
    would be indistinguishable from an unvisited node.
    """

    q_value: float
    valid_q: bool
    gate_failures: tuple[str, ...]
    metrics: Mapping[str, Any]
    cached: bool = False


class LocalBacktestEvaluator:
    """Run the real engine on the cached panels.

    The panels are loaded once per evaluator and held for the run: a search
    evaluates a dozen candidates over the same window, and re-reading the
    parquet on every call would dominate the wall clock while adding nothing.
    """

    def __init__(
        self,
        *,
        cache: SearchDataCache,
        evaluation_start: str,
        evaluation_end: str,
        backtest_spec: BacktestSpec | None = None,
        rebalance_frequency: str = "monthly",
        policy: StrategyAcceptancePolicy | None = None,
    ) -> None:
        self.cache = cache
        self.evaluation_start = str(evaluation_start)
        self.evaluation_end = str(evaluation_end)
        # The runtime-level execution assumptions.  They are a constant of the
        # run, never a node field: two nodes must differ in what they select,
        # not in how the same selection is executed, or their Qs are not
        # comparable.
        self.backtest_spec = backtest_spec or BacktestSpec()
        if rebalance_frequency not in ("daily", "weekly", "monthly", "quarterly"):
            raise ValueError(f"unsupported rebalance_frequency: {rebalance_frequency!r}")
        self.rebalance_frequency = rebalance_frequency
        self.policy = policy or StrategyAcceptancePolicy()
        close = self.cache.close_wide()
        window = self._window(close)
        self.returns = close.loc[window].pct_change(fill_method=None).iloc[1:]
        self.benchmark = self.returns.mean(axis=1, skipna=True)
        self.rebalance_dates = self._rebalance_dates(self.returns.index)
        self.factor_panels: dict[str, pd.DataFrame] = {}
        self.factor_directions: dict[str, int] = {}

    def _window(self, close: pd.DataFrame) -> pd.DatetimeIndex:
        window = close.loc[pd.Timestamp(self.evaluation_start) : pd.Timestamp(self.evaluation_end)].index
        if window.empty:
            raise ValueError(
                f"search cache does not cover {self.evaluation_start}..{self.evaluation_end}"
            )
        return window

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if self.rebalance_frequency == "daily":
            return index
        periods = index.to_period({"weekly": "W", "monthly": "M", "quarterly": "Q"}[self.rebalance_frequency])
        selected = pd.Series(index, index=index).groupby(periods).last().to_list()
        return pd.DatetimeIndex(sorted(set([index[0], *selected])))

    def usable_rebalance_dates(self) -> pd.DatetimeIndex:
        """Rebalance dates on which every registered factor has a cross-section.

        A rolling factor needs its window before it can rank anything, so the
        first rebalance date of a two-year window has no valid value for a
        20-day factor.  Leaving that date in the calendar does not degrade
        gracefully: a gate layer over an empty cross-section has zero survivors,
        so it fails closed on a warm-up artefact and every gated candidate is
        refused for a reason that has nothing to do with its configuration.

        The fix is to drop such dates from the calendar, never to fabricate a
        factor value for them -- a filled warm-up window would be invented data
        driving a real position.
        """

        if not self.factor_panels:
            return self.rebalance_dates
        keep = pd.Series(True, index=self.rebalance_dates)
        for panel in self.factor_panels.values():
            aligned = panel.reindex(self.rebalance_dates)
            keep &= aligned.notna().any(axis=1)
        usable = self.rebalance_dates[keep.to_numpy()]
        if len(usable) < 2:
            raise ValueError(
                "fewer than two rebalance dates carry a cross-section for every registered factor; "
                "widen the cached window or drop the longest-window factor"
            )
        return usable

    def register_factor(self, name: str, panel: pd.DataFrame, direction: int) -> None:
        """Make one factor panel available to the engine.

        Directions are frozen by the factor stage and read here rather than
        re-derived: re-deriving a sign from the same data the score is built
        from would let the search flip a factor to make its own number look
        better.
        """

        if direction not in (1, -1):
            raise ValueError("factor direction must be +1 or -1")
        self.factor_panels[name] = panel.reindex(index=self.rebalance_dates).astype(float)
        self.factor_directions[name] = int(direction)

    def __call__(self, candidate: StrategyCandidate) -> StrategyEvaluation:
        missing = [name for name in candidate.factor_names if name not in self.factor_panels]
        if missing:
            raise ValueError(f"factors have no registered panel: {missing[:3]}")
        # The decision calendar is restricted to dates on which *every
        # registered* factor has a cross-section, not just the candidate's own:
        # a shared calendar is what makes two candidates' metrics comparable, and
        # a per-candidate calendar would let one node be scored over more days
        # than another for reasons unrelated to its configuration.
        dates = self.usable_rebalance_dates()
        score = build_gated_factor_score(
            {name: self.factor_panels[name] for name in candidate.factor_names},
            {name: self.factor_directions[name] for name in candidate.factor_names},
            candidate.signal,
            candidate.gating,
            index=dates,
            columns=self.returns.columns,
            returns=self.returns,
            lag=int(candidate.backtest.signal_lag),
        )
        targets = build_target_weights(
            candidate.composition,
            score,
            returns=self.returns,
        )
        composition = candidate.composition
        selection_operators = (
            composition.holding_buffer_entry is not None
            or composition.diversity_max_correlation is not None
        )
        result = run_portfolio_backtest(
            targets,
            self.returns,
            spec=self.backtest_spec,
            benchmark_returns=self.benchmark,
            composition=composition if selection_operators else None,
            score=score if selection_operators else None,
        )
        metrics = {
            **dict(result.metrics),
            "metrics_by_year": dict(result.metrics_by_year),
        }
        return StrategyEvaluation.from_result(result, metrics=metrics)


def _invalid(reason: str) -> SearchEvaluation:
    return SearchEvaluation(
        q_value=0.0,
        valid_q=False,
        gate_failures=(reason,),
        metrics={},
    )


def evaluate_candidate(
    candidate: StrategyCandidate,
    *,
    evaluator: LocalBacktestEvaluator,
    policy: StrategyAcceptancePolicy,
) -> SearchEvaluation:
    """Evaluate one candidate, converting any engine refusal into an invalid Q.

    A configuration the engine refuses is not a crash: it is a node whose gate
    failed, and the design is explicit that such a node stays counted and
    expandable.  The refusal reason is preserved verbatim as the gate failure so
    the reason a branch died is visible in the sidecar rather than lost.
    """

    try:
        evaluation = evaluator(candidate)
    except (ValueError, KeyError, ZeroDivisionError) as error:
        return _invalid(f"{type(error).__name__}: {error}")
    passed, failures = policy.check(dict(evaluation.metrics), candidate)
    q = evaluation.metrics.get("excess_annualized_return")
    try:
        q_value = float(q)
    except (TypeError, ValueError):
        return SearchEvaluation(
            q_value=0.0,
            valid_q=False,
            gate_failures=tuple(failures) + ("excess_annualized_return_missing",),
            metrics=dict(evaluation.metrics),
        )
    if not passed:
        return SearchEvaluation(
            q_value=q_value,
            valid_q=False,
            gate_failures=tuple(failures),
            metrics=dict(evaluation.metrics),
        )
    return SearchEvaluation(
        q_value=q_value,
        valid_q=True,
        gate_failures=(),
        metrics=dict(evaluation.metrics),
    )


class RemoteSearchEvaluator:
    """Evaluate a candidate on the data machine through the existing protocol.

    The strategy side reuses ``strategy.remote.RemoteStrategyEvaluator``
    unchanged: it already builds the schema-v1 request, uploads the worker,
    executes it over the pinned SSH connection and verifies the result's request
    checksum.  What this class adds is the mapping from the search's runtime-level
    configuration onto that evaluator's constructor, so a run's evaluation window
    and execution assumptions travel as one constant exactly as they do locally.

    It also refuses to pretend.  Connection details come from
    :class:`RemoteSettings` (environment or ``GF_SCENARIO_SETTINGS``); nothing is
    hard-coded, and an unreachable machine raises a message naming what failed
    rather than degrading to an invented Q.
    """

    def __init__(
        self,
        *,
        snapshot: Any,
        working_set: Any,
        artifact_set: Any,
        evaluation_start: str,
        evaluation_end: str,
        settings: Any = None,
        rebalance_frequency: str = "monthly",
        remote_evaluate: Callable[..., dict[str, Any]] | None = None,
        worker_path: Path | None = None,
    ) -> None:
        from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
        from rdagent.scenarios.gf_quant.strategy.remote import (
            RemoteStrategyEvaluator,
            evaluate_strategy_remote,
        )

        self._delegate = RemoteStrategyEvaluator(
            snapshot=snapshot,
            working_set=working_set,
            artifact_set=artifact_set,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            settings=settings or GF_SCENARIO_SETTINGS,
            rebalance_frequency=rebalance_frequency,
            remote_evaluate=remote_evaluate or evaluate_strategy_remote,
            worker_path=worker_path,
        )

    def request_for(self, candidate: StrategyCandidate, **kwargs: Any) -> dict[str, Any]:
        return self._delegate.request_for(candidate, **kwargs)

    def manifest_facts(self) -> dict[str, Any]:
        return self._delegate.manifest_facts()

    def __call__(self, candidate: StrategyCandidate) -> StrategyEvaluation:
        return self._delegate(candidate)


def remote_evaluator_unavailable(*_args: Any, **_kwargs: Any) -> SearchEvaluation:
    """The remote backend with no data machine to talk to.

    Kept as an explicit, named refusal rather than the previous
    ``remote_evaluator_stub``: the class above is the implementation, and this is
    only what :func:`build_evaluator` returns when the caller asks for the remote
    backend without supplying the snapshot/working-set/artifact triple the remote
    protocol requires.  Refusing here is what keeps "remote selected but
    unconfigured" from silently becoming "local instead".
    """

    raise RuntimeError(
        "the remote search evaluator was selected but no search context "
        "(snapshot / working set / artifact set) was provided.  The data machine path needs "
        "those three from the factor stage; build it with RemoteSearchEvaluator directly, or "
        "run with backend='local' against the cached panels."
    )


def build_evaluator(
    backend: str,
    *,
    cache: SearchDataCache | None = None,
    evaluation_start: str,
    evaluation_end: str,
    backtest_spec: BacktestSpec | None = None,
    rebalance_frequency: str = "monthly",
    policy: StrategyAcceptancePolicy | None = None,
    factor_evaluator: LocalBacktestEvaluator | None = None,
    remote: Callable[[StrategyCandidate], StrategyEvaluation] | None = None,
) -> Callable[[StrategyCandidate], SearchEvaluation]:
    """The one switch point between the two evaluation backends.

    ``factor_evaluator`` lets the caller pass the engine it already built, so one
    run holds exactly one loaded panel set instead of two.  ``remote`` is the
    matching injection for the data-machine path: a caller that has assembled a
    :class:`RemoteSearchEvaluator` passes its ``__call__`` here, and the switch
    becomes a one-line change at the run level rather than a rewrite of the loop.
    """

    if backend == LOCAL_BACKEND:
        if cache is None:
            raise ValueError("the local backend requires a search cache")
        local = factor_evaluator or LocalBacktestEvaluator(
            cache=cache,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            backtest_spec=backtest_spec,
            rebalance_frequency=rebalance_frequency,
            policy=policy,
        )

        def evaluate(candidate: StrategyCandidate) -> SearchEvaluation:
            return evaluate_candidate(candidate, evaluator=local, policy=local.policy)

        evaluate.local = local  # type: ignore[attr-defined]
        return evaluate
    if backend == REMOTE_BACKEND:
        return remote or remote_evaluator_unavailable
    raise ValueError(f"unsupported search evaluation backend: {backend!r}")
