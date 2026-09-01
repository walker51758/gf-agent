"""Double-timescale orchestration for factor-to-portfolio research.

The inner loop proposes and evaluates portfolio strategies against one frozen
factor working set.  Only after those trials finish does the outer loop emit
factor demands and invoke the existing factor-research capability.  Any factors
created by those calls remain in :class:`FactorRegistry` but cannot enter the
already-frozen snapshot; they become visible in the next epoch.

This module deliberately depends on small callable protocols.  The current
RD-Agent factor loop and the forthcoming LLM strategy proposer/remote worker
can be adapted without making the orchestrator own either implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.service import StrategyAcceptanceService, StrategyProcessingResult
from rdagent.scenarios.gf_quant.strategy.snapshot import (
    FactorUniverseSnapshot,
    FactorWorkingSet,
    build_factor_universe_snapshot,
    build_factor_working_set,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True)
class OuterEpochConfig:
    strategy_trials_per_epoch: int = 10
    targeted_factor_rounds_max: int = 5
    exploratory_factor_rounds: int = 2
    working_set_core: int = 16
    working_set_conditional: int = 8
    min_factors_per_strategy: int = 2
    max_factors_per_strategy: int = 8
    max_factor_demands: int = 3

    def __post_init__(self) -> None:
        if self.strategy_trials_per_epoch < 1:
            raise ValueError("strategy_trials_per_epoch must be positive")
        if self.targeted_factor_rounds_max < 0 or self.exploratory_factor_rounds < 0:
            raise ValueError("factor research round counts must be non-negative")
        if self.working_set_core < 0 or self.working_set_conditional < 0:
            raise ValueError("working-set quotas must be non-negative")
        if not 1 <= self.min_factors_per_strategy <= self.max_factors_per_strategy <= 8:
            raise ValueError("strategy factor count must satisfy 1 <= min <= max <= 8")
        if self.max_factor_demands < 0:
            raise ValueError("max_factor_demands must be non-negative")


@dataclass(frozen=True)
class TrialOutcomeSummary:
    trial_index: int
    strategy_name: str
    status: str
    gate_failures: tuple[str, ...]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class StrategyProposalContext:
    epoch_id: str
    trial_index: int
    data_cutoff: str
    snapshot_id: str
    working_set: FactorWorkingSet
    prior_outcomes: tuple[TrialOutcomeSummary, ...]


class StrategyProposalEngine(Protocol):
    def __call__(self, context: StrategyProposalContext) -> StrategyCandidate: ...


@dataclass(frozen=True)
class FactorContribution:
    factor_name: str
    factor_id: str | None
    strategy_trials: int
    accepted_trials: int
    mean_marginal_annualized_return: float
    mean_marginal_sharpe: float
    mean_marginal_drawdown_improvement: float
    mean_marginal_turnover_reduction: float


@dataclass(frozen=True)
class FactorDemand:
    demand_id: str
    epoch_id: str
    intended_role: str
    economic_theme: str
    target_family: str
    priority: float
    rationale: str
    evidence: Mapping[str, Any]
    avoid_factor_ids: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        epoch_id: str,
        intended_role: str,
        economic_theme: str = "any",
        target_family: str = "any",
        priority: float,
        rationale: str,
        evidence: Mapping[str, Any],
        avoid_factor_ids: Sequence[str] = (),
    ) -> "FactorDemand":
        body = {
            "epoch_id": epoch_id,
            "intended_role": intended_role,
            "economic_theme": economic_theme,
            "target_family": target_family,
            "priority": priority,
            "rationale": rationale,
            "evidence": dict(evidence),
            "avoid_factor_ids": list(avoid_factor_ids),
        }
        digest = hashlib.sha256(_canonical_json(body).encode()).hexdigest()[:16]
        return cls(
            demand_id=f"gf_factor_demand_{digest}",
            epoch_id=epoch_id,
            intended_role=intended_role,
            economic_theme=economic_theme,
            target_family=target_family,
            priority=priority,
            rationale=rationale,
            evidence=dict(evidence),
            avoid_factor_ids=tuple(avoid_factor_ids),
        )


@dataclass(frozen=True)
class FactorResearchRequest:
    epoch_id: str
    round_index: int
    mode: str
    demand: FactorDemand | None
    frozen_snapshot_id: str


class FactorResearchEngine(Protocol):
    def __call__(self, request: FactorResearchRequest) -> Any: ...


@dataclass(frozen=True)
class StrategyTrial:
    trial_index: int
    candidate: StrategyCandidate | None
    result: StrategyProcessingResult | None
    error: str | None = None


@dataclass(frozen=True)
class OuterEpochResult:
    epoch_id: str
    created_at: str
    snapshot: FactorUniverseSnapshot
    working_set: FactorWorkingSet
    strategy_trials: tuple[StrategyTrial, ...]
    contributions: tuple[FactorContribution, ...]
    factor_demands: tuple[FactorDemand, ...]
    factor_research_results: tuple[Any, ...] = field(default_factory=tuple)

    def compact_summary(self) -> dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "created_at": self.created_at,
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_checksum": self.snapshot.checksum,
            "snapshot_factor_count": len(self.snapshot.factors),
            "working_set_id": self.working_set.working_set_id,
            "working_set_factor_ids": [factor.factor_id for factor in self.working_set.factors],
            "strategy_trials": [
                {
                    "trial_index": trial.trial_index,
                    "strategy_name": trial.candidate.name if trial.candidate else None,
                    "status": trial.result.record.status if trial.result else "error",
                    "gate_failures": list(trial.result.gate_failures) if trial.result else [],
                    "error": trial.error,
                }
                for trial in self.strategy_trials
            ],
            "factor_demands": [demand.demand_id for demand in self.factor_demands],
            "factor_research_rounds": len(self.factor_research_results),
        }


def aggregate_factor_contributions(
    trials: Sequence[StrategyTrial], working_set: FactorWorkingSet
) -> tuple[FactorContribution, ...]:
    """Aggregate remote/local ablation metrics across all completed trials."""

    factor_by_name = {factor.name: factor for factor in working_set.factors}
    values: dict[str, list[tuple[bool, Mapping[str, Any]]]] = defaultdict(list)
    for trial in trials:
        if trial.result is None or trial.result.evaluation is None:
            continue
        accepted = trial.result.record.status == "accepted"
        for factor_name, contribution in trial.result.evaluation.factor_contributions.items():
            if factor_name in factor_by_name and isinstance(contribution, Mapping):
                values[factor_name].append((accepted, contribution))

    aggregated: list[FactorContribution] = []
    for factor_name, observations in values.items():
        factor = factor_by_name[factor_name]

        def mean(key: str) -> float:
            return sum(_finite(payload.get(key)) for _, payload in observations) / len(observations)

        aggregated.append(
            FactorContribution(
                factor_name=factor_name,
                factor_id=factor.factor_id,
                strategy_trials=len(observations),
                accepted_trials=sum(accepted for accepted, _ in observations),
                mean_marginal_annualized_return=mean("marginal_annualized_return"),
                mean_marginal_sharpe=mean("marginal_sharpe"),
                mean_marginal_drawdown_improvement=mean("marginal_drawdown_improvement"),
                mean_marginal_turnover_reduction=mean("marginal_turnover_reduction"),
            )
        )
    return tuple(sorted(aggregated, key=lambda item: (-item.strategy_trials, item.factor_name)))


def derive_factor_demands(
    *,
    epoch_id: str,
    trials: Sequence[StrategyTrial],
    contributions: Sequence[FactorContribution],
    working_set: FactorWorkingSet,
    limit: int = 3,
) -> tuple[FactorDemand, ...]:
    """Translate portfolio failures/ablations into bounded factor research requests."""

    if limit <= 0:
        return ()
    failure_counts = Counter(
        failure
        for trial in trials
        if trial.result is not None
        for failure in trial.result.gate_failures
    )
    demands: list[FactorDemand] = []
    mappings = (
        (("annualized_return", "sharpe"), "alpha", "return_or_sharpe_gate_failed"),
        (("max_drawdown",), "drawdown_control", "drawdown_gate_failed"),
        (("annual_turnover",), "turnover_control", "turnover_gate_failed"),
        (("strategy_holdings", "strategy_observations"), "coverage", "coverage_gate_failed"),
    )
    for failure_names, role, rationale in mappings:
        count = sum(failure_counts[name] for name in failure_names)
        if count:
            demands.append(
                FactorDemand.create(
                    epoch_id=epoch_id,
                    intended_role=role,
                    priority=float(count),
                    rationale=rationale,
                    evidence={"failed_trials": count, "gate_failures": list(failure_names)},
                )
            )

    factor_by_id = {factor.factor_id: factor for factor in working_set.factors}
    weak = sorted(
        (
            contribution
            for contribution in contributions
            if contribution.strategy_trials > 0 and contribution.mean_marginal_annualized_return <= 0
        ),
        key=lambda item: (item.mean_marginal_annualized_return, item.factor_name),
    )
    for contribution in weak:
        factor = factor_by_id.get(contribution.factor_id or "")
        demands.append(
            FactorDemand.create(
                epoch_id=epoch_id,
                intended_role="replacement_or_complement",
                economic_theme=factor.theme if factor else "any",
                target_family=factor.family if factor else "any",
                priority=1.0 + abs(contribution.mean_marginal_annualized_return),
                rationale="non_positive_marginal_portfolio_contribution",
                evidence={
                    "factor_name": contribution.factor_name,
                    "strategy_trials": contribution.strategy_trials,
                    "mean_marginal_annualized_return": contribution.mean_marginal_annualized_return,
                },
                avoid_factor_ids=(contribution.factor_id,) if contribution.factor_id else (),
            )
        )

    unique: dict[tuple[str, str, str], FactorDemand] = {}
    for demand in demands:
        key = (demand.intended_role, demand.economic_theme, demand.target_family)
        if key not in unique or demand.priority > unique[key].priority:
            unique[key] = demand
    ranked = sorted(unique.values(), key=lambda demand: (-demand.priority, demand.demand_id))
    return tuple(ranked[:limit])


class GFQuantOrchestrator:
    """Coordinate one or more frozen factor-to-strategy outer epochs."""

    def __init__(
        self,
        *,
        factor_registry: FactorRegistry,
        strategy_service: StrategyAcceptanceService,
        proposal_engine: StrategyProposalEngine,
        factor_research_engine: FactorResearchEngine | None = None,
        config: OuterEpochConfig | None = None,
        snapshot_dir: str | Path | None = None,
    ) -> None:
        self.factor_registry = factor_registry
        self.strategy_service = strategy_service
        self.proposal_engine = proposal_engine
        self.factor_research_engine = factor_research_engine
        self.config = config or OuterEpochConfig()
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else None

    def _validate_candidate(self, candidate: StrategyCandidate, working_set: FactorWorkingSet) -> None:
        if not candidate.factor_names:
            return
        if not self.config.min_factors_per_strategy <= len(candidate.factor_names) <= self.config.max_factors_per_strategy:
            raise ValueError(
                f"strategy must reference {self.config.min_factors_per_strategy}-"
                f"{self.config.max_factors_per_strategy} factors"
            )
        allowed = {factor.name for factor in working_set.factors}
        unknown = sorted(set(candidate.factor_names).difference(allowed))
        if unknown:
            raise ValueError(f"strategy references factors outside the frozen working set: {unknown[:3]}")

    def _bind_epoch(self, snapshot: FactorUniverseSnapshot, working_set: FactorWorkingSet, epoch_id: str) -> None:
        """Publish the freshly frozen snapshot to the strategy service, if it wants it.

        An evaluator that materializes remote artifacts has to know which
        snapshot an epoch is running against, but the snapshot only exists once
        ``run_epoch`` has built it.  Services that care expose a ``bind_epoch``
        method; everything else is unaffected and keeps working unchanged.
        """

        bind = getattr(self.strategy_service, "bind_epoch", None)
        if callable(bind):
            bind(snapshot=snapshot, working_set=working_set, epoch_id=epoch_id)

    @staticmethod
    def _outcome_summary(trial: StrategyTrial) -> TrialOutcomeSummary:
        assert trial.candidate is not None and trial.result is not None
        metrics = trial.result.record.metrics
        keep = {
            key: metrics[key]
            for key in ("annualized_return", "sharpe", "max_drawdown", "annual_turnover", "avg_holdings")
            if key in metrics
        }
        return TrialOutcomeSummary(
            trial_index=trial.trial_index,
            strategy_name=trial.candidate.name,
            status=trial.result.record.status,
            gate_failures=trial.result.gate_failures,
            metrics=keep,
        )

    def run_epoch(self, *, data_cutoff: str, epoch_id: str | None = None) -> OuterEpochResult:
        created_at = datetime.now(timezone.utc).isoformat()
        snapshot = build_factor_universe_snapshot(self.factor_registry, data_cutoff=data_cutoff, created_at=created_at)
        working_set = build_factor_working_set(
            snapshot,
            core_limit=self.config.working_set_core,
            conditional_limit=self.config.working_set_conditional,
        )
        resolved_epoch_id = epoch_id or f"gf_epoch_{snapshot.checksum[:12]}_{created_at[:10].replace('-', '')}"
        if self.snapshot_dir is not None:
            snapshot.write_json(self.snapshot_dir / f"{snapshot.snapshot_id}.json")
        self._bind_epoch(snapshot, working_set, resolved_epoch_id)

        trials: list[StrategyTrial] = []
        prior: list[TrialOutcomeSummary] = []
        for trial_index in range(self.config.strategy_trials_per_epoch):
            candidate: StrategyCandidate | None = None
            try:
                candidate = self.proposal_engine(
                    StrategyProposalContext(
                        epoch_id=resolved_epoch_id,
                        trial_index=trial_index,
                        data_cutoff=data_cutoff,
                        snapshot_id=snapshot.snapshot_id,
                        working_set=working_set,
                        prior_outcomes=tuple(prior),
                    )
                )
                self._validate_candidate(candidate, working_set)
                result = self.strategy_service.process(candidate)
                trial = StrategyTrial(trial_index, candidate, result)
                prior.append(self._outcome_summary(trial))
            except Exception as error:  # one failed proposal/evaluation must not abort the epoch
                trial = StrategyTrial(trial_index, candidate, None, f"{type(error).__name__}: {error}")
            trials.append(trial)

        contributions = aggregate_factor_contributions(trials, working_set)
        demands = derive_factor_demands(
            epoch_id=resolved_epoch_id,
            trials=trials,
            contributions=contributions,
            working_set=working_set,
            limit=self.config.max_factor_demands,
        )

        research_results: list[Any] = []
        if self.factor_research_engine is not None:
            targeted_rounds = min(self.config.targeted_factor_rounds_max, len(demands))
            for round_index in range(targeted_rounds):
                research_results.append(
                    self.factor_research_engine(
                        FactorResearchRequest(
                            epoch_id=resolved_epoch_id,
                            round_index=round_index,
                            mode="targeted",
                            demand=demands[round_index],
                            frozen_snapshot_id=snapshot.snapshot_id,
                        )
                    )
                )
            for offset in range(self.config.exploratory_factor_rounds):
                research_results.append(
                    self.factor_research_engine(
                        FactorResearchRequest(
                            epoch_id=resolved_epoch_id,
                            round_index=targeted_rounds + offset,
                            mode="exploratory",
                            demand=None,
                            frozen_snapshot_id=snapshot.snapshot_id,
                        )
                    )
                )

        return OuterEpochResult(
            epoch_id=resolved_epoch_id,
            created_at=created_at,
            snapshot=snapshot,
            working_set=working_set,
            strategy_trials=tuple(trials),
            contributions=contributions,
            factor_demands=demands,
            factor_research_results=tuple(research_results),
        )
