"""Remote-first GF strategy research loop.

This is the strategy counterpart of :mod:`rdagent.app.gf_rd_loop.factor`: instead
of mining single factors it runs the double-timescale loop from
``docs/gf_strategy_agent_design.md`` (batch 5) --

    frozen snapshot -> working set -> inner portfolio trials -> ablations ->
    factor demands -> demand-driven factor research -> epoch report

New factors registered by the outer round stay out of the already-frozen
snapshot by construction; they surface in the next epoch's snapshot.

Two entry points with different audiences:

* :func:`run_strategy_epochs` is pure orchestration over an already-built
  orchestrator and is what the tests exercise;
* :func:`build_strategy_orchestrator` wires the real LLM proposer, the remote
  portfolio evaluator and the demand-driven factor researcher.  It needs a live
  SSH host and API key, so nothing here tries to degrade gracefully -- a missing
  piece fails loudly instead of silently producing an unbacktested strategy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

# Load the repository .env before importing module-level Pydantic settings.
load_dotenv()

from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.app.gf_rd_loop.conf import configure_gf_llm_backend
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.orchestrator import (
    GFQuantOrchestrator,
    OuterEpochConfig,
    OuterEpochResult,
)
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry
from rdagent.scenarios.gf_quant.strategy.report import write_epoch_report
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
)

REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")


def epoch_summary(result: OuterEpochResult) -> dict[str, Any]:
    """One-line-per-epoch view used for the console and for chaining epochs."""

    statuses: dict[str, int] = {}
    for trial in result.strategy_trials:
        status = trial.result.record.status if trial.result is not None else "error"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "epoch_id": result.epoch_id,
        "snapshot_id": result.snapshot.snapshot_id,
        "working_set_id": result.working_set.working_set_id,
        "strategy_trials": len(result.strategy_trials),
        "strategy_statuses": statuses,
        "factor_demands": len(result.factor_demands),
        "factor_research_rounds": len(result.factor_research_results),
    }


def run_strategy_epochs(
    orchestrator: GFQuantOrchestrator,
    *,
    epochs: int,
    data_cutoff: str,
    report_dir: str | Path,
) -> list[dict[str, Any]]:
    """Run whole outer epochs and persist one compact report per epoch.

    A failing epoch is recorded as a report with a ``status`` of ``error`` rather
    than aborting the run: the loop is meant to be left running over many epochs,
    and one unreachable remote must not discard the epochs that already
    completed.  Successive epochs intentionally see each other's newly
    registered factors, because every epoch refreezes the snapshot.
    """

    if epochs < 1:
        raise ValueError("epochs must be positive")
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for index in range(epochs):
        result: OuterEpochResult | None = None
        error: str | None = None
        try:
            result = orchestrator.run_epoch(data_cutoff=data_cutoff)
        except Exception as exc:  # keep going, but say so in the report
            error = f"{type(exc).__name__}: {exc}"

        if result is None:
            report: dict[str, Any] = {"status": "error", "error": error, "epoch_id": None}
        else:
            path = write_epoch_report(result, destination / f"{index:03d}_{result.epoch_id}.json")
            report = {**epoch_summary(result), "report_path": str(path), "status": "ok", "error": None}

        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
    return reports


class EpochBoundStrategyService:
    """``StrategyAcceptanceService`` plus knowledge of the current epoch.

    The remote portfolio evaluator needs the snapshot that the orchestrator
    froze *for this epoch* -- a snapshot rebuilt later would carry a different
    ``created_at`` and therefore a different id, which would silently break the
    provenance chain the worker echoes back.  The orchestrator calls
    ``bind_epoch`` right after freezing (see
    ``GFQuantOrchestrator._bind_epoch``); every other call is delegated.
    """

    def __init__(self, service: StrategyAcceptanceService, evaluator_factory: Callable[..., Any]) -> None:
        self._service = service
        self._evaluator_factory = evaluator_factory
        self._bound: bool = False

    def bind_epoch(self, *, snapshot: Any, working_set: Any, epoch_id: str) -> None:
        self._bound = True
        self._service.evaluator = self._evaluator_factory(snapshot=snapshot, working_set=working_set)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)

    def process(self, candidate: StrategyCandidate, **kwargs: Any):
        if not self._bound:
            raise RuntimeError("no epoch is bound; the orchestrator must call bind_epoch first")
        return self._service.process(candidate, **kwargs)


class MaterializingRemoteStrategyEvaluator:
    """Evaluate one candidate against an artifact set materialized for it.

    ``RemoteStrategyEvaluator`` requires the artifact set to hold exactly the
    candidate's factors, and each trial in an epoch proposes a different factor
    subset, so the artifact set is built per candidate rather than once per
    epoch.  Materialization is injectable so tests never touch SSH.
    """

    def __init__(
        self,
        *,
        snapshot: Any,
        working_set: Any,
        evaluation_start: str,
        evaluation_end: str,
        settings: Any = GF_SCENARIO_SETTINGS,
        rebalance_frequency: str = "monthly",
        materializer: Callable[..., Any] | None = None,
        remote_evaluator_factory: Callable[..., Any] | None = None,
    ) -> None:
        if rebalance_frequency not in REBALANCE_FREQUENCIES:
            raise ValueError(f"rebalance_frequency must be one of {REBALANCE_FREQUENCIES}")
        self.snapshot = snapshot
        self.working_set = working_set
        self.evaluation_start = evaluation_start
        self.evaluation_end = evaluation_end
        self.settings = settings
        self.rebalance_frequency = rebalance_frequency
        self.remote_evaluator_factory = remote_evaluator_factory
        from rdagent.scenarios.gf_quant.strategy.artifacts import (
            FactorArtifactCache,
            materialize_factor_artifact_set,
        )

        start_tag = re.sub(r"[^0-9]", "", evaluation_start)
        end_tag = re.sub(r"[^0-9]", "", evaluation_end)
        remote_root = settings.remote_work_dir.rstrip("/\\")
        versioned_dir = (
            f"{remote_root}/strategy_factor_artifacts/"
            f"{snapshot.snapshot_id}_{start_tag}_{end_tag}"
        )
        self.artifact_cache = FactorArtifactCache(
            working_set=working_set,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            settings=settings,
            remote_artifact_dir=versioned_dir,
            materializer=materializer or materialize_factor_artifact_set,
        )

    def __call__(self, candidate: StrategyCandidate) -> StrategyEvaluation:
        from rdagent.scenarios.gf_quant.strategy.remote import RemoteStrategyEvaluator

        factor_names = sorted(candidate.factor_names)
        artifact_set = self.artifact_cache.artifact_set_for(factor_names)
        evaluator = (
            self.remote_evaluator_factory(artifact_set)
            if self.remote_evaluator_factory is not None
            else RemoteStrategyEvaluator(
                snapshot=self.snapshot,
                working_set=self.working_set,
                artifact_set=artifact_set,
                evaluation_start=self.evaluation_start,
                evaluation_end=self.evaluation_end,
                settings=self.settings,
                rebalance_frequency=self.rebalance_frequency,
            )
        )
        return evaluator(candidate)


# Backward-compatible name used by the first integration tests.
RemotePortfolioEvaluator = MaterializingRemoteStrategyEvaluator


def build_strategy_orchestrator(
    *,
    data_cutoff: str,
    evaluation_start: str,
    evaluation_end: str,
    rebalance_frequency: str = "monthly",
    config: OuterEpochConfig | None = None,
    settings: Any = GF_SCENARIO_SETTINGS,
    factor_registry: FactorRegistry | None = None,
    strategy_registry: StrategyRegistry | None = None,
    proposal_engine: Any = None,
    factor_proposer: Any = None,
    factor_acceptance_service: Any = None,
    artifact_materializer: Callable[..., Any] | None = None,
    remote_evaluator_factory: Callable[..., Any] | None = None,
    snapshot_dir: str | Path | None = None,
) -> GFQuantOrchestrator:
    """Assemble the real loop: LLM proposals, remote evaluation, demand research.

    ``proposal_engine``, ``factor_proposer`` and ``factor_acceptance_service``
    default to their RD-Agent implementations; passing substitutes is how tests
    and experiments replace a single collaborator without stubbing the rest.
    """

    from rdagent.scenarios.gf_quant.mining import RDAGentDSLProposer
    from rdagent.scenarios.gf_quant.remote import GFFactorEvaluator
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.factor_research import DemandDrivenFactorResearchEngine
    from rdagent.scenarios.gf_quant.strategy.proposal import LLMStrategyProposalEngine

    factor_registry = factor_registry or FactorRegistry(settings.registry_path)
    strategy_registry = strategy_registry or StrategyRegistry(settings.strategy_registry_path)

    def make_remote_evaluator(*, snapshot: Any, working_set: Any) -> MaterializingRemoteStrategyEvaluator:
        return MaterializingRemoteStrategyEvaluator(
            snapshot=snapshot,
            working_set=working_set,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            settings=settings,
            rebalance_frequency=rebalance_frequency,
            materializer=artifact_materializer,
            remote_evaluator_factory=remote_evaluator_factory,
        )

    service = EpochBoundStrategyService(
        StrategyAcceptanceService(
            strategy_registry,
            None,  # replaced on bind_epoch, once the snapshot is frozen
            policy=StrategyAcceptancePolicy(
                min_annualized_return=0.08,
                min_excess_annualized_return=0.0,
                min_positive_return_years=2,
            ),
            factor_registry=factor_registry,
        ),
        make_remote_evaluator,
    )

    if proposal_engine is None:
        proposal_engine = LLMStrategyProposalEngine()
    if factor_acceptance_service is None:
        factor_acceptance_service = FactorAcceptanceService(
            factor_registry,
            GFFactorEvaluator(settings=settings, mode="remote"),
        )
    factor_research = DemandDrivenFactorResearchEngine(
        proposer=factor_proposer or RDAGentDSLProposer(),
        factor_service=factor_acceptance_service,
        factor_registry=factor_registry,
    )

    return GFQuantOrchestrator(
        factor_registry=factor_registry,
        strategy_service=service,
        proposal_engine=proposal_engine,
        factor_research_engine=factor_research,
        config=config or OuterEpochConfig(),
        snapshot_dir=snapshot_dir,
    )


def main(
    epochs: Annotated[int, typer.Option(min=1, help="Number of outer epochs to run.")] = 1,
    data_cutoff: Annotated[str, typer.Option(help="Snapshot data cutoff (ISO date).")] = "2023-12-31",
    evaluation_start: Annotated[str, typer.Option(help="Portfolio evaluation window start.")] = "2020-01-01",
    evaluation_end: Annotated[str, typer.Option(help="Portfolio evaluation window end.")] = "2023-12-31",
    rebalance_frequency: Annotated[
        str, typer.Option(help="Rebalance schedule for the portfolio trials.")
    ] = "monthly",
    trials_per_epoch: Annotated[int, typer.Option(min=1, help="Inner strategy trials per epoch.")] = 5,
    working_set_core: Annotated[int, typer.Option(min=0, help="Core factors kept in the working set.")] = 16,
    working_set_conditional: Annotated[int, typer.Option(min=0, help="Conditional factors kept.")] = 8,
    max_factor_demands: Annotated[int, typer.Option(min=0, help="Factor research demands per epoch.")] = 3,
    targeted_factor_rounds: Annotated[
        int, typer.Option(min=0, help="Maximum demand-driven factor rounds per epoch.")
    ] = 5,
    exploratory_factor_rounds: Annotated[
        int, typer.Option(min=0, help="Untargeted factor exploration rounds per epoch.")
    ] = 2,
    report_dir: Annotated[
        Path, typer.Option(help="Directory receiving one JSON report per epoch.")
    ] = Path("artifacts/live/strategy_epochs"),
    snapshot_dir: Annotated[Path, typer.Option(help="Directory receiving frozen snapshot JSON.")] = Path(
        "git_ignore_folder/gf_quant/snapshots"
    ),
) -> None:
    """Run the remote-first GF strategy research loop over N outer epochs."""
    if rebalance_frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(f"rebalance_frequency must be one of {REBALANCE_FREQUENCIES}")
    configure_gf_llm_backend()

    orchestrator = build_strategy_orchestrator(
        data_cutoff=data_cutoff,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        rebalance_frequency=rebalance_frequency,
        config=OuterEpochConfig(
            strategy_trials_per_epoch=trials_per_epoch,
            working_set_core=working_set_core,
            working_set_conditional=working_set_conditional,
            max_factor_demands=max_factor_demands,
            targeted_factor_rounds_max=targeted_factor_rounds,
            exploratory_factor_rounds=exploratory_factor_rounds,
        ),
        snapshot_dir=snapshot_dir,
    )
    run_strategy_epochs(
        orchestrator,
        epochs=epochs,
        data_cutoff=data_cutoff,
        report_dir=report_dir,
    )


if __name__ == "__main__":
    typer.run(main)
