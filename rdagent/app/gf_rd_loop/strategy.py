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
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

# Load the repository .env before importing module-level Pydantic settings.
load_dotenv()

from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.app.gf_rd_loop.conf import configure_gf_llm_backend
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.deployment import DeploymentPolicy
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


def _parse_iso_date(value: str, *, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD): {value!r}") from exc


def _validate_epoch_window(window: Mapping[str, str], *, index: int | None = None) -> None:
    prefix = f"epoch window {index} " if index is not None else "epoch window "
    # The snapshot cutoff may be later than a historical evaluation window:
    # factor panels are point-in-time by observation date, while the cutoff
    # controls which registry versions are visible.  Validate its syntax, but
    # do not impose a false chronological relationship with the backtest.
    _parse_iso_date(window["data_cutoff"], label=f"{prefix}data_cutoff")
    start = _parse_iso_date(window["evaluation_start"], label=f"{prefix}evaluation_start")
    end = _parse_iso_date(window["evaluation_end"], label=f"{prefix}evaluation_end")
    if start > end:
        raise ValueError(f"{prefix}has evaluation_start after evaluation_end")


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
        "evaluation_start": result.evaluation_start,
        "evaluation_end": result.evaluation_end,
    }


def run_strategy_epochs(
    orchestrator: GFQuantOrchestrator,
    *,
    epochs: int,
    data_cutoff: str,
    report_dir: str | Path,
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
    epoch_window_factory: Callable[[int], Mapping[str, str]] | None = None,
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

    # Direct callers may omit the window even though the real assembled
    # orchestrator has an explicit default.  Resolve that default here so the
    # report and the evaluator receive identical effective dates.
    effective_default_start = evaluation_start
    effective_default_end = evaluation_end
    if effective_default_start is None and effective_default_end is None:
        effective_default_start = getattr(orchestrator, "default_evaluation_start", None)
        effective_default_end = getattr(orchestrator, "default_evaluation_end", None)
    if (effective_default_start is None) != (effective_default_end is None):
        raise ValueError("evaluation window requires both start and end")
    reports: list[dict[str, Any]] = []
    for index in range(epochs):
        result: OuterEpochResult | None = None
        error: str | None = None
        window: dict[str, str | None] = {
            "data_cutoff": data_cutoff,
            "evaluation_start": effective_default_start,
            "evaluation_end": effective_default_end,
        }
        if epoch_window_factory is not None:
            overrides = dict(epoch_window_factory(index))
            unknown = sorted(set(overrides).difference(window))
            if unknown:
                raise ValueError(f"epoch window contains unsupported keys: {unknown}")
            window.update(overrides)
        try:
            if window["evaluation_start"] is None or window["evaluation_end"] is None:
                raise ValueError("evaluation window requires both start and end")
            _validate_epoch_window(
                {
                    "data_cutoff": str(window["data_cutoff"]),
                    "evaluation_start": str(window["evaluation_start"]),
                    "evaluation_end": str(window["evaluation_end"]),
                },
            )
            result = orchestrator.run_epoch(
                data_cutoff=str(window["data_cutoff"]),
                evaluation_start=window["evaluation_start"],
                evaluation_end=window["evaluation_end"],
            )
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


def load_epoch_windows(path: str | Path) -> list[dict[str, str]]:
    """Load an explicit per-epoch PIT/evaluation schedule from JSON.

    Keeping the schedule in a small local file makes advancing outer epochs
    auditable and reproducible without embedding calendar arithmetic in the
    orchestrator.  Each entry must contain ``data_cutoff``,
    ``evaluation_start`` and ``evaluation_end``.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("epoch window file must contain a non-empty JSON list")
    required = {"data_cutoff", "evaluation_start", "evaluation_end"}
    windows: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"epoch window {index} must be a JSON object")
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"epoch window {index} is missing: {missing}")
        window = {key: str(item[key]) for key in required}
        if any(not value for value in window.values()):
            raise ValueError(f"epoch window {index} contains an empty date")
        _validate_epoch_window(window, index=index)
        if windows:
            previous = windows[-1]
            for field in ("data_cutoff", "evaluation_start", "evaluation_end"):
                if _parse_iso_date(window[field], label=f"epoch window {index} {field}") < _parse_iso_date(
                    previous[field], label=f"epoch window {index - 1} {field}"
                ):
                    raise ValueError(f"epoch window {index} moves {field} backwards")
            previous_end = _parse_iso_date(
                previous["evaluation_end"], label=f"epoch window {index - 1} evaluation_end"
            )
            current_start = _parse_iso_date(
                window["evaluation_start"], label=f"epoch window {index} evaluation_start"
            )
            if current_start <= previous_end:
                raise ValueError(f"epoch window {index} overlaps the previous evaluation window")
        windows.append(window)
    return windows


def _load_json_object(path: str | Path | None, label: str) -> dict[str, Any] | None:
    """Load an optional request-level configuration object from JSON.

    ``None`` means the section was not requested.  A file that exists must hold
    a JSON object: the two consumers (the audit sections and the walk-forward
    window sizes) both carry calibrated inputs this builder must not invent, so
    a malformed file is refused rather than silently skipped.
    """

    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} file must contain a JSON object")
    return dict(payload)


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

    def bind_epoch(
        self,
        *,
        snapshot: Any,
        working_set: Any,
        epoch_id: str,
        evaluation_start: str | None = None,
        evaluation_end: str | None = None,
    ) -> None:
        if evaluation_start is None or evaluation_end is None:
            raise ValueError("bound strategy epoch requires an evaluation window")
        self._service.bind_evaluation_window(evaluation_start, evaluation_end)
        self._bound = True
        self._service.evaluator = self._evaluator_factory(
            snapshot=snapshot,
            working_set=working_set,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )

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

    The inner evaluator is **kept**, not discarded.  It is the only object that
    holds the frozen provenance (snapshot, working set, artifact set, evaluation
    window), and the governance manifest needs exactly those facts -- a wrapper
    that built the evaluator, called it and dropped it would leave every
    provenance field empty while still looking like a healthy manifest.
    ``manifest_facts`` therefore delegates to the inner instance and raises when
    there is nothing to delegate to, so a missing provenance can never be
    mistaken for an empty one.
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
        audit: Mapping[str, Any] | None = None,
        walk_forward: Mapping[str, Any] | None = None,
        manifest: Mapping[str, Any] | None = None,
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
        # Request-level audit objects (governance + result-report audits).  They
        # are configuration, not constants: the data-leakage tables and the
        # walk-forward window sizes are calibrated upstream, and the emission
        # switch asks for the worker-side attestation.  All default to None,
        # which means "do not request that section" -- nothing is invented here.
        self.audit = audit
        self.walk_forward = walk_forward
        self.manifest = manifest
        # The most recent inner evaluator, so ``manifest_facts`` can read the
        # provenance of the artifact set that actually produced the last result.
        self._last_evaluator: Any = None
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

    def manifest_facts(self) -> dict[str, Any]:
        """Provenance of the artifact set behind the last evaluation.

        Raises when nothing has been evaluated yet or the inner evaluator cannot
        supply facts: an absent provenance must surface as a failure, never as a
        manifest whose provenance fields are quietly empty.
        """

        inner = self._last_evaluator
        if inner is None:
            raise ValueError(
                "strategy evaluator has no provenance yet: manifest facts are only available after an evaluation"
            )
        facts = getattr(inner, "manifest_facts", None)
        if not callable(facts):
            raise ValueError("strategy evaluator does not provide manifest facts")
        return dict(facts())

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
        result = evaluator(
            candidate,
            audit=self.audit,
            walk_forward=self.walk_forward,
            manifest=self.manifest,
        )
        # Only remember an evaluator that can actually answer the provenance
        # question, so a substitute that cannot is reported as absent rather
        # than being cached and then treated as a fact source.
        self._last_evaluator = evaluator if callable(getattr(evaluator, "manifest_facts", None)) else None
        return result


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
    audit: Mapping[str, Any] | None = None,
    walk_forward: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> GFQuantOrchestrator:
    """Assemble the real loop: LLM proposals, remote evaluation, demand research.

    ``proposal_engine``, ``factor_proposer`` and ``factor_acceptance_service``
    default to their RD-Agent implementations; passing substitutes is how tests
    and experiments replace a single collaborator without stubbing the rest.

    Transaction costs are owned by the strategy execution layer and cannot be
    changed by this builder, settings, environment variables or the CLI.

    ``audit`` / ``walk_forward`` / ``manifest`` are request-level configuration
    for the audit, walk-forward and manifest sections.  They default to
    ``None``, which means the section is simply not requested: the two injected
    inputs they need (the data-leakage availability tables and the walk-forward
    window sizes) are calibrated upstream and are deliberately not invented
    here.  The three governance switches on the acceptance service follow the
    same rule -- ``emit_manifest`` and ``reproducibility_check`` turn on exactly
    when a manifest was asked for, and the deployment gate is enabled only
    alongside them.
    """

    from rdagent.scenarios.gf_quant.mining import RDAGentDSLProposer
    from rdagent.scenarios.gf_quant.remote import GFFactorEvaluator
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.factor_research import DemandDrivenFactorResearchEngine
    from rdagent.scenarios.gf_quant.strategy.proposal import LLMStrategyProposalEngine

    factor_registry = factor_registry or FactorRegistry(settings.registry_path)
    strategy_registry = strategy_registry or StrategyRegistry(settings.strategy_registry_path)
    default_evaluation_start = evaluation_start
    default_evaluation_end = evaluation_end

    config = config or OuterEpochConfig()

    def make_remote_evaluator(
        *,
        snapshot: Any,
        working_set: Any,
        evaluation_start: str | None = None,
        evaluation_end: str | None = None,
    ) -> MaterializingRemoteStrategyEvaluator:
        return MaterializingRemoteStrategyEvaluator(
            snapshot=snapshot,
            working_set=working_set,
            evaluation_start=evaluation_start or default_evaluation_start,
            evaluation_end=evaluation_end or default_evaluation_end,
            settings=settings,
            rebalance_frequency=rebalance_frequency,
            materializer=artifact_materializer,
            remote_evaluator_factory=remote_evaluator_factory,
            audit=audit,
            walk_forward=walk_forward,
            manifest=manifest,
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
            # The governance reports only exist when a manifest was asked for:
            # without the worker attestation there is nothing to check
            # reproducibility against, and a deployment verdict built on absent
            # evidence would be a claim this builder did not earn.
            emit_manifest=manifest is not None,
            reproducibility_check=manifest is not None,
            deployment_policy=DeploymentPolicy() if manifest is not None else None,
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
        default_evaluation_start=default_evaluation_start,
        default_evaluation_end=default_evaluation_end,
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
    epoch_windows: Annotated[
        Path | None,
        typer.Option(help="Optional JSON list of per-epoch data/evaluation windows."),
    ] = None,
    audit_config: Annotated[
        Path | None,
        typer.Option(help="Optional JSON object with the request-level audit sections."),
    ] = None,
    walk_forward_config: Annotated[
        Path | None,
        typer.Option(help="Optional JSON object with the walk-forward window sizes."),
    ] = None,
    governance_report: Annotated[
        bool,
        typer.Option(
            "--governance-report",
            help=(
                "Emit all three governance reports together: the strategy manifest, the reproducibility "
                "verdict and the deployment band. Without --audit-config and --walk-forward-config the "
                "deployment band is always research_only, because the evidence those sections carry is "
                "then absent."
            ),
        ),
    ] = False,
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
        # The audit tables and walk-forward window sizes come from files the
        # upstream calibration owns; this CLI only forwards them, never guesses
        # them.  A missing file means that section is not requested.
        audit=_load_json_object(audit_config, "audit_config"),
        walk_forward=_load_json_object(walk_forward_config, "walk_forward_config"),
        manifest={} if governance_report else None,
    )
    windows = load_epoch_windows(epoch_windows) if epoch_windows is not None else None
    if windows is not None and len(windows) < epochs:
        raise ValueError(f"epoch window file contains {len(windows)} windows but epochs={epochs}")
    run_strategy_epochs(
        orchestrator,
        epochs=epochs,
        data_cutoff=data_cutoff,
        report_dir=report_dir,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        epoch_window_factory=(lambda index: windows[index]) if windows is not None else None,
    )


if __name__ == "__main__":
    typer.run(main)
