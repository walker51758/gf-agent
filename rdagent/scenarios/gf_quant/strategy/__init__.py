"""GF strategy layer: factor discovery -> composition -> portfolio evaluation -> registry.

The package is additive to the factor loop: nothing here modifies
``rdagent.scenarios.gf_quant`` behaviour, and the only shared objects are the
deduplicator, the factor registry (read-only) and the panel conventions.
See ``docs/gf_strategy_agent_design.md`` for the architecture decision and the
batch plan.
"""

from rdagent.scenarios.gf_quant.strategy.backtest import PortfolioResult, run_portfolio_backtest
from rdagent.scenarios.gf_quant.strategy.artifacts import FactorArtifactCache
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.composition import build_target_weights
from rdagent.scenarios.gf_quant.strategy.registry import StrategyRecord, StrategyRegistry, strategy_id_for
from rdagent.scenarios.gf_quant.strategy.proposal import LLMStrategyProposalEngine
from rdagent.scenarios.gf_quant.strategy.factor_research import (
    FactorAttemptSummary,
    FactorResearchOutcome,
    DemandDrivenFactorResearchEngine,
    render_demand_feedback,
)
from rdagent.scenarios.gf_quant.strategy.orchestrator import (
    FactorContribution,
    FactorDemand,
    FactorResearchRequest,
    GFQuantOrchestrator,
    OuterEpochConfig,
    OuterEpochResult,
    StrategyProposalContext,
    aggregate_factor_contributions,
    derive_factor_demands,
)
from rdagent.scenarios.gf_quant.strategy.report import epoch_report, write_epoch_report
from rdagent.scenarios.gf_quant.strategy.service import (
    StrategyAcceptancePolicy,
    StrategyAcceptanceService,
    StrategyEvaluation,
    StrategyProcessingResult,
)
from rdagent.scenarios.gf_quant.strategy.snapshot import (
    FactorSnapshotItem,
    FactorUniverseSnapshot,
    FactorWorkingSet,
    build_factor_universe_snapshot,
    build_factor_working_set,
    factor_eligibility,
)

__all__ = [
    "BacktestSpec",
    "CompositionSpec",
    "FactorSnapshotItem",
    "FactorAttemptSummary",
    "FactorContribution",
    "FactorArtifactCache",
    "FactorDemand",
    "FactorResearchOutcome",
    "FactorResearchRequest",
    "FactorUniverseSnapshot",
    "FactorWorkingSet",
    "GFQuantOrchestrator",
    "LLMStrategyProposalEngine",
    "DemandDrivenFactorResearchEngine",
    "OuterEpochConfig",
    "OuterEpochResult",
    "PortfolioResult",
    "StrategyAcceptancePolicy",
    "StrategyAcceptanceService",
    "StrategyCandidate",
    "StrategyEvaluation",
    "StrategyProcessingResult",
    "StrategyProposalContext",
    "StrategyRecord",
    "StrategyRegistry",
    "build_target_weights",
    "aggregate_factor_contributions",
    "build_factor_universe_snapshot",
    "build_factor_working_set",
    "epoch_report",
    "factor_eligibility",
    "derive_factor_demands",
    "render_demand_feedback",
    "run_portfolio_backtest",
    "strategy_id_for",
    "write_epoch_report",
]
