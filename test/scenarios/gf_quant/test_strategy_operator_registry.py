"""Registry sanity: the strategy operator catalogue matches both code sides.

The catalogue in ``rdagent.scenarios.gf_quant.strategy.operators`` is only
useful if every anchor it claims actually exists.  These tests import the
referenced modules (including the standalone single-file worker) and resolve
every dotted symbol, so a rename or a move breaks the registry test instead of
silently rotting the inventory.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from rdagent.scenarios.gf_quant.strategy.operators import (
    OPERATORS,
    OPERATOR_GROUPS,
    WORKER_PATH,
    implemented_operators,
    query,
)

_REPO_ROOT = Path(__file__).parents[3]


def _import_module(repo_relative_file: str):
    module_name = repo_relative_file.removesuffix(".py").replace("/", ".")
    return importlib.import_module(module_name)


def _resolve_symbol(module, dotted_symbol: str) -> None:
    target = module
    for attribute in dotted_symbol.split("."):
        target = getattr(target, attribute)
    assert target is not None


def test_registry_lists_exactly_the_implemented_catalogue() -> None:
    assert tuple(OPERATORS) == (
        "DirectionMap",
        "CrossSectionalRank",
        "Winsorize",
        "CrossSectionalStandardize",
        "FactorTransform",
        "MissingValuePolicy",
        "FactorBlend",
        "FactorGroupBlend",
        "FactorInteraction",
        "IndustryNeutralize",
        "StyleNeutralize",
        "RegimeConditionalSignal",
        "TopKSelector",
        "RankTieBreaker",
        "QuantileGate",
        "CascadedFactorGating",
        "HoldingBuffer",
        "DiversitySelector",
        "SingleNameCap",
        "SingleNameFloor",
        "IndustryWeightCap",
        "CashResidualPolicy",
        "IndustryDeviationConstraint",
        "StyleExposureConstraint",
        "BetaConstraint",
        "FactorExposureConstraint",
        "ConstrainedWeightProjection",
        "EqualWeight",
        "RankWeight",
        "ScoreProportionalWeight",
        "ScoreTransformWeight",
        "InverseVolatilityWeight",
        "HierarchicalWeighting",
        "RiskParityWeight",
        "FactorRiskBudgetWeight",
        "RebalanceSchedule",
        "SignalLag",
        "DriftBandRebalance",
        "TurnoverShrinkage",
        "MinimumHoldingPeriod",
        "ParticipationCap",
        "LotRounding",
        "SideAwareTradability",
        "StagedExecution",
        "FixedExecutionCost",
        "PortfolioVolatilityTarget",
        "DrawdownStateMachine",
        "FactorDecayMonitor",
        "ReturnProfileDedup",
        "StressScenarioCheck",
        "ConcentrationMonitor",
        "CostSensitivityAudit",
        "ParameterPerturbationAudit",
        "AblationAudit",
        "FeasibilityAudit",
        "DataLeakageAudit",
        "WalkForwardEvaluation",
        "StrategyManifestEmitter",
        "ReproducibilityCheck",
        "DeploymentGate",
        "BaseUniverse",
        "IndexMembershipFilter",
        "ListingAgeFilter",
        "StatusExclusionFilter",
        "PriceLimitTradabilityFilter",
        "LiquidityFilter",
        "PriceRangeFilter",
        "DataQualityFilter",
    )
    assert len(implemented_operators()) == 68


def test_registry_groups_cover_the_catalogue_exactly() -> None:
    grouped = [name for names in OPERATOR_GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped)) == len(OPERATORS)
    assert set(grouped) == set(OPERATORS)


def test_every_local_anchor_resolves() -> None:
    for name, spec in OPERATORS.items():
        # A side may be empty on purpose, but only when notes explain the gap.
        assert spec.local_anchors or spec.notes, f"{name} must record a local implementation or explain the gap in notes"
        for file, symbol in spec.local_anchors:
            assert (_REPO_ROOT / file).is_file(), f"{name}: missing local file {file}"
            module = _import_module(file)
            try:
                _resolve_symbol(module, symbol)
            except AttributeError as exc:  # pragma: no cover - failure reporting
                raise AssertionError(f"{name}: {file} does not define {symbol}") from exc


def test_every_worker_anchor_resolves() -> None:
    for name, spec in OPERATORS.items():
        for file, symbol in spec.worker_anchors:
            assert (_REPO_ROOT / file).is_file(), f"{name}: missing worker file {file}"
            module = _import_module(file)
            try:
                _resolve_symbol(module, symbol)
            except AttributeError as exc:  # pragma: no cover - failure reporting
                raise AssertionError(f"{name}: {file} does not define {symbol}") from exc


def test_one_side_only_operators_are_annotated() -> None:
    assert OPERATORS["CrossSectionalRank"].local_anchors == ()
    assert "worker" in OPERATORS["CrossSectionalRank"].notes.lower()
    assert "worker" not in OPERATORS["ReturnProfileDedup"].worker_anchors
    assert OPERATORS["ReturnProfileDedup"].notes


def test_every_entry_points_at_the_strategy_worker_path() -> None:
    for name, spec in OPERATORS.items():
        if spec.worker_anchors:
            assert {file for file, _ in spec.worker_anchors} == {WORKER_PATH}, name


def test_doc_sections_exist_in_the_roadmap() -> None:
    roadmap = (_REPO_ROOT / "docs" / "gf_strategy_structured_operators.md").read_text(encoding="utf-8")
    for name, spec in OPERATORS.items():
        assert spec.doc_section, name
        header = f"### {spec.doc_section} `{name}`"
        assert header in roadmap, f"{name}: roadmap header {header!r} not found"


def test_query_rejects_unknown_names() -> None:
    with pytest.raises(KeyError, match="NonexistentOperator"):
        query("NonexistentOperator")
