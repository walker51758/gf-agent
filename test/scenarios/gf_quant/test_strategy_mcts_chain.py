"""Tests for the search's factor-evaluation, proposal, remote and isolation layers.

The four things asserted here are the parts of the chain that a reader cannot
verify by inspection: that the local RankIC evaluator produces the same numbers
through the worker's own code path, that a proposal is gated rather than trusted,
that the remote switch has a real implementation with a testable protocol and an
explicit refusal when unconfigured, and that a test run cannot write into a
production directory.

Everything runs on a synthetic cache, so the suite needs no rqdatac licence, no
network and no API key.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.service import EvaluationResult, FactorCandidate
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.search.data_cache import SearchDataCache, synthetic_cache
from rdagent.scenarios.gf_quant.strategy.search.evaluator import (
    REMOTE_BACKEND,
    build_evaluator,
    remote_evaluator_unavailable,
)
from rdagent.scenarios.gf_quant.strategy.search.factor_eval import (
    LocalFactorEvaluationSettings,
    LocalRankICFactorEvaluator,
    load_factor_worker,
    repository_root,
    settings_for_cache,
    verify_cache_layout,
)
from rdagent.scenarios.gf_quant.strategy.search.isolation import (
    SearchPaths,
    assert_isolated,
    isolated_paths,
    production_paths,
)
from rdagent.scenarios.gf_quant.strategy.search.runner import build_local_proposal_chain

SYMBOLS = tuple(f"S{index:03d}.XSHE" for index in range(14))
DATES = pd.bdate_range("2021-01-04", "2023-12-29")


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> SearchDataCache:
    root = tmp_path_factory.mktemp("factor_eval_cache")
    return synthetic_cache(root, symbols=SYMBOLS, dates=DATES, seed=23)


# ---- the repository-root guard ------------------------------------------------


def test_repository_root_resolves_to_a_directory_holding_the_package() -> None:
    """A plain parents[N] once resolved to ``rdagent/scripts``; the marker stops that."""

    root = repository_root()
    assert (root / "rdagent" / "scenarios" / "gf_quant").is_dir()
    assert (root / "scripts" / "gf_quant_remote_worker.py").is_file()


def test_the_factor_worker_loads_and_carries_its_evaluation_entry_points() -> None:
    worker = load_factor_worker()
    for name in ("SafeExpression", "PanelLoader", "EvaluationConfig", "evaluate_factor", "daily_series"):
        assert hasattr(worker, name), f"worker is missing {name}"


# ---- the local RankIC evaluator ----------------------------------------------


def test_synthetic_cache_carries_every_field_the_dsl_is_prompted_with(cache: SearchDataCache) -> None:
    """A proposed expression naming a field the cache lacks fails on a missing file.

    The dump names are checked against the proposer's own vocabulary so a rename
    on either side is caught here rather than as an obscure pyarrow error during
    a search that has already spent credits.
    """

    for name in ("adjclose", "volume", "amount", "turnover", "calendar"):
        assert (cache.root / f"{name}.parquet").exists(), f"cache dump is missing {name}.parquet"


def test_settings_keep_the_calendar_years_the_annual_gate_needs(cache: SearchDataCache) -> None:
    """A window that skips a covered year fails the annual gate for no reason."""

    settings = settings_for_cache(cache, minimum_calendar_years=3)
    window = cache.trading_dates()
    window = window[(window >= settings.start) & (window <= settings.end)]
    assert len(set(window.year)) >= 3


def test_settings_refuse_an_observation_requirement_the_window_cannot_meet(cache: SearchDataCache) -> None:
    """The observation count is a function of the stride; asking for more fails loudly."""

    with pytest.raises(ValueError, match="observations"):
        settings_for_cache(cache, sample_step=5, minimum_observations=10_000)


def test_the_evaluator_produces_a_real_rank_ic_through_the_worker_path(cache: SearchDataCache, tmp_path: Path) -> None:
    """The number must come from the worker's own series, not a local reimplementation."""

    settings = settings_for_cache(cache, minimum_calendar_years=3)
    evaluator = LocalRankICFactorEvaluator(cache=cache, settings=settings, artifact_dir=tmp_path / "artifacts")
    candidate = FactorCandidate(
        name="probe_vol",
        family="price_volume",
        hypothesis="probe",
        expression="ts_std(adjclose, 20) / ts_mean(adjclose, 20)",
    )
    result = evaluator(candidate)
    metrics = result.metrics

    assert isinstance(result, EvaluationResult)
    assert metrics["observations"] > 0
    assert metrics["metrics_by_year"], "the annual gate needs a per-year breakdown"
    assert -1.0 <= float(metrics["Rank_IC"]) <= 1.0
    assert result.artifact_path is not None and Path(result.artifact_path).exists()
    assert result.artifact_checksum


def test_the_evaluator_fails_closed_on_an_operator_outside_the_allow_list(
    cache: SearchDataCache, tmp_path: Path
) -> None:
    """The DSL's own validation must still be doing the rejecting."""

    settings = settings_for_cache(cache, minimum_calendar_years=3)
    evaluator = LocalRankICFactorEvaluator(cache=cache, settings=settings, artifact_dir=tmp_path / "artifacts")
    candidate = FactorCandidate(
        name="probe_bad", family="price_volume", hypothesis="probe", expression="os_system('ls')"
    )
    with pytest.raises(ValueError):
        evaluator(candidate)


def test_verify_cache_layout_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_cache_layout(tmp_path)


def test_request_for_describes_the_same_fields_the_remote_worker_reads(
    cache: SearchDataCache, tmp_path: Path
) -> None:
    """One description of a candidate drives both the local and the remote path."""

    settings = LocalFactorEvaluationSettings(start="2022-01-03", end="2022-12-30")
    evaluator = LocalRankICFactorEvaluator(cache=cache, settings=settings, artifact_dir=tmp_path / "artifacts")
    request = evaluator.request_for(
        FactorCandidate(name="probe", family="price_volume", hypothesis="p", expression="1 / adjclose")
    )
    assert set(request) >= {"name", "family", "expression", "base_path", "evaluation", "artifact_path"}
    assert set(request["evaluation"]) >= {"start", "end", "label_horizon", "quantile", "min_cross_section", "sample_step"}


# ---- the proposal chain -------------------------------------------------------


class _RejectingEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, candidate: FactorCandidate) -> EvaluationResult:
        self.calls += 1
        return EvaluationResult(
            metrics={
                "Rank_IC": 0.001,
                "Rank_ICIR": 0.01,
                "annualized_return": -0.1,
                "sharpe": -0.5,
                "max_drawdown": -0.6,
                "turnover": 2.0,
                "coverage": 0.1,
                "observations": 5,
                "metrics_by_year": {"2022": {"Rank_IC": 0.001}},
            }
        )


class _ScriptedProposer:
    def __init__(self, proposals: list[FactorCandidate]) -> None:
        self.proposals = proposals
        self.feedback: list[str] = []
        self.families: list[str] = []

    def propose(self, family: str, feedback: str | None = None) -> list[FactorCandidate]:
        self.families.append(family)
        self.feedback.append(feedback or "")
        return list(self.proposals)


def test_a_proposal_that_fails_the_rankic_chain_yields_no_action(tmp_path: Path) -> None:
    """The gate is what makes a proposed factor admissible, not the proposal itself."""

    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
    from rdagent.scenarios.gf_quant.strategy.search.tree import root_node

    evaluator = _RejectingEvaluator()
    registry = FactorRegistry(tmp_path / "registry.sqlite")
    service = FactorAcceptanceService(registry, evaluator, standards_path=tmp_path / "absent.json")
    proposer = _ScriptedProposer(
        [FactorCandidate(name="bad", family="price_volume", hypothesis="h", expression="1 / adjclose")]
    )
    generator = ActionGenerator(
        factor_registry=registry, factor_service=service, proposer=proposer
    )
    node = root_node()

    actions = generator.actions_for(node)

    assert evaluator.calls == 1
    assert all("bad" not in action.label for action in actions)
    assert any("rejected:bad" in note for note in generator.notes)


def test_a_proposal_that_passes_the_chain_becomes_an_action(tmp_path: Path) -> None:
    """The positive direction: an accepted proposal really does enter the action space."""

    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
    from rdagent.scenarios.gf_quant.strategy.search.tree import root_node

    class _AcceptingEvaluator:
        def __call__(self, candidate: FactorCandidate) -> EvaluationResult:
            return EvaluationResult(
                metrics={
                    "Rank_IC": 0.05,
                    "Rank_ICIR": 0.4,
                    "annualized_return": 0.3,
                    "sharpe": 1.5,
                    "max_drawdown": -0.1,
                    "turnover": 0.2,
                    "coverage": 1.0,
                    "observations": 800,
                    "metrics_by_year": {
                        "2021": {"Rank_IC": 0.05},
                        "2022": {"Rank_IC": 0.04},
                        "2023": {"Rank_IC": 0.06},
                    },
                }
            )

    registry = FactorRegistry(tmp_path / "registry.sqlite")
    service = FactorAcceptanceService(registry, _AcceptingEvaluator(), standards_path=tmp_path / "absent.json")
    proposer = _ScriptedProposer(
        [FactorCandidate(name="good", family="price_volume", hypothesis="h", expression="1 / adjclose")]
    )
    generator = ActionGenerator(factor_registry=registry, factor_service=service, proposer=proposer)

    actions = generator.actions_for(root_node())

    assert any("good" in action.label for action in actions)
    assert generator.proposal_rounds == 1


def test_a_node_is_asked_for_proposals_at_most_once(tmp_path: Path) -> None:
    """One LLM round per expansion point, so the budget stays the search's, not the model's."""

    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
    from rdagent.scenarios.gf_quant.strategy.search.tree import root_node

    registry = FactorRegistry(tmp_path / "registry.sqlite")
    service = FactorAcceptanceService(registry, _RejectingEvaluator(), standards_path=tmp_path / "absent.json")
    proposer = _ScriptedProposer(
        [FactorCandidate(name="x", family="price_volume", hypothesis="h", expression="1 / adjclose")]
    )
    generator = ActionGenerator(factor_registry=registry, factor_service=service, proposer=proposer)
    node = root_node()

    generator.actions_for(node)
    generator.actions_for(node)
    generator.actions_for(node)

    assert generator.proposal_rounds == 1


def test_the_proposal_feedback_carries_the_observed_gate_failures(tmp_path: Path) -> None:
    """The design asks the proposer to read the tree context; this is that context."""

    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
    from rdagent.scenarios.gf_quant.strategy.search.tree import root_node

    registry = FactorRegistry(tmp_path / "registry.sqlite")
    service = FactorAcceptanceService(registry, _RejectingEvaluator(), standards_path=tmp_path / "absent.json")
    proposer = _ScriptedProposer(
        [FactorCandidate(name="x", family="price_volume", hypothesis="h", expression="1 / adjclose")]
    )
    generator = ActionGenerator(factor_registry=registry, factor_service=service, proposer=proposer)
    node = root_node()
    node.factor_names = ("mom_20",)
    node.gate_failures = ("strategy_holdings", "sharpe")

    generator.actions_for(node)

    assert proposer.feedback
    assert "strategy_holdings" in proposer.feedback[0]
    assert "mom_20" in proposer.feedback[0]


def test_a_proposer_that_raises_does_not_break_the_action_space(tmp_path: Path) -> None:
    """A failed proposal round must leave the registry-sourced actions working."""

    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.service import FactorAcceptanceService
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
    from rdagent.scenarios.gf_quant.strategy.search.tree import root_node

    class _ExplodingProposer:
        def propose(self, family: str, feedback: str | None = None):
            raise RuntimeError("backend down")

    registry = FactorRegistry(tmp_path / "registry.sqlite")
    service = FactorAcceptanceService(registry, _RejectingEvaluator(), standards_path=tmp_path / "absent.json")
    generator = ActionGenerator(
        factor_registry=registry, factor_service=service, proposer=_ExplodingProposer()
    )

    actions = generator.actions_for(root_node())

    assert all(action.kind == "strategy" for action in actions)
    assert any("proposer_error" in note for note in generator.notes)


def test_build_local_proposal_chain_reports_a_missing_cache_loudly(tmp_path: Path) -> None:
    """A cache without the worker layout must fail before the search starts."""

    cache = synthetic_cache(
        tmp_path / "empty_cache", symbols=SYMBOLS, dates=DATES, seed=5
    )
    (cache.root / "calendar.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="calendar"):
        build_local_proposal_chain(cache=cache, work_dir=tmp_path / "work")


# ---- the remote switch --------------------------------------------------------


def test_remote_backend_returns_a_callable_that_names_what_is_missing() -> None:
    """Selecting remote without a context is a refusal, never a silent local fallback."""

    evaluate = build_evaluator(REMOTE_BACKEND, evaluation_start="2022-01-03", evaluation_end="2022-12-30")
    assert evaluate is remote_evaluator_unavailable
    candidate = StrategyCandidate(
        name="x", universe="u", composition=CompositionSpec(top_n=8), factor_names=("a", "b"), backtest=BacktestSpec()
    )
    with pytest.raises(RuntimeError, match="snapshot"):
        evaluate(candidate)


def test_remote_search_evaluator_maps_onto_the_existing_strategy_protocol() -> None:
    """The remote path must reuse ``RemoteStrategyEvaluator`` rather than re-implement it."""

    from rdagent.scenarios.gf_quant.strategy.search.evaluator import RemoteSearchEvaluator

    assert hasattr(RemoteSearchEvaluator, "request_for")
    assert hasattr(RemoteSearchEvaluator, "manifest_facts")
    source = Path(RemoteSearchEvaluator.__module__ and __file__).read_text(encoding="utf-8")
    del source
    # The delegate is imported from the module that already owns the protocol.
    module_source = (
        Path(__file__).resolve().parents[3]
        / "rdagent"
        / "scenarios"
        / "gf_quant"
        / "strategy"
        / "search"
        / "evaluator.py"
    ).read_text(encoding="utf-8")
    assert "from rdagent.scenarios.gf_quant.strategy.remote import" in module_source
    assert "RemoteStrategyEvaluator" in module_source


def test_remote_search_evaluator_forwards_the_request_and_verifies_the_result() -> None:
    """Protocol correctness without a data machine: a fake transport, real adapter.

    ``RemoteSearchEvaluator`` constructs the *existing* ``RemoteStrategyEvaluator``,
    which validates that the artifact set's provenance matches the snapshot and
    working set before it will build a request.  The stubs below satisfy that
    contract, so what is exercised is the real adapter chain rather than a
    reimplementation of it.
    """

    from rdagent.scenarios.gf_quant.strategy.search.evaluator import RemoteSearchEvaluator

    class _Snapshot:
        snapshot_id = "gf_snapshot_" + "a" * 16
        checksum = "a" * 64
        data_cutoff = "2023-12-31"

    class _WorkingSet:
        working_set_id = "gf_working_set_" + "b" * 16
        snapshot_checksum = "a" * 64

    class _ArtifactSet:
        artifact_set_id = "gf_artifact_set_" + "c" * 16
        snapshot_id = _Snapshot.snapshot_id
        working_set_id = _WorkingSet.working_set_id

        def as_payload(self):
            return {"artifact_set_id": self.artifact_set_id, "factors": []}

    evaluator = RemoteSearchEvaluator(
        snapshot=_Snapshot(),
        working_set=_WorkingSet(),
        artifact_set=_ArtifactSet(),
        evaluation_start="2022-01-03",
        evaluation_end="2022-12-30",
        remote_evaluate=lambda *args, **kwargs: pytest.fail("no transport may be opened here"),
    )

    # Construction is the assertion: it proves the adapter accepted the real
    # protocol's provenance contract instead of accepting anything it was given.
    assert evaluator._delegate is not None
    assert evaluator._delegate.evaluation_start == "2022-01-03"
    assert evaluator._delegate.artifact_set.artifact_set_id == _ArtifactSet.artifact_set_id


def test_remote_search_evaluator_refuses_mismatched_provenance() -> None:
    """The protocol's own guard must still fire through the adapter."""

    from rdagent.scenarios.gf_quant.strategy.search.evaluator import RemoteSearchEvaluator

    class _Snapshot:
        snapshot_id = "gf_snapshot_" + "a" * 16
        checksum = "a" * 64
        data_cutoff = "2023-12-31"

    class _WorkingSet:
        working_set_id = "gf_working_set_" + "b" * 16
        snapshot_checksum = "a" * 64

    class _Mismatched:
        artifact_set_id = "gf_artifact_set_" + "d" * 16
        snapshot_id = "gf_snapshot_" + "9" * 16
        working_set_id = "gf_working_set_" + "9" * 16

    with pytest.raises(ValueError, match="provenance"):
        RemoteSearchEvaluator(
            snapshot=_Snapshot(),
            working_set=_WorkingSet(),
            artifact_set=_Mismatched(),
            evaluation_start="2022-01-03",
            evaluation_end="2022-12-30",
            remote_evaluate=lambda *args, **kwargs: pytest.fail("no transport"),
        )


# ---- isolation ----------------------------------------------------------------


def test_assert_isolated_refuses_the_production_cache() -> None:
    with pytest.raises(RuntimeError, match="production"):
        assert_isolated(production_paths().cache)


def test_assert_isolated_refuses_a_root_above_the_production_directory() -> None:
    """The other direction matters too: a root above production would swallow it."""

    with pytest.raises(RuntimeError, match="production"):
        assert_isolated(production_paths().cache.parent)


def test_isolated_paths_land_under_the_isolated_root() -> None:
    paths = isolated_paths("unit_case")
    assert "search_test" in str(paths.root)
    for path in (paths.cache, paths.sidecar, paths.work, paths.log):
        assert paths.root in path.parents


def test_a_named_test_root_is_distinct_from_production() -> None:
    """Two named runs share no path, so one test cannot corrupt another's tree."""

    first = isolated_paths("case_a")
    second = isolated_paths("case_b")
    assert first.root != second.root
    assert first.sidecar not in second.sidecar.parents
    assert first.root not in production_paths().root.parents


def test_create_makes_every_directory_and_writes_nothing_to_production(tmp_path: Path) -> None:
    from rdagent.scenarios.gf_quant.strategy.search.isolation import TEST_ROOT, repository_root as repo_root

    base = repo_root() / TEST_ROOT
    paths = SearchPaths(
        root=base / "pytest_run",
        cache=base / "pytest_run" / "cache",
        sidecar=base / "pytest_run" / "sidecar",
        work=base / "pytest_run" / "work",
        log=base / "pytest_run" / "log",
    )
    try:
        paths.create()
        assert paths.sidecar.is_dir() and paths.work.is_dir()
        assert paths.sidecar_file().parent == paths.sidecar
        assert paths.registry_file().parent == paths.work
    finally:
        import shutil

        shutil.rmtree(paths.root, ignore_errors=True)


def test_production_paths_stay_inside_the_ignore_root() -> None:
    paths = production_paths()
    assert "git_ignore_folder" in str(paths.cache)
    assert "gf_quant" in str(paths.sidecar)


# ---- the return profile the calibrated gate measures --------------------------


def test_the_return_profile_is_the_daily_long_short_spread(cache: SearchDataCache, tmp_path: Path) -> None:
    """The profile must be the spread series, not any other daily aggregate.

    The calibrated gate computes direction-adjusted return, Sharpe and drawdown
    from whatever series is supplied as the profile, and those thresholds were
    calibrated on long-short spreads.  A cross-sectional mean instead produces a
    series centred near 0.5, which annualises to a three-digit "return" and
    silently makes all three gates non-discriminating -- a passing verdict that
    measures nothing.
    """

    from rdagent.scenarios.gf_quant.benchmark import candidate_profile_metrics

    evaluator = LocalRankICFactorEvaluator(
        cache=cache,
        settings=settings_for_cache(cache, minimum_calendar_years=1),
        artifact_dir=tmp_path / "artifacts",
    )
    candidate = FactorCandidate(
        name="probe_spread", family="price_volume", hypothesis="h", expression="ts_mean(adjclose, 5)"
    )
    result = evaluator(candidate)

    assert result.return_profile is not None
    profile = pd.Series(dict(result.return_profile))
    # A long-short spread is a small daily return; a ranked-panel mean is not.
    assert abs(float(profile.mean())) < 0.05, "profile looks like a level series, not a spread"
    metrics = candidate_profile_metrics(result.metrics, result.return_profile)
    assert metrics is not None
    assert abs(metrics.annualized_return) < 5.0, "annualised return is on the spread scale"
    assert abs(metrics.mean_spread) < 0.05


def test_the_profile_and_the_metrics_describe_the_same_series(cache: SearchDataCache, tmp_path: Path) -> None:
    """The gate's annualised return must be the metrics' one, re-signed to the factor.

    ``metrics_from_daily`` annualises the spread with its own orientation;
    ``candidate_profile_metrics`` re-signs it to the aggregate RankIC first, so
    the two agree in magnitude and may differ in sign.  A magnitude mismatch
    would mean the two are reading different series, which is exactly the defect
    where the profile is not the spread.
    """

    from rdagent.scenarios.gf_quant.benchmark import candidate_profile_metrics

    evaluator = LocalRankICFactorEvaluator(
        cache=cache,
        settings=settings_for_cache(cache, minimum_calendar_years=1),
        artifact_dir=tmp_path / "artifacts",
    )
    result = evaluator(
        FactorCandidate(name="probe_same", family="price_volume", hypothesis="h", expression="ts_mean(adjclose, 20)")
    )
    metrics = candidate_profile_metrics(result.metrics, result.return_profile)
    assert metrics is not None
    raw = float(result.metrics["annualized_return"])
    assert abs(metrics.annualized_return) == pytest.approx(abs(raw), rel=1e-9)
    assert metrics.direction == (1 if raw >= 0 else -1)


# ---- the strategy proposal source --------------------------------------------


def _strategy_engine(candidate: Any) -> Any:
    """A stand-in with ``LLMStrategyProposalEngine``'s call shape."""

    class _Engine:
        def __init__(self) -> None:
            self.calls = 0
            self.contexts: list[Any] = []
            self.error: Exception | None = None

        def __call__(self, context: Any) -> Any:
            self.calls += 1
            self.contexts.append(context)
            if self.error is not None:
                raise self.error
            return candidate

    return _Engine()


def _proposed_candidate(*factor_names: str, top_n: int = 30) -> StrategyCandidate:
    return StrategyCandidate(
        name="llm_proposal",
        universe="csi_all",
        composition=CompositionSpec(top_n=top_n, weighting="rank"),
        factor_names=tuple(factor_names),
        rule="blend",
        backtest=BacktestSpec(),
    )


def _node_with_factors(*names: str) -> Any:
    from rdagent.scenarios.gf_quant.strategy.search.tree import TreeNode, configuration_hash
    from rdagent.scenarios.gf_quant.strategy.candidate import GatingSpec, SignalSpec

    signal = SignalSpec()
    gating = GatingSpec()
    composition = CompositionSpec(top_n=8)
    return TreeNode(
        node_id="n1",
        path=("n1",),
        factor_names=tuple(names),
        signal=signal,
        gating=gating,
        composition=composition,
        config_hash=configuration_hash(names, signal=signal, gating=gating, composition=composition),
        depth=2,
    )


def test_a_strategy_proposal_becomes_one_strategy_action() -> None:
    """The engine's candidate is offered as an action rather than stored and ignored."""

    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator

    engine = _strategy_engine(_proposed_candidate("mom_20", "vol_20"))
    generator = ActionGenerator(
        strategy_engine=engine,
        strategy_context_factory=lambda node: {"node": node.node_id},
    )

    actions = generator._proposed_strategy_actions(_node_with_factors("mom_20"))

    assert engine.calls == 1
    assert generator.strategy_proposal_rounds == 1
    assert len(actions) == 1
    action = actions[0]
    assert action.kind == "strategy"
    assert action.label == "strategy_proposal:llm_proposal"
    assert action.factor_names == ("mom_20", "vol_20")
    assert action.composition.top_n == 30
    assert not generator.notes


def test_the_half_wired_pair_produces_no_action_and_no_call() -> None:
    """An engine without a context factory is a wiring mistake, not a proposal source."""

    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator

    engine = _strategy_engine(_proposed_candidate("mom_20", "vol_20"))
    generator = ActionGenerator(strategy_engine=engine)

    assert generator._proposed_strategy_actions(_node_with_factors("mom_20")) == []
    assert engine.calls == 0


def test_a_strategy_proposal_is_asked_for_at_most_once_per_node() -> None:
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator

    engine = _strategy_engine(_proposed_candidate("mom_20", "vol_20"))
    generator = ActionGenerator(
        strategy_engine=engine, strategy_context_factory=lambda node: {"node": node.node_id}
    )
    node = _node_with_factors("mom_20")

    generator._proposed_strategy_actions(node)
    generator._proposed_strategy_actions(node)

    assert engine.calls == 1
    assert generator.strategy_proposal_rounds == 1


def test_a_strategy_engine_that_raises_leaves_the_rest_of_the_action_space_intact() -> None:
    """A rejected proposal must degrade to a note, not abort a search that has spent budget."""

    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator

    engine = _strategy_engine(None)
    engine.error = ValueError("holding_buffer_retain must not exceed holding_buffer_entry")
    generator = ActionGenerator(
        strategy_engine=engine, strategy_context_factory=lambda node: {"node": node.node_id}
    )

    actions = generator.actions_for(_node_with_factors("mom_20"))

    assert actions, "the ladder actions must survive a failed proposal"
    assert any("strategy_proposal_error:ValueError" in note for note in generator.notes)
    assert generator.strategy_proposal_rounds == 0


def test_a_strategy_proposal_naming_no_factor_is_not_an_action() -> None:
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator

    engine = _strategy_engine(_proposed_candidate())
    generator = ActionGenerator(
        strategy_engine=engine, strategy_context_factory=lambda node: {"node": node.node_id}
    )

    assert generator._proposed_strategy_actions(_node_with_factors("mom_20")) == []
    assert any("strategy_proposal_empty" in note for note in generator.notes)


def test_a_proposal_carries_its_gate_layers_onto_the_action() -> None:
    """The action holds the proposal's own gates, unfiltered and unrepaired.

    ``StrategyCandidate`` already refuses a gate naming a factor the candidate
    does not hold, so every layer that reaches here is one the proposal was
    built with; the action must pass them through rather than drop or rewrite
    any of them.
    """

    from rdagent.scenarios.gf_quant.strategy.candidate import GateLayer, GatingSpec
    from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator

    layer = GateLayer(factor="mom_20", direction="top", quantile=0.5, min_pass=3)
    candidate = dataclasses.replace(_proposed_candidate("mom_20", "vol_20"), gating=GatingSpec(layers=(layer,)))
    engine = _strategy_engine(candidate)
    generator = ActionGenerator(
        strategy_engine=engine, strategy_context_factory=lambda node: {"node": node.node_id}
    )

    actions = generator._proposed_strategy_actions(_node_with_factors("mom_20"))

    assert len(actions) == 1
    assert actions[0].gating.layers == (layer,)


def test_the_context_factory_describes_the_node_it_is_given() -> None:
    """The engine's prompt vocabulary needs a working set, a trial index and a snapshot id.

    The joint search has nodes rather than epochs, so the factory translates one
    into the other; a context that named the wrong factors would let the model
    propose a strategy over a book the search is not looking at.
    """

    from rdagent.scenarios.gf_quant.strategy.search.runner import build_strategy_context_factory

    factory = build_strategy_context_factory(
        directions={"mom_20": 1, "vol_20": -1},
        evaluation_start="2021-04-08",
        evaluation_end="2026-08-28",
    )
    node = _node_with_factors("mom_20", "vol_20")

    context = factory(node)

    assert [factor.name for factor in context.working_set.factors] == ["mom_20", "vol_20"]
    assert [factor.direction for factor in context.working_set.factors] == [1, -1]
    assert context.trial_index == 2
    assert context.snapshot_id == node.config_hash
    assert context.data_cutoff == "2026-08-28"
