"""Local-vs-worker parity for the GF strategy layer.

The portfolio pipeline exists twice: the local executor
(``composition.build_target_weights`` + ``backtest.run_portfolio_backtest``)
and the standalone single-file remote worker
(``scripts/gf_quant_strategy_worker.py``).  Nothing else guarantees the two do
not drift, so these tests run the *same* request through both and compare the
resulting per-day start-of-day weights, headline metrics and yearly metrics.

The signal front-end exists on both sides since batch A1: the local pure
functions in ``rdagent/scenarios/gf_quant/strategy/signal.py`` and the worker's
``_apply_transform`` / ``_apply_winsorize`` / ``_apply_standardize`` plus the
rank+blend in ``run_request``.  The rebalance calendar is still derived with
the worker's own ``_rebalance_dates`` (no local counterpart), and the frozen
directions come from the request; parity therefore covers the full
signal -> composition -> backtest chain that exists on both sides.

Everything here is offline: the worker is imported and executed in-process, no
SSH and no data machine.
"""

from __future__ import annotations

import copy
import json
import math
import re
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.artifacts import (
    FactorArtifactSet,
    FactorArtifactVersion,
    artifact_set_id_for,
)
from rdagent.scenarios.gf_quant.strategy.audit import (
    DEFAULT_CONCENTRATION_TOP_K,
    DEFAULT_COST_SENSITIVITY_SCENARIOS,
    DEFAULT_STRESS_MODE,
    ablation_report,
    concentration_report,
    cost_sensitivity_report,
    data_leakage_report,
    ensure_stress_passed,
    feasibility_days,
    feasibility_report,
    leakage_declarations,
    parameter_perturbation_report,
    stress_scenario_report,
    validate_audit_config,
)
from rdagent.scenarios.gf_quant.strategy.backtest import (
    BacktestSpec,
    _apply_min_holding as _local_apply_min_holding,
    run_portfolio_backtest,
)
from rdagent.scenarios.gf_quant.strategy.walk_forward import (
    WALK_FORWARD_METRIC_KEYS,
    validate_walk_forward_config,
    walk_forward_report,
)
from rdagent.scenarios.gf_quant.strategy.candidate import (
    DRAWDOWN_STATES,
    EXECUTION_DATA_KEYS,
    PRICE_LIMIT_SIDES,
    CompositionSpec,
    ExecutionDataSpec,
    GatingSpec,
    IndexMembershipRule,
    ListingAgeRule,
    SignalSpec,
    StrategyCandidate,
    UniverseSpec,
)
from rdagent.scenarios.gf_quant.strategy.composition import (
    _PROJECTION_FEASIBILITY_CEILING as _local_projection_ceiling,
    _apply_constrained_projection as _local_apply_constrained_projection,
    _apply_industry_cap as _local_apply_industry_cap,
    _apply_weight_floor as _local_apply_weight_floor,
    _factor_risk_budget_weights as _local_factor_risk_budget_weights,
    _hierarchical_weights as _local_hierarchical_weights,
    _inverse_volatility_weights as _local_inverse_volatility_weights,
    _projection_params as _local_projection_params,
    _risk_parity_weights as _local_risk_parity_weights,
    _score_transform_weights as _local_score_transform_weights,
    build_buffered_target_row as _local_build_buffered_target_row,
    build_target_weights,
)
from rdagent.scenarios.gf_quant.strategy.execution import (
    apply_execution_layer as _local_apply_execution_layer,
    apply_lot_rounding as _local_apply_lot_rounding,
    apply_participation_cap as _local_apply_participation_cap,
    apply_side_aware_tradability as _local_apply_side_aware_tradability,
    drawdown_state_machine as _local_drawdown_state_machine,
    factor_decay_scales as _local_factor_decay_scales,
    factor_decay_state as _local_factor_decay_state,
    factor_ic_series as _local_factor_ic_series,
    plan_order as _local_plan_order,
    stage_orders as _local_stage_orders,
    volatility_scale as _local_volatility_scale,
)
from rdagent.scenarios.gf_quant.strategy.gating import apply_gating, build_gated_factor_score
from rdagent.scenarios.gf_quant.strategy.remote import RemoteStrategyEvaluator
from rdagent.scenarios.gf_quant.strategy.registry import strategy_id_for
from rdagent.scenarios.gf_quant.strategy.signal import (
    apply_industry_neutralize as _local_apply_industry_neutralize,
    apply_missing_value,
    apply_neutral_rank_fill,
    apply_standardize,
    apply_style_neutralize as _local_apply_style_neutralize,
    apply_transform,
    apply_winsorize,
    blend_member_panels,
    blend_rank_panels,
    blend_regime_panels as _local_blend_regime_panels,
    build_factor_rank_panels,
    build_factor_score,
    interaction_rank_panels,
)
from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set
from rdagent.scenarios.gf_quant.strategy.universe import build_eligible_mask
from scripts import gf_quant_strategy_worker
from scripts.gf_quant_strategy_worker import (
    BACKTEST_KEYS as _worker_backtest_keys,
    DRAWDOWN_STATES as _worker_drawdown_states,
    EXECUTION_DATA_KEYS as _worker_execution_data_keys,
    PRICE_LIMIT_SIDES as _worker_price_limit_sides,
    SIGNAL_KEYS as _worker_signal_keys,
    _PROJECTION_FEASIBILITY_CEILING as _worker_projection_ceiling,
    _apply_constrained_projection,
    _apply_execution_layer,
    _apply_gating,
    _apply_industry_cap,
    _apply_industry_neutralize,
    _apply_lot_rounding,
    _apply_min_holding,
    _apply_missing_value,
    _apply_neutral_rank_fill,
    _apply_participation_cap,
    _apply_side_aware_tradability,
    _apply_standardize,
    _apply_style_neutralize,
    _apply_transform,
    _apply_weight_floor,
    _apply_winsorize,
    _blend_regime_panels,
    _build_buffered_target_row,
    _build_rank_panels,
    _build_universe_eligible,
    _drawdown_state_machine,
    _factor_decay_scales,
    _factor_decay_state,
    _factor_ic_series,
    _factor_risk_budget_weights,
    _hierarchical_weights,
    _interaction_rank_panels,
    _inverse_volatility_weights,
    _load_close,
    _plan_order,
    _projection_params_from_payload,
    _rebalance_dates,
    _risk_parity_weights,
    _score,
    _score_transform_weights,
    _stage_orders,
    _target_weights,
    _validate_execution_data,
    _validate_order_layer,
    _validate_walk_forward,
    _volatility_scale,
    _walk_forward_report,
    _worker_file_digest,
    run_request,
)

EVAL_DATES = pd.bdate_range("2024-01-02", periods=65)
SYMBOLS = tuple(f"S{position}" for position in range(6))
FACTOR_NAMES = ("value", "quality", "momentum")
PRICE_DATES = pd.bdate_range("2023-12-01", periods=90)

# One factor is registered with a negative Rank_IC so the snapshot freezes a
# -1 direction for it: that exercises DirectionMap on both paths.
FACTOR_RANK_IC = {"value": 0.03, "quality": 0.04, "momentum": -0.03}


def _factor_panel(name: str) -> pd.DataFrame:
    """Deterministic wide panel; symbol position drives a stable cross-section."""

    position_of = {symbol: position for position, symbol in enumerate(SYMBOLS)}
    if name == "value":
        return pd.DataFrame(
            {symbol: np.linspace(0.0, 1.0, len(EVAL_DATES)) + 0.1 * position for symbol, position in position_of.items()},
            index=EVAL_DATES,
        )
    if name == "quality":
        return pd.DataFrame(
            {symbol: np.linspace(1.0, 0.0, len(EVAL_DATES)) + 0.05 * (5 - position) for symbol, position in position_of.items()},
            index=EVAL_DATES,
        )
    return pd.DataFrame(
        {symbol: np.cos(0.05 * position + np.linspace(0.0, 2.0, len(EVAL_DATES))) for symbol, position in position_of.items()},
        index=EVAL_DATES,
    )


def _parquet_checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _world_prices(drawdown: bool = False) -> pd.DataFrame:
    """Close panel for the parity world.

    ``drawdown=False`` reproduces the historical monotone path bit for bit.
    ``drawdown=True`` adds a give-back of 0.6% per session across 15 sessions
    (price rows 46..60, i.e. evaluation positions 24..38) and recovers at 0.6%
    per session afterwards; the deepest loss stays near 8%, i.e. below the 10%
    ``halt`` tier, so the risk cases walk ``warning`` / ``reduce`` and back to
    ``normal`` without freezing the NAV in the absorbing halt state.
    """

    symbols = list(SYMBOLS)
    shocks = [1.0] * len(PRICE_DATES)
    if drawdown:
        shock = 1.0
        for day in range(len(PRICE_DATES)):
            if 46 <= day < 61:
                shock *= 0.994
            elif 61 <= day < 81:
                shock *= 1.006
            shocks[day] = shock
    return pd.DataFrame(
        [
            {
                "dt": date,
                "symbol": symbol,
                "value": 100.0 * (1.0 + 0.0005 * day + 0.0008 * position) * shocks[day],
            }
            for day, date in enumerate(PRICE_DATES)
            for position, symbol in enumerate(symbols)
        ]
    )


def _build_world(tmp_path: Path, prices: pd.DataFrame) -> dict:
    """Frozen snapshot / working set / artifact set plus factor panel paths."""

    base = tmp_path / "base"
    base.mkdir()
    symbols = list(SYMBOLS)
    prices.to_parquet(base / "adjclose.parquet")

    registry = FactorRegistry(tmp_path / "factors.sqlite")
    versions = []
    for position, name in enumerate(FACTOR_NAMES):
        original = tmp_path / f"old_{name}.parquet"
        pd.DataFrame(0.0, index=EVAL_DATES, columns=symbols).to_parquet(original)
        registry.upsert(
            FactorRecord(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                status="accepted",
                metrics={"Rank_IC": FACTOR_RANK_IC[name]},
                artifact_path=str(original),
                artifact_checksum=_parquet_checksum(original),
            )
        )
        path = tmp_path / f"{name}.parquet"
        _factor_panel(name).to_parquet(path)
        versions.append(
            FactorArtifactVersion(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                artifact_path=str(path),
                artifact_checksum=_parquet_checksum(path),
                data_start=EVAL_DATES[0].strftime("%Y-%m-%d"),
                data_end=EVAL_DATES[-1].strftime("%Y-%m-%d"),
                rows=len(EVAL_DATES),
                columns=len(symbols),
            )
        )
    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")
    working = build_factor_working_set(snapshot, core_limit=3, conditional_limit=0)
    start, end = EVAL_DATES[0].strftime("%Y-%m-%d"), EVAL_DATES[-1].strftime("%Y-%m-%d")
    artifact_set = FactorArtifactSet(
        artifact_set_id=artifact_set_id_for(snapshot.snapshot_id, working.working_set_id, start, end, versions),
        snapshot_id=snapshot.snapshot_id,
        working_set_id=working.working_set_id,
        evaluation_start=start,
        evaluation_end=end,
        created_at="2024-01-01T00:00:00Z",
        factors=tuple(versions),
    )
    settings = SimpleNamespace(
        remote_work_dir=str(tmp_path).replace("\\", "/"),
        data_base_path=str(base),
        evaluation_close_field="adjclose",
        data_host="host",
        data_port=22,
        data_user="user",
        data_password_env="PASSWORD",
        data_host_key_sha256="",
        remote_python="python",
    )
    return {
        "tmp_path": tmp_path,
        "base": base,
        "evaluator": RemoteStrategyEvaluator(
            snapshot=snapshot,
            working_set=working,
            artifact_set=artifact_set,
            evaluation_start=start,
            evaluation_end=end,
            settings=settings,
        ),
        "start": start,
        "end": end,
    }


@pytest.fixture
def world(tmp_path: Path):
    return _build_world(tmp_path, _world_prices())


@pytest.fixture
def drawdown_world(tmp_path: Path):
    """Same world, but the close panel really draws down and recovers."""

    return _build_world(tmp_path, _world_prices(drawdown=True))


def _local_portfolio(
    world: dict,
    request: dict,
    composition: CompositionSpec,
    signal: SignalSpec,
    gating: GatingSpec | None = None,
    backtest: BacktestSpec | None = None,
    execution_trace: list | None = None,
):
    """Run the local composition+backtest path on the local signal front-end.

    With an enabled ``gating`` spec the score comes from the gated front-end
    (rank pipeline + blend + cascade); without one it is the untouched
    default path.  ``backtest`` carries the full execution spec (lag plus the
    drift-band / turnover-shrinkage operators); it defaults to lag 1 only for
    the legacy cases.  ``execution_trace`` is the optional observation sink the
    FeasibilityAudit reads; omitting it keeps the pre-change call shape.
    """

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    symbols = sorted(set(close.columns))

    # The local front-end (signal.build_factor_score) reindexes each frozen
    # factor panel onto the same (rebalance dates x common symbols) grid, runs
    # transform -> winsorize -> standardize -> direction -> percentile rank and
    # blends -- the mirror image of the worker's run_request.
    symbols = sorted(set(close.columns))
    score = _score_panel_for(world, request, signal, rebalance_dates, symbols, gating=gating)

    daily_returns = close.reindex(columns=symbols).pct_change(fill_method=None).loc[
        pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])
    ]
    benchmark = daily_returns.mean(axis=1, skipna=True)
    # The inverse_volatility weighting consumes the realised-return panel, so
    # it is always handed to the target builder; other schemes ignore it.
    targets = build_target_weights(composition, score, returns=daily_returns)
    # HoldingBuffer / DiversitySelector need the score panel and the executed
    # holdings state inside the main loop, so they are passed only when
    # enabled; the disabled path keeps the pre-change call shape exactly.
    has_selection_operators = composition.holding_buffer_entry is not None or (
        composition.diversity_max_correlation is not None
    )
    # The batch-6 execution layer (8.6-8.9) reads the injected execution data
    # panels.  They travel in the candidate payload exactly like the batch-E
    # universe rules, so the request is the single source of truth here too and
    # an absent section keeps the pre-change call shape.
    execution_payload = request["candidate"].get("execution_data")
    result = run_portfolio_backtest(
        targets,
        daily_returns,
        spec=backtest if backtest is not None else BacktestSpec(signal_lag=int(request["candidate"]["backtest"]["signal_lag"])),
        benchmark_returns=benchmark,
        composition=composition if has_selection_operators else None,
        score=score if has_selection_operators else None,
        execution_data=ExecutionDataSpec.from_payload(execution_payload) if execution_payload is not None else None,
        execution_trace=execution_trace,
    )
    return result


def _score_panel_for(
    world: dict,
    request: dict,
    signal: SignalSpec,
    rebalance_dates: pd.DatetimeIndex,
    symbols: list[str],
    gating: GatingSpec | None = None,
) -> pd.DataFrame:
    """The local front-end score panel on the (rebalance dates x symbols) grid."""

    factor_names = list(request["candidate"]["factor_names"])
    factor_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    panels = {name: pd.read_parquet(factor_items[name]["artifact_path"]) for name in factor_names}
    directions = {name: int(request["factor_directions"][name]) for name in factor_names}
    # Candidate-pool filters (batch E): the eligible mask rides in the
    # candidate payload and feeds the rank pipeline exactly like the worker
    # does; with the section absent the mask is None and nothing changes.
    universe_payload = request["candidate"].get("universe_rules")
    eligible = None
    if universe_payload is not None:
        eligible = build_eligible_mask(
            UniverseSpec.from_payload(universe_payload),
            factor_panels=panels,
            index=rebalance_dates,
            columns=symbols,
        )
    # FactorDecayMonitor (batch 6, roadmap 9.5) reads the engine's own realised
    # daily-return panel and aligns the IC window with the backtest signal lag,
    # exactly like the worker does.  The panel is only loaded while the monitor
    # is enabled, so the disabled path keeps its previous I/O pattern.
    returns = None
    lag = 1
    if signal.decay_window is not None:
        returns = _returns_panel_for(world)
        lag = int(request["candidate"]["backtest"].get("signal_lag") or 1)
    if gating is not None and gating.layers:
        return build_gated_factor_score(
            panels,
            directions,
            signal,
            gating,
            index=rebalance_dates,
            columns=symbols,
            eligible=eligible,
            returns=returns,
            lag=lag,
        )
    return build_factor_score(
        panels,
        directions,
        signal,
        index=rebalance_dates,
        columns=symbols,
        eligible=eligible,
        returns=returns,
        lag=lag,
    )


def _returns_panel_for(world: dict) -> pd.DataFrame:
    """The realised daily-return panel the worker derives from the close data."""

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    symbols = sorted(set(close.columns))
    return close.reindex(columns=symbols).pct_change(fill_method=None).loc[
        pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])
    ]


def _assert_metrics_parity(local: dict, remote: dict, label: str) -> None:
    assert set(local) == set(remote), f"{label}: metric key sets drifted: local-only={set(local) - set(remote)}, worker-only={set(remote) - set(local)}"
    for key, value in local.items():
        assert float(value) == pytest.approx(float(remote[key]), rel=1e-9, abs=1e-12), f"{label}.{key}"


def _assert_end_to_end_parity(local, payload: dict, portfolio_artifact_path: Path) -> None:
    """Shared local-vs-worker comparison for one executed request."""

    # Per-day start-of-day weights: the worker writes exactly this frame to the
    # portfolio artifact, so it is the shared observable for both engines.
    worker_weights = pd.read_parquet(portfolio_artifact_path)
    assert list(worker_weights.columns) == list(local.weights.columns)
    assert list(worker_weights.index) == list(local.weights.index)
    np.testing.assert_allclose(
        worker_weights.to_numpy(dtype=float),
        local.weights.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-15,
        err_msg="start-of-day weights drifted between local and worker",
    )

    # NOTE: no cap assertion on these frames -- the artifact holds *drifted*
    # start-of-day weights, and drift between rebalances legitimately pushes a
    # holding above the cap on both sides identically.  Cap semantics are
    # covered by the target-weight tests in test_strategy_layer.py.

    _assert_metrics_parity(dict(local.metrics), dict(payload["metrics"]), "metrics")
    assert set(local.metrics_by_year) == set(payload["metrics_by_year"])
    for year, year_local in local.metrics_by_year.items():
        _assert_metrics_parity(dict(year_local), dict(payload["metrics_by_year"][year]), f"metrics_by_year[{year}]")

    # The worker's daily net-return profile is rounded to 10 decimals
    # (scripts/gf_quant_strategy_worker.py return_profile), so compare with a
    # matching tolerance instead of exact equality.
    profile_dates = [pd.Timestamp(date) for date in payload["return_profile"]["dates"]]
    assert profile_dates == list(local.daily_returns.index)
    np.testing.assert_allclose(
        np.asarray(payload["return_profile"]["returns"], dtype=float),
        local.daily_returns.to_numpy(dtype=float),
        atol=5e-11,
        rtol=0.0,
        err_msg="daily net-return profile drifted between local and worker",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("weighting", "weight_cap", "top_n", "signal_lag", "signal_payload"),
    [
        ("equal", None, 4, 1, None),
        ("rank", None, 4, 1, None),
        ("proportional", None, 4, 1, None),
        pytest.param("equal", 0.2, 4, 1, None, id="equal-cap-cash-residual"),  # 4 x 0.25 > 0.2: everything caps, cash stays.
        pytest.param("proportional", 0.4, 4, 1, None, id="proportional-cap-water-filling"),
        pytest.param("equal", None, 4, 2, None, id="signal-lag-2"),
        pytest.param(
            "equal",
            None,
            4,
            1,
            {"winsorize_lower": 0.1, "winsorize_upper": 0.9, "winsorize_min_samples": 4},
            id="winsorize",
        ),
        pytest.param("equal", None, 4, 1, {"standardize": "zscore"}, id="standardize-zscore"),
        pytest.param("equal", None, 4, 1, {"standardize": "robust_z"}, id="standardize-robust-z"),
        pytest.param("equal", None, 4, 1, {"standardize": "rank_z"}, id="standardize-rank-z"),
        pytest.param("equal", None, 4, 1, {"transform": "signed_log"}, id="transform-signed-log"),
        pytest.param(
            "equal",
            None,
            4,
            1,
            {
                "transform": "linear",
                "transform_scale": 1.5,
                "transform_shift": 0.25,
                "winsorize_lower": 0.05,
                "winsorize_upper": 0.95,
                "winsorize_min_samples": 4,
                "standardize": "robust_z",
            },
            id="combo-winsorize-standardize-transform",
        ),
    ],
)
def test_local_and_worker_agree_end_to_end(world, weighting, weight_cap, top_n, signal_lag, signal_payload) -> None:
    tmp_path: Path = world["tmp_path"]
    suffix = f"{weighting}" + (f"-cap{weight_cap}" if weight_cap is not None else "") + f"-lag{signal_lag}"
    candidate = StrategyCandidate(
        name=f"parity_{suffix}",
        universe="csi_all",
        composition=CompositionSpec(top_n=top_n, weighting=weighting, weight_cap=weight_cap),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=signal_lag),
        signal=SignalSpec.from_payload({} if signal_payload is None else signal_payload),
    )

    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio.parquet")
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("gating_payload",),
    [
        pytest.param(
            # value ranks increase with symbol position: keeps S4, S5.
            {"layers": [{"factor": "value", "direction": "top", "quantile": 0.75}]},
            id="quantile-top",
        ),
        pytest.param(
            # quality ranks decrease with symbol position: keeps S3, S4, S5.
            {"layers": [{"factor": "quality", "direction": "bottom", "quantile": 0.5}]},
            id="quantile-bottom",
        ),
        pytest.param(
            # Intersection of the value top half with the quality bottom half
            # across two factors: a true two-layer cascade.
            {
                "layers": [
                    {"factor": "value", "direction": "top", "quantile": 0.5},
                    {"factor": "quality", "direction": "bottom", "quantile": 0.5},
                ]
            },
            id="cascade-two-layers",
        ),
    ],
)
def test_gating_cases_are_discriminating_and_agree_end_to_end(world, gating_payload) -> None:
    """QuantileGate / CascadedFactorGating end-to-end parity (roadmap 5.1/5.2).

    Gating has real discriminative power end to end: non-surviving symbols
    become NaN in the score and therefore never receive a weight, so the
    holding count drops below ``top_n``.  The request without a ``gating`` key
    still runs the untouched default path (covered by the cases above).
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_gating",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        gating=GatingSpec.from_payload(gating_payload),
    )

    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_gating.parquet")
    request_path, result_path = tmp_path / "request_gating.json", tmp_path / "result_gating.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, candidate.gating, candidate.backtest)

    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Discrimination: with 6 symbols and top_n=4, the surviving set is smaller
    # than top_n on every case below, so no more than that many names ever
    # hold a weight.
    assert int((local.weights != 0).sum(axis=1).max()) < 4


def test_signal_stages_match_bit_for_bit_between_local_and_worker() -> None:
    """Feed the same panel straight into both implementations of each stage.

    The end-to-end cases above cannot discriminate the monotone signal stages
    (their effect is absorbed by the percentile rank), so the signal pipeline
    is guarded by this direct comparison instead.
    """

    panel = pd.DataFrame(
        {
            "A": [1.0, -2.0, np.nan, 4.0],
            "B": [0.5, 1.5, 2.5, 99.0],
            "C": [-1.0, 0.0, 3.0, 10.0],
            "D": [7.0, 8.0, 9.0, np.nan],
        },
        index=pd.bdate_range("2024-01-02", periods=4),
    )

    transform_cases = [
        # Missing keys must fall back to the same neutral defaults on both
        # sides (this is the request {"transform": "linear"} regression).
        {},
        {"transform": "linear"},
        {"transform": "linear", "transform_scale": 2.0, "transform_shift": -1.0},
        {"transform": "signed_log"},
    ]
    for signal_payload in transform_cases:
        spec = SignalSpec.from_payload(signal_payload)
        np.testing.assert_array_equal(
            apply_transform(panel, spec).to_numpy(dtype=float),
            _apply_transform(panel, signal_payload).to_numpy(dtype=float),
        )

    winsorize_cases = [
        {"winsorize_lower": 0.25, "winsorize_upper": 0.75},
        {"winsorize_lower": 0.1, "winsorize_upper": 0.9, "winsorize_min_samples": 3},
    ]
    for signal_payload in winsorize_cases:
        spec = SignalSpec.from_payload(signal_payload)
        np.testing.assert_array_equal(
            apply_winsorize(panel, spec).to_numpy(dtype=float),
            _apply_winsorize(panel, signal_payload).to_numpy(dtype=float),
        )

    standardize_cases = [
        {"standardize": "zscore"},
        {"standardize": "rank_z"},
        {"standardize": "robust_z"},
        {"standardize": "robust_z", "standardize_min_std": 0.01},
    ]
    for signal_payload in standardize_cases:
        spec = SignalSpec.from_payload(signal_payload)
        np.testing.assert_array_equal(
            apply_standardize(panel, spec).to_numpy(dtype=float),
            _apply_standardize(panel, signal_payload).to_numpy(dtype=float),
        )


@pytest.mark.parametrize(
    ("signal_payload", "match"),
    [
        ({"transform": "sqrt"}, "sqrt transform requires non-negative"),
        ({"transform": "log"}, "log transform requires strictly positive"),
        ({"winsorize_lower": 0.1, "winsorize_upper": 0.9, "winsorize_min_samples": 100}, "at least 100"),
        ({"standardize": "zscore", "standardize_min_std": 5.0}, "dispersion of at least 5"),
    ],
)
def test_signal_failures_fail_closed_on_both_sides(world, signal_payload, match) -> None:
    """Out-of-domain data must raise on the local front-end and in the worker alike.

    The fixture's value panel contains a zero and the momentum panel negative
    values (sqrt/log domain violations); 6 symbols make min_samples=100 and a
    dispersion floor of 5 unreachable.
    """

    candidate = StrategyCandidate(
        name="parity_signal_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        signal=SignalSpec.from_payload(signal_payload),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_fail.parquet")
    request_path, result_path = tmp_path / "request_fail.json", tmp_path / "result_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        run_request(request_path, result_path)

    signal = SignalSpec.from_payload(signal_payload)
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    symbols = sorted(set(close.columns))
    factor_names = list(request["candidate"]["factor_names"])
    factor_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    panels = {name: pd.read_parquet(factor_items[name]["artifact_path"]) for name in factor_names}
    directions = {name: int(request["factor_directions"][name]) for name in factor_names}
    with pytest.raises(ValueError, match=match):
        build_factor_score(panels, directions, signal, index=rebalance_dates, columns=symbols)


def test_gating_matches_bit_for_bit_between_local_and_worker() -> None:
    """Feed the same score/rank panels straight into both gating implementations.

    The end-to-end cases cover selection discrimination, but ties and exact
    threshold boundaries deserve an explicit per-cell comparison: this runs
    the local cascade and the worker's ``_apply_gating`` on identical panels
    over tie, boundary, NaN, all-pass and multi-layer groups.
    """

    dates = pd.bdate_range("2024-01-02", periods=2)
    columns = ["A", "B", "C", "D", "E"]
    score = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0, 50.0], [1.0, 2.0, np.nan, 4.0, 5.0]],
        index=dates,
        columns=columns,
    )
    # Row 0 ties exactly on the 0.4 threshold; row 1 has a NaN rank and a
    # 1.0 / 0.2 boundary pair.
    ranks = {
        "f1": pd.DataFrame(
            [[0.2, 0.4, 0.4, 0.6, 1.0], [1.0, 0.8, 0.6, np.nan, 0.2]],
            index=dates,
            columns=columns,
        ),
        "f2": pd.DataFrame(
            [[0.9, 0.7, 0.5, 0.3, 0.1], [0.2, 0.4, 0.5, 0.6, 0.8]],
            index=dates,
            columns=columns,
        ),
    }

    cases = [
        # Inclusive comparison on the average rank: the tie at 0.4 survives
        # the top gate from above and the bottom gate from below.
        {"layers": [{"factor": "f1", "direction": "top", "quantile": 0.4}]},
        {"layers": [{"factor": "f1", "direction": "bottom", "quantile": 0.4}]},
        # The threshold itself is a kept rank on either side.
        {"layers": [{"factor": "f1", "direction": "top", "quantile": 0.6}]},
        # NaN ranks are excluded in both directions on both sides.
        {"layers": [{"factor": "f1", "direction": "bottom", "quantile": 0.5}]},
        # Zero-exclusion group: everyone survives.
        {"layers": [{"factor": "f1", "direction": "bottom", "quantile": 0.9}]},
        # Multi-layer cascade with an explicit min_pass.
        {
            "layers": [
                {"factor": "f1", "direction": "top", "quantile": 0.5},
                {"factor": "f2", "direction": "bottom", "quantile": 0.5, "min_pass": 2},
            ]
        },
    ]
    for gating_payload in cases:
        spec = GatingSpec.from_payload(gating_payload)
        np.testing.assert_array_equal(
            apply_gating(score, ranks, spec).to_numpy(dtype=float),
            _apply_gating(score, ranks, gating_payload).to_numpy(dtype=float),
        )


def test_gating_failures_fail_closed_on_both_sides(world) -> None:
    """A min_pass the surviving set cannot reach must raise on both sides.

    Six symbols make min_pass=10 unreachable on every date; the worker raises
    while building the strategy weights and the local gated front-end raises
    the same message on the same panels.
    """

    gating_payload = {"layers": [{"factor": "value", "direction": "top", "quantile": 0.75, "min_pass": 10}]}
    candidate = StrategyCandidate(
        name="parity_gating_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        gating=GatingSpec.from_payload(gating_payload),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_gating_fail.parquet")
    request_path, result_path = tmp_path / "request_gating_fail.json", tmp_path / "result_gating_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="fewer than 10 surviving names"):
        run_request(request_path, result_path)

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    symbols = sorted(set(close.columns))
    factor_names = list(request["candidate"]["factor_names"])
    factor_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    panels = {name: pd.read_parquet(factor_items[name]["artifact_path"]) for name in factor_names}
    directions = {name: int(request["factor_directions"][name]) for name in factor_names}
    with pytest.raises(ValueError, match="fewer than 10 surviving names"):
        build_gated_factor_score(
            panels,
            directions,
            SignalSpec(),
            GatingSpec.from_payload(gating_payload),
            index=rebalance_dates,
            columns=symbols,
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("composition", "label"),
    [
        # Pure floor on rank weights [0.4, 0.3, 0.2, 0.1]: D is raised, the
        # donors fund it.
        (CompositionSpec(top_n=4, weighting="rank", weight_floor=0.15), "rank-floor"),
        # Cap and floor both active: the floor runs after the cap and must not
        # re-breach it (roadmap 6.2 order rule).
        (CompositionSpec(top_n=4, weighting="rank", weight_cap=0.3, weight_floor=0.15), "cap-floor-stack"),
        (CompositionSpec(top_n=4, weighting="score_transform", score_transform_kind="sigmoid", score_transform_slope=4.0, score_transform_cut=0.5), "score-transform-sigmoid"),
        (CompositionSpec(top_n=4, weighting="score_transform", score_transform_kind="truncated", score_transform_cut=0.3), "score-transform-truncated"),
        (CompositionSpec(top_n=4, weighting="score_transform", score_transform_kind="linear", score_transform_slope=2.0), "score-transform-linear"),
    ],
)
def test_floor_and_score_transform_agree_end_to_end(world, composition: CompositionSpec, label: str) -> None:
    """SingleNameFloor (6.2) / ScoreTransformWeight (7.4) end-to-end parity."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name=f"parity_{label}",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )

    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{label}.parquet")
    request_path, result_path = tmp_path / f"request_{label}.json", tmp_path / f"result_{label}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))


def test_floor_and_score_transform_match_bit_for_bit_between_local_and_worker() -> None:
    """Feed identical weight/score series into both implementations.

    The end-to-end cases exercise the pipeline, but floor boundaries (a weight
    exactly on the floor), zero-weight names and transform degeneracies deserve
    a direct per-cell comparison, mirroring the gating test above.
    """

    floor_cases = [
        # Boundary weight exactly on the floor, zero names, everyone below.
        (pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}), 0.15),
        (pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}), 0.2),
        (pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}), 0.1),
        (pd.Series({"A": 0.1, "B": 0.1, "C": 0.0, "D": 0.0}), 0.25),
        # Deficit larger than the donor excess: donors freeze at the floor and
        # the remainder stays as cash.
        (pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}), 0.28),
    ]
    for weights, floor in floor_cases:
        np.testing.assert_array_equal(
            _local_apply_weight_floor(weights, floor).to_numpy(dtype=float),
            _apply_weight_floor(weights.copy(), floor).to_numpy(dtype=float),
        )

    transform_cases = [
        ("linear", 2.0, 0.0, pd.Series({"A": 0.9, "B": 0.6, "C": 0.3})),
        ("linear", 0.5, 0.0, pd.Series({"A": 0.9, "B": 0.6, "C": 0.3})),
        ("truncated", 1.0, 0.5, pd.Series({"A": 0.9, "B": 0.7, "C": 0.3, "D": 0.1})),
        ("sigmoid", 4.0, 0.5, pd.Series({"A": 0.9, "B": 0.6, "C": 0.4, "D": 0.1})),
        ("sigmoid", 1.0, 0.0, pd.Series({"A": 0.3, "B": -0.2, "C": 0.9})),
    ]
    for kind, slope, cut, scores in transform_cases:
        spec = CompositionSpec(
            top_n=len(scores),
            weighting="score_transform",
            score_transform_kind=kind,  # type: ignore[arg-type]
            score_transform_slope=slope,
            score_transform_cut=cut,
        )
        worker_composition = {
            "top_n": len(scores),
            "weighting": "score_transform",
            "score_transform_kind": kind,
            "score_transform_slope": slope,
            "score_transform_cut": cut,
        }
        np.testing.assert_array_equal(
            _local_score_transform_weights(spec, scores).to_numpy(dtype=float),
            _score_transform_weights(scores.copy(), worker_composition).to_numpy(dtype=float),
        )

    # Degenerate inputs fail closed with the same message on both sides.
    for kind, slope, cut, scores in [
        ("linear", 2.0, 0.0, pd.Series({"A": 1.0, "B": 0.0})),
        ("truncated", 1.0, 2.0, pd.Series({"A": 1.0, "B": 0.5})),
        ("sigmoid", 1e6, 0.5, pd.Series({"A": 0.0, "B": 1.0})),
    ]:
        spec = CompositionSpec(
            top_n=len(scores),
            weighting="score_transform",
            score_transform_kind=kind,  # type: ignore[arg-type]
            score_transform_slope=slope,
            score_transform_cut=cut,
        )
        worker_composition = {
            "top_n": len(scores),
            "weighting": "score_transform",
            "score_transform_kind": kind,
            "score_transform_slope": slope,
            "score_transform_cut": cut,
        }
        with pytest.raises(ValueError) as local_error:
            _local_score_transform_weights(spec, scores)
        with pytest.raises(ValueError) as worker_error:
            _score_transform_weights(scores.copy(), worker_composition)
        assert str(local_error.value) == str(worker_error.value)


@pytest.mark.parametrize(
    ("composition", "match"),
    [
        # Freezing every rank holding at the floor needs 1.4 of budget.
        (
            CompositionSpec(top_n=4, weighting="rank", weight_floor=0.35),
            "weight_floor is infeasible",
        ),
        # The rank-blend scores never exceed 1, so this cut truncates the whole
        # selection on every date.
        (
            CompositionSpec(top_n=4, weighting="score_transform", score_transform_kind="truncated", score_transform_cut=1.1),
            "at least one score above the cut",
        ),
    ],
)
def test_floor_and_score_transform_failures_fail_closed_on_both_sides(
    world, composition: CompositionSpec, match: str
) -> None:
    candidate = StrategyCandidate(
        name="parity_floor_transform_fail_closed",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_floor_fail.parquet")
    request_path, result_path = tmp_path / "request_floor_fail.json", tmp_path / "result_floor_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        run_request(request_path, result_path)

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    symbols = sorted(set(close.columns))
    factor_names = list(request["candidate"]["factor_names"])
    factor_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    panels = {name: pd.read_parquet(factor_items[name]["artifact_path"]) for name in factor_names}
    directions = {name: int(request["factor_directions"][name]) for name in factor_names}
    score = build_factor_score(panels, directions, SignalSpec(), index=rebalance_dates, columns=symbols)
    with pytest.raises(ValueError, match=match):
        build_target_weights(composition, score)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("backtest", "label"),
    [
        # The portfolio band (0.3 of half-L1 turnover incl. cash) lets the
        # small drift-only deviations skip a rebalance while full or partial
        # switches still trade.
        (BacktestSpec(signal_lag=1, drift_band_portfolio=0.3), "drift-band-portfolio-skip"),
        # A band no deviation can ever cross would freeze the book, but a
        # forced cadence of 1 trading day overrides it: every rebalance is
        # executed and the result must equal the untouched baseline.
        (BacktestSpec(signal_lag=1, drift_band_name=0.5, drift_band_forced_days=1), "drift-band-forced"),
        (BacktestSpec(signal_lag=1, turnover_shrinkage=0.5), "turnover-shrink-soft"),
        (BacktestSpec(signal_lag=1, turnover_max_step=0.1), "turnover-shrink-clip"),
        # Both execution operators stacked: the band decides, the shrink sizes.
        (BacktestSpec(signal_lag=1, drift_band_portfolio=0.3, turnover_shrinkage=0.5), "band-plus-shrinkage"),
        # Stacked with the batch A3 composition operators: the floor and the
        # cap keep judging target weights while the carried state feeds the
        # band/shrinkage path.
        (
            BacktestSpec(signal_lag=1, drift_band_name=0.05, turnover_shrinkage=0.3),
            "band-shrinkage-floor-stack",
        ),
    ],
)
def test_drift_band_and_shrinkage_agree_end_to_end(world, backtest: BacktestSpec, label: str) -> None:
    """DriftBandRebalance (8.3) / TurnoverShrinkage (8.5) end-to-end parity.

    Both operators live inside the two backtest main loops (local
    ``run_portfolio_backtest`` and the worker's ``_backtest``), so the
    per-day start-of-day weight comparison below is what actually guards the
    carried-state path; the direct helper comparison lives in the unit suite.
    """

    tmp_path: Path = world["tmp_path"]
    composition = (
        CompositionSpec(top_n=4, weighting="rank", weight_floor=0.15)
        if label == "band-shrinkage-floor-stack"
        else CompositionSpec(top_n=4, weighting="equal")
    )
    candidate = StrategyCandidate(
        name=f"parity_{label}",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=backtest,
    )

    request = world["evaluator"].request_for(candidate)
    assert request["candidate"]["backtest"] == candidate.backtest.as_payload()
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{label}.parquet")
    request_path, result_path = tmp_path / f"request_{label}.json", tmp_path / f"result_{label}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Discrimination checks against the untouched baseline run (same panels,
    # default execution spec) so a case that accidentally stops exercising the
    # operator fails loudly instead of passing as vacuous parity.
    baseline = _local_portfolio(world, request, composition, candidate.signal)
    if label == "drift-band-portfolio-skip":
        assert local.turnover.sum() < baseline.turnover.sum()
    elif label == "drift-band-forced":
        np.testing.assert_allclose(local.weights.to_numpy(), baseline.weights.to_numpy(), atol=1e-12)
    elif label == "turnover-shrink-soft":
        assert 0.0 < local.turnover.sum() < baseline.turnover.sum()
    elif label == "turnover-shrink-clip":
        traded = local.turnover[local.turnover > 0]
        assert len(traded) > 0
        assert float(traded.max()) <= 0.1 + 1e-12


@pytest.mark.parametrize(
    ("payload_override", "message"),
    [
        ({"turnover_shrinkage": 1.0}, "turnover_shrinkage must be in [0, 1) when provided"),
        ({"turnover_shrinkage": -0.1}, "turnover_shrinkage must be in [0, 1) when provided"),
        ({"turnover_max_step": 0.0}, "turnover_max_step must be in (0, 1] when provided"),
        ({"turnover_max_step": 1.5}, "turnover_max_step must be in (0, 1] when provided"),
        ({"drift_band_name": 1.5}, "drift_band_name must be in (0, 1] when provided"),
        ({"drift_band_portfolio": 0.0}, "drift_band_portfolio must be in (0, 1] when provided"),
        ({"drift_band_forced_days": 0}, "drift_band_forced_days must be at least 1 when provided"),
        # A fractional cadence would be silently int()-normalised by one side
        # and read raw by the other; it must fail closed on both instead.
        ({"drift_band_forced_days": 2.5}, "drift_band_forced_days must be a whole number when provided"),
        (
            {"drift_band_forced_days": 4},
            "drift_band_forced_days requires drift_band_name or drift_band_portfolio",
        ),
        # Unknown keys fail closed on the worker only: the local from_payload
        # deliberately tolerates legacy junk keys (see the round-trip test in
        # test_strategy_layer.py), so there is no local counterpart.
        ({"drift_band_typo": 0.1}, "candidate backtest has unknown keys"),
    ],
)
def test_execution_band_failures_fail_closed_on_both_sides(world, payload_override: dict, message: str) -> None:
    """Invalid execution-adjustment parameters raise ValueError on both sides.

    The worker sees the parameter through the request payload (mutated after
    ``request_for``), the local side through ``BacktestSpec`` validation; the
    messages must be identical where both sides implement the check.
    """

    candidate = StrategyCandidate(
        name="parity_execution_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_exec_fail.parquet")
    request["candidate"]["backtest"].update(payload_override)
    request_path, result_path = tmp_path / "request_exec_fail.json", tmp_path / "result_exec_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    if payload_override == {"drift_band_typo": 0.1}:
        return
    (key, value), = payload_override.items()
    with pytest.raises(ValueError) as local_error:
        BacktestSpec(signal_lag=1, **{key: value})
    assert str(local_error.value) == message

# ---------------------------------------------------------------------------
# Batch B1: signal DAG operators (roadmap 4.5 / 4.9 / 4.10 / 4.11).
# ---------------------------------------------------------------------------

# Reference captured from the pre-batch code: the default candidate (no
# signal/gating/execution operators at all) run through the worker and the
# local executor.  If any new signal stage leaks into the disabled default
# path, one of these assertions fails.
REDLINE_METRICS: dict = {
    "annual_buy_turnover": 3.876969173104735,
    "annual_sell_turnover": 4.609618166010722e-05,
    "annual_turnover": 3.8769691731047367,
    "annualized_return": 0.11663351767543924,
    "annualized_vol": 0.002162936432313754,
    "average_transaction_cost": 1.5384981227326967e-05,
    "avg_holdings": 3.9384615384615387,
    "buy_cost_bps": 10.0,
    "buy_turnover": 0.01538479830597117,
    "cagr": 0.12367462815935792,
    "excess_annualized_return": -0.005885399230863315,
    "information_ratio": -2.6972196712681162,
    "max_drawdown": -0.0005068056922616915,
    "observations": 65,
    "round_trip_cost_bps": 20.0,
    "sell_cost_bps": 10.0,
    "sell_turnover": 1.8292135579407625e-07,
    "sharpe": 53.923691853797614,
    "signal_lag": 1,
    "total_return": 0.0305333550224558,
    "total_transaction_cost": 0.001000023779776253,
    "tracking_error": 0.0021820244355908376,
    "turnover": 0.015384798305971177,
    "win_rate": 0.9692307692307692,
}
REDLINE_WEIGHTS_SHA256 = "d8a27cc91c3ccc3346b42234a673ee3c3e9e77f2a90ca9fc202cb6a3c969ec4c"


@pytest.mark.slow
def test_disabled_signal_dag_matches_the_pre_change_reference(world, tmp_path: Path) -> None:
    """Regression red line: with every new field disabled the end-to-end
    outputs (worker metrics and both engines' start-of-day weights) are
    bit-for-bit the pre-batch reference."""

    candidate = StrategyCandidate(
        name="parity_redline",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    # Fact 10: disabled signal-DAG fields stay out of the canonical payload.
    assert set(candidate.signal.as_payload()) == {
        "transform",
        "transform_scale",
        "transform_shift",
        "winsorize_lower",
        "winsorize_upper",
        "winsorize_min_samples",
        "standardize",
        "standardize_min_std",
    }
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_redline.parquet")
    request_path, result_path = tmp_path / "request_redline.json", tmp_path / "result_redline.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal)

    worker_csv = pd.read_parquet(request["portfolio_artifact_path"]).to_csv(float_format="%.17g")
    local_csv = local.weights.to_csv(float_format="%.17g")
    assert sha256(worker_csv.encode()).hexdigest() == REDLINE_WEIGHTS_SHA256
    assert sha256(local_csv.encode()).hexdigest() == REDLINE_WEIGHTS_SHA256
    assert {key: float(value) for key, value in payload["metrics"].items()} == {
        key: float(value) for key, value in REDLINE_METRICS.items()
    }


def _make_symbol_missing(request: dict, factor_name: str, symbol: str, *, partial: bool = True) -> None:
    """Blank one symbol out of one frozen factor panel and re-freeze the
    artifact checksums so the mutated request still passes provenance.

    Partial blanking hides the symbol on every other date only, which is what
    makes ``drop_stock`` end-to-end discriminating: percentile ranks are
    invariant to removing a NaN name, so a fully-blank symbol looks identical
    with and without the policy."""

    item = next(item for item in request["artifact_set"]["factors"] if str(item["name"]) == factor_name)
    path = Path(item["artifact_path"])
    frame = pd.read_parquet(path)
    if partial:
        frame.loc[frame.index[1::2], symbol] = np.nan
    else:
        frame[symbol] = np.nan
    frame.to_parquet(path)
    item["artifact_checksum"] = _parquet_checksum(path)
    versions = tuple(
        FactorArtifactVersion.from_payload(dict(payload)) for payload in request["artifact_set"]["factors"]
    )
    request["artifact_set"]["artifact_set_id"] = artifact_set_id_for(
        request["artifact_set"]["snapshot_id"],
        request["artifact_set"]["working_set_id"],
        request["artifact_set"]["evaluation_start"],
        request["artifact_set"]["evaluation_end"],
        versions,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("signal_payload", "composition", "label", "blank_factor", "blank_symbol"),
    [
        pytest.param(
            {"missing_value": "cross_section_median"},
            CompositionSpec(top_n=4, weighting="equal"),
            "missing-median-fill",
            "value",
            "S0",
            id="missing-median-fill",
        ),
        pytest.param(
            {"missing_value": "neutral_rank"},
            CompositionSpec(top_n=4, weighting="equal"),
            "missing-neutral-rank",
            "value",
            "S0",
            id="missing-neutral-rank",
        ),
        pytest.param(
            {"missing_value": "drop_stock"},
            CompositionSpec(top_n=4, weighting="equal"),
            "missing-drop-stock",
            "value",
            "S4",
            id="missing-drop-stock",
        ),
        pytest.param(
            {"blend_weights": {"value": 0.6, "quality": 0.3, "momentum": 0.1}},
            CompositionSpec(top_n=4, weighting="proportional"),
            "blend-flat-weights",
            None,
            None,
            id="blend-flat-weights",
        ),
        pytest.param(
            {
                "blend_groups": {"group_value": ["value"], "group_rest": ["quality", "momentum"]},
                "blend_group_weights": {"group_value": 0.8, "group_rest": 0.2},
                "blend_within_weights": {"group_rest": {"quality": 0.9, "momentum": 0.1}},
            },
            CompositionSpec(top_n=4, weighting="proportional"),
            "blend-groups",
            None,
            None,
            id="blend-group-weights",
        ),
        pytest.param(
            {"interaction_pairs": [["value", "quality", "product"]]},
            CompositionSpec(top_n=4, weighting="proportional"),
            "interaction-product",
            None,
            None,
            id="interaction-product",
        ),
        pytest.param(
            {"interaction_pairs": [["value", "quality", "min"], ["quality", "momentum", "geometric_mean"]]},
            CompositionSpec(top_n=4, weighting="proportional"),
            "interaction-multi",
            None,
            None,
            id="interaction-multi-kinds",
        ),
        pytest.param(
            {
                "missing_value": "cross_section_median",
                "blend_weights": {"value": 0.6, "quality": 0.3, "momentum": 0.1},
            },
            CompositionSpec(top_n=4, weighting="proportional"),
            "missing-plus-flat-blend",
            "value",
            "S0",
            id="missing-plus-flat-blend",
        ),
        pytest.param(
            {
                "interaction_pairs": [["value", "quality", "product"]],
                "blend_groups": {"group_value": ["value"], "group_rest": ["quality", "momentum"]},
                "blend_group_weights": {"group_value": 0.2, "group_rest": 0.05},
            },
            CompositionSpec(top_n=4, weighting="equal"),
            "interaction-plus-group-blend",
            None,
            None,
            id="interaction-plus-group-blend",
        ),
    ],
)
def test_signal_dag_operators_agree_end_to_end(
    world, signal_payload, composition: CompositionSpec, label, blank_factor, blank_symbol
) -> None:
    """MissingValuePolicy / FactorBlend / FactorGroupBlend / FactorInteraction
    end-to-end parity, including the two two-operator stacks.

    The equal top-4 selection absorbs pure score *weight* changes on these
    panels, so the blend and interaction cases run under the ``proportional``
    weighting (or an interaction-dominated group blend) where the score level
    actually reaches the portfolio; the probe that picked these payloads is
    part of the delivery note."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name=f"parity_{label}",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        signal=SignalSpec.from_payload(signal_payload),
    )
    request = world["evaluator"].request_for(candidate)
    if blank_symbol is not None:
        _make_symbol_missing(request, blank_factor, blank_symbol)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{label}.parquet")
    request_path, result_path = tmp_path / f"request_{label}.json", tmp_path / f"result_{label}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal)

    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Discrimination: the same (possibly blanked) panels through the disabled
    # signal spec must produce different holdings, so a case that accidentally
    # stops exercising the operator fails loudly instead of passing vacuously.
    baseline = _local_portfolio(world, request, composition, SignalSpec())
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float)), label

    if label == "missing-drop-stock":
        # S4 is partially blanked in the value panel: without the policy it is
        # still held on the dates where its value exists, with the policy it is
        # dropped from the panel and can never be held.
        assert float(baseline.weights["S4"].abs().max()) > 0.0
        assert float(local.weights["S4"].abs().max()) == 0.0
    if label in ("missing-median-fill", "missing-neutral-rank", "missing-plus-flat-blend"):
        # Filling rescues the blanked name: without the policy it can never
        # receive a weight because its value rank is NaN on the blanked dates.
        assert float((local.weights["S0"] != 0).sum()) > 0, label


def test_signal_dag_stages_match_bit_for_bit_between_local_and_worker(world) -> None:
    """Feed identical panels into both implementations of every new stage.

    The end-to-end cases exercise the pipeline, but the fill values, the
    dropped-symbol sets, the interaction arithmetic and the weighted blend
    deserve the same direct per-cell comparison the A1 stages and the gating
    cascade already have.
    """

    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = ["A", "B", "C", "D"]
    panel = pd.DataFrame(
        {
            "A": [1.0, np.nan, 3.0],
            "B": [np.nan, 2.0, 6.0],
            "C": [3.0, 4.0, np.nan],
            "D": [4.0, 8.0, 12.0],
        },
        index=dates,
        columns=columns,
    )
    clean = pd.DataFrame(np.arange(12.0).reshape(3, 4), index=dates, columns=columns)

    # --- value-space MissingValuePolicy -----------------------------------
    for values, signal_payload in [
        (panel, {"missing_value": "cross_section_median"}),
        (panel, {"missing_value": "cross_section_median", "missing_max_fill_ratio": 0.5}),
        (panel, {"missing_value": "drop_stock"}),
        (clean, {"missing_value": "drop_stock"}),
        (panel, {}),  # disabled: the identity on both sides
    ]:
        spec = SignalSpec.from_payload(signal_payload)
        np.testing.assert_array_equal(
            apply_missing_value(values, spec).to_numpy(dtype=float),
            _apply_missing_value(values.copy(), signal_payload).to_numpy(dtype=float),
        )

    # --- rank-space neutral fill ------------------------------------------
    rank_panel = pd.DataFrame(
        {
            "A": [0.1, np.nan, 0.5],
            "B": [np.nan, 0.4, 0.2],
            "C": [0.7, 0.9, np.nan],
            "D": [0.2, 0.3, 0.6],
        },
        index=dates,
        columns=columns,
    )
    for signal_payload in [
        {"missing_value": "neutral_rank"},
        {"missing_value": "neutral_rank", "missing_max_fill_ratio": 0.5},
        {},
    ]:
        spec = SignalSpec.from_payload(signal_payload)
        np.testing.assert_array_equal(
            apply_neutral_rank_fill(rank_panel, spec).to_numpy(dtype=float),
            _apply_neutral_rank_fill(rank_panel.copy(), signal_payload).to_numpy(dtype=float),
        )

    # --- the whole per-panel rank build on the frozen fixture panels ------
    directions = {"value": 1, "quality": 1, "momentum": int(np.sign(FACTOR_RANK_IC["momentum"]))}
    panels_with_nan = {name: _factor_panel(name).copy() for name in FACTOR_NAMES}
    panels_with_nan["value"]["S0"] = np.nan
    for panels in ({name: _factor_panel(name) for name in FACTOR_NAMES}, panels_with_nan):
        for signal_payload in [
            {},
            {"missing_value": "cross_section_median"},
            {"missing_value": "neutral_rank"},
            {"missing_value": "drop_stock"},
            {"standardize": "zscore", "missing_value": "cross_section_median"},
        ]:
            spec = SignalSpec.from_payload(signal_payload)
            local = build_factor_rank_panels(panels, directions, spec, index=EVAL_DATES, columns=list(SYMBOLS))
            worker = _build_rank_panels(
                {name: frame.copy() for name, frame in panels.items()},
                directions,
                signal_payload,
                EVAL_DATES,
                list(SYMBOLS),
            )
            assert set(local) == set(worker)
            for name in local:
                np.testing.assert_array_equal(local[name].to_numpy(dtype=float), worker[name].to_numpy(dtype=float))

    # --- interaction terms -------------------------------------------------
    ranks = {
        "f1": pd.DataFrame(
            {"A": [0.1, 0.5, np.nan], "B": [0.9, 0.4, 0.3], "C": [0.2, np.nan, 0.8], "D": [0.6, 0.7, 0.1]},
            index=dates,
            columns=columns,
        ),
        "f2": pd.DataFrame(
            {"A": [0.3, 0.2, 0.5], "B": [np.nan, 0.8, 0.4], "C": [0.6, 0.1, 0.9], "D": [0.4, 0.3, np.nan]},
            index=dates,
            columns=columns,
        ),
    }
    for signal_payload in [
        {"interaction_pairs": [["f1", "f2", "product"]]},
        {"interaction_pairs": [["f1", "f2", "min"]]},
        {"interaction_pairs": [["f1", "f2", "geometric_mean"]]},
    ]:
        spec = SignalSpec.from_payload(signal_payload)
        local = interaction_rank_panels(ranks, spec)
        worker = _interaction_rank_panels(ranks, signal_payload)
        assert set(local) == set(worker)
        for name in local:
            np.testing.assert_array_equal(local[name].to_numpy(dtype=float), worker[name].to_numpy(dtype=float))
        member = local["f1*f2"]
        combined = member.to_numpy(dtype=float)
        left, right = ranks["f1"].to_numpy(dtype=float), ranks["f2"].to_numpy(dtype=float)
        # A NaN in either input keeps the interaction cell missing.
        assert bool(np.isnan(combined[np.isnan(left) | np.isnan(right)]).all())
        assert bool(((combined[~np.isnan(combined)] > 0) & (combined[~np.isnan(combined)] <= 1)).all())

    # --- blend configurations ----------------------------------------------
    for signal_payload in [
        {},
        {"blend_weights": {"f1": 0.7, "f2": 0.3}},
        {"blend_groups": {"g1": ["f1"], "g2": ["f2"]}},
        {"blend_groups": {"g1": ["f1"], "g2": ["f2"]}, "blend_group_weights": {"g1": 0.8, "g2": 0.2}},
        {"blend_groups": {"g1": ["f1", "f2"]}, "blend_within_weights": {"g1": {"f1": 0.25, "f2": 0.75}}},
        {"interaction_pairs": [["f1", "f2", "product"]], "blend_weights": {"f1": 0.5, "f2": 0.25, "f1*f2": 0.25}},
        {
            "interaction_pairs": [["f1", "f2", "min"]],
            "blend_groups": {"g1": ["f1"], "g2": ["f2"]},
            "blend_group_weights": {"g1": 0.6, "g2": 0.4},
        },
    ]:
        spec = SignalSpec.from_payload(signal_payload)
        local = blend_member_panels(ranks, ["f1", "f2"], spec)
        worker = _score(ranks, ["f1", "f2"], signal_payload)
        np.testing.assert_array_equal(local.to_numpy(dtype=float), worker.to_numpy(dtype=float))

    # --- the strict missing contract survives every blend configuration ----
    ranks_all_missing = {"f1": ranks["f1"], "f2": ranks["f2"] * np.nan}
    for signal_payload in [
        {},
        {"blend_weights": {"f1": 0.7, "f2": 0.3}},
        {"blend_groups": {"g1": ["f1"], "g2": ["f2"]}},
    ]:
        spec = SignalSpec.from_payload(signal_payload)
        assert blend_member_panels(ranks_all_missing, ["f1", "f2"], spec).isna().all().all()
        assert _score(ranks_all_missing, ["f1", "f2"], signal_payload).isna().all().all()


@pytest.mark.parametrize(
    ("signal_payload", "blank_value_symbols", "match"),
    [
        pytest.param(
            {"missing_value": "cross_section_median", "missing_max_fill_ratio": 0.01},
            ["S0"],
            "missing_value fill exceeds 0.01 of the cross-section on some date",
            id="fill-ratio-exceeded",
        ),
        pytest.param(
            {"missing_value": "drop_stock"},
            list(SYMBOLS),
            "drop_stock removed every symbol from a factor panel",
            id="drop-stock-empties-the-panel",
        ),
        pytest.param(
            {"blend_weights": {"value": 0.5, "quality": 0.5}},
            [],
            "blend_weights must cover exactly the blend members",
            id="flat-weights-partial",
        ),
        pytest.param(
            {"blend_weights": {"value": -1.0, "quality": 0.5, "momentum": 0.5}},
            [],
            "blend weights must be finite positive numbers",
            id="flat-weights-negative",
        ),
        pytest.param(
            {
                "blend_weights": {"value": 0.4, "quality": 0.3, "momentum": 0.3},
                "blend_groups": {"g1": ["value", "quality", "momentum"]},
            },
            [],
            "blend_weights and blend_groups are mutually exclusive",
            id="flat-and-group-exclusive",
        ),
        pytest.param(
            {"blend_groups": {"g1": ["value"]}},
            [],
            "blend_groups must partition the candidate factors exactly",
            id="groups-not-partitioning",
        ),
        pytest.param(
            {"blend_groups": {"g1": ["value", "momentum"], "g2": ["value", "quality"]}},
            [],
            "blend_groups must not assign a factor to two groups",
            id="groups-overlapping",
        ),
        pytest.param(
            {"interaction_pairs": [["value", "value", "product"]]},
            [],
            "interaction terms need two distinct non-empty factor names",
            id="interaction-degenerate",
        ),
        pytest.param(
            {"interaction_pairs": [["value", "quality", "tanh"]]},
            [],
            "interaction kind must be one of",
            id="interaction-unknown-kind",
        ),
        pytest.param(
            {"interaction_pairs": [["value", "invented", "product"]]},
            [],
            "interaction terms must reference candidate factors",
            id="interaction-unknown-factor",
        ),
        pytest.param(
            {"missing_value": "median"},
            [],
            "missing_value must be one of",
            id="unknown-missing-policy",
        ),
        pytest.param(
            {"missing_max_fill_ratio": 0.5},
            [],
            "missing_max_fill_ratio requires a missing_value policy",
            id="ratio-without-policy",
        ),
        pytest.param(
            {"missing_value": "drop_stock", "missing_max_fill_ratio": 0.5},
            [],
            "missing_max_fill_ratio does not apply to the drop_stock policy",
            id="ratio-with-drop-stock",
        ),
        pytest.param(
            {"missing_value": "cross_section_median", "missing_max_fill_ratio": 1.5},
            [],
            "missing_max_fill_ratio must be in (0, 1] when provided",
            id="ratio-out-of-range",
        ),
    ],
)
def test_signal_dag_failures_fail_closed_on_both_sides(world, signal_payload, blank_value_symbols, match) -> None:
    """Invalid signal-DAG parameters or data raise ValueError on both sides.

    Parameter failures raise at candidate validation locally and in
    ``_validate_signal`` inside the worker; data failures (over-wide fills, an
    emptied panel) raise at the rank build on both engines.
    """

    candidate = StrategyCandidate(
        name="parity_signal_dag_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    for symbol in blank_value_symbols:
        _make_symbol_missing(request, "value", symbol)
    request["candidate"]["signal"] = signal_payload
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_dag_fail.parquet")
    request_path, result_path = tmp_path / "request_dag_fail.json", tmp_path / "result_dag_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(match)):
        run_request(request_path, result_path)

    if blank_value_symbols:
        close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
        calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
        rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
        symbols = sorted(set(close.columns))
        factor_names = list(request["candidate"]["factor_names"])
        factor_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
        panels = {name: pd.read_parquet(factor_items[name]["artifact_path"]) for name in factor_names}
        directions = {name: int(request["factor_directions"][name]) for name in factor_names}
        with pytest.raises(ValueError, match=re.escape(match)):
            build_factor_score(
                panels,
                directions,
                SignalSpec.from_payload(signal_payload),
                index=rebalance_dates,
                columns=symbols,
            )
    else:
        with pytest.raises(ValueError, match=re.escape(match)):
            StrategyCandidate(
                name="parity_signal_dag_fail_closed",
                universe="csi_all",
                composition=CompositionSpec(top_n=4, weighting="equal"),
                factor_names=FACTOR_NAMES,
                rule="parity fixture",
                signal=SignalSpec.from_payload(signal_payload),
            )


# ---------------------------------------------------------------------------
# Batch B2: HoldingBuffer (5.5) / DiversitySelector (5.6) / MinimumHoldingPeriod
# (8.4).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("composition", "label", "daily_with_blank"),
    [
        # A held name that drops out of the natural top_n but stays above the
        # retention threshold is rescued (vacated seat first, then extra).
        (
            CompositionSpec(top_n=4, weighting="equal", holding_buffer_entry=0.6, holding_buffer_retain=0.15, holding_buffer_max_extra=1),
            "holding-buffer-rescue",
            True,
        ),
        # The retention threshold collapses onto the entry threshold when
        # unset; a rescued holding then needs an extra seat.
        (
            CompositionSpec(top_n=4, weighting="equal", holding_buffer_entry=0.33, holding_buffer_max_extra=1),
            "holding-buffer-default-retain",
            True,
        ),
        # Greedy max-correlation screen changes the selected set.
        (
            CompositionSpec(top_n=4, weighting="equal", diversity_max_correlation=0.4),
            "diversity-correlation",
            False,
        ),
        # Both selection operators stacked: the correlation screen also binds
        # the rescued holdings.
        (
            CompositionSpec(
                top_n=3,
                weighting="equal",
                holding_buffer_entry=0.6,
                holding_buffer_retain=0.15,
                holding_buffer_max_extra=1,
                diversity_max_correlation=0.6,
            ),
            "buffer-plus-diversity",
            False,
        ),
    ],
)
def test_selection_operators_agree_end_to_end(world, composition: CompositionSpec, label: str, daily_with_blank: bool) -> None:
    """HoldingBuffer (5.5) / DiversitySelector (5.6) end-to-end parity."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name=f"parity_{label}",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )

    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{label}.parquet")
    if daily_with_blank:
        # The static fixture almost never churns the top-4 on monthly
        # rebalances, so the buffer would have nothing to bite on (fact 23).
        # Alternating-NaN on one top value name forces daily membership
        # churn: the seat-filler is bought and sold on consecutive days.
        request["evaluation"]["rebalance_frequency"] = "daily"
        _make_symbol_missing(request, "value", "S5")
    request_path, result_path = tmp_path / f"request_{label}.json", tmp_path / f"result_{label}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Discrimination: the held-name set must actually change versus the
    # untouched composition on the same panels (fact 23: an operator case that
    # cannot move the book is a tautology).
    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=composition.top_n, weighting=composition.weighting),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float)), label


@pytest.mark.slow
def test_minimum_holding_period_agrees_end_to_end(world) -> None:
    """MinimumHoldingPeriod (8.4) end-to-end parity, alone and stacked with the
    A4 execution operators.

    Daily rebalances make the lock bite: a name bought within the last
    ``min_holding_days`` days cannot be sold even though the fresh target
    exits it.  The stacked case runs band + shrinkage + lock in one loop in
    the documented order (band decides, shrink sizes, lock vetoes last).
    """

    tmp_path: Path = world["tmp_path"]

    def _run(backtest: BacktestSpec, label: str, composition: CompositionSpec | None = None):
        composition = composition or CompositionSpec(top_n=4, weighting="equal")
        candidate = StrategyCandidate(
            name=f"parity_{label}",
            universe="csi_all",
            composition=composition,
            factor_names=FACTOR_NAMES,
            rule="parity fixture",
            backtest=backtest,
        )
        request = world["evaluator"].request_for(candidate)
        request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{label}.parquet")
        # Daily decisions are what make the lock bind at all: with the monthly
        # default the entry ages past every lock window between rebalances.
        request["evaluation"]["rebalance_frequency"] = "daily"
        # The static fixture almost never sells anything, so an alternating
        # NaN on one top value name forces daily buy/sell churn and the lock
        # genuinely blocks sells the baseline makes.
        _make_symbol_missing(request, "value", "S5")
        request_path, result_path = tmp_path / f"request_{label}.json", tmp_path / f"result_{label}.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        payload = run_request(request_path, result_path)
        assert payload["status"] == "success", payload
        local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
        _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))
        baseline = _local_portfolio(
            world,
            request,
            candidate.composition,
            candidate.signal,
            backtest=BacktestSpec(signal_lag=1),
        )
        return local, baseline

    locked, baseline = _run(BacktestSpec(signal_lag=1, min_holding_days=3), "min-holding")
    # Discrimination: the lock must have kept at least one name the baseline
    # sold (a blocked sell), so the two books genuinely diverge.
    kept_but_baseline_flat = (locked.weights > 0) & (baseline.weights == 0)
    assert bool(kept_but_baseline_flat.to_numpy().any()), "the lock never blocked a sell the baseline made"
    # Conservation (blocking fix): every executed row of every min-holding run
    # stays within the 100% budget -- the lock can never inflate the book.
    assert bool((locked.weights.sum(axis=1) <= 1.0 + 1e-9).all()), "min-holding inflated a row above 1.0"
    assert bool((baseline.weights.sum(axis=1) <= 1.0 + 1e-9).all())

    stacked, stacked_baseline = _run(
        BacktestSpec(signal_lag=1, min_holding_days=3, drift_band_name=0.02, turnover_shrinkage=0.3),
        "min-holding-band-shrink",
    )
    assert not np.allclose(
        stacked.weights.to_numpy(dtype=float), stacked_baseline.weights.to_numpy(dtype=float)
    ), "stacked execution operators never diverged from the baseline"
    assert bool((stacked.weights.sum(axis=1) <= 1.0 + 1e-9).all()), "stacked run inflated a row above 1.0"
    assert bool((stacked_baseline.weights.sum(axis=1) <= 1.0 + 1e-9).all())

    # The verifier's original inflation repro (daily + mhd=5 + top_n=2, row
    # sums peaked at 2.3521 pre-fix): the conservation assertion is the
    # regression red line for the unfunded-raise abandonment.
    repro, repro_baseline = _run(
        BacktestSpec(signal_lag=1, min_holding_days=5),
        "min-holding-unfunded-repro",
        composition=CompositionSpec(top_n=2, weighting="equal"),
    )
    assert bool((repro.weights.sum(axis=1) <= 1.0 + 1e-9).all()), "min-holding inflated a row above 1.0 (pre-fix repro)"
    assert bool((repro_baseline.weights.sum(axis=1) <= 1.0 + 1e-9).all())
    assert not np.allclose(repro.weights.to_numpy(dtype=float), repro_baseline.weights.to_numpy(dtype=float))


@pytest.mark.parametrize(
    ("payload_override", "message", "composition_override"),
    [
        ({"min_holding_days": 0}, "min_holding_days must be at least 1 when provided", False),
        ({"min_holding_days": 2.5}, "min_holding_days must be a whole number when provided", False),
        ({"holding_buffer_entry": 0.0}, "holding_buffer_entry must be in (0, 1] when provided", True),
        ({"holding_buffer_entry": 1.5}, "holding_buffer_entry must be in (0, 1] when provided", True),
        ({"holding_buffer_retain": 1.5}, "holding_buffer_retain must be in (0, 1] when provided", True),
        (
            {"holding_buffer_entry": 0.3, "holding_buffer_retain": 0.8},
            "holding_buffer_retain must not exceed holding_buffer_entry",
            True,
        ),
        (
            {"holding_buffer_retain": 0.5},
            "holding_buffer_retain and holding_buffer_max_extra require holding_buffer_entry",
            True,
        ),
        (
            {"holding_buffer_max_extra": -1},
            "holding_buffer_max_extra must not be negative",
            True,
        ),
        (
            {"holding_buffer_max_extra": 1.5},
            "holding_buffer_max_extra must be a whole number",
            True,
        ),
        ({"diversity_max_correlation": 1.5}, "diversity_max_correlation must be in [-1, 1] when provided", True),
        ({"diversity_max_correlation": -1.5}, "diversity_max_correlation must be in [-1, 1] when provided", True),
    ],
)
def test_selection_and_holding_operators_fail_closed_on_both_sides(
    world,
    payload_override: dict,
    message: str,
    composition_override: bool,
) -> None:
    """Invalid buffer/diversity/min-holding parameters raise on both sides."""

    candidate = StrategyCandidate(
        name="parity_b2_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_b2_fail.parquet")
    target = request["candidate"]["composition" if composition_override else "backtest"]
    target.update(payload_override)
    request_path, result_path = tmp_path / "request_b2_fail.json", tmp_path / "result_b2_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    if composition_override:
        with pytest.raises(ValueError) as local_error:
            CompositionSpec(top_n=4, weighting="equal", **payload_override)
    else:
        with pytest.raises(ValueError) as local_error:
            BacktestSpec(signal_lag=1, **payload_override)
    assert str(local_error.value) == message


@pytest.mark.slow
def test_min_holding_and_buffered_row_match_bit_for_bit_between_local_and_worker() -> None:
    """Feed identical inputs into both implementations of the two new mirrored
    helpers: the per-date buffered target row and the min-holding clamp."""

    dates = pd.bdate_range("2024-01-02", periods=3)
    columns = ["A", "B", "C", "D", "E"]
    scores = pd.DataFrame(
        [
            [50.0, 40.0, 30.0, 20.0, 10.0],
            [10.0, 50.0, np.nan, 30.0, 20.0],
            [20.0, 20.0, 20.0, 50.0, np.nan],
        ],
        index=dates,
        columns=columns,
    )
    returns_history = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.01, 0.04, 0.02],
            "B": [0.01, -0.02, 0.03, 0.01, 0.04, 0.02],
            "C": [0.05, 0.01, -0.03, 0.02, 0.01, -0.01],
            "D": [-0.02, 0.03, 0.01, -0.01, 0.02, 0.03],
            "E": [0.0, 0.0, 0.0, 0.01, -0.01, 0.0],
        }
    )
    composition_cases = [
        CompositionSpec(top_n=3, weighting="equal"),
        CompositionSpec(top_n=3, weighting="rank"),
        CompositionSpec(top_n=3, weighting="equal", holding_buffer_entry=0.5),
        CompositionSpec(top_n=3, weighting="equal", holding_buffer_entry=0.6, holding_buffer_retain=0.2, holding_buffer_max_extra=2),
        CompositionSpec(top_n=3, weighting="rank", holding_buffer_entry=0.6, holding_buffer_retain=0.4),
        CompositionSpec(top_n=3, weighting="equal", diversity_max_correlation=0.5),
        CompositionSpec(top_n=3, weighting="equal", diversity_max_correlation=-0.2),
        CompositionSpec(
            top_n=3,
            weighting="equal",
            weight_cap=0.5,
            holding_buffer_entry=0.5,
            holding_buffer_retain=0.2,
            holding_buffer_max_extra=2,
            diversity_max_correlation=0.7,
        ),
    ]
    held_cases = [frozenset(), frozenset({"A", "B"}), frozenset({"C", "E"}), frozenset({"A", "B", "C", "D", "E"})]
    for spec in composition_cases:
        payload_composition = spec.as_payload()
        for position in range(len(dates)):
            for held in held_cases:
                history = returns_history if position % 2 == 0 else returns_history.iloc[:-1]
                local_row = _local_build_buffered_target_row(spec, scores.iloc[position], held, history)
                worker_row = _build_buffered_target_row(scores.iloc[position], payload_composition, held, history)
                np.testing.assert_array_equal(
                    local_row.to_numpy(dtype=float),
                    worker_row.to_numpy(dtype=float),
                    err_msg=f"buffered target row drifted ({spec}, position={position}, held={sorted(held)})",
                )

    carried = pd.Series({"A": 0.3, "B": 0.25, "C": 0.25, "D": 0.1, "E": 0.0})
    min_holding_cases = [
        # Nothing locked: the executed vector passes through untouched.
        (carried, pd.Series({"A": 0.2, "B": 0.3, "C": 0.3, "D": 0.2, "E": 0.0}), {"A": 0, "B": 9, "C": 9}, 5, 3),
        # A is locked but not being sold: identical no-op.
        (carried, pd.Series({"A": 0.3, "B": 0.3, "C": 0.25, "D": 0.15, "E": 0.0}), {"A": 1, "B": 9, "C": 9}, 2, 3),
        # A and B are locked and being sold: they keep their carried weights,
        # the unlocked names give the budget back proportionally.
        (carried, pd.Series({"A": 0.0, "B": 0.1, "C": 0.5, "D": 0.3, "E": 0.1}), {"A": 0, "B": 1, "C": 9}, 2, 3),
        # Everything locked: no unlocked name left to scale.
        (carried, pd.Series({"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.3, "E": 0.1}), {"A": 0, "B": 0, "C": 0, "D": 0}, 1, 5),
        # The locked take-back exceeds the executed total: the whole trade is
        # abandoned and both sides return None (the caller keeps carried).
        (
            pd.Series({"A": 0.6, "B": 0.2, "C": 0.2}),
            pd.Series({"A": 0.0, "B": 0.05, "C": 0.05}),
            {"A": 0},
            1,
            5,
        ),
        # Exactly min_holding_days completed days since entry: the sell goes
        # through (strict <).
        (carried, pd.Series({"A": 0.0, "B": 0.3, "C": 0.4, "D": 0.2, "E": 0.1}), {"A": 0, "B": 9, "C": 9}, 3, 3),
    ]
    for carried_row, executed_row, entry_index, position, min_holding_days in min_holding_cases:
        local_adjusted = _local_apply_min_holding(carried_row, executed_row, entry_index, position, min_holding_days)
        worker_adjusted = _apply_min_holding(carried_row, executed_row, entry_index, position, min_holding_days)
        assert (local_adjusted is None) == (worker_adjusted is None), (
            f"min-holding abandonment disagrees (position={position}, entry={entry_index})"
        )
        if local_adjusted is None:
            continue
        np.testing.assert_array_equal(
            local_adjusted.to_numpy(dtype=float),
            worker_adjusted.to_numpy(dtype=float),
            err_msg=f"min-holding clamp drifted (position={position}, entry={entry_index})",
        )


@pytest.mark.parametrize(
    ("signal_override", "message"),
    [
        (
            {"interaction_pairs": [["value", "quality", "product", "extra"]]},
            "each interaction pair must be a [first, second, kind] triple",
        ),
        (
            {"interaction_pairs": [["value", "quality"]]},
            "each interaction pair must be a [first, second, kind] triple",
        ),
        ({"blend_weights": [1.0, 1.0, 1.0]}, "blend_weights must not be empty"),
    ],
)
def test_b2_signal_payload_failures_fail_closed_on_both_sides(world, signal_override: dict, message: str) -> None:
    """Malformed interaction pairs and blend weights fail closed everywhere.

    The worker used to silently truncate a 4-element interaction pair to its
    first three entries; both ends now reject any non-triple with the same
    message.  A list-valued blend_weights raised AttributeError locally and
    ValueError in the worker; the local side now raises the same ValueError.
    """

    candidate = StrategyCandidate(
        name="parity_b2_signal_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_b2_signal_fail.parquet")
    request["candidate"]["signal"].update(signal_override)
    request_path, result_path = tmp_path / "request_b2_signal_fail.json", tmp_path / "result_b2_signal_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError, match=re.escape(message)):
        SignalSpec.from_payload(signal_override)


# ---------------------------------------------------------------------------
# Batch C (roadmap 6.3 / 6.8 / 7.5 / 7.8): IndustryWeightCap,
# CashResidualPolicy, InverseVolatilityWeight, HierarchicalWeighting.
# ---------------------------------------------------------------------------

_INDUSTRY_MAP = {"S5": "A", "S4": "A", "S3": "A", "S2": "B", "S1": "B", "S0": "B"}


@pytest.mark.slow
def test_industry_weight_cap_agrees_and_binds_end_to_end(world) -> None:
    """IndustryWeightCap (6.3) end-to-end parity.

    ``top_n=6`` selects the whole universe, so the industry sums before the
    cap are 15/21 (industry A) and 6/21 (industry B) regardless of the score
    ordering: the cap binds on every rebalance date by construction.
    """

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(
        top_n=6,
        weighting="rank",
        weight_cap=0.3,
        weight_floor=0.05,
        industry_weight_cap=0.45,
        industry_map=dict(_INDUSTRY_MAP),
    )
    candidate = StrategyCandidate(
        name="parity_industry_cap",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_industry_cap.parquet")
    request_path, result_path = tmp_path / "request_industry_cap.json", tmp_path / "result_industry_cap.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # The cap chain (single-name cap -> industry cap -> floor) is judged on
    # the target weights, so recompute them and check every row.
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    score = _score_panel_for(world, request, candidate.signal, rebalance_dates, sorted(set(close.columns)))
    targets = build_target_weights(composition, score, returns=_returns_panel_for(world))
    for date, row in targets.iterrows():
        held = row[row > 0]
        assert held.max() <= composition.weight_cap + 1e-9, date
        assert held.min() >= composition.weight_floor - 1e-9, date
        for industry in ("A", "B"):
            assert held.groupby(_INDUSTRY_MAP).sum().get(industry, 0.0) <= composition.industry_weight_cap + 1e-9, date

    # Discrimination: without the industry cap the book differs (fact 23).
    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=6, weighting="rank", weight_cap=0.3, weight_floor=0.05),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_inverse_volatility_weighting_agrees_end_to_end(world) -> None:
    """InverseVolatilityWeight (7.5) end-to-end parity."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(top_n=4, weighting="inverse_volatility")
    candidate = StrategyCandidate(
        name="parity_inverse_vol",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_inverse_vol.parquet")
    request_path, result_path = tmp_path / "request_inverse_vol.json", tmp_path / "result_inverse_vol.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Discrimination (fact 23): with the fixture's near-deterministic prices
    # the inverse-volatility tilt must move the book away from the equal
    # weighting after the first (fallback) decision date.
    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=4, weighting="equal"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_hierarchical_weighting_agrees_end_to_end(world) -> None:
    """HierarchicalWeighting (7.8) end-to-end parity."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(
        top_n=4,
        weighting="hierarchical",
        hierarchical_groups={"gA": ("S5", "S4", "S3"), "gB": ("S2", "S1", "S0")},
        hierarchical_group_weights={"gA": 0.7, "gB": 0.3},
    )
    candidate = StrategyCandidate(
        name="parity_hierarchical",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_hierarchical.parquet")
    request_path, result_path = tmp_path / "request_hierarchical.json", tmp_path / "result_hierarchical.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Group mass: 70/30 of every invested row, on the recomputed targets.
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    score = _score_panel_for(world, request, candidate.signal, rebalance_dates, sorted(set(close.columns)))
    targets = build_target_weights(composition, score, returns=_returns_panel_for(world))
    for date, row in targets.iterrows():
        invested = row.sum()
        assert row[["S5", "S4", "S3"]].sum() == pytest.approx(0.7 * invested, abs=1e-9), date
        assert row[["S2", "S1", "S0"]].sum() == pytest.approx(0.3 * invested, abs=1e-9), date

    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=4, weighting="equal"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_industry_cap_cash_floor_chain_stacks_end_to_end(world) -> None:
    """Industry cap + single-name cap + floor + explicit cash bound."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(
        top_n=6,
        weighting="rank",
        weight_cap=0.25,
        weight_floor=0.05,
        industry_weight_cap=0.45,
        industry_map=dict(_INDUSTRY_MAP),
        cash_residual_max=0.25,
    )
    candidate = StrategyCandidate(
        name="parity_industry_cap_stack",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_industry_cap_stack.parquet")
    request_path, result_path = tmp_path / "request_industry_cap_stack.json", tmp_path / "result_industry_cap_stack.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    score = _score_panel_for(world, request, candidate.signal, rebalance_dates, sorted(set(close.columns)))
    targets = build_target_weights(composition, score, returns=_returns_panel_for(world))
    for date, row in targets.iterrows():
        held = row[row > 0]
        assert held.max() <= composition.weight_cap + 1e-9, date
        assert held.min() >= composition.weight_floor - 1e-9, date
        assert held.groupby(_INDUSTRY_MAP).sum().max() <= composition.industry_weight_cap + 1e-9, date
        assert 1.0 - row.sum() <= composition.cash_residual_max + 1e-9, date


@pytest.mark.slow
def test_inverse_volatility_and_min_holding_stack_end_to_end(world) -> None:
    """InverseVolatilityWeight (7.5) stacked with MinimumHoldingPeriod (8.4)."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(top_n=4, weighting="inverse_volatility")
    candidate = StrategyCandidate(
        name="parity_invvol_min_holding",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1, min_holding_days=3),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_invvol_min_holding.parquet")
    # Daily decisions make the lock bite at all (fact: with the monthly
    # default the entry ages past every lock window between rebalances).
    request["evaluation"]["rebalance_frequency"] = "daily"
    _make_symbol_missing(request, "value", "S5")
    request_path, result_path = tmp_path / "request_invvol_min_holding.json", tmp_path / "result_invvol_min_holding.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))
    assert bool((local.weights.sum(axis=1) <= 1.0 + 1e-9).all()), "min-holding inflated a row above 1.0"

    baseline = _local_portfolio(
        world,
        request,
        composition,
        candidate.signal,
        backtest=BacktestSpec(signal_lag=1),
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_inverse_volatility_flows_through_the_buffered_row_path(world) -> None:
    """HoldingBuffer (5.5) on top of the inverse-volatility weighting: the
    per-date buffered row builder must reproduce the weighting from the same
    return history."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(top_n=4, weighting="inverse_volatility", holding_buffer_entry=0.5)
    candidate = StrategyCandidate(
        name="parity_invvol_buffer",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_invvol_buffer.parquet")
    request["evaluation"]["rebalance_frequency"] = "daily"
    _make_symbol_missing(request, "value", "S5")
    request_path, result_path = tmp_path / "request_invvol_buffer.json", tmp_path / "result_invvol_buffer.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=4, weighting="inverse_volatility"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


def test_batch_c_helpers_match_bit_for_bit_between_local_and_worker() -> None:
    """Feed identical inputs into the three mirrored batch C helpers plus the
    buffered row builder running the inverse-volatility weighting."""

    weights_cases = [
        pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}),
        pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        pd.Series({"A": 0.5, "B": 0.2, "C": 0.2, "D": 0.1}),
    ]
    cap_cases = [0.45, 0.55, 0.3]
    map_cases = [
        {"A": "X", "B": "X", "C": "Y", "D": "Y"},
        {"A": "X"},
        {},
        {"A": "X", "B": "Y", "C": "Z", "D": "W"},
    ]
    for weights in weights_cases:
        for cap in cap_cases:
            for industry_map in map_cases:
                for name_cap in (None, 0.35):
                    local_out = _local_apply_industry_cap(weights, cap, industry_map, name_cap=name_cap)
                    worker_out = _apply_industry_cap(weights, cap, industry_map, name_cap=name_cap)
                    np.testing.assert_array_equal(
                        local_out.to_numpy(dtype=float),
                        worker_out.to_numpy(dtype=float),
                        err_msg=f"industry cap drifted (cap={cap}, map={industry_map}, name_cap={name_cap})",
                    )

    history = pd.DataFrame(
        {
            "A": [0.04, -0.04, 0.04, -0.04, 0.04, -0.04],
            "B": [0.01, -0.01, 0.01, -0.01, 0.01, -0.01],
            "C": [0.02, 0.0, -0.02, 0.01, -0.01, 0.02],
            "D": [np.nan, np.nan, 0.01, -0.01, 0.02, -0.02],
        }
    )
    local_spec_cases = [
        CompositionSpec(top_n=4, weighting="inverse_volatility"),
        CompositionSpec(top_n=4, weighting="inverse_volatility", inverse_vol_window=3),
        CompositionSpec(top_n=4, weighting="inverse_volatility", inverse_vol_min=0.05),
        CompositionSpec(top_n=4, weighting="inverse_volatility", inverse_vol_multiply_score=True),
    ]
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    history_cases = [history, history.iloc[:1], history.iloc[:5], None]
    for spec in local_spec_cases:
        payload = spec.as_payload()
        for returns_history in history_cases:
            if returns_history is None:
                with pytest.raises(
                    ValueError, match=re.escape("inverse_volatility weighting requires a returns panel")
                ):
                    _local_inverse_volatility_weights(spec, scores, None)
                with pytest.raises(
                    ValueError, match=re.escape("inverse_volatility weighting requires a returns panel")
                ):
                    _inverse_volatility_weights(scores, None, payload)
                continue
            local_out = _local_inverse_volatility_weights(spec, scores, returns_history)
            worker_out = _inverse_volatility_weights(scores, returns_history, payload)
            np.testing.assert_array_equal(
                local_out.to_numpy(dtype=float),
                worker_out.to_numpy(dtype=float),
                err_msg=f"inverse volatility drifted (spec={spec}, rows={len(returns_history)})",
            )
    degenerate = history.copy()
    degenerate["B"] = 0.01
    with pytest.raises(
        ValueError,
        match=re.escape("inverse_volatility weighting requires a positive volatility estimate for every selected name"),
    ):
        _local_inverse_volatility_weights(local_spec_cases[0], scores, degenerate)
    with pytest.raises(
        ValueError,
        match=re.escape("inverse_volatility weighting requires a positive volatility estimate for every selected name"),
    ):
        _inverse_volatility_weights(scores, degenerate, local_spec_cases[0].as_payload())

    hierarchical_cases = [
        CompositionSpec(
            top_n=3,
            weighting="hierarchical",
            hierarchical_groups={"g1": ("A", "B"), "g2": ("C", "D")},
        ),
        CompositionSpec(
            top_n=3,
            weighting="hierarchical",
            hierarchical_groups={"g1": ("A", "B"), "g2": ("C", "D")},
            hierarchical_group_weights={"g1": 0.7, "g2": 0.3},
        ),
        CompositionSpec(top_n=3, weighting="hierarchical", hierarchical_groups={"solo": ("A", "B", "C", "D")}),
    ]
    hier_scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0})
    for spec in hierarchical_cases:
        local_out = _local_hierarchical_weights(spec, hier_scores)
        worker_out = _hierarchical_weights(hier_scores, spec.as_payload())
        np.testing.assert_array_equal(
            local_out.to_numpy(dtype=float),
            worker_out.to_numpy(dtype=float),
            err_msg=f"hierarchical weighting drifted ({spec.hierarchical_groups})",
        )
    outside = CompositionSpec(top_n=2, weighting="hierarchical", hierarchical_groups={"g1": ("A", "B")})
    with pytest.raises(
        ValueError, match=re.escape("hierarchical weighting requires every selected name to belong to a group")
    ):
        _local_hierarchical_weights(outside, pd.Series({"A": 5.0, "Z": 1.0}))
    with pytest.raises(
        ValueError, match=re.escape("hierarchical weighting requires every selected name to belong to a group")
    ):
        _hierarchical_weights(pd.Series({"A": 5.0, "Z": 1.0}), outside.as_payload())

    # The buffered row builder must produce identical rows when the weighting
    # is inverse_volatility and a return history is supplied.
    dates = pd.bdate_range("2024-01-02", periods=3)
    row_scores = pd.DataFrame(
        [[50.0, 40.0, 30.0, 20.0], [10.0, 50.0, 20.0, 30.0], [20.0, 20.0, 50.0, 10.0]],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    buffered_specs = [
        CompositionSpec(top_n=3, weighting="inverse_volatility"),
        CompositionSpec(
            top_n=3,
            weighting="inverse_volatility",
            inverse_vol_window=3,
            inverse_vol_multiply_score=True,
            holding_buffer_entry=0.5,
            holding_buffer_retain=0.3,
            holding_buffer_max_extra=1,
        ),
        CompositionSpec(
            top_n=3,
            weighting="inverse_volatility",
            weight_cap=0.4,
            weight_floor=0.1,
            industry_weight_cap=0.5,
            industry_map={"A": "X", "B": "X", "C": "Y", "D": "Y"},
            holding_buffer_entry=0.6,
        ),
    ]
    for spec in buffered_specs:
        payload = spec.as_payload()
        for position in range(len(dates)):
            for held in (frozenset(), frozenset({"A", "C"})):
                local_row = _local_build_buffered_target_row(spec, row_scores.iloc[position], held, history)
                worker_row = _build_buffered_target_row(row_scores.iloc[position], payload, held, history)
                np.testing.assert_array_equal(
                    local_row.to_numpy(dtype=float),
                    worker_row.to_numpy(dtype=float),
                    err_msg=f"buffered inverse-vol row drifted (position={position}, held={sorted(held)})",
                )


@pytest.mark.parametrize(
    ("payload_override", "message"),
    [
        ({"weight_cap": 0.0}, "weight_cap must be in (0, 1] when provided"),
        ({"weight_cap": 1.5}, "weight_cap must be in (0, 1] when provided"),
        ({"industry_weight_cap": 0.0}, "industry_weight_cap must be in (0, 1] when provided"),
        ({"industry_weight_cap": 1.5}, "industry_weight_cap must be in (0, 1] when provided"),
        ({"industry_map": {"S5": "X"}}, "industry_map is only valid with industry_weight_cap"),
        (
            {"industry_weight_cap": 0.4, "industry_map": {"S5": ""}},
            "industry_map must map non-empty symbols to non-empty industry labels",
        ),
        ({"cash_residual_max": 0.0}, "cash_residual_max must be in (0, 1] when provided"),
        ({"cash_residual_max": 1.5}, "cash_residual_max must be in (0, 1] when provided"),
        ({"inverse_vol_window": 1}, "inverse_vol_window must be at least 2"),
        ({"inverse_vol_window": 2.5}, "inverse_vol_window must be a whole number"),
        ({"inverse_vol_min": 0.0}, "inverse_vol_min must be a positive finite number when provided"),
        ({"inverse_vol_multiply_score": "yes"}, "inverse_vol_multiply_score must be a boolean"),
        ({"weighting": "hierarchical"}, "hierarchical weighting requires hierarchical_groups"),
        (
            {"weighting": "hierarchical", "hierarchical_groups": {"g1": ["A", "B"], "g2": ["B", "C"]}},
            "hierarchical_groups members must not appear in two groups",
        ),
        (
            {
                "weighting": "hierarchical",
                "hierarchical_groups": {"g1": ["A", "B"], "g2": ["C", "D"]},
                "hierarchical_group_weights": {"g1": 1.0},
            },
            "hierarchical_group_weights must cover exactly the hierarchical_groups keys",
        ),
        (
            {"weight_floor": 0.5, "industry_weight_cap": 0.3},
            "weight_floor must not exceed industry_weight_cap",
        ),
    ],
)
def test_batch_c_operators_fail_closed_on_both_sides(world, payload_override: dict, message: str) -> None:
    """Invalid batch C composition parameters raise on both sides."""

    candidate = StrategyCandidate(
        name="parity_batch_c_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch_c_fail.parquet")
    request["candidate"]["composition"].update(payload_override)
    request_path, result_path = tmp_path / "request_batch_c_fail.json", tmp_path / "result_batch_c_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        CompositionSpec(top_n=4, **payload_override)
    assert str(local_error.value) == message


@pytest.mark.slow
def test_cash_residual_bound_fails_closed_end_to_end(world) -> None:
    """A target row whose residual exceeds cash_residual_max fails closed on
    both sides with the same message (an execution-time, not spec-time,
    failure)."""

    tmp_path: Path = world["tmp_path"]
    composition_payload = {
        "top_n": 4,
        "weighting": "equal",
        "weight_cap": 0.22,
        "cash_residual_max": 0.1,
    }
    candidate = StrategyCandidate(
        name="parity_cash_residual_fail",
        universe="csi_all",
        composition=CompositionSpec.from_payload(composition_payload),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_cash_residual_fail.parquet")
    request_path, result_path = tmp_path / "request_cash_residual_fail.json", tmp_path / "result_cash_residual_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    # 4 x 0.22 after the cap leaves 0.12 in cash: above the bound.
    with pytest.raises(ValueError, match=re.escape("cash residual exceeds cash_residual_max")):
        run_request(request_path, result_path)

    score = _score_panel_for(
        world,
        request,
        candidate.signal,
        _rebalance_dates(
            _load_close(str(world["base"]), "adjclose", world["start"], world["end"]).loc[
                pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])
            ].index,
            request["evaluation"]["rebalance_frequency"],
        ),
        list(SYMBOLS),
    )
    with pytest.raises(ValueError) as local_error:
        build_target_weights(
            CompositionSpec.from_payload(composition_payload),
            score,
            returns=_returns_panel_for(world),
        )
    assert str(local_error.value) == "cash residual exceeds cash_residual_max"


# ---------------------------------------------------------------------------
# Batch D: result-report audits (roadmap 9.4 / 10.2 / 10.3 / 10.4).  The audit
# configuration rides at the request level, outside the candidate payload, so
# none of these tests changes any strategy identity hash.
# ---------------------------------------------------------------------------


def _run_worker_with_audit(world, tmp_path: Path, candidate, audit, tag: str):
    request = world["evaluator"].request_for(candidate, audit=audit)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{tag}.parquet")
    request_path, result_path = tmp_path / f"request_{tag}.json", tmp_path / f"result_{tag}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload
    return request, payload


def _local_perturbation_evaluate(world, request):
    """Re-run the local pipeline for one perturbed candidate payload copy.

    Mirror of the worker's evaluate closure: deep-copy the request, apply the
    dotted key, rebuild the specs (which revalidates them) and re-run.
    """

    def evaluate(dotted_key: str, value):
        variant = copy.deepcopy(request)
        node = variant["candidate"]
        segments = dotted_key.split(".")
        for segment in segments[:-1]:
            node = node[segment]
        node[segments[-1]] = value
        payload = variant["candidate"]
        result = _local_portfolio(
            world,
            variant,
            CompositionSpec.from_payload(dict(payload["composition"])),
            SignalSpec.from_payload(dict(payload.get("signal") or {})),
            GatingSpec.from_payload(payload.get("gating")),
            BacktestSpec.from_payload(dict(payload["backtest"])),
        )
        return dict(result.metrics)

    return evaluate


def _local_ablation_metrics(world, request, candidate):
    """Per-factor ablated metrics through the local ungated two-argument path.

    Mirror of the worker's ablation loop: equal-weight blend of the kept
    final rank panels (no gating, no selection-layer operators), full
    composition spec, unchanged backtest spec -- the pre-existing ablation
    semantics the audit must formalise, not change.
    """

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    symbols = sorted(set(close.columns))
    factor_names = list(candidate.factor_names)
    factor_items = {str(item["name"]): item for item in request["artifact_set"]["factors"]}
    panels = {name: pd.read_parquet(factor_items[name]["artifact_path"]) for name in factor_names}
    directions = {name: int(request["factor_directions"][name]) for name in factor_names}
    ranks = build_factor_rank_panels(panels, directions, candidate.signal, index=rebalance_dates, columns=symbols)
    daily_returns = _returns_panel_for(world)
    benchmark = daily_returns.mean(axis=1, skipna=True)
    out = {}
    for removed in factor_names:
        kept = [name for name in factor_names if name != removed]
        score = blend_rank_panels(ranks, kept)
        targets = build_target_weights(candidate.composition, score, returns=daily_returns)
        ablated = run_portfolio_backtest(targets, daily_returns, spec=candidate.backtest, benchmark_returns=benchmark)
        out[removed] = dict(ablated.metrics)
    return out


def _local_audit_report(world, request, candidate, result, audit):
    report = {}
    if "concentration" in audit:
        section = audit["concentration"]
        report["concentration"] = concentration_report(
            result.weights,
            top_k=list(section.get("top_k") or DEFAULT_CONCENTRATION_TOP_K),
            industry_map=section.get("industry_map"),
            industry_threshold=section.get("industry_threshold"),
        )
    if "cost_sensitivity" in audit:
        section = audit["cost_sensitivity"]
        scenarios = [tuple(pair) for pair in (section.get("scenarios") or DEFAULT_COST_SENSITIVITY_SCENARIOS)]
        report["cost_sensitivity"] = cost_sensitivity_report(
            result.daily_gross_returns,
            result.turnover,
            result.buy_turnover,
            result.sell_turnover,
            scenarios,
            candidate.backtest.trading_days_per_year,
        )
    if "parameter_perturbation" in audit:
        report["parameter_perturbation"] = parameter_perturbation_report(
            candidate.as_payload(),
            audit["parameter_perturbation"]["perturbations"],
            dict(result.metrics),
            _local_perturbation_evaluate(world, request),
        )
    if "ablation" in audit:
        report["ablation"] = ablation_report(
            dict(result.metrics), _local_ablation_metrics(world, request, candidate)
        )
    if "stress" in audit:
        report["stress"] = stress_scenario_report(result.weights, audit["stress"])
    return report


def _assert_concentration_parity(worker_section: dict, local_section: dict) -> None:
    assert set(worker_section) == set(local_section)
    assert worker_section["by_date"]["dates"] == local_section["by_date"]["dates"]
    np.testing.assert_allclose(
        worker_section["by_date"]["hhi"], local_section["by_date"]["hhi"], atol=5e-11, rtol=0.0
    )
    for key, share in worker_section["by_date"]["top_k"].items():
        np.testing.assert_allclose(share, local_section["by_date"]["top_k"][key], atol=5e-11, rtol=0.0)
    for stat in ("mean", "max", "min"):
        assert worker_section["hhi"][stat] == pytest.approx(local_section["hhi"][stat], rel=1e-9, abs=1e-12)
    for stat in ("mean", "max"):
        assert worker_section["effective_names"][stat] == pytest.approx(
            local_section["effective_names"][stat], rel=1e-9, abs=1e-12
        )
    for key, summary in worker_section["top_k"].items():
        for stat in ("mean", "max"):
            assert summary[stat] == pytest.approx(local_section["top_k"][key][stat], rel=1e-9, abs=1e-12)
    if "industry" in worker_section:
        assert worker_section["industry"]["threshold"] == local_section["industry"]["threshold"]
        assert worker_section["industry"]["breach_count"] == local_section["industry"]["breach_count"]
        np.testing.assert_allclose(
            worker_section["industry"]["by_date"], local_section["industry"]["by_date"], atol=5e-11, rtol=0.0
        )


@pytest.mark.slow
def test_concentration_audit_matches_end_to_end(world) -> None:
    """ConcentrationMonitor (9.4): per-day indicators of the executed book.

    Discrimination: a top-2 book concentrates far more than a capped six-name
    book, so the audit numbers must genuinely track the configuration.
    """

    tmp_path: Path = world["tmp_path"]
    audit = {
        "concentration": {
            "top_k": [2, 5],
            "industry_map": {f"S{position}": ("X" if position < 3 else "Y") for position in range(len(SYMBOLS))},
            "industry_threshold": 0.55,
        }
    }
    concentrated = StrategyCandidate(
        name="parity_conc_tight",
        universe="csi_all",
        composition=CompositionSpec(top_n=2, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    diversified = StrategyCandidate(
        name="parity_conc_wide",
        universe="csi_all",
        composition=CompositionSpec(top_n=6, weighting="equal", weight_cap=0.2),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )

    reports = {}
    for candidate in (concentrated, diversified):
        request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, f"audit_{candidate.name}")
        local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
        expected = concentration_report(
            local.weights,
            top_k=[2, 5],
            industry_map=audit["concentration"]["industry_map"],
            industry_threshold=audit["concentration"]["industry_threshold"],
        )
        _assert_concentration_parity(payload["audit_report"]["concentration"], expected)
        reports[candidate.name] = payload["audit_report"]["concentration"]

    assert reports["parity_conc_tight"]["hhi"]["mean"] > reports["parity_conc_wide"]["hhi"]["mean"]
    assert reports["parity_conc_tight"]["top_k"]["2"]["mean"] > reports["parity_conc_wide"]["top_k"]["2"]["mean"]


@pytest.mark.slow
def test_cost_sensitivity_audit_matches_and_respects_the_cost_contract(world) -> None:
    """CostSensitivityAudit (10.2): the scenarios re-price the executed book.

    The production-rate scenario (10/10 bps) must reproduce the production
    metrics exactly -- the audit consumes the same gross and turnover series
    the engine produced -- and dearer scenarios must strictly degrade it.
    A cost key inside the candidate backtest is still rejected even when the
    audit asks for cost scenarios.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_cost_audit",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    audit = {"cost_sensitivity": {}}
    request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, "audit_cost")
    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    section = payload["audit_report"]["cost_sensitivity"]
    expected = cost_sensitivity_report(
        local.daily_gross_returns,
        local.turnover,
        local.buy_turnover,
        local.sell_turnover,
        [tuple(pair) for pair in DEFAULT_COST_SENSITIVITY_SCENARIOS],
        candidate.backtest.trading_days_per_year,
    )
    assert section["production"] == {"buy_bps": 10.0, "sell_bps": 10.0}
    assert [(row["buy_bps"], row["sell_bps"]) for row in section["scenarios"]] == [(10.0, 10.0), (15.0, 15.0), (20.0, 20.0)]
    for worker_row, local_row in zip(section["scenarios"], expected["scenarios"]):
        assert set(worker_row) == set(local_row)
        for key, value in local_row.items():
            assert float(worker_row[key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12), key

    # The production-rate scenario reproduces the production metrics.
    production_row = section["scenarios"][0]
    for key in ("total_return", "annualized_return", "cagr", "annualized_vol", "sharpe", "max_drawdown", "annual_turnover", "total_transaction_cost"):
        assert production_row[key] == pytest.approx(float(payload["metrics"][key]), rel=1e-12, abs=1e-15), key

    returns = [row["annualized_return"] for row in section["scenarios"]]
    costs = [row["total_transaction_cost"] for row in section["scenarios"]]
    assert returns[0] > returns[1] > returns[2]
    assert costs[0] < costs[1] < costs[2]

    # Cost-key ban interaction: the audit lives at the request level, so the
    # candidate payload stays cost-free and a smuggled cost key still fails.
    bad_request = world["evaluator"].request_for(candidate, audit=audit)
    bad_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_audit_cost_bad.parquet")
    bad_request["candidate"]["backtest"]["buy_cost_bps"] = 5.0
    bad_path, bad_result = tmp_path / "request_audit_cost_bad.json", tmp_path / "result_audit_cost_bad.json"
    bad_path.write_text(json.dumps(bad_request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("candidate backtest must not contain transaction-cost fields")):
        run_request(bad_path, bad_result)


def test_parameter_perturbation_audit_lazy_keys_fail_closed(world) -> None:
    """ParameterPerturbationAudit (10.3): lazy payload keys are rejected.

    ``composition.inverse_vol_window`` travels in the canonical payload only
    under the inverse-volatility weighting; perturbing it under another
    weighting would silently change nothing and get misreported as
    insensitivity, so the audit fails closed on both ends instead.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_perturb_lazy",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    audit = {"parameter_perturbation": {"perturbations": {"composition.inverse_vol_window": [10]}}}
    request = world["evaluator"].request_for(candidate, audit=audit)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_audit_perturb_lazy.parquet")
    request_path, result_path = tmp_path / "request_audit_perturb_lazy.json", tmp_path / "result_audit_perturb_lazy.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    message = "parameter perturbation key 'composition.inverse_vol_window' is not present in the candidate payload"
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    # Same rule on the local side, before any variant is evaluated.
    with pytest.raises(ValueError, match=re.escape(message)):
        parameter_perturbation_report(
            candidate.as_payload(),
            audit["parameter_perturbation"]["perturbations"],
            dict(_local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest).metrics),
            _local_perturbation_evaluate(world, request),
        )


@pytest.mark.slow
def test_parameter_perturbation_audit_matches_end_to_end(world) -> None:
    """ParameterPerturbationAudit (10.3): variants re-run each engine.

    Valid variants agree between local and worker on metrics and marginals;
    a variant that fails validation is recorded as an error variant on both
    ends (each end surfaces its own engine's message) without aborting the
    audit, and a genuinely sensitive parameter moves the metrics.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_perturb",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal", weight_cap=0.4),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    audit = {
        "parameter_perturbation": {
            "perturbations": {
                "composition.weight_cap": [0.25, 0.3, 1.5],
                "composition.top_n": [2],
            }
        }
    }
    request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, "audit_perturb")
    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    expected = parameter_perturbation_report(
        candidate.as_payload(),
        audit["parameter_perturbation"]["perturbations"],
        dict(local.metrics),
        _local_perturbation_evaluate(world, request),
    )

    section = payload["audit_report"]["parameter_perturbation"]
    assert section["baseline"]["metrics"] == {key: pytest.approx(value) for key, value in expected["baseline"]["metrics"].items()}
    assert [(v["key"], v["value"]) for v in section["variants"]] == [(v["key"], v["value"]) for v in expected["variants"]]
    assert [v["status"] for v in section["variants"]] == ["ok", "ok", "error", "ok"]
    # The invalid cap is recorded as an error variant by both engines with the
    # same message (the worker's validator is aligned with CompositionSpec).
    assert section["variants"][2]["error"] == "weight_cap must be in (0, 1] when provided"
    assert expected["variants"][2]["error"] == "weight_cap must be in (0, 1] when provided"
    for worker_variant, local_variant in zip(section["variants"], expected["variants"]):
        if worker_variant["status"] != "ok":
            continue
        for key, value in local_variant["metrics"].items():
            assert float(worker_variant["metrics"][key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12), key
            assert float(worker_variant["marginal"][key]) == pytest.approx(
                float(local_variant["marginal"][key]), rel=1e-9, abs=1e-12
            ), key

    # Discrimination: a genuinely different parameter value moves the book.
    top2 = next(v for v in section["variants"] if v["key"] == "composition.top_n")
    assert abs(top2["marginal"]["annualized_return"]) > 1e-9


@pytest.mark.slow
def test_ablation_audit_matches_the_contributions_path(world) -> None:
    """AblationAudit (10.4): formalise the existing ungated ablation loop.

    The audit's marginals must be exactly the existing
    ``factor_contributions`` (same loop, same formulas), the ablated metrics
    must agree between local and worker, and a request without an audit
    object must produce a result without ``audit_report``.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_ablation",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    audit = {"ablation": {}}
    request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, "audit_ablation")
    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    section = payload["audit_report"]["ablation"]
    assert section["mode"] == "factors"
    assert set(section["factors"]) == set(candidate.factor_names)
    for name, factor_section in section["factors"].items():
        # Same loop, same four formulas: exact equality with the legacy output.
        assert factor_section["marginal"] == payload["factor_contributions"][name]

    expected = ablation_report(
        dict(local.metrics), _local_ablation_metrics(world, request, candidate)
    )
    for name, factor_section in expected["factors"].items():
        for key, value in factor_section["ablated"].items():
            assert float(section["factors"][name]["ablated"][key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12), key
        for key, value in factor_section["marginal"].items():
            assert float(section["factors"][name]["marginal"][key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12), key

    # Disabled pass-through: no audit object, no audit_report field at all.
    plain_request = world["evaluator"].request_for(candidate)
    plain_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_no_audit.parquet")
    plain_path, plain_result = tmp_path / "request_no_audit.json", tmp_path / "result_no_audit.json"
    plain_path.write_text(json.dumps(plain_request), encoding="utf-8")
    plain_payload = run_request(plain_path, plain_result)
    assert plain_payload["status"] == "success", plain_payload
    assert "audit_report" not in plain_payload


_STRESS_TABLE_VERSION = "2024Q1"


def _stress_section(*, mode: str | None = None, threshold: float = 0.2) -> dict:
    """A versioned scenario table covering the fixture's whole universe.

    ``broad`` hits every name through ``default_shock`` only (explicit
    coverage zero), ``named`` names every symbol explicitly at twice the
    shock, so a worker that ignored ``symbol_shocks`` cannot reproduce the
    local report.
    """

    section = {
        "scenario_table_version": _STRESS_TABLE_VERSION,
        "scenarios": {
            "broad": {"default_shock": -0.2},
            "named": {"symbol_shocks": {symbol: -0.4 for symbol in SYMBOLS}},
        },
        "loss_threshold": threshold,
    }
    if mode is not None:
        section["mode"] = mode
    return section


@pytest.mark.slow
def test_stress_audit_matches_end_to_end(world) -> None:
    """StressScenarioCheck (9.3): replay the executed book on a scenario table.

    The table is audit-level data and never enters the candidate payload; the
    report is computed from the last row of the executed weights panel on both
    ends.  Discrimination: an all-default scenario must be strictly milder and
    strictly less explicitly covered than one that names every symbol.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_stress",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    audit = {"stress": _stress_section()}
    request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, "audit_stress")
    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)

    section = payload["audit_report"]["stress"]
    expected = stress_scenario_report(local.weights, audit["stress"])
    assert section["scenario_table_version"] == _STRESS_TABLE_VERSION
    assert expected["scenario_table_version"] == _STRESS_TABLE_VERSION
    assert section["mode"] == expected["mode"] == DEFAULT_STRESS_MODE
    assert section["as_of"] == expected["as_of"]
    assert section["held_names"] == expected["held_names"] > 0
    assert [row["scenario_id"] for row in section["scenarios"]] == ["broad", "named"]
    for worker_row, local_row in zip(section["scenarios"], expected["scenarios"]):
        assert set(worker_row) == set(local_row)
        for key, value in local_row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(worker_row[key]) == pytest.approx(float(value), rel=1e-9, abs=1e-12), key
            else:
                assert worker_row[key] == value, key

    # Discrimination: the named table is both dearer and explicitly covered.
    broad, named = section["scenarios"]
    assert named["explicit_coverage"] > broad["explicit_coverage"] == 0.0
    assert named["loss"] > broad["loss"]
    assert broad["passed"] is True
    assert named["passed"] is False
    assert section["passed"] is False
    assert section["worst_scenario"] == "named"
    assert section["max_loss"] == pytest.approx(named["loss"])
    # An all-default table cannot be book-specific: the loss is the shock
    # times whatever share is invested.
    assert broad["loss"] == pytest.approx(0.2 * sum(local.weights.iloc[-1].clip(lower=0.0)))


@pytest.mark.slow
def test_stress_audit_warn_reports_without_touching_weights_and_reject_aborts(world) -> None:
    """``warn`` writes the report only; ``reject`` fails the audit closed.

    Neither mode may move a weight: the warn run's metrics must be identical
    to the same candidate evaluated without any audit, and the reject run must
    produce no result file at all.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_stress_mode",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )

    plain_request = world["evaluator"].request_for(candidate)
    plain_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_stress_plain.parquet")
    plain_path, plain_result = tmp_path / "request_stress_plain.json", tmp_path / "result_stress_plain.json"
    plain_path.write_text(json.dumps(plain_request), encoding="utf-8")
    plain_payload = run_request(plain_path, plain_result)
    assert plain_payload["status"] == "success", plain_payload
    assert "audit_report" not in plain_payload

    warn_audit = {"stress": _stress_section(mode="warn")}
    warn_request, warn_payload = _run_worker_with_audit(world, tmp_path, candidate, warn_audit, "audit_stress_warn")
    warn_section = warn_payload["audit_report"]["stress"]
    assert warn_section["mode"] == "warn"
    assert warn_section["passed"] is False
    # A breaching warn audit still delivers the untouched production metrics.
    assert set(warn_payload["metrics"]) == set(plain_payload["metrics"])
    for key, value in plain_payload["metrics"].items():
        assert float(warn_payload["metrics"][key]) == pytest.approx(float(value), rel=0.0, abs=0.0), key

    local = _local_portfolio(
        world, warn_request, candidate.composition, candidate.signal, backtest=candidate.backtest
    )
    for key, value in plain_payload["metrics"].items():
        assert float(local.metrics[key]) == pytest.approx(float(value), rel=0.0, abs=0.0), key
    warn_report = stress_scenario_report(local.weights, warn_audit["stress"])
    assert warn_report["passed"] is False
    ensure_stress_passed(warn_report)  # warn never aborts

    reject_audit = {"stress": _stress_section(mode="reject")}
    reject_request = world["evaluator"].request_for(candidate, audit=reject_audit)
    reject_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_stress_reject.parquet")
    reject_path, reject_result = tmp_path / "request_stress_reject.json", tmp_path / "result_stress_reject.json"
    reject_path.write_text(json.dumps(reject_request), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        run_request(reject_path, reject_result)
    message = str(failure.value)
    assert message.startswith("stress scenario check failed: loss ")
    assert "exceeds loss_threshold 0.2 for scenario 'named'" in message
    assert f"(scenario_table_version {_STRESS_TABLE_VERSION!r})" in message
    # Fail-closed means no result artifact at all.
    assert not reject_result.exists()

    # The local mirror aborts on the same report with the same message shape.
    reject_report = stress_scenario_report(local.weights, reject_audit["stress"])
    assert reject_report["passed"] is False
    with pytest.raises(ValueError) as local_failure:
        ensure_stress_passed(reject_report)
    assert str(local_failure.value).startswith("stress scenario check failed: loss ")
    assert "for scenario 'named'" in str(local_failure.value)


def test_stress_scenario_table_stays_out_of_the_candidate_payload(world) -> None:
    """The audit config must not move a strategy identity hash.

    The scenario table is versioned data owned by the caller; an agent never
    sees it (it is not in the proposal vocabulary) and enabling the check
    cannot change the canonical payload the identity is hashed from.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_stress_payload",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    payload = candidate.as_payload()
    assert "audit" not in payload
    assert "stress" not in payload
    assert "scenario_table_version" not in json.dumps(payload)

    with_audit = world["evaluator"].request_for(candidate, audit={"stress": _stress_section()})
    without_audit = world["evaluator"].request_for(candidate)
    assert with_audit["candidate"] == without_audit["candidate"]
    assert with_audit["audit"] == {"stress": _stress_section()}
    assert "audit" not in without_audit
    assert strategy_id_for(candidate.name, payload) == strategy_id_for(candidate.name, with_audit["candidate"])


# ---------------------------------------------------------------------------
# Batch 7: FeasibilityAudit (10.6) and DataLeakageAudit (10.7).  Both are
# report-only sections riding in the request-level audit object; the
# feasibility trace is an observation sink inside the order layer, so the
# executed book must stay bit-identical whether or not it is collected.
# ---------------------------------------------------------------------------


def _feasibility_candidate(name: str, backtest: BacktestSpec, execution_data: ExecutionDataSpec):
    return StrategyCandidate(
        name=name,
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=backtest,
        signal=SignalSpec(),
        execution_data=execution_data,
    )


#: The batch-7 block sits above the batch-6 helpers, so it carries its own copy
#: of the order-layer calendar keys and flat-panel helper.
_BATCH7_DAY_KEYS = tuple(pd.Timestamp(date).date().isoformat() for date in EVAL_DATES)


def _batch7_flat_panel(value: float) -> dict:
    return {day: dict.fromkeys(SYMBOLS, value) for day in _BATCH7_DAY_KEYS}


_FEASIBILITY_CASES = {
    "participation": (
        BacktestSpec(
            signal_lag=1,
            participation_max_rate=0.5,
            participation_max_days=3,
            participation_unfilled="carry",
            participation_amount_window=1,
        ),
        ExecutionDataSpec(trading_dates=list(_BATCH7_DAY_KEYS), amount=_batch7_flat_panel(0.02)),
    ),
    "staging": (
        BacktestSpec(signal_lag=1, staged_max_days=3, staged_priority="weight"),
        ExecutionDataSpec(trading_dates=list(_BATCH7_DAY_KEYS)),
    ),
    "lot_rounding": (
        BacktestSpec(signal_lag=1, lot_rounding=True, lot_rounding_mode="down"),
        ExecutionDataSpec(
            price=_batch7_flat_panel(0.0375),
            lot_sizes={symbol: 1 for symbol in SYMBOLS},
        ),
    ),
    "tradability": (
        BacktestSpec(signal_lag=1, side_aware_tradability="block_both"),
        ExecutionDataSpec(
            price=_batch7_flat_panel(2.0),
            halt={_BATCH7_DAY_KEYS[1]: dict.fromkeys(SYMBOLS, "suspended")},
        ),
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("label", sorted(_FEASIBILITY_CASES))
def test_feasibility_audit_matches_end_to_end(world, label: str) -> None:
    """FeasibilityAudit (10.6): both ends agree and the reason split is real.

    Each case enables exactly one order-layer operator, so the day table must
    attribute the shortfall to that operator's bucket and leave the others at
    zero -- the discrimination that proves the report tracks the execution
    chain rather than reporting a constant.
    """

    tmp_path: Path = world["tmp_path"]
    backtest, execution_data = _FEASIBILITY_CASES[label]
    candidate = _feasibility_candidate(f"parity_feas_{label}", backtest, execution_data)
    audit = {"feasibility": {"min_executable_ratio": 0.99, "allow_cash_residual": False}}
    request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, f"feas_{label}")

    trace: list = []
    _local_portfolio(
        world,
        request,
        candidate.composition,
        candidate.signal,
        candidate.gating,
        candidate.backtest,
        execution_trace=trace,
    )
    expected = feasibility_report(
        feasibility_days(trace), min_executable_ratio=0.99, allow_cash_residual=False
    )
    section = payload["audit_report"]["feasibility"]
    assert json.dumps(section, sort_keys=True) == json.dumps(expected, sort_keys=True)

    # Discrimination: the live operator is the one that shows up.
    reasons = section["summary"]["unfilled_by_reason"]
    expected_bucket = {"staging": "staging"}.get(label, label)
    assert reasons[expected_bucket] > 0.0, (label, reasons)
    for other, amount in reasons.items():
        if other != expected_bucket:
            assert amount == 0.0, (label, other, amount)

    # The configured floor is breached, and it is breached as a *record*: the
    # executed book is what the engine produced either way.
    assert section["min_executable_ratio"] == 0.99
    assert section["allow_cash_residual"] is False
    assert any(v["kind"] == "min_executable_ratio" for v in section["violations"])
    assert abs(sum(v["unfilled"] for v in section["violations"] if v["kind"] == "cash_residual_not_allowed")) > 0.0


@pytest.mark.slow
def test_feasibility_trace_does_not_move_the_executed_book(world) -> None:
    """The observation sink is pure: weights are bit-identical with and without.

    This is the guard that keeps the audit honest -- a trace that perturbed the
    run would make the reported shortfall describe a different strategy than the
    one whose metrics were also reported.
    """

    tmp_path: Path = world["tmp_path"]
    backtest, execution_data = _FEASIBILITY_CASES["participation"]
    candidate = _feasibility_candidate("parity_feas_pure", backtest, execution_data)
    request = world["evaluator"].request_for(candidate, audit={"feasibility": {}})
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_feas_pure.parquet")

    plain = _local_portfolio(
        world, request, candidate.composition, candidate.signal, candidate.gating, candidate.backtest
    )
    trace: list = []
    traced = _local_portfolio(
        world,
        request,
        candidate.composition,
        candidate.signal,
        candidate.gating,
        candidate.backtest,
        execution_trace=trace,
    )
    assert (plain.weights.to_numpy() == traced.weights.to_numpy()).all()
    assert trace, "the sink must observe the order layer"
    for entry in trace:
        assert {"allocator", "tradability", "lot_rounding"} <= set(entry["stages"])


def test_feasibility_audit_absent_without_audit(world) -> None:
    """A request with no audit object carries no audit_report key at all."""

    tmp_path: Path = world["tmp_path"]
    backtest, execution_data = _FEASIBILITY_CASES["participation"]
    candidate = _feasibility_candidate("parity_feas_no_audit", backtest, execution_data)
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_feas_no_audit.parquet")
    request_path, result_path = tmp_path / "request_feas_no_audit.json", tmp_path / "result_feas_no_audit.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success"
    assert "audit_report" not in payload
    assert "audit" not in request


@pytest.mark.slow
def test_data_leakage_audit_matches_end_to_end(world) -> None:
    """DataLeakageAudit (10.7): both ends agree on declarations and risks.

    Two candidates differ only in what they declare: one leaves every lag at
    the engine default, the other shifts the membership lag past the injected
    availability standard.  The risk list must move with the declaration, which
    is what proves the audit reads the payload rather than reporting a constant.
    """

    tmp_path: Path = world["tmp_path"]
    audit = {
        "data_leakage": {
            "field_availability_lag": {"signal_lag": 2, "index_membership": 3, "listing_age": 5},
            "announcement_to_effective": {"signal_lag": 0, "index_membership": 3},
        }
    }
    base = StrategyCandidate(
        name="parity_leak_short",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    membership = {day.strftime("%Y-%m-%d"): dict.fromkeys(SYMBOLS, True) for day in EVAL_DATES}
    lagged = StrategyCandidate(
        name="parity_leak_lagged",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=3),
        universe_rules=UniverseSpec(
            trading_dates=list(_BATCH7_DAY_KEYS),
            index_membership=IndexMembershipRule(membership=membership, lag_days=3),
            listing_age=ListingAgeRule(listing_dates={symbol: "2023-01-01" for symbol in SYMBOLS}, min_days=5),
        ),
    )

    reports = {}
    for candidate in (base, lagged):
        request, payload = _run_worker_with_audit(world, tmp_path, candidate, audit, f"leak_{candidate.name}")
        section = payload["audit_report"]["data_leakage"]
        expected = data_leakage_report(
            leakage_declarations(request["candidate"]),
            field_availability_lag=audit["data_leakage"]["field_availability_lag"],
            announcement_to_effective=audit["data_leakage"]["announcement_to_effective"],
            provenance=section["provenance"],
        )
        assert json.dumps(section, sort_keys=True) == json.dumps(expected, sort_keys=True)
        # Provenance comes from the request identity, not an invented version.
        assert section["provenance"]["snapshot_id"] == request["snapshot_id"]
        assert section["provenance"]["snapshot_checksum"] == request["snapshot_checksum"]
        reports[candidate.name] = section

    # The undeclared-field path: listing_age declares no lag, so it is reported
    # as unknown rather than silently passed or fail-closed.
    lagged_report = reports["parity_leak_lagged"]
    listing_entry = next(entry for entry in lagged_report["undeclared"] if entry["field"] == "listing_age")
    assert listing_entry["declared_lag"] is None
    assert listing_entry["severity"] == "unknown"
    assert listing_entry["location"] == "universe_rules.listing_age.min_days"
    assert all(entry["severity"] == "unknown" for entry in lagged_report["undeclared"])
    assert any(risk["reason"] == "unknown" for risk in lagged_report["risks"])

    # Discrimination: shifting the membership lag up to the standard clears the
    # breach the short candidate shows for signal_lag.
    short_fields = {risk["field"] for risk in reports["parity_leak_short"]["risks"]}
    lagged_fields = {
        risk["field"]
        for risk in lagged_report["risks"]
        if risk["reason"] == "declared_lag_below_available_lag"
    }
    assert "signal_lag" in short_fields
    assert "signal_lag" not in lagged_fields


@pytest.mark.slow
def test_data_leakage_declarations_track_the_payload(world) -> None:
    """Every harvested declaration must resolve to a real payload location."""

    tmp_path: Path = world["tmp_path"]
    membership = {day.strftime("%Y-%m-%d"): dict.fromkeys(SYMBOLS, True) for day in EVAL_DATES}
    candidate = StrategyCandidate(
        name="parity_leak_declared",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        universe_rules=UniverseSpec(index_membership=IndexMembershipRule(membership=membership, lag_days=2)),
        signal=SignalSpec(regime_lag=2),
    )
    request, payload = _run_worker_with_audit(
        world, tmp_path, candidate, {"data_leakage": {"field_availability_lag": {"signal_lag": 1}, "announcement_to_effective": {"x": 1}}}, "leak_declared"
    )
    section = payload["audit_report"]["data_leakage"]
    declared = {entry["field"]: entry for entry in section["declarations"]}
    assert declared["index_membership"]["declared_lag"] == 2
    assert declared["signal_lag"]["declared_lag"] == 1
    payload_node = request["candidate"]
    for entry in section["declarations"]:
        node = payload_node
        for segment in entry["location"].split("."):
            assert isinstance(node, dict) and segment in node, entry
            node = node[segment]


@pytest.mark.parametrize(
    ("audit", "message"),
    [
        (
            {"feasibility": {"min_executable_ratio": 0.0}},
            "audit feasibility min_executable_ratio must be in (0, 1]",
        ),
        (
            {"feasibility": {"min_executable_ratio": 1.5}},
            "audit feasibility min_executable_ratio must be in (0, 1]",
        ),
        (
            {"feasibility": {"min_executable_ratio": True}},
            "audit feasibility min_executable_ratio must be in (0, 1]",
        ),
        (
            {"feasibility": {"allow_cash_residual": "yes"}},
            "audit feasibility allow_cash_residual must be a boolean",
        ),
        (
            {"feasibility": []},
            "request audit section 'feasibility' must be an object",
        ),
        (
            {"data_leakage": {}},
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {"data_leakage": {"field_availability_lag": {}, "announcement_to_effective": {"x": 1}}},
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {
                "data_leakage": {
                    "field_availability_lag": {"signal_lag": -1},
                    "announcement_to_effective": {"x": 1},
                }
            },
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {
                "data_leakage": {
                    "field_availability_lag": {"signal_lag": 1.5},
                    "announcement_to_effective": {"x": 1},
                }
            },
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {
                "data_leakage": {
                    "field_availability_lag": {"signal_lag": True},
                    "announcement_to_effective": {"x": 1},
                }
            },
            "audit data_leakage field_availability_lag must map non-empty field names to non-negative whole days",
        ),
        (
            {
                "data_leakage": {
                    "field_availability_lag": {"signal_lag": 1},
                    "announcement_to_effective": {},
                }
            },
            "audit data_leakage announcement_to_effective must map non-empty rule names to non-negative whole days",
        ),
        (
            {"data_leakage": {"field_availability_lag": {"signal_lag": 1}}},
            "audit data_leakage announcement_to_effective must map non-empty rule names to non-negative whole days",
        ),
        (
            {"feasibility": {}, "data_leakage": []},
            "request audit section 'data_leakage' must be an object",
        ),
    ],
)
def test_batch7_audit_sections_fail_closed_on_both_sides(world, audit: dict, message: str) -> None:
    """Illegal parameters in either new section raise on both sides."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_batch7_fail",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate, audit=audit)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch7_fail.parquet")
    request_path, result_path = tmp_path / "request_batch7_fail.json", tmp_path / "result_batch7_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        validate_audit_config(audit)
    assert str(local_error.value) == message


def test_batch7_audit_sections_stay_out_of_the_candidate_payload(world) -> None:
    """Neither new section moves a strategy identity hash."""

    candidate = StrategyCandidate(
        name="parity_batch7_payload",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    payload = candidate.as_payload()
    for key in ("feasibility", "data_leakage", "field_availability_lag", "min_executable_ratio"):
        assert key not in json.dumps(payload)

    audit = {
        "feasibility": {"min_executable_ratio": 0.5, "allow_cash_residual": False},
        "data_leakage": {"field_availability_lag": {"signal_lag": 1}, "announcement_to_effective": {"signal_lag": 0}},
    }
    with_audit = world["evaluator"].request_for(candidate, audit=audit)
    without_audit = world["evaluator"].request_for(candidate)
    assert with_audit["candidate"] == without_audit["candidate"]
    assert with_audit["audit"] == audit
    assert "audit" not in without_audit
    assert strategy_id_for(candidate.name, payload) == strategy_id_for(candidate.name, with_audit["candidate"])


# ---------------------------------------------------------------------------
# Batch E: candidate-pool / tradability filters (roadmap 3.1-3.8).  One
# shared eligible mask consumed by the signal front-end before ranking; the
# data panels ride in the candidate payload (industry_map injection pattern).
# ---------------------------------------------------------------------------


def _world_raw_prices() -> dict[str, dict[str, float]]:
    """The fixture's raw close series rebuilt from its deterministic formula."""

    return {
        day.strftime("%Y-%m-%d"): {
            symbol: 100.0 * (1.0 + 0.0005 * position_day + 0.0008 * position)
            for position, symbol in enumerate(SYMBOLS)
        }
        for position_day, day in enumerate(PRICE_DATES)
    }


def _world_price_dates() -> list[str]:
    return [day.strftime("%Y-%m-%d") for day in PRICE_DATES]


def _world_decision_dates(world) -> pd.DatetimeIndex:
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    return _rebalance_dates(calendar, "monthly")


def _universe_candidate(name: str, universe_payload: dict | None, *, top_n: int = 6, gating: GatingSpec | None = None) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        universe="csi_all",
        composition=CompositionSpec(top_n=top_n, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        gating=gating if gating is not None else GatingSpec(),
        universe_rules=None if universe_payload is None else UniverseSpec.from_payload(universe_payload),
    )


def _run_universe_e2e(world, tmp_path: Path, candidate: StrategyCandidate, tag: str):
    """Run the candidate through both engines and assert full parity."""

    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{tag}.parquet")
    request_path, result_path = tmp_path / f"request_{tag}.json", tmp_path / f"result_{tag}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload
    local = _local_portfolio(
        world,
        request,
        candidate.composition,
        candidate.signal,
        gating=candidate.gating,
        backtest=candidate.backtest,
    )
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))
    return local


def _baseline_local(world, tmp_path: Path, tag: str, *, top_n: int = 6, gating: GatingSpec | None = None):
    baseline = _universe_candidate(f"parity_universe_base_{tag}", None, top_n=top_n, gating=gating)
    request = world["evaluator"].request_for(baseline)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_base_{tag}.parquet")
    return _local_portfolio(
        world, request, baseline.composition, baseline.signal, gating=baseline.gating, backtest=baseline.backtest
    )


@pytest.mark.slow
def test_universe_base_membership_listing_status_filters_parity(world, tmp_path: Path) -> None:
    price_dates = _world_price_dates()
    payloads = {
        # 3.1: only S0..S3 stay in the pool.
        "base": {"base_universe": {"symbols": ["S0", "S1", "S2", "S3"]}},
        # 3.2: S4/S5 are never members of the injected PIT panel.
        "membership": {
            "index_membership": {
                "membership": {day: {symbol: True for symbol in ("S0", "S1", "S2", "S3")} for day in price_dates},
                "lag_days": 0,
                "exclude_recent_days": 0,
            }
        },
        # 3.3: S4/S5 list late and cannot reach the 30-trading-day minimum.
        "listing_age": {
            "trading_dates": price_dates,
            "listing_age": {
                "listing_dates": {
                    **{symbol: "1990-01-01" for symbol in ("S0", "S1", "S2", "S3")},
                    "S4": price_dates[70],
                    "S5": price_dates[70],
                },
                "min_days": 30,
            },
        },
        # 3.4: S4/S5 carry an ST flag on every injected day.
        "status": {
            "status_exclusion": {
                "statuses": {day: {"S4": "ST", "S5": "ST"} for day in price_dates},
                "excluded_statuses": ["ST"],
            }
        },
    }
    baseline = _baseline_local(world, tmp_path, "filters")
    for tag, payload in payloads.items():
        candidate = _universe_candidate(f"parity_universe_{tag}", payload)
        local = _run_universe_e2e(world, tmp_path, candidate, f"universe_{tag}")
        # Discrimination (fact 23): with top_n=6 the baseline holds the whole
        # cross-section, so any real exclusion must change the book.
        assert not local.weights.equals(baseline.weights), tag
        assert set(local.weights.columns) == set(baseline.weights.columns)


@pytest.mark.slow
def test_universe_price_limit_liquidity_range_quality_filters_parity(world, tmp_path: Path) -> None:
    prices = _world_raw_prices()
    price_dates = _world_price_dates()
    limit_up = {
        day: {symbol: value + (0.0 if symbol in ("S4", "S5") else 1.0) for symbol, value in row.items()}
        for day, row in prices.items()
    }
    amount = {
        day: {symbol: (1.0 if symbol in ("S4", "S5") else 1_000_000.0) for symbol in SYMBOLS}
        for day in price_dates
    }
    non_positive = {
        day: {symbol: (-1.0 if symbol in ("S4", "S5") else value) for symbol, value in row.items()}
        for day, row in prices.items()
    }
    payloads = {
        # 3.5: S4/S5 close exactly at their limit-up price on every day.
        "price_limit": {
            "price": prices,
            "price_limit": {"limit_up": limit_up, "limit_down": None, "exclude_limit_up": True, "exclude_limit_down": False},
        },
        # 3.6: S4/S5 never clear the trailing mean-amount floor.
        "liquidity": {
            "trading_dates": price_dates,
            "liquidity": {"amount": amount, "window": 20, "min_amount": 100_000.0, "quantile": None, "missing": "exclude"},
        },
        # 3.7: S4/S5 eventually break the upper price band.
        "price_range": {
            "trading_dates": price_dates,
            "price": prices,
            "price_range": {"min_price": None, "max_price": 102.0, "consecutive_days": 1},
        },
        # 3.8: the injected raw price of S4/S5 is non-positive.
        "data_quality": {
            "price": non_positive,
            "data_quality": {"min_coverage": None, "max_missing_run": None, "window": 20, "non_positive_price": True},
        },
    }
    baseline = _baseline_local(world, tmp_path, "data_filters")
    for tag, payload in payloads.items():
        candidate = _universe_candidate(f"parity_universe_{tag}", payload)
        local = _run_universe_e2e(world, tmp_path, candidate, f"universe_{tag}")
        assert not local.weights.equals(baseline.weights), tag


@pytest.mark.slow
def test_universe_chain_and_gating_combinations_parity(world, tmp_path: Path) -> None:
    prices = _world_raw_prices()
    price_dates = _world_price_dates()
    amount = {
        day: {symbol: (1.0 if symbol == "S5" else 1_000_000.0) for symbol in SYMBOLS}
        for day in price_dates
    }
    chain = {
        "trading_dates": price_dates,
        "base_universe": {"symbols": ["S0", "S1", "S2", "S3", "S4"]},
        "price": prices,
        "liquidity": {"amount": amount, "window": 20, "min_amount": 100_000.0, "quantile": None, "missing": "exclude"},
        # 104.5 keeps at least two names eligible on the last decision date
        # (S2..S4 fall out only up there) while still excluding late dates.
        "price_range": {"min_price": None, "max_price": 104.5, "consecutive_days": 1},
        "status_exclusion": {
            "statuses": {day: {"S3": "suspended"} for day in price_dates},
            "excluded_statuses": ["suspended"],
        },
    }
    candidate = _universe_candidate("parity_universe_chain", chain)
    local = _run_universe_e2e(world, tmp_path, candidate, "universe_chain")
    baseline = _baseline_local(world, tmp_path, "chain")
    assert not local.weights.equals(baseline.weights)

    # Filter + gating: the cascade sees the post-filter rank cross-section.
    gated_payload = {"base_universe": {"symbols": ["S0", "S1", "S2", "S3", "S4"]}}
    gating = GatingSpec.from_payload({"layers": [{"factor": "value", "direction": "top", "quantile": 0.5, "min_pass": 2}]})
    gated = _universe_candidate("parity_universe_gated", gated_payload, top_n=4, gating=gating)
    gated_local = _run_universe_e2e(world, tmp_path, gated, "universe_gated")
    assert not gated_local.weights.equals(baseline.weights)


@pytest.mark.slow
def test_universe_integral_float_params_execute_identically(world, tmp_path: Path) -> None:
    """Integral floats in whole-number fields are normalised on both ends.

    The pre-fix symptom: the local spec kept ``lag_days=1.0`` and the
    executor crashed on ``shift(1.0)`` while the worker's ``int()`` call ran
    fine.  After the write-back both engines see the same int and agree.
    """

    price_dates = _world_price_dates()
    payload = {
        "index_membership": {
            "membership": {day: {symbol: True for symbol in ("S0", "S1", "S2", "S3")} for day in price_dates},
            "lag_days": 1.0,
            "exclude_recent_days": 0.0,
        }
    }
    candidate = _universe_candidate("parity_universe_float_params", payload)
    local = _run_universe_e2e(world, tmp_path, candidate, "universe_float_params")
    baseline = _baseline_local(world, tmp_path, "float_params")
    assert not local.weights.equals(baseline.weights)


def test_universe_direct_mask_comparison_boundaries() -> None:
    """The two mask engines agree bit for bit, including boundary rows."""

    calendar = pd.bdate_range("2024-01-01", periods=15)
    dates = [day.strftime("%Y-%m-%d") for day in calendar]
    decision = pd.DatetimeIndex([calendar[7], calendar[10], calendar[14]])
    symbols = ["S0", "S1", "S2", "S3"]
    base_panel = pd.DataFrame(
        np.arange(60, dtype=float).reshape(len(calendar), len(symbols)), index=calendar, columns=symbols
    )
    holey = base_panel.copy()
    holey.loc[calendar[9], "S2"] = np.nan
    holey.loc[[calendar[8], calendar[9]], "S3"] = np.nan
    loaded = {"value": holey, "quality": base_panel}

    price = {day: {symbol: 10.0 + position for position, symbol in enumerate(symbols)} for day in dates}
    amount = {day: {symbol: 100.0 + 10.0 * position for position, symbol in enumerate(symbols)} for day in dates}

    cases = {
        # 3.1 static list.
        "base": {"base_universe": {"symbols": ["S0", "S1", "S3"]}},
        # 3.2 as-of lookup, lag shift, and the exact inclusion-window boundary.
        "membership": {
            "index_membership": {
                "membership": {
                    dates[0]: {"S0": True},
                    dates[5]: {"S0": True, "S1": True},
                    dates[9]: {"S0": True, "S1": True, "S2": True},
                },
                "lag_days": 0,
                "exclude_recent_days": 2,
            }
        },
        "membership_lagged": {
            "index_membership": {
                "membership": {
                    dates[0]: {"S0": True},
                    dates[5]: {"S0": True, "S1": True},
                    dates[9]: {"S0": True, "S1": True, "S2": True},
                },
                "lag_days": 2,
                "exclude_recent_days": 0,
            }
        },
        # 3.3 boundary: S1 lists on CALENDAR[6], so it becomes eligible on
        # exactly the 3rd trading day after listing (CALENDAR[9]) and stays
        # eligible from the second decision date on; the flip point sits
        # between the two engines' shared decision dates.
        "listing_age": {
            "trading_dates": dates,
            "listing_age": {"listing_dates": {"S1": dates[6], "S2": "1990-01-01"}, "min_days": 3},
        },
        # 3.4 exact-date statuses; sparse rows are normal.
        "status": {
            "status_exclusion": {
                "statuses": {dates[7]: {"S0": "ST"}, dates[3]: {"S1": "ST"}},
                "excluded_statuses": ["ST"],
            }
        },
        # 3.5 boundary: S1's price is exactly its limit (touching excludes),
        # S2 sits one tick below the limit.
        "price_limit": {
            "price": price,
            "price_limit": {
                "limit_up": {day: {symbol: 12.0 for symbol in symbols} for day in dates},
                "limit_down": {day: {symbol: 1.0 for symbol in symbols} for day in dates},
                "exclude_limit_up": True,
                "exclude_limit_down": True,
            },
        },
        # 3.6 boundary: the quantile leg is 0.0 (its own inclusive boundary,
        # the cross-section minimum) so the min_amount leg decides, and S1's
        # trailing mean is exactly the floor.
        "liquidity": {
            "trading_dates": dates,
            "liquidity": {"amount": amount, "window": 3, "min_amount": 110.0, "quantile": 0.0, "missing": "exclude"},
        },
        # 3.7 boundary: S2 sits exactly on max_price (inclusive).
        "price_range": {
            "trading_dates": dates,
            "price": price,
            "price_range": {"min_price": 10.0, "max_price": 12.0, "consecutive_days": 2},
        },
        # 3.8 boundary: S2's trailing coverage is exactly 5/6, S3's missing
        # run is exactly 2.
        "data_quality": {"data_quality": {"min_coverage": 5 / 6, "max_missing_run": 2, "window": 3, "non_positive_price": False}},
    }

    for tag, payload in cases.items():
        local_mask = build_eligible_mask(
            UniverseSpec.from_payload(payload), factor_panels=loaded, index=decision, columns=symbols
        )
        worker_mask = _build_universe_eligible(payload, loaded, decision, symbols)
        pd.testing.assert_frame_equal(local_mask, worker_mask, check_exact=True), tag


def test_universe_timing_mismatch_fails_closed_identically(world, tmp_path: Path) -> None:
    """A decision date off the injected calendar is a data error on both ends."""

    payload = {
        "trading_dates": _world_price_dates()[:5],
        "listing_age": {"listing_dates": {symbol: "1990-01-01" for symbol in SYMBOLS}, "min_days": 1},
    }
    candidate = _universe_candidate("parity_universe_mismatch", payload)
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_universe_mismatch.parquet")
    request_path, result_path = tmp_path / "request_universe_mismatch.json", tmp_path / "result_universe_mismatch.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError) as worker_error:
        run_request(request_path, result_path)
    assert "universe trading_dates do not contain decision date" in str(worker_error.value)

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    decision = _rebalance_dates(calendar, "monthly")
    panels = {
        name: pd.read_parquet(item["artifact_path"])
        for name, item in ((n, next(i for i in request["artifact_set"]["factors"] if i["name"] == n)) for n in FACTOR_NAMES)
    }
    with pytest.raises(ValueError) as local_error:
        build_eligible_mask(
            UniverseSpec.from_payload(payload), factor_panels=panels, index=decision, columns=list(SYMBOLS)
        )
    assert str(local_error.value) == str(worker_error.value)


@pytest.mark.parametrize(
    ("universe_payload", "message"),
    [
        ({"unknown_section": {}}, "candidate universe_rules has unknown keys: ['unknown_section']"),
        ({"base_universe": {"symbols": []}}, "universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings"),
        ({"listing_age": {"listing_dates": {"S0": "2020-01-01"}, "min_days": 0}}, "universe_rules.listing_age.min_days must be a whole number >= 1"),
        ({"liquidity": {"amount": {"2024-01-01": {"S0": 1.0}}}}, "universe_rules.liquidity requires min_amount or quantile"),
        ({"price_range": {}}, "universe_rules.price_range requires min_price or max_price"),
        ({"price_limit": {"limit_up": {"2024-01-01": {"S0": 1.0}}, "limit_down": None, "exclude_limit_up": False, "exclude_limit_down": False}}, "universe_rules.price_limit must exclude at least one of limit_up and limit_down"),
        ({"data_quality": {}}, "universe_rules.data_quality requires min_coverage, max_missing_run or non_positive_price"),
        # Malformed-typing family (verifier C1): both ends must reject with
        # the same message -- and accept integral floats identically.
        ({"index_membership": {"membership": {"2024-01-01": {"S0": True}}, "lag_days": 1.5}}, "universe_rules.index_membership.lag_days must be a whole number >= 0"),
        ({"base_universe": {"symbols": "S0"}}, "universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings"),
        ({"data_quality": {"non_positive_price": "yes"}}, "universe_rules.data_quality.non_positive_price must be a boolean"),
        ({"liquidity": {"amount": {"2024-01-01": {"S0": 1.0}}, "min_amount": 1.0, "missing": 123}}, "universe_rules.liquidity.missing must be one of ('exclude', 'include'), got 123"),
    ],
)
def test_universe_validation_fails_closed_identically(world, tmp_path: Path, universe_payload, message) -> None:
    # Build a candidate without universe rules, then smuggle the invalid
    # payload straight into the request so the *worker's* validator is what
    # raises; the local validator is exercised on the same raw payload below.
    candidate = _universe_candidate("parity_universe_bad", None)
    request = world["evaluator"].request_for(candidate)
    request["candidate"]["universe_rules"] = universe_payload
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_universe_bad.parquet")
    request_path, result_path = tmp_path / "request_universe_bad.json", tmp_path / "result_universe_bad.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        UniverseSpec.from_payload(universe_payload)
    assert str(local_error.value) == message


# ---------------------------------------------------------------------------
# Batch C2: roadmap 4.6 IndustryNeutralize / 4.7 StyleNeutralize /
# 4.12 RegimeConditionalSignal / 7.6 RiskParityWeight / 7.7
# FactorRiskBudgetWeight.
# ---------------------------------------------------------------------------

_C2_INDUSTRY_MAP = {"S0": "X", "S1": "X", "S2": "Y", "S3": "Y", "S4": "Z"}


def _c2_values_panel() -> pd.DataFrame:
    """Deterministic wide value panel: two full industries, one thin one, an
    unmapped symbol and a sparse column."""

    dates = pd.bdate_range("2024-01-02", periods=6)
    return pd.DataFrame(
        {
            "S0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "S1": [2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            "S2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "S3": [np.nan, 1.0, 2.0, 2.0, 1.0, np.nan],
            "S4": [0.5, 0.5, 0.5, 1.0, 1.5, 2.0],
            "S5": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        },
        index=dates,
    )


def test_batch_c2_signal_helpers_match_bit_for_bit_between_local_and_worker() -> None:
    values = _c2_values_panel()
    exposures = {
        "size": {"S0": 1.0, "S1": 2.0, "S2": 3.0, "S3": 4.0, "S4": 5.0, "S5": 6.0},
        "beta": {"S0": 0.5, "S1": -0.5, "S2": 1.0, "S3": 0.0, "S4": 2.0, "S5": -1.0},
    }

    industry_cases = [
        SignalSpec(industry_neutralize=True, industry_neutralize_industry_map=dict(_C2_INDUSTRY_MAP)),
        SignalSpec(
            industry_neutralize=True,
            industry_neutralize_industry_map=dict(_C2_INDUSTRY_MAP),
            industry_neutralize_min_group=2,
        ),
        SignalSpec(
            industry_neutralize=True,
            industry_neutralize_industry_map=dict(_C2_INDUSTRY_MAP),
            industry_neutralize_method="regression",
        ),
        SignalSpec(
            industry_neutralize=True,
            industry_neutralize_industry_map=dict(_C2_INDUSTRY_MAP),
            industry_neutralize_method="regression",
            industry_neutralize_exposures=exposures,
            industry_neutralize_version="sw2021",
        ),
    ]
    for spec in industry_cases:
        local_out = _local_apply_industry_neutralize(values, spec)
        worker_out = _apply_industry_neutralize(values, spec.as_payload())
        np.testing.assert_array_equal(
            local_out.to_numpy(dtype=float),
            worker_out.to_numpy(dtype=float),
            err_msg=f"industry neutralize drifted (method={spec.industry_neutralize_method}, "
            f"min_group={spec.industry_neutralize_min_group}, "
            f"exposures={spec.industry_neutralize_exposures is not None})",
        )

    style_cases = [
        SignalSpec(style_neutralize=True, style_neutralize_exposures={"size": exposures["size"]}),
        SignalSpec(
            style_neutralize=True,
            style_neutralize_exposures=exposures,
            style_neutralize_min_samples=4,
        ),
    ]
    for spec in style_cases:
        local_out = _local_apply_style_neutralize(values, spec)
        worker_out = _apply_style_neutralize(values, spec.as_payload())
        np.testing.assert_array_equal(
            local_out.to_numpy(dtype=float),
            worker_out.to_numpy(dtype=float),
            err_msg=f"style neutralize drifted (styles={sorted(spec.style_neutralize_exposures)})",
        )

    rank_dates = pd.bdate_range("2024-01-02", periods=8)
    symbols = list(SYMBOLS)
    ranks = {
        name: pd.DataFrame(
            {symbol: np.linspace(0.0, 1.0, 8) + 0.05 * position for position, symbol in enumerate(symbols)},
            index=rank_dates,
            columns=symbols,
        ).rank(axis=1, pct=True)
        for position, name in enumerate(FACTOR_NAMES)
    }
    # One sparse cell so the any-missing-kills contract is exercised.
    ranks["quality"].iloc[3, 2] = np.nan
    # States start *before* the rank dates so the lag shifts still resolve
    # for the earliest decision date (as-of semantics with a row shift).
    state_dates = pd.bdate_range(rank_dates[0] - pd.Timedelta(days=14), rank_dates[-1])
    states = {
        date.strftime("%Y-%m-%d"): ("on" if position % 2 == 0 else "off")
        for position, date in enumerate(state_dates)
    }
    regime_cases = [
        SignalSpec(regime_states=states, regime_factor_groups={"on": ["value", "quality"], "off": ["value", "momentum"]}),
        SignalSpec(regime_states=states, regime_factor_groups={"on": ["value", "quality"], "off": ["value", "momentum"]}, regime_lag=0),
        SignalSpec(regime_states=states, regime_factor_groups={"on": ["value"], "off": ["momentum"]}, regime_lag=3),
    ]
    for spec in regime_cases:
        local_out = _local_blend_regime_panels(ranks, spec)
        worker_out = _blend_regime_panels(ranks, spec.as_payload())
        np.testing.assert_array_equal(
            local_out.to_numpy(dtype=float),
            worker_out.to_numpy(dtype=float),
            err_msg=f"regime blend drifted (lag={spec.regime_lag})",
        )

    # Fail-closed parity: a date without a resolvable state, and a label the
    # group mapping does not define, abort on both sides with one message.
    late_states = {key: label for key, label in states.items() if pd.Timestamp(key) >= rank_dates[5]}
    missing_spec = SignalSpec(regime_states=late_states, regime_factor_groups={"on": ["value"], "off": ["momentum"]})
    # lag=0 so the first date resolves and the *label* branch (not the missing
    # state branch) is what fails on "off".
    unmapped_spec = SignalSpec(regime_states=states, regime_factor_groups={"on": ["value"]}, regime_lag=0)
    for spec in (missing_spec, unmapped_spec):
        with pytest.raises(ValueError) as local_error:
            _local_blend_regime_panels(ranks, spec)
        with pytest.raises(ValueError) as worker_error:
            _blend_regime_panels(ranks, spec.as_payload())
        assert str(local_error.value) == str(worker_error.value)


def test_batch_c2_weight_helpers_match_bit_for_bit_between_local_and_worker() -> None:
    dates = pd.bdate_range("2024-01-02", periods=12)
    history = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, 0.03, -0.01, 0.005, 0.02, -0.03, 0.01, 0.0, 0.025, -0.005],
            "B": [0.005, 0.01, -0.005, 0.02, 0.015, -0.01, 0.0, 0.01, 0.005, 0.02, -0.015, 0.01],
            "C": [np.nan, 0.01, 0.02, -0.01, 0.005, 0.03, -0.02, 0.01, 0.015, -0.005, 0.02, 0.01],
            "D": [0.02, 0.02, 0.02, -0.02, 0.03, 0.0, 0.01, 0.015, -0.01, 0.02, 0.005, 0.01],
        },
        index=dates,
    )
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})

    risk_cases = [
        CompositionSpec(top_n=4, weighting="risk_parity"),
        CompositionSpec(top_n=4, weighting="risk_parity", risk_parity_window=5),
        CompositionSpec(top_n=4, weighting="risk_parity", risk_parity_shrinkage_kind="none"),
        CompositionSpec(top_n=4, weighting="risk_parity", risk_parity_shrinkage=0.5),
        CompositionSpec(
            top_n=4,
            weighting="risk_parity",
            risk_parity_budgets={"A": 2.0, "B": 1.0, "C": 1.0, "D": 3.0},
        ),
        CompositionSpec(top_n=4, weighting="risk_parity", risk_parity_tolerance=1e-10),
    ]
    for spec in risk_cases:
        payload = spec.as_payload()
        local_out = _local_risk_parity_weights(spec, scores, history)
        worker_out = _risk_parity_weights(scores, history, payload)
        np.testing.assert_array_equal(
            local_out.to_numpy(dtype=float),
            worker_out.to_numpy(dtype=float),
            err_msg=f"risk parity drifted (window={spec.risk_parity_window})",
        )

    with pytest.raises(ValueError, match=re.escape("risk_parity weighting requires a returns panel")):
        _local_risk_parity_weights(risk_cases[0], scores, None)
    with pytest.raises(ValueError, match=re.escape("risk_parity weighting requires a returns panel")):
        _risk_parity_weights(scores, None, risk_cases[0].as_payload())
    fallback_local = _local_risk_parity_weights(risk_cases[0], scores, history.iloc[:1])
    fallback_worker = _risk_parity_weights(scores, history.iloc[:1], risk_cases[0].as_payload())
    np.testing.assert_array_equal(fallback_local.to_numpy(dtype=float), fallback_worker.to_numpy(dtype=float))
    assert np.allclose(fallback_local.to_numpy(dtype=float), 0.25)

    constant = history.copy()
    constant["B"] = 0.01
    single_observation = history.copy()
    single_observation.loc[single_observation.index[1:], "D"] = np.nan
    # Both degenerate inputs must produce *identical* outcomes on both sides:
    # an exact-zero variance raises (the inverse-volatility convention) and
    # the single-observation name raises deterministically; where numpy's
    # floating-point residue makes the variance tiny-but-positive instead,
    # both ends converge on the same concentrated weights.
    for degenerate in (constant, single_observation):
        def local_outcome():
            try:
                return ("ok", _local_risk_parity_weights(risk_cases[0], scores, degenerate))
            except ValueError as exc:
                return ("error", str(exc))

        def worker_outcome():
            try:
                return ("ok", _risk_parity_weights(scores, degenerate, risk_cases[0].as_payload()))
            except ValueError as exc:
                return ("error", str(exc))

        local_result, worker_result = local_outcome(), worker_outcome()
        assert local_result[0] == worker_result[0], (local_result, worker_result)
        if local_result[0] == "error":
            assert local_result[1] == worker_result[1]
        else:
            np.testing.assert_array_equal(
                local_result[1].to_numpy(dtype=float), worker_result[1].to_numpy(dtype=float)
            )
    assert single_observation["D"].notna().sum() == 1

    budgets_missing = CompositionSpec(
        top_n=4, weighting="risk_parity", risk_parity_budgets={"A": 1.0, "B": 1.0}
    )
    with pytest.raises(
        ValueError, match=re.escape("risk_parity_budgets must cover every selected name")
    ):
        _local_risk_parity_weights(budgets_missing, scores, history)
    with pytest.raises(
        ValueError, match=re.escape("risk_parity_budgets must cover every selected name")
    ):
        _risk_parity_weights(scores, history, budgets_missing.as_payload())

    frb_budgets = {"value": 1.0, "quality": 2.0, "momentum": 1.0}
    frb_exposures = {
        "value": {"A": 1.2, "B": 0.8, "C": 1.5, "D": 0.4},
        "quality": {"A": 0.3, "B": 1.1, "C": 0.9, "D": 1.4},
        "momentum": {"A": 0.9, "B": 0.2, "C": 1.1, "D": 0.7},
    }
    frb_cov = {"value": {"value": 1.0, "quality": 0.2, "momentum": -0.1}, "quality": {"value": 0.2, "quality": 0.8, "momentum": 0.1}, "momentum": {"value": -0.1, "quality": 0.1, "momentum": 1.2}}
    frb_idio = {"A": 0.04, "B": 0.05, "C": 0.03, "D": 0.06}
    frb_cases = [
        CompositionSpec(
            top_n=4,
            weighting="factor_risk_budget",
            frb_budgets=frb_budgets,
            frb_exposures=frb_exposures,
            frb_factor_covariance=frb_cov,
            frb_idiosyncratic=frb_idio,
        ),
        CompositionSpec(
            top_n=4,
            weighting="factor_risk_budget",
            frb_budgets=frb_budgets,
            frb_exposures=frb_exposures,
            frb_factor_covariance=frb_cov,
            frb_idiosyncratic=frb_idio,
            frb_ridge=0.1,
        ),
        CompositionSpec(
            top_n=4,
            weighting="factor_risk_budget",
            frb_budgets=frb_budgets,
            frb_exposures=frb_exposures,
            frb_factor_covariance=frb_cov,
            frb_idiosyncratic=frb_idio,
            frb_idio_max=0.045,
        ),
    ]
    for spec in frb_cases:
        local_out = _local_factor_risk_budget_weights(spec, scores)
        worker_out = _factor_risk_budget_weights(scores, spec.as_payload())
        np.testing.assert_array_equal(
            local_out.to_numpy(dtype=float),
            worker_out.to_numpy(dtype=float),
            err_msg=f"factor risk budget drifted (ridge={spec.frb_ridge}, cap={spec.frb_idio_max})",
        )
        assert float(local_out.sum()) == pytest.approx(1.0, abs=1e-12)

    singular_frb = CompositionSpec(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 1.0},
        frb_exposures={"value": frb_exposures["value"], "quality": frb_exposures["quality"]},
        frb_factor_covariance={"value": {"value": 1.0, "quality": 1.0}, "quality": {"value": 1.0, "quality": 1.0}},
        frb_idiosyncratic=frb_idio,
    )
    with pytest.raises(
        ValueError,
        match=re.escape("factor_risk_budget weighting requires a positive definite factor covariance"),
    ):
        _local_factor_risk_budget_weights(singular_frb, scores)
    with pytest.raises(
        ValueError,
        match=re.escape("factor_risk_budget weighting requires a positive definite factor covariance"),
    ):
        _factor_risk_budget_weights(scores, singular_frb.as_payload())

    missing_exposure = CompositionSpec(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 1.0},
        frb_exposures={"value": frb_exposures["value"], "quality": {"A": 1.0, "B": 1.0, "C": 1.0}},
        frb_factor_covariance={"value": {"value": 1.0, "quality": 0.0}, "quality": {"value": 0.0, "quality": 1.0}},
        frb_idiosyncratic=frb_idio,
    )
    with pytest.raises(
        ValueError,
        match=re.escape("factor_risk_budget weighting requires an exposure for every selected name and factor"),
    ):
        _local_factor_risk_budget_weights(missing_exposure, scores)
    with pytest.raises(
        ValueError,
        match=re.escape("factor_risk_budget weighting requires an exposure for every selected name and factor"),
    ):
        _factor_risk_budget_weights(scores, missing_exposure.as_payload())

    # The buffered row builder must reproduce both new schemes from the same
    # return history when HoldingBuffer is stacked on top.
    row_scores = pd.DataFrame(
        [[50.0, 40.0, 30.0, 20.0], [10.0, 50.0, 20.0, 30.0], [20.0, 20.0, 50.0, 10.0]],
        index=dates[:3],
        columns=["A", "B", "C", "D"],
    )
    buffered_specs = [
        CompositionSpec(top_n=3, weighting="risk_parity", holding_buffer_entry=0.6),
        CompositionSpec(
            top_n=3,
            weighting="risk_parity",
            risk_parity_window=5,
            risk_parity_budgets={"A": 2.0, "B": 1.0, "C": 1.0, "D": 1.0},
            holding_buffer_entry=0.5,
            holding_buffer_retain=0.3,
            holding_buffer_max_extra=1,
        ),
        CompositionSpec(
            top_n=3,
            weighting="factor_risk_budget",
            frb_budgets=frb_budgets,
            frb_exposures={"value": {"A": 1.2, "B": 0.8, "C": 1.5, "D": 0.4}, "quality": {"A": 0.3, "B": 1.1, "C": 0.9, "D": 1.4}, "momentum": {"A": 0.9, "B": 0.2, "C": 1.1, "D": 0.7}},
            frb_factor_covariance=frb_cov,
            frb_idiosyncratic=frb_idio,
            holding_buffer_entry=0.6,
        ),
    ]
    for spec in buffered_specs:
        payload = spec.as_payload()
        for position in range(len(row_scores)):
            for held in (frozenset(), frozenset({"A", "C"})):
                local_row = _local_build_buffered_target_row(spec, row_scores.iloc[position], held, history)
                worker_row = _build_buffered_target_row(row_scores.iloc[position], payload, held, history)
                np.testing.assert_array_equal(
                    local_row.to_numpy(dtype=float),
                    worker_row.to_numpy(dtype=float),
                    err_msg=f"buffered batch C2 row drifted (position={position}, held={sorted(held)})",
                )


def test_factor_risk_budget_negative_projection_relu_parity() -> None:
    """The declared non-negative projection (relu) actually holds: negative
    exposures are a legal input, so a name projecting negative must get a
    zero weight -- never a short leg.  Direct comparison + hand-computed
    values + the frb_idio_max interaction."""

    # Hand-computable: a single factor makes the CCD target exactly [1.0]
    # (scale-invariant), so the raw projection is the exposure column itself.
    scores = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5})
    single = CompositionSpec(
        top_n=3,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0},
        frb_exposures={"value": {"A": 2.0, "B": -1.0, "C": 1.0}},
        frb_factor_covariance={"value": {"value": 1.0}},
        frb_idiosyncratic={"A": 0.04, "B": 0.04, "C": 0.04},
    )
    payload = single.as_payload()
    local_out = _local_factor_risk_budget_weights(single, scores)
    worker_out = _factor_risk_budget_weights(scores, payload)
    np.testing.assert_array_equal(
        local_out.to_numpy(dtype=float),
        worker_out.to_numpy(dtype=float),
        err_msg="single-factor relu drifted",
    )
    # relu(B @ e) / sum == [2/3, 0, 1/3] -- bit for bit.
    np.testing.assert_array_equal(local_out.to_numpy(dtype=float), np.array([2.0 / 3.0, 0.0, 1.0 / 3.0]))
    assert bool((local_out >= 0.0).all()), "relu projection produced a negative weight"

    # Multi-factor: the verifier's reproduction shape -- one name negative on
    # every factor -- must land on exactly zero, both sides.
    scores4 = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    multi = CompositionSpec(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 1.0},
        frb_exposures={
            "value": {"A": 1.0, "B": 0.8, "C": -1.2, "D": 0.5},
            "quality": {"A": 0.4, "B": 1.1, "C": -0.6, "D": 0.9},
        },
        frb_factor_covariance={"value": {"value": 1.0, "quality": 0.1}, "quality": {"value": 0.1, "quality": 0.9}},
        frb_idiosyncratic={"A": 0.04, "B": 0.05, "C": 0.03, "D": 0.06},
    )
    local_out = _local_factor_risk_budget_weights(multi, scores4)
    worker_out = _factor_risk_budget_weights(scores4, multi.as_payload())
    np.testing.assert_array_equal(
        local_out.to_numpy(dtype=float),
        worker_out.to_numpy(dtype=float),
        err_msg="multi-factor relu drifted",
    )
    assert bool((local_out >= 0.0).all())
    assert float(local_out["C"]) == 0.0

    # frb_idio_max stacked on negative projections: capping the last two
    # positive names zeroes the whole book -> the original fail-closed guard
    # must still fire (zeroing and relu commute, the guard is not bypassed).
    capped = CompositionSpec(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 1.0},
        frb_exposures=multi.frb_exposures,
        frb_factor_covariance=multi.frb_factor_covariance,
        frb_idiosyncratic={"A": 0.20, "B": 0.20, "C": 0.03, "D": 0.20},
        frb_idio_max=0.1,
    )
    with pytest.raises(
        ValueError,
        match=re.escape("factor_risk_budget weighting requires at least one positive projected exposure"),
    ):
        _local_factor_risk_budget_weights(capped, scores4)
    with pytest.raises(
        ValueError,
        match=re.escape("factor_risk_budget weighting requires at least one positive projected exposure"),
    ):
        _factor_risk_budget_weights(scores4, capped.as_payload())

    # A cap that only removes some positives keeps the rest: zeroed names and
    # relu'd names coexist, both sides identical and all non-negative.
    partial = CompositionSpec(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 1.0},
        frb_exposures=multi.frb_exposures,
        frb_factor_covariance=multi.frb_factor_covariance,
        frb_idiosyncratic={"A": 0.20, "B": 0.05, "C": 0.03, "D": 0.06},
        frb_idio_max=0.1,
    )
    local_out = _local_factor_risk_budget_weights(partial, scores4)
    worker_out = _factor_risk_budget_weights(scores4, partial.as_payload())
    np.testing.assert_array_equal(
        local_out.to_numpy(dtype=float),
        worker_out.to_numpy(dtype=float),
        err_msg="partial idio cap over relu drifted",
    )
    assert bool((local_out >= 0.0).all())
    assert float(local_out["A"]) == 0.0 and float(local_out["C"]) == 0.0
    assert float(local_out.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.slow
def test_industry_neutralization_agrees_end_to_end(world) -> None:
    """IndustryNeutralize (4.6) end-to-end parity, proportional weighting per
    the discrimination rule for monotone-insensitive schemes."""

    tmp_path: Path = world["tmp_path"]
    industry_map = {"S0": "X", "S1": "X", "S2": "Y", "S3": "Y", "S4": "Z", "S5": "Z"}
    # min_group=2 so every industry actually neutralizes (default 3 would
    # pass every pair through and the case would not discriminate).
    signal = SignalSpec(
        industry_neutralize=True,
        industry_neutralize_industry_map=industry_map,
        industry_neutralize_min_group=2,
    )
    composition = CompositionSpec(top_n=4, weighting="proportional")
    candidate = StrategyCandidate(
        name="parity_industry_neutralize",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        signal=signal,
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_industry_neutralize.parquet")
    request_path, result_path = tmp_path / "request_industry_neutralize.json", tmp_path / "result_industry_neutralize.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    baseline = _local_portfolio(
        world,
        request,
        composition,
        SignalSpec(),
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_style_neutralization_agrees_end_to_end(world) -> None:
    """StyleNeutralize (4.7) end-to-end parity with two injected styles."""

    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)
    signal = SignalSpec(
        style_neutralize=True,
        style_neutralize_exposures={
            "size": {symbol: 0.5 + 0.2 * position for position, symbol in enumerate(symbols)},
            "beta": {symbol: (-1.0) ** position * (0.5 + 0.1 * position) for position, symbol in enumerate(symbols)},
        },
        style_neutralize_min_samples=4,
    )
    composition = CompositionSpec(top_n=4, weighting="proportional")
    candidate = StrategyCandidate(
        name="parity_style_neutralize",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        signal=signal,
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_style_neutralize.parquet")
    request_path, result_path = tmp_path / "request_style_neutralize.json", tmp_path / "result_style_neutralize.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    baseline = _local_portfolio(world, request, composition, SignalSpec(), backtest=candidate.backtest)
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_regime_conditional_signal_agrees_end_to_end(world) -> None:
    """RegimeConditionalSignal (4.12) end-to-end parity: the injected state
    rotates the factor mix mid-window and the holdings follow."""

    tmp_path: Path = world["tmp_path"]
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, "monthly")
    # States cover the whole price calendar; the switch lands between the
    # monthly decision dates so the group rotation is observable.
    price_dates = pd.bdate_range(PRICE_DATES[0], PRICE_DATES[-1])
    switch = price_dates[len(price_dates) // 2]
    states = {
        date.strftime("%Y-%m-%d"): ("off" if date >= switch else "on")
        for date in price_dates
    }
    signal = SignalSpec(
        regime_states=states,
        regime_factor_groups={"on": ["value", "quality"], "off": ["value", "momentum"]},
        regime_lag=1,
    )
    composition = CompositionSpec(top_n=4, weighting="equal")
    candidate = StrategyCandidate(
        name="parity_regime_signal",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        signal=signal,
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_regime.parquet")
    request_path, result_path = tmp_path / "request_regime.json", tmp_path / "result_regime.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    baseline = _local_portfolio(world, request, composition, SignalSpec(), backtest=candidate.backtest)
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))

    # The audit block reports the state-conditional combination truthfully.
    assert payload["audit"]["score_combination"] == (
        "state_conditional_equal_weight_blend_of_cross_sectional_percentile_rank"
    )


@pytest.mark.slow
def test_regime_missing_state_fails_closed_end_to_end(world) -> None:
    """A decision date without a resolvable state aborts on both sides with
    the same message (execution-time fail-closed)."""

    tmp_path: Path = world["tmp_path"]
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, "monthly")
    # With lag=0 the only state sits on the *second* decision date: the first
    # decision date has no state at or before it and must fail closed.
    states = {rebalance_dates[1].strftime("%Y-%m-%d"): "on"}
    signal = SignalSpec(
        regime_states=states, regime_factor_groups={"on": list(FACTOR_NAMES)}, regime_lag=0
    )
    candidate = StrategyCandidate(
        name="parity_regime_missing",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        signal=signal,
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_regime_missing.parquet")
    request_path, result_path = tmp_path / "request_regime_missing.json", tmp_path / "result_regime_missing.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("regime state is missing for decision date")):
        run_request(request_path, result_path)

    panels = {
        str(item["name"]): pd.read_parquet(item["artifact_path"])
        for item in request["artifact_set"]["factors"]
    }
    directions = {name: int(request["factor_directions"][name]) for name in FACTOR_NAMES}
    with pytest.raises(ValueError, match=re.escape("regime state is missing for decision date")):
        build_factor_score(panels, directions, signal, index=rebalance_dates, columns=list(SYMBOLS))


@pytest.mark.slow
def test_risk_parity_weighting_agrees_end_to_end(world) -> None:
    """RiskParityWeight (7.6) end-to-end parity."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(top_n=4, weighting="risk_parity")
    candidate = StrategyCandidate(
        name="parity_risk_parity",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_risk_parity.parquet")
    request_path, result_path = tmp_path / "request_risk_parity.json", tmp_path / "result_risk_parity.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Ablation path (fact 29 convention): the worker's ablation loop passes
    # the *same* composition payload and returns panel as the main path, so
    # the per-factor marginal metrics must reproduce from the local engine
    # running the risk-parity scheme on the ungated equal blend.  Were the
    # ablation loop to drop the scheme, these marginals would drift.
    local_ablation = _local_ablation_metrics(world, request, candidate)
    for removed, contribution in payload["factor_contributions"].items():
        assert contribution["marginal_annualized_return"] == pytest.approx(
            float(local.metrics["annualized_return"]) - float(local_ablation[removed]["annualized_return"]),
            rel=1e-9,
            abs=1e-12,
        ), removed
        assert contribution["marginal_sharpe"] == pytest.approx(
            float(local.metrics["sharpe"]) - float(local_ablation[removed]["sharpe"]),
            rel=1e-9,
            abs=1e-12,
        ), removed

    # Discrimination: after the first (fallback) decision date the risk-parity
    # tilt moves the book away from the equal weighting.
    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=4, weighting="equal"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_factor_risk_budget_weighting_agrees_end_to_end(world) -> None:
    """FactorRiskBudgetWeight (7.7) end-to-end parity with injected exposures."""

    tmp_path: Path = world["tmp_path"]
    composition = CompositionSpec(
        top_n=4,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 2.0, "momentum": 1.0},
        frb_exposures={
            # One negative exposure exercises the declared relu projection
            # end to end (negative loads are a legal input).
            "value": {"S0": 1.2, "S1": 0.8, "S2": 1.5, "S3": -0.4, "S4": 0.9, "S5": 1.1},
            "quality": {"S0": 0.3, "S1": 1.1, "S2": 0.9, "S3": 1.4, "S4": 0.6, "S5": 0.8},
            "momentum": {"S0": 0.9, "S1": 0.2, "S2": 1.1, "S3": 0.7, "S4": 1.3, "S5": 0.5},
        },
        frb_factor_covariance={
            "value": {"value": 1.0, "quality": 0.2, "momentum": -0.1},
            "quality": {"value": 0.2, "quality": 0.8, "momentum": 0.1},
            "momentum": {"value": -0.1, "quality": 0.1, "momentum": 1.2},
        },
        frb_idiosyncratic={symbol: 0.04 + 0.01 * position for position, symbol in enumerate(SYMBOLS)},
    )
    candidate = StrategyCandidate(
        name="parity_factor_risk_budget",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_frb.parquet")
    request_path, result_path = tmp_path / "request_frb.json", tmp_path / "result_frb.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=4, weighting="equal"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.parametrize(
    ("payload_override", "message"),
    [
        ({"weighting": "equal", "risk_parity_shrinkage": 0.5}, "risk_parity parameters require the risk_parity weighting"),
        ({"weighting": "equal", "risk_parity_window": 30}, "risk_parity parameters require the risk_parity weighting"),
        ({"weighting": "equal", "risk_parity_budgets": {"S0": 1.0}}, "risk_parity parameters require the risk_parity weighting"),
        ({"weighting": "risk_parity", "risk_parity_window": 1}, "risk_parity_window must be at least 2"),
        ({"weighting": "risk_parity", "risk_parity_window": 2.5}, "risk_parity_window must be a whole number"),
        ({"weighting": "risk_parity", "risk_parity_budgets": 5}, "risk_parity_budgets must map non-empty symbols to finite positive numbers"),
        ({"weighting": "risk_parity", "risk_parity_budgets": {"S0": -1.0}}, "risk_parity_budgets must map non-empty symbols to finite positive numbers"),
        ({"weighting": "risk_parity", "risk_parity_budgets": {"S0": 0}}, "risk_parity_budgets must map non-empty symbols to finite positive numbers"),
        ({"weighting": "risk_parity", "risk_parity_shrinkage_kind": "ledoit"}, "risk_parity_shrinkage_kind must be one of ('none', 'diagonal'), got 'ledoit'"),
        ({"weighting": "risk_parity", "risk_parity_shrinkage": 1.0}, "risk_parity_shrinkage must be in [0, 1) when provided"),
        ({"weighting": "risk_parity", "risk_parity_shrinkage": "high"}, "risk_parity_shrinkage must be in [0, 1) when provided"),
        ({"weighting": "risk_parity", "risk_parity_tolerance": 0.0}, "risk_parity_tolerance must be a finite positive number"),
        ({"weighting": "equal", "frb_budgets": {"value": 1.0}}, "factor_risk_budget parameters require the factor_risk_budget weighting"),
        ({"weighting": "equal", "frb_idio_max": 0.1}, "factor_risk_budget parameters require the factor_risk_budget weighting"),
        ({"weighting": "factor_risk_budget"}, "factor_risk_budget weighting requires frb_budgets"),
        ({"weighting": "factor_risk_budget", "frb_budgets": {"value": 1.0, "quality": 1.0, "momentum": 1.0}}, "factor_risk_budget weighting requires frb_exposures"),
        (
            {
                "weighting": "factor_risk_budget",
                "frb_budgets": {"value": 1.0, "quality": 1.0, "momentum": 1.0},
                "frb_exposures": {"value": {"S0": 1.0}, "quality": {"S0": 1.0}, "momentum": {"S0": 1.0}},
                "frb_factor_covariance": {"value": {"value": 1.0, "quality": 0.0, "momentum": 0.0}, "quality": {"value": 0.0, "quality": 1.0, "momentum": 0.0}, "momentum": {"value": 0.0, "quality": 0.0, "momentum": 1.0}},
                "frb_idiosyncratic": {},
            },
            "frb_idiosyncratic must map non-empty symbols to finite non-negative numbers",
        ),
        (
            {
                "weighting": "factor_risk_budget",
                "frb_budgets": {"value": 1.0, "quality": 1.0, "momentum": 1.0},
                "frb_exposures": {"value": {"S0": 1.0}, "quality": {"S0": 1.0}, "momentum": {"S0": 1.0}},
                "frb_factor_covariance": {"value": {"value": 1.0, "quality": 0.0, "momentum": 0.0}, "quality": {"value": 0.0, "quality": 1.0, "momentum": 0.0}, "momentum": {"value": 0.0, "quality": 0.0, "momentum": 1.0}},
                "frb_idiosyncratic": {"S0": 0.04, "S1": 0.04, "S2": 0.04, "S3": 0.04, "S4": 0.04, "S5": 0.04},
                "frb_ridge": -0.1,
            },
            "frb_ridge must be a finite non-negative number",
        ),
    ],
)
def test_batch_c2_composition_failures_fail_closed_on_both_sides(world, payload_override: dict, message: str) -> None:
    """Invalid batch C2 composition parameters raise on both sides."""

    candidate = StrategyCandidate(
        name="parity_batch_c2_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch_c2_fail.parquet")
    request["candidate"]["composition"].update(payload_override)
    request_path, result_path = tmp_path / "request_batch_c2_fail.json", tmp_path / "result_batch_c2_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        CompositionSpec(top_n=4, **payload_override)
    assert str(local_error.value) == message


@pytest.mark.parametrize(
    ("signal_override", "message"),
    [
        ({"industry_neutralize": True}, "industry neutralize requires industry_neutralize_industry_map"),
        ({"industry_neutralize_industry_map": {"S0": "X"}}, "industry neutralize parameters require industry_neutralize"),
        ({"industry_neutralize": True, "industry_neutralize_industry_map": {"S0": ""}}, "industry_neutralize_industry_map must map non-empty symbols to non-empty industry labels"),
        ({"industry_neutralize": True, "industry_neutralize_industry_map": {"S0": "X"}, "industry_neutralize_min_group": 1}, "industry_neutralize_min_group must be at least 2"),
        ({"industry_neutralize": True, "industry_neutralize_industry_map": {"S0": "X"}, "industry_neutralize_min_group": 2.5}, "industry_neutralize_min_group must be a whole number"),
        ({"industry_neutralize": True, "industry_neutralize_industry_map": {"S0": "X"}, "industry_neutralize_method": "robust"}, "industry_neutralize_method must be one of ('demean', 'regression'), got 'robust'"),
        ({"industry_neutralize": True, "industry_neutralize_industry_map": {"S0": "X"}, "industry_neutralize_exposures": {"size": {"S0": 1.0}}}, "industry_neutralize_exposures is only valid with the regression method"),
        ({"style_neutralize": True}, "style neutralize requires style_neutralize_exposures"),
        ({"style_neutralize_exposures": {"size": {"S0": 1.0}}}, "style neutralize parameters require style_neutralize"),
        ({"style_neutralize": True, "style_neutralize_exposures": {"size": {"S0": 1.0}}, "style_neutralize_min_samples": 1}, "style_neutralize_min_samples must be at least 2"),
        ({"regime_states": {"2024-01-02": "on"}}, "regime_states and regime_factor_groups must be provided together"),
        ({"regime_factor_groups": {"on": ["value"]}}, "regime_states and regime_factor_groups must be provided together"),
        ({"regime_states": {"2024-01-02": "on"}, "regime_factor_groups": {"on": ["value"]}, "blend_weights": {"value": 1.0, "quality": 1.0, "momentum": 1.0}}, "regime_factor_groups and blend_weights are mutually exclusive"),
        ({"regime_states": {"2024-01-02": "on"}, "regime_factor_groups": {"on": ["value"]}, "interaction_pairs": [["value", "quality", "product"]]}, "regime_factor_groups and interaction_pairs are mutually exclusive"),
        ({"regime_states": {"not-a-date": "on"}, "regime_factor_groups": {"on": ["value"]}}, "regime_states contains an invalid date: 'not-a-date'"),
        ({"regime_states": {"2024-01-02": "on"}, "regime_factor_groups": {"on": []}}, "regime state 'on' must name at least one factor"),
        ({"regime_states": {"2024-01-02": "on"}, "regime_factor_groups": {"on": ["value"]}, "regime_lag": -1}, "regime_lag must not be negative"),
    ],
)
def test_batch_c2_signal_failures_fail_closed_on_both_sides(world, signal_override: dict, message: str) -> None:
    """Invalid batch C2 signal parameters raise on both sides with one message."""

    candidate = StrategyCandidate(
        name="parity_batch_c2_signal_fail",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch_c2_signal_fail.parquet")
    request["candidate"]["signal"].update(signal_override)
    request_path, result_path = tmp_path / "request_batch_c2_signal_fail.json", tmp_path / "result_batch_c2_signal_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        SignalSpec.from_payload(signal_override)
    assert str(local_error.value) == message


# ---------------------------------------------------------------------------
# Batch C3: roadmap 6.4 IndustryDeviationConstraint / 6.5
# StyleExposureConstraint / 6.6 BetaConstraint / 6.7 FactorExposureConstraint
# / 6.9 ConstrainedWeightProjection.
# ---------------------------------------------------------------------------

_C3_SYMBOLS = ["A", "B", "C", "D"]
_C3_INDUSTRY_MAP = {"A": "X", "B": "X", "C": "Y", "D": "Y"}


def _c3_params(**overrides) -> dict:
    params = dict(
        industry_deviation_max={"X": 0.1, "Y": 0.15},
        industry_deviation_industry_map=dict(_C3_INDUSTRY_MAP),
        industry_deviation_exempt=(),
        industry_deviation_benchmark={"X": 0.5, "Y": 0.5},
        projection_max_iter=512,
        projection_tolerance=1e-9,
    )
    params.update(overrides)
    return {key: value for key, value in params.items() if value is not None or key in ("industry_deviation_exempt",)}


@pytest.mark.slow
def test_batch_c3_boxed_projection_matches_between_local_and_worker() -> None:
    """The box constraints of the chain stages must reach the worker projector.

    Both sides build their params mapping through a different route -- the
    local side from the spec, the worker side from the canonical payload -- so
    a key that only one route carries changes the constraint set and therefore
    the projected point.  The cap and the held-name floor are exactly such
    keys: the chain applies them one stage earlier and the projector has to
    respect them, otherwise a required industry move pushes a name back over
    the cap (or under the floor).
    """

    for boxed, values, expected in (
        (
            CompositionSpec(
                top_n=4,
                weighting="equal",
                weight_cap=0.25,
                industry_deviation_max={"X": 0.1, "Y": 0.1},
                industry_deviation_industry_map=dict(_C3_INDUSTRY_MAP),
                industry_deviation_benchmark={"X": 0.5, "Y": 0.5},
            ),
            np.array([0.4, 0.35, 0.15, 0.1]),
            np.array([0.25, 0.25, 0.225, 0.175]),
        ),
        (
            CompositionSpec(
                top_n=4,
                weighting="equal",
                weight_cap=0.4,
                weight_floor=0.2,
                industry_deviation_max={"X": 0.1, "Y": 0.1},
                industry_deviation_industry_map=dict(_C3_INDUSTRY_MAP),
                industry_deviation_benchmark={"X": 0.5, "Y": 0.5},
            ),
            np.array([0.7, 0.2, 0.05, 0.05]),
            np.array([0.4, 0.2, 0.2, 0.2]),
        ),
    ):
        assert _local_projection_params(boxed) == _projection_params_from_payload(boxed.as_payload())
        local_out = _local_apply_constrained_projection(values, _C3_SYMBOLS, _local_projection_params(boxed))
        worker_out = _apply_constrained_projection(values, _C3_SYMBOLS, _projection_params_from_payload(boxed.as_payload()))
        np.testing.assert_array_equal(local_out, worker_out, err_msg=f"boxed projection drifted (cap={boxed.weight_cap})")
        np.testing.assert_allclose(local_out, expected, rtol=0.0, atol=1e-12)
        assert float(local_out.max()) <= boxed.weight_cap + 1e-12
        if boxed.weight_floor is not None:
            assert float(local_out[values > 0.0].min()) >= boxed.weight_floor - 1e-12


def test_batch_c3_projection_helpers_match_bit_for_bit_between_local_and_worker() -> None:
    # The params mapping built from the spec and from the canonical payload
    # must be the same object graph (the worker never sees the spec).
    spec = CompositionSpec(
        top_n=4,
        weighting="equal",
        industry_deviation_max={"X": 0.1, "Y": 0.15},
        industry_deviation_industry_map=dict(_C3_INDUSTRY_MAP),
        industry_deviation_benchmark={"X": 0.5, "Y": 0.5},
    )
    assert _local_projection_params(spec) == _projection_params_from_payload(spec.as_payload())

    # The pre-projection chain stages enter the projector as box constraints,
    # so the cap and the held-name floor must travel through the payload
    # mapping as well: dropping them silently widens the worker's constraint
    # set and the projection can push a name back over the cap (or under the
    # floor) that the chain enforced one stage earlier.
    for extra in ({"weight_cap": 0.25}, {"weight_floor": 0.12}, {"weight_cap": 0.3, "weight_floor": 0.12}):
        boxed = CompositionSpec(
            top_n=4,
            weighting="equal",
            industry_deviation_max={"X": 0.1, "Y": 0.15},
            industry_deviation_industry_map=dict(_C3_INDUSTRY_MAP),
            industry_deviation_benchmark={"X": 0.5, "Y": 0.5},
            **extra,
        )
        assert _local_projection_params(boxed) == _projection_params_from_payload(boxed.as_payload()), sorted(extra)

    violating = np.array([0.4, 0.35, 0.15, 0.1])  # X=0.75 vs bench 0.5
    feasible = np.array([0.3, 0.2, 0.3, 0.2])  # X=0.5, Y=0.5: already feasible

    cases = [
        # Industry deviation: an over-deviated book is pulled back, an
        # exactly-feasible book passes through bit for bit (the boundary is
        # inclusive).
        (_c3_params(), violating),
        (_c3_params(), feasible),
        # Exempting X leaves only the Y constraint active.
        (_c3_params(industry_deviation_exempt=["X"]), violating),
        # Missing benchmark industries count as benchmark weight 0.
        (_c3_params(industry_deviation_benchmark={"X": 0.3}), violating),
        # Style exposure interval with an exactly-binding upper bound.
        (
            _c3_params(
                industry_deviation_max=None,
                industry_deviation_industry_map=None,
                industry_deviation_benchmark=None,
                style_exposure_bounds={"size": [-0.05, 0.05], "beta": [-0.2, 0.2]},
                style_exposure_matrix={
                    "size": {"A": 1.0, "B": 0.5, "C": -0.5, "D": -1.0},
                    "beta": {"A": 0.3, "B": -0.2, "C": 0.1, "D": 0.4},
                },
                style_exposure_benchmark={"size": 0.1, "beta": 0.0},
            ),
            np.array([0.4, 0.3, 0.2, 0.1]),
        ),
        # Beta: two-sided absolute interval.
        (
            _c3_params(
                industry_deviation_max=None,
                industry_deviation_industry_map=None,
                industry_deviation_benchmark=None,
                beta_constraint_min=0.8,
                beta_constraint_max=1.0,
                beta_by_symbol={"A": 1.4, "B": 1.1, "C": 0.7, "D": 0.4},
            ),
            np.array([0.4, 0.3, 0.2, 0.1]),  # beta = 1.05 > 1.0
        ),
        # Factor exposure: absolute interval over the injected matrix.
        (
            _c3_params(
                industry_deviation_max=None,
                industry_deviation_industry_map=None,
                industry_deviation_benchmark=None,
                factor_exposure_bounds={"value": [0.0, 0.3]},
                factor_exposure_matrix={"value": {"A": 1.2, "B": 0.9, "C": 0.4, "D": -0.3}},
            ),
            np.array([0.5, 0.25, 0.15, 0.1]),  # value exposure = 0.785
        ),
        # The single-name cap participates as a box constraint: the
        # projection can never push a name back over the cap.
        (_c3_params(weight_cap=0.25), violating),
        # The held-name floor participates as well.
        (_c3_params(weight_floor=0.12), np.array([0.5, 0.2, 0.2, 0.1])),
    ]
    for params, values in cases:
        local_out = _local_apply_constrained_projection(values, _C3_SYMBOLS, params)
        worker_out = _apply_constrained_projection(values, _C3_SYMBOLS, params)
        np.testing.assert_array_equal(
            local_out,
            worker_out,
            err_msg=f"projection drifted (keys={sorted(params)})",
        )

    # Boundary semantics: an exactly-feasible row is returned unchanged.
    np.testing.assert_array_equal(
        _local_apply_constrained_projection(feasible, _C3_SYMBOLS, _c3_params()),
        feasible,
    )
    # Post-conditions on the violating row: every industry deviation bound
    # holds (inclusive boundary) and the budget/non-negativity survive.
    projected = _local_apply_constrained_projection(violating, _C3_SYMBOLS, _c3_params())
    industry_active = {
        industry: float(sum(weight for weight, symbol in zip(projected, _C3_SYMBOLS) if _C3_INDUSTRY_MAP[symbol] == industry))
        - float(_c3_params()["industry_deviation_benchmark"][industry])
        for industry in ("X", "Y")
    }
    for industry, active in industry_active.items():
        assert abs(active) <= _c3_params()["industry_deviation_max"][industry] + 1e-9, industry
    assert float(projected.sum()) <= 1.0 + 1e-9
    assert bool((projected >= -1e-12).all())
    assert not np.allclose(projected, violating)

    # Fail-closed parity: an infeasible constraint set and a missing exposure
    # abort on both sides with one message.
    infeasible = _c3_params(
        industry_deviation_max=None,
        industry_deviation_industry_map=None,
        industry_deviation_benchmark=None,
        beta_constraint_min=2.5,
        beta_constraint_max=3.0,
        beta_by_symbol={"A": 1.4, "B": 1.1, "C": 0.7, "D": 0.4},
    )
    missing_style = _c3_params(
        industry_deviation_max=None,
        industry_deviation_industry_map=None,
        industry_deviation_benchmark=None,
        style_exposure_bounds={"size": [-0.05, 0.05]},
        style_exposure_matrix={"size": {"A": 1.0, "B": 0.5, "C": -0.5}},
        style_exposure_benchmark={"size": 0.1},
    )

    def local_outcome(values, params):
        try:
            return ("ok", _local_apply_constrained_projection(values, _C3_SYMBOLS, params))
        except ValueError as exc:
            return ("error", str(exc))

    def worker_outcome(values, params):
        try:
            return ("ok", _apply_constrained_projection(values, _C3_SYMBOLS, params))
        except ValueError as exc:
            return ("error", str(exc))

    for params in (infeasible, missing_style):
        local_result, worker_result = local_outcome(violating, params), worker_outcome(violating, params)
        assert local_result[0] == worker_result[0] == "error", (local_result, worker_result)
        assert local_result[1] == worker_result[1]
        assert local_result[1].startswith("constrained weight projection") or "requires an exposure" in local_result[1]


@pytest.mark.slow
def test_industry_deviation_constraint_agrees_end_to_end(world) -> None:
    tmp_path: Path = world["tmp_path"]
    industry_map = {"S0": "X", "S1": "X", "S2": "X", "S3": "Y", "S4": "Y", "S5": "Y"}
    composition = CompositionSpec(
        top_n=6,
        weighting="proportional",
        industry_deviation_max={"X": 0.05, "Y": 0.05},
        industry_deviation_industry_map=industry_map,
        industry_deviation_benchmark={"X": 0.5, "Y": 0.5},
    )
    candidate = StrategyCandidate(
        name="parity_industry_deviation",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_industry_deviation.parquet")
    request_path, result_path = tmp_path / "request_industry_deviation.json", tmp_path / "result_industry_deviation.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Post-condition: the projection output lands on the next trading day
    # after each rebalance (the earliest start-of-day image of the *target*
    # weights; later rows drift with prices and are deliberately not
    # re-constrained -- fact 8 semantics, same as cap/floor).
    weights = local.weights
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, "monthly")
    next_rows = weights.index.intersection(calendar.intersection(pd.DatetimeIndex(rebalance_dates) + pd.Timedelta(days=1)))
    invested = weights.loc[(weights.sum(axis=1) > 0) & weights.index.isin(next_rows)]
    for industry in ("X", "Y"):
        columns = [symbol for symbol in weights.columns if industry_map.get(symbol) == industry]
        active = invested[columns].sum(axis=1) - 0.5
        assert bool((active.abs() <= 0.05 + 2e-3).all()), industry

    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=6, weighting="proportional"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_cap_and_floor_boxes_survive_the_projection_agrees_end_to_end(world) -> None:
    """The chain stages before the projection must keep holding after it.

    ``rank`` weighting at ``top_n=6`` puts the best name at 6/21 above the cap,
    so the single-name cap binds on every rebalance date by construction, and
    the industry benchmark demands an X weight the pre-projection book cannot
    reach without moving money.  The cap and the held-name floor enter the
    projector as box constraints on both ends, so a target row produced by the
    projection may never sit above the cap or below the floor -- a mapping that
    loses those two keys widens the worker's constraint set and returns exactly
    such a row on the same date.
    """

    tmp_path: Path = world["tmp_path"]
    industry_map = {"S0": "X", "S1": "X", "S2": "X", "S3": "Y", "S4": "Y", "S5": "Y"}
    composition = CompositionSpec(
        top_n=6,
        weighting="rank",
        weight_cap=0.25,
        weight_floor=0.05,
        industry_deviation_max={"X": 0.05, "Y": 0.05},
        industry_deviation_industry_map=industry_map,
        industry_deviation_benchmark={"X": 0.7, "Y": 0.3},
    )
    candidate = StrategyCandidate(
        name="parity_projection_box",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_projection_box.parquet")
    request_path, result_path = tmp_path / "request_projection_box.json", tmp_path / "result_projection_box.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
    score = _score_panel_for(world, request, candidate.signal, rebalance_dates, sorted(set(close.columns)))
    targets = build_target_weights(composition, score, returns=_returns_panel_for(world))
    for date, row in targets.iterrows():
        held = row[row > 0]
        assert held.max() <= composition.weight_cap + 1e-9, date
        assert held.min() >= composition.weight_floor - 1e-9, date
        for industry in ("X", "Y"):
            columns = [symbol for symbol in targets.columns if industry_map.get(symbol) == industry]
            active = float(held.reindex(columns).fillna(0.0).sum()) - composition.industry_deviation_benchmark[industry]
            assert abs(active) <= composition.industry_deviation_max[industry] + 2e-3, date


@pytest.mark.slow
def test_style_exposure_constraint_agrees_end_to_end(world) -> None:
    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)
    composition = CompositionSpec(
        top_n=6,
        weighting="proportional",
        style_exposure_bounds={"size": [-0.08, 0.08]},
        style_exposure_matrix={
            "size": {symbol: 1.0 + 0.4 * position for position, symbol in enumerate(symbols)},
        },
        style_exposure_benchmark={"size": 1.7},
    )
    candidate = StrategyCandidate(
        name="parity_style_exposure",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_style_exposure.parquet")
    request_path, result_path = tmp_path / "request_style_exposure.json", tmp_path / "result_style_exposure.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    weights = local.weights
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, "monthly")
    next_rows = weights.index.intersection(calendar.intersection(pd.DatetimeIndex(rebalance_dates) + pd.Timedelta(days=1)))
    invested = weights.loc[(weights.sum(axis=1) > 0) & weights.index.isin(next_rows)]
    exposures = invested[[f"S{position}" for position in range(6)]].apply(
        lambda row: sum(float(row[symbol]) * (1.0 + 0.4 * position) for position, symbol in enumerate(symbols)) - 1.7,
        axis=1,
    )
    assert bool((exposures.abs() <= 0.08 + 2e-3).all())

    baseline = _local_portfolio(
        world,
        request,
        CompositionSpec(top_n=6, weighting="proportional"),
        candidate.signal,
        backtest=candidate.backtest,
    )
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline.weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_beta_constraint_agrees_end_to_end(world) -> None:
    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)

    # Stage 1: measure the unconstrained book's beta on the post-rebalance
    # rows, so the interval below provably cuts it (a static interval may
    # already be satisfied and the case would not discriminate).
    baseline_request = world["evaluator"].request_for(
        StrategyCandidate(
            name="parity_beta_baseline",
            universe="csi_all",
            composition=CompositionSpec(top_n=6, weighting="proportional"),
            factor_names=FACTOR_NAMES,
            rule="parity fixture",
            backtest=BacktestSpec(signal_lag=1),
        )
    )
    baseline_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_beta_baseline.parquet")
    baseline = _local_portfolio(
        world,
        baseline_request,
        CompositionSpec(top_n=6, weighting="proportional"),
        SignalSpec(),
        backtest=BacktestSpec(signal_lag=1),
    )
    baseline_weights = baseline.weights
    baseline_betas = baseline_weights[symbols].apply(
        lambda row: sum(float(row[symbol]) * (0.6 + 0.15 * position) for position, symbol in enumerate(symbols)),
        axis=1,
    )
    beta_ceiling = float(baseline_betas.max())
    assert beta_ceiling > 0.5

    composition = CompositionSpec(
        top_n=6,
        weighting="proportional",
        beta_constraint_min=max(beta_ceiling - 0.20, 0.1),
        beta_constraint_max=beta_ceiling - 0.05,
        beta_by_symbol={symbol: 0.6 + 0.15 * position for position, symbol in enumerate(symbols)},
    )
    candidate = StrategyCandidate(
        name="parity_beta_constraint",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_beta.parquet")
    request_path, result_path = tmp_path / "request_beta.json", tmp_path / "result_beta.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Post-condition: the projection output lands on the next trading day
    # after each rebalance (the earliest start-of-day image of the *target*
    # weights; later rows drift with prices and are deliberately not
    # re-constrained -- fact 8 semantics, same as cap/floor).
    weights = local.weights
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, "monthly")
    next_rows = weights.index.intersection(calendar.intersection(pd.DatetimeIndex(rebalance_dates) + pd.Timedelta(days=1)))
    invested = weights.loc[(weights.sum(axis=1) > 0) & weights.index.isin(next_rows)]
    betas = invested[symbols].apply(
        lambda row: sum(float(row[symbol]) * (0.6 + 0.15 * position) for position, symbol in enumerate(symbols)),
        axis=1,
    )
    assert bool(((betas >= composition.beta_constraint_min - 2e-3) & (betas <= composition.beta_constraint_max + 2e-3)).all())
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline_weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_factor_exposure_constraint_agrees_end_to_end(world) -> None:
    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)

    # Stage 1: measure the unconstrained book's factor exposures so the
    # intervals below provably cut them (a static interval may already be
    # satisfied and the case would not discriminate).
    baseline_request = world["evaluator"].request_for(
        StrategyCandidate(
            name="parity_factor_exposure_baseline",
            universe="csi_all",
            composition=CompositionSpec(top_n=6, weighting="proportional"),
            factor_names=FACTOR_NAMES,
            rule="parity fixture",
            backtest=BacktestSpec(signal_lag=1),
        )
    )
    baseline_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_factor_exposure_baseline.parquet")
    baseline = _local_portfolio(
        world,
        baseline_request,
        CompositionSpec(top_n=6, weighting="proportional"),
        SignalSpec(),
        backtest=BacktestSpec(signal_lag=1),
    )
    baseline_weights = baseline.weights
    baseline_value = baseline_weights[symbols].apply(
        lambda row: sum(float(row[symbol]) * (1.0 - 0.25 * position) for position, symbol in enumerate(symbols)),
        axis=1,
    )
    baseline_quality = baseline_weights[symbols].apply(
        lambda row: sum(float(row[symbol]) * ((-1.0) ** position) * 0.8 for position, symbol in enumerate(symbols)),
        axis=1,
    )
    value_ceiling = float(baseline_value.max())
    quality_floor = float(baseline_quality.min())
    assert value_ceiling > 0.05 and quality_floor < 0.2

    composition = CompositionSpec(
        top_n=6,
        weighting="proportional",
        factor_exposure_bounds={
            "value": [max(value_ceiling - 0.18, 0.0), value_ceiling - 0.04],
            "quality": [quality_floor + 0.04, quality_floor + 0.15],
        },
        factor_exposure_matrix={
            "value": {symbol: 1.0 - 0.25 * position for position, symbol in enumerate(symbols)},
            "quality": {symbol: (-1.0) ** position * 0.8 for position, symbol in enumerate(symbols)},
        },
    )
    candidate = StrategyCandidate(
        name="parity_factor_exposure",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_factor_exposure.parquet")
    request_path, result_path = tmp_path / "request_factor_exposure.json", tmp_path / "result_factor_exposure.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))

    # Post-condition: the projection output lands on the next trading day
    # after each rebalance (the earliest start-of-day image of the *target*
    # weights; later rows drift with prices and are deliberately not
    # re-constrained -- fact 8 semantics, same as cap/floor).
    weights = local.weights
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    calendar = close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index
    rebalance_dates = _rebalance_dates(calendar, "monthly")
    next_rows = weights.index.intersection(calendar.intersection(pd.DatetimeIndex(rebalance_dates) + pd.Timedelta(days=1)))
    invested = weights.loc[(weights.sum(axis=1) > 0) & weights.index.isin(next_rows)]
    value = invested[symbols].apply(
        lambda row: sum(float(row[symbol]) * (1.0 - 0.25 * position) for position, symbol in enumerate(symbols)),
        axis=1,
    )
    quality = invested[symbols].apply(
        lambda row: sum(float(row[symbol]) * ((-1.0) ** position) * 0.8 for position, symbol in enumerate(symbols)),
        axis=1,
    )
    value_bounds, quality_bounds = composition.factor_exposure_bounds["value"], composition.factor_exposure_bounds["quality"]
    assert bool(((value >= value_bounds[0] - 2e-3) & (value <= value_bounds[1] + 2e-3)).all())
    assert bool(((quality >= quality_bounds[0] - 2e-3) & (quality <= quality_bounds[1] + 2e-3)).all())
    assert not np.allclose(local.weights.to_numpy(dtype=float), baseline_weights.to_numpy(dtype=float))


@pytest.mark.slow
def test_constrained_projection_fails_closed_end_to_end(world) -> None:
    """An infeasible beta interval aborts on both sides with one message."""

    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)
    composition = CompositionSpec(
        top_n=6,
        weighting="proportional",
        beta_constraint_min=2.5,
        beta_constraint_max=3.0,
        beta_by_symbol={symbol: 0.6 + 0.15 * position for position, symbol in enumerate(symbols)},
    )
    candidate = StrategyCandidate(
        name="parity_projection_infeasible",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_projection_infeasible.parquet")
    request_path, result_path = tmp_path / "request_projection_infeasible.json", tmp_path / "result_projection_infeasible.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("constrained weight projection")):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    assert "constrained weight projection" in str(local_error.value)


#: The two fail-closed messages of the projection stage.  The widened-tolerance
#: cases below pin the first one exactly: an early-stopped solve that misses the
#: feasible set must still report infeasibility, never a silent book.
_INFEASIBLE_PROJECTION_MESSAGE = "constrained weight projection did not reach a feasible point"


def _floor_over_budget_score() -> pd.DataFrame:
    """The verifier's chain-level repro panel: four names scored 4/3/2/1."""

    return pd.DataFrame(
        [[4.0, 3.0, 2.0, 1.0]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=["A", "B", "C", "D"],
    )


def _floor_over_budget_composition(tolerance: float | None, floor: float | None = 0.2) -> CompositionSpec:
    """``floor * 4 = 0.8`` against an exposure bound of ``0.65``: infeasible.

    ``projection_tolerance`` is the only knob that varies, so the parametrised
    cases below isolate the feasibility check from the stopping accuracy.
    """

    symbols = list(_floor_over_budget_score().columns)
    extra = {} if tolerance is None else {"projection_tolerance": tolerance}
    return CompositionSpec(
        top_n=4,
        weighting="equal",
        weight_floor=floor,
        weight_floor_allow_cash=True,
        factor_exposure_bounds={"value": [0.0, 0.65]},
        factor_exposure_matrix={"value": {symbol: 1.0 for symbol in symbols}},
        **extra,
    )


@pytest.mark.parametrize("tolerance", [None, 1e-9, 1e-3, 1e-2, 0.1])
def test_projection_feasibility_check_does_not_follow_a_widened_tolerance(tolerance) -> None:
    """``projection_tolerance`` is a stopping-accuracy knob only.

    The feasibility allowance used to be proportional to it, so a tolerance of
    ``1e-2`` early-stopped the solver and then declared the infeasible point
    feasible on both ends (``[0.1625] * 4``, every name 0.0375 below the
    floor).  The allowance is now capped by a fixed ceiling: widening the
    tolerance can turn a feasible run into a conservative failure, never the
    other way round.
    """

    score = _floor_over_budget_score()
    composition = _floor_over_budget_composition(tolerance)
    with pytest.raises(ValueError) as local_error:
        build_target_weights(composition, score)
    with pytest.raises(ValueError) as worker_error:
        _target_weights(score, composition.as_payload())
    assert str(local_error.value) == str(worker_error.value) == _INFEASIBLE_PROJECTION_MESSAGE

    # Control: drop the floor and the very same exposure bound becomes
    # feasible, so both engines must return the capped book instead of
    # erroring -- the abort above tracks the infeasibility, not the tolerance.
    control = _floor_over_budget_composition(tolerance, floor=None)
    local_book = build_target_weights(control, score).to_numpy(dtype=float)
    worker_book = _target_weights(score, control.as_payload()).to_numpy(dtype=float)
    np.testing.assert_array_equal(local_book, worker_book)
    assert float(local_book.sum()) <= 0.65 + 1e-9
    assert float(local_book.max()) < 0.2


@pytest.mark.slow
def test_widened_projection_tolerance_still_fails_closed_end_to_end(world) -> None:
    """The chain-level repro through the wire: ``tolerance=1e-2`` plus a floor
    the exposure bound cannot fund.  Both engines abort with one message."""

    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)
    composition = CompositionSpec(
        top_n=4,
        weighting="equal",
        weight_floor=0.2,
        weight_floor_allow_cash=True,
        factor_exposure_bounds={"value": [0.0, 0.65]},
        factor_exposure_matrix={"value": {symbol: 1.0 for symbol in symbols}},
        projection_tolerance=1e-2,
    )
    candidate = StrategyCandidate(
        name="parity_projection_wide_tolerance",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_projection_wide_tolerance.parquet")
    request_path = tmp_path / "request_projection_wide_tolerance.json"
    result_path = tmp_path / "result_projection_wide_tolerance.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(_INFEASIBLE_PROJECTION_MESSAGE)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    assert str(local_error.value) == _INFEASIBLE_PROJECTION_MESSAGE


@pytest.mark.slow
def test_factor_risk_budget_all_capped_fails_closed_end_to_end(world) -> None:
    """e2e coverage of the relu + frb_idio_max interaction shape: every
    positively-projected name is capped away by the idiosyncratic bound while
    a negatively-projected name sits in the book (relu -> zero), so the
    leftover set is empty and both engines fail closed with one message."""

    tmp_path: Path = world["tmp_path"]
    symbols = list(SYMBOLS)
    composition = CompositionSpec(
        top_n=6,
        weighting="factor_risk_budget",
        frb_budgets={"value": 1.0, "quality": 1.0, "momentum": 1.0},
        frb_exposures={
            # S0-S3 project positively; S4/S5 are negative on every factor
            # and therefore relu to zero.
            "value": {"S0": 1.2, "S1": 0.9, "S2": 1.4, "S3": 0.7, "S4": -1.0, "S5": -0.8},
            "quality": {"S0": 0.8, "S1": 1.1, "S2": 0.6, "S3": 1.3, "S4": -0.7, "S5": -0.9},
            "momentum": {"S0": 1.0, "S1": 0.7, "S2": 1.2, "S3": 0.9, "S4": -0.6, "S5": -1.1},
        },
        frb_factor_covariance={
            "value": {"value": 1.0, "quality": 0.1, "momentum": 0.0},
            "quality": {"value": 0.1, "quality": 0.9, "momentum": 0.0},
            "momentum": {"value": 0.0, "quality": 0.0, "momentum": 1.0},
        },
        # The idiosyncratic cap removes every positively-projected name.
        frb_idiosyncratic={"S0": 0.2, "S1": 0.2, "S2": 0.2, "S3": 0.2, "S4": 0.01, "S5": 0.01},
        frb_idio_max=0.1,
    )
    candidate = StrategyCandidate(
        name="parity_frb_all_capped",
        universe="csi_all",
        composition=composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_frb_all_capped.parquet")
    request_path, result_path = tmp_path / "request_frb_all_capped.json", tmp_path / "result_frb_all_capped.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    message = "factor_risk_budget weighting requires at least one positive projected exposure"
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        _local_portfolio(world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    assert str(local_error.value) == message

    # Discrimination control: the raise must come from the capped positive
    # projections, not from the negative-projection names alone.  Dropping the
    # idiosyncratic cap leaves the same exposure matrix and the same relu-
    # zeroed names, and the date must then execute -- with the negatively
    # projected names carrying zero target weight on both sides.
    control_composition = CompositionSpec(
        top_n=6,
        weighting="factor_risk_budget",
        frb_budgets=composition.frb_budgets,
        frb_exposures=composition.frb_exposures,
        frb_factor_covariance=composition.frb_factor_covariance,
        frb_idiosyncratic=composition.frb_idiosyncratic,
    )
    control_candidate = StrategyCandidate(
        name="parity_frb_all_capped_control",
        universe="csi_all",
        composition=control_composition,
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    control_request = world["evaluator"].request_for(control_candidate)
    control_request["portfolio_artifact_path"] = str(tmp_path / "portfolio_frb_all_capped_control.parquet")
    control_request_path = tmp_path / "request_frb_all_capped_control.json"
    control_result_path = tmp_path / "result_frb_all_capped_control.json"
    control_request_path.write_text(json.dumps(control_request), encoding="utf-8")
    control_payload = run_request(control_request_path, control_result_path)
    assert control_payload["status"] == "success", control_payload

    control_local = _local_portfolio(
        world,
        control_request,
        control_candidate.composition,
        control_candidate.signal,
        backtest=control_candidate.backtest,
    )
    _assert_end_to_end_parity(control_local, control_payload, Path(control_request["portfolio_artifact_path"]))
    control_weights = control_local.weights[["S4", "S5"]].to_numpy(dtype=float)
    assert bool((control_weights == 0.0).all())
    assert bool((control_local.weights.sum(axis=1) > 0.0).any())


@pytest.mark.parametrize(
    ("payload_override", "message"),
    [
        ({"industry_deviation_max": {"X": 0.1}}, "industry deviation requires industry_deviation_industry_map"),
        ({"industry_deviation_max": {"X": 0.1}, "industry_deviation_industry_map": {"A": "X"}}, "industry deviation requires industry_deviation_benchmark"),
        ({"industry_deviation_industry_map": {"A": "X"}}, "industry deviation parameters require industry_deviation_max"),
        ({"industry_deviation_max": {"X": 0}}, "industry_deviation_max must map non-empty industry names to finite positive deviations"),
        ({"industry_deviation_max": {"X": -0.1}}, "industry_deviation_max must map non-empty industry names to finite positive deviations"),
        ({"industry_deviation_max": 5}, "industry_deviation_max must map non-empty industry names to finite positive deviations"),
        (
            {
                "industry_deviation_max": {"X": 0.1, "Y": 0.1},
                "industry_deviation_industry_map": {"A": "X"},
                "industry_deviation_benchmark": {"X": 0.5},
                "industry_deviation_exempt": ["Z"],
            },
            "industry_deviation_exempt industries must appear in industry_deviation_max",
        ),
        (
            {
                "industry_deviation_max": {"X": 0.1},
                "industry_deviation_industry_map": {"A": "X"},
                "industry_deviation_benchmark": {"X": 0.5},
                "industry_deviation_exempt": "X",
            },
            "industry_deviation_exempt must be a list of non-empty industry names",
        ),
        ({"style_exposure_bounds": {"size": [0.1, 0.0]}}, "style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper"),
        ({"style_exposure_bounds": {"size": [0.0, "high"]}}, "style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper"),
        ({"style_exposure_matrix": {"size": {"A": 1.0}}}, "style exposure parameters require style_exposure_bounds"),
        (
            {"style_exposure_bounds": {"size": [-0.1, 0.1]}},
            "style exposure constraint requires style_exposure_matrix",
        ),
        (
            {"style_exposure_bounds": {"size": [-0.1, 0.1]}, "style_exposure_matrix": {"size": {"A": 1.0}}},
            "style exposure constraint requires style_exposure_benchmark",
        ),
        (
            {
                "style_exposure_bounds": {"size": [-0.1, 0.1]},
                "style_exposure_matrix": {"beta": {"A": 1.0}},
                "style_exposure_benchmark": {"size": 0.0},
            },
            "style_exposure_matrix must cover every bounded style",
        ),
        (
            {
                "style_exposure_bounds": {"size": [-0.1, 0.1]},
                "style_exposure_matrix": {"size": {"A": 1.0}},
                "style_exposure_benchmark": {"beta": 0.0},
            },
            "style_exposure_benchmark must cover every bounded style",
        ),
        ({"beta_constraint_min": 0.5}, "beta constraint requires beta_by_symbol"),
        ({"beta_by_symbol": {"A": 1.0}}, "beta constraint parameters require beta_constraint_min or beta_constraint_max"),
        ({"beta_constraint_min": 1.0, "beta_constraint_max": 0.5}, "beta_constraint_min must not exceed beta_constraint_max"),
        ({"beta_constraint_min": "high"}, "beta_constraint_min must be a finite number when provided"),
        ({"factor_exposure_bounds": {"value": [0.0, 0.2]}}, "factor exposure constraint requires factor_exposure_matrix"),
        ({"factor_exposure_matrix": {"value": {"A": 1.0}}}, "factor exposure parameters require factor_exposure_bounds"),
        (
            {
                "factor_exposure_bounds": {"value": [0.0, 0.2], "quality": [-0.1, 0.1]},
                "factor_exposure_matrix": {"value": {"A": 1.0}},
            },
            "factor_exposure_matrix must cover every bounded factor",
        ),
        ({"projection_max_iter": 0}, "projection parameters require an exposure constraint"),
        ({"projection_tolerance": 1e-3}, "projection parameters require an exposure constraint"),
        (
            {"beta_constraint_min": 0.5, "beta_constraint_max": 1.5, "beta_by_symbol": {"A": 1.0}, "projection_max_iter": 0},
            "projection_max_iter must be at least 1",
        ),
        (
            {"beta_constraint_min": 0.5, "beta_constraint_max": 1.5, "beta_by_symbol": {"A": 1.0}, "projection_max_iter": 2.5},
            "projection_max_iter must be a whole number",
        ),
        (
            {"beta_constraint_min": 0.5, "beta_constraint_max": 1.5, "beta_by_symbol": {"A": 1.0}, "projection_tolerance": 0},
            "projection_tolerance must be a finite positive number",
        ),
    ],
)
def test_batch_c3_composition_failures_fail_closed_on_both_sides(world, payload_override: dict, message: str) -> None:
    candidate = StrategyCandidate(
        name="parity_batch_c3_fail_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    tmp_path: Path = world["tmp_path"]
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch_c3_fail.parquet")
    request["candidate"]["composition"].update(payload_override)
    request_path, result_path = tmp_path / "request_batch_c3_fail.json", tmp_path / "result_batch_c3_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        CompositionSpec(top_n=4, **payload_override)
    assert str(local_error.value) == message


#: The non-finite guards added after the C3 review.  The projector rejects a
#: NaN/inf weight row outright; the score panel rejects inf while NaN keeps its
#: meaning of "excluded on this date".
_NON_FINITE_PROJECTION_INPUT_MESSAGE = "constrained weight projection input must contain only finite weights"
_INFINITE_SCORE_PANEL_MESSAGE = "score panel must not contain infinite values"

#: Boundary residuals for the feasibility ceiling.  ``_apply_weight_floor``
#: lifts the equal-weighted book onto the floor and the factor exposure bound
#: then pulls every name back below it by ``delta``; with
#: ``projection_max_iter=1`` that single-pass residual survives, so the
#: feasibility check is the only thing deciding.  ``delta`` is tunable through
#: the exposure bound (``upper = 0.8 - 4 * delta`` for four names at floor 0.2).
_CEILING_DELTA_BELOW = 5e-8
_CEILING_DELTA_ABOVE = 1.3e-7
_CEILING_BOUNDARY_TOLERANCE = 1e-8


def _two_name_projection_composition() -> CompositionSpec:
    return CompositionSpec(
        top_n=2,
        weighting="equal",
        factor_exposure_bounds={"value": [0.0, 1.0]},
        factor_exposure_matrix={"value": {"A": 1.0, "B": 1.0}},
    )


def _all_projection_families_composition() -> CompositionSpec:
    """Every projection family switched on, so the payload does carry all the
    projection keys -- and still none that could move the ceiling."""

    symbols = list(_floor_over_budget_score().columns)
    return CompositionSpec(
        top_n=4,
        weighting="equal",
        weight_cap=0.6,
        weight_floor=0.1,
        weight_floor_allow_cash=True,
        industry_deviation_max={"X": 0.2},
        industry_deviation_industry_map={symbol: "X" for symbol in symbols},
        industry_deviation_benchmark={"X": 1.0},
        style_exposure_bounds={"size": [-1.0, 1.0]},
        style_exposure_matrix={"size": {symbol: 1.0 for symbol in symbols}},
        style_exposure_benchmark={"size": 0.0},
        beta_constraint_min=0.5,
        beta_constraint_max=1.5,
        beta_by_symbol={symbol: 1.0 for symbol in symbols},
        factor_exposure_bounds={"value": [0.0, 1.0]},
        factor_exposure_matrix={"value": {symbol: 1.0 for symbol in symbols}},
        projection_max_iter=7,
        projection_tolerance=1e-8,
    )


def _floor_gap_composition(delta: float, tolerance: float) -> CompositionSpec:
    """Four names at floor 0.2 whose exposure bound leaves a ``delta`` gap."""

    symbols = list(_floor_over_budget_score().columns)
    upper = len(symbols) * 0.2 - len(symbols) * delta
    return CompositionSpec(
        top_n=4,
        weighting="equal",
        weight_floor=0.2,
        weight_floor_allow_cash=True,
        factor_exposure_bounds={"value": [0.0, upper]},
        factor_exposure_matrix={"value": {symbol: 1.0 for symbol in symbols}},
        projection_max_iter=1,
        projection_tolerance=tolerance,
    )


def test_projection_feasibility_ceiling_is_one_shared_literal() -> None:
    """The feasibility ceiling is the same module constant on both ends.

    It is what stops a widened ``projection_tolerance`` from also widening the
    feasibility check, so a drifted literal on either side would quietly
    reopen the hole.  Pinned here: same number on both ends, and unreachable
    from the canonical payload (no key of the composition section -- even with
    every projection family enabled -- can override it).
    """

    assert _local_projection_ceiling == 1e-7
    assert _worker_projection_ceiling == 1e-7
    assert _local_projection_ceiling == _worker_projection_ceiling

    composition = _all_projection_families_composition()
    payload = composition.as_payload()
    local_params = _local_projection_params(composition)
    worker_params = _projection_params_from_payload(payload)
    assert local_params is not None
    assert set(local_params) == set(worker_params)
    for mapping in (payload, local_params, worker_params):
        assert [key for key in mapping if "ceiling" in key or "feasib" in key] == []


@pytest.mark.parametrize(
    ("delta", "accepted"),
    [(_CEILING_DELTA_BELOW, True), (_CEILING_DELTA_ABOVE, False)],
    ids=["under_the_ceiling", "over_the_ceiling"],
)
def test_projection_feasibility_ceiling_boundary_is_the_same_on_both_ends(delta: float, accepted: bool) -> None:
    """A residual just under the ceiling executes, just over it fails closed.

    The tolerance is ``1e-8``, so the uncapped allowance would be
    ``16 * tolerance = 1.6e-7``: the rejected residual sits inside the band the
    ceiling exists to close, and without the ceiling both residuals would
    execute.  This is the standing version of the fuzz evidence that no real
    violation is waved through.
    """

    score = _floor_over_budget_score()
    composition = _floor_gap_composition(delta, _CEILING_BOUNDARY_TOLERANCE)
    if accepted:
        local_book = build_target_weights(composition, score).to_numpy(dtype=float)
        worker_book = _target_weights(score, composition.as_payload()).to_numpy(dtype=float)
        np.testing.assert_array_equal(local_book, worker_book)
        gap = float(np.max(0.2 - local_book))
        assert 0.0 < gap <= _local_projection_ceiling
        return

    with pytest.raises(ValueError) as local_error:
        build_target_weights(composition, score)
    with pytest.raises(ValueError) as worker_error:
        _target_weights(score, composition.as_payload())
    message = f"constrained weight projection did not converge within {composition.projection_max_iter} iterations"
    assert str(local_error.value) == str(worker_error.value) == message
    # Discrimination self-check: without the ceiling this residual would have
    # fitted inside the allowance and the book would have executed.
    assert _local_projection_ceiling < delta < 16.0 * _CEILING_BOUNDARY_TOLERANCE


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_projection_input_fails_closed_on_both_ends(bad: float) -> None:
    """A non-finite weight row reaches the projector only through an upstream
    bug: both ends abort with one message instead of projecting a NaN book."""

    composition = _two_name_projection_composition()
    values = np.array([0.5, bad])
    with pytest.raises(ValueError) as local_error:
        _local_apply_constrained_projection(values, ["A", "B"], _local_projection_params(composition))
    with pytest.raises(ValueError) as worker_error:
        _apply_constrained_projection(values, ["A", "B"], _projection_params_from_payload(composition.as_payload()))
    assert str(local_error.value) == str(worker_error.value) == _NON_FINITE_PROJECTION_INPUT_MESSAGE


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_score_panel_fails_closed_on_both_ends(bad: float) -> None:
    """An infinite score is rejected instead of ranked -- on both ends.

    The control keeps NaN in the same cell: NaN is the "excluded on this date"
    marker and must keep working, so the new guard covers inf only.
    """

    composition = CompositionSpec(top_n=4, weighting="equal")
    score = _floor_over_budget_score()
    score.iloc[0, 0] = bad
    with pytest.raises(ValueError) as local_error:
        build_target_weights(composition, score)
    with pytest.raises(ValueError) as worker_error:
        _target_weights(score, composition.as_payload())
    assert str(local_error.value) == str(worker_error.value) == _INFINITE_SCORE_PANEL_MESSAGE

    control = _floor_over_budget_score()
    control.iloc[0, 0] = np.nan
    local_book = build_target_weights(composition, control).to_numpy(dtype=float)
    worker_book = _target_weights(control, composition.as_payload()).to_numpy(dtype=float)
    np.testing.assert_array_equal(local_book, worker_book)
    assert float(local_book[0, 0]) == 0.0
    assert float(local_book[0, 1:].sum()) == pytest.approx(1.0)


@pytest.mark.slow
def test_infinite_score_panel_fails_closed_inside_run_portfolio_backtest() -> None:
    """The selection-layer branch repeats the builder's score guard.

    With the selection-layer operators enabled ``run_portfolio_backtest`` reads
    the score panel straight off its ``score`` argument, so it carries its own
    non-finite guard.  The worker's ``_backtest`` has no twin: its caller
    validates the rank panels before the loop, which makes this guard
    unreachable through the wire and leaves a direct call as the only cover.
    """

    dates = pd.bdate_range("2024-01-02", periods=3)
    symbols = ["A", "B"]
    target = pd.DataFrame([[0.5, 0.5]], index=dates[:1], columns=symbols)
    returns = pd.DataFrame(0.0, index=dates, columns=symbols)
    composition = CompositionSpec(top_n=2, weighting="equal")
    score = pd.DataFrame([[1.0, np.inf]], index=dates[:1], columns=symbols)
    with pytest.raises(ValueError) as error:
        run_portfolio_backtest(
            target,
            returns,
            spec=BacktestSpec(signal_lag=1),
            composition=composition,
            score=score,
        )
    assert str(error.value) == _INFINITE_SCORE_PANEL_MESSAGE

    # Discrimination control: the identical direct call with a finite panel
    # executes, so the abort above tracks the inf cell, not the call shape.
    control = pd.DataFrame([[1.0, 0.5]], index=dates[:1], columns=symbols)
    result = run_portfolio_backtest(
        target,
        returns,
        spec=BacktestSpec(signal_lag=1),
        composition=composition,
        score=control,
    )
    assert list(result.weights.columns) == symbols
    assert float(result.weights.iloc[-1].sum()) == pytest.approx(1.0)


#: The lower clamp of the projection feasibility allowance.  ``tolerance=1e-9``
#: already implies ``16t = 1.6e-8`` on its own, so no earlier case ever reached
#: the ``1e-8`` floor; the tiny tolerance below puts the allowance exactly on it.
_PROJECTION_ALLOWANCE_FLOOR = 1e-8
_PROJECTION_TINY_TOLERANCE = 1e-12
_PROJECTION_FLOOR_DELTA = 5e-9


@pytest.mark.slow
def test_projection_allowance_floor_admits_a_residual_the_raw_tolerance_rejects() -> None:
    """A one-pass residual of ``5e-9`` executes on both ends.

    The raw allowance ``16 * 1e-12`` is two orders of magnitude below that
    residual, so the fixed ``1e-8`` floor is the only reason both engines admit
    this book: the clamp is load-bearing, not cosmetic.
    """

    score = _floor_over_budget_score()
    composition = _floor_gap_composition(_PROJECTION_FLOOR_DELTA, _PROJECTION_TINY_TOLERANCE)
    local_book = build_target_weights(composition, score).to_numpy(dtype=float)
    worker_book = _target_weights(score, composition.as_payload()).to_numpy(dtype=float)
    np.testing.assert_array_equal(local_book, worker_book)
    gap = float(np.max(0.2 - local_book))
    assert 16.0 * _PROJECTION_TINY_TOLERANCE < gap <= _PROJECTION_ALLOWANCE_FLOOR
    # The floor must stay strictly under the ceiling, otherwise the clamp order
    # ``min(max(16t, floor), ceiling)`` would resolve to the ceiling instead.
    assert _PROJECTION_ALLOWANCE_FLOOR < _local_projection_ceiling


# ---------------------------------------------------------------------------
# Batch 6: order layer 8.6-8.9, portfolio-risk scalings 9.1/9.2 and the signal
# side decay monitor 9.5.
#
# The four execution helpers share one mutable state (the pending order table)
# and are reached by the main loop through a single entry point, so the wire
# contract is checked in three layers here: the shared literals and key sets,
# a value-for-value direct comparison of each pure helper, and an end-to-end
# run of a request carrying a real ``execution_data`` section.
# ---------------------------------------------------------------------------

#: A two-sided mask needs all three inputs at once, so the direct comparisons
#: below build a tiny book with a name that trades up, a name that trades down
#: and a name that does not move.
_ORDER_SYMBOLS = ("A", "B", "C")


def _order_series(**weights: float) -> pd.Series:
    return pd.Series({symbol: weights.get(symbol, 0.0) for symbol in _ORDER_SYMBOLS}, dtype=float)


def _assert_series_identical(local: pd.Series, worker: pd.Series, label: str) -> None:
    assert list(local.index) == list(worker.index), f"{label}: index drifted"
    np.testing.assert_array_equal(local.to_numpy(dtype=float), worker.to_numpy(dtype=float), err_msg=label)


#: The backtest keys earlier batches own.  Batch 6's two allocators are
#: mutually exclusive, so no single spec can open every order-layer key; the
#: union of an 8.6-armed spec and an 8.9-armed spec is the widest payload the
#: batch can emit, and the residual below is what the whitelist keeps for the
#: execution-layer keys landed before it (8.3 drift band, 8.4 minimum holding
#: period, 8.5 turnover shrinkage).
_PRE_BATCH6_BACKTEST_KEYS = frozenset(
    {
        "drift_band_forced_days",
        "drift_band_name",
        "drift_band_portfolio",
        "min_holding_days",
        "turnover_max_step",
        "turnover_shrinkage",
    }
)


def test_order_layer_shared_literals_and_key_sets_are_one_on_both_ends() -> None:
    """The three shared literals and the two key whitelists must not drift.

    The whitelists are the worker-side enforcement of the canonical payload: a
    local payload key the whitelist does not carry fails closed on the wire, and
    a whitelist key no payload can produce is a knob nothing validates.  Both
    directions are asserted, over the union of the two mutually exclusive
    allocators, so the equality is not accidental.
    """

    assert tuple(DRAWDOWN_STATES) == tuple(_worker_drawdown_states)
    assert tuple(EXECUTION_DATA_KEYS) == tuple(_worker_execution_data_keys)
    assert tuple(PRICE_LIMIT_SIDES) == tuple(_worker_price_limit_sides)

    participation = BacktestSpec(
        signal_lag=1,
        participation_max_rate=0.5,
        participation_max_days=3,
        participation_unfilled="carry",
        participation_amount_window=2,
        lot_rounding=True,
        lot_rounding_mode="down",
        lot_min_notional=0.05,
        side_aware_tradability="block_buy",
        vol_target_annual=0.15,
        vol_target_window=8,
        vol_target_min_scale=0.2,
        vol_target_max_scale=2.0,
        drawdown_thresholds={"warning": 0.03, "reduce": 0.06, "halt": 0.10},
        drawdown_scales={"warning": 0.8, "reduce": 0.5, "halt": 0.0},
        drawdown_confirm_days=2,
        drawdown_recovery=0.02,
    )
    staged = BacktestSpec(signal_lag=1, staged_max_days=2, staged_priority="score")
    emitted = set(participation.as_payload()) | set(staged.as_payload())
    assert emitted <= set(_worker_backtest_keys)
    assert set(_worker_backtest_keys) - emitted == _PRE_BATCH6_BACKTEST_KEYS
    # The two allocators must not be able to open the same key set: if they
    # could, the exclusion above would be the only thing keeping the payload
    # unambiguous and this case would stop proving anything about it.
    assert set(staged.as_payload()) - set(participation.as_payload()) == {"staged_max_days", "staged_priority"}

    # The signal side carries the 9.5 keys the same way: present while the
    # monitor is on, absent (and therefore identity-preserving) while it is off.
    decay = SignalSpec(decay_window=12, decay_fail_periods=2, decay_ic_threshold=0.01, decay_scale=0.25)
    assert set(decay.as_payload()) <= set(_worker_signal_keys)
    assert set(SignalSpec().as_payload()) <= set(_worker_signal_keys)
    assert set(decay.as_payload()) - set(SignalSpec().as_payload()) == {
        "decay_window",
        "decay_fail_periods",
        "decay_ic_threshold",
        "decay_scale",
    }


def test_side_aware_tradability_agrees_value_for_value() -> None:
    carried = _order_series(A=0.2, B=0.2, C=0.2)
    target = _order_series(A=0.5, B=0.0, C=0.3)
    halt = {"d0": {"A": "suspended", "B": ""}}
    limits = {"d0": {"A": {"upper": 10.0}, "C": {"lower": 1.0}}}
    price = {"d0": {"A": 12.0, "B": 5.0, "C": 1.0}}

    for mode in ("block_buy", "block_sell", "block_both"):
        local = _local_apply_side_aware_tradability(carried, target, halt, limits, price, mode=mode, day="d0")
        worker = _apply_side_aware_tradability(carried, target, halt, limits, price, mode=mode, day="d0")
        _assert_series_identical(local, worker, f"side_aware[{mode}]")


def test_lot_rounding_agrees_value_for_value() -> None:
    carried = _order_series(A=0.0, B=0.2, C=0.1)
    executed = _order_series(A=0.137, B=0.205, C=0.1)
    price = {"d0": {"A": 0.05, "B": 0.07, "C": 3.0}}
    lot_sizes = {"A": 1, "B": 10, "C": 100}

    for mode in ("down", "nearest"):
        for min_notional in (None, 0.05):
            local = _local_apply_lot_rounding(
                executed, carried, price, lot_sizes, min_notional=min_notional, mode=mode, day="d0"
            )
            worker = _apply_lot_rounding(executed, carried, price, lot_sizes, min_notional=min_notional, mode=mode, day="d0")
            _assert_series_identical(local, worker, f"lot[{mode},{min_notional}]")


def test_participation_cap_agrees_value_for_value_across_a_whole_plan() -> None:
    """Walk a plan past its own expiry and compare the state at every step."""

    amount = {f"d{day}": {"A": 0.1, "B": 0.4} for day in range(6)}
    target = _order_series(A=0.5, B=0.5)
    local_carried = worker_carried = _order_series()
    local_pending: dict[str, list[float]] = {}
    worker_pending: dict[str, list[float]] = {}

    for position in range(6):
        decision = target if position == 0 else None
        kwargs = {
            "position": position,
            "day": f"d{position}",
            "window_keys": [f"d{day}" for day in range(position + 1)],
            "rate": 0.5,
            "window": 2,
            "unfilled": "carry",
            "max_days": 3,
        }
        local_carried, local_pending = _local_apply_participation_cap(
            local_carried, local_pending, decision, amount, **kwargs
        )
        worker_carried, worker_pending = _apply_participation_cap(worker_carried, worker_pending, decision, amount, **kwargs)
        _assert_series_identical(local_carried, worker_carried, f"cap book day {position}")
        assert local_pending == worker_pending, f"cap pending day {position}"

    # The plan must have been carried and then cancelled, not silently refilled.
    assert local_pending == worker_pending == {}
    assert float(local_carried.sum()) < float(target.sum())


def test_staged_execution_agrees_value_for_value_across_a_whole_plan() -> None:
    target = _order_series(A=0.3, B=0.2, C=0.1)
    local_carried = worker_carried = _order_series()
    local_pending: dict[str, list[float]] = {}
    worker_pending: dict[str, list[float]] = {}

    for position in range(4):
        decision = target if position == 0 else None
        local_carried, local_pending = _local_stage_orders(
            local_carried, local_pending, decision, position=position, max_days=3, order=["C", "B", "A"]
        )
        worker_carried, worker_pending = _stage_orders(
            worker_carried, worker_pending, decision, position=position, max_days=3, order=["C", "B", "A"]
        )
        _assert_series_identical(local_carried, worker_carried, f"staged book day {position}")
        assert local_pending == worker_pending, f"staged pending day {position}"

    assert local_pending == worker_pending == {}


def test_plan_order_and_execution_layer_entry_point_agree() -> None:
    carried = _order_series(A=0.1, B=0.2)
    executed = _order_series(A=0.4, B=0.2)
    score = _order_series(A=0.1, B=0.9)

    for priority, score_row in (("weight", None), (None, None), ("score", score), (None, score)):
        assert _local_plan_order(executed, carried, priority=priority, score_row=score_row) == _plan_order(
            executed, carried, priority=priority, score_row=score_row
        ), f"service order[{priority}]"

    amount = {"d0": {"A": 0.1}, "d1": {"A": 0.1}}
    price = {"d0": {"A": 2.0}, "d1": {"A": 2.0}}
    call = {
        "position": 0,
        "day": "d0",
        "calendar": ["d0"],
        "order": ["A"],
        "participation_rate": 0.5,
        "participation_window": 1,
        "participation_unfilled": "carry",
        "participation_max_days": 2,
        "staged_max_days": None,
        "lot_rounding": True,
        "lot_min_notional": None,
        "lot_rounding_mode": "down",
        "side_aware_tradability": "block_buy",
        "amount": amount,
        "price": price,
        "lot_sizes": {"A": 100},
        "halt": {"d0": {"A": "suspended"}},
        "limits": None,
    }
    local_book, local_pending = _local_apply_execution_layer(_order_series(), {}, _order_series(A=0.5), **call)
    worker_book, worker_pending = _apply_execution_layer(_order_series(), {}, _order_series(A=0.5), **call)
    _assert_series_identical(local_book, worker_book, "execution layer book")
    assert local_pending == worker_pending, "execution layer pending"


def test_portfolio_volatility_target_agrees_value_for_value() -> None:
    samples = [
        [],
        [0.01],
        [0.5, 0.5, 0.5, 0.5],
        [0.1, -0.1, 0.1, -0.1],
        [0.008, -0.008] * 4,
        [0.0, 0.0, 0.0],
        [0.02, 0.03, -0.01, 0.005, -0.02],
    ]
    for values in samples:
        for target_annual, window in ((0.1, 8), (0.2, 4), (0.05, 2)):
            local = _local_volatility_scale(
                values, target_annual=target_annual, window=window, min_scale=0.2, max_scale=3.0, periods_per_year=252
            )
            worker = _volatility_scale(
                values, target_annual=target_annual, window=window, min_scale=0.2, max_scale=3.0, periods_per_year=252
            )
            assert float(local) == float(worker), f"vol scale[{values},{target_annual},{window}]"


def test_drawdown_state_machine_agrees_value_for_value_along_a_path() -> None:
    thresholds = {"warning": 0.03, "reduce": 0.06, "halt": 0.10}
    scales = {"warning": 0.8, "reduce": 0.5, "halt": 0.0}
    path = [1.0, 0.95, 0.95, 0.93, 0.93, 0.89, 0.89, 0.91, 0.91, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 0.88, 0.88]

    local_machine: dict[str, object] = {}
    worker_machine: dict[str, object] = {}
    for nav in path:
        local = _local_drawdown_state_machine(
            [nav], machine=local_machine, thresholds=thresholds, scales=scales, confirm_days=2, recovery=0.02
        )
        worker = _drawdown_state_machine(
            [nav], machine=worker_machine, thresholds=thresholds, scales=scales, confirm_days=2, recovery=0.02
        )
        assert float(local) == float(worker), f"drawdown scale at nav {nav}"
        assert local_machine == worker_machine, f"drawdown state at nav {nav}"


# ---------------------------------------------------------------------------
# Batch 6 end-to-end runs: one request per operator, built so the candidate
# payload really carries the section the operator reads (``execution_data`` for
# the order layer, the risk configuration for 9.1/9.2, the signal spec for 9.5).
#
# Every case is run twice.  The first run is the parity run: the worker executes
# the request and the local engine re-runs the same request, and the two books
# must agree bit for bit.  The second run is the discrimination run: the same
# request, same panels, operator switched off -- earlier batches showed that a
# broken operator satisfies a parity assertion exactly as loudly as a working
# one, so a case that cannot show the operator moving the book proves nothing.
#
# 9.1/9.2 run on the daily cadence on purpose.  Both read a realised-return
# history; on the monthly default the first decision would be the only one with
# a short window and the declared fallback would swallow most of the operator.
# ---------------------------------------------------------------------------

_TRADING_DAY_KEYS = tuple(pd.Timestamp(date).date().isoformat() for date in EVAL_DATES)

#: 9.5's discriminating blend.  An equal-weight blend shrinks every factor by
#: the same amount, which a uniform row scale cannot show; weighting the
#: factors differently is what makes the monitor visible in the book.
_DECAY_BLEND = {"value": 0.6, "quality": 0.3, "momentum": 0.1}


def _flat_execution_panel(value: float) -> dict:
    return {day: dict.fromkeys(SYMBOLS, value) for day in _TRADING_DAY_KEYS}


def _flat_price_limits(upper: float, lower: float) -> dict:
    return {day: {symbol: {"upper": upper, "lower": lower} for symbol in SYMBOLS} for day in _TRADING_DAY_KEYS}


def _run_batch6_end_to_end(world, tmp_path: Path, candidate: StrategyCandidate, tag: str, *, daily=False, blank_symbol=None):
    """Execute one batch-6 request on both ends and assert they agree."""

    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{tag}.parquet")
    if daily:
        request["evaluation"]["rebalance_frequency"] = "daily"
    if blank_symbol is not None:
        # Partial blanking is what makes the rank pipeline's missing handling
        # visible; a fully blanked symbol survives percentile ranking untouched.
        _make_symbol_missing(request, "value", blank_symbol)
    request_path, result_path = tmp_path / f"request_{tag}.json", tmp_path / f"result_{tag}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload
    local = _local_portfolio(
        world, request, candidate.composition, candidate.signal, candidate.gating, candidate.backtest
    )
    _assert_end_to_end_parity(local, payload, Path(request["portfolio_artifact_path"]))
    return request, local, payload


def _local_operator_off(world, request, candidate: StrategyCandidate, *, signal: SignalSpec | None = None):
    """The same request with the operator under test switched off.

    The local front-end reads its configuration off the request payload, so the
    control has to be a *different spec object* handed to the same request --
    re-running an unchanged request would compare the candidate against itself
    and report a perfectly meaningless zero difference.
    """

    return _local_portfolio(
        world,
        request,
        candidate.composition,
        signal if signal is not None else candidate.signal,
        None,
        BacktestSpec(signal_lag=int(request["candidate"]["backtest"]["signal_lag"])),
    )


def _max_weight_gap(local, other) -> float:
    return float((local.weights - other.weights).abs().to_numpy().max())


def _book_scale_per_decision(local, baseline) -> tuple[dict, float]:
    """Row scaling implied by two runs, plus how far it spreads across names.

    A row the operator multiplied by a single number shows up as the same
    per-name ratio for every name both runs hold; ``spread`` is the largest
    disagreement between those ratios anywhere in the panel, so a scaler that is
    uniform by construction keeps it at the floating-point noise floor.
    """

    ratios: dict = {}
    spread = 0.0
    for date in local.weights.index:
        base_row = baseline.weights.loc[date]
        mask = base_row.abs() > 1e-12
        if not mask.any():
            continue
        per_name = (local.weights.loc[date][mask] / base_row[mask]).to_numpy()
        ratios[str(date)[:10]] = float(per_name.mean())
        spread = max(spread, float(per_name.max() - per_name.min()))
    return ratios, spread


_ORDER_LAYER_END_TO_END_CASES = [
    pytest.param(
        "lot_rounding",
        BacktestSpec(signal_lag=1, lot_rounding=True, lot_rounding_mode="down"),
        ExecutionDataSpec(price=_flat_execution_panel(0.0375), lot_sizes={symbol: 1 for symbol in SYMBOLS}),
        id="8.7-lot-rounding",
    ),
    pytest.param(
        "side_aware",
        BacktestSpec(signal_lag=1, side_aware_tradability="block_both"),
        ExecutionDataSpec(
            price=_flat_execution_panel(2.0),
            halt={_TRADING_DAY_KEYS[1]: dict.fromkeys(SYMBOLS, "suspended")},
        ),
        id="8.8-side-aware-tradability",
    ),
    pytest.param(
        "participation",
        BacktestSpec(
            signal_lag=1,
            participation_max_rate=0.5,
            participation_max_days=3,
            participation_unfilled="carry",
            participation_amount_window=1,
        ),
        ExecutionDataSpec(trading_dates=list(_TRADING_DAY_KEYS), amount=_flat_execution_panel(0.02)),
        id="8.6-participation-cap",
    ),
    pytest.param(
        "staged",
        BacktestSpec(signal_lag=1, staged_max_days=3, staged_priority="weight"),
        ExecutionDataSpec(trading_dates=list(_TRADING_DAY_KEYS)),
        id="8.9-staged-execution",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(("label", "backtest", "execution_data"), _ORDER_LAYER_END_TO_END_CASES)
def test_order_layer_operators_agree_end_to_end(world, label: str, backtest: BacktestSpec, execution_data) -> None:
    """8.6-8.9 end to end, with the injected execution panels on the wire."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name=f"parity_order_{label}",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=backtest,
        execution_data=execution_data,
    )
    request, local, _ = _run_batch6_end_to_end(world, tmp_path, candidate, f"order_{label}")
    assert request["candidate"]["execution_data"] is not None
    baseline = _local_operator_off(world, request, candidate)

    if label == "lot_rounding":
        # The injected price panel turns the book into whole lots of 0.0375, so
        # the executed weights are exact multiples of it.
        tail = local.weights.iloc[-1]
        for symbol, weight in (("S2", 0.1875), ("S3", 0.225), ("S4", 0.225), ("S5", 0.2625)):
            assert float(tail[symbol]) == pytest.approx(weight, abs=1e-12), symbol
        assert float(local.weights.sum(axis=1).max()) < float(baseline.weights.sum(axis=1).max())
    elif label == "side_aware":
        # The whole first trade day is halted, so the book stays empty far
        # longer than it does without the mask.
        empty_local = int((local.weights.sum(axis=1) == 0).sum())
        assert empty_local > int((baseline.weights.sum(axis=1) == 0).sum())
        assert _max_weight_gap(local, baseline) > 0.25
    elif label == "participation":
        # The cap is a containment: the plan is cancelled, never queued, so the
        # final book stays near the participation budget instead of filling up.
        assert float(local.weights.iloc[-1].sum()) == pytest.approx(0.4051969, rel=1e-6)
        assert float(local.weights.sum(axis=1).max()) <= 1.0 + 1e-9
        assert _max_weight_gap(local, baseline) > 0.25
    else:
        # Staged execution serves one target over several days, so the first
        # trade day is only partially filled.
        rows = local.weights.sum(axis=1) - baseline.weights.sum(axis=1)
        assert float(rows.iloc[1]) < -1e-9
        assert _max_weight_gap(local, baseline) > 0.2


@pytest.mark.slow
def test_portfolio_volatility_target_agrees_end_to_end(world) -> None:
    """9.1 PortfolioVolatilityTarget on the daily cadence.

    The target is deliberately far below the realised volatility of the fixture,
    so the upper clamp is reached and the case pins both ends of the scale band
    instead of only the untouched fallback row.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_vol_target",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(
            signal_lag=1,
            vol_target_annual=0.004,
            vol_target_window=8,
            vol_target_min_scale=0.1,
            vol_target_max_scale=2.0,
        ),
    )
    request, local, _ = _run_batch6_end_to_end(world, tmp_path, candidate, "vol_target", daily=True, blank_symbol="S5")
    baseline = _local_operator_off(world, request, candidate)

    ratios, spread = _book_scale_per_decision(local, baseline)
    values = sorted({round(value, 10) for value in ratios.values()})
    assert spread <= 1e-12
    assert len(values) == 10
    assert values[-1] == pytest.approx(2.0)
    assert all(value > 0.1 + 1e-12 for value in values)
    assert sum(1 for value in values if 0.1 < value < 2.0) == 9
    # Exactly one decision is untouched: the first one, which has a single
    # realised observation and therefore falls back to "no scaling".
    assert [date for date, value in ratios.items() if abs(value - 1.0) < 1e-12] == ["2024-01-03"]
    assert len(ratios) == 64
    assert _max_weight_gap(local, baseline) > 0.29
    # A scale above 1 is leverage: the row sum is allowed to exceed the book.
    assert float(local.weights.sum(axis=1).max()) == pytest.approx(2.0)


@pytest.mark.slow
def test_drawdown_state_machine_agrees_end_to_end(drawdown_world) -> None:
    """9.2 DrawdownStateMachine on a world that really draws down and recovers.

    The scaler only ever reduces, so every row stays at or below the baseline.
    Turnover is *not* flat while a tier is held -- the design note claims that
    and the engine disagrees: holding a tier still trades against drift, so the
    held days sit in a narrow band, and a tier change lands as one day that
    clears the band on both sides of itself.  The case pins that shape.
    """

    tmp_path: Path = drawdown_world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_drawdown",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(
            signal_lag=1,
            drawdown_thresholds={"warning": 0.03, "reduce": 0.06, "halt": 0.10},
            drawdown_scales={"warning": 0.8, "reduce": 0.5, "halt": 0.0},
            drawdown_confirm_days=2,
            drawdown_recovery=0.02,
        ),
    )
    request, local, _ = _run_batch6_end_to_end(
        drawdown_world, tmp_path, candidate, "drawdown", daily=True, blank_symbol="S5"
    )
    baseline = _local_operator_off(drawdown_world, request, candidate)

    ratios, spread = _book_scale_per_decision(local, baseline)
    assert spread <= 1e-12
    assert sorted({round(value, 10) for value in ratios.values()}) == [0.5, 0.8, 1.0]
    assert _max_weight_gap(local, baseline) == pytest.approx(0.1450892857142857, rel=1e-9)
    assert float(local.weights.sum(axis=1).max()) <= 1.0 + 1e-9

    # The tier sequence is read back out of the book, not restated: a decision
    # whose scale differs from the previous one is where the ladder moved.
    dates = list(ratios)
    tiers = list(ratios.values())
    change_days = [
        date for index, date in enumerate(dates) if index and abs(tiers[index] - tiers[index - 1]) > 1e-12
    ]
    assert change_days == ["2024-02-13", "2024-02-21", "2024-03-08", "2024-03-18"]

    # Every decision day stays inside the held band except the four changes --
    # and 2024-01-03, which builds the book from cash and is not a "held" day.
    # The first row (2024-01-02) has no book at all and turns over nothing.
    assert float(local.turnover.iloc[0]) == 0.0
    amplified = [
        str(date)[:10] for date in local.turnover.index if float(local.turnover.loc[date]) > 0.3
    ]
    assert amplified == ["2024-01-03", *change_days]

    # A change is one amplified day, not a new level: it clears its neighbours.
    for day in change_days:
        position = int(local.turnover.index.get_loc(pd.Timestamp(day)))
        assert float(local.turnover.iloc[position]) > float(local.turnover.iloc[position - 1])
        assert float(local.turnover.iloc[position]) > float(local.turnover.iloc[position + 1])


@pytest.mark.slow
def test_factor_decay_monitor_agrees_end_to_end(world) -> None:
    """9.5 FactorDecayMonitor with a non-uniform blend on the wire."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_decay",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
        signal=SignalSpec(
            decay_window=8,
            decay_fail_periods=1,
            decay_ic_threshold=0.5,
            decay_scale=0.0,
            blend_weights=_DECAY_BLEND,
        ),
    )
    request, local, _ = _run_batch6_end_to_end(world, tmp_path, candidate, "decay", daily=True, blank_symbol="S5")
    # The control keeps the blend and drops the monitor, which is the only way
    # to tell a shrunk factor apart from a factor that was never weighted.
    control = SignalSpec(blend_weights=_DECAY_BLEND)
    plain = _local_operator_off(world, request, candidate, signal=control)

    changed = [date for date in local.weights.index if not local.weights.loc[date].equals(plain.weights.loc[date])]
    assert len(changed) == 62
    assert str(changed[0])[:10] == "2024-01-05"
    assert len(local.weights.index) == 65
    assert _max_weight_gap(local, plain) == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert float(local.weights.sum(axis=1).max()) <= 1.0 + 1e-9


def test_factor_decay_helpers_agree_value_for_value() -> None:
    """The three 9.5 helpers, compared directly on the same inputs."""

    dates = pd.bdate_range("2024-01-02", periods=6)
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.03, 0.0, -0.02, 0.01],
            "B": [-0.01, 0.005, -0.02, 0.01, 0.03, -0.015],
            "C": [0.03, 0.02, 0.01, -0.01, -0.03, 0.02],
        },
        index=dates,
    )
    panel = pd.DataFrame(
        {"A": [1.0, 0.5, 0.75], "B": [0.5, 1.0, 0.25], "C": [0.25, 0.75, 0.5]},
        index=dates[:3],
    )

    histories = [
        [],
        [0.4],
        [0.4, 0.5, 0.6],
        [0.0, 0.0],
        [-0.01, -0.02, 0.3],
        [0.2, float("nan"), -0.5],
        [None, None, -0.3],
    ]
    for values in histories:
        for window, fail_periods, threshold, scale in ((2, 1, 0.0, 0.0), (4, 2, 0.05, 0.25), (3, 1, 0.5, 1.0)):
            local = _local_factor_decay_state(
                values, window=window, fail_periods=fail_periods, threshold=threshold, scale=scale
            )
            worker = _factor_decay_state(
                values, window=window, fail_periods=fail_periods, threshold=threshold, scale=scale
            )
            assert float(local) == float(worker), f"decay state[{values},{window}]"

    # Two pinned conventions: a period counts as a failure only when its IC is
    # strictly below the threshold, and a history shorter than fail_periods
    # cannot fail.
    for factory in (_local_factor_decay_state, _factor_decay_state):
        assert factory([0.0, 0.0], window=4, fail_periods=2, threshold=0.0, scale=0.5) == 1.0
        assert factory([-0.01, -0.02], window=4, fail_periods=2, threshold=0.0, scale=0.5) == 0.5
        assert factory([-0.01], window=4, fail_periods=2, threshold=0.0, scale=0.5) == 1.0

    local_ic = _local_factor_ic_series(panel, returns, positions=[0, 2, 4], lag=1)
    worker_ic = _factor_ic_series(panel, returns, positions=[0, 2, 4], lag=1)
    assert local_ic == worker_ic
    assert local_ic[-1] is None  # the last decision has no complete holding window

    ranks = {"value": panel, "momentum": panel.iloc[::-1]}
    call = {"lag": 1, "window": 2, "fail_periods": 1, "threshold": 0.0, "scale": 0.25}
    local_scales = _local_factor_decay_scales(ranks, returns, **call)
    worker_scales = _factor_decay_scales(ranks, returns, **call)
    assert set(local_scales) == set(worker_scales) == set(ranks)
    for name, series in local_scales.items():
        pd.testing.assert_series_equal(series, worker_scales[name])
    assert _local_factor_decay_scales({}, returns, **call) == {}
    assert _factor_decay_scales({}, returns, **call) == {}

    outside = pd.DataFrame({"A": [1.0], "B": [0.5]}, index=[dates[-1] + pd.Timedelta(days=30)])
    for factory in (_local_factor_decay_scales, _factor_decay_scales):
        with pytest.raises(ValueError) as failure:
            factory({"value": outside}, returns, **call)
        assert str(failure.value) == "factor decay monitor requires the decision dates to be present in the returns panel"


#: Every batch-6 validation branch, written the way the worker sees it (as a
#: payload override) and the way the local spec sees it (as a constructor
#: keyword).  Both ends must refuse the same value with the same sentence.
_BATCH6_BACKTEST_VALIDATION_ERRORS = [
    ({"participation_max_rate": 0.0}, "participation_max_rate must be in (0, 1] when provided"),
    ({"participation_max_rate": 1.5}, "participation_max_rate must be in (0, 1] when provided"),
    ({"participation_max_rate": True}, "participation_max_rate must be in (0, 1] when provided"),
    (
        {"participation_max_rate": 0.5, "participation_unfilled": "hold"},
        "participation_unfilled must be one of ('cancel', 'carry') when provided",
    ),
    (
        {"participation_max_rate": 0.5, "participation_unfilled": "carry"},
        "participation_max_days must be at least 1 when participation_unfilled is carry",
    ),
    (
        {"participation_max_rate": 0.5, "participation_max_days": 0},
        "participation_max_days must be at least 1 when provided",
    ),
    ({"participation_max_days": 3}, "participation_max_days requires participation_max_rate"),
    ({"lot_rounding": "yes"}, "lot_rounding must be a boolean"),
    ({"lot_min_notional": 0.0}, "lot_min_notional must be positive when provided"),
    ({"lot_rounding": True, "lot_rounding_mode": "up"}, "lot_rounding_mode must be one of ('down', 'nearest') when provided"),
    ({"lot_min_notional": 1.0}, "lot_min_notional requires lot_rounding"),
    (
        {"side_aware_tradability": "block"},
        "side_aware_tradability must be one of ('block_buy', 'block_sell', 'block_both') when provided",
    ),
    ({"staged_max_days": 0}, "staged_max_days must be at least 1 when provided"),
    ({"staged_priority": "score"}, "staged_priority requires staged_max_days"),
    (
        {"participation_max_rate": 0.5, "staged_max_days": 2},
        "participation_cap and staged_execution are mutually exclusive",
    ),
    ({"vol_target_annual": 0.0}, "vol_target_annual must be positive when provided"),
    ({"vol_target_window": 1}, "vol_target_window must be at least 2 when provided"),
    ({"vol_target_window": 5}, "vol_target_window requires vol_target_annual"),
    ({"vol_target_annual": 0.2}, "vol_target_window is required when vol_target_annual is provided"),
    (
        {"vol_target_annual": 0.2, "vol_target_window": 5, "vol_target_min_scale": 0.5, "vol_target_max_scale": 0.9},
        "vol_target_max_scale must be at least 1 when provided",
    ),
    (
        {"vol_target_annual": 0.2, "vol_target_window": 5, "vol_target_min_scale": 1.0, "vol_target_max_scale": 2.0},
        "vol_target_min_scale must be below 1 and vol_target_max_scale at least 1",
    ),
    (
        {"drawdown_thresholds": {"warning": 0.0, "reduce": 0.06, "halt": 0.10}},
        "drawdown_thresholds must be in (0, 1] when provided",
    ),
    (
        {"drawdown_thresholds": {"warning": 0.06, "reduce": 0.03, "halt": 0.10}},
        "drawdown_thresholds must be strictly increasing",
    ),
    (
        {"drawdown_thresholds": {"warning": 0.03, "reduce": 0.06}},
        "drawdown_thresholds must cover exactly the drawdown states",
    ),
    ({"drawdown_confirm_days": 2}, "drawdown_confirm_days requires drawdown_thresholds"),
    (
        {"drawdown_thresholds": {"warning": 0.03, "reduce": 0.06, "halt": 0.10}},
        "drawdown_confirm_days is required when drawdown_thresholds is provided",
    ),
    (
        {
            "drawdown_thresholds": {"warning": 0.03, "reduce": 0.06, "halt": 0.10},
            "drawdown_confirm_days": 2,
            "drawdown_recovery": 2.0,
        },
        "drawdown_recovery must be in (0, 1] when provided",
    ),
]

_BATCH6_SIGNAL_VALIDATION_ERRORS = [
    ({"decay_fail_periods": 2}, "decay_fail_periods requires decay_window"),
    ({"decay_ic_threshold": 0.01}, "decay_ic_threshold requires decay_window"),
    ({"decay_scale": 0.5}, "decay_scale requires decay_window"),
    ({"decay_window": 1}, "decay_window must be at least 2"),
    ({"decay_window": 2.5}, "decay_window must be a whole number"),
    ({"decay_window": 4, "decay_fail_periods": 0}, "decay_fail_periods must be at least 1"),
    ({"decay_window": 4, "decay_ic_threshold": float("nan")}, "decay_ic_threshold must be finite"),
    ({"decay_window": 4, "decay_scale": 1.5}, "decay_scale must be in [0, 1]"),
]


@pytest.mark.parametrize(("override", "message"), _BATCH6_BACKTEST_VALIDATION_ERRORS)
def test_batch6_backtest_keys_fail_closed_identically(world, override: dict, message: str) -> None:
    """The worker's 8.6-9.2 validation and ``BacktestSpec`` must refuse alike.

    The worker reads the value out of the request payload (mutated after
    ``request_for``), the local side through the spec constructor; the two
    sentences are compared literally, so a reworded check on one end only is a
    failure rather than a nicety.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_batch6_backtest_fail",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch6_backtest_fail.parquet")
    request["candidate"]["backtest"].update(override)
    request_path, result_path = tmp_path / "request_batch6_backtest_fail.json", tmp_path / "result_batch6_backtest_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        BacktestSpec(signal_lag=1, **override)
    assert str(local_error.value) == message


@pytest.mark.parametrize(("override", "message"), _BATCH6_SIGNAL_VALIDATION_ERRORS)
def test_batch6_signal_decay_keys_fail_closed_identically(world, override: dict, message: str) -> None:
    """The 9.5 signal keys are validated the same way on both ends."""

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_batch6_signal_fail",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=1),
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_batch6_signal_fail.parquet")
    request["candidate"]["signal"].update(override)
    request_path, result_path = tmp_path / "request_batch6_signal_fail.json", tmp_path / "result_batch6_signal_fail.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        run_request(request_path, result_path)

    with pytest.raises(ValueError) as local_error:
        SignalSpec(**override)
    assert str(local_error.value) == message


@pytest.mark.slow
def test_the_widest_batch6_request_still_carries_no_audit_report(world) -> None:
    """9.3 stays opt-in: arming everything in the batch must not conjure a report.

    The widest payload this batch can emit is built here -- 8.6, 8.7, 8.8, 9.1,
    9.2 and 9.5 at once (8.9 is excluded by the allocator exclusion, and 9.3 has
    no candidate-side switch at all) -- and the same candidate is then run with
    an audit section, so the missing key is shown to be a real absence rather
    than a key the worker simply never writes.
    """

    tmp_path: Path = world["tmp_path"]
    candidate = StrategyCandidate(
        name="parity_batch6_widest",
        universe="csi_all",
        composition=CompositionSpec(top_n=4, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(
            signal_lag=1,
            participation_max_rate=0.5,
            participation_max_days=3,
            participation_unfilled="carry",
            participation_amount_window=1,
            lot_rounding=True,
            lot_rounding_mode="down",
            side_aware_tradability="block_both",
            vol_target_annual=0.15,
            vol_target_window=8,
            vol_target_min_scale=0.5,
            vol_target_max_scale=2.0,
            drawdown_thresholds={"warning": 0.03, "reduce": 0.06, "halt": 0.10},
            drawdown_scales={"warning": 0.8, "reduce": 0.5, "halt": 0.0},
            drawdown_confirm_days=2,
            drawdown_recovery=0.02,
        ),
        signal=SignalSpec(decay_window=8, decay_fail_periods=1, decay_ic_threshold=0.0, decay_scale=0.5),
        execution_data=ExecutionDataSpec(
            trading_dates=list(_TRADING_DAY_KEYS),
            price=_flat_execution_panel(0.0375),
            amount=_flat_execution_panel(0.5),
            lot_sizes={symbol: 1 for symbol in SYMBOLS},
            # 8.8 needs one of the two tradability panels; the bounds are
            # deliberately wide so the widest pipeline still fills a book
            # instead of being blocked into cash.
            price_limits=_flat_price_limits(upper=99.0, lower=-99.0),
        ),
    )
    emitted = set(candidate.backtest.as_payload())
    assert {"participation_max_rate", "lot_rounding", "side_aware_tradability", "vol_target_annual"} <= emitted
    assert {"drawdown_thresholds", "drawdown_scales", "drawdown_confirm_days"} <= emitted
    # The exclusion still holds inside the widest payload.
    assert "staged_max_days" not in emitted
    assert set(candidate.signal.as_payload()) >= {"decay_window", "decay_fail_periods", "decay_scale"}

    request, _, payload = _run_batch6_end_to_end(
        world, tmp_path, candidate, "widest", daily=True, blank_symbol="S5"
    )
    assert "audit" not in request
    assert "audit_report" not in payload

    # The audited run reuses *this* request on purpose: blanking a symbol
    # rewrites the frozen factor artifact on disk, so a freshly built request for
    # the same candidate would carry the pre-blanking checksum and be rejected by
    # the worker's provenance gate.
    request["audit"] = {"stress": _stress_section()}
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_widest_audit.parquet")
    audited_path = tmp_path / "request_widest_audit.json"
    audited_path.write_text(json.dumps(request), encoding="utf-8")
    audited = run_request(audited_path, tmp_path / "result_widest_audit.json")
    assert audited["status"] == "success", audited
    assert "audit_report" in audited


# ---------------------------------------------------------------------------
# Batch 6 red line: the order layer left at its defaults.
# ---------------------------------------------------------------------------

# Captured from the engine shipped before this batch on the same inputs, and
# confirmed to be reproduced by both current ends afterwards.  The shape is
# deliberately not the default one (proportional weighting, ``signal_lag=2``,
# ``top_n=3``): the equal-weight monthly default already carries the pre-batch
# red line above, so reusing it would only restate that one.
BATCH6_REDLINE_WEIGHTS_SHA256 = "6d472dda066b6a371d36f8ff4d97cdbc6e0a6516b3647bc49ba1dd296565a3df"
BATCH6_REDLINE_METRICS: dict = {
    "annual_buy_turnover": 3.8769442080327163,
    "annual_sell_turnover": 2.113110963673119e-05,
    "annual_turnover": 3.8769442080327163,
    "annualized_return": 0.1146702779985226,
    "annualized_vol": 0.0023460284228221722,
    "average_transaction_cost": 1.5384783091834735e-05,
    "avg_holdings": 2.9076923076923076,
    "buy_cost_bps": 10.0,
    "buy_turnover": 0.015384699238225065,
    "cagr": 0.12147130408758522,
    "excess_annualized_return": -0.007848638907779959,
    "information_ratio": -3.3126395050743076,
    "max_drawdown": -0.0005072648990879225,
    "observations": 65.0,
    "round_trip_cost_bps": 20.0,
    "sell_cost_bps": 10.0,
    "sell_turnover": 8.385360966956822e-08,
    "sharpe": 48.878469196284996,
    "signal_lag": 2.0,
    "total_return": 0.03001176547559692,
    "total_transaction_cost": 0.0010000109009692577,
    "tracking_error": 0.002369300642511025,
    "turnover": 0.015384699238225065,
    "win_rate": 0.9538461538461539,
}


@pytest.mark.slow
def test_disabled_order_layer_matches_the_pre_batch6_reference(world, tmp_path: Path) -> None:
    """Regression red line for this batch: with every batch-6 field left at its
    default the worker metrics and both engines' start-of-day weights are
    bit-for-bit the pre-batch reference."""

    candidate = StrategyCandidate(
        name="parity_batch6_redline",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="proportional"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
        backtest=BacktestSpec(signal_lag=2),
        signal=SignalSpec(),
    )
    # Fact 10: a disabled section is absent from the payload, not present empty.
    assert candidate.execution_data is None
    assert set(candidate.backtest.as_payload()) == {"signal_lag", "trading_days_per_year"}
    assert set(candidate.as_payload()) == {
        "backtest",
        "composition",
        "factor_names",
        "name",
        "rule",
        "signal",
        "universe",
    }

    request, local, payload = _run_batch6_end_to_end(world, tmp_path, candidate, "batch6_redline")

    worker_csv = pd.read_parquet(request["portfolio_artifact_path"]).to_csv(float_format="%.17g")
    local_csv = local.weights.to_csv(float_format="%.17g")
    assert sha256(worker_csv.encode()).hexdigest() == BATCH6_REDLINE_WEIGHTS_SHA256
    assert sha256(local_csv.encode()).hexdigest() == BATCH6_REDLINE_WEIGHTS_SHA256
    assert {key: float(value) for key, value in payload["metrics"].items()} == {
        key: float(value) for key, value in BATCH6_REDLINE_METRICS.items()
    }


# ---------------------------------------------------------------------------
# Batch 8: WalkForwardEvaluation (roadmap 10.1).
#
# The one evaluation that re-runs the engine, once per segment.  Unlike every
# other audit section its configuration rides at the request level beside
# ``evaluation`` rather than inside the report-only ``audit`` object, so a
# request that configures it still leaves the candidate payload -- and hence
# the strategy identity hash -- untouched.
# ---------------------------------------------------------------------------


def _run_worker_with_walk_forward(world, tmp_path: Path, candidate, walk_forward, tag: str):
    request = world["evaluator"].request_for(candidate, walk_forward=walk_forward)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{tag}.parquet")
    request_path, result_path = tmp_path / f"request_{tag}.json", tmp_path / f"result_{tag}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload
    return request, payload


def _local_walk_forward(world, request, candidate):
    """Local mirror of the worker segment evaluator: same windows, same engine.

    ``_local_portfolio`` always runs the world's full evaluation window, so a
    segment is expressed the way the worker expresses it -- by narrowing the
    request's evaluation window and handing the local engine that window's own
    close panel and score grid.
    """

    def evaluate_segment(role, start, end):
        result = _local_segment_portfolio(world, request, candidate, start.isoformat(), end.isoformat())
        return dict(result.metrics)

    def _local_segment_portfolio(world, request, candidate, start, end):
        close = _load_close(str(world["base"]), "adjclose", start, end)
        calendar = close.loc[pd.Timestamp(start) : pd.Timestamp(end)].index
        rebalance_dates = _rebalance_dates(calendar, request["evaluation"]["rebalance_frequency"])
        symbols = sorted(set(close.columns))
        score = _score_panel_for(world, request, candidate.signal, rebalance_dates, symbols, gating=candidate.gating)
        daily_returns = close.reindex(columns=symbols).pct_change(fill_method=None).loc[
            pd.Timestamp(start) : pd.Timestamp(end)
        ]
        benchmark = daily_returns.mean(axis=1, skipna=True)
        targets = build_target_weights(candidate.composition, score, returns=daily_returns)
        has_selection_operators = candidate.composition.holding_buffer_entry is not None or (
            candidate.composition.diversity_max_correlation is not None
        )
        return run_portfolio_backtest(
            targets,
            daily_returns,
            spec=candidate.backtest,
            benchmark_returns=benchmark,
            composition=candidate.composition if has_selection_operators else None,
            score=score if has_selection_operators else None,
        )

    config = validate_walk_forward_config(request["walk_forward"])
    return walk_forward_report(config, _segment_calendar(world, request), evaluate_segment)


def _segment_calendar(world, request):
    close = _load_close(str(world["base"]), "adjclose", world["start"], world["end"])
    return list(close.loc[pd.Timestamp(world["start"]) : pd.Timestamp(world["end"])].index)


def _assert_walk_forward_parity(worker_section: dict, local_section: dict) -> None:
    assert set(worker_section) == set(local_section)
    for key in ("mode", "fit_on_train", "parameter_refit", "expanding_train"):
        assert worker_section[key] == local_section[key], key
    for key in ("train_size", "valid_size", "test_size", "step_size", "min_oos_segments", "max_segments"):
        assert worker_section[key] == local_section[key], key

    assert len(worker_section["segments"]) == len(local_section["segments"])
    for worker_segment, local_segment in zip(worker_section["segments"], local_section["segments"]):
        for key in ("segment", "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"):
            assert worker_segment[key] == local_segment[key], key
        for role in ("train", "valid", "test"):
            assert worker_segment[role]["window"] == local_segment[role]["window"], role
            assert worker_segment[role]["in_sample"] == local_segment[role]["in_sample"], role
            worker_metrics = dict(worker_segment[role]["metrics"])
            local_metrics = dict(local_segment[role]["metrics"])
            assert set(worker_metrics) == set(local_metrics), f"segment{worker_segment['segment']}.{role}"
            for key, value in local_metrics.items():
                remote_value = worker_metrics[key]
                # A one-day window leaves some ratios undefined; the worker's
                # JSON round-trip maps a non-finite float to null, so both
                # sides must agree on "undefined" as well as on the number.
                if remote_value is None or not math.isfinite(float(value)):
                    assert not math.isfinite(float(value)) and remote_value is None, (
                        f"segment{worker_segment['segment']}.{role}.{key}"
                    )
                    continue
                assert float(remote_value) == pytest.approx(float(value), rel=1e-9, abs=1e-12), (
                    f"segment{worker_segment['segment']}.{role}.{key}"
                )

    for scope in ("oos", "is"):
        worker_summary, local_summary = worker_section["summary"][scope], local_section["summary"][scope]
        assert worker_summary["segments"] == local_summary["segments"], scope
        assert worker_summary["observations"] == local_summary["observations"], scope
        assert worker_summary["positive_return_ratio"] == pytest.approx(
            local_summary["positive_return_ratio"], rel=1e-12, abs=1e-15
        ), scope
        for key, value in local_summary["mean_metrics"].items():
            assert float(worker_summary["mean_metrics"][key]) == pytest.approx(
                float(value), rel=1e-9, abs=1e-12
            ), f"{scope}.{key}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("walk_forward", "label"),
    [
        ({"train_size": 20, "valid_size": 5, "test_size": 5}, "tiling-step"),
        ({"train_size": 20, "valid_size": 5, "test_size": 5, "step_size": 10}, "overlapping-step"),
        ({"train_size": 20, "valid_size": 1, "test_size": 4, "step_size": 4, "max_segments": 12}, "minimal-valid"),
        (
            {"train_size": 30, "valid_size": 5, "test_size": 5, "step_size": 5, "expanding_train": False},
            "rolling-train",
        ),
        ({"train_size": 20, "valid_size": 5, "test_size": 5, "step_size": 5, "max_segments": 12}, "capped-step"),
    ],
)
def test_walk_forward_evaluation_agrees_end_to_end(world, tmp_path: Path, walk_forward, label: str) -> None:
    """Walk-forward (10.1) parity: the same frozen strategy, segment by segment.

    Each segment re-runs the ordinary pipeline on a narrowed evaluation
    window, so the metrics must match the local engine's window-for-window --
    boundaries included.  Discrimination: the summary must actually move with
    the segmentation, which the cross-case comparison below checks.
    """

    candidate = StrategyCandidate(
        name=f"parity_wf_{label}",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    request, payload = _run_worker_with_walk_forward(world, tmp_path, candidate, walk_forward, f"wf_{label}")

    # The configuration never enters the candidate payload.
    assert "walk_forward" not in json.dumps(request["candidate"])
    assert "walk_forward" not in json.dumps(candidate.as_payload())

    section = payload["audit_report"]["walk_forward"]
    expected = _local_walk_forward(world, request, candidate)
    _assert_walk_forward_parity(section, expected)

    # The base run is untouched: the main metrics still come from the single
    # evaluation window, not from any segment.
    assert payload["metrics"]["observations"] == len(_segment_calendar(world, request))
    assert section["mode"] == "rolling"
    assert section["fit_on_train"] is False
    assert section["parameter_refit"] is False


@pytest.mark.slow
def test_walk_forward_segmentation_moves_the_report(world, tmp_path: Path) -> None:
    """Discrimination: a finer step must produce more, shorter OOS windows.

    Without this the parity cases above would still pass if the segmentation
    were silently ignored and the whole window reported once.
    """

    candidate = StrategyCandidate(
        name="parity_wf_discriminating",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    reports = {}
    for label, walk_forward in (
        ("coarse", {"train_size": 20, "valid_size": 5, "test_size": 10, "step_size": 10}),
        ("fine", {"train_size": 20, "valid_size": 5, "test_size": 5, "step_size": 5}),
    ):
        _, payload = _run_worker_with_walk_forward(world, tmp_path, candidate, walk_forward, f"wf_disc_{label}")
        reports[label] = payload["audit_report"]["walk_forward"]

    coarse, fine = reports["coarse"], reports["fine"]
    assert len(fine["segments"]) > len(coarse["segments"])
    # Ten-day OOS windows see twice the days of five-day windows.
    assert fine["segments"][0]["test"]["metrics"]["observations"] < coarse["segments"][0]["test"]["metrics"]["observations"]
    assert coarse["test_size"] == 10 and fine["test_size"] == 5
    # The two segmentations really describe different windows.
    assert [segment["test_start"] for segment in fine["segments"]] != [
        segment["test_start"] for segment in coarse["segments"]
    ]


@pytest.mark.slow
def test_walk_forward_absent_without_the_request_key(world, tmp_path: Path) -> None:
    """Disabled path: no key requested, no section written, no re-run."""

    candidate = StrategyCandidate(
        name="parity_wf_absent",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    request = world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio_wf_absent.parquet")
    request_path, result_path = tmp_path / "request_wf_absent.json", tmp_path / "result_wf_absent.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload
    assert "audit_report" not in payload
    assert "walk_forward" not in request


@pytest.mark.parametrize(
    "start",
    ["2023-12-25", "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29"],
)
@pytest.mark.slow
def test_walk_forward_year_with_one_trading_day_matches_the_local_engine(world, tmp_path: Path, start: str) -> None:
    """A calendar year holding a single trading day must not split the two ends.

    The worker computes each year's excess standard deviation inline while the
    local engine reaches every year through ``_relative_performance``, whose
    ``len(excess) > 1`` guard reports 0.0 for one observation.  Unguarded,
    pandas returns NaN for that year on the worker side, ``_json_safe`` turns
    it into ``null``, and ``metrics_by_year[year]["tracking_error"]`` diverges
    between the engines -- reachable as soon as a window straddles a year
    boundary with a one-day year inside it.  Several calendar variants are
    checked so the guard cannot pass by accident of where the boundary lands.
    """

    # Exactly one trading day in the earlier calendar year and 39 in the next.
    # ``start`` walks the first regular session across the week, so the
    # boundary between the two years moves with it and the guard cannot pass
    # by accident of where the split lands.
    single_year_day = pd.Timestamp("2023-12-29")
    dates = pd.DatetimeIndex(
        [single_year_day, *pd.bdate_range(pd.Timestamp(start) + pd.offsets.Day(5), periods=39)]
    )
    per_year: dict[str, int] = {}
    for ts in dates:
        per_year[str(ts.year)] = per_year.get(str(ts.year), 0) + 1
    short_year = str(single_year_day.year)
    assert per_year[short_year] == 1, per_year
    assert len(dates) == 40 and dates.is_monotonic_increasing, dates

    original_eval_dates, original_price_dates = EVAL_DATES, PRICE_DATES
    globals()["EVAL_DATES"], globals()["PRICE_DATES"] = dates, pd.bdate_range("2023-12-01", periods=90)
    try:
        local_root = tmp_path / "local"
        local_root.mkdir()
        local_world = _build_world(local_root, _world_prices())
    finally:
        globals()["EVAL_DATES"], globals()["PRICE_DATES"] = original_eval_dates, original_price_dates

    candidate = StrategyCandidate(
        name=f"parity_wf_year_{short_year}",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    request = local_world["evaluator"].request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_wf_year_{start}.parquet")
    request_path = tmp_path / f"request_wf_year_{start}.json"
    result_path = tmp_path / f"result_wf_year_{start}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload

    local = _local_portfolio(local_world, request, candidate.composition, candidate.signal, backtest=candidate.backtest)
    worker_year = payload["metrics_by_year"][short_year]
    local_year = local.metrics_by_year[short_year]
    assert worker_year["observations"] == 1, worker_year
    # The guard's whole point: the same year must read as the same number on
    # both ends, and that number is the declared degenerate 0.0 -- not null.
    assert worker_year["tracking_error"] == local_year["tracking_error"] == 0.0
    assert worker_year["tracking_error"] is not None
    assert worker_year["information_ratio"] == local_year["information_ratio"] == 0.0
    # A multi-day year in the same report still reports a real standard
    # deviation, so the guard cannot be satisfied by zeroing every year.
    other_year = max(per_year)
    assert local.metrics_by_year[other_year]["tracking_error"] > 0.0
    assert payload["metrics_by_year"][other_year]["tracking_error"] == pytest.approx(
        local.metrics_by_year[other_year]["tracking_error"], rel=1e-9, abs=1e-12
    )


@pytest.mark.slow
def test_walk_forward_transport_refuses_a_non_object_config(world) -> None:
    """``request_for`` must not coerce, and must say what the validator says.

    ``dict([])`` quietly becomes ``{}``, ``dict("abc")`` raises a sequence
    ValueError and ``dict(7)`` a TypeError: three unrelated surfaces for one
    caller mistake, none of them the message the worker's validator raises.
    The guard makes the request-level contract "same message on both ends"
    hold before the request is ever serialised.
    """

    candidate = StrategyCandidate(
        name="parity_wf_transport",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    for value in ([], "abc", 7, (), set()):
        with pytest.raises(ValueError, match=re.escape("request walk_forward must be an object")):
            world["evaluator"].request_for(candidate, walk_forward=value)
    # None still means "this evaluation is off" and writes no key at all.
    request = world["evaluator"].request_for(candidate, walk_forward=None)
    assert "walk_forward" not in request
    # A Mapping -- not just a dict -- is accepted.
    request = world["evaluator"].request_for(candidate, walk_forward=MappingProxyType({"train_size": 20, "test_size": 5}))
    assert request["walk_forward"] == {"train_size": 20, "test_size": 5}
    # The same guard covers the audit object, which had the identical hole.
    for value in ([], "abc", 7):
        with pytest.raises(ValueError, match=re.escape("request audit must be an object")):
            world["evaluator"].request_for(candidate, audit=value)
    request = world["evaluator"].request_for(candidate, audit=MappingProxyType({"concentration": {}}))
    assert request["audit"] == {"concentration": {}}


@pytest.mark.parametrize(
    ("walk_forward", "message"),
    [
        ([], "request walk_forward must be an object"),
        ({"mystery": 1, "train_size": 5, "test_size": 5}, "request walk_forward has unknown keys: ['mystery']"),
        ({"test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": 5}, "walk_forward test_size must be a positive integer"),
        ({"train_size": 0, "test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": True, "test_size": 5}, "walk_forward train_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "valid_size": 0}, "walk_forward valid_size must be a positive integer"),
        ({"train_size": 5, "test_size": 5, "step_size": 0}, "walk_forward step_size must be a positive integer"),
        (
            {"train_size": 5, "test_size": 5, "min_oos_segments": 0},
            "walk_forward min_oos_segments must be a positive integer",
        ),
        ({"train_size": 5, "test_size": 5, "max_segments": 0}, "walk_forward max_segments must be a positive integer"),
        # Roadmap 10.1 lists no warm-up term, and this key used to be accepted,
        # echoed into the report and read by nothing.  Unknown-key refusal on
        # both ends is the fail-closed replacement (see D1 in
        # temp/forge_batch8fix/).
        (
            {"train_size": 20, "test_size": 5, "warmup_days": 3},
            "request walk_forward has unknown keys: ['warmup_days']",
        ),
        (
            {"train_size": 5, "test_size": 5, "expanding_train": "yes"},
            "walk_forward expanding_train must be a boolean",
        ),
    ],
)
def test_walk_forward_config_failures_fail_closed_on_both_sides(world, walk_forward, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_walk_forward_config(walk_forward)
    with pytest.raises(ValueError, match=re.escape(message)):
        _validate_walk_forward(walk_forward)


@pytest.mark.slow
def test_walk_forward_fails_closed_end_to_end(world, tmp_path: Path) -> None:
    """Execution-time refusals: too few dates, too many or too few segments."""

    candidate = StrategyCandidate(
        name="parity_wf_closed",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    cases = (
        ({"train_size": 50, "valid_size": 10, "test_size": 10}, "Need at least 70 dates, got 65"),
        (
            {"train_size": 5, "valid_size": 2, "test_size": 2, "step_size": 1, "max_segments": 3},
            "walk_forward would build 57 segments, more than max_segments 3",
        ),
        (
            {"train_size": 20, "valid_size": 5, "test_size": 5, "min_oos_segments": 20},
            "walk_forward produced 8 OOS segments, fewer than min_oos_segments 20",
        ),
    )
    for index, (walk_forward, message) in enumerate(cases):
        request = world["evaluator"].request_for(candidate, walk_forward=walk_forward)
        request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_wf_closed_{index}.parquet")
        request_path = tmp_path / f"request_wf_closed_{index}.json"
        result_path = tmp_path / f"result_wf_closed_{index}.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape(message)):
            run_request(request_path, result_path)
        # The local side raises the same message from the same splitter.
        with pytest.raises(ValueError, match=re.escape(message)):
            _local_walk_forward(world, request, candidate)


@pytest.mark.slow
def test_walk_forward_leaves_the_executed_book_and_metrics_untouched(world, tmp_path: Path) -> None:
    """The report is additional: the base run must be bit-identical with it."""

    candidate = StrategyCandidate(
        name="parity_wf_untouched",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    plain = world["evaluator"].request_for(candidate)
    plain["portfolio_artifact_path"] = str(tmp_path / "portfolio_wf_plain.parquet")
    plain_path, plain_result = tmp_path / "request_wf_plain.json", tmp_path / "result_wf_plain.json"
    plain_path.write_text(json.dumps(plain), encoding="utf-8")
    baseline = run_request(plain_path, plain_result)
    assert baseline["status"] == "success", baseline

    request, segmented = _run_worker_with_walk_forward(
        world,
        tmp_path,
        candidate,
        {"train_size": 20, "valid_size": 5, "test_size": 5},
        "wf_untouched_segmented",
    )
    assert segmented["metrics"] == baseline["metrics"]
    assert segmented["return_profile"] == baseline["return_profile"]
    assert segmented["factor_contributions"] == baseline["factor_contributions"]

    plain_weights = pd.read_parquet(plain["portfolio_artifact_path"]).to_csv(float_format="%.17g")
    segmented_weights = pd.read_parquet(request["portfolio_artifact_path"]).to_csv(float_format="%.17g")
    assert sha256(plain_weights.encode()).hexdigest() == sha256(segmented_weights.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Batch 9 (governance): StrategyManifestEmitter (10.8)
#
# ReproducibilityCheck (10.9) and DeploymentGate (10.10) have no request
# surface and no worker counterpart at all, so they have nothing to mirror and
# nothing to compare here -- their coverage lives in
# test_strategy_reproducibility.py and test_strategy_deployment.py.  The
# manifest switch is the one governance object that does cross the wire, so it
# is the one that gets a two-side section.
# ---------------------------------------------------------------------------


def _run_worker_with_manifest(world, tmp_path: Path, candidate, manifest, tag: str):
    request = world["evaluator"].request_for(candidate, manifest=manifest)
    request["portfolio_artifact_path"] = str(tmp_path / f"portfolio_{tag}.parquet")
    request_path, result_path = tmp_path / f"request_{tag}.json", tmp_path / f"result_{tag}.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    assert payload["status"] == "success", payload
    return request, payload


@pytest.mark.slow
def test_manifest_section_is_filled_only_when_requested(world, tmp_path: Path) -> None:
    """Two states: the switch on yields the attestation, the switch off adds nothing."""

    candidate = StrategyCandidate(
        name="parity_manifest_states",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )

    plain = world["evaluator"].request_for(candidate)
    plain["portfolio_artifact_path"] = str(tmp_path / "portfolio_manifest_off.parquet")
    plain_path, plain_result = tmp_path / "request_manifest_off.json", tmp_path / "result_manifest_off.json"
    plain_path.write_text(json.dumps(plain), encoding="utf-8")
    baseline = run_request(plain_path, plain_result)
    assert baseline["status"] == "success", baseline
    assert "audit_report" not in baseline
    assert "manifest" not in plain

    _, attested = _run_worker_with_manifest(world, tmp_path, candidate, {}, "manifest_on")
    section = attested["audit_report"]["manifest"]
    assert isinstance(section, dict)
    assert section["worker_file_sha256"] == _worker_file_digest(Path(gf_quant_strategy_worker.__file__).resolve())
    assert set(section["worker_runtime"]) == {"python", "numpy", "pandas", "pyarrow"}
    assert section["trading_calendar"]["rows"] > 0
    assert section["request_checksum"]


@pytest.mark.slow
def test_manifest_section_is_a_dict_and_flows_into_the_audit_report(world, tmp_path: Path) -> None:
    """The result adapter coerces each section with ``dict(value)``, so it must be an object."""

    candidate = StrategyCandidate(
        name="parity_manifest_shape",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    request, payload = _run_worker_with_manifest(world, tmp_path, candidate, {}, "manifest_shape")
    assert isinstance(payload["audit_report"], dict)
    assert isinstance(payload["audit_report"]["manifest"], dict)

    evaluation = RemoteStrategyEvaluator._result_from_payload(candidate, request, payload)
    assert evaluation.audit_report is not None
    assert isinstance(evaluation.audit_report["manifest"], dict)


@pytest.mark.slow
def test_manifest_leaves_the_executed_book_and_metrics_untouched(world, tmp_path: Path) -> None:
    """The manifest is additional: the base run must be bit-identical with it."""

    candidate = StrategyCandidate(
        name="parity_manifest_untouched",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    plain = world["evaluator"].request_for(candidate)
    plain["portfolio_artifact_path"] = str(tmp_path / "portfolio_manifest_base.parquet")
    plain_path, plain_result = tmp_path / "request_manifest_base.json", tmp_path / "result_manifest_base.json"
    plain_path.write_text(json.dumps(plain), encoding="utf-8")
    baseline = run_request(plain_path, plain_result)
    assert baseline["status"] == "success", baseline

    request, attested = _run_worker_with_manifest(world, tmp_path, candidate, {}, "manifest_untouched")
    assert attested["metrics"] == baseline["metrics"]
    assert attested["return_profile"] == baseline["return_profile"]
    assert attested["factor_contributions"] == baseline["factor_contributions"]

    base_weights = pd.read_parquet(plain["portfolio_artifact_path"]).to_csv(float_format="%.17g")
    attested_weights = pd.read_parquet(request["portfolio_artifact_path"]).to_csv(float_format="%.17g")
    assert sha256(base_weights.encode()).hexdigest() == sha256(attested_weights.encode()).hexdigest()


@pytest.mark.slow
def test_manifest_attestation_tracks_the_worker_source_and_the_request(world, tmp_path: Path) -> None:
    """The digest names the file that ran; the checksum names the request it ran on."""

    from rdagent.scenarios.gf_quant.strategy.remote import _canonical_json

    candidate = StrategyCandidate(
        name="parity_manifest_identity",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    request, payload = _run_worker_with_manifest(world, tmp_path, candidate, {}, "manifest_identity")
    section = payload["audit_report"]["manifest"]
    worker_source = Path(gf_quant_strategy_worker.__file__).resolve()
    assert section["worker_file_sha256"] == _worker_file_digest(worker_source)
    assert section["worker_file_name"] == worker_source.name
    # The result's own request checksum and the attestation's must agree: both
    # are taken over the same request object in the same run.
    import hashlib

    assert section["request_checksum"] == hashlib.sha256(_canonical_json(request).encode()).hexdigest()
    assert section["request_checksum"] == payload["request_checksum"]


@pytest.mark.parametrize(
    "manifest",
    [
        {"schema_version": 1},
        {"operator_version": "x"},
        {"a": None, "b": None},
    ],
)
def test_manifest_config_failures_fail_closed_on_both_sides(tmp_path: Path, manifest: dict) -> None:
    """Same message on both ends, and the transport layer refuses it first."""

    from rdagent.scenarios.gf_quant.strategy.manifest import validate_manifest_config
    from scripts.gf_quant_strategy_worker import _validate_manifest

    with pytest.raises(ValueError) as local:
        validate_manifest_config(manifest)
    with pytest.raises(ValueError) as worker:
        _validate_manifest(manifest)
    assert str(local.value) == str(worker.value)
    assert "request manifest has unknown keys" in str(local.value)


def test_manifest_transport_refuses_a_non_object(tmp_path: Path, world) -> None:
    """``request_for`` guards the section before it is coerced, like audit/walk_forward."""

    candidate = StrategyCandidate(
        name="parity_manifest_transport",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal"),
        factor_names=FACTOR_NAMES,
        rule="parity fixture",
    )
    for value in ([], 7, "manifest"):
        with pytest.raises(ValueError) as excinfo:
            world["evaluator"].request_for(candidate, manifest=value)
        assert str(excinfo.value) == "request manifest must be an object"


def test_manifest_keyset_matches_the_worker() -> None:
    """The switch is the empty-object contract on both ends."""

    from rdagent.scenarios.gf_quant.strategy.manifest import MANIFEST_KEYS

    assert MANIFEST_KEYS == gf_quant_strategy_worker.MANIFEST_KEYS == frozenset()
