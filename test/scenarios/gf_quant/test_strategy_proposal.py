from __future__ import annotations

import json

import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import (
    BacktestSpec,
    CompositionSpec,
    GateLayer,
    GatingSpec,
    SignalSpec,
)
from rdagent.scenarios.gf_quant.strategy.orchestrator import StrategyProposalContext, TrialOutcomeSummary
from rdagent.scenarios.gf_quant.strategy.proposal import LLMStrategyProposalEngine
from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def build_messages_and_create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.payload)


def _context(tmp_path) -> StrategyProposalContext:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    for position, name in enumerate(("value", "quality", "cashflow")):
        registry.upsert(
            FactorRecord(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                status="accepted",
                metrics={"Rank_IC": 0.02 + position * 0.01, "sharpe": 0.5},
                artifact_path=f"G:/artifacts/{name}.parquet",
                artifact_checksum=(str(position + 1) * 64),
                metadata={"search_direction": name},
            )
        )
    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")
    working = build_factor_working_set(snapshot, core_limit=3, conditional_limit=0)
    return StrategyProposalContext(
        epoch_id="epoch_001",
        trial_index=2,
        data_cutoff="2023-12-31",
        snapshot_id=snapshot.snapshot_id,
        working_set=working,
        prior_outcomes=(
            TrialOutcomeSummary(1, "old", "rejected", ("sharpe",), {"sharpe": -0.1}),
        ),
    )


def _valid_payload():
    return {
        "name": "Value + Quality",
        "hypothesis": "Cash-generative value with a quality filter should diversify valuation traps.",
        "factor_names": ["value", "quality"],
        "rule": "equal rank blend then top 50",
        "top_n": 50,
        "weighting": "equal",
        "weight_cap": 0.05,
        "signal_lag": 1,
    }


def test_llm_strategy_proposal_uses_strict_frozen_contract(tmp_path) -> None:
    backend = FakeBackend(_valid_payload())
    context = _context(tmp_path)

    candidate = LLMStrategyProposalEngine(backend)(context)

    assert candidate.factor_names == ("value", "quality")
    assert candidate.composition.top_n == 50
    assert "cost_bps" not in candidate.backtest.as_payload()
    assert candidate.name.endswith("_epoch001_t02")
    assert candidate.metadata["snapshot_id"] == context.snapshot_id
    assert backend.calls[0]["json_mode"] is True
    assert "old" in backend.calls[0]["user_prompt"]
    assert "cashflow" in backend.calls[0]["user_prompt"]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"factor_names": ["value", "invented"]}, "invented"),
        ({"factor_names": ["value", "value"]}, "duplicate"),
        ({"top_n": 500}, "top_n"),
        ({"signal_lag": 0}, "look-ahead"),
        ({"signal_standardize": "mad"}, "standardize must be one of"),
        ({"signal_winsorize_lower": 0.9, "signal_winsorize_upper": 0.1}, "0 <= lower < upper <= 1"),
        # Batch C2 keys travel raw into spec validation (fact 37): a
        # non-integer must fail closed with the canonical message, not be
        # silently truncated by int() on the proposal path.
        (
            {
                "signal_industry_neutralize": True,
                "signal_industry_neutralize_industry_map": {"S0": "X"},
                "signal_industry_neutralize_min_group": 2.5,
            },
            "industry_neutralize_min_group must be a whole number",
        ),
        (
            {
                "signal_style_neutralize": True,
                "signal_style_neutralize_exposures": {"size": {"S0": 1.0}},
                "signal_style_neutralize_min_samples": 3.7,
            },
            "style_neutralize_min_samples must be a whole number",
        ),
        (
            {
                "signal_regime_states": {"2024-01-02": "on"},
                "signal_regime_factor_groups": {"on": ["value"]},
                "signal_regime_lag": 1.9,
            },
            "regime_lag must be a whole number",
        ),
        (
            {"weighting": "risk_parity", "risk_parity_window": 30.7},
            "risk_parity_window must be a whole number",
        ),
        (
            {"weighting": "risk_parity", "risk_parity_window": "abc"},
            "risk_parity_window must be a whole number",
        ),
    ],
)
def test_llm_strategy_proposal_fails_closed(tmp_path, update, message) -> None:
    payload = _valid_payload()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        LLMStrategyProposalEngine(FakeBackend(payload))(_context(tmp_path))


def test_model_cost_quote_is_ignored_and_not_part_of_the_prompt_contract(tmp_path) -> None:
    backend = FakeBackend({**_valid_payload(), "cost_bps": 3})
    context = _context(tmp_path)

    candidate = LLMStrategyProposalEngine(backend)(context)

    assert "cost_bps" not in candidate.as_payload()["backtest"]
    prompt = backend.calls[0]["user_prompt"]
    assert "10 bps on buys and 10 bps on sells" in prompt
    assert "cost_bps:" not in prompt

    free = LLMStrategyProposalEngine(FakeBackend({**_valid_payload(), "cost_bps": 0}))(context)
    assert free == candidate


def test_llm_strategy_proposal_maps_the_signal_vocabulary(tmp_path) -> None:
    payload = {
        **_valid_payload(),
        "signal_transform": "signed_log",
        "signal_transform_scale": 2.0,
        "signal_transform_shift": 0.5,
        "signal_winsorize_lower": 0.05,
        "signal_winsorize_upper": 0.95,
        "signal_winsorize_min_samples": 10,
        "signal_standardize": "robust_z",
        "signal_standardize_min_std": 0.01,
    }
    backend = FakeBackend(payload)

    candidate = LLMStrategyProposalEngine(backend)(_context(tmp_path))

    assert candidate.signal == SignalSpec(
        transform="signed_log",
        transform_scale=2.0,
        transform_shift=0.5,
        winsorize_lower=0.05,
        winsorize_upper=0.95,
        winsorize_min_samples=10,
        standardize="robust_z",
        standardize_min_std=0.01,
    )
    assert "signal_standardize" in backend.calls[0]["user_prompt"]


def test_llm_strategy_proposal_defaults_to_a_disabled_signal(tmp_path) -> None:
    candidate = LLMStrategyProposalEngine(FakeBackend(_valid_payload()))(_context(tmp_path))
    assert candidate.signal == SignalSpec()


def test_llm_strategy_proposal_maps_the_gating_vocabulary(tmp_path) -> None:
    payload = {
        **_valid_payload(),
        "gating_layers": [
            {"factor": "value", "direction": "top", "quantile": 0.5},
            {"factor": "quality", "direction": "bottom", "quantile": 0.25, "min_pass": 3},
        ],
    }
    backend = FakeBackend(payload)

    candidate = LLMStrategyProposalEngine(backend)(_context(tmp_path))

    assert candidate.gating == GatingSpec(
        layers=(
            GateLayer(factor="value", direction="top", quantile=0.5),
            GateLayer(factor="quality", direction="bottom", quantile=0.25, min_pass=3),
        )
    )
    assert "gating_layers" in backend.calls[0]["user_prompt"]

    # An empty or null list stays the disabled default (no gating key in the
    # canonical payload at all).
    plain = LLMStrategyProposalEngine(FakeBackend({**_valid_payload(), "gating_layers": []}))(_context(tmp_path))
    assert plain.gating == GatingSpec()
    assert "gating" not in plain.as_payload()


def test_llm_strategy_proposal_maps_the_floor_and_score_transform_vocabulary(tmp_path) -> None:
    payload = {
        **_valid_payload(),
        "weighting": "score_transform",
        "score_transform_kind": "sigmoid",
        "score_transform_slope": 2.5,
        "score_transform_cut": 0.4,
        "weight_floor": 0.01,
        "weight_floor_allow_cash": False,
    }
    backend = FakeBackend(payload)

    candidate = LLMStrategyProposalEngine(backend)(_context(tmp_path))

    assert candidate.composition == CompositionSpec(
        top_n=50,
        weighting="score_transform",
        weight_cap=0.05,
        weight_floor=0.01,
        weight_floor_allow_cash=False,
        score_transform_kind="sigmoid",
        score_transform_slope=2.5,
        score_transform_cut=0.4,
    )
    prompt = backend.calls[0]["user_prompt"]
    assert "score_transform_kind" in prompt
    assert "weight_floor" in prompt

    # Absent vocabulary keys fall back to the disabled defaults.
    plain = LLMStrategyProposalEngine(FakeBackend(_valid_payload()))(_context(tmp_path))
    assert plain.composition == CompositionSpec(top_n=50, weighting="equal", weight_cap=0.05)
    assert "weight_floor" not in plain.as_payload()["composition"]
    assert "score_transform_kind" not in plain.as_payload()["composition"]


def test_llm_strategy_proposal_maps_the_execution_band_and_shrinkage_vocabulary(tmp_path) -> None:
    """The A4 execution operators travel the full vocabulary -> payload -> spec path.

    Guards the three-way key alignment (prompt wording, payload keys,
    ``BacktestSpec`` fields) the same way the gating and floor tests guard
    their batches: every execution vocabulary key must be offered to the
    model, map into the spec when the model picks it, and stay absent from
    the canonical payload while disabled.
    """

    execution_keys = (
        "drift_band_name",
        "drift_band_portfolio",
        "drift_band_forced_days",
        "turnover_shrinkage",
        "turnover_max_step",
    )
    payload = {
        **_valid_payload(),
        "drift_band_name": 0.05,
        "drift_band_portfolio": None,
        "drift_band_forced_days": 20,
        "turnover_shrinkage": 0.5,
        "turnover_max_step": None,
    }
    backend = FakeBackend(payload)

    candidate = LLMStrategyProposalEngine(backend)(_context(tmp_path))

    assert candidate.backtest == BacktestSpec(
        signal_lag=1,
        drift_band_name=0.05,
        drift_band_forced_days=20,
        turnover_shrinkage=0.5,
    )
    prompt = backend.calls[0]["user_prompt"]
    for key in execution_keys:
        assert key in prompt

    # The canonical payload carries exactly the enabled fields.
    backtest_payload = candidate.as_payload()["backtest"]
    assert set(backtest_payload) == {
        "signal_lag",
        "trading_days_per_year",
        "drift_band_name",
        "drift_band_forced_days",
        "turnover_shrinkage",
    }
    assert BacktestSpec.from_payload(backtest_payload) == candidate.backtest

    # Absent vocabulary keys fall back to the disabled defaults: no
    # execution-adjustment key in the payload at all.
    plain = LLMStrategyProposalEngine(FakeBackend(_valid_payload()))(_context(tmp_path))
    assert plain.backtest == BacktestSpec(signal_lag=1)
    for key in execution_keys:
        assert key not in plain.as_payload()["backtest"]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"weighting": "score_transform"}, "score_transform_kind is required"),
        ({"weighting": "score_transform", "score_transform_kind": "softmax"}, "score_transform_kind must be one of"),
        ({"score_transform_kind": "sigmoid"}, "only valid with the score_transform weighting"),
        ({"weight_floor": 0.5}, "weight_floor must not exceed weight_cap"),
        ({"weight_floor": 1.5}, "weight_floor must be in"),
    ],
)
def test_llm_strategy_proposal_floor_and_transform_fail_closed(tmp_path, update, message) -> None:
    payload = _valid_payload()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        LLMStrategyProposalEngine(FakeBackend(payload))(_context(tmp_path))


def test_llm_strategy_proposal_maps_the_signal_dag_vocabulary(tmp_path) -> None:
    """The B1 signal-DAG operators travel the vocabulary -> spec -> payload path."""

    payload = {
        **_valid_payload(),
        "signal_missing_value": "cross_section_median",
        "signal_missing_max_fill_ratio": 0.4,
        "signal_interaction_pairs": [["value", "quality", "product"]],
        "signal_blend_weights": {"value": 0.5, "quality": 0.3, "value*quality": 0.2},
    }
    backend = FakeBackend(payload)

    candidate = LLMStrategyProposalEngine(backend)(_context(tmp_path))

    assert candidate.signal == SignalSpec(
        missing_value="cross_section_median",
        missing_max_fill_ratio=0.4,
        interaction_pairs=(("value", "quality", "product"),),
        blend_weights={"value": 0.5, "quality": 0.3, "value*quality": 0.2},
    )
    prompt = backend.calls[0]["user_prompt"]
    for key in (
        "signal_missing_value",
        "signal_missing_max_fill_ratio",
        "signal_interaction_pairs",
        "signal_blend_weights",
        "signal_blend_groups",
    ):
        assert key in prompt

    signal_payload = candidate.as_payload()["signal"]
    assert set(signal_payload) == {
        "transform",
        "transform_scale",
        "transform_shift",
        "winsorize_lower",
        "winsorize_upper",
        "winsorize_min_samples",
        "standardize",
        "standardize_min_std",
        "missing_value",
        "missing_max_fill_ratio",
        "blend_weights",
        "interaction_pairs",
    }
    assert SignalSpec.from_payload(signal_payload) == candidate.signal

    # The group-blend vocabulary maps too (mutually exclusive with flat weights).
    grouped = {
        **_valid_payload(),
        "signal_blend_groups": {"g_value": ["value"], "g_quality": ["quality"]},
        "signal_blend_group_weights": {"g_value": 0.7, "g_quality": 0.3},
        "signal_blend_within_weights": {"g_quality": {"quality": 1.0}},
    }
    grouped_candidate = LLMStrategyProposalEngine(FakeBackend(grouped))(_context(tmp_path))
    assert grouped_candidate.signal.blend_groups == {"g_value": ["value"], "g_quality": ["quality"]}
    assert grouped_candidate.signal.blend_group_weights == {"g_value": 0.7, "g_quality": 0.3}

    # Absent vocabulary keys fall back to the disabled defaults, and the
    # canonical payload carries none of the new keys at all.
    plain = LLMStrategyProposalEngine(FakeBackend(_valid_payload()))(_context(tmp_path))
    assert plain.signal == SignalSpec()
    for key in (
        "missing_value",
        "missing_max_fill_ratio",
        "blend_weights",
        "blend_groups",
        "interaction_pairs",
    ):
        assert key not in plain.as_payload()["signal"]


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"signal_missing_value": "median"}, "missing_value must be one of"),
        ({"signal_missing_value": "drop_stock", "signal_missing_max_fill_ratio": 0.5}, "does not apply to the drop_stock policy"),
        ({"signal_missing_max_fill_ratio": 0.5}, "missing_max_fill_ratio requires a missing_value policy"),
        ({"signal_blend_weights": {"value": 0.5}}, "blend_weights must cover exactly the blend members"),
        (
            {
                "signal_blend_weights": {"value": 0.5, "quality": 0.5},
                "signal_blend_groups": {"g_value": ["value"], "g_quality": ["quality"]},
            },
            "mutually exclusive",
        ),
        ({"signal_interaction_pairs": [["value", "invented", "product"]]}, "interaction terms must reference candidate factors"),
        ({"signal_interaction_pairs": [["value", "quality", "tanh"]]}, "interaction kind must be one of"),
        # A pair that is not an exact [first, second, kind] triple is rejected
        # outright instead of being silently truncated to its first three
        # entries (batch B2 fix, same message on every side).
        (
            {"signal_interaction_pairs": [["value"]]},
            r"each interaction pair must be a \[first, second, kind\] triple",
        ),
        (
            {"signal_interaction_pairs": [["value", "quality", "product", "extra"]]},
            r"each interaction pair must be a \[first, second, kind\] triple",
        ),
        ({"signal_blend_weights": [1.0, 1.0]}, "blend_weights must not be empty"),
    ],
)
def test_llm_strategy_proposal_signal_dag_fails_closed(tmp_path, update, message) -> None:
    payload = _valid_payload()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        LLMStrategyProposalEngine(FakeBackend(payload))(_context(tmp_path))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"gating_layers": [{"factor": "invented", "direction": "top", "quantile": 0.5}]}, "gating layers must reference candidate factors"),
        ({"gating_layers": [{"factor": "value", "direction": "side", "quantile": 0.5}]}, "direction must be one of"),
        ({"gating_layers": [{"factor": "value", "direction": "top", "quantile": 1.0}]}, "strictly between 0 and 1"),
        ({"gating_layers": [{"factor": "value", "direction": "top"}]}, "missing the"),
        ({"gating_layers": "top half"}, "must be a list"),
    ],
)
def test_llm_strategy_proposal_gating_fails_closed(tmp_path, update, message) -> None:
    payload = _valid_payload()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        LLMStrategyProposalEngine(FakeBackend(payload))(_context(tmp_path))
