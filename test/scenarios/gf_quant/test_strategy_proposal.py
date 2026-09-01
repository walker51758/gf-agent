from __future__ import annotations

import json

import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
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
        "cost_bps": 10,
    }


def test_llm_strategy_proposal_uses_strict_frozen_contract(tmp_path) -> None:
    backend = FakeBackend(_valid_payload())
    context = _context(tmp_path)

    candidate = LLMStrategyProposalEngine(backend)(context)

    assert candidate.factor_names == ("value", "quality")
    assert candidate.composition.top_n == 50
    assert candidate.backtest.cost_bps == 10
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
        ({"cost_bps": 0}, "cost_bps"),
        ({"signal_lag": 0}, "look-ahead"),
    ],
)
def test_llm_strategy_proposal_fails_closed(tmp_path, update, message) -> None:
    payload = _valid_payload()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        LLMStrategyProposalEngine(FakeBackend(payload))(_context(tmp_path))
