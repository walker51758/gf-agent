from __future__ import annotations

import json
from types import SimpleNamespace

from rdagent.app.gf_rd_loop.conf import GFFactorPropSetting
from rdagent.core.proposal import Hypothesis, Trace
from rdagent.scenarios.gf_quant.proposal import GFFactorHypothesis2Experiment, GFFactorHypothesisGen
from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.search_policy import (
    GFSearchPolicy,
    collect_tested_factors,
    infer_search_direction,
)


class _Scenario:
    def get_scenario_all_desc(self, filtered_tag=None):  # type: ignore[no-untyped-def]
        return "GF SCENARIO"


def _hypothesis(text: str = "SECRET_HYPOTHESIS") -> Hypothesis:
    return Hypothesis(text, "SECRET_REASON", "", "", "", "")


def _experiment(direction: str, index: int = 0):  # type: ignore[no-untyped-def]
    task = SimpleNamespace(
        factor_name=f"factor_{direction}_{index}",
        factor_description=f"[{direction}] description",
        factor_formulation="-pct_change(adjclose, 20)" if direction == "reversal" else "pct_change(adjclose, 20)",
        search_direction=direction,
    )
    return SimpleNamespace(
        hypothesis=_hypothesis(),
        sub_tasks=[task],
        result=None,
        stdout="SECRET_STDOUT",
        search_direction=direction,
    )


def _feedback(decision: bool):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        decision=decision,
        observations="SECRET_OBSERVATION",
        hypothesis_evaluation="SECRET_EVALUATION",
        new_hypothesis="SECRET_NEW_HYPOTHESIS",
        reason="SECRET_FEEDBACK_REASON",
    )


def _trace(directions: list[str], decisions: list[bool] | None = None) -> Trace:
    trace = Trace(_Scenario())
    decisions = decisions or [True] * len(directions)
    trace.hist = [(_experiment(direction, i), _feedback(decisions[i])) for i, direction in enumerate(directions)]
    return trace


def _record(name: str, expression: str, direction: str) -> FactorRecord:
    return FactorRecord(
        factor_id=name,
        name=name,
        family="fundamental" if direction == "fundamental" else "price_volume",
        expression=expression,
        status="accepted",
        metadata={"search_direction": direction},
    )


def test_search_setting_defaults_match_documented_policy() -> None:
    settings = GFFactorPropSetting(_env_file=None)
    assert settings.exploration_probability == 0.30
    assert settings.max_same_family_streak == 6
    assert settings.stagnation_patience == 3
    assert settings.search_random_seed is None


def test_probability_selects_improve_or_explore_deterministically() -> None:
    trace = _trace(["reversal"])

    improve = GFSearchPolicy(
        exploration_probability=0.0,
        max_same_family_streak=99,
        stagnation_patience=99,
        random_seed=1,
    ).decide(trace)
    explore = GFSearchPolicy(
        exploration_probability=1.0,
        max_same_family_streak=99,
        stagnation_patience=99,
        random_seed=1,
    ).decide(trace)

    assert (improve.mode, improve.direction, improve.reason) == (
        "improve",
        "reversal",
        "continue_best_direction",
    )
    assert explore.mode == "explore"
    assert explore.reason == "random_exploration"
    assert explore.direction != "reversal"


def test_failed_exploration_does_not_replace_the_improvement_mainline() -> None:
    trace = _trace(["reversal", "fundamental"], [True, False])
    decision = GFSearchPolicy(
        exploration_probability=0.0,
        max_same_family_streak=99,
        stagnation_patience=99,
        random_seed=1,
    ).decide(trace)

    assert decision.mode == "improve"
    assert decision.direction == "reversal"


def test_same_direction_and_stagnation_each_force_exploration() -> None:
    same = GFSearchPolicy(
        exploration_probability=0.0,
        max_same_family_streak=3,
        stagnation_patience=99,
        random_seed=2,
    ).decide(_trace(["reversal", "reversal", "reversal"]))
    stagnant = GFSearchPolicy(
        exploration_probability=0.0,
        max_same_family_streak=99,
        stagnation_patience=3,
        random_seed=2,
    ).decide(_trace(["reversal", "momentum", "liquidity"], [False, False, False]))

    assert same.mode == "explore"
    assert same.reason == "same_direction_limit"
    assert same.same_direction_streak == 3
    assert stagnant.mode == "explore"
    assert stagnant.reason == "stagnation_limit"
    assert stagnant.stagnation_streak == 3


def test_exploration_chooses_the_only_least_tested_direction() -> None:
    records = [
        _record("r", "-pct_change(adjclose, 5)", "reversal"),
        _record("m", "pct_change(adjclose, 20)", "momentum"),
        _record("l", "cs_rank(turnover)", "liquidity"),
        _record("v", "ts_std(adjclose, 20)", "volatility"),
    ]
    decision = GFSearchPolicy(exploration_probability=1.0, random_seed=7).decide(
        _trace(["reversal"]), records
    )

    assert decision.direction == "fundamental"
    assert decision.direction_counts["fundamental"] == 0


def test_legacy_direction_inference_covers_main_families() -> None:
    assert infer_search_direction("short-term reversal", "-pct_change(adjclose, 10)") == "reversal"
    assert infer_search_direction("trend strength", "pct_change(adjclose, 20)") == "momentum"
    assert infer_search_direction("turnover shock", "cs_rank(turnover)") == "liquidity"
    assert infer_search_direction("realized volatility", "ts_std(adjclose, 20)") == "volatility"
    assert infer_search_direction("book value", "cs_rank(balance__parent_equity / mv)") == "fundamental"

    legacy_trace = _trace([])
    delattr_exp = _experiment("reversal")
    del delattr_exp.search_direction
    del delattr_exp.sub_tasks[0].search_direction
    legacy_trace.hist = [(delattr_exp, _feedback(True))]
    assert collect_tested_factors(legacy_trace)[0].direction == "reversal"


def test_exploration_context_is_metric_and_feedback_free_at_both_llm_stages(tmp_path) -> None:
    trace = _trace(["reversal"])
    registry = FactorRegistry(tmp_path / "registry.sqlite")
    registry.upsert(_record("registered_reversal", "-pct_change(adjclose, 10)", "reversal"))
    policy = GFSearchPolicy(
        exploration_probability=1.0,
        max_same_family_streak=99,
        stagnation_patience=99,
        random_seed=4,
    )

    generator = GFFactorHypothesisGen(_Scenario(), search_policy=policy, registry=registry)
    hypothesis_context, _ = generator.prepare_context(trace)
    hypothesis = generator.convert_response(json.dumps({"hypothesis": "Independent idea", "reason": "test"}))
    converter = GFFactorHypothesis2Experiment(registry=registry)
    experiment_context, _ = converter.prepare_context(hypothesis, trace)

    joined = json.dumps([hypothesis_context, experiment_context], ensure_ascii=False)
    assert hypothesis.search_mode == "explore"
    assert "registered_reversal" in joined
    assert "factor_reversal_0" in joined
    for secret in (
        "SECRET_HYPOTHESIS",
        "SECRET_REASON",
        "SECRET_OBSERVATION",
        "SECRET_EVALUATION",
        "SECRET_NEW_HYPOTHESIS",
        "SECRET_FEEDBACK_REASON",
        "SECRET_STDOUT",
    ):
        assert secret not in joined


def test_improvement_context_preserves_trace_and_metadata_propagates(tmp_path) -> None:
    trace = _trace(["reversal"])
    registry = FactorRegistry(tmp_path / "registry.sqlite")
    policy = GFSearchPolicy(
        exploration_probability=0.0,
        max_same_family_streak=99,
        stagnation_patience=99,
        random_seed=1,
    )
    generator = GFFactorHypothesisGen(_Scenario(), search_policy=policy, registry=registry)
    hypothesis_context, _ = generator.prepare_context(trace)
    hypothesis = generator.convert_response(json.dumps({"hypothesis": "Refine reversal", "reason": "test"}))
    converter = GFFactorHypothesis2Experiment(registry=registry)
    experiment_context, _ = converter.prepare_context(hypothesis, trace)

    assert hypothesis.search_mode == "improve"
    assert hypothesis.search_direction == "reversal"
    assert "SECRET_HYPOTHESIS" in hypothesis_context["hypothesis_and_feedback"]
    assert "SECRET_OBSERVATION" in hypothesis_context["hypothesis_and_feedback"]
    assert "SECRET_HYPOTHESIS" in experiment_context["hypothesis_and_feedback"]

    response = json.dumps(
        {
            "refined_reversal": {
                "description": "[Reversal] refined window",
                "family": "price_volume",
                "formulation": "-pct_change(adjclose, 15)",
                "variables": {"adjclose": "adjusted close"},
            }
        }
    )
    experiment = converter.convert_response(response, hypothesis, trace)
    assert experiment.search_mode == "improve"
    assert experiment.search_direction == "reversal"
    assert experiment.sub_tasks[0].search_mode == "improve"
    assert experiment.sub_tasks[0].search_direction == "reversal"
