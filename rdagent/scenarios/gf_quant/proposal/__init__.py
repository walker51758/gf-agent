"""GF proposal classes that require executable remote-DSL formulations."""

import json
from typing import Any

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.search_policy import (
    GFSearchPolicy,
    SearchDecision,
    collect_tested_factors,
    format_tested_archive,
)
from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
    QlibFactorHypothesisGen,
)
from rdagent.utils.agent.tpl import T

__all__ = ["GFFactorHypothesisGen", "GFFactorHypothesis2Experiment"]


def _registry_records(registry: FactorRegistry | None) -> list[Any]:
    """Read the audit archive without making registry availability a loop blocker."""

    try:
        return (registry or FactorRegistry(GF_SCENARIO_SETTINGS.registry_path)).list()
    except Exception as exc:  # pragma: no cover - defensive path for damaged external DBs
        logger.warning(f"GF search archive is unavailable; continuing with trace only: {exc}")
        return []


def _search_instruction(decision: SearchDecision) -> str:
    return T("scenarios.gf_quant.experiment.prompts:gf_factor_search_instruction").r(
        search_mode=decision.mode,
        search_direction=decision.direction,
        search_reason=decision.reason,
    )


class GFFactorHypothesisGen(QlibFactorHypothesisGen):
    """Generate hypotheses constrained to the fields available on ``G:/base``."""

    def __init__(
        self,
        scen,
        *,
        search_policy: GFSearchPolicy | None = None,
        registry: FactorRegistry | None = None,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(scen)
        if search_policy is None:
            # Local import avoids making the reusable GF scenario depend on the
            # application settings while that settings module is initialising.
            from rdagent.app.gf_rd_loop.conf import GF_FACTOR_PROP_SETTING

            search_policy = GFSearchPolicy(
                exploration_probability=GF_FACTOR_PROP_SETTING.exploration_probability,
                max_same_family_streak=GF_FACTOR_PROP_SETTING.max_same_family_streak,
                stagnation_patience=GF_FACTOR_PROP_SETTING.stagnation_patience,
                random_seed=GF_FACTOR_PROP_SETTING.search_random_seed,
            )
        self.search_policy = search_policy
        self.registry = registry
        self._active_search_decision: SearchDecision | None = None

    def prepare_context(self, trace):  # type: ignore[no-untyped-def]
        records = _registry_records(self.registry)
        decision = self.search_policy.decide(trace, records)
        self._active_search_decision = decision
        logger.log_object(decision.as_log_dict(), tag="search policy")

        instruction = _search_instruction(decision)
        specification = T("scenarios.gf_quant.experiment.prompts:gf_factor_hypothesis_specification").r()
        if decision.mode == "explore":
            # Deliberately do not call the Qlib parent here: it renders the
            # complete historical hypotheses, metrics and feedback twice.
            factors = collect_tested_factors(trace, records)
            context = {
                "hypothesis_and_feedback": format_tested_archive(factors),
                "last_hypothesis_and_feedback": "",
                "RAG": instruction,
                "hypothesis_output_format": T(
                    "scenarios.qlib.prompts:factor_hypothesis_output_format"
                ).r(),
                "hypothesis_specification": specification + "\n\n" + instruction,
            }
            return context, True

        context, json_mode = super().prepare_context(trace)
        prior_rag = context.get("RAG") or ""
        context["RAG"] = (prior_rag + "\n\n" + instruction).strip()
        context["hypothesis_specification"] = specification + "\n\n" + instruction
        return context, json_mode

    def convert_response(self, response):  # type: ignore[no-untyped-def]
        hypothesis = super().convert_response(response)
        decision = self._active_search_decision
        if decision is None:  # direct test/debug calls that bypass ``gen``
            decision = SearchDecision("improve", "other", "missing_prepare_context", 0, 0, {})
        hypothesis.search_mode = decision.mode
        hypothesis.search_direction = decision.direction
        hypothesis.search_reason = decision.reason
        hypothesis.search_policy = decision.as_log_dict()
        return hypothesis


class GFFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    """Convert hypotheses into named safe-DSL factor tasks."""

    def __init__(self, *, registry: FactorRegistry | None = None) -> None:
        super().__init__()
        self.registry = registry

    def prepare_context(self, hypothesis, trace):  # type: ignore[no-untyped-def]
        mode = getattr(hypothesis, "search_mode", "improve")
        direction = getattr(hypothesis, "search_direction", "other")
        reason = getattr(hypothesis, "search_reason", "legacy_hypothesis")
        decision = SearchDecision(mode, direction, reason, 0, 0, {})
        instruction = _search_instruction(decision)
        output_format = T("scenarios.gf_quant.experiment.prompts:gf_factor_experiment_output_format").r()

        if mode == "explore":
            records = _registry_records(self.registry)
            factors = collect_tested_factors(trace, records)
            # This is the second isolation boundary.  Only the current
            # hypothesis and a metric-free dedup archive reach the LLM.
            return {
                "target_hypothesis": str(hypothesis) + "\n\n" + instruction,
                "scenario": trace.scen.get_scenario_all_desc(),
                "hypothesis_and_feedback": format_tested_archive(factors),
                "experiment_output_format": output_format + "\n\n" + instruction,
                "target_list": [],
                "RAG": None,
            }, True

        context, json_mode = super().prepare_context(hypothesis, trace)
        context["target_hypothesis"] = context["target_hypothesis"] + "\n\n" + instruction
        context["experiment_output_format"] = output_format + "\n\n" + instruction
        return context, json_mode

    def convert_response(self, response, hypothesis, trace):  # type: ignore[no-untyped-def]
        payload = json.loads(response)
        exp = super().convert_response(response, hypothesis, trace)
        mode = getattr(hypothesis, "search_mode", "improve")
        direction = getattr(hypothesis, "search_direction", "other")
        exp.search_mode = mode
        exp.search_direction = direction
        for task in exp.sub_tasks:
            item = payload.get(task.factor_name, {})
            declared = item.get("family") if isinstance(item, dict) else None
            if declared not in {"price_volume", "fundamental"}:
                declared = "fundamental" if "__" in str(task.factor_formulation) else "price_volume"
            task.gf_family = declared
            task.search_mode = mode
            task.search_direction = direction
        return exp
