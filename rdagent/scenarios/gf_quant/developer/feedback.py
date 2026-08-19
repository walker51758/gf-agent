"""GF factor feedback exposing the full evaluation vector to the LLM."""

from __future__ import annotations

import json
from typing import Any

from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.gf_quant.metrics import format_factor_feedback
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T


def _mapping(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        data = result.to_dict()
        # pandas Series -> scalar mapping; a one-column DataFrame -> nested.
        if data and all(isinstance(value, dict) and len(value) == 1 for value in data.values()):
            return {key: next(iter(value.values())) for key, value in data.items()}
        return dict(data)
    return dict(result)


class GFFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp, trace) -> HypothesisFeedback:  # type: ignore[no-untyped-def]
        hypothesis = exp.hypothesis
        tasks = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        baseline = exp.based_experiments[-1].result if exp.based_experiments else None
        combined_result = format_factor_feedback(_mapping(exp.result), _mapping(baseline) if baseline is not None else None)

        scenario = self.scen.get_scenario_all_desc()
        system_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(scenario=scenario)
        user_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis.hypothesis,
            task_details=tasks,
            combined_result=combined_result,
        )
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=True,
            json_target_type=dict[str, str | bool | int],
        )
        payload = json.loads(response)
        return HypothesisFeedback(
            observations=payload.get("Observations", "No observations provided"),
            hypothesis_evaluation=payload.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=payload.get("New Hypothesis", "No new hypothesis provided"),
            reason=payload.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(payload.get("Replace Best Result", "no")),
        )
