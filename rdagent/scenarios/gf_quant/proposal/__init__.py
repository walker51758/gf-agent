"""GF proposal classes that require executable remote-DSL formulations."""

import json

from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
    QlibFactorHypothesisGen,
)
from rdagent.utils.agent.tpl import T

__all__ = ["GFFactorHypothesisGen", "GFFactorHypothesis2Experiment"]


class GFFactorHypothesisGen(QlibFactorHypothesisGen):
    """Generate hypotheses constrained to the fields available on ``G:/base``."""

    def prepare_context(self, trace):  # type: ignore[no-untyped-def]
        context, json_mode = super().prepare_context(trace)
        context["hypothesis_specification"] = T(
            "scenarios.gf_quant.experiment.prompts:gf_factor_hypothesis_specification"
        ).r()
        return context, json_mode


class GFFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    """Convert hypotheses into named safe-DSL factor tasks."""

    def prepare_context(self, hypothesis, trace):  # type: ignore[no-untyped-def]
        context, json_mode = super().prepare_context(hypothesis, trace)
        context["experiment_output_format"] = T(
            "scenarios.gf_quant.experiment.prompts:gf_factor_experiment_output_format"
        ).r()
        return context, json_mode

    def convert_response(self, response, hypothesis, trace):  # type: ignore[no-untyped-def]
        payload = json.loads(response)
        exp = super().convert_response(response, hypothesis, trace)
        for task in exp.sub_tasks:
            item = payload.get(task.factor_name, {})
            declared = item.get("family") if isinstance(item, dict) else None
            if declared not in {"price_volume", "fundamental"}:
                declared = "fundamental" if "__" in str(task.factor_formulation) else "price_volume"
            task.gf_family = declared
        return exp
