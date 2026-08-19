"""
GF (GuangFa Securities) Quant extension.

A self-contained scenario on top of Microsoft RD-Agent that tailors the Qlib
quant factor research stack to GF Securities' Chinese A-share data.

* :class:`GFFactorScenario` - factor research scenario
* :class:`GFFactorHypothesisGen` / :class:`GFFactorHypothesis2Experiment` -
  proposal classes

The factor-loop entry point lives in
:mod:`rdagent.app.gf_rd_loop`.
"""

__all__ = [
    "GFFactorScenario",
    "GFFactorHypothesisGen",
    "GFFactorHypothesis2Experiment",
]


def __getattr__(name: str):
    """Keep lightweight registry/split imports independent of Qlib extras."""

    if name == "GFFactorScenario":
        from rdagent.scenarios.gf_quant.experiment.gf_factor_experiment import GFFactorScenario

        return GFFactorScenario
    if name in {"GFFactorHypothesisGen", "GFFactorHypothesis2Experiment"}:
        from rdagent.scenarios.gf_quant import proposal

        return getattr(proposal, name)
    raise AttributeError(name)
