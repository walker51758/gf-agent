"""
GF factor proposal classes.

The proposal logic is identical to the Qlib factor proposal: the GF-specific
context reaches the LLM through :meth:`Scenario.get_scenario_all_desc`, which is
provided by the GF scenarios. These thin subclasses exist so the GF extension
has explicit, stable extension points for future GF-specific prompt tweaks.
"""

from rdagent.scenarios.qlib.proposal.factor_proposal import (
    QlibFactorHypothesis2Experiment,
    QlibFactorHypothesisGen,
)

__all__ = ["GFFactorHypothesisGen", "GFFactorHypothesis2Experiment"]


class GFFactorHypothesisGen(QlibFactorHypothesisGen):
    """GF factor hypothesis generator (inherits the Qlib proposal logic)."""


class GFFactorHypothesis2Experiment(QlibFactorHypothesis2Experiment):
    """GF factor hypothesis -> experiment converter (inherits the Qlib logic)."""
