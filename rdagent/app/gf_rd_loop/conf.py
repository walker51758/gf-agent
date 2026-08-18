"""
GF (GuangFa Securities) loop configuration.

The GF factor / quant loops are the Qlib factor / quant loops pointed at the GF
scenario classes. Settings use the ``GF_FACTOR_`` / ``GF_QUANT_`` environment
prefixes (with the Qlib prefixes kept as fallbacks via inheritance).
"""

import os

from pydantic_settings import SettingsConfigDict

from rdagent.app.qlib_rd_loop.conf import (
    FactorBasePropSetting,
    QuantBasePropSetting,
)

GF_LLM_BACKEND = "rdagent.oai.backend.LiteLLMAPIBackend"
GF_LLM_CHAT_MODEL = "deepseek/deepseek-v4-flash"


class GFFactorPropSetting(FactorBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="GF_FACTOR_", protected_namespaces=())

    # 1) override base settings for the GF scenario
    scen: str = "rdagent.scenarios.gf_quant.experiment.gf_factor_experiment.GFFactorScenario"
    """Scenario class for GF Factor"""

    hypothesis_gen: str = "rdagent.scenarios.gf_quant.proposal.GFFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.gf_quant.proposal.GFFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class (reused from Qlib)"""

    runner: str = "rdagent.scenarios.gf_quant.developer.factor_runner.GFFactorRunner"
    """Runner class (reused from Qlib)"""

    summarizer: str = "rdagent.scenarios.gf_quant.developer.feedback.GFFactorExperiment2Feedback"
    """Summarizer class (reused from Qlib)"""


class GFQuantPropSetting(QuantBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="GF_QUANT_", protected_namespaces=())

    # 1) override base settings for the GF scenario
    scen: str = "rdagent.scenarios.gf_quant.experiment.gf_quant_experiment.GFQuantScenario"
    """Scenario class for GF Quant"""

    quant_hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.quant_proposal.QlibQuantHypothesisGen"
    """Hypothesis generation class (reused from Qlib)"""

    model_hypothesis2experiment: str = (
        "rdagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    )
    """Hypothesis to experiment class for models (reused from Qlib)"""

    model_coder: str = "rdagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    """Coder class for models (reused from Qlib)"""

    model_runner: str = "rdagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    """Runner class for models (reused from Qlib)"""

    model_summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibModelExperiment2Feedback"
    """Summarizer class for models (reused from Qlib)"""

    factor_hypothesis2experiment: str = "rdagent.scenarios.gf_quant.proposal.GFFactorHypothesis2Experiment"
    """Hypothesis to experiment class for factors"""

    factor_coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class for factors (reused from Qlib)"""

    factor_runner: str = "rdagent.scenarios.gf_quant.developer.factor_runner.GFFactorRunner"
    """Runner class for factors (reused from Qlib)"""

    factor_summarizer: str = "rdagent.scenarios.gf_quant.developer.feedback.GFFactorExperiment2Feedback"
    """Summarizer class for factors (reused from Qlib)"""


GF_FACTOR_PROP_SETTING = GFFactorPropSetting()
GF_QUANT_PROP_SETTING = GFQuantPropSetting()


def configure_gf_llm_backend() -> None:
    """
    Wire the GF LLM backend (DeepSeek through LiteLLM) into RD-Agent.

    Reads ``GFAGENT_DEEPSEEK_API_KEY`` from the environment once and exposes it
    to LiteLLM as ``DEEPSEEK_API_KEY``; sets the chat model and backend on the
    shared settings singletons. It is safe to call multiple times.
    """
    from rdagent.oai.backend.litellm import LITELLM_SETTINGS
    from rdagent.oai.llm_conf import LLM_SETTINGS

    from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS

    api_key = os.environ.get(GF_SCENARIO_SETTINGS.deepseek_api_key_env, "")
    if api_key:
        # LiteLLM's deepseek provider reads DEEPSEEK_API_KEY from the environment
        os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
    os.environ.setdefault("DEEPSEEK_API_BASE", GF_SCENARIO_SETTINGS.llm_base_url)

    LLM_SETTINGS.backend = GF_SCENARIO_SETTINGS.llm_backend
    LITELLM_SETTINGS.chat_model = GF_SCENARIO_SETTINGS.llm_chat_model
    LITELLM_SETTINGS.chat_openai_base_url = GF_SCENARIO_SETTINGS.llm_base_url
