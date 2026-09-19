"""GF factor-loop configuration."""

import os

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting

GF_LLM_BACKEND = "rdagent.oai.backend.LiteLLMAPIBackend"
GF_LLM_CHAT_MODEL = "deepseek/deepseek-flash"


class GFFactorPropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="GF_FACTOR_", protected_namespaces=())

    # 1) override base settings for the GF scenario
    scen: str = "rdagent.scenarios.gf_quant.experiment.gf_factor_experiment.GFFactorScenario"
    """Scenario class for GF Factor"""

    hypothesis_gen: str = "rdagent.scenarios.gf_quant.proposal.GFFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.gf_quant.proposal.GFFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.gf_quant.developer.coder.GFFactorCoder"
    """Lightweight coder that validates the remote GF factor DSL"""

    runner: str = "rdagent.scenarios.gf_quant.developer.factor_runner.GFFactorRunner"
    """Remote SSH runner for GF factor expressions"""

    summarizer: str = "rdagent.scenarios.gf_quant.developer.feedback.GFFactorExperiment2Feedback"
    """Summarizer class (reused from Qlib)"""

    exploration_probability: float = Field(default=0.30, ge=0.0, le=1.0)
    """Chance that a non-forced round starts an independent research direction."""

    max_same_family_streak: int = Field(default=6, ge=1)
    """Force exploration after this many rounds in one fine-grained research direction."""

    stagnation_patience: int = Field(default=3, ge=1)
    """Force exploration after this many consecutive rounds fail to improve SOTA."""

    search_random_seed: int | None = None
    """Optional reproducible seed for mode selection and least-tested direction ties."""


GF_FACTOR_PROP_SETTING = GFFactorPropSetting()


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
