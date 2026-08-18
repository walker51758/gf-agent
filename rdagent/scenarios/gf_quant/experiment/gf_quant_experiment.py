"""
GF (GuangFa Securities) quant (factor + model) scenario.

Extends the Qlib quant scenario by pointing the factor-coder data folders at the
GF data folder and by using GF-tailored prompt templates. The action-selection
hypothesis generator, factor/model coding stack, runners and feedback are all
inherited from the Qlib quant scenario.
"""

from copy import deepcopy

from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.model_coder.conf import get_model_env
from rdagent.scenarios.gf_quant.data.loader import ensure_gf_data
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class GFQuantScenario(QlibQuantScenario):
    """GF Securities quant (factor + model) scenario.

    .. note::
       We deliberately do **not** call ``super().__init__()`` because the Qlib
       quant scenario loads its prompt templates with ``T(".prompts:qlib_*")``
       which resolves relative to the *caller* module (see
       :class:`GFFactorScenario` for details).
    """

    def __init__(self) -> None:
        # Make the GF data folders available to the whole factor stack
        ensure_gf_data(raise_on_missing=False)

        self._source_data = deepcopy(get_data_folder_intro())
        self._rich_style_description = deepcopy(T(".prompts:gf_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:gf_factor_experiment_setting").r())

    def background(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        quant_background = "The background of the scenario is as follows:\n" + T(".prompts:gf_quant_background").r(
            runtime_environment=self.get_runtime_environment(),
        )
        factor_background = (
            "This time, I need your help with the research and development of the factor. "
            "The background of the factor scenario is as follows:\n"
            + T(".prompts:gf_factor_background").r(runtime_environment=self.get_runtime_environment(tag="factor"))
        )
        model_background = (
            "This time, I need your help with the research and development of the model. "
            "The background of the model scenario is as follows:\n"
            + T(".prompts:gf_model_background").r(runtime_environment=self.get_runtime_environment(tag="model"))
        )

        if tag is None:
            return quant_background + "\n" + factor_background + "\n" + model_background
        elif tag == "factor":
            return factor_background
        else:
            return model_background

    def output_format(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor_output_format = "The factor code should output the following format:\n" + T(
            ".prompts:gf_factor_output_format"
        ).r()
        model_output_format = "The model code should output the following format:\n" + T(
            ".prompts:gf_model_output_format"
        ).r()

        if tag is None:
            return factor_output_format + "\n" + model_output_format
        elif tag == "factor":
            return factor_output_format
        else:
            return model_output_format

    def interface(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor_interface = "The factor code should be written in the following interface:\n" + T(
            ".prompts:gf_factor_interface"
        ).r()
        model_interface = "The model code should be written in the following interface:\n" + T(
            ".prompts:gf_model_interface"
        ).r()

        if tag is None:
            return factor_interface + "\n" + model_interface
        elif tag == "factor":
            return factor_interface
        else:
            return model_interface

    def simulator(self, tag=None) -> str:
        assert tag in [None, "factor", "model"]
        factor_simulator = "The factor code will be sent to the simulator:\n" + T(
            ".prompts:gf_factor_simulator"
        ).r()
        model_simulator = "The model code will be sent to the simulator:\n" + T(".prompts:gf_model_simulator").r()

        if tag is None:
            return factor_simulator + "\n" + model_simulator
        elif tag == "factor":
            return factor_simulator
        else:
            return model_simulator

    def get_runtime_environment(self, tag: str = None) -> str:
        assert tag in [None, "factor", "model"]

        def _safe(env_getter, name: str) -> str:
            try:
                return get_runtime_environment_by_env(env=env_getter())
            except Exception as e:  # noqa: BLE001 - runtime detection is best-effort
                return f"(Runtime environment detection for {name} failed, continuing without it: {e})"

        if tag is None:
            return (
                "=== [Environment to generate the factors] ===\n"
                + _safe(get_factor_env, "factor").strip()
                + "\n\n=== [Environment to train the models] ===\n"
                + _safe(get_model_env, "model").strip()
            )
        elif tag == "factor":
            return _safe(get_factor_env, "factor")
        else:
            return _safe(get_model_env, "model")
