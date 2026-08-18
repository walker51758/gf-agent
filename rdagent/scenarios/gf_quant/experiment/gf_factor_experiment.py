"""
GF (GuangFa Securities) factor scenario.

Extends the Qlib factor scenario by

* pointing the factor-coder data folders at the GF data folder (and building it
  from the GF source when needed),
* using GF-tailored prompt templates,
* describing the GF A-share source data.

The rest of the factor stack (proposal, CoSTEER coding, runner, feedback) is
inherited unchanged from the Qlib scenario.
"""

from copy import deepcopy

from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.core.experiment import Task
from rdagent.scenarios.gf_quant.data.loader import ensure_gf_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class GFFactorScenario(QlibFactorScenario):
    """GF Securities factor research scenario.

    .. note::
       We deliberately do **not** call ``super().__init__()``: the Qlib scenario
       loads its prompt templates with ``T(".prompts:qlib_*")`` which resolves
       relative to the *caller* module, so calling it from the GF package would
       look for ``qlib_*`` keys in the GF ``prompts.yaml`` and fail.
    """

    def __init__(self) -> None:
        # Make the GF data folders available to the whole factor stack
        ensure_gf_data(raise_on_missing=False)

        self._background = deepcopy(
            T(".prompts:gf_factor_background").r(runtime_environment=self.get_runtime_environment())
        )
        self._source_data = deepcopy(get_data_folder_intro())
        self._output_format = deepcopy(T(".prompts:gf_factor_output_format").r())
        self._interface = deepcopy(T(".prompts:gf_factor_interface").r())
        self._strategy = deepcopy(T(".prompts:gf_factor_strategy").r())
        self._simulator = deepcopy(T(".prompts:gf_factor_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:gf_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:gf_factor_experiment_setting").r())

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    def get_runtime_environment(self):
        # Detection requires a working local conda env; degrade gracefully when
        # the runtime is not provisioned (e.g. Windows dev box).
        try:
            factor_env = get_factor_env()
            return get_runtime_environment_by_env(env=factor_env)
        except Exception as e:  # noqa: BLE001 - runtime detection is best-effort
            return f"(Runtime environment detection failed, continuing without it: {e})"
