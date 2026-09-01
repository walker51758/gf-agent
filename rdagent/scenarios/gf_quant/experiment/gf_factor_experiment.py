"""
GF (GuangFa Securities) factor scenario.

The proposal is expressed in a small safe DSL and evaluated beside the remote
``G:/base`` Parquet store.  No local HDF5 materialization or CoSTEER execution
is part of the production loop.
"""

from copy import deepcopy

from rdagent.core.experiment import Task
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
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
        self._background = deepcopy(
            T(".prompts:gf_factor_background").r(runtime_environment=self.get_runtime_environment())
        )
        self._source_data = deepcopy(T(".prompts:gf_factor_source_data").r())
        self._output_format = deepcopy(T(".prompts:gf_factor_output_format").r())
        self._interface = deepcopy(T(".prompts:gf_factor_interface").r())
        self._strategy = deepcopy(T(".prompts:gf_factor_strategy").r())
        self._simulator = deepcopy(T(".prompts:gf_factor_simulator").r())
        self._acceptance_standards = deepcopy(T(".prompts:gf_factor_acceptance_standards").r())
        self._rich_style_description = deepcopy(T(".prompts:gf_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:gf_factor_experiment_setting").r())

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def acceptance_standards(self) -> str:
        """How acceptance is decided: calibrated family benchmark standards."""
        return self._acceptance_standards

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        """Same as the Qlib scenario, plus the calibrated acceptance policy."""
        base = super().get_scenario_all_desc(task, filtered_tag, simple_background)
        if simple_background:
            return base
        return f"""{base}
How acceptance is decided:
{self.acceptance_standards}
"""

    def get_runtime_environment(self):
        return (
            "Factor expressions run on the pinned GF Windows data host using "
            "G:/.venv/Scripts/python.exe and the safe GF DSL worker."
        )
