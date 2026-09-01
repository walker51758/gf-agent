from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.gf_quant.developer.coder import GFFactorCoder
from rdagent.scenarios.gf_quant.developer.factor_runner import GFFactorRunner
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.service import EvaluationResult, FactorAcceptanceService


def _task(name: str = "reversal_20", expression: str = "-pct_change(adjclose, 20)") -> FactorTask:
    return FactorTask(
        factor_name=name,
        factor_description="[Reversal] twenty-day adjusted-close reversal",
        factor_formulation=expression,
        variables={"adjclose": "adjusted close"},
    )


def _experiment(task: FactorTask) -> SimpleNamespace:
    return SimpleNamespace(
        sub_tasks=[task],
        sub_workspace_list=[None],
        hypothesis=SimpleNamespace(hypothesis="Short-horizon reversal should predict returns."),
        sub_results={},
        result=None,
        stdout="",
    )


def test_dsl_coder_marks_valid_task_implemented() -> None:
    exp = _experiment(_task())

    result = GFFactorCoder(SimpleNamespace()).develop(exp)

    assert result is exp
    assert exp.sub_tasks[0].factor_implementation is True
    assert exp.sub_tasks[0].dsl_fields == ("adjclose",)
    assert exp.sub_workspace_list == [None]


def test_dsl_coder_rejects_arbitrary_python() -> None:
    exp = _experiment(_task(expression="__import__('os').system('whoami')"))

    with pytest.raises(FactorEmptyError, match="Invalid GF DSL"):
        GFFactorCoder(SimpleNamespace()).develop(exp)


def test_remote_runner_writes_trace_metrics_and_registry(tmp_path: Path) -> None:
    seen = []

    def evaluator(candidate):
        seen.append(candidate)
        return EvaluationResult(
            metrics={
                "IC": 0.03,
                "ICIR": 0.4,
                "Rank_IC": 0.05,
                "Rank_ICIR": 0.8,
                "annualized_return": 0.12,
                "sharpe": 0.9,
                "max_drawdown": -0.10,
                "turnover": 0.25,
                "coverage": 0.92,
                "win_rate": 0.54,
                "metrics_by_year": {
                    "2020": {"Rank_IC": 0.04},
                    "2021": {"Rank_IC": 0.06},
                    "2022": {"Rank_IC": 0.05},
                    "2023": {"Rank_IC": 0.05},
                },
                "stability_by_year": {"2022": {"annualized_return": 0.08}},
            },
            artifact_path="G:/gf_quant_agent/artifacts/reversal_20.parquet",
            artifact_checksum="abc123",
        )

    exp = _experiment(_task())
    exp.search_mode = "explore"
    exp.search_direction = "reversal"
    exp.sub_tasks[0].search_mode = "explore"
    exp.sub_tasks[0].search_direction = "reversal"
    GFFactorCoder(SimpleNamespace()).develop(exp)
    registry = FactorRegistry(tmp_path / "factor_registry.sqlite")
    service = FactorAcceptanceService(registry, evaluator, calibrations={})  # explicit development fallback
    result = GFFactorRunner(SimpleNamespace(), evaluator=evaluator, registry=registry, service=service).develop(exp)

    assert seen[0].family == "price_volume"
    assert seen[0].expression == "-pct_change(adjclose, 20)"
    assert result.result.loc["IC"] == pytest.approx(0.03)
    assert result.result.loc["1day.excess_return_without_cost.annualized_return"] == pytest.approx(0.12)
    assert result.result.loc["1day.excess_return_without_cost.max_drawdown"] == pytest.approx(-0.10)
    assert result.remote_results[0]["status"] == "accepted"
    stored = registry.get_by_name("reversal_20")
    assert stored.artifact_checksum == "abc123"
    assert stored.metadata["search_mode"] == "explore"
    assert stored.metadata["search_direction"] == "reversal"
    assert result.remote_results[0]["search_mode"] == "explore"
    assert result.remote_results[0]["search_direction"] == "reversal"


def test_factor_entrypoint_keeps_loop_n_and_loads_dotenv_first(monkeypatch) -> None:
    from rdagent.app.gf_rd_loop import factor

    calls = {}

    class DummyLoop:
        def __init__(self, settings):
            calls["settings"] = settings

        async def run(self, **kwargs):
            calls["run"] = kwargs

    monkeypatch.setattr(factor, "GFFactorRDLoop", DummyLoop)
    monkeypatch.setattr(factor, "configure_gf_llm_backend", lambda: calls.setdefault("configured", True))

    factor.main(loop_n=2)

    source = Path(factor.__file__).read_text(encoding="utf-8")
    assert source.index("load_dotenv()") < source.index("from rdagent.app.gf_rd_loop.conf")
    assert calls["configured"] is True
    assert calls["run"]["loop_n"] == 2
