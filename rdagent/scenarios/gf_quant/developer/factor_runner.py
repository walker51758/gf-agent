"""Remote runner for GF factor DSL experiments."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.developer.coder import validate_dsl_expression
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.remote import GFFactorEvaluator
from rdagent.scenarios.gf_quant.service import (
    FactorAcceptanceService,
    FactorCandidate,
    ProcessingResult,
)


def _number(metrics: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    try:
        value = metrics.get(key)
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class GFFactorRunner(Developer):
    """Evaluate every proposed DSL expression beside the remote ``G:/base`` data."""

    def __init__(
        self,
        scen,
        *,
        evaluator: GFFactorEvaluator | None = None,
        registry: FactorRegistry | None = None,
        service: FactorAcceptanceService | None = None,
    ) -> None:
        super().__init__(scen)
        self.registry = registry or FactorRegistry(GF_SCENARIO_SETTINGS.registry_path)
        self.evaluator = evaluator or GFFactorEvaluator()
        self.service = service or FactorAcceptanceService(self.registry, self.evaluator)

    @staticmethod
    def _family(task, fields: tuple[str, ...]) -> str:  # type: ignore[no-untyped-def]
        declared = getattr(task, "gf_family", None)
        if declared in {"price_volume", "fundamental"}:
            return declared
        return "fundamental" if any("__" in field for field in fields) else "price_volume"

    @staticmethod
    def _best_key(result: ProcessingResult) -> tuple[bool, float, float]:
        metrics = result.record.metrics
        return (
            result.record.status == "accepted",
            abs(_number(metrics, "Rank_IC")),
            _number(metrics, "annualized_return"),
        )

    def develop(self, exp):  # type: ignore[no-untyped-def]
        if not exp.sub_tasks:
            raise FactorEmptyError("The GF experiment contains no factor tasks")

        completed: list[tuple[Any, ProcessingResult]] = []
        failures: list[dict[str, str]] = []
        hypothesis = getattr(exp.hypothesis, "hypothesis", "") if exp.hypothesis is not None else ""

        for task in exp.sub_tasks:
            try:
                expression = str(task.factor_formulation).strip()
                fields = tuple(getattr(task, "dsl_fields", ())) or validate_dsl_expression(expression)
                candidate = FactorCandidate(
                    name=str(task.factor_name),
                    family=self._family(task, fields),
                    expression=expression,
                    hypothesis=hypothesis,
                    metadata={"fields": list(fields), "source": "gf_rd_loop"},
                )
                processed = self.service.process(candidate)
                completed.append((task, processed))
                task.factor_implementation = True
                annualized = processed.record.metrics.get("annualized_return")
                if annualized is not None:
                    exp.sub_results[str(task.factor_name)] = float(annualized)
            except Exception as exc:  # one bad candidate must not hide successful siblings
                task.factor_implementation = False
                failures.append({"factor": str(task.factor_name), "error": str(exc)})

        if not completed:
            detail = "; ".join(f"{item['factor']}: {item['error']}" for item in failures)
            raise FactorEmptyError(f"All remote GF factor evaluations failed. {detail}")

        _, best = max(completed, key=lambda item: self._best_key(item[1]))
        metrics: dict[str, Any] = dict(best.record.metrics)
        metrics["1day.excess_return_without_cost.annualized_return"] = metrics.get("annualized_return")
        metrics["1day.excess_return_without_cost.max_drawdown"] = metrics.get("max_drawdown")
        metrics["selected_factor"] = best.record.name
        metrics["registry_status"] = best.record.status
        exp.result = pd.Series(metrics, dtype=object)

        remote_results = [
            {
                "name": result.record.name,
                "family": result.record.family,
                "expression": result.record.expression,
                "status": result.record.status,
                "metrics": result.record.metrics,
                "artifact_path": result.record.artifact_path,
                "artifact_checksum": result.record.artifact_checksum,
                "gate_failures": list(result.gate_failures),
            }
            for _, result in completed
        ]
        exp.remote_results = remote_results
        exp.stdout = json.dumps(
            {"remote_results": remote_results, "failures": failures},
            ensure_ascii=False,
            indent=2,
        )
        return exp
