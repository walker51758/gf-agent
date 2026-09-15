"""JSON-safe epoch reports for the factor-to-portfolio research loop.

An epoch touches every layer (snapshot, strategy trials, ablations, demands,
factor research) and mixes SQLite records, pandas traces and LLM output.  The
report is deliberately a *flat, self-describing* projection of
:class:`~rdagent.scenarios.gf_quant.strategy.orchestrator.OuterEpochResult`:

* it never embeds a panel, a weight matrix or a daily return trace -- those live
  in the strategy registry and in the remote artifacts the records point at;
* every float is coerced to a JSON-safe value, because backtest metrics contain
  ``NaN``/``inf`` for degenerate windows and a report that cannot be written is
  a report that was never audited;
* factor research outcomes are stored through ``as_payload()`` when available so
  the demand -> factor provenance chain survives without importing the class.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rdagent.scenarios.gf_quant.strategy.costs import execution_costs_metrics
from rdagent.scenarios.gf_quant.strategy.orchestrator import OuterEpochResult, StrategyTrial

SCHEMA_VERSION = 1

_HEADLINE_METRICS = (
    "annualized_return",
    "excess_annualized_return",
    "sharpe",
    "max_drawdown",
    "annual_turnover",
    "avg_holdings",
    "observations",
    "signal_lag",
    "buy_cost_bps",
    "sell_cost_bps",
    "round_trip_cost_bps",
    "buy_turnover",
    "sell_turnover",
    "total_transaction_cost",
)


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value into something ``json.dumps`` accepts."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    # Dataclasses, timestamps and numpy scalars are reduced to their text form
    # rather than dropped: an unrepresentable field is still evidence.
    return str(value)


def _headline(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(metrics[key]) for key in _HEADLINE_METRICS if key in metrics}


def _trial_payload(trial: StrategyTrial) -> dict[str, Any]:
    result = trial.result
    payload: dict[str, Any] = {
        "trial_index": trial.trial_index,
        "strategy_name": trial.candidate.name if trial.candidate is not None else None,
        "factor_names": list(trial.candidate.factor_names) if trial.candidate is not None else [],
        "universe": trial.candidate.universe if trial.candidate is not None else None,
        "rule": trial.candidate.rule if trial.candidate is not None else None,
        "signal_lag": trial.candidate.backtest.signal_lag if trial.candidate is not None else None,
        "execution_costs": execution_costs_metrics(),
        "status": result.record.status if result is not None else "error",
        "gate_failures": list(result.gate_failures) if result is not None else [],
        "dedup_of": result.record.dedup_of if result is not None else None,
        "error": trial.error,
        "metrics": {},
        "metrics_by_year": {},
        "artifact_path": None,
        "artifact_checksum": None,
        "annual_gate_requirements": {},
    }
    if result is not None and result.evaluation is not None:
        metrics = result.evaluation.metrics
        payload["metrics"] = _headline(metrics)
        by_year = metrics.get("metrics_by_year")
        if isinstance(by_year, Mapping):
            payload["metrics_by_year"] = {
                str(year): _headline(values) if isinstance(values, Mapping) else _json_safe(values)
                for year, values in by_year.items()
            }
        payload["artifact_path"] = result.evaluation.artifact_path
        payload["artifact_checksum"] = result.evaluation.artifact_checksum
    if result is not None:
        requirements = result.record.metadata.get("annual_gate_requirements", {})
        if isinstance(requirements, Mapping):
            payload["annual_gate_requirements"] = _json_safe(requirements)
        recorded_costs = result.record.metadata.get("execution_costs", {})
        if isinstance(recorded_costs, Mapping):
            payload["execution_costs"] = _json_safe(recorded_costs)
    return payload


def _research_payload(entry: Any) -> dict[str, Any]:
    """Project one factor research outcome without importing its concrete type."""

    if isinstance(entry, Mapping):
        return _json_safe(dict(entry))
    payload = getattr(entry, "as_payload", None)
    if callable(payload):
        return _json_safe(payload())
    return {"representation": _json_safe(str(entry))}


def _demand_payload(demand: Any) -> dict[str, Any]:
    """Serialize a factor demand.

    ``FactorDemand`` is a plain dataclass without an ``as_payload``; ``asdict``
    keeps the demand -> evidence chain intact without importing the class here.
    """

    return _json_safe(asdict(demand) if is_dataclass(demand) and not isinstance(demand, type) else {"representation": str(demand)})


def epoch_report(result: OuterEpochResult) -> dict[str, Any]:
    """Build the compact, auditable report for one completed outer epoch."""

    statuses: dict[str, int] = {}
    for trial in result.strategy_trials:
        status = trial.result.record.status if trial.result is not None else "error"
        statuses[status] = statuses.get(status, 0) + 1

    tiers: dict[str, int] = {}
    for factor in result.working_set.factors:
        tiers[factor.eligibility_tier] = tiers.get(factor.eligibility_tier, 0) + 1

    research = [_research_payload(entry) for entry in result.factor_research_results]
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch_id": result.epoch_id,
        "created_at": result.created_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution_costs": execution_costs_metrics(),
        "evaluation": {
            "start": result.evaluation_start,
            "end": result.evaluation_end,
        },
        "snapshot": {
            "snapshot_id": result.snapshot.snapshot_id,
            "checksum": result.snapshot.checksum,
            "data_cutoff": result.snapshot.data_cutoff,
            "factor_count": len(result.snapshot.factors),
        },
        "working_set": {
            "working_set_id": result.working_set.working_set_id,
            "snapshot_checksum": result.working_set.snapshot_checksum,
            "factor_count": len(result.working_set.factors),
            "eligibility_tiers": tiers,
            "portfolio_feedback": _json_safe(dict(result.working_set.portfolio_feedback)),
        },
        "summary": {
            "strategy_trials": len(result.strategy_trials),
            "strategy_statuses": statuses,
            "factor_contributions": len(result.contributions),
            "factor_demands": len(result.factor_demands),
            "factor_research_rounds": len(research),
            "accepted_factors_from_research": sum(
                len(payload.get("accepted_factor_ids", []))
                for payload in research
                if isinstance(payload, Mapping)
            ),
        },
        "strategy_trials": [_trial_payload(trial) for trial in result.strategy_trials],
        "factor_contributions": [
            {
                "factor_name": contribution.factor_name,
                "factor_id": contribution.factor_id,
                "strategy_trials": contribution.strategy_trials,
                "accepted_trials": contribution.accepted_trials,
                "mean_marginal_annualized_return": _json_safe(contribution.mean_marginal_annualized_return),
                "mean_marginal_sharpe": _json_safe(contribution.mean_marginal_sharpe),
                "mean_marginal_drawdown_improvement": _json_safe(contribution.mean_marginal_drawdown_improvement),
                "mean_marginal_turnover_reduction": _json_safe(contribution.mean_marginal_turnover_reduction),
            }
            for contribution in result.contributions
        ],
        "factor_demands": [_demand_payload(demand) for demand in result.factor_demands],
        "factor_research": research,
    }


def write_epoch_report(result: OuterEpochResult, path: str | Path) -> Path:
    """Persist one epoch report.  Writes atomically enough for a report file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(epoch_report(result), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return destination
