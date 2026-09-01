"""LLM strategy proposal engine for one frozen factor working set.

The model produces declarative JSON only.  It may choose factors and portfolio
parameters, but it never writes executable backtest code; deterministic local
or remote framework code remains responsible for composition, lag and costs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.orchestrator import StrategyProposalContext


def _metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _safe_name(value: Any, context: StrategyProposalContext) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "strategy").strip()).strip("_").lower()
    base = base[:48] or "strategy"
    epoch_suffix = re.sub(r"[^A-Za-z0-9]+", "", context.epoch_id)[-8:] or "epoch"
    return f"{base}_{epoch_suffix}_t{context.trial_index:02d}"


class LLMStrategyProposalEngine:
    """Convert a real RD-Agent API response into a validated strategy candidate."""

    def __init__(
        self,
        api_backend: Any | None = None,
        *,
        universe: str = "csi_all",
        min_factors: int = 2,
        max_factors: int = 6,
        min_top_n: int = 20,
        max_top_n: int = 100,
        default_cost_bps: float = 10.0,
    ) -> None:
        if api_backend is None:
            from rdagent.oai.llm_utils import APIBackend

            api_backend = APIBackend()
        if not 1 <= min_factors <= max_factors <= 8:
            raise ValueError("factor bounds must satisfy 1 <= min <= max <= 8")
        if not 1 <= min_top_n <= max_top_n:
            raise ValueError("top_n bounds are invalid")
        if default_cost_bps <= 0:
            raise ValueError("default_cost_bps must be positive")
        self.api_backend = api_backend
        self.universe = universe
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.min_top_n = min_top_n
        self.max_top_n = max_top_n
        self.default_cost_bps = default_cost_bps

    @staticmethod
    def _factor_payload(context: StrategyProposalContext) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []
        for factor in context.working_set.factors:
            metrics = factor.metrics
            factors.append(
                {
                    "name": factor.name,
                    "factor_id": factor.factor_id,
                    "family": factor.family,
                    "eligibility_tier": factor.eligibility_tier,
                    "theme": factor.theme,
                    "intended_role": factor.intended_role,
                    "direction": factor.direction,
                    "expression": factor.expression,
                    "rank_ic": _metric(metrics, "Rank_IC", "rank_ic"),
                    "rank_icir": _metric(metrics, "Rank_ICIR", "rank_icir"),
                    "annualized_return": _metric(metrics, "annualized_return"),
                    "sharpe": _metric(metrics, "sharpe"),
                    "max_drawdown": _metric(metrics, "max_drawdown"),
                    "turnover": _metric(metrics, "turnover"),
                    "coverage": _metric(metrics, "coverage"),
                }
            )
        return factors

    @staticmethod
    def _history_payload(context: StrategyProposalContext) -> list[dict[str, Any]]:
        return [
            {
                "strategy_name": outcome.strategy_name,
                "status": outcome.status,
                "gate_failures": list(outcome.gate_failures),
                "metrics": dict(outcome.metrics),
            }
            for outcome in context.prior_outcomes
        ]

    def _prompt(self, context: StrategyProposalContext) -> str:
        factors = json.dumps(self._factor_payload(context), ensure_ascii=False, indent=2)
        history = json.dumps(self._history_payload(context), ensure_ascii=False, indent=2)
        return f"""
Design one auditable China A-share long-only portfolio strategy for inner trial
{context.trial_index} of epoch {context.epoch_id}.  The factor snapshot is
immutable ({context.snapshot_id}, data cutoff {context.data_cutoff}).

Available frozen factors:
{factors}

Prior trials in this same epoch (do not repeat a failed specification):
{history}

Return exactly one JSON object with these keys:
- name: short ASCII strategy name;
- hypothesis: why these factors should complement each other economically;
- factor_names: {self.min_factors}-{self.max_factors} distinct names copied exactly from the list;
- rule: concise declarative score/selection description, never Python code;
- top_n: integer in [{self.min_top_n}, {self.max_top_n}];
- weighting: one of equal, rank, proportional;
- weight_cap: null or a number in (0, 1];
- signal_lag: integer at least 1;
- cost_bps: positive one-way transaction cost in basis points.

Use the frozen direction supplied for each factor.  Prefer complementary themes
and explain any conditional factor.  Do not use future data, do not invent a
factor, do not emit executable code, and do not optimize against a final holdout.
""".strip()

    def __call__(self, context: StrategyProposalContext) -> StrategyCandidate:
        response = self.api_backend.build_messages_and_create_chat_completion(
            user_prompt=self._prompt(context),
            system_prompt=(
                "You are the portfolio-construction component of a fundamental quant research agent. "
                "Return strict JSON specifications; deterministic framework code performs every backtest."
            ),
            json_mode=True,
            json_target_type=dict[str, Any],
        )
        payload = json.loads(response)
        if not isinstance(payload, Mapping):
            raise ValueError("strategy proposal must be a JSON object")

        factor_names = tuple(str(value) for value in payload.get("factor_names", []))
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("strategy proposal contains duplicate factor names")
        if not self.min_factors <= len(factor_names) <= self.max_factors:
            raise ValueError(f"strategy proposal must use {self.min_factors}-{self.max_factors} factors")
        available = {factor.name for factor in context.working_set.factors}
        unknown = sorted(set(factor_names).difference(available))
        if unknown:
            raise ValueError(f"strategy proposal invented factors: {unknown[:3]}")

        top_n = int(payload.get("top_n", 0))
        if not self.min_top_n <= top_n <= self.max_top_n:
            raise ValueError(f"top_n must be in [{self.min_top_n}, {self.max_top_n}]")
        cost_bps = float(payload.get("cost_bps", self.default_cost_bps))
        if cost_bps <= 0:
            raise ValueError("cost_bps must be positive")

        return StrategyCandidate(
            name=_safe_name(payload.get("name"), context),
            universe=self.universe,
            composition=CompositionSpec(
                top_n=top_n,
                weighting=str(payload.get("weighting", "equal")),
                weight_cap=(None if payload.get("weight_cap") is None else float(payload["weight_cap"])),
            ),
            factor_names=factor_names,
            rule=str(payload.get("rule", "cross-sectional rank blend")).strip(),
            hypothesis=str(payload.get("hypothesis", "")).strip(),
            backtest=BacktestSpec(signal_lag=int(payload.get("signal_lag", 1)), cost_bps=cost_bps),
            metadata={
                "proposal_backend": "rdagent_llm",
                "epoch_id": context.epoch_id,
                "trial_index": context.trial_index,
                "snapshot_id": context.snapshot_id,
                "working_set_id": context.working_set.working_set_id,
            },
        )
