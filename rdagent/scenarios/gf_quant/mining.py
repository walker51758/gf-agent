"""Small orchestration layer connecting RD-Agent proposals to the V1 gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.gf_quant.conf import FACTOR_PROPOSAL_LOGGER_TAG
from rdagent.scenarios.gf_quant.developer.coder import describe_operators
from rdagent.scenarios.gf_quant.metrics import format_factor_feedback
from rdagent.scenarios.gf_quant.service import FactorAcceptanceService, FactorCandidate, ProcessingResult


class CandidateProposer(Protocol):
    def propose(self, family: str, feedback: str | None = None) -> list[FactorCandidate]: ...


class RDAGentDSLProposer:
    """Use RD-Agent's configured API backend to propose safe remote DSL factors."""

    def __init__(self, api_backend: Any | None = None) -> None:
        if api_backend is None:
            from rdagent.oai.llm_utils import APIBackend

            api_backend = APIBackend()
        self.api_backend = api_backend

    def propose(self, family: str, feedback: str | None = None) -> list[FactorCandidate]:
        if family not in {"price_volume", "fundamental"}:
            raise ValueError(f"Unsupported family: {family}")
        fields = (
            "adjclose, adjopen, adjhigh, adjlow, volume, amount, turnover, mv"
            if family == "price_volume"
            else "mv, pb, pe, ps, balance__parent_equity, balance__total_assets, "
            "income__net_profit, cashflow__net_oper_cash_flow, industry"
        )
        user_prompt = f"""
Propose 1-3 causal China A-share {family} factor candidates.
Available fields: {fields}.
Safe operators, one group per line:
{describe_operators()}
plus numeric arithmetic +, -, *, /, ** and parentheses. Windows and shifts must
be positive integer literals and the winsorize quantile a literal in (0, 0.5).
Financial fields are point-in-time and become available on announcement date.
Capitalised quarterly operators require raw financial fields and align by the
accounting report period; never imitate YOY/QOQ with trading-day shifts.
Return a JSON object with key candidates. Each candidate must contain name,
family, hypothesis, and expression. Do not use future data.
Previous multi-metric feedback:
{feedback or 'No previous attempt.'}
""".strip()
        # The tag is what lets the backend raise the temperature for factor
        # proposals alone (see ``FACTOR_PROPOSAL_LOGGER_TAG``).
        with logger.tag(FACTOR_PROPOSAL_LOGGER_TAG):
            response = self.api_backend.build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt="You are the factor proposal component of Microsoft RD-Agent(Q). Be concise and testable.",
                json_mode=True,
                json_target_type=dict[str, list[dict[str, str]]],
            )
        payload = json.loads(response)
        candidates = payload.get("candidates", [])
        return [
            FactorCandidate(
                name=item["name"],
                family=family,  # enforce the requested arm
                hypothesis=item.get("hypothesis", ""),
                expression=item["expression"],
                metadata={"proposal_backend": "rdagent"},
            )
            for item in candidates
        ]


@dataclass(frozen=True)
class MiningSummary:
    accepted: dict[str, ProcessingResult]
    attempts: tuple[ProcessingResult, ...]


class GFMiningLoop:
    """Iterate proposals until one accepted factor exists in each family."""

    def __init__(self, proposer: CandidateProposer, service: FactorAcceptanceService) -> None:
        self.proposer = proposer
        self.service = service

    def run(self, max_rounds_per_family: int = 5) -> MiningSummary:
        accepted: dict[str, ProcessingResult] = {}
        attempts: list[ProcessingResult] = []
        for family in ("price_volume", "fundamental"):
            feedback: str | None = None
            for _ in range(max_rounds_per_family):
                proposals = self.proposer.propose(family, feedback)
                if not proposals:
                    feedback = "No parseable candidate was returned; produce a simpler expression."
                    continue
                round_results: list[ProcessingResult] = []
                for candidate in proposals:
                    result = self.service.process(candidate)
                    attempts.append(result)
                    round_results.append(result)
                    if result.record.status == "accepted":
                        accepted[family] = result
                        break
                if family in accepted:
                    break
                last = round_results[-1]
                feedback = format_factor_feedback(last.record.metrics) + "\nGate failures: " + ", ".join(last.gate_failures)
        return MiningSummary(accepted, tuple(attempts))
