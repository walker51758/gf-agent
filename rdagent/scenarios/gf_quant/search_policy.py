"""Small explore-or-improve policy for the GF factor RDLoop."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence


SearchMode = Literal["explore", "improve"]
SearchDirection = Literal["reversal", "momentum", "liquidity", "volatility", "fundamental", "other"]

# ``other`` is an inference fallback, not a scheduled arm.  Keeping the
# scheduled set small makes the policy understandable and gives every arm a
# concrete set of fields/operators in the GF DSL.
SEARCH_DIRECTIONS: tuple[SearchDirection, ...] = (
    "reversal",
    "momentum",
    "liquidity",
    "volatility",
    "fundamental",
)


@dataclass(frozen=True)
class TestedFactor:
    name: str
    expression: str
    direction: SearchDirection


@dataclass(frozen=True)
class SearchDecision:
    mode: SearchMode
    direction: SearchDirection
    reason: str
    same_direction_streak: int
    stagnation_streak: int
    direction_counts: dict[str, int]

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "direction": self.direction,
            "reason": self.reason,
            "same_direction_streak": self.same_direction_streak,
            "stagnation_streak": self.stagnation_streak,
            "direction_counts": self.direction_counts,
        }


def infer_search_direction(*parts: object) -> SearchDirection:
    """Infer a research direction for legacy records without policy metadata."""

    text = " ".join(str(part) for part in parts if part is not None).lower()
    if not text:
        return "other"

    # Reversal must be checked before volume/volatility because many of the
    # existing reversal variants also contain turnover or ts_std.
    if any(token in text for token in ("reversal", "mean reversion", "mean-reversion", "反转")):
        return "reversal"
    compact = "".join(text.split())
    if "-pct_change(" in compact or "-delta(" in compact:
        return "reversal"

    if any(
        token in text
        for token in (
            "balance__",
            "income__",
            "cashflow__",
            "fundamental",
            "valuation",
            "value factor",
            "quality",
            "profitability",
            "book-to-market",
            "book to market",
            "基本面",
            "价值",
            "质量",
        )
    ):
        return "fundamental"
    if any(token in text for token in ("volatility", "ts_std(", "price range", "true range", "波动")):
        return "volatility"
    if any(token in text for token in ("liquidity", "turnover", "volume", "amount", "换手", "流动性", "成交量")):
        return "liquidity"
    if any(token in text for token in ("momentum", "trend", "pct_change(", "delta(", "动量", "趋势")):
        return "momentum"
    if any(token in text for token in (" pb", " pe", " ps", "parent_equity", "net_profit")):
        return "fundamental"
    return "other"


def experiment_direction(experiment: object) -> SearchDirection:
    """Return policy metadata, or infer a direction from an old experiment."""

    declared = getattr(experiment, "search_direction", None)
    if declared in (*SEARCH_DIRECTIONS, "other"):
        return declared

    hypothesis = getattr(experiment, "hypothesis", None)
    declared = getattr(hypothesis, "search_direction", None)
    if declared in (*SEARCH_DIRECTIONS, "other"):
        return declared

    parts: list[object] = [
        getattr(hypothesis, "hypothesis", ""),
        getattr(hypothesis, "reason", ""),
    ]
    for task in getattr(experiment, "sub_tasks", ()) or ():
        task_direction = getattr(task, "search_direction", None)
        if task_direction in (*SEARCH_DIRECTIONS, "other"):
            return task_direction
        parts.extend(
            (
                getattr(task, "factor_name", ""),
                getattr(task, "factor_description", ""),
                getattr(task, "factor_formulation", ""),
            )
        )
    return infer_search_direction(*parts)


def collect_tested_factors(trace: object, registry_records: Iterable[object] = ()) -> list[TestedFactor]:
    """Collect a unique, metric-free archive from the trace and registry."""

    unique: dict[tuple[str, str], TestedFactor] = {}
    for experiment, _feedback in getattr(trace, "hist", ()) or ():
        exp_direction = experiment_direction(experiment)
        for task in getattr(experiment, "sub_tasks", ()) or ():
            name = str(getattr(task, "factor_name", "")).strip()
            expression = str(getattr(task, "factor_formulation", "")).strip()
            if not name and not expression:
                continue
            direction = getattr(task, "search_direction", None)
            if direction not in (*SEARCH_DIRECTIONS, "other"):
                direction = exp_direction if exp_direction != "other" else infer_search_direction(name, expression)
            unique[(name, expression)] = TestedFactor(name, expression, direction)

    for record in registry_records:
        name = str(getattr(record, "name", "")).strip()
        expression = str(getattr(record, "expression", "")).strip()
        if not name and not expression:
            continue
        metadata = getattr(record, "metadata", {}) or {}
        direction = metadata.get("search_direction") if isinstance(metadata, dict) else None
        if direction not in (*SEARCH_DIRECTIONS, "other"):
            direction = infer_search_direction(name, expression, getattr(record, "family", ""))
        unique.setdefault((name, expression), TestedFactor(name, expression, direction))
    return list(unique.values())


def format_tested_archive(factors: Sequence[TestedFactor], *, limit: int = 200) -> str:
    """Render only information needed to avoid duplicates during exploration."""

    counts = Counter(factor.direction for factor in factors)
    count_line = ", ".join(f"{direction}={counts.get(direction, 0)}" for direction in SEARCH_DIRECTIONS)
    if not factors:
        tested = "- none"
    else:
        recent = factors[-limit:]
        tested = "\n".join(
            f"- {factor.name or '(unnamed)'} | {factor.direction} | {factor.expression}" for factor in recent
        )
        if len(factors) > limit:
            tested = f"- {len(factors) - limit} older entries omitted; registry deduplication remains active\n" + tested
    return (
        "Direction counts (deduplication context only): "
        + count_line
        + "\nAlready-tested names and expressions (do not infer performance from this list):\n"
        + tested
    )


class GFSearchPolicy:
    """Choose one auditable search mode for the next completed RDLoop round."""

    def __init__(
        self,
        *,
        exploration_probability: float = 0.30,
        max_same_family_streak: int = 6,
        stagnation_patience: int = 3,
        random_seed: int | None = None,
    ) -> None:
        if not 0.0 <= exploration_probability <= 1.0:
            raise ValueError("exploration_probability must be between 0 and 1")
        if max_same_family_streak < 1:
            raise ValueError("max_same_family_streak must be at least 1")
        if stagnation_patience < 1:
            raise ValueError("stagnation_patience must be at least 1")
        self.exploration_probability = exploration_probability
        self.max_same_family_streak = max_same_family_streak
        self.stagnation_patience = stagnation_patience
        self._rng = random.Random(random_seed)

    @staticmethod
    def _history_directions(trace: object) -> list[SearchDirection]:
        return [experiment_direction(experiment) for experiment, _ in getattr(trace, "hist", ()) or ()]

    @staticmethod
    def _tail_streak(values: Sequence[object]) -> int:
        if not values:
            return 0
        last = values[-1]
        count = 0
        for value in reversed(values):
            if value != last:
                break
            count += 1
        return count

    @staticmethod
    def _stagnation_streak(trace: object) -> int:
        count = 0
        for _experiment, feedback in reversed(getattr(trace, "hist", ()) or ()):
            if bool(getattr(feedback, "decision", False)):
                break
            count += 1
        return count

    @staticmethod
    def _best_direction(trace: object, fallback: SearchDirection) -> SearchDirection:
        """Use the latest SOTA-winning direction, not a failed exploration arm."""

        for experiment, feedback in reversed(getattr(trace, "hist", ()) or ()):
            if bool(getattr(feedback, "decision", False)):
                return experiment_direction(experiment)
        return fallback

    def _least_tested_direction(self, factors: Sequence[TestedFactor]) -> SearchDirection:
        counts = Counter(factor.direction for factor in factors)
        minimum = min(counts.get(direction, 0) for direction in SEARCH_DIRECTIONS)
        choices = [direction for direction in SEARCH_DIRECTIONS if counts.get(direction, 0) == minimum]
        return self._rng.choice(choices)

    def decide(self, trace: object, registry_records: Iterable[object] = ()) -> SearchDecision:
        factors = collect_tested_factors(trace, registry_records)
        counts = Counter(factor.direction for factor in factors)
        count_dict = {direction: counts.get(direction, 0) for direction in SEARCH_DIRECTIONS}
        directions = self._history_directions(trace)
        same_streak = self._tail_streak(directions)
        stagnation = self._stagnation_streak(trace)

        if not directions:
            mode: SearchMode = "explore"
            reason = "initial_round"
        elif same_streak >= self.max_same_family_streak:
            mode = "explore"
            reason = "same_direction_limit"
        elif stagnation >= self.stagnation_patience:
            mode = "explore"
            reason = "stagnation_limit"
        elif self._rng.random() < self.exploration_probability:
            mode = "explore"
            reason = "random_exploration"
        else:
            mode = "improve"
            reason = "continue_best_direction"

        direction = (
            self._least_tested_direction(factors)
            if mode == "explore"
            else self._best_direction(trace, directions[-1])
        )
        if direction == "other":
            # Old unclassified work should not turn "other" into a permanent
            # exploitation arm; choose a concrete under-tested direction.
            direction = self._least_tested_direction(factors)
            reason = "unclassified_previous_direction"
        return SearchDecision(mode, direction, reason, same_streak, stagnation, count_dict)
