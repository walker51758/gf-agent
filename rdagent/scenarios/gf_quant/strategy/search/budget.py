"""Unified budget counter and the budget-sufficiency signal φ.

Three remote-call kinds share one counter -- a combination backtest, a factor
remote evaluation and an audit re-run -- because they compete for the same
scarce resource and a search that only counted backtests would overspend as soon
as it started proposing factors.  In the first version only the backtest is
actually spent, but the counter already charges all three so nothing has to
change when the other two come back.

φ = 1 - used / total is a *continuous signal*, not a stop flag: it is read by the
selection bonus, by the depth correction and by the expansion motivation, and it
falls from 1 to 0 as the budget drains.  The counter is per-run: a second run
starts at φ = 1 again, and continuity across runs is carried by the tree's
accumulated statistics instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-run search budget.  The design's budget-sufficiency signal is
# ``φ = 1 - spent / total``, so a larger total makes φ decay more slowly and the
# exploration term stays alive longer; the single source of truth is this
# constant, and every entry point reads it.
DEFAULT_TOTAL_CREDITS = 24

BACKTEST_CREDIT = 1
FACTOR_EVALUATION_CREDIT = 1
AUDIT_CREDIT = 1


@dataclass
class CreditLedger:
    """One search run's spending, by category, against a fixed total."""

    total: int = DEFAULT_TOTAL_CREDITS
    spent: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError("search budget must be at least 1 credit")
        if not 0 <= self.spent <= self.total:
            raise ValueError("spent credits must lie within the budget")

    @property
    def remaining(self) -> int:
        return max(self.total - self.spent, 0)

    @property
    def phi(self) -> float:
        """Budget sufficiency in [0, 1]; 1.0 at the start, 0.0 when exhausted."""

        return max(1.0 - self.spent / self.total, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def can_afford(self, credits: int = BACKTEST_CREDIT) -> bool:
        if credits < 1:
            raise ValueError("a spend must be at least one credit")
        return self.spent + credits <= self.total

    def spend(self, kind: str = "backtest", credits: int = BACKTEST_CREDIT) -> None:
        """Charge the ledger, refusing to overdraw.

        Overdrawing is refused rather than clamped: a clamped ledger would let
        the loop keep spending while φ reads 0, which is precisely the state the
        budget term is supposed to make impossible.
        """

        if not self.can_afford(credits):
            raise RuntimeError(
                f"search budget exhausted: {self.spent}/{self.total} credits spent, "
                f"cannot charge {credits} more for {kind!r}"
            )
        self.spent += credits
        self.by_kind[kind] = self.by_kind.get(kind, 0) + credits

    def as_payload(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "spent": self.spent,
            "remaining": self.remaining,
            "phi": self.phi,
            "by_kind": dict(self.by_kind),
        }
