"""Memory-bounded multi-signal factor de-duplication."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def normalize_expression(expression: str) -> str:
    try:
        return ast.dump(ast.parse(expression.strip(), mode="eval"), annotate_fields=True, include_attributes=False)
    except SyntaxError:
        return re.sub(r"\s+", "", expression)


def expression_hash(expression: str) -> str:
    return hashlib.sha256(normalize_expression(expression).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactorSignature:
    factor_id: str
    expression: str
    values: pd.Series | None = None
    return_profile: pd.Series | None = None


@dataclass(frozen=True)
class DedupResult:
    is_duplicate: bool
    duplicate_of: str | None
    reason: str
    expression_match: bool = False
    pearson: float | None = None
    spearman: float | None = None
    return_correlation: float | None = None
    overlap: int = 0


class FactorDeduplicator:
    def __init__(self, threshold: float = 0.88, min_overlap: int = 100) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if min_overlap < 2:
            raise ValueError("min_overlap must be at least 2")
        self.threshold = threshold
        self.min_overlap = min_overlap

    @staticmethod
    def _corr(left: pd.Series | None, right: pd.Series | None, *, rank: bool = False) -> tuple[float | None, int]:
        if left is None or right is None:
            return None, 0
        pair = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
        if pair.empty or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
            return None, len(pair)
        if rank:
            pair = pair.rank(method="average")
        return float(pair["left"].corr(pair["right"])), len(pair)

    def compare(self, candidate: FactorSignature, existing: Iterable[FactorSignature]) -> DedupResult:
        candidate_hash = expression_hash(candidate.expression)
        for other in existing:  # stream one reference at a time; no NxN matrix
            if candidate_hash == expression_hash(other.expression):
                return DedupResult(True, other.factor_id, "normalized_expression_match", expression_match=True)

            pearson, overlap = self._corr(candidate.values, other.values)
            spearman, rank_overlap = self._corr(candidate.values, other.values, rank=True)
            return_corr, return_overlap = self._corr(candidate.return_profile, other.return_profile)
            sufficient_value_overlap = max(overlap, rank_overlap) >= self.min_overlap
            sufficient_return_overlap = return_overlap >= self.min_overlap
            if sufficient_value_overlap and pearson is not None and abs(pearson) >= self.threshold:
                return DedupResult(True, other.factor_id, "absolute_pearson", pearson=pearson, spearman=spearman, return_correlation=return_corr, overlap=overlap)
            if sufficient_value_overlap and spearman is not None and abs(spearman) >= self.threshold:
                return DedupResult(True, other.factor_id, "absolute_spearman", pearson=pearson, spearman=spearman, return_correlation=return_corr, overlap=rank_overlap)
            if sufficient_return_overlap and return_corr is not None and abs(return_corr) >= self.threshold:
                return DedupResult(True, other.factor_id, "return_profile_correlation", pearson=pearson, spearman=spearman, return_correlation=return_corr, overlap=return_overlap)
        return DedupResult(False, None, "no_duplicate_signal")
