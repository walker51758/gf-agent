"""Deterministic portfolio construction from cross-sectional scores.

The composition layer is a pure function: ``score``/``eligible`` panels in,
target weights out.  No I/O, no calendar inference and no look-ahead -- the
caller decides which dates are rebalance dates.  Panel orientation matches the
remote worker exactly (``index=dt`` trading dates, ``columns=symbol``).
"""

from __future__ import annotations

import pandas as pd
from rdagent.scenarios.gf_quant.strategy.candidate import CompositionSpec


def _scheme_weights(spec: CompositionSpec, scores: pd.Series) -> pd.Series:
    """Unconstrained weights for the selected names, summing to one."""

    count = len(scores)
    if spec.weighting == "equal":
        return pd.Series(1.0 / count, index=scores.index)
    if spec.weighting == "rank":
        # Rank 1 (best score) gets the largest weight; ties are impossible here
        # because selection already ordered equal scores deterministically.
        ranks = pd.Series(range(count, 0, -1), index=scores.index, dtype=float)
        return ranks / ranks.sum()
    # Proportional weighting (free-cash-flow-rate style) only makes sense for
    # non-negative scores; a negative score would silently become a negative
    # position, so fail closed instead.
    if (scores <= 0).any():
        raise ValueError("proportional weighting requires strictly positive scores")
    return scores / scores.sum()


def _apply_weight_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Cap single holdings and redistribute the excess proportionally.

    Redistribution can push another name above the cap, so capped names are
    frozen one round at a time and only the remaining names absorb the excess.
    Each round freezes at least one name, which bounds the loop by the number
    of holdings and keeps the result deterministic.  When
    ``len(weights) * cap < 1`` the book cannot be fully invested inside the cap:
    the residual is deliberately left as uninvested cash rather than forced
    into already-capped names.
    """

    if (weights <= cap).all():
        return weights
    result = weights.astype(float).copy()
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        over = ~frozen & (result > cap)
        if not over.any():
            break
        frozen |= over
        excess = float((result - cap)[over].sum())
        result[over] = cap
        free = ~frozen
        room = float(result[free].sum())
        if not free.any() or room <= 0:
            break
        result[free] = result[free] + excess * result[free] / room
    return result.clip(upper=cap)


def build_target_weights(
    spec: CompositionSpec,
    score: pd.DataFrame,
    *,
    eligible: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build target weights (one row per rebalance date) from a score panel.

    Parameters
    ----------
    spec:
        Selection and weighting rule.
    score:
        Cross-sectional score panel; a higher score means a stronger
        intention to hold the name.  ``NaN`` excludes the name on that date.
    eligible:
        Optional boolean panel of the same orientation marking investable
        names (liquidity / industry / PIT screens).  Missing entries are
        treated as not eligible.

    Returns
    -------
    ``pd.DataFrame`` with the same index and columns as ``score`` holding the
    target weights.  Unselected names are ``0.0`` so downstream turnover is
    computable without re-deriving the selection.
    """

    if score.empty:
        raise ValueError("score panel is empty")
    if eligible is not None:
        missing_dates = score.index.difference(eligible.index)
        if len(missing_dates):
            raise ValueError(f"eligible panel is missing score dates: {[str(date) for date in missing_dates[:3]]}")
        extra_columns = eligible.columns.difference(score.columns)
        if len(extra_columns):
            raise ValueError(f"eligible panel has unknown symbols: {list(extra_columns)[:3]}")

    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for date, row in score.iterrows():
        candidates = row
        if eligible is not None:
            mask = eligible.loc[date].reindex(row.index)
            candidates = row.where(mask.fillna(False).astype(bool))
        candidates = candidates.dropna()
        if candidates.empty:
            continue
        # Deterministic ordering: best score first, ties broken by symbol so a
        # re-run on the same panel always selects the same names.
        ordered = (
            pd.DataFrame({"score": candidates})
            .rename_axis("symbol")
            .reset_index()
            .sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
        )
        selected = ordered.head(spec.top_n).set_index("symbol")["score"]
        row_weights = _scheme_weights(spec, selected)
        if spec.weight_cap is not None:
            row_weights = _apply_weight_cap(row_weights, spec.weight_cap)
        weights.loc[date, selected.index] = row_weights.to_numpy()
    return weights
