"""Local candidate-pool filter chain: one ``eligible[t, symbol]`` panel.

Mirrors the worker's ``_build_universe_eligible`` /
``scripts/gf_quant_strategy_worker.py`` value for value; the parity suite
(``test_strategy_worker_parity.py``) runs both on identical injected panels
and fails on any drift.

Roadmap section 3 defines eight filters (3.1 BaseUniverse ... 3.8
DataQualityFilter) that intersect into a single boolean eligibility panel.
Neither engine carries membership, status, limit-price or amount data of its
own, so every data input is injected explicitly (the industry_map pattern):
the panels travel inside the canonical ``universe_rules`` payload section and
are validated by :class:`UniverseSpec`.  The mask is consumed by the signal
front-end (``build_factor_rank_panels``), so excluded names never enter any
cross-sectional statistic -- the rank (roadmap 4.2) sees exactly the
post-filter universe, and a failing name reaches the weight builders as a
missing score through the existing dropna path, never as an order to delete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from rdagent.scenarios.gf_quant.strategy.candidate import (
    DataQualityRule,
    IndexMembershipRule,
    ListingAgeRule,
    LiquidityRule,
    PriceLimitRule,
    PriceRangeRule,
    StatusExclusionRule,
    UniverseSpec,
)

#: Touching the limit price counts as being at the limit; the tolerance only
#: absorbs float representation noise.
PRICE_LIMIT_TOLERANCE = 1e-9


def panel_from_payload(data: Mapping[str, Mapping[str, object]], *, dtype: str) -> pd.DataFrame:
    """Dense panel from a sparse ``{date: {symbol: value}}`` payload panel.

    Dates are parsed and normalised to midnight, symbols are sorted; the
    panel's own dates are the only rows it contributes.  ``dtype`` is one of
    ``"float"`` (number panels), ``"bool"`` (membership, ``False`` fill) or
    ``"object"`` (status strings, ``None`` fill).
    """

    dates = sorted({pd.Timestamp(key).normalize() for key in data})
    symbols = sorted({str(symbol) for row in data.values() for symbol in row})
    if dtype == "bool":
        frame = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=symbols, dtype=bool)
    elif dtype == "object":
        frame = pd.DataFrame(None, index=pd.DatetimeIndex(dates), columns=symbols, dtype=object)
    else:
        frame = pd.DataFrame(float("nan"), index=pd.DatetimeIndex(dates), columns=symbols, dtype=float)
    for key, row in data.items():
        day = pd.Timestamp(key).normalize()
        for symbol, value in row.items():
            frame.at[day, symbol] = value
    return frame


def build_eligible_mask(
    spec: UniverseSpec | None,
    *,
    factor_panels: Mapping[str, pd.DataFrame],
    index: pd.Index,
    columns: Sequence[str],
) -> pd.DataFrame | None:
    """Intersect every enabled filter into one boolean panel.

    ``index`` holds the decision (rebalance) dates and ``columns`` the
    candidate symbols; the returned frame shares both and is ``True`` where
    the name may participate.  ``None`` (and no computation at all) when no
    filter is enabled -- the disabled path stays bit for bit identical.
    Filters apply in roadmap order 3.1..3.8; intersection makes the order
    semantically inert, but the quantile leg of the liquidity filter reads
    only the data-available cross-section, never a partially filtered one.
    """

    if spec is None or not spec.enabled:
        return None
    idx = pd.DatetimeIndex(index)
    cols = list(columns)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    calendar = (
        pd.DatetimeIndex([pd.Timestamp(day).normalize() for day in spec.trading_dates])
        if spec.trading_dates is not None
        else None
    )
    price = panel_from_payload(spec.price, dtype="float") if spec.price is not None else None
    if spec.listing_age is not None or spec.liquidity is not None or spec.price_range is not None:
        _require_calendar_coverage(calendar, idx)
    if spec.base_universe is not None:
        members = set(spec.base_universe.symbols)
        mask &= pd.DataFrame([[symbol in members for symbol in cols]] * len(idx), index=idx, columns=cols)
    if spec.index_membership is not None:
        mask &= _membership_mask(spec.index_membership, idx, cols)
    if spec.listing_age is not None:
        mask &= _listing_age_mask(spec.listing_age, calendar, idx, cols)
    if spec.status_exclusion is not None:
        mask &= _status_mask(spec.status_exclusion, idx, cols)
    if spec.price_limit is not None:
        mask &= _price_limit_mask(spec.price_limit, price, idx, cols)
    if spec.liquidity is not None:
        mask &= _liquidity_mask(spec.liquidity, calendar, idx, cols)
    if spec.price_range is not None:
        mask &= _price_range_mask(spec.price_range, price, calendar, idx, cols)
    if spec.data_quality is not None:
        mask &= _data_quality_mask(spec.data_quality, price, factor_panels, idx, cols)
    return mask


def _require_calendar_coverage(calendar: pd.DatetimeIndex | None, index: pd.DatetimeIndex) -> None:
    """A history-dependent filter needs every decision date on the calendar.

    A decision date outside (or simply not on) the injected calendar makes
    its trailing windows ill-defined; that is a data mismatch, not an
    eligibility question, so it fails closed instead of silently excluding
    the whole date.
    """

    assert calendar is not None
    missing = index.difference(calendar)
    if len(missing):
        raise ValueError(f"universe trading_dates do not contain decision date {missing[0].date().isoformat()}")


def _membership_mask(rule: IndexMembershipRule, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    panel = panel_from_payload(rule.membership, dtype="bool")
    if rule.lag_days > 0:
        panel = panel.shift(rule.lag_days).fillna(False).astype(bool)
    if rule.exclude_recent_days > 0:
        excluded = pd.DataFrame(False, index=panel.index, columns=panel.columns)
        for symbol in panel.columns:
            member = panel[symbol]
            inclusion = member & ~member.shift(1, fill_value=False).astype(bool)
            run = inclusion.cumsum()
            since = run.groupby(run).cumcount()
            # Group 0 is the pre-existing membership stretch at the panel
            # start -- never a "new" inclusion -- so run > 0 guards it.
            excluded[symbol] = member & (run > 0) & (since < rule.exclude_recent_days)
        panel = panel & ~excluded
    combined = panel.index.union(index)
    asof = panel.reindex(combined).ffill().loc[index]
    return asof.reindex(columns=columns).fillna(False).astype(bool)


def _listing_age_mask(rule: ListingAgeRule, calendar: pd.DatetimeIndex, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    position = calendar.searchsorted(index, side="right")
    mask = pd.DataFrame(False, index=index, columns=columns)
    for symbol in columns:
        day = rule.listing_dates.get(symbol)
        if day is None:
            continue
        listed = pd.Timestamp(day).normalize()
        # Trailing trading days strictly after the listing day: the listing
        # day itself is day 0, so the boundary date is exactly the
        # min_days-th trading day after it.
        mask[symbol] = (position - calendar.searchsorted(listed, side="right")) >= rule.min_days
    return mask


def _status_mask(rule: StatusExclusionRule, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    panel = panel_from_payload(rule.statuses, dtype="object")
    # Sparse panel: only abnormal states are recorded, so the lookup is
    # exact-date by construction and an absent entry means "normal".
    at = panel.reindex(index=index).reindex(columns=columns)
    return ~at.isin(set(rule.excluded_statuses))


def _price_limit_mask(rule: PriceLimitRule, price: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    close = price.reindex(index=index).reindex(columns=columns)
    excluded = pd.DataFrame(False, index=index, columns=columns)
    if rule.exclude_limit_up:
        limit = panel_from_payload(rule.limit_up, dtype="float").reindex(index=index).reindex(columns=columns)
        excluded |= close.ge(limit - PRICE_LIMIT_TOLERANCE)
        excluded |= close.isna() | limit.isna()
    if rule.exclude_limit_down:
        limit = panel_from_payload(rule.limit_down, dtype="float").reindex(index=index).reindex(columns=columns)
        excluded |= close.le(limit + PRICE_LIMIT_TOLERANCE)
        excluded |= close.isna() | limit.isna()
    return ~excluded


def _liquidity_mask(rule: LiquidityRule, calendar: pd.DatetimeIndex, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    amount = panel_from_payload(rule.amount, dtype="float")
    aligned = amount.reindex(index=calendar).reindex(columns=columns)
    rolling = aligned.rolling(rule.window, min_periods=1).mean()
    position = calendar.get_indexer(index)
    rows = rolling.iloc[position].set_axis(index, axis=0)
    ok = pd.DataFrame(True, index=index, columns=columns)
    if rule.min_amount is not None:
        ok &= rows.ge(float(rule.min_amount))
    if rule.quantile is not None:
        threshold = rows.quantile(float(rule.quantile), axis=1)
        leg = rows.ge(threshold, axis=0)
        # A date without any valid rolling mean has no quantile to compare
        # against; the quantile leg passes there and the min_amount / missing
        # legs keep their say.
        ok &= leg.where(threshold.notna(), True)
    if rule.missing == "exclude":
        ok &= rows.notna()
    else:
        ok |= rows.isna()
    return ok


def _price_range_mask(
    rule: PriceRangeRule,
    price: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    index: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    close = price.reindex(index=calendar).reindex(columns=columns)
    position = calendar.get_indexer(index)
    low = float("-inf") if rule.min_price is None else float(rule.min_price)
    high = float("inf") if rule.max_price is None else float(rule.max_price)
    ok = np.ones((len(index), len(columns)), dtype=bool)
    for offset in range(rule.consecutive_days):
        source = position - offset
        has_row = source >= 0
        window = close.iloc[np.clip(source, 0, max(len(close) - 1, 0))].to_numpy(dtype=float)
        ok &= (~np.isnan(window)) & (window >= low) & (window <= high) & has_row[:, None]
    return pd.DataFrame(ok, index=index, columns=columns)


def _data_quality_mask(
    rule: DataQualityRule,
    price: pd.DataFrame | None,
    factor_panels: Mapping[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    columns: list[str],
) -> pd.DataFrame:
    ok = pd.DataFrame(True, index=index, columns=columns)
    if rule.non_positive_price:
        assert price is not None  # guaranteed by UniverseSpec validation
        close = price.reindex(index=index).reindex(columns=columns)
        ok &= close.gt(0)
    if rule.min_coverage is None and rule.max_missing_run is None:
        return ok
    if not factor_panels:
        raise ValueError("universe data_quality coverage requires factor panels")
    union = pd.DatetimeIndex(index)
    for frame in factor_panels.values():
        union = union.union(pd.DatetimeIndex(frame.index))
    union = union.sort_values()
    missing = [
        frame.reindex(index=union).reindex(columns=columns).isna() for frame in factor_panels.values()
    ]
    if rule.min_coverage is not None:
        miss_count = missing[0].astype(int)
        for miss in missing[1:]:
            miss_count = miss_count + miss.astype(int)
        windowed = miss_count.rolling(rule.window, min_periods=1).sum()
        rows_in_window = np.minimum(rule.window, np.arange(1, len(union) + 1))
        coverage = 1 - windowed.div(pd.Series(len(missing) * rows_in_window, index=union), axis=0)
        ok &= coverage.loc[index].ge(float(rule.min_coverage))
    if rule.max_missing_run is not None:
        bad = pd.DataFrame(False, index=union, columns=columns)
        for miss in missing:
            for symbol in columns:
                column = miss[symbol].astype(bool)
                group = (~column).cumsum()
                run = column.groupby(group).cumsum()
                bad[symbol] |= run.gt(rule.max_missing_run)
        ok &= ~bad.loc[index]
    return ok
