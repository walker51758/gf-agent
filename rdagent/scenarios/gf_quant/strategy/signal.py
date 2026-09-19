"""Local signal front-end: per-date cross-sectional transforms before ranking.

Mirrors the worker's signal stage (``scripts/gf_quant_strategy_worker.py``:
``_apply_transform`` / ``_apply_winsorize`` / ``_apply_standardize`` /
``_apply_missing_value`` / ``_apply_neutral_rank_fill`` / ``_build_rank_panels``
/ ``_interaction_rank_panels`` and the rank+blend in ``run_request``) value for
value; the parity suite (``test_strategy_worker_parity.py``) runs both on
identical requests and fails on any drift.  Every stage is a pure per-date
function over a wide panel (``index=dt`` trading dates, ``columns=symbol``)
with the same failure semantics: invalid parameters are rejected at spec
validation, invalid *data* (log/sqrt of out-of-domain values, too-thin
cross-sections, near-zero dispersion, over-wide fills) raise at run time
instead of being silently repaired.

Stage order (roadmap order, batch B1): transform -> winsorize -> standardize
-> missing-value policy (value space) -> direction map -> percentile rank ->
neutral-rank fill (rank space) -> interaction terms (rank space) -> blend
(flat fixed weights or two-level group weights).

Batch C2 inserts one value-space stage between the missing-value/eligible
stage and the direction map: IndustryNeutralize (roadmap 4.6, within-industry
de-mean or regression residual) followed by StyleNeutralize (roadmap 4.7,
cross-sectional residual on the injected style exposures), and one blend-space
operator: RegimeConditionalSignal (roadmap 4.12) replaces the static blend
with a per-date, state-conditional equal-weight blend.
"""

from __future__ import annotations

import bisect

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from rdagent.scenarios.gf_quant.strategy.candidate import SignalSpec
from rdagent.scenarios.gf_quant.strategy.execution import factor_decay_scales

# Reserved group for symbols the injected industry mapping does not cover
# (same convention as the composition layer's ``_UNKNOWN_INDUSTRY``).
_NO_INDUSTRY = "__no_industry__"


def apply_transform(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """FactorTransform (roadmap 4.8): a monotone transform of raw factor values."""

    if spec.transform is None:
        return values
    if spec.transform == "linear":
        return values * spec.transform_scale + spec.transform_shift
    if spec.transform == "log":
        if bool((values <= 0).any().any()):
            raise ValueError("log transform requires strictly positive factor values")
        return np.log(values)
    if spec.transform == "signed_log":
        return np.sign(values) * np.log1p(np.abs(values))
    if spec.transform == "sqrt":
        if bool((values < 0).any().any()):
            raise ValueError("sqrt transform requires non-negative factor values")
        return np.sqrt(values)
    raise ValueError(f"unsupported signal transform: {spec.transform!r}")


def apply_winsorize(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """Winsorize (roadmap 4.3): clip each date's cross-section at its quantiles.

    The clip boundaries come only from the valid (non-NaN) values of the same
    date; fewer valid values than ``winsorize_min_samples`` fails closed.
    """

    if spec.winsorize_lower is None:
        return values
    counts = values.notna().sum(axis=1)
    if bool((counts < spec.winsorize_min_samples).any()):
        raise ValueError(
            f"winsorize requires at least {spec.winsorize_min_samples} valid cross-sectional values on every date"
        )
    lower = values.quantile(spec.winsorize_lower, axis=1)
    upper = values.quantile(spec.winsorize_upper, axis=1)
    return values.clip(lower=lower, upper=upper, axis=0)


def apply_standardize(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """CrossSectionalStandardize (roadmap 4.4): z-score, rank-z or robust z-score.

    The dispersion denominator (std for z/rank-z, 1.4826 * MAD for robust z)
    must be finite and at least ``standardize_min_std`` on every date;
    constant or single-value cross-sections fail closed instead of exploding.
    """

    if spec.standardize is None:
        return values
    if spec.standardize == "rank_z":
        values = values.rank(axis=1, pct=True, method="average")
    if spec.standardize == "robust_z":
        center = values.median(axis=1)
        scale = values.sub(center, axis=0).abs().median(axis=1) * 1.4826
    else:
        center = values.mean(axis=1)
        scale = values.std(axis=1)
    scale = scale.astype(float)
    if bool((~np.isfinite(scale)).any()) or bool((scale < spec.standardize_min_std).any()):
        raise ValueError(
            f"cross-sectional standardize requires a dispersion of at least {spec.standardize_min_std:g} on every date"
        )
    return values.sub(center, axis=0).div(scale, axis=0)


def apply_signal(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """Run the full per-date signal pipeline in roadmap order on one panel."""

    return apply_standardize(apply_winsorize(apply_transform(values, spec), spec), spec)


def apply_missing_value(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """MissingValuePolicy (roadmap 4.5), value-space half.

    ``drop_stock`` removes every symbol that has at least one missing value in
    the panel (the roadmap names no threshold, so "any missing" is the
    conservative reading); ``cross_section_median`` fills each missing cell
    with the median of that date's valid cross-section -- after the A1 signal
    stages and before the direction map, on both sides.  The neutral-rank
    policy fills in rank space and is applied by
    :func:`apply_neutral_rank_fill` after ranking.  When
    ``missing_max_fill_ratio`` is set, a date that would need more than that
    share of the cross-section filled fails closed.
    """

    if spec.missing_value is None:
        return values
    if spec.missing_value == "drop_stock":
        keep = [column for column in values.columns if bool(values[column].notna().all())]
        if not keep:
            raise ValueError("drop_stock removed every symbol from a factor panel")
        return values[keep]
    if spec.missing_max_fill_ratio is not None:
        ratio = values.isna().sum(axis=1) / len(values.columns)
        if bool((ratio > spec.missing_max_fill_ratio).any()):
            raise ValueError(
                f"missing_value fill exceeds {spec.missing_max_fill_ratio:g} of the cross-section on some date"
            )
    if spec.missing_value == "cross_section_median":
        # A fully-missing date keeps its NaN cells (median of nothing), which
        # the blend's any-missing-kills mask consumes unchanged.
        return values.where(values.notna(), values.median(axis=1), axis=0)
    return values


def _industry_demean(values: pd.DataFrame, labels: Mapping[str, str], min_group: int) -> pd.DataFrame:
    """De-mean ``values`` within the injected industry groups, per date.

    Group means are computed over the valid cells only; missing cells stay
    missing.  A group with fewer than ``min_group`` valid values on a date is
    passed through unchanged -- the declared fallback for thin industries (the
    roadmap explicitly demands a fallback, so a thin group never fails the
    whole date).
    """

    result = values.copy()
    groups: dict[str, list[str]] = {}
    for symbol in values.columns:
        groups.setdefault(labels.get(symbol) or _NO_INDUSTRY, []).append(symbol)
    for label in sorted(groups):
        members = groups[label]
        block = values[members]
        ok = block.notna().sum(axis=1) >= min_group
        result[members] = block.sub(block.mean(axis=1), axis=0).where(ok, block)
    return result


def _cross_section_demean(series: pd.Series, labels: Mapping[str, str], min_group: int) -> pd.Series:
    """One-shot (no time dimension) industry de-mean of an exposure column.

    The thin-group pass-through applies with the group size judged once over
    the whole cross-section, mirroring the panel version's rule.
    """

    groups: dict[str, list[str]] = {}
    for symbol in series.index:
        groups.setdefault(labels.get(symbol) or _NO_INDUSTRY, []).append(symbol)
    result = series.copy()
    for label in sorted(groups):
        members = groups[label]
        block = series[members]
        if int(block.notna().sum()) < min_group:
            continue
        result[members] = block - block.mean()
    return result


def _cross_sectional_residual(values: pd.DataFrame, exposure_matrix: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """Per-date least-squares residual of ``values`` on the exposure columns.

    Rows missing any exposure keep their value (listwise deletion); a date
    with fewer joint-valid rows than ``max(min_rows, n_exposures + 1)`` keeps
    the whole row unchanged (the declared thin-sample fallback -- a sample too
    thin to identify the regression cannot produce a trustworthy residual).
    Both the values and the exposures are de-meaned across the regression
    sample first, so the residual has zero cross-sectional mean by
    construction (the implicit intercept).  ``numpy.linalg.lstsq`` keeps the
    solve deterministic and identical on both engine sides.
    """

    result = values.copy()
    if exposure_matrix is None or exposure_matrix.empty:
        return result
    needed = max(int(min_rows), exposure_matrix.shape[1] + 1)
    valid_rows = exposure_matrix.notna().all(axis=1)
    for date in values.index:
        row = values.loc[date]
        joint = row.index[row.notna() & valid_rows]
        if len(joint) < needed:
            continue
        y = row.loc[joint].to_numpy(dtype=float)
        x = exposure_matrix.loc[joint].to_numpy(dtype=float)
        y_demeaned = y - float(y.mean())
        x_demeaned = x - x.mean(axis=0)
        beta, _, _, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
        fitted = x_demeaned @ beta
        if not bool(np.all(np.isfinite(fitted))):
            raise ValueError("neutralization regression produced a non-finite fit")
        result.loc[date, joint] = y_demeaned - fitted
    return result


def _exposure_matrix(exposures: Mapping[str, Mapping[str, float]], values: pd.DataFrame) -> pd.DataFrame:
    """Wide (symbol x exposure) frame of the injected per-symbol exposures."""

    frame = pd.DataFrame(
        {name: pd.Series({symbol: float(number) for symbol, number in mapping.items()}) for name, mapping in exposures.items()}
    )
    return frame.reindex(values.columns)


def apply_industry_neutralize(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """IndustryNeutralize (roadmap 4.6), value space.

    ``demean`` de-means the panel within the injected industry groups;
    ``regression`` first de-means within industries and then takes the
    residual of the industry-demeaned values on the industry-demeaned
    exposures (Frisch-Waugh: the residual of a regression on industry dummies
    plus exposures, so with no exposures configured it is exactly the
    de-mean).  Symbols missing from the mapping share the reserved
    ``_NO_INDUSTRY`` group, which is neutralized like any other industry.
    Disabled, the panel passes through untouched bit for bit.
    """

    if not spec.industry_neutralize:
        return values
    industry_map = spec.industry_neutralize_industry_map or {}
    labels = {symbol: industry_map.get(symbol) or _NO_INDUSTRY for symbol in values.columns}
    demeaned = _industry_demean(values, labels, int(spec.industry_neutralize_min_group))
    if spec.industry_neutralize_method == "regression" and spec.industry_neutralize_exposures:
        exposure_matrix = _exposure_matrix(spec.industry_neutralize_exposures, values)
        for name in exposure_matrix.columns:
            exposure_matrix[name] = _cross_section_demean(exposure_matrix[name], labels, int(spec.industry_neutralize_min_group))
        return _cross_sectional_residual(demeaned, exposure_matrix, int(spec.industry_neutralize_min_group))
    return demeaned


def apply_style_neutralize(values: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """StyleNeutralize (roadmap 4.7), value space.

    Residual of every date's cross-section on the injected per-symbol style
    exposures (size/beta/momentum style columns as explicit point-in-time
    configuration inputs).  Applied after the industry stage when both are
    enabled, so the style regression neutralizes the industry-neutralized
    values.  Disabled, the panel passes through untouched bit for bit.
    """

    if not spec.style_neutralize:
        return values
    exposure_matrix = _exposure_matrix(spec.style_neutralize_exposures or {}, values)
    return _cross_sectional_residual(values, exposure_matrix, int(spec.style_neutralize_min_samples))


def _regime_state_series(states: Mapping[str, str], lag: int) -> tuple[list[pd.Timestamp], list[str]]:
    """The lag-shifted state series: the state observed at row ``i`` of the
    injected (date-sorted) series becomes effective ``lag`` rows later, the
    same convention as the membership panel's ``lag_days`` shift."""

    parsed = sorted((pd.Timestamp(date_key), str(label)) for date_key, label in states.items())
    dates = [date for date, _ in parsed]
    labels = [label for _, label in parsed]
    if lag <= 0:
        return dates, labels
    if lag >= len(dates):
        return [], []
    return dates[lag:], labels[:-lag]


def _regime_label_for(date: pd.Timestamp, effective_dates: list[pd.Timestamp], effective_labels: list[str]) -> str:
    position = bisect.bisect_right(effective_dates, date) - 1
    if position < 0:
        raise ValueError(f"regime state is missing for decision date {date:%Y-%m-%d}")
    return effective_labels[position]


def blend_regime_panels(ranks: Mapping[str, pd.DataFrame], spec: SignalSpec) -> pd.DataFrame:
    """RegimeConditionalSignal (roadmap 4.12): per-date state-conditional blend.

    On every decision date the injected PIT state (read as-of the date, after
    the ``regime_lag`` row shift) selects one factor group; the score row is
    the equal-weight blend of that group's rank panels with the strict
    missing contract -- a cell is valid only when every member of the date's
    group has a value there.  A date without a resolvable state fails closed
    (the state is a required operator input; defaulting one would fabricate
    information), and so does a state label the group mapping does not define.
    """

    groups = spec.regime_factor_groups or {}
    effective_dates, effective_labels = _regime_state_series(spec.regime_states or {}, int(spec.regime_lag))
    first = next(iter(ranks.values()))
    result = pd.DataFrame(np.nan, index=first.index, columns=first.columns)
    for date in first.index:
        label = _regime_label_for(date, effective_dates, effective_labels)
        members = groups.get(label)
        if members is None:
            raise ValueError(f"regime_factor_groups does not define state {label!r}")
        panels = [ranks[name].loc[date] for name in members]
        total = panels[0].copy()
        valid = panels[0].notna()
        for panel in panels[1:]:
            total = total.add(panel, fill_value=0.0)
            valid &= panel.notna()
        result.loc[date] = (total / len(panels)).where(valid)
    return result


def apply_neutral_rank_fill(ranks: pd.DataFrame, spec: SignalSpec) -> pd.DataFrame:
    """MissingValuePolicy (roadmap 4.5), rank-space half.

    With ``missing_value="neutral_rank"`` every missing percentile rank is
    replaced by the neutral 0.5, so a filled name neither benefits nor suffers
    from the missing value regardless of the cross-section's shape.  The fill
    ratio guard sees the same pre-fill NaN count as the value-space policies.
    """

    if spec.missing_value != "neutral_rank":
        return ranks
    if spec.missing_max_fill_ratio is not None:
        ratio = ranks.isna().sum(axis=1) / len(ranks.columns)
        if bool((ratio > spec.missing_max_fill_ratio).any()):
            raise ValueError(
                f"missing_value fill exceeds {spec.missing_max_fill_ratio:g} of the cross-section on some date"
            )
    return ranks.where(ranks.notna(), 0.5)


def interaction_rank_panels(
    ranks: Mapping[str, pd.DataFrame],
    spec: SignalSpec,
) -> dict[str, pd.DataFrame]:
    """FactorInteraction (roadmap 4.11): one extra blend member per term.

    Each term combines the *final* post-direction percentile ranks of its two
    factors (product, elementwise minimum or geometric mean) and is itself
    re-ranked to a percentile, so it enters the blend on exactly the same
    scale as every factor.  Inputs stay in rank space -- the conservative
    reading of the roadmap's "rank first" parameter, which avoids the noise
    amplification the roadmap warns about for raw z-score products.  A NaN in
    either input keeps the combined cell NaN, which the blend's
    any-missing-kills mask consumes (the missing-value contract of the
    worker's ``_score``).
    """

    panels: dict[str, pd.DataFrame] = {}
    for first, second, kind in spec.interaction_pairs:
        left, right = ranks[first], ranks[second]
        if kind == "product":
            combined = left * right
        elif kind == "min":
            # np.minimum propagates NaN elementwise; pandas min would skip it.
            combined = pd.DataFrame(np.minimum(left, right), index=left.index, columns=left.columns)
        else:  # geometric_mean
            combined = np.sqrt(left * right)
        panels[f"{first}*{second}"] = combined.rank(axis=1, pct=True, method="average")
    return panels


def _weighted_blend(weighted: Sequence[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """Blend (panel, weight) pairs; any member missing on a cell kills it.

    The weights are normalised by their sum.  With all weights 1.0 this is
    numerically identical to :func:`blend_rank_panels`.
    """

    total: pd.DataFrame | None = None
    valid: pd.DataFrame | None = None
    for panel, weight in weighted:
        term = panel * float(weight)
        if total is None:
            total, valid = term, panel.notna()
        else:
            total = total.add(term, fill_value=0.0)
            valid = valid & panel.notna()
    assert total is not None and valid is not None
    normalizer = float(sum(weight for _, weight in weighted))
    return (total / normalizer).where(valid)


def blend_member_panels(
    ranks: Mapping[str, pd.DataFrame],
    names: Sequence[str],
    spec: SignalSpec,
    *,
    decay: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Blend factor and interaction rank panels per the signal spec.

    Without blend configuration this is the untouched equal-weight
    :func:`blend_rank_panels` over the factors plus every interaction term.
    ``blend_weights`` (roadmap 4.9 FactorBlend) allocates fixed frozen weights
    (normalised to sum one) over exactly the same members.
    ``blend_groups`` (roadmap 4.10 FactorGroupBlend) averages within groups
    first (``blend_within_weights``, default equal) and then across the group
    scores plus every interaction term as a singleton member
    (``blend_group_weights``, default equal).  All three preserve the strict
    missing contract: a cell is valid only when *every* factor -- and hence
    every group and interaction term -- has a value there.

    ``decay`` (roadmap 9.5 FactorDecayMonitor) maps a factor name to its
    per-decision-date scale; that factor's rank panel is multiplied by its
    scale before the blend, so a decaying factor contributes less while the
    rest of the blend keeps its defined weights (the factor is scaled, never
    replaced, and no member is dropped).  ``None`` -- the disabled default --
    leaves every panel untouched.
    """

    interactions = interaction_rank_panels(ranks, spec)
    merged = {**ranks, **interactions}
    members = list(names) + list(interactions)
    if decay:
        for name in names:
            scale = decay.get(name)
            if scale is not None:
                merged[name] = merged[name].mul(scale.reindex(merged[name].index).astype(float), axis=0)
    if spec.regime_factor_groups is not None:
        # RegimeConditionalSignal (roadmap 4.12) replaces the static blend
        # outright; spec validation keeps it mutually exclusive with every
        # blend/interaction configuration, so nothing else can be set here.
        return blend_regime_panels({name: merged[name] for name in ranks}, spec)
    if spec.blend_weights is None and spec.blend_groups is None:
        return blend_rank_panels(merged, members)
    if spec.blend_weights is not None:
        return _weighted_blend([(merged[name], spec.blend_weights[name]) for name in members])
    assert spec.blend_groups is not None
    top: list[tuple[pd.DataFrame, float]] = []
    for group, factors in spec.blend_groups.items():
        within = (spec.blend_within_weights or {}).get(group)
        weighted = [(merged[factor], (within[factor] if within else 1.0)) for factor in factors]
        top.append((_weighted_blend(weighted), (spec.blend_group_weights or {}).get(group, 1.0)))
    for panel in interactions.values():
        top.append((panel, 1.0))
    return _weighted_blend(top)


def blend_rank_panels(ranks: Mapping[str, pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    """Equal blend of per-factor percentile ranks; any missing factor kills the cell.

    Verbatim mirror of the worker's ``_score``: a symbol/date is valid only
    when *every* factor panel has a value there.
    """

    panels = [ranks[name] for name in names]
    total = panels[0].copy()
    valid = panels[0].notna()
    for panel in panels[1:]:
        total = total.add(panel, fill_value=0.0)
        valid &= panel.notna()
    return (total / len(panels)).where(valid)


def build_factor_rank_panels(
    panels: Mapping[str, pd.DataFrame],
    directions: Mapping[str, int],
    spec: SignalSpec,
    *,
    index: pd.Index,
    columns: pd.Index,
    eligible: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Direction map + signal pipeline + percentile rank for every panel.

    Every panel is first reindexed onto ``(index, columns)`` -- the caller's
    rebalance dates and the common symbol set -- so the cross-sectional
    statistics see exactly the same universe as the worker does.  The gating
    layer (``rdagent.scenarios.gf_quant.strategy.gating``) consumes these same
    panels, so the gate thresholds see exactly what the blend sees.

    ``eligible`` (roadmap section 3 candidate-pool filters) is the shared
    boolean panel produced by ``universe.build_eligible_mask``; it is applied
    twice, once after the value-space missing-value stage (so the percentile
    rank -- roadmap 4.2 -- is computed over the post-filter universe only)
    and once after the rank-space neutral fill (which would otherwise
    resurrect an excluded name at the neutral 0.5).  ``None`` -- the disabled
    default -- leaves the pipeline untouched bit for bit.
    """

    if not panels:
        raise ValueError("factor panel mapping is empty")
    ranks: dict[str, pd.DataFrame] = {}
    for name, frame in panels.items():
        aligned = frame.reindex(index=index, columns=columns)
        values = apply_missing_value(apply_signal(aligned, spec), spec)
        if eligible is not None:
            values = values.where(eligible)
        # Batch C2 value-space stage: neutralize the post-filter cross-section
        # (the same universe the percentile rank sees), before the direction
        # map.  Both helpers are no-ops while disabled, so the disabled
        # pipeline is bit-for-bit the pre-change one.
        values = apply_industry_neutralize(values, spec)
        values = apply_style_neutralize(values, spec)
        panel = (int(directions[name]) * values).rank(axis=1, pct=True, method="average")
        panel = apply_neutral_rank_fill(panel, spec)
        if eligible is not None:
            panel = panel.where(eligible)
        ranks[name] = panel
    return ranks


def factor_decay_weights(
    ranks: Mapping[str, pd.DataFrame],
    spec: SignalSpec,
    *,
    returns: pd.DataFrame | None,
    lag: int = 1,
) -> dict[str, pd.Series] | None:
    """Per-factor decay scales for the score blend, or ``None`` when disabled.

    Roadmap 9.5 FactorDecayMonitor.  The monitor reads the trailing rank-IC
    history of each factor and scales its blend contribution once the
    consecutive weak periods reach ``decay_fail_periods``.  The realised
    returns the IC is measured on are the engine's own daily-return panel, so
    the caller passes it in rather than the spec carrying a data panel.

    ``returns`` is mandatory while the monitor is on: an enabled monitor with
    no return panel would silently do nothing, so it fails closed instead.
    """

    if spec.decay_window is None:
        return None
    if returns is None:
        raise ValueError("factor decay monitor requires a returns panel")
    return factor_decay_scales(
        ranks,
        returns,
        lag=int(lag),
        window=int(spec.decay_window),
        fail_periods=int(spec.decay_fail_periods or 1),
        threshold=float(spec.decay_ic_threshold or 0.0),
        scale=0.0 if spec.decay_scale is None else float(spec.decay_scale),
    )


def build_factor_score(
    panels: Mapping[str, pd.DataFrame],
    directions: Mapping[str, int],
    spec: SignalSpec,
    *,
    index: pd.Index,
    columns: pd.Index,
    eligible: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Direction map + signal pipeline + percentile rank + blend, one call.

    ``returns``/``lag`` only enter while the roadmap 9.5 decay monitor is
    enabled: the IC windows are measured on this realised-return panel with
    the same ``signal_lag`` the position itself uses.  Disabled -- the default
    -- both are ignored and the blend is the untouched one.
    """

    ranks = build_factor_rank_panels(panels, directions, spec, index=index, columns=columns, eligible=eligible)
    decay = factor_decay_weights(ranks, spec, returns=returns, lag=lag)
    return blend_member_panels(ranks, list(panels), spec, decay=decay)
