"""Metric normalization and deterministic feedback for the factor LLM."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("IC", ("IC", "ic", "ic_mean")),
    ("IC std", ("IC_std", "ic_std")),
    ("ICIR", ("ICIR", "icir")),
    ("RankIC", ("Rank_IC", "Rank IC", "rank_ic", "rank_ic_mean")),
    ("RankIC std", ("Rank_IC_std", "rank_ic_std")),
    ("RankICIR", ("Rank_ICIR", "Rank ICIR", "rank_icir")),
    (
        "Annualized return",
        ("annualized_return", "1day.excess_return_with_cost.annualized_return", "arr"),
    ),
    ("Sharpe / information ratio", ("sharpe", "information_ratio", "1day.excess_return_with_cost.information_ratio")),
    ("Max drawdown", ("max_drawdown", "1day.excess_return_with_cost.max_drawdown", "mdd")),
    ("Turnover", ("turnover",)),
    ("Coverage", ("coverage",)),
    ("Long-short spread (daily)", ("long_short_spread_daily", "long_short_spread", "spread")),
    ("Win rate", ("win_rate",)),
)


def _value(metrics: Mapping[str, Any] | None, aliases: tuple[str, ...]) -> float | None:
    if not metrics:
        return None
    for alias in aliases:
        value = metrics.get(alias)
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            return number if math.isfinite(number) else None
    return None


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def format_factor_feedback(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> str:
    """Return stable, full feedback instead of the upstream three metrics."""

    lines = ["Factor evaluation metrics:"]
    for label, aliases in METRIC_ALIASES:
        current_value = _value(current, aliases)
        if baseline is None:
            lines.append(f"- {label}: {_fmt(current_value)}")
        else:
            baseline_value = _value(baseline, aliases)
            delta = current_value - baseline_value if current_value is not None and baseline_value is not None else None
            lines.append(f"- {label}: current={_fmt(current_value)}, baseline={_fmt(baseline_value)}, delta={_fmt(delta)}")

    # Deterministic per-year rendering: years sorted as strings, metric keys in
    # the worker's canonical order first, then any extra keys alphabetically.
    lines.append("- Metrics by year:")
    by_year = current.get("metrics_by_year")
    if isinstance(by_year, Mapping) and by_year:
        for year in sorted(by_year, key=str):
            values = by_year[year]
            if not isinstance(values, Mapping):
                lines.append(f"  - {year}: {values}")
                continue
            known = [key for _, aliases in METRIC_ALIASES for key in aliases if key in values]
            extra = sorted(set(values) - set(known))
            rendered = ", ".join(f"{key}={_fmt(_value(values, (key,)))}" for key in (*known, *extra))
            lines.append(f"  - {year}: {rendered}")
    else:
        lines.append("  - N/A")

    # Legacy compatibility: older payloads only carried the compact
    # stability_by_year summary, which is still rendered when present.
    stability = current.get("stability_by_year") or current.get("stability_by_subperiod")
    lines.append("- Stability by subperiod:")
    if isinstance(stability, Mapping) and stability:
        for period in sorted(stability, key=str):
            values = stability[period]
            if isinstance(values, Mapping):
                rendered = ", ".join(f"{key}={_fmt(_value(values, (key,)))}" for key in sorted(values))
            else:
                rendered = str(values)
            lines.append(f"  - {period}: {rendered}")
    else:
        lines.append("  - N/A")
    return "\n".join(lines)
