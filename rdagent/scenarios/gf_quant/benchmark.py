"""Family-aware benchmark calibration for GF factor acceptance standards.

The remote benchmark worker (``scripts/gf_quant_benchmark_worker.py``) turns the
existing benchmark panels into compact per-factor summaries.  This module turns
those summaries into *numeric* acceptance standards:

1. **Exclusions** - unusable benchmark factors are dropped with an explicit
   reason; nothing is discarded silently.
2. **Clustering** - factors are grouped by absolute correlation of their compact
   daily profiles so a large family of near-identical benchmarks cannot dominate
   the empirical distribution.  Every factor in a cluster carries the inverse
   cluster weight ``1 / (cluster_size * cluster_count)``, which is mathematically
   equivalent to counting one representative per cluster.
3. **Standards** - quantiles of the cluster-balanced distributions define the
   floors and ceilings that candidates must clear.

Everything is deterministic and JSON-serializable so a calibration can be
audited, diffed and reloaded.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

SUPPORTED_FAMILIES: tuple[str, ...] = ("price_volume", "fundamental")
SCHEMA_VERSION = "1.0"

#: Profile kinds that can drive correlation clustering.
PROFILE_KINDS: tuple[str, ...] = ("spread", "rank_ic")

#: Aggregated per-factor quantities calibrated per family and their roles.
QUALITY_FLOOR_QUANTILE = 0.50
COVERAGE_FLOOR_QUANTILE = 0.25
RISK_CEILING_QUANTILE = 0.75

#: A profile-derived standard needs this many valid benchmark
#: observations before a standard is calibrated from it; otherwise the gate is
#: explicitly skipped instead of being silently invented.
MIN_SAMPLE_FOR_STANDARD = 10

DEFAULT_CLUSTER_THRESHOLD = 0.88
DEFAULT_CLUSTER_MIN_OVERLAP = 100
DEFAULT_MIN_OBSERVATIONS = 100
DEFAULT_MIN_VALID_YEARS = 2


class StandardsValidationError(ValueError):
    """Raised when a stored calibration is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------------
# per-factor benchmark results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkFactorResult:
    """Compact summary of one benchmark factor produced by the remote worker."""

    name: str
    family: str
    source_dir: str
    file_name: str
    metrics: Mapping[str, Any]
    metrics_by_year: Mapping[str, Mapping[str, Any]]
    spread_profile: Mapping[str, float | None]
    rank_ic_profile: Mapping[str, float | None]
    status: str = "ok"
    exclusion_reason: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, default_family: str = "") -> "BenchmarkFactorResult":
        profile = payload.get("profile") or {}
        dates = profile.get("dates") or []
        spread = profile.get("spread") or []
        rank_ic = profile.get("rank_ic") or []
        spread_profile = {str(dt): value for dt, value in zip(dates, spread)}
        rank_ic_profile = {str(dt): value for dt, value in zip(dates, rank_ic)}
        return cls(
            name=str(payload.get("name") or ""),
            family=str(payload.get("family") or default_family),
            source_dir=str(payload.get("source_dir") or ""),
            file_name=str(payload.get("file_name") or ""),
            metrics=payload.get("metrics") or {},
            metrics_by_year=payload.get("metrics_by_year") or {},
            spread_profile=spread_profile,
            rank_ic_profile=rank_ic_profile,
            status=str(payload.get("status") or ("ok" if payload.get("metrics") else "excluded")),
            exclusion_reason=payload.get("exclusion_reason"),
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialise the summary without re-encoding the worker payload."""
        return {
            "name": self.name,
            "family": self.family,
            "source_dir": self.source_dir,
            "file_name": self.file_name,
            "status": self.status,
            "exclusion_reason": self.exclusion_reason,
            "metrics": dict(self.metrics),
            "metrics_by_year": {str(year): dict(values) for year, values in self.metrics_by_year.items()},
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def annual_rank_ic(results: Mapping[str, Any]) -> dict[str, float]:
    """Finite per-year RankIC values keyed by year, sorted deterministically."""
    by_year = results.get("metrics_by_year")
    if not isinstance(by_year, Mapping):
        return {}
    values: dict[str, float] = {}
    for year in sorted(by_year, key=str):
        value = _finite((by_year[year] or {}).get("Rank_IC"))
        if value is not None:
            values[str(year)] = value
    return values


# ---------------------------------------------------------------------------
# direction handling and profile metrics
# ---------------------------------------------------------------------------


def orientation_sign(rank_ic: float | None, spread: Iterable[float | None] = ()) -> int:
    """Return the predictive orientation of a factor as ``+1``/``-1``.

    Benchmark factors and candidates may be built with either polarity, so any
    signed comparison must be made relative to the factor's own predictive
    direction.  The aggregate RankIC defines it; the mean daily long-short
    return is the fallback when RankIC is unavailable, and ``+1`` when both are
    silent.
    """
    value = _finite(rank_ic)
    if value is not None and value != 0.0:
        return 1 if value > 0 else -1
    total = 0.0
    count = 0
    for item in spread:
        number = _finite(item)
        if number is None:
            continue
        total += number
        count += 1
    if count and total != 0.0:
        return 1 if total > 0 else -1
    return 1


@dataclass(frozen=True)
class ProfileMetrics:
    """Direction-adjusted performance quantities derived from a daily profile.

    Benchmark factors and candidates may carry either polarity, so return and
    risk are measured on the series re-signed to the factor's own predictive
    direction (its aggregate RankIC).  ``max_drawdown_abs`` is the drawdown
    magnitude of that direction-adjusted series: it is an exact
    direction-adjusted quantity, never a raw signed return comparison across
    arbitrarily oriented factors.
    """

    direction: int
    annualized_return: float
    sharpe: float
    max_drawdown_abs: float
    observations: int
    mean_spread: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "direction": self.direction,
            "annualized_return": self.annualized_return,
            "sharpe": self.sharpe,
            "max_drawdown_abs": self.max_drawdown_abs,
            "observations": self.observations,
            "mean_spread": self.mean_spread,
        }


def _max_drawdown(values: Sequence[float]) -> float:
    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    return drawdown


def _profile_values(spread: Mapping[str, float | None] | pd.Series) -> list[float]:
    if isinstance(spread, Mapping):
        raw = list(spread.values())
    else:
        raw = list(pd.Series(spread).dropna())
    values: list[float] = []
    for value in raw:
        number = _finite(value)
        if number is not None:
            values.append(number)
    return values


def profile_metrics(
    spread: Mapping[str, float | None] | pd.Series | None,
    *,
    rank_ic: float | None,
) -> ProfileMetrics | None:
    """Compute direction-adjusted return metrics from a daily spread profile.

    ``rank_ic`` fixes the orientation: the long-short series is sign-adjusted to
    the factor's own predictive direction before return and Sharpe are computed.
    Returns ``None`` when the profile carries no usable observation.
    """
    if spread is None:
        return None
    values = _profile_values(spread)
    if not values:
        return None
    direction = orientation_sign(rank_ic, values)
    adjusted = [direction * value for value in values]
    mean_spread = sum(adjusted) / len(adjusted)
    std = (
        math.sqrt(sum((value - mean_spread) ** 2 for value in adjusted) / (len(adjusted) - 1))
        if len(adjusted) > 1
        else 0.0
    )
    sharpe = mean_spread / std * math.sqrt(252.0) if std > 0 else math.nan
    return ProfileMetrics(
        direction=direction,
        annualized_return=mean_spread * 252.0,
        sharpe=sharpe,
        max_drawdown_abs=abs(_max_drawdown(adjusted)),
        observations=len(values),
        mean_spread=mean_spread,
    )


def candidate_profile_metrics(
    metrics: Mapping[str, Any],
    return_profile: Mapping[str, float | None] | pd.Series | None,
) -> ProfileMetrics | None:
    """Same computation for a candidate, keyed off its aggregate RankIC."""

    if return_profile is None:
        return None
    if isinstance(return_profile, pd.Series):
        profile = {pd.Timestamp(dt).strftime("%Y-%m-%d"): value for dt, value in return_profile.items()}
    else:
        profile = dict(return_profile)
    return profile_metrics(profile, rank_ic=_finite(metrics.get("Rank_IC")))


# ---------------------------------------------------------------------------
# deterministic clustering
# ---------------------------------------------------------------------------


def profile_correlation(
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
    *,
    min_overlap: int,
) -> float | None:
    """Pearson correlation of two compact daily profiles over their overlap."""
    shared = [key for key in left if key in right]
    pair = [
        (float(left[key]), float(right[key]))
        for key in shared
        if _finite(left[key]) is not None and _finite(right[key]) is not None
    ]
    if len(pair) < max(2, min_overlap):
        return None
    xs = [a for a, _ in pair]
    ys = [b for _, b in pair]
    n = float(len(pair))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pair)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return float(cov / math.sqrt(var_x * var_y))


@dataclass(frozen=True)
class BenchmarkCluster:
    """One redundancy cluster of the family benchmark pool."""

    cluster_id: int
    members: tuple[str, ...]
    representative: str

    @property
    def size(self) -> int:
        return len(self.members)

    def to_payload(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "representative": self.representative,
            "members": list(self.members),
            "size": self.size,
        }


def cluster_by_correlation(
    results: Sequence[BenchmarkFactorResult],
    *,
    threshold: float,
    min_overlap: int,
    profile_kind: str = "spread",
) -> list[BenchmarkCluster]:
    """Single-linkage clustering on absolute profile correlation.

    Pairs are processed in descending correlation with a deterministic name
    tie-break, so the output never depends on the input ordering.  Sign is
    ignored (absolute correlation) because a factor and its negation are the
    same signal.
    """
    if profile_kind not in PROFILE_KINDS:
        raise ValueError(f"Unsupported clustering profile: {profile_kind!r}")
    if not 0 < threshold <= 1:
        raise ValueError("cluster threshold must be in (0, 1]")
    ordered = sorted(results, key=lambda item: item.name)
    attr = f"{profile_kind}_profile"
    pairs: list[tuple[float, str, str]] = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            left_profile = getattr(left, attr)
            right_profile = getattr(right, attr)
            if not left_profile or not right_profile:
                continue
            smaller, larger = (left_profile, right_profile) if len(left_profile) <= len(right_profile) else (
                right_profile,
                left_profile,
            )
            correlation = profile_correlation(smaller, larger, min_overlap=min_overlap)
            if correlation is not None and abs(correlation) >= threshold:
                pairs.append((abs(correlation), left.name, right.name))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    parent: dict[str, str] = {item.name: item.name for item in ordered}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for _, left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Deterministic root: the lexicographically smaller name wins.
            keep, drop = sorted((root_left, root_right))
            parent[drop] = keep

    grouped: dict[str, list[str]] = {}
    for item in ordered:
        grouped.setdefault(find(item.name), []).append(item.name)

    clusters = [sorted(members) for members in grouped.values()]
    clusters.sort(key=lambda members: (-len(members), members[0]))
    return [
        BenchmarkCluster(cluster_id=index, members=tuple(members), representative=members[0])
        for index, members in enumerate(clusters)
    ]


def inverse_cluster_weights(clusters: Sequence[BenchmarkCluster]) -> dict[str, float]:
    """Weight every factor by ``1 / (cluster_size * cluster_count)``.

    Each cluster therefore carries the same total weight, which is the weighted
    formulation of "one representative per cluster": the weights sum to one and
    a cluster with many near-identical members cannot outweigh a singleton.
    """
    if not clusters:
        return {}
    count = len(clusters)
    weights: dict[str, float] = {}
    for cluster in clusters:
        for member in cluster.members:
            weights[member] = 1.0 / (cluster.size * count)
    return weights


def representative_sample(clusters: Sequence[BenchmarkCluster]) -> tuple[str, ...]:
    """One deterministic representative per cluster, ordered by cluster id."""
    return tuple(cluster.representative for cluster in clusters)


# ---------------------------------------------------------------------------
# weighted quantiles
# ---------------------------------------------------------------------------


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Linearly interpolated weighted quantile.

    Sort the values ascending (stable on ties, i.e. input order), place each
    observation at the midpoint of its cumulative weight
    ``p_i = (c_i - w_i / 2) / total`` and interpolate.  With equal weights this
    is the usual empirical quantile; with inverse-cluster weights it is the
    cluster-balanced quantile.
    """
    if not values:
        raise ValueError("weighted_quantile requires at least one value")
    if len(values) != len(weights):
        raise ValueError("weighted_quantile requires one weight per value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    total = math.fsum(float(weight) for weight in weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("weighted_quantile requires positive finite weights")
    ordered = sorted(zip((float(v) for v in values), (float(w) for w in weights)), key=lambda item: item[0])
    cumulative = 0.0
    positions: list[float] = []
    for value, weight in ordered:
        cumulative += weight
        positions.append((cumulative - weight / 2.0) / total)
    if q <= positions[0]:
        return ordered[0][0]
    if q >= positions[-1]:
        return ordered[-1][0]
    for index in range(1, len(ordered)):
        lower_position, upper_position = positions[index - 1], positions[index]
        if lower_position <= q <= upper_position:
            lower_value, upper_value = ordered[index - 1][0], ordered[index][0]
            if upper_position == lower_position:
                return lower_value
            share = (q - lower_position) / (upper_position - lower_position)
            return float(lower_value + share * (upper_value - lower_value))
    return ordered[-1][0]


def cluster_balanced_quantile(
    values: Mapping[str, float],
    clusters: Sequence[BenchmarkCluster],
    q: float,
) -> float:
    """Quantile of ``values`` under the inverse-cluster weighting."""
    weights = inverse_cluster_weights(clusters)
    selected = [(name, value) for name, value in values.items() if name in weights]
    if not selected:
        raise ValueError("cluster_balanced_quantile received no weighted values")
    return weighted_quantile(
        [value for _, value in selected],
        [weights[name] for name, _ in selected],
        q,
    )


def benchmark_percentile(values: Sequence[float], value: float | None) -> float | None:
    """Fraction of the reference distribution at or below ``value``.

    Ties share half of their mass, so a candidate identical to the median
    benchmark reports ``0.5``.  ``None`` means the value is not comparable.
    """
    number = _finite(value)
    if number is None or not values:
        return None
    ordered = sorted(float(item) for item in values)
    below = sum(1 for item in ordered if item < number)
    ties = sum(1 for item in ordered if item == number)
    return (below + 0.5 * ties) / len(ordered)


# ---------------------------------------------------------------------------
# exclusions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exclusion:
    name: str
    reason: str
    detail: str

    def to_payload(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason, "detail": self.detail}


@dataclass(frozen=True)
class ExclusionRule:
    """Transparent, pre-declared exclusion rules applied before calibration."""

    min_observations: int = DEFAULT_MIN_OBSERVATIONS
    min_valid_years: int = DEFAULT_MIN_VALID_YEARS
    min_profile_observations: int = DEFAULT_CLUSTER_MIN_OVERLAP

    def apply(self, results: Sequence[BenchmarkFactorResult]) -> tuple[list[BenchmarkFactorResult], list[Exclusion]]:
        valid: list[BenchmarkFactorResult] = []
        exclusions: list[Exclusion] = []
        for result in results:
            reason = self._reason(result)
            if reason is None:
                valid.append(result)
                continue
            exclusions.append(Exclusion(result.name, reason, self._detail(result, reason)))
        return valid, exclusions

    def _reason(self, result: BenchmarkFactorResult) -> str | None:
        if result.status != "ok":
            return result.exclusion_reason or "worker_error"
        metrics = result.metrics
        for key in ("Rank_IC", "coverage", "turnover"):
            if _finite(metrics.get(key)) is None:
                return "nonfinite_core_metrics"
        observations = _finite(metrics.get("observations"))
        if observations is None or observations < self.min_observations:
            return "insufficient_observations"
        if len(annual_rank_ic(metrics)) < self.min_valid_years:
            return "insufficient_valid_years"
        if not result.spread_profile:
            return "missing_profile"
        usable = sum(1 for value in result.spread_profile.values() if _finite(value) is not None)
        if usable < self.min_profile_observations:
            return "insufficient_profile_observations"
        return None

    @staticmethod
    def _detail(result: BenchmarkFactorResult, reason: str) -> str:
        if reason == "nonfinite_core_metrics":
            missing = [key for key in ("Rank_IC", "coverage", "turnover") if _finite(result.metrics.get(key)) is None]
            return f"missing or non-finite: {', '.join(missing)}"
        if reason == "insufficient_observations":
            return f"observations={result.metrics.get('observations')}"
        if reason == "insufficient_valid_years":
            return f"valid_years={len(annual_rank_ic(result.metrics))}"
        if reason == "insufficient_profile_observations":
            return f"profile_observations={len(result.spread_profile)}"
        return str(result.exclusion_reason or reason)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyStandards:
    """Numeric acceptance standards for one factor family.

    Every number is derived from the cluster-balanced benchmark distribution;
    see :class:`QuantilePolicy` for the role each quantile plays.
    """

    family: str
    min_abs_rank_ic: float
    min_coverage: float
    max_turnover: float
    max_abs_drawdown: float
    min_observations: int
    annual_min_valid_years: int
    annual_min_matching_sign_years: int
    min_abs_rank_icir: float | None = None
    min_direction_adjusted_annualized_return: float | None = None
    min_direction_adjusted_sharpe: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "min_abs_rank_ic": self.min_abs_rank_ic,
            "min_abs_rank_icir": self.min_abs_rank_icir,
            "min_coverage": self.min_coverage,
            "max_turnover": self.max_turnover,
            "max_abs_drawdown": self.max_abs_drawdown,
            "min_observations": self.min_observations,
            "annual_min_valid_years": self.annual_min_valid_years,
            "annual_min_matching_sign_years": self.annual_min_matching_sign_years,
            "min_direction_adjusted_annualized_return": self.min_direction_adjusted_annualized_return,
            "min_direction_adjusted_sharpe": self.min_direction_adjusted_sharpe,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FamilyStandards":
        required = (
            "family",
            "min_abs_rank_ic",
            "min_coverage",
            "max_turnover",
            "max_abs_drawdown",
            "min_observations",
            "annual_min_valid_years",
            "annual_min_matching_sign_years",
        )
        family = str(payload.get("family") or "")
        if family not in SUPPORTED_FAMILIES:
            raise StandardsValidationError(f"unsupported standards family: {family!r}")
        missing = [key for key in required if key not in payload]
        if missing:
            raise StandardsValidationError(f"standards payload missing keys: {', '.join(missing)}")
        optional_floats = {}
        for key in (
            "min_abs_rank_icir",
            "min_direction_adjusted_annualized_return",
            "min_direction_adjusted_sharpe",
        ):
            raw = payload.get(key)
            if raw is not None:
                number = _finite(raw)
                if number is None:
                    raise StandardsValidationError(f"{key} must be a finite number, got {raw!r}")
                optional_floats[key] = number
        family = str(payload["family"])
        return cls(
            family=family,
            min_abs_rank_ic=_required_float(payload, "min_abs_rank_ic"),
            min_coverage=_required_float(payload, "min_coverage"),
            max_turnover=_required_float(payload, "max_turnover"),
            max_abs_drawdown=_required_float(payload, "max_abs_drawdown"),
            min_observations=_required_int(payload, "min_observations", minimum=1),
            annual_min_valid_years=_required_int(payload, "annual_min_valid_years", minimum=1),
            annual_min_matching_sign_years=_required_int(payload, "annual_min_matching_sign_years", minimum=1),
            **optional_floats,
        )


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    number = _finite(payload.get(key))
    if number is None:
        raise StandardsValidationError(f"{key} must be a finite number, got {payload.get(key)!r}")
    return number


def _required_int(payload: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or float(raw) != int(raw):
        raise StandardsValidationError(f"{key} must be a whole number, got {raw!r}")
    value = int(raw)
    if value < minimum:
        raise StandardsValidationError(f"{key} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class QuantilePolicy:
    """The deterministic quantile rule behind every standard."""

    quality_floor: float = QUALITY_FLOOR_QUANTILE
    coverage_floor: float = COVERAGE_FLOOR_QUANTILE
    risk_ceiling: float = RISK_CEILING_QUANTILE
    annual_integer_rule: str = "floor_of_cluster_balanced_median"

    def to_payload(self) -> dict[str, Any]:
        return {
            "quality_floor_quantile": self.quality_floor,
            "coverage_floor_quantile": self.coverage_floor,
            "risk_ceiling_quantile": self.risk_ceiling,
            "annual_integer_rule": self.annual_integer_rule,
            "weighting": "inverse_cluster (1 / (cluster_size * cluster_count))",
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "QuantilePolicy":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise StandardsValidationError("quantile_policy must be an object")
        for key in ("quality_floor_quantile", "coverage_floor_quantile", "risk_ceiling_quantile"):
            number = _finite(payload.get(key))
            if number is None or not 0.0 < number < 1.0:
                raise StandardsValidationError(f"quantile_policy.{key} must be a fraction in (0, 1)")
        return cls(
            quality_floor=float(payload["quality_floor_quantile"]),
            coverage_floor=float(payload["coverage_floor_quantile"]),
            risk_ceiling=float(payload["risk_ceiling_quantile"]),
            annual_integer_rule=str(payload.get("annual_integer_rule", cls.annual_integer_rule)),
        )


@dataclass(frozen=True)
class Distribution:
    """A calibrated reference distribution over cluster representatives."""

    metric: str
    role: str
    quantile: float
    values: tuple[float, ...]

    def to_payload(self) -> dict[str, Any]:
        return {"metric": self.metric, "role": self.role, "quantile": self.quantile, "values": list(self.values)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Distribution":
        for key in ("metric", "role", "quantile", "values"):
            if key not in payload:
                raise StandardsValidationError(f"distribution missing key: {key}")
        values = payload["values"]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise StandardsValidationError(f"distribution {payload['metric']!r} must be a non-empty list")
        numbers = [_finite(value) for value in values]
        if any(number is None for number in numbers):
            raise StandardsValidationError(f"distribution {payload['metric']!r} contains a non-finite value")
        quantile = _finite(payload["quantile"])
        if quantile is None or not 0.0 <= quantile <= 1.0:
            raise StandardsValidationError(f"distribution {payload['metric']!r} has an invalid quantile")
        return cls(
            metric=str(payload["metric"]),
            role=str(payload["role"]),
            quantile=float(quantile),
            values=tuple(float(value) for value in numbers),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class FamilyCalibration:
    """Everything needed to audit one family's calibrated standards."""

    family: str
    standards: FamilyStandards
    clusters: tuple[BenchmarkCluster, ...]
    distributions: tuple[Distribution, ...]
    exclusions: tuple[Exclusion, ...]
    cluster_profile: str
    correlation_threshold: float
    min_overlap: int
    sample: Mapping[str, Any]
    quantile_policy: QuantilePolicy = field(default_factory=QuantilePolicy)

    def distribution_values(self, metric: str) -> tuple[float, ...] | None:
        for distribution in self.distributions:
            if distribution.metric == metric:
                return distribution.values
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "standards": self.standards.to_payload(),
            "clusters": [cluster.to_payload() for cluster in self.clusters],
            "distributions": [distribution.to_payload() for distribution in self.distributions],
            "exclusions": [exclusion.to_payload() for exclusion in self.exclusions],
            "cluster_profile": self.cluster_profile,
            "correlation_threshold": self.correlation_threshold,
            "min_overlap": self.min_overlap,
            "sample": dict(self.sample),
            "quantile_policy": self.quantile_policy.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FamilyCalibration":
        for key in ("standards", "clusters", "distributions", "exclusions", "sample"):
            if key not in payload:
                raise StandardsValidationError(f"family calibration missing key: {key}")
        standards = FamilyStandards.from_payload(payload["standards"])
        clusters = tuple(
            BenchmarkCluster(
                cluster_id=_required_int(cluster, "cluster_id"),
                members=tuple(str(member) for member in cluster["members"]),
                representative=str(cluster["representative"]),
            )
            for cluster in payload["clusters"]
        )
        if not clusters:
            raise StandardsValidationError(f"{standards.family} calibration has no clusters")
        seen: set[str] = set()
        for cluster in clusters:
            if not cluster.members:
                raise StandardsValidationError(f"{standards.family} cluster {cluster.cluster_id} is empty")
            if cluster.representative not in cluster.members:
                raise StandardsValidationError(
                    f"{standards.family} cluster {cluster.cluster_id} representative is not a member"
                )
            for member in cluster.members:
                if member in seen:
                    raise StandardsValidationError(f"{standards.family} factor {member} appears in two clusters")
                seen.add(member)
        distributions = tuple(
            Distribution.from_payload(item) for item in _sequence(payload["distributions"], "distributions")
        )
        if not distributions:
            raise StandardsValidationError(f"{standards.family} calibration has no distributions")
        exclusions = tuple(
            Exclusion(str(item.get("name", "")), str(item.get("reason", "")), str(item.get("detail", "")))
            for item in _sequence(payload["exclusions"], "exclusions")
        )
        return cls(
            family=standards.family,
            standards=standards,
            clusters=clusters,
            distributions=distributions,
            exclusions=exclusions,
            cluster_profile=str(payload.get("cluster_profile", "spread")),
            correlation_threshold=_required_float(payload, "correlation_threshold")
            if "correlation_threshold" in payload
            else DEFAULT_CLUSTER_THRESHOLD,
            min_overlap=_required_int(payload, "min_overlap") if "min_overlap" in payload else DEFAULT_CLUSTER_MIN_OVERLAP,
            sample=payload["sample"],
            quantile_policy=QuantilePolicy.from_payload(payload.get("quantile_policy")),
        )


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StandardsValidationError(f"{label} must be a list")
    return list(value)


def calibrate_family(
    results: Sequence[BenchmarkFactorResult],
    *,
    family: str,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    min_overlap: int = DEFAULT_CLUSTER_MIN_OVERLAP,
    cluster_profile: str = "spread",
    exclusion_rule: ExclusionRule | None = None,
    quantile_policy: QuantilePolicy | None = None,
    min_sample_for_standard: int = MIN_SAMPLE_FOR_STANDARD,
) -> FamilyCalibration:
    """Derive one family's standards from its benchmark pool."""
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported family: {family}")
    wrong_family = [result.name for result in results if result.family != family]
    if wrong_family:
        raise ValueError(f"Benchmark results of other families supplied to {family}: {wrong_family[:5]}")
    policy = quantile_policy or QuantilePolicy()
    rule = exclusion_rule or ExclusionRule()
    valid, exclusions = rule.apply(results)
    if not valid:
        raise StandardsValidationError(
            f"Family {family} has no valid benchmark factors after exclusions "
            f"({len(exclusions)} excluded); refusing to invent standards"
        )

    clusters = cluster_by_correlation(
        valid, threshold=threshold, min_overlap=min_overlap, profile_kind=cluster_profile
    )
    weights = inverse_cluster_weights(clusters)

    def weighted(metric_values: Mapping[str, float]) -> tuple[list[float], list[float]]:
        selected = [(name, value) for name, value in metric_values.items() if name in weights]
        return [value for _, value in selected], [weights[name] for name, _ in selected]

    abs_rank_ic = {result.name: abs(float(result.metrics["Rank_IC"])) for result in valid}
    abs_rank_icir = {
        result.name: abs(ir) for result in valid if (ir := _finite(result.metrics.get("Rank_ICIR"))) is not None
    }
    coverage = {result.name: float(result.metrics["coverage"]) for result in valid}
    turnover = {result.name: float(result.metrics["turnover"]) for result in valid}
    observations = {result.name: float(result.metrics["observations"]) for result in valid}

    profile_by_name: dict[str, ProfileMetrics] = {}
    for result in valid:
        computed = profile_metrics(result.spread_profile, rank_ic=_finite(result.metrics.get("Rank_IC")))
        if computed is not None:
            profile_by_name[result.name] = computed
    adjusted_return = {name: item.annualized_return for name, item in profile_by_name.items()}
    adjusted_sharpe = {name: item.sharpe for name, item in profile_by_name.items() if math.isfinite(item.sharpe)}
    drawdown_abs = {name: item.max_drawdown_abs for name, item in profile_by_name.items()}

    valid_years = {result.name: float(len(annual_rank_ic(result.metrics))) for result in valid}
    matching_sign_years = {}
    for result in valid:
        yearly = annual_rank_ic(result.metrics)
        aggregate = _finite(result.metrics.get("Rank_IC"))
        if aggregate is None or aggregate == 0.0:
            matching_sign_years[result.name] = 0.0
            continue
        matching_sign_years[result.name] = float(
            sum(1 for value in yearly.values() if value != 0.0 and (value > 0) == (aggregate > 0))
        )

    def quantile_of(values: Mapping[str, float], q: float) -> float | None:
        if not values:
            return None
        selected, selected_weights = weighted(values)
        if not selected:
            return None
        return weighted_quantile(selected, selected_weights, q)

    def integer_rule(values: Mapping[str, float], cap: int) -> int:
        median = quantile_of(values, policy.quality_floor)
        if median is None:
            raise StandardsValidationError(f"Family {family} cannot calibrate the annual integer rules")
        return max(1, min(cap, int(math.floor(median))))

    available_years = int(max(valid_years.values())) if valid_years else 0
    if available_years < 1:
        raise StandardsValidationError(f"Family {family} benchmarks report no valid years")

    icir_value = quantile_of(abs_rank_icir, policy.quality_floor)
    min_icir = icir_value if len(abs_rank_icir) >= min_sample_for_standard else None
    return_value = quantile_of(adjusted_return, policy.quality_floor)
    min_return = return_value if len(adjusted_return) >= min_sample_for_standard else None
    sharpe_value = quantile_of(adjusted_sharpe, policy.quality_floor)
    min_sharpe = sharpe_value if len(adjusted_sharpe) >= min_sample_for_standard else None

    standards = FamilyStandards(
        family=family,
        min_abs_rank_ic=float(quantile_of(abs_rank_ic, policy.quality_floor)),
        min_abs_rank_icir=min_icir,
        min_coverage=float(quantile_of(coverage, policy.coverage_floor)),
        max_turnover=float(quantile_of(turnover, policy.risk_ceiling)),
        max_abs_drawdown=float(quantile_of(drawdown_abs, policy.risk_ceiling)),
        min_observations=max(1, int(math.floor(quantile_of(observations, policy.coverage_floor)))),
        annual_min_valid_years=integer_rule(valid_years, available_years),
        annual_min_matching_sign_years=integer_rule(matching_sign_years, available_years),
        min_direction_adjusted_annualized_return=min_return,
        min_direction_adjusted_sharpe=min_sharpe,
    )

    representatives = representative_sample(clusters)

    def representative_distribution(metric: str, role: str, quantile: float, values: Mapping[str, float]) -> Distribution:
        return Distribution(metric, role, quantile, tuple(sorted(values[name] for name in representatives if name in values)))

    distributions = (
        representative_distribution("abs_rank_ic", "quality_floor", policy.quality_floor, abs_rank_ic),
        representative_distribution("abs_rank_icir", "quality_floor", policy.quality_floor, abs_rank_icir),
        representative_distribution("coverage", "coverage_floor", policy.coverage_floor, coverage),
        representative_distribution("turnover", "risk_ceiling", policy.risk_ceiling, turnover),
        representative_distribution("abs_drawdown", "risk_ceiling", policy.risk_ceiling, drawdown_abs),
        representative_distribution(
            "direction_adjusted_annualized_return", "quality_floor", policy.quality_floor, adjusted_return
        ),
        representative_distribution("direction_adjusted_sharpe", "quality_floor", policy.quality_floor, adjusted_sharpe),
        representative_distribution("observations", "coverage_floor", policy.coverage_floor, observations),
        representative_distribution("valid_years", "annual_rule", policy.quality_floor, valid_years),
        representative_distribution("matching_sign_years", "annual_rule", policy.quality_floor, matching_sign_years),
    )

    source_dirs = sorted({result.source_dir for result in results if result.source_dir})
    sample = {
        "total_inputs": len(results),
        "valid": len(valid),
        "excluded": len(exclusions),
        "cluster_count": len(clusters),
        "representative_count": len(representative_sample(clusters)),
        "source_dirs": source_dirs,
        "exclusion_rule": {
            "min_observations": rule.min_observations,
            "min_valid_years": rule.min_valid_years,
            "min_profile_observations": rule.min_profile_observations,
        },
        "min_sample_for_standard": min_sample_for_standard,
        "standards_sample_sizes": {
            "abs_rank_ic": len(abs_rank_ic),
            "abs_rank_icir": len(abs_rank_icir),
            "direction_adjusted_annualized_return": len(adjusted_return),
            "direction_adjusted_sharpe": len(adjusted_sharpe),
        },
    }
    return FamilyCalibration(
        family=family,
        standards=standards,
        clusters=tuple(clusters),
        distributions=distributions,
        exclusions=tuple(exclusions),
        cluster_profile=cluster_profile,
        correlation_threshold=threshold,
        min_overlap=min_overlap,
        sample=sample,
        quantile_policy=policy,
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def calibration_to_document(
    calibrations: Mapping[str, FamilyCalibration],
    *,
    generated_at: str,
    evaluation: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialise a full calibration into the tracked standards document."""
    missing = [family for family in SUPPORTED_FAMILIES if family not in calibrations]
    if missing:
        raise StandardsValidationError(f"calibration document is missing families: {', '.join(missing)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "evaluation": dict(evaluation or {}),
        "provenance": dict(provenance or {}),
        "families": {family: calibrations[family].to_payload() for family in SUPPORTED_FAMILIES},
    }


def load_family_calibrations(
    path: str | Path,
    *,
    families: Sequence[str] = SUPPORTED_FAMILIES,
) -> dict[str, FamilyCalibration]:
    """Load and validate a stored calibration document, loudly.

    Every structural problem - unreadable JSON, unsupported schema, missing
    family, bad number, inconsistent cluster - raises
    :class:`StandardsValidationError`.  Production calibration mode must never
    fall back to invented defaults.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StandardsValidationError(f"Cannot read standards file {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise StandardsValidationError(f"Standards file {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StandardsValidationError(f"Standards file {path} must contain a JSON object")
    version = str(payload.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        raise StandardsValidationError(
            f"Standards file {path} has unsupported schema_version {version!r} (expected {SCHEMA_VERSION!r})"
        )
    families_payload = payload.get("families")
    if not isinstance(families_payload, Mapping):
        raise StandardsValidationError(f"Standards file {path} has no families object")
    loaded: dict[str, FamilyCalibration] = {}
    for family in families:
        if family not in SUPPORTED_FAMILIES:
            raise StandardsValidationError(f"Unsupported family requested: {family}")
        if family not in families_payload:
            raise StandardsValidationError(f"Standards file {path} is missing the {family} family")
        try:
            loaded[family] = FamilyCalibration.from_payload(families_payload[family])
        except StandardsValidationError as exc:
            raise StandardsValidationError(f"Standards file {path}, family {family}: {exc}") from exc
    return loaded


def load_family_standards(
    path: str | Path,
    *,
    families: Sequence[str] = SUPPORTED_FAMILIES,
) -> dict[str, FamilyStandards]:
    """Convenience wrapper returning just the numeric standards per family."""
    return {family: item.standards for family, item in load_family_calibrations(path, families=families).items()}


def load_family_signatures(directory: str | Path, family: str) -> list[BenchmarkFactorResult]:
    """Load the compact benchmark profiles of one family for redundancy checks.

    The signatures file is written by
    ``scripts/gf_quant_benchmark_controller.py``; it holds only daily profiles,
    never a benchmark panel.
    """
    if family not in SUPPORTED_FAMILIES:
        raise StandardsValidationError(f"Unsupported family: {family}")
    path = Path(directory) / f"{family}_signatures.json"
    if not path.exists():
        raise StandardsValidationError(f"Benchmark signature file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StandardsValidationError(f"Benchmark signature file {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("family") != family:
        raise StandardsValidationError(f"Benchmark signature file {path} does not describe family {family!r}")
    factors = payload.get("factors")
    if not isinstance(factors, Mapping) or not factors:
        raise StandardsValidationError(f"Benchmark signature file {path} has no factors")
    results: list[BenchmarkFactorResult] = []
    for name, item in factors.items():
        if not isinstance(item, Mapping):
            continue
        dates = item.get("dates") or []
        results.append(
            BenchmarkFactorResult(
                name=str(name),
                family=family,
                source_dir=str(item.get("source_dir") or ""),
                file_name=f"{name}.parquet",
                metrics={"Rank_IC": item.get("rank_ic_mean")},
                metrics_by_year={},
                spread_profile={str(dt): value for dt, value in zip(dates, item.get("spread") or [])},
                rank_ic_profile={str(dt): value for dt, value in zip(dates, item.get("rank_ic") or [])},
            )
        )
    return results


# ---------------------------------------------------------------------------
# candidate gating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateReport:
    """Auditable outcome of one candidate against its family standards."""

    family: str
    passed: bool
    failures: tuple[str, ...]
    percentiles: Mapping[str, float | None]
    profile_available: bool
    skipped_gates: tuple[str, ...]
    standards: FamilyStandards | None = None
    source: str = "calibrated"

    def to_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "passed": self.passed,
            "gate_failures": list(self.failures),
            "benchmark_percentiles": dict(self.percentiles),
            "profile_available": self.profile_available,
            "skipped_gates": list(self.skipped_gates),
            "standards_source": self.source,
            "standards": self.standards.to_payload() if self.standards else None,
        }


def annual_gate_failures(
    metrics: Mapping[str, Any],
    *,
    min_valid_years: int,
    min_matching_sign_years: int,
) -> list[str]:
    """Calibrated annual gate over ``metrics_by_year`` (fails closed)."""
    failures: list[str] = []
    by_year = metrics.get("metrics_by_year")
    if not isinstance(by_year, Mapping) or not by_year:
        return ["annual_metrics_by_year_missing"]
    yearly = annual_rank_ic(metrics)
    if len(yearly) < min_valid_years:
        failures.append("annual_rank_ic_years")
    aggregate = _finite(metrics.get("Rank_IC"))
    if aggregate is None or aggregate == 0.0:
        failures.append("annual_rank_ic_sign_alignment")
        return failures
    aligned = sum(1 for value in yearly.values() if value != 0.0 and (value > 0) == (aggregate > 0))
    if aligned < min_matching_sign_years:
        failures.append("annual_rank_ic_sign_alignment")
    return failures


def evaluate_against_calibration(
    metrics: Mapping[str, Any],
    calibration: FamilyCalibration,
    *,
    return_profile: Mapping[str, float | None] | pd.Series | None = None,
    require_profile: bool = False,
) -> GateReport:
    """Gate one candidate against its family's calibrated standards.

    Direction handling: ``abs(RankIC)``/``abs(RankICIR)`` are sign-free; return,
    Sharpe and drawdown are measured on the candidate's own daily long-short
    profile re-signed to its aggregate RankIC, so arbitrarily oriented factors
    stay comparable.  Raw signed benchmark returns are never compared directly.
    """
    standards = calibration.standards
    failures: list[str] = []
    skipped: list[str] = []

    rank_ic = _finite(metrics.get("Rank_IC"))
    if rank_ic is None or abs(rank_ic) < standards.min_abs_rank_ic:
        failures.append("abs_rank_ic")
    if standards.min_abs_rank_icir is None:
        skipped.append("abs_rank_icir")
    else:
        icir = _finite(metrics.get("Rank_ICIR"))
        if icir is None or abs(icir) < standards.min_abs_rank_icir:
            failures.append("abs_rank_icir")

    coverage = _finite(metrics.get("coverage"))
    if coverage is None or coverage < standards.min_coverage:
        failures.append("coverage")
    turnover = _finite(metrics.get("turnover"))
    if turnover is None or turnover > standards.max_turnover:
        failures.append("turnover")
    observations = _finite(metrics.get("observations"))
    if observations is None or observations < standards.min_observations:
        failures.append("observations")

    profile = candidate_profile_metrics(metrics, return_profile)
    if profile is None:
        if require_profile:
            failures.append("return_profile_missing")
        else:
            skipped.extend(
                [
                    "direction_adjusted_annualized_return",
                    "direction_adjusted_sharpe",
                    "abs_drawdown",
                ]
            )
    else:
        if (
            standards.min_direction_adjusted_annualized_return is not None
            and profile.annualized_return < standards.min_direction_adjusted_annualized_return
        ):
            failures.append("direction_adjusted_annualized_return")
        if standards.min_direction_adjusted_sharpe is not None and profile.sharpe < standards.min_direction_adjusted_sharpe:
            failures.append("direction_adjusted_sharpe")
        if profile.max_drawdown_abs > standards.max_abs_drawdown:
            failures.append("abs_drawdown")

    failures.extend(
        annual_gate_failures(
            metrics,
            min_valid_years=standards.annual_min_valid_years,
            min_matching_sign_years=standards.annual_min_matching_sign_years,
        )
    )

    profile_metrics_map = {
        "abs_rank_ic": None if rank_ic is None else abs(rank_ic),
        "abs_rank_icir": None if _finite(metrics.get("Rank_ICIR")) is None else abs(float(metrics["Rank_ICIR"])),
        "coverage": coverage,
        "turnover": turnover,
        "observations": observations,
        "direction_adjusted_annualized_return": None if profile is None else profile.annualized_return,
        "direction_adjusted_sharpe": None if profile is None else profile.sharpe,
        "abs_drawdown": None if profile is None else profile.max_drawdown_abs,
    }
    percentiles = {
        metric: benchmark_percentile(calibration.distribution_values(metric), value)
        for metric, value in profile_metrics_map.items()
    }
    return GateReport(
        family=calibration.family,
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
        percentiles=percentiles,
        profile_available=profile is not None,
        skipped_gates=tuple(dict.fromkeys(skipped)),
        standards=standards,
        source="calibrated",
    )
