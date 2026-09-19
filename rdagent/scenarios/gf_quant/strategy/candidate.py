"""Strategy candidate specification for the GF portfolio layer.

A strategy candidate is a **specification**, not generated code: it names the
accepted factors or the selection rule it relies on, and how the resulting
cross-sectional score is turned into a portfolio.  Keeping the candidate
declarative is what makes the strategy loop auditable -- the LLM chooses rules
and parameters while the portfolio construction and the backtest stay
deterministic framework code that can be unit tested once.

This dataclass is deliberately independent of
:class:`rdagent.scenarios.gf_quant.service.FactorCandidate`.  Factor fields
(expression, family, benchmark percentiles) do not transfer to a portfolio, and
mixing them would invite reusing single-factor IC thresholds on strategies.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

WeightingScheme = Literal[
    "equal", "rank", "proportional", "score_transform", "inverse_volatility", "hierarchical"
]

WEIGHTING_SCHEMES: tuple[str, ...] = (
    "equal",
    "rank",
    "proportional",
    "score_transform",
    "inverse_volatility",
    "hierarchical",
    "risk_parity",
    "factor_risk_budget",
)

# Roadmap 7.6 RiskParityWeight: the covariance shrinkage targets the diagonal
# of the sample covariance (variances kept, correlations shrunk away), a
# deterministic closed-form target that needs no external dependency.
RISK_PARITY_SHRINKAGE_KINDS: tuple[str, ...] = ("none", "diagonal")

# The CCD solver's iteration budget is a solver detail, not a strategy
# parameter: fixed on both engine sides instead of travelling in the payload.
RISK_PARITY_MAX_ITER = 512

# Roadmap 7.4 ScoreTransformWeight: the transformed score is allocated
# proportionally, so every transform must consume (kind, slope, cut) with
# ``cut`` as the truncation point (truncated) or the sigmoid midpoint.
ScoreTransformKind = Literal["linear", "truncated", "sigmoid"]

SCORE_TRANSFORM_KINDS: tuple[str, ...] = ("linear", "truncated", "sigmoid")

TransformKind = Literal["linear", "log", "signed_log", "sqrt"]

TRANSFORM_KINDS: tuple[str, ...] = ("linear", "log", "signed_log", "sqrt")

StandardizeKind = Literal["zscore", "rank_z", "robust_z"]

STANDARDIZE_KINDS: tuple[str, ...] = ("zscore", "rank_z", "robust_z")

# Roadmap 4.5 MissingValuePolicy: explicit per-panel handling of missing
# factor values.  ``drop_stock`` removes symbols with any missing value from
# the factor's panel, ``cross_section_median`` fills value-space missing cells
# with that date's cross-sectional median (after the A1 signal stages, before
# the direction map), and ``neutral_rank`` fills missing percentile ranks with
# the neutral 0.5 after ranking.
MissingValuePolicy = Literal["drop_stock", "cross_section_median", "neutral_rank"]

MISSING_VALUE_POLICIES: tuple[str, ...] = ("drop_stock", "cross_section_median", "neutral_rank")

# Roadmap 4.11 FactorInteraction: each term combines two factors'
# post-direction percentile ranks into one extra synthetic blend member.
InteractionKind = Literal["product", "min", "geometric_mean"]

INTERACTION_KINDS: tuple[str, ...] = ("product", "min", "geometric_mean")

# The roadmap requires the number of interaction terms to be bounded; this is
# the conservative hard cap (same order as the 2-8 factor whitelist).
MAX_INTERACTION_TERMS = 8

# Roadmap 4.6 IndustryNeutralize: within-group de-mean (the regression on
# industry dummies alone is mathematically the de-mean) or a residual from a
# regression on industry dummies plus the injected exposures.
INDUSTRY_NEUTRALIZE_METHODS: tuple[str, ...] = ("demean", "regression")

GateDirection = Literal["top", "bottom"]

GATE_DIRECTIONS: tuple[str, ...] = ("top", "bottom")

GATING_PAYLOAD_KEYS: frozenset[str] = frozenset({"layers"})

GATE_LAYER_PAYLOAD_KEYS: frozenset[str] = frozenset({"factor", "direction", "quantile", "min_pass"})

# ``standardize`` is enabled whenever this is the denominator floor; a smaller
# cross-sectional dispersion (or a non-finite one) must fail closed instead of
# producing infinite z-scores.
DEFAULT_STANDARDIZE_MIN_STD = 1e-12


@dataclass(frozen=True)
class SignalSpec:
    """Per-date cross-sectional signal transforms applied before ranking.

    Pipeline on every rebalance date, in order: FactorTransform (raw factor
    values, monotone only) -> Winsorize (quantile clip) ->
    CrossSectionalStandardize, MissingValuePolicy (drop / fill), then the
    frozen factor direction and the percentile rank; interaction terms are
    built from the final rank panels and the blend (flat fixed weights or
    two-level group weights) combines factors and interaction terms.
    Parameters are global: one setting covers every factor of the candidate.
    All fields default to the disabled state, which leaves the raw factor
    values untouched and the blend equal-weight.
    """

    transform: TransformKind | None = None
    transform_scale: float = 1.0
    transform_shift: float = 0.0
    winsorize_lower: float | None = None
    winsorize_upper: float | None = None
    winsorize_min_samples: int = 1
    standardize: StandardizeKind | None = None
    standardize_min_std: float = DEFAULT_STANDARDIZE_MIN_STD
    missing_value: MissingValuePolicy | None = None
    missing_max_fill_ratio: float | None = None
    blend_weights: dict[str, float] | None = None
    blend_groups: dict[str, list[str]] | None = None
    blend_group_weights: dict[str, float] | None = None
    blend_within_weights: dict[str, dict[str, float]] | None = None
    interaction_pairs: tuple[tuple[str, str, str], ...] = ()
    # Roadmap 4.6 IndustryNeutralize (value space, applied right after the
    # eligible mask and before the direction map).  The industry mapping and
    # the optional regression exposures are explicit configuration inputs --
    # the engine has no industry data of its own, so the upper layer injects
    # them; they are execution data and stay out of the LLM prompt vocabulary
    # (same convention as ``industry_map``), but they travel in the canonical
    # payload and the worker whitelist while the operator is enabled.
    industry_neutralize: bool = False
    industry_neutralize_industry_map: dict[str, str] | None = None
    industry_neutralize_method: str = "demean"  # demean | regression
    industry_neutralize_min_group: int = 3
    industry_neutralize_version: str | None = None
    industry_neutralize_exposures: dict[str, dict[str, float]] | None = None
    # Roadmap 4.7 StyleNeutralize (value space, applied right after the
    # industry stage and before the direction map).  Style exposures are
    # explicit per-symbol configuration inputs (point-in-time values supplied
    # by the upper layer); the roadmap's estimation window and weighting have
    # no engine-meaningful source here and are deliberately absent.
    style_neutralize: bool = False
    style_neutralize_exposures: dict[str, dict[str, float]] | None = None
    style_neutralize_min_samples: int = 5
    # Roadmap 4.12 RegimeConditionalSignal: the score is, per decision date,
    # the equal-weight blend of the rank panels of the factors named for the
    # date's injected market state.  The state series ({date: label}) is an
    # explicit PIT configuration input; mutually exclusive with every static
    # blend configuration (FactorBlend / FactorGroupBlend / FactorInteraction).
    regime_states: dict[str, str] | None = None
    regime_factor_groups: dict[str, list[str]] | None = None
    regime_lag: int = 1
    # Roadmap 9.5 FactorDecayMonitor: while a factor's trailing rank-IC history
    # is failing, its blend contribution is scaled by ``decay_scale`` (the
    # factor is never replaced -- handing the book to a factor chosen by the
    # failing signal would use the same information twice).  ``decay_window``
    # is the enabler; the other three keys collapse onto their conservative
    # defaults once it is set.  The monitor consumes the realised-return panel
    # the engine already builds, so no data panel travels in the payload.
    decay_window: int | None = None
    decay_fail_periods: int | None = None
    decay_ic_threshold: float | None = None
    decay_scale: float | None = None

    def __post_init__(self) -> None:
        if self.transform not in (None, *TRANSFORM_KINDS):
            raise ValueError(f"transform must be one of {TRANSFORM_KINDS} or None, got {self.transform!r}")
        if not math.isfinite(self.transform_scale) or self.transform_scale <= 0:
            raise ValueError("transform_scale must be a finite positive number")
        if not math.isfinite(self.transform_shift):
            raise ValueError("transform_shift must be finite")
        if (self.winsorize_lower is None) != (self.winsorize_upper is None):
            raise ValueError("winsorize_lower and winsorize_upper must be provided together")
        if self.winsorize_lower is not None and not (
            0 <= self.winsorize_lower < self.winsorize_upper <= 1
        ):
            raise ValueError("winsorize quantiles must satisfy 0 <= lower < upper <= 1")
        if int(self.winsorize_min_samples) < 1:
            raise ValueError("winsorize_min_samples must be at least 1")
        if self.standardize not in (None, *STANDARDIZE_KINDS):
            raise ValueError(f"standardize must be one of {STANDARDIZE_KINDS} or None, got {self.standardize!r}")
        if not math.isfinite(self.standardize_min_std) or self.standardize_min_std <= 0:
            raise ValueError("standardize_min_std must be a finite positive number")
        if self.missing_value not in (None, *MISSING_VALUE_POLICIES):
            raise ValueError(
                f"missing_value must be one of {MISSING_VALUE_POLICIES} or None, got {self.missing_value!r}"
            )
        if self.missing_max_fill_ratio is not None:
            if not math.isfinite(self.missing_max_fill_ratio) or not 0 < self.missing_max_fill_ratio <= 1:
                raise ValueError("missing_max_fill_ratio must be in (0, 1] when provided")
            if self.missing_value is None:
                raise ValueError("missing_max_fill_ratio requires a missing_value policy")
            if self.missing_value == "drop_stock":
                raise ValueError("missing_max_fill_ratio does not apply to the drop_stock policy")
        if self.blend_weights is not None and self.blend_groups is not None:
            raise ValueError("blend_weights and blend_groups are mutually exclusive")
        if self.blend_weights is not None:
            # A list (or any non-mapping) must fail closed with the same
            # message the worker raises, not with an AttributeError.
            if not isinstance(self.blend_weights, Mapping) or not self.blend_weights:
                raise ValueError("blend_weights must not be empty")
            for name, weight in self.blend_weights.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("blend_weights keys must be non-empty blend member names")
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("blend weights must be finite positive numbers")
        if self.blend_groups is not None:
            if not self.blend_groups:
                raise ValueError("blend_groups must not be empty")
            assigned: set[str] = set()
            for group, factors in self.blend_groups.items():
                if not isinstance(group, str) or not group:
                    raise ValueError("blend_groups keys must be non-empty group names")
                if not factors:
                    raise ValueError(f"blend group {group!r} must name at least one factor")
                for factor_name in factors:
                    if factor_name in assigned:
                        raise ValueError("blend_groups must not assign a factor to two groups")
                    assigned.add(factor_name)
            if self.blend_group_weights is not None:
                if set(self.blend_group_weights) != set(self.blend_groups):
                    raise ValueError("blend_group_weights keys must match the blend_groups exactly")
                for weight in self.blend_group_weights.values():
                    if not math.isfinite(weight) or weight <= 0:
                        raise ValueError("blend group weights must be finite positive numbers")
            if self.blend_within_weights is not None:
                unknown_groups = set(self.blend_within_weights) - set(self.blend_groups)
                if unknown_groups:
                    raise ValueError(f"blend_within_weights names unknown groups: {sorted(unknown_groups)[:3]}")
                for group, weights in self.blend_within_weights.items():
                    if set(weights) != set(self.blend_groups[group]):
                        raise ValueError(f"blend_within_weights for group {group!r} must cover its factors exactly")
                    for weight in weights.values():
                        if not math.isfinite(weight) or weight <= 0:
                            raise ValueError("blend within-group weights must be finite positive numbers")
        if len(self.interaction_pairs) > MAX_INTERACTION_TERMS:
            raise ValueError(f"at most {MAX_INTERACTION_TERMS} interaction terms are allowed")
        seen_members: set[str] = set()
        for pair in self.interaction_pairs:
            # Every pair must be an exact (first, second, kind) triple; a
            # longer or shorter pair fails closed with the same message the
            # worker raises instead of unpacking wrong or crashing.
            if not isinstance(pair, tuple) or len(pair) != 3:
                raise ValueError("each interaction pair must be a [first, second, kind] triple")
            first, second, kind = pair
            if kind not in INTERACTION_KINDS:
                raise ValueError(f"interaction kind must be one of {INTERACTION_KINDS}, got {kind!r}")
            if not first or not second or first == second:
                raise ValueError("interaction terms need two distinct non-empty factor names")
            member = f"{first}*{second}"
            if member in seen_members:
                raise ValueError("interaction terms must be distinct")
            seen_members.add(member)
        # Roadmap 4.6 IndustryNeutralize.
        if not isinstance(self.industry_neutralize, bool):
            raise ValueError("industry_neutralize must be a boolean")
        if not self.industry_neutralize:
            lazy_industry = (
                self.industry_neutralize_industry_map is not None
                or self.industry_neutralize_exposures is not None
                or self.industry_neutralize_version is not None
                or self.industry_neutralize_method != "demean"
                or self.industry_neutralize_min_group != 3
            )
            if lazy_industry:
                raise ValueError("industry neutralize parameters require industry_neutralize")
        else:
            if self.industry_neutralize_industry_map is None:
                raise ValueError("industry neutralize requires industry_neutralize_industry_map")
            if not isinstance(self.industry_neutralize_industry_map, dict) or not self.industry_neutralize_industry_map:
                raise ValueError("industry_neutralize_industry_map must map non-empty symbols to non-empty industry labels")
            for symbol, label in self.industry_neutralize_industry_map.items():
                if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                    raise ValueError("industry_neutralize_industry_map must map non-empty symbols to non-empty industry labels")
            object.__setattr__(self, "industry_neutralize_industry_map", dict(self.industry_neutralize_industry_map))
        if self.industry_neutralize_method not in INDUSTRY_NEUTRALIZE_METHODS:
            raise ValueError(
                f"industry_neutralize_method must be one of {INDUSTRY_NEUTRALIZE_METHODS}, "
                f"got {self.industry_neutralize_method!r}"
            )
        min_group = _whole_universe_number(
            self.industry_neutralize_min_group, "industry_neutralize_min_group must be a whole number"
        )
        if min_group < 2:
            raise ValueError("industry_neutralize_min_group must be at least 2")
        if not isinstance(self.industry_neutralize_min_group, int) or isinstance(self.industry_neutralize_min_group, bool):
            object.__setattr__(self, "industry_neutralize_min_group", min_group)
        if self.industry_neutralize_version is not None:
            if not self.industry_neutralize:
                raise ValueError("industry neutralize parameters require industry_neutralize")
            if not isinstance(self.industry_neutralize_version, str) or not self.industry_neutralize_version:
                raise ValueError("industry_neutralize_version must be a non-empty string when provided")
        if self.industry_neutralize_exposures is not None:
            if not self.industry_neutralize:
                raise ValueError("industry neutralize parameters require industry_neutralize")
            if self.industry_neutralize_method != "regression":
                raise ValueError("industry_neutralize_exposures is only valid with the regression method")
            _validate_exposure_map(
                self.industry_neutralize_exposures,
                "industry_neutralize_exposures must map non-empty exposure names to non-empty symbol -> finite number objects",
            )
        # Roadmap 4.7 StyleNeutralize.
        if not isinstance(self.style_neutralize, bool):
            raise ValueError("style_neutralize must be a boolean")
        if not self.style_neutralize:
            if (
                self.style_neutralize_exposures is not None
                or self.style_neutralize_min_samples != 5
            ):
                raise ValueError("style neutralize parameters require style_neutralize")
        else:
            if self.style_neutralize_exposures is None:
                raise ValueError("style neutralize requires style_neutralize_exposures")
            _validate_exposure_map(
                self.style_neutralize_exposures,
                "style_neutralize_exposures must map non-empty style names to non-empty symbol -> finite number objects",
            )
        min_samples = _whole_universe_number(
            self.style_neutralize_min_samples, "style_neutralize_min_samples must be a whole number"
        )
        if min_samples < 2:
            raise ValueError("style_neutralize_min_samples must be at least 2")
        if not isinstance(self.style_neutralize_min_samples, int) or isinstance(self.style_neutralize_min_samples, bool):
            object.__setattr__(self, "style_neutralize_min_samples", min_samples)
        # Roadmap 4.12 RegimeConditionalSignal.
        if (self.regime_states is None) != (self.regime_factor_groups is None):
            raise ValueError("regime_states and regime_factor_groups must be provided together")
        if self.regime_states is not None:
            if self.blend_weights is not None:
                raise ValueError("regime_factor_groups and blend_weights are mutually exclusive")
            if self.blend_groups is not None:
                raise ValueError("regime_factor_groups and blend_groups are mutually exclusive")
            if self.interaction_pairs:
                raise ValueError("regime_factor_groups and interaction_pairs are mutually exclusive")
            if not isinstance(self.regime_states, dict) or not self.regime_states:
                raise ValueError("regime_states must map date strings to non-empty state labels")
            for date_key, label in self.regime_states.items():
                _parse_universe_date(date_key, f"regime_states contains an invalid date: {date_key!r}")
                if not isinstance(label, str) or not label:
                    raise ValueError("regime_states must map date strings to non-empty state labels")
            object.__setattr__(self, "regime_states", dict(self.regime_states))
            if not isinstance(self.regime_factor_groups, dict) or not self.regime_factor_groups:
                raise ValueError("regime_factor_groups must map non-empty state labels to non-empty factor lists")
            for label, factors in self.regime_factor_groups.items():
                if not isinstance(label, str) or not label:
                    raise ValueError("regime_factor_groups must map non-empty state labels to non-empty factor lists")
                if not isinstance(factors, list) or not factors:
                    raise ValueError(f"regime state {label!r} must name at least one factor")
                for factor_name in factors:
                    if not isinstance(factor_name, str) or not factor_name:
                        raise ValueError("regime_factor_groups members must be non-empty factor names")
            object.__setattr__(self, "regime_factor_groups", {label: list(factors) for label, factors in self.regime_factor_groups.items()})
        regime_lag = _whole_universe_number(self.regime_lag, "regime_lag must be a whole number")
        if regime_lag < 0:
            raise ValueError("regime_lag must not be negative")
        if not isinstance(self.regime_lag, int) or isinstance(self.regime_lag, bool):
            object.__setattr__(self, "regime_lag", regime_lag)
        # Roadmap 9.5 FactorDecayMonitor.  decay_window enables the monitor;
        # the three dependent keys follow the "x requires y" convention this
        # file uses everywhere else and collapse onto their conservative
        # defaults once the monitor is on (never fill a period, keep nothing).
        for dependent, value in (
            ("decay_fail_periods", self.decay_fail_periods),
            ("decay_ic_threshold", self.decay_ic_threshold),
            ("decay_scale", self.decay_scale),
        ):
            if self.decay_window is None and value is not None:
                raise ValueError(f"{dependent} requires decay_window")
        if self.decay_window is not None:
            window = _whole_universe_number(self.decay_window, "decay_window must be a whole number")
            if window < 2:
                raise ValueError("decay_window must be at least 2")
            object.__setattr__(self, "decay_window", window)
            fail_periods = (
                1
                if self.decay_fail_periods is None
                else _whole_universe_number(self.decay_fail_periods, "decay_fail_periods must be a whole number")
            )
            if fail_periods < 1:
                raise ValueError("decay_fail_periods must be at least 1")
            object.__setattr__(self, "decay_fail_periods", fail_periods)
            threshold = (
                0.0
                if self.decay_ic_threshold is None
                else _finite_universe_number(self.decay_ic_threshold, "decay_ic_threshold must be finite")
            )
            object.__setattr__(self, "decay_ic_threshold", threshold)
            scale = (
                0.0
                if self.decay_scale is None
                else _finite_universe_number(self.decay_scale, "decay_scale must be in [0, 1]")
            )
            if not 0.0 <= scale <= 1.0:
                raise ValueError("decay_scale must be in [0, 1]")
            object.__setattr__(self, "decay_scale", scale)

    def as_payload(self) -> dict[str, Any]:
        """Canonical JSON-serialisable form; also feeds ``strategy_id_for``.

        The signal-DAG fields travel only while enabled, so every pre-existing
        candidate keeps its canonical payload (and therefore its strategy
        identity hash) unchanged.
        """

        payload = {
            "transform": self.transform,
            "transform_scale": self.transform_scale,
            "transform_shift": self.transform_shift,
            "winsorize_lower": self.winsorize_lower,
            "winsorize_upper": self.winsorize_upper,
            "winsorize_min_samples": self.winsorize_min_samples,
            "standardize": self.standardize,
            "standardize_min_std": self.standardize_min_std,
        }
        if self.missing_value is not None:
            payload["missing_value"] = self.missing_value
            if self.missing_max_fill_ratio is not None:
                payload["missing_max_fill_ratio"] = self.missing_max_fill_ratio
        if self.blend_weights is not None:
            payload["blend_weights"] = dict(self.blend_weights)
        if self.blend_groups is not None:
            payload["blend_groups"] = {group: list(factors) for group, factors in self.blend_groups.items()}
            if self.blend_group_weights is not None:
                payload["blend_group_weights"] = dict(self.blend_group_weights)
            if self.blend_within_weights is not None:
                payload["blend_within_weights"] = {
                    group: dict(weights) for group, weights in self.blend_within_weights.items()
                }
        if self.interaction_pairs:
            payload["interaction_pairs"] = [list(pair) for pair in self.interaction_pairs]
        if self.industry_neutralize:
            payload["industry_neutralize"] = True
            payload["industry_neutralize_industry_map"] = dict(self.industry_neutralize_industry_map or {})
            payload["industry_neutralize_method"] = self.industry_neutralize_method
            payload["industry_neutralize_min_group"] = self.industry_neutralize_min_group
            if self.industry_neutralize_version is not None:
                payload["industry_neutralize_version"] = self.industry_neutralize_version
            if self.industry_neutralize_exposures is not None:
                payload["industry_neutralize_exposures"] = {
                    name: dict(values) for name, values in self.industry_neutralize_exposures.items()
                }
        if self.style_neutralize:
            payload["style_neutralize"] = True
            payload["style_neutralize_exposures"] = {
                name: dict(values) for name, values in (self.style_neutralize_exposures or {}).items()
            }
            payload["style_neutralize_min_samples"] = self.style_neutralize_min_samples
        if self.regime_states is not None:
            payload["regime_states"] = dict(self.regime_states)
            payload["regime_factor_groups"] = {
                label: list(factors) for label, factors in (self.regime_factor_groups or {}).items()
            }
            payload["regime_lag"] = self.regime_lag
        if self.decay_window is not None:
            # Enabled-only, so every candidate that leaves the monitor off
            # keeps its canonical payload (and its identity hash) untouched.
            payload["decay_window"] = self.decay_window
            payload["decay_fail_periods"] = self.decay_fail_periods
            payload["decay_ic_threshold"] = self.decay_ic_threshold
            payload["decay_scale"] = self.decay_scale
        return payload

    def interaction_member_names(self) -> tuple[str, ...]:
        """Blend member names of the interaction terms, in declaration order."""

        return tuple(f"{first}*{second}" for first, second, _ in self.interaction_pairs)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SignalSpec":
        blend_groups = payload.get("blend_groups")
        blend_within = payload.get("blend_within_weights")
        raw_blend_weights = payload.get("blend_weights")
        if raw_blend_weights is not None and not isinstance(raw_blend_weights, Mapping):
            # Same fail-closed shape as the worker for a malformed payload.
            raise ValueError("blend_weights must not be empty")
        raw_interaction_pairs = payload.get("interaction_pairs") or ()
        for pair in raw_interaction_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 3:
                raise ValueError("each interaction pair must be a [first, second, kind] triple")
        raw_industry_map = payload.get("industry_neutralize_industry_map")
        if raw_industry_map is not None and not isinstance(raw_industry_map, Mapping):
            raise ValueError("industry_neutralize_industry_map must map non-empty symbols to non-empty industry labels")
        raw_industry_exposures = payload.get("industry_neutralize_exposures")
        if raw_industry_exposures is not None and not isinstance(raw_industry_exposures, Mapping):
            raise ValueError(
                "industry_neutralize_exposures must map non-empty exposure names to non-empty symbol -> finite number objects"
            )
        raw_style_exposures = payload.get("style_neutralize_exposures")
        if raw_style_exposures is not None and not isinstance(raw_style_exposures, Mapping):
            raise ValueError(
                "style_neutralize_exposures must map non-empty style names to non-empty symbol -> finite number objects"
            )
        raw_regime_states = payload.get("regime_states")
        if raw_regime_states is not None and not isinstance(raw_regime_states, Mapping):
            raise ValueError("regime_states must map date strings to non-empty state labels")
        raw_regime_groups = payload.get("regime_factor_groups")
        if raw_regime_groups is not None and not isinstance(raw_regime_groups, Mapping):
            raise ValueError("regime_factor_groups must map non-empty state labels to non-empty factor lists")
        return cls(
            transform=None if payload.get("transform") is None else str(payload["transform"]),
            transform_scale=float(payload.get("transform_scale", 1.0)),
            transform_shift=float(payload.get("transform_shift", 0.0)),
            winsorize_lower=None if payload.get("winsorize_lower") is None else float(payload["winsorize_lower"]),
            winsorize_upper=None if payload.get("winsorize_upper") is None else float(payload["winsorize_upper"]),
            winsorize_min_samples=int(payload.get("winsorize_min_samples", 1)),
            standardize=None if payload.get("standardize") is None else str(payload["standardize"]),
            standardize_min_std=float(payload.get("standardize_min_std", DEFAULT_STANDARDIZE_MIN_STD)),
            missing_value=None if payload.get("missing_value") is None else str(payload["missing_value"]),
            missing_max_fill_ratio=(
                None if payload.get("missing_max_fill_ratio") is None else float(payload["missing_max_fill_ratio"])
            ),
            blend_weights=(
                None
                if raw_blend_weights is None
                else {str(key): float(value) for key, value in raw_blend_weights.items()}
            ),
            blend_groups=(
                None
                if blend_groups is None
                else {str(group): [str(factor) for factor in factors] for group, factors in blend_groups.items()}
            ),
            blend_group_weights=(
                None
                if payload.get("blend_group_weights") is None
                else {str(key): float(value) for key, value in payload["blend_group_weights"].items()}
            ),
            blend_within_weights=(
                None
                if blend_within is None
                else {
                    str(group): {str(factor): float(weight) for factor, weight in weights.items()}
                    for group, weights in blend_within.items()
                }
            ),
            interaction_pairs=tuple(
                (str(pair[0]), str(pair[1]), str(pair[2])) for pair in raw_interaction_pairs
            ),
            industry_neutralize=payload.get("industry_neutralize", False),
            # Raw passthrough (batch C1 lesson): __post_init__ is the
            # validator, so from_payload must not cast -- int() would silently
            # truncate 2.5 and str()/float() would mask malformed shapes with
            # uncontrolled errors.  Integral floats are normalized (and
            # written back) in __post_init__.
            industry_neutralize_industry_map=raw_industry_map,
            industry_neutralize_method=payload.get("industry_neutralize_method", "demean"),
            industry_neutralize_min_group=payload.get("industry_neutralize_min_group", 3),
            industry_neutralize_version=(
                None if payload.get("industry_neutralize_version") is None else payload["industry_neutralize_version"]
            ),
            industry_neutralize_exposures=raw_industry_exposures,
            style_neutralize=payload.get("style_neutralize", False),
            style_neutralize_exposures=raw_style_exposures,
            style_neutralize_min_samples=payload.get("style_neutralize_min_samples", 5),
            regime_states=raw_regime_states,
            regime_factor_groups=raw_regime_groups,
            regime_lag=payload.get("regime_lag", 1),
            # Raw passthrough (fact 37 discipline): int()/float() here would
            # silently truncate a non-whole payload and bypass the fail-closed
            # validation in __post_init__.
            decay_window=payload.get("decay_window"),
            decay_fail_periods=payload.get("decay_fail_periods"),
            decay_ic_threshold=payload.get("decay_ic_threshold"),
            decay_scale=payload.get("decay_scale"),
        )


@dataclass(frozen=True)
class GateLayer:
    """One quantile screen of the gating cascade (roadmap 5.1 / 5.2).

    ``factor`` must be one of the candidate's factors.  Per date the layer
    keeps the symbols whose post-direction percentile rank of that factor
    passes ``quantile`` (``top`` keeps ``rank >= quantile``, ``bottom`` keeps
    ``rank <= quantile``); the comparison is inclusive, so a threshold that
    falls on a tie keeps every tied name on that side.  A date whose surviving
    set drops below ``min_pass`` fails closed at run time instead of silently
    shrinking the book.
    """

    factor: str
    direction: GateDirection
    quantile: float
    min_pass: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.factor, str) or not self.factor:
            raise ValueError("gate layer requires a non-empty factor name")
        if self.direction not in GATE_DIRECTIONS:
            raise ValueError(f"gate layer direction must be one of {GATE_DIRECTIONS}, got {self.direction!r}")
        if (
            isinstance(self.quantile, bool)
            or not isinstance(self.quantile, (int, float))
            or not math.isfinite(self.quantile)
            or not 0 < self.quantile < 1
        ):
            raise ValueError("gate layer quantile must be a finite number strictly between 0 and 1")
        if int(self.min_pass) < 1:
            raise ValueError("gate layer min_pass must be at least 1")

    def as_payload(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "direction": self.direction,
            "quantile": self.quantile,
            "min_pass": self.min_pass,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GateLayer":
        unknown = set(payload) - GATE_LAYER_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"gating layer has unknown keys: {sorted(unknown)[:3]}")
        # Missing required keys fail closed as ValueError (never a KeyError):
        # proposal-side and worker-side validation share the error shape.
        for key in ("factor", "direction", "quantile"):
            if key not in payload:
                raise ValueError(f"gating layer is missing the {key!r} key")
        return cls(
            factor=str(payload["factor"]),
            direction=str(payload["direction"]),
            quantile=float(payload["quantile"]),
            min_pass=int(payload.get("min_pass", 1)),
        )


@dataclass(frozen=True)
class GatingSpec:
    """Ordered quantile gating cascade (roadmap 5.1 QuantileGate as the
    single-layer case, 5.2 CascadedFactorGating in full).

    Layers apply in order with *and* semantics: a symbol survives a date only
    when it passes every layer, and a symbol that fails any layer has its
    score set to ``NaN`` on that date (the weight builders consume the
    surviving set through their existing ``dropna``).  The ``and/or`` and
    ``final Top-K`` knobs of roadmap 5.2 are deliberately absent from this
    batch: only the intersection semantics is implemented, and the final
    Top-K stays with ``CompositionSpec.top_n`` (TopKSelector, 5.3).

    An empty ``layers`` tuple is the disabled state: ``as_payload`` then emits
    no ``gating`` key at all, so the canonical payload -- and therefore the
    strategy identity hash -- of every pre-gating candidate is unchanged.
    """

    layers: tuple[GateLayer, ...] = ()

    def as_payload(self) -> dict[str, Any] | None:
        if not self.layers:
            return None
        return {"layers": [layer.as_payload() for layer in self.layers]}

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "GatingSpec":
        if payload is None:
            return cls()
        if not isinstance(payload, dict) or "layers" not in payload:
            raise ValueError("candidate gating must be an object with a layers list")
        unknown = set(payload) - GATING_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"candidate gating has unknown keys: {sorted(unknown)[:3]}")
        layers = payload["layers"]
        if not isinstance(layers, list) or not layers:
            raise ValueError("candidate gating layers must be a non-empty list")
        return cls(layers=tuple(GateLayer.from_payload(dict(layer)) for layer in layers))


@dataclass(frozen=True)
class CompositionSpec:
    """Turn a cross-sectional score panel into target portfolio weights.

    ``top_n`` is the number of holdings selected per rebalance date,
    ``weighting`` picks the allocation scheme and ``weight_cap`` optionally
    limits a single holding (the CSI free-cash-flow style indices use this).

    ``weight_floor`` (roadmap 6.2 SingleNameFloor) raises every held name
    below the floor up to it; the raise is funded proportionally by the
    excess of the names above the floor and an infeasible floor fails closed
    instead of pushing the book past 100%.  It must not exceed ``weight_cap``
    when both are provided.  ``weight_floor_allow_cash`` keeps the roadmap's
    cash-residual knob: with ``False`` the book must remain fully invested.

    ``score_transform_*`` (roadmap 7.4 ScoreTransformWeight) only apply with
    ``weighting="score_transform"``: the selected scores are transformed
    (linear ``slope * s``, truncated ``max(s - cut, 0)`` or sigmoid
    ``1 / (1 + exp(-slope * (s - cut)))``) and allocated proportionally.
    """

    top_n: int
    weighting: WeightingScheme = "equal"
    weight_cap: float | None = None
    weight_floor: float | None = None
    weight_floor_allow_cash: bool = True
    score_transform_kind: ScoreTransformKind | None = None
    score_transform_slope: float = 1.0
    score_transform_cut: float = 0.0
    # Roadmap 5.5 HoldingBuffer: a held name survives as long as its score
    # percentile stays at or above ``holding_buffer_retain`` (>= the easier
    # ``holding_buffer_entry`` threshold a fresh name needs to enter), and the
    # book may keep up to ``holding_buffer_max_extra`` rescued holdings beyond
    # ``top_n``.  Percentiles run from the top (best name = 1.0) over the
    # date's valid cross-section.
    holding_buffer_entry: float | None = None
    holding_buffer_retain: float | None = None
    holding_buffer_max_extra: int = 0
    # Roadmap 5.6 DiversitySelector (conservative subset): a candidate is
    # skipped when its correlation with an already-admitted name exceeds this
    # bound; the greedy screen walks the deterministic score order.  The
    # roadmap's per-industry counts need industry labels the strategy
    # pipeline does not have on either side.
    diversity_max_correlation: float | None = None
    # Roadmap 6.3 IndustryWeightCap: the per-industry sum of the target
    # weights must not exceed ``industry_weight_cap``.  The industry mapping
    # (symbol -> industry label) is an explicit configuration input -- the
    # engine has no industry data of its own, so the upper layer injects it.
    # The mapping is an execution detail and stays out of the LLM prompt
    # vocabulary (same convention as ``weight_floor_allow_cash``), but it
    # travels in the canonical payload and the worker whitelist.  Names that
    # are absent from the mapping share one implicit group and the cap applies
    # to that group like any other (conservative reading of an undefined case).
    industry_weight_cap: float | None = None
    industry_map: dict[str, str] | None = None
    # Roadmap 6.8 CashResidualPolicy: the existing convention leaves the
    # post-constraint residual as cash; this knob makes the convention an
    # explicit bound.  When set, every target row whose residual cash exceeds
    # ``cash_residual_max`` fails closed instead of silently holding the
    # under-investment.  ``None`` keeps the pre-change behaviour bit for bit.
    cash_residual_max: float | None = None
    # Roadmap 7.5 InverseVolatilityWeight (weighting scheme
    # ``inverse_volatility``): weights proportional to the inverse of the
    # realised-return volatility observed up to each decision date.  The
    # window is an upper bound on the lookback (shorter history is used as
    # is); ``inverse_vol_min`` floors every volatility estimate;
    # ``inverse_vol_multiply_score`` additionally multiplies by the score.
    # The three fields travel in the canonical payload only while the scheme
    # is selected.
    inverse_vol_window: int = 20
    inverse_vol_min: float | None = None
    inverse_vol_multiply_score: bool = False
    # Roadmap 7.8 HierarchicalWeighting (weighting scheme ``hierarchical``):
    # weight is allocated per group first (equal, or the explicit
    # ``hierarchical_group_weights`` normalised over the groups that actually
    # hold selected names) and split equally inside each group.  The grouping
    # is an explicit configuration input like the industry map; a selected
    # name outside every group fails closed.
    hierarchical_groups: dict[str, tuple[str, ...]] | None = None
    hierarchical_group_weights: dict[str, float] | None = None
    # Roadmap 7.6 RiskParityWeight (weighting scheme ``risk_parity``): risk
    # contributions of the selected names are equal, or match the explicit
    # ``risk_parity_budgets`` when provided.  The covariance is estimated from
    # the realised daily returns up to the decision date (the same returns
    # panel ``inverse_volatility`` consumes) with a ``risk_parity_window``
    # lookback upper bound and a deterministic diagonal shrinkage.  The CCD
    # iteration budget is fixed on both engine sides (RISK_PARITY_MAX_ITER);
    # an unconverged solve fails closed.  ``risk_parity_budgets`` is keyed by
    # symbol, an injection the LLM cannot know, so it stays out of the prompt
    # vocabulary.
    risk_parity_window: int = 60
    risk_parity_budgets: dict[str, float] | None = None
    risk_parity_shrinkage_kind: str = "diagonal"
    risk_parity_shrinkage: float = 0.1
    risk_parity_tolerance: float = 1e-8
    # Roadmap 7.7 FactorRiskBudgetWeight (weighting scheme
    # ``factor_risk_budget``): factor risk budgets are matched in factor space
    # and mapped onto the selected names through the injected exposure matrix.
    # The exposures, factor covariance and idiosyncratic variances are
    # explicit risk-model injections (the engine has no risk-model data of its
    # own); declared simplifications versus a full risk model: the factor
    # budgeting ignores idiosyncratic risk (which enters only through the
    # ``frb_idio_max`` exclusion cap) and the exposure mapping projects
    # non-negatively instead of solving a constrained quadratic program.
    frb_budgets: dict[str, float] | None = None
    frb_exposures: dict[str, dict[str, float]] | None = None
    frb_factor_covariance: dict[str, dict[str, float]] | None = None
    frb_idiosyncratic: dict[str, float] | None = None
    frb_idio_max: float | None = None
    frb_ridge: float = 0.0
    # Roadmap 6.4 IndustryDeviationConstraint: per-industry max *active*
    # deviation |portfolio industry weight - benchmark industry weight|, the
    # industry membership map and the explicit exempt list.  The benchmark
    # industry weights ({industry: weight}) are a frozen PIT injection; an
    # industry absent from the benchmark counts as benchmark weight 0 (the
    # declared convention: industries outside the benchmark universe carry
    # their full portfolio weight as active weight).  The exemption is an
    # explicit industry list, not a boolean -- a boolean has no engine
    # meaning; exempted industries must appear in industry_deviation_max.
    industry_deviation_max: dict[str, float] | None = None
    industry_deviation_industry_map: dict[str, str] | None = None
    industry_deviation_exempt: tuple[str, ...] = ()
    industry_deviation_benchmark: dict[str, float] | None = None
    # Roadmap 6.5 StyleExposureConstraint: per-style active exposure
    # (portfolio exposure minus benchmark exposure) boxed into [lower, upper].
    # Style exposures ({style: {symbol: value}}) and the benchmark exposures
    # ({style: value}) are frozen risk-model injections; both must cover the
    # bounds key set exactly, and missing benchmark styles fail closed (the
    # matrix and the benchmark come from the same risk model, so a missing
    # style is a configuration error, not a sparse-universe convention).
    style_exposure_bounds: dict[str, tuple[float, float]] | None = None
    style_exposure_matrix: dict[str, dict[str, float]] | None = None
    style_exposure_benchmark: dict[str, float] | None = None
    # Roadmap 6.6 BetaConstraint: the portfolio beta sum(beta_i * w_i) boxed
    # into an absolute interval (the conservative reading of "beta 上下限或
    # 主动 beta 上限" -- absolute, not active-vs-benchmark).  Per-symbol PIT
    # betas are an explicit injection.  The roadmap's collinearity risk is
    # handled by the declared downgrade policy: infeasibility fails closed.
    beta_constraint_min: float | None = None
    beta_constraint_max: float | None = None
    beta_by_symbol: dict[str, float] | None = None
    # Roadmap 6.7 FactorExposureConstraint: absolute per-factor exposure
    # intervals over the injected risk-factor exposure matrix (no benchmark:
    # the roadmap states absolute target ranges for this operator).
    factor_exposure_bounds: dict[str, tuple[float, float]] | None = None
    factor_exposure_matrix: dict[str, dict[str, float]] | None = None
    # Roadmap 6.9 ConstrainedWeightProjection knobs.  All four enabled
    # constraint families are projected jointly by one deterministic Dykstra
    # alternating projection (equal-weight L2 distance, fixed constraint
    # order, no constraint priorities -- every constraint is hard, the
    # roadmap's "priority / soft penalty" knobs are deliberately absent); an
    # infeasible or unconverged solve fails closed.  Only present in the
    # payload while at least one constraint family is enabled.
    projection_max_iter: int = 512
    projection_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.weighting not in WEIGHTING_SCHEMES:
            raise ValueError(f"weighting must be one of {WEIGHTING_SCHEMES}, got {self.weighting!r}")
        if self.weight_cap is not None and not 0 < self.weight_cap <= 1:
            raise ValueError("weight_cap must be in (0, 1] when provided")
        if self.weight_floor is not None and not 0 < self.weight_floor <= 1:
            raise ValueError("weight_floor must be in (0, 1] when provided")
        if self.weight_floor is not None and self.weight_cap is not None and self.weight_floor > self.weight_cap:
            raise ValueError("weight_floor must not exceed weight_cap")
        if not isinstance(self.weight_floor_allow_cash, bool):
            raise ValueError("weight_floor_allow_cash must be a boolean")
        if self.score_transform_kind not in (None, *SCORE_TRANSFORM_KINDS):
            raise ValueError(
                f"score_transform_kind must be one of {SCORE_TRANSFORM_KINDS} or None, got {self.score_transform_kind!r}"
            )
        if not math.isfinite(self.score_transform_slope) or self.score_transform_slope <= 0:
            raise ValueError("score_transform_slope must be a finite positive number")
        if not math.isfinite(self.score_transform_cut):
            raise ValueError("score_transform_cut must be finite")
        if self.weighting == "score_transform" and self.score_transform_kind is None:
            raise ValueError("score_transform_kind is required when weighting is score_transform")
        if self.weighting != "score_transform" and self.score_transform_kind is not None:
            raise ValueError("score_transform_kind is only valid with the score_transform weighting")
        if self.holding_buffer_entry is not None and not (
            math.isfinite(self.holding_buffer_entry) and 0 < self.holding_buffer_entry <= 1
        ):
            raise ValueError("holding_buffer_entry must be in (0, 1] when provided")
        if self.holding_buffer_retain is not None and not (
            math.isfinite(self.holding_buffer_retain) and 0 < self.holding_buffer_retain <= 1
        ):
            raise ValueError("holding_buffer_retain must be in (0, 1] when provided")
        if self.holding_buffer_max_extra != int(self.holding_buffer_max_extra):
            raise ValueError("holding_buffer_max_extra must be a whole number")
        if int(self.holding_buffer_max_extra) < 0:
            raise ValueError("holding_buffer_max_extra must not be negative")
        if self.holding_buffer_entry is None:
            if self.holding_buffer_retain is not None or int(self.holding_buffer_max_extra) != 0:
                raise ValueError("holding_buffer_retain and holding_buffer_max_extra require holding_buffer_entry")
        else:
            if self.holding_buffer_retain is None:
                # Staying is never harder than entering; an unset retention
                # threshold therefore collapses onto the entry threshold.
                object.__setattr__(self, "holding_buffer_retain", self.holding_buffer_entry)
            elif self.holding_buffer_retain > self.holding_buffer_entry:
                raise ValueError("holding_buffer_retain must not exceed holding_buffer_entry")
        if self.diversity_max_correlation is not None and not (
            math.isfinite(self.diversity_max_correlation) and -1 <= self.diversity_max_correlation <= 1
        ):
            raise ValueError("diversity_max_correlation must be in [-1, 1] when provided")
        if self.industry_weight_cap is not None and not 0 < self.industry_weight_cap <= 1:
            raise ValueError("industry_weight_cap must be in (0, 1] when provided")
        if self.industry_map is not None:
            if self.industry_weight_cap is None:
                raise ValueError("industry_map is only valid with industry_weight_cap")
            for symbol, label in self.industry_map.items():
                if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                    raise ValueError("industry_map must map non-empty symbols to non-empty industry labels")
            object.__setattr__(self, "industry_map", dict(self.industry_map))
        if self.weight_floor is not None and self.industry_weight_cap is not None and self.weight_floor > self.industry_weight_cap:
            raise ValueError("weight_floor must not exceed industry_weight_cap")
        if self.cash_residual_max is not None and not 0 < self.cash_residual_max <= 1:
            raise ValueError("cash_residual_max must be in (0, 1] when provided")
        if self.inverse_vol_window != int(self.inverse_vol_window):
            raise ValueError("inverse_vol_window must be a whole number")
        if int(self.inverse_vol_window) < 2:
            raise ValueError("inverse_vol_window must be at least 2")
        if self.inverse_vol_min is not None and not (math.isfinite(self.inverse_vol_min) and self.inverse_vol_min > 0):
            raise ValueError("inverse_vol_min must be a positive finite number when provided")
        if not isinstance(self.inverse_vol_multiply_score, bool):
            raise ValueError("inverse_vol_multiply_score must be a boolean")
        if self.weighting == "hierarchical" and self.hierarchical_groups is None:
            raise ValueError("hierarchical weighting requires hierarchical_groups")
        if self.hierarchical_groups is not None:
            if self.weighting != "hierarchical":
                raise ValueError("hierarchical_groups is only valid with the hierarchical weighting")
            if not isinstance(self.hierarchical_groups, dict) or not self.hierarchical_groups:
                raise ValueError("hierarchical_groups must be a non-empty object when provided")
            seen: dict[str, str] = {}
            for group, members in self.hierarchical_groups.items():
                if not isinstance(group, str) or not group:
                    raise ValueError("hierarchical_groups group names must be non-empty strings")
                if not len(members):
                    raise ValueError(f"hierarchical group {group!r} must not be empty")
                for member in members:
                    if not isinstance(member, str) or not member:
                        raise ValueError("hierarchical_groups members must be non-empty strings")
                    if member in seen and seen[member] != group:
                        raise ValueError("hierarchical_groups members must not appear in two groups")
                    seen[member] = group
            object.__setattr__(
                self, "hierarchical_groups", {group: tuple(members) for group, members in self.hierarchical_groups.items()}
            )
        if self.hierarchical_group_weights is not None:
            if self.hierarchical_groups is None:
                raise ValueError("hierarchical_group_weights is only valid with hierarchical_groups")
            if not isinstance(self.hierarchical_group_weights, dict) or set(self.hierarchical_group_weights) != set(
                self.hierarchical_groups
            ):
                raise ValueError("hierarchical_group_weights must cover exactly the hierarchical_groups keys")
            for value in self.hierarchical_group_weights.values():
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError("hierarchical_group_weights values must be finite positive numbers")
            object.__setattr__(self, "hierarchical_group_weights", dict(self.hierarchical_group_weights))
        # Roadmap 7.6 RiskParityWeight.
        if self.weighting != "risk_parity" and (
            self.risk_parity_budgets is not None
            or self.risk_parity_window != 60
            or self.risk_parity_shrinkage_kind != "diagonal"
            or self.risk_parity_shrinkage != 0.1
            or self.risk_parity_tolerance != 1e-8
        ):
            raise ValueError("risk_parity parameters require the risk_parity weighting")
        risk_window = _whole_universe_number(self.risk_parity_window, "risk_parity_window must be a whole number")
        if risk_window < 2:
            raise ValueError("risk_parity_window must be at least 2")
        if not isinstance(self.risk_parity_window, int) or isinstance(self.risk_parity_window, bool):
            object.__setattr__(self, "risk_parity_window", risk_window)
        if self.risk_parity_budgets is not None:
            if not isinstance(self.risk_parity_budgets, dict) or not self.risk_parity_budgets:
                raise ValueError("risk_parity_budgets must map non-empty symbols to finite positive numbers")
            for symbol, budget in self.risk_parity_budgets.items():
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError("risk_parity_budgets must map non-empty symbols to finite positive numbers")
                if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(float(budget)) or float(budget) <= 0:
                    raise ValueError("risk_parity_budgets must map non-empty symbols to finite positive numbers")
            object.__setattr__(
                self, "risk_parity_budgets", {str(symbol): float(budget) for symbol, budget in self.risk_parity_budgets.items()}
            )
        if self.risk_parity_shrinkage_kind not in RISK_PARITY_SHRINKAGE_KINDS:
            raise ValueError(
                f"risk_parity_shrinkage_kind must be one of {RISK_PARITY_SHRINKAGE_KINDS}, "
                f"got {self.risk_parity_shrinkage_kind!r}"
            )
        if (
            isinstance(self.risk_parity_shrinkage, bool)
            or not isinstance(self.risk_parity_shrinkage, (int, float))
            or not math.isfinite(float(self.risk_parity_shrinkage))
            or not 0 <= float(self.risk_parity_shrinkage) < 1
        ):
            raise ValueError("risk_parity_shrinkage must be in [0, 1) when provided")
        object.__setattr__(self, "risk_parity_shrinkage", float(self.risk_parity_shrinkage))
        if (
            isinstance(self.risk_parity_tolerance, bool)
            or not isinstance(self.risk_parity_tolerance, (int, float))
            or not math.isfinite(float(self.risk_parity_tolerance))
            or float(self.risk_parity_tolerance) <= 0
        ):
            raise ValueError("risk_parity_tolerance must be a finite positive number")
        object.__setattr__(self, "risk_parity_tolerance", float(self.risk_parity_tolerance))
        # Roadmap 7.7 FactorRiskBudgetWeight.
        if self.weighting == "factor_risk_budget":
            for field, message in (
                ("frb_budgets", "factor_risk_budget weighting requires frb_budgets"),
                ("frb_exposures", "factor_risk_budget weighting requires frb_exposures"),
                ("frb_factor_covariance", "factor_risk_budget weighting requires frb_factor_covariance"),
                ("frb_idiosyncratic", "factor_risk_budget weighting requires frb_idiosyncratic"),
            ):
                if getattr(self, field) is None:
                    raise ValueError(message)
        elif any(
            getattr(self, field) is not None
            for field in (
                "frb_budgets",
                "frb_exposures",
                "frb_factor_covariance",
                "frb_idiosyncratic",
                "frb_idio_max",
            )
        ) or self.frb_ridge != 0.0:
            raise ValueError("factor_risk_budget parameters require the factor_risk_budget weighting")
        if self.frb_budgets is not None:
            if not isinstance(self.frb_budgets, dict) or not self.frb_budgets:
                raise ValueError("frb_budgets must map non-empty factor names to finite positive numbers")
            for factor_name, budget in self.frb_budgets.items():
                if not isinstance(factor_name, str) or not factor_name:
                    raise ValueError("frb_budgets must map non-empty factor names to finite positive numbers")
                if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(float(budget)) or float(budget) <= 0:
                    raise ValueError("frb_budgets must map non-empty factor names to finite positive numbers")
            object.__setattr__(
                self, "frb_budgets", {str(factor_name): float(budget) for factor_name, budget in self.frb_budgets.items()}
            )
        if self.frb_exposures is not None:
            _validate_exposure_map(
                self.frb_exposures,
                "frb_exposures must map non-empty factor names to non-empty symbol -> finite number objects",
            )
        if self.frb_factor_covariance is not None:
            _validate_exposure_map(
                self.frb_factor_covariance,
                "frb_factor_covariance must map non-empty factor names to non-empty factor -> finite number objects",
            )
        if self.frb_idiosyncratic is not None:
            if not isinstance(self.frb_idiosyncratic, dict) or not self.frb_idiosyncratic:
                raise ValueError("frb_idiosyncratic must map non-empty symbols to finite non-negative numbers")
            for symbol, variance in self.frb_idiosyncratic.items():
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError("frb_idiosyncratic must map non-empty symbols to finite non-negative numbers")
                if (
                    isinstance(variance, bool)
                    or not isinstance(variance, (int, float))
                    or not math.isfinite(float(variance))
                    or float(variance) < 0
                ):
                    raise ValueError("frb_idiosyncratic must map non-empty symbols to finite non-negative numbers")
            object.__setattr__(
                self, "frb_idiosyncratic", {str(symbol): float(variance) for symbol, variance in self.frb_idiosyncratic.items()}
            )
        if self.frb_idio_max is not None and (
            isinstance(self.frb_idio_max, bool)
            or not isinstance(self.frb_idio_max, (int, float))
            or not math.isfinite(float(self.frb_idio_max))
            or float(self.frb_idio_max) <= 0
        ):
            raise ValueError("frb_idio_max must be a positive finite number when provided")
        if self.frb_idio_max is not None:
            object.__setattr__(self, "frb_idio_max", float(self.frb_idio_max))
        if (
            isinstance(self.frb_ridge, bool)
            or not isinstance(self.frb_ridge, (int, float))
            or not math.isfinite(float(self.frb_ridge))
            or float(self.frb_ridge) < 0
        ):
            raise ValueError("frb_ridge must be a finite non-negative number")
        object.__setattr__(self, "frb_ridge", float(self.frb_ridge))
        # Roadmap 6.4 IndustryDeviationConstraint (batch C3).
        any_constraint = self.industry_deviation_max is not None
        if self.industry_deviation_max is not None:
            if not isinstance(self.industry_deviation_max, dict) or not self.industry_deviation_max:
                raise ValueError("industry_deviation_max must map non-empty industry names to finite positive deviations")
            for industry, deviation in self.industry_deviation_max.items():
                if not isinstance(industry, str) or not industry:
                    raise ValueError("industry_deviation_max must map non-empty industry names to finite positive deviations")
                if (
                    isinstance(deviation, bool)
                    or not isinstance(deviation, (int, float))
                    or not math.isfinite(float(deviation))
                    or float(deviation) <= 0
                ):
                    raise ValueError("industry_deviation_max must map non-empty industry names to finite positive deviations")
            object.__setattr__(
                self,
                "industry_deviation_max",
                {str(industry): float(deviation) for industry, deviation in self.industry_deviation_max.items()},
            )
            if self.industry_deviation_industry_map is None:
                raise ValueError("industry deviation requires industry_deviation_industry_map")
            if not isinstance(self.industry_deviation_industry_map, dict) or not self.industry_deviation_industry_map:
                raise ValueError("industry_deviation_industry_map must map non-empty symbols to non-empty industry labels")
            for symbol, label in self.industry_deviation_industry_map.items():
                if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                    raise ValueError("industry_deviation_industry_map must map non-empty symbols to non-empty industry labels")
            object.__setattr__(self, "industry_deviation_industry_map", dict(self.industry_deviation_industry_map))
            if not isinstance(self.industry_deviation_exempt, (tuple, list)):
                raise ValueError("industry_deviation_exempt must be a list of non-empty industry names")
            for exempt in self.industry_deviation_exempt:
                if not isinstance(exempt, str) or not exempt:
                    raise ValueError("industry_deviation_exempt must be a list of non-empty industry names")
                if exempt not in self.industry_deviation_max:
                    raise ValueError("industry_deviation_exempt industries must appear in industry_deviation_max")
            object.__setattr__(self, "industry_deviation_exempt", tuple(self.industry_deviation_exempt))
            if self.industry_deviation_benchmark is None:
                raise ValueError("industry deviation requires industry_deviation_benchmark")
            if not isinstance(self.industry_deviation_benchmark, dict) or not self.industry_deviation_benchmark:
                raise ValueError("industry_deviation_benchmark must map non-empty industry names to finite non-negative weights")
            for industry, weight in self.industry_deviation_benchmark.items():
                if not isinstance(industry, str) or not industry:
                    raise ValueError("industry_deviation_benchmark must map non-empty industry names to finite non-negative weights")
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(float(weight))
                    or float(weight) < 0
                ):
                    raise ValueError("industry_deviation_benchmark must map non-empty industry names to finite non-negative weights")
            object.__setattr__(
                self,
                "industry_deviation_benchmark",
                {str(industry): float(weight) for industry, weight in self.industry_deviation_benchmark.items()},
            )
        elif (
            self.industry_deviation_industry_map is not None
            or self.industry_deviation_exempt
            or self.industry_deviation_benchmark is not None
        ):
            raise ValueError("industry deviation parameters require industry_deviation_max")
        # Roadmap 6.5 StyleExposureConstraint (batch C3).
        any_constraint = any_constraint or self.style_exposure_bounds is not None
        if self.style_exposure_bounds is not None:
            _validate_bounds_map(
                self.style_exposure_bounds,
                "style_exposure_bounds must map non-empty style names to [lower, upper] pairs with lower <= upper",
            )
            object.__setattr__(
                self,
                "style_exposure_bounds",
                {str(name): (float(lower), float(upper)) for name, (lower, upper) in self.style_exposure_bounds.items()},
            )
            if self.style_exposure_matrix is None:
                raise ValueError("style exposure constraint requires style_exposure_matrix")
            _validate_exposure_map(
                self.style_exposure_matrix,
                "style_exposure_matrix must map non-empty style names to non-empty symbol -> finite number objects",
            )
            if not set(self.style_exposure_bounds) <= set(self.style_exposure_matrix):
                raise ValueError("style_exposure_matrix must cover every bounded style")
            if self.style_exposure_benchmark is None:
                raise ValueError("style exposure constraint requires style_exposure_benchmark")
            if not isinstance(self.style_exposure_benchmark, dict) or not self.style_exposure_benchmark:
                raise ValueError("style_exposure_benchmark must map non-empty style names to finite numbers")
            for style, benchmark in self.style_exposure_benchmark.items():
                if not isinstance(style, str) or not style:
                    raise ValueError("style_exposure_benchmark must map non-empty style names to finite numbers")
                if isinstance(benchmark, bool) or not isinstance(benchmark, (int, float)) or not math.isfinite(float(benchmark)):
                    raise ValueError("style_exposure_benchmark must map non-empty style names to finite numbers")
            if not set(self.style_exposure_bounds) <= set(self.style_exposure_benchmark):
                raise ValueError("style_exposure_benchmark must cover every bounded style")
            object.__setattr__(
                self,
                "style_exposure_benchmark",
                {str(style): float(benchmark) for style, benchmark in self.style_exposure_benchmark.items()},
            )
        elif self.style_exposure_matrix is not None or self.style_exposure_benchmark is not None:
            raise ValueError("style exposure parameters require style_exposure_bounds")
        # Roadmap 6.6 BetaConstraint (batch C3).
        beta_enabled = self.beta_constraint_min is not None or self.beta_constraint_max is not None
        any_constraint = any_constraint or beta_enabled
        if beta_enabled:
            if self.beta_constraint_min is not None and (
                isinstance(self.beta_constraint_min, bool)
                or not isinstance(self.beta_constraint_min, (int, float))
                or not math.isfinite(float(self.beta_constraint_min))
            ):
                raise ValueError("beta_constraint_min must be a finite number when provided")
            if self.beta_constraint_max is not None and (
                isinstance(self.beta_constraint_max, bool)
                or not isinstance(self.beta_constraint_max, (int, float))
                or not math.isfinite(float(self.beta_constraint_max))
            ):
                raise ValueError("beta_constraint_max must be a finite number when provided")
            if (
                self.beta_constraint_min is not None
                and self.beta_constraint_max is not None
                and float(self.beta_constraint_min) > float(self.beta_constraint_max)
            ):
                raise ValueError("beta_constraint_min must not exceed beta_constraint_max")
            if self.beta_constraint_min is not None:
                object.__setattr__(self, "beta_constraint_min", float(self.beta_constraint_min))
            if self.beta_constraint_max is not None:
                object.__setattr__(self, "beta_constraint_max", float(self.beta_constraint_max))
            if self.beta_by_symbol is None:
                raise ValueError("beta constraint requires beta_by_symbol")
            if not isinstance(self.beta_by_symbol, dict) or not self.beta_by_symbol:
                raise ValueError("beta_by_symbol must map non-empty symbols to finite numbers")
            for symbol, beta in self.beta_by_symbol.items():
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError("beta_by_symbol must map non-empty symbols to finite numbers")
                if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
                    raise ValueError("beta_by_symbol must map non-empty symbols to finite numbers")
            object.__setattr__(self, "beta_by_symbol", {str(symbol): float(beta) for symbol, beta in self.beta_by_symbol.items()})
        elif self.beta_by_symbol is not None:
            raise ValueError("beta constraint parameters require beta_constraint_min or beta_constraint_max")
        # Roadmap 6.7 FactorExposureConstraint (batch C3).
        any_constraint = any_constraint or self.factor_exposure_bounds is not None
        if self.factor_exposure_bounds is not None:
            _validate_bounds_map(
                self.factor_exposure_bounds,
                "factor_exposure_bounds must map non-empty factor names to [lower, upper] pairs with lower <= upper",
            )
            object.__setattr__(
                self,
                "factor_exposure_bounds",
                {str(name): (float(lower), float(upper)) for name, (lower, upper) in self.factor_exposure_bounds.items()},
            )
            if self.factor_exposure_matrix is None:
                raise ValueError("factor exposure constraint requires factor_exposure_matrix")
            _validate_exposure_map(
                self.factor_exposure_matrix,
                "factor_exposure_matrix must map non-empty factor names to non-empty symbol -> finite number objects",
            )
            if not set(self.factor_exposure_bounds) <= set(self.factor_exposure_matrix):
                raise ValueError("factor_exposure_matrix must cover every bounded factor")
        elif self.factor_exposure_matrix is not None:
            raise ValueError("factor exposure parameters require factor_exposure_bounds")
        # Roadmap 6.9 ConstrainedWeightProjection knobs (batch C3).
        if not any_constraint and (self.projection_max_iter != 512 or self.projection_tolerance != 1e-9):
            raise ValueError("projection parameters require an exposure constraint")
        projection_max_iter = _whole_universe_number(self.projection_max_iter, "projection_max_iter must be a whole number")
        if projection_max_iter < 1:
            raise ValueError("projection_max_iter must be at least 1")
        if not isinstance(self.projection_max_iter, int) or isinstance(self.projection_max_iter, bool):
            object.__setattr__(self, "projection_max_iter", projection_max_iter)
        if (
            isinstance(self.projection_tolerance, bool)
            or not isinstance(self.projection_tolerance, (int, float))
            or not math.isfinite(float(self.projection_tolerance))
            or float(self.projection_tolerance) <= 0
        ):
            raise ValueError("projection_tolerance must be a finite positive number")
        object.__setattr__(self, "projection_tolerance", float(self.projection_tolerance))

    def as_payload(self) -> dict[str, Any]:
        """Canonical JSON-serialisable form; also feeds ``strategy_id_for``.

        The floor and score-transform fields travel only while enabled, so
        every pre-existing candidate keeps its canonical payload (and
        therefore its strategy identity hash) unchanged.
        """

        payload = {"top_n": self.top_n, "weighting": self.weighting, "weight_cap": self.weight_cap}
        if self.weighting == "score_transform":
            payload["score_transform_kind"] = self.score_transform_kind
            payload["score_transform_slope"] = self.score_transform_slope
            payload["score_transform_cut"] = self.score_transform_cut
        if self.weight_floor is not None:
            payload["weight_floor"] = self.weight_floor
            payload["weight_floor_allow_cash"] = self.weight_floor_allow_cash
        if self.holding_buffer_entry is not None:
            payload["holding_buffer_entry"] = self.holding_buffer_entry
            payload["holding_buffer_retain"] = self.holding_buffer_retain
            payload["holding_buffer_max_extra"] = self.holding_buffer_max_extra
        if self.diversity_max_correlation is not None:
            payload["diversity_max_correlation"] = self.diversity_max_correlation
        if self.industry_weight_cap is not None:
            payload["industry_weight_cap"] = self.industry_weight_cap
            payload["industry_map"] = dict(self.industry_map) if self.industry_map is not None else {}
        if self.cash_residual_max is not None:
            payload["cash_residual_max"] = self.cash_residual_max
        if self.weighting == "inverse_volatility":
            payload["inverse_vol_window"] = self.inverse_vol_window
            payload["inverse_vol_min"] = self.inverse_vol_min
            payload["inverse_vol_multiply_score"] = self.inverse_vol_multiply_score
        if self.weighting == "hierarchical":
            # Group *members* are sorted so that two specs differing only in
            # the order inside a group share one canonical payload (and hence
            # one strategy identity hash); the weighting itself never reads
            # member order.  Group keys are handled by the canonical JSON
            # dump, which sorts object keys.
            payload["hierarchical_groups"] = {group: sorted(members) for group, members in self.hierarchical_groups.items()}
            if self.hierarchical_group_weights is not None:
                payload["hierarchical_group_weights"] = dict(self.hierarchical_group_weights)
        if self.weighting == "risk_parity":
            payload["risk_parity_window"] = self.risk_parity_window
            if self.risk_parity_budgets is not None:
                payload["risk_parity_budgets"] = dict(self.risk_parity_budgets)
            payload["risk_parity_shrinkage_kind"] = self.risk_parity_shrinkage_kind
            payload["risk_parity_shrinkage"] = self.risk_parity_shrinkage
            payload["risk_parity_tolerance"] = self.risk_parity_tolerance
        if self.weighting == "factor_risk_budget":
            payload["frb_budgets"] = dict(self.frb_budgets or {})
            payload["frb_exposures"] = {
                name: dict(values) for name, values in (self.frb_exposures or {}).items()
            }
            payload["frb_factor_covariance"] = {
                name: dict(values) for name, values in (self.frb_factor_covariance or {}).items()
            }
            payload["frb_idiosyncratic"] = dict(self.frb_idiosyncratic or {})
            if self.frb_idio_max is not None:
                payload["frb_idio_max"] = self.frb_idio_max
            payload["frb_ridge"] = self.frb_ridge
        constraint_enabled = False
        if self.industry_deviation_max is not None:
            constraint_enabled = True
            payload["industry_deviation_max"] = dict(self.industry_deviation_max)
            payload["industry_deviation_industry_map"] = dict(self.industry_deviation_industry_map or {})
            if self.industry_deviation_exempt:
                payload["industry_deviation_exempt"] = list(self.industry_deviation_exempt)
            payload["industry_deviation_benchmark"] = dict(self.industry_deviation_benchmark or {})
        if self.style_exposure_bounds is not None:
            constraint_enabled = True
            payload["style_exposure_bounds"] = {
                name: [lower, upper] for name, (lower, upper) in self.style_exposure_bounds.items()
            }
            payload["style_exposure_matrix"] = {
                name: dict(values) for name, values in (self.style_exposure_matrix or {}).items()
            }
            payload["style_exposure_benchmark"] = dict(self.style_exposure_benchmark or {})
        if self.beta_constraint_min is not None or self.beta_constraint_max is not None:
            constraint_enabled = True
            if self.beta_constraint_min is not None:
                payload["beta_constraint_min"] = self.beta_constraint_min
            if self.beta_constraint_max is not None:
                payload["beta_constraint_max"] = self.beta_constraint_max
            payload["beta_by_symbol"] = dict(self.beta_by_symbol or {})
        if self.factor_exposure_bounds is not None:
            constraint_enabled = True
            payload["factor_exposure_bounds"] = {
                name: [lower, upper] for name, (lower, upper) in self.factor_exposure_bounds.items()
            }
            payload["factor_exposure_matrix"] = {
                name: dict(values) for name, values in (self.factor_exposure_matrix or {}).items()
            }
        if constraint_enabled:
            payload["projection_max_iter"] = self.projection_max_iter
            payload["projection_tolerance"] = self.projection_tolerance
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CompositionSpec":
        return cls(
            top_n=int(payload["top_n"]),
            weighting=str(payload.get("weighting", "equal")),
            weight_cap=None if payload.get("weight_cap") is None else float(payload["weight_cap"]),
            weight_floor=None if payload.get("weight_floor") is None else float(payload["weight_floor"]),
            weight_floor_allow_cash=bool(payload.get("weight_floor_allow_cash", True)),
            score_transform_kind=(
                None if payload.get("score_transform_kind") is None else str(payload["score_transform_kind"])
            ),
            score_transform_slope=float(payload.get("score_transform_slope", 1.0)),
            score_transform_cut=float(payload.get("score_transform_cut", 0.0)),
            holding_buffer_entry=(
                None if payload.get("holding_buffer_entry") is None else float(payload["holding_buffer_entry"])
            ),
            holding_buffer_retain=(
                None if payload.get("holding_buffer_retain") is None else float(payload["holding_buffer_retain"])
            ),
            holding_buffer_max_extra=int(payload.get("holding_buffer_max_extra", 0)),
            diversity_max_correlation=(
                None if payload.get("diversity_max_correlation") is None else float(payload["diversity_max_correlation"])
            ),
            industry_weight_cap=(
                None if payload.get("industry_weight_cap") is None else float(payload["industry_weight_cap"])
            ),
            industry_map=(
                None
                if payload.get("industry_map") is None
                else {str(symbol): str(label) for symbol, label in payload["industry_map"].items()}
            ),
            cash_residual_max=None if payload.get("cash_residual_max") is None else float(payload["cash_residual_max"]),
            inverse_vol_window=20 if payload.get("inverse_vol_window") is None else int(payload["inverse_vol_window"]),
            inverse_vol_min=None if payload.get("inverse_vol_min") is None else float(payload["inverse_vol_min"]),
            inverse_vol_multiply_score=bool(payload.get("inverse_vol_multiply_score", False)),
            hierarchical_groups=(
                None
                if payload.get("hierarchical_groups") is None
                else {
                    str(group): tuple(str(member) for member in members)
                    for group, members in payload["hierarchical_groups"].items()
                }
            ),
            hierarchical_group_weights=(
                None
                if payload.get("hierarchical_group_weights") is None
                else {str(group): float(value) for group, value in payload["hierarchical_group_weights"].items()}
            ),
            risk_parity_window=payload.get("risk_parity_window", 60),
            risk_parity_budgets=payload.get("risk_parity_budgets"),
            risk_parity_shrinkage_kind=payload.get("risk_parity_shrinkage_kind", "diagonal"),
            risk_parity_shrinkage=payload.get("risk_parity_shrinkage", 0.1),
            risk_parity_tolerance=payload.get("risk_parity_tolerance", 1e-8),
            frb_budgets=payload.get("frb_budgets"),
            frb_exposures=payload.get("frb_exposures"),
            frb_factor_covariance=payload.get("frb_factor_covariance"),
            frb_idiosyncratic=payload.get("frb_idiosyncratic"),
            frb_idio_max=payload.get("frb_idio_max"),
            frb_ridge=payload.get("frb_ridge", 0.0),
            # Batch C3 raw passthrough (fact 37): __post_init__ is the
            # validator; normalization write-backs happen there.
            industry_deviation_max=payload.get("industry_deviation_max"),
            industry_deviation_industry_map=payload.get("industry_deviation_industry_map"),
            industry_deviation_exempt=payload.get("industry_deviation_exempt") or (),
            industry_deviation_benchmark=payload.get("industry_deviation_benchmark"),
            style_exposure_bounds=payload.get("style_exposure_bounds"),
            style_exposure_matrix=payload.get("style_exposure_matrix"),
            style_exposure_benchmark=payload.get("style_exposure_benchmark"),
            beta_constraint_min=payload.get("beta_constraint_min"),
            beta_constraint_max=payload.get("beta_constraint_max"),
            beta_by_symbol=payload.get("beta_by_symbol"),
            factor_exposure_bounds=payload.get("factor_exposure_bounds"),
            factor_exposure_matrix=payload.get("factor_exposure_matrix"),
            projection_max_iter=payload.get("projection_max_iter", 512),
            projection_tolerance=payload.get("projection_tolerance", 1e-9),
        )


def _whole_backtest_number(value: Any, name: str, *, minimum: int) -> int:
    """Validate an integer-valued execution knob and return it as ``int``.

    A whole-valued float (``2.0``) denotes the same quantity as the integer and
    is accepted and normalised -- normalising keeps both mirrors' canonical
    payloads hash-identical -- while booleans, non-numeric values and genuinely
    fractional numbers fail closed with the documented messages.
    """

    message = f"{name} must be a whole number when provided"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(message)
    if float(value) != int(value):
        raise ValueError(message)
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum} when provided")
    return number


def _bounded_backtest_number(
    value: Any,
    message: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    lower_open: bool = True,
    upper_open: bool = False,
) -> float:
    """Validate a finite execution knob against a bounded range.

    Defaults express the usual ``(0, 1]`` shape; ``upper=None`` drops the upper
    bound and ``lower_open=False`` admits the lower bound itself.  Booleans are
    rejected on purpose: ``True`` is not a rate.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(message)
    number = float(value)
    if lower is not None and (number <= lower if lower_open else number < lower):
        raise ValueError(message)
    if upper is not None and (number >= upper if upper_open else number > upper):
        raise ValueError(message)
    return number


#: The three escalatable drawdown states.  ``normal`` is deliberately absent:
#: it is the machine's resting state and always scales by 1.0.
DRAWDOWN_STATES = ("warning", "reduce", "halt")


def _drawdown_map(
    value: Any, name: str, *, message: str, upper: float, positive: bool = False
) -> dict[str, float]:
    """Validate a ``{state: number}`` map covering exactly the three states.

    ``positive`` picks the threshold range ``(0, upper]`` over the scale range
    ``[0, upper]``: a threshold of zero would be armed from the very first day,
    so the boundary belongs to the scales only.
    """

    if not isinstance(value, Mapping) or set(value) != set(DRAWDOWN_STATES):
        raise ValueError(f"{name} must cover exactly the drawdown states")
    out: dict[str, float] = {}
    for state in DRAWDOWN_STATES:
        entry = value[state]
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(float(entry)):
            raise ValueError(message)
        number = float(entry)
        lower = 0.0 < number if positive else 0.0 <= number
        if not lower or number > upper:
            raise ValueError(message)
        out[state] = number
    return out


@dataclass(frozen=True)
class BacktestSpec:
    """Execution assumptions of a strategy backtest.

    ``signal_lag`` is the number of trading days between a decision date and
    the first day the resulting weights can be traded.  ``1`` is the minimum on
    purpose: ``0`` would mean trading on a signal computed with the same day's
    prices, i.e. look-ahead. Transaction costs are deliberately absent: they
    are an immutable system execution contract, not candidate state.

    ``drift_band_name`` / ``drift_band_portfolio`` (roadmap 8.3
    DriftBandRebalance) let a rebalance be skipped while the drifted book
    stays inside the band: the per-name band tolerates an absolute weight
    change on any single name, the portfolio band tolerates the portfolio-wide
    half-L1 change *including cash* -- the same measurement convention the
    engine reports as turnover, so the two numbers are directly comparable.
    The band comparison is the one deliberate consumer of drifted carried
    weights (tolerating drift is its business purpose); the composition-layer
    cap and floor keep judging target weights.  A rebalance is skipped when
    neither band is exceeded, and ``drift_band_forced_days`` forces a trade at
    least this many trading days after the previous executed one.  At least
    one band is required before a forced cadence may be declared.

    ``turnover_shrinkage`` (roadmap 8.5 TurnoverShrinkage, soft) executes only
    ``1 - turnover_shrinkage`` of the move from the carried weights to the
    target on every rebalance; ``turnover_max_step`` (hard) additionally caps
    the executed step at this much half-L1 turnover including cash.  Both
    reshape the executed trade -- the cost formulas stay untouched and simply
    see a smaller delta.

    ``min_holding_days`` (roadmap 8.4 MinimumHoldingPeriod) blocks the sale of
    any name for this many completed trading days after the trade that entered
    it: the executed weight of a locked name is raised back to its carried
    weight and the remaining names are scaled down proportionally so the book
    never exceeds its executed total.  When the unlocked book cannot fund the
    raise-back, the whole rebalance is abandoned and the book keeps its
    carried weights (the same skip semantics as a band skip).  The clamp runs
    after the drift-band gate and the turnover shrinkage, and it never
    applies to a name the selection never held.

    All six fields default to the disabled state and travel in the canonical
    payload only while enabled, so every pre-existing candidate keeps its
    strategy identity hash unchanged.

    The order layer and the two portfolio-risk scalings (roadmap 8.6-8.9 and
    9.1/9.2) add a further group of optional fields, all disabled by default
    and therefore absent from the canonical payload of every earlier
    candidate:

    * ``participation_*`` (8.6 ParticipationCap): every name's target move is
      split across days at ``participation_max_rate`` of its trailed traded
      amount; ``participation_max_days`` bounds how long an unfilled slice
      survives, ``participation_unfilled`` chooses carry-or-cancel, and
      ``participation_amount_window`` is the trailing window the amount is
      averaged over.  Mutually exclusive with ``staged_max_days`` -- both
      allocate a multi-day plan and the engine supports one allocator.
    * ``lot_rounding`` / ``lot_min_notional`` / ``lot_rounding_mode`` (8.7
      LotRounding): snap traded amounts onto the lot grid, ``down`` by default
      so the rounded amount never exceeds the target.
    * ``side_aware_tradability`` (8.8): ``block_both`` / ``block_buy`` /
      ``block_sell``, the state-to-blocked-direction rule rather than a list
      of directions to inspect.
    * ``staged_*`` (8.9 StagedExecution): spread a fresh target over at most
      ``staged_max_days`` days, served in ``staged_priority`` order.
    * ``vol_target_*`` (9.1 PortfolioVolatilityTarget): scale the whole target
      row so realised volatility tracks ``vol_target_annual``.
    * ``drawdown_*`` (9.2 DrawdownStateMachine): the frozen state ladder
      (thresholds, confirmation days, recovery width and per-state scales).

    Every one of these reads the injected ``execution_data`` panels; none of
    them enters the LLM prompt vocabulary.
    """

    signal_lag: int = 1
    trading_days_per_year: int = 252
    drift_band_name: float | None = None
    drift_band_portfolio: float | None = None
    drift_band_forced_days: int | None = None
    turnover_shrinkage: float | None = None
    turnover_max_step: float | None = None
    min_holding_days: int | None = None
    participation_max_rate: float | None = None
    participation_max_days: int | None = None
    participation_unfilled: str | None = None
    participation_amount_window: int | None = None
    lot_rounding: bool = False
    lot_min_notional: float | None = None
    lot_rounding_mode: str | None = None
    side_aware_tradability: str | None = None
    staged_max_days: int | None = None
    staged_priority: str | None = None
    vol_target_annual: float | None = None
    vol_target_window: int | None = None
    vol_target_min_scale: float | None = None
    vol_target_max_scale: float | None = None
    drawdown_thresholds: dict[str, float] | None = None
    drawdown_confirm_days: int | None = None
    drawdown_recovery: float | None = None
    drawdown_scales: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.signal_lag < 1:
            raise ValueError("signal_lag must be at least 1; 0 would be look-ahead")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")
        if self.drift_band_name is not None and not (
            math.isfinite(self.drift_band_name) and 0 < self.drift_band_name <= 1
        ):
            raise ValueError("drift_band_name must be in (0, 1] when provided")
        if self.drift_band_portfolio is not None and not (
            math.isfinite(self.drift_band_portfolio) and 0 < self.drift_band_portfolio <= 1
        ):
            raise ValueError("drift_band_portfolio must be in (0, 1] when provided")
        if self.drift_band_forced_days is not None:
            if self.drift_band_forced_days != int(self.drift_band_forced_days):
                raise ValueError("drift_band_forced_days must be a whole number when provided")
            if int(self.drift_band_forced_days) < 1:
                raise ValueError("drift_band_forced_days must be at least 1 when provided")
            if self.drift_band_name is None and self.drift_band_portfolio is None:
                raise ValueError("drift_band_forced_days requires drift_band_name or drift_band_portfolio")
        if self.turnover_shrinkage is not None and not (
            math.isfinite(self.turnover_shrinkage) and 0 <= self.turnover_shrinkage < 1
        ):
            raise ValueError("turnover_shrinkage must be in [0, 1) when provided")
        if self.turnover_max_step is not None and not (
            math.isfinite(self.turnover_max_step) and 0 < self.turnover_max_step <= 1
        ):
            raise ValueError("turnover_max_step must be in (0, 1] when provided")
        if self.min_holding_days is not None:
            if self.min_holding_days != int(self.min_holding_days):
                raise ValueError("min_holding_days must be a whole number when provided")
            if int(self.min_holding_days) < 1:
                raise ValueError("min_holding_days must be at least 1 when provided")
        self._validate_order_layer()

    def _validate_order_layer(self) -> None:
        """Validate the 8.6-8.9 / 9.1 / 9.2 group and resolve its companions.

        Each group is enabled by exactly one key -- the group's presence in the
        canonical payload is that key -- and the companions an enabled group
        needs are filled in here, so the payload carries the values the
        executors actually read and the worker mirror only has to re-run the
        same checks.  A companion offered without its enabler is a contradiction
        rather than a tightening: nothing would read it, and silently ignoring a
        declared risk knob is the one failure mode worth being loud about.
        """

        # 8.6 ParticipationCap.  Both it and 8.9 allocate a multi-day plan and
        # the engine runs one allocator, so asking for both is a contradiction
        # (the shape of drift_band_forced_days above).
        if self.participation_max_rate is not None and self.staged_max_days is not None:
            raise ValueError("participation_cap and staged_execution are mutually exclusive")
        if self.participation_max_rate is not None:
            object.__setattr__(
                self,
                "participation_max_rate",
                _bounded_backtest_number(
                    self.participation_max_rate, "participation_max_rate must be in (0, 1] when provided", lower=0.0, upper=1.0
                ),
            )
        if self.participation_max_days is not None:
            object.__setattr__(
                self,
                "participation_max_days",
                _whole_backtest_number(self.participation_max_days, "participation_max_days", minimum=1),
            )
        if self.participation_amount_window is not None:
            object.__setattr__(
                self,
                "participation_amount_window",
                _whole_backtest_number(self.participation_amount_window, "participation_amount_window", minimum=1),
            )
        if self.participation_unfilled is not None and self.participation_unfilled not in {"cancel", "carry"}:
            raise ValueError("participation_unfilled must be one of ('cancel', 'carry') when provided")
        if self.participation_max_rate is None:
            for name, value in (
                ("participation_max_days", self.participation_max_days),
                ("participation_unfilled", self.participation_unfilled),
                ("participation_amount_window", self.participation_amount_window),
            ):
                if value is not None:
                    raise ValueError(f"{name} requires participation_max_rate")
        else:
            # Conservative default per the design: an unfilled slice is carried
            # (never quietly filled) and the participation amount is the
            # execution day's own traded amount.
            if self.participation_unfilled is None:
                object.__setattr__(self, "participation_unfilled", "carry")
            if self.participation_amount_window is None:
                object.__setattr__(self, "participation_amount_window", 1)
            if self.participation_unfilled == "carry" and self.participation_max_days is None:
                raise ValueError("participation_max_days must be at least 1 when participation_unfilled is carry")

        # 8.7 LotRounding.
        if not isinstance(self.lot_rounding, bool):
            raise ValueError("lot_rounding must be a boolean")
        if self.lot_min_notional is not None:
            object.__setattr__(
                self,
                "lot_min_notional",
                _bounded_backtest_number(self.lot_min_notional, "lot_min_notional must be positive when provided", lower=0.0),
            )
        if self.lot_rounding_mode is not None and self.lot_rounding_mode not in {"down", "nearest"}:
            raise ValueError("lot_rounding_mode must be one of ('down', 'nearest') when provided")
        if not self.lot_rounding:
            for name, value in (("lot_min_notional", self.lot_min_notional), ("lot_rounding_mode", self.lot_rounding_mode)):
                if value is not None:
                    raise ValueError(f"{name} requires lot_rounding")
        elif self.lot_rounding_mode is None:
            # ``down`` keeps ``rounded <= target`` an invariant, so rounding can
            # never overspend the budget the composition chain just set.
            object.__setattr__(self, "lot_rounding_mode", "down")

        # 8.8 SideAwareTradability.
        if self.side_aware_tradability is not None and self.side_aware_tradability not in {
            "block_buy",
            "block_sell",
            "block_both",
        }:
            raise ValueError("side_aware_tradability must be one of ('block_buy', 'block_sell', 'block_both') when provided")

        # 8.9 StagedExecution.
        if self.staged_max_days is not None:
            object.__setattr__(
                self, "staged_max_days", _whole_backtest_number(self.staged_max_days, "staged_max_days", minimum=1)
            )
        if self.staged_priority is not None and self.staged_priority not in {"score", "weight"}:
            raise ValueError("staged_priority must be one of ('score', 'weight') when provided")
        if self.staged_max_days is None:
            if self.staged_priority is not None:
                raise ValueError("staged_priority requires staged_max_days")
        elif self.staged_priority is None:
            # ``weight`` needs no score panel, so it is the default; a caller
            # that wants score-ordered service has to ask for it and supply the
            # scores the order reads.
            object.__setattr__(self, "staged_priority", "weight")

        # 9.1 PortfolioVolatilityTarget.
        if self.vol_target_annual is not None:
            object.__setattr__(
                self,
                "vol_target_annual",
                _bounded_backtest_number(self.vol_target_annual, "vol_target_annual must be positive when provided", lower=0.0),
            )
        if self.vol_target_window is not None:
            object.__setattr__(
                self, "vol_target_window", _whole_backtest_number(self.vol_target_window, "vol_target_window", minimum=2)
            )
        if self.vol_target_min_scale is not None:
            object.__setattr__(
                self,
                "vol_target_min_scale",
                _bounded_backtest_number(
                    self.vol_target_min_scale, "vol_target_min_scale must be in (0, 1] when provided", lower=0.0, upper=1.0
                ),
            )
        if self.vol_target_max_scale is not None:
            object.__setattr__(
                self,
                "vol_target_max_scale",
                _bounded_backtest_number(
                    self.vol_target_max_scale, "vol_target_max_scale must be at least 1 when provided", lower=1.0, lower_open=False
                ),
            )
        vol_target_companions = ("vol_target_window", "vol_target_min_scale", "vol_target_max_scale")
        if self.vol_target_annual is None:
            for name in vol_target_companions:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} requires vol_target_annual")
        else:
            for name in vol_target_companions:
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required when vol_target_annual is provided")
            # Both bounds are mandatory (they are what stops a single day from
            # flipping the book between fully invested and fully in cash) and
            # they must actually bracket the no-op scale of 1.0.
            if not self.vol_target_min_scale < 1 <= self.vol_target_max_scale:
                raise ValueError("vol_target_min_scale must be below 1 and vol_target_max_scale at least 1")

        # 9.2 DrawdownStateMachine.
        if self.drawdown_thresholds is not None:
            object.__setattr__(
                self,
                "drawdown_thresholds",
                _drawdown_map(
                    self.drawdown_thresholds,
                    "drawdown_thresholds",
                    message="drawdown_thresholds must be in (0, 1] when provided",
                    upper=1.0,
                    positive=True,
                ),
            )
            ordered = [self.drawdown_thresholds[state] for state in DRAWDOWN_STATES]
            if not ordered[0] < ordered[1] < ordered[2]:
                raise ValueError("drawdown_thresholds must be strictly increasing")
        if self.drawdown_scales is not None:
            object.__setattr__(
                self,
                "drawdown_scales",
                _drawdown_map(
                    self.drawdown_scales,
                    "drawdown_scales",
                    message="drawdown_scales must be in [0, 1] when provided",
                    upper=1.0,
                ),
            )
        if self.drawdown_confirm_days is not None:
            object.__setattr__(
                self,
                "drawdown_confirm_days",
                _whole_backtest_number(self.drawdown_confirm_days, "drawdown_confirm_days", minimum=1),
            )
        if self.drawdown_recovery is not None:
            object.__setattr__(
                self,
                "drawdown_recovery",
                _bounded_backtest_number(
                    self.drawdown_recovery, "drawdown_recovery must be in (0, 1] when provided", lower=0.0, upper=1.0
                ),
            )
        drawdown_companions = ("drawdown_confirm_days", "drawdown_recovery", "drawdown_scales")
        if self.drawdown_thresholds is None:
            for name in drawdown_companions:
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} requires drawdown_thresholds")
        else:
            for name in drawdown_companions:
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required when drawdown_thresholds is provided")

    def as_payload(self) -> dict[str, Any]:
        """Canonical JSON-serialisable form; also feeds ``strategy_id_for``.

        The execution-adjustment fields travel only while enabled, so every
        pre-existing candidate keeps its canonical payload (and therefore its
        strategy identity hash) unchanged.
        """

        payload = {
            "signal_lag": self.signal_lag,
            "trading_days_per_year": self.trading_days_per_year,
        }
        if self.drift_band_name is not None:
            payload["drift_band_name"] = self.drift_band_name
        if self.drift_band_portfolio is not None:
            payload["drift_band_portfolio"] = self.drift_band_portfolio
        if self.drift_band_forced_days is not None:
            payload["drift_band_forced_days"] = self.drift_band_forced_days
        if self.turnover_shrinkage is not None:
            payload["turnover_shrinkage"] = self.turnover_shrinkage
        if self.turnover_max_step is not None:
            payload["turnover_max_step"] = self.turnover_max_step
        if self.min_holding_days is not None:
            payload["min_holding_days"] = self.min_holding_days
        for key in (
            "participation_max_rate",
            "participation_max_days",
            "participation_unfilled",
            "participation_amount_window",
            "lot_min_notional",
            "lot_rounding_mode",
            "side_aware_tradability",
            "staged_max_days",
            "staged_priority",
            "vol_target_annual",
            "vol_target_window",
            "vol_target_min_scale",
            "vol_target_max_scale",
            "drawdown_thresholds",
            "drawdown_confirm_days",
            "drawdown_recovery",
            "drawdown_scales",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = dict(value) if isinstance(value, Mapping) else value
        if self.lot_rounding:
            # A boolean enabler rather than a nullable one: ``False`` is the
            # disabled state, so only the enabled value travels.
            payload["lot_rounding"] = True
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BacktestSpec":
        return cls(
            signal_lag=int(payload.get("signal_lag", 1)),
            trading_days_per_year=int(payload.get("trading_days_per_year", 252)),
            drift_band_name=None if payload.get("drift_band_name") is None else float(payload["drift_band_name"]),
            drift_band_portfolio=(
                None if payload.get("drift_band_portfolio") is None else float(payload["drift_band_portfolio"])
            ),
            drift_band_forced_days=(
                None if payload.get("drift_band_forced_days") is None else int(payload["drift_band_forced_days"])
            ),
            turnover_shrinkage=(
                None if payload.get("turnover_shrinkage") is None else float(payload["turnover_shrinkage"])
            ),
            turnover_max_step=None if payload.get("turnover_max_step") is None else float(payload["turnover_max_step"]),
            min_holding_days=None if payload.get("min_holding_days") is None else int(payload["min_holding_days"]),
            # Batch 6 raw passthrough (fact 37): __post_init__ validates and
            # writes the normalised value back, so nothing here may coerce a
            # malformed payload into something the validators would accept.
            participation_max_rate=payload.get("participation_max_rate"),
            participation_max_days=payload.get("participation_max_days"),
            participation_unfilled=payload.get("participation_unfilled"),
            participation_amount_window=payload.get("participation_amount_window"),
            lot_rounding=payload.get("lot_rounding", False),
            lot_min_notional=payload.get("lot_min_notional"),
            lot_rounding_mode=payload.get("lot_rounding_mode"),
            side_aware_tradability=payload.get("side_aware_tradability"),
            staged_max_days=payload.get("staged_max_days"),
            staged_priority=payload.get("staged_priority"),
            vol_target_annual=payload.get("vol_target_annual"),
            vol_target_window=payload.get("vol_target_window"),
            vol_target_min_scale=payload.get("vol_target_min_scale"),
            vol_target_max_scale=payload.get("vol_target_max_scale"),
            drawdown_thresholds=payload.get("drawdown_thresholds"),
            drawdown_confirm_days=payload.get("drawdown_confirm_days"),
            drawdown_recovery=payload.get("drawdown_recovery"),
            drawdown_scales=payload.get("drawdown_scales"),
        )


# ---------------------------------------------------------------------------
# Roadmap section 3: candidate-pool / tradability universe filters.
#
# Every filter consumes data panels the upper layer injects explicitly (the
# industry_map pattern): neither engine carries membership, status, limit or
# amount data of its own.  The filters share one ``universe_rules`` payload
# section; each filter section travels only while enabled, so the canonical
# payload -- and therefore the strategy identity hash -- of every pre-batch
# candidate is unchanged.  None of it enters the LLM prompt vocabulary: the
# panels are configuration inputs, not agent-selectable parameters.
# ---------------------------------------------------------------------------

UNIVERSE_TOP_LEVEL_KEYS = frozenset(
    {
        "trading_dates",
        "price",
        "base_universe",
        "index_membership",
        "listing_age",
        "status_exclusion",
        "price_limit",
        "liquidity",
        "price_range",
        "data_quality",
    }
)

_UNIVERSE_SECTION_KEYS: dict[str, frozenset[str]] = {
    "base_universe": frozenset({"symbols"}),
    "index_membership": frozenset({"membership", "lag_days", "exclude_recent_days"}),
    "listing_age": frozenset({"listing_dates", "min_days"}),
    "status_exclusion": frozenset({"statuses", "excluded_statuses"}),
    "price_limit": frozenset({"limit_up", "limit_down", "exclude_limit_up", "exclude_limit_down"}),
    "liquidity": frozenset({"amount", "window", "min_amount", "quantile", "missing"}),
    "price_range": frozenset({"min_price", "max_price", "consecutive_days"}),
    "data_quality": frozenset({"min_coverage", "max_missing_run", "window", "non_positive_price"}),
}


def _parse_universe_date(value: Any, message: str) -> "date":
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(message)
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(message) from None


def _whole_universe_number(value: Any, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(message)
    if float(value) != int(value):
        raise ValueError(message)
    return int(value)


def _finite_universe_number(value: Any, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(message)
    return float(value)


def _validate_exposure_map(value: Any, message: str) -> None:
    """Shape check for a ``{name: {symbol: number}}`` injected exposure mapping.

    The neutralization/regime data panels are explicit configuration inputs
    (roadmap 4.6/4.7); malformed shapes fail closed with the same message the
    worker raises instead of crashing deeper in the pipeline.
    """

    if not isinstance(value, dict) or not value:
        raise ValueError(message)
    for name, row in value.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict) or not row:
            raise ValueError(message)
        for symbol, number in row.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(message)
            _finite_universe_number(number, message)


def _validate_bounds_map(value: Any, message: str) -> None:
    """Shape check for a ``{name: (lower, upper)`` interval map (roadmap
    6.5/6.7): non-empty dict, non-empty string keys, finite lower <= upper.
    """

    if not isinstance(value, dict) or not value:
        raise ValueError(message)
    for name, bounds in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(message)
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(message)
        lower, upper = bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) > float(upper)
        ):
            raise ValueError(message)


def _validate_universe_number_panel(value: Any, prefix: str, *, allow_empty: bool = False) -> None:
    """Shape check for a sparse {date: {symbol: number}} payload panel."""

    if not isinstance(value, dict) or (not value and not allow_empty):
        raise ValueError(f"{prefix} must map date strings to objects of symbol -> number")
    for date_key, row in value.items():
        _parse_universe_date(date_key, f"{prefix} contains an invalid date: {date_key!r}")
        if not isinstance(row, dict):
            raise ValueError(f"{prefix} must map date strings to objects of symbol -> number")
        for symbol, number in row.items():
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(f"{prefix} must map date strings to objects of symbol -> number")
            _finite_universe_number(number, f"{prefix} must map date strings to objects of symbol -> number")


def _validate_universe_true_panel(value: Any, prefix: str) -> None:
    """Shape check for a sparse {date: {symbol: True}} membership panel."""

    if not isinstance(value, dict) or not value:
        raise ValueError(f"{prefix} must map date strings to objects of symbol -> true")
    for date_key, row in value.items():
        _parse_universe_date(date_key, f"{prefix} contains an invalid date: {date_key!r}")
        if not isinstance(row, dict) or not row:
            raise ValueError(f"{prefix} must map date strings to objects of symbol -> true")
        for symbol, flag in row.items():
            if not isinstance(symbol, str) or not symbol or flag is not True:
                raise ValueError(f"{prefix} must map date strings to objects of symbol -> true")


def _validate_universe_string_panel(value: Any, prefix: str) -> None:
    """Shape check for a sparse {date: {symbol: status-string}} panel."""

    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must map date strings to objects of symbol -> non-empty string")
    for date_key, row in value.items():
        _parse_universe_date(date_key, f"{prefix} contains an invalid date: {date_key!r}")
        if not isinstance(row, dict):
            raise ValueError(f"{prefix} must map date strings to objects of symbol -> non-empty string")
        for symbol, label in row.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(label, str) or not label:
                raise ValueError(f"{prefix} must map date strings to objects of symbol -> non-empty string")


def _validate_universe_symbol_dates(value: Any, prefix: str) -> None:
    """Shape check for a {symbol: date-string} mapping (listing dates)."""

    if not isinstance(value, dict) or not value:
        raise ValueError(f"{prefix} must map non-empty symbols to date strings")
    for symbol, day in value.items():
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"{prefix} must map non-empty symbols to date strings")
        _parse_universe_date(day, f"{prefix} contains an invalid date for {symbol!r}")


@dataclass(frozen=True)
class BaseUniverseRule:
    """Roadmap 3.1 BaseUniverse: the static initial member list.

    The member list is an explicit configuration input (the engine has no
    securities master table); point-in-time membership is the business of the
    IndexMembershipFilter.  A symbol absent from the list never enters the
    candidate pool on any date.
    """

    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.symbols, str) or not isinstance(self.symbols, (list, tuple)) or not self.symbols:
            raise ValueError("universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.symbols):
            raise ValueError("universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings")
        object.__setattr__(self, "symbols", tuple(self.symbols))

    def as_payload(self) -> dict[str, Any]:
        return {"symbols": list(self.symbols)}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BaseUniverseRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["base_universe"]
        if unknown:
            raise ValueError(f"universe_rules.base_universe has unknown keys: {sorted(unknown)[:3]}")
        if "symbols" not in payload:
            raise ValueError("universe_rules.base_universe is missing the symbols key")
        # No list() coercion: a string would be character-split silently and
        # slip past the __post_init__ container check -- the raw value is
        # validated as-is (str rejected) exactly like the worker.
        return cls(symbols=payload["symbols"])


@dataclass(frozen=True)
class IndexMembershipRule:
    """Roadmap 3.2 IndexMembershipFilter: point-in-time index membership.

    ``membership`` is a sparse {date: {symbol: True}} panel whose effective
    dates are already resolved by the data layer; a decision date receives the
    membership of the last panel row at or before it (as-of semantics), rows
    before the panel start count as not-a-member.  ``lag_days`` shifts the
    panel later by that many panel rows (an effective-lag conservatism);
    ``exclude_recent_days`` bars a name for that many panel rows after a
    False->True inclusion transition (a name that is a member at the panel
    start is pre-existing and never "new").  The index codes themselves stay
    with the data layer -- the panel is the union the upper layer chose.
    """

    membership: dict[str, dict[str, bool]] = field(default_factory=dict)
    lag_days: int = 0
    exclude_recent_days: int = 0

    def __post_init__(self) -> None:
        _validate_universe_true_panel(self.membership, "universe_rules.index_membership.membership")
        object.__setattr__(
            self,
            "membership",
            {str(day): dict(row) for day, row in self.membership.items()},
        )
        lag = _whole_universe_number(self.lag_days, "universe_rules.index_membership.lag_days must be a whole number >= 0")
        if lag < 0:
            raise ValueError("universe_rules.index_membership.lag_days must be a whole number >= 0")
        # Normalisation is written back so the spec never carries a 1.0 that
        # the executor would trip over (mirror of the worker's int() call).
        object.__setattr__(self, "lag_days", lag)
        recent = _whole_universe_number(
            self.exclude_recent_days,
            "universe_rules.index_membership.exclude_recent_days must be a whole number >= 0",
        )
        if recent < 0:
            raise ValueError("universe_rules.index_membership.exclude_recent_days must be a whole number >= 0")
        object.__setattr__(self, "exclude_recent_days", recent)

    def as_payload(self) -> dict[str, Any]:
        return {
            "membership": {day: dict(row) for day, row in self.membership.items()},
            "lag_days": self.lag_days,
            "exclude_recent_days": self.exclude_recent_days,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IndexMembershipRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["index_membership"]
        if unknown:
            raise ValueError(f"universe_rules.index_membership has unknown keys: {sorted(unknown)[:3]}")
        if "membership" not in payload:
            raise ValueError("universe_rules.index_membership is missing the membership key")
        return cls(
            membership=dict(payload["membership"]),
            lag_days=payload.get("lag_days", 0),
            exclude_recent_days=payload.get("exclude_recent_days", 0),
        )


@dataclass(frozen=True)
class ListingAgeRule:
    """Roadmap 3.3 ListingAgeFilter: ``min_listing_days`` trading days on book.

    A symbol is eligible on date ``t`` when the number of trading-calendar
    days ``d`` with ``listing_date < d <= t`` is at least ``min_listing_days``
    (the listing day itself is day 0, so the boundary date is exactly the
    ``min_listing_days``-th trading day after listing).  A symbol without a
    listing date can never be age-verified and is excluded (conservative).
    """

    listing_dates: dict[str, str] = field(default_factory=dict)
    min_days: int = 1

    def __post_init__(self) -> None:
        _validate_universe_symbol_dates(self.listing_dates, "universe_rules.listing_age.listing_dates")
        object.__setattr__(self, "listing_dates", {str(symbol): str(day) for symbol, day in self.listing_dates.items()})
        min_days = _whole_universe_number(self.min_days, "universe_rules.listing_age.min_days must be a whole number >= 1")
        if min_days < 1:
            raise ValueError("universe_rules.listing_age.min_days must be a whole number >= 1")
        object.__setattr__(self, "min_days", min_days)

    def as_payload(self) -> dict[str, Any]:
        return {"listing_dates": dict(self.listing_dates), "min_days": self.min_days}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ListingAgeRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["listing_age"]
        if unknown:
            raise ValueError(f"universe_rules.listing_age has unknown keys: {sorted(unknown)[:3]}")
        if "listing_dates" not in payload:
            raise ValueError("universe_rules.listing_age is missing the listing_dates key")
        if "min_days" not in payload:
            raise ValueError("universe_rules.listing_age is missing the min_days key")
        return cls(listing_dates=dict(payload["listing_dates"]), min_days=payload["min_days"])


@dataclass(frozen=True)
class StatusExclusionRule:
    """Roadmap 3.4 StatusExclusionFilter: exclude listed abnormal statuses.

    ``statuses`` is a sparse {date: {symbol: status}} panel that records only
    abnormal states, so an absent entry means "normal on that date" and the
    lookup is exact-date by construction (no as-of backfill of a stale status
    -- the survivorship risk the roadmap warns about).  A symbol is excluded
    on a date when its recorded status is one of ``excluded_statuses``.
    """

    statuses: dict[str, dict[str, str]] = field(default_factory=dict)
    excluded_statuses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_universe_string_panel(self.statuses, "universe_rules.status_exclusion.statuses")
        object.__setattr__(
            self,
            "statuses",
            {str(day): dict(row) for day, row in self.statuses.items()},
        )
        if isinstance(self.excluded_statuses, str) or not isinstance(self.excluded_statuses, (list, tuple)):
            raise ValueError(
                "universe_rules.status_exclusion.excluded_statuses must be a non-empty list of distinct non-empty strings"
            )
        if not self.excluded_statuses or any(not isinstance(name, str) or not name for name in self.excluded_statuses):
            raise ValueError(
                "universe_rules.status_exclusion.excluded_statuses must be a non-empty list of distinct non-empty strings"
            )
        if len(set(self.excluded_statuses)) != len(self.excluded_statuses):
            raise ValueError(
                "universe_rules.status_exclusion.excluded_statuses must be a non-empty list of distinct non-empty strings"
            )
        object.__setattr__(self, "excluded_statuses", tuple(self.excluded_statuses))

    def as_payload(self) -> dict[str, Any]:
        return {
            "statuses": {day: dict(row) for day, row in self.statuses.items()},
            "excluded_statuses": list(self.excluded_statuses),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StatusExclusionRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["status_exclusion"]
        if unknown:
            raise ValueError(f"universe_rules.status_exclusion has unknown keys: {sorted(unknown)[:3]}")
        if "statuses" not in payload:
            raise ValueError("universe_rules.status_exclusion is missing the statuses key")
        if "excluded_statuses" not in payload:
            raise ValueError("universe_rules.status_exclusion is missing the excluded_statuses key")
        # Same no-coercion rule as base_universe.symbols: the raw value is
        # validated as-is so a string is rejected, never character-split.
        return cls(statuses=dict(payload["statuses"]), excluded_statuses=payload["excluded_statuses"])


@dataclass(frozen=True)
class PriceLimitRule:
    """Roadmap 3.5 PriceLimitTradabilityFilter (conservative single mask).

    A name at its limit price on a decision date cannot participate in that
    date's selection: the limit is breached when the decision-date price is
    within 1e-9 of the limit (touching counts).  Missing price or limit data
    for an excluded side means tradability cannot be established and excludes
    the name (conservative).  The roadmap's buy/sell asymmetry ("hold but do
    not buy") needs a two-sided execution mask the engine's weight builders
    do not have and is not fabricated here; both excluded sides share the
    top-level raw-close ``price`` panel.
    """

    limit_up: dict[str, dict[str, float]] | None = None
    limit_down: dict[str, dict[str, float]] | None = None
    exclude_limit_up: bool = True
    exclude_limit_down: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.exclude_limit_up, bool) or not isinstance(self.exclude_limit_down, bool):
            raise ValueError("universe_rules.price_limit.exclude_limit_up and exclude_limit_down must be booleans")
        if not self.exclude_limit_up and not self.exclude_limit_down:
            raise ValueError("universe_rules.price_limit must exclude at least one of limit_up and limit_down")
        if self.exclude_limit_up and self.limit_up is None:
            raise ValueError("universe_rules.price_limit.limit_up is required when exclude_limit_up is true")
        if not self.exclude_limit_up and self.limit_up is not None:
            raise ValueError("universe_rules.price_limit.limit_up is only valid when exclude_limit_up is true")
        if self.exclude_limit_down and self.limit_down is None:
            raise ValueError("universe_rules.price_limit.limit_down is required when exclude_limit_down is true")
        if not self.exclude_limit_down and self.limit_down is not None:
            raise ValueError("universe_rules.price_limit.limit_down is only valid when exclude_limit_down is true")
        for panel, prefix in ((self.limit_up, "universe_rules.price_limit.limit_up"), (self.limit_down, "universe_rules.price_limit.limit_down")):
            if panel is not None:
                _validate_universe_number_panel(panel, prefix)
        object.__setattr__(
            self,
            "limit_up",
            None if self.limit_up is None else {str(day): dict(row) for day, row in self.limit_up.items()},
        )
        object.__setattr__(
            self,
            "limit_down",
            None if self.limit_down is None else {str(day): dict(row) for day, row in self.limit_down.items()},
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "limit_up": None if self.limit_up is None else {day: dict(row) for day, row in self.limit_up.items()},
            "limit_down": None if self.limit_down is None else {day: dict(row) for day, row in self.limit_down.items()},
            "exclude_limit_up": self.exclude_limit_up,
            "exclude_limit_down": self.exclude_limit_down,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PriceLimitRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["price_limit"]
        if unknown:
            raise ValueError(f"universe_rules.price_limit has unknown keys: {sorted(unknown)[:3]}")
        return cls(
            limit_up=payload.get("limit_up"),
            limit_down=payload.get("limit_down"),
            exclude_limit_up=payload.get("exclude_limit_up", True),
            exclude_limit_down=payload.get("exclude_limit_down", True),
        )


@dataclass(frozen=True)
class LiquidityRule:
    """Roadmap 3.6 LiquidityFilter: trailing mean-turnover floor.

    The trailing mean of the injected daily amount over the last ``window``
    trading-calendar days ending at the decision date (short history is used
    as is) must clear ``min_amount`` and/or the date's cross-sectional
    ``quantile`` of the same rolling-mean series; both comparisons are
    inclusive.  A name whose rolling mean is missing over the whole window is
    excluded unless ``missing="include"``.  The quantile leg applies only on
    dates where at least one valid rolling mean exists.
    """

    amount: dict[str, dict[str, float]] = field(default_factory=dict)
    window: int = 20
    min_amount: float | None = None
    quantile: float | None = None
    missing: str = "exclude"

    def __post_init__(self) -> None:
        _validate_universe_number_panel(self.amount, "universe_rules.liquidity.amount")
        object.__setattr__(self, "amount", {str(day): dict(row) for day, row in self.amount.items()})
        window = _whole_universe_number(self.window, "universe_rules.liquidity.window must be a whole number >= 1")
        if window < 1:
            raise ValueError("universe_rules.liquidity.window must be a whole number >= 1")
        object.__setattr__(self, "window", window)
        if self.min_amount is not None:
            if _finite_universe_number(self.min_amount, "universe_rules.liquidity.min_amount must be a finite number >= 0 when provided") < 0:
                raise ValueError("universe_rules.liquidity.min_amount must be a finite number >= 0 when provided")
        if self.quantile is not None and not (
            isinstance(self.quantile, (int, float))
            and not isinstance(self.quantile, bool)
            and math.isfinite(float(self.quantile))
            and 0 <= float(self.quantile) <= 1
        ):
            raise ValueError("universe_rules.liquidity.quantile must be in [0, 1] when provided")
        if self.missing not in ("exclude", "include"):
            raise ValueError(f"universe_rules.liquidity.missing must be one of ('exclude', 'include'), got {self.missing!r}")
        if self.min_amount is None and self.quantile is None:
            raise ValueError("universe_rules.liquidity requires min_amount or quantile")

    def as_payload(self) -> dict[str, Any]:
        return {
            "amount": {day: dict(row) for day, row in self.amount.items()},
            "window": self.window,
            "min_amount": self.min_amount,
            "quantile": self.quantile,
            "missing": self.missing,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LiquidityRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["liquidity"]
        if unknown:
            raise ValueError(f"universe_rules.liquidity has unknown keys: {sorted(unknown)[:3]}")
        if "amount" not in payload:
            raise ValueError("universe_rules.liquidity is missing the amount key")
        return cls(
            amount=dict(payload["amount"]),
            window=payload.get("window", 20),
            min_amount=payload.get("min_amount"),
            quantile=payload.get("quantile"),
            # Raw, never str()-coerced: a non-string would otherwise change
            # repr inside the error message and diverge from the worker.
            missing=payload.get("missing", "exclude"),
        )


@dataclass(frozen=True)
class PriceRangeRule:
    """Roadmap 3.7 PriceRangeFilter: price inside ``[min_price, max_price]``.

    Both bounds are inclusive and either may be omitted; a name must sit
    inside the band on each of the last ``consecutive_days`` trading-calendar
    days ending at the decision date.  A missing price inside the window, or
    a window extending before the trading calendar starts, cannot confirm the
    band and excludes the name (conservative).  The band reads the injected
    raw-close ``price`` panel, never the engine's adjusted close.
    """

    min_price: float | None = None
    max_price: float | None = None
    consecutive_days: int = 1

    def __post_init__(self) -> None:
        if self.min_price is not None:
            if _finite_universe_number(self.min_price, "universe_rules.price_range.min_price must be a finite positive number when provided") <= 0:
                raise ValueError("universe_rules.price_range.min_price must be a finite positive number when provided")
        if self.max_price is not None:
            if _finite_universe_number(self.max_price, "universe_rules.price_range.max_price must be a finite positive number when provided") <= 0:
                raise ValueError("universe_rules.price_range.max_price must be a finite positive number when provided")
        if self.min_price is None and self.max_price is None:
            raise ValueError("universe_rules.price_range requires min_price or max_price")
        if self.min_price is not None and self.max_price is not None and float(self.min_price) > float(self.max_price):
            raise ValueError("universe_rules.price_range.min_price must not exceed max_price")
        consecutive = _whole_universe_number(
            self.consecutive_days, "universe_rules.price_range.consecutive_days must be a whole number >= 1"
        )
        if consecutive < 1:
            raise ValueError("universe_rules.price_range.consecutive_days must be a whole number >= 1")
        object.__setattr__(self, "consecutive_days", consecutive)

    def as_payload(self) -> dict[str, Any]:
        return {
            "min_price": self.min_price,
            "max_price": self.max_price,
            "consecutive_days": self.consecutive_days,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PriceRangeRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["price_range"]
        if unknown:
            raise ValueError(f"universe_rules.price_range has unknown keys: {sorted(unknown)[:3]}")
        return cls(
            min_price=payload.get("min_price"),
            max_price=payload.get("max_price"),
            consecutive_days=payload.get("consecutive_days", 1),
        )


@dataclass(frozen=True)
class DataQualityRule:
    """Roadmap 3.8 DataQualityFilter (conservative subset).

    Runs on the candidate's own factor panels plus the optional raw-close
    ``price`` panel.  ``min_coverage`` requires the share of non-missing
    factor cells over the trailing ``window`` rows of the factors' combined
    date grid to reach the bound (inclusive); ``max_missing_run`` excludes a
    name whose current trailing missing run in any factor is longer than the
    bound (a run of exactly the bound still passes);
    ``non_positive_price`` excludes names whose decision-date raw price is
    non-positive or missing (which requires the price panel).  The roadmap's
    generic outlier rules have no engine-independent definition and are
    deliberately absent.
    """

    min_coverage: float | None = None
    max_missing_run: int | None = None
    window: int = 20
    non_positive_price: bool = False

    def __post_init__(self) -> None:
        if self.min_coverage is not None and not (
            isinstance(self.min_coverage, (int, float))
            and not isinstance(self.min_coverage, bool)
            and math.isfinite(float(self.min_coverage))
            and 0 <= float(self.min_coverage) <= 1
        ):
            raise ValueError("universe_rules.data_quality.min_coverage must be in [0, 1] when provided")
        if self.max_missing_run is not None:
            missing_run = _whole_universe_number(
                self.max_missing_run,
                "universe_rules.data_quality.max_missing_run must be a whole number >= 1 when provided",
            )
            if missing_run < 1:
                raise ValueError("universe_rules.data_quality.max_missing_run must be a whole number >= 1 when provided")
            object.__setattr__(self, "max_missing_run", missing_run)
        quality_window = _whole_universe_number(self.window, "universe_rules.data_quality.window must be a whole number >= 1")
        if quality_window < 1:
            raise ValueError("universe_rules.data_quality.window must be a whole number >= 1")
        object.__setattr__(self, "window", quality_window)
        if not isinstance(self.non_positive_price, bool):
            raise ValueError("universe_rules.data_quality.non_positive_price must be a boolean")
        if self.min_coverage is None and self.max_missing_run is None and not self.non_positive_price:
            raise ValueError("universe_rules.data_quality requires min_coverage, max_missing_run or non_positive_price")

    def as_payload(self) -> dict[str, Any]:
        return {
            "min_coverage": self.min_coverage,
            "max_missing_run": self.max_missing_run,
            "window": self.window,
            "non_positive_price": self.non_positive_price,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DataQualityRule":
        unknown = set(payload) - _UNIVERSE_SECTION_KEYS["data_quality"]
        if unknown:
            raise ValueError(f"universe_rules.data_quality has unknown keys: {sorted(unknown)[:3]}")
        return cls(
            min_coverage=payload.get("min_coverage"),
            max_missing_run=payload.get("max_missing_run"),
            window=payload.get("window", 20),
            # Raw, never bool()-coerced: bool() would turn any truthy value
            # (e.g. "yes") into True and silently bypass the isinstance check.
            non_positive_price=payload.get("non_positive_price", False),
        )


@dataclass(frozen=True)
class UniverseSpec:
    """Candidate-pool filter chain (roadmap sections 3.1-3.8).

    One configuration object; the eight filters apply in roadmap order and
    intersect into a single ``eligible[t, symbol]`` boolean panel.  Shared
    inputs live at the top level: ``trading_dates`` (the calendar every
    history-dependent filter -- listing age, liquidity, price range -- counts
    on) and ``price`` (the raw close panel the price-limit, price-range and
    non-positive-price checks read).  ``as_payload`` emits ``None`` while no
    filter is enabled, so the canonical payload of every pre-batch candidate
    is unchanged.
    """

    trading_dates: tuple[str, ...] | None = None
    price: dict[str, dict[str, float]] | None = None
    base_universe: BaseUniverseRule | None = None
    index_membership: IndexMembershipRule | None = None
    listing_age: ListingAgeRule | None = None
    status_exclusion: StatusExclusionRule | None = None
    price_limit: PriceLimitRule | None = None
    liquidity: LiquidityRule | None = None
    price_range: PriceRangeRule | None = None
    data_quality: DataQualityRule | None = None

    def __post_init__(self) -> None:
        if self.trading_dates is not None:
            if isinstance(self.trading_dates, str) or not isinstance(self.trading_dates, (list, tuple)) or not self.trading_dates:
                raise ValueError("universe_rules.trading_dates must be a non-empty list of date strings")
            parsed = [
                _parse_universe_date(day, "universe_rules.trading_dates must be a non-empty list of date strings")
                for day in self.trading_dates
            ]
            if any(b <= a for a, b in zip(parsed, parsed[1:])):
                raise ValueError("universe_rules.trading_dates must be strictly increasing")
            object.__setattr__(self, "trading_dates", tuple(str(day) for day in self.trading_dates))
        if self.price is not None:
            _validate_universe_number_panel(self.price, "universe_rules.price")
            object.__setattr__(self, "price", {str(day): dict(row) for day, row in self.price.items()})
        if self.listing_age is not None or self.liquidity is not None or self.price_range is not None:
            if self.trading_dates is None:
                raise ValueError(
                    "universe_rules.trading_dates is required when listing_age, liquidity or price_range is enabled"
                )
        if self.price_limit is not None and self.price is None:
            raise ValueError("universe_rules.price is required when price_limit is enabled")
        if self.price_range is not None and self.price is None:
            raise ValueError("universe_rules.price is required when price_range is enabled")
        if self.data_quality is not None and self.data_quality.non_positive_price and self.price is None:
            raise ValueError("universe_rules.price is required when data_quality requests the non-positive-price check")

    @property
    def enabled(self) -> bool:
        return any(
            section is not None
            for section in (
                self.base_universe,
                self.index_membership,
                self.listing_age,
                self.status_exclusion,
                self.price_limit,
                self.liquidity,
                self.price_range,
                self.data_quality,
            )
        )

    def as_payload(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        payload: dict[str, Any] = {}
        if self.trading_dates is not None:
            payload["trading_dates"] = list(self.trading_dates)
        if self.price is not None:
            payload["price"] = {day: dict(row) for day, row in self.price.items()}
        if self.base_universe is not None:
            payload["base_universe"] = self.base_universe.as_payload()
        if self.index_membership is not None:
            payload["index_membership"] = self.index_membership.as_payload()
        if self.listing_age is not None:
            payload["listing_age"] = self.listing_age.as_payload()
        if self.status_exclusion is not None:
            payload["status_exclusion"] = self.status_exclusion.as_payload()
        if self.price_limit is not None:
            payload["price_limit"] = self.price_limit.as_payload()
        if self.liquidity is not None:
            payload["liquidity"] = self.liquidity.as_payload()
        if self.price_range is not None:
            payload["price_range"] = self.price_range.as_payload()
        if self.data_quality is not None:
            payload["data_quality"] = self.data_quality.as_payload()
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "UniverseSpec":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("candidate universe_rules must be an object when provided")
        unknown = set(payload) - UNIVERSE_TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"candidate universe_rules has unknown keys: {sorted(unknown)[:3]}")
        sections: dict[str, Any] = {}
        for section, rule_cls in (
            ("base_universe", BaseUniverseRule),
            ("index_membership", IndexMembershipRule),
            ("listing_age", ListingAgeRule),
            ("status_exclusion", StatusExclusionRule),
            ("price_limit", PriceLimitRule),
            ("liquidity", LiquidityRule),
            ("price_range", PriceRangeRule),
            ("data_quality", DataQualityRule),
        ):
            raw = payload.get(section)
            if raw is None:
                sections[section] = None
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"universe_rules.{section} must be an object when provided")
            sections[section] = rule_cls.from_payload(raw)
        return cls(
            trading_dates=payload.get("trading_dates"),
            price=payload.get("price"),
            base_universe=sections["base_universe"],
            index_membership=sections["index_membership"],
            listing_age=sections["listing_age"],
            status_exclusion=sections["status_exclusion"],
            price_limit=sections["price_limit"],
            liquidity=sections["liquidity"],
            price_range=sections["price_range"],
            data_quality=sections["data_quality"],
        )


EXECUTION_DATA_KEYS = frozenset({"trading_dates", "price", "amount", "lot_sizes", "halt", "price_limits"})

#: The two sides of a price limit; a name may carry either or both.
PRICE_LIMIT_SIDES = ("upper", "lower")


@dataclass(frozen=True)
class ExecutionDataSpec:
    """Explicitly injected order-layer data (roadmap 8.6-8.9).

    The four execution operators need prices, lot sizes, traded amounts and
    halt/limit states, none of which a weight vector can express, and neither
    engine carries market data of its own -- so the upper layer injects it here,
    exactly as ``universe_rules`` / ``industry_map`` do.  The panels are
    configuration inputs rather than agent-selectable parameters and never enter
    the LLM prompt vocabulary.

    ``trading_dates`` is the calendar the participation and staging windows
    count on and the as-of lookups use; it is deliberately a second, independent
    calendar rather than a re-reading of the universe section's, so a caller can
    supply order-layer dates without enabling a single pool filter.  The whole
    section is absent while nothing is injected, so the canonical payload of
    every pre-batch candidate keeps its strategy identity hash.

    ``halt`` is a **sparse exception panel, not a full-panel state table**: it
    records only the names that are *not* tradable on a given day, and a symbol
    absent from a day's row is tradable.  The side-aware mask reads a cell as
    ``state is not None and str(state) != ""``, so an empty status string means
    tradable and any non-empty cell means halted -- a caller that populated
    every cell with a status name would mark the whole book halted.  Omit the
    entry, or leave the cell empty, to say "this name trades normally".
    """

    trading_dates: tuple[str, ...] | None = None
    price: dict[str, dict[str, float]] | None = None
    amount: dict[str, dict[str, float]] | None = None
    lot_sizes: dict[str, int] | None = None
    halt: dict[str, dict[str, str]] | None = None
    price_limits: dict[str, dict[str, dict[str, float]]] | None = None

    def __post_init__(self) -> None:
        if self.trading_dates is not None:
            if isinstance(self.trading_dates, str) or not isinstance(self.trading_dates, (list, tuple)) or not self.trading_dates:
                raise ValueError("execution_data.trading_dates must be a non-empty list of date strings")
            parsed = [
                _parse_universe_date(day, "execution_data.trading_dates must be a non-empty list of date strings")
                for day in self.trading_dates
            ]
            if any(b <= a for a, b in zip(parsed, parsed[1:])):
                raise ValueError("execution_data.trading_dates must be strictly increasing")
            object.__setattr__(self, "trading_dates", tuple(str(day) for day in self.trading_dates))
        if self.price is not None:
            _validate_universe_number_panel(self.price, "execution_data.price")
            object.__setattr__(self, "price", {str(day): dict(row) for day, row in self.price.items()})
        if self.amount is not None:
            _validate_universe_number_panel(self.amount, "execution_data.amount")
            object.__setattr__(self, "amount", {str(day): dict(row) for day, row in self.amount.items()})
        if self.lot_sizes is not None:
            shape = "execution_data.lot_sizes must map non-empty symbols to whole positive lot sizes"
            if not isinstance(self.lot_sizes, dict) or not self.lot_sizes:
                raise ValueError(shape)
            sizes: dict[str, int] = {}
            for symbol, size in self.lot_sizes.items():
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError(shape)
                if (
                    isinstance(size, bool)
                    or not isinstance(size, (int, float))
                    or not math.isfinite(float(size))
                    or float(size) != int(size)
                    or int(size) < 1
                ):
                    raise ValueError(shape)
                sizes[str(symbol)] = int(size)
            object.__setattr__(self, "lot_sizes", sizes)
        if self.halt is not None:
            _validate_universe_string_panel(self.halt, "execution_data.halt")
            object.__setattr__(self, "halt", {str(day): dict(row) for day, row in self.halt.items()})
        if self.price_limits is not None:
            object.__setattr__(self, "price_limits", self._validated_price_limits())

    def _validated_price_limits(self) -> dict[str, dict[str, dict[str, float | None]]]:
        shape = "execution_data.price_limits must map date strings to objects of symbol -> {upper, lower}"
        if not isinstance(self.price_limits, dict) or not self.price_limits:
            raise ValueError(shape)
        limits: dict[str, dict[str, dict[str, float | None]]] = {}
        for date_key, row in self.price_limits.items():
            _parse_universe_date(date_key, f"execution_data.price_limits contains an invalid date: {date_key!r}")
            if not isinstance(row, dict) or not row:
                raise ValueError(shape)
            clean_row: dict[str, dict[str, float | None]] = {}
            for symbol, bounds in row.items():
                if not isinstance(symbol, str) or not symbol or not isinstance(bounds, Mapping):
                    raise ValueError(shape)
                unknown = {str(key) for key in set(bounds) - set(PRICE_LIMIT_SIDES)}
                if unknown:
                    raise ValueError(f"execution_data.price_limits has unknown keys: {sorted(unknown)[:3]}")
                clean_bounds: dict[str, float | None] = {}
                for side in PRICE_LIMIT_SIDES:
                    raw = bounds.get(side)
                    clean_bounds[side] = None if raw is None else _finite_universe_number(raw, shape)
                if clean_bounds["upper"] is None and clean_bounds["lower"] is None:
                    raise ValueError(shape)
                if (
                    clean_bounds["upper"] is not None
                    and clean_bounds["lower"] is not None
                    and clean_bounds["lower"] > clean_bounds["upper"]
                ):
                    raise ValueError(f"execution_data.price_limits lower must not exceed upper for {symbol} on {date_key}")
                clean_row[str(symbol)] = clean_bounds
            limits[str(date_key)] = clean_row
        return limits

    @property
    def enabled(self) -> bool:
        return any(
            section is not None
            for section in (self.trading_dates, self.price, self.amount, self.lot_sizes, self.halt, self.price_limits)
        )

    def as_payload(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        payload: dict[str, Any] = {}
        if self.trading_dates is not None:
            payload["trading_dates"] = list(self.trading_dates)
        for panel in ("price", "amount", "halt", "price_limits"):
            value = getattr(self, panel)
            if value is not None:
                payload[panel] = {day: dict(row) for day, row in value.items()}
        if self.lot_sizes is not None:
            payload["lot_sizes"] = dict(self.lot_sizes)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ExecutionDataSpec":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("candidate execution_data must be an object when provided")
        unknown = set(payload) - EXECUTION_DATA_KEYS
        if unknown:
            raise ValueError(f"candidate execution_data has unknown keys: {sorted(unknown)[:3]}")
        return cls(
            trading_dates=payload.get("trading_dates"),
            price=payload.get("price"),
            amount=payload.get("amount"),
            lot_sizes=payload.get("lot_sizes"),
            halt=payload.get("halt"),
            price_limits=payload.get("price_limits"),
        )


@dataclass(frozen=True)
class StrategyCandidate:
    """A portfolio strategy to be evaluated and registered.

    Either ``factor_names`` (references into the factor registry) or ``rule``
    (a declarative selection rule such as the free-cash-flow-rate ranking) must
    be given; both may be present when a rule is applied on top of factor
    scores.  Neither is interpreted here -- the composition layer receives the
    resulting score panel, so this class never touches market data.
    """

    name: str
    universe: str
    composition: CompositionSpec
    factor_names: tuple[str, ...] = ()
    rule: str = ""
    hypothesis: str = ""
    backtest: BacktestSpec = field(default_factory=BacktestSpec)
    signal: SignalSpec = field(default_factory=SignalSpec)
    gating: GatingSpec = field(default_factory=GatingSpec)
    universe_rules: UniverseSpec | None = None
    execution_data: ExecutionDataSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Strategy candidate requires a name")
        if not self.universe:
            raise ValueError("Strategy candidate requires a universe")
        if not self.factor_names and not self.rule:
            raise ValueError("Strategy candidate needs factor_names or a rule")
        if any(not name for name in self.factor_names):
            raise ValueError("factor_names must not contain empty names")
        if any(layer.factor not in self.factor_names for layer in self.gating.layers):
            raise ValueError("gating layers must reference candidate factors")
        if any(name not in self.factor_names for pair in self.signal.interaction_pairs for name in pair[:2]):
            raise ValueError("interaction terms must reference candidate factors")
        if self.signal.blend_weights is not None:
            expected = set(self.factor_names) | set(self.signal.interaction_member_names())
            if set(self.signal.blend_weights) != expected:
                raise ValueError(
                    "blend_weights must cover exactly the blend members: "
                    "the candidate factors plus every interaction term"
                )
        if self.signal.blend_groups is not None:
            grouped = {
                factor_name for factors in self.signal.blend_groups.values() for factor_name in factors
            }
            if grouped != set(self.factor_names):
                raise ValueError("blend_groups must partition the candidate factors exactly")
        if self.signal.regime_factor_groups is not None:
            for label, factors in self.signal.regime_factor_groups.items():
                unknown_regime = [name for name in factors if name not in self.factor_names]
                if unknown_regime:
                    raise ValueError("regime_factor_groups members must reference candidate factors")
        if self.composition.weighting == "factor_risk_budget" and self.composition.frb_budgets is not None:
            if set(self.composition.frb_budgets) != set(self.factor_names):
                raise ValueError("frb_budgets must cover exactly the candidate factors")

    def as_payload(self) -> dict[str, Any]:
        """Canonical payload used for identity hashing and registry storage."""

        payload = {
            "name": self.name,
            "universe": self.universe,
            "composition": self.composition.as_payload(),
            "factor_names": list(self.factor_names),
            "rule": self.rule,
            "backtest": self.backtest.as_payload(),
            "signal": self.signal.as_payload(),
        }
        # Absent while the gate is disabled, so the canonical payload (and the
        # strategy_id hash) of every pre-gating candidate stays unchanged.
        gating = self.gating.as_payload()
        if gating is not None:
            payload["gating"] = gating
        # Same lazy-key convention for the candidate-pool filter chain (batch
        # E, roadmap section 3): the whole section is absent while every
        # filter is disabled.
        universe_payload = None if self.universe_rules is None else self.universe_rules.as_payload()
        if universe_payload is not None:
            payload["universe_rules"] = universe_payload
        # And for the injected order-layer panels (batch 6, roadmap 8.6-8.9):
        # absent until some panel is actually supplied.
        execution_payload = None if self.execution_data is None else self.execution_data.as_payload()
        if execution_payload is not None:
            payload["execution_data"] = execution_payload
        return payload

    def canonical_payload(self) -> str:
        return json.dumps(self.as_payload(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StrategyCandidate":
        return cls(
            name=str(payload["name"]),
            universe=str(payload["universe"]),
            composition=CompositionSpec.from_payload(dict(payload["composition"])),
            factor_names=tuple(str(value) for value in payload.get("factor_names", [])),
            rule=str(payload.get("rule", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            backtest=BacktestSpec.from_payload(dict(payload.get("backtest", {}))),
            signal=SignalSpec.from_payload(dict(payload.get("signal", {}))),
            gating=GatingSpec.from_payload(payload.get("gating")),
            # An absent (or all-disabled) universe section restores as None,
            # the disabled default -- not as an empty spec object, so the
            # payload round-trip stays equality-preserving.
            universe_rules=(
                UniverseSpec.from_payload(payload["universe_rules"])
                if payload.get("universe_rules") is not None
                else None
            ),
            execution_data=(
                ExecutionDataSpec.from_payload(payload["execution_data"])
                if payload.get("execution_data") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )
