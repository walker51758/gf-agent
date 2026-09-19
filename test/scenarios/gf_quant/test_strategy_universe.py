"""Unit tests for the candidate-pool filter chain (roadmap sections 3.1-3.8).

Covers per-filter semantics and boundaries of ``universe.build_eligible_mask``,
the ``UniverseSpec`` payload contract (lazy key sets, validation, round-trip)
and the wiring into the local signal front-end.  End-to-end local-vs-worker
parity for the same filters lives in ``test_strategy_worker_parity.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.candidate import (
    BaseUniverseRule,
    CompositionSpec,
    DataQualityRule,
    ListingAgeRule,
    SignalSpec,
    StrategyCandidate,
    UniverseSpec,
)
from rdagent.scenarios.gf_quant.strategy.proposal import LLMStrategyProposalEngine
from rdagent.scenarios.gf_quant.strategy.signal import build_factor_rank_panels, build_factor_score
from rdagent.scenarios.gf_quant.strategy.universe import build_eligible_mask

SYMBOLS = ("S0", "S1", "S2", "S3")
CALENDAR = pd.bdate_range("2024-01-01", periods=20)
DECISION_DATES = pd.DatetimeIndex([CALENDAR[9], CALENDAR[13], CALENDAR[17], CALENDAR[19]])


def _d(day) -> str:
    return pd.Timestamp(day).strftime("%Y-%m-%d")


def _factor_panels() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(len(CALENDAR), len(SYMBOLS))), index=CALENDAR, columns=list(SYMBOLS))
    return {"value": frame, "quality": frame + 1.0}


def _trading_dates() -> list[str]:
    return [_d(day) for day in CALENDAR]


def _sparse(values: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return values


# ---------------------------------------------------------------------------
# Payload contract: lazy key sets, validation, round-trip.
# ---------------------------------------------------------------------------


def test_disabled_spec_is_absent_from_the_payload() -> None:
    spec = UniverseSpec()
    assert not spec.enabled
    assert spec.as_payload() is None
    assert UniverseSpec.from_payload(None).as_payload() is None
    candidate = StrategyCandidate(
        name="u",
        universe="csi_all",
        composition=CompositionSpec(top_n=4),
        factor_names=("a", "b"),
        rule="r",
    )
    assert "universe_rules" not in candidate.as_payload()


def test_payload_round_trip_and_lazy_sections() -> None:
    spec = UniverseSpec(
        trading_dates=tuple(_trading_dates()),
        price=_sparse({_d(day): {symbol: 10.0 for symbol in SYMBOLS} for day in CALENDAR}),
        base_universe=BaseUniverseRule(symbols=["S0"]),
        listing_age=ListingAgeRule(listing_dates={"S0": "2020-01-01"}, min_days=5),
        data_quality=DataQualityRule(min_coverage=0.5),
    )
    payload = spec.as_payload()
    assert set(payload) == {"trading_dates", "price", "base_universe", "listing_age", "data_quality"}
    restored = UniverseSpec.from_payload(json.loads(json.dumps(payload)))
    assert restored.as_payload() == payload


def test_disabled_filters_are_inert_in_the_mask() -> None:
    assert build_eligible_mask(None, factor_panels=_factor_panels(), index=DECISION_DATES, columns=SYMBOLS) is None
    assert build_eligible_mask(UniverseSpec(), factor_panels=_factor_panels(), index=DECISION_DATES, columns=SYMBOLS) is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"universe_rules_unknown": 1}, "candidate universe_rules has unknown keys: ['universe_rules_unknown']"),
        ({"base_universe": {"symbols": []}}, "universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings"),
        ({"base_universe": {}}, "universe_rules.base_universe is missing the symbols key"),
        ({"listing_age": {"listing_dates": {"S0": "2020-01-01"}, "min_days": 0}}, "universe_rules.listing_age.min_days must be a whole number >= 1"),
        ({"listing_age": {"min_days": 5}}, "universe_rules.listing_age is missing the listing_dates key"),
        ({"index_membership": {"membership": {}}}, "universe_rules.index_membership.membership must map date strings to objects of symbol -> true"),
        ({"index_membership": {"membership": {"2024-01-01": {"S0": False}}}}, "universe_rules.index_membership.membership must map date strings to objects of symbol -> true"),
        ({"index_membership": {"membership": {"2024-01-01": {"S0": True}}, "lag_days": -1}}, "universe_rules.index_membership.lag_days must be a whole number >= 0"),
        ({"status_exclusion": {"statuses": {}, "excluded_statuses": []}}, "universe_rules.status_exclusion.excluded_statuses must be a non-empty list of distinct non-empty strings"),
        ({"price_limit": {"limit_up": {"2024-01-01": {"S0": 1.0}}, "limit_down": None, "exclude_limit_up": False, "exclude_limit_down": False}}, "universe_rules.price_limit must exclude at least one of limit_up and limit_down"),
        ({"liquidity": {"amount": {"2024-01-01": {"S0": 1.0}}}}, "universe_rules.liquidity requires min_amount or quantile"),
        ({"liquidity": {"amount": {}, "min_amount": 1.0}}, "universe_rules.liquidity.amount must map date strings to objects of symbol -> number"),
        ({"liquidity": {"amount": {"2024-01-01": {"S0": 1.0}}, "min_amount": 1.0, "missing": "both"}}, "universe_rules.liquidity.missing must be one of ('exclude', 'include'), got 'both'"),
        ({"price_range": {}}, "universe_rules.price_range requires min_price or max_price"),
        ({"price_range": {"min_price": 5.0, "max_price": 1.0}}, "universe_rules.price_range.min_price must not exceed max_price"),
        ({"data_quality": {}}, "universe_rules.data_quality requires min_coverage, max_missing_run or non_positive_price"),
        ({"data_quality": {"min_coverage": 1.5}}, "universe_rules.data_quality.min_coverage must be in [0, 1] when provided"),
        ({"listing_age": {"listing_dates": {"S0": "2020-01-01"}, "min_days": 5}}, "universe_rules.trading_dates is required when listing_age, liquidity or price_range is enabled"),
        ({"price_limit": {"limit_up": {"2024-01-01": {"S0": 1.0}}, "exclude_limit_up": True, "exclude_limit_down": False}}, "universe_rules.price is required when price_limit is enabled"),
        ({"price_range": {"min_price": 1.0}, "trading_dates": ["2024-01-01"]}, "universe_rules.price is required when price_range is enabled"),
        ({"data_quality": {"non_positive_price": True}}, "universe_rules.price is required when data_quality requests the non-positive-price check"),
        ({"trading_dates": ["2024-01-02", "2024-01-01"]}, "universe_rules.trading_dates must be strictly increasing"),
        # Malformed-typing family (verifier C1): non-integral floats, strings
        # into list fields, non-bools into bool fields, bools into whole
        # numbers -- same message on both ends.
        ({"index_membership": {"membership": {"2024-01-01": {"S0": True}}, "lag_days": 1.5}}, "universe_rules.index_membership.lag_days must be a whole number >= 0"),
        ({"index_membership": {"membership": {"2024-01-01": {"S0": True}}, "lag_days": True}}, "universe_rules.index_membership.lag_days must be a whole number >= 0"),
        ({"listing_age": {"listing_dates": {"S0": "2020-01-01"}, "min_days": "3"}}, "universe_rules.listing_age.min_days must be a whole number >= 1"),
        ({"liquidity": {"amount": {"2024-01-01": {"S0": 1.0}}, "min_amount": 1.0, "missing": 123}}, "universe_rules.liquidity.missing must be one of ('exclude', 'include'), got 123"),
        ({"base_universe": {"symbols": "S0"}}, "universe_rules.base_universe.symbols must be a non-empty list of distinct non-empty strings"),
        ({"status_exclusion": {"statuses": {}, "excluded_statuses": "ST"}}, "universe_rules.status_exclusion.excluded_statuses must be a non-empty list of distinct non-empty strings"),
        ({"data_quality": {"non_positive_price": "yes"}}, "universe_rules.data_quality.non_positive_price must be a boolean"),
    ],
)
def test_spec_validation_fails_closed_with_stable_messages(payload, message) -> None:
    with pytest.raises(ValueError) as error:
        UniverseSpec.from_payload(payload)
    assert str(error.value) == message


def test_whole_number_fields_are_normalised_after_validation() -> None:
    """An integral float (1.0) is semantically 1: accepted on both ends and
    written back as an int, so the spec never carries a value the executor
    would trip over (``shift(1.0)`` raised TypeError before this)."""

    spec = UniverseSpec.from_payload(
        {
            "trading_dates": _trading_dates(),
            "index_membership": {
                "membership": {_d(CALENDAR[0]): {"S0": True}},
                "lag_days": 1.0,
                "exclude_recent_days": 2.0,
            },
            "listing_age": {"listing_dates": {"S0": "1990-01-01"}, "min_days": 3.0},
            "liquidity": {"amount": {_d(CALENDAR[0]): {"S0": 1.0}}, "window": 5.0, "min_amount": 1.0},
            "price": {_d(day): {symbol: 10.0 for symbol in SYMBOLS} for day in CALENDAR},
            "price_range": {"min_price": 1.0, "consecutive_days": 2.0},
            "data_quality": {"min_coverage": 0.5, "max_missing_run": 2.0, "window": 4.0},
        }
    )
    assert type(spec.index_membership.lag_days) is int and spec.index_membership.lag_days == 1
    assert type(spec.index_membership.exclude_recent_days) is int and spec.index_membership.exclude_recent_days == 2
    assert type(spec.listing_age.min_days) is int and spec.listing_age.min_days == 3
    assert type(spec.liquidity.window) is int and spec.liquidity.window == 5
    assert type(spec.price_range.consecutive_days) is int and spec.price_range.consecutive_days == 2
    assert type(spec.data_quality.max_missing_run) is int and spec.data_quality.max_missing_run == 2
    assert type(spec.data_quality.window) is int and spec.data_quality.window == 4
    # The canonical payload carries the normalised ints, so 1.0 and 1 in the
    # incoming JSON hash to the same strategy identity.
    payload = json.loads(json.dumps(spec.as_payload()))
    assert payload["index_membership"]["lag_days"] == 1
    assert payload["listing_age"]["min_days"] == 3
    assert payload["price_range"]["consecutive_days"] == 2
    # The normalised spec still executes (the original TypeError regression).
    mask = build_eligible_mask(spec, factor_panels=_factor_panels(), index=DECISION_DATES, columns=SYMBOLS)
    assert mask is not None


def test_listing_age_without_calendar_and_quality_without_panels_fail_closed() -> None:
    with pytest.raises(ValueError, match="universe trading_dates do not contain decision date"):
        build_eligible_mask(
            UniverseSpec.from_payload(
                {
                    "trading_dates": _trading_dates()[:5],
                    "listing_age": {"listing_dates": {symbol: "2020-01-01" for symbol in SYMBOLS}, "min_days": 1},
                }
            ),
            factor_panels=_factor_panels(),
            index=DECISION_DATES,
            columns=SYMBOLS,
        )
    with pytest.raises(ValueError, match="universe data_quality coverage requires factor panels"):
        build_eligible_mask(
            UniverseSpec.from_payload({"data_quality": {"min_coverage": 0.5}}),
            factor_panels={},
            index=DECISION_DATES,
            columns=SYMBOLS,
        )


# ---------------------------------------------------------------------------
# 3.1 BaseUniverse.
# ---------------------------------------------------------------------------


def test_base_universe_excludes_unknown_symbols_on_every_date() -> None:
    mask = build_eligible_mask(
        UniverseSpec.from_payload({"base_universe": {"symbols": ["S0", "S2"]}}),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert mask is not None
    assert mask["S0"].all() and mask["S2"].all()
    assert not mask["S1"].any() and not mask["S3"].any()


# ---------------------------------------------------------------------------
# 3.2 IndexMembershipFilter.
# ---------------------------------------------------------------------------


def _membership_payload(lag_days: int = 0, exclude_recent_days: int = 0) -> dict:
    # S0/S1 members from the second calendar day on; S2 only from the 12th;
    # S3 never.
    membership = {
        _d(day): {"S0": True, "S1": True} for day in CALENDAR[1:6]
    }
    membership.update({_d(day): {"S0": True, "S1": True, "S2": True} for day in CALENDAR[11:]})
    payload = {
        "index_membership": {
            "membership": membership,
            "lag_days": lag_days,
            "exclude_recent_days": exclude_recent_days,
        }
    }
    return payload


def test_membership_as_of_lookup_and_column_coverage() -> None:
    mask = build_eligible_mask(
        UniverseSpec.from_payload(_membership_payload()),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert mask is not None
    # S3 is never a member; S2 joins at CALENDAR[11] and the first decision
    # date (CALENDAR[9]) precedes it.
    assert not mask.loc[DECISION_DATES[0], "S2"]
    assert not mask["S3"].any()
    assert mask.loc[DECISION_DATES[1], "S2"]
    assert mask["S0"].all() and mask["S1"].all()


def test_membership_lag_and_new_inclusion_window() -> None:
    # lag_days shifts the panel one row later: CALENDAR[11] membership becomes
    # effective at CALENDAR[12].
    lagged = build_eligible_mask(
        UniverseSpec.from_payload(_membership_payload(lag_days=1)),
        factor_panels=_factor_panels(),
        index=pd.DatetimeIndex([CALENDAR[11], CALENDAR[12]]),
        columns=SYMBOLS,
    )
    assert not lagged.loc[CALENDAR[11], "S2"]
    assert lagged.loc[CALENDAR[12], "S2"]

    # exclude_recent_days=1 bars the inclusion row itself only.
    fresh = build_eligible_mask(
        UniverseSpec.from_payload(_membership_payload(exclude_recent_days=1)),
        factor_panels=_factor_panels(),
        index=pd.DatetimeIndex([CALENDAR[11], CALENDAR[12]]),
        columns=SYMBOLS,
    )
    assert not fresh.loc[CALENDAR[11], "S2"]
    assert fresh.loc[CALENDAR[12], "S2"]
    # Pre-existing members (member at the panel start) are never "new".
    assert fresh["S0"].all()


# ---------------------------------------------------------------------------
# 3.3 ListingAgeFilter.
# ---------------------------------------------------------------------------


def test_listing_age_boundary_is_inclusive_and_missing_dates_exclude() -> None:
    payload = {
        "trading_dates": _trading_dates(),
        "listing_age": {
            "listing_dates": {"S0": _d(CALENDAR[5]), "S1": "1990-01-01"},
            "min_days": 3,
        },
    }
    mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=pd.DatetimeIndex([CALENDAR[7], CALENDAR[8]]),
        columns=SYMBOLS,
    )
    # S0 lists at CALENDAR[5]; trailing trading days strictly after it are
    # CALENDAR[6..] -- the 3rd is CALENDAR[8], so CALENDAR[7] fails and
    # CALENDAR[8] passes.
    assert not mask.loc[CALENDAR[7], "S0"]
    assert mask.loc[CALENDAR[8], "S0"]
    assert mask["S1"].all()
    # S2/S3 have no listing date and can never be age-verified.
    assert not mask["S2"].any() and not mask["S3"].any()


# ---------------------------------------------------------------------------
# 3.4 StatusExclusionFilter.
# ---------------------------------------------------------------------------


def test_status_exclusion_is_exact_date_and_sparse_is_normal() -> None:
    payload = {
        "status_exclusion": {
            "statuses": {
                _d(DECISION_DATES[0]): {"S0": "ST", "S1": "suspended"},
                _d(CALENDAR[2]): {"S2": "ST"},  # recovered before the decisions
            },
            "excluded_statuses": ["ST", "suspended", "delisting"],
        }
    }
    mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert not mask.loc[DECISION_DATES[0], "S0"]
    assert not mask.loc[DECISION_DATES[0], "S1"]
    assert mask.loc[DECISION_DATES[1], "S0"]  # status lifted
    assert mask["S2"].all() and mask["S3"].all()


# ---------------------------------------------------------------------------
# 3.5 PriceLimitTradabilityFilter.
# ---------------------------------------------------------------------------


def _price_and_limit_payload(limit: float, prices: dict[str, float], *, side: str = "up") -> dict:
    price = _sparse({_d(day): dict(prices) for day in CALENDAR})
    limit_panel = _sparse({_d(day): {symbol: limit for symbol in prices} for day in CALENDAR})
    section = {f"limit_{side}": limit_panel, f"exclude_limit_{side}": True}
    other = "down" if side == "up" else "up"
    section[f"limit_{other}"] = None
    section[f"exclude_limit_{other}"] = False
    return {"trading_dates": _trading_dates(), "price": price, "price_limit": section}


def test_price_limit_touching_excludes_and_missing_data_fails_closed() -> None:
    prices = {"S0": 110.0, "S1": 109.0, "S2": 111.0, "S3": 110.0}
    mask = build_eligible_mask(
        UniverseSpec.from_payload(_price_and_limit_payload(110.0, prices, side="up")),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    # Touching (==) and breaching (>) both exclude; strictly below passes.
    assert not mask["S0"].any() and not mask["S2"].any()
    assert mask["S1"].all()

    # A decision date the panels do not cover cannot establish tradability.
    partial_price_panel = _sparse({_d(day): dict(prices) for day in CALENDAR[:5]})
    payload = _price_and_limit_payload(110.0, prices, side="up")
    payload["price"] = partial_price_panel
    payload["price_limit"]["limit_up"] = _sparse({_d(day): {symbol: 110.0 for symbol in prices} for day in CALENDAR[:5]})
    late_mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert not late_mask.any().any()


# ---------------------------------------------------------------------------
# 3.6 LiquidityFilter.
# ---------------------------------------------------------------------------


def _amount_payload(amounts: dict[str, float], **overrides) -> dict:
    amount = _sparse({_d(day): {symbol: value for symbol, value in amounts.items()} for day in CALENDAR})
    section = {"amount": amount, "window": 3, "min_amount": None, "quantile": None, "missing": "exclude"}
    section.update(overrides)
    return {"trading_dates": _trading_dates(), "liquidity": section}


def test_liquidity_mean_threshold_boundary_is_inclusive() -> None:
    amounts = {"S0": 100.0, "S1": 99.999999, "S2": 50.0, "S3": 100.0}
    payload = _amount_payload(amounts, min_amount=100.0)
    mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert mask["S0"].all()  # exactly at the threshold passes
    assert not mask["S1"].any() and not mask["S2"].any()
    assert mask["S3"].all()


def test_liquidity_quantile_leg_and_missing_policy() -> None:
    amounts = {"S0": 100.0, "S1": 50.0, "S2": 75.0, "S3": 25.0}
    payload = _amount_payload(amounts, quantile=0.5)
    mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    # The rolling mean equals the constant amount; the cross-sectional median
    # is 75, so S0 passes and S2 sits exactly on it (inclusive).
    assert mask["S0"].all() and mask["S2"].all()
    assert not mask["S1"].any() and not mask["S3"].any()

    # A symbol with no amount data at all: excluded by default, kept with
    # missing="include".
    empty = _amount_payload({"S0": 100.0}, min_amount=50.0)
    out = build_eligible_mask(
        UniverseSpec.from_payload(empty),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert mask is not None
    assert not out["S1"].any()
    included = build_eligible_mask(
        UniverseSpec.from_payload(_amount_payload({"S0": 100.0}, min_amount=50.0, missing="include")),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert included["S1"].all() and included["S2"].all() and included["S3"].all()


# ---------------------------------------------------------------------------
# 3.7 PriceRangeFilter.
# ---------------------------------------------------------------------------


def _price_range_payload(prices: dict[str, float], **overrides) -> dict:
    price = _sparse({_d(day): {symbol: value for symbol, value in prices.items()} for day in CALENDAR})
    section = {"min_price": None, "max_price": None, "consecutive_days": 1}
    section.update(overrides)
    return {"trading_dates": _trading_dates(), "price": price, "price_range": section}


def test_price_range_bounds_are_inclusive_and_streaks_break_on_missing() -> None:
    payload = _price_range_payload({"S0": 5.0, "S1": 4.999999, "S2": 20.000001, "S3": 10.0}, min_price=5.0, max_price=20.0)
    mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert mask["S0"].all() and mask["S3"].all()  # exactly on either bound passes
    assert not mask["S1"].any() and not mask["S2"].any()

    # A single out-of-band day inside the streak window breaks it.
    streak_price = _sparse({_d(day): {symbol: 10.0 for symbol in SYMBOLS} for day in CALENDAR})
    streak_price[_d(CALENDAR[18])]["S0"] = 1.0
    streak = build_eligible_mask(
        UniverseSpec.from_payload({"trading_dates": _trading_dates(), "price": streak_price, "price_range": {"min_price": 5.0, "consecutive_days": 3}}),
        factor_panels=_factor_panels(),
        index=pd.DatetimeIndex([CALENDAR[19]]),
        columns=SYMBOLS,
    )
    assert not streak.loc[CALENDAR[19], "S0"]
    assert streak.loc[CALENDAR[19], "S1"]

    # A window reaching before the calendar start cannot confirm the band.
    early = build_eligible_mask(
        UniverseSpec.from_payload({"trading_dates": _trading_dates()[:3], "price": {k: v for k, v in _price_range_payload({"S0": 10.0}, min_price=1.0)["price"].items() if k in _trading_dates()[:3]}, "price_range": {"min_price": 1.0, "consecutive_days": 5}}),
        factor_panels=_factor_panels(),
        index=pd.DatetimeIndex([CALENDAR[2]]),
        columns=SYMBOLS,
    )
    assert not early.loc[CALENDAR[2], "S0"]


# ---------------------------------------------------------------------------
# 3.8 DataQualityFilter.
# ---------------------------------------------------------------------------


def test_data_quality_coverage_boundary_and_missing_run() -> None:
    panels = _factor_panels()
    # S0 loses exactly one of the 2*3 = 6 factor cells in the trailing
    # 3-row window before the first decision date: coverage 5/6.
    panels["value"].loc[CALENDAR[9], "S0"] = np.nan
    spec = UniverseSpec.from_payload({"data_quality": {"min_coverage": 5 / 6, "window": 3}})
    mask = build_eligible_mask(spec, factor_panels=panels, index=DECISION_DATES, columns=SYMBOLS)
    assert mask.loc[DECISION_DATES[0], "S0"]  # exactly at the bound passes
    tighter = UniverseSpec.from_payload({"data_quality": {"min_coverage": 0.9, "window": 3}})
    strict = build_eligible_mask(tighter, factor_panels=panels, index=DECISION_DATES, columns=SYMBOLS)
    assert not strict.loc[DECISION_DATES[0], "S0"]
    assert strict.loc[DECISION_DATES[0], "S1"]

    # A trailing missing run of exactly max_missing_run passes; one longer
    # fails.
    run_panels = _factor_panels()
    run_panels["value"].loc[[CALENDAR[8], CALENDAR[9]], "S1"] = np.nan
    two_ok = build_eligible_mask(
        UniverseSpec.from_payload({"data_quality": {"max_missing_run": 2, "window": 3}}),
        factor_panels=run_panels,
        index=pd.DatetimeIndex([CALENDAR[9]]),
        columns=SYMBOLS,
    )
    assert two_ok.loc[CALENDAR[9], "S1"]
    run_panels["value"].loc[CALENDAR[7], "S1"] = np.nan
    three_bad = build_eligible_mask(
        UniverseSpec.from_payload({"data_quality": {"max_missing_run": 2, "window": 3}}),
        factor_panels=run_panels,
        index=pd.DatetimeIndex([CALENDAR[9]]),
        columns=SYMBOLS,
    )
    assert not three_bad.loc[CALENDAR[9], "S1"]


def test_data_quality_non_positive_price_check() -> None:
    price = _sparse(
        {
            _d(day): {"S0": 10.0, "S1": 0.0, "S2": -1.0, "S3": 10.0}
            for day in CALENDAR
        }
    )
    mask = build_eligible_mask(
        UniverseSpec.from_payload({"price": price, "data_quality": {"non_positive_price": True}}),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    assert mask["S0"].all() and mask["S3"].all()
    assert not mask["S1"].any() and not mask["S2"].any()


# ---------------------------------------------------------------------------
# Chain composition and signal-front-end wiring.
# ---------------------------------------------------------------------------


def test_filters_intersect_and_order_leaves_the_mask_inert() -> None:
    payload = {
        "trading_dates": _trading_dates(),
        "base_universe": {"symbols": ["S0", "S1", "S2"]},
        "price": _sparse({_d(day): {symbol: 10.0 for symbol in SYMBOLS} for day in CALENDAR}),
        "price_range": {"min_price": 5.0, "max_price": 20.0, "consecutive_days": 1},
        "status_exclusion": {
            "statuses": {_d(DECISION_DATES[0]): {"S1": "ST"}},
            "excluded_statuses": ["ST"],
        },
    }
    mask = build_eligible_mask(
        UniverseSpec.from_payload(payload),
        factor_panels=_factor_panels(),
        index=DECISION_DATES,
        columns=SYMBOLS,
    )
    expected = pd.DataFrame(True, index=DECISION_DATES, columns=list(SYMBOLS))
    expected["S3"] = False
    expected.loc[DECISION_DATES[0], "S1"] = False
    pd.testing.assert_frame_equal(mask, expected)


def test_eligible_mask_masks_factor_ranks_and_disabled_path_is_untouched() -> None:
    panels = _factor_panels()
    directions = {"value": 1, "quality": 1}
    spec = SignalSpec()
    mask = pd.DataFrame(True, index=DECISION_DATES, columns=list(SYMBOLS))
    mask["S3"] = False
    masked = build_factor_rank_panels(panels, directions, spec, index=DECISION_DATES, columns=SYMBOLS, eligible=mask)
    assert masked["value"]["S3"].isna().all()
    # Percentile ranks of the survivors are computed within the post-filter
    # cross-section (three valid names), so the top name sits at 1.0.
    top_row = masked["value"].loc[DECISION_DATES[0]].dropna()
    assert float(top_row.max()) == pytest.approx(1.0)
    assert len(top_row) == 3

    baseline = build_factor_score(panels, directions, spec, index=DECISION_DATES, columns=SYMBOLS)
    untouched = build_factor_score(
        panels, directions, spec, index=DECISION_DATES, columns=SYMBOLS, eligible=None
    )
    pd.testing.assert_frame_equal(baseline, untouched)


# ---------------------------------------------------------------------------
# Proposal passthrough (deliberately absent from the LLM vocabulary).
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, payload):
        self.payload = payload

    def build_messages_and_create_chat_completion(self, **kwargs):
        return json.dumps(self.payload)


def _proposal_context(tmp_path: Path):
    from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set
    from rdagent.scenarios.gf_quant.strategy.orchestrator import StrategyProposalContext

    registry = FactorRegistry(tmp_path / "factors.sqlite")
    for position, name in enumerate(("value", "quality", "cashflow")):
        registry.upsert(
            FactorRecord(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                status="accepted",
                metrics={"Rank_IC": 0.02 + position * 0.01, "sharpe": 0.5},
                artifact_path=f"G:/artifacts/{name}.parquet",
                artifact_checksum=(str(position + 1) * 64),
                metadata={"search_direction": name},
            )
        )
    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")
    working = build_factor_working_set(snapshot, core_limit=3, conditional_limit=0)
    return StrategyProposalContext(
        epoch_id="epoch_001",
        trial_index=2,
        data_cutoff="2023-12-31",
        snapshot_id=snapshot.snapshot_id,
        working_set=working,
        prior_outcomes=(),
    )


def _proposal_payload(**extra) -> dict:
    return {
        "name": "Value + Quality",
        "factor_names": ["value", "quality"],
        "rule": "equal rank blend then top 50",
        "top_n": 50,
        "weighting": "equal",
        "signal_lag": 1,
        **extra,
    }


def test_proposal_passes_universe_rules_through_and_fails_closed(tmp_path) -> None:
    context = _proposal_context(tmp_path)
    universe_payload = {"base_universe": {"symbols": ["S0", "S1"]}}
    candidate = LLMStrategyProposalEngine(_FakeBackend(_proposal_payload(universe_rules=universe_payload)))(context)
    assert candidate.universe_rules is not None
    assert candidate.universe_rules.base_universe.symbols == ("S0", "S1")

    with pytest.raises(ValueError, match="universe_rules.base_universe.symbols"):
        LLMStrategyProposalEngine(_FakeBackend(_proposal_payload(universe_rules={"base_universe": {"symbols": []}})))(context)

    plain = LLMStrategyProposalEngine(_FakeBackend(_proposal_payload()))(context)
    assert plain.universe_rules is None
