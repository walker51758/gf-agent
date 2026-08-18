from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.dedup import FactorDeduplicator, FactorSignature
from rdagent.scenarios.gf_quant.data.loader import build_daily_pv_from_base, build_fundamental_from_base
from rdagent.scenarios.gf_quant.metrics import format_factor_feedback
from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.service import (
    EvaluationResult,
    FactorAcceptanceService,
    FactorCandidate,
)
from rdagent.scenarios.gf_quant.split import DateSplit, Period, build_ratio_split, build_rolling_splits


def _dates(count: int) -> list[date]:
    start = date(2020, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def test_split_validation_and_data_driven_boundaries() -> None:
    split = build_ratio_split(_dates(100), train_ratio=0.6, valid_ratio=0.2)
    assert split.train.end < split.valid.start < split.test.start
    assert split.train.start == date(2020, 1, 1)
    assert split.test.end == _dates(100)[-1]
    with pytest.raises(ValueError, match="overlap"):
        DateSplit(Period.parse("2024-01-01", "2024-06-01"), Period.parse("2024-06-01", "2024-07-01"), Period.parse("2024-08-01", "2024-09-01"))


def test_rolling_splits_are_chronological() -> None:
    splits = build_rolling_splits(_dates(120), train_size=50, valid_size=20, test_size=10, step_size=10)
    assert len(splits) == 5
    assert all(split.train.end < split.valid.start < split.test.start for split in splits)
    assert splits[1].train.start == splits[0].train.start  # expanding window
    assert splits[1].test.start > splits[0].test.start


def test_metric_feedback_is_complete_and_handles_missing_values() -> None:
    text = format_factor_feedback(
        {
            "IC": 0.03,
            "ICIR": 0.2,
            "Rank_IC": 0.04,
            "Rank_ICIR": 0.3,
            "annualized_return": 0.1,
            "sharpe": 0.7,
            "max_drawdown": -0.2,
            "turnover": 0.3,
            "coverage": 0.9,
            "long_short_spread_daily": 0.001,
            "win_rate": 0.52,
            "stability_by_year": {"2025": {"annualized_return": 0.12, "days": 240}},
        }
    )
    for label in ("ICIR", "RankICIR", "Annualized return", "Turnover", "Coverage", "Win rate", "2025"):
        assert label in text
    assert "N/A" in text  # absent standard deviations are rendered, not crashed


def test_dedup_exact_and_statistical_reasons() -> None:
    dedup = FactorDeduplicator(threshold=0.88, min_overlap=5)
    exact = dedup.compare(FactorSignature("new", "a + b"), [FactorSignature("old", "a+b")])
    assert exact.is_duplicate and exact.reason == "normalized_expression_match"

    values = pd.Series(range(10), dtype=float)
    correlated = dedup.compare(
        FactorSignature("new", "x", values=values),
        [FactorSignature("old", "y", values=-values)],
    )
    assert correlated.is_duplicate and correlated.reason == "absolute_pearson"
    assert correlated.pearson == pytest.approx(-1.0)

    return_duplicate = dedup.compare(
        FactorSignature("new", "x", return_profile=values),
        [FactorSignature("old", "y", return_profile=values * 2.0)],
    )
    assert return_duplicate.is_duplicate and return_duplicate.reason == "return_profile_correlation"

    distinct = dedup.compare(
        FactorSignature("new", "x", values=pd.Series([0, 1, 0, 1, 0, 1], dtype=float)),
        [FactorSignature("old", "y", values=pd.Series([0, 0, 1, 1, 1, 0], dtype=float))],
    )
    assert not distinct.is_duplicate


def test_registry_roundtrip_and_filters(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    original = FactorRecord(
        factor_id="gf_one",
        name="one",
        family="price_volume",
        expression="-pct_change(adjclose, 20)",
        status="accepted",
        metrics={"Rank_IC": 0.05},
        artifact_path="G:/artifacts/one.parquet",
        artifact_checksum="abc",
    )
    stored = registry.upsert(original)
    assert stored.metrics == {"Rank_IC": 0.05}
    assert registry.get_by_name("one") == stored
    assert registry.list(family="price_volume", status="accepted") == [stored]


def test_acceptance_service_supports_both_families_and_rejection(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")

    def evaluator(candidate: FactorCandidate) -> EvaluationResult:
        good = candidate.name != "weak"
        return EvaluationResult(
            metrics={
                "Rank_IC": 0.05 if good else 0.001,
                "annualized_return": 0.08 if good else -0.10,
                "sharpe": 0.55 if good else -0.5,
                "max_drawdown": -0.16,
                "turnover": 0.2,
                "coverage": 0.99,
            },
            artifact_path=f"G:/artifacts/{candidate.name}.parquet",
            artifact_checksum="checksum",
        )

    service = FactorAcceptanceService(registry, evaluator)
    pv = service.process(FactorCandidate("pv", "price_volume", "-pct_change(adjclose, 20)"))
    fundamental = service.process(FactorCandidate("fund", "fundamental", "cs_rank(balance__parent_equity / mv)"))
    weak = service.process(FactorCandidate("weak", "fundamental", "cs_rank(income__net_profit / mv)"))
    assert pv.record.status == fundamental.record.status == "accepted"
    assert weak.record.status == "rejected"
    assert "rank_ic" in weak.gate_failures


def test_base_adapter_uses_adjusted_prices_and_announcement_dates(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    for field, values in {
        "adjopen": [10.0, 10.5],
        "adjclose": [10.4, 10.7],
        "adjhigh": [10.6, 10.9],
        "adjlow": [9.9, 10.3],
        "volume": [1000.0, 1200.0],
    }.items():
        pd.DataFrame(
            {"dt": dates, "symbol": ["000001.SZ", "000001.SZ"], "update_time": dates, "value": values}
        ).to_parquet(base / f"{field}.parquet")
    pd.DataFrame(
        {
            "dt": pd.to_datetime(["2025-01-03"]),
            "factor": ["net_profit"],
            "report": pd.to_datetime(["2024-12-31"]),
            "symbol": ["000001.SZ"],
            "update_time": pd.to_datetime(["2025-01-03"]),
            "value": [123.0],
        }
    ).to_parquet(base / "income.parquet")
    settings = SimpleNamespace(
        data_base_path=str(base),
        data_direct_field_files={},
        data_end_date="2025-01",
        data_financial_pattern="",
    )
    daily = build_daily_pv_from_base(settings)
    assert daily.loc[(pd.Timestamp("2025-01-02"), "000001.SZ"), "$close"] == 10.4
    assert (daily["$factor"] == 1.0).all()
    fundamental = build_fundamental_from_base(settings, daily.index.get_level_values("datetime").unique())
    assert fundamental is not None
    assert pd.isna(fundamental.loc[(pd.Timestamp("2025-01-02"), "000001.SZ"), "net_profit"])
    assert fundamental.loc[(pd.Timestamp("2025-01-03"), "000001.SZ"), "net_profit"] == 123.0
