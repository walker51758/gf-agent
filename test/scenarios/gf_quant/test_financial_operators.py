from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.developer.coder import validate_dsl_expression
from scripts.gf_quant_remote_worker import (
    A2B,
    CAGR,
    Delta,
    DeltaA2B,
    DeltaProp,
    DeltaRegA2B,
    IndusProp,
    OPERATOR_SPECS,
    PanelLoader,
    PercentileA2B,
    QOQ,
    RegA2B,
    SUD,
    Self,
    YOY,
    YOYChange,
    YOYA_YOYB,
    YOYOQOQ,
)


FINANCIAL_OPERATOR_NAMES = {
    "Self",
    "YOY",
    "QOQ",
    "YOYOQOQ",
    "Delta",
    "SUD",
    "IndusProp",
    "DeltaProp",
    "A2B",
    "PercentileA2B",
    "RegA2B",
    "DeltaA2B",
    "YOYA_YOYB",
    "DeltaRegA2B",
    "YOYChange",
    "CAGR",
}


def _financial_panel(values: dict[str, list[float]], reports: pd.DatetimeIndex) -> pd.DataFrame:
    announcements = reports + pd.to_timedelta(45, unit="D")
    rows: list[dict[str, object]] = []
    for symbol, series in values.items():
        rows.extend(
            {"dt": dt, "report": report, "symbol": symbol, "value": value}
            for dt, report, value in zip(announcements, reports, series, strict=True)
        )
    events = pd.DataFrame(rows)
    frame = events.pivot(index="dt", columns="symbol", values="value").sort_index().astype(float)
    frame.attrs["gf_financial_events"] = events
    frame.attrs["gf_calendar"] = frame.index
    frame.attrs["gf_financial_field"] = "income__net_profit"
    frame.attrs["gf_quarter_lag_cache"] = {}
    return frame


def test_all_directed_financial_operators_are_registered() -> None:
    assert FINANCIAL_OPERATOR_NAMES <= set(OPERATOR_SPECS)
    assert len(OPERATOR_SPECS) == 58


def test_quarterly_growth_and_change_use_report_periods() -> None:
    reports = pd.date_range("2019-03-31", periods=13, freq="QE")
    values = [80.0, 90.0, 95.0, 100.0, 100.0, 120.0, 125.0, 140.0, 130.0, 150.0, 165.0, 175.0, 190.0]
    panel = _financial_panel({"A": values}, reports)
    current = panel.index[-1]

    assert Self(panel).loc[current, "A"] == 190.0
    assert Delta(panel).loc[current, "A"] == pytest.approx(60.0)
    assert YOY(panel).loc[current, "A"] == pytest.approx(60.0 / 130.0)
    assert QOQ(panel).loc[current, "A"] == pytest.approx(15.0 / 175.0)
    prior_yoy = (175.0 - 140.0) / 140.0
    assert YOYOQOQ(panel).loc[current, "A"] == pytest.approx(60.0 / 130.0 - prior_yoy)
    four_quarters_ago_yoy = (130.0 - 100.0) / 100.0
    assert YOYChange(panel, 4).loc[current, "A"] == pytest.approx(60.0 / 130.0 - four_quarters_ago_yoy)
    assert CAGR(panel, 4).loc[current, "A"] == pytest.approx(190.0 / 130.0 - 1.0)


def test_cagr_requires_positive_endpoints() -> None:
    reports = pd.date_range("2020-03-31", periods=5, freq="QE")
    nonpositive_current = _financial_panel({"A": [10.0, 11.0, 12.0, 13.0, 0.0]}, reports)
    nonpositive_base = _financial_panel({"A": [-10.0, 11.0, 12.0, 13.0, 20.0]}, reports)
    assert np.isnan(CAGR(nonpositive_current, 4).iloc[-1, 0])
    assert np.isnan(CAGR(nonpositive_base, 4).iloc[-1, 0])


def test_sud_excludes_current_change_and_uses_previous_eight_quarters() -> None:
    reports = pd.date_range("2019-03-31", periods=13, freq="QE")
    values = np.array([70, 75, 82, 90, 100, 108, 119, 131, 146, 158, 175, 193, 220], dtype=float)
    panel = _financial_panel({"A": values.tolist()}, reports)
    current_change = values[12] - values[8]
    previous_changes = np.array([values[12 - k] - values[12 - k - 4] for k in range(1, 9)])
    expected = (current_change - previous_changes.mean()) / previous_changes.std(ddof=1)
    assert SUD(panel).iloc[-1, 0] == pytest.approx(expected)


def test_period_operators_never_emit_before_announcement(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    calendar = pd.date_range("2020-04-01", "2021-05-31", freq="B")
    pd.DataFrame({"trade_dt": calendar}).to_parquet(base / "calendar.parquet")
    reports = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31"])
    announcements = pd.to_datetime(["2020-04-30", "2020-08-14", "2020-10-30", "2021-03-31", "2021-04-30"])
    pd.DataFrame(
        {
            "dt": announcements,
            "report": reports,
            "symbol": ["A"] * 5,
            "factor": ["net_profit"] * 5,
            "value": [10.0, 12.0, 15.0, 20.0, 16.0],
        }
    ).to_parquet(base / "income.parquet")

    panel = PanelLoader(base, "2020-04-01", "2021-05-31", lookback_days=500).load("income__net_profit")
    yoy = YOY(panel)
    assert yoy.loc[:"2021-04-29", "A"].isna().all()
    assert yoy.loc["2021-04-30", "A"] == pytest.approx(0.6)


def test_weekend_announcement_becomes_visible_on_next_trade_day(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    calendar = pd.date_range("2022-04-01", "2023-05-12", freq="B")
    pd.DataFrame({"trade_dt": calendar}).to_parquet(base / "calendar.parquet")
    pd.DataFrame(
        {
            "dt": pd.to_datetime(["2022-04-29", "2022-08-12", "2022-10-28", "2023-03-31", "2023-05-06"]),
            "report": pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31"]),
            "symbol": ["A"] * 5,
            "factor": ["net_profit"] * 5,
            "value": [10.0, 12.0, 15.0, 20.0, 16.0],
        }
    ).to_parquet(base / "income.parquet")

    panel = PanelLoader(base, "2022-04-01", "2023-05-12").load("income__net_profit")
    yoy = YOY(panel)
    assert np.isnan(yoy.loc["2023-05-05", "A"])
    assert yoy.loc["2023-05-08", "A"] == pytest.approx(0.6)


def test_default_financial_history_is_long_enough_for_sud(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    calendar = pd.date_range("2022-01-03", "2023-05-31", freq="B")
    pd.DataFrame({"trade_dt": calendar}).to_parquet(base / "calendar.parquet")
    reports = pd.date_range("2020-03-31", periods=13, freq="QE")
    values = np.array([70, 75, 82, 90, 100, 108, 119, 131, 146, 158, 175, 193, 220], dtype=float)
    pd.DataFrame(
        {
            "dt": reports + pd.to_timedelta(45, unit="D"),
            "report": reports,
            "symbol": ["A"] * len(reports),
            "factor": ["net_profit"] * len(reports),
            "value": values,
        }
    ).to_parquet(base / "income.parquet")

    # The ordinary 400-day market-data warm-up starts in 2022, but SUD still
    # receives the older quarterly announcements through the bounded financial
    # history path.
    panel = PanelLoader(base, "2023-04-01", "2023-05-31").load("income__net_profit")
    assert np.isfinite(SUD(panel).loc["2023-05-15", "A"])


def test_loader_builds_citic_level_one_panel_and_excludes_hk_mirrors(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    calendar = pd.date_range("2022-01-03", "2022-01-10", freq="B")
    pd.DataFrame({"trade_dt": calendar}).to_parquet(base / "calendar.parquet")
    rows = [
        # Mainland and CIHK mirror rows are parallel in the verified remote table.
        ("A", "2020-01-01", "2022-01-05", "CI005001.WI", "中信一级"),
        ("A", "2020-01-01", "2022-01-05", "CIHK5001.WI", "中信一级"),
        ("A", "2022-01-06", "2262-04-11", "CI005002.WI", "中信一级"),
        ("A", "2022-01-06", "2262-04-11", "CIHK5002.WI", "中信一级"),
        ("B", "2020-01-01", "2262-04-11", "CI005001.WI", "中信一级"),
        ("B", "2020-01-01", "2262-04-11", "CIHK5001.WI", "中信一级"),
        ("C", "2020-01-01", "2262-04-11", "CI005101.WI", "中信二级"),
    ]
    pd.DataFrame(rows, columns=["symbol", "beg_date", "end_date", "indus_code", "level"]).assign(
        beg_date=lambda frame: pd.to_datetime(frame["beg_date"]),
        end_date=lambda frame: pd.to_datetime(frame["end_date"]),
    ).to_parquet(base / "industry_ci.parquet")

    industry = PanelLoader(base, "2022-01-03", "2022-01-10", lookback_days=0).load("industry")

    assert list(industry.columns) == ["A", "B"]
    assert industry.loc["2022-01-05", "A"] == "CI005001.WI"
    assert industry.loc["2022-01-06", "A"] == "CI005002.WI"
    assert industry.loc["2022-01-10", "B"] == "CI005001.WI"
    assert not industry.isin(["CIHK5001.WI", "CIHK5002.WI"]).any().any()
    assert industry.attrs == {
        "gf_industry_source": "industry_ci.parquet",
        "gf_industry_level": "中信一级",
        "gf_industry_value": "indus_code",
    }
    values = pd.DataFrame({"A": [1.0] * len(calendar), "B": [3.0] * len(calendar)}, index=calendar)
    proportions = IndusProp(values, industry)
    assert proportions.loc["2022-01-05", ["A", "B"]].tolist() == pytest.approx([0.25, 0.75])
    assert proportions.loc["2022-01-06", ["A", "B"]].tolist() == pytest.approx([1.0, 1.0])


def test_industry_proportion_and_yearly_change() -> None:
    reports = pd.date_range("2020-03-31", periods=5, freq="QE")
    panel = _financial_panel(
        {
            "A": [1, 2, 3, 4, 3],
            "B": [3, 3, 3, 3, 9],
            "C": [2, 2, 2, 2, 4],
            "D": [2, 2, 2, 2, 4],
        },
        reports,
    )
    industry = pd.DataFrame(
        [["bank", "bank", "tech", "tech"]] * len(panel), index=panel.index, columns=panel.columns
    )
    current = IndusProp(panel, industry).iloc[-1]
    assert current[["A", "B", "C", "D"]].tolist() == pytest.approx([0.25, 0.75, 0.5, 0.5])
    change = DeltaProp(panel, industry).iloc[-1]
    assert change[["A", "B", "C", "D"]].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_ratio_percentile_and_regression_compounds() -> None:
    date = pd.DatetimeIndex(["2024-01-02"])
    columns = list("ABCD")
    b = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=date, columns=columns)
    a = pd.DataFrame([[5.0, 7.5, 10.0, 12.0]], index=date, columns=columns)

    assert A2B(a, b).iloc[0].tolist() == pytest.approx([5.0, 3.75, 10 / 3, 3.0])
    percentile = PercentileA2B(a, b)
    assert percentile.iloc[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    residual = RegA2B(a, b).iloc[0]
    assert residual.mean() == pytest.approx(0.0, abs=1e-12)
    assert float((residual * (b.iloc[0] - b.iloc[0].mean())).sum()) == pytest.approx(0.0, abs=1e-12)


def test_delta_compound_operators() -> None:
    reports = pd.date_range("2020-03-31", periods=5, freq="QE")
    a = _financial_panel(
        {"A": [1, 1, 1, 1, 3], "B": [2, 2, 2, 2, 6], "C": [3, 3, 3, 3, 8], "D": [4, 4, 4, 4, 11]},
        reports,
    )
    b_financial = _financial_panel(
        {"A": [2, 2, 2, 2, 4], "B": [4, 4, 4, 4, 8], "C": [6, 6, 6, 6, 9], "D": [8, 8, 8, 8, 13]},
        reports,
    )
    denominator = pd.DataFrame([[2.0, 4.0, 5.0, 7.0]] * len(a), index=a.index, columns=a.columns)

    assert DeltaA2B(a, denominator).iloc[-1].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    expected_yoy_gap = YOY(a).iloc[-1] - YOY(b_financial).iloc[-1]
    pd.testing.assert_series_equal(YOYA_YOYB(a, b_financial).iloc[-1], expected_yoy_gap)
    residual = DeltaRegA2B(a, denominator).iloc[-1]
    assert residual.mean() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "expression",
    [
        "YOY(income__net_profit)",
        "SUD(cashflow__net_oper_cash_flow, 8)",
        "IndusProp(income__net_profit, industry)",
        "YOYA_YOYB(income__net_profit, income__revenue)",
        "YOYChange(income__net_profit, 4)",
        "YOYChange(income__net_profit, 1)",
        "CAGR(income__revenue, 12)",
        "CAGR(income__revenue, 1)",
    ],
)
def test_financial_dsl_contract_accepts_valid_expressions(expression: str) -> None:
    assert validate_dsl_expression(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "YOY(close)",
        "YOY(ts_mean(income__net_profit, 5))",
        "DeltaProp(income__net_profit, mv)",
        "YOYA_YOYB(income__net_profit, close)",
        "YOYChange(close, 4)",
        "YOYChange(income__net_profit, 0)",
        "CAGR(ts_mean(income__revenue, 5), 12)",
    ],
)
def test_financial_dsl_contract_rejects_ambiguous_inputs(expression: str) -> None:
    with pytest.raises(ValueError):
        validate_dsl_expression(expression)
