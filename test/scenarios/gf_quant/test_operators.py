from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.gf_quant.developer import coder as coder_module
from rdagent.scenarios.gf_quant.developer.coder import (
    DIRECT_FIELDS as CODER_FIELDS,
)
from rdagent.scenarios.gf_quant.developer.coder import (
    GFFactorCoder,
    describe_operators,
    validate_dsl_expression,
)
from scripts.gf_quant_remote_worker import (
    DIRECT_FIELDS as WORKER_FIELDS,
)
from scripts.gf_quant_remote_worker import (
    SafeExpression,
    _abs,
    _inverse,
    _log,
    _sign,
    _sqrt,
    clip,
    cs_demean,
    cs_industry_demean,
    cs_industry_rank,
    cs_industry_zscore,
    cs_scale,
    cs_zscore_mad,
    cs_zscore,
    run_request,
    ts_argmax,
    ts_argmin,
    ts_above_mean,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_ewm,
    ts_kurt,
    ts_mad,
    ts_max,
    ts_median,
    ts_min,
    ts_quantile,
    ts_rank,
    ts_residual,
    ts_skew,
    ts_slope,
    ts_sum,
    ts_downside_std,
    ts_upside_std,
    ts_zscore,
)


def _panel(values: dict[str, list[float]], periods: int = 6) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.date_range("2022-01-03", periods=periods, freq="D"))


def _column(frame: pd.DataFrame) -> list[float]:
    return frame.iloc[:, 0].tolist()


# --------------------------------------------------------------------------- #
# Catalog agreement between the local coder and the standalone remote worker.
# --------------------------------------------------------------------------- #


def test_coder_and_worker_operator_catalogs_agree() -> None:
    from scripts.gf_quant_remote_worker import OPERATOR_SPECS as worker_specs

    assert set(worker_specs) == set(coder_module.OPERATORS)
    for name, local in coder_module.OPERATORS.items():
        remote = worker_specs[name]
        assert (remote.name, remote.signature, remote.min_args, remote.max_args) == (
            local.name,
            local.signature,
            local.min_args,
            local.max_args,
        )
        # The two ArgRule classes live in different modules, so compare as data.
        assert [asdict(rule) for rule in remote.arguments] == [asdict(rule) for rule in local.arguments], name
    assert CODER_FIELDS == WORKER_FIELDS


def test_operator_groups_cover_the_catalog_exactly() -> None:
    grouped = [name for names in coder_module.OPERATOR_GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped)) == len(coder_module.OPERATORS)
    assert set(grouped) == set(coder_module.OPERATORS)


def test_describe_operators_lists_every_signature() -> None:
    text = describe_operators()
    for spec in coder_module.OPERATORS.values():
        assert spec.signature in text


def test_human_catalog_lists_every_operator() -> None:
    catalog = (Path(__file__).parents[3] / "docs" / "factor_operator_catalog.md").read_text(encoding="utf-8")
    assert "58 个函数算子" in catalog
    formal_catalog = catalog.split("## 建议新加的算子", 1)[0]
    for name in coder_module.OPERATORS:
        assert f"`{name}(" in formal_catalog, name


# --------------------------------------------------------------------------- #
# Operator semantics on small panels.
# --------------------------------------------------------------------------- #


def test_rolling_statistics_match_their_definitions() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert _column(ts_sum(panel, 3))[:2] == pytest.approx([np.nan, np.nan], nan_ok=True)
    assert _column(ts_sum(panel, 3))[2:] == pytest.approx([6.0, 9.0, 12.0, 15.0])
    assert _column(ts_min(panel, 3))[2:] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert _column(ts_max(panel, 3))[2:] == pytest.approx([3.0, 4.0, 5.0, 6.0])
    assert _column(ts_median(panel, 3))[2:] == pytest.approx([2.0, 3.0, 4.0, 5.0])


def test_rolling_higher_moments_need_the_stability_floor() -> None:
    symmetric = _panel({"A": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0]})
    assert ts_skew(symmetric, 5).iloc[:4, 0].tolist() == pytest.approx([np.nan, np.nan, 0.0, 0.0], nan_ok=True)
    skewed = _panel({"A": [1.0, 2.0, 3.0, 4.0, 20.0, 5.0]})
    assert ts_skew(skewed, 5).iloc[4, 0] == pytest.approx(2.1251, abs=1e-3)
    assert ts_kurt(skewed, 5).iloc[4, 0] == pytest.approx(4.6061, abs=1e-3)


def test_ts_quantile_interpolates_inside_the_window() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0, 10.0, 5.0, 6.0]})
    assert _column(ts_quantile(panel, 3, 0.5))[2:] == pytest.approx([2.0, 3.0, 5.0, 6.0])
    assert _column(ts_quantile(panel, 3, 0.75))[3] == pytest.approx(6.5)  # midpoint of 3 and 10
    assert _column(ts_quantile(panel, 3, 0.25))[2:] == pytest.approx([1.5, 2.5, 4.0, 5.5])
    assert _column(ts_quantile(panel, 3, 0.5))[:2] == pytest.approx([np.nan, np.nan], nan_ok=True)


def test_robust_and_regression_rolling_operators() -> None:
    panel = _panel({"A": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0], "B": [7.0] * 6})
    assert ts_mad(panel, 3).iloc[2, 0] == pytest.approx(1.0)
    assert ts_slope(panel[["B"]], 3).iloc[2:, 0].tolist() == pytest.approx([0.0] * 4)

    linear = _panel({"A": [1.0, 3.0, 5.0, 8.0, 9.0, np.nan]})
    assert ts_slope(linear, 3).iloc[2, 0] == pytest.approx(2.0)
    assert ts_residual(linear, 3).iloc[2, 0] == pytest.approx(0.0)
    assert ts_residual(linear, 3).iloc[3, 0] == pytest.approx(1.0 / 6.0)
    assert np.isnan(ts_residual(linear, 3).iloc[-1, 0])


def test_directional_volatility_and_above_mean_fraction() -> None:
    panel = _panel({"A": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]}, periods=6)
    assert ts_upside_std(panel, 6).iloc[-1, 0] == pytest.approx(1.0)
    assert ts_downside_std(panel, 6).iloc[-1, 0] == pytest.approx(1.0)
    assert ts_upside_std(panel, 5).iloc[-1, 0] != ts_upside_std(panel, 5).iloc[-1, 0]

    rising = _panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, periods=5)
    result = ts_above_mean(rising, 3)
    assert result.iloc[:4, 0].isna().all()
    assert result.iloc[-1, 0] == pytest.approx(1.0)


def test_ts_zscore_standardises_and_rejects_flat_windows() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [2.0] * 6})
    result = ts_zscore(panel, 3)
    assert result.iloc[2:, 0].tolist() == pytest.approx([1.0] * 4)
    assert result["B"].isna().all()


def test_ts_rank_uses_average_ties_like_cs_rank() -> None:
    panel = _panel({"A": [5.0, 1.0, 4.0], "B": [1.0, 3.0, 3.0], "C": [2.0, 2.0, 7.0]}, periods=3)
    result = ts_rank(panel, 3)
    # With a window covering the whole panel, the trailing-window percentile of
    # each column equals its own time-series rank, i.e. the cs_rank definition.
    expected = panel.rank(pct=True, method="average") - 0.5
    pd.testing.assert_series_equal(result.loc["2022-01-05"], expected.loc["2022-01-05"])
    assert result.loc["2022-01-05", "C"] == pytest.approx(0.5)  # window maximum
    assert result.loc["2022-01-05", "B"] == pytest.approx(2.5 / 3 - 0.5)  # tied pair shares rank 2.5 of 3


def test_ts_rank_does_not_rank_a_missing_current_value() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0, np.nan]}, periods=4)
    assert np.isnan(ts_rank(panel, 3).iloc[-1, 0])


def test_ts_decay_linear_weights_recent_days_most() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    result = ts_decay_linear(panel, 3)
    assert result.iloc[2, 0] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6.0)
    assert result.iloc[5, 0] == pytest.approx((4 * 1 + 5 * 2 + 6 * 3) / 6.0)


def test_ts_ewm_matches_a_span_of_three() -> None:
    panel = _panel({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    result = ts_ewm(panel, 3)
    assert np.isnan(result.iloc[:2, 0]).all()
    assert result.iloc[2, 0] == pytest.approx(2.4285714, abs=1e-6)


def test_ts_argmax_and_ts_argmin_count_days_since_the_extreme() -> None:
    panel = _panel({"A": [5.0, 1.0, 4.0, 2.0, 6.0, 3.0]})
    assert _column(ts_argmax(panel, 3))[2:] == pytest.approx([2.0, 1.0, 0.0, 1.0])
    assert _column(ts_argmin(panel, 3))[2:] == pytest.approx([1.0, 2.0, 1.0, 2.0])
    assert np.isnan(ts_argmax(panel, 3).iloc[:2, 0]).all()
    ties = _panel({"A": [5.0, 1.0, 5.0], "B": [1.0, 5.0, 1.0]}, periods=3)
    assert ts_argmax(ties, 3).iloc[-1, 0] == pytest.approx(0.0)
    assert ts_argmin(ties, 3).iloc[-1, 1] == pytest.approx(0.0)


def test_two_panel_rolling_statistics_align_symbols() -> None:
    dates = pd.date_range("2022-01-03", periods=6, freq="D")
    left = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]}, index=dates)
    right = pd.DataFrame({"A": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0], "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates)
    assert ts_corr(left, right, 3).iloc[2:, 0].tolist() == pytest.approx([1.0] * 4)
    assert ts_cov(left, right, 3).iloc[2, 0] == pytest.approx(2.0)
    disjoint = pd.DataFrame({"C": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates)
    assert list(ts_corr(left, disjoint, 3).columns) == []


def test_cross_sectional_transforms_are_row_wise() -> None:
    panel = _panel({"A": [1.0, 4.0, 3.0], "B": [4.0, 1.0, 3.0], "C": [3.0, 3.0, 3.0]}, periods=3)
    zscore = cs_zscore(panel)
    assert zscore.iloc[0, 0] == pytest.approx(-1.0911, abs=1e-4)
    assert np.isfinite(zscore.iloc[2]).sum() == 0  # a flat cross-section has no z-score
    assert cs_demean(panel).sum(axis=1).abs().max() == pytest.approx(0.0, abs=1e-12)
    scaled = cs_scale(panel)
    assert scaled.iloc[0].tolist() == pytest.approx([1 / 8, 4 / 8, 3 / 8])
    signed = cs_scale(_panel({"A": [-1.0, 4.0, 3.0], "B": [4.0, 1.0, 3.0], "C": [3.0, 3.0, 3.0]}, periods=3))
    assert signed.iloc[0, 0] == pytest.approx(-1 / 8)


def test_industry_and_mad_cross_sectional_transforms() -> None:
    dates = pd.date_range("2022-01-03", periods=2, freq="D")
    values = pd.DataFrame([[1.0, 3.0, 10.0, 20.0], [2.0, 2.0, 5.0, np.nan]], index=dates, columns=list("ABCD"))
    industry = pd.DataFrame([["bank", "bank", "tech", "solo"], ["bank", "bank", "tech", "tech"]], index=dates, columns=list("ABCD"))
    rank = cs_industry_rank(values, industry)
    assert rank.iloc[0].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.5])
    demeaned = cs_industry_demean(values, industry)
    assert demeaned.iloc[0].tolist() == pytest.approx([-1.0, 1.0, 0.0, 0.0])
    zscore = cs_industry_zscore(values, industry)
    assert zscore.iloc[0, :2].tolist() == pytest.approx([-2**-0.5, 2**-0.5])
    assert zscore.iloc[0, 2:].isna().all()

    robust = cs_zscore_mad(pd.DataFrame([[1.0, 2.0, 100.0], [3.0, 3.0, 3.0]], index=dates, columns=list("ABC")))
    assert robust.iloc[0].tolist() == pytest.approx([-1.0, 0.0, 98.0])
    assert robust.iloc[1].isna().all()


def test_elementwise_math_is_total_and_nan_preserving() -> None:
    panel = _panel({"A": [1.0, np.e, -1.0, 0.0, 4.0, -4.0]})
    assert _column(_log(panel))[:2] == pytest.approx([0.0, 1.0])
    assert _column(_sqrt(panel))[:1] == pytest.approx([1.0])
    assert _column(_sqrt(panel))[4] == pytest.approx(2.0)
    assert np.isnan(_log(panel).iloc[2:4, 0]).all()  # logarithm needs a positive input
    assert np.isnan(_sqrt(panel).iloc[2, 0])
    assert _sqrt(panel).iloc[3, 0] == pytest.approx(0.0)
    assert _column(_inverse(panel))[:2] == pytest.approx([1.0, 1 / np.e])
    assert np.isnan(_inverse(panel).iloc[3, 0])  # 1 / 0 becomes NaN, never inf
    assert _column(_sign(panel)) == pytest.approx([1.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    assert _column(_abs(panel)) == pytest.approx([1.0, np.e, 1.0, 0.0, 4.0, 4.0])
    assert _column(clip(panel, -0.5, 2.0)) == pytest.approx([1.0, 2.0, -0.5, 0.0, 2.0, -0.5])


# --------------------------------------------------------------------------- #
# Argument validation, identical locally and remotely.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expression",
    [
        "ts_zscore(adjclose, 20)",
        "ts_quantile(ts_mean(adjclose, 5), 60, 0.75)",
        "ts_rank(ts_mean(adjclose, 5), 60)",
        "cs_scale(abs(income__net_profit))",
        "ts_corr(ts_mean(adjclose, 5), volume, 20)",
        "winsorize(pct_change(adjclose, 5), 0.02)",
        "inverse(cs_demean(mv))",
        "log(sqrt(amount)) + sign(volume)",
        "ts_mad(adjclose, 20) + ts_slope(adjclose, 20)",
        "ts_residual(adjclose, 20) + ts_above_mean(adjclose, 20)",
        "ts_upside_std(pct_change(adjclose, 2), 20) - ts_downside_std(pct_change(adjclose, 2), 20)",
        "cs_industry_zscore(mv, industry) + cs_zscore_mad(pb)",
        "clip(cs_industry_demean(mv, industry), -3, 3)",
    ],
)
def test_valid_new_operator_expressions_are_accepted(expression: str) -> None:
    fields = validate_dsl_expression(expression)
    assert fields
    assert SafeExpression(expression).fields == list(fields)  # both gates agree


@pytest.mark.parametrize(
    "expression",
    [
        "ts_zscore(adjclose)",  # missing window
        "ts_zscore(adjclose, 20, 5)",  # too many arguments
        "ts_corr(adjclose, 20)",  # missing second panel
        "ts_mad(adjclose, 2)",  # robust windows need at least three dates
        "ts_zscore(adjclose, adjclose)",  # window must be a literal
        "ts_zscore(adjclose, 0)",  # window must be positive
        "ts_zscore(adjclose, -5)",
        "ts_zscore(adjclose, 5.5)",  # window must be a whole number
        "ts_quantile(adjclose, 20, 0.0)",  # quantile must stay inside (0, 1)
        "ts_quantile(adjclose, 20, 1.0)",
        "ts_quantile(adjclose, 20)",  # missing quantile
        "ts_zscore(adjclose, n)",  # window must be a constant, not a name
        "winsorize(adjclose, 0.5)",  # quantile must stay below one half
        "winsorize(adjclose, 0.0)",
        "winsorize(adjclose, p=0.01)",  # keyword arguments stay unsupported
        "cs_rank",  # operator referenced without a call
        "ts_zscore(close_to_maturity, 20)",  # unknown field
        "clip(adjclose, 1, 1)",
        "clip(adjclose, -1, lo)",
        "cs_industry_rank(mv, pb)",
    ],
)
def test_invalid_expressions_are_rejected_locally(expression: str) -> None:
    with pytest.raises(ValueError):
        validate_dsl_expression(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "ts_zscore(adjclose, 0)",
        "ts_corr(adjclose, volume)",
        "winsorize(adjclose, 0.5)",
        "unknown_operator(adjclose, 5)",
    ],
)
def test_worker_enforces_the_same_argument_rules(expression: str) -> None:
    with pytest.raises(ValueError):
        SafeExpression(expression)


# --------------------------------------------------------------------------- #
# End-to-end evaluation through the remote expression evaluator.
# --------------------------------------------------------------------------- #


class _StaticLoader:
    def __init__(self, panels: dict[str, pd.DataFrame]) -> None:
        self.panels = panels

    def load(self, name: str) -> pd.DataFrame:
        if name not in self.panels:
            raise ValueError(f"Unsupported field: {name}")
        return self.panels[name]


def test_safe_expression_evaluates_a_composed_expression() -> None:
    dates = pd.date_range("2022-01-03", periods=30, freq="D")
    symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
    trend = np.arange(30, dtype=float)
    close = pd.DataFrame({symbol: 10.0 + trend for symbol in symbols}, index=dates)
    # Each symbol relates to the common trend differently, so the rolling
    # price/volume correlation separates them cross-sectionally.
    volume = pd.DataFrame(
        {"000001.SZ": 1000 + 10 * trend, "000002.SZ": 1000 - 10 * trend, "000003.SZ": 1000 + 30 * np.sin(trend / 3)},
        index=dates,
    )
    loader = _StaticLoader({"adjclose": close, "volume": volume})

    factor = SafeExpression("cs_zscore(ts_corr(ts_mean(adjclose, 5), ts_mean(volume, 5), 10))").evaluate(loader)

    assert factor.shape == close.shape
    assert np.isnan(factor.iloc[:3]).all().all()  # warm-up windows stay NaN
    assert np.isfinite(factor.iloc[20:]).all().all()
    late = factor.iloc[20:]
    assert (late["000001.SZ"] > late["000002.SZ"]).all()  # co-movement ranks above divergence
    assert 0.0 <= late["000003.SZ"].abs().max() < late["000001.SZ"].abs().max()


def test_coder_develops_a_task_using_the_expanded_operators() -> None:
    task = FactorTask(
        factor_name="volume_price_corr",
        factor_description="[Volume price] rolling correlation of price and volume",
        factor_formulation="cs_zscore(ts_corr(ts_mean(adjclose, 5), ts_mean(volume, 5), 20))",
        variables={"adjclose": "adjusted close", "volume": "traded volume"},
    )
    exp = SimpleNamespace(sub_tasks=[task], sub_workspace_list=[None])

    result = GFFactorCoder(SimpleNamespace()).develop(exp)

    assert result.sub_tasks[0].factor_implementation is True
    assert result.sub_tasks[0].dsl_fields == ("adjclose", "volume")


def _write_parquet_base(base: Path) -> None:
    """Build a tiny G:/base look-alike whose symbols move in opposite directions."""
    base.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2022-01-03", periods=60, freq="D")
    trend = np.arange(60, dtype=float)
    prices = {"000001.SZ": 10.0 + trend, "000002.SZ": 60.0 - 0.5 * trend}
    volumes = {"000001.SZ": 1000.0 + 10 * trend, "000002.SZ": 1000.0 + 5 * trend}
    for field, paths in (("adjclose", prices), ("volume", volumes)):
        rows = [
            {"dt": day, "symbol": symbol, "update_time": day, "value": value}
            for symbol, path in paths.items()
            for day, value in zip(dates, path)
        ]
        pd.DataFrame(rows).to_parquet(base / f"{field}.parquet")
    pd.DataFrame({"trade_dt": dates}).to_parquet(base / "calendar.parquet")


def test_run_request_evaluates_the_expanded_operators_end_to_end(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _write_parquet_base(base)
    artifact = tmp_path / "artifacts" / "volume_price_corr.parquet"
    request = {
        "name": "volume_price_corr",
        "family": "price_volume",
        "expression": "cs_zscore(ts_corr(ts_mean(adjclose, 5), ts_mean(volume, 5), 10))",
        "base_path": str(base),
        "lookback_days": 60,
        "artifact_path": str(artifact),
        "evaluation": {
            "start": "2022-01-20",
            "end": "2022-03-01",
            "label_horizon": 5,
            "quantile": 0.5,  # a two-symbol fixture needs both legs populated
            "min_cross_section": 2,
            "sample_step": 1,
        },
    }
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    result = run_request(request_path, result_path)

    assert result["fields"] == ["adjclose", "volume"]
    assert result["artifact_checksum"] == sha256(artifact.read_bytes()).hexdigest()
    panel = pd.read_parquet(artifact)
    assert panel.attrs == {}
    assert panel.shape[1] == 2
    assert panel.index.min() == pd.Timestamp("2022-01-20")
    assert panel.notna().to_numpy().any()
    metrics = result["metrics"]
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["observations"] > 0
    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["name"] == "volume_price_corr"


def test_coder_reports_the_operator_in_the_error_message() -> None:
    task = FactorTask(
        factor_name="broken",
        factor_description="uses an unsupported operator",
        factor_formulation="unknown_mad(adjclose, 20)",
        variables={"adjclose": "adjusted close"},
    )
    exp = SimpleNamespace(sub_tasks=[task], sub_workspace_list=[None])

    with pytest.raises(FactorEmptyError, match="unknown_mad"):
        GFFactorCoder(SimpleNamespace()).develop(exp)
