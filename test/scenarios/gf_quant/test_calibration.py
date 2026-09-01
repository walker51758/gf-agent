"""Unit tests for the family-aware benchmark calibration module."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.benchmark import (
    DEFAULT_CLUSTER_THRESHOLD,
    BenchmarkCluster,
    BenchmarkFactorResult,
    Distribution,
    ExclusionRule,
    FamilyCalibration,
    FamilyStandards,
    QuantilePolicy,
    StandardsValidationError,
    annual_gate_failures,
    benchmark_percentile,
    calibrate_family,
    calibration_to_document,
    cluster_balanced_quantile,
    cluster_by_correlation,
    evaluate_against_calibration,
    inverse_cluster_weights,
    load_family_calibrations,
    load_family_standards,
    orientation_sign,
    profile_correlation,
    profile_metrics,
    representative_sample,
    weighted_quantile,
)
from rdagent.scenarios.gf_quant.service import (
    AcceptancePolicy,
    EvaluationResult,
    FactorAcceptanceService,
    FactorCandidate,
)
from rdagent.scenarios.gf_quant.registry import FactorRegistry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _result(
    name: str,
    *,
    rank_ic: float = 0.03,
    rank_icir: float = 0.30,
    coverage: float = 0.98,
    turnover: float = 0.30,
    observations: int = 900,
    spread: dict[str, float] | None = None,
    yearly: dict[str, float] | None = None,
    family: str = "price_volume",
    status: str = "ok",
    exclusion_reason: str | None = None,
    source_dir: str = "G:/pool",
) -> BenchmarkFactorResult:
    profile = spread if spread is not None else {
        f"2020-{1 + index // 28:02d}-{1 + index % 28:02d}": 0.001 * (1 + 0.1 * (index % 5)) for index in range(300)
    }
    metrics = {
        "Rank_IC": rank_ic,
        "Rank_ICIR": rank_icir,
        "coverage": coverage,
        "turnover": turnover,
        "observations": observations,
    }
    if status != "ok":
        metrics = {}
    metrics_by_year = {
        year: {"Rank_IC": value, "observations": 240}
        for year, value in (yearly or {"2020": rank_ic, "2021": rank_ic}).items()
    }
    # the worker embeds the yearly breakdown inside the aggregate metrics
    metrics = {**metrics, "metrics_by_year": metrics_by_year}
    return BenchmarkFactorResult(
        name=name,
        family=family,
        source_dir=source_dir,
        file_name=f"{name}.parquet",
        metrics=metrics,
        metrics_by_year=metrics_by_year,
        spread_profile=profile,
        rank_ic_profile={key: rank_ic for key in profile},
        status=status,
        exclusion_reason=exclusion_reason,
    )


def _profile(values: dict[str, float]) -> dict[str, float]:
    return dict(sorted(values.items()))


def _dates(count: int, start: str = "2020-01-01") -> list[str]:
    return [pd.Timestamp(start) + pd.Timedelta(days=offset) for offset in range(count)]


def _iso(dates: list[pd.Timestamp]) -> list[str]:
    return [date.strftime("%Y-%m-%d") for date in dates]


def _varying_spread(seed: int, *, count: int = 300, base: float = 0.001) -> dict[str, float]:
    """Two clearly distinct daily shapes so clustering stays deterministic.

    Constant profiles are numerically degenerate (near-zero variance makes the
    Pearson correlation meaningless), so fixtures always use one of two shapes:
    even seeds ramp, odd seeds zigzag.  Same-shape factors cluster together,
    different shapes stay apart.
    """
    dates = _iso(_dates(count))
    if seed % 2 == 0:
        return {date: base * (1.0 + 0.01 * index) for index, date in enumerate(dates)}
    return {date: base * (1.0 + 0.02 * (index % 2)) for index, date in enumerate(dates)}


# ---------------------------------------------------------------------------
# weighted / cluster-balanced quantiles
# ---------------------------------------------------------------------------


def test_weighted_quantile_interpolates_linearly() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    weights = [1.0, 1.0, 1.0, 1.0]
    assert weighted_quantile(values, weights, 0.5) == pytest.approx(2.5)
    assert weighted_quantile(values, weights, 0.0) == 1.0
    assert weighted_quantile(values, weights, 1.0) == 4.0
    # midpoint cumulative positions are (0.125, 0.375, 0.625, 0.875)
    assert weighted_quantile(values, weights, 0.375) == pytest.approx(2.0)
    assert weighted_quantile([10.0], [5.0], 0.5) == 10.0


def test_weighted_quantile_respects_weights_and_rejects_bad_input() -> None:
    # equal weights: P50 is the interpolated median of the two observations
    assert weighted_quantile([1.0, 10.0], [1.0, 1.0], 0.5) == pytest.approx(5.5)
    # a heavy tail observation pulls the quantile towards itself
    assert weighted_quantile([1.0, 10.0], [0.1, 0.9], 0.5) > 8.0
    assert weighted_quantile([1.0, 10.0], [0.1, 0.9], 0.05) == 1.0
    assert weighted_quantile([1.0, 10.0], [0.1, 0.9], 0.55) == 10.0
    with pytest.raises(ValueError, match="one weight per value"):
        weighted_quantile([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(ValueError, match="positive finite weights"):
        weighted_quantile([1.0], [0.0], 0.5)
    with pytest.raises(ValueError, match="q must be"):
        weighted_quantile([1.0], [1.0], 1.5)


def test_cluster_balanced_quantile_is_invariant_to_cluster_size() -> None:
    values = {"a": 1.0, "a2": 1.5, "b": 9.0}
    dates = _iso(_dates(200))
    ramp = {date: 1.0 + 0.01 * index for index, date in enumerate(dates)}
    clusters = cluster_by_correlation(
        [
            _result("a", spread=_profile(ramp)),
            _result("a2", spread=_profile({date: -value for date, value in ramp.items()})),
            _result("b", spread=_profile({date: 0.01 * (index % 2) for index, date in enumerate(dates)})),
        ],
        threshold=0.9,
        min_overlap=10,
    )
    assert len(clusters) == 2
    weights = inverse_cluster_weights(clusters)
    assert weights["a"] == pytest.approx(weights["a2"])
    assert weights["a"] + weights["a2"] == pytest.approx(0.5)
    assert weights["b"] == pytest.approx(0.5)
    # the singleton cluster contributes half of the mass, so the balanced P50 is
    # far above the unweighted median (1.5) of the three observations
    assert cluster_balanced_quantile(values, clusters, 0.5) == pytest.approx(4.0)
    assert cluster_balanced_quantile(values, clusters, 0.5) > 2.0
    assert representative_sample(clusters) == ("a", "b")


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def _ramp(seed: float, count: int = 200) -> dict[str, float]:
    return {date: seed + 0.01 * index for index, date in enumerate(_iso(_dates(count)))}


def _zigzag(count: int = 200) -> dict[str, float]:
    return {date: 0.01 * (index % 2) for index, date in enumerate(_iso(_dates(count)))}


def test_clustering_groups_correlated_factors_and_is_deterministic() -> None:
    results = [
        _result("negated", spread={key: -value for key, value in _ramp(1.0).items()}),
        _result("twin_b", spread=_ramp(1.0 + 1e-9)),
        _result("twin_a", spread=_ramp(1.0)),
        _result("other", spread=_zigzag()),
    ]
    first = cluster_by_correlation(results, threshold=0.9, min_overlap=50)
    second = cluster_by_correlation(list(reversed(results)), threshold=0.9, min_overlap=50)
    assert [cluster.members for cluster in first] == [cluster.members for cluster in second]
    assert first[0].members == ("negated", "twin_a", "twin_b")  # sorted inside the cluster
    assert first[1].members == ("other",)
    assert first[0].representative == "negated"  # lexicographic, deterministic


def test_clustering_requires_minimum_overlap_and_valid_profile_kind() -> None:
    left = _result("left", spread=_profile({date: 1.0 for date in _iso(_dates(50))}))
    right = _result("right", spread=_profile({date: 1.0 for date in _iso(_dates(200))}))
    assert cluster_by_correlation([left, right], threshold=0.9, min_overlap=100) == [
        BenchmarkCluster(cluster_id=0, members=("left",), representative="left"),
        BenchmarkCluster(cluster_id=1, members=("right",), representative="right"),
    ]
    with pytest.raises(ValueError, match="Unsupported clustering profile"):
        cluster_by_correlation([left, right], threshold=0.9, min_overlap=10, profile_kind="values")


def test_profile_correlation_uses_absolute_overlap_and_handles_degenerate_series() -> None:
    dates = _iso(_dates(10))
    left = _profile({date: float(index) for index, date in enumerate(dates)})
    right = _profile({date: -2.0 * index for index, date in enumerate(dates)})
    assert right is not None
    assert profile_correlation(left, right, min_overlap=5) == pytest.approx(-1.0)
    assert profile_correlation(left, {}, min_overlap=5) is None
    flat = _profile({date: 1.0 for date in dates})
    assert profile_correlation(left, flat, min_overlap=5) is None  # zero variance


# ---------------------------------------------------------------------------
# sign handling
# ---------------------------------------------------------------------------


def test_orientation_sign_and_direction_adjusted_profile_metrics() -> None:
    assert orientation_sign(-0.04) == -1
    assert orientation_sign(0.0, [1.0, -2.0]) == -1
    assert orientation_sign(None, [0.0, 0.0]) == 1

    dates = _iso(_dates(300))
    down = _profile({date: -0.001 for date in dates})
    metrics = profile_metrics(down, rank_ic=-0.05)
    assert metrics is not None and metrics.direction == -1
    assert metrics.annualized_return == pytest.approx(0.252, rel=1e-6)
    assert metrics.max_drawdown_abs == pytest.approx(0.0)  # monotonic after re-signing
    assert metrics.sharpe > 0

    flipped = profile_metrics(down, rank_ic=0.05)
    assert flipped is not None and flipped.direction == 1
    assert flipped.annualized_return == pytest.approx(-0.252, rel=1e-6)  # sign-adjusted, not ignored

    assert profile_metrics({}, rank_ic=0.1) is None
    assert profile_metrics(None, rank_ic=0.1) is None


def test_benchmark_percentile_handles_ties_and_missing_values() -> None:
    values = [0.1, 0.2, 0.2, 0.4]
    assert benchmark_percentile(values, 0.2) == pytest.approx(0.5)  # 1 below + 2 tied, halved
    assert benchmark_percentile(values, 0.05) == 0.0
    assert benchmark_percentile(values, 0.5) == 1.0
    assert benchmark_percentile(values, None) is None
    assert benchmark_percentile([], 0.2) is None


# ---------------------------------------------------------------------------
# exclusions
# ---------------------------------------------------------------------------


def test_exclusions_are_explicit_and_complete() -> None:
    rule = ExclusionRule(min_observations=100, min_valid_years=2, min_profile_observations=50)
    results = [
        _result("healthy"),
        _result("bad_ic", rank_ic=float("nan")),
        _result("few_obs", observations=50),
        _result("one_year", yearly={"2020": 0.03}),
        _result("no_profile", spread={}),
        _result("thin_profile", spread=_profile({date: 0.001 for date in _iso(_dates(20))})),
        _result("worker_failed", status="excluded", exclusion_reason="worker_error: ValueError"),
    ]
    valid, exclusions = rule.apply(results)
    assert [item.name for item in valid] == ["healthy"]
    reasons = {item.name: item.reason for item in exclusions}
    assert reasons == {
        "bad_ic": "nonfinite_core_metrics",
        "few_obs": "insufficient_observations",
        "one_year": "insufficient_valid_years",
        "no_profile": "missing_profile",
        "thin_profile": "insufficient_profile_observations",
        "worker_failed": "worker_error: ValueError",
    }
    assert all(item.detail for item in exclusions)


def test_calibration_refuses_to_invent_standards_from_an_empty_pool() -> None:
    with pytest.raises(StandardsValidationError, match="no valid benchmark factors"):
        calibrate_family([_result("excluded", status="excluded", exclusion_reason="x")], family="price_volume")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_calibrate_family_produces_family_specific_standards() -> None:
    def pool(
        base_ic: float, base_turnover: float, coverage: float, family: str = "price_volume"
    ) -> list[BenchmarkFactorResult]:
        return [
            _result(
                f"f{index}",
                family=family,
                rank_ic=base_ic + 0.01 * index,
                rank_icir=base_ic * 10 + index,
                turnover=base_turnover,
                coverage=coverage,
                observations=800 + index,
                spread=_varying_spread(index, base=base_ic * 0.1),
                yearly={year: base_ic + 0.01 * index for year in ("2020", "2021", "2022", "2023")},
            )
            for index in range(12)
        ]

    pv = calibrate_family(pool(0.03, 0.40, 0.97), family="price_volume", min_overlap=50)
    fundamental = calibrate_family(pool(0.01, 0.02, 0.99, family="fundamental"), family="fundamental", min_overlap=50)

    assert pv.standards.min_abs_rank_ic > fundamental.standards.min_abs_rank_ic
    assert pv.standards.max_turnover > fundamental.standards.max_turnover
    assert pv.standards.min_coverage != fundamental.standards.min_coverage
    assert pv.standards.annual_min_valid_years == 4
    assert pv.standards.min_abs_rank_icir is not None
    assert pv.standards.min_direction_adjusted_sharpe is not None
    # cross-family contamination is rejected loudly
    with pytest.raises(ValueError, match="other families"):
        calibrate_family(pool(0.03, 0.4, 0.97), family="fundamental", min_overlap=50)  # wrong-family pool


def test_standards_are_skipped_when_the_sample_is_too_small() -> None:
    pool = [
        _result(
            f"f{index}",
            rank_ic=0.05,
            rank_icir=float("nan") if index < 8 else 0.4,
            spread={date: 0.001 for date in _iso(_dates(200))},
        )
        for index in range(12)
    ]
    calibration = calibrate_family(pool, family="price_volume", min_overlap=50)
    assert calibration.standards.min_abs_rank_icir is None  # 8 finite values < MIN_SAMPLE_FOR_STANDARD
    assert "abs_rank_icir" in calibration.sample["standards_sample_sizes"]


# ---------------------------------------------------------------------------
# persistence and validation
# ---------------------------------------------------------------------------


def _document(tmp_path) -> tuple[dict, object]:
    pool = [
        _result(
            f"pv{index}",
            rank_ic=0.03 + 0.005 * index,
            turnover=0.3 + 0.01 * index,
            spread=_varying_spread(index),
            yearly={year: 0.03 + 0.005 * index for year in ("2020", "2021", "2022", "2023")},
        )
        for index in range(12)
    ] + [
        _result(
            f"fund{index}",
            family="fundamental",
            rank_ic=0.01 + 0.001 * index,
            turnover=0.02,
            spread=_varying_spread(index, base=0.0005),
            yearly={year: 0.01 for year in ("2020", "2021", "2022", "2023")},
        )
        for index in range(12)
    ]
    calibrations = {
        family: calibrate_family(
            [item for item in pool if item.family == family],
            family=family,
            min_overlap=50,
        )
        for family in ("price_volume", "fundamental")
    }
    document = calibration_to_document(calibrations, generated_at="2026-08-30T00:00:00Z", evaluation={"start": "2020-01-01"})
    path = tmp_path / "standards.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return document, path


def test_roundtrip_through_json_and_validation_failures(tmp_path) -> None:
    document, path = _document(tmp_path)

    loaded = load_family_calibrations(path)
    assert set(loaded) == {"price_volume", "fundamental"}
    assert loaded["price_volume"].standards.min_abs_rank_ic == pytest.approx(
        document["families"]["price_volume"]["standards"]["min_abs_rank_ic"]
    )
    standards = load_family_standards(path)
    assert isinstance(standards["fundamental"], FamilyStandards)
    assert standards["fundamental"].family == "fundamental"

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(StandardsValidationError, match="not valid JSON"):
        load_family_calibrations(corrupt)

    missing = tmp_path / "missing.json"
    with pytest.raises(StandardsValidationError, match="Cannot read"):
        load_family_calibrations(missing)

    broken = json.loads(json.dumps(document))
    del broken["families"]["fundamental"]
    broken_path = tmp_path / "broken.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(StandardsValidationError, match="fundamental"):
        load_family_calibrations(broken_path)

    bad_version = dict(document)
    bad_version["schema_version"] = "9.9"
    version_path = tmp_path / "version.json"
    version_path.write_text(json.dumps(bad_version), encoding="utf-8")
    with pytest.raises(StandardsValidationError, match="schema_version"):
        load_family_calibrations(version_path)

    bad_number = json.loads(json.dumps(document))
    bad_number["families"]["price_volume"]["standards"]["min_abs_rank_ic"] = "high"
    number_path = tmp_path / "number.json"
    number_path.write_text(json.dumps(bad_number), encoding="utf-8")
    with pytest.raises(StandardsValidationError, match="min_abs_rank_ic"):
        load_family_calibrations(number_path)

    bad_clusters = json.loads(json.dumps(document))
    bad_clusters["families"]["price_volume"]["clusters"][0]["members"] = []
    cluster_path = tmp_path / "cluster.json"
    cluster_path.write_text(json.dumps(bad_clusters), encoding="utf-8")
    with pytest.raises(StandardsValidationError, match="empty"):
        load_family_calibrations(cluster_path)

    shared = json.loads(json.dumps(document))
    shared["families"]["price_volume"]["clusters"][0]["members"] = ["pv0", "pv1"]
    shared["families"]["price_volume"]["clusters"][1]["members"] = ["pv1"]
    shared_path = tmp_path / "shared.json"
    shared_path.write_text(json.dumps(shared), encoding="utf-8")
    with pytest.raises(StandardsValidationError, match="two clusters"):
        load_family_calibrations(shared_path)


def test_distribution_payload_validation(tmp_path) -> None:
    distribution = Distribution("abs_rank_ic", "quality_floor", 0.5, (0.1, 0.2))
    assert Distribution.from_payload(distribution.to_payload()).values == (0.1, 0.2)
    with pytest.raises(StandardsValidationError, match="non-empty list"):
        Distribution.from_payload({"metric": "x", "role": "r", "quantile": 0.5, "values": []})
    with pytest.raises(StandardsValidationError, match="non-finite"):
        Distribution.from_payload({"metric": "x", "role": "r", "quantile": 0.5, "values": [1.0, "NaN"]})
    with pytest.raises(StandardsValidationError, match="invalid quantile"):
        Distribution.from_payload({"metric": "x", "role": "r", "quantile": 2.0, "values": [1.0]})


def test_family_standards_reject_unknown_family_and_bad_integers() -> None:
    with pytest.raises(StandardsValidationError, match="unsupported standards family"):
        FamilyStandards.from_payload({"family": "other", "min_abs_rank_ic": 0.02})
    payload = FamilyStandards(
        family="price_volume",
        min_abs_rank_ic=0.02,
        min_coverage=0.9,
        max_turnover=0.8,
        max_abs_drawdown=0.3,
        min_observations=400,
        annual_min_valid_years=4,
        annual_min_matching_sign_years=2,
    ).to_payload()
    assert FamilyStandards.from_payload(payload) == FamilyStandards(
        family="price_volume",
        min_abs_rank_ic=0.02,
        min_coverage=0.9,
        max_turnover=0.8,
        max_abs_drawdown=0.3,
        min_observations=400,
        annual_min_valid_years=4,
        annual_min_matching_sign_years=2,
    )
    for key, value in (("annual_min_valid_years", 2.5), ("min_observations", 0)):
        broken = dict(payload)
        broken[key] = value
        with pytest.raises(StandardsValidationError):
            FamilyStandards.from_payload(broken)


def test_quantile_policy_defaults_and_validation() -> None:
    policy = QuantilePolicy.from_payload(None)
    assert policy.quality_floor == 0.5 and policy.coverage_floor == 0.25 and policy.risk_ceiling == 0.75
    with pytest.raises(StandardsValidationError, match="quality_floor_quantile"):
        QuantilePolicy.from_payload({"quality_floor_quantile": 1.5})


# ---------------------------------------------------------------------------
# candidate gating
# ---------------------------------------------------------------------------


def _calibration_for(standards: FamilyStandards, values: dict[str, tuple[float, ...]]) -> FamilyCalibration:
    distributions = tuple(
        Distribution(metric, "quality_floor", 0.5, values.get(metric, (0.0, 1.0)))
        for metric in (
            "abs_rank_ic",
            "abs_rank_icir",
            "coverage",
            "turnover",
            "observations",
            "direction_adjusted_annualized_return",
            "direction_adjusted_sharpe",
            "abs_drawdown",
        )
    )
    return FamilyCalibration(
        family=standards.family,
        standards=standards,
        clusters=(),
        distributions=distributions,
        exclusions=(),
        cluster_profile="spread",
        correlation_threshold=DEFAULT_CLUSTER_THRESHOLD,
        min_overlap=100,
        sample={},
        quantile_policy=QuantilePolicy(),
    )


def _candidate_metrics(rank_ic: float) -> dict:
    metrics = {
        "Rank_IC": rank_ic,
        "Rank_ICIR": rank_ic * 8,
        "coverage": 0.99,
        "turnover": 0.2,
        "observations": 960,
        "metrics_by_year": {
            "2020": {"Rank_IC": rank_ic},
            "2021": {"Rank_IC": rank_ic},
            "2022": {"Rank_IC": rank_ic},
            "2023": {"Rank_IC": rank_ic},
        },
    }
    return metrics


def test_candidate_gate_is_direction_invariant() -> None:
    standards = FamilyStandards(
        family="price_volume",
        min_abs_rank_ic=0.02,
        min_abs_rank_icir=0.2,
        min_coverage=0.9,
        max_turnover=0.5,
        max_abs_drawdown=0.3,
        min_observations=400,
        annual_min_valid_years=4,
        annual_min_matching_sign_years=2,
        min_direction_adjusted_annualized_return=0.0,
        min_direction_adjusted_sharpe=0.5,
    )
    calibration = _calibration_for(
        standards,
        {
            "abs_rank_ic": (0.02, 0.08),
            "abs_rank_icir": (0.2, 0.9),
            "coverage": (0.9, 1.0),
            "turnover": (0.1, 0.5),
            "observations": (400, 900),
            "direction_adjusted_annualized_return": (0.0, 0.4),
            "direction_adjusted_sharpe": (0.5, 2.0),
            "abs_drawdown": (0.0, 0.3),
        },
    )

    dates = _iso(_dates(250))
    def varying(step: float) -> pd.Series:
        values = [step * (1 + 0.25 * (index % 4)) for index in range(len(dates))]
        return pd.Series(values, index=pd.DatetimeIndex(dates))

    negative_profile = varying(-0.002)
    positive_profile = varying(0.002)

    reversed_candidate = evaluate_against_calibration(
        _candidate_metrics(-0.05),
        calibration,
        return_profile=negative_profile,
    )
    assert reversed_candidate.passed, reversed_candidate.failures
    assert reversed_candidate.profile_available
    assert reversed_candidate.percentiles["abs_rank_ic"] == pytest.approx(0.5)

    # the same factor with the opposite orientation and the mirrored profile is
    # accepted exactly the same way: no signed comparison leaks in
    mirrored = evaluate_against_calibration(
        _candidate_metrics(0.05),
        calibration,
        return_profile=positive_profile,
    )
    assert mirrored.passed and mirrored.failures == ()
    assert mirrored.percentiles == reversed_candidate.percentiles

    weak = evaluate_against_calibration(
        _candidate_metrics(0.01), calibration, return_profile=positive_profile
    )
    assert not weak.passed and "abs_rank_ic" in weak.failures

    no_profile = evaluate_against_calibration(
        _candidate_metrics(0.05), calibration, return_profile=None
    )
    assert no_profile.passed  # profile gates skipped, recorded explicitly
    assert set(no_profile.skipped_gates) == {
        "direction_adjusted_annualized_return",
        "direction_adjusted_sharpe",
        "abs_drawdown",
    }

    strict = evaluate_against_calibration(
        _candidate_metrics(0.05), calibration, return_profile=None, require_profile=True
    )
    assert not strict.passed and "return_profile_missing" in strict.failures

    loss = evaluate_against_calibration(
        _candidate_metrics(0.05),
        calibration,
        return_profile=varying(-0.01),
    )
    assert not loss.passed and "direction_adjusted_annualized_return" in loss.failures


def test_annual_gate_failures_fail_closed() -> None:
    assert annual_gate_failures({"metrics_by_year": {}}, min_valid_years=2, min_matching_sign_years=1) == [
        "annual_metrics_by_year_missing"
    ]
    metrics = {
        "Rank_IC": 0.05,
        "metrics_by_year": {"2020": {"Rank_IC": 0.04}, "2021": {"Rank_IC": -0.03}},
    }
    assert annual_gate_failures(metrics, min_valid_years=3, min_matching_sign_years=1) == ["annual_rank_ic_years"]
    assert annual_gate_failures(metrics, min_valid_years=2, min_matching_sign_years=2) == [
        "annual_rank_ic_sign_alignment"
    ]
    zero = {"Rank_IC": 0.0, "metrics_by_year": {"2020": {"Rank_IC": 0.04}, "2021": {"Rank_IC": 0.04}}}
    assert annual_gate_failures(zero, min_valid_years=2, min_matching_sign_years=1) == [
        "annual_rank_ic_sign_alignment"
    ]


# ---------------------------------------------------------------------------
# service integration
# ---------------------------------------------------------------------------


def test_service_selects_standards_by_family_and_records_audit_metadata(tmp_path) -> None:
    standards_pv = FamilyStandards(
        family="price_volume",
        min_abs_rank_ic=0.02,
        min_coverage=0.9,
        max_turnover=0.5,
        max_abs_drawdown=0.3,
        min_observations=400,
        annual_min_valid_years=2,
        annual_min_matching_sign_years=1,
    )
    standards_fund = FamilyStandards(
        family="fundamental",
        min_abs_rank_ic=0.10,  # far stricter than the price_volume family
        min_coverage=0.9,
        max_turnover=0.5,
        max_abs_drawdown=0.3,
        min_observations=400,
        annual_min_valid_years=2,
        annual_min_matching_sign_years=1,
    )
    calibrations = {
        "price_volume": _calibration_for(standards_pv, {}),
        "fundamental": _calibration_for(standards_fund, {}),
    }
    registry = FactorRegistry(tmp_path / "registry.sqlite")

    def evaluator(candidate: FactorCandidate) -> EvaluationResult:
        rank_ic = 0.05  # clears the price_volume floor but not the fundamental one
        metrics = _candidate_metrics(rank_ic)
        return EvaluationResult(metrics=metrics, artifact_path=None, artifact_checksum=None)

    service = FactorAcceptanceService(registry, evaluator, calibrations=calibrations)
    pv = service.process(FactorCandidate("pv", "price_volume", "-pct_change(adjclose, 20)"))
    fund = service.process(FactorCandidate("fund", "fundamental", "cs_rank(balance__x / mv)"))

    assert pv.record.status == "accepted"
    assert fund.record.status == "rejected" and "abs_rank_ic" in fund.gate_failures
    assert pv.record.metadata["standards_source"] == "calibrated"
    assert pv.record.metadata["standards"]["family"] == "price_volume"
    assert "abs_rank_ic" in pv.record.metadata["benchmark_percentiles"]
    assert pv.benchmark_gate is not None and pv.benchmark_gate.standards is standards_pv

    fallback = FactorAcceptanceService(registry, evaluator, calibrations={})
    assert fallback.process(FactorCandidate("legacy", "price_volume", "-pct_change(adjclose, 20)")).record.status in {
        "accepted",
        "rejected",
    }
    with pytest.raises(RuntimeError, match="No calibrated benchmark standards"):
        FactorAcceptanceService(registry, evaluator, calibrations={"price_volume": calibrations["price_volume"]}).process(
            FactorCandidate("fund2", "fundamental", "cs_rank(balance__x / mv)")
        )


def test_service_falls_back_when_standards_are_missing_or_corrupt(tmp_path, capsys) -> None:
    from types import SimpleNamespace

    registry = FactorRegistry(tmp_path / "registry.sqlite")
    settings = SimpleNamespace(
        benchmark_standards_path=tmp_path / "absent.json",
        benchmark_signature_dir=tmp_path / "signatures",
        benchmark_calibration_required=False,
        benchmark_profile_required=False,
    )
    service = FactorAcceptanceService(
        FactorRegistry(tmp_path / "r1.sqlite"),
        lambda candidate: EvaluationResult(
            metrics={
                "Rank_IC": 0.05,
                "annualized_return": 0.1,
                "sharpe": 0.5,
                "max_drawdown": -0.1,
                "turnover": 0.2,
                "coverage": 0.99,
                "metrics_by_year": {"2020": {"Rank_IC": 0.04}, "2021": {"Rank_IC": 0.04}, "2022": {"Rank_IC": 0.04}},
            }
        ),
        settings=settings,
    )
    assert service.standards_source == "development_fallback"
    assert service.process(FactorCandidate("dev", "price_volume", "adjclose")).record.status == "accepted"
    assert "development fallback" in capsys.readouterr().err

    strict_settings = SimpleNamespace(**{**settings.__dict__, "benchmark_calibration_required": True})
    with pytest.raises(RuntimeError, match="required but unusable"):
        FactorAcceptanceService(
            FactorRegistry(tmp_path / "r2.sqlite"), lambda candidate: EvaluationResult(metrics={}), settings=strict_settings
        )

    (tmp_path / "corrupt.json").write_text("{", encoding="utf-8")
    corrupt_settings = SimpleNamespace(**{**settings.__dict__, "benchmark_standards_path": tmp_path / "corrupt.json"})
    FactorAcceptanceService(
        FactorRegistry(tmp_path / "r3.sqlite"), lambda candidate: EvaluationResult(metrics={}), settings=corrupt_settings
    )
    assert "development fallback" in capsys.readouterr().err


def test_service_flags_benchmark_family_redundancy(tmp_path) -> None:
    standards = FamilyStandards(
        family="price_volume",
        min_abs_rank_ic=0.01,
        min_coverage=0.5,
        max_turnover=1.0,
        max_abs_drawdown=1.0,
        min_observations=10,
        annual_min_valid_years=1,
        annual_min_matching_sign_years=1,
    )
    calibration = _calibration_for(standards, {})
    dates = _iso(_dates(200))
    twin_values = {date: 0.001 * (1 + 0.2 * (index % 3)) for index, date in enumerate(dates)}
    signature = _result("benchmark_twin", spread=_profile(twin_values))

    def evaluator(candidate: FactorCandidate) -> EvaluationResult:
        profile = pd.Series(list(twin_values.values()), index=pd.DatetimeIndex(dates))
        return EvaluationResult(metrics=_candidate_metrics(0.05), return_profile=profile)

    service = FactorAcceptanceService(
        FactorRegistry(tmp_path / "registry.sqlite"),
        evaluator,
        calibrations={"price_volume": calibration},
        benchmark_signatures=[signature],
    )
    result = service.process(FactorCandidate("twin", "price_volume", "-pct_change(adjclose, 20)"))
    assert result.record.status == "duplicate"
    assert result.record.metadata["benchmark_redundant_of"] == "benchmark_twin"
    assert result.record.metadata["benchmark_redundancy_reason"] == "benchmark_family_profile_correlation"
    assert result.benchmark_redundancy is not None

    unrelated = FactorAcceptanceService(
        FactorRegistry(tmp_path / "registry2.sqlite"),
        evaluator,
        calibrations={"price_volume": calibration},
        benchmark_signatures=[
            _result("other", spread=_profile({date: 0.02 * ((index % 2) - 0.5) for index, date in enumerate(dates)}))
        ],
    )
    candidate_profile = {date: 0.001 * (1 + 0.2 * (index % 3)) for index, date in enumerate(dates)}
    correlation = profile_correlation(
        candidate_profile, _result("other").spread_profile, min_overlap=100
    )
    assert correlation is None or abs(correlation) < 0.88
    clean = unrelated.process(FactorCandidate("fresh", "price_volume", "-pct_change(adjclose, 5)"))
    assert clean.record.status == "accepted"


def test_development_policy_still_enforces_its_temporary_annual_gate() -> None:
    policy = AcceptancePolicy()
    metrics = {
        "Rank_IC": 0.05,
        "annualized_return": 0.1,
        "sharpe": 0.5,
        "max_drawdown": -0.1,
        "turnover": 0.2,
        "coverage": 0.99,
    }
    passed, failures = policy.check(metrics)
    assert not passed and failures == ["annual_metrics_by_year_missing"]
