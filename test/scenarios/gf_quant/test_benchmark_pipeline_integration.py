"""End-to-end integration test: synthetic benchmark panels -> calibration -> acceptance.

No remote host and no real data are involved: the test writes a tiny synthetic
``G:/base``-shaped directory (``calendar.parquet`` + ``adjclose.parquet``) and
small wide factor panels, runs the *real* benchmark worker over them, calibrates
with the real controller code, persists the standards document, and finally
gates candidates through :class:`FactorAcceptanceService`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.benchmark import calibration_to_document, load_family_calibrations
from rdagent.scenarios.gf_quant.dedup import FactorDeduplicator
from rdagent.scenarios.gf_quant.registry import FactorRegistry
from rdagent.scenarios.gf_quant.service import EvaluationResult, FactorAcceptanceService, FactorCandidate

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import gf_quant_benchmark_controller as controller  # noqa: E402  (scripts helper module)
import gf_quant_benchmark_worker as worker  # noqa: E402
from gf_quant_remote_worker import EvaluationConfig  # noqa: E402


N_DAYS = 320
N_SYMBOLS = 6
START = "2020-01-01"


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=N_DAYS)


def _write_base(base: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Write a synthetic base directory and return the wide adjusted-close panel."""
    rng = np.random.default_rng(7)
    symbols = [f"00000{index}.SZ" for index in range(N_SYMBOLS)]
    # Next-day returns load on the same day's cross-sectional dispersion so the
    # benchmark panels get a real, deterministic RankIC.
    signal = rng.normal(size=(len(dates), N_SYMBOLS))
    returns = pd.DataFrame(
        -0.02 * (signal - signal.mean(axis=1, keepdims=True)) + 0.001 * rng.normal(size=signal.shape)
    )
    close = 10.0 * (1.0 + returns.cumsum())
    close.index = dates
    close.columns = symbols
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_dt": dates}).to_parquet(base / "calendar.parquet", index=False)
    # the real base tables store dt/symbol/value as plain columns
    close.rename_axis("dt").stack().rename("value").astype(float).rename_axis(["dt", "symbol"]).reset_index().to_parquet(
        base / "adjclose.parquet", index=False
    )
    return close


def _write_family_dir(close: pd.DataFrame, directory: Path, *, include_empty: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    momentum = -close.pct_change(5)
    momentum.index.name = "trade_dt"
    momentum.to_parquet(directory / "momentum.parquet")
    # A near-identical twin so the clustering has real work to do.
    momentum.add(1e-9).to_parquet(directory / "momentum_twin.parquet")
    # A structurally different signal: a cross-sectional level, not a reversal.
    level = pd.DataFrame(
        np.arange(float(len(close)))[:, None] + np.arange(float(close.shape[1]))[None, :],
        index=close.index,
        columns=close.columns,
    )
    level.index.name = "trade_dt"
    level.to_parquet(directory / "level.parquet")
    if include_empty:
        # a panel with no rows at all (e.g. written outside the screening window)
        momentum.iloc[:0].to_parquet(directory / "empty.parquet")


def _run_worker(tmp_path: Path) -> Path:
    """Run the real worker over both synthetic source directories."""
    out_dir = tmp_path / "benchmark"
    cfg = EvaluationConfig(
        start="2020-01-01",
        end="2021-12-31",
        label_horizon=1,
        quantile=0.2,
        min_cross_section=3,
        sample_step=1,
    )
    for family, source in (
        ("price_volume", tmp_path / "pools" / "artifacts"),
        ("fundamental", tmp_path / "pools" / "basic_fundamental"),
    ):
        manifest = worker.evaluate_source(
            family,
            source,
            out_dir / family / source.name,
            cfg,
            base_path=str(tmp_path / "base"),
            close_field="adjclose",
        )
        assert manifest["total_files"] >= 2
    return out_dir


def _candidate_metrics(rank_ic: float) -> dict:
    return {
        "Rank_IC": rank_ic,
        "Rank_ICIR": rank_ic * 5,
        "coverage": 0.99,
        "turnover": 0.2,
        "observations": N_DAYS - 5,
        "metrics_by_year": {
            "2020": {"Rank_IC": rank_ic},
            "2021": {"Rank_IC": rank_ic},
        },
    }


@pytest.mark.slow
def test_synthetic_benchmark_pool_calibrates_and_gates_candidates(tmp_path: Path) -> None:
    dates = _calendar()
    close = _write_base(tmp_path / "base", dates)
    _write_family_dir(close, tmp_path / "pools" / "artifacts", include_empty=True)
    _write_family_dir(close, tmp_path / "pools" / "basic_fundamental", include_empty=False)

    # ---- 1. the real worker evaluates the synthetic panels -------------------
    out_dir = _run_worker(tmp_path)
    pv_summaries = sorted(path for path in (out_dir / "price_volume").rglob("*.json") if path.name != "_manifest.json")
    assert {path.stem for path in pv_summaries} == {"momentum", "momentum_twin", "level", "empty"}
    excluded = json.loads((out_dir / "price_volume" / "artifacts" / "empty.json").read_text(encoding="utf-8"))
    assert excluded["status"] == "excluded" and excluded["exclusion_reason"] == "empty_panel_in_window"

    # ---- 2. the real controller loads and calibrates both families ----------
    source_dirs = {
        "price_volume": [str(tmp_path / "pools" / "artifacts")],
        "fundamental": [str(tmp_path / "pools" / "basic_fundamental")],
    }
    calibrations = controller.calibrate(
        out_dir,
        source_dirs=source_dirs,
        threshold=0.9,
        min_overlap=100,
        cluster_profile="spread",
    )
    pv, fundamental = calibrations["price_volume"], calibrations["fundamental"]
    # the empty panel is dropped loudly, the three usable ones stay
    assert {exclusion.name for exclusion in pv.exclusions} == {"empty"}
    assert pv.sample["valid"] == 3
    # momentum and its twin cluster together instead of double-counting
    twin_cluster = next(cluster for cluster in pv.clusters if "momentum" in cluster.members)
    assert {"momentum", "momentum_twin"} <= set(twin_cluster.members)
    assert pv.standards.family == "price_volume" and fundamental.standards.family == "fundamental"
    assert pv.standards.min_abs_rank_ic > 0 and fundamental.standards.min_abs_rank_ic > 0
    # the small sample keeps the optional standards explicitly skipped, never invented
    assert pv.standards.min_abs_rank_icir is None
    assert "abs_rank_icir" in pv.sample["standards_sample_sizes"]

    # ---- 3. the standards document round-trips through JSON ------------------
    document = calibration_to_document(
        calibrations,
        generated_at="2026-08-30T00:00:00Z",
        evaluation={"start": "2020-01-01", "end": "2021-12-31"},
        provenance={"mode": "synthetic integration test"},
    )
    standards_path = tmp_path / "benchmark_standards.json"
    standards_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    loaded = load_family_calibrations(standards_path)
    assert set(loaded) == {"price_volume", "fundamental"}
    assert loaded["price_volume"].standards == pv.standards

    # ---- 4. candidate acceptance against the calibrated standards ------------
    signature_dir = tmp_path / "signatures"
    controller.write_signatures(calibrations, out_dir, signature_dir)
    assert (signature_dir / "price_volume_signatures.json").exists()

    momentum_payload = json.loads(
        (out_dir / "price_volume" / "artifacts" / "momentum.json").read_text(encoding="utf-8")
    )["profile"]
    observed = [(date, value) for date, value in zip(momentum_payload["dates"], momentum_payload["spread"]) if value is not None]
    momentum_profile = pd.Series(
        {pd.Timestamp(date): value for date, value in observed}, dtype=float
    )

    def evaluator(candidate: FactorCandidate) -> EvaluationResult:
        if candidate.name == "benchmark_twin":
            # same daily long-short profile as the benchmark momentum factor
            return EvaluationResult(metrics=_candidate_metrics(0.4), return_profile=momentum_profile)
        return EvaluationResult(metrics=_candidate_metrics(0.4))

    registry = FactorRegistry(tmp_path / "registry.sqlite")
    service = FactorAcceptanceService(
        registry,
        evaluator,
        standards_path=standards_path,
        signature_dir=signature_dir,
        deduplicator=FactorDeduplicator(threshold=0.88, min_overlap=30),
    )
    assert service.standards_source == "calibrated"

    accepted = service.process(FactorCandidate("fresh_alpha", "price_volume", "-pct_change(adjclose, 10)"))
    assert accepted.record.status == "accepted", accepted.gate_failures
    assert accepted.record.metadata["standards_source"] == "calibrated"
    assert accepted.record.metadata["standards"]["family"] == "price_volume"
    assert "abs_rank_ic" in accepted.record.metadata["benchmark_percentiles"]

    def weak_evaluator(candidate: FactorCandidate) -> EvaluationResult:
        return EvaluationResult(metrics=_candidate_metrics(0.0))

    rejecting_service = FactorAcceptanceService(
        FactorRegistry(tmp_path / "registry2.sqlite"),
        weak_evaluator,
        standards_path=standards_path,
        signature_dir=signature_dir,
        deduplicator=FactorDeduplicator(threshold=0.88, min_overlap=30),
    )
    rejected = rejecting_service.process(FactorCandidate("weak", "price_volume", "-pct_change(adjclose, 3)"))
    assert rejected.record.status == "rejected"
    assert "abs_rank_ic" in rejected.gate_failures
    assert "annual_rank_ic_sign_alignment" in rejected.gate_failures

    duplicate = service.process(FactorCandidate("benchmark_twin", "price_volume", "-pct_change(adjclose, 5)"))
    assert duplicate.record.status == "duplicate"
    assert duplicate.record.metadata["benchmark_redundant_of"] in {"momentum", "momentum_twin"}
    assert duplicate.record.metadata["benchmark_redundancy_reason"] == "benchmark_family_profile_correlation"

    # fundamental candidates are held to their own family's standards
    fundamental_result = service.process(FactorCandidate("bp", "fundamental", "cs_rank(book_value / mv)"))
    assert fundamental_result.record.status == "accepted", fundamental_result.gate_failures
    assert fundamental_result.record.metadata["standards"]["family"] == "fundamental"
