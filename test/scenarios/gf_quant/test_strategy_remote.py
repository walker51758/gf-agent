from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.artifacts import (
    FactorArtifactSet,
    FactorArtifactVersion,
    artifact_set_id_for,
)
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec, StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.remote import RemoteStrategyEvaluator
from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set
from scripts.gf_quant_strategy_worker import run_request


def _fixture(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=65)
    symbols = [f"S{position}" for position in range(6)]
    base = tmp_path / "base"
    base.mkdir()
    long = pd.DataFrame(
        [
            {"dt": date, "symbol": symbol, "value": 100 * (1.0 + 0.0005 * day + 0.0008 * position)}
            for day, date in enumerate(pd.bdate_range("2023-12-20", periods=75))
            for position, symbol in enumerate(symbols)
        ]
    )
    long.to_parquet(base / "adjclose.parquet")

    registry = FactorRegistry(tmp_path / "factors.sqlite")
    versions = []
    for position, name in enumerate(("value", "quality")):
        original = tmp_path / f"old_{name}.parquet"
        pd.DataFrame(0.0, index=dates, columns=symbols).to_parquet(original)
        registry.upsert(
            FactorRecord(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                status="accepted",
                metrics={"Rank_IC": 0.03},
                artifact_path=str(original),
                artifact_checksum=sha256(original.read_bytes()).hexdigest(),
            )
        )
        path = tmp_path / f"{name}.parquet"
        values = pd.DataFrame(
            {
                symbol: np.linspace(0, 1, len(dates)) + (position + 1) * symbol_position
                for symbol_position, symbol in enumerate(symbols)
            },
            index=dates,
        )
        values.to_parquet(path)
        versions.append(
            FactorArtifactVersion(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                artifact_path=str(path),
                artifact_checksum=sha256(path.read_bytes()).hexdigest(),
                data_start=dates[0].strftime("%Y-%m-%d"),
                data_end=dates[-1].strftime("%Y-%m-%d"),
                rows=len(dates),
                columns=len(symbols),
            )
        )
    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")
    working = build_factor_working_set(snapshot, core_limit=2, conditional_limit=0)
    start, end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    artifact_set = FactorArtifactSet(
        artifact_set_id=artifact_set_id_for(snapshot.snapshot_id, working.working_set_id, start, end, versions),
        snapshot_id=snapshot.snapshot_id,
        working_set_id=working.working_set_id,
        evaluation_start=start,
        evaluation_end=end,
        created_at="2024-01-01T00:00:00Z",
        factors=tuple(versions),
    )
    candidate = StrategyCandidate(
        name="remote_smoke",
        universe="csi_all",
        composition=CompositionSpec(top_n=3, weighting="equal", weight_cap=0.5),
        factor_names=("value", "quality"),
        rule="equal rank blend",
        backtest=BacktestSpec(signal_lag=1),
    )
    settings = SimpleNamespace(
        remote_work_dir=str(tmp_path).replace("\\", "/"),
        data_base_path=str(base),
        evaluation_close_field="adjclose",
        data_host="host",
        data_port=22,
        data_user="user",
        data_password_env="PASSWORD",
        data_host_key_sha256="",
        remote_python="python",
    )
    evaluator = RemoteStrategyEvaluator(
        snapshot=snapshot,
        working_set=working,
        artifact_set=artifact_set,
        evaluation_start=start,
        evaluation_end=end,
        settings=settings,
    )
    return evaluator, candidate


def test_standalone_strategy_worker_smoke_and_result_parser(tmp_path) -> None:
    evaluator, candidate = _fixture(tmp_path)
    request = evaluator.request_for(candidate)
    assert request["execution_costs"] == {"buy_bps": 10.0, "sell_bps": 10.0}
    assert "cost_bps" not in request["candidate"]["backtest"]
    assert "2024-01-02".replace("-", "") in request["portfolio_artifact_path"]
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio.parquet")
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    payload = run_request(request_path, result_path)
    evaluation = evaluator._result_from_payload(candidate, request, payload)

    assert payload["status"] == "success"
    assert payload["audit"]["factor_score_weights_supported"] is False
    assert set(payload["factor_contributions"]) == {"value", "quality"}
    assert evaluation.metrics["observations"] == 65
    assert evaluation.metrics["metrics_by_year"]
    assert len(evaluation.returns) == 65
    assert (tmp_path / "portfolio.parquet").exists()


def test_strategy_worker_fails_closed_on_provenance_and_coverage(tmp_path) -> None:
    evaluator, candidate = _fixture(tmp_path)
    request = evaluator.request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio.parquet")
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"

    tampered = json.loads(json.dumps(request))
    tampered["snapshot_checksum"] = "0" * 64
    request_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_id"):
        run_request(request_path, result_path)

    incomplete = json.loads(json.dumps(request))
    incomplete["evaluation"]["end"] = "2025-01-01"
    request_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="cover"):
        run_request(request_path, result_path)

    tampered_cost = json.loads(json.dumps(request))
    tampered_cost["execution_costs"]["sell_bps"] = 1.0
    request_path.write_text(json.dumps(tampered_cost), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed 10/10"):
        run_request(request_path, result_path)

    candidate_cost = json.loads(json.dumps(request))
    candidate_cost["candidate"]["backtest"]["cost_bps"] = 0.0
    request_path.write_text(json.dumps(candidate_cost), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate backtest"):
        run_request(request_path, result_path)


def test_result_parser_rejects_request_checksum_mismatch(tmp_path) -> None:
    evaluator, candidate = _fixture(tmp_path)
    request = evaluator.request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio.parquet")
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    payload["request_checksum"] = "tampered"

    with pytest.raises(RuntimeError, match="request checksum"):
        evaluator._result_from_payload(candidate, request, payload)


def test_result_parser_rejects_tampered_cost_metrics(tmp_path) -> None:
    evaluator, candidate = _fixture(tmp_path)
    request = evaluator.request_for(candidate)
    request["portfolio_artifact_path"] = str(tmp_path / "portfolio.parquet")
    request_path, result_path = tmp_path / "request.json", tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    payload = run_request(request_path, result_path)
    payload["metrics"]["buy_cost_bps"] = 1.0

    with pytest.raises(RuntimeError, match="fixed execution costs"):
        evaluator._result_from_payload(candidate, request, payload)
