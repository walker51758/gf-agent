from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.remote import GFFactorEvaluator
from rdagent.scenarios.gf_quant.service import FactorCandidate
from scripts.gf_quant_remote_worker import EvaluationConfig, evaluate_factor


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        factor_evaluation_mode="remote",
        remote_artifact_dir="G:/gf_quant_agent/artifacts",
        local_artifact_dir="artifacts/live",
        data_base_path="G:/base",
        evaluation_lookback_days=1200,
        evaluation_close_field="adjclose",
        evaluation_start="2022-01-01",
        evaluation_end="2023-12-31",
        evaluation_label_horizon=5,
        evaluation_quantile=0.2,
        evaluation_min_cross_section=100,
        evaluation_sample_step=5,
        data_host="host.example",
        data_port=7022,
        data_user="sshuser",
        data_password_env="SSH_CODE",
        data_host_key_sha256="SHA256:pinned",
        remote_python="G:/.venv/Scripts/python.exe",
        remote_work_dir="G:/gf_quant_agent",
    )


def _payload(request: dict) -> dict:
    return {
        "name": request["name"],
        "family": request["family"],
        "expression": request["expression"],
        "metrics": {"Rank_IC": 0.04, "coverage": 0.95},
        "artifact_path": request["artifact_path"],
        "artifact_checksum": "checksum",
    }


def test_remote_evaluator_is_default_and_uses_screening_window() -> None:
    settings = _settings()
    received: dict = {}

    def remote(request, **kwargs):
        received["request"] = request
        received["kwargs"] = kwargs
        return _payload(request)

    evaluator = GFFactorEvaluator(settings=settings, remote_evaluate=remote)
    candidate = FactorCandidate("mean reversion / 20", "price_volume", "-pct_change(adjclose, 20)")

    result = evaluator(candidate)

    assert evaluator.mode == "remote"
    assert received["request"]["name"] == "mean_reversion_20"
    assert received["request"]["evaluation"] == {
        "start": "2022-01-01",
        "end": "2023-12-31",
        "label_horizon": 5,
        "quantile": 0.2,
        "min_cross_section": 100,
        "sample_step": 5,
    }
    assert received["request"]["artifact_path"] == "G:/gf_quant_agent/artifacts/mean_reversion_20.parquet"
    assert received["kwargs"]["settings"].host == "host.example"
    assert received["kwargs"]["worker_path"].as_posix().endswith("scripts/gf_quant_remote_worker.py")
    assert result.artifact_checksum == "checksum"


def test_remote_evaluator_rejects_mismatched_worker_result() -> None:
    settings = _settings()

    def remote(request, **kwargs):
        result = _payload(request)
        result["expression"] = "adjclose"
        return result

    evaluator = GFFactorEvaluator(settings=settings, remote_evaluate=remote)
    with pytest.raises(RuntimeError, match="expression"):
        evaluator(FactorCandidate("candidate", "price_volume", "-pct_change(adjclose, 20)"))


def test_local_mode_is_explicit() -> None:
    evaluator = GFFactorEvaluator(settings=_settings(), mode="local")
    request = evaluator.request_for(FactorCandidate("candidate", "fundamental", "cs_rank(balance__parent_equity / mv)"))

    assert evaluator.mode == "local"
    assert request["artifact_path"] == "artifacts/live/candidate.parquet"


def test_worker_coverage_cannot_exceed_one() -> None:
    dates = pd.date_range("2022-01-03", periods=8, freq="D")
    close = pd.DataFrame(
        {"A": np.arange(10.0, 18.0), "B": [np.nan, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]},
        index=dates,
    )
    factor = pd.DataFrame({"A": np.arange(8.0), "B": np.arange(8.0)}, index=dates)
    config = EvaluationConfig(
        start="2022-01-03",
        end="2022-01-10",
        label_horizon=1,
        quantile=0.5,
        min_cross_section=1,
        sample_step=1,
    )

    metrics, _ = evaluate_factor(factor, close, config)

    assert 0.0 <= metrics["coverage"] <= 1.0
