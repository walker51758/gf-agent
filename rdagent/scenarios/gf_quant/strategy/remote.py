"""Local request/result adapter for the standalone GF strategy worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.remote import RemoteSettings, _connect, _safe_remote
from rdagent.scenarios.gf_quant.strategy.artifacts import FactorArtifactSet
from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
from rdagent.scenarios.gf_quant.strategy.service import StrategyEvaluation
from rdagent.scenarios.gf_quant.strategy.snapshot import FactorUniverseSnapshot, FactorWorkingSet

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evaluate_strategy_remote(
    request: Mapping[str, Any], *, worker_path: Path, settings: RemoteSettings
) -> dict[str, Any]:
    work_dir = _safe_remote(settings.work_dir.rstrip("/"))
    python_path = _safe_remote(settings.python_path)
    name = _SAFE_NAME.sub("_", str(request["candidate"]["name"])).strip("._") or "strategy"
    request_remote = f"{work_dir}/{name}_strategy_request.json"
    result_remote = f"{work_dir}/{name}_strategy_result.json"
    worker_remote = f"{work_dir}/gf_quant_strategy_worker.py"
    transport = _connect(settings)
    try:
        sftp = transport.open_sftp_client()
        try:
            sftp.put(str(worker_path), worker_remote)
            with sftp.open(request_remote, "wb") as stream:
                stream.write(json.dumps(dict(request), ensure_ascii=False).encode("utf-8"))
            command = " ".join(
                [
                    shlex.quote(python_path),
                    shlex.quote(worker_remote),
                    "--request",
                    shlex.quote(request_remote),
                    "--result",
                    shlex.quote(result_remote),
                ]
            )
            channel = transport.open_session()
            channel.exec_command(command)
            stdout = channel.makefile("rb", -1).read().decode("utf-8", "replace")
            stderr = channel.makefile_stderr("rb", -1).read().decode("utf-8", "replace")
            exit_status = channel.recv_exit_status()
            if exit_status != 0:
                raise RuntimeError(f"Remote strategy worker exited {exit_status}: {stderr[-4000:]}")
            with sftp.open(result_remote, "rb") as stream:
                payload = json.loads(stream.read().decode("utf-8"))
        finally:
            sftp.close()
    finally:
        transport.close()
    if not isinstance(payload, dict):
        raise RuntimeError("Remote strategy worker returned a non-object result")
    payload["remote_stdout_present"] = bool(stdout.strip())
    return payload


class RemoteStrategyEvaluator:
    def __init__(
        self,
        *,
        snapshot: FactorUniverseSnapshot,
        working_set: FactorWorkingSet,
        artifact_set: FactorArtifactSet,
        evaluation_start: str,
        evaluation_end: str,
        settings: Any = GF_SCENARIO_SETTINGS,
        rebalance_frequency: str = "monthly",
        remote_evaluate: Callable[..., dict[str, Any]] = evaluate_strategy_remote,
        worker_path: Path | None = None,
    ) -> None:
        if artifact_set.snapshot_id != snapshot.snapshot_id or artifact_set.working_set_id != working_set.working_set_id:
            raise ValueError("artifact set provenance does not match snapshot/working set")
        self.snapshot = snapshot
        self.working_set = working_set
        self.artifact_set = artifact_set
        self.evaluation_start = evaluation_start
        self.evaluation_end = evaluation_end
        self.settings = settings
        self.rebalance_frequency = rebalance_frequency
        self.remote_evaluate = remote_evaluate
        self.worker_path = worker_path or Path(__file__).parents[4] / "scripts" / "gf_quant_strategy_worker.py"

    def request_for(self, candidate: StrategyCandidate) -> dict[str, Any]:
        factor_by_name = {factor.name: factor for factor in self.working_set.factors}
        artifact_by_name = {factor.name: factor for factor in self.artifact_set.factors}
        if set(candidate.factor_names) != set(artifact_by_name):
            raise ValueError("candidate factors must match the materialized artifact set exactly")
        if not set(candidate.factor_names).issubset(factor_by_name):
            raise ValueError("candidate references factors outside working set")
        safe_name = _SAFE_NAME.sub("_", candidate.name).strip("._") or "strategy"
        start_tag = re.sub(r"[^0-9]", "", self.evaluation_start)
        end_tag = re.sub(r"[^0-9]", "", self.evaluation_end)
        portfolio_path = (
            f"{self.settings.remote_work_dir.rstrip('/')}/strategy_artifacts/"
            f"{safe_name}_{start_tag}_{end_tag}.parquet"
        )
        return {
            "schema_version": 1,
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_checksum": self.snapshot.checksum,
            "working_set_id": self.working_set.working_set_id,
            "working_set_snapshot_checksum": self.working_set.snapshot_checksum,
            "artifact_set": self.artifact_set.as_payload(),
            "factor_directions": {name: factor_by_name[name].direction for name in candidate.factor_names},
            "candidate": candidate.as_payload(),
            "evaluation": {
                "start": self.evaluation_start,
                "end": self.evaluation_end,
                "rebalance_frequency": self.rebalance_frequency,
            },
            "base_path": self.settings.data_base_path,
            "close_field": self.settings.evaluation_close_field,
            "portfolio_artifact_path": portfolio_path,
        }

    def __call__(self, candidate: StrategyCandidate) -> StrategyEvaluation:
        request = self.request_for(candidate)
        payload = self.remote_evaluate(
            request,
            worker_path=self.worker_path,
            settings=RemoteSettings.from_gf_settings(self.settings),
        )
        return self._result_from_payload(candidate, request, payload)

    @staticmethod
    def _result_from_payload(
        candidate: StrategyCandidate, request: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> StrategyEvaluation:
        if payload.get("status") != "success" or int(payload.get("schema_version", 0)) != 1:
            raise RuntimeError("strategy worker did not return a successful schema-v1 result")
        for key in ("snapshot_id", "snapshot_checksum", "working_set_id"):
            if payload.get(key) != request.get(key):
                raise RuntimeError(f"strategy result {key} does not match request")
        if payload.get("artifact_set_id") != request["artifact_set"]["artifact_set_id"]:
            raise RuntimeError("strategy result artifact_set_id does not match request")
        if payload.get("name") != candidate.name:
            raise RuntimeError("strategy result name does not match candidate")
        expected_request_checksum = hashlib.sha256(_canonical_json(request).encode()).hexdigest()
        if payload.get("request_checksum") != expected_request_checksum:
            raise RuntimeError("strategy result request checksum mismatch")
        metrics = payload.get("metrics")
        by_year = payload.get("metrics_by_year")
        profile = payload.get("return_profile")
        contributions = payload.get("factor_contributions")
        if not isinstance(metrics, Mapping) or not isinstance(by_year, Mapping):
            raise RuntimeError("strategy result metrics are malformed")
        if not isinstance(profile, Mapping) or not isinstance(contributions, Mapping):
            raise RuntimeError("strategy result return profile or contributions are malformed")
        dates, values = profile.get("dates"), profile.get("returns")
        if not isinstance(dates, list) or not isinstance(values, list) or not dates or len(dates) != len(values):
            raise RuntimeError("strategy return profile is malformed")
        if set(contributions) != set(candidate.factor_names):
            raise RuntimeError("strategy result contributions do not match candidate factors")
        if not payload.get("artifact_path") or not payload.get("artifact_checksum"):
            raise RuntimeError("strategy result is missing remote artifact provenance")
        returns = pd.Series([float(value) for value in values], index=pd.to_datetime(dates), name="strategy_returns")
        return StrategyEvaluation(
            metrics={**dict(metrics), "metrics_by_year": dict(by_year)},
            returns=returns,
            artifact_path=str(payload["artifact_path"]),
            artifact_checksum=str(payload["artifact_checksum"]),
            factor_contributions={str(key): dict(value) for key, value in contributions.items()},
        )
