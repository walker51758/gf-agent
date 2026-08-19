"""Remote-first evaluation for GF factor candidates.

The RD-Agent proposal loop runs locally, while factor expressions are evaluated
next to the GF ``G:/base`` data dump.  Local evaluation is available only as an
explicit development mode; remote evaluation is the production default.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
from rdagent.scenarios.gf_quant.service import EvaluationResult, FactorCandidate


DEFAULT_HOST_KEY = "SHA256:6MqioxKcyZXMd1dx+t+DJOgSWXIqJUtb0j4Zaay/3x0"
SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./:-]+$")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
EvaluationMode = Literal["remote", "local"]


@dataclass(frozen=True)
class RemoteSettings:
    """Connection details for the pinned GF evaluation host."""

    host: str = "8.138.237.193"
    port: int = 7022
    user: str = "sshuser"
    password_env: str = "SSH_CODE"
    host_key_sha256: str = DEFAULT_HOST_KEY
    python_path: str = "G:/.venv/Scripts/python.exe"
    work_dir: str = "G:/gf_quant_agent"

    @classmethod
    def from_env(cls) -> "RemoteSettings":
        """Build controller settings from the standalone controller env contract."""
        return cls(
            host=os.getenv("GF_DATA_HOST", cls.host),
            port=int(os.getenv("GF_DATA_PORT", str(cls.port))),
            user=os.getenv("GF_DATA_USER", cls.user),
            password_env=os.getenv("GF_DATA_PASSWORD_ENV", cls.password_env),
            host_key_sha256=os.getenv("GF_DATA_HOST_KEY_SHA256", cls.host_key_sha256),
            python_path=os.getenv("GF_REMOTE_PYTHON", cls.python_path),
            work_dir=os.getenv("GF_REMOTE_WORK_DIR", cls.work_dir),
        )

    @classmethod
    def from_gf_settings(cls, settings: Any = GF_SCENARIO_SETTINGS) -> "RemoteSettings":
        """Build controller settings from :class:`GFScenarioSettings`."""
        return cls(
            host=settings.data_host,
            port=settings.data_port,
            user=settings.data_user,
            password_env=settings.data_password_env,
            host_key_sha256=settings.data_host_key_sha256,
            python_path=settings.remote_python,
            work_dir=settings.remote_work_dir,
        )


def _fingerprint(key: Any) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _safe_remote(path: str) -> str:
    if not SAFE_REMOTE_PATH.fullmatch(path):
        raise ValueError(f"Remote path contains unsupported characters: {path!r}")
    return path


def _connect(settings: RemoteSettings) -> Any:
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - depends on deployment extra
        raise RuntimeError("Install the optional 'paramiko' dependency for SSH evaluation") from exc

    password = os.getenv(settings.password_env)
    if not password:
        raise RuntimeError(f"Missing SSH password environment variable: {settings.password_env}")

    transport = paramiko.Transport((settings.host, settings.port))
    try:
        transport.connect(username=settings.user, password=password)
        actual = _fingerprint(transport.get_remote_server_key())
        if settings.host_key_sha256 and actual != settings.host_key_sha256:
            raise RuntimeError(f"SSH host key mismatch for {settings.host}:{settings.port}; got {actual}")
        return transport
    except Exception:
        transport.close()
        raise


def evaluate_remote(
    request: Mapping[str, Any],
    *,
    worker_path: Path,
    settings: RemoteSettings | None = None,
) -> dict[str, Any]:
    """Upload one request, execute it remotely, and return parsed JSON.

    The request is data-only and the worker evaluates an allow-listed expression
    DSL.  Credentials stay in the controller process and are never uploaded.
    """
    settings = settings or RemoteSettings.from_env()
    work_dir = _safe_remote(settings.work_dir.rstrip("/"))
    python_path = _safe_remote(settings.python_path)
    request_name = _SAFE_NAME.sub("_", str(request.get("name", "candidate"))).strip("._") or "candidate"
    request_remote = f"{work_dir}/{request_name}_request.json"
    result_remote = f"{work_dir}/{request_name}_result.json"
    worker_remote = f"{work_dir}/gf_quant_remote_worker.py"

    transport = _connect(settings)
    try:
        sftp = transport.open_sftp_client()
        try:
            try:
                sftp.stat(work_dir)
            except OSError:
                sftp.mkdir(work_dir)
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
                raise RuntimeError(f"Remote evaluator exited {exit_status}: {stderr[-4000:]}")
            with sftp.open(result_remote, "rb") as stream:
                result = json.loads(stream.read().decode("utf-8"))
        finally:
            sftp.close()
        if not isinstance(result, dict):
            raise RuntimeError("Remote evaluator returned a non-object JSON result")
        result["remote_stdout_present"] = bool(stdout.strip())
        return result
    finally:
        transport.close()


class GFFactorEvaluator:
    """Evaluate GF candidates remotely by default and adapt the worker result."""

    def __init__(
        self,
        *,
        settings: Any = GF_SCENARIO_SETTINGS,
        mode: EvaluationMode | None = None,
        remote_evaluate: Callable[..., dict[str, Any]] = evaluate_remote,
        worker_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.mode: EvaluationMode = mode or settings.factor_evaluation_mode
        if self.mode not in ("remote", "local"):
            raise ValueError(f"Unsupported GF factor evaluation mode: {self.mode!r}")
        self.remote_evaluate = remote_evaluate
        self.worker_path = worker_path or Path(__file__).parents[3] / "scripts" / "gf_quant_remote_worker.py"

    @staticmethod
    def _safe_name(name: str) -> str:
        return _SAFE_NAME.sub("_", name).strip("._") or "candidate"

    def request_for(self, candidate: FactorCandidate) -> dict[str, Any]:
        """Build the data-only request consumed by the isolated evaluator."""
        name = self._safe_name(candidate.name)
        artifact_root = (
            self.settings.remote_artifact_dir if self.mode == "remote" else self.settings.local_artifact_dir
        ).rstrip("/\\")
        return {
            "name": name,
            "family": candidate.family,
            "expression": candidate.expression,
            "base_path": self.settings.data_base_path,
            "lookback_days": self.settings.evaluation_lookback_days,
            "close_field": self.settings.evaluation_close_field,
            "artifact_path": f"{artifact_root}/{name}.parquet",
            "evaluation": {
                "start": self.settings.evaluation_start,
                "end": self.settings.evaluation_end,
                "label_horizon": self.settings.evaluation_label_horizon,
                "quantile": self.settings.evaluation_quantile,
                "min_cross_section": self.settings.evaluation_min_cross_section,
                "sample_step": self.settings.evaluation_sample_step,
            },
        }

    def __call__(self, candidate: FactorCandidate) -> EvaluationResult:
        request = self.request_for(candidate)
        if self.mode == "remote":
            payload = self.remote_evaluate(
                request,
                worker_path=self.worker_path,
                settings=RemoteSettings.from_gf_settings(self.settings),
            )
        else:
            payload = self._evaluate_locally(request)
        return self._result_from_payload(candidate, payload)

    @staticmethod
    def _result_from_payload(candidate: FactorCandidate, payload: Mapping[str, Any]) -> EvaluationResult:
        if payload.get("name") != GFFactorEvaluator._safe_name(candidate.name):
            raise RuntimeError("Evaluator result name does not match the submitted factor candidate")
        if payload.get("family") != candidate.family:
            raise RuntimeError("Evaluator result family does not match the submitted factor candidate")
        if payload.get("expression") != candidate.expression:
            raise RuntimeError("Evaluator result expression does not match the submitted factor candidate")
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError("Evaluator result does not contain a metrics object")
        return EvaluationResult(
            metrics=dict(metrics),
            artifact_path=payload.get("artifact_path"),
            artifact_checksum=payload.get("artifact_checksum"),
        )

    def _evaluate_locally(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Run the exact remote worker locally for deliberate offline debugging."""
        spec = importlib.util.spec_from_file_location("gf_quant_remote_worker", self.worker_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load local GF worker: {self.worker_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        run_request = module.run_request

        with tempfile.TemporaryDirectory(prefix="gf-factor-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            result_path = root / "result.json"
            request_path.write_text(json.dumps(dict(request), ensure_ascii=False), encoding="utf-8")
            payload = run_request(request_path, result_path)
        if not isinstance(payload, dict):  # defensive guard for a custom worker implementation
            raise RuntimeError("Local evaluator returned a non-object result")
        return payload
