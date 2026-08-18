"""Deploy and invoke the lightweight GF Quant evaluator over SSH.

Secrets are read by *name* from the environment.  They are never serialized
into the request or printed.  The default server key is pinned to the key
observed during the 2026-08-17 integration run; override it explicitly after a
legitimate host-key rotation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HOST_KEY = "SHA256:6MqioxKcyZXMd1dx+t+DJOgSWXIqJUtb0j4Zaay/3x0"
SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./:-]+$")


@dataclass(frozen=True)
class RemoteSettings:
    host: str = "8.138.237.193"
    port: int = 7022
    user: str = "sshuser"
    password_env: str = "SSH_CODE"
    host_key_sha256: str = DEFAULT_HOST_KEY
    python_path: str = "G:/.venv/Scripts/python.exe"
    work_dir: str = "G:/gf_quant_agent"

    @classmethod
    def from_env(cls) -> "RemoteSettings":
        return cls(
            host=os.getenv("GF_DATA_HOST", cls.host),
            port=int(os.getenv("GF_DATA_PORT", str(cls.port))),
            user=os.getenv("GF_DATA_USER", cls.user),
            password_env=os.getenv("GF_DATA_PASSWORD_ENV", cls.password_env),
            host_key_sha256=os.getenv("GF_DATA_HOST_KEY_SHA256", cls.host_key_sha256),
            python_path=os.getenv("GF_REMOTE_PYTHON", cls.python_path),
            work_dir=os.getenv("GF_REMOTE_WORK_DIR", cls.work_dir),
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
    transport.connect(username=settings.user, password=password)
    actual = _fingerprint(transport.get_remote_server_key())
    if settings.host_key_sha256 and actual != settings.host_key_sha256:
        transport.close()
        raise RuntimeError(f"SSH host key mismatch for {settings.host}:{settings.port}; got {actual}")
    return transport


def evaluate_remote(
    request: dict[str, Any],
    *,
    worker_path: Path,
    settings: RemoteSettings | None = None,
) -> dict[str, Any]:
    """Upload one request, execute it remotely, and return parsed JSON."""

    settings = settings or RemoteSettings.from_env()
    work_dir = _safe_remote(settings.work_dir.rstrip("/"))
    python_path = _safe_remote(settings.python_path)
    request_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(request.get("name", "candidate")))
    request_remote = f"{work_dir}/{request_name}_request.json"
    result_remote = f"{work_dir}/{request_name}_result.json"
    worker_remote = f"{work_dir}/gf_quant_remote_worker.py"

    transport = _connect(settings)
    try:
        sftp = transport.open_sftp_client()
        try:
            sftp.stat(work_dir)
        except OSError:
            sftp.mkdir(work_dir)
        sftp.put(str(worker_path), worker_remote)
        with sftp.open(request_remote, "wb") as stream:
            stream.write(json.dumps(request, ensure_ascii=False).encode("utf-8"))

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
            # stderr contains no credentials; truncate bulky tracebacks.
            raise RuntimeError(f"Remote evaluator exited {exit_status}: {stderr[-4000:]}")
        with sftp.open(result_remote, "rb") as stream:
            result = json.loads(stream.read().decode("utf-8"))
        result["remote_stdout_present"] = bool(stdout.strip())
        return result
    finally:
        transport.close()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Run one GF Quant request on the pinned remote data host")
    parser.add_argument("--request", required=True, type=Path, help="Local JSON request")
    parser.add_argument("--result", required=True, type=Path, help="Local JSON result destination")
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("gf_quant_remote_worker.py"),
        help="Remote worker source",
    )
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = evaluate_remote(request, worker_path=args.worker)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed={result.get('name')} family={result.get('family')} result={args.result}")


if __name__ == "__main__":
    main()
