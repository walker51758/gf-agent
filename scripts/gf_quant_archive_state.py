"""Archive the calibration-era GF factor/strategy state and start clean.

The archive is intentionally kept outside the active registry/artifact paths.
Large remote artifacts are renamed on the remote G: drive; they are never
downloaded or duplicated locally.  Local SQLite registries are moved intact,
exported to readable JSON, and recreated with empty production schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_STATE_ROOT = PROJECT_ROOT / "git_ignore_folder" / "gf_quant"
LOCAL_ARCHIVE_PARENT = LOCAL_STATE_ROOT / "state_archives"
ACTIVE_LOCAL_ARTIFACT_DIRS = (
    PROJECT_ROOT / "artifacts" / "live",
    PROJECT_ROOT / "artifacts" / "strategy",
)
REMOTE_ROOT = PurePosixPath("G:/gf_quant_agent")
REMOTE_STATE_DIRS = ("artifacts", "strategy_artifacts", "strategy_factor_artifacts")
ARCHIVE_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def _file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        result.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return result


def _read_table(path: Path, table: str) -> list[dict[str, Any]]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY 1')]


def _mkdir_sftp(sftp: Any, path: PurePosixPath) -> None:
    current = PurePosixPath(path.parts[0] + "/")
    for part in path.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except OSError:
            sftp.mkdir(str(current))


def _remote_archive(archive_id: str) -> dict[str, Any]:
    from rdagent.scenarios.gf_quant.remote import RemoteSettings, _connect

    settings = RemoteSettings.from_env()
    expected_root = PurePosixPath(settings.work_dir.replace("\\", "/"))
    if expected_root != REMOTE_ROOT:
        raise RuntimeError(f"refusing unexpected remote root: {expected_root}")

    archive_root = REMOTE_ROOT / "baseline_standards_archive" / archive_id
    moved: list[dict[str, str]] = []
    transport = _connect(settings)
    try:
        sftp = transport.open_sftp_client()
        try:
            try:
                sftp.stat(str(archive_root))
            except OSError:
                pass
            else:
                raise FileExistsError(f"remote archive already exists: {archive_root}")

            _mkdir_sftp(sftp, archive_root)
            request_result_root = archive_root / "requests_results"
            _mkdir_sftp(sftp, request_result_root)

            for name in REMOTE_STATE_DIRS:
                source = REMOTE_ROOT / name
                destination = archive_root / name
                sftp.stat(str(source))
                sftp.rename(str(source), str(destination))
                moved.append({"from": str(source), "to": str(destination)})
                sftp.mkdir(str(source))

            for attr in sorted(sftp.listdir_attr(str(REMOTE_ROOT)), key=lambda item: item.filename):
                name = attr.filename
                if not (name.endswith("_request.json") or name.endswith("_result.json")):
                    continue
                source = REMOTE_ROOT / name
                destination = request_result_root / name
                sftp.rename(str(source), str(destination))
                moved.append({"from": str(source), "to": str(destination)})

            remote_manifest = {
                "archive_id": archive_id,
                "purpose": "baseline standards only; excluded from active production state",
                "moved": moved,
            }
            with sftp.open(str(archive_root / "REMOTE_ARCHIVE_MANIFEST.json"), "wb") as stream:
                stream.write(json.dumps(remote_manifest, ensure_ascii=False, indent=2).encode("utf-8"))
            return remote_manifest
        finally:
            sftp.close()
    finally:
        transport.close()


def _move_contents(source: Path, destination: Path, moved: list[dict[str, str]]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    source.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        target = destination / item.name
        shutil.move(str(item), str(target))
        moved.append({"from": str(item), "to": str(target)})


def _local_archive(archive_id: str, remote_manifest: dict[str, Any]) -> Path:
    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.strategy.registry import StrategyRegistry

    factor_registry = LOCAL_STATE_ROOT / "factor_registry.sqlite"
    strategy_registry = LOCAL_STATE_ROOT / "strategy_registry.sqlite"
    for path in (factor_registry, strategy_registry):
        if not path.is_file():
            raise FileNotFoundError(f"active registry is missing: {path}")

    archive_root = LOCAL_ARCHIVE_PARENT / archive_id
    if archive_root.exists():
        raise FileExistsError(f"local archive already exists: {archive_root}")

    factors = _read_table(factor_registry, "factors")
    strategies = _read_table(strategy_registry, "strategies")
    snapshot_dirs = sorted(
        path for path in LOCAL_STATE_ROOT.iterdir() if path.is_dir() and path.name.startswith("snapshots")
    )
    local_sources = [factor_registry, strategy_registry, *snapshot_dirs, *ACTIVE_LOCAL_ARTIFACT_DIRS]
    before_files = _files_under(factor_registry) + _files_under(strategy_registry)
    for path in snapshot_dirs + list(ACTIVE_LOCAL_ARTIFACT_DIRS):
        before_files.extend(_files_under(path))
    before_manifest = _file_manifest(before_files)

    archive_root.mkdir(parents=True)
    exports = archive_root / "exports"
    exports.mkdir()
    (exports / "factors.json").write_text(
        json.dumps(factors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (exports / "strategies.json").write_text(
        json.dumps(strategies, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    moved: list[dict[str, str]] = []
    registries_dir = archive_root / "registries"
    registries_dir.mkdir()
    for source in (factor_registry, strategy_registry):
        destination = registries_dir / source.name
        shutil.move(str(source), str(destination))
        moved.append({"from": str(source), "to": str(destination)})

    snapshots_dir = archive_root / "snapshots"
    snapshots_dir.mkdir()
    for source in snapshot_dirs:
        destination = snapshots_dir / source.name
        shutil.move(str(source), str(destination))
        moved.append({"from": str(source), "to": str(destination)})

    artifacts_dir = archive_root / "local_artifacts"
    artifacts_dir.mkdir()
    for source in ACTIVE_LOCAL_ARTIFACT_DIRS:
        _move_contents(source, artifacts_dir / source.name, moved)

    FactorRegistry(factor_registry)
    StrategyRegistry(strategy_registry)
    if FactorRegistry(factor_registry).list() or StrategyRegistry(strategy_registry).list():
        raise RuntimeError("fresh registries are not empty")

    manifest = {
        "schema_version": 1,
        "archive_id": archive_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "calibration-era factors and strategies retained only as standards, not production outputs",
        "pre_reset": {
            "factor_records": len(factors),
            "strategy_records": len(strategies),
            "local_files": before_manifest,
            "local_sources": [str(path) for path in local_sources],
        },
        "local_moves": moved,
        "remote": remote_manifest,
        "post_reset": {
            "factor_records": 0,
            "strategy_records": 0,
            "active_local_artifact_files": sum(
                len(_files_under(path)) for path in ACTIVE_LOCAL_ARTIFACT_DIRS
            ),
        },
    }
    (archive_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return archive_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-id",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Timestamp-shaped archive id (YYYYMMDD-HHMMSS)",
    )
    args = parser.parse_args()
    if not ARCHIVE_ID_RE.fullmatch(args.archive_id):
        raise ValueError("archive id must use YYYYMMDD-HHMMSS")

    load_dotenv(PROJECT_ROOT / ".env")
    remote_manifest = _remote_archive(args.archive_id)
    archive_root = _local_archive(args.archive_id, remote_manifest)
    print(json.dumps({"archive_root": str(archive_root), "factor_records": 0, "strategy_records": 0}))


if __name__ == "__main__":
    main()
