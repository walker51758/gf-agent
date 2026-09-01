"""Versioned factor artifacts for an independent strategy-evaluation window.

The factor registry freezes factor logic and its original research artifact.
Those artifacts may come from different historical runs and need not overlap in
dates.  Before a real portfolio backtest, this module re-evaluates only the
referenced expressions into a new remote directory, leaving every original
artifact untouched, and records a compact checksummed manifest locally.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rdagent.scenarios.gf_quant.remote import GFFactorEvaluator, RemoteSettings, _connect
from rdagent.scenarios.gf_quant.service import EvaluationResult, FactorCandidate
from rdagent.scenarios.gf_quant.strategy.snapshot import FactorWorkingSet


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FactorArtifactVersion:
    factor_id: str
    name: str
    family: str
    expression: str
    artifact_path: str
    artifact_checksum: str
    data_start: str
    data_end: str
    rows: int
    columns: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "family": self.family,
            "expression": self.expression,
            "artifact_path": self.artifact_path,
            "artifact_checksum": self.artifact_checksum,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "rows": self.rows,
            "columns": self.columns,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FactorArtifactVersion":
        return cls(
            factor_id=str(payload["factor_id"]),
            name=str(payload["name"]),
            family=str(payload["family"]),
            expression=str(payload["expression"]),
            artifact_path=str(payload["artifact_path"]),
            artifact_checksum=str(payload["artifact_checksum"]),
            data_start=str(payload["data_start"]),
            data_end=str(payload["data_end"]),
            rows=int(payload["rows"]),
            columns=int(payload["columns"]),
        )


def artifact_set_id_for(
    snapshot_id: str,
    working_set_id: str,
    evaluation_start: str,
    evaluation_end: str,
    factors: Sequence[FactorArtifactVersion],
) -> str:
    body = {
        "snapshot_id": snapshot_id,
        "working_set_id": working_set_id,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "factors": [factor.as_payload() for factor in sorted(factors, key=lambda item: item.factor_id)],
    }
    return f"gf_artifact_set_{hashlib.sha256(_canonical_json(body).encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class FactorArtifactSet:
    artifact_set_id: str
    snapshot_id: str
    working_set_id: str
    evaluation_start: str
    evaluation_end: str
    created_at: str
    factors: tuple[FactorArtifactVersion, ...]

    def __post_init__(self) -> None:
        names = [factor.name for factor in self.factors]
        if len(names) != len(set(names)):
            raise ValueError("artifact set contains duplicate factor names")
        expected = artifact_set_id_for(
            self.snapshot_id,
            self.working_set_id,
            self.evaluation_start,
            self.evaluation_end,
            self.factors,
        )
        if self.artifact_set_id != expected:
            raise ValueError("artifact set id does not match its frozen contents")
        for factor in self.factors:
            if factor.data_start > self.evaluation_start or factor.data_end < self.evaluation_end:
                raise ValueError(f"factor artifact {factor.name} does not cover the requested evaluation window")

    def as_payload(self) -> dict[str, Any]:
        return {
            "artifact_set_id": self.artifact_set_id,
            "snapshot_id": self.snapshot_id,
            "working_set_id": self.working_set_id,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "created_at": self.created_at,
            "factors": [factor.as_payload() for factor in self.factors],
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FactorArtifactSet":
        factors = tuple(FactorArtifactVersion.from_payload(value) for value in payload.get("factors", []))
        return cls(
            artifact_set_id=str(payload["artifact_set_id"]),
            snapshot_id=str(payload["snapshot_id"]),
            working_set_id=str(payload["working_set_id"]),
            evaluation_start=str(payload["evaluation_start"]),
            evaluation_end=str(payload["evaluation_end"]),
            created_at=str(payload["created_at"]),
            factors=factors,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "FactorArtifactSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("artifact set JSON must contain an object")
        return cls.from_payload(payload)


def inspect_remote_factor_artifact(path: str, *, settings: RemoteSettings) -> dict[str, Any]:
    """Return compact remote parquet provenance without downloading the panel."""

    code = f"""
import hashlib
import json
from pathlib import Path
import pandas as pd
path = Path({path!r})
if not path.exists():
    raise FileNotFoundError(str(path))
frame = pd.read_parquet(path)
index = pd.to_datetime(frame.index)
print(json.dumps({{
    'path': str(path),
    'checksum': hashlib.sha256(path.read_bytes()).hexdigest(),
    'rows': int(frame.shape[0]),
    'columns': int(frame.shape[1]),
    'start': str(index.min().date()),
    'end': str(index.max().date()),
}}))
""".strip()
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    command = f'{shlex.quote(settings.python_path)} -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'
    transport = _connect(settings)
    try:
        channel = transport.open_session()
        channel.exec_command(command)
        stdout = channel.makefile("rb", -1).read().decode("utf-8", "replace")
        stderr = channel.makefile_stderr("rb", -1).read().decode("utf-8", "replace")
        exit_status = channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(f"Remote artifact inspection exited {exit_status}: {stderr[-2000:]}")
        payload = json.loads(stdout)
    finally:
        transport.close()
    if not isinstance(payload, Mapping):
        raise RuntimeError("Remote artifact inspection returned a non-object result")
    return dict(payload)


def materialize_factor_artifact_set(
    *,
    working_set: FactorWorkingSet,
    factor_names: Sequence[str],
    evaluation_start: str,
    evaluation_end: str,
    settings: Any,
    remote_artifact_dir: str,
    evaluator_factory: Callable[[Any], Callable[[FactorCandidate], EvaluationResult]] | None = None,
    inspector: Callable[[str], Mapping[str, Any]] | None = None,
) -> FactorArtifactSet:
    """Re-evaluate selected frozen expressions into a versioned remote folder."""

    if not factor_names or len(factor_names) != len(set(factor_names)):
        raise ValueError("factor_names must be a non-empty distinct sequence")
    available = {factor.name: factor for factor in working_set.factors}
    unknown = sorted(set(factor_names).difference(available))
    if unknown:
        raise ValueError(f"factors are outside the working set: {unknown[:3]}")

    versioned_settings = settings.model_copy(
        update={
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "remote_artifact_dir": remote_artifact_dir.rstrip("/\\"),
            "factor_evaluation_mode": "remote",
        }
    )
    evaluator = (
        evaluator_factory(versioned_settings)
        if evaluator_factory is not None
        else GFFactorEvaluator(settings=versioned_settings, mode="remote")
    )
    remote_settings = RemoteSettings.from_gf_settings(versioned_settings)
    inspect = inspector or (lambda path: inspect_remote_factor_artifact(path, settings=remote_settings))

    versions: list[FactorArtifactVersion] = []
    for name in factor_names:
        factor = available[name]
        result = evaluator(
            FactorCandidate(
                name=factor.name,
                family=factor.family,
                hypothesis="Versioned strategy-evaluation materialization of frozen factor logic.",
                expression=factor.expression,
                metadata={"snapshot_id": working_set.snapshot_id, "working_set_id": working_set.working_set_id},
            )
        )
        if not result.artifact_path or not result.artifact_checksum:
            raise RuntimeError(f"factor evaluator did not return frozen artifact provenance for {name}")
        provenance = dict(inspect(result.artifact_path))
        if provenance.get("checksum") != result.artifact_checksum:
            raise RuntimeError(f"remote artifact checksum mismatch after materializing {name}")
        version = FactorArtifactVersion(
            factor_id=factor.factor_id,
            name=factor.name,
            family=factor.family,
            expression=factor.expression,
            artifact_path=result.artifact_path,
            artifact_checksum=result.artifact_checksum,
            data_start=str(provenance["start"]),
            data_end=str(provenance["end"]),
            rows=int(provenance["rows"]),
            columns=int(provenance["columns"]),
        )
        if version.data_start > evaluation_start or version.data_end < evaluation_end:
            raise RuntimeError(f"materialized factor {name} does not cover {evaluation_start}..{evaluation_end}")
        versions.append(version)

    artifact_set_id = artifact_set_id_for(
        working_set.snapshot_id,
        working_set.working_set_id,
        evaluation_start,
        evaluation_end,
        versions,
    )
    return FactorArtifactSet(
        artifact_set_id=artifact_set_id,
        snapshot_id=working_set.snapshot_id,
        working_set_id=working_set.working_set_id,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        created_at=datetime.now(timezone.utc).isoformat(),
        factors=tuple(versions),
    )


class FactorArtifactCache:
    """Materialize each frozen factor at most once inside an outer epoch.

    Strategy trials generally overlap in their factor selections.  Re-running
    the remote factor worker for every candidate is both expensive and makes
    provenance harder to audit.  This cache stores immutable
    :class:`FactorArtifactVersion` objects and builds an exact manifest for the
    factor subset used by each candidate.
    """

    def __init__(
        self,
        *,
        working_set: FactorWorkingSet,
        evaluation_start: str,
        evaluation_end: str,
        settings: Any,
        remote_artifact_dir: str,
        materializer: Callable[..., FactorArtifactSet] = materialize_factor_artifact_set,
    ) -> None:
        source_dir = str(settings.remote_artifact_dir).rstrip("/\\")
        target_dir = remote_artifact_dir.rstrip("/\\")
        if not target_dir or target_dir == source_dir:
            raise ValueError("strategy factor artifacts require a versioned directory separate from source artifacts")
        self.working_set = working_set
        self.evaluation_start = evaluation_start
        self.evaluation_end = evaluation_end
        self.settings = settings
        self.remote_artifact_dir = target_dir
        self.materializer = materializer
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._versions: dict[str, FactorArtifactVersion] = {}

    @property
    def cached_factor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._versions))

    def artifact_set_for(self, factor_names: Sequence[str]) -> FactorArtifactSet:
        names = tuple(factor_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("factor_names must be a non-empty distinct sequence")
        available = {factor.name for factor in self.working_set.factors}
        unknown = sorted(set(names).difference(available))
        if unknown:
            raise ValueError(f"factors are outside the working set: {unknown[:3]}")

        for name in names:
            if name in self._versions:
                continue
            materialized = self.materializer(
                working_set=self.working_set,
                factor_names=(name,),
                evaluation_start=self.evaluation_start,
                evaluation_end=self.evaluation_end,
                settings=self.settings,
                remote_artifact_dir=self.remote_artifact_dir,
            )
            if (
                materialized.snapshot_id != self.working_set.snapshot_id
                or materialized.working_set_id != self.working_set.working_set_id
            ):
                raise RuntimeError(f"materialized factor {name} has mismatched frozen provenance")
            if len(materialized.factors) != 1 or materialized.factors[0].name != name:
                raise RuntimeError(f"materializer did not return exactly the requested factor {name}")
            self._versions[name] = materialized.factors[0]

        factors = tuple(self._versions[name] for name in names)
        return FactorArtifactSet(
            artifact_set_id=artifact_set_id_for(
                self.working_set.snapshot_id,
                self.working_set.working_set_id,
                self.evaluation_start,
                self.evaluation_end,
                factors,
            ),
            snapshot_id=self.working_set.snapshot_id,
            working_set_id=self.working_set.working_set_id,
            evaluation_start=self.evaluation_start,
            evaluation_end=self.evaluation_end,
            created_at=self.created_at,
            factors=factors,
        )
