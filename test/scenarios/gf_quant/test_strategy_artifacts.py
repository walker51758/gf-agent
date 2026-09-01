from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.service import EvaluationResult
from rdagent.scenarios.gf_quant.strategy.artifacts import (
    FactorArtifactCache,
    FactorArtifactSet,
    FactorArtifactVersion,
    artifact_set_id_for,
    materialize_factor_artifact_set,
)
from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set


class Settings(BaseModel):
    evaluation_start: str = "2020-01-01"
    evaluation_end: str = "2023-12-31"
    remote_artifact_dir: str = "G:/old"
    factor_evaluation_mode: str = "remote"
    data_host: str = "host"
    data_port: int = 22
    data_user: str = "user"
    data_password_env: str = "PASSWORD"
    data_host_key_sha256: str = ""
    remote_python: str = "python"
    remote_work_dir: str = "work"


def _working_set(tmp_path):
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    for position, name in enumerate(("value", "quality")):
        registry.upsert(
            FactorRecord(
                factor_id=f"f{position}",
                name=name,
                family="fundamental",
                expression=f"cs_rank({name})",
                status="accepted",
                metrics={"Rank_IC": 0.02},
                artifact_path=f"G:/old/{name}.parquet",
                artifact_checksum=str(position + 1) * 64,
            )
        )
    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")
    return build_factor_working_set(snapshot, core_limit=2, conditional_limit=0)


def test_materialization_creates_versioned_checksummed_manifest(tmp_path) -> None:
    working = _working_set(tmp_path)
    seen_settings = []

    def factory(settings):
        seen_settings.append(settings)

        def evaluate(candidate):
            return EvaluationResult(
                metrics={},
                artifact_path=f"G:/versioned/{candidate.name}.parquet",
                artifact_checksum=("a" if candidate.name == "value" else "b") * 64,
            )

        return evaluate

    def inspect(path):
        name = path.rsplit("/", 1)[-1].split(".", 1)[0]
        return {
            "checksum": ("a" if name == "value" else "b") * 64,
            "start": "2024-01-01",
            "end": "2025-12-31",
            "rows": 500,
            "columns": 4000,
        }

    artifact_set = materialize_factor_artifact_set(
        working_set=working,
        factor_names=("value", "quality"),
        evaluation_start="2024-01-01",
        evaluation_end="2025-12-31",
        settings=Settings(),
        remote_artifact_dir="G:/versioned",
        evaluator_factory=factory,
        inspector=inspect,
    )

    assert artifact_set.artifact_set_id.startswith("gf_artifact_set_")
    assert len(artifact_set.factors) == 2
    assert seen_settings[0].remote_artifact_dir == "G:/versioned"
    assert all(factor.data_start == "2024-01-01" for factor in artifact_set.factors)
    path = artifact_set.write_json(tmp_path / "artifact_set.json")
    assert FactorArtifactSet.read_json(path) == artifact_set


def test_materialization_rejects_incomplete_or_tampered_artifacts(tmp_path) -> None:
    working = _working_set(tmp_path)

    def factory(settings):
        return lambda candidate: EvaluationResult(
            metrics={}, artifact_path=f"G:/{candidate.name}.parquet", artifact_checksum="a" * 64
        )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        materialize_factor_artifact_set(
            working_set=working,
            factor_names=("value",),
            evaluation_start="2024-01-01",
            evaluation_end="2025-12-31",
            settings=Settings(),
            remote_artifact_dir="G:/versioned",
            evaluator_factory=factory,
            inspector=lambda path: {
                "checksum": "b" * 64,
                "start": "2024-01-01",
                "end": "2025-12-31",
                "rows": 500,
                "columns": 4000,
            },
        )

    with pytest.raises(RuntimeError, match="does not cover"):
        materialize_factor_artifact_set(
            working_set=working,
            factor_names=("value",),
            evaluation_start="2024-01-01",
            evaluation_end="2025-12-31",
            settings=Settings(),
            remote_artifact_dir="G:/versioned",
            evaluator_factory=factory,
            inspector=lambda path: {
                "checksum": "a" * 64,
                "start": "2024-06-01",
                "end": "2025-12-31",
                "rows": 400,
                "columns": 4000,
            },
        )


def test_epoch_artifact_cache_materializes_each_factor_once(tmp_path) -> None:
    working = _working_set(tmp_path)
    calls = []

    def materializer(**kwargs):
        calls.append(kwargs)
        name = kwargs["factor_names"][0]
        factor = next(value for value in working.factors if value.name == name)
        version = FactorArtifactVersion(
            factor_id=factor.factor_id,
            name=name,
            family=factor.family,
            expression=factor.expression,
            artifact_path=f"G:/strategy_versioned/{name}.parquet",
            artifact_checksum=("a" if name == "value" else "b") * 64,
            data_start="2024-01-01",
            data_end="2025-12-31",
            rows=500,
            columns=4000,
        )
        return FactorArtifactSet(
            artifact_set_id=artifact_set_id_for(
                working.snapshot_id,
                working.working_set_id,
                "2024-01-01",
                "2025-12-31",
                (version,),
            ),
            snapshot_id=working.snapshot_id,
            working_set_id=working.working_set_id,
            evaluation_start="2024-01-01",
            evaluation_end="2025-12-31",
            created_at="2024-01-01T00:00:00Z",
            factors=(version,),
        )

    cache = FactorArtifactCache(
        working_set=working,
        evaluation_start="2024-01-01",
        evaluation_end="2025-12-31",
        settings=Settings(),
        remote_artifact_dir="G:/strategy_versioned",
        materializer=materializer,
    )
    first = cache.artifact_set_for(("value", "quality"))
    second = cache.artifact_set_for(("quality",))

    assert [call["factor_names"] for call in calls] == [("value",), ("quality",)]
    assert all(call["remote_artifact_dir"] == "G:/strategy_versioned" for call in calls)
    assert [factor.name for factor in first.factors] == ["value", "quality"]
    assert [factor.name for factor in second.factors] == ["quality"]
    assert cache.cached_factor_names == ("quality", "value")


def test_epoch_artifact_cache_rejects_original_artifact_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="separate from source artifacts"):
        FactorArtifactCache(
            working_set=_working_set(tmp_path),
            evaluation_start="2024-01-01",
            evaluation_end="2025-12-31",
            settings=Settings(),
            remote_artifact_dir="G:/old",
        )
