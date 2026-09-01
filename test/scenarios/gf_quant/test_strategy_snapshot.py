from __future__ import annotations

import json

import pytest

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry
from rdagent.scenarios.gf_quant.strategy.snapshot import (
    FactorUniverseSnapshot,
    build_factor_universe_snapshot,
    build_factor_working_set,
    factor_eligibility,
)


def _record(
    factor_id: str,
    *,
    status: str = "accepted",
    rank_ic: float = 0.03,
    failures: list[str] | None = None,
    theme: str = "value",
    artifact: bool = True,
) -> FactorRecord:
    return FactorRecord(
        factor_id=factor_id,
        name=f"factor_{factor_id}",
        family="fundamental",
        expression=f"cs_rank(field_{factor_id})",
        status=status,
        metrics={"Rank_IC": rank_ic, "Rank_ICIR": rank_ic * 10, "coverage": 0.99},
        artifact_path=f"G:/artifacts/{factor_id}.parquet" if artifact else None,
        artifact_checksum=(factor_id * 64)[:64] if artifact else None,
        metadata={"gate_failures": failures or [], "search_direction": theme},
    )


def test_factor_eligibility_separates_quality_from_hard_failures() -> None:
    assert factor_eligibility(_record("a"))[0] == "core"
    assert factor_eligibility(_record("b", status="rejected", failures=["rank_ic", "sharpe"]))[0] == "conditional"
    assert factor_eligibility(_record("c", status="rejected", failures=["coverage"]))[0] == "ineligible"
    assert factor_eligibility(_record("d", artifact=False))[0] == "ineligible"
    assert factor_eligibility(_record("e", status="duplicate"))[0] == "ineligible"


def test_snapshot_freezes_registry_provenance_and_detects_tampering(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    registry.upsert(_record("a", rank_ic=-0.04))
    registry.upsert(_record("b", status="rejected", failures=["rank_ic"]))
    registry.upsert(_record("c", status="rejected", failures=["coverage"]))

    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31", created_at="2024-01-01T00:00:00Z")

    assert len(snapshot.factors) == 2
    assert {factor.eligibility_tier for factor in snapshot.factors} == {"core", "conditional"}
    assert next(factor for factor in snapshot.factors if factor.factor_id == "a").direction == -1
    assert all(factor.artifact_checksum for factor in snapshot.factors)

    path = snapshot.write_json(tmp_path / "snapshot.json")
    assert FactorUniverseSnapshot.read_json(path) == snapshot

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["factors"][0]["artifact_checksum"] = "tampered"
    with pytest.raises(ValueError, match="checksum"):
        FactorUniverseSnapshot.from_payload(payload)


def test_working_set_is_bounded_diverse_and_snapshot_scoped(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    for position in range(8):
        registry.upsert(_record(f"c{position}", rank_ic=0.08 - position * 0.005, theme=f"theme{position % 3}"))
    for position in range(5):
        registry.upsert(
            _record(
                f"q{position}",
                status="rejected",
                rank_ic=0.04 - position * 0.005,
                failures=["rank_ic"],
                theme=f"conditional{position % 2}",
            )
        )

    snapshot = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")
    working = build_factor_working_set(snapshot, core_limit=4, conditional_limit=2)

    assert len(working.factors) == 6
    assert sum(factor.eligibility_tier == "core" for factor in working.factors) == 4
    assert sum(factor.eligibility_tier == "conditional" for factor in working.factors) == 2
    assert working.snapshot_id == snapshot.snapshot_id
    assert working.snapshot_checksum == snapshot.checksum
    assert len({factor.theme for factor in working.factors[:3]}) == 3


def test_new_registry_factor_only_enters_the_next_snapshot(tmp_path) -> None:
    registry = FactorRegistry(tmp_path / "factors.sqlite")
    registry.upsert(_record("old"))
    first = build_factor_universe_snapshot(registry, data_cutoff="2023-12-31")

    registry.upsert(_record("new"))
    second = build_factor_universe_snapshot(registry, data_cutoff="2024-01-31")

    assert {factor.factor_id for factor in first.factors} == {"old"}
    assert {factor.factor_id for factor in second.factors} == {"old", "new"}
    assert first.snapshot_id != second.snapshot_id
