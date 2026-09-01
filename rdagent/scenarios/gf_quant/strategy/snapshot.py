"""Immutable factor snapshots consumed by the GF strategy loop.

The factor registry remains the complete mutable history.  A strategy epoch
never reads it directly after starting: it freezes every research-eligible
factor into a :class:`FactorUniverseSnapshot`, then selects a smaller
:class:`FactorWorkingSet` for the inner strategy trials.  New factor records
therefore become visible only when the next epoch creates a new snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from rdagent.scenarios.gf_quant.registry import FactorRecord, FactorRegistry

EligibilityTier = Literal["core", "conditional", "ineligible"]

ELIGIBILITY_TIERS: tuple[str, ...] = ("core", "conditional", "ineligible")

# These failures mean that the factor cannot safely be composed into a
# portfolio.  Ordinary single-factor quality failures (RankIC, return, Sharpe,
# drawdown or turnover) deliberately do not appear here: those are exactly the
# factors that deserve a second chance in the conditional pool.
_HARD_FAILURE_TOKENS: tuple[str, ...] = (
    "artifact",
    "checksum",
    "coverage",
    "observation",
    "pit",
    "point_in_time",
    "remote",
    "reproduc",
    "data_missing",
    "metrics_by_year_missing",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(metrics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _finite_number(metrics.get(name))
        if value is not None:
            return value
    return None


def _metadata_text(metadata: Mapping[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if value is not None and str(value).strip() else fallback


def factor_eligibility(record: FactorRecord) -> tuple[EligibilityTier, str]:
    """Classify a registry record without changing the factor registry schema."""

    explicit = record.metadata.get("eligibility_tier") if isinstance(record.metadata, Mapping) else None
    if explicit in ELIGIBILITY_TIERS:
        return explicit, "explicit_registry_metadata"

    if not record.artifact_path or not record.artifact_checksum:
        return "ineligible", "missing_frozen_artifact"
    if record.status == "accepted":
        return "core", "accepted_by_factor_gate"
    if record.status != "rejected":
        return "ineligible", f"factor_status_{record.status}"

    failures = record.metadata.get("gate_failures", []) if isinstance(record.metadata, Mapping) else []
    failure_names = [str(value).lower() for value in failures] if isinstance(failures, Sequence) else []
    if any(token in failure for failure in failure_names for token in _HARD_FAILURE_TOKENS):
        return "ineligible", "hard_factor_contract_failure"
    return "conditional", "standalone_quality_failure_only"


def _factor_direction(metrics: Mapping[str, Any]) -> int:
    value = _metric(metrics, "Rank_IC", "rank_ic", "IC", "ic")
    return -1 if value is not None and value < 0 else 1


def _quality_score(metrics: Mapping[str, Any]) -> float:
    rank_ic = abs(_metric(metrics, "Rank_IC", "rank_ic") or 0.0)
    rank_icir = abs(_metric(metrics, "Rank_ICIR", "rank_icir") or 0.0)
    sharpe = max(_metric(metrics, "sharpe") or 0.0, 0.0)
    coverage = max(_metric(metrics, "coverage") or 0.0, 0.0)
    return rank_ic + 0.25 * rank_icir + 0.05 * sharpe + 0.01 * coverage


def _stability_score(metrics: Mapping[str, Any], direction: int) -> float:
    by_year = metrics.get("metrics_by_year")
    if not isinstance(by_year, Mapping):
        return abs(_metric(metrics, "Rank_IC", "rank_ic") or 0.0)
    signs: list[int] = []
    for values in by_year.values():
        if not isinstance(values, Mapping):
            continue
        value = _metric(values, "Rank_IC", "rank_ic")
        if value is not None and value != 0:
            signs.append(1 if value > 0 else -1)
    return sum(sign == direction for sign in signs) / len(signs) if signs else 0.0


def _cluster(metadata: Mapping[str, Any]) -> str:
    direct = metadata.get("correlation_cluster") or metadata.get("cluster_id")
    if direct is not None:
        return str(direct)
    for key in ("benchmark_redundancy", "benchmark_calibration", "calibration"):
        nested = metadata.get(key)
        if isinstance(nested, Mapping):
            value = nested.get("cluster_id") or nested.get("cluster")
            if value is not None:
                return str(value)
    return ""


@dataclass(frozen=True)
class FactorSnapshotItem:
    factor_id: str
    name: str
    family: str
    expression: str
    eligibility_tier: Literal["core", "conditional"]
    eligibility_reason: str
    artifact_path: str
    artifact_checksum: str
    direction: int
    theme: str
    intended_role: str
    correlation_cluster: str
    quality_score: float
    stability_score: float
    metrics_json: str
    source_updated_at: str

    def __post_init__(self) -> None:
        if self.eligibility_tier not in ("core", "conditional"):
            raise ValueError("snapshot items must be core or conditional")
        if self.direction not in (-1, 1):
            raise ValueError("factor direction must be -1 or 1")
        if not self.factor_id or not self.name or not self.artifact_path or not self.artifact_checksum:
            raise ValueError("snapshot factors require identity and frozen artifact provenance")
        json.loads(self.metrics_json)

    @property
    def metrics(self) -> dict[str, Any]:
        return json.loads(self.metrics_json)

    def as_payload(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "family": self.family,
            "expression": self.expression,
            "eligibility_tier": self.eligibility_tier,
            "eligibility_reason": self.eligibility_reason,
            "artifact_path": self.artifact_path,
            "artifact_checksum": self.artifact_checksum,
            "direction": self.direction,
            "theme": self.theme,
            "intended_role": self.intended_role,
            "correlation_cluster": self.correlation_cluster,
            "quality_score": self.quality_score,
            "stability_score": self.stability_score,
            "metrics": self.metrics,
            "source_updated_at": self.source_updated_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FactorSnapshotItem":
        values = dict(payload)
        metrics = values.pop("metrics", {})
        return cls(metrics_json=_canonical_json(metrics), **values)


def _item_from_record(record: FactorRecord, tier: Literal["core", "conditional"], reason: str) -> FactorSnapshotItem:
    metrics = dict(record.metrics)
    metadata = dict(record.metadata)
    direction = _factor_direction(metrics)
    return FactorSnapshotItem(
        factor_id=record.factor_id,
        name=record.name,
        family=record.family,
        expression=record.expression,
        eligibility_tier=tier,
        eligibility_reason=reason,
        artifact_path=str(record.artifact_path),
        artifact_checksum=str(record.artifact_checksum),
        direction=direction,
        theme=_metadata_text(metadata, "economic_theme", _metadata_text(metadata, "search_direction", record.family)),
        intended_role=_metadata_text(metadata, "intended_role", "alpha"),
        correlation_cluster=_cluster(metadata),
        quality_score=_quality_score(metrics),
        stability_score=_stability_score(metrics, direction),
        metrics_json=_canonical_json(metrics),
        source_updated_at=record.updated_at or record.created_at or "",
    )


def _snapshot_body(data_cutoff: str, factors: Sequence[FactorSnapshotItem]) -> dict[str, Any]:
    return {
        "data_cutoff": data_cutoff,
        "factors": [factor.as_payload() for factor in sorted(factors, key=lambda item: item.factor_id)],
    }


@dataclass(frozen=True)
class FactorUniverseSnapshot:
    snapshot_id: str
    checksum: str
    created_at: str
    data_cutoff: str
    factors: tuple[FactorSnapshotItem, ...]

    def __post_init__(self) -> None:
        factor_ids = [factor.factor_id for factor in self.factors]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("snapshot contains duplicate factor ids")
        expected = hashlib.sha256(_canonical_json(_snapshot_body(self.data_cutoff, self.factors)).encode()).hexdigest()
        if self.checksum != expected or self.snapshot_id != f"gf_snapshot_{expected[:16]}":
            raise ValueError("snapshot checksum or id does not match its frozen contents")

    def as_payload(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "checksum": self.checksum,
            "created_at": self.created_at,
            **_snapshot_body(self.data_cutoff, self.factors),
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FactorUniverseSnapshot":
        factors = tuple(FactorSnapshotItem.from_payload(value) for value in payload.get("factors", []))
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            checksum=str(payload["checksum"]),
            created_at=str(payload["created_at"]),
            data_cutoff=str(payload["data_cutoff"]),
            factors=factors,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "FactorUniverseSnapshot":
        return cls.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def build_factor_universe_snapshot(
    registry: FactorRegistry,
    *,
    data_cutoff: str,
    created_at: str | None = None,
) -> FactorUniverseSnapshot:
    """Freeze all core and conditional factors currently in the registry."""

    if not data_cutoff:
        raise ValueError("data_cutoff is required for a point-in-time snapshot")
    items: list[FactorSnapshotItem] = []
    for record in registry.list():
        tier, reason = factor_eligibility(record)
        if tier in ("core", "conditional"):
            items.append(_item_from_record(record, tier, reason))
    body = _snapshot_body(data_cutoff, items)
    checksum = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
    return FactorUniverseSnapshot(
        snapshot_id=f"gf_snapshot_{checksum[:16]}",
        checksum=checksum,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        data_cutoff=data_cutoff,
        factors=tuple(sorted(items, key=lambda item: item.factor_id)),
    )


def _diverse_take(factors: Sequence[FactorSnapshotItem], limit: int) -> list[FactorSnapshotItem]:
    """Take strong factors round-robin across cluster/theme buckets."""

    if limit <= 0:
        return []
    buckets: dict[str, deque[FactorSnapshotItem]] = defaultdict(deque)
    ordered = sorted(factors, key=lambda item: (-item.quality_score, -item.stability_score, item.factor_id))
    for factor in ordered:
        key = factor.correlation_cluster or factor.theme or factor.family
        buckets[key].append(factor)
    keys = sorted(buckets, key=lambda key: (-buckets[key][0].quality_score, key))
    selected: list[FactorSnapshotItem] = []
    while keys and len(selected) < limit:
        next_keys: list[str] = []
        for key in keys:
            selected.append(buckets[key].popleft())
            if buckets[key]:
                next_keys.append(key)
            if len(selected) >= limit:
                break
        keys = next_keys
    return selected


@dataclass(frozen=True)
class FactorWorkingSet:
    working_set_id: str
    snapshot_id: str
    snapshot_checksum: str
    core_limit: int
    conditional_limit: int
    factors: tuple[FactorSnapshotItem, ...]

    def __post_init__(self) -> None:
        factor_ids = [factor.factor_id for factor in self.factors]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("working set contains duplicate factors")
        if sum(factor.eligibility_tier == "core" for factor in self.factors) > self.core_limit:
            raise ValueError("working set exceeds core limit")
        if sum(factor.eligibility_tier == "conditional" for factor in self.factors) > self.conditional_limit:
            raise ValueError("working set exceeds conditional limit")
        expected = working_set_id_for(self.snapshot_checksum, factor_ids, self.core_limit, self.conditional_limit)
        if self.working_set_id != expected:
            raise ValueError("working set id does not match its contents")

    def as_payload(self) -> dict[str, Any]:
        return {
            "working_set_id": self.working_set_id,
            "snapshot_id": self.snapshot_id,
            "snapshot_checksum": self.snapshot_checksum,
            "core_limit": self.core_limit,
            "conditional_limit": self.conditional_limit,
            "factors": [factor.as_payload() for factor in self.factors],
        }


def working_set_id_for(snapshot_checksum: str, factor_ids: Sequence[str], core_limit: int, conditional_limit: int) -> str:
    payload = {
        "snapshot_checksum": snapshot_checksum,
        "factor_ids": list(factor_ids),
        "core_limit": core_limit,
        "conditional_limit": conditional_limit,
    }
    return f"gf_working_set_{hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:16]}"


def build_factor_working_set(
    snapshot: FactorUniverseSnapshot,
    *,
    core_limit: int = 16,
    conditional_limit: int = 8,
) -> FactorWorkingSet:
    """Select the default 16 core + 8 conditional factors deterministically."""

    if core_limit < 0 or conditional_limit < 0 or core_limit + conditional_limit < 1:
        raise ValueError("working-set limits must be non-negative with a positive total")
    core = _diverse_take([factor for factor in snapshot.factors if factor.eligibility_tier == "core"], core_limit)
    conditional = _diverse_take(
        [factor for factor in snapshot.factors if factor.eligibility_tier == "conditional"], conditional_limit
    )
    factors = tuple([*core, *conditional])
    factor_ids = [factor.factor_id for factor in factors]
    return FactorWorkingSet(
        working_set_id=working_set_id_for(snapshot.checksum, factor_ids, core_limit, conditional_limit),
        snapshot_id=snapshot.snapshot_id,
        snapshot_checksum=snapshot.checksum,
        core_limit=core_limit,
        conditional_limit=conditional_limit,
        factors=factors,
    )
