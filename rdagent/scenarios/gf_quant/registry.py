"""SQLite metadata registry for evaluated GF factors."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


FactorFamily = Literal["price_volume", "fundamental"]
FactorStatus = Literal["candidate", "accepted", "rejected", "duplicate", "retired"]


@dataclass(frozen=True)
class FactorRecord:
    factor_id: str
    name: str
    family: FactorFamily
    expression: str
    status: FactorStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_checksum: str | None = None
    dedup_of: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class FactorRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS factors (
                    factor_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    family TEXT NOT NULL CHECK (family IN ('price_volume', 'fundamental')),
                    expression TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('candidate','accepted','rejected','duplicate','retired')),
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    artifact_path TEXT,
                    artifact_checksum TEXT,
                    dedup_of TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (dedup_of) REFERENCES factors(factor_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_factors_family_status ON factors(family, status)")

    @staticmethod
    def _row(row: sqlite3.Row | None) -> FactorRecord | None:
        if row is None:
            return None
        return FactorRecord(
            factor_id=row["factor_id"],
            name=row["name"],
            family=row["family"],
            expression=row["expression"],
            status=row["status"],
            metrics=json.loads(row["metrics_json"]),
            artifact_path=row["artifact_path"],
            artifact_checksum=row["artifact_checksum"],
            dedup_of=row["dedup_of"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(self, record: FactorRecord) -> FactorRecord:
        now = datetime.now(timezone.utc).isoformat()
        created_at = record.created_at or now
        with self._connect() as connection:  # transactional commit/rollback
            connection.execute(
                """
                INSERT INTO factors (
                    factor_id, name, family, expression, status, metrics_json,
                    artifact_path, artifact_checksum, dedup_of, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(factor_id) DO UPDATE SET
                    name=excluded.name,
                    family=excluded.family,
                    expression=excluded.expression,
                    status=excluded.status,
                    metrics_json=excluded.metrics_json,
                    artifact_path=excluded.artifact_path,
                    artifact_checksum=excluded.artifact_checksum,
                    dedup_of=excluded.dedup_of,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.factor_id,
                    record.name,
                    record.family,
                    record.expression,
                    record.status,
                    json.dumps(record.metrics, ensure_ascii=False, sort_keys=True),
                    record.artifact_path,
                    record.artifact_checksum,
                    record.dedup_of,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    created_at,
                    now,
                ),
            )
        stored = self.get(record.factor_id)
        assert stored is not None
        return stored

    def get(self, factor_id: str) -> FactorRecord | None:
        with self._connect() as connection:
            return self._row(connection.execute("SELECT * FROM factors WHERE factor_id = ?", (factor_id,)).fetchone())

    def get_by_name(self, name: str) -> FactorRecord | None:
        with self._connect() as connection:
            return self._row(connection.execute("SELECT * FROM factors WHERE name = ?", (name,)).fetchone())

    def list(self, *, family: FactorFamily | None = None, status: FactorStatus | None = None) -> list[FactorRecord]:
        clauses: list[str] = []
        parameters: list[str] = []
        if family:
            clauses.append("family = ?")
            parameters.append(family)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(f"SELECT * FROM factors{where} ORDER BY created_at, factor_id", parameters).fetchall()
        return [record for row in rows if (record := self._row(row)) is not None]


def factor_id_for(name: str, expression: str) -> str:
    import hashlib

    digest = hashlib.sha256(f"{name}\n{expression}".encode("utf-8")).hexdigest()[:16]
    return f"gf_{digest}"
