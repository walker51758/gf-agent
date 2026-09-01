"""SQLite registry for evaluated GF portfolio strategies.

Deliberately a separate table (and default database file) from the factor
registry: a strategy row references factors instead of being one, and its
"expression" is a composition specification.  Keeping the two apart means the
factor schema, its ``family`` CHECK constraint and its acceptance metadata can
evolve without touching strategies, and vice versa.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

StrategyStatus = Literal["candidate", "accepted", "rejected", "duplicate", "retired"]


@dataclass(frozen=True)
class StrategyRecord:
    strategy_id: str
    name: str
    universe: str
    rule: str
    spec_json: str
    factor_names: list[str]
    status: StrategyStatus
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    artifact_checksum: str | None = None
    dedup_of: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class StrategyRegistry:
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
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    universe TEXT NOT NULL,
                    rule TEXT NOT NULL DEFAULT '',
                    spec_json TEXT NOT NULL,
                    factor_names_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL CHECK (
                        status IN ('candidate','accepted','rejected','duplicate','retired')
                    ),
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    artifact_path TEXT,
                    artifact_checksum TEXT,
                    dedup_of TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (dedup_of) REFERENCES strategies(strategy_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategies_universe_status ON strategies(universe, status)"
            )

    @staticmethod
    def _row(row: sqlite3.Row | None) -> StrategyRecord | None:
        if row is None:
            return None
        return StrategyRecord(
            strategy_id=row["strategy_id"],
            name=row["name"],
            universe=row["universe"],
            rule=row["rule"],
            spec_json=row["spec_json"],
            factor_names=json.loads(row["factor_names_json"]),
            status=row["status"],
            metrics=json.loads(row["metrics_json"]),
            artifact_path=row["artifact_path"],
            artifact_checksum=row["artifact_checksum"],
            dedup_of=row["dedup_of"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(self, record: StrategyRecord) -> StrategyRecord:
        now = datetime.now(timezone.utc).isoformat()
        created_at = record.created_at or now
        with self._connect() as connection:  # transactional commit/rollback
            connection.execute(
                """
                INSERT INTO strategies (
                    strategy_id, name, universe, rule, spec_json, factor_names_json,
                    status, metrics_json, artifact_path, artifact_checksum, dedup_of,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    name=excluded.name,
                    universe=excluded.universe,
                    rule=excluded.rule,
                    spec_json=excluded.spec_json,
                    factor_names_json=excluded.factor_names_json,
                    status=excluded.status,
                    metrics_json=excluded.metrics_json,
                    artifact_path=excluded.artifact_path,
                    artifact_checksum=excluded.artifact_checksum,
                    dedup_of=excluded.dedup_of,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.strategy_id,
                    record.name,
                    record.universe,
                    record.rule,
                    record.spec_json,
                    json.dumps(record.factor_names, ensure_ascii=False),
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
        stored = self.get(record.strategy_id)
        assert stored is not None
        return stored

    def get(self, strategy_id: str) -> StrategyRecord | None:
        with self._connect() as connection:
            return self._row(
                connection.execute("SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)).fetchone()
            )

    def get_by_name(self, name: str) -> StrategyRecord | None:
        with self._connect() as connection:
            return self._row(connection.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone())

    def list(self, *, status: StrategyStatus | None = None) -> list[StrategyRecord]:
        where = " WHERE status = ?" if status else ""
        parameters: tuple[str, ...] = (status,) if status else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM strategies{where} ORDER BY created_at, strategy_id", parameters
            ).fetchall()
        return [record for row in rows if (record := self._row(row)) is not None]


def strategy_id_for(name: str, spec: str | dict[str, Any]) -> str:
    """Stable identity for a strategy, mirroring :func:`factor_id_for`.

    The spec is hashed in its canonical JSON form so two candidates that differ
    only in dict ordering or in a defaulted field of an identical value produce
    the same id, while any real rule/weighting/lag change produces a new one.
    """

    payload = spec if isinstance(spec, str) else json.dumps(spec, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(f"{name}\n{payload}".encode()).hexdigest()[:16]
    return f"gf_strategy_{digest}"
