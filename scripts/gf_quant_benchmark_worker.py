"""Standalone remote benchmark worker for GF factor calibration.

The controller uploads this file next to ``gf_quant_remote_worker.py`` on the
pinned GF data host and runs it once per benchmark source directory.  It walks
the *existing* wide parquet panels one at a time, reuses the candidate metric
definitions verbatim (``daily_series`` / ``metrics_from_daily``), and writes one
compact JSON summary per factor plus a checkpoint manifest.  Factor panels are
never written and never transferred; only these summaries are downloaded.

Every summary is written immediately after its factor finishes, so the run can
be interrupted and resumed: existing summaries are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

# The controller delivers this file next to gf_quant_remote_worker.py on the
# remote host, so the local directory is on the import path first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_quant_remote_worker as evaluator  # noqa: E402  (deliberately local)

MANIFEST_NAME = "_manifest.json"
PROFILE_SOURCE_KEYS = ("dt", "trade_dt", "date", "datetime")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _index_column(path: Path) -> str:
    """Return the datetime index column of a wide panel written by pandas."""
    schema = pq.ParquetFile(path).schema_arrow
    metadata = schema.metadata or {}
    raw = metadata.get(b"pandas")
    if raw:
        try:
            index_columns = json.loads(raw.decode("utf-8")).get("index_columns") or []
        except (ValueError, UnicodeDecodeError):
            index_columns = []
        if index_columns:
            return str(index_columns[0])
    for name in PROFILE_SOURCE_KEYS:
        if name in schema.names:
            return name
    raise ValueError(f"No datetime index column found in {path.name}")


def load_wide_panel(path: Path, start: str, end: str) -> pd.DataFrame:
    """Read one wide factor panel restricted to ``[start, end]``.

    Only the rows inside the screening window are materialised; symbols are
    kept as columns so the panel can be evaluated exactly like a candidate
    factor.
    """
    index_column = _index_column(path)
    start_ts = evaluator._as_timestamp(start)
    end_ts = evaluator._as_timestamp(end)
    table = pq.read_table(
        path,
        filters=[(index_column, ">=", start_ts.to_pydatetime()), (index_column, "<=", end_ts.to_pydatetime())],
    )
    frame = table.to_pandas()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame[index_column]))
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.astype(float)


def _json_safe(value: Any) -> Any:
    return evaluator._json_safe(value)


def evaluate_benchmark_file(
    path: Path,
    close: pd.DataFrame,
    cfg: evaluator.EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate one existing wide panel and return its compact summary."""
    factor = load_wide_panel(path, cfg.start, cfg.end)
    if factor.empty:
        return {"status": "excluded", "exclusion_reason": "empty_panel_in_window", "metrics": {}, "metrics_by_year": {}}
    daily, restricted = evaluator.daily_series(factor, close, cfg)
    metrics = evaluator.metrics_from_daily(daily, restricted.index)
    summary = {
        "status": "ok",
        "exclusion_reason": None,
        "metrics": metrics,
        "metrics_by_year": metrics.get("metrics_by_year", {}),
        "profile": evaluator.compact_profile(daily),
        "start_date": pd.Timestamp(restricted.index.min()).strftime("%Y-%m-%d") if len(restricted.index) else None,
        "end_date": pd.Timestamp(restricted.index.max()).strftime("%Y-%m-%d") if len(restricted.index) else None,
        "panel_rows": int(len(restricted.index)),
        "panel_columns": int(restricted.shape[1]),
    }
    return summary


def evaluate_source(
    family: str,
    source_dir: Path,
    out_dir: Path,
    cfg: evaluator.EvaluationConfig,
    *,
    base_path: str,
    close_field: str,
    force: bool = False,
    progress_every: int = 10,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate every parquet panel of one source directory, checkpointed."""
    source_dir = Path(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(source_dir.glob("*.parquet"))
    if limit is not None:
        files = files[:limit]
    manifest_path = out_dir / MANIFEST_NAME
    manifest: dict[str, Any] = {
        "family": family,
        "source_dir": str(source_dir).replace("\\", "/"),
        "total_files": len(files),
        "completed": 0,
        "excluded": 0,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
    }

    # Close data is shared by every factor of the source: load it once.
    loader = evaluator.PanelLoader(base_path, cfg.start, cfg.end, lookback_days=10)
    close = loader.load(close_field)

    started = time.time()
    for position, path in enumerate(files, start=1):
        name = path.stem
        result_path = out_dir / f"{name}.json"
        if result_path.exists() and not force:
            manifest["completed"] += 1
            continue
        record: dict[str, Any] = {
            "name": name,
            "family": family,
            "source_dir": manifest["source_dir"],
            "file_name": path.name,
            "file_size_bytes": int(path.stat().st_size),
            "close_field": close_field,
            "evaluation": {
                "start": cfg.start,
                "end": cfg.end,
                "label_horizon": cfg.label_horizon,
                "quantile": cfg.quantile,
                "min_cross_section": cfg.min_cross_section,
                "sample_step": cfg.sample_step,
            },
            "generated_at": _utc_now(),
        }
        try:
            record.update(evaluate_benchmark_file(path, close, cfg))
        except Exception as exc:  # noqa: BLE001 - one bad panel must not stop the run
            record.update(
                {
                    "status": "excluded",
                    "exclusion_reason": f"worker_error: {type(exc).__name__}: {exc}",
                    "metrics": {},
                    "metrics_by_year": {},
                }
            )
        record = json.loads(json.dumps(_json_safe(record), ensure_ascii=False))
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        if record["status"] == "ok":
            manifest["completed"] += 1
        else:
            manifest["excluded"] += 1
        manifest["updated_at"] = _utc_now()
        manifest["elapsed_seconds"] = round(time.time() - started, 1)
        if position % progress_every == 0 or position == len(files):
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            rate = position / max(manifest["elapsed_seconds"], 1e-9)
            remaining = (len(files) - position) / max(rate, 1e-9)
            print(
                f"PROGRESS family={family} source={source_dir.name} {position}/{len(files)} "
                f"elapsed_s={manifest['elapsed_seconds']} eta_s={remaining:.0f}",
                flush=True,
            )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing GF benchmark panels into compact summaries")
    parser.add_argument(
        "--family-dir",
        action="append",
        required=True,
        metavar="FAMILY=DIR",
        help="Benchmark source directory, e.g. price_volume=G:/gf_quant_agent/artifacts",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("G:/gf_quant_agent/benchmark"))
    parser.add_argument("--base-path", default="G:/base")
    parser.add_argument("--close-field", default="adjclose")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--min-cross-section", type=int, default=100)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Re-evaluate factors that already have a summary")
    parser.add_argument("--only-source", default=None, help="Restrict the run to a source directory substring")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test cap on the number of panels per source")
    args = parser.parse_args()

    cfg = evaluator.EvaluationConfig(
        start=args.start,
        end=args.end,
        label_horizon=args.label_horizon,
        quantile=args.quantile,
        min_cross_section=args.min_cross_section,
        sample_step=args.sample_step,
    )
    for binding in args.family_dir:
        family, _, raw_dir = binding.partition("=")
        if not family or not raw_dir:
            raise SystemExit(f"--family-dir must be FAMILY=DIR, got {binding!r}")
        if family not in ("price_volume", "fundamental"):
            raise SystemExit(f"Unsupported family: {family}")
        if args.only_source and args.only_source not in raw_dir:
            continue
        manifest = evaluate_source(
            family,
            Path(raw_dir),
            args.out_dir / family / Path(raw_dir).name,
            cfg,
            base_path=args.base_path,
            close_field=args.close_field,
            force=args.force,
            limit=args.limit,
        )
        print(json.dumps({"source_manifest": manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
