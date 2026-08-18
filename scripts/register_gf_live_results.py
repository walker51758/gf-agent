"""Apply the V1 gate and register remote GF factor evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdagent.scenarios.gf_quant.dedup import FactorSignature
from rdagent.scenarios.gf_quant.registry import FactorRegistry, factor_id_for
from rdagent.scenarios.gf_quant.service import (
    EvaluationResult,
    FactorAcceptanceService,
    FactorCandidate,
)


def register_result(path: Path, registry: FactorRegistry) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = FactorCandidate(
        name=payload["name"],
        family=payload["family"],
        expression=payload["expression"],
        metadata={"evaluation": payload.get("evaluation", {}), "source_result": str(path)},
    )
    evaluation = EvaluationResult(
        metrics=payload["metrics"],
        artifact_path=payload.get("artifact_path"),
        artifact_checksum=payload.get("artifact_checksum"),
    )
    service = FactorAcceptanceService(registry, lambda _: evaluation)
    current_id = factor_id_for(candidate.name, candidate.expression)
    existing = [
        FactorSignature(record.factor_id, record.expression)
        for record in registry.list(status="accepted")
        if record.factor_id != current_id
    ]
    processed = service.process(candidate, existing_signatures=existing)
    return f"{processed.record.factor_id} {processed.record.family} {processed.record.status} {processed.record.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("git_ignore_folder/gf_quant/factor_registry.sqlite"))
    parser.add_argument("--export", type=Path, help="Optional JSON snapshot of registry metadata")
    args = parser.parse_args()
    registry = FactorRegistry(args.registry)
    for result in args.results:
        print(register_result(result, registry))
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "factor_id": record.factor_id,
                "name": record.name,
                "family": record.family,
                "expression": record.expression,
                "status": record.status,
                "metrics": record.metrics,
                "artifact_path": record.artifact_path,
                "artifact_checksum": record.artifact_checksum,
                "dedup_of": record.dedup_of,
                "metadata": record.metadata,
            }
            for record in registry.list()
        ]
        args.export.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accepted={len(registry.list(status='accepted'))} registry={args.registry}")


if __name__ == "__main__":
    main()
