"""Run one materialized strategy on the pinned remote host and save compact evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--artifact-set", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-cutoff", default="2023-12-31")
    parser.add_argument("--rebalance-frequency", default="monthly", choices=["daily", "weekly", "monthly", "quarterly"])
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.remote import RemoteSettings
    from rdagent.scenarios.gf_quant.strategy.artifacts import FactorArtifactSet
    from rdagent.scenarios.gf_quant.strategy.candidate import StrategyCandidate
    from rdagent.scenarios.gf_quant.strategy.remote import RemoteStrategyEvaluator, evaluate_strategy_remote
    from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set

    candidate_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    candidate = StrategyCandidate.from_payload(candidate_payload)
    artifact_set = FactorArtifactSet.read_json(args.artifact_set)
    registry = FactorRegistry(GF_SCENARIO_SETTINGS.registry_path)
    snapshot = build_factor_universe_snapshot(registry, data_cutoff=args.data_cutoff)
    working_set = build_factor_working_set(snapshot)
    if candidate_payload.get("snapshot_id") not in (None, snapshot.snapshot_id):
        raise RuntimeError("candidate snapshot_id does not match current reconstructed snapshot")
    if candidate_payload.get("working_set_id") not in (None, working_set.working_set_id):
        raise RuntimeError("candidate working_set_id does not match current reconstructed working set")

    evaluator = RemoteStrategyEvaluator(
        snapshot=snapshot,
        working_set=working_set,
        artifact_set=artifact_set,
        evaluation_start=args.start,
        evaluation_end=args.end,
        settings=GF_SCENARIO_SETTINGS,
        rebalance_frequency=args.rebalance_frequency,
    )
    request = evaluator.request_for(candidate)
    payload = evaluate_strategy_remote(
        request,
        worker_path=evaluator.worker_path,
        settings=RemoteSettings.from_gf_settings(GF_SCENARIO_SETTINGS),
    )
    evaluation = evaluator._result_from_payload(candidate, request, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "name": candidate.name,
                "window": [args.start, args.end],
                "annualized_return": evaluation.metrics.get("annualized_return"),
                "excess_annualized_return": evaluation.metrics.get("excess_annualized_return"),
                "sharpe": evaluation.metrics.get("sharpe"),
                "max_drawdown": evaluation.metrics.get("max_drawdown"),
                "artifact_path": evaluation.artifact_path,
                "artifact_checksum": evaluation.artifact_checksum,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
