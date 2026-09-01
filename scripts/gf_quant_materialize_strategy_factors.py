"""Materialize selected frozen factors into a common remote evaluation window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", nargs="+", required=True, help="Exact factor names from the frozen working set")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-cutoff", default="2023-12-31", help="Cutoff used to reconstruct the logic snapshot")
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS
    from rdagent.scenarios.gf_quant.registry import FactorRegistry
    from rdagent.scenarios.gf_quant.strategy.artifacts import materialize_factor_artifact_set
    from rdagent.scenarios.gf_quant.strategy.snapshot import build_factor_universe_snapshot, build_factor_working_set

    registry = FactorRegistry(GF_SCENARIO_SETTINGS.registry_path)
    snapshot = build_factor_universe_snapshot(registry, data_cutoff=args.data_cutoff)
    working_set = build_factor_working_set(snapshot)
    artifact_set = materialize_factor_artifact_set(
        working_set=working_set,
        factor_names=args.factors,
        evaluation_start=args.start,
        evaluation_end=args.end,
        settings=GF_SCENARIO_SETTINGS,
        remote_artifact_dir=args.remote_dir,
    )
    artifact_set.write_json(args.output)
    print(
        f"artifact_set={artifact_set.artifact_set_id} factors={len(artifact_set.factors)} "
        f"window={artifact_set.evaluation_start}..{artifact_set.evaluation_end} output={args.output}"
    )


if __name__ == "__main__":
    main()
