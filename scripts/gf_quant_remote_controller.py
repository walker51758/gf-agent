"""Backward-compatible CLI wrapper for the packaged GF remote controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdagent.scenarios.gf_quant.remote import RemoteSettings, evaluate_remote


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Run one GF Quant request on the pinned remote data host")
    parser.add_argument("--request", required=True, type=Path, help="Local JSON request")
    parser.add_argument("--result", required=True, type=Path, help="Local JSON result destination")
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("gf_quant_remote_worker.py"),
        help="Remote worker source",
    )
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = evaluate_remote(request, worker_path=args.worker, settings=RemoteSettings.from_env())
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed={result.get('name')} family={result.get('family')} result={args.result}")


if __name__ == "__main__":
    main()
