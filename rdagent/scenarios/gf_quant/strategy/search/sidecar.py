"""Sidecar persistence for one search tree.

The design keeps tree statistics *outside* the strategy registry: the registry
answers "what was accepted", the sidecar answers "what did the search try and
what did each attempt cost".  A run writes its whole tree after the loop, and
the next run loads it and continues -- statistics accumulate across runs while
the budget restarts.

The file is a single JSON object with a version tag.  A version the loader does
not recognise is refused rather than half-read: a silently mis-parsed tree would
show up as a search that mysteriously re-evaluates everything, which is the
kind of failure that looks like a working run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rdagent.scenarios.gf_quant.strategy.search.tree import TreeNode, tree_node_from_payload

SIDECAR_VERSION = 1

SIDECAR_FILENAME = "mcts_tree.json"


@dataclass(frozen=True)
class SidecarState:
    """What was loaded: the nodes plus the run-level facts beside them."""

    version: int
    run_id: str
    nodes: dict[str, TreeNode]
    root_id: str
    runs_completed: int
    credits_spent_total: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": SIDECAR_VERSION,
            "run_id": self.run_id,
            "root_id": self.root_id,
            "runs_completed": self.runs_completed,
            "credits_spent_total": self.credits_spent_total,
            "nodes": [node.as_payload() for node in self.nodes.values()],
        }


def sidecar_path(directory: str | Path) -> Path:
    return Path(directory) / SIDECAR_FILENAME


def load_state(path: str | Path) -> SidecarState | None:
    """Read a sidecar, or ``None`` when there is nothing to continue from."""

    target = Path(path)
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("search sidecar must be a JSON object")
    version = int(payload.get("version", 0))
    if version != SIDECAR_VERSION:
        raise ValueError(
            f"unsupported search sidecar version {version}; this build reads version {SIDECAR_VERSION}"
        )
    nodes = {
        str(item["node_id"]): tree_node_from_payload(item)
        for item in payload.get("nodes", ())
    }
    if not nodes:
        raise ValueError("search sidecar carries no nodes")
    return SidecarState(
        version=version,
        run_id=str(payload.get("run_id", "")),
        nodes=nodes,
        root_id=str(payload.get("root_id", "root")),
        runs_completed=int(payload.get("runs_completed", 0)),
        credits_spent_total=int(payload.get("credits_spent_total", 0)),
    )


def write_state(
    path: str | Path,
    *,
    run_id: str,
    nodes: Mapping[str, TreeNode],
    root_id: str,
    runs_completed: int,
    credits_spent_total: int,
) -> Path:
    """Write the whole tree atomically enough for a search to resume from it.

    The file is written to a temporary name and then replaced, so an interrupted
    write leaves the previous sidecar intact instead of a truncated one that
    would fail to parse on the next run.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = SidecarState(
        version=SIDECAR_VERSION,
        run_id=run_id,
        nodes=dict(nodes),
        root_id=root_id,
        runs_completed=runs_completed,
        credits_spent_total=credits_spent_total,
    )
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state.as_payload(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
