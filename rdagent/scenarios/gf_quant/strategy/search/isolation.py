"""Test isolation for search runs: every artifact under one disposable root.

A search writes four kinds of state -- the tree sidecar, the factor registry, the
factor artifacts and the run log -- and a test that lets any of them land in the
production paths corrupts a real run's continuation.  This module owns the single
answer to "where does a run write", so a test can point one root at a temporary
directory and be certain nothing else was touched.

The separation is structural rather than conventional: :class:`SearchPaths`
derives every path from one root, so a caller cannot accidentally mix a
production sidecar with a test registry by passing two different arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The production root, under the repository's existing ignore directory.
PRODUCTION_ROOT = Path("git_ignore_folder") / "gf_quant"
PRODUCTION_CACHE = PRODUCTION_ROOT / "search_cache"
PRODUCTION_SIDECAR = PRODUCTION_ROOT / "search_sidecar"

# The test root.  Deliberately a sibling of the production paths inside the same
# ignore directory: one ignore rule covers both, and a test run is trivially
# removed without touching anything a real run wrote.
TEST_ROOT = PRODUCTION_ROOT / "search_test"

# Directories a test must never write into.  Checked before a run rather than
# after, because the damage of writing first and checking later is a corrupted
# continuation that reads as a legitimate tree.
PRODUCTION_DIRECTORIES = (PRODUCTION_CACHE, PRODUCTION_SIDECAR)


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / "rdagent" / "scenarios" / "gf_quant").is_dir():
        raise RuntimeError(f"cannot locate the repository root from {__file__}: tried {root}")
    return root


@dataclass(frozen=True)
class SearchPaths:
    """Every path one run writes, derived from a single root."""

    root: Path
    cache: Path
    sidecar: Path
    work: Path
    log: Path

    def as_payload(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "cache": str(self.cache),
            "sidecar": str(self.sidecar),
            "work": str(self.work),
            "log": str(self.log),
        }

    def create(self) -> "SearchPaths":
        """Make every directory the run needs, and prove none is production."""

        assert_isolated(self.root)
        for path in (self.cache, self.sidecar, self.work, self.log):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def sidecar_file(self) -> Path:
        return self.sidecar / "mcts_tree.json"

    def registry_file(self) -> Path:
        return self.work / "factor_registry.sqlite"


def production_paths() -> SearchPaths:
    root = repository_root() / PRODUCTION_ROOT
    return SearchPaths(
        root=root,
        cache=repository_root() / PRODUCTION_CACHE,
        sidecar=repository_root() / PRODUCTION_SIDECAR,
        work=root / "search_work",
        log=root / "search_logs",
    )


def isolated_paths(name: str, *, base: str | Path | None = None) -> SearchPaths:
    """A fresh isolated root for one test or one named demo run.

    Named ``isolated_paths`` rather than ``test_paths`` on purpose: a module-level
    function whose name starts with ``test_`` is collected by pytest as a test
    case, so a production helper wearing that name turns every suite that imports
    it into a spurious collection error.
    """

    root = (Path(base) if base is not None else repository_root() / TEST_ROOT) / name
    return SearchPaths(
        root=root,
        cache=root / "cache",
        sidecar=root / "sidecar",
        work=root / "work",
        log=root / "log",
    )


def assert_isolated(root: str | Path) -> None:
    """Refuse a root that is, or contains, a production directory.

    Symmetric containment is checked because both mistakes are real: pointing a
    test at the production root, and pointing a run at a directory *above* it
    (which would swallow the production tree on a recursive write).
    """

    resolved = Path(root).resolve()
    repo = repository_root()
    for production in PRODUCTION_DIRECTORIES:
        target = (repo / production).resolve()
        if resolved == target or resolved in target.parents or target in resolved.parents:
            raise RuntimeError(
                f"refusing to use {resolved} as a run root: it overlaps the production "
                f"directory {target}.  Test runs write under {TEST_ROOT} instead."
            )


def write_manifest(paths: SearchPaths, payload: dict[str, Any]) -> Path:
    """Record what a run wrote, beside it, for later inspection."""

    paths.create()
    target = paths.log / "run_manifest.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target
