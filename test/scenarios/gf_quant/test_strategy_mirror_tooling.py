"""Red lines for the worker mirror tooling.

The strategy layer keeps its execution-layer logic twice: locally in
``rdagent/scenarios/gf_quant/strategy/execution.py`` and mirrored in
``scripts/gf_quant_strategy_worker.py``.  ``scripts/gf_quant_mirror.py`` owns both
halves of that contract -- ``--write`` regenerates the worker block from the local
file, ``--check`` verifies it -- and these tests hold the tool to its promise:

* the block currently in the worker must be byte-identical to a fresh
  regeneration, so a hand edit (or a stale splice after a local change) fails here
  instead of silently drifting;
* every mirrored function pair must have an identical normalised body;
* the shared constants and audit key sets must agree.

Everything runs off paths derived from ``__file__``, so the suite works from any
checkout and any working directory.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MIRROR_TOOL = ROOT / "scripts" / "gf_quant_mirror.py"

assert MIRROR_TOOL.is_file(), f"mirror tool not found at {MIRROR_TOOL}"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("gf_quant_mirror", MIRROR_TOOL)
mirror = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mirror)


# ---------------------------------------------------------------------------
# the spliced block
# ---------------------------------------------------------------------------


def test_worker_execution_mirror_is_generated() -> None:
    """The worker's mirror block must be exactly what ``--write`` would produce.

    The failure message carries the first differing line so a stale splice is
    diagnosable without re-running the tool by hand.
    """

    local_text = mirror.LOCAL_EXECUTION.read_text(encoding="utf-8")
    worker_text = mirror.WORKER.read_text(encoding="utf-8", newline="")

    expected = mirror.render_execution_block(local_text)
    actual = mirror.current_worker_block(worker_text)

    if expected != actual:
        width = max(len(expected), len(actual))
        for index in range(width):
            want = expected[index] if index < len(expected) else "<missing>"
            got = actual[index] if index < len(actual) else "<missing>"
            if want != got:
                pytest.fail(
                    "worker mirror block is stale at block line "
                    f"{index + 1}:\n  regenerated: {want!r}\n  worker     : {got!r}\n"
                    f"(regenerated {len(expected)} lines, worker block {len(actual)} lines; "
                    "run `python scripts/gf_quant_mirror.py --write`)"
                )
        pytest.fail("worker mirror block differs in length only")  # pragma: no cover


def test_regeneration_is_idempotent() -> None:
    """Applying ``--write`` to the already-current worker is a no-op."""

    worker_text = mirror.WORKER.read_text(encoding="utf-8", newline="")
    assert worker_text == mirror.generated_worker_text(worker_text, mirror.LOCAL_EXECUTION.read_text(encoding="utf-8"))


def test_mirror_tool_derives_its_own_root() -> None:
    """The tool must resolve the repo from ``__file__``, never a hard-coded path."""

    text = MIRROR_TOOL.read_text(encoding="utf-8")
    assert "D:\\" not in text and "D:/" not in text
    assert mirror.ROOT == ROOT
    assert mirror.WORKER == ROOT / "scripts" / "gf_quant_strategy_worker.py"


# ---------------------------------------------------------------------------
# symbol-by-symbol identity
# ---------------------------------------------------------------------------


def _pairs() -> list[tuple[str, str, str]]:
    out = [("execution", left, right) for left, right in mirror.EXECUTION_PAIRS]
    out += [("audit", left, right) for left, right in mirror.AUDIT_PAIRS]
    out += [("walk_forward", left, right) for left, right in mirror.WALK_FORWARD_PAIRS]
    out += [("manifest", left, right) for left, right in mirror.MANIFEST_PAIRS]
    return out


@pytest.fixture(scope="module")
def modules():
    return mirror.import_pair()


def _module_for(modules, group: str):
    local_execution, local_audit, local_walk_forward, local_manifest, _worker = modules
    return {
        "execution": local_execution,
        "audit": local_audit,
        "walk_forward": local_walk_forward,
        "manifest": local_manifest,
    }[group]


@pytest.mark.parametrize(("group", "left", "right"), _pairs(), ids=lambda v: v if isinstance(v, str) else "")
def test_mirrored_function_bodies_are_identical(modules, group: str, left: str, right: str) -> None:
    """``left`` in the local module and ``right`` in the worker must share a body."""

    _local_execution, _local_audit, _local_walk_forward, _local_manifest, worker = modules
    module = _module_for(modules, group)

    local_func = getattr(module, left, None)
    worker_func = getattr(worker, right, None)
    assert local_func is not None, f"local {group}.{left} is missing"
    assert worker_func is not None, f"worker.{right} is missing"

    body_local = mirror.body_text(local_func)
    body_worker = mirror.body_text(worker_func)
    if body_local != body_worker:
        pytest.fail(
            f"{group}.{left} / worker.{right} drifted:\n"
            + "\n".join(mirror.first_difference(body_local, body_worker))
        )


def test_remapped_constants_agree(modules) -> None:
    """The three names RENAMES touches must carry the same values on both sides."""

    local_execution, _local_audit, _local_walk_forward, _local_manifest, worker = modules
    assert local_execution.PRICE_LIMIT_TOLERANCE == worker._UNIVERSE_PRICE_LIMIT_TOLERANCE
    assert local_execution.CAPITAL == worker._CAPITAL
    assert local_execution.LOT_SNAP == worker._LOT_SNAP


def test_audit_section_keysets_agree(modules) -> None:
    """The audit section vocabulary is a mirror of its own (registry-style) list."""

    _local_execution, local_audit, _local_walk_forward, _local_manifest, worker = modules
    assert local_audit.AUDIT_SECTIONS == worker.AUDIT_KEYS
    assert local_audit.STRESS_MODES == worker.STRESS_MODES
    assert local_audit.DEFAULT_STRESS_MODE == worker.DEFAULT_STRESS_MODE


def test_walk_forward_keysets_agree(modules) -> None:
    """WalkForwardEvaluation's configuration surface is shared, so it is checked."""

    _local_execution, _local_audit, local_walk_forward, _local_manifest, worker = modules
    assert local_walk_forward.WALK_FORWARD_KEYS == worker.WALK_FORWARD_KEYS
    assert local_walk_forward.WALK_FORWARD_METRIC_KEYS == worker.WALK_FORWARD_METRIC_KEYS
    assert local_walk_forward.DEFAULT_MAX_SEGMENTS == worker.DEFAULT_MAX_SEGMENTS
    assert local_walk_forward.DEFAULT_MIN_OOS_SEGMENTS == worker.DEFAULT_MIN_OOS_SEGMENTS
    assert local_walk_forward.DEFAULT_VALID_SIZE == worker.DEFAULT_VALID_SIZE


def test_manifest_keysets_agree(modules) -> None:
    """The manifest switch is the empty-object contract on both ends."""

    _local_execution, _local_audit, _local_walk_forward, local_manifest, worker = modules
    assert local_manifest.MANIFEST_KEYS == worker.MANIFEST_KEYS
    assert local_manifest.MANIFEST_KEYS == frozenset()


def test_mirror_groups_carry_the_expected_pair_counts(modules) -> None:
    """Each mirror group's size is asserted, so a silently dropped pair fails here."""

    assert len(mirror.EXECUTION_PAIRS) == 21
    assert len(mirror.AUDIT_PAIRS) == 10
    assert len(mirror.WALK_FORWARD_PAIRS) == 5
    assert len(mirror.MANIFEST_PAIRS) == 2
    assert len(mirror.import_pair()) == 5


def test_every_rename_rule_is_still_live(modules) -> None:
    """Each RENAMES rule must still match the local source and resolve on the worker.

    Guards against a rule that silently stops matching after a refactor: an
    unmatched rule leaves the public name inside the mirror block, where the
    local/worker body comparison would then never notice it.  The rules are
    matched against ``execution.py`` itself, because that is the text they are
    applied to (``PRICE_LIMIT_TOLERANCE`` -> ``_UNIVERSE_PRICE_LIMIT_TOLERANCE``
    is a genuine rename, not just an underscore prefix).
    """

    _local_execution, _local_audit, _local_walk_forward, _local_manifest, worker = modules
    local_source = mirror.LOCAL_EXECUTION.read_text(encoding="utf-8")
    for pattern, replacement in mirror.RENAMES:
        assert re.search(pattern, local_source), f"rename rule no longer matches execution.py: {pattern}"
        assert hasattr(worker, replacement), f"rename target {replacement} is not defined in the worker"
