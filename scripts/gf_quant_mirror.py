"""Generate and verify the worker-side verbatim mirrors of the local strategy layer.

The strategy layer keeps two implementations of the same logic: the local package
``rdagent/scenarios/gf_quant/strategy/`` and the standalone remote worker
``scripts/gf_quant_strategy_worker.py``.  A subset of the two files has to stay
character-for-character comparable, and this script owns both halves of that
contract:

``--write``
    Re-generate the worker mirror block from ``strategy/execution.py``.  The block
    is bordered by its banner comment on top and the ``_build_buffered_target_row``
    anchor below, so re-running is idempotent: the current block is simply replaced
    with a freshly renamed copy of the execution-layer tail (public names get a
    leading underscore, shared constants are remapped onto the worker's copies).

``--check``
    Compare only, write nothing.  Two independent checks run in one pass:

    1. ``execution.py`` -> worker splice fidelity: rebuild the block in memory and
       require it to be byte-identical to the block currently in the worker.
    2. symbol-by-symbol text identity for the function pairs that live outside the
       spliced block (execution 21, audit 10, walk_forward 5, manifest 2),
       normalised by name so that ``apply_lot_rounding`` and
       ``_apply_lot_rounding`` compare equal.

    ``--check`` deliberately does **not** cover the two worker-only manifest
    symbols (``_worker_runtime_snapshot``, ``_build_manifest_section``) and the
    ``feasibility_days`` helper: the former have no local counterpart at all and
    the latter is a local-side reader of the trace the worker writes, so neither
    belongs to the mirrored set.  They are recorded here rather than silently
    dropped.

    Any mismatch prints the offending symbol and the first differing lines, and the
    process exits non-zero.

Usage::

    python scripts/gf_quant_mirror.py --check
    python scripts/gf_quant_mirror.py --write

The repository root is derived from this file's location, so the script works from
any working directory.  The same checks are enforced by
``test/scenarios/gf_quant/test_strategy_mirror_tooling.py``.

There is deliberately no ``[project.scripts]`` console-script entry for this tool:
``scripts/`` carries no ``__init__.py`` and is excluded from
``[tool.setuptools.packages.find]`` (``include = ["rdagent*"]``), so a
``scripts.gf_quant_mirror:main`` entry point raises ``ModuleNotFoundError`` once
the package is installed and invoked outside the checkout.  Invoke it as a file.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCAL_EXECUTION = ROOT / "rdagent" / "scenarios" / "gf_quant" / "strategy" / "execution.py"
LOCAL_AUDIT = ROOT / "rdagent" / "scenarios" / "gf_quant" / "strategy" / "audit.py"
LOCAL_WALK_FORWARD = ROOT / "rdagent" / "scenarios" / "gf_quant" / "strategy" / "walk_forward.py"
LOCAL_MANIFEST = ROOT / "rdagent" / "scenarios" / "gf_quant" / "strategy" / "manifest.py"
WORKER = ROOT / "scripts" / "gf_quant_strategy_worker.py"

ANCHOR = "def _build_buffered_target_row("
BANNER_MARK = "# Order layer, portfolio-risk scaling and the factor-decay monitor"

#: Where the spliced block starts inside ``execution.py``.
BLOCK_START_MARK = "CAPITAL = 1.0"

#: Public-to-mirror renames applied to the execution tail.  Order matters: the
#: constants come first so that a later ``\bday_key\b`` style rule cannot
#: partially rewrite an already-prefixed name.
RENAMES = [
    (r"\bPRICE_LIMIT_TOLERANCE\b", "_UNIVERSE_PRICE_LIMIT_TOLERANCE"),
    (r"\bCAPITAL\b", "_CAPITAL"),
    (r"\bLOT_SNAP\b", "_LOT_SNAP"),
    (r"\bday_key\b", "_day_key"),
    (r"\bplan_order\b", "_plan_order"),
    (r"\bapply_execution_layer\b", "_apply_execution_layer"),
    (r"\bapply_side_aware_tradability\b", "_apply_side_aware_tradability"),
    (r"\bapply_lot_rounding\b", "_apply_lot_rounding"),
    (r"\bapply_participation_cap\b", "_apply_participation_cap"),
    (r"\bstage_orders\b", "_stage_orders"),
    (r"\bvolatility_scale\b", "_volatility_scale"),
    (r"\bdrawdown_state_machine\b", "_drawdown_state_machine"),
    (r"\bfactor_decay_state\b", "_factor_decay_state"),
    (r"\bfactor_ic_series\b", "_factor_ic_series"),
    (r"\bfactor_decay_scales\b", "_factor_decay_scales"),
]

BANNER = [
    "# ---------------------------------------------------------------------------",
    "# Order layer, portfolio-risk scaling and the factor-decay monitor",
    "# (8.6-8.9, 9.1-9.2, 9.5).  Verbatim mirror of",
    "# ``rdagent/scenarios/gf_quant/strategy/execution.py``: the public helpers",
    "# keep the same bodies under a leading underscore, so the two engines can be",
    "# compared value for value by test/scenarios/gf_quant/",
    "# test_strategy_worker_parity.py.",
    "# ---------------------------------------------------------------------------",
    "",
]

#: ``(local name, worker name)`` inside ``strategy/execution.py``.
EXECUTION_PAIRS = [
    ("day_key", "_day_key"),
    ("_row", "_row"),
    ("_finite", "_finite"),
    ("apply_side_aware_tradability", "_apply_side_aware_tradability"),
    ("apply_lot_rounding", "_apply_lot_rounding"),
    ("_empty_pending", "_empty_pending"),
    ("_order_names", "_order_names"),
    ("_apply_pending", "_apply_pending"),
    ("_reap", "_reap"),
    ("_mean_amount", "_mean_amount"),
    ("apply_participation_cap", "_apply_participation_cap"),
    ("_advance_participation", "_advance_participation"),
    ("stage_orders", "_stage_orders"),
    ("_advance_staged", "_advance_staged"),
    ("volatility_scale", "_volatility_scale"),
    ("drawdown_state_machine", "_drawdown_state_machine"),
    ("plan_order", "_plan_order"),
    ("apply_execution_layer", "_apply_execution_layer"),
    ("factor_decay_state", "_factor_decay_state"),
    ("factor_ic_series", "_factor_ic_series"),
    ("factor_decay_scales", "_factor_decay_scales"),
]

#: ``(local name, worker name)`` inside ``strategy/audit.py``.
AUDIT_PAIRS = [
    ("_validate_stress", "_validate_stress"),
    ("stress_scenario_report", "_stress_scenario_report"),
    ("ensure_stress_passed", "_ensure_stress_passed"),
    ("_validate_lag_table", "_validate_lag_table"),
    ("_validate_feasibility", "_validate_feasibility"),
    ("_validate_data_leakage", "_validate_data_leakage"),
    ("attribute_execution_shortfall", "_attribute_execution_shortfall"),
    ("feasibility_report", "_feasibility_report"),
    ("data_leakage_report", "_data_leakage_report"),
    ("leakage_declarations", "_leakage_declarations"),
]

#: ``(local name, worker name)`` inside ``strategy/walk_forward.py``.  The
#: worker imports no project package, so its rolling splitter is mirrored text
#: rather than a shared import -- these pairs are what keeps the two window
#: builders and the two report assemblers from drifting apart.
WALK_FORWARD_PAIRS = [
    ("_positive_int", "_positive_walk_forward_int"),
    ("validate_walk_forward_config", "_validate_walk_forward"),
    ("_mean_metric", "_walk_forward_mean_metric"),
    ("_aggregate", "_walk_forward_aggregate"),
    ("walk_forward_report", "_walk_forward_report"),
]

#: ``(local name, worker name)`` inside ``strategy/manifest.py``.  The manifest
#: switch validator and the worker-digest helper are the two symbols both ends
#: implement; the rest of the worker's attestation (``_worker_runtime_snapshot``,
#: ``_build_manifest_section``) reads the worker's own process and file, so it has
#: no local counterpart to mirror.
MANIFEST_PAIRS = [
    ("validate_manifest_config", "_validate_manifest"),
    ("worker_file_digest", "_worker_file_digest"),
]

#: Constants whose two spellings must be normalised away before comparing bodies.
MIRROR_CONSTANTS = [
    "_UNIVERSE_PRICE_LIMIT_TOLERANCE",
    "PRICE_LIMIT_TOLERANCE",
    "_CAPITAL",
    "CAPITAL",
    "_LOT_SNAP",
    "LOT_SNAP",
]

MIRROR_NAMES = sorted(
    {
        name
        for _, pairs in (
            (None, EXECUTION_PAIRS),
            (None, AUDIT_PAIRS),
            (None, WALK_FORWARD_PAIRS),
            (None, MANIFEST_PAIRS),
        )
        for pair in pairs
        for name in pair
    }
    | set(MIRROR_CONSTANTS),
    key=len,
    reverse=True,
)


class MirrorError(RuntimeError):
    """Raised when the worker file does not have the layout the tool expects."""


# ---------------------------------------------------------------------------
# block generation
# ---------------------------------------------------------------------------


def render_execution_block(local_text: str) -> list[str]:
    """Return the worker mirror block for ``strategy/execution.py`` as lines."""

    lines = local_text.split("\n")
    start = None
    for index, line in enumerate(lines):
        if line.startswith(BLOCK_START_MARK):
            start = index
            break
    if start is None:
        raise MirrorError(f"{BLOCK_START_MARK!r} not found in {LOCAL_EXECUTION.name}")
    # Pull in the ``#:`` block comment that documents the constant.
    while start > 0 and lines[start - 1].startswith("#:"):
        start -= 1
    block = lines[start:]
    while block and not block[-1].strip():
        block.pop()
    text = "\n".join(block)
    for pattern, replacement in RENAMES:
        text = re.sub(pattern, replacement, text)
    return BANNER + text.split("\n")


def split_worker(worker_text: str) -> tuple[list[str], int, int, str]:
    """Locate the mirror block.  Returns ``(lines, block_start, anchor_at, newline)``."""

    newline = "\r\n" if "\r\n" in worker_text else "\n"
    lines = worker_text.split(newline)

    banner_at = next((i for i, line in enumerate(lines) if line.strip() == BANNER_MARK), None)
    if banner_at is None:
        raise MirrorError(f"mirror banner {BANNER_MARK!r} not found in {WORKER.name}")
    block_start = banner_at - 1
    if not lines[block_start].startswith("# ----"):
        raise MirrorError("unexpected banner layout: the rule comment is not directly above the marker")

    anchor_at = next((i for i, line in enumerate(lines) if line.startswith(ANCHOR)), None)
    if anchor_at is None:
        raise MirrorError(f"anchor {ANCHOR!r} not found in {WORKER.name}")
    if anchor_at < block_start:
        raise MirrorError("anchor appears before the mirror banner")
    return lines, block_start, anchor_at, newline


def generated_worker_text(worker_text: str, local_text: str) -> str:
    """Return the worker file with its mirror block regenerated."""

    lines, block_start, anchor_at, newline = split_worker(worker_text)
    mirror = render_execution_block(local_text)
    merged = lines[:block_start] + mirror + ["", ""] + lines[anchor_at:]
    return newline.join(merged)


def current_worker_block(worker_text: str) -> list[str]:
    """Return the mirror block currently living in the worker file."""

    lines, block_start, anchor_at, _ = split_worker(worker_text)
    block = lines[block_start:anchor_at]
    while block and not block[-1].strip():
        block.pop()
    return block


# ---------------------------------------------------------------------------
# symbol comparison
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Erase the local/worker spelling difference so bodies compare equal."""

    for name in MIRROR_NAMES:
        text = re.sub(r"\b" + re.escape(name) + r"\b", "_N_", text)
    return text


def body_text(func) -> str:
    """Function source without its docstring (worker mirrors keep a short one)."""

    source = textwrap.dedent(inspect.getsource(func))
    lines = source.splitlines()
    node = ast.parse(source).body[0]
    docstring = node.body[0] if node.body else None
    if (
        isinstance(docstring, ast.Expr)
        and isinstance(docstring.value, ast.Constant)
        and isinstance(docstring.value.value, str)
    ):
        for index in range(docstring.lineno - 1, docstring.end_lineno):
            lines[index] = ""
    return normalise("\n".join(line for line in lines if line.strip()))


def first_difference(left: str, right: str) -> list[str]:
    """Human-readable report of the first differing lines of two bodies."""

    report = []
    for index, (line_a, line_b) in enumerate(zip(left.splitlines(), right.splitlines()), start=1):
        if line_a != line_b:
            report.append(f"    line {index}")
            report.append(f"      local : {line_a}")
            report.append(f"      worker: {line_b}")
            break
    else:
        report.append("    bodies share a common prefix; line counts differ")
    report.append(f"    local lines={len(left.splitlines())} worker lines={len(right.splitlines())}")
    return report


def import_pair():
    """Import the local modules and the worker.  Requires the repo root on sys.path.

    Returns ``(local_execution, local_audit, local_walk_forward, local_manifest,
    worker)`` -- five elements, one per mirror group.
    """

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import rdagent.scenarios.gf_quant.strategy.audit as local_audit
    import rdagent.scenarios.gf_quant.strategy.execution as local_execution
    import rdagent.scenarios.gf_quant.strategy.manifest as local_manifest
    import rdagent.scenarios.gf_quant.strategy.walk_forward as local_walk_forward
    import scripts.gf_quant_strategy_worker as worker

    return local_execution, local_audit, local_walk_forward, local_manifest, worker


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def command_write() -> int:
    local_text = LOCAL_EXECUTION.read_text(encoding="utf-8")
    worker_text = WORKER.read_text(encoding="utf-8", newline="")
    regenerated = generated_worker_text(worker_text, local_text)
    if regenerated == worker_text:
        print("mirror block already current; nothing to write")
        return 0

    _lines, block_start, anchor_at, newline = split_worker(worker_text)
    mirror = render_execution_block(local_text)
    WORKER.write_text(regenerated, encoding="utf-8", newline="")
    print(f"replaced worker lines {block_start + 1}..{anchor_at} with {len(mirror) + 2} lines")
    return 0


def command_check() -> int:
    local_text = LOCAL_EXECUTION.read_text(encoding="utf-8")
    worker_text = WORKER.read_text(encoding="utf-8", newline="")

    failures: list[str] = []

    # 1. the spliced block must be exactly what --write would produce.
    expected = render_execution_block(local_text)
    actual = current_worker_block(worker_text)
    if expected != actual:
        failures.append("spliced execution block differs from a fresh regeneration")
        for index, (want, got) in enumerate(
            zip(expected + [""] * max(0, len(actual) - len(expected)), actual + [""] * max(0, len(expected) - len(actual)))
        ):
            if want != got:
                failures.append(f"    first differing block line {index + 1}")
                failures.append(f"      regenerated: {want!r}")
                failures.append(f"      worker     : {got!r}")
                break
        failures.append(f"    regenerated lines={len(expected)} worker block lines={len(actual)}")
    else:
        print(f"ok   spliced execution block ({len(actual)} lines, {sum(len(line) + 1 for line in actual)} bytes)")

    # 2. symbol-by-symbol identity outside the block.
    try:
        local_execution, local_audit, local_walk_forward, local_manifest, worker = import_pair()
    except Exception as exc:  # pragma: no cover - environment failure, surfaced verbatim
        print(f"FAIL cannot import the mirror modules: {exc!r}")
        return 2

    for module, pairs in (
        (local_execution, EXECUTION_PAIRS),
        (local_audit, AUDIT_PAIRS),
        (local_walk_forward, WALK_FORWARD_PAIRS),
        (local_manifest, MANIFEST_PAIRS),
    ):
        for left, right in pairs:
            a = getattr(module, left, None)
            b = getattr(worker, right, None)
            if a is None or b is None:
                failures.append(f"MISSING {left} / {right}")
                continue
            body_a, body_b = body_text(a), body_text(b)
            if body_a != body_b:
                failures.append(f"DRIFT {left}")
                failures.extend(first_difference(body_a, body_b))
            else:
                print(f"ok   {left}")

    # 3. shared configuration surfaces that are not functions.
    try:
        if local_audit.AUDIT_SECTIONS != worker.AUDIT_KEYS:
            failures.append(f"DRIFT audit sections: {local_audit.AUDIT_SECTIONS} != {worker.AUDIT_KEYS}")
        if local_audit.STRESS_MODES != worker.STRESS_MODES:
            failures.append("DRIFT STRESS_MODES")
        if local_audit.DEFAULT_STRESS_MODE != worker.DEFAULT_STRESS_MODE:
            failures.append("DRIFT DEFAULT_STRESS_MODE")
        if local_walk_forward.WALK_FORWARD_KEYS != worker.WALK_FORWARD_KEYS:
            failures.append(
                f"DRIFT walk_forward keys: {local_walk_forward.WALK_FORWARD_KEYS} != {worker.WALK_FORWARD_KEYS}"
            )
        if local_walk_forward.WALK_FORWARD_METRIC_KEYS != worker.WALK_FORWARD_METRIC_KEYS:
            failures.append("DRIFT WALK_FORWARD_METRIC_KEYS")
        if local_walk_forward.DEFAULT_MAX_SEGMENTS != worker.DEFAULT_MAX_SEGMENTS:
            failures.append("DRIFT DEFAULT_MAX_SEGMENTS")
        if local_walk_forward.DEFAULT_MIN_OOS_SEGMENTS != worker.DEFAULT_MIN_OOS_SEGMENTS:
            failures.append("DRIFT DEFAULT_MIN_OOS_SEGMENTS")
        if local_manifest.MANIFEST_KEYS != worker.MANIFEST_KEYS:
            failures.append(f"DRIFT manifest keys: {local_manifest.MANIFEST_KEYS} != {worker.MANIFEST_KEYS}")
    except AttributeError as exc:
        failures.append(f"MISSING shared constant: {exc}")
    else:
        print("ok   audit section key sets, stress modes, walk-forward and manifest constants")

    # 4. the remapped constants must carry the same values.
    constants_ok = (
        local_execution.PRICE_LIMIT_TOLERANCE == worker._UNIVERSE_PRICE_LIMIT_TOLERANCE
        and local_execution.CAPITAL == worker._CAPITAL
        and local_execution.LOT_SNAP == worker._LOT_SNAP
    )
    if not constants_ok:
        failures.append(
            "DRIFT constants: "
            f"PRICE_LIMIT_TOLERANCE={local_execution.PRICE_LIMIT_TOLERANCE!r}/"
            f"{worker._UNIVERSE_PRICE_LIMIT_TOLERANCE!r} "
            f"CAPITAL={local_execution.CAPITAL!r}/{worker._CAPITAL!r} "
            f"LOT_SNAP={local_execution.LOT_SNAP!r}/{worker._LOT_SNAP!r}"
        )
    else:
        print(
            f"ok   constants PRICE_LIMIT_TOLERANCE={local_execution.PRICE_LIMIT_TOLERANCE!r} "
            f"CAPITAL={local_execution.CAPITAL!r} LOT_SNAP={local_execution.LOT_SNAP!r}"
        )

    if failures:
        print("\nMIRROR DRIFT")
        for line in failures:
            print(line)
        print(f"\n{len(failures)} problem line(s); run --write to regenerate the spliced block")
        return 1

    print("\nmirrors are identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gf_quant_mirror",
        description="Generate or verify the worker-side verbatim mirrors of the local strategy layer.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="re-generate the worker mirror block in place (idempotent)")
    group.add_argument("--check", action="store_true", help="compare only; exit non-zero on any drift")
    args = parser.parse_args(argv)

    try:
        if args.write:
            return command_write()
        return command_check()
    except MirrorError as exc:
        print(f"mirror tool cannot proceed: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
