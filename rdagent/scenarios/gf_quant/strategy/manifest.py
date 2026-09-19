"""Strategy manifest emission (roadmap 10.8): the immutable identity record.

A manifest answers one question: *is this artifact reproducible?*  Everything
in it is either a hard identity fact -- the strategy identity hash, the frozen
snapshot and artifact-set checksums, the evaluation window, the execution-cost
contract, and the worker-side attestation that names the exact file digest and
runtime stack which produced the numbers -- or a human-readable semantic label.
The label fields are declared once in :data:`MANIFEST_LABEL_FIELDS` and are
skipped by the equality check, so re-labelling never turns a reproducible
artifact into an irreproducible one.

The request-level ``manifest`` object carries no configuration at all:
:data:`MANIFEST_KEYS` is empty, because every roadmap parameter (schema
version, operator version, parameter checksum, factor ids, cost facts, data
window) is a *fact to be collected*, not a knob a caller sets.  The object is
therefore purely a switch -- "ask the worker to fill in its attestation" --
and it rides at the request level, never inside the candidate payload, so
enabling it cannot move a strategy identity hash.

The worker mirrors :func:`validate_manifest_config` and
:func:`worker_file_digest` verbatim (``scripts/gf_quant_strategy_worker.py``);
their error messages are part of the two-side contract and must stay
character-for-character identical.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Keys of the request-level ``manifest`` object.  Empty by design: every
#: roadmap 10.8 parameter is a fact to be collected, never a caller setting, so
#: the object is a bare switch and any key is a caller mistake.
MANIFEST_KEYS: frozenset[str] = frozenset()

#: Schema version of the manifest layout itself.  A semantic label: it is
#: recorded so a reader can tell which layout produced the record, but two
#: manifests that differ only here describe the same artifact.
MANIFEST_SCHEMA_VERSION = 1

#: Human-readable label for the operator semantics this engine implements.
#: A semantic label, not an identity: the identity is the worker file digest.
OPERATOR_SEMANTICS_LABEL = "lda_equivalent_structured_operator_dag"

#: Keys that carry meaning for a reader but are excluded from equality: a
#: re-labelled manifest still describes the same artifact, so these must never
#: by themselves turn a reproducible artifact into an irreproducible one.
MANIFEST_LABEL_FIELDS: frozenset[str] = frozenset(
    {
        "manifest_schema_version",
        "operator_semantics_label",
        "attestation_schema_version",
    }
)

#: Section key under which the worker's self-attestation is embedded.
WORKER_ATTESTATION_KEY = "worker_attestation"


#: Field recording whether the evaluator could supply provenance facts at all.
#: A boolean *flag*, not a value: it says "the facts below are real" or "they
#: are absent".  Without it an absent provenance would be indistinguishable from
#: a genuine one whose ids happen to be empty, which is how a manifest can look
#: healthy while carrying nothing.
PROVENANCE_AVAILABLE_KEY = "provenance_available"


def validate_manifest_config(manifest: Mapping[str, Any]) -> None:
    """Validate the request-level ``manifest`` object.

    Mirrored by the worker's ``_validate_manifest``; both sides raise the same
    message for the same malformed input.  The only accepted value is the empty
    object -- see :data:`MANIFEST_KEYS`.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("request manifest must be an object")
    unknown = set(manifest) - MANIFEST_KEYS
    if unknown:
        raise ValueError(f"request manifest has unknown keys: {sorted(unknown)[:3]}")


def worker_file_digest(path: Path) -> str:
    """SHA-256 of the worker source file: the hard identity of the code that ran."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Canonical SHA-256 over the manifest, following the repo's JSON convention."""

    body = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Refuse a manifest that is not shaped like the 10.8 record.

    Checks the *envelope* only: the label fields must be present and the worker
    attestation must be an object or ``None``.  Whether the identity fields
    agree with another manifest is :mod:`reproducibility`'s business.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("strategy manifest must be an object")
    for field in sorted(MANIFEST_LABEL_FIELDS):
        if field == "attestation_schema_version":
            continue
        if field not in manifest:
            raise ValueError(f"strategy manifest is missing {field!r}")
    attestation = manifest.get(WORKER_ATTESTATION_KEY)
    if attestation is not None and not isinstance(attestation, Mapping):
        raise ValueError("strategy manifest worker_attestation must be an object or null")


def build_strategy_manifest(
    *,
    strategy_id: str,
    strategy_name: str,
    canonical_payload: str,
    evaluation: Mapping[str, Any],
    execution_costs: Mapping[str, Any],
    artifact_set_id: str,
    factor_ids: Sequence[Mapping[str, Any]],
    snapshot_id: str,
    snapshot_checksum: str,
    working_set_id: str,
    working_set_snapshot_checksum: str,
    data_cutoff: str,
    acceptance: Mapping[str, Any],
    worker_attestation: Mapping[str, Any] | None = None,
    provenance_available: bool = True,
) -> dict[str, Any]:
    """Assemble the complete 10.8 manifest from facts already on hand.

    Nothing here is invented: every argument is copied from the request, the
    frozen snapshot, the working set, the artifact set, the acceptance policy
    or the worker's own result.  ``worker_attestation`` stays ``None`` when the
    request did not ask for it, which :mod:`reproducibility` reads as "cannot
    prove reproducibility" rather than as a pass.

    ``canonical_payload`` is the strategy's canonical JSON *text* --
    :meth:`StrategyCandidate.canonical_payload` returns exactly that, and it is
    the identity string the registry already keys on, so it is stored verbatim
    rather than re-serialised into a second, possibly different spelling.
    """

    if not isinstance(canonical_payload, str) or not canonical_payload:
        raise ValueError("strategy manifest canonical_payload must be the canonical JSON string")

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "operator_semantics_label": OPERATOR_SEMANTICS_LABEL,
        PROVENANCE_AVAILABLE_KEY: bool(provenance_available),
        "strategy_id": str(strategy_id),
        "strategy_name": str(strategy_name),
        "canonical_payload": canonical_payload,
        "snapshot_id": str(snapshot_id),
        "snapshot_checksum": str(snapshot_checksum),
        "working_set_id": str(working_set_id),
        "working_set_snapshot_checksum": str(working_set_snapshot_checksum),
        "data_cutoff": str(data_cutoff),
        "artifact_set_id": str(artifact_set_id),
        "factor_ids": [dict(item) for item in factor_ids],
        "evaluation": dict(evaluation),
        "execution_costs": dict(execution_costs),
        "acceptance": dict(acceptance),
        WORKER_ATTESTATION_KEY: None if worker_attestation is None else dict(worker_attestation),
    }


def write_strategy_manifest(manifest: Mapping[str, Any], directory: str | Path) -> Path:
    """Write a manifest under its strategy id, refusing to overwrite a different one.

    "Immutable" is taken at face value: the file is named by the strategy id it
    describes, and re-writing that name with different content raises
    ``ValueError`` naming the differing keys, rather than silently replacing what
    an earlier run recorded.  Identical content is a no-op, so re-emitting the
    same manifest is safe.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest['strategy_id']}.json"
    body = json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(manifest):
            differing = sorted(
                key for key in set(existing) | set(manifest) if existing.get(key) != dict(manifest).get(key)
            )
            raise ValueError(f"strategy manifest {path.name} already exists with different content: {differing}")
        return path
    path.write_text(body, encoding="utf-8")
    return path
