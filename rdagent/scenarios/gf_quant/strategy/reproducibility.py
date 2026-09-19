"""Reproducibility check (roadmap 10.9): identity comparison, not a re-run.

The roadmap asks for a "reproducible / not reproducible" verdict over a
manifest, a snapshot, an artifact checksum and a worker version.  This engine
deliberately answers that question by **comparing identities**, never by
re-running the pipeline: a re-run that agrees to within a float tolerance is
the weaker claim, and the frozen schema means there is nothing stochastic to
re-run anyway.  The roadmap's tolerance parameter is therefore not accepted --
there is no numeric comparison here to loosen.

The compared set is :data:`REPRODUCIBILITY_IDENTITY_FIELDS`.  Two fields are
excluded on purpose:

* ``request_checksum`` covers the *whole* request, including which audit
  sections this particular run asked for and whether the manifest switch was
  on.  Two runs of the same strategy that requested different audits are the
  same artifact, so folding the request hash in would report a spurious
  irreproducibility.
* ``worker_attestation.trading_calendar`` is an *observation* of the data
  window the run actually saw, which the artifact set's own evaluation window
  already covers.

In one sentence: ``request_checksum`` answers "what request was this", while
the identity set answers "can this artifact still be produced by the same
code, on the same data, in the same environment".  The label fields declared
in :mod:`manifest` are skipped for the same reason.

A missing field on either side is *not* a pass: it is recorded under
``missing_fields`` and forces ``not_reproducible``, following the repo's
fail-closed habit.  A worker attestation that is absent (the request never
asked for one) is the ``worker_attestation_missing`` case.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rdagent.scenarios.gf_quant.strategy.manifest import (
    MANIFEST_LABEL_FIELDS,
    PROVENANCE_AVAILABLE_KEY,
    WORKER_ATTESTATION_KEY,
)

#: Verdicts of the reproducibility check.
REPRODUCIBILITY_VERDICTS: tuple[str, ...] = ("reproducible", "not_reproducible")

#: Dotted paths that must agree between two manifests for the artifact to count
#: as reproducible.  This is the whole contract of roadmap 10.9; see the module
#: docstring for why ``request_checksum`` and ``trading_calendar`` are not here.
REPRODUCIBILITY_IDENTITY_FIELDS: tuple[str, ...] = (
    "strategy_id",
    "snapshot_checksum",
    "working_set_snapshot_checksum",
    "artifact_set_id",
    "evaluation",
    "execution_costs",
    "worker_attestation.worker_file_sha256",
    "worker_attestation.worker_runtime",
)

#: Reason code recorded when the manifest carries no worker attestation.
REASON_WORKER_ATTESTATION_MISSING = "worker_attestation_missing"

#: Reason code recorded when the manifest itself declares its provenance absent.
REASON_PROVENANCE_UNAVAILABLE = "provenance_unavailable"


def _is_carried(value: Any) -> bool:
    """Whether an identity value actually carries information.

    ``None``, ``""``, ``{}`` and ``[]`` are all "no value": comparing two of them
    is ``"" == ""``, which is true but proves nothing.  Treating that as a
    checked field is what lets a manifest of empty strings report
    ``reproducible``, so those values are reported as missing instead.
    """

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    return True


def _lookup(manifest: Mapping[str, Any], dotted_path: str) -> tuple[bool, Any]:
    """Return ``(present, value)`` for a dotted path into a manifest."""

    target: Any = manifest
    for part in dotted_path.split("."):
        if not isinstance(target, Mapping) or part not in target:
            return False, None
        target = target[part]
    return True, target


def reproducibility_report(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two manifests on their identity fields and report the verdict.

    ``reference`` is the manifest on record, ``candidate`` the one being
    checked.  The result carries the two-value verdict plus ``reasons``,
    ``checked_fields`` and ``missing_fields`` (and the label fields that were
    skipped, so a reader can see they were not compared).

    A field counts as checked only when **both** sides carry a non-empty value.
    Anything else lands in ``missing_fields`` with a ``field_unknown:<field>``
    reason and forces ``not_reproducible``: agreement between two absent values
    is not evidence, and a manifest that declares its own provenance absent
    (``provenance_available: false``) is reported separately rather than being
    silently read as a match.
    """

    for label, manifest in (("reference", reference), ("candidate", candidate)):
        if not isinstance(manifest, Mapping):
            raise ValueError(f"reproducibility {label} manifest must be an object")

    reasons: list[str] = []
    checked: list[str] = []
    missing: list[str] = []

    for field in REPRODUCIBILITY_IDENTITY_FIELDS:
        present_reference, value_reference = _lookup(reference, field)
        present_candidate, value_candidate = _lookup(candidate, field)
        if not (_is_carried(value_reference) and _is_carried(value_candidate)):
            reason = f"field_unknown:{field}"
            reasons.append(reason)
            missing.append(field)
            continue
        checked.append(field)
        if value_reference != value_candidate:
            reasons.append(f"field_mismatch:{field}")

    if reference.get(WORKER_ATTESTATION_KEY) is None or candidate.get(WORKER_ATTESTATION_KEY) is None:
        reasons.append(REASON_WORKER_ATTESTATION_MISSING)

    if not reference.get(PROVENANCE_AVAILABLE_KEY, True) or not candidate.get(PROVENANCE_AVAILABLE_KEY, True):
        reasons.append(REASON_PROVENANCE_UNAVAILABLE)

    # Preserve the declaration order of the identity set, and report each reason
    # once even when two paths land on the same code.
    seen: set[str] = set()
    ordered_reasons = [reason for reason in reasons if not (reason in seen or seen.add(reason))]

    return {
        "verdict": "reproducible" if not ordered_reasons else "not_reproducible",
        "reasons": ordered_reasons,
        "checked_fields": checked,
        "missing_fields": missing,
        "skipped_label_fields": sorted(MANIFEST_LABEL_FIELDS),
        "excluded_fields": ["request_checksum", "worker_attestation.trading_calendar"],
    }
