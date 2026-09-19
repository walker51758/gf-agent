"""StrategyManifestEmitter (roadmap 10.8): the switch, the record and the digest.

Three things are pinned here.  The request-level ``manifest`` object is a bare
switch -- :data:`MANIFEST_KEYS` is empty and both ends reject any key with the
same message.  The assembled record carries the hard identity facts and marks
the schema/semantics versions as labels so re-labelling can never change a
verdict.  And "immutable" is taken literally: a manifest file is named by the
strategy id and is never overwritten with different content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rdagent.scenarios.gf_quant.strategy.manifest import (
    MANIFEST_KEYS,
    MANIFEST_LABEL_FIELDS,
    MANIFEST_SCHEMA_VERSION,
    OPERATOR_SEMANTICS_LABEL,
    PROVENANCE_AVAILABLE_KEY,
    WORKER_ATTESTATION_KEY,
    build_strategy_manifest,
    manifest_checksum,
    validate_manifest,
    validate_manifest_config,
    worker_file_digest,
    write_strategy_manifest,
)
from scripts.gf_quant_strategy_worker import (
    MANIFEST_KEYS as _WORKER_MANIFEST_KEYS,
    _validate_manifest,
    _worker_file_digest,
)


def _manifest(**overrides):
    payload = {
        "strategy_id": "strategy_abc",
        "strategy_name": "parity_manifest",
        "canonical_payload": '{"composition":{"top_n":3}}',
        "evaluation": {"start": "2020-01-01", "end": "2023-12-31", "rebalance_frequency": "monthly"},
        "execution_costs": {"buy_bps": 10.0, "sell_bps": 10.0},
        "artifact_set_id": "artifact_set_1",
        "factor_ids": [{"factor_id": "f1", "name": "alpha", "artifact_checksum": "deadbeef"}],
        "snapshot_id": "gf_snapshot_1",
        "snapshot_checksum": "aaaa",
        "working_set_id": "ws_1",
        "working_set_snapshot_checksum": "aaaa",
        "data_cutoff": "2023-12-31",
        "acceptance": {"status": "accepted", "gate_failures": []},
        "worker_attestation": {"worker_file_sha256": "abc", "worker_runtime": {"python": "3.13.0"}},
    }
    payload.update(overrides)
    return build_strategy_manifest(**payload)


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


def test_manifest_keys_is_empty_by_design() -> None:
    """Every roadmap 10.8 parameter is a fact to be collected, not a setting."""

    assert MANIFEST_KEYS == frozenset()
    assert MANIFEST_KEYS == _WORKER_MANIFEST_KEYS


def test_empty_manifest_object_is_the_only_accepted_value() -> None:
    validate_manifest_config({})
    _validate_manifest({})


@pytest.mark.parametrize("manifest", [{"schema_version": 1}, {"anything": None}])
def test_unknown_manifest_keys_fail_closed_identically(manifest: dict) -> None:
    with pytest.raises(ValueError) as local:
        validate_manifest_config(manifest)
    with pytest.raises(ValueError) as worker:
        _validate_manifest(manifest)
    assert str(local.value) == str(worker.value)
    assert "request manifest has unknown keys" in str(local.value)


def test_non_object_manifest_fails_closed_identically() -> None:
    for value in ([], 7, "manifest"):
        with pytest.raises(ValueError) as local:
            validate_manifest_config(value)
        with pytest.raises(ValueError) as worker:
            _validate_manifest(value)
        assert str(local.value) == str(worker.value) == "request manifest must be an object"


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


def test_manifest_carries_identity_facts_and_empty_attestation() -> None:
    manifest = _manifest(worker_attestation=None)
    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["operator_semantics_label"] == OPERATOR_SEMANTICS_LABEL
    assert manifest["strategy_id"] == "strategy_abc"
    assert manifest["evaluation"]["rebalance_frequency"] == "monthly"
    assert manifest["execution_costs"] == {"buy_bps": 10.0, "sell_bps": 10.0}
    assert manifest[WORKER_ATTESTATION_KEY] is None


def test_manifest_records_provenance_availability_explicitly() -> None:
    """Absent provenance must be stated, never left to be inferred from empty fields."""

    assert _manifest()["provenance_available"] is True
    assert _manifest(provenance_available=False)["provenance_available"] is False


def test_provenance_availability_is_not_a_label_field() -> None:
    """It is an identity fact about the record, so it must not be skipped by comparisons."""

    assert "provenance_available" not in MANIFEST_LABEL_FIELDS


def test_manifest_labels_are_declared_as_non_identity() -> None:
    assert MANIFEST_LABEL_FIELDS == {
        "manifest_schema_version",
        "operator_semantics_label",
        "attestation_schema_version",
    }


def test_manifest_stores_the_canonical_payload_verbatim() -> None:
    """The identity string the registry keys on is stored, not re-serialised."""

    canonical = '{"composition":{"top_n":3},"universe":"csi_all"}'
    manifest = _manifest(canonical_payload=canonical)
    assert manifest["canonical_payload"] == canonical


def test_manifest_refuses_a_non_string_canonical_payload() -> None:
    """``StrategyCandidate.canonical_payload`` returns JSON text, so nothing else fits."""

    with pytest.raises(ValueError, match="canonical JSON string"):
        _manifest(canonical_payload={"composition": {"top_n": 3}})


def test_manifest_is_json_round_trippable() -> None:
    manifest = _manifest()
    restored = json.loads(json.dumps(manifest, sort_keys=True))
    assert restored == manifest
    assert manifest_checksum(restored) == manifest_checksum(manifest)


def test_manifest_checksum_moves_with_an_identity_field() -> None:
    """The checksum is an identity, so a changed fact must change it."""

    base = _manifest()
    changed = _manifest(artifact_set_id="artifact_set_2")
    assert manifest_checksum(base) != manifest_checksum(changed)


#: The assembled record's complete envelope, spelled out in literals on purpose.
#: ``MANIFEST_KEYS`` above is the *request* whitelist and is empty by design; this
#: is the *record*'s key set, which nothing used to police.  A key added here rides
#: into every record and into the registry, and none of the provenance or identity
#: assertions would notice: they read named fields, and the checksum's coverage was
#: only ever asserted for keys that already existed.
#:
#: It deliberately does **not** reference ``PROVENANCE_AVAILABLE_KEY`` or
#: ``WORKER_ATTESTATION_KEY``, even though those are the very constants the builder
#: writes those two keys from.  A pin assembled out of the builder's own constants
#: moves with them: rename one of those constants and this set and the record move
#: together, which leaves the test green.  Literal text here is what forces such a
#: rename to be made twice, visibly.
MANIFEST_ENVELOPE_KEYS = frozenset(
    {
        "manifest_schema_version",
        "operator_semantics_label",
        "provenance_available",
        "strategy_id",
        "strategy_name",
        "canonical_payload",
        "snapshot_id",
        "snapshot_checksum",
        "working_set_id",
        "working_set_snapshot_checksum",
        "data_cutoff",
        "artifact_set_id",
        "factor_ids",
        "evaluation",
        "execution_costs",
        "acceptance",
        "worker_attestation",
    }
)


def test_the_manifest_key_set_is_pinned_exactly() -> None:
    """Adding or dropping a key has to be a deliberate, visible edit.

    The two emission shapes differ in the *value* of ``worker_attestation`` -- an
    object when the request asked for one, ``None`` otherwise -- and never in the
    key set, so one assertion covers both and neither shape is a false positive.
    """

    assert set(_manifest()) == MANIFEST_ENVELOPE_KEYS
    assert set(_manifest(worker_attestation=None)) == MANIFEST_ENVELOPE_KEYS
    assert len(MANIFEST_ENVELOPE_KEYS) == 17

    # The flag is a key of the envelope, not an optional extra: it is written
    # whichever way ``provenance_available`` goes, so a manifest that says
    # ``False`` still has the same shape as one that says ``True``.
    assert set(_manifest(provenance_available=False)) == MANIFEST_ENVELOPE_KEYS


def test_manifest_checksum_covers_the_whole_envelope() -> None:
    """Hashing only the label fields would make the checksum blind to the envelope."""

    manifest = _manifest()
    extra = {**manifest, "extra_probe_key": "x"}
    dropped = {key: value for key, value in manifest.items() if key != "acceptance"}
    assert manifest_checksum(extra) != manifest_checksum(manifest)
    assert manifest_checksum(dropped) != manifest_checksum(manifest)


def test_manifest_validation_requires_the_label_fields() -> None:
    manifest = _manifest()
    validate_manifest(manifest)
    with pytest.raises(ValueError, match="manifest_schema_version"):
        validate_manifest({key: value for key, value in manifest.items() if key != "manifest_schema_version"})


def test_manifest_validation_refuses_a_non_object_attestation() -> None:
    with pytest.raises(ValueError, match="worker_attestation"):
        validate_manifest({**_manifest(), WORKER_ATTESTATION_KEY: ["not", "an", "object"]})


def test_manifest_validation_refuses_a_non_object() -> None:
    with pytest.raises(ValueError, match="strategy manifest must be an object"):
        validate_manifest([])


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------


def test_worker_file_digest_matches_the_local_helper(tmp_path: Path) -> None:
    """Both ends compute the same SHA-256 over the same bytes."""

    path = tmp_path / "worker.py"
    path.write_bytes(b"# stand-in for the worker source\n" * 3)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert worker_file_digest(path) == expected
    assert _worker_file_digest(path) == expected


def test_worker_file_digest_moves_with_content(tmp_path: Path) -> None:
    path = tmp_path / "worker.py"
    path.write_bytes(b"one")
    first = worker_file_digest(path)
    path.write_bytes(b"two")
    assert worker_file_digest(path) != first


def test_worker_file_digest_reads_the_real_worker_source() -> None:
    """The digest is over the shipped file, so it changes when that file changes."""

    from scripts import gf_quant_strategy_worker

    path = Path(gf_quant_strategy_worker.__file__)
    assert worker_file_digest(path) == _worker_file_digest(path)
    assert len(worker_file_digest(path)) == 64


# ---------------------------------------------------------------------------
# immutability
# ---------------------------------------------------------------------------


def test_write_manifest_names_the_file_by_strategy_id(tmp_path: Path) -> None:
    path = write_strategy_manifest(_manifest(), tmp_path)
    assert path == tmp_path / "strategy_abc.json"
    assert path.is_file()


def test_rewriting_identical_content_is_a_no_op(tmp_path: Path) -> None:
    first = write_strategy_manifest(_manifest(), tmp_path)
    marker = first.stat().st_mtime_ns
    second = write_strategy_manifest(_manifest(), tmp_path)
    assert first == second
    assert second.stat().st_mtime_ns == marker


def test_rewriting_different_content_is_refused(tmp_path: Path) -> None:
    """An existing manifest is a record, not a cache."""

    write_strategy_manifest(_manifest(), tmp_path)
    with pytest.raises(ValueError) as excinfo:
        write_strategy_manifest(_manifest(artifact_set_id="artifact_set_2"), tmp_path)
    message = str(excinfo.value)
    assert "strategy_abc.json" in message
    assert "artifact_set_id" in message


def test_the_refused_rewrite_leaves_the_original_on_disk(tmp_path: Path) -> None:
    write_strategy_manifest(_manifest(), tmp_path)
    with pytest.raises(ValueError):
        write_strategy_manifest(_manifest(artifact_set_id="artifact_set_2"), tmp_path)
    on_disk = json.loads((tmp_path / "strategy_abc.json").read_text(encoding="utf-8"))
    assert on_disk["artifact_set_id"] == "artifact_set_1"
