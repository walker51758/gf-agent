"""ReproducibilityCheck (roadmap 10.9): identity comparison over the manifest.

The check never re-runs the engine, so these tests are about what the identity
set contains and, just as importantly, what it deliberately leaves out.  The
two exclusions carry the design: ``request_checksum`` covers which audit
sections this particular run asked for, and the observed ``trading_calendar``
is already covered by the artifact set's window.
"""

from __future__ import annotations

import pytest

from rdagent.scenarios.gf_quant.strategy.manifest import WORKER_ATTESTATION_KEY
from rdagent.scenarios.gf_quant.strategy.reproducibility import (
    REASON_WORKER_ATTESTATION_MISSING,
    REPRODUCIBILITY_IDENTITY_FIELDS,
    REPRODUCIBILITY_VERDICTS,
    reproducibility_report,
)

_RUNTIME = {"python": "3.13.0", "numpy": "2.5.3", "pandas": "3.0.5", "pyarrow": "25.0.1"}


def _manifest(**overrides):
    payload = {
        "manifest_schema_version": 1,
        "operator_semantics_label": "lda_equivalent_structured_operator_dag",
        "strategy_id": "strategy_abc",
        "snapshot_checksum": "aaaa",
        "working_set_snapshot_checksum": "aaaa",
        "artifact_set_id": "artifact_set_1",
        "evaluation": {"start": "2020-01-01", "end": "2023-12-31", "rebalance_frequency": "monthly"},
        "execution_costs": {"buy_bps": 10.0, "sell_bps": 10.0},
        "request_checksum": "run_one",
        WORKER_ATTESTATION_KEY: {
            "attestation_schema_version": 1,
            "worker_file_sha256": "digest_one",
            "worker_runtime": dict(_RUNTIME),
            "trading_calendar": {"first_date": "2020-01-02", "last_date": "2023-12-29", "rows": 971},
            "request_checksum": "run_one",
        },
    }
    payload.update(overrides)
    return payload


def test_identical_manifests_are_reproducible() -> None:
    report = reproducibility_report(_manifest(), _manifest())
    assert report["verdict"] == "reproducible"
    assert report["reasons"] == []
    assert report["missing_fields"] == []


def test_identity_set_is_exactly_the_declared_fields() -> None:
    assert REPRODUCIBILITY_IDENTITY_FIELDS == (
        "strategy_id",
        "snapshot_checksum",
        "working_set_snapshot_checksum",
        "artifact_set_id",
        "evaluation",
        "execution_costs",
        "worker_attestation.worker_file_sha256",
        "worker_attestation.worker_runtime",
    )
    report = reproducibility_report(_manifest(), _manifest())
    assert sorted(report["checked_fields"]) == sorted(REPRODUCIBILITY_IDENTITY_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strategy_id", "strategy_other"),
        ("snapshot_checksum", "bbbb"),
        ("working_set_snapshot_checksum", "bbbb"),
        ("artifact_set_id", "artifact_set_2"),
        ("evaluation", {"start": "2020-01-01", "end": "2023-12-30", "rebalance_frequency": "monthly"}),
        ("execution_costs", {"buy_bps": 15.0, "sell_bps": 15.0}),
    ],
)
def test_each_identity_field_mismatch_fails(field: str, value) -> None:
    report = reproducibility_report(_manifest(), _manifest(**{field: value}))
    assert report["verdict"] == "not_reproducible"
    assert f"field_mismatch:{field}" in report["reasons"]


def test_worker_file_digest_mismatch_fails() -> None:
    other = _manifest()
    other[WORKER_ATTESTATION_KEY] = {**other[WORKER_ATTESTATION_KEY], "worker_file_sha256": "digest_two"}
    report = reproducibility_report(_manifest(), other)
    assert report["verdict"] == "not_reproducible"
    assert "field_mismatch:worker_attestation.worker_file_sha256" in report["reasons"]


def test_worker_runtime_mismatch_fails() -> None:
    """A different library stack can move a number, so it is an identity fact."""

    other = _manifest()
    other[WORKER_ATTESTATION_KEY] = {
        **other[WORKER_ATTESTATION_KEY],
        "worker_runtime": {**_RUNTIME, "pandas": "2.3.3"},
    }
    report = reproducibility_report(_manifest(), other)
    assert report["verdict"] == "not_reproducible"
    assert "field_mismatch:worker_attestation.worker_runtime" in report["reasons"]


# ---------------------------------------------------------------------------
# the two deliberate exclusions
# ---------------------------------------------------------------------------


def test_request_checksum_is_excluded() -> None:
    """The same strategy under a different audit configuration is the same artifact."""

    reference = _manifest()
    reference["request_checksum"] = "run_one"
    candidate = _manifest()
    candidate["request_checksum"] = "run_two"
    candidate[WORKER_ATTESTATION_KEY] = {
        **candidate[WORKER_ATTESTATION_KEY],
        "request_checksum": "run_two",
    }
    report = reproducibility_report(reference, candidate)
    assert report["verdict"] == "reproducible"
    assert "request_checksum" in report["excluded_fields"]


def test_observed_trading_calendar_is_excluded() -> None:
    """It is an observation of the window the artifact set already describes."""

    other = _manifest()
    other[WORKER_ATTESTATION_KEY] = {
        **other[WORKER_ATTESTATION_KEY],
        "trading_calendar": {"first_date": "2020-01-03", "last_date": "2023-12-28", "rows": 969},
    }
    report = reproducibility_report(_manifest(), other)
    assert report["verdict"] == "reproducible"
    assert "worker_attestation.trading_calendar" in report["excluded_fields"]


def test_label_fields_are_skipped() -> None:
    """Re-labelling a record must not turn it irreproducible."""

    other = _manifest(manifest_schema_version=2, operator_semantics_label="relabelled")
    report = reproducibility_report(_manifest(), other)
    assert report["verdict"] == "reproducible"
    assert "manifest_schema_version" in report["skipped_label_fields"]
    assert "operator_semantics_label" in report["skipped_label_fields"]


# ---------------------------------------------------------------------------
# fail-closed cases
# ---------------------------------------------------------------------------


def test_missing_field_on_the_candidate_fails_closed() -> None:
    candidate = _manifest()
    candidate.pop("artifact_set_id")
    report = reproducibility_report(_manifest(), candidate)
    assert report["verdict"] == "not_reproducible"
    assert "field_unknown:artifact_set_id" in report["reasons"]
    assert "artifact_set_id" in report["missing_fields"]


def test_missing_field_on_the_reference_fails_closed() -> None:
    reference = _manifest()
    reference.pop("snapshot_checksum")
    report = reproducibility_report(reference, _manifest())
    assert report["verdict"] == "not_reproducible"
    assert "field_unknown:snapshot_checksum" in report["reasons"]


def test_absent_worker_attestation_fails_closed() -> None:
    """A request that never asked for a manifest cannot be proven reproducible."""

    report = reproducibility_report(_manifest(worker_attestation=None), _manifest())
    assert report["verdict"] == "not_reproducible"
    assert REASON_WORKER_ATTESTATION_MISSING in report["reasons"]
    # The two nested fields are also unknown, which is reported separately.
    assert "worker_attestation.worker_file_sha256" in report["missing_fields"]
    assert "worker_attestation.worker_runtime" in report["missing_fields"]


def test_partial_attestation_fails_closed() -> None:
    partial = _manifest()
    partial[WORKER_ATTESTATION_KEY] = {"worker_file_sha256": "digest_one"}
    report = reproducibility_report(_manifest(), partial)
    assert report["verdict"] == "not_reproducible"
    assert "worker_attestation.worker_runtime" in report["missing_fields"]


def test_reasons_are_reported_once() -> None:
    report = reproducibility_report(_manifest(worker_attestation=None), _manifest(worker_attestation=None))
    assert report["reasons"].count(REASON_WORKER_ATTESTATION_MISSING) == 1


# ---------------------------------------------------------------------------
# empty values are not evidence (fail-closed identity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "empty"),
    [
        ("snapshot_checksum", ""),
        ("working_set_snapshot_checksum", ""),
        ("artifact_set_id", ""),
        ("evaluation", {}),
    ],
)
def test_an_empty_identity_field_is_missing_not_equal(field: str, empty) -> None:
    """``'' == ''`` is true but proves nothing, so it must not count as checked."""

    blank = _manifest(**{field: empty})
    report = reproducibility_report(blank, blank)
    assert report["verdict"] == "not_reproducible"
    assert f"field_unknown:{field}" in report["reasons"]
    assert field in report["missing_fields"]
    assert field not in report["checked_fields"]


@pytest.mark.parametrize("value", [None, "", {}, []])
def test_none_and_empty_containers_count_as_missing(value) -> None:
    report = reproducibility_report(_manifest(snapshot_checksum=value), _manifest())
    assert report["verdict"] == "not_reproducible"
    assert "snapshot_checksum" in report["missing_fields"]


def test_one_side_empty_is_still_missing() -> None:
    """An empty value on either side is enough to make the comparison worthless."""

    report = reproducibility_report(_manifest(), _manifest(evaluation={}))
    assert report["verdict"] == "not_reproducible"
    assert "evaluation" in report["missing_fields"]


def test_a_genuinely_empty_comparison_can_never_report_reproducible() -> None:
    """The whole point: a manifest of empty strings must not read as a match."""

    skeleton = {
        "manifest_schema_version": 1,
        "operator_semantics_label": "label",
        "strategy_id": "",
        "snapshot_checksum": "",
        "working_set_snapshot_checksum": "",
        "artifact_set_id": "",
        "evaluation": {},
        "execution_costs": {},
        WORKER_ATTESTATION_KEY: None,
    }
    report = reproducibility_report(skeleton, skeleton)
    assert report["verdict"] == "not_reproducible"
    assert report["checked_fields"] == []
    # Every identity field is named, so the report says which values were absent.
    assert sorted(report["missing_fields"]) == sorted(REPRODUCIBILITY_IDENTITY_FIELDS)


def test_provenance_unavailable_is_reported_separately() -> None:
    """A manifest that says its facts are absent is named as such, not read as a match."""

    from rdagent.scenarios.gf_quant.strategy.manifest import PROVENANCE_AVAILABLE_KEY

    flagged = _manifest(**{PROVENANCE_AVAILABLE_KEY: False})
    report = reproducibility_report(flagged, flagged)
    assert report["verdict"] == "not_reproducible"
    assert "provenance_unavailable" in report["reasons"]


def test_absent_provenance_flag_defaults_to_available() -> None:
    """A manifest that never sets the flag is treated as carrying real facts."""

    assert "provenance_unavailable" not in reproducibility_report(_manifest(), _manifest())["reasons"]


@pytest.mark.parametrize("value", [[], 7, "manifest"])
def test_non_object_manifests_fail_closed(value) -> None:
    with pytest.raises(ValueError, match="reference manifest must be an object"):
        reproducibility_report(value, _manifest())
    with pytest.raises(ValueError, match="candidate manifest must be an object"):
        reproducibility_report(_manifest(), value)


def test_verdicts_are_the_two_documented_values() -> None:
    assert REPRODUCIBILITY_VERDICTS == ("reproducible", "not_reproducible")
    assert reproducibility_report(_manifest(), _manifest())["verdict"] in REPRODUCIBILITY_VERDICTS
