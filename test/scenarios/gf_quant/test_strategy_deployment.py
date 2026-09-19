"""DeploymentGate (roadmap 10.10): three verdicts from evidence, never from P&L.

The roadmap fixes the shape of the decision and leaves the numbers unstated, so
these tests pin the shape: which evidence is required, which defects are hard,
that a missing section is only research-only, and that no threshold was
invented.  ``deployment_verdict`` takes no metrics argument at all, which is
the structural answer to the roadmap's stated risk.
"""

from __future__ import annotations

import pytest

from rdagent.scenarios.gf_quant.strategy.deployment import (
    DEPLOYMENT_REQUIRED_EVIDENCE,
    DEPLOYMENT_VERDICTS,
    REASON_THRESHOLDS_UNDEFINED,
    THRESHOLDS_NOTE,
    DeploymentPolicy,
    deployment_verdict,
)

_REPRODUCIBLE = {"verdict": "reproducible", "reasons": []}


def _audit_report(**overrides):
    report = {
        "walk_forward": {"segments": [], "summary": {"oos": {}, "is": {}}},
        "feasibility": {"days": [], "violations": []},
        "cost_sensitivity": {"production": {}, "scenarios": []},
        "data_leakage": {"findings": []},
    }
    report.update(overrides)
    return report


def _verdict(**overrides):
    payload = {
        "gate_failures": (),
        "audit_report": _audit_report(),
        "dedup_is_duplicate": False,
        "reproducibility": _REPRODUCIBLE,
    }
    payload.update(overrides)
    return deployment_verdict(**payload)


def test_complete_evidence_deploys() -> None:
    report = _verdict()
    assert report["verdict"] == "allow_deploy"
    assert report["missing_evidence"] == []


def test_verdict_vocabulary_is_the_three_documented_values() -> None:
    assert DEPLOYMENT_VERDICTS == ("reject", "research_only", "allow_deploy")
    assert _verdict()["verdict"] in DEPLOYMENT_VERDICTS


def test_required_evidence_is_the_roadmap_input_list() -> None:
    assert DEPLOYMENT_REQUIRED_EVIDENCE == ("walk_forward", "feasibility", "cost_sensitivity", "data_leakage")


# ---------------------------------------------------------------------------
# reject: hard defects only
# ---------------------------------------------------------------------------


def test_failed_acceptance_gate_rejects() -> None:
    report = _verdict(gate_failures=("min_sharpe",))
    assert report["verdict"] == "reject"
    assert "acceptance_gate_failed" in report["reasons"]
    assert report["gate_failures"] == ["min_sharpe"]


def test_high_severity_leakage_finding_rejects() -> None:
    audit = _audit_report(
        data_leakage={"findings": [{"field": "index_membership_lag", "severity": "high"}]}
    )
    report = _verdict(audit_report=audit)
    assert report["verdict"] == "reject"
    assert "data_leakage_breach" in report["reasons"]
    assert report["leakage_breaches"] == ["index_membership_lag"]


def test_medium_severity_leakage_finding_does_not_reject() -> None:
    audit = _audit_report(
        data_leakage={"findings": [{"field": "signal_lag", "severity": "medium"}]}
    )
    report = _verdict(audit_report=audit)
    assert report["verdict"] == "allow_deploy"
    assert report["leakage_breaches"] == []


def test_feasibility_violation_rejects() -> None:
    audit = _audit_report(feasibility={"days": [], "violations": [{"reason": "participation"}]})
    report = _verdict(audit_report=audit)
    assert report["verdict"] == "reject"
    assert report["feasibility_violations"] == ["participation"]


def test_reject_beats_missing_evidence() -> None:
    """A defect is a defect even when the rest of the evidence is absent."""

    report = _verdict(gate_failures=("min_sharpe",), audit_report=None)
    assert report["verdict"] == "reject"


# ---------------------------------------------------------------------------
# research_only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", DEPLOYMENT_REQUIRED_EVIDENCE)
def test_each_missing_section_is_research_only(section: str) -> None:
    audit = _audit_report()
    audit.pop(section)
    report = _verdict(audit_report=audit)
    assert report["verdict"] == "research_only"
    assert "required_evidence_missing" in report["reasons"]
    assert report["missing_evidence"] == [section]


def test_absent_audit_report_is_research_only_not_reject() -> None:
    """Missing evidence is not a defect; the roadmap's middle band is for exactly this."""

    report = _verdict(audit_report=None)
    assert report["verdict"] == "research_only"
    assert "required_evidence_missing" in report["reasons"]
    assert report["missing_evidence"] == list(DEPLOYMENT_REQUIRED_EVIDENCE)


def test_an_empty_section_counts_as_absent() -> None:
    report = _verdict(audit_report=_audit_report(feasibility={}))
    assert report["verdict"] == "research_only"
    assert "feasibility" in report["missing_evidence"]


def test_duplicate_strategy_is_research_only() -> None:
    report = _verdict(dedup_is_duplicate=True)
    assert report["verdict"] == "research_only"
    assert "duplicate_strategy" in report["reasons"]


def test_unconfirmed_reproducibility_is_research_only() -> None:
    report = _verdict(reproducibility={"verdict": "not_reproducible", "reasons": ["field_mismatch:snapshot_checksum"]})
    assert report["verdict"] == "research_only"
    assert "reproducibility_not_confirmed" in report["reasons"]
    assert report["reproducibility_verdict"] == "not_reproducible"


def test_absent_reproducibility_report_is_research_only() -> None:
    report = _verdict(reproducibility=None)
    assert report["verdict"] == "research_only"
    assert "reproducibility_not_confirmed" in report["reasons"]


# ---------------------------------------------------------------------------
# the undefined-threshold posture
# ---------------------------------------------------------------------------


def test_thresholds_are_declared_undefined() -> None:
    report = _verdict()
    assert report["thresholds_defined"] is False
    assert THRESHOLDS_NOTE in report["notes"]
    assert REASON_THRESHOLDS_UNDEFINED in report["reasons"]
    assert DEPLOYMENT_REQUIRED_EVIDENCE == tuple(report["required_evidence"])


def test_a_policy_cannot_claim_defined_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds are not defined"):
        DeploymentPolicy(thresholds_defined=True)


def test_policy_rejects_an_empty_evidence_list() -> None:
    with pytest.raises(ValueError, match="required_evidence"):
        DeploymentPolicy(required_evidence=())


def test_policy_rejects_blank_evidence_names() -> None:
    with pytest.raises(ValueError, match="required_evidence"):
        DeploymentPolicy(required_evidence=("",))


def test_default_policy_matches_the_module_constant() -> None:
    assert DeploymentPolicy().required_evidence == DEPLOYMENT_REQUIRED_EVIDENCE


# ---------------------------------------------------------------------------
# structural guarantee: returns are not an input
# ---------------------------------------------------------------------------


def test_returns_are_not_an_input() -> None:
    """The roadmap's risk is mistaking a research-window return for a permit.

    This is enforced structurally: the function has no parameter that could
    carry a metric, so no strong backtest can promote a candidate.
    """

    import inspect

    parameters = set(inspect.signature(deployment_verdict).parameters)
    assert parameters == {
        "gate_failures",
        "audit_report",
        "dedup_is_duplicate",
        "reproducibility",
        "policy",
    }
    for forbidden in ("metrics", "returns", "sharpe", "annualized_return"):
        assert forbidden not in parameters
