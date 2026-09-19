"""Deployment gate (roadmap 10.10): verdict from evidence, never from P&L.

The roadmap asks for three outcomes -- deploy, research-only, reject -- and
says the thresholds are system-owned and cannot be lowered by a strategy.  It
does *not* give any number, and this module does not invent one: the report
hard-codes ``thresholds_defined: false`` and states why.  What this gate does
check is **evidence completeness and internal consistency**, which is what the
roadmap's inputs actually name (walk-forward, feasibility, cost sensitivity,
dedup, data leakage).

The three outcomes, in the order they are decided:

``reject``
    Two hard defects, drawn only from fields existing reports already carry: a
    failed strategy acceptance gate, or a ``data_leakage`` finding at the
    breach severity (the existing ``LEAKAGE_BREACH_SEVERITY`` constant, reused
    rather than re-declared) or a recorded feasibility violation.
``research_only``
    Any required evidence section is absent, or the candidate is a duplicate of
    an already-registered strategy.  Missing evidence is not a defect, and a
    duplicate is not an error -- it simply carries no incremental value
    to deploy.
``allow_deploy``
    Everything required is present, there is no hard defect, the candidate is
    not a duplicate, and the reproducibility check passed.

A high research-period return never promotes a candidate: returns are not an
input to this function at all, which is exactly how the roadmap's stated risk
("mistaking a research-window high return for a deployment permit") is closed.
The policy object carries no thresholds for the same reason -- there is nothing
to lower because nothing was set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The three deployment outcomes, most permissive last.
DEPLOYMENT_VERDICTS: tuple[str, ...] = ("reject", "research_only", "allow_deploy")

#: Evidence sections the gate requires.  Section names are the keys of the
#: ``audit_report`` produced by the worker (roadmap 10.1 / 10.2 / 10.6 / 10.7).
DEPLOYMENT_REQUIRED_EVIDENCE: tuple[str, ...] = (
    "walk_forward",
    "feasibility",
    "cost_sensitivity",
    "data_leakage",
)

#: Reason recorded while the roadmap gives no numeric threshold for this gate.
REASON_THRESHOLDS_UNDEFINED = "thresholds_undefined_roadmap_10_10"

#: The statement a reader must see next to every verdict.
THRESHOLDS_NOTE = "roadmap 10.10 does not state numeric thresholds; this gate reports evidence completeness and consistency only"


@dataclass(frozen=True)
class DeploymentPolicy:
    """The system-owned deployment policy.

    Deliberately threshold-free: the roadmap fixes the *shape* of the decision
    (which evidence is required, which defects are hard) and leaves the numbers
    to a future calibration, so this object records that choice rather than
    inventing thresholds that a reader would then mistake for calibrated ones.
    """

    required_evidence: tuple[str, ...] = DEPLOYMENT_REQUIRED_EVIDENCE
    thresholds_defined: bool = False
    notes: tuple[str, ...] = (THRESHOLDS_NOTE,)

    def __post_init__(self) -> None:
        if not isinstance(self.required_evidence, tuple) or not self.required_evidence:
            raise ValueError("deployment policy required_evidence must be a non-empty tuple")
        if any(not isinstance(name, str) or not name for name in self.required_evidence):
            raise ValueError("deployment policy required_evidence must be non-empty strings")
        if self.thresholds_defined:
            raise ValueError("deployment policy thresholds are not defined by roadmap 10.10; do not claim otherwise")


def _present(audit_report: Mapping[str, Any] | None, name: str) -> bool:
    """Whether an evidence section is actually present.

    ``None`` and an empty mapping both count as absent: a section that reports
    nothing is not evidence.
    """

    if not isinstance(audit_report, Mapping):
        return False
    section = audit_report.get(name)
    return isinstance(section, Mapping) and bool(section)


def _data_leakage_breach(audit_report: Mapping[str, Any] | None) -> list[str]:
    """Names of leakage findings that are at the existing breach severity."""

    from rdagent.scenarios.gf_quant.strategy.audit import LEAKAGE_BREACH_SEVERITY

    if not isinstance(audit_report, Mapping):
        return []
    section = audit_report.get("data_leakage")
    if not isinstance(section, Mapping):
        return []
    findings = section.get("findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
        return []
    breaches: list[str] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        if finding.get("severity") == LEAKAGE_BREACH_SEVERITY:
            breaches.append(str(finding.get("field", "unknown")))
    return breaches


def _feasibility_violations(audit_report: Mapping[str, Any] | None) -> list[str]:
    """Reasons of feasibility violations already recorded by that report."""

    if not isinstance(audit_report, Mapping):
        return []
    section = audit_report.get("feasibility")
    if not isinstance(section, Mapping):
        return []
    violations = section.get("violations")
    if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes)):
        return []
    return [str(item.get("reason", "unknown")) if isinstance(item, Mapping) else str(item) for item in violations]


def deployment_verdict(
    *,
    gate_failures: Sequence[str],
    audit_report: Mapping[str, Any] | None,
    dedup_is_duplicate: bool,
    reproducibility: Mapping[str, Any] | None,
    policy: DeploymentPolicy | None = None,
) -> dict[str, Any]:
    """Return the roadmap 10.10 verdict plus the evidence trail behind it."""

    policy = policy or DeploymentPolicy()
    failures = [str(item) for item in gate_failures]
    breaches = _data_leakage_breach(audit_report)
    violations = _feasibility_violations(audit_report)
    missing = [name for name in policy.required_evidence if not _present(audit_report, name)]

    reproducible = (
        isinstance(reproducibility, Mapping) and reproducibility.get("verdict") == "reproducible"
    )

    reasons: list[str] = []
    if failures:
        reasons.append("acceptance_gate_failed")
    if breaches:
        reasons.append("data_leakage_breach")
    if violations:
        reasons.append("feasibility_violation")
    if failures or breaches or violations:
        verdict = "reject"
    elif missing:
        reasons.append("required_evidence_missing")
        verdict = "research_only"
    elif dedup_is_duplicate:
        reasons.append("duplicate_strategy")
        verdict = "research_only"
    elif not reproducible:
        reasons.append("reproducibility_not_confirmed")
        verdict = "research_only"
    else:
        verdict = "allow_deploy"

    if policy.thresholds_defined:
        reasons.append("thresholds_defined")
    else:
        reasons.append(REASON_THRESHOLDS_UNDEFINED)

    return {
        "verdict": verdict,
        "reasons": reasons,
        "thresholds_defined": policy.thresholds_defined,
        "notes": list(policy.notes),
        "required_evidence": list(policy.required_evidence),
        "missing_evidence": missing,
        "gate_failures": failures,
        "leakage_breaches": breaches,
        "feasibility_violations": violations,
        "duplicate": bool(dedup_is_duplicate),
        "reproducibility_verdict": (
            reproducibility.get("verdict") if isinstance(reproducibility, Mapping) else None
        ),
    }
