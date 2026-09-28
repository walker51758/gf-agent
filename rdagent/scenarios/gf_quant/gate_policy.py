"""TEMPORARY gate selection for the 22-23 search regime.

One calibrated requirement is switched off while the data machine is unreachable
and the benchmark pools have to be re-measured: ``annual_min_valid_years``, the
demand that a candidate's RankIC be finite in at least N calendar years.

The reason is arithmetic rather than qualitative.  The 22-23 evaluation window
covers two calendar years, so ``metrics_by_year`` can hold at most two entries,
while the calibrated standard asks for four.  On this window that requirement is
unreachable for every candidate, and a gate that no candidate can satisfy
describes the window rather than the factor -- so it is set aside until the
window it was calibrated for is measurable again.

Everything else stays on, including the annual *sign* requirement
(``annual_rank_ic_sign_alignment``), which a two-year window can still satisfy.

**To restore the full gate set, set** ``TEMPORARY_DISABLED_GATES = ()``.  That is
the whole revert; ``TEMPORARY_ACTIVE_GATES`` is derived from it, so there is one
place to edit.  No threshold in ``benchmark_standards.json`` is touched, and no
gate implementation is deleted -- a switched-off requirement is still evaluated
and its verdict is still reported, it just cannot fail a candidate.
"""

from __future__ import annotations

#: The ten calibrated gate names in the order ``evaluate_against_calibration``
#: appends them to its failure list.  Kept here as the one authoritative order:
#: report tables group them differently, so an ordinal slice of one ordering is
#: not an ordinal slice of the other.  Selection below is by name for that reason.
GATE_ORDER: tuple[str, ...] = (
    "abs_rank_ic",
    "abs_rank_icir",
    "coverage",
    "turnover",
    "observations",
    "direction_adjusted_annualized_return",
    "direction_adjusted_sharpe",
    "abs_drawdown",
    "annual_rank_ic_years",
    "annual_rank_ic_sign_alignment",
)

#: TEMPORARY: the requirements switched off in this regime.
#:
#: ``annual_rank_ic_years`` is the failure name produced by the
#: ``annual_min_valid_years`` standard -- the standard's name and the gate's
#: failure name differ, and the switch has to name the failure.  Reverting is
#: ``TEMPORARY_DISABLED_GATES = ()``.
TEMPORARY_DISABLED_GATES: tuple[str, ...] = (
    "annual_rank_ic_years",
)

#: Maps the switched-off standard to the failure name it produces, so a reader
#: can go from the standard in ``benchmark_standards.json`` to the switch here.
STANDARD_TO_FAILURE = {
    "annual_min_valid_years": "annual_rank_ic_years",
}


def active_gates() -> tuple[str, ...] | None:
    """The gate names enforced, or ``None`` when every gate is enforced."""

    disabled = set(TEMPORARY_DISABLED_GATES)
    if not disabled:
        return None
    return tuple(name for name in GATE_ORDER if name not in disabled)


def is_active_under(name: str, selected: tuple[str, ...] | None) -> bool:
    """Whether ``name`` is enforced given an explicit selection (``None`` = all).

    A name outside :data:`GATE_ORDER` is always enforced: the switch governs the
    ten calibrated gates, and a structural failure such as a missing return
    profile, or the annual block's own ``annual_metrics_by_year_missing`` guard,
    is not one of them and must not be hidden by it.
    """

    if selected is None or name not in GATE_ORDER:
        return True
    return name in selected


def partition(names: tuple[str, ...], selected: tuple[str, ...] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split ``names`` into (enforced, suppressed) for ``selected``.

    ``selected=None`` enforces everything, so the split is ``(names, ())``.
    """

    if selected is None:
        return tuple(names), ()
    kept = tuple(name for name in names if is_active_under(name, selected))
    dropped = tuple(name for name in names if not is_active_under(name, selected))
    return kept, dropped


def assert_switch_is_well_formed() -> None:
    """Every disabled name must be a real calibrated gate, and at least one stay on.

    Called by the tests rather than at import: a wrong switch value should fail a
    check, not prevent a module from loading.
    """

    unknown = sorted(set(TEMPORARY_DISABLED_GATES) - set(GATE_ORDER))
    if unknown:
        raise AssertionError(f"switch names gates that do not exist: {unknown}")
    for standard, failure in STANDARD_TO_FAILURE.items():
        if failure not in GATE_ORDER:
            raise AssertionError(f"{standard!r} maps to unknown gate {failure!r}")
    selected = active_gates()
    if selected is not None and not selected:
        raise AssertionError("the switch disables every gate; that is not a supported regime")
