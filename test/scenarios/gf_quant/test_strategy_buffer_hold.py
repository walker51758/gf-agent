"""Local semantics for batch B2: HoldingBuffer (5.5), DiversitySelector (5.6)
and MinimumHoldingPeriod (8.4).

The cross-end contracts (bit-for-bit parity, identical fail-closed messages)
live in ``test_strategy_worker_parity.py``; this module pins the local-side
behaviour: buffer seat algebra, percentile boundaries, correlation screening,
the lock window arithmetic and the canonical-payload shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.gf_quant.strategy.backtest import _apply_min_holding, run_portfolio_backtest
from rdagent.scenarios.gf_quant.strategy.candidate import BacktestSpec, CompositionSpec
from rdagent.scenarios.gf_quant.strategy.composition import build_buffered_target_row, build_target_weights


# ---------------------------------------------------------------------------
# Spec validation and canonical payload shape.
# ---------------------------------------------------------------------------


def test_composition_buffer_and_diversity_validation() -> None:
    with pytest.raises(ValueError, match="holding_buffer_entry must be in \\(0, 1\\] when provided"):
        CompositionSpec(top_n=2, holding_buffer_entry=0.0)
    with pytest.raises(ValueError, match="holding_buffer_retain must be in \\(0, 1\\] when provided"):
        CompositionSpec(top_n=2, holding_buffer_entry=0.5, holding_buffer_retain=1.5)
    with pytest.raises(ValueError, match="holding_buffer_retain must not exceed holding_buffer_entry"):
        CompositionSpec(top_n=2, holding_buffer_entry=0.3, holding_buffer_retain=0.8)
    with pytest.raises(ValueError, match="require holding_buffer_entry"):
        CompositionSpec(top_n=2, holding_buffer_retain=0.5)
    with pytest.raises(ValueError, match="require holding_buffer_entry"):
        CompositionSpec(top_n=2, holding_buffer_max_extra=1)
    with pytest.raises(ValueError, match="holding_buffer_max_extra must be a whole number"):
        CompositionSpec(top_n=2, holding_buffer_entry=0.5, holding_buffer_max_extra=1.5)
    with pytest.raises(ValueError, match="holding_buffer_max_extra must not be negative"):
        CompositionSpec(top_n=2, holding_buffer_entry=0.5, holding_buffer_max_extra=-1)
    with pytest.raises(ValueError, match="diversity_max_correlation must be in \\[-1, 1\\] when provided"):
        CompositionSpec(top_n=2, diversity_max_correlation=1.5)
    with pytest.raises(ValueError, match="diversity_max_correlation must be in \\[-1, 1\\] when provided"):
        CompositionSpec(top_n=2, diversity_max_correlation=-1.5)


def test_buffer_retain_collapses_onto_entry_when_unset() -> None:
    spec = CompositionSpec(top_n=2, holding_buffer_entry=0.4)
    assert spec.holding_buffer_retain == 0.4
    # Identical resolved configuration, one identity.
    assert spec == CompositionSpec(top_n=2, holding_buffer_entry=0.4, holding_buffer_retain=0.4)


def test_min_holding_days_validation() -> None:
    with pytest.raises(ValueError, match="min_holding_days must be a whole number when provided"):
        BacktestSpec(signal_lag=1, min_holding_days=2.5)
    with pytest.raises(ValueError, match="min_holding_days must be at least 1 when provided"):
        BacktestSpec(signal_lag=1, min_holding_days=0)
    assert BacktestSpec(signal_lag=1, min_holding_days=3).min_holding_days == 3


def test_new_fields_travel_only_while_enabled() -> None:
    # Disabled: no new key may leak into the canonical payload (fact 10).
    plain_composition = CompositionSpec(top_n=4).as_payload()
    assert "holding_buffer_entry" not in plain_composition
    assert "holding_buffer_retain" not in plain_composition
    assert "holding_buffer_max_extra" not in plain_composition
    assert "diversity_max_correlation" not in plain_composition
    assert "min_holding_days" not in BacktestSpec(signal_lag=1).as_payload()

    enabled = CompositionSpec(
        top_n=4,
        holding_buffer_entry=0.6,
        holding_buffer_retain=0.3,
        holding_buffer_max_extra=2,
        diversity_max_correlation=0.5,
    ).as_payload()
    assert enabled["holding_buffer_entry"] == 0.6
    assert enabled["holding_buffer_retain"] == 0.3
    assert enabled["holding_buffer_max_extra"] == 2
    assert enabled["diversity_max_correlation"] == 0.5
    # Round-trip preserves the configuration.
    assert CompositionSpec.from_payload(enabled) == CompositionSpec(
        top_n=4,
        holding_buffer_entry=0.6,
        holding_buffer_retain=0.3,
        holding_buffer_max_extra=2,
        diversity_max_correlation=0.5,
    )
    assert BacktestSpec.from_payload({"signal_lag": 1, "min_holding_days": 4}).min_holding_days == 4
    assert "min_holding_days" not in BacktestSpec.from_payload({"signal_lag": 1}).as_payload()


# ---------------------------------------------------------------------------
# HoldingBuffer seat algebra (per-date row builder).
# ---------------------------------------------------------------------------


def _score_row(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_buffer_disabled_row_equals_the_panel_pass() -> None:
    spec = CompositionSpec(top_n=2)
    score = pd.DataFrame(
        {"A": [3.0, 1.0], "B": [2.0, 3.0], "C": [1.0, 2.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )
    for position, date in enumerate(score.index):
        expected = build_target_weights(spec, score).loc[date]
        row = build_buffered_target_row(spec, score.iloc[position], frozenset({"A"}), None)
        np.testing.assert_allclose(row.to_numpy(), expected.to_numpy())


def test_buffered_row_held_name_inside_top_n_is_kept() -> None:
    spec = CompositionSpec(top_n=2, holding_buffer_entry=0.9, holding_buffer_retain=0.9)
    # Percentiles: A 1.0, B 0.75, C 0.5, D 0.25.  B is held and inside the
    # natural top_n, so the strict entry threshold cannot expel it.
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset({"B"}), None)
    assert set(row[row > 0].index) == {"A", "B"}


def test_buffered_row_blocks_fresh_entrants_below_the_entry_threshold() -> None:
    spec = CompositionSpec(top_n=3, holding_buffer_entry=0.8, holding_buffer_retain=0.5)
    # Entrants need pct >= 0.8, i.e. the top rank only; C and D (pct 0.5 and
    # 0.25) are blocked and their seats stay open for retained holdings.
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset(), None)
    assert set(row[row > 0].index) == {"A"}


def test_buffered_row_rescues_a_held_name_via_vacated_seat_and_extra_seat() -> None:
    # D is held, dropped to the last rank (pct 0.25) but above the retention
    # threshold 0.2.  The entry threshold 0.6 keeps the fresh names at ranks
    # 1-2 (pct 1.0 / 0.75) and vacates rank 3 (pct 0.5 < 0.6): D fills the
    # vacated seat, so the book stays at top_n.
    spec = CompositionSpec(top_n=3, holding_buffer_entry=0.6, holding_buffer_retain=0.2, holding_buffer_max_extra=0)
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset({"D"}), None)
    assert set(row[row > 0].index) == {"A", "B", "D"}

    # With no seat to fill (all entrants pass) the rescued held name needs an
    # extra seat; max_extra=1 allows a fourth holding, max_extra=0 drops it.
    spec_extra = CompositionSpec(top_n=3, holding_buffer_entry=0.2, holding_buffer_retain=0.2, holding_buffer_max_extra=1)
    row_extra = build_buffered_target_row(spec_extra, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset({"D"}), None)
    assert set(row_extra[row_extra > 0].index) == {"A", "B", "C", "D"}
    spec_no_extra = CompositionSpec(top_n=3, holding_buffer_entry=0.2, holding_buffer_retain=0.2, holding_buffer_max_extra=0)
    row_no_extra = build_buffered_target_row(
        spec_no_extra, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset({"D"}), None
    )
    assert set(row_no_extra[row_no_extra > 0].index) == {"A", "B", "C"}


def test_buffered_row_drops_held_names_below_the_retention_threshold() -> None:
    # D sits at pct 0.25, below the retention threshold 0.3: nothing rescues
    # it even though an extra seat would be available.
    spec = CompositionSpec(top_n=3, holding_buffer_entry=0.35, holding_buffer_retain=0.3, holding_buffer_max_extra=1)
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset({"D"}), None)
    assert set(row[row > 0].index) == {"A", "B", "C"}


def test_buffered_row_thresholds_are_inclusive_and_rank_weighting_follows_the_selection() -> None:
    # C sits exactly on both thresholds (pct 0.5): inclusive comparisons keep
    # it as an entrant (rank weighting gives it the smallest weight).
    spec = CompositionSpec(top_n=3, weighting="rank", holding_buffer_entry=0.5, holding_buffer_retain=0.5)
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset(), None)
    assert set(row[row > 0].index) == {"A", "B", "C"}
    # Rank weights for three names: 3/6, 2/6, 1/6, best score first.
    assert row["A"] == pytest.approx(3.0 / 6.0)
    assert row["B"] == pytest.approx(2.0 / 6.0)
    assert row["C"] == pytest.approx(1.0 / 6.0)


def test_buffered_row_cannot_rescue_a_name_without_a_valid_score() -> None:
    spec = CompositionSpec(top_n=2, holding_buffer_entry=0.4, holding_buffer_retain=0.2)
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": np.nan}), frozenset({"C"}), None)
    assert set(row[row > 0].index) == {"A", "B"}


def test_buffered_row_applies_cap_and_floor_after_the_selection() -> None:
    # Buffered selection of three names: the cap/floor stack runs on the
    # buffered set exactly as on a natural one.  Equal weights 1/3 stay below
    # the 0.5 cap, so the row is untouched and D is never pulled in.
    spec = CompositionSpec(top_n=3, weight_cap=0.5, holding_buffer_entry=0.6, holding_buffer_retain=0.2, holding_buffer_max_extra=0)
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset({"C", "D"}), None)
    # A and B enter; C is held and stays inside the natural top_n; D (held,
    # above the retention threshold) finds no vacated seat and no extra seat.
    assert set(row[row > 0].index) == {"A", "B", "C"}
    np.testing.assert_allclose(row[["A", "B", "C"]], 1.0 / 3.0)
    assert row["D"] == 0.0


# ---------------------------------------------------------------------------
# DiversitySelector (per-date row builder).
# ---------------------------------------------------------------------------


def _returns_history() -> pd.DataFrame:
    rng = np.random.RandomState(7)
    market = rng.normal(size=40)
    return pd.DataFrame(
        {
            "A": market,
            "B": market,  # perfectly correlated with A
            "C": rng.normal(size=40),
            "D": -market,  # perfectly negatively correlated with A
        }
    )


def test_diversity_screen_skips_names_too_correlated_with_admitted_ones() -> None:
    spec = CompositionSpec(top_n=3, diversity_max_correlation=0.5)
    row = build_buffered_target_row(spec, _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}), frozenset(), _returns_history())
    chosen = set(row[row > 0].index)
    assert "A" in chosen  # best score is always admitted first
    assert "B" not in chosen  # corr(A, B) = 1 > 0.5 is blocked
    assert "D" in chosen  # corr(A, D) = -1 <= 0.5 passes (negative corr diversifies)
    assert row.sum() == pytest.approx(1.0)


def test_diversity_boundaries_and_disabled_paths() -> None:
    scores = _score_row({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
    history = _returns_history()
    # A bound of 1.0 can never be exceeded: the screen is inert.
    inert = build_buffered_target_row(
        CompositionSpec(top_n=3, diversity_max_correlation=1.0), scores, frozenset(), history
    )
    plain = build_buffered_target_row(CompositionSpec(top_n=3), scores, frozenset(), history)
    np.testing.assert_allclose(inert.to_numpy(), plain.to_numpy())
    # Without a returns history the screen has no labels and never binds.
    no_history = build_buffered_target_row(
        CompositionSpec(top_n=3, diversity_max_correlation=0.5), scores, frozenset(), None
    )
    np.testing.assert_allclose(no_history.to_numpy(), plain.to_numpy())
    # A single-row history has no correlation evidence either.
    one_row = build_buffered_target_row(
        CompositionSpec(top_n=3, diversity_max_correlation=0.5), scores, frozenset(), history.iloc[:1]
    )
    np.testing.assert_allclose(one_row.to_numpy(), plain.to_numpy())


# ---------------------------------------------------------------------------
# MinimumHoldingPeriod (execution layer).
# ---------------------------------------------------------------------------


def test_min_holding_blocks_a_sell_and_releases_after_n_days() -> None:
    dates = pd.bdate_range("2024-01-02", periods=7)
    zeros = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    targets = pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        },
        index=dates,
    )
    result = run_portfolio_backtest(
        targets, zeros, spec=BacktestSpec(signal_lag=1, min_holding_days=2)
    )
    weights = result.weights
    # p1: A is bought.  p2: the target exits A but it is 1 day old -> the
    # sell is blocked and the budget for B is squeezed to zero.  p3: A has
    # completed exactly 2 days -> the sell goes through.
    assert weights.loc[dates[1], "A"] == pytest.approx(1.0)
    assert weights.loc[dates[2], "A"] == pytest.approx(1.0)
    assert weights.loc[dates[2], "B"] == pytest.approx(0.0)
    assert weights.loc[dates[3], "B"] == pytest.approx(1.0)
    # p4: B was bought on p3, so the exit decided on d3 is blocked again;
    # p5: released, A re-entered on p5 gets a *fresh* lock (its old entry was
    # cleared when it was sold), so the exit decided on d5 stays blocked on p6.
    assert weights.loc[dates[4], "B"] == pytest.approx(1.0)
    assert weights.loc[dates[4], "A"] == pytest.approx(0.0)
    assert weights.loc[dates[5], "A"] == pytest.approx(1.0)
    assert weights.loc[dates[6], "A"] == pytest.approx(1.0)
    assert weights.loc[dates[6], "B"] == pytest.approx(0.0)
    # Conservation: no executed row ever exceeds the 100% budget.
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_min_holding_redistributes_the_budget_proportionally() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    zeros = pd.DataFrame(0.0, index=dates, columns=["A", "B", "C"])
    targets = pd.DataFrame(
        {
            "A": [0.5, 0.0, 0.0],
            "B": [0.5, 0.5, 0.5],
            "C": [0.0, 0.5, 0.5],
        },
        index=dates,
    )
    result = run_portfolio_backtest(targets, zeros, spec=BacktestSpec(signal_lag=1, min_holding_days=2))
    weights = result.weights
    # p2: A *and* B were bought on the same day, so both are locked; A is
    # being sold and keeps its carried 0.5, B is not selling and stays at its
    # planned 0.5, so the fresh C buy is squeezed to zero by the budget.
    assert weights.loc[dates[2], "A"] == pytest.approx(0.5)
    assert weights.loc[dates[2], "B"] == pytest.approx(0.5)
    assert weights.loc[dates[2], "C"] == pytest.approx(0.0)


def test_apply_min_holding_redistributes_across_lock_ages() -> None:
    # Mixed lock ages: B (entered yesterday) is locked and being sold, D
    # (entered yesterday) is locked but only buying, A (entered two days ago)
    # has aged out.  B keeps its carried weight, the unlocked names fund the
    # take-back proportionally.
    carried = pd.Series({"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2})
    executed = pd.Series({"A": 0.0, "B": 0.0, "C": 0.5, "D": 0.5})
    adjusted = _apply_min_holding(carried, executed, {"A": 0, "B": 1, "D": 1}, 2, 2)
    assert adjusted["B"] == pytest.approx(0.3)  # sell blocked by the lock
    assert adjusted["A"] == pytest.approx(0.0)  # aged out, the sell executes
    assert adjusted["C"] == pytest.approx(0.2)  # 0.5 scaled by the 0.4 factor
    assert adjusted["D"] == pytest.approx(0.5)  # a locked buy is never touched
    assert adjusted.sum() == pytest.approx(1.0)


def test_min_holding_requires_composition_and_score_together() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    targets = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=dates)
    score = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    with pytest.raises(ValueError, match="composition and score must be provided together"):
        run_portfolio_backtest(targets, returns, spec=BacktestSpec(signal_lag=1), composition=CompositionSpec(top_n=1))
    with pytest.raises(ValueError, match="composition and score must be provided together"):
        run_portfolio_backtest(targets, returns, spec=BacktestSpec(signal_lag=1), score=score)
    with pytest.raises(ValueError, match="score panel is missing target weight dates"):
        run_portfolio_backtest(
            targets,
            returns,
            spec=BacktestSpec(signal_lag=1),
            composition=CompositionSpec(top_n=1),
            score=score.iloc[:1],
        )


def test_min_holding_and_buffer_stack_with_band_and_shrinkage() -> None:
    """band -> shrink -> lock coexist in one loop on a deterministic fixture.

    All returns are zero, so carried always equals the previous executed
    weights and every number below is hand-traceable: the shrink halves each
    move, the lock then pushes a being-sold name back to its carried weight,
    and the remaining budget is scaled down proportionally (the executed
    total, already reduced by the shrinkage, is conserved).
    """

    dates = pd.bdate_range("2024-01-02", periods=6)
    zeros = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    targets = pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "B": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )
    spec = BacktestSpec(
        signal_lag=1,
        min_holding_days=2,
        drift_band_name=0.01,
        turnover_shrinkage=0.5,
    )
    result = run_portfolio_backtest(targets, zeros, spec=spec)
    weights = result.weights
    # p1: the initial buy of A is shrunk to 0.5.
    assert weights.loc[dates[1], "A"] == pytest.approx(0.5)
    # p2: shrink halves the A->B move (A 0.25 / B 0.5), then the lock pushes
    # the one-day-old A back to 0.5 and B is squeezed to 0.25 by the budget.
    assert weights.loc[dates[2], "A"] == pytest.approx(0.5)
    assert weights.loc[dates[2], "B"] == pytest.approx(0.25)
    # p3: A has aged out of the lock; the shrink-scaled switch executes.
    assert weights.loc[dates[3], "A"] == pytest.approx(0.25)
    assert weights.loc[dates[3], "B"] == pytest.approx(0.625)
    # p4: B was entered on p2, so it has completed 2 days and the sell decided
    # on d3 goes through after the shrink only.
    assert weights.loc[dates[4], "A"] == pytest.approx(0.625)
    assert weights.loc[dates[4], "B"] == pytest.approx(0.3125)
    # p5: the switch back to A continues.
    assert weights.loc[dates[5], "A"] == pytest.approx(0.8125)
    assert weights.loc[dates[5], "B"] == pytest.approx(0.15625)
    # Conservation: no executed row ever exceeds the 100% budget.
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_min_holding_abandons_the_trade_when_the_locked_book_cannot_be_funded() -> None:
    """All-locked scenario: the raise-back has no funding source.

    A and B are bought together and both are still locked when the target
    doubles B and exits A.  Raising A back to its carried weight needs 0.5 of
    budget, but the unlocked book is empty -- partially raising A would mean
    partially selling it, so the whole rebalance is abandoned and the book
    keeps its carried weights.  The baseline (no lock) trades normally, and
    every executed row of the locked run stays within the 100% budget.
    """

    dates = pd.bdate_range("2024-01-02", periods=4)
    zeros = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    targets = pd.DataFrame(
        {
            "A": [0.5, 0.0, 0.0, 0.5],
            "B": [0.5, 1.0, 1.0, 0.5],
        },
        index=dates,
    )
    locked = run_portfolio_backtest(targets, zeros, spec=BacktestSpec(signal_lag=1, min_holding_days=5))
    baseline = run_portfolio_backtest(targets, zeros, spec=BacktestSpec(signal_lag=1))
    # Baseline rows are exactly the targets: always fully invested.
    assert (baseline.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    # p2: both names are locked, the raise-back of A is unfundable -> the
    # trade is abandoned and the book keeps {A: 0.5, B: 0.5} (row sum 1.0)
    # instead of the pre-fix inflation {A: 0.5, B: 1.0}.
    assert locked.weights.loc[dates[2], "A"] == pytest.approx(0.5)
    assert locked.weights.loc[dates[2], "B"] == pytest.approx(0.5)
    assert locked.weights.loc[dates[2]].sum() == pytest.approx(1.0)
    # Conservation invariant on the whole panel: no executed row ever exceeds
    # the 100% budget.
    assert (locked.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    # The abandonment is a skip, not a freeze of the budget math: p4's target
    # equals the carried book, and the row stays inside the budget either way.
    p4 = locked.weights.loc[dates[3]]
    assert p4["A"] + p4["B"] <= 1.0 + 1e-9


def test_apply_min_holding_abandons_when_unlocked_budget_is_insufficient() -> None:
    # A is locked and being sold; the unlocked names can only give up 0.1 of
    # executed weight against the 0.6 needed: the clamp refuses the whole
    # trade and the caller keeps the carried weights.
    carried = pd.Series({"A": 0.6, "B": 0.2, "C": 0.2})
    executed = pd.Series({"A": 0.0, "B": 0.05, "C": 0.05})
    assert _apply_min_holding(carried, executed, {"A": 0}, 1, 5) is None
    # Exactly sufficient budget stays on the scaling branch (strict <).
    executed_tight = pd.Series({"A": 0.0, "B": 0.3, "C": 0.3})
    adjusted = _apply_min_holding(carried, executed_tight, {"A": 0}, 1, 5)
    assert adjusted is not None
    assert adjusted["A"] == pytest.approx(0.6)
    assert adjusted["B"] == pytest.approx(0.0)
    assert adjusted["C"] == pytest.approx(0.0)
    # The executed total (0.6 -- the target holds 40% cash) is conserved.
    assert adjusted.sum() == pytest.approx(0.6)
