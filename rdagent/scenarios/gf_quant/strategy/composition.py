"""Deterministic portfolio construction from cross-sectional scores.

The composition layer is a pure function: ``score``/``eligible`` panels in,
target weights out.  No I/O, no calendar inference and no look-ahead -- the
caller decides which dates are rebalance dates.  Panel orientation matches the
remote worker exactly (``index=dt`` trading dates, ``columns=symbol``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from rdagent.scenarios.gf_quant.strategy.candidate import RISK_PARITY_MAX_ITER, CompositionSpec

#: Reserved label for selected names that the industry map does not cover
#: (roadmap 6.3): they share one implicit group and the industry cap applies
#: to that group like any other.  Mirrored verbatim in the worker.
_UNKNOWN_INDUSTRY = "__no_industry__"


def _scheme_weights(
    spec: CompositionSpec, scores: pd.Series, returns_history: pd.DataFrame | None = None
) -> pd.Series:
    """Unconstrained weights for the selected names, summing to one.

    ``returns_history`` (realised daily returns up to and including the
    decision date) is only consumed by the ``inverse_volatility`` scheme; the
    other schemes ignore it.
    """

    count = len(scores)
    if spec.weighting == "equal":
        return pd.Series(1.0 / count, index=scores.index)
    if spec.weighting == "rank":
        # Rank 1 (best score) gets the largest weight; ties are impossible here
        # because selection already ordered equal scores deterministically.
        ranks = pd.Series(range(count, 0, -1), index=scores.index, dtype=float)
        return ranks / ranks.sum()
    if spec.weighting == "score_transform":
        return _score_transform_weights(spec, scores)
    if spec.weighting == "inverse_volatility":
        return _inverse_volatility_weights(spec, scores, returns_history)
    if spec.weighting == "hierarchical":
        return _hierarchical_weights(spec, scores)
    if spec.weighting == "risk_parity":
        return _risk_parity_weights(spec, scores, returns_history)
    if spec.weighting == "factor_risk_budget":
        return _factor_risk_budget_weights(spec, scores)
    # Proportional weighting (free-cash-flow-rate style) only makes sense for
    # non-negative scores; a negative score would silently become a negative
    # position, so fail closed instead.
    if (scores <= 0).any():
        raise ValueError("proportional weighting requires strictly positive scores")
    return scores / scores.sum()


def _inverse_volatility_weights(
    spec: CompositionSpec, scores: pd.Series, returns_history: pd.DataFrame | None
) -> pd.Series:
    """InverseVolatilityWeight (roadmap 7.5): weights proportional to 1/vol.

    The volatility of every selected name is the standard deviation of its
    realised daily returns over the last ``inverse_vol_window`` rows of the
    history passed in (realised returns up to and including the decision date,
    never beyond it).  A window shorter than the history is honoured by
    slicing; a *panel* with fewer than two rows cannot define any volatility,
    so the row falls back to equal weighting -- a declared conservative
    fallback for the first decision dates of a backtest, where only one
    realised return exists.  With a usable panel, a selected name with fewer
    than two valid observations (single-point history) or a non-positive
    volatility (zero variance) fails closed with the same message on both
    engine sides; ``inverse_vol_min`` floors positive-but-tiny estimates.
    """

    if returns_history is None:
        raise ValueError("inverse_volatility weighting requires a returns panel")
    if len(returns_history) < 2:
        return pd.Series(1.0 / len(scores), index=scores.index)
    window = returns_history.tail(int(spec.inverse_vol_window))
    vols: dict[str, float] = {}
    for symbol in scores.index:
        series = window[symbol].dropna() if symbol in window.columns else pd.Series(dtype=float)
        vol = float(series.std()) if len(series) >= 2 else float("nan")
        if not math.isfinite(vol) or vol <= 0.0:
            raise ValueError(
                "inverse_volatility weighting requires a positive volatility estimate for every selected name"
            )
        if spec.inverse_vol_min is not None:
            vol = max(vol, float(spec.inverse_vol_min))
        vols[symbol] = vol
    if spec.inverse_vol_multiply_score:
        raw = pd.Series({symbol: float(scores[symbol]) / vols[symbol] for symbol in scores.index})
    else:
        raw = pd.Series({symbol: 1.0 / vols[symbol] for symbol in scores.index})
    if float(raw.sum()) <= 0.0:
        raise ValueError("inverse_volatility weighting requires a positive combined weight")
    return raw / raw.sum()


def _hierarchical_weights(spec: CompositionSpec, scores: pd.Series) -> pd.Series:
    """HierarchicalWeighting (roadmap 7.8): group weight first, then inside.

    The groups come from the explicit ``hierarchical_groups`` configuration.
    Groups without selected names receive nothing (declared empty-group
    fallback); the inter-group weights are equal by default, or the explicit
    ``hierarchical_group_weights`` normalised over the groups that actually
    hold selected names.  Inside a group the weight is split equally (the
    roadmap's within-group rule is fixed to equal, a declared conservative
    reading).  A selected name outside every group has no place in the
    hierarchy and fails closed with the same message on both engine sides.
    """

    membership: dict[str, str] = {}
    for group, members in (spec.hierarchical_groups or {}).items():
        for symbol in members:
            membership[symbol] = group
    missing = sorted(set(scores.index).difference(membership))
    if missing:
        raise ValueError(f"hierarchical weighting requires every selected name to belong to a group: {missing[:3]}")
    active = sorted({membership[symbol] for symbol in scores.index})
    group_weights = spec.hierarchical_group_weights
    if group_weights is None:
        share = {group: 1.0 / len(active) for group in active}
    else:
        total = sum(float(group_weights[group]) for group in active)
        share = {group: float(group_weights[group]) / total for group in active}
    result = pd.Series(0.0, index=scores.index, dtype=float)
    for group in active:
        members = [symbol for symbol in scores.index if membership[symbol] == group]
        result[members] = share[group] / len(members)
    return result


def _risk_parity_ccd(covariance: np.ndarray, budgets: np.ndarray, tolerance: float, label: str = "risk_parity") -> np.ndarray:
    """Cyclical coordinate descent for the risk-budgeting problem.

    Solves ``min 0.5 x' S x - sum(b_i * ln(x_i))`` over ``x > 0``; at the
    optimum ``x_i * (S x)_i`` equals the budget ``b_i`` (up to the sweep
    tolerance), which is exactly the risk-contribution condition.  The
    iteration is deterministic (fixed sweep order, fixed start) so both engine
    sides produce bit-identical weights; an unconverged solve fails closed.
    """

    count = len(budgets)
    x = (budgets / budgets.sum()).astype(float)
    for _ in range(RISK_PARITY_MAX_ITER):
        max_step = 0.0
        for position in range(count):
            others = float(covariance[position] @ x) - float(covariance[position, position]) * float(x[position])
            diagonal = float(covariance[position, position])
            updated = (-others + math.sqrt(others * others + 4.0 * diagonal * float(budgets[position]))) / (
                2.0 * diagonal
            )
            max_step = max(max_step, abs(updated - float(x[position])))
            x[position] = updated
        if max_step <= tolerance:
            return x
    raise ValueError(f"{label} weighting did not converge within {RISK_PARITY_MAX_ITER} iterations")


def _risk_parity_weights(
    spec: CompositionSpec, scores: pd.Series, returns_history: pd.DataFrame | None
) -> pd.Series:
    """RiskParityWeight (roadmap 7.6): equal risk contributions, or the
    explicit ``risk_parity_budgets`` when provided.

    The covariance of the selected names is the sample covariance (ddof=1) of
    the realised daily returns up to and including the decision date over the
    last ``risk_parity_window`` rows (the window is an upper bound, mirroring
    ``inverse_volatility``), shrunk towards its own diagonal by
    ``risk_parity_shrinkage`` when ``risk_parity_shrinkage_kind="diagonal"``
    (variances kept, correlations shrunk away -- a deterministic target that
    needs no external dependency).  A history with fewer than two rows cannot
    define any covariance, so the row falls back to equal weighting (the
    declared first-decision-date fallback).  A selected name without two valid
    observations, a non-positive variance, or a still non-positive-definite
    covariance fails closed with the same message on both engine sides.
    """

    if returns_history is None:
        raise ValueError("risk_parity weighting requires a returns panel")
    if len(returns_history) < 2:
        return pd.Series(1.0 / len(scores), index=scores.index)
    window = returns_history.tail(int(spec.risk_parity_window))
    selected = list(scores.index)
    # Per-name variance gates on the pandas std (bit-for-bit the
    # inverse-volatility convention); np.cov alone leaves a tiny positive
    # residue for a constant column, so the diagonal check cannot catch it.
    for symbol in selected:
        series = window[symbol].dropna() if symbol in window.columns else pd.Series(dtype=float)
        variance = float(series.var()) if len(series) >= 2 else float("nan")
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                "risk_parity weighting requires a positive variance estimate for every selected name"
            )
    observations = window.reindex(columns=selected).dropna()
    if len(observations) < 2:
        raise ValueError(
            "risk_parity weighting requires a positive variance estimate for every selected name"
        )
    covariance = np.cov(observations.to_numpy(dtype=float), rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    if not bool(np.all(np.isfinite(covariance))):
        raise ValueError("risk_parity weighting requires a positive variance estimate for every selected name")
    if spec.risk_parity_shrinkage_kind == "diagonal" and spec.risk_parity_shrinkage > 0.0:
        intensity = float(spec.risk_parity_shrinkage)
        covariance = (1.0 - intensity) * covariance + intensity * np.diag(np.diag(covariance))
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("risk_parity weighting requires a positive definite covariance estimate") from exc
    if spec.risk_parity_budgets is not None:
        missing = [symbol for symbol in selected if symbol not in spec.risk_parity_budgets]
        if missing:
            raise ValueError(f"risk_parity_budgets must cover every selected name: {missing[:3]}")
        budgets = np.array([float(spec.risk_parity_budgets[symbol]) for symbol in selected], dtype=float)
    else:
        budgets = np.ones(len(selected), dtype=float)
    solved = _risk_parity_ccd(covariance, budgets, float(spec.risk_parity_tolerance))
    weights = pd.Series(solved / solved.sum(), index=selected)
    if not bool(np.all(np.isfinite(weights.to_numpy(dtype=float)))) or float(weights.sum()) <= 0.0:
        raise ValueError("risk_parity weighting requires a positive combined weight")
    return weights


def _factor_risk_budget_weights(spec: CompositionSpec, scores: pd.Series) -> pd.Series:
    """FactorRiskBudgetWeight (roadmap 7.7): factor risk budgets mapped onto
    the selected names through the injected exposure matrix.

    Stage one solves the risk-budgeting problem in *factor* space: the
    factor covariance (ridged by ``frb_ridge``) and the ``frb_budgets`` feed
    the same deterministic CCD solver, yielding target factor exposures
    whose risk contributions match the budgets.  Stage two maps the target
    exposures onto the selected names: every name's raw weight is its
    non-negative projection ``relu(B @ e)`` onto the target factor mix
    (a negative projected exposure cannot fund a long-only book; if every
    selected name projects non-positive the date fails closed).  Declared
    simplifications versus a full risk model: idiosyncratic risk does not
    enter the budgeting objective (it is bounded only by the ``frb_idio_max``
    exclusion cap) and the exposure mapping replaces a constrained quadratic
    program.  Selected names without an exposure or an idiosyncratic variance
    fail closed, as does a non-positive-definite (ridged) factor covariance.
    """

    factor_names = list(spec.frb_budgets or {})
    selected = list(scores.index)
    missing_budgets = [symbol for symbol in factor_names if symbol not in spec.frb_exposures]
    if missing_budgets:
        raise ValueError("frb_exposures must cover every budgeted factor")
    exposure_matrix = np.column_stack(
        [
            [float(spec.frb_exposures[name].get(symbol, float("nan"))) for symbol in selected]
            for name in factor_names
        ]
    )
    missing_exposure = [symbol for position, symbol in enumerate(selected) if not np.all(np.isfinite(exposure_matrix[position]))]
    if missing_exposure:
        raise ValueError("factor_risk_budget weighting requires an exposure for every selected name and factor")
    covariance = np.array(
        [[float(spec.frb_factor_covariance[row][column]) for column in factor_names] for row in factor_names],
        dtype=float,
    )
    if spec.frb_ridge > 0.0:
        covariance = covariance + float(spec.frb_ridge) * np.eye(len(factor_names))
    if not bool(np.all(np.isfinite(covariance))):
        raise ValueError("factor_risk_budget weighting requires a finite factor covariance")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("factor_risk_budget weighting requires a positive definite factor covariance") from exc
    budgets = np.array([float(spec.frb_budgets[name]) for name in factor_names], dtype=float)
    target = _risk_parity_ccd(covariance, budgets, 1e-8, "factor_risk_budget")
    # Non-negative projection (the declared relu): negative exposures are a
    # legal input and a negative projected exposure cannot fund a long-only
    # book, so it is floored to zero weight before normalisation.
    projected = np.maximum(exposure_matrix @ target, 0.0)
    for symbol in selected:
        if symbol not in spec.frb_idiosyncratic:
            raise ValueError("factor_risk_budget weighting requires an idiosyncratic variance for every selected name")
    if spec.frb_idio_max is not None:
        for position, symbol in enumerate(selected):
            if float(spec.frb_idiosyncratic[symbol]) > float(spec.frb_idio_max):
                projected[position] = 0.0
    if not bool(np.all(np.isfinite(projected))):
        raise ValueError("factor_risk_budget weighting requires finite projected exposures")
    if not bool((projected > 0.0).any()):
        raise ValueError("factor_risk_budget weighting requires at least one positive projected exposure")
    weights = pd.Series(projected / projected.sum(), index=selected)
    return weights


_NO_INDUSTRY_PROJECTION = "__no_industry__"

#: Composition keys consumed by the exposure projection stage (roadmap
#: 6.4/6.5/6.6/6.7/6.9).  The worker builds the params mapping from exactly
#: these payload keys; the local side builds the same-shaped mapping from the
#: spec via ``_projection_params``.
_PROJECTION_PARAM_KEYS = (
    "industry_deviation_max",
    "industry_deviation_industry_map",
    "industry_deviation_exempt",
    "industry_deviation_benchmark",
    "style_exposure_bounds",
    "style_exposure_matrix",
    "style_exposure_benchmark",
    "beta_constraint_min",
    "beta_constraint_max",
    "beta_by_symbol",
    "factor_exposure_bounds",
    "factor_exposure_matrix",
    "projection_max_iter",
    "projection_tolerance",
)

#: Ceiling on the float-noise allowance of the feasibility check (roadmap 6.9
#: "无解处理").  The allowance must not follow a user-widened
#: ``projection_tolerance``: doing so lets an early-stopped solve report a
#: genuinely infeasible point as feasible (measured before this ceiling
#: existed: tolerance 1e-2 returned a book whose every name sat 0.0375 below
#: the floor).  The ceiling is fixed and orders of magnitude below any real
#: violation (the tightest observed genuine violation is 0.015).
_PROJECTION_FEASIBILITY_CEILING = 1e-7


def _projection_params(spec: CompositionSpec) -> dict | None:
    """The projection params mapping for a spec, or None when every exposure
    constraint family is disabled (the stage is then a bit-for-bit no-op)."""

    if (
        spec.industry_deviation_max is None
        and spec.style_exposure_bounds is None
        and spec.beta_constraint_min is None
        and spec.beta_constraint_max is None
        and spec.factor_exposure_bounds is None
    ):
        return None
    params: dict = {}
    if spec.industry_deviation_max is not None:
        params["industry_deviation_max"] = spec.industry_deviation_max
        params["industry_deviation_industry_map"] = spec.industry_deviation_industry_map
        params["industry_deviation_exempt"] = spec.industry_deviation_exempt
        params["industry_deviation_benchmark"] = spec.industry_deviation_benchmark
    if spec.style_exposure_bounds is not None:
        params["style_exposure_bounds"] = spec.style_exposure_bounds
        params["style_exposure_matrix"] = spec.style_exposure_matrix
        params["style_exposure_benchmark"] = spec.style_exposure_benchmark
    if spec.beta_constraint_min is not None or spec.beta_constraint_max is not None:
        params["beta_constraint_min"] = spec.beta_constraint_min
        params["beta_constraint_max"] = spec.beta_constraint_max
        params["beta_by_symbol"] = spec.beta_by_symbol
    if spec.factor_exposure_bounds is not None:
        params["factor_exposure_bounds"] = spec.factor_exposure_bounds
        params["factor_exposure_matrix"] = spec.factor_exposure_matrix
    params["projection_max_iter"] = spec.projection_max_iter
    params["projection_tolerance"] = spec.projection_tolerance
    if spec.weight_cap is not None:
        params["weight_cap"] = spec.weight_cap
    if spec.weight_floor is not None:
        params["weight_floor"] = spec.weight_floor
    return params


def _projection_halfspace_slack(x: np.ndarray, a: np.ndarray, b: float) -> float:
    return float(a @ x) - float(b)


def _apply_constrained_projection(values: np.ndarray, symbols: Sequence[str], params: Mapping) -> np.ndarray:
    """Project the post-chain target weights onto the enabled linear
    constraint set (roadmap 6.4/6.5/6.6/6.7 jointly; the projector itself is
    roadmap 6.9 ConstrainedWeightProjection).

    Declared and locked: distance = equal-weight L2 in weight space; every
    constraint is hard (the roadmap's priority / soft-penalty knobs are
    deliberately absent -- they require a penalised objective, which would
    break the "closest feasible weights" semantics); the solver is Dykstra's
    alternating projection over the fixed constraint order below, which is
    deterministic, dependency-free and version-stable; infeasibility and
    non-convergence fail closed (the roadmap 6.6 "降级策略" reading).
    ``projection_tolerance`` only sets the stopping accuracy: the feasibility
    check uses a fixed float-noise ceiling, so a widened tolerance can turn a
    feasible run into a conservative failure but can never let an infeasible
    point through.  The
    single-name cap and the floor of the already-held names participate as
    box constraints, so the projection can never silently break the earlier
    stages of the chain; non-negativity and the full-investment budget
    (sum <= 1, cash keeps the residual) are always part of the set.
    A non-finite input row fails closed: NaN and +/-inf are upstream bugs, not
    weights, and projecting one would hand the caller a NaN book.
    """

    count = len(values)
    x = np.array(values, dtype=float, copy=True)
    if not bool(np.isfinite(x).all()):
        raise ValueError("constrained weight projection input must contain only finite weights")
    constraints: list[tuple] = []
    cap = params.get("weight_cap")
    if cap is not None:
        constraints.append(("upper", np.full(count, float(cap))))
    floor = params.get("weight_floor")
    if floor is not None:
        constraints.append(("lower", np.where(x > 0.0, float(floor), 0.0)))
    constraints.append(("lower", np.zeros(count)))
    constraints.append(("halfspace", np.ones(count), 1.0))
    deviation_map = params.get("industry_deviation_max")
    if deviation_map:
        industry_map = params.get("industry_deviation_industry_map") or {}
        exempt = set(params.get("industry_deviation_exempt") or ())
        benchmark = params.get("industry_deviation_benchmark") or {}
        groups: dict[str, list[int]] = {}
        for position, symbol in enumerate(symbols):
            groups.setdefault(industry_map.get(symbol) or _NO_INDUSTRY_PROJECTION, []).append(position)
        for industry in sorted(groups):
            if industry in exempt or industry not in deviation_map:
                continue
            rows = groups[industry]
            a = np.zeros(count)
            a[rows] = 1.0
            bench = float(benchmark.get(industry, 0.0))
            deviation = float(deviation_map[industry])
            constraints.append(("halfspace", a, bench + deviation))
            constraints.append(("halfspace", -a, deviation - bench))
    style_bounds = params.get("style_exposure_bounds")
    if style_bounds:
        matrix = params.get("style_exposure_matrix") or {}
        benchmark = params.get("style_exposure_benchmark") or {}
        for style in sorted(style_bounds):
            lower, upper = style_bounds[style]
            column = []
            for symbol in symbols:
                number = matrix.get(style, {}).get(symbol)
                if number is None:
                    raise ValueError("style exposure constraint requires an exposure for every selected name and style")
                column.append(float(number))
            a = np.array(column)
            bench = float(benchmark[style])
            constraints.append(("halfspace", a, float(upper) + bench))
            constraints.append(("halfspace", -a, bench - float(lower)))
    if params.get("beta_constraint_min") is not None or params.get("beta_constraint_max") is not None:
        betas = params.get("beta_by_symbol") or {}
        column = []
        for symbol in symbols:
            if symbol not in betas:
                raise ValueError("beta constraint requires a beta for every selected name")
            column.append(float(betas[symbol]))
        a = np.array(column)
        if params.get("beta_constraint_max") is not None:
            constraints.append(("halfspace", a, float(params["beta_constraint_max"])))
        if params.get("beta_constraint_min") is not None:
            constraints.append(("halfspace", -a, -float(params["beta_constraint_min"])))
    factor_bounds = params.get("factor_exposure_bounds")
    if factor_bounds:
        matrix = params.get("factor_exposure_matrix") or {}
        for factor in sorted(factor_bounds):
            lower, upper = factor_bounds[factor]
            column = []
            for symbol in symbols:
                number = matrix.get(factor, {}).get(symbol)
                if number is None:
                    raise ValueError("factor exposure constraint requires an exposure for every selected name and factor")
                column.append(float(number))
            a = np.array(column)
            constraints.append(("halfspace", a, float(upper)))
            constraints.append(("halfspace", -a, -float(lower)))
    max_iter = int(params.get("projection_max_iter", 512))
    tolerance = float(params.get("projection_tolerance", 1e-9))
    corrections = [np.zeros(count) for _ in constraints]
    converged = False
    for _ in range(max_iter):
        previous = x.copy()
        for position, constraint in enumerate(constraints):
            y = x + corrections[position]
            if constraint[0] == "halfspace":
                _, a, b = constraint
                slack = _projection_halfspace_slack(y, a, b)
                if slack > 0.0:
                    y = y - (slack / float(a @ a)) * a
            elif constraint[0] == "upper":
                y = np.minimum(y, constraint[1])
            else:
                y = np.maximum(y, constraint[1])
            corrections[position] = x + corrections[position] - y
            x = y
        if float(np.max(np.abs(x - previous))) <= tolerance:
            converged = True
            break
    # The projection only guarantees ~`tolerance` accuracy, so the feasibility
    # check must tolerate the accumulated float noise of a convergent solve
    # (observed ~1.5e-9 on the budget boundary with tolerance=1e-9) while
    # staying far below any real violation.  The allowance is therefore capped
    # by a fixed ceiling: `projection_tolerance` is only a stopping-accuracy
    # (performance) knob, so widening it must never widen the feasibility
    # check -- an early-stopped solve that misses the feasible set has to fail
    # closed rather than silently return a point that breaks a box constraint.
    feasibility_tolerance = min(max(16.0 * tolerance, 1e-8), _PROJECTION_FEASIBILITY_CEILING)
    feasible = True
    for constraint in constraints:
        if constraint[0] == "halfspace":
            _, a, b = constraint
            if _projection_halfspace_slack(x, a, b) > feasibility_tolerance:
                feasible = False
                break
        elif constraint[0] == "upper":
            if float(np.max(x - constraint[1])) > feasibility_tolerance:
                feasible = False
                break
        else:
            if float(np.max(constraint[1] - x)) > feasibility_tolerance:
                feasible = False
                break
    if not feasible:
        if converged:
            raise ValueError("constrained weight projection did not reach a feasible point")
        raise ValueError(f"constrained weight projection did not converge within {max_iter} iterations")
    return x


def _apply_industry_cap(
    weights: pd.Series,
    cap: float,
    industry_map: Mapping[str, str],
    name_cap: float | None = None,
) -> pd.Series:
    """IndustryWeightCap (roadmap 6.3): cap each industry's weight sum.

    Applied after the single-name cap and before the floor (declared order).
    The loop mirrors ``_apply_weight_cap``: every round freezes the names of
    each over-cap industry (scaled down proportionally so the group sits
    exactly on the cap), then hands the freed excess to the remaining names.
    Because the redistribution can push another industry over its cap, the
    frozen groups are revisited round by round, each round freezing at least
    one group, which bounds the loop and keeps it deterministic.  Names the
    map does not cover share the reserved ``_UNKNOWN_INDUSTRY`` group and the
    cap applies to it like to any other industry.

    With ``name_cap`` set (the single-name cap applied just before), the freed
    excess is distributed proportional to each receiving name's remaining
    headroom under the single-name cap, so a receiving name can never be
    pushed back above it; without ``name_cap`` the distribution is
    proportional to the free weights, mirroring the single-name cap exactly.
    Whatever cannot be placed stays as uninvested cash -- the same residual
    convention as the single-name cap.
    """

    result = weights.astype(float).copy()
    labels = pd.Series(
        [industry_map.get(symbol) or _UNKNOWN_INDUSTRY for symbol in result.index], index=result.index
    )
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        free = ~frozen
        if not free.any():
            break
        sums = result[free].groupby(labels[free]).sum()
        over = [group for group in sums.index if sums[group] > cap]
        if not over:
            break
        excess = 0.0
        for group in over:  # groupby keys are sorted, so the order is deterministic
            members = free & (labels == group)
            frozen |= members
            group_sum = float(result[members].sum())
            excess += group_sum - cap
            result[members] = result[members] * (cap / group_sum)
        free = ~frozen
        if not free.any():
            break
        if name_cap is not None:
            headroom = (float(name_cap) - result[free]).clip(lower=0.0)
            budget = float(headroom.sum())
            if budget <= 0.0:
                break
            ratio = min(1.0, excess / budget)
            result[free] = result[free] + ratio * headroom
        else:
            room = float(result[free].sum())
            if room <= 0.0:
                break
            result[free] = result[free] + excess * result[free] / room
    return result


def _score_transform_weights(spec: CompositionSpec, scores: pd.Series) -> pd.Series:
    """ScoreTransformWeight (roadmap 7.4): transform scores, allocate
    proportionally.

    ``linear`` scales by the slope, ``truncated`` zeroes every score at or
    below the cut point (those names keep a zero weight and are not held)
    and ``sigmoid`` maps through the logistic with the cut as its midpoint.
    A non-positive transformed score cannot be held long-only, so linear and
    sigmoid fail closed on it; truncated fails closed only when *every*
    selected score is truncated away on that date.  The full curve
    parameters travel inside the canonical payload (roadmap 7.4 requires
    recording them).
    """

    kind = spec.score_transform_kind
    if kind == "linear":
        transformed = spec.score_transform_slope * scores
        if not bool((transformed > 0).all()):
            raise ValueError("score_transform weighting requires strictly positive transformed scores")
    elif kind == "truncated":
        transformed = (scores - spec.score_transform_cut).clip(lower=0.0)
        if not bool((transformed > 0).any()):
            raise ValueError("score_transform truncated weighting requires at least one score above the cut")
    else:  # sigmoid; CompositionSpec validation leaves no other kind
        with np.errstate(over="ignore"):
            transformed = 1.0 / (1.0 + np.exp(-spec.score_transform_slope * (scores - spec.score_transform_cut)))
        if not bool((transformed > 0).all()):
            raise ValueError("score_transform weighting requires strictly positive transformed scores")
    return transformed / transformed.sum()


def _apply_weight_floor(weights: pd.Series, floor: float) -> pd.Series:
    """SingleNameFloor (roadmap 6.2): raise held names below ``floor`` up to it.

    Applied *after* the cap: the floor never exceeds the cap (spec-level
    validation), so the result stays inside the cap.  Only held names
    (weight > 0) participate -- unselected names stay at exactly zero and are
    never pulled into the book.  The raise is funded proportionally by the
    excess above the floor of the names that can spare it; donors are scaled
    down towards (never below) the floor, so the floor is never violated and
    a cap is never re-breached.  When even the frozen floors do not fit the
    budget the specification is infeasible and fails closed instead of
    shrinking anyone below the floor.  A remaining under-investment is left
    as cash (the same residual convention as the cap), which
    ``weight_floor_allow_cash=False`` rejects at the call site.
    """

    result = weights.astype(float).copy()
    held = result > 0.0
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        below = held & ~frozen & (result < floor)
        if not below.any():
            break
        frozen |= below
        deficit = float((floor - result[below]).sum())
        result[below] = floor
        donors = held & ~frozen
        excess = float((result[donors] - floor).clip(lower=0.0).sum())
        if not donors.any() or excess <= 0.0:
            break
        ratio = min(1.0, deficit / excess)
        result[donors] = floor + (result[donors] - floor) * (1.0 - ratio)
        if deficit <= excess:
            break
    if float(result.sum()) > 1.0 + 1e-9:
        raise ValueError("weight_floor is infeasible: the floors of the held names exceed the budget")
    return result


def _apply_weight_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Cap single holdings and redistribute the excess proportionally.

    Redistribution can push another name above the cap, so capped names are
    frozen one round at a time and only the remaining names absorb the excess.
    Each round freezes at least one name, which bounds the loop by the number
    of holdings and keeps the result deterministic.  When
    ``len(weights) * cap < 1`` the book cannot be fully invested inside the cap:
    the residual is deliberately left as uninvested cash rather than forced
    into already-capped names.
    """

    if (weights <= cap).all():
        return weights
    result = weights.astype(float).copy()
    frozen = pd.Series(False, index=result.index)
    for _ in range(len(result)):
        over = ~frozen & (result > cap)
        if not over.any():
            break
        frozen |= over
        excess = float((result - cap)[over].sum())
        result[over] = cap
        free = ~frozen
        room = float(result[free].sum())
        if not free.any() or room <= 0:
            break
        result[free] = result[free] + excess * result[free] / room
    return result.clip(upper=cap)


def build_buffered_target_row(
    spec: CompositionSpec,
    score_row: pd.Series,
    held_names: frozenset[str],
    returns_history: pd.DataFrame | None = None,
) -> pd.Series:
    """HoldingBuffer (roadmap 5.5) and DiversitySelector (roadmap 5.6) for one date.

    Called from the backtest main loops (local and worker, mirrored verbatim)
    whenever either operator is enabled, because both need information the
    pure ``build_target_weights`` panel pass cannot see: the buffer consumes
    the names still held after the previous executed trade, and the diversity
    filter consumes the realised-return correlation labels.  With both
    operators disabled the function is not called and the panel pass above
    keeps its pre-change behaviour bit for bit.

    Conservative readings of the roadmap (documented in AGENTS.md):

    * the percentile thresholds compare against the *full* valid
      cross-section of the date, best name = 1.0;
    * a held name that still makes the natural top ``top_n`` is always kept;
      the retention threshold only rescues held names that dropped out of it;
    * names expelled by the diversity filter cannot be rescued by the buffer
      (diversity is the hard risk constraint), and every admitted name --
      entrants and rescued holdings alike -- passes through the same greedy
      correlation screen;
    * the roadmap's per-industry minimum/maximum counts are not implemented:
      the strategy pipeline has no industry label data on either side.
    """

    candidates = score_row.dropna()
    if candidates.empty:
        return pd.Series(0.0, index=score_row.index, dtype=float)
    ordered = (
        pd.DataFrame({"score": candidates})
        .rename_axis("symbol")
        .reset_index()
        .sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
    )
    names_ordered = ordered["symbol"].tolist()
    count = len(names_ordered)
    # Percentile from the top: best name 1.0, worst 1/count.
    percentile = {sym: (count - position + 1) / count for position, sym in enumerate(names_ordered, start=1)}

    if spec.diversity_max_correlation is not None and returns_history is not None and len(returns_history) >= 2:
        correlations = returns_history.corr(min_periods=2)
        eligible_order: list[str] = []
        for sym in names_ordered:
            passes = True
            for chosen in eligible_order:
                value = np.nan
                if sym in correlations.index and chosen in correlations.columns:
                    value = correlations.at[sym, chosen]
                if pd.notna(value) and float(value) > float(spec.diversity_max_correlation):
                    passes = False
                    break
            if passes:
                eligible_order.append(sym)
    else:
        eligible_order = names_ordered

    base = eligible_order[: spec.top_n]
    if spec.holding_buffer_entry is None:
        selection = list(base)
    else:
        entry_threshold = float(spec.holding_buffer_entry)
        retain_threshold = float(spec.holding_buffer_retain)
        max_extra = int(spec.holding_buffer_max_extra)
        base_set = set(base)
        kept = [sym for sym in base if sym in held_names]
        entrants = [sym for sym in base if sym not in held_names and percentile[sym] >= entry_threshold]
        open_seats = len(base) - len(kept) - len(entrants)
        selection = kept + entrants
        for sym in eligible_order:
            if sym not in held_names or sym in base_set or percentile[sym] < retain_threshold:
                continue
            if open_seats > 0:
                selection.append(sym)
                open_seats -= 1
            elif len(selection) < spec.top_n + max_extra:
                selection.append(sym)
            else:
                break

    if not selection:
        return pd.Series(0.0, index=score_row.index, dtype=float)
    selection_set = set(selection)
    selection_ordered = [sym for sym in names_ordered if sym in selection_set]
    row_weights = _scheme_weights(spec, candidates[selection_ordered], returns_history)
    if spec.weight_cap is not None:
        row_weights = _apply_weight_cap(row_weights, spec.weight_cap)
    if spec.industry_weight_cap is not None:
        row_weights = _apply_industry_cap(
            row_weights, spec.industry_weight_cap, spec.industry_map or {}, name_cap=spec.weight_cap
        )
    if spec.weight_floor is not None:
        row_weights = _apply_weight_floor(row_weights, spec.weight_floor)
        if not spec.weight_floor_allow_cash and float(row_weights.sum()) < 1.0 - 1e-9:
            raise ValueError("weight_floor forbids a cash residual but the book is not fully invested")
    projection_params = _projection_params(spec)
    if projection_params is not None:
        row_weights = pd.Series(
            _apply_constrained_projection(row_weights.to_numpy(dtype=float), list(row_weights.index), projection_params),
            index=row_weights.index,
        )
    if spec.cash_residual_max is not None and 1.0 - float(row_weights.sum()) > float(spec.cash_residual_max) + 1e-9:
        raise ValueError("cash residual exceeds cash_residual_max")
    result = pd.Series(0.0, index=score_row.index, dtype=float)
    result.loc[row_weights.index] = row_weights.to_numpy()
    return result


def build_target_weights(
    spec: CompositionSpec,
    score: pd.DataFrame,
    *,
    eligible: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build target weights (one row per rebalance date) from a score panel.

    Parameters
    ----------
    spec:
        Selection and weighting rule.
    score:
        Cross-sectional score panel; a higher score means a stronger
        intention to hold the name.  ``NaN`` excludes the name on that date;
        an infinite score is rejected instead of ranked.
    eligible:
        Optional boolean panel of the same orientation marking investable
        names (liquidity / industry / PIT screens).  Missing entries are
        treated as not eligible.
    returns:
        Realised daily-return panel (``index=dt`` / ``columns=symbol``).
        Only the ``inverse_volatility`` weighting consumes it (the volatility
        of each selected name over the realised returns up to and including
        the decision date); it is required whenever that scheme is selected
        and ignored otherwise.

    Returns
    -------
    ``pd.DataFrame`` with the same index and columns as ``score`` holding the
    target weights.  Unselected names are ``0.0`` so downstream turnover is
    computable without re-deriving the selection.
    """

    if score.empty:
        raise ValueError("score panel is empty")
    numeric_score = score.apply(pd.to_numeric, errors="coerce")
    if bool(np.isinf(numeric_score.to_numpy(dtype=float)).any()):
        raise ValueError("score panel must not contain infinite values")
    if spec.weighting == "inverse_volatility" and returns is None:
        raise ValueError("inverse_volatility weighting requires a returns panel")
    if eligible is not None:
        missing_dates = score.index.difference(eligible.index)
        if len(missing_dates):
            raise ValueError(f"eligible panel is missing score dates: {[str(date) for date in missing_dates[:3]]}")
        extra_columns = eligible.columns.difference(score.columns)
        if len(extra_columns):
            raise ValueError(f"eligible panel has unknown symbols: {list(extra_columns)[:3]}")

    weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for date, row in score.iterrows():
        candidates = row
        if eligible is not None:
            mask = eligible.loc[date].reindex(row.index)
            candidates = row.where(mask.fillna(False).astype(bool))
        candidates = candidates.dropna()
        if candidates.empty:
            continue
        # Deterministic ordering: best score first, ties broken by symbol so a
        # re-run on the same panel always selects the same names.
        ordered = (
            pd.DataFrame({"score": candidates})
            .rename_axis("symbol")
            .reset_index()
            .sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
        )
        selected = ordered.head(spec.top_n).set_index("symbol")["score"]
        returns_history = None if returns is None else returns.loc[returns.index <= date]
        row_weights = _scheme_weights(spec, selected, returns_history)
        if spec.weight_cap is not None:
            row_weights = _apply_weight_cap(row_weights, spec.weight_cap)
        if spec.industry_weight_cap is not None:
            row_weights = _apply_industry_cap(
                row_weights, spec.industry_weight_cap, spec.industry_map or {}, name_cap=spec.weight_cap
            )
        if spec.weight_floor is not None:
            row_weights = _apply_weight_floor(row_weights, spec.weight_floor)
            if not spec.weight_floor_allow_cash and float(row_weights.sum()) < 1.0 - 1e-9:
                raise ValueError("weight_floor forbids a cash residual but the book is not fully invested")
        projection_params = _projection_params(spec)
        if projection_params is not None:
            row_weights = pd.Series(
                _apply_constrained_projection(row_weights.to_numpy(dtype=float), list(row_weights.index), projection_params),
                index=row_weights.index,
            )
        if spec.cash_residual_max is not None and 1.0 - float(row_weights.sum()) > float(spec.cash_residual_max) + 1e-9:
            raise ValueError("cash residual exceeds cash_residual_max")
        weights.loc[date, selected.index] = row_weights.to_numpy()
    return weights
