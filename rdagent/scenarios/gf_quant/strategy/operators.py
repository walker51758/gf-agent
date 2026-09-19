"""Unified registry of the implemented GF strategy-layer operators.

The strategy roadmap (``docs/gf_strategy_structured_operators.md``) names every
operator once, but its behaviour is spread over four places: the local spec,
the local executor, the standalone remote worker and the LLM prompt vocabulary.
This module is the single inventory that says, for each *implemented* operator,
where the local implementation and the worker implementation live and which
roadmap section it belongs to.

Scope is deliberately narrow: it is a catalogue plus query helpers.  It does
not dispatch execution, does not validate payloads and does not change any
runtime path -- future batches extend the entries (and only then) wire new
operators into the four touch points listed in the roadmap.

Anchors are ``(repo_relative_file, dotted_symbol)`` pairs.  An empty anchor
tuple means that side has no implementation of the operator; the ``notes``
field says why (some signal-side operators are worker-only, dedup is
local-only).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Repo-relative path of the standalone remote worker (single-file deliverable).
WORKER_PATH = "scripts/gf_quant_strategy_worker.py"

_STRATEGY_PACKAGE = "rdagent/scenarios/gf_quant/strategy"


@dataclass(frozen=True)
class StrategyOperatorSpec:
    """Catalogue entry for one implemented strategy operator.

    ``local_anchors`` / ``worker_anchors`` are ``(file, dotted_symbol)`` pairs;
    the dotted symbol may point at a module-level object, a class or a
    ``Class.method``.  An empty tuple means that side intentionally has no
    implementation -- never fill it with a guess.
    """

    name: str
    category: str
    doc_section: str
    local_anchors: tuple[tuple[str, str], ...] = ()
    worker_anchors: tuple[tuple[str, str], ...] = ()
    notes: str = ""


_OPERATOR_LIST: tuple[StrategyOperatorSpec, ...] = (
    # Signal-side transforms: the direction payload and the percentile ranks are
    # built when the worker consumes the request, so only the worker computes.
    StrategyOperatorSpec(
        name="DirectionMap",
        category="signal",
        doc_section="4.1",
        local_anchors=((f"{_STRATEGY_PACKAGE}/remote.py", "RemoteStrategyEvaluator.request_for"),),
        worker_anchors=((WORKER_PATH, "run_request"),),
        notes="Frozen directions travel in the request (factor_directions); the direction multiplication itself happens only in the worker.",
    ),
    StrategyOperatorSpec(
        name="CrossSectionalRank",
        category="signal",
        doc_section="4.2",
        worker_anchors=((WORKER_PATH, "run_request"),),
        notes="Cross-sectional percentile ranking is worker-only; the local composition layer consumes an already-blended score panel.",
    ),
    StrategyOperatorSpec(
        name="Winsorize",
        category="signal",
        doc_section="4.3",
        local_anchors=((f"{_STRATEGY_PACKAGE}/signal.py", "apply_winsorize"),),
        worker_anchors=((WORKER_PATH, "_apply_winsorize"),),
        notes="Quantile clip per date from that date's valid cross-section only; min_samples failures close on both sides.",
    ),
    StrategyOperatorSpec(
        name="CrossSectionalStandardize",
        category="signal",
        doc_section="4.4",
        local_anchors=((f"{_STRATEGY_PACKAGE}/signal.py", "apply_standardize"),),
        worker_anchors=((WORKER_PATH, "_apply_standardize"),),
        notes="z-score / rank-z / robust z-score; near-zero or non-finite dispersion fails closed on both sides.",
    ),
    StrategyOperatorSpec(
        name="FactorTransform",
        category="signal",
        doc_section="4.8",
        local_anchors=((f"{_STRATEGY_PACKAGE}/signal.py", "apply_transform"),),
        worker_anchors=((WORKER_PATH, "_apply_transform"),),
        notes="Monotone transforms only (linear/log/signed_log/sqrt); out-of-domain values fail closed, never silently repaired.",
    ),
    StrategyOperatorSpec(
        name="MissingValuePolicy",
        category="signal",
        doc_section="4.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/signal.py", "apply_missing_value"),
            (f"{_STRATEGY_PACKAGE}/signal.py", "apply_neutral_rank_fill"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_missing_value"),
            (WORKER_PATH, "_apply_neutral_rank_fill"),
            (WORKER_PATH, "_build_rank_panels"),
        ),
        notes="drop_stock removes symbols with any missing value; cross_section_median fills in value space after the A1 stages and before the direction map; neutral_rank fills missing percentile ranks with the neutral 0.5. missing_max_fill_ratio fails closed when a date would need more than that share filled. With the policy off the panels are untouched.",
    ),
    StrategyOperatorSpec(
        name="FactorBlend",
        category="signal",
        doc_section="4.9",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/signal.py", "blend_member_panels"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_score"),
            (WORKER_PATH, "_weighted_blend"),
        ),
        notes="Fixed frozen weights (blend_weights) normalised to sum one over the candidate factors plus every interaction term; the strict any-missing-kills contract of the default equal blend is preserved, and the default path stays numerically identical to the pre-batch equal blend.",
    ),
    StrategyOperatorSpec(
        name="FactorGroupBlend",
        category="signal",
        doc_section="4.10",
        local_anchors=((f"{_STRATEGY_PACKAGE}/signal.py", "blend_member_panels"),),
        worker_anchors=((WORKER_PATH, "_score"),),
        notes="Two-level blend: within-group (blend_within_weights, default equal) then across the group scores plus every interaction term as a singleton member (blend_group_weights, default equal). Mutually exclusive with the flat blend_weights; the groups must partition the candidate factors exactly; a group score is missing whenever any of its factors is missing, so the any-missing-kills contract is preserved.",
    ),
    StrategyOperatorSpec(
        name="FactorInteraction",
        category="signal",
        doc_section="4.11",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/signal.py", "interaction_rank_panels"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_interaction_rank_panels"),
            (WORKER_PATH, "_validate_signal"),
        ),
        notes="One extra blend member per term: product, elementwise min or geometric mean of the two factors' final post-direction percentile ranks (the conservative rank-first reading of the roadmap parameter), re-ranked to a percentile. A NaN in either input keeps the cell missing; at most 8 terms; the gated-interaction kind of the roadmap is not implemented.",
    ),
    StrategyOperatorSpec(
        name="IndustryNeutralize",
        category="signal",
        doc_section="4.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/signal.py", "apply_industry_neutralize"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_industry_neutralize"),
            (WORKER_PATH, "_validate_signal"),
        ),
        notes="Value-space stage between the eligible mask and the direction map. demean: within-industry de-mean (group means over valid cells, missing cells stay missing); regression: Frisch-Waugh residual of the industry-demeaned values on the industry-demeaned injected exposures (identical to the de-mean without exposures). The industry map is an explicit configuration injection. Declared fallback: an industry with fewer than min_group valid values on a date passes through unchanged; unmapped symbols share the reserved __no_industry__ group. The roadmap's weighting mode is deliberately absent (no engine-meaningful weight source).",
    ),
    StrategyOperatorSpec(
        name="StyleNeutralize",
        category="signal",
        doc_section="4.7",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/signal.py", "apply_style_neutralize"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_style_neutralize"),
            (WORKER_PATH, "_validate_signal"),
        ),
        notes="Value-space stage after the industry stage (when both are enabled): per-date least-squares residual on the injected per-symbol style exposures, de-meaned over the regression sample (zero-mean residual by construction). Rows missing any exposure keep their value; a date with fewer than max(min_samples, n_styles + 1) joint-valid rows passes through unchanged. The roadmap's estimation window and weighting are deliberately absent: exposures are point-in-time injections without history, so no estimation window exists.",
    ),
    StrategyOperatorSpec(
        name="RegimeConditionalSignal",
        category="signal",
        doc_section="4.12",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/signal.py", "blend_regime_panels"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_blend_regime_panels"),
            (WORKER_PATH, "_score"),
        ),
        notes="Replaces the static blend: per decision date the injected PIT state ({date: label}, read as-of the date after the regime_lag row shift) selects one factor group and the score is the equal-weight blend of that group's rank panels with the strict any-missing-kills contract. Mutually exclusive with blend_weights/blend_groups/interaction_pairs; a date without a resolvable state and a label missing from regime_factor_groups both fail closed. State thresholds and derivation belong to the data layer (only discrete labels are consumed).",
    ),
    # Selection.
    StrategyOperatorSpec(
        name="TopKSelector",
        category="selection",
        doc_section="5.3",
        local_anchors=((f"{_STRATEGY_PACKAGE}/composition.py", "build_target_weights"),),
        worker_anchors=((WORKER_PATH, "_target_weights"),),
        notes="Both sides keep the head(top_n) selection over the deterministic ordering.",
    ),
    StrategyOperatorSpec(
        name="RankTieBreaker",
        category="selection",
        doc_section="5.4",
        local_anchors=((f"{_STRATEGY_PACKAGE}/composition.py", "build_target_weights"),),
        worker_anchors=((WORKER_PATH, "_target_weights"),),
        notes="Both sides break score ties with a stable mergesort on symbol, ascending.",
    ),
    StrategyOperatorSpec(
        name="QuantileGate",
        category="selection",
        doc_section="5.1",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/gating.py", "apply_gating"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "GateLayer"),
        ),
        worker_anchors=((WORKER_PATH, "_apply_gating"),),
        notes="Single-layer cascade on the post-direction percentile rank of one candidate factor; the inclusive comparison on the average rank is the tie rule.",
    ),
    StrategyOperatorSpec(
        name="CascadedFactorGating",
        category="selection",
        doc_section="5.2",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/gating.py", "apply_gating"),
            (f"{_STRATEGY_PACKAGE}/gating.py", "build_gated_factor_score"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "GatingSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_gating"),
            (WORKER_PATH, "_validate_gating"),
        ),
        notes="Layers combine with and-semantics only (or-semantics is not implemented); failing symbols become NaN in the score and the final Top-K stays with CompositionSpec.top_n. The ablation contribution paths in the worker stay ungated.",
    ),
    StrategyOperatorSpec(
        name="HoldingBuffer",
        category="selection",
        doc_section="5.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "build_buffered_target_row"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_build_buffered_target_row"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="A held name inside the natural top_n is always kept; the retention threshold only rescues held names that dropped out of it, funded first by the seats vacated by entrants below the entry threshold and then by up to holding_buffer_max_extra extra seats beyond top_n. Needs the names still held after the previous executed trade, so it runs inside the backtest main loops; percentiles run from the top over the date's valid cross-section.",
    ),
    StrategyOperatorSpec(
        name="DiversitySelector",
        category="selection",
        doc_section="5.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "build_buffered_target_row"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_build_buffered_target_row"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Conservative subset: greedy max-pairwise-correlation screen over the deterministic score order, correlations from the realised-return history up to the decision date (undefined pairs pass). The roadmap's per-industry minimum/maximum counts need industry labels the strategy pipeline does not carry on either side and are deliberately absent.",
    ),
    # Constraints.
    StrategyOperatorSpec(
        name="SingleNameCap",
        category="constraint",
        doc_section="6.1",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_weight_cap"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "build_target_weights"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_cap"),
            (WORKER_PATH, "_target_weights"),
        ),
        notes="Deterministic water-filling; residual stays as cash when len(weights) * cap < 1.",
    ),
    StrategyOperatorSpec(
        name="SingleNameFloor",
        category="constraint",
        doc_section="6.2",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_weight_floor"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "build_target_weights"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_weight_floor"),
            (WORKER_PATH, "_target_weights"),
        ),
        notes="Applied after the cap, on target weights only (never drifted start-of-day weights). Held names below the floor are raised and funded by the proportional excess of the donors; infeasible floors fail closed, residual cash follows the cap convention. The roadmap's minimum-trade-amount parameter has no weight-layer counterpart and is deliberately absent.",
    ),
    StrategyOperatorSpec(
        name="IndustryWeightCap",
        category="constraint",
        doc_section="6.3",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_industry_cap"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "build_target_weights"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_industry_cap"),
            (WORKER_PATH, "_target_weights"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Runs after the single-name cap and before the floor (declared order): each round freezes the names of every over-cap industry, scales the group back to the cap and hands the freed excess to the remaining names (proportional to their single-name-cap headroom when a name cap is active). The symbol->industry mapping is an explicit configuration input (industry_map) because neither engine carries industry data; unmapped names share one implicit group that the cap also binds.",
    ),
    StrategyOperatorSpec(
        name="CashResidualPolicy",
        category="constraint",
        doc_section="6.8",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "build_target_weights"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_target_weights"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Explicit bound on the existing retain-as-cash convention: cash_residual_max fails a target row closed when its residual cash exceeds the bound; disabled keeps the pre-change path bit for bit. Redistribution of the residual beyond the caps is mathematically unavailable in this engine, so the roadmap's redistribute/order parameters are deliberately absent.",
    ),
    StrategyOperatorSpec(
        name="IndustryDeviationConstraint",
        category="constraint",
        doc_section="6.4",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_constrained_projection"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_constrained_projection"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Per-industry max active deviation |portfolio - benchmark| inside the joint exposure projection stage (runs after the floor, before the cash-residual check). Benchmark industry weights are a frozen PIT injection; an industry absent from the benchmark counts as benchmark weight 0 (declared convention). Exemptions are an explicit industry list (a boolean has no engine meaning); exempted industries must appear in industry_deviation_max. Unmapped symbols share the reserved __no_industry__ group.",
    ),
    StrategyOperatorSpec(
        name="StyleExposureConstraint",
        category="constraint",
        doc_section="6.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_constrained_projection"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_constrained_projection"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Active style exposure (portfolio minus injected benchmark exposure) boxed per style inside the joint projection stage. The style matrix and benchmark are frozen risk-model injections and must cover the bounded styles (missing benchmark styles fail closed -- the matrix and benchmark come from the same risk model, so a gap is a configuration error). The roadmap's constraint priority and soft-warning knobs are deliberately absent: every constraint is hard.",
    ),
    StrategyOperatorSpec(
        name="BetaConstraint",
        category="constraint",
        doc_section="6.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_constrained_projection"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_constrained_projection"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Portfolio beta boxed into an absolute interval (the conservative reading of the roadmap's 'absolute or active' parameter). Per-symbol PIT betas are an explicit injection covering every selected name. The declared downgrade policy for the collinearity/infeasibility risk: the joint projection fails closed instead of silently degrading.",
    ),
    StrategyOperatorSpec(
        name="FactorExposureConstraint",
        category="constraint",
        doc_section="6.7",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_constrained_projection"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_constrained_projection"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Absolute per-factor exposure intervals over the injected risk-factor exposure matrix (no benchmark: the roadmap states absolute target ranges). Same joint projection stage and hard-constraint discipline as the style family; the soft-penalty coefficient is deliberately absent.",
    ),
    StrategyOperatorSpec(
        name="ConstrainedWeightProjection",
        category="constraint",
        doc_section="6.9",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_apply_constrained_projection"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "_projection_params"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_constrained_projection"),
            (WORKER_PATH, "_projection_params_from_payload"),
        ),
        notes="The joint projector for all enabled linear constraint families (single-name cap and held-name floor participate as box constraints, non-negativity and sum<=1 always). Locked: equal-weight L2 distance in weight space, Dykstra alternating projection over a fixed constraint order (deterministic, dependency-free, solver-version-stable), no constraint priorities (all hard), infeasible/unconverged fail closed. projection_max_iter/projection_tolerance are the only knobs and travel in the payload only while a family is enabled; projection_tolerance sets the stopping accuracy alone, while the feasibility check uses a fixed float-noise ceiling, so widening the tolerance can only turn a feasible run into a conservative failure; no code ever enters the payload, so the roadmap's ban on LLM-supplied objectives holds by construction.",
    ),
    # Weighting schemes.
    StrategyOperatorSpec(
        name="EqualWeight",
        category="weight",
        doc_section="7.1",
        local_anchors=((f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),),
        worker_anchors=((WORKER_PATH, "_target_weights"),),
    ),
    StrategyOperatorSpec(
        name="RankWeight",
        category="weight",
        doc_section="7.2",
        local_anchors=((f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),),
        worker_anchors=((WORKER_PATH, "_target_weights"),),
        notes="Linear decreasing weights by rank, best score first.",
    ),
    StrategyOperatorSpec(
        name="ScoreProportionalWeight",
        category="weight",
        doc_section="7.3",
        local_anchors=((f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),),
        worker_anchors=((WORKER_PATH, "_target_weights"),),
        notes="Non-positive scores must fail closed on both sides; no implicit shorting.",
    ),
    StrategyOperatorSpec(
        name="ScoreTransformWeight",
        category="weight",
        doc_section="7.4",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_score_transform_weights"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_score_transform_weights"),
            (WORKER_PATH, "_target_weights"),
        ),
        notes="linear/truncated/sigmoid transform of the selected scores allocated proportionally; the full curve parameters (kind, slope, cut) travel in the canonical payload. Non-positive transformed scores fail closed except that truncated keeps zero-weight names while at least one score survives.",
    ),
    StrategyOperatorSpec(
        name="InverseVolatilityWeight",
        category="weight",
        doc_section="7.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_inverse_volatility_weights"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_inverse_volatility_weights"),
            (WORKER_PATH, "_target_weights"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Weights proportional to the inverse realised-return volatility up to each decision date (window is an upper bound on the lookback). Needs the returns panel, so build_target_weights/_target_weights take an optional returns argument and the ablation path passes the same panel. A panel with fewer than two rows falls back to equal weighting (declared); a selected name with a single valid observation or zero variance fails closed with the same message on both sides; inverse_vol_min floors tiny estimates.",
    ),
    StrategyOperatorSpec(
        name="HierarchicalWeighting",
        category="weight",
        doc_section="7.8",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_hierarchical_weights"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_hierarchical_weights"),
            (WORKER_PATH, "_target_weights"),
        ),
        notes="Group weight first (equal, or hierarchical_group_weights normalised over the groups that hold selected names), equal split inside each group. The grouping is an explicit configuration input (hierarchical_groups); empty groups receive nothing and a selected name outside every group fails closed. Pure weighting scheme: no returns or loop state needed.",
    ),
    StrategyOperatorSpec(
        name="RiskParityWeight",
        category="weight",
        doc_section="7.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_risk_parity_weights"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_risk_parity_weights"),
            (WORKER_PATH, "_target_weights"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Risk contributions of the selected names equal (or matching risk_parity_budgets). Covariance from the realised daily returns up to the decision date (window is an upper bound), shrunk towards its own diagonal when risk_parity_shrinkage_kind='diagonal'; solved by a deterministic cyclical-coordinate-descent fixed-point (iteration budget fixed on both sides, no solver dependency). History under two rows falls back to equal weighting; a selected name without a positive variance and a still non-positive-definite covariance fail closed; the unconverged solve fails closed. Cap/industry-cap/floor/cash chain applies afterwards as for every scheme.",
    ),
    StrategyOperatorSpec(
        name="FactorRiskBudgetWeight",
        category="weight",
        doc_section="7.7",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/composition.py", "_factor_risk_budget_weights"),
            (f"{_STRATEGY_PACKAGE}/composition.py", "_scheme_weights"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "CompositionSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_factor_risk_budget_weights"),
            (WORKER_PATH, "_target_weights"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Factor risk budgets matched in factor space (CCD on the ridged injected factor covariance) and mapped onto the selected names through the injected exposure matrix as relu(B @ e). Declared simplifications: idiosyncratic risk stays out of the budgeting objective (bounded only by the frb_idio_max exclusion cap) and the non-negative projection replaces a constrained quadratic program. Exposures/factor covariance/idiosyncratic variances are explicit risk-model injections; missing per-name data, a non-positive-definite covariance and an all-non-positive projection fail closed.",
    ),
    # Execution.
    StrategyOperatorSpec(
        name="RebalanceSchedule",
        category="execution",
        doc_section="8.1",
        local_anchors=((f"{_STRATEGY_PACKAGE}/remote.py", "RemoteStrategyEvaluator"),),
        worker_anchors=((WORKER_PATH, "_rebalance_dates"),),
        notes="The calendar derivation is worker-only; locally only the frequency is declared on the request.",
    ),
    StrategyOperatorSpec(
        name="SignalLag",
        category="execution",
        doc_section="8.2",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="System floor of 1 trading day, enforced in BacktestSpec and re-validated by the worker.",
    ),
    StrategyOperatorSpec(
        name="DriftBandRebalance",
        category="execution",
        doc_section="8.3",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/backtest.py", "_drift_band_triggers"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_drift_band_triggers"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Band comparison deliberately reads drifted carried weights against the target -- the one explicit exception to the target-weight judgement rule (caps and floors still judge target weights). Inside the band a rebalance is skipped entirely and the next target only arrives at the next rebalance date; drift_band_forced_days counts trading days since the last executed rebalance.",
    ),
    StrategyOperatorSpec(
        name="TurnoverShrinkage",
        category="execution",
        doc_section="8.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/backtest.py", "_shrink_turnover"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_shrink_turnover"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Soft shrinkage scales the whole weight delta, so truncated zero-weight exits participate like any other leg; the hard clip caps half-L1 turnover including cash, leaving the immutable cost formulas untouched. Runs after the drift-band gate on a trade day.",
    ),
    StrategyOperatorSpec(
        name="MinimumHoldingPeriod",
        category="execution",
        doc_section="8.4",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/backtest.py", "_apply_min_holding"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_min_holding"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="A name entered by a trade cannot be sold for min_holding_days completed trading days: its executed weight is raised back to the carried weight and the remaining names are scaled down proportionally so the book never exceeds its executed total. The clamp runs after the drift-band gate and the turnover shrinkage, so the lock is the last execution veto; costs simply observe the smaller sell delta. The roadmap's forced-exit exception parameter has no defined trigger in the engine and is deliberately absent.",
    ),
    # Order layer (batch 6): a separate execution-intent stage that consumes the
    # injected market panels.  All four fields default closed, so a candidate
    # that does not use them keeps its payload -- and its identity hash --
    # unchanged.
    StrategyOperatorSpec(
        name="ParticipationCap",
        category="execution",
        doc_section="8.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_participation_cap"),
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_execution_layer"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_participation_cap"),
            (WORKER_PATH, "_apply_execution_layer"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_order_layer"),
        ),
        notes="One day's traded amount is capped at participation_max_rate times the trailing mean amount of the injected amount panel; the unfilled remainder becomes an explicit pending plan (unfilled=carry, the default, bounded by participation_max_days and cancelled on expiry) rather than a silent fill later. Needs execution_data.trading_dates and is mutually exclusive with StagedExecution.",
    ),
    StrategyOperatorSpec(
        name="LotRounding",
        category="execution",
        doc_section="8.7",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_lot_rounding"),
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_execution_layer"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_lot_rounding"),
            (WORKER_PATH, "_apply_execution_layer"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_order_layer"),
        ),
        notes="The day's execution slice is snapped onto the injected lot grid at the raw execution-day price (capital base pinned at 1.0, never the drifted net asset value). The default down mode makes 'rounded notional never exceeds the intended one' an invariant and keeps the cash residual non-negative; lot_min_notional is judged on the rounded notional, so a sub-minimum order is not placed at all. Rounding runs after the allocator -- a staged plan therefore cannot split a whole lot back onto a fractional grid -- and the shaved part is cancelled, not re-queued.",
    ),
    StrategyOperatorSpec(
        name="SideAwareTradability",
        category="execution",
        doc_section="8.8",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_side_aware_tradability"),
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_execution_layer"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_apply_side_aware_tradability"),
            (WORKER_PATH, "_apply_execution_layer"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_order_layer"),
        ),
        notes="An asymmetric mask (block_buy / block_sell / block_both) over the injected halt and price-limit panels, deliberately not compressed into one symmetric boolean. It clamps only the day's execution slice and never the carried book, so 'may hold but may not buy' can never degrade into a forced sale. A blocked leg is cancelled, not queued.",
    ),
    StrategyOperatorSpec(
        name="StagedExecution",
        category="execution",
        doc_section="8.9",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "stage_orders"),
            (f"{_STRATEGY_PACKAGE}/execution.py", "plan_order"),
            (f"{_STRATEGY_PACKAGE}/execution.py", "apply_execution_layer"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_stage_orders"),
            (WORKER_PATH, "_plan_order"),
            (WORKER_PATH, "_apply_execution_layer"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_order_layer"),
        ),
        notes="A deterministic multi-day order plan: the target delta is spread over staged_max_days trading days in a resolved service order (staged_priority score/weight by descending adjustment size, fixed when the plan is created and reused while it is alive), and the plan is carried inside the backtest loop rather than folded into the final returns. Needs execution_data.trading_dates, is mutually exclusive with ParticipationCap, and stacks on TurnoverShrinkage -- the combination trades less than either alone, which is intended. Budget-accounting note: the plan's per-day budget is the remaining half-L1 adjustment divided by the remaining days; only the order in which the individual orders drain that budget changed (each order now fills its own slice greedily).",
    ),
    StrategyOperatorSpec(
        name="FixedExecutionCost",
        category="execution",
        doc_section="8.10",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/costs.py", "BUY_COST_BPS"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_request"),
        ),
        notes="Immutable 10/10 bps buy/sell contract; no agent-selectable cost field anywhere.",
    ),
    # Risk controls (batch 6).  Unlike the audit group these do move weights:
    # both scalers multiply the day's whole target row, and the factor monitor
    # scales a factor's contribution to the score.  All of them read realised
    # data only -- never the day they steer.
    StrategyOperatorSpec(
        name="PortfolioVolatilityTarget",
        category="risk",
        doc_section="9.1",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "volatility_scale"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_volatility_scale"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_order_layer"),
        ),
        notes="Scales the whole target row by target annual volatility over the realised annualised volatility of the daily net-return path, clamped to [min_scale, max_scale]; the net series is advanced with the engine's own immutable cost formula, so the estimate is the net-of-cost one the metrics report. Fewer than two observations passes through declared-unchanged -- the same declared degradation inverse-volatility weighting uses for a short panel -- while a portfolio-level zero-volatility estimate takes max_scale instead of failing closed: that is a sample property, not the single-name data defect the weighting scheme rejects.",
    ),
    StrategyOperatorSpec(
        name="DrawdownStateMachine",
        category="risk",
        doc_section="9.2",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "drawdown_state_machine"),
            (f"{_STRATEGY_PACKAGE}/backtest.py", "run_portfolio_backtest"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BacktestSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_drawdown_state_machine"),
            (WORKER_PATH, "_backtest"),
            (WORKER_PATH, "_validate_order_layer"),
        ),
        notes="Frozen three-state machine (warning / reduce / halt) over the realised NAV path: strictly increasing thresholds, consecutive-day confirmation for both entry and recovery, and recovery measured as the drawdown narrowing from the trough of the current segment. It scales the target row, and the two risk scalers compose as min(dd_scale, vol_scale) so the circuit breaker always outranks the tactical volatility scaler. Declared consequence: a halt_scale of 0.0 freezes the book at its current holdings, and a frozen book has no further return variation -- the recovery gate then has nothing to read and the machine stays in its highest state.",
    ),
    StrategyOperatorSpec(
        name="FactorDecayMonitor",
        category="risk",
        doc_section="9.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/execution.py", "factor_decay_scales"),
            (f"{_STRATEGY_PACKAGE}/execution.py", "factor_decay_state"),
            (f"{_STRATEGY_PACKAGE}/signal.py", "build_factor_score"),
            (f"{_STRATEGY_PACKAGE}/candidate.py", "SignalSpec"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_factor_decay_scales"),
            (WORKER_PATH, "_factor_decay_state"),
            (WORKER_PATH, "_score"),
        ),
        notes="Trailing rank IC of each factor over the holding window its decision actually earned; once the run of consecutive failing periods reaches decay_fail_periods, that factor's rank panel is scaled by decay_scale. The factor is never replaced -- swapping in another factor chosen by the failing signal would reuse the same information twice -- and an IC window that is not complete by the decision day is not read. A period counts as failing only when its IC is strictly below the threshold, and the ablation path (two-argument score call) does not consume the configuration.",
    ),
    # Audit: dedup runs only in the local acceptance flow, never in the worker.
    StrategyOperatorSpec(
        name="ReturnProfileDedup",
        category="audit",
        doc_section="10.5",
        local_anchors=((f"{_STRATEGY_PACKAGE}/service.py", "StrategyAcceptanceService._dedup"),),
        notes="Return-profile dedup is local-only; the standalone worker performs no deduplication.",
    ),
    # Result-report audits (batch D): read-only consumers of the executed
    # portfolio; the configuration rides at the request level, outside the
    # candidate payload, so enabling one never changes a strategy identity hash.
    StrategyOperatorSpec(
        name="StressScenarioCheck",
        category="audit",
        doc_section="9.3",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/audit.py", "stress_scenario_report"),
            (f"{_STRATEGY_PACKAGE}/audit.py", "ensure_stress_passed"),
            (f"{_STRATEGY_PACKAGE}/audit.py", "validate_audit_config"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_stress_scenario_report"),
            (WORKER_PATH, "_ensure_stress_passed"),
            (WORKER_PATH, "_validate_audit"),
        ),
        notes="Replays the executed start-of-day weights through an injected, versioned scenario table and reports the loss plus a pass flag; warn mode writes report fields only, while reject mode fails the whole audit closed without rewriting a single weight. The scenario table is an audit-level data input (request-level audit object, never the candidate payload) and the report echoes scenario_table_version so swapping the table cannot pass unnoticed.",
    ),
    StrategyOperatorSpec(
        name="ConcentrationMonitor",
        category="audit",
        doc_section="9.4",
        local_anchors=((f"{_STRATEGY_PACKAGE}/audit.py", "concentration_report"),),
        worker_anchors=((WORKER_PATH, "_concentration_report"),),
        notes="Per-day HHI, effective number of holdings and top-k share of the executed start-of-day weights plus an optional single-industry threshold alert (industry labels are an explicit audit config input, like the composition layer's industry_map). Report fields only -- the weights are never rewritten.",
    ),
    StrategyOperatorSpec(
        name="CostSensitivityAudit",
        category="audit",
        doc_section="10.2",
        local_anchors=((f"{_STRATEGY_PACKAGE}/audit.py", "cost_sensitivity_report"),),
        worker_anchors=((WORKER_PATH, "_cost_sensitivity_report"),),
        notes="Re-prices the already-executed buy/sell legs under audit-only cost scenarios (default 10/15/20 bps) from the gross and turnover series the backtest already produced; no strategy computation is re-run. Scenario rates never enter the candidate payload -- the worker's cost-key rejection there is untouched -- and the production 10/10 contract stays the only tradable assumption.",
    ),
    StrategyOperatorSpec(
        name="ParameterPerturbationAudit",
        category="audit",
        doc_section="10.3",
        local_anchors=((f"{_STRATEGY_PACKAGE}/audit.py", "parameter_perturbation_report"),),
        worker_anchors=((WORKER_PATH, "_parameter_perturbation_report"),),
        notes="Pre-registered dotted keys into the canonical payload (composition/backtest/signal/gating only) re-run the full pipeline per variant; a variant that fails validation is recorded as an error variant, and a key absent from the payload (lazy fields such as inverse_vol_window under another weighting) fails closed instead of reporting a fake insensitivity.",
    ),
    StrategyOperatorSpec(
        name="AblationAudit",
        category="audit",
        doc_section="10.4",
        local_anchors=((f"{_STRATEGY_PACKAGE}/audit.py", "ablation_report"),),
        worker_anchors=((WORKER_PATH, "_ablation_report"),),
        notes="Formalises the existing ungated two-argument factor-ablation loop into report fields (per-factor ablated metrics plus the four marginal deltas, exact formulas of factor_contributions); factor-level only -- operator-level removal has no engine disable switches and is not fabricated.",
    ),
    StrategyOperatorSpec(
        name="FeasibilityAudit",
        category="audit",
        doc_section="10.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/audit.py", "feasibility_report"),
            (f"{_STRATEGY_PACKAGE}/audit.py", "feasibility_days"),
            (f"{_STRATEGY_PACKAGE}/audit.py", "attribute_execution_shortfall"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_feasibility_report"),
            (WORKER_PATH, "_feasibility_days"),
            (WORKER_PATH, "_attribute_execution_shortfall"),
        ),
        notes="Reports the theoretical-versus-actual traded amount per rebalance day (executable ratio, unfilled amount, cash residual) and splits the shortfall across the execution chain's own stages: participation cap / staged plan, side-aware tradability, lot rounding. The stage books come from an observation sink the order layer fills only while this section is requested, so the executed path is bit-identical with and without it. Report-only by construction: allow_cash_residual records a violation and changes nothing -- it is deliberately not CompositionSpec.cash_residual_max, which bounds the target-row residual at the composition layer and fails the run closed.",
    ),
    StrategyOperatorSpec(
        name="DataLeakageAudit",
        category="audit",
        doc_section="10.7",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/audit.py", "data_leakage_report"),
            (f"{_STRATEGY_PACKAGE}/audit.py", "leakage_declarations"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_data_leakage_report"),
            (WORKER_PATH, "_leakage_declarations"),
        ),
        notes="Collects the point-in-time declarations the candidate payload actually makes (index-membership lag, listing age, signal lag, regime lag) and compares them against injected field-availability lags and announcement-to-effective rules, emitting a severity-tagged look-ahead risk list with each finding's payload location. A field with no declaration is reported as unknown, never silently passed and never fail-closed: this is a reporter. Provenance reuses the request's snapshot_id/snapshot_checksum rather than inventing a version number.",
    ),
    StrategyOperatorSpec(
        name="WalkForwardEvaluation",
        category="audit",
        doc_section="10.1",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/walk_forward.py", "walk_forward_report"),
            (f"{_STRATEGY_PACKAGE}/walk_forward.py", "validate_walk_forward_config"),
            (f"{_STRATEGY_PACKAGE}/walk_forward.py", "_build_rolling_splits"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_walk_forward_report"),
            (WORKER_PATH, "_validate_walk_forward"),
            (WORKER_PATH, "_build_rolling_splits"),
        ),
        notes="The one audit that re-runs the engine: the frozen candidate is described segment by segment over rolling train/valid/test windows, so its configuration rides at the request level next to ``evaluation`` rather than inside the report-only ``audit`` object. Windows come from the shared tested splitter (mirrored into the worker, which imports no project package); the report is explicitly ex-post -- fit_on_train and parameter_refit are false, because a frozen schema has nothing to fit and reading the OOS segments to re-tune is the risk the roadmap names. Segment count is bounded by max_segments (each segment is a full engine run) and a shortfall against min_oos_segments fails closed. Nothing here touches the main run: the artifact, metrics and return profile still come from the single ``evaluation`` window.",
    ),
    # Governance (batch 9): manifests, reproducibility and the deployment gate.
    # None of the three touches portfolio behaviour and none enters the
    # candidate payload, so enabling them cannot move a strategy identity hash.
    StrategyOperatorSpec(
        name="StrategyManifestEmitter",
        category="audit",
        doc_section="10.8",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/manifest.py", "build_strategy_manifest"),
            (f"{_STRATEGY_PACKAGE}/manifest.py", "validate_manifest_config"),
            (f"{_STRATEGY_PACKAGE}/manifest.py", "write_strategy_manifest"),
        ),
        worker_anchors=(
            (WORKER_PATH, "_build_manifest_section"),
            (WORKER_PATH, "_validate_manifest"),
            (WORKER_PATH, "_worker_file_digest"),
        ),
        notes="Emits the immutable identity record for an accepted strategy. The request-level ``manifest`` object carries no settings at all -- MANIFEST_KEYS is empty and any key fails closed on both ends -- because every roadmap parameter (schema version, operator version, parameter checksum, factor ids, cost facts, data window) is a fact to be collected rather than a knob. Hard identity is the worker file digest plus the runtime stack; the schema and semantics versions are labels, declared in MANIFEST_LABEL_FIELDS and skipped by the equality check. The worker attests what it observed: its own file hash, its imported module versions, the calendar it actually read and the request checksum. Immutability is taken literally: the file is named by strategy id and re-writing it with different content is refused rather than overwriting an earlier record.",
    ),
    StrategyOperatorSpec(
        name="ReproducibilityCheck",
        category="audit",
        doc_section="10.9",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/reproducibility.py", "reproducibility_report"),
            (f"{_STRATEGY_PACKAGE}/service.py", "StrategyAcceptanceService.process"),
        ),
        notes="Local-only (like ReturnProfileDedup): reproducibility is a property of the record on hand, not of a run, so there is no request surface and nothing for the worker to mirror. It compares identities rather than re-running the pipeline -- the roadmap's float-tolerance parameter is therefore not accepted, because there is no numeric comparison to loosen -- over REPRODUCIBILITY_IDENTITY_FIELDS. request_checksum and the observed trading_calendar are excluded: the former covers which audit sections this particular run asked for, so folding it in would report a spurious irreproducibility for the same strategy under a different audit configuration; the latter is an observation already covered by the artifact set's window. A field missing on either side fails closed with field_unknown:<field>, and an absent worker attestation is worker_attestation_missing.",
    ),
    StrategyOperatorSpec(
        name="DeploymentGate",
        category="audit",
        doc_section="10.10",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/deployment.py", "deployment_verdict"),
            (f"{_STRATEGY_PACKAGE}/deployment.py", "DeploymentPolicy"),
            (f"{_STRATEGY_PACKAGE}/service.py", "StrategyAcceptanceService.process"),
        ),
        notes="Local-only: it consumes reports already produced (acceptance gate, walk_forward, feasibility, cost_sensitivity, data_leakage, dedup and the reproducibility verdict) and returns allow_deploy / research_only / reject. The roadmap states no numeric threshold, so none is invented: DeploymentPolicy is threshold-free and the report hard-codes thresholds_defined false with a note saying why. reject needs a hard defect only -- a failed acceptance gate, a data_leakage finding at the existing LEAKAGE_BREACH_SEVERITY, or a recorded feasibility violation. Missing evidence or a duplicate strategy is research_only, never reject. Returns are not an input at all, which is how the roadmap's stated risk (mistaking a research-window high return for a deployment permit) is closed.",
    ),
    # Candidate pool / tradability (batch E): one shared boolean eligibility
    # panel consumed by the signal front-end before ranking.  Every data
    # input is an explicit payload panel (the industry_map injection
    # pattern); none of it enters the LLM prompt vocabulary.
    StrategyOperatorSpec(
        name="BaseUniverse",
        category="universe",
        doc_section="3.1",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "BaseUniverseRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Static explicit member list injected as configuration (the engine has no securities master table); point-in-time membership is the IndexMembershipFilter's business. A symbol absent from the list never enters the candidate pool.",
    ),
    StrategyOperatorSpec(
        name="IndexMembershipFilter",
        category="universe",
        doc_section="3.2",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "IndexMembershipRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Sparse point-in-time membership panel with as-of lookup on the decision date, an optional effective-lag shift in panel rows, and an optional exclusion window after a False->True inclusion transition (a name that is a member at the panel start is pre-existing and never new). Index codes stay with the data layer -- the panel is the union the upper layer chose.",
    ),
    StrategyOperatorSpec(
        name="ListingAgeFilter",
        category="universe",
        doc_section="3.3",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "ListingAgeRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Eligible once the count of trading-calendar days strictly after the injected listing date reaches min_listing_days (the listing day is day 0, so the boundary date is exactly the N-th trading day after it). A symbol without a listing date can never be age-verified and is excluded.",
    ),
    StrategyOperatorSpec(
        name="StatusExclusionFilter",
        category="universe",
        doc_section="3.4",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "StatusExclusionRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Sparse {date: {symbol: status}} panel recording only abnormal states, so the lookup is exact-date by construction -- no as-of backfill of a stale status, the survivorship risk the roadmap warns about. A symbol is excluded when its recorded status is in excluded_statuses.",
    ),
    StrategyOperatorSpec(
        name="PriceLimitTradabilityFilter",
        category="universe",
        doc_section="3.5",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "PriceLimitRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Conservative single-mask subset: a name at its limit price on the decision date (touching counts, 1e-9 tolerance) is excluded from that date's selection, and missing price or limit data for an excluded side excludes it too. The roadmap's buy/sell asymmetry (hold but do not buy) needs a two-sided execution mask the weight builders do not have and is not fabricated.",
    ),
    StrategyOperatorSpec(
        name="LiquidityFilter",
        category="universe",
        doc_section="3.6",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "LiquidityRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Trailing mean of the injected daily amount over the last window trading-calendar days (short history used as is) must clear min_amount and/or the date's cross-sectional quantile of the same rolling series, both inclusive; a name missing over the whole window is excluded unless missing=include.",
    ),
    StrategyOperatorSpec(
        name="PriceRangeFilter",
        category="universe",
        doc_section="3.7",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "PriceRangeRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Price inside [min_price, max_price] (inclusive, either bound omittable) on each of the last consecutive_days trading-calendar days; a missing price or a window extending before the calendar cannot confirm the band and excludes the name. Reads the injected raw-close panel, never the engine's adjusted close.",
    ),
    StrategyOperatorSpec(
        name="DataQualityFilter",
        category="universe",
        doc_section="3.8",
        local_anchors=(
            (f"{_STRATEGY_PACKAGE}/candidate.py", "DataQualityRule"),
            (f"{_STRATEGY_PACKAGE}/universe.py", "build_eligible_mask"),
        ),
        worker_anchors=((WORKER_PATH, "_build_universe_eligible"),),
        notes="Conservative subset on the candidate's own factor panels plus the optional raw-close panel: trailing-window non-missing cell coverage (inclusive bound), trailing missing-run length in any factor (a run of exactly the bound still passes) and the non-positive-price check (missing price excluded). The roadmap's generic outlier rules have no engine-independent definition and are deliberately absent.",
    ),
)

#: Catalogue keyed by operator name; mirrors ``coder.OPERATORS`` organisation.
OPERATORS: dict[str, StrategyOperatorSpec] = {spec.name: spec for spec in _OPERATOR_LIST}

#: Grouping by roadmap category, preserving catalogue order inside each group.
OPERATOR_GROUPS: dict[str, tuple[str, ...]] = {
    category: tuple(spec.name for spec in _OPERATOR_LIST if spec.category == category)
    for category in dict.fromkeys(spec.category for spec in _OPERATOR_LIST)
}


def implemented_operators() -> tuple[str, ...]:
    """Names of every registered implemented operator, in catalogue order."""

    return tuple(spec.name for spec in _OPERATOR_LIST)


def query(name: str) -> StrategyOperatorSpec:
    """Return the catalogue entry for ``name`` or raise a keyed LookupError."""

    try:
        return OPERATORS[name]
    except KeyError:
        raise KeyError(f"unknown strategy operator: {name!r}; known: {sorted(OPERATORS)}") from None
