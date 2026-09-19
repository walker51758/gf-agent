"""LLM strategy proposal engine for one frozen factor working set.

The model produces declarative JSON only. It may choose factors and portfolio
parameters, but it never writes executable backtest code; deterministic local
or remote framework code remains responsible for composition, lag and the
immutable system execution-cost contract.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from rdagent.scenarios.gf_quant.strategy.candidate import (
    DEFAULT_STANDARDIZE_MIN_STD,
    BacktestSpec,
    CompositionSpec,
    ExecutionDataSpec,
    GateLayer,
    GatingSpec,
    SignalSpec,
    StrategyCandidate,
    UniverseSpec,
)
from rdagent.scenarios.gf_quant.strategy.costs import BUY_COST_BPS, SELL_COST_BPS
from rdagent.scenarios.gf_quant.strategy.orchestrator import StrategyProposalContext


def _metric(metrics: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _safe_name(value: Any, context: StrategyProposalContext) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "strategy").strip()).strip("_").lower()
    base = base[:48] or "strategy"
    epoch_suffix = re.sub(r"[^A-Za-z0-9]+", "", context.epoch_id)[-8:] or "epoch"
    return f"{base}_{epoch_suffix}_t{context.trial_index:02d}"


class LLMStrategyProposalEngine:
    """Convert a real RD-Agent API response into a validated strategy candidate."""

    def __init__(
        self,
        api_backend: Any | None = None,
        *,
        universe: str = "csi_all",
        min_factors: int = 2,
        max_factors: int = 6,
        min_top_n: int = 20,
        max_top_n: int = 100,
    ) -> None:
        if api_backend is None:
            from rdagent.oai.llm_utils import APIBackend

            api_backend = APIBackend()
        if not 1 <= min_factors <= max_factors <= 8:
            raise ValueError("factor bounds must satisfy 1 <= min <= max <= 8")
        if not 1 <= min_top_n <= max_top_n:
            raise ValueError("top_n bounds are invalid")
        self.api_backend = api_backend
        self.universe = universe
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.min_top_n = min_top_n
        self.max_top_n = max_top_n

    @staticmethod
    def _factor_payload(context: StrategyProposalContext) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []
        for factor in context.working_set.factors:
            metrics = factor.metrics
            factors.append(
                {
                    "name": factor.name,
                    "factor_id": factor.factor_id,
                    "family": factor.family,
                    "eligibility_tier": factor.eligibility_tier,
                    "theme": factor.theme,
                    "intended_role": factor.intended_role,
                    "direction": factor.direction,
                    "expression": factor.expression,
                    "rank_ic": _metric(metrics, "Rank_IC", "rank_ic"),
                    "rank_icir": _metric(metrics, "Rank_ICIR", "rank_icir"),
                    "annualized_return": _metric(metrics, "annualized_return"),
                    "sharpe": _metric(metrics, "sharpe"),
                    "max_drawdown": _metric(metrics, "max_drawdown"),
                    "turnover": _metric(metrics, "turnover"),
                    "coverage": _metric(metrics, "coverage"),
                }
            )
        return factors

    @staticmethod
    def _history_payload(context: StrategyProposalContext) -> list[dict[str, Any]]:
        return [
            {
                "strategy_name": outcome.strategy_name,
                "status": outcome.status,
                "gate_failures": list(outcome.gate_failures),
                "metrics": dict(outcome.metrics),
            }
            for outcome in context.prior_outcomes
        ]

    def _prompt(self, context: StrategyProposalContext) -> str:
        factors = json.dumps(self._factor_payload(context), ensure_ascii=False, indent=2)
        history = json.dumps(self._history_payload(context), ensure_ascii=False, indent=2)
        return f"""
Design one auditable China A-share long-only portfolio strategy for inner trial
{context.trial_index} of epoch {context.epoch_id}.  The factor snapshot is
immutable ({context.snapshot_id}, data cutoff {context.data_cutoff}).

Available frozen factors:
{factors}

Prior trials in this same epoch (do not repeat a failed specification):
{history}

Return exactly one JSON object with these keys:
- name: short ASCII strategy name;
- hypothesis: why these factors should complement each other economically;
- factor_names: {self.min_factors}-{self.max_factors} distinct names copied exactly from the list;
- rule: concise declarative score/selection description, never Python code;
- top_n: integer in [{self.min_top_n}, {self.max_top_n}];
- weighting: one of equal, rank, proportional, score_transform,
  inverse_volatility, risk_parity;
- weight_cap: null or a number in (0, 1];
- weight_floor: null or a number in (0, 1] raising every held name to at
  least this weight (it must not exceed weight_cap); leave it null unless a
  minimum effective position is part of the hypothesis;
- score_transform_kind: only with weighting score_transform, one of linear,
  truncated, sigmoid; score_transform_slope: positive number used by linear
  and sigmoid; score_transform_cut: number used as the truncation point
  (truncated) or the sigmoid midpoint;
- inverse_vol_window: integer >= 2 (default 20) used only with weighting
  inverse_volatility -- the volatility lookback over the realised daily
  returns up to each decision date; inverse_vol_min: null or a positive
  number flooring every volatility estimate; inverse_vol_multiply_score:
  boolean (default false) -- multiply the inverse-volatility weights by the
  score before normalising;
- risk_parity_window: integer >= 2 (default 60) used only with weighting
  risk_parity -- the covariance lookback over the realised daily returns up
  to each decision date; risk_parity_shrinkage_kind: only with weighting
  risk_parity, one of none, diagonal; risk_parity_shrinkage: number in
  [0, 1) used only with weighting risk_parity -- the intensity of shrinking
  the covariance towards its diagonal;
- drift_band_name: null or a number in (0, 1] -- on a rebalance date, when no
  single name's target weight deviates from its drifted weight by more than
  this absolute band, the rebalance is skipped and the book keeps its drifted
  positions; drift_band_portfolio: null or a number in (0, 1] -- the same
  skip rule measured portfolio-wide as half the L1 weight change including
  cash (the same convention as the reported turnover); provide at least one
  of the two bands before using the next field;
- drift_band_forced_days: null or an integer >= 1 -- force a rebalance at
  least this many trading days after the previously executed one;
- turnover_shrinkage: null or a number in [0, 1) -- every executed rebalance
  moves only (1 - turnover_shrinkage) of the way from the drifted weights to
  the target weights, reducing turnover at the cost of tracking the signal
  more slowly; turnover_max_step: null or a number in (0, 1] -- hard cap on
  the per-rebalance turnover (half-L1 including cash) after shrinking;
- holding_buffer_entry: null or a number in (0, 1] -- percentile-from-the-top
  threshold a fresh name needs before it may enter the book on a rebalance
  date (the name must rank within the top this share of the valid
  cross-section); holding_buffer_retain: null or a number in (0, 1] no
  greater than holding_buffer_entry -- a held name that dropped out of the
  top_n keeps its seat while its percentile stays at or above this easier
  threshold (it collapses onto holding_buffer_entry when null);
  holding_buffer_max_extra: integer >= 0 -- how many rescued holdings may
  push the book beyond top_n; all three are ignored unless
  holding_buffer_entry is set;
- diversity_max_correlation: null or a number in [-1, 1] -- candidates are
  admitted in score order and a candidate whose correlation with an
  already-admitted name (measured on the realised daily returns available up
  to the decision date) exceeds this bound is skipped;
- min_holding_days: null or an integer >= 1 -- a name cannot be sold for this
  many completed trading days after the trade that entered it; the remaining
  book rescales proportionally to fund the blocked sale;
- participation_max_rate: null or a number in (0, 1] -- the share of a name's
  average traded amount a single day's order may consume; the remainder is
  worked over the following trading days (needs the injected traded-amount
  panels); participation_max_days: null or an integer >= 1 -- how many trading
  days an unfilled order may stay pending before it is cancelled;
  participation_amount_window: null or an integer >= 1 -- the trailing window
  the average traded amount is measured over; participation_unfilled: null or
  one of cancel, carry -- what happens to the unfilled remainder, carry (the
  default) keeps it pending under participation_max_days; cannot be combined
  with staged_max_days;
- lot_rounding: boolean (default false) -- round every executed weight down to
  a whole number of tradable lots; lot_rounding_mode: null or one of down,
  nearest (down is the default and never trades more than the target);
  lot_min_notional: null or a positive number -- a rounded order below this
  notional is dropped and stays in cash;
- side_aware_tradability: null or one of block_buy, block_sell, block_both --
  a suspended name, or a name printing at its price limit, cannot be traded on
  the blocked side; the mask only blocks a trade, a held name keeps its
  position;
- staged_max_days: null or an integer >= 1 -- spread one rebalance over at
  most this many trading days, the remainder stays pending; staged_priority:
  null or one of score, weight (default score) -- the service order of the
  staged orders; cannot be combined with participation_max_rate;
- vol_target_annual: null or a positive number -- the annualised volatility the
  realised book is scaled towards; vol_target_window: null or an integer >= 2
  (default 20) -- the realised-return lookback; vol_target_min_scale: null or
  a number in (0, 1] and vol_target_max_scale: null or a number >= 1 bound the
  scaling factor and must be provided together;
- drawdown_thresholds: null or an object mapping warning, reduce and halt to
  strictly increasing numbers in (0, 1] -- the drawdown levels that drive the
  state machine; drawdown_confirm_days: null or an integer >= 1 -- how many
  consecutive eligible days a level must hold before the state moves;
  drawdown_scales: null or an object mapping the same three state names to
  numbers in [0, 1] -- the target-weight scale of each state (a halt scale of
  0.0 freezes the book, it does not liquidate it); drawdown_recovery: null or
  a number in (0, 1] -- how far the drawdown must retrace from its deepest
  point, held for drawdown_confirm_days, before the state steps back down;
- signal_decay_window: null or an integer >= 2 -- the trailing number of
  decision periods the factor-decay monitor reads; signal_decay_fail_periods:
  null or an integer >= 1 -- how many consecutive weak rank-IC periods mark a
  factor as decaying; signal_decay_ic_threshold: null or a finite number --
  the rank-IC level a period must stay strictly below to count as weak;
  signal_decay_scale: null or a number in [0, 1] -- the share of its blend
  contribution a decaying factor keeps (the factor is scaled, never replaced);
- signal_lag: integer at least 1;
- signal_transform: null or one of linear, log, signed_log, sqrt -- one
  optional monotone cross-sectional transform of every raw factor value
  before ranking (signal_transform_scale > 0 and signal_transform_shift are
  numbers used only by linear);
- signal_winsorize_lower and signal_winsorize_upper: null or quantiles with
  0 <= lower < upper <= 1; signal_winsorize_min_samples: integer >= 1;
- signal_standardize: null or one of zscore, rank_z, robust_z;
- signal_standardize_min_std: positive number;
- signal_missing_value: null or one of drop_stock, cross_section_median,
  neutral_rank -- how missing factor values are handled before ranking:
  drop_stock removes any symbol with a missing value from that factor's
  panel, cross_section_median fills each missing value with that date's
  cross-sectional median, neutral_rank fills missing percentile ranks with
  the neutral 0.5; signal_missing_max_fill_ratio: null or a number in
  (0, 1] -- fail instead of filling when more than this share of a date's
  cross-section would be filled (never with drop_stock);
- signal_interaction_pairs: null or a list of at most 8 interaction terms,
  each a JSON list [factor_a, factor_b, kind] with two distinct factor names
  copied exactly from factor_names and kind one of product, min,
  geometric_mean; each term builds one extra synthetic blend member from the
  two factors' post-direction percentile ranks and joins the final blend as
  an additional member;
- signal_blend_weights: null or an object mapping every blend member to a
  positive weight; the blend members are the candidate factors plus every
  interaction term named "factor_a*factor_b"; weights are normalized to sum
  to one; cannot be combined with signal_blend_groups;
- signal_blend_groups: null or an object mapping group names to disjoint
  factor-name lists that together partition exactly the candidate factors;
  the blend averages each group first (signal_blend_within_weights: null or
  an object mapping group name to that group's factor weights, default
  equal) and then averages the group scores plus every interaction term as
  singletons (signal_blend_group_weights: null or an object mapping group
  name to a positive weight, default equal); cannot be combined with
  signal_blend_weights;
- gating_layers: null or a non-empty list of quantile gate layers applied in
  order to the score panel, each a JSON object with exactly these keys:
  factor (copied exactly from factor_names), direction (top or bottom),
  quantile (a number strictly between 0 and 1), min_pass (integer >= 1).
  A symbol survives a date only when it passes every layer; a symbol that
  fails any layer receives no weight on that date;

Use the frozen direction supplied for each factor.  Prefer complementary themes
and explain any conditional factor.  Do not use future data, do not invent a
factor, do not emit executable code, and do not optimize against a final holdout.
Transaction costs are not proposal fields. The system always charges
{BUY_COST_BPS:g} bps on buys and {SELL_COST_BPS:g} bps on sells; do not emit any
cost or fee key.
""".strip()

    def __call__(self, context: StrategyProposalContext) -> StrategyCandidate:
        response = self.api_backend.build_messages_and_create_chat_completion(
            user_prompt=self._prompt(context),
            system_prompt=(
                "You are the portfolio-construction component of a fundamental quant research agent. "
                "Return strict JSON specifications; deterministic framework code performs every backtest."
            ),
            json_mode=True,
            json_target_type=dict[str, Any],
        )
        payload = json.loads(response)
        if not isinstance(payload, Mapping):
            raise ValueError("strategy proposal must be a JSON object")

        factor_names = tuple(str(value) for value in payload.get("factor_names", []))
        if len(factor_names) != len(set(factor_names)):
            raise ValueError("strategy proposal contains duplicate factor names")
        if not self.min_factors <= len(factor_names) <= self.max_factors:
            raise ValueError(f"strategy proposal must use {self.min_factors}-{self.max_factors} factors")
        available = {factor.name for factor in context.working_set.factors}
        unknown = sorted(set(factor_names).difference(available))
        if unknown:
            raise ValueError(f"strategy proposal invented factors: {unknown[:3]}")

        top_n = int(payload.get("top_n", 0))
        if not self.min_top_n <= top_n <= self.max_top_n:
            raise ValueError(f"top_n must be in [{self.min_top_n}, {self.max_top_n}]")

        gating_layers = payload.get("gating_layers") or []
        if not isinstance(gating_layers, list):
            raise ValueError("gating_layers must be a list of gate layers")
        try:
            gating = GatingSpec(
                layers=tuple(
                    GateLayer(
                        factor=str(layer["factor"]),
                        direction=str(layer["direction"]),
                        quantile=float(layer["quantile"]),
                        min_pass=int(layer.get("min_pass", 1)),
                    )
                    for layer in gating_layers
                )
            )
        except KeyError as exc:
            raise ValueError(f"gating layer is missing the {exc} key") from exc
        except TypeError as exc:
            raise ValueError("each gating layer must be a JSON object") from exc

        try:
            raw_interaction_pairs = payload.get("signal_interaction_pairs") or ()
            for pair in raw_interaction_pairs:
                # Same fail-closed rule as SignalSpec and the worker: no
                # silent truncation of an over-long pair.
                if not isinstance(pair, (list, tuple)) or len(pair) != 3:
                    raise ValueError("each interaction pair must be a [first, second, kind] triple")
            signal_interaction_pairs = tuple(
                (str(pair[0]), str(pair[1]), str(pair[2])) for pair in raw_interaction_pairs
            )
            signal_missing_value = payload.get("signal_missing_value")
            signal_blend_weights = payload.get("signal_blend_weights")
            if signal_blend_weights is not None and not isinstance(signal_blend_weights, Mapping):
                # Same message as SignalSpec/worker for a list-valued payload.
                raise ValueError("blend_weights must not be empty")
            signal_blend_groups = payload.get("signal_blend_groups")
            signal_group_weights = payload.get("signal_blend_group_weights")
            signal_within_weights = payload.get("signal_blend_within_weights")
            signal = SignalSpec(
                transform=None if payload.get("signal_transform") is None else str(payload["signal_transform"]),
                transform_scale=(
                    1.0 if payload.get("signal_transform_scale") is None else float(payload["signal_transform_scale"])
                ),
                transform_shift=(
                    0.0 if payload.get("signal_transform_shift") is None else float(payload["signal_transform_shift"])
                ),
                winsorize_lower=(
                    None if payload.get("signal_winsorize_lower") is None else float(payload["signal_winsorize_lower"])
                ),
                winsorize_upper=(
                    None if payload.get("signal_winsorize_upper") is None else float(payload["signal_winsorize_upper"])
                ),
                winsorize_min_samples=int(payload.get("signal_winsorize_min_samples", 1)),
                standardize=None if payload.get("signal_standardize") is None else str(payload["signal_standardize"]),
                standardize_min_std=(
                    DEFAULT_STANDARDIZE_MIN_STD
                    if payload.get("signal_standardize_min_std") is None
                    else float(payload["signal_standardize_min_std"])
                ),
                missing_value=None if signal_missing_value is None else str(signal_missing_value),
                missing_max_fill_ratio=(
                    None
                    if payload.get("signal_missing_max_fill_ratio") is None
                    else float(payload["signal_missing_max_fill_ratio"])
                ),
                blend_weights=(
                    None
                    if signal_blend_weights is None
                    else {str(key): float(value) for key, value in signal_blend_weights.items()}
                ),
                blend_groups=(
                    None
                    if signal_blend_groups is None
                    else {
                        str(group): [str(factor) for factor in factors]
                        for group, factors in signal_blend_groups.items()
                    }
                ),
                blend_group_weights=(
                    None
                    if signal_group_weights is None
                    else {str(key): float(value) for key, value in signal_group_weights.items()}
                ),
                blend_within_weights=(
                    None
                    if signal_within_weights is None
                    else {
                        str(group): {str(factor): float(weight) for factor, weight in weights.items()}
                        for group, weights in signal_within_weights.items()
                    }
                ),
                interaction_pairs=signal_interaction_pairs,
                # Batch C2 signal operators (roadmap 4.6/4.7/4.12) are
                # data-dependent configuration inputs -- deliberately absent
                # from the prompt vocabulary (the industry map, style/risk
                # exposures and regime states are injections the agent cannot
                # know), but a hallucinated key still passes through and fails
                # closed in spec validation.
                industry_neutralize=bool(payload.get("signal_industry_neutralize", False)),
                industry_neutralize_industry_map=(
                    None
                    if payload.get("signal_industry_neutralize_industry_map") is None
                    else {
                        str(symbol): str(label)
                        for symbol, label in payload["signal_industry_neutralize_industry_map"].items()
                    }
                ),
                industry_neutralize_method=str(payload.get("signal_industry_neutralize_method", "demean")),
                # Raw passthrough (fact 37 discipline): int() would silently
                # truncate 2.5 and bypass the whole-number fail-closed the
                # canonical payload path raises.
                industry_neutralize_min_group=payload.get("signal_industry_neutralize_min_group", 3),
                industry_neutralize_version=(
                    None
                    if payload.get("signal_industry_neutralize_version") is None
                    else str(payload["signal_industry_neutralize_version"])
                ),
                industry_neutralize_exposures=(
                    None
                    if payload.get("signal_industry_neutralize_exposures") is None
                    else {
                        str(name): {str(symbol): float(number) for symbol, number in values.items()}
                        for name, values in payload["signal_industry_neutralize_exposures"].items()
                    }
                ),
                style_neutralize=bool(payload.get("signal_style_neutralize", False)),
                style_neutralize_exposures=(
                    None
                    if payload.get("signal_style_neutralize_exposures") is None
                    else {
                        str(name): {str(symbol): float(number) for symbol, number in values.items()}
                        for name, values in payload["signal_style_neutralize_exposures"].items()
                    }
                ),
                style_neutralize_min_samples=payload.get("signal_style_neutralize_min_samples", 5),
                regime_states=(
                    None
                    if payload.get("signal_regime_states") is None
                    else {
                        str(date_key): str(label)
                        for date_key, label in payload["signal_regime_states"].items()
                    }
                ),
                regime_factor_groups=(
                    None
                    if payload.get("signal_regime_factor_groups") is None
                    else {
                        str(label): [str(factor) for factor in factors]
                        for label, factors in payload["signal_regime_factor_groups"].items()
                    }
                ),
                regime_lag=payload.get("signal_regime_lag", 1),
                # Batch 6 FactorDecayMonitor (roadmap 9.5).  Raw passthrough
                # (fact 37 discipline): the four values are whole/range
                # checked in SignalSpec, so int()/float() here would silently
                # truncate 2.5 and bypass the fail-closed path.
                decay_window=payload.get("signal_decay_window"),
                decay_fail_periods=payload.get("signal_decay_fail_periods"),
                decay_ic_threshold=payload.get("signal_decay_ic_threshold"),
                decay_scale=payload.get("signal_decay_scale"),
            )
        except (TypeError, IndexError) as exc:
            raise ValueError(f"invalid signal DAG field in the strategy proposal: {exc}") from exc

        return StrategyCandidate(
            name=_safe_name(payload.get("name"), context),
            universe=self.universe,
            composition=CompositionSpec(
                top_n=top_n,
                weighting=str(payload.get("weighting", "equal")),
                weight_cap=(None if payload.get("weight_cap") is None else float(payload["weight_cap"])),
                weight_floor=(None if payload.get("weight_floor") is None else float(payload["weight_floor"])),
                weight_floor_allow_cash=bool(payload.get("weight_floor_allow_cash", True)),
                score_transform_kind=(
                    None if payload.get("score_transform_kind") is None else str(payload["score_transform_kind"])
                ),
                score_transform_slope=(
                    1.0 if payload.get("score_transform_slope") is None else float(payload["score_transform_slope"])
                ),
                score_transform_cut=(
                    0.0 if payload.get("score_transform_cut") is None else float(payload["score_transform_cut"])
                ),
                holding_buffer_entry=(
                    None if payload.get("holding_buffer_entry") is None else float(payload["holding_buffer_entry"])
                ),
                holding_buffer_retain=(
                    None if payload.get("holding_buffer_retain") is None else float(payload["holding_buffer_retain"])
                ),
                holding_buffer_max_extra=int(payload.get("holding_buffer_max_extra", 0)),
                diversity_max_correlation=(
                    None
                    if payload.get("diversity_max_correlation") is None
                    else float(payload["diversity_max_correlation"])
                ),
                industry_weight_cap=(
                    None if payload.get("industry_weight_cap") is None else float(payload["industry_weight_cap"])
                ),
                industry_map=(
                    None
                    if payload.get("industry_map") is None
                    else {str(symbol): str(label) for symbol, label in payload["industry_map"].items()}
                ),
                cash_residual_max=(
                    None if payload.get("cash_residual_max") is None else float(payload["cash_residual_max"])
                ),
                inverse_vol_window=(
                    20 if payload.get("inverse_vol_window") is None else int(payload["inverse_vol_window"])
                ),
                inverse_vol_min=(
                    None if payload.get("inverse_vol_min") is None else float(payload["inverse_vol_min"])
                ),
                inverse_vol_multiply_score=bool(payload.get("inverse_vol_multiply_score", False)),
                hierarchical_groups=(
                    None
                    if payload.get("hierarchical_groups") is None
                    else {
                        str(group): tuple(str(member) for member in members)
                        for group, members in payload["hierarchical_groups"].items()
                    }
                ),
                hierarchical_group_weights=(
                    None
                    if payload.get("hierarchical_group_weights") is None
                    else {str(group): float(value) for group, value in payload["hierarchical_group_weights"].items()}
                ),
                risk_parity_window=payload.get("risk_parity_window", 60),
                risk_parity_budgets=(
                    None
                    if payload.get("risk_parity_budgets") is None
                    else {str(symbol): float(budget) for symbol, budget in payload["risk_parity_budgets"].items()}
                ),
                risk_parity_shrinkage_kind=str(payload.get("risk_parity_shrinkage_kind", "diagonal")),
                risk_parity_shrinkage=(
                    0.1 if payload.get("risk_parity_shrinkage") is None else float(payload["risk_parity_shrinkage"])
                ),
                risk_parity_tolerance=(
                    1e-8 if payload.get("risk_parity_tolerance") is None else float(payload["risk_parity_tolerance"])
                ),
                frb_budgets=(
                    None
                    if payload.get("frb_budgets") is None
                    else {str(factor): float(budget) for factor, budget in payload["frb_budgets"].items()}
                ),
                frb_exposures=(
                    None
                    if payload.get("frb_exposures") is None
                    else {
                        str(name): {str(symbol): float(number) for symbol, number in values.items()}
                        for name, values in payload["frb_exposures"].items()
                    }
                ),
                frb_factor_covariance=(
                    None
                    if payload.get("frb_factor_covariance") is None
                    else {
                        str(name): {str(symbol): float(number) for symbol, number in values.items()}
                        for name, values in payload["frb_factor_covariance"].items()
                    }
                ),
                frb_idiosyncratic=(
                    None
                    if payload.get("frb_idiosyncratic") is None
                    else {str(symbol): float(variance) for symbol, variance in payload["frb_idiosyncratic"].items()}
                ),
                frb_idio_max=None if payload.get("frb_idio_max") is None else float(payload["frb_idio_max"]),
                frb_ridge=(0.0 if payload.get("frb_ridge") is None else float(payload["frb_ridge"])),
                # Batch C3 exposure-constraint families (roadmap
                # 6.4/6.5/6.6/6.7/6.9) are data-dependent configuration inputs
                # -- deliberately absent from the prompt vocabulary (the
                # benchmark weights, exposure matrices and betas are
                # injections the agent cannot know), but a hallucinated key
                # still passes through and fails closed in spec validation.
                industry_deviation_max=payload.get("industry_deviation_max"),
                industry_deviation_industry_map=payload.get("industry_deviation_industry_map"),
                industry_deviation_exempt=payload.get("industry_deviation_exempt") or (),
                industry_deviation_benchmark=payload.get("industry_deviation_benchmark"),
                style_exposure_bounds=payload.get("style_exposure_bounds"),
                style_exposure_matrix=payload.get("style_exposure_matrix"),
                style_exposure_benchmark=payload.get("style_exposure_benchmark"),
                beta_constraint_min=payload.get("beta_constraint_min"),
                beta_constraint_max=payload.get("beta_constraint_max"),
                beta_by_symbol=payload.get("beta_by_symbol"),
                factor_exposure_bounds=payload.get("factor_exposure_bounds"),
                factor_exposure_matrix=payload.get("factor_exposure_matrix"),
                projection_max_iter=payload.get("projection_max_iter", 512),
                projection_tolerance=payload.get("projection_tolerance", 1e-9),
            ),
            signal=signal,
            gating=gating,
            # Candidate-pool filters (roadmap section 3) are configuration
            # inputs, not agent-selectable parameters: deliberately absent
            # from the prompt vocabulary, but a hallucinated ``universe_rules``
            # key is still passed through and fails closed in spec validation.
            universe_rules=None if payload.get("universe_rules") is None else UniverseSpec.from_payload(
                dict(payload["universe_rules"])
            ),
            # The order-layer market panels (roadmap 8.6-8.9) are data
            # injections the agent cannot know: deliberately absent from the
            # prompt vocabulary, passed through in full, and fail-closed in
            # ExecutionDataSpec when malformed.
            execution_data=(
                None
                if payload.get("execution_data") is None
                else ExecutionDataSpec.from_payload(dict(payload["execution_data"]))
            ),
            factor_names=factor_names,
            rule=str(payload.get("rule", "cross-sectional rank blend")).strip(),
            hypothesis=str(payload.get("hypothesis", "")).strip(),
            backtest=BacktestSpec(
                signal_lag=int(payload.get("signal_lag", 1)),
                drift_band_name=(
                    None if payload.get("drift_band_name") is None else float(payload["drift_band_name"])
                ),
                drift_band_portfolio=(
                    None if payload.get("drift_band_portfolio") is None else float(payload["drift_band_portfolio"])
                ),
                drift_band_forced_days=(
                    None if payload.get("drift_band_forced_days") is None else int(payload["drift_band_forced_days"])
                ),
                turnover_shrinkage=(
                    None if payload.get("turnover_shrinkage") is None else float(payload["turnover_shrinkage"])
                ),
                turnover_max_step=(
                    None if payload.get("turnover_max_step") is None else float(payload["turnover_max_step"])
                ),
                min_holding_days=(
                    None if payload.get("min_holding_days") is None else int(payload["min_holding_days"])
                ),
                # Batch 6 order layer and portfolio-risk operators.  Every key
                # travels raw (fact 37 discipline): whole-number and range
                # checks live in BacktestSpec.__post_init__, so int()/float()
                # here would truncate a non-whole payload before the
                # fail-closed message could fire.
                participation_max_rate=payload.get("participation_max_rate"),
                participation_max_days=payload.get("participation_max_days"),
                participation_amount_window=payload.get("participation_amount_window"),
                participation_unfilled=payload.get("participation_unfilled"),
                lot_rounding=payload.get("lot_rounding", False),
                lot_min_notional=payload.get("lot_min_notional"),
                lot_rounding_mode=payload.get("lot_rounding_mode"),
                side_aware_tradability=payload.get("side_aware_tradability"),
                staged_max_days=payload.get("staged_max_days"),
                staged_priority=payload.get("staged_priority"),
                vol_target_annual=payload.get("vol_target_annual"),
                vol_target_window=payload.get("vol_target_window"),
                vol_target_min_scale=payload.get("vol_target_min_scale"),
                vol_target_max_scale=payload.get("vol_target_max_scale"),
                drawdown_thresholds=payload.get("drawdown_thresholds"),
                drawdown_confirm_days=payload.get("drawdown_confirm_days"),
                drawdown_recovery=payload.get("drawdown_recovery"),
                drawdown_scales=payload.get("drawdown_scales"),
            ),
            metadata={
                "proposal_backend": "rdagent_llm",
                "epoch_id": context.epoch_id,
                "trial_index": context.trial_index,
                "snapshot_id": context.snapshot_id,
                "working_set_id": context.working_set.working_set_id,
            },
        )
