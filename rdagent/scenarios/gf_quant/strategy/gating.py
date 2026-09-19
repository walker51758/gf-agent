"""Local gating front-end: quantile screens applied on top of the blended score.

Mirrors the worker's gating stage (``scripts/gf_quant_strategy_worker.py``:
``_validate_gating`` / ``_apply_gating``) value for value; the parity suite
(``test_strategy_worker_parity.py``) runs both on identical panels and fails on
any drift.

Semantics (roadmap 5.1 ``QuantileGate`` / 5.2 ``CascadedFactorGating``): every
layer names a candidate factor and keeps, per date, the symbols whose
post-direction percentile rank of that factor passes the layer quantile
(``top`` keeps ``rank >= quantile``, ``bottom`` keeps ``rank <= quantile``).
Layers combine with *and* semantics -- a symbol survives a date only when it
passes every layer -- and a symbol that does not survive is set to ``NaN`` in
the score panel, which the downstream weight builders consume through their
existing ``dropna`` (no eligible panel is introduced).

The per-factor rank panels are exactly the ones the blend consumes, so ties at
the threshold are resolved by the inclusive comparison on the *average*
percentile rank: deterministic and identical on both sides by construction.
A date whose surviving set drops below the layer's ``min_pass`` fails closed
(same philosophy as the signal stages: bad data raises instead of being
silently repaired).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from rdagent.scenarios.gf_quant.strategy.candidate import GatingSpec, SignalSpec
from rdagent.scenarios.gf_quant.strategy.signal import (
    blend_member_panels,
    build_factor_rank_panels,
    factor_decay_weights,
)


def apply_gating(score: pd.DataFrame, ranks: Mapping[str, pd.DataFrame], spec: GatingSpec) -> pd.DataFrame:
    """Apply the cascade to a blended score panel; non-survivors become ``NaN``.

    ``ranks`` holds the post-direction percentile-rank panel of every gating
    factor (the same panels the rank blend consumed).  The score panel must
    share its index and columns with those panels.  With no layers the score
    passes through unchanged, so the gating-off path is the untouched default.
    """

    if not spec.layers:
        return score
    survivors: pd.DataFrame | None = None
    for position, layer in enumerate(spec.layers):
        mask = ranks[layer.factor]
        mask = mask >= layer.quantile if layer.direction == "top" else mask <= layer.quantile
        survivors = mask if survivors is None else survivors & mask
        counts = survivors.sum(axis=1)
        if bool((counts < layer.min_pass).any()):
            raise ValueError(
                f"gating layer {position} ({layer.factor} {layer.direction} {layer.quantile:g}) "
                f"leaves fewer than {layer.min_pass} surviving names on some date"
            )
    assert survivors is not None
    return score.where(survivors)


def build_gated_factor_score(
    panels: Mapping[str, pd.DataFrame],
    directions: Mapping[str, int],
    signal_spec: SignalSpec,
    gating_spec: GatingSpec,
    *,
    index: pd.Index,
    columns: pd.Index,
    eligible: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    lag: int = 1,
) -> pd.DataFrame:
    """Rank pipeline + blend + gating cascade, one call.

    With a disabled ``gating_spec`` this equals
    :func:`rdagent.scenarios.gf_quant.strategy.signal.build_factor_score`.
    ``eligible`` (roadmap section 3) is handed to the rank pipeline, so the
    gating cascade and its quantile thresholds see exactly the post-filter
    cross-section: candidate-pool filtering runs before gating by
    construction.  ``returns``/``lag`` feed the roadmap 9.5 decay monitor the
    same way they do there; the gate thresholds read the *unscaled* rank
    panels, so a decaying factor still gates on its own cross-section.
    """

    ranks = build_factor_rank_panels(panels, directions, signal_spec, index=index, columns=columns, eligible=eligible)
    decay = factor_decay_weights(ranks, signal_spec, returns=returns, lag=lag)
    score = blend_member_panels(ranks, list(panels), signal_spec, decay=decay)
    return apply_gating(score, ranks, gating_spec)
