"""MCTS joint search over factor sets and strategy structures.

The package is additive: it drives the existing factor and strategy capability
layers (the proposal engines, the acceptance services, the local backtest
engine) without changing any of them, and it keeps its own state in a sidecar
file rather than in the strategy registry.

Entry points:

* :mod:`data_cache` -- the local rqdatac cache the local engine reads;
* :mod:`tree` -- node configuration, identity and the action edge;
* :mod:`actions` -- the two proposers wired into one action space;
* :mod:`evaluator` -- the local/remote evaluation switch;
* :mod:`budget` -- the unified credit counter and φ;
* :mod:`engine` -- the selection / expansion / simulation / backpropagation loop;
* :mod:`sidecar` -- tree persistence and continuation.
"""

from rdagent.scenarios.gf_quant.strategy.search.actions import ActionGenerator
from rdagent.scenarios.gf_quant.strategy.search.budget import (
    BACKTEST_CREDIT,
    DEFAULT_TOTAL_CREDITS,
    CreditLedger,
)
from rdagent.scenarios.gf_quant.strategy.search.data_cache import (
    CacheManifest,
    SearchDataCache,
    default_cache_root,
    update_cache,
)
from rdagent.scenarios.gf_quant.strategy.search.engine import (
    EXPLORATION_C,
    REFERENCE_DEPTH,
    SearchEngine,
    SearchRunResult,
)
from rdagent.scenarios.gf_quant.strategy.search.evaluator import (
    LOCAL_BACKEND,
    REMOTE_BACKEND,
    LocalBacktestEvaluator,
    RemoteSearchEvaluator,
    SearchEvaluation,
    build_evaluator,
)
from rdagent.scenarios.gf_quant.strategy.search.factor_eval import (
    EVALUATION_WINDOW_END,
    EVALUATION_WINDOW_START,
    LocalFactorEvaluationSettings,
    LocalRankICFactorEvaluator,
    load_factor_worker,
    settings_for_cache,
    verify_cache_layout,
    window_for_cache,
)
from rdagent.scenarios.gf_quant.strategy.search.isolation import (
    SearchPaths,
    assert_isolated,
    isolated_paths,
    production_paths,
)
from rdagent.scenarios.gf_quant.strategy.search.sidecar import (
    SIDECAR_VERSION,
    SidecarState,
    load_state,
    sidecar_path,
    write_state,
)
from rdagent.scenarios.gf_quant.strategy.search.tree import (
    MAX_FACTORS,
    MIN_FACTORS,
    Action,
    TreeNode,
    configuration_hash,
    root_node,
    tree_node_from_payload,
)

__all__ = [
    "Action",
    "ActionGenerator",
    "BACKTEST_CREDIT",
    "CacheManifest",
    "CreditLedger",
    "DEFAULT_TOTAL_CREDITS",
    "EXPLORATION_C",
    "LOCAL_BACKEND",
    "LocalBacktestEvaluator",
    "MAX_FACTORS",
    "MIN_FACTORS",
    "REFERENCE_DEPTH",
    "REMOTE_BACKEND",
    "SIDECAR_VERSION",
    "SearchDataCache",
    "SearchEngine",
    "SearchEvaluation",
    "SearchRunResult",
    "SidecarState",
    "TreeNode",
    "build_evaluator",
    "configuration_hash",
    "default_cache_root",
    "load_state",
    "root_node",
    "sidecar_path",
    "tree_node_from_payload",
    "update_cache",
    "write_state",
]
