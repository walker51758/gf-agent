"""Local RankIC factor evaluator for the search's factor-proposal path.

The strategy worker evaluates candidates on the data machine.  While that machine
is unreachable, the search still has to gate a proposed factor or the factor half
of the action space is dead -- so this module runs the *existing* GF factor
evaluator locally, against the cached market panels, and hands the result to the
unchanged :class:`FactorAcceptanceService`.

Reimplementation was deliberately avoided.  ``scripts/gf_quant_remote_worker.py``
already contains the authoritative expression engine (``SafeExpression``), the
panel loader (``PanelLoader``), the per-date series (``daily_series``) and the
metric aggregation (``metrics_from_daily``); the isolation harness imports that
module and calls it, so the numbers a local run reports are the numbers the data
machine would report for the same expression and window.  A second implementation
of RankIC would drift from the first and make the two incomparable, which is the
exact failure the parity suite exists to prevent.

The evaluator satisfies the protocol :class:`FactorAcceptanceService` expects:
``__call__(candidate) -> EvaluationResult``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.scenarios.gf_quant.remote import RemoteSettings
from rdagent.scenarios.gf_quant.service import EvaluationResult
from rdagent.scenarios.gf_quant.strategy.search.data_cache import (
    WORKER_CALENDAR_NAME,
    WORKER_CLOSE_NAME,
    SearchDataCache,
)

# The worker module is a script, not an installed package, so it is loaded from
# its path the same way ``GFFactorEvaluator._evaluate_locally`` does.  Loading it
# once per process keeps the panel cache inside it warm across candidates.
_WORKER_MODULE_NAME = "gf_quant_remote_worker"


# The window a proposed factor is judged on.  Kept as a literal so the interval
# is stated in exactly one place: a gate verdict depends on it, and a window that
# drifts with the cached calendar would make two runs incomparable.
#
# This is the interval a *proposal* is evaluated on.  ``benchmark_standards.json``
# records a different interval -- ``provenance.evaluation`` says
# ``2020-01-01 .. 2023-12-31`` -- and that one is the sampling interval the
# thresholds themselves were derived from, not the interval used here.  The two
# are deliberately independent: changing this constant must not move a threshold.
EVALUATION_WINDOW_START = "2022-01-01"
EVALUATION_WINDOW_END = "2023-12-31"


def window_for_cache(
    cache: SearchDataCache,
    start: str = EVALUATION_WINDOW_START,
    end: str = EVALUATION_WINDOW_END,
) -> pd.DatetimeIndex:
    """The cached trading dates inside ``[start, end]``.

    A cache that does not reach the requested interval at all is an error rather
    than a silent substitution: judging a candidate outside the interval the
    verdict is meant to describe is the mismatch a fixed window exists to remove.
    """

    dates = pd.DatetimeIndex(cache.trading_dates())
    inside = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if len(inside) == 0:
        raise ValueError(
            f"cache covers {dates.min().date()}..{dates.max().date()}, which has no overlap with the "
            f"evaluation window {start}..{end}; the calibrated standards cannot be applied to it"
        )
    return inside


def repository_root() -> Path:
    """The repository root, resolved from this file and checked against a marker.

    This module sits five levels below the root
    (``<repo>/rdagent/scenarios/gf_quant/strategy/search/``), one deeper than the
    rest of the strategy package; resolving from a plain parents[N] without a
    marker is how a sibling module in this package silently produced
    ``rdagent/scripts/...``.
    """

    root = Path(__file__).resolve().parents[5]
    if not (root / "rdagent" / "scenarios" / "gf_quant").is_dir():
        raise RuntimeError(f"cannot locate the repository root from {__file__}: tried {root}")
    return root


def load_factor_worker(worker_path: Path | None = None) -> Any:
    """Import the GF factor worker as a module, caching it per process."""

    if _WORKER_MODULE_NAME in sys.modules:
        return sys.modules[_WORKER_MODULE_NAME]
    path = worker_path or repository_root() / "scripts" / "gf_quant_remote_worker.py"
    if not path.exists():
        raise FileNotFoundError(f"GF factor worker not found: {path}")
    spec = importlib.util.spec_from_file_location(_WORKER_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load the GF factor worker: {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module's own dataclasses resolve their
    # annotations through ``sys.modules`` and fail without this entry.
    sys.modules[_WORKER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class LocalFactorEvaluationSettings:
    """Evaluation window and sampling, mirroring the worker's request fields."""

    start: str
    end: str
    label_horizon: int = 5
    quantile: float = 0.2
    min_cross_section: int = 10
    sample_step: int = 5
    lookback_days: int = 400
    close_field: str = WORKER_CLOSE_NAME

    def as_payload(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "label_horizon": self.label_horizon,
            "quantile": self.quantile,
            "min_cross_section": self.min_cross_section,
            "sample_step": self.sample_step,
        }


class LocalRankICFactorEvaluator:
    """Evaluate a factor expression on the cached panels, then gate it.

    Artifacts land under ``artifact_dir`` (one parquet per accepted factor), so a
    factor that passes carries the same kind of provenance the remote path
    produces and can be re-read by the strategy side without a second pull.
    """

    def __init__(
        self,
        *,
        cache: SearchDataCache,
        settings: LocalFactorEvaluationSettings,
        artifact_dir: str | Path,
        worker_path: Path | None = None,
    ) -> None:
        self.cache = cache
        self.settings = settings
        self.base_path = Path(cache.root)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._worker = load_factor_worker(worker_path)
        self._config = self._worker.EvaluationConfig(
            start=settings.start,
            end=settings.end,
            label_horizon=settings.label_horizon,
            quantile=settings.quantile,
            min_cross_section=settings.min_cross_section,
            sample_step=settings.sample_step,
        )
        self._loader: Any | None = None
        self._close: pd.DataFrame | None = None
        self.calls = 0

    def _panels(self) -> tuple[Any, pd.DataFrame]:
        """One loader and one close panel per evaluation window, reused."""

        if self._loader is None:
            self._loader = self._worker.PanelLoader(
                self.base_path,
                self.settings.start,
                self.settings.end,
                lookback_days=self.settings.lookback_days,
            )
            self._close = self._loader.load(self.settings.close_field)
        assert self._close is not None
        return self._loader, self._close

    def __call__(self, candidate: Any) -> EvaluationResult:
        self.calls += 1
        expression = str(candidate.expression)
        loader, close = self._panels()
        parsed = self._worker.SafeExpression(expression)
        factor = parsed.evaluate(loader)
        # ``daily_series`` is called directly rather than ``evaluate_factor``
        # because the compact profile has to come off the same per-date series
        # that produced the metrics; the benchmark worker builds its own
        # profiles the same way, so the two stay on one definition.
        daily, window = self._worker.daily_series(factor, close, self._config)
        metrics = self._worker.metrics_from_daily(daily, window.index)
        artifact_path, checksum = self._write_artifact(str(candidate.name), window)
        return EvaluationResult(
            metrics=dict(metrics),
            artifact_path=str(artifact_path),
            artifact_checksum=checksum,
            # The daily long-short spread, which is what the calibrated gate
            # measures direction-adjusted return, Sharpe and drawdown on.  Any
            # other series would put those three gates on a definition the
            # standards were never calibrated against.
            return_profile=self._profile(daily),
        )

    def _write_artifact(self, name: str, window: pd.DataFrame) -> tuple[Path, str]:
        safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in name)[:64]
        path = self.artifact_dir / f"{safe or 'factor'}.parquet"
        frame = window.copy()
        # Financial/quarterly operators attach non-tabular PIT caches to
        # ``DataFrame.attrs``, which pyarrow cannot serialise into parquet
        # metadata; the cache is execution provenance, not factor content.
        frame.attrs.clear()
        frame.to_parquet(path, engine="pyarrow")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _profile(daily: Any) -> pd.Series | None:
        """The daily long-short spread series, as the calibrated gate expects.

        The gate's direction-adjusted return, Sharpe and drawdown terms are
        computed from whatever series is supplied here, and the calibrated
        standards were derived from long-short spreads.  The spread is already
        computed inside the worker's ``daily_series`` -- this only reads it, so
        the profile and the metrics in one result describe one series.
        """

        spread = getattr(daily, "spread", None)
        if spread is None or len(spread) == 0:
            return None
        profile = spread.dropna()
        return None if profile.empty else profile

    def request_for(self, candidate: Any) -> dict[str, Any]:
        """The data-only request the remote worker would receive.

        Built here as well so the local and remote paths are driven by one
        description of a candidate: the remote evaluator takes this dict, and a
        test can assert the two agree without a second, hand-maintained payload.
        """

        safe = "".join(
            character if character.isalnum() or character in "._-" else "_" for character in str(candidate.name)
        ).strip("._") or "candidate"
        return {
            "name": safe,
            "family": str(candidate.family),
            "expression": str(candidate.expression),
            "base_path": str(self.base_path),
            "lookback_days": self.settings.lookback_days,
            "close_field": self.settings.close_field,
            "artifact_path": str(self.artifact_dir / f"{safe}.parquet"),
            "evaluation": self.settings.as_payload(),
        }


def settings_for_cache(
    cache: SearchDataCache,
    *,
    warmup_days: int = 130,
    label_horizon: int = 5,
    min_cross_section: int = 10,
    sample_step: int = 1,
    trailing_trim_days: int = 10,
    minimum_calendar_years: int | None = None,
    minimum_observations: int | None = None,
    window: tuple[str, str] | None = (EVALUATION_WINDOW_START, EVALUATION_WINDOW_END),
) -> LocalFactorEvaluationSettings:
    """Derive an evaluation window that the cache can actually serve.

    ``window`` (the default) pins the interval to
    ``EVALUATION_WINDOW_START .. EVALUATION_WINDOW_END`` intersected with the
    cached calendar.  A fixed interval is what makes two runs comparable and lets
    a gate verdict be attributed to the factor rather than to how far the cached
    calendar happened to reach.  Passing ``None`` restores the previous
    behaviour of walking back from the trimmed cache end until the window covers
    ``minimum_calendar_years`` years.

    ``minimum_calendar_years``, when given explicitly, widens the fixed window
    backwards rather than being silently violated: the annual gate needs a finite
    RankIC in several calendar years, and a caller that asks for that guarantee
    must receive it.

    Four adjustments are mandatory rather than cosmetic:

    * a warm-up at the front, because the DSL's rolling operators need history
      before the first valid value;
    * a trim at the back of the label horizon, because the last
      ``label_horizon`` dates have no forward return and would otherwise count
      as valid observations with a missing label;
    * a warm-up *ceiling* driven by ``minimum_calendar_years``.  The acceptance
      gate requires a finite RankIC in at least ``annual_min_valid_years``
      calendar years, and a window that skips a year the cache covers can fail
      that gate for a reason that has nothing to do with the factor.
    * ``sample_step = 1`` by default.  The evaluator's sampling stride is what
      sets the observation count, and the calibrated standard demands
      ``min_observations`` in the high hundreds; a stride of 5 silently divides
      the available observations by five and makes the observation gate
      unreachable.  ``minimum_observations`` states that requirement explicitly
      when a calibration is in play.
    """

    from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS

    if window is not None:
        inside = window_for_cache(cache, *window)
        if minimum_calendar_years is not None:
            dates = pd.DatetimeIndex(cache.trading_dates())
            first = int(dates.get_loc(inside[0]))
            last = int(dates.get_loc(inside[-1]))
            while first > 0 and len({int(year) for year in dates[first : last + 1].year}) < minimum_calendar_years:
                first -= 1
            inside = dates[first : last + 1]
        settings = LocalFactorEvaluationSettings(
            start=inside[0].strftime("%Y-%m-%d"),
            end=inside[-1].strftime("%Y-%m-%d"),
            label_horizon=label_horizon,
            min_cross_section=min_cross_section,
            sample_step=sample_step,
        )
        if minimum_observations is not None:
            observations = len(inside[::sample_step])
            if observations < minimum_observations:
                raise ValueError(
                    f"the fixed window yields {observations} observations at sample_step={sample_step}, "
                    f"but the acceptance gate requires {minimum_observations}; pass window=None to widen "
                    f"the window, or lower sample_step"
                )
        return settings

    required_years = (
        minimum_calendar_years
        if minimum_calendar_years is not None
        else int(GF_SCENARIO_SETTINGS.annual_gate_min_years_with_rank_ic)
    )
    dates = cache.trading_dates()
    end_index = len(dates) - 1 - (trailing_trim_days + label_horizon)
    if end_index <= 0:
        raise ValueError(
            f"cache holds only {len(dates)} trading dates; too few for a "
            f"{trailing_trim_days + label_horizon}-day trailing trim"
        )
    # Step back from the trimmed end until the window covers the required number
    # of distinct calendar years, but never past the warm-up the operators need.
    start_index = warmup_days
    while start_index < end_index:
        span = pd.DatetimeIndex(dates[start_index : end_index + 1])
        if len(set(span.year)) >= required_years:
            break
        start_index += 1
    window = pd.DatetimeIndex(dates[start_index : end_index + 1])
    window_years = sorted({int(year) for year in window.year})
    if len(window_years) < required_years:
        raise ValueError(
            f"cache covers {sorted({int(year) for year in dates.year})}, which cannot supply "
            f"{required_years} calendar years for the acceptance gate's annual check; "
            f"pull a longer window"
        )
    observations = len(window[::sample_step])
    if minimum_observations is not None and observations < minimum_observations:
        raise ValueError(
            f"the derived window yields {observations} observations at sample_step={sample_step}, "
            f"but the acceptance gate requires {minimum_observations}; pull a longer window or "
            f"lower sample_step"
        )
    return LocalFactorEvaluationSettings(
        start=window[0].strftime("%Y-%m-%d"),
        end=window[-1].strftime("%Y-%m-%d"),
        label_horizon=label_horizon,
        min_cross_section=min_cross_section,
        sample_step=sample_step,
    )


def remote_settings_payload(settings: Any = None) -> dict[str, Any]:
    """The connection fields a remote evaluation needs, minus the secret.

    Built through :class:`RemoteSettings` so the switch to the data machine needs
    no new configuration surface; the password is named, never read here.
    """

    resolved = settings or RemoteSettings.from_env()
    return {
        "host": resolved.host,
        "port": resolved.port,
        "user": resolved.user,
        "password_env": resolved.password_env,
        "python_path": resolved.python_path,
        "work_dir": resolved.work_dir,
        "host_key_sha256": resolved.host_key_sha256,
    }


def digest_payload(payload: Any) -> str:
    """Stable digest of a request payload, for provenance and tests."""

    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def verify_cache_layout(root: str | Path) -> dict[str, Any]:
    """Check a cache directory holds everything the worker loaders require.

    Called before a run rather than at the first evaluation: a missing calendar
    or close file otherwise surfaces as an obscure pyarrow error in the middle of
    a search that has already spent credits.
    """

    base = Path(root)
    required = [
        f"{WORKER_CALENDAR_NAME}.parquet",
        f"{WORKER_CLOSE_NAME}.parquet",
    ]
    present = {name: (base / name).exists() for name in required}
    missing = [name for name, ok in present.items() if not ok]
    if missing:
        raise FileNotFoundError(
            f"search cache at {base} is missing {missing}; re-pull with --refresh-data"
        )
    return {"root": str(base), "present": present}
