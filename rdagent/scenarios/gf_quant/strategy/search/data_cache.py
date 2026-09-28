"""Local rqdatac cache for the MCTS search loop.

The search evaluates candidates against a *local* backtest engine, so it needs
one local copy of the market panels the engine reads: a long-format close table
(``dt`` / ``symbol`` / ``value``, identical to the GF dump layout) and one wide
return panel per symbol.  Both are produced from a single rqdatac batch pull and
written under ``git_ignore_folder/`` -- the repository's existing ignore root --
so a cache never reaches version control.

The layout deliberately mirrors what the two readers already expect:

* :meth:`SearchDataCache.close_wide` returns ``index=dt`` / ``columns=symbol``,
  the shape ``backtest.run_portfolio_backtest`` consumes as ``returns``;
* :meth:`SearchDataCache.write_dump` writes ``adjclose.parquet`` in the same
  long format ``scripts/gf_quant_strategy_worker.py:_load_close`` reads, so the
  same cache can serve a future worker run without conversion.

Every panel is written with the adjusted close field, because the backtest
labels returns from it: a raw close would silently mix in dividend and split
jumps and make every candidate's Q incomparable to a future data-machine run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

# The cache defaults are the smallest set that still produces a usable
# cross-section: 40 index members, two calendar years of daily bars.  Both are
# runtime arguments so a demo can widen them without touching this module.
DEFAULT_UNIVERSE_INDEX = "000300.XSHG"
DEFAULT_SYMBOL_COUNT = 40
DEFAULT_YEARS = 2
# The close field under its DSL name, matching ``WORKER_CLOSE_NAME`` and the
# worker's ``evaluation_close_field`` default.  Storing and referring to it by
# the same name is what lets one cache serve the search, the factor evaluator and
# a future worker run without a rename step in between.
DEFAULT_FIELD = "adjclose"

# rqdatac's stock price API returns one row per symbol/date with the fields
# below.  The cache stores them so the factor DSL has the vocabulary it is
# prompted with: ``adjclose`` for returns, ``volume`` for liquidity, and the two
# that the proposer actually emits -- ``amount`` (traded value) and ``turnover``
# (share turnover).  rqdatac names those two ``total_turnover`` and
# ``num_trades`` respectively, so the pull asks for its names and the dump
# renames them to the DSL's.  Without this mapping a proposed expression such as
# ``ts_mean(amount, 20)`` would fail on a missing file rather than on its merit.
_PRICE_FIELDS = (
    "close",
    "volume",
    "total_turnover",
    "num_trades",
)

# rqdatac field -> the name the factor DSL and the worker loaders use.
_FIELD_ALIASES = {
    "close": "adjclose",
    "volume": "volume",
    "total_turnover": "amount",
    "num_trades": "turnover",
}

# The dump additionally exposes two worker-compatible names so the same cache
# directory can serve as a ``base_path`` for the existing GF worker loaders
# without a conversion step:
#
# * ``adjclose.parquet`` -- the close column under the worker's canonical field
#   name (``evaluation_close_field`` defaults to ``adjclose``);
# * ``volume.parquet``    -- the volume column, same long layout;
# * ``calendar.parquet``  -- the trading calendar ``PanelLoader.calendar`` reads,
#   needing only a ``trade_dt`` column.
#
# All three use the GF dump's long layout (``dt``/``symbol``/``value``, with
# ``calendar`` the documented exception), which is what makes the directory
# drop-in compatible rather than merely similar.
WORKER_CLOSE_NAME = "adjclose"
WORKER_VOLUME_NAME = "volume"
WORKER_CALENDAR_NAME = "calendar"

CACHE_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class CacheManifest:
    """What was pulled, when, and with how many rqdatac calls.

    ``api_calls`` is recorded rather than inferred: the acceptance criterion for
    the demo is a stated call count, and guessing it after the fact would be a
    number with no evidence behind it.
    """

    index_code: str
    symbols: tuple[str, ...]
    start: str
    end: str
    field: str
    api_calls: int
    rows: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "index_code": self.index_code,
            "symbols": list(self.symbols),
            "symbol_count": len(self.symbols),
            "start": self.start,
            "end": self.end,
            "field": self.field,
            "api_calls": self.api_calls,
            "rows": self.rows,
        }


def default_cache_root() -> Path:
    """The repository's ignore root, resolved from this file's location.

    This module sits at ``<repo>/rdagent/scenarios/gf_quant/strategy/search/``,
    so the repository root is five levels up, not four -- the four-level jump
    used elsewhere in the strategy package lands on ``rdagent/`` itself.  The
    resolved path is checked against a marker before it is used, so a future
    move of this file fails loudly here instead of quietly writing a cache into
    a directory that is not ignored.
    """

    root = Path(__file__).resolve().parents[5]
    if not (root / "rdagent").is_dir():
        raise RuntimeError(
            f"cannot locate the repository root from {__file__}: {root} has no rdagent/ directory"
        )
    return root / "git_ignore_folder" / "gf_quant" / "search_cache"


def _require_rqdatac() -> Any:
    try:
        import rqdatac
    except ImportError as exc:  # pragma: no cover - depends on the local licence
        raise RuntimeError(
            "rqdatac is required to pull the search cache; install and licence it, "
            "or point the search at an existing cache directory"
        ) from exc
    return rqdatac


def fetch_panels(
    *,
    index_code: str = DEFAULT_UNIVERSE_INDEX,
    symbol_count: int = DEFAULT_SYMBOL_COUNT,
    years: int = DEFAULT_YEARS,
    end: str | None = None,
    field: str = DEFAULT_FIELD,
) -> tuple[pd.DataFrame, CacheManifest]:
    """Pull one close/volume batch and return a long frame plus its manifest.

    Exactly two rqdatac calls are made regardless of the requested width: one
    index-components lookup to pick a stable universe and one
    ``get_price`` batch for every symbol at once.  Pulling per symbol would
    multiply the API count by the universe size for no extra information, and
    the quota is the binding constraint here.
    """

    rqdatac = _require_rqdatac()
    rqdatac.init()

    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.DateOffset(years=years)

    components = rqdatac.index_components(index_code, date=end_ts.strftime("%Y-%m-%d"))
    if not components:
        raise ValueError(f"index {index_code!r} has no components at {end_ts.date()}")
    symbols = tuple(sorted(components)[:symbol_count])
    if len(symbols) < 2:
        raise ValueError("at least two symbols are required to build a cross-section")

    frame = rqdatac.get_price(
        list(symbols),
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        frequency="1d",
        fields=list(_PRICE_FIELDS),
        adjust_type="pre",
        expect_df=True,
    )
    if frame is None or frame.empty:
        raise ValueError("rqdatac returned an empty price frame")
    # get_price returns a two-level (order_book_id, date) index; both names are
    # normalised explicitly so the dump does not depend on rqdatac's index
    # naming, which is a library detail rather than part of this contract.
    frame = frame.reset_index()
    frame = frame.rename(columns={"order_book_id": "symbol", "date": "dt"})
    if "dt" not in frame.columns:
        raise ValueError(f"rqdatac price frame has no date column; got {sorted(map(str, frame.columns))}")
    frame["dt"] = pd.to_datetime(frame["dt"]).dt.normalize()
    frame = frame.sort_values(["dt", "symbol"], kind="mergesort")
    # Every pulled field is kept and renamed to its DSL name in one step:
    # dropping or deferring a rename here is what turns a legal proposed
    # expression into a missing-file error at evaluation time.
    present = {name: _FIELD_ALIASES.get(name, name) for name in _PRICE_FIELDS if name in frame.columns}
    frame = frame.loc[:, ["dt", "symbol", *present]].rename(columns=present).reset_index(drop=True)
    if field not in frame.columns:
        raise ValueError(f"rqdatac did not return the requested field {field!r}; got {sorted(frame.columns)}")
    frame = frame.replace([float("inf"), float("-inf")], pd.NA)
    frame = frame.dropna(subset=[field])
    manifest = CacheManifest(
        index_code=index_code,
        symbols=symbols,
        start=frame["dt"].min().strftime("%Y-%m-%d"),
        end=frame["dt"].max().strftime("%Y-%m-%d"),
        field=field,
        api_calls=2,
        rows=int(len(frame)),
    )
    return frame, manifest


@dataclass(frozen=True)
class CachedManifest(CacheManifest):
    """A manifest that also reports where and how large its dump is.

    The size is measured from the file rather than estimated from the row count:
    the report states the on-disk size, and an estimate would be a different
    number wearing the same label.
    """

    dump_path: str = ""
    dump_bytes: int = 0

    def as_payload(self) -> dict[str, Any]:
        return {**super().as_payload(), "dump_path": self.dump_path, "dump_bytes": self.dump_bytes}


def update_cache(
    root: str | Path | None = None,
    *,
    index_code: str = DEFAULT_UNIVERSE_INDEX,
    symbol_count: int = DEFAULT_SYMBOL_COUNT,
    years: int = DEFAULT_YEARS,
    end: str | None = None,
    field: str = DEFAULT_FIELD,
    force: bool = False,
) -> CacheManifest:
    """Pull once and write the cache; no-op when a manifest already exists.

    ``force`` re-pulls.  Without it a second call is the cheap path the demo
    wants: the cache is the artifact being demonstrated, not the download.
    """

    target = Path(root) if root is not None else default_cache_root()
    manifest_path = target / CACHE_MANIFEST
    if manifest_path.exists() and not force:
        return _with_dump(target, load_manifest(target))

    frame, manifest = fetch_panels(
        index_code=index_code, symbol_count=symbol_count, years=years, end=end, field=field
    )
    target.mkdir(parents=True, exist_ok=True)
    # The search's own copy keeps just the requested field; the worker layout
    # carries every pulled field under the names the GF worker loaders read.
    _write_dump(target, field, frame.loc[:, ["dt", "symbol", field]])
    _write_worker_layout(target, frame)
    manifest_path.write_text(
        json.dumps(manifest.as_payload(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return _with_dump(target, manifest)


def _write_dump(root: Path, field: str, frame: pd.DataFrame) -> None:
    path = root / f"{field}.parquet"
    frame.to_parquet(path, engine="pyarrow", index=False)


def _write_worker_layout(root: Path, frame: pd.DataFrame) -> None:
    """Write the worker-compatible dump beside the search's own copy.

    The long frame carries ``dt``/``symbol`` plus one column per pulled field.
    Each field is emitted as its own ``(dt, symbol, value)`` file under the
    worker's field name, which is the layout ``PanelLoader._load_direct`` reads.
    The calendar is derived from the dates actually present rather than from a
    separate fetch: rqdatac's own trading calendar is exactly the set of dates
    the batch returned for these symbols.
    """

    columns = ["dt", "symbol"]
    for name in sorted(set(_FIELD_ALIASES.values())):
        if name not in frame.columns:
            continue
        long = frame.loc[:, [*columns, name]].rename(columns={name: "value"})
        long.to_parquet(root / f"{name}.parquet", engine="pyarrow", index=False)
    calendar = pd.DataFrame({"trade_dt": pd.DatetimeIndex(sorted(frame["dt"].unique()))})
    calendar.to_parquet(root / f"{WORKER_CALENDAR_NAME}.parquet", engine="pyarrow", index=False)


def _with_dump(root: Path, manifest: CacheManifest) -> CacheManifest:
    """Attach the dump's path and measured size to a manifest."""

    path = root / f"{manifest.field}.parquet"
    return CachedManifest(
        index_code=manifest.index_code,
        symbols=manifest.symbols,
        start=manifest.start,
        end=manifest.end,
        field=manifest.field,
        api_calls=manifest.api_calls,
        rows=manifest.rows,
        dump_path=str(path),
        dump_bytes=path.stat().st_size if path.exists() else 0,
    )


def load_manifest(root: str | Path | None = None) -> CacheManifest:
    target = Path(root) if root is not None else default_cache_root()
    payload = json.loads((target / CACHE_MANIFEST).read_text(encoding="utf-8"))
    return CacheManifest(
        index_code=str(payload["index_code"]),
        symbols=tuple(str(value) for value in payload["symbols"]),
        start=str(payload["start"]),
        end=str(payload["end"]),
        field=str(payload["field"]),
        api_calls=int(payload["api_calls"]),
        rows=int(payload["rows"]),
    )


class SearchDataCache:
    """Read side of the cache: the panels a backtest run needs.

    Constructed from a directory rather than pulling anything, so a run is
    reproducible from the cache alone and cannot silently hit the network.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_root()
        self.manifest = load_manifest(self.root)

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.manifest.symbols

    @property
    def field(self) -> str:
        return self.manifest.field

    def _dump(self) -> pd.DataFrame:
        """The close dump, keyed under the manifest's field name.

        The stored column is expected to be named after the field; the worker
        layout file of the same name is the same data in ``value`` form, so this
        reader accepts either rather than depending on which writer ran last.
        """

        path = self.root / f"{self.manifest.field}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"search cache dump is missing: {path}; re-pull with --refresh-data"
            )
        frame = pd.read_parquet(path, engine="pyarrow")
        if self.manifest.field not in frame.columns:
            if "value" in frame.columns:
                frame = frame.rename(columns={"value": self.manifest.field})
            else:
                raise ValueError(
                    f"cache dump {path} has no {self.manifest.field!r} column; "
                    f"got {sorted(map(str, frame.columns))}"
                )
        frame["dt"] = pd.to_datetime(frame["dt"]).dt.normalize()
        return frame

    def close_wide(self) -> pd.DataFrame:
        """``index=dt`` / ``columns=symbol`` close prices, wide and float."""

        frame = self._dump()
        wide = frame.pivot_table(index="dt", columns="symbol", values=self.manifest.field, aggfunc="last")
        return wide.sort_index().astype(float)

    def returns(self) -> pd.DataFrame:
        """Daily simple returns from the adjusted close panel.

        ``fill_method=None`` is deliberate: forward-filling a suspended name
        before differencing would turn a real gap into a fabricated flat day,
        which is exactly the kind of invented data the engine refuses to make
        up elsewhere.
        """

        return self.close_wide().pct_change(fill_method=None).iloc[1:]

    def trading_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close_wide().index)

    def summary(self) -> dict[str, Any]:
        return {
            **self.manifest.as_payload(),
            "root": str(self.root),
        }


def synthetic_cache(
    root: str | Path,
    *,
    symbols: Sequence[str],
    dates: pd.DatetimeIndex,
    seed: int = 7,
    drift: float = 0.0004,
    vol: float = 0.015,
) -> SearchDataCache:
    """Write a deterministic pseudo-market cache without rqdatac.

    Used by the tests and by ``--offline`` demos: a fixed seed makes the panel
    byte-reproducible, so a search run can be replayed without a licence or a
    network.  It is *not* a substitute for real data in a research conclusion --
    all it proves is that the loop closes.
    """

    import numpy as np

    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_dates, n_symbols = len(dates), len(symbols)
    shocks = rng.normal(loc=drift, scale=vol, size=(n_dates, n_symbols))
    # One common market factor so the cross-section is not pure noise; without
    # it every candidate ranks names the same way and the search has nothing to
    # discriminate on.
    market = rng.normal(loc=0.0, scale=vol * 0.8, size=(n_dates, 1))
    returns = shocks + market
    prices = 10.0 * (1.0 + pd.DataFrame(returns, index=dates, columns=list(symbols))).cumprod()
    # Volume is synthetic but plausibly scaled and, crucially, correlated with
    # the absolute return: a pure-noise volume would make every volume-based
    # expression score exactly zero and tell a test nothing about the chain.
    base_volume = 1.0e6 * (1.0 + np.abs(returns) * 10.0) * rng.lognormal(0.0, 0.3, size=(n_dates, n_symbols))
    volume = pd.DataFrame(base_volume, index=dates, columns=list(symbols))
    amount = volume * prices
    turnover = volume / volume.rolling(20, min_periods=1).mean()

    index = pd.DataFrame({
        "dt": prices.index.repeat(n_symbols),
        "symbol": list(symbols) * n_dates,
    })
    frame = index.assign(
        adjclose=prices.to_numpy().reshape(-1),
        volume=volume.to_numpy().reshape(-1),
        amount=amount.to_numpy().reshape(-1),
        turnover=turnover.to_numpy().reshape(-1),
    ).sort_values(["dt", "symbol"], kind="mergesort").reset_index(drop=True)

    _write_dump(target, DEFAULT_FIELD, frame.loc[:, ["dt", "symbol", DEFAULT_FIELD]])
    # The worker layout is written too: the local RankIC evaluator reads
    # ``calendar.parquet`` and ``adjclose.parquet``, so a synthetic cache without
    # them could not exercise the factor chain at all.
    _write_worker_layout(target, frame)
    manifest = CacheManifest(
        index_code="synthetic",
        symbols=tuple(symbols),
        start=dates.min().strftime("%Y-%m-%d"),
        end=dates.max().strftime("%Y-%m-%d"),
        field=DEFAULT_FIELD,
        api_calls=0,
        rows=int(len(frame)),
    )
    (target / CACHE_MANIFEST).write_text(
        json.dumps(manifest.as_payload(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return SearchDataCache(target)
