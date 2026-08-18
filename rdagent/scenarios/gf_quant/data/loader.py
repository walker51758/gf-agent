"""
GF (GuangFa Securities) data loader.

Builds the standard factor source data consumed by RD-Agent's factor
implementation stack from the GF Securities data dump.

The primary agent confirmed the dump (mounted as ``G:/base``) uses two
long-format table schemas:

1. **Direct field tables** - one table per price/volume field with columns
   ``(dt, symbol, update_time, value)``. The field name comes from the table
   (file) name. The adapter uses ``adjopen`` / ``adjclose`` / ``adjhigh`` /
   ``adjlow`` plus ``volume`` and emits ``$factor=1`` because prices are already
   adjusted.
2. **Financial tables** - fundamental data with columns
   ``(dt announcement date, factor, report, symbol, update_time, value)``,
   where ``factor`` is the financial item name (e.g. ``total_assets``) and
   ``report`` is the reporting period (e.g. ``2025Q4``). They are loaded into a
   point-in-time ``fundamental.h5`` panel available for future fundamental
   factors.

The dump ends ``2026-04`` (``GF_DATA_END_DATE`` clips rows past that month).

The daily price/volume output is intentionally identical to
``rdagent/scenarios/qlib/experiment/factor_data_template`` so the whole Qlib
factor implementation stack works unchanged:

    ``pandas.DataFrame`` with a ``MultiIndex`` of ``["datetime", "instrument"]``
    and columns ``$open $close $high $low $volume $factor``.

Two fallback sources are still supported for single wide raw files (offline /
SSH): a local raw CSV / parquet / HDF5 file (``GF_DATA_LOCAL_RAW_PATH``) or a
fetch from the remote GF data server over SSH (``GF_DATA_HOST``). Those go
through :func:`normalize_gf_raw`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.log import rdagent_logger as logger

try:  # `paramiko` is an optional dependency of the GF extension
    import paramiko  # noqa: F401

    _PARAMIKO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _PARAMIKO_AVAILABLE = False

# The canonical columns a generated ``daily_pv.h5`` must expose.
DAILY_PV_COLUMNS = ["$open", "$close", "$high", "$low", "$volume", "$factor"]

# Canonical daily-pv column -> primary stem used to probe the G:/base dump.
_DIRECT_FIELD_NAMES = {
    "$open": "adjopen",
    "$close": "adjclose",
    "$high": "adjhigh",
    "$low": "adjlow",
    "$volume": "volume",
}

# Canonical stem -> alternative file stems probed when the primary one is absent.
_DIRECT_FIELD_ALIASES = {
    "open": ["open_price", "open_px"],
    "close": ["close_price", "close_px"],
    "high": ["high_price", "high_px"],
    "low": ["low_price", "low_px"],
    "volume": ["vol", "volume_total", "volume_amt"],
}

_TABLE_SUFFIXES = (".csv", ".parquet", ".h5", ".hdf")

GF_README = """# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename         | Description                                                          |
| ---------------- | ---------------------------------------------------------------------|
| "daily_pv.h5"    | Adjusted daily price and volume data (GF Securities A-share data).    |
| "fundamental.h5" | Point-in-time fundamental factors built from the G:/base financial    |
|                  | tables (may be absent when no financial table is found).              |

# For different data, We have some basic knowledge for them

## Daily price and volume data
$open: open price of the stock on that day.
$close: close price of the stock on that day.
$high: high price of the stock on that day.
$low: low price of the stock on that day.
$volume: volume of the stock on that day.
$factor: factor value of the stock on that day.

## Fundamental factors
Each column is one fundamental factor (e.g. total_assets, net_profit).
The value of an instrument on a date is the most recent announcement on or
before that date (point-in-time, no look-ahead). The announcement date is the
official `dt` from the G:/base financial tables.
"""


def _normalize_column_name(name: str) -> str:
    """Map a raw GF column name to a canonical lowercase one."""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_columns(cols) -> dict:
    return {col: _normalize_column_name(col) for col in cols}


# Alias map used by the wide-format fallback path (:func:`normalize_gf_raw`).
_COLUMN_ALIASES = {
    "datetime": "datetime",
    "date": "datetime",
    "trade_date": "datetime",
    "time": "datetime",
    "instrument": "instrument",
    "code": "instrument",
    "stock_code": "instrument",
    "symbol": "instrument",
    "secucode": "instrument",
    "open": "$open",
    "open_price": "$open",
    "close": "$close",
    "close_price": "$close",
    "high": "$high",
    "high_price": "$high",
    "low": "$low",
    "low_price": "$low",
    "volume": "$volume",
    "vol": "$volume",
    "factor": "$factor",
    "adjust_factor": "$factor",
    "adj_factor": "$factor",
}


def normalize_direct_field_long(raw_df: pd.DataFrame, field: str, source_name: str = "") -> pd.Series:
    """Normalize one long-format direct-field table.

    Confirmed G:/base schema: ``(dt, symbol, update_time, value)``. Each table
    holds a single price/volume field, identified by its file name (the
    ``field`` argument). ``update_time`` is only used to keep the latest update
    when several rows exist for the same ``(dt, symbol)``.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        The raw long table.
    field : str
        The canonical column name this table feeds (e.g. ``$open``).
    source_name : str
        Optional human-readable source (e.g. file name) for error messages.

    Returns
    -------
    pandas.Series
        Named ``field``, indexed by ``["datetime", "instrument"]``.
    """
    df = raw_df.rename(columns=_canonical_columns(raw_df.columns))
    missing = [c for c in ("dt", "symbol", "value") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Direct-field table {source_name or field!r} is missing columns {missing}; "
            f"expected the long schema (dt, symbol, update_time, value). "
            f"Found: {list(raw_df.columns)}"
        )

    df["dt"] = pd.to_datetime(df["dt"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "update_time" in df.columns:
        df = df.sort_values("update_time")
    df = df.dropna(subset=["dt", "symbol", "value"]).drop_duplicates(subset=["dt", "symbol"], keep="last")

    series = df.set_index(["dt", "symbol"])["value"].rename(field)
    series.index.names = ["datetime", "instrument"]
    return series


def normalize_financial_long(raw_df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """Normalize one long-format financial (fundamental) table.

    Confirmed G:/base schema: ``(dt announcement date, factor, report, symbol,
    update_time, value)``. ``update_time`` is used to keep the latest update,
    and ``report`` is only used for dedup (the most recent announcement per
    ``(datetime, instrument, factor)`` wins).

    Returns
    -------
    pandas.DataFrame
        Long frame with columns ``[datetime, instrument, factor, value]``.
    """
    df = raw_df.rename(columns=_canonical_columns(raw_df.columns))
    missing = [c for c in ("dt", "factor", "symbol", "value") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Financial table {source_name!r} is missing columns {missing}; "
            f"expected (dt, factor, report, symbol, update_time, value). "
            f"Found: {list(raw_df.columns)}"
        )

    df["dt"] = pd.to_datetime(df["dt"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["factor"] = df["factor"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "update_time" in df.columns:
        df = df.sort_values("update_time")
    df = df.dropna(subset=["dt", "symbol", "factor", "value"]).drop_duplicates(
        subset=["dt", "symbol", "factor"], keep="last"
    )
    return df[["dt", "symbol", "factor", "value"]].rename(columns={"dt": "datetime", "symbol": "instrument"})


def normalize_gf_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw *wide* GF daily price/volume frame into the canonical layout.

    This is the fallback path for a single wide raw file (local or SSH
    download). The primary G:/base dump uses long-format tables instead - see
    :func:`normalize_direct_field_long`.

    Returns a frame indexed by ``["datetime", "instrument"]`` with columns
    ``["$open", "$close", "$high", "$low", "$volume", "$factor"]``.
    """
    rename = {}
    for col in raw_df.columns:
        norm = _normalize_column_name(col)
        if norm in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[norm]
    if "datetime" not in rename.values():
        raise ValueError(
            f"Raw GF data is missing a datetime/date/trade_date column. Found columns: {list(raw_df.columns)}"
        )
    if "instrument" not in rename.values():
        raise ValueError(
            f"Raw GF data is missing an instrument/code/stock_code column. Found columns: {list(raw_df.columns)}"
        )

    df = raw_df.rename(columns=rename)
    # Keep only the columns we know about; fill any missing price/volume fields
    # (e.g. a source that only provides OHLC without an adjustment factor).
    for col in DAILY_PV_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str).str.upper()

    df = (
        df[["datetime", "instrument", *DAILY_PV_COLUMNS]]
        .dropna(subset=["datetime", "instrument"])
        .drop_duplicates(subset=["datetime", "instrument"], keep="last")
        .set_index(["datetime", "instrument"])
        .sort_index()
    )
    return df


def _read_table(path: Path) -> pd.DataFrame:
    """Read a raw GF data table (CSV / parquet / HDF5)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".h5", ".hdf"):
        # try key "data" first, then fall back to the first key available
        with pd.HDFStore(path) as store:
            key = "data" if "data" in store.keys() else store.keys()[0]
        return pd.read_hdf(path, key=key)
    raise NotImplementedError(f"Unsupported raw GF data file type: {suffix} (from {path})")


# Backward-compatible alias for :func:`_read_table`.
_read_raw = _read_table


def _parse_end_month(end_date: str) -> pd.Timestamp | None:
    """Parse an inclusive ``YYYY-MM`` boundary into the end-of-month timestamp."""
    if not end_date:
        return None
    return pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)


def _clip_datetime(df: pd.DataFrame | pd.Series, end_date: str) -> pd.DataFrame | pd.Series:
    """Drop rows after the confirmed end of the GF data dump."""
    end = _parse_end_month(end_date)
    if end is None:
        return df
    return df[df.index.get_level_values("datetime") <= end]


def _find_direct_field_file(base_dir: Path, stem: str, overrides: dict) -> Path | None:
    """Locate the table file for one direct field inside the base dump."""
    candidates = [overrides.get(stem, stem)] + _DIRECT_FIELD_ALIASES.get(stem, [])
    for candidate in candidates:
        for suffix in _TABLE_SUFFIXES:
            path = base_dir / f"{candidate}{suffix}"
            if path.exists():
                return path
    # case-insensitive / extension-agnostic fallback
    for candidate in candidates:
        for path in base_dir.iterdir():
            if path.is_file() and path.suffix.lower() in _TABLE_SUFFIXES and path.stem.lower() == candidate:
                return path
    return None


def build_daily_pv_from_base(settings: Any) -> pd.DataFrame:
    """Assemble ``daily_pv`` from the direct-field tables of the G:/base dump.

    Each direct-field table (``(dt, symbol, update_time, value)``) is pivoted
    into its canonical column. Returns a
    frame indexed by ``["datetime", "instrument"]`` with the canonical
    ``$open $close $high $low $volume $factor`` columns, clipped to
    ``settings.data_end_date``.
    """
    base_dir = Path(settings.data_base_path)
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"GF data dump base path not found: {base_dir}. Set GF_DATA_BASE_PATH "
            "to the mounted G:/base directory."
        )

    overrides = dict(settings.data_direct_field_files or {})
    series: dict[str, pd.Series | None] = {}
    missing: list[str] = []
    for canonical, stem in _DIRECT_FIELD_NAMES.items():
        path = _find_direct_field_file(base_dir, stem, overrides)
        if path is None:
            missing.append(f"{stem} ({canonical})")
            series[canonical] = None
            continue
        series[canonical] = normalize_direct_field_long(_read_table(path), canonical, path.name)

    if missing:
        logger.warning(
            f"GF direct-field tables not found in {base_dir}: {', '.join(missing)}. "
            "Missing fields will be NaN."
        )

    index = None
    for s in series.values():
        if s is not None:
            index = s.index.union(index) if index is not None else s.index
    if index is None:
        raise RuntimeError(f"No direct-field tables found under {base_dir}.")

    df = pd.DataFrame(index=index, columns=DAILY_PV_COLUMNS, dtype=float)
    for canonical, s in series.items():
        if s is not None:
            df[canonical] = s.reindex(index).astype(float)
    df["$factor"] = 1.0

    df = _clip_datetime(df, settings.data_end_date)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def build_fundamental_from_base(settings: Any, trading_dates: pd.Index | None = None) -> pd.DataFrame | None:
    """Assemble a point-in-time fundamental panel from the G:/base financial tables.

    Every file under ``settings.data_base_path`` whose name contains
    ``settings.data_financial_pattern`` is normalized and concatenated into a
    long frame, then pivoted to ``(datetime, instrument) x factor``. When
    ``trading_dates`` is provided the panel is forward-filled across the full
    trading calendar so each date carries the most recent announcement
    (point-in-time, no look-ahead). Returns ``None`` when no financial table is
    found.
    """
    base_dir = Path(settings.data_base_path)
    if not base_dir.is_dir():
        return None

    pattern = (settings.data_financial_pattern or "").lower()
    tables: list[pd.DataFrame] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _TABLE_SUFFIXES:
            continue
        if pattern and pattern not in path.stem.lower():
            continue
        try:
            long_df = normalize_financial_long(_read_table(path), path.name)
        except ValueError:
            continue  # direct-field or unrelated table that happens to match
        if not long_df.empty:
            tables.append(long_df)

    if not tables:
        return None

    long = pd.concat(tables, ignore_index=True)
    long = long[long["datetime"] <= _parse_end_month(settings.data_end_date)]
    if long.empty:
        return None

    panel = long.pivot_table(index=["datetime", "instrument"], columns="factor", values="value", aggfunc="last")
    panel = panel.sort_index()

    if trading_dates is not None and len(trading_dates):
        instruments = panel.index.get_level_values("instrument").unique()
        n_cells = len(trading_dates) * len(instruments) * panel.shape[1]
        if n_cells > 50_000_000:
            logger.warning(
                f"GF fundamental panel would be {n_cells / 1e6:.0f}M cells; skipping the "
                "full trading-calendar expansion (announcement-grid ffill only)."
            )
        else:
            idx = pd.MultiIndex.from_product(
                [sorted(trading_dates), sorted(instruments)],
                names=["datetime", "instrument"],
            )
            panel = panel.reindex(idx)

    # forward-fill per instrument so each date carries the latest known value
    panel = panel.sort_index(level=["instrument", "datetime"]).groupby(level="instrument", sort=False).ffill()
    return panel.sort_index()


def write_gf_data_files(df: pd.DataFrame, settings: Any, fundamental: pd.DataFrame | None = None) -> None:
    """Write ``daily_pv.h5`` (+ ``README.md``, + ``fundamental.h5`` when given)
    into the configured full and debug GF data folders."""
    full_folder = Path(settings.data_folder)
    debug_folder = Path(settings.data_folder_debug)
    full_folder.mkdir(parents=True, exist_ok=True)
    debug_folder.mkdir(parents=True, exist_ok=True)

    df.to_hdf(full_folder / "daily_pv.h5", key="data", mode="w")
    if fundamental is not None:
        fundamental.to_hdf(full_folder / "fundamental.h5", key="data", mode="w")
    (full_folder / "README.md").write_text(GF_README, encoding="utf-8")

    # Debug sample: keep the full history of a small set of instruments so the
    # factor implementation can run in seconds.
    instruments = df.index.get_level_values("instrument").unique()
    sample_instruments = instruments[: settings.build_debug_sample_n]
    debug_df = df.loc[pd.IndexSlice[:, sample_instruments], :]
    debug_df.to_hdf(debug_folder / "daily_pv.h5", key="data", mode="w")
    if fundamental is not None:
        fundamental.loc[pd.IndexSlice[:, sample_instruments], :].to_hdf(
            debug_folder / "fundamental.h5", key="data", mode="w"
        )
    (debug_folder / "README.md").write_text(GF_README, encoding="utf-8")

    logger.info(
        f"GF factor source data written: full={full_folder / 'daily_pv.h5'} "
        f"({len(instruments)} instruments), debug={debug_folder / 'daily_pv.h5'} "
        f"({len(sample_instruments)} instruments)."
    )


def build_daily_pv_files(raw_df: pd.DataFrame, settings: Any) -> None:
    """Backward-compatible entry: normalize a *wide* raw GF frame and write the folders."""
    write_gf_data_files(normalize_gf_raw(raw_df), settings, fundamental=None)


def build_gf_data_files(settings: Any) -> bool:
    """Build the GF factor source data files into the configured folders.

    Priority order:

    1. ``settings.data_base_path`` - the confirmed G:/base long-format dump
       (direct-field tables plus, when present, financial tables).
    2. ``settings.data_local_raw_path`` - a single wide raw file.
    3. remote SSH fetch (last resort).

    Returns ``True`` when the data was written.
    """
    base_dir = Path(settings.data_base_path)
    if base_dir.is_dir():
        df = build_daily_pv_from_base(settings)
        fundamental = build_fundamental_from_base(settings, trading_dates=df.index.get_level_values("datetime").unique())
        write_gf_data_files(df, settings, fundamental=fundamental)
        return True

    try:
        raw_path = resolve_gf_raw(settings)
    except Exception as e:  # noqa: BLE001 - surface a single actionable message
        logger.warning(f"GF factor source data is not ready (continue without data): {e}")
        return False

    write_gf_data_files(normalize_gf_raw(_read_table(raw_path)), settings, fundamental=None)
    return True


def fetch_remote_gf_raw(settings: Any) -> Path:
    """Download the raw GF daily data from the remote server into the local raw path.

    Uses ``paramiko``/SFTP. The exact remote layout is configurable through the
    ``GF_DATA_*`` settings. Returns the path of the downloaded local file.
    """
    if not settings.data_host:
        raise RuntimeError("No remote GF data host configured (GF_DATA_HOST is empty).")
    if not _PARAMIKO_AVAILABLE:
        raise RuntimeError(
            "`paramiko` is required to fetch GF data over SSH. Install it with "
            "`pip install paramiko` or set GF_DATA_LOCAL_RAW_PATH to a local raw file."
        )

    local_raw = Path(settings.data_local_raw_path) if settings.data_local_raw_path else None
    if local_raw is None:
        raise RuntimeError("GF_DATA_LOCAL_RAW_PATH must be set so remote data can be downloaded into it.")
    local_raw.parent.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if settings.data_ssh_key_path:
        client.connect(
            hostname=settings.data_host,
            port=settings.data_port,
            username=settings.data_user,
            key_filename=settings.data_ssh_key_path,
        )
    else:
        client.connect(
            hostname=settings.data_host,
            port=settings.data_port,
            username=settings.data_user,
            password=settings.data_ssh_code,
        )

    try:
        remote_path = settings.data_remote_path
        if not remote_path:
            raise RuntimeError("GF_DATA_REMOTE_PATH must be set to fetch data from the remote server.")
        sftp = client.open_sftp()
        try:
            sftp.get(remote_path, str(local_raw))
        finally:
            sftp.close()
    finally:
        client.close()

    logger.info(f"Downloaded GF raw data from {settings.data_host}:{settings.data_remote_path} to {local_raw}")
    return local_raw


def resolve_gf_raw(settings: Any) -> Path:
    """Resolve the raw GF data file (fallback path), fetching from the remote server if needed."""
    local_raw = Path(settings.data_local_raw_path) if settings.data_local_raw_path else None
    if local_raw is not None and local_raw.exists():
        return local_raw
    if settings.enable_remote_fetch and settings.data_host:
        return fetch_remote_gf_raw(settings)
    raise FileNotFoundError(
        "GF factor source data is not available. Configure either:\n"
        "  - GF_DATA_BASE_PATH=<G:/base dump> (preferred), or\n"
        "  - GF_DATA_LOCAL_RAW_PATH=<path to local GF raw csv/parquet/h5>, or\n"
        "  - GF_DATA_HOST / GF_DATA_USER / GF_DATA_REMOTE_PATH (plus credentials)\n"
        "and re-run. See rdagent/scenarios/gf_quant/data/loader.py for the expected schema."
    )


def ensure_gf_data(raise_on_missing: bool = False) -> bool:
    """
    Make the GF factor source data folders available to the RD-Agent stack.

    It is idempotent and performs three steps:

    1. Point ``FACTOR_COSTEER_SETTINGS.data_folder`` / ``data_folder_debug`` at
       the GF folders (the folders every factor implementation links in).
    2. If ``daily_pv.h5`` is already present, do nothing more.
    3. Otherwise build the data from the configured GF source (G:/base dump, a
       local raw file, or the remote server).

    Parameters
    ----------
    raise_on_missing : bool
        When True and no data source is available, raise a descriptive error.
        Set False in the scenario constructors so a scenario can still be built
        for tooling/CI; set True in the app entry points to fail the run fast.

    Returns
    -------
    bool
        True if the GF data folders are ready.
    """
    from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS as settings

    # 1) point the generic factor-coder data folders at the GF folders and make
    # sure they exist. Creating them even when data is missing keeps the generic
    # `get_data_folder_intro()` from falling back to the *Qlib* data generation.
    full_folder = Path(settings.data_folder)
    debug_folder = Path(settings.data_folder_debug)
    full_folder.mkdir(parents=True, exist_ok=True)
    debug_folder.mkdir(parents=True, exist_ok=True)

    FACTOR_COSTEER_SETTINGS.data_folder = str(full_folder.resolve())
    FACTOR_COSTEER_SETTINGS.data_folder_debug = str(debug_folder.resolve())

    full_pv = full_folder / "daily_pv.h5"
    if full_pv.exists():
        return True

    try:
        built = build_gf_data_files(settings)
    except Exception as e:  # noqa: BLE001 - surface a single actionable message
        if raise_on_missing:
            raise RuntimeError(f"GF factor source data is not ready: {e}") from e
        logger.warning(f"GF factor source data is not ready (continue without data): {e}")
        return False

    if not built and raise_on_missing:
        raise RuntimeError("GF factor source data is not ready: no source was found.")
    return built


def copy_gf_data_from_local_folder(src_folder: Path, settings: Any | None = None) -> None:
    """
    Copy a pre-built GF data folder (holding ``daily_pv.h5`` + ``README.md``)
    into the configured GF data folders.

    This is the fastest way for the primary agent to share a verified data dump
    without going through the raw-schema normalization path.
    """
    if settings is None:  # delayed import to avoid circularity at module load
        from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS as settings

    for sub in ("daily_pv.h5", "README.md"):
        src = Path(src_folder) / sub
        if not src.exists():
            raise FileNotFoundError(f"{src} not found in the GF data folder to copy.")
        for dst in (Path(settings.data_folder), Path(settings.data_folder_debug)):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst / sub)

    # Re-point the generic folders and finish
    ensure_gf_data(raise_on_missing=True)
    logger.info(f"Copied GF data from {src_folder} to {settings.data_folder} and {settings.data_folder_debug}.")
