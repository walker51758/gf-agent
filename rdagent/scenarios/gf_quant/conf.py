"""
GF (GuangFa Securities) Quant extension settings.

All settings live under the ``GF_`` environment prefix so the extension never
collides with the Qlib factor/model scenario variables.
"""

from typing import Literal

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class GFScenarioSettings(ExtendedBaseSettings):
    """Settings that parameterize the GF Quant extension.

    Two concerns are grouped here:

    * **GF source data** - where the factor source data comes from. The remote
      GF data server is reachable over SSH (``data_host`` + ``data_ssh_*``).
      When no remote is configured the loader falls back to a local raw file
      (``data_local_raw_path``) which is convenient for offline development and
      for the primary agent to iterate on the discovered schema without a live
      connection.
    * **Local GF data folders** - the folders holding the built ``daily_pv.h5``
      factor source data. They are the folders linked into every factor
      implementation workspace, mirroring
      ``FACTOR_COSTEER_data_folder`` / ``FACTOR_COSTEER_data_folder_debug``.
    """

    model_config = SettingsConfigDict(env_prefix="GF_", protected_namespaces=())

    # ---- GF source data dump (G:/base) ------------------------------------
    data_base_path: str = "G:/base"
    """Root directory of the GF Securities data dump.

    The primary agent confirmed the dump uses long-format tables:

    * **direct fields** - one table per price/volume field with columns
      ``(dt, symbol, update_time, value)`` (the field name is the file name,
      e.g. ``open`` / ``close`` / ``high`` / ``low`` / ``volume`` / ``factor``);
    * **financial tables** - fundamental data with columns
      ``(dt announcement date, factor, report, symbol, update_time, value)``.

    Data ends ``2026-04`` (see ``data_end_date``).
    """

    data_direct_field_files: dict[str, str] = {}
    """Optional per-field file-name overrides for the direct-field tables,
    keyed by canonical stem (e.g. ``{"open": "open_px"}``). When empty the
    loader probes for ``open`` / ``close`` / ``high`` / ``low`` / ``volume`` /
    ``factor`` (plus common aliases) among the files in ``data_base_path``."""

    data_financial_pattern: str = ""
    """Substring matched (case-insensitive) against file names under
    ``data_base_path`` to identify financial (fundamental) tables. Set to empty
    to disable building ``fundamental.h5``."""

    data_end_date: str = "2026-04"
    """Inclusive end of the GF data dump (``YYYY-MM``), as confirmed by the
    primary agent. Rows after the end of this month are dropped at build time
    so the factor source data never reaches past the verified boundary. Set to
    empty to disable clipping."""

    # ---- GF remote data source -------------------------------------------
    data_host: str = "8.138.237.193"
    """Remote GF data server host. Empty means "no SSH fetch, use local raw file"."""

    data_port: int = 7022
    """SSH port of the remote GF data server."""

    data_user: str = "sshuser"
    """SSH user of the remote GF data server."""

    data_ssh_code: str = ""
    """SSH password/token for the remote GF data server (alternative to an SSH key)."""

    data_ssh_key_path: str = ""
    """Path to an SSH private key for the remote GF data server."""

    data_remote_path: str = ""
    """Remote path (single file or directory) on the GF server holding the raw
    daily price/volume records. The schema of the remote file is described in
    :mod:`rdagent.scenarios.gf_quant.data.loader`."""

    # ---- local raw fallback ----------------------------------------------
    data_local_raw_path: str = ""
    """Local path to a raw GF data file (CSV / parquet / HDF5) used when no
    remote server is configured. It is also the landing path for SSH downloads."""

    # ---- local GF data folders -------------------------------------------
    data_folder: str = "git_ignore_folder/gf_factor_data"
    """Folder holding the full GF factor source data (``daily_pv.h5``)."""

    data_folder_debug: str = "git_ignore_folder/gf_factor_data_debug"
    """Folder holding the partial GF factor source data used for fast debugging."""

    # ---- data build options ----------------------------------------------
    enable_remote_fetch: bool = True
    """Whether to try fetching from the remote GF server when local data is missing."""

    build_debug_sample_n: int = 100
    """Number of instruments kept in the debug sample (the whole history is kept)."""

    # ---- LLM backend ------------------------------------------------------
    llm_backend: str = "rdagent.oai.backend.LiteLLMAPIBackend"
    """LLM backend class used by the GF agent (default: LiteLLM)."""

    llm_chat_model: str = "deepseek/deepseek-flash"
    """LiteLLM chat model used by the GF agent."""

    factor_proposal_temperature: float = 2.0
    """Chat temperature used for factor proposals only.

    The vendor's parameter-settings page states only the default (1.0) and
    per-task suggestions (up to 1.5), not the accepted range, so the bound was
    measured: ``temperature=2.0`` is accepted by ``deepseek/deepseek-flash``
    while 2.1 and 3.0 are rejected with "the valid range of temperature is
    [0, 2]".  2.0 is therefore the endpoint's upper bound rather than a chosen
    value.  It reaches the API
    through the tag-scoped ``chat_model_map`` entry wired in
    :func:`rdagent.app.gf_rd_loop.conf.configure_gf_llm_backend`; the strategy
    proposer keeps the global ``chat_temperature``.
    """

    deepseek_api_key_env: str = "GFAGENT_DEEPSEEK_API_KEY"
    """Env var holding the DeepSeek API key. It is wired into LiteLLM's
    ``DEEPSEEK_API_KEY`` at startup so no duplicate key is needed."""

    llm_base_url: str = "https://api.deepseek.com"
    """OpenAI-compatible DeepSeek API base URL."""

    data_password_env: str = "SSH_CODE"
    """Name of the environment variable holding the SSH password."""

    data_host_key_sha256: str = "SHA256:6MqioxKcyZXMd1dx+t+DJOgSWXIqJUtb0j4Zaay/3x0"
    """Pinned SHA-256 host-key fingerprint for the remote evaluation server."""

    remote_python: str = "G:/.venv/Scripts/python.exe"
    """Python executable used by the remote evaluator."""

    remote_work_dir: str = "G:/gf_quant_agent"
    """Remote directory used for evaluator source, requests, and results."""

    remote_artifact_dir: str = "G:/gf_quant_agent/artifacts"
    """Remote directory where accepted candidate panels are written."""

    local_artifact_dir: str = "artifacts/live"
    """Local artifact directory, used only with ``GF_FACTOR_EVALUATION_MODE=local``."""

    factor_evaluation_mode: Literal["remote", "local"] = "remote"
    """Evaluation backend. Remote is the production default; local is explicit development mode."""

    evaluation_start: str = "2020-01-01"
    evaluation_end: str = "2023-12-31"
    evaluation_lookback_days: int = 1200
    evaluation_close_field: str = "adjclose"
    evaluation_label_horizon: int = 5
    evaluation_quantile: float = 0.2
    evaluation_min_cross_section: int = 100
    evaluation_sample_step: int = 5

    registry_path: str = "git_ignore_folder/gf_quant/factor_registry.sqlite"
    """Local SQLite metadata registry. Factor arrays stay in remote artifacts."""

    strategy_registry_path: str = "git_ignore_folder/gf_quant/strategy_registry.sqlite"
    """Local SQLite metadata registry for portfolio strategies. Kept separate from
    ``registry_path`` so a strategy record can never overwrite a factor record;
    the two are linked only through metadata references."""

    dedup_threshold: float = 0.88
    dedup_min_overlap: int = 100

    # ---- TEMPORARY annual screening gate ----------------------------------
    # The screening window is now four calendar years (2020-2023), so a
    # candidate must also hold up inside individual years instead of only in
    # aggregate.  These knobs only apply while a family has *no* calibrated
    # benchmark standards (development fallback); the calibrated standards
    # replace them as soon as ``benchmark_standards_path`` exists.
    annual_gate_min_years_with_rank_ic: int = 3
    """TEMPORARY fallback: require at least this many calendar years with a
    finite Rank_IC in ``metrics_by_year`` before a candidate can be accepted."""

    annual_gate_min_matching_sign_years: int = 2
    """TEMPORARY fallback: require at least this many yearly Rank_IC values to
    share the same nonzero sign as the aggregate Rank_IC."""

    # ---- benchmark calibration --------------------------------------------
    benchmark_standards_path: str = "configs/gf_quant/benchmark_standards.json"
    """Machine-readable calibrated standards produced by
    ``scripts/gf_quant_benchmark_controller.py``.  Relative paths resolve
    against the repository root, not the current working directory."""

    benchmark_signature_dir: str = "git_ignore_folder/gf_benchmark/signatures"
    """Directory holding the compact benchmark profiles used for family-aware
    candidate redundancy checks (no benchmark panel is ever downloaded)."""

    benchmark_calibration_required: bool = False
    """Fail closed when True: acceptance raises if the calibrated standards file
    is missing or invalid instead of using the development fallback."""

    benchmark_profile_required: bool = False
    """Fail closed when True: a candidate without a compact daily profile fails
    the direction-adjusted return/Sharpe/drawdown gates.  When False those gates
    are skipped and the skip is recorded in the registry metadata."""

    benchmark_cluster_threshold: float = 0.88
    """Absolute-profile correlation threshold used to cluster benchmark factors
    inside a family during calibration."""

    benchmark_cluster_min_overlap: int = 100
    """Minimum overlapping trading days for two benchmark profiles to be
    correlated during calibration."""


GF_SCENARIO_SETTINGS = GFScenarioSettings()

FACTOR_PROPOSAL_LOGGER_TAG = "gf_factor_proposal"
"""Logger tag the factor proposer runs under.

The tag is the only per-call-site signal the LLM backend exposes, so it is what
scopes the factor proposal temperature: ``LiteLLMAPIBackend.get_complete_kwargs``
matches the tag as a substring and applies the matching ``chat_model_map`` entry
to that call alone.  It is defined here, next to the setting it carries, and the
proposer and the backend wiring both import it rather than repeating the string.
"""

