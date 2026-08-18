"""
GF (GuangFa Securities) Quant extension settings.

All settings live under the ``GF_`` environment prefix so the extension never
collides with the Qlib factor/model scenario variables.
"""

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

    llm_chat_model: str = "deepseek/deepseek-v4-flash"
    """LiteLLM chat model used by the GF agent."""

    deepseek_api_key_env: str = "GFAGENT_DEEPSEEK_API_KEY"
    """Env var holding the DeepSeek API key. It is wired into LiteLLM's
    ``DEEPSEEK_API_KEY`` at startup so no duplicate key is needed."""

    llm_base_url: str = "https://api.deepseek.com"
    """OpenAI-compatible DeepSeek API base URL."""

    data_password_env: str = "SSH_CODE"
    """Name of the environment variable holding the SSH password."""

    registry_path: str = "git_ignore_folder/gf_quant/factor_registry.sqlite"
    """Local SQLite metadata registry. Factor arrays stay in remote artifacts."""

    dedup_threshold: float = 0.88
    dedup_min_overlap: int = 100


GF_SCENARIO_SETTINGS = GFScenarioSettings()
