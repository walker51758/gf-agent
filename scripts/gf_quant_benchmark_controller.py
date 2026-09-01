"""Local controller for the GF benchmark calibration run.

The controller drives :mod:`scripts.gf_quant_benchmark_worker` over the pinned
SSH host, downloads *only* the compact per-factor JSON summaries, and calibrates
family-specific acceptance standards from them:

* ``price_volume`` is calibrated from the merged ``artifacts`` + ``alpha158``
  pools;
* ``fundamental`` is calibrated from the ``basic_fundamental`` pool;
* correlation/redundancy calibration happens inside each family pool only.

Outputs:

* ``configs/gf_quant/benchmark_standards.json`` - machine-readable standards;
* ``docs/gf_benchmark_calibration.md`` - auditable human-readable report;
* ``git_ignore_folder/gf_benchmark/raw/...`` - downloaded compact summaries;
* ``git_ignore_folder/gf_benchmark/signatures/...`` - compact profiles used for
  family-aware candidate redundancy checks.

Credentials are read from ``.env`` / the ``GF_DATA_*`` contract and never leave
the local process.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdagent.scenarios.gf_quant.benchmark import (  # noqa: E402
    SUPPORTED_FAMILIES,
    BenchmarkFactorResult,
    FamilyCalibration,
    calibrate_family,
    calibration_to_document,
)
from rdagent.scenarios.gf_quant.conf import GF_SCENARIO_SETTINGS  # noqa: E402
from rdagent.scenarios.gf_quant.remote import RemoteSettings, _connect, _safe_remote  # noqa: E402

DEFAULT_SOURCE_DIRS: dict[str, list[str]] = {
    "price_volume": ["G:/gf_quant_agent/artifacts", "G:/gf_quant_agent/alpha158"],
    "fundamental": ["G:/gf_quant_agent/basic_fundamental"],
}
SOURCE_FAMILY_OVERRIDES = {
    "artifacts": "price_volume",
    "alpha158": "price_volume",
    "basic_fundamental": "fundamental",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# remote execution
# ---------------------------------------------------------------------------


def run_remote_worker(
    settings: RemoteSettings,
    *,
    worker_path: Path,
    family_dirs: Sequence[tuple[str, str]],
    out_dir: str,
    base_path: str,
    close_field: str,
    start: str,
    end: str,
    extra: Sequence[str] = (),
) -> None:
    """Upload the worker (if needed) and run it remotely to completion."""
    work_dir = settings.work_dir.rstrip("/")
    remote_worker = _safe_remote(f"{work_dir}/gf_quant_benchmark_worker.py")
    remote_evaluator = _safe_remote(f"{work_dir}/gf_quant_remote_worker.py")
    transport = _connect(settings)
    try:
        sftp = transport.open_sftp_client()
        try:
            sftp.stat(work_dir)
        except OSError:
            sftp.mkdir(work_dir)
        sftp.put(str(worker_path), remote_worker)
        evaluator_path = worker_path.with_name("gf_quant_remote_worker.py")
        sftp.put(str(evaluator_path), remote_evaluator)

        bindings = [f"--family-dir={family}={directory}" for family, directory in family_dirs]
        command = " ".join(
            [
                shlex.quote(settings.python_path),
                shlex.quote(remote_worker),
                *bindings,
                "--out-dir",
                shlex.quote(out_dir),
                "--base-path",
                shlex.quote(base_path),
                "--close-field",
                shlex.quote(close_field),
                "--start",
                shlex.quote(start),
                "--end",
                shlex.quote(end),
                *extra,
            ]
        )
        channel = transport.open_session()
        channel.exec_command(command)
        stdout = channel.makefile("rb", -1)
        # Stream progress so a long calibration run stays observable.
        for line in iter(stdout.readline, b""):
            text = line.decode("utf-8", "replace").strip()
            if text:
                print(f"[remote] {text}", flush=True)
        stderr = channel.makefile_stderr("rb", -1).read().decode("utf-8", "replace")
        status = channel.recv_exit_status()
        if status != 0:
            raise RuntimeError(f"Remote benchmark worker exited {status}: {stderr[-4000:]}")
        sftp.close()
    finally:
        transport.close()


def download_summaries(
    settings: RemoteSettings,
    *,
    remote_out_dir: str,
    local_raw_dir: Path,
) -> list[Path]:
    """Download the compact per-factor summaries, skipping already-fetched ones."""
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    transport = _connect(settings)
    try:
        sftp = transport.open_sftp_client()

        def walk(remote_dir: str, local_dir: Path) -> None:
            local_dir.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(remote_dir):
                remote_path = f"{remote_dir}/{entry.filename}"
                local_path = local_dir / entry.filename
                if entry.st_mode and (entry.st_mode & 0o040000):  # directory
                    walk(remote_path, local_path)
                elif entry.filename.endswith(".json") and entry.filename != "_manifest.json":
                    if local_path.exists():
                        continue
                    with sftp.open(remote_path, "rb") as stream:
                        local_path.write_bytes(stream.read())
                    downloaded.append(local_path)

        walk(_safe_remote(remote_out_dir), local_raw_dir)
        sftp.close()
    finally:
        transport.close()
    return downloaded


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def load_results(raw_dir: Path, family: str, source_dirs: Sequence[str]) -> list[BenchmarkFactorResult]:
    """Load every downloaded summary belonging to ``family``."""
    results: list[BenchmarkFactorResult] = []
    for path in sorted(raw_dir.rglob("*.json")):
        if path.name.startswith("_"):
            continue  # worker checkpoint manifests are not factor summaries
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"[warn] skipping unreadable summary {path}: {exc}", file=sys.stderr)
            continue
        result = BenchmarkFactorResult.from_payload(payload, default_family=family)
        if result.family != family:
            continue
        results.append(result)
    if source_dirs:
        expected = {str(directory).replace("\\", "/").rstrip("/") for directory in source_dirs}
        missing = sorted(
            {result.name for result in results if result.source_dir.replace("\\", "/").rstrip("/") not in expected}
        )
        if missing:
            print(
                f"[warn] {len(missing)} {family} summaries come from unexpected source directories "
                f"(first: {missing[:3]})",
                file=sys.stderr,
            )
    return results


def calibrate(
    raw_dir: Path,
    *,
    source_dirs: Mapping[str, Sequence[str]],
    threshold: float,
    min_overlap: int,
    cluster_profile: str,
) -> dict[str, FamilyCalibration]:
    calibrations: dict[str, FamilyCalibration] = {}
    for family in SUPPORTED_FAMILIES:
        results = load_results(raw_dir / family, family, source_dirs.get(family, ()))
        calibrations[family] = calibrate_family(
            results,
            family=family,
            threshold=threshold,
            min_overlap=min_overlap,
            cluster_profile=cluster_profile,
        )
    return calibrations


def render_report(
    calibrations: Mapping[str, FamilyCalibration],
    *,
    generated_at: str,
    evaluation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Render the auditable Markdown report."""
    lines: list[str] = [
        "# GF 基准因子校准报告（benchmark calibration）",
        "",
        f"生成时间：`{generated_at}`（UTC）",
        "",
        "## 口径",
        "",
        "- 评估窗口：`{start}` 至 `{end}`；5 日前瞻 RankIC 每 5 日采样；多空组合使用 1 日信号滞后与现行的分位规则。".format(
            **evaluation
        ),
        "- 指标定义与候选因子完全一致：远端 worker `scripts/gf_quant_remote_worker.py` 的"
        "`daily_series` / `metrics_from_daily`。",
        "- 基准池按族划分：`price_volume` = artifacts ∪ alpha158，`fundamental` = basic_fundamental；"
        "相关性/冗余度校准只在族内进行，注册表全局去重仍作为第二道安全检查。",
        "- 每个因子先按透明的规则剔除（见下文），再在族内用「绝对相关 ≥ 阈值」的单链聚类分组；"
        "同一簇内的相似因子不会主导分布：每个因子的权重为 `1 / (簇内样本数 × 簇数)`，"
        "等价于每簇取一个代表样本。",
        "- 分位规则：质量下限 = 簇平衡 P50；覆盖率下限 = P25；风险/成本上限 = P75；"
        "年度整数规则 = 簇平衡中位数向下取整并封顶到可用年数。",
        "- 方向处理：候选因子与基准因子的符号是任意的，因此所有带符号的比较都改用方向不变量："
        "`abs(RankIC)`、`abs(RankICIR)`、方向调整后的年化收益/Sharpe（按各自聚合 RankIC 的符号调整）、"
        "以及「按自身预测方向调整后的序列的回撤幅度」。不会跨因子比较原始带符号收益。",
        "",
    ]
    for family, calibration in calibrations.items():
        standards = calibration.standards
        lines += [
            f"## 族：`{family}`",
            "",
            f"- 样本：输入 {calibration.sample['total_inputs']} 个，剔除 {calibration.sample['excluded']} 个，"
            f"有效 {calibration.sample['valid']} 个，聚类 {calibration.sample['cluster_count']} 簇"
            f"（代表样本 {calibration.sample['representative_count']} 个）。",
            f"- 聚类：`{calibration.cluster_profile}` 日度曲线的绝对 Pearson 相关 ≥ "
            f"`{calibration.correlation_threshold}`，重叠 ≥ {calibration.min_overlap} 个交易日。",
            f"- 数据来源：{', '.join(f'`{item}`' for item in calibration.sample['source_dirs'])}",
            "",
            "| 标准 | 数值 | 规则 |",
            "| --- | --- | --- |",
            f"| 最小 |RankIC| | `{standards.min_abs_rank_ic:.6f}` | 簇平衡 P{int(calibration.quantile_policy.quality_floor * 100)} |",
            (
                f"| 最小 |RankICIR| | `{standards.min_abs_rank_icir:.6f}` | 簇平衡 P"
                f"{int(calibration.quantile_policy.quality_floor * 100)}（样本 "
                f"{calibration.sample['standards_sample_sizes']['abs_rank_icir']}） |"
                if standards.min_abs_rank_icir is not None
                else "| 最小 |RankICIR| | 不启用 | 有限 RankICIR 的样本不足，未校准（显式跳过，不虚构） |"
            ),
            f"| 最小覆盖率 | `{standards.min_coverage:.6f}` | 簇平衡 P{int(calibration.quantile_policy.coverage_floor * 100)} |",
            f"| 最大换手率 | `{standards.max_turnover:.6f}` | 簇平衡 P{int(calibration.quantile_policy.risk_ceiling * 100)} |",
            f"| 最大回撤幅度 | `{standards.max_abs_drawdown:.6f}` | 簇平衡 P{int(calibration.quantile_policy.risk_ceiling * 100)} |",
            f"| 最少观测数 | `{standards.min_observations}` | 簇平衡 P{int(calibration.quantile_policy.coverage_floor * 100)} |",
            f"| 年度最少有效年数 | `{standards.annual_min_valid_years}` | 簇平衡中位数向下取整，封顶 "
            f"{max(calibration.distribution_values('valid_years') or (0,))} 年 |",
            f"| 年度最少同号年数 | `{standards.annual_min_matching_sign_years}` | 簇平衡中位数向下取整 |",
            (
                f"| 方向调整后最小年化收益 | `{standards.min_direction_adjusted_annualized_return:.6f}` | 簇平衡 P"
                f"{int(calibration.quantile_policy.quality_floor * 100)} |"
                if standards.min_direction_adjusted_annualized_return is not None
                else "| 方向调整后最小年化收益 | 不启用 | 可用样本不足，未校准 |"
            ),
            (
                f"| 方向调整后最小 Sharpe | `{standards.min_direction_adjusted_sharpe:.6f}` | 簇平衡 P"
                f"{int(calibration.quantile_policy.quality_floor * 100)} |"
                if standards.min_direction_adjusted_sharpe is not None
                else "| 方向调整后最小 Sharpe | 不启用 | 可用样本不足，未校准 |"
            ),
            "",
            "### 剔除明细",
            "",
        ]
        if calibration.exclusions:
            lines += ["| 因子 | 原因 | 说明 |", "| --- | --- | --- |"]
            lines += [
                f"| `{item.name}` | `{item.reason}` | {item.detail} |" for item in calibration.exclusions
            ]
        else:
            lines.append("- 无剔除。")
        lines += ["", "### 代表样本分布（用于百分位）", ""]
        for distribution in calibration.distributions:
            values = list(distribution.values)
            if not values:
                lines.append(f"- `{distribution.metric}`：无可用样本。")
                continue
            lines.append(
                f"- `{distribution.metric}`（{distribution.role}，P{int(distribution.quantile * 100)}）：n={len(values)}, "
                f"min={values[0]:.6g}, P25={values[int(0.25 * (len(values) - 1))]:.6g}, "
                f"P50={values[len(values) // 2]:.6g}, max={values[-1]:.6g}"
            )
        lines.append("")
    lines += [
        "## 产物",
        "",
        f"- 机读标准：`{provenance.get('standards_path', 'configs/gf_quant/benchmark_standards.json')}`",
        f"- 紧凑摘要（本地）：`{provenance.get('raw_dir', 'git_ignore_folder/gf_benchmark/raw')}`",
        f"- 紧凑曲线签名（本地）：`{provenance.get('signature_dir', 'git_ignore_folder/gf_benchmark/signatures')}`",
        f"- 远端摘要：`{provenance.get('remote_out_dir', 'G:/gf_quant_agent/benchmark')}`",
        "",
    ]
    return "\n".join(lines)


def write_signatures(
    calibrations: Mapping[str, FamilyCalibration],
    raw_dir: Path,
    signature_dir: Path,
) -> None:
    """Store compact daily profiles per family for candidate redundancy checks."""
    signature_dir.mkdir(parents=True, exist_ok=True)
    for family, calibration in calibrations.items():
        payload = {
            "family": family,
            "cluster_profile": calibration.cluster_profile,
            "correlation_threshold": calibration.correlation_threshold,
            "min_overlap": calibration.min_overlap,
            "factors": {},
        }
        for path in sorted((raw_dir / family).rglob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if record.get("status") != "ok":
                continue
            profile = record.get("profile") or {}
            if not profile.get("dates"):
                continue
            payload["factors"][str(record.get("name") or path.stem)] = {
                "source_dir": record.get("source_dir"),
                "dates": profile["dates"],
                "spread": profile.get("spread") or [],
                "rank_ic": profile.get("rank_ic") or [],
                "rank_ic_mean": record.get("metrics", {}).get("Rank_IC"),
            }
        (signature_dir / f"{family}_signatures.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )


def parse_family_dirs(values: Sequence[str]) -> dict[str, list[str]]:
    source_dirs: dict[str, list[str]] = {family: [] for family in SUPPORTED_FAMILIES}
    for value in values:
        family, _, directory = value.partition("=")
        if not family or not directory:
            raise SystemExit(f"--family-dir must be FAMILY=DIR, got {value!r}")
        if family not in SUPPORTED_FAMILIES:
            raise SystemExit(f"Unsupported family: {family}")
        if directory not in source_dirs[family]:
            source_dirs[family].append(directory)
    return source_dirs


def main(argv: Sequence[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:  # pragma: no cover - dotenv is a hard dependency of the loop
        pass
    parser = argparse.ArgumentParser(description="Calibrate GF family benchmark standards from the remote host")
    parser.add_argument(
        "--family-dir",
        action="append",
        default=None,
        metavar="FAMILY=DIR",
        help="Benchmark source directory (repeatable). Defaults to the confirmed three directories.",
    )
    parser.add_argument("--remote-out-dir", default="G:/gf_quant_agent/benchmark")
    parser.add_argument("--raw-dir", type=Path, default=Path("git_ignore_folder/gf_benchmark/raw"))
    parser.add_argument("--signature-dir", type=Path, default=Path("git_ignore_folder/gf_benchmark/signatures"))
    parser.add_argument("--standards-path", type=Path, default=Path("configs/gf_quant/benchmark_standards.json"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/gf_benchmark_calibration.md"))
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("gf_quant_benchmark_worker.py"))
    parser.add_argument("--correlation-threshold", type=float, default=0.88)
    parser.add_argument("--min-overlap", type=int, default=100)
    parser.add_argument("--cluster-profile", choices=("spread", "rank_ic"), default="spread")
    parser.add_argument("--skip-remote", action="store_true", help="Reuse downloaded summaries and skip the remote run")
    parser.add_argument("--extra-worker-args", nargs="*", default=(), help="Extra flags passed to the remote worker")
    args = parser.parse_args(argv)

    settings = RemoteSettings.from_env()
    source_dirs = (
        parse_family_dirs(args.family_dir)
        if args.family_dir
        else {family: list(DEFAULT_SOURCE_DIRS[family]) for family in SUPPORTED_FAMILIES}
    )

    if not args.skip_remote:
        family_dirs = [(family, directory) for family, directories in source_dirs.items() for directory in directories]
        run_remote_worker(
            settings,
            worker_path=args.worker,
            family_dirs=family_dirs,
            out_dir=args.remote_out_dir,
            base_path=GF_SCENARIO_SETTINGS.data_base_path,
            close_field=GF_SCENARIO_SETTINGS.evaluation_close_field,
            start=GF_SCENARIO_SETTINGS.evaluation_start,
            end=GF_SCENARIO_SETTINGS.evaluation_end,
            extra=list(args.extra_worker_args),
        )
    downloaded = download_summaries(settings, remote_out_dir=args.remote_out_dir, local_raw_dir=args.raw_dir)
    print(f"downloaded {len(downloaded)} new summaries into {args.raw_dir}")

    calibrations = calibrate(
        args.raw_dir,
        source_dirs=source_dirs,
        threshold=args.correlation_threshold,
        min_overlap=args.min_overlap,
        cluster_profile=args.cluster_profile,
    )

    generated_at = utc_now()
    evaluation = {
        "start": GF_SCENARIO_SETTINGS.evaluation_start,
        "end": GF_SCENARIO_SETTINGS.evaluation_end,
        "label_horizon": GF_SCENARIO_SETTINGS.evaluation_label_horizon,
        "quantile": GF_SCENARIO_SETTINGS.evaluation_quantile,
        "min_cross_section": GF_SCENARIO_SETTINGS.evaluation_min_cross_section,
        "sample_step": GF_SCENARIO_SETTINGS.evaluation_sample_step,
        "close_field": GF_SCENARIO_SETTINGS.evaluation_close_field,
    }
    document = calibration_to_document(
        calibrations,
        generated_at=generated_at,
        evaluation=evaluation,
        provenance={
            "remote_out_dir": args.remote_out_dir,
            "raw_dir": str(args.raw_dir).replace("\\", "/"),
            "signature_dir": str(args.signature_dir).replace("\\", "/"),
            "standards_path": str(args.standards_path).replace("\\", "/"),
            "worker": "scripts/gf_quant_benchmark_worker.py",
            "controller": "scripts/gf_quant_benchmark_controller.py",
            "source_dirs": {family: list(directories) for family, directories in source_dirs.items()},
        },
    )
    args.standards_path.parent.mkdir(parents=True, exist_ok=True)
    args.standards_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8"
    )
    write_signatures(calibrations, args.raw_dir, args.signature_dir)
    report = render_report(
        calibrations,
        generated_at=generated_at,
        evaluation=evaluation,
        provenance={
            "standards_path": str(args.standards_path).replace("\\", "/"),
            "raw_dir": str(args.raw_dir).replace("\\", "/"),
            "signature_dir": str(args.signature_dir).replace("\\", "/"),
            "remote_out_dir": args.remote_out_dir,
        },
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")

    for family, calibration in calibrations.items():
        standards = calibration.standards
        print(
            f"{family}: valid={calibration.sample['valid']} clusters={calibration.sample['cluster_count']} "
            f"min|RankIC|={standards.min_abs_rank_ic:.6f} min|RankICIR|={standards.min_abs_rank_icir} "
            f"minCoverage={standards.min_coverage:.6f} maxTurnover={standards.max_turnover:.6f} "
            f"maxAbsDrawdown={standards.max_abs_drawdown:.6f} annualYears={standards.annual_min_valid_years} "
            f"annualSignYears={standards.annual_min_matching_sign_years}"
        )
    print(f"standards: {args.standards_path}")
    print(f"report: {args.report_path}")


if __name__ == "__main__":
    main()
