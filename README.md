# 广发 A 股因子智能体

这是从 Microsoft RD-Agent 精简出的专用因子研究项目，只保留与广发 A 股数据、因子生成、远程评估、去重和登记直接相关的代码。

## 核心流程

1. DeepSeek 驱动 RD-Agent 因子研究循环生成候选因子。
2. 本地控制器通过固定主机指纹的 SSH 连接数据机。
3. 远端 worker 按需读取 `G:/base` 的 Parquet 列，并按公告日处理财务数据；迭代筛选窗口固定默认 `2020-01-01` 至 `2023-12-31`。
4. 一次返回四年总体指标及 2020–2023 逐年的 IC、RankIC、收益、Sharpe、回撤、换手率和覆盖率。
5. 本地执行多指标门槛、相关性去重，并写入 SQLite 因子登记表。

目前年度稳定性门槛是临时口径：至少 3 个年度有有限 RankIC，且
至少 2 个年度的 RankIC 与四年总体 RankIC 同号。可通过
`GF_ANNUAL_GATE_MIN_YEARS_WITH_RANK_IC` 和
`GF_ANNUAL_GATE_MIN_MATCHING_SIGN_YEARS` 调整，仅在某个家族**没有**已校准
基准标准（开发兜底）时生效。

## 基准校准（正式筛选标准）

接受门槛不再是拍脑袋常数，而是从基准因子池的实证分布校准出来的：

1. 远端基准 worker（`scripts/gf_quant_benchmark_worker.py`）逐个读取既有宽表
   基准面板（`artifacts`、`alpha158`、`basic_fundamental`），复用候选因子完全
   相同的指标口径，产出**紧凑的逐因子摘要**（总体 + 逐年指标 + 每日多空收益
   与 RankIC 轮廓）；因子面板本身永不下传，支持断点续跑，且不删除远端文件。
2. 本地控制器（`scripts/gf_quant_benchmark_controller.py`）只下载这些紧凑摘要，
   按**家族**分别校准：`price_volume` 用 artifacts + alpha158，
   `fundamental` 用 basic_fundamental。家族内按每日轮廓的**绝对相关**
   （默认阈值 0.88）聚类，用逆簇权重（等价于每簇一个代表样本）计算分位数，
   得到家族各自的 |RankIC| / |RankICIR| / 覆盖率 / 换手率 / 回撤 /
   观测数 / 年度稳定性的数值标准。
3. `FactorAcceptanceService` 按 `candidate.family` 选用对应家族的标准打分，
   并把基准百分位与未通过项写入登记表元数据；同时用同家族基准轮廓做相关性
   冗余检查（全局登记表去重仍是独立的一道兜底）。

机器可读标准在 `configs/gf_quant/benchmark_standards.json`，可审计的口径与
数值报告在 `docs/gf_benchmark_calibration.md`。相关开关：
`GF_BENCHMARK_STANDARDS_PATH`、`GF_BENCHMARK_SIGNATURE_DIR`、
`GF_BENCHMARK_CALIBRATION_REQUIRED`（为 true 时标准缺失或损坏会直接报错，
而不是退回开发兜底）、`GF_BENCHMARK_PROFILE_REQUIRED`、
`GF_BENCHMARK_CLUSTER_THRESHOLD`、`GF_BENCHMARK_CLUSTER_MIN_OVERLAP`。

重新校准（远端跑基准 worker + 本地汇总，可断点续跑）：

```powershell
python scripts/gf_quant_benchmark_controller.py
```

## 安装

```powershell
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

在 `.env` 中填写 DeepSeek 和远端数据机凭据。真实 `.env`、本地数据库、日志和大体量数据均被 Git 忽略。

## 运行

启动远程优先的因子研究循环（远端不可用时不会静默回退到本地）：

```powershell
python -m rdagent.app.gf_rd_loop.factor --loop-n 5
```

入口会自动读取项目根目录的 `.env`。主循环每轮依次完成 DeepSeek 假设与
DSL 因子生成、远端 SSH 评估、SQLite 登记和反馈写入；不要求本机存在
`daily_pv.h5`、Qlib 或 Docker。

循环默认以 70% 概率继续改进上一研究方向、以 30% 概率独立探索当前测试
较少的方向。连续六轮属于同一方向，或连续三轮没有刷新最佳结果时，下一轮
会强制探索。探索轮不会把以往假设、指标和反馈叙事交给 LLM，只提供名称与
表达式用于防止重复。可在 `.env` 中通过 `GF_FACTOR_EXPLORATION_PROBABILITY`、
`GF_FACTOR_MAX_SAME_FAMILY_STREAK`、`GF_FACTOR_STAGNATION_PATIENCE` 和可选的
`GF_FACTOR_SEARCH_RANDOM_SEED` 调整该行为。

执行一个远程候选因子：

```powershell
python scripts/gf_quant_remote_controller.py `
  --request configs/gf_quant/candidates/price_volume_reversal_20.json `
  --result artifacts/live/pv_reversal_20.json
```

## 因子 DSL 算子

候选因子是一个只含数据字段与算子的表达式。算子按滚动统计、滚动标准化与
位置、滞后变化、定向财务、双面板滚动、截面变换、稳健变换、逐元素数学组织，完整
语义见 `docs/factor_operator_catalog.md`。当前共有 58 个函数算子；窗口与位移必须是正整数字面量，
`winsorize` 的分位数必须是 `(0, 0.5)` 内的字面量。

财务季度算子按会计期间 `report` 对齐，但只从公告日 `dt` 起生效，不能用
63/252 个交易日位移替代 QOQ/YOY。行业占比算子使用显式的 `industry`
面板；该面板由已核验的 `G:/base/industry_ci.parquet` 生成，固定选择
“中信一级”，排除 `CIHK...` 港股镜像记录，并按 `beg_date`/`end_date`
有效区间把 `indus_code` 映射到交易日。

算子契约同时存在于远端 worker 与本地校验器两处（worker 是单文件交付物，
不能 import 本地包）；`test/scenarios/gf_quant/test_operators.py` 比对两份
副本，新增算子时需同步更新 `docs/factor_operator_catalog.md` 与
`prompts.yaml`。

登记评估结果：

```powershell
python -m scripts.register_gf_live_results `
  artifacts/live/pv_reversal_20.json `
  artifacts/live/fund_book_to_market_pit.json `
  --export artifacts/live/registry_snapshot.json
```

## 策略层（第一批）

`rdagent/scenarios/gf_quant/strategy/` 开始把纯单因子工具扩展为
“因子发现—因子整合—组合规则—组合评价/登记”全链路。第一批是纯新增模块，
尚未接入研究循环，也不改动任何既有流程：

- `candidate.py`：`StrategyCandidate` 规格——引用的因子或选样规则、Top-N、
  加权方式、权重上限、信号滞后与成本假设；与 `FactorCandidate` 字段独立。
- `composition.py`：`build_target_weights` 由截面打分与可投资掩码生成目标
  权重（等权 / 名次 / 比例加权，含单票权重上限）。
- `backtest.py`：`run_portfolio_backtest` 强制信号滞后、持仓漂移与单边换手
  成本，输出总体与逐年组合指标及超额收益指标。
- `registry.py` / `service.py`：独立的 SQLite 策略登记表、fail-closed 验收门，
  以及基于日度收益轨迹的策略去重（复用因子去重器）。

架构评估（一体式端到端 / 分层模块化 / 折中）、与中证自由现金流指数编制规则
的映射，以及后续批次计划见 `docs/gf_strategy_agent_design.md`。

运行核心测试：

```powershell
python -m pytest -q
```

## 保留目录

- `rdagent/scenarios/gf_quant/`：GF 数据、切分、指标、去重、登记与研究场景。
- `rdagent/scenarios/gf_quant/strategy/`：策略层第一批（组合规则、组合回测、策略登记）。
- `rdagent/components/`、`rdagent/core/`：因子生成循环所需的最小 RD-Agent 基础设施。
- `rdagent/scenarios/qlib/`：GF 因子循环仍需复用的 Qlib 因子基类和回测模板。
- `scripts/`：远程控制器、远端 worker 与结果登记脚本。
- `configs/`、`artifacts/live/`：候选配置与可复核评估结果。
- `test/scenarios/gf_quant/`：项目专属核心测试。

本项目保留上游 MIT License；评估结果仅用于研究筛选，不代表生产收益承诺。
