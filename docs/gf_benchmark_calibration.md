# GF 基准因子校准报告（benchmark calibration）

生成时间：`2026-08-30T07:38:07Z`（UTC）

## 口径

- 评估窗口：`2020-01-01` 至 `2023-12-31`；5 日前瞻 RankIC 每 5 日采样；多空组合使用 1 日信号滞后与现行的分位规则。
- 指标定义与候选因子完全一致：远端 worker `scripts/gf_quant_remote_worker.py` 的`daily_series` / `metrics_from_daily`。
- 基准池按族划分：`price_volume` = artifacts ∪ alpha158，`fundamental` = basic_fundamental；相关性/冗余度校准只在族内进行，注册表全局去重仍作为第二道安全检查。
- 每个因子先按透明的规则剔除（见下文），再在族内用「绝对相关 ≥ 阈值」的单链聚类分组；同一簇内的相似因子不会主导分布：每个因子的权重为 `1 / (簇内样本数 × 簇数)`，等价于每簇取一个代表样本。
- 分位规则：质量下限 = 簇平衡 P50；覆盖率下限 = P25；风险/成本上限 = P75；年度整数规则 = 簇平衡中位数向下取整并封顶到可用年数。
- 方向处理：候选因子与基准因子的符号是任意的，因此所有带符号的比较都改用方向不变量：`abs(RankIC)`、`abs(RankICIR)`、方向调整后的年化收益/Sharpe（按各自聚合 RankIC 的符号调整）、以及「按自身预测方向调整后的序列的回撤幅度」。不会跨因子比较原始带符号收益。

## 族：`price_volume`

- 样本：输入 326 个，剔除 8 个，有效 318 个，聚类 107 簇（代表样本 107 个）。
- 聚类：`spread` 日度曲线的绝对 Pearson 相关 ≥ `0.88`，重叠 ≥ 100 个交易日。
- 数据来源：`G:/gf_quant_agent/alpha158`, `G:/gf_quant_agent/artifacts`

| 标准 | 数值 | 规则 |
| --- | --- | --- |
| 最小 |RankIC| | `0.017433` | 簇平衡 P50 |
| 最小 |RankICIR| | `0.217482` | 簇平衡 P50（样本 318） |
| 最小覆盖率 | `0.938668` | 簇平衡 P25 |
| 最大换手率 | `0.727267` | 簇平衡 P75 |
| 最大回撤幅度 | `0.196002` | 簇平衡 P75 |
| 最少观测数 | `483` | 簇平衡 P25 |
| 年度最少有效年数 | `4` | 簇平衡中位数向下取整，封顶 4.0 年 |
| 年度最少同号年数 | `2` | 簇平衡中位数向下取整 |
| 方向调整后最小年化收益 | `0.082061` | 簇平衡 P50 |
| 方向调整后最小 Sharpe | `1.112711` | 簇平衡 P50 |

### 剔除明细

| 因子 | 原因 | 说明 |
| --- | --- | --- |
| `fund_book_to_market_pit` | `empty_panel_in_window` | empty_panel_in_window |
| `fund_book_to_price` | `empty_panel_in_window` | empty_panel_in_window |
| `fund_cash_income_stability` | `insufficient_observations` | observations=0 |
| `fund_cashflow_stability` | `insufficient_observations` | observations=0 |
| `fund_roa_stability` | `insufficient_observations` | observations=0 |
| `fund_roe_stability` | `insufficient_observations` | observations=0 |
| `pv_reversal_20` | `empty_panel_in_window` | empty_panel_in_window |
| `short_term_reversal` | `empty_panel_in_window` | empty_panel_in_window |

### 代表样本分布（用于百分位）

- `abs_rank_ic`（quality_floor，P50）：n=107, min=0.000555232, P25=0.00738714, P50=0.0174395, max=0.0736412
- `abs_rank_icir`（quality_floor，P50）：n=107, min=0.0116047, P25=0.103416, P50=0.21813, max=0.774309
- `coverage`（coverage_floor，P25）：n=107, min=0.155783, P25=0.938901, P50=0.980706, max=1
- `turnover`（risk_ceiling，P75）：n=107, min=0.00585468, P25=0.246953, P50=0.485537, max=1.52918
- `abs_drawdown`（risk_ceiling，P75）：n=107, min=0.0265362, P25=0.0604335, P50=0.105126, max=0.998624
- `direction_adjusted_annualized_return`（quality_floor，P50）：n=107, min=-1.57859, P25=0.000628696, P50=0.076128, max=0.606412
- `direction_adjusted_sharpe`（quality_floor，P50）：n=107, min=-5.41111, P25=0.00889237, P50=1.04245, max=5.25192
- `observations`（coverage_floor，P25）：n=107, min=146, P25=483, P50=483, max=951
- `valid_years`（annual_rule，P50）：n=107, min=2, P25=2, P50=4, max=4
- `matching_sign_years`（annual_rule，P50）：n=107, min=1, P25=2, P50=2, max=4

## 族：`fundamental`

- 样本：输入 111 个，剔除 0 个，有效 111 个，聚类 40 簇（代表样本 40 个）。
- 聚类：`spread` 日度曲线的绝对 Pearson 相关 ≥ `0.88`，重叠 ≥ 100 个交易日。
- 数据来源：`G:/gf_quant_agent/basic_fundamental`

| 标准 | 数值 | 规则 |
| --- | --- | --- |
| 最小 |RankIC| | `0.005892` | 簇平衡 P50 |
| 最小 |RankICIR| | `0.085450` | 簇平衡 P50（样本 111） |
| 最小覆盖率 | `0.977443` | 簇平衡 P25 |
| 最大换手率 | `0.020149` | 簇平衡 P75 |
| 最大回撤幅度 | `0.201124` | 簇平衡 P75 |
| 最少观测数 | `969` | 簇平衡 P25 |
| 年度最少有效年数 | `4` | 簇平衡中位数向下取整，封顶 4.0 年 |
| 年度最少同号年数 | `3` | 簇平衡中位数向下取整 |
| 方向调整后最小年化收益 | `0.024825` | 簇平衡 P50 |
| 方向调整后最小 Sharpe | `0.390497` | 簇平衡 P50 |

### 剔除明细

- 无剔除。

### 代表样本分布（用于百分位）

- `abs_rank_ic`（quality_floor，P50）：n=40, min=0.000132754, P25=0.00301695, P50=0.00601734, max=0.0474568
- `abs_rank_icir`（quality_floor，P50）：n=40, min=0.00140793, P25=0.0488267, P50=0.0838889, max=0.29774
- `coverage`（coverage_floor，P25）：n=40, min=0.380512, P25=0.978309, P50=0.99049, max=1
- `turnover`（risk_ceiling，P75）：n=40, min=0.00315826, P25=0.00514937, P50=0.00873864, max=0.0352881
- `abs_drawdown`（risk_ceiling，P75）：n=40, min=0.036668, P25=0.0982774, P50=0.146497, max=0.286218
- `direction_adjusted_annualized_return`（quality_floor，P50）：n=40, min=-0.041296, P25=0.000735455, P50=0.0250825, max=0.146036
- `direction_adjusted_sharpe`（quality_floor，P50）：n=40, min=-0.466436, P25=0.0066289, P50=0.400939, max=1.38662
- `observations`（coverage_floor，P25）：n=40, min=969, P25=969, P50=969, max=969
- `valid_years`（annual_rule，P50）：n=40, min=4, P25=4, P50=4, max=4
- `matching_sign_years`（annual_rule，P50）：n=40, min=1, P25=3, P50=3, max=4

## 产物

- 机读标准：`configs/gf_quant/benchmark_standards.json`
- 紧凑摘要（本地）：`git_ignore_folder/gf_benchmark/raw`
- 紧凑曲线签名（本地）：`git_ignore_folder/gf_benchmark/signatures`
- 远端摘要：`G:/gf_quant_agent/benchmark`
