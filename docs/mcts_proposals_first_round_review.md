# MCTS 联合搜索 39 条 LLM 因子提案明细

来源：D:\BigFiles\gf-agent 仓库内 2026-09-22 阶段二的三轮 MCTS 搜索运行（搜索运行 ID 依次为 search_990dd621958e → search_bcb45afc129c → search_aefed4b5f3c1）。

## 摘要

- 累计真实 LLM 提案：39 条，全部归入 price_volume 家族，全部 `status = rejected`。
- 因 5 轮 fundamental 提案均抛 FileNotFoundError（缺基准签名），仅 price_volume 提案进入本地 RankIC 验收链；39 条 = run1 的 17 + run2 的 13 + run3 的 9。
- 树节点：17 → 27 → 37 续树；3 轮搜索均因 every evaluated configuration failed the strategy gate 返回 `no_best_node`，搜索树从未携带因子。
- 评估窗口：本地 RankIC 窗口 2021-04-08..2026-08-28（取自 `temp/round1.json` 的 notes）。

## 数据来源

| 用途 | 路径 | 行/大小 |
| --- | --- | --- |
| 主日志（run1，6 轮 PV LLM 提案 + 5 轮 fundamental LLM 提案 + 11 轮 strategy 提案） | `D:\BigFiles\gf-agent\temp\round1.log` | 3 272 行 / 913 991 字节 |
| run1 搜索结果摘要 | `D:\BigFiles\gf-agent\temp\round1.json` | 80 行 / 3 513 字节 |
| run2 搜索结果摘要 | `D:\BigFiles\gf-agent\temp\round2.json` | 76 行 / 3 309 字节 |
| run3 搜索结果摘要 | `D:\BigFiles\gf-agent\temp\round3.json` | 76 行 / 3 261 字节 |
| 39 条提案明细（指标 JSON、表达式、hypothesis、被拒原因） | `D:\BigFiles\gf-agent\git_ignore_folder\gf_quant\search_test\round1\work\factor_registry.sqlite`，`factors` 表 39 行 | sqlite |
| 39 条精简 JSON（用于本明细的二次渲染） | `D:\BigFiles\gf-agent\temp\factors_39_dump.json` | 39 条 |
| 设计文档（门槛与方案依据） | `D:\BigFiles\gf-agent\docs\gf_strategy_mcts_redesign.md` | — |
| price_volume 校准门槛 | `D:\BigFiles\gf-agent\configs\gf_quant\benchmark_standards.json`，`families.price_volume.standards` | — |

校准门槛（直接决定 39 条全拒的 6 道门槛）：

| 门槛名 | 阈值 | 说明 |
| --- | --- | --- |
| `min_abs_rank_ic` | 0.0174326921 | |Rank_IC| 下限 |
| `min_abs_rank_icir` | 0.2174823025 | |Rank_ICIR| 下限 |
| `min_coverage` | 0.9386682604 | 截面覆盖下限 |
| `max_turnover` | 0.7272665214 | 换手率上限 |
| `max_abs_drawdown` | 0.1960024818 | 长短组合 |max_drawdown| 上限（profile 路径） |
| `min_direction_adjusted_annualized_return` | 0.0820614461 | profile 路径的方向调整后年化收益下限 |
| `min_direction_adjusted_sharpe` | 1.1127106469 | profile 路径的方向调整后 Sharpe 下限 |
| `annual_min_valid_years` | 4 | 至少 4 个年度的 Rank_IC 非空 |
| `annual_min_matching_sign_years` | 2 | 至少 2 个年度的 Rank_IC 符号与全样本一致 |

## 39 条提案逐条明细

列含义：`abs_rank_icir` = |Rank_ICIR|；`abs_drawdown` = |max_drawdown|（profile 路径）；`ann_ret` = `annualized_return`；`sharpe` 为非 profile 路径的 sharpe 标量；`turnover` 为日均换手率；`coverage` 为截面覆盖率；`obs` 为本地 RankIC 评估窗口（2021-04-08..2026-08-28）的观测样本数。`Fail` 为被卡的门槛列表。`rank_ic/rank_icir` 列出原值（含符号）。


### Run 1（搜索 ID search_990dd621958e，节点 0→17，本轮 17 条）

**[1] amihud_illiq_20**  （PV#1）
- 表达式：`ts_mean(abs(pct_change(adjclose, 1)) / (amount + 1), 20)`
- 假设：Stocks with higher recent price impact per unit traded amount are less liquid and tend to earn higher expected returns in China A-shares; the 20-day mean of absolute return over amount captures this illiquidity premium.
- rank_ic = 0.037987，rank_icir = 0.188937
- abs_rank_ic = 0.037987，abs_rank_icir = 0.188937，abs_drawdown = 0.397190
- IC = 0.024391，ICIR = 0.148512，ann_ret = 0.160004，sharpe = 0.668791
- turnover = 0.075029，coverage = 0.999743，win_rate = 0.531346，spread = 0.000635，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[2] signed_volume_pressure_20**  （PV#1）
- 表达式：`ts_sum(sign(delta(adjclose, 1)) * volume, 20) / (ts_sum(volume, 20) + 1)`
- 假设：Net signed trading volume over the past 20 days, normalized by total volume, proxies for buying pressure and predicts short-term price continuation or reversal in A-shares.
- rank_ic = -0.016127，rank_icir = -0.078588
- abs_rank_ic = 0.016127，abs_rank_icir = 0.078588，abs_drawdown = 0.447255
- IC = 0.003753，ICIR = 0.018244，ann_ret = 0.024517，sharpe = 0.109555
- turnover = 0.392303，coverage = 0.999743，win_rate = 0.496942，spread = 0.000097，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[3] volume_return_corr_20**  （PV#1）
- 表达式：`ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20)`
- 假设：The rolling 20-day correlation between daily returns and volume changes measures whether volume confirms price moves; low or negative correlation indicates weak conviction and predicts subsequent reversal.
- rank_ic = -0.020388，rank_icir = -0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.371897
- IC = -0.006482，ICIR = -0.036143，ann_ret = 0.002712，sharpe = 0.014127
- turnover = 0.269782，coverage = 0.998360，win_rate = 0.480887，spread = 0.000011，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[4] Volume-Price Change Correlation (20D)**  （PV#2）
- 表达式：`ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20)`
- 假设：A-share stocks with high co-movement between daily price changes and volume changes over the past month tend to be driven by attention/herding and subsequently underperform; low correlation indicates healthier price discovery and higher future returns.
- rank_ic = -0.020388，rank_icir = -0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.371897
- IC = -0.006482，ICIR = -0.036143，ann_ret = 0.002712，sharpe = 0.014127
- turnover = 0.269782，coverage = 0.998360，win_rate = 0.480887，spread = 0.000011，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[5] Amihud Illiquidity (20D)**  （PV#2）
- 表达式：`cs_rank(ts_mean(abs(pct_change(adjclose, 1)) / amount, 20))`
- 假设：Stocks with higher average absolute return per unit of trading amount over the past month are less liquid and should earn an illiquidity premium.
- rank_ic = 0.037939，rank_icir = 0.188037
- abs_rank_ic = 0.037939，abs_rank_icir = 0.188037，abs_drawdown = 0.397190
- IC = 0.038092，ICIR = 0.190248，ann_ret = 0.204216，sharpe = 0.974283
- turnover = 0.074553，coverage = 0.998373，win_rate = 0.530581，spread = 0.000810，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[6] Turnover-Amplified Short-Term Reversal**  （PV#2）
- 表达式：`-1 * ts_zscore(adjclose, 20) * ts_rank(turnover, 20)`
- 假设：Short-term price reversal is stronger when recent turnover is high; stocks that are oversold on high attention/turnover tend to rebound.
- rank_ic = 0.009501，rank_icir = 0.057616
- abs_rank_ic = 0.009501，abs_rank_icir = 0.057616，abs_drawdown = 0.276495
- IC = 0.002648，ICIR = 0.015594，ann_ret = 0.021469，sharpe = 0.112464
- turnover = 1.112297，coverage = 0.998662，win_rate = 0.503058，spread = 0.000085，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[7] Amihud_Illiquidity_20D**  （PV#3）
- 表达式：`cs_rank(ts_mean(abs(pct_change(adjclose, 1)) / amount, 20))`
- 假设：Higher Amihud illiquidity (absolute return per unit amount) indicates lower liquidity and higher liquidity premium; cross-sectional long high illiquidity stocks earns positive returns. All inputs are point-in-time and rolling windows use only past data.
- rank_ic = 0.037939，rank_icir = 0.188037
- abs_rank_ic = 0.037939，abs_rank_icir = 0.188037，abs_drawdown = 0.397190
- IC = 0.038092，ICIR = 0.190248，ann_ret = 0.204216，sharpe = 0.974283
- turnover = 0.074553，coverage = 0.998373，win_rate = 0.530581，spread = 0.000810，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[8] Volume_Weighted_Momentum_20D**  （PV#3）
- 表达式：`cs_rank(ts_sum(pct_change(adjclose, 1) * volume, 20) / ts_sum(volume, 20))`
- 假设：Volume-weighted average return over 20 days captures price trends confirmed by trading volume; stocks with higher volume-supported momentum continue to outperform. Uses only lagged and rolling data, no future information.
- rank_ic = -0.020598，rank_icir = -0.094609
- abs_rank_ic = 0.020598，abs_rank_icir = 0.094609，abs_drawdown = 0.465729
- IC = -0.000085，ICIR = -0.000382，ann_ret = 0.098238，sharpe = 0.381833
- turnover = 0.329665，coverage = 0.998637，win_rate = 0.504587，spread = 0.000390，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[9] Turnover_Reversal_20D**  （PV#3）
- 表达式：`cs_rank(-ts_zscore(turnover, 20))`
- 假设：High abnormal turnover (turnover z-score over 20 days) signals speculative attention and overreaction, predicting lower future returns; long low turnover z-score stocks. Rolling z-score uses only historical turnover, causal.
- rank_ic = 0.005267，rank_icir = 0.031626
- abs_rank_ic = 0.005267，abs_rank_icir = 0.031626，abs_drawdown = 0.414164
- IC = 0.000402，ICIR = 0.002413，ann_ret = 0.000387，sharpe = 0.001973
- turnover = 1.139613，coverage = 0.998662，win_rate = 0.516820，spread = 0.000002，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[10] PriceVolumeCorrelationReversal**  （PV#4）
- 表达式：`-1 * ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20)`
- 假设：High positive rolling correlation between daily price changes and volume changes reflects speculative momentum that subsequently reverses; negative correlation indicates informed accumulation, so short high correlation and long low correlation.
- rank_ic = 0.020388，rank_icir = 0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.440160
- IC = 0.006482，ICIR = 0.036143，ann_ret = -0.036202，sharpe = -0.186690
- turnover = 0.270053，coverage = 0.998360，win_rate = 0.510703，spread = -0.000144，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[11] TurnoverWeightedShortTermReversal**  （PV#4）
- 表达式：`-1 * ts_sum(pct_change(adjclose, 1), 5) * ts_rank(turnover, 20)`
- 假设：Short-term price moves accompanied by abnormally high turnover are more likely driven by retail speculation and mean-revert; scaling the 5-day return by the 20-day turnover rank gives a contrarian signal.
- rank_ic = 0.013901，rank_icir = 0.082071
- abs_rank_ic = 0.013901，abs_rank_icir = 0.082071，abs_drawdown = 0.450128
- IC = 0.004719，ICIR = 0.023787，ann_ret = -0.090880，sharpe = -0.460378
- turnover = 1.139319，coverage = 0.999769，win_rate = 0.500000，spread = -0.000361，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[12] AmihudIlliquidityPremium**  （PV#4）
- 表达式：`cs_zscore(ts_mean(abs(pct_change(adjclose, 1)) / (amount + 1), 20))`
- 假设：Stocks with higher recent price impact per unit of traded amount are less liquid and earn an illiquidity premium; the 20-day mean of absolute daily return divided by amount captures this effect cross-sectionally.
- rank_ic = 0.037987，rank_icir = 0.188937
- abs_rank_ic = 0.037987，abs_rank_icir = 0.188937，abs_drawdown = 0.397190
- IC = 0.024391，ICIR = 0.148512，ann_ret = 0.160004，sharpe = 0.668791
- turnover = 0.075029，coverage = 0.999743，win_rate = 0.531346，spread = 0.000635，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[13] turnover_price_corr_reversal_20**  （PV#5）
- 表达式：`cs_rank(-ts_corr(pct_change(adjclose, 5), pct_change(turnover, 5), 20))`
- 假设：A-share speculative volume-price co-movement tends to overheat; stocks with lower rolling correlation between 5-day price change and 5-day turnover change are less crowded and should outperform on a short horizon. Cross-sectional rank makes high values mean lower co-movement.
- rank_ic = 0.022027，rank_icir = 0.116943
- abs_rank_ic = 0.022027，abs_rank_icir = 0.116943，abs_drawdown = 0.429589
- IC = 0.003654，ICIR = 0.019972，ann_ret = 0.008099，sharpe = 0.039471
- turnover = 0.247918，coverage = 0.998205，win_rate = 0.516055，spread = 0.000032，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[14] vwap_deviation_reversal_20**  （PV#5）
- 表达式：`-cs_rank((adjclose - ts_sum(adjclose * volume, 20) / ts_sum(volume, 20)) / (ts_std(adjclose, 20) + 1e-9))`
- 假设：When price is far above its 20-day volume-weighted average price, recent buying is likely overextended and vulnerable to reversal; the negative cross-sectional rank favors stocks trading below VWAP support.
- rank_ic = 0.011367，rank_icir = 0.054461
- abs_rank_ic = 0.011367，abs_rank_icir = 0.054461，abs_drawdown = 0.499600
- IC = 0.000736，ICIR = 0.003576，ann_ret = 0.002631，sharpe = 0.011134
- turnover = 0.597712，coverage = 0.998662，win_rate = 0.493119，spread = 0.000010，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[15] amihud_illiquidity_20**  （PV#5）
- 表达式：`cs_rank(ts_mean(abs(pct_change(adjclose, 1)) / (amount + 1), 20))`
- 假设：Average absolute daily return per unit amount measures illiquidity; in China A-shares, less liquid names can earn an illiquidity premium, so cross-sectional rank favors higher Amihud illiquidity.
- rank_ic = 0.037987，rank_icir = 0.188937
- abs_rank_ic = 0.037987，abs_rank_icir = 0.188937，abs_drawdown = 0.397190
- IC = 0.037216，ICIR = 0.186076，ann_ret = 0.160004，sharpe = 0.668791
- turnover = 0.075029，coverage = 0.999743，win_rate = 0.531346，spread = 0.000635，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[16] pv_corr_20_reversal**  （PV#6）
- 表达式：`(0 - ts_corr(pct_change(adjclose, 1), ts_zscore(volume, 20), 20))`
- 假设：Low 20-day correlation between daily returns and volume z-score identifies stocks with less speculative volume confirmation; in China A-shares such stocks tend to outperform.
- rank_ic = 0.008669，rank_icir = 0.048948
- abs_rank_ic = 0.008669，abs_rank_icir = 0.048948，abs_drawdown = 0.488392
- IC = -0.004812，ICIR = -0.027983，ann_ret = -0.082881，sharpe = -0.446298
- turnover = 0.319597，coverage = 0.998313，win_rate = 0.501529，spread = -0.000329，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[17] volume_conditioned_reversal_5_20**  （PV#6）
- 表达式：`(0 - pct_change(adjclose, 5)) * cs_rank(ts_mean(volume, 5) / (ts_mean(volume, 20) + 1))`
- 假设：Short-term price reversals are amplified by high recent volume; losers with elevated 5-day versus 20-day volume tend to rebound.
- rank_ic = -0.003186，rank_icir = -0.020335
- abs_rank_ic = 0.003186，abs_rank_icir = 0.020335，abs_drawdown = 0.566423
- IC = -0.014979，ICIR = -0.076774，ann_ret = -0.048689，sharpe = -0.261230
- turnover = 0.785828，coverage = 0.999769，win_rate = 0.483180，spread = -0.000193，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown


### Run 2（搜索 ID search_bcb45afc129c，节点 17→27，本轮 13 条）

**[18] turnover_shock_price_interaction**  （PV#1）
- 表达式：`-1 * cs_zscore(ts_zscore(turnover, 20)) * cs_zscore(pct_change(adjclose, 5))`
- 假设：换手率相对自身历史的极端抬升代表注意力驱动的过度交易;若同时伴随短期价格走强(追涨),后续收益倾向反转,而放量下跌后则倾向修复。用换手z-score与5日收益z-score的交互刻画这一量价共振,取负号后做多'高换手+超跌'组合。
- rank_ic = 0.010509，rank_icir = 0.070214
- abs_rank_ic = 0.010509，abs_rank_icir = 0.070214，abs_drawdown = 0.386461
- IC = 0.002692，ICIR = 0.014106，ann_ret = -0.060092，sharpe = -0.334867
- turnover = 1.167833，coverage = 0.998662，win_rate = 0.492355，spread = -0.000238，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[19] vwap_gap_reversal**  （PV#1）
- 表达式：`(ts_mean(amount / volume, 20) - adjclose) / ts_mean(amount / volume, 20)`
- 假设：以成交额/成交量构造20日成交量加权均价(近端资金成本),收盘价显著低于该均线说明近期买盘成本被套、抛压释放较充分,存在向资金成本回归的均值回复;价格大幅高于VWAP则相反。
- rank_ic = 0.012790，rank_icir = 0.053519
- abs_rank_ic = 0.012790，abs_rank_icir = 0.053519，abs_drawdown = 0.685652
- IC = -0.003277，ICIR = -0.012713，ann_ret = -0.137109，sharpe = -0.493300
- turnover = 0.303953，coverage = 0.998399，win_rate = 0.488532，spread = -0.000544，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[20] pv_corr_divergence_reversal**  （PV#3）
- 表达式：`-1 * cs_rank(ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20))`
- 假设：Stocks whose daily return changes and daily volume changes are negatively correlated over the past 20 trading days (volume expanding on down moves, a distribution/divergence pattern) tend to subsequently outperform; cross-sectional rank of this correlation isolates the divergence signal.
- rank_ic = 0.020388，rank_icir = 0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.440160
- IC = 0.007072，ICIR = 0.039142，ann_ret = -0.036202，sharpe = -0.186690
- turnover = 0.270053，coverage = 0.998360，win_rate = 0.510703，spread = -0.000144，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[21] amihud_illiq_turnover_20**  （PV#3）
- 表达式：`cs_rank(ts_mean(abs(pct_change(adjclose, 1)) / (turnover + 0.01), 20))`
- 假设：Illiquidity, measured as the 20-day average absolute daily return per unit of turnover, carries a positive risk premium in China A-shares; cross-sectional rank captures the premium without relying on price level or market value.
- rank_ic = 0.026954，rank_icir = 0.151161
- abs_rank_ic = 0.026954，abs_rank_icir = 0.151161，abs_drawdown = 0.313309
- IC = 0.030639，ICIR = 0.174966，ann_ret = 0.159484，sharpe = 0.731230
- turnover = 0.101429，coverage = 0.999743，win_rate = 0.532110，spread = 0.000633，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[22] volume_amplified_short_reversal**  （PV#3）
- 表达式：`-1 * cs_zscore(pct_change(adjclose, 5) * ts_zscore(volume, 20))`
- 假设：Short-term 5-day reversals are stronger when accompanied by abnormally high volume (ts_zscore of volume), because high-volume moves are more likely driven by transient liquidity demand than by information; shorting past winners and buying past losers conditioned on volume shock.
- rank_ic = 0.012737，rank_icir = 0.073292
- abs_rank_ic = 0.012737，abs_rank_icir = 0.073292，abs_drawdown = 0.468508
- IC = 0.002046，ICIR = 0.010399，ann_ret = -0.089685，sharpe = -0.458612
- turnover = 1.169241，coverage = 0.998662，win_rate = 0.493119，spread = -0.000356，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[23] pv_corr_reversal_20d**  （PV#5）
- 表达式：`cs_rank(-1 * ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20))`
- 假设：A negative 20-day correlation between daily returns and daily volume changes marks stocks whose recent price advances lack volume confirmation (weak sponsorship / distribution). Such unconfirmed rallies tend to reverse, so cross-sectionally ranking this negative correlation should isolate future underperformers.
- rank_ic = 0.020388，rank_icir = 0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.440160
- IC = 0.007072，ICIR = 0.039142，ann_ret = -0.036202，sharpe = -0.186690
- turnover = 0.270053，coverage = 0.998360，win_rate = 0.510703，spread = -0.000144，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[24] turnover_crowding_reversal**  （PV#5）
- 表达式：`cs_rank(-1 * ts_zscore(turnover, 20))`
- 假设：Abnormally high turnover relative to a stock's own 20-day history signals speculative crowding and attention-driven overreaction. Since such flows are transient in China A-shares, the z-score of turnover is expected to negatively predict subsequent short-horizon returns.
- rank_ic = 0.005267，rank_icir = 0.031626
- abs_rank_ic = 0.005267，abs_rank_icir = 0.031626，abs_drawdown = 0.414164
- IC = 0.000402，ICIR = 0.002413，ann_ret = 0.000387，sharpe = 0.001973
- turnover = 1.139613，coverage = 0.998662，win_rate = 0.516820，spread = 0.000002，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[25] amihud_illiquidity_20d**  （PV#5）
- 表达式：`cs_rank(ts_mean(abs(pct_change(adjclose, 1)), 20) / (ts_mean(amount, 20) + 1))`
- 假设：Price impact per unit of traded value (Amihud-style illiquidity, built from absolute daily return scaled by average amount) captures the illiquidity premium: less liquid stocks command higher expected returns. Cross-sectional ranking of this ratio gives a slow-moving liquidity exposure that is largely orthogonal to pure momentum.
- rank_ic = 0.037657，rank_icir = 0.185005
- abs_rank_ic = 0.037657，abs_rank_icir = 0.185005，abs_drawdown = 0.399132
- IC = 0.036139，ICIR = 0.178061，ann_ret = 0.223995，sharpe = 0.923242
- turnover = 0.082788，coverage = 0.999743，win_rate = 0.538991，spread = 0.000889，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[26] price_volume_corr_reversal_20**  （PV#7）
- 表达式：`-ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20)`
- 假设：When daily price changes co-move strongly with volume changes, the move is attention-driven and subsequently reverses; hence the negative of the 20-day price-volume correlation predicts returns.
- rank_ic = 0.020388，rank_icir = 0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.440160
- IC = 0.006482，ICIR = 0.036143，ann_ret = -0.036202，sharpe = -0.186690
- turnover = 0.270053，coverage = 0.998360，win_rate = 0.510703，spread = -0.000144，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[27] abnormal_turnover_reversal**  （PV#7）
- 表达式：`-ts_zscore(turnover, 20)`
- 假设：Short-term turnover surges relative to a stock's own baseline reflect speculative overbuying, so high abnormal turnover predicts negative short-horizon returns in A-shares.
- rank_ic = 0.005267，rank_icir = 0.031626
- abs_rank_ic = 0.005267，abs_rank_icir = 0.031626，abs_drawdown = 0.414164
- IC = 0.002604，ICIR = 0.015123，ann_ret = 0.000387，sharpe = 0.001973
- turnover = 1.139613，coverage = 0.998662，win_rate = 0.516820，spread = 0.000002，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[28] pv_volume_price_absorption**  （PV#9）
- 表达式：`ts_zscore(volume, 20) - ts_zscore(abs(pct_change(adjclose, 1)), 20)`
- 假设：When trading volume spikes but absolute return stays muted, supply is being absorbed (accumulation/distribution), which predicts subsequent directional drift as the imbalance resolves.
- rank_ic = -0.008400，rank_icir = -0.049881
- abs_rank_ic = 0.008400，abs_rank_icir = 0.049881，abs_drawdown = 0.496421
- IC = -0.002851，ICIR = -0.016601，ann_ret = -0.092494，sharpe = -0.493560
- turnover = 1.346048，coverage = 0.998637，win_rate = 0.485474，spread = -0.000367，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_sharpe, abs_drawdown

**[29] pv_volume_return_confirmation**  （PV#9）
- 表达式：`ts_corr(signed_log(volume), pct_change(adjclose, 1), 20)`
- 假设：The rolling correlation between signed volume and one-day return measures whether volume confirms price direction; persistently positive confirmation signals informed order flow and trend continuation, while negative confirmation flags exhaustion.
- rank_ic = -0.009156，rank_icir = -0.051095
- abs_rank_ic = 0.009156，abs_rank_icir = 0.051095，abs_drawdown = 0.243594
- IC = 0.002522，ICIR = 0.014536，ann_ret = 0.055086，sharpe = 0.293059
- turnover = 0.348325，coverage = 0.998637，win_rate = 0.493884，spread = 0.000219，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[30] pv_turnover_acceleration**  （PV#9）
- 表达式：`ts_zscore(ts_mean(turnover, 5), 60) - ts_zscore(ts_mean(turnover, 20), 60)`
- 假设：A short-window turnover mean rising above its longer-window baseline reflects accelerating attention/positioning, which precedes return continuation in A-shares before crowding reverses.
- rank_ic = -0.002885，rank_icir = -0.017066
- abs_rank_ic = 0.002885，abs_rank_icir = 0.017066，abs_drawdown = 0.463001
- IC = -0.001770，ICIR = -0.010216，ann_ret = -0.071981，sharpe = -0.398208
- turnover = 0.380858，coverage = 0.997918，win_rate = 0.500765，spread = -0.000286，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown


### Run 3（搜索 ID search_aefed4b5f3c1，节点 27→37，本轮 9 条）

**[31] pv_reversal_corr_20**  （PV#1）
- 表达式：`-ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20)`
- 假设：When daily returns and volume changes co-move strongly over the past month, speculative flow drives prices away from fundamentals; the negative 20-day correlation captures subsequent reversal.
- rank_ic = 0.020388，rank_icir = 0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.440160
- IC = 0.006482，ICIR = 0.036143，ann_ret = -0.036202，sharpe = -0.186690
- turnover = 0.270053，coverage = 0.998360，win_rate = 0.510703，spread = -0.000144，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[32] turnover_weighted_mom_20**  （PV#1）
- 表达式：`ts_mean(pct_change(adjclose, 1) * ts_zscore(turnover, 20), 20)`
- 假设：Momentum is more informed when accompanied by elevated turnover; weighting daily returns by the 20-day turnover z-score isolates volume-driven trends.
- rank_ic = -0.009715，rank_icir = -0.051434
- abs_rank_ic = 0.009715，abs_rank_icir = 0.051434，abs_drawdown = 0.440481
- IC = 0.004395，ICIR = 0.021260，ann_ret = 0.118786，sharpe = 0.550255
- turnover = 0.270266，coverage = 0.998313，win_rate = 0.506881，spread = 0.000471，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[33] amihud_illiq_20d**  （PV#3）
- 表达式：`cs_rank(ts_mean(abs(pct_change(adjclose, 1)) / (amount + 1), 20))`
- 假设：The Amihud illiquidity premium: stocks with larger absolute daily returns per unit of trading amount are less liquid and earn higher expected returns; ranking this 20-day average cross-sectionally captures the premium.
- rank_ic = 0.037987，rank_icir = 0.188937
- abs_rank_ic = 0.037987，abs_rank_icir = 0.188937，abs_drawdown = 0.397190
- IC = 0.037216，ICIR = 0.186076，ann_ret = 0.160004，sharpe = 0.668791
- turnover = 0.075029，coverage = 0.999743，win_rate = 0.531346，spread = 0.000635，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_sharpe, abs_drawdown

**[34] turnover_zscore_reversal_20d**  （PV#3）
- 表达式：`cs_rank(-1 * ts_zscore(turnover, 20))`
- 假设：A positive short-term turnover z-score relative to its own trailing history signals speculative overattention and subsequent reversal; stocks with low recent turnover relative to their norm tend to outperform.
- rank_ic = 0.005267，rank_icir = 0.031626
- abs_rank_ic = 0.005267，abs_rank_icir = 0.031626，abs_drawdown = 0.414164
- IC = 0.000402，ICIR = 0.002413，ann_ret = 0.000387，sharpe = 0.001973
- turnover = 1.139613，coverage = 0.998662，win_rate = 0.516820，spread = 0.000002，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[35] pv_corr_20**  （PV#5）
- 表达式：`cs_rank(ts_corr(pct_change(adjclose, 1), pct_change(volume, 1), 20))`
- 假设：The rolling correlation between daily returns and volume changes captures price-volume confirmation: positive correlation suggests accumulation/trending, negative suggests distribution; cross-sectional ranking isolates relative confirmation strength.
- rank_ic = -0.020388，rank_icir = -0.110319
- abs_rank_ic = 0.020388，abs_rank_icir = 0.110319，abs_drawdown = 0.371897
- IC = -0.007072，ICIR = -0.039142，ann_ret = 0.002712，sharpe = 0.014127
- turnover = 0.269782，coverage = 0.998360，win_rate = 0.480887，spread = 0.000011，obs = 1308
- 被拒门槛：abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[36] turnover_reversal_20**  （PV#5）
- 表达式：`-ts_zscore(turnover, 20)`
- 假设：Abnormally high turnover relative to a stock's own recent history reflects attention-driven overreaction, predicting short-term reversal; hence the negative of the 20-day turnover z-score is expected to be positively related to future returns.
- rank_ic = 0.005267，rank_icir = 0.031626
- abs_rank_ic = 0.005267，abs_rank_icir = 0.031626，abs_drawdown = 0.414164
- IC = 0.002604，ICIR = 0.015123，ann_ret = 0.000387，sharpe = 0.001973
- turnover = 1.139613，coverage = 0.998662，win_rate = 0.516820，spread = 0.000002，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[37] pv_corr_reversal_20**  （PV#7）
- 表达式：`-1 * ts_corr(signed_log(volume), pct_change(adjclose, 1), 20)`
- 假设：The trailing 20-day correlation between signed-log volume changes and daily returns measures speculative chasing: stocks whose volume spikes co-move with positive returns are overcrowded and prone to reversal, so the negative of this correlation is a causal cross-sectional alpha.
- rank_ic = 0.009156，rank_icir = 0.051095
- abs_rank_ic = 0.009156，abs_rank_icir = 0.051095，abs_drawdown = 0.433187
- IC = -0.002522，ICIR = -0.014536，ann_ret = -0.072143，sharpe = -0.380015
- turnover = 0.347732，coverage = 0.998637，win_rate = 0.500765，spread = -0.000286，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[38] abnormal_turnover_reversal_5**  （PV#7）
- 表达式：`-1 * cs_rank(ts_sum(pct_change(adjclose, 1), 5) * clip(ts_zscore(turnover, 60), 0, 3))`
- 假设：A 5-day price gain accompanied by turnover abnormally above its own 60-day norm signals retail-driven speculative inflow that subsequently unwinds; taking the negative of the abnormal-turnover-weighted 5-day return captures this reversal.
- rank_ic = 0.011652，rank_icir = 0.063737
- abs_rank_ic = 0.011652，abs_rank_icir = 0.063737，abs_drawdown = 0.419301
- IC = 0.003094，ICIR = 0.016385，ann_ret = -0.009264，sharpe = -0.034843
- turnover = 1.032913，coverage = 0.998149，win_rate = 0.491037，spread = -0.000037，obs = 1283
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

**[39] turnover_cond_reversal_5_20**  （PV#9）
- 表达式：`-1 * ts_zscore(pct_change(adjclose, 5), 20) * cs_rank(ts_zscore(turnover, 20))`
- 假设：Short-term price run-ups accompanied by abnormally high turnover (relative to the stock's own turnover norm and the cross-section of that week) reflect crowded, liquidity-driven buying that subsequently reverses; conditioning the 5-day reversal on turnover intensity isolates this effect.
- rank_ic = 0.005886，rank_icir = 0.038813
- abs_rank_ic = 0.005886，abs_rank_icir = 0.038813，abs_drawdown = 0.291066
- IC = -0.001248，ICIR = -0.007962，ann_ret = 0.017803，sharpe = 0.104165
- turnover = 1.183504，coverage = 0.998534，win_rate = 0.496942，spread = 0.000071，obs = 1308
- 被拒门槛：abs_rank_ic, abs_rank_icir, turnover, direction_adjusted_annualized_return, direction_adjusted_sharpe, abs_drawdown

## 汇总统计

### 39 条因子在 6 道门槛上的通过/被拒

| 门槛名 | 通过数 | 被拒数 | 阈值 | 说明 |
| --- | --- | --- | --- | --- |
| `abs_rank_ic` | 18 | 21 | 0.0174326921 | |Rank_IC| 下限 |
| `abs_rank_icir` | 0 | 39 | 0.2174823025 | |Rank_ICIR| 下限 |
| `abs_drawdown` | 0 | 39 | 0.1960024818 | |max_drawdown| 上限 |
| `direction_adjusted_annualized_return` | 9 | 30 | 0.0820614461 | 方向调整后年化收益下限 |
| `direction_adjusted_sharpe` | 0 | 39 | 1.1127106469 | 方向调整后 Sharpe 下限 |
| `turnover` | 26 | 13 | 0.7272665214 | 换手率上限 |
| `coverage` | 39 | 0 | 0.9386682604 | 覆盖率下限 |

### 被拒门槛数量的分布

- 命中 3 道门槛：8 条
- 命中 4 道门槛：10 条
- 命中 5 道门槛：9 条
- 命中 6 道门槛：12 条

具体组合分布：
- `abs_drawdown+abs_rank_ic+abs_rank_icir+direction_adjusted_annualized_return+direction_adjusted_sharpe+turnover`：12 条
- `abs_drawdown+abs_rank_icir+direction_adjusted_annualized_return+direction_adjusted_sharpe`：10 条
- `abs_drawdown+abs_rank_icir+direction_adjusted_sharpe`：8 条
- `abs_drawdown+abs_rank_ic+abs_rank_icir+direction_adjusted_annualized_return+direction_adjusted_sharpe`：8 条
- `abs_drawdown+abs_rank_ic+abs_rank_icir+direction_adjusted_sharpe+turnover`：1 条

### 关键指标的总体分布

| 指标 | 最小 | 最大 | 均值 | 中位数 |
| --- | --- | --- | --- | --- |
| abs_rank_ic | 0.002885 | 0.037987 | 0.017439 | 0.013901 |
| abs_rank_icir | 0.017066 | 0.188937 | 0.091856 | 0.078588 |
| abs_drawdown | 0.243594 | 0.685652 | 0.421389 | 0.414164 |
| annualized_return | -0.137109 | 0.223995 | 0.021856 | 0.000387 |
| sharpe | -0.493560 | 0.974283 | 0.083016 | 0.001973 |
| turnover | 0.074553 | 1.346048 | 0.537601 | 0.319597 |
| coverage | 0.997918 | 0.999769 | 0.998762 | 0.998637 |
| win_rate | 0.480887 | 0.538991 | 0.507337 | 0.506881 |
| long_short_spread_daily | -0.000544 | 0.000889 | 0.000087 | 0.000002 |

### 简短结论

- 三道主导门槛全部 0/39 通过：`abs_rank_icir`（阈值 0.2174823025）、`abs_drawdown`（阈值 0.1960024818）、`direction_adjusted_sharpe`（阈值 1.1127106469）。任一道不放开，39 条提案都无法翻盘。
- `abs_rank_ic`（阈值 0.0174326921）有 18/39 通过；剩下 21/39 卡在 |Rank_IC| 上，是第三大原因。即便 18 条 |Rank_IC| 达标，仍全部卡在 ICIR。
- `direction_adjusted_annualized_return`（阈值 0.0820614461）有 9/39 通过；`turnover`（阈值 0.7272665214）有 26/39 通过；`coverage`（阈值 0.9386682604）有 39/39 通过（覆盖率 ≥ 0.998）。
- 39 条中有 8 条只命中 3 道门槛、10 条命中 4 道、9 条命中 5 道、12 条命中 6 道。
- 校准门槛（`max_abs_drawdown=0.196`、`min_abs_rank_icir=0.217`、`min_direction_adjusted_sharpe=1.113`）以合并的 artifacts+alpha158 基准分布的 quality_floor / risk_ceiling 分位数派生（见 `rdagent/scenarios/gf_quant/benchmark.py:896-909`）；数据机不可达期间只能跑本地副本，门槛基于 2020-2023 区间校准，2021-2026 评估窗口的样本分布与校准分布不一致，是当前 39 条全部卡在校准门槛之下的根本原因。
