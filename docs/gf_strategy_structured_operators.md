# GF 策略智能体：结构化组合算子实现记录

> 本文档是 68 个策略算子的实现记录，条目与 `rdagent/scenarios/gf_quant/strategy/operators.py`
> 的注册表逐条对应；两端实现锚点、注册表计数与本文标题的一致性由
> `test/scenarios/gf_quant/test_strategy_operator_registry.py` 守护。
>
> 固定执行事实：买入 10 bps、卖出 10 bps、完整往返 20 bps。交易成本不是策略
> 算子参数，不能出现在 LLM 输出、`StrategyCandidate`、CLI、环境变量或 settings
> 中。所有算子都是声明式、确定性的：模型只给出参数，执行由确定性代码完成，
> 任何位置都不接受模型生成的 Python 或回测代码。

## 1. 数据流

一个策略在引擎中被表达为下列数据流。每一段只读它声明的输入，段落之间的衔接点
即 payload 的分节边界（见 §12）。

```text
显式注入数据面：因子面板、收益面板、候选池面板、行业映射、执行面板
        ↓
候选池过滤（§3）：静态名单 → PIT 成分 → 上市年龄 → 异常状态
        → 涨跌停可交易性 → 流动性 → 价格区间 → 数据质量
        产出 eligible[t, symbol] 布尔面板
        ↓
信号值空间阶段（§4，顺序固定）：
        transform → winsorize → standardize → missing_value
        → industry_neutralize → style_neutralize → 冻结方向乘法
        （eligible 掩码在 missing_value 之后生效一次，
          决定 percentile rank 的分母）
        ↓
信号 rank 空间阶段（§4）：
        percentile rank → neutral_rank 填充 → eligible 掩码再生效一次
        → interaction 交互项重排
        ↓
blend（§4）：等权 / 固定权重 / 两级分组 / 状态条件
        （四种模式的互斥关系按实现校验，缺失语义统一为任一成员缺失即整格缺失）
        ↓
门控（§5）：有序的分位层，层间 and，不达标置 NaN
        ↓
选择（§5）：Top-N → 名次并列稳定排序 → 持仓缓冲 → 分散度筛选
        ↓
配权档（§7）：等权 / 名次 / 分数比例 / 分数变换 / 逆波动 /
         分层 / 风险平价 / 因子风险预算
        ↓
约束链（§6，顺序固定）：单名 cap → 行业 cap → 名次下限 floor
        → 联合投影（行业偏离 / 风格暴露 / beta / 因子暴露）
        → 现金残差检查
        ↓
执行层（§8）：漂移带判定 → 换手收缩 → 参与率上限或分阶段执行
        → 侧向可交易性掩码 → 整手取整 → 最低持有期锁
        （成交量与整手读执行日状态，成本公式不参与以上任何一步）
        ↓
组合回测：信号滞后、持仓漂移、固定买卖成本、净值与指标
        ↓
风险缩放（§9，与执行层同批）：回撤状态机 → 波动率目标（取 min）
        → 因子衰减缩放（信号侧）
        ↓
审计与治理（§9–§10）：验收、去重、walk-forward、成本敏感性、
        参数扰动、消融、可行性、数据泄漏、压力情景、集中度、
        manifest、可复现判定、部署档位
```

`rule` 字段是审计说明文字，执行器不解释它。新增算子要走受限 schema 加确定性
执行器这条路，不能只把名称塞进 `rule`。

## 2. 算子统一契约

### 2.1 算子落在哪几类位置

| 落点 | 形态 | 覆盖范围 |
|---|---|---|
| spec 字段 | `SignalSpec` / `CompositionSpec` / `BacktestSpec` / `GatingSpec` 上的可选字段 | §4 的信号与中性化算子、§5、§6、§8、§9.1、§9.2、§9.5 |
| weighting 档 | `CompositionSpec.weighting` 的枚举值加其伴生参数 | §7 全部 |
| request 级对象 | 与 `evaluation` 平级、不进 candidate payload 的请求节 | `audit`（§9.3、§9.4、§10.2–§10.4、§10.6、§10.7）、`walk_forward`（§10.1）、`manifest`（§10.8） |
| 显式注入数据面 | payload 里的面板或映射，由上层配置注入，引擎自己不生成 | `industry_map`、`universe_rules`、`execution_data`、暴露矩阵与基准、beta、regime 状态、因子风险预算的暴露与协方差 |

本地链路是 `strategy/` 包加回测循环，远端是单文件 worker
（`scripts/gf_quant_strategy_worker.py`）。同一算子在两端各有一份实现，锚点记在
注册表里；两侧函数逐字镜像，错误消息逐字符相同。市场数据、成分、状态、涨跌停、
成交量这类引擎不持有的输入，全部以显式配置注入，不由引擎自行生成。

### 2.2 进不进 payload 与身份哈希的关系

`StrategyCandidate.canonical_payload()` 是 `strategy_id` 身份哈希的输入，规则统一：

- 可选字段默认关闭时**不进** canonical payload，因此禁用不改变身份哈希，历史
  registry 记录不失效。`GatingSpec` 空层、universe 节缺席、`execution_data` 整节
  缺席、信号 DAG 的每个新字段、约束链的每个新族都按此模式落地。
- 启用时字段进 payload，身份随之改变。这是有意的：换了配置就是另一个策略。
- request 级对象（`audit`、`walk_forward`、`manifest`）**永不进** candidate
  payload，所以开一个审计节不会让同一个策略换一个 id。

### 2.3 fail-closed 纪律

校验走「发现非法输入即抛错」：越界参数、缺失的必要伴生参数、无法核实的数据、被
引用但不存在的键，一律拒绝执行，不降级、不静默修复、不产生无穷值。少数**声明的
降级**是有意为之且在条目里写明理由，例如样本不足的行业中性化按原值穿过、不足两天
历史的逆波动与风险平价整行等权回退、组合层面零波动率取上界。降级与 fail-closed
的分界是「这是数据缺陷还是样本性质」，两者的判据都不得由用户可调的性能旋钮反向
放宽。

### 2.4 两端镜像

新增或改动任一算子必须两端同改：

- 落点按类别固定。信号侧同时落 `signal.py` 与 worker 的镜像函数；weighting 档要
  同步 `candidate.py` 的枚举、`proposal.py` 的 payload→spec，以及 `composition.py`
  与 worker 两侧的执行分支；执行层落 `execution.py` 与 worker 的镜像块，由
  `scripts/gf_quant_mirror.py --check` 守护。
- 参数汇聚函数的键集两侧必须逐字符对齐，用符号定位而不是行号。
- 校验通过的归一值要写回 spec，canonical payload 才能哈希稳定。
- 改完必须复跑 `test/scenarios/gf_quant/test_strategy_worker_parity.py`；新增阶段要
  同时加两端直连比对用例，因为端到端数值用例对单调变换没有鉴别力。

### 2.5 参数所有权

| 所有权 | 含义 |
|---|---|
| `agent` | 策略智能体可以在允许范围内声明，仍须 schema 校验和样本外审计。当前是离散选项、阈值与账户级数字，例如 weighting 档、`top_n`、分位阈值、漂移带宽、换手收缩比例 |
| `system` | 系统固定事实，智能体不可修改：买卖各 10 bps、信号滞后下限 1 个交易日、PIT 规则、长期存在的硬交易约束 |
| `calibration` | 由研究流程或人工标准确定，改值须产生新版本。行业映射、暴露矩阵、基准权重、regime 状态标签、因子风险预算这类注入数据面属于本类，不进 LLM 词表 |

各算子记录输入快照 ID、参数 canonical JSON、schema 版本、输出 checksum、数据截止
日、缺失与降级数量以及失败原因。这些是待采集的事实，`manifest` 节因此不带任何
设置项。

### 2.6 LLM 边界

模型给出低维、离散、有金融含义的参数，执行器负责其余全部。模型不能给出成本、滞后
下限、PIT 规则，不能给代码、数学表达式或优化目标，不能引用冻结 working set 之外的
因子，也不能看样本外结果回头调参。参数由谁拥有决定它能不能进词表：`agent` 类进，
`system` 与 `calibration` 类不进；不进词表的键仍经 payload→spec 透传，幻觉输出被
spec 校验 fail-closed。

同一份 canonical payload 在同一份数据上必须给出同一结果，因此求解器、迭代预算、
约束顺序这类执行细节都锁成模块常量，而不是 payload 里的旋钮。

## 3. 候选池与可交易性算子

八个过滤段合成一个候选池配置对象，payload 节名 `universe_rules`；本地执行器是
`strategy/universe.py` 的 `build_eligible_mask`，worker 侧是
`_build_universe_eligible`。共享输入 `trading_dates` 与 `price` 在节顶层，各段启用
才进 payload。过滤结果是 `eligible[t, symbol]` 布尔面板，不可交易的名在权重生成
之前就被排除，而不是事后从订单里删掉。

### 3.1 `BaseUniverse`

- 实现状态：已实现，两端。静态显式名单，名单外的 symbol 在任何日期都不进入候选池。
- 参数：`symbols`

### 3.2 `IndexMembershipFilter`

- 实现状态：已实现，两端。按 as-of 查找注入的 PIT 成分面板；面板起点即为成员的名不算新纳入。
- 参数：`membership`、`lag_days`、`exclude_recent_days`

### 3.3 `ListingAgeFilter`

- 实现状态：已实现，两端。上市日计为第 0 天，边界恰为上市后第 N 个交易日；缺上市日期即剔除。
- 参数：`listing_dates`、`min_days`

### 3.4 `StatusExclusionFilter`

- 实现状态：已实现，两端。稀疏异常态面板，按精确日期查找而不做 as-of 回填，避免过期状态被复用。
- 参数：`statuses`、`excluded_statuses`

### 3.5 `PriceLimitTradabilityFilter`

- 实现状态：已实现，两端。单掩码保守子集，触及涨跌停价即剔除（容差 1e-9），任一排除侧缺价或缺停牌价也剔除；买卖不对称掩码不在本引擎。
- 参数：`limit_up`、`limit_down`、`exclude_limit_up`、`exclude_limit_down`

### 3.6 `LiquidityFilter`

- 实现状态：已实现，两端。滚动均值 `min_periods=1`，日额与截面分位两条腿都含等号；`missing=exclude` 时全窗缺失即剔除。
- 参数：`amount`、`window`、`min_amount`、`quantile`、`missing`

### 3.7 `PriceRangeFilter`

- 实现状态：已实现，两端。区间双端含等号；窗口内缺价或日历史不足即剔除，读注入的原始收盘价。
- 参数：`min_price`、`max_price`、`consecutive_days`

### 3.8 `DataQualityFilter`

- 实现状态：已实现，两端。覆盖率与缺失游程以因子面板自身日期的并集网格计，游程恰等于上界仍通过；通用异常值规则没有与引擎无关的定义，未实现。
- 参数：`min_coverage`、`max_missing_run`、`window`、`non_positive_price`

## 4. 因子与信号变换算子

信号管线顺序固定，两端一致：值空间阶段（transform → winsorize → standardize →
missing_value → industry_neutralize → style_neutralize → 冻结方向乘法）在前，rank
空间阶段（percentile rank → neutral_rank 填充 → interaction）在后，最后 blend。
全部横截面统计只用信号日可得数据，有效样本数与缺失情况进报告。

### 4.1 `DirectionMap`

- 实现状态：已实现，两端。方向来自冻结的 registry 并随请求下发（`factor_directions`）；两端信号管线都做方向乘法，生产主链路由 worker 执行。
- 参数：无（方向取自 registry，不是可配置参数）

### 4.2 `CrossSectionalRank`

- 实现状态：已实现，仅 worker。截面百分位排名在 worker 侧计算，本地组合层消费的已是混合好的评分面板。
- 参数：无

### 4.3 `Winsorize`

- 实现状态：已实现，两端。逐日只用当日有效截面定分位边界；有效样本数不足时两端同消息 fail-closed。
- 参数：`winsorize_lower`、`winsorize_upper`、`winsorize_min_samples`

### 4.4 `CrossSectionalStandardize`

- 实现状态：已实现，两端。z-score / rank-z / robust z-score 三档；离散度接近零或非有限时 fail-closed，不产生无穷值。
- 参数：`standardize`、`standardize_min_std`

### 4.5 `MissingValuePolicy`

- 实现状态：已实现，两端。`drop_stock` 取「任一缺失即剔除该名」的保守读法；填充超过比例上限时两端同消息 fail-closed。
- 参数：`missing_value`、`missing_max_fill_ratio`

### 4.6 `IndustryNeutralize`

- 实现状态：已实现，两端。回归法为 Frisch-Waugh 残差；某日某组有效值少于 `min_group` 时该组原值穿过，属声明式 fallback 故不 fail-closed。
- 参数：`industry_neutralize`、`industry_neutralize_industry_map`、`industry_neutralize_method`、`industry_neutralize_min_group`、`industry_neutralize_version`、`industry_neutralize_exposures`

### 4.7 `StyleNeutralize`

- 实现状态：已实现，两端。逐日对注入的暴露列回归取残差；联合有效行数不足时整日原值穿过，缺任一暴露的名保持原值。
- 参数：`style_neutralize`、`style_neutralize_exposures`、`style_neutralize_min_samples`

### 4.8 `FactorTransform`

- 实现状态：已实现，两端。只做单调变换（linear / log / signed_log / sqrt），越域值 fail-closed，绝不静默修补。
- 参数：`transform`、`transform_scale`、`transform_shift`

### 4.9 `FactorBlend`

- 实现状态：已实现，两端。固定权重按和归一化，缺任一成员即整格缺失；默认等权路径与引入固定权重之前逐位相同。
- 参数：`blend_weights`

### 4.10 `FactorGroupBlend`

- 实现状态：已实现，两端。组必须精确划分候选因子；与平铺的 `blend_weights` 互斥。
- 参数：`blend_groups`、`blend_within_weights`、`blend_group_weights`

### 4.11 `FactorInteraction`

- 实现状态：已实现，两端。输入取两端最终的 rank 面板（保守的 rank-first 读法），上限 8 项；文档里的门控交互未实现。
- 参数：`interaction_pairs`

### 4.12 `RegimeConditionalSignal`

- 实现状态：已实现，两端。逐决策日按 as-of 查找注入的状态标签；状态查不到或标签未定义时两端同消息 fail-closed。
- 参数：`regime_states`、`regime_factor_groups`、`regime_lag`

## 5. 选择与排序算子

选择层在评分截面上工作，产出被选中的名。门控不达标的名在评分中为 NaN，下游
`dropna` 自然衔接。Top-N 与并列排序是选择层的确定性基础。

### 5.1 `QuantileGate`

- 实现状态：已实现，两端。在单个候选因子的方向后百分位上做单层门控，平均名次上的含等号比较是并列规则。
- 参数：`factor`、`direction`、`quantile`、`min_pass`

### 5.2 `CascadedFactorGating`

- 实现状态：已实现，两端。层间只有 and 语义（or 未实现），不达标的名在评分中置 NaN；最终 Top-K 仍由 `CompositionSpec.top_n` 决定。
- 参数：`layers`（每层为 §5.1 的四键）

### 5.3 `TopKSelector`

- 实现状态：已实现，两端。两端都取确定性排序后的 `head(top_n)`。
- 参数：`top_n`

### 5.4 `RankTieBreaker`

- 实现状态：已实现，两端。分数并列时按 symbol 升序做稳定归并排序。
- 参数：无（排序规则固定，不是参数）

### 5.5 `HoldingBuffer`

- 实现状态：已实现，两端。百分位从顶部计，持有名在自然 top_n 内必留，retain 只救掉出 top_n 的名；需要上期执行后仍持有的信息，故运行在回测主循环内。
- 参数：`holding_buffer_entry`、`holding_buffer_retain`、`holding_buffer_max_extra`

### 5.6 `DiversitySelector`

- 实现状态：已实现，两端。硬约束的贪心最大相关屏筛，相关用决策日已实现的日收益；按行业的最少与最多持仓数因两端都无行业标签数据，未实现。
- 参数：`diversity_max_correlation`

## 6. 行业、风格与组合约束算子

约束链在一行目标权重上依次施加，顺序固定为单名 cap → 行业 cap → 名次下限 floor →
联合投影 → 现金残差检查，两端同位置调用。约束的判定对象是**目标权重**，漂移后的
start-of-day 权重可以越界，这是有意的（唯一例外见 §8.3）。约束不改因子分数。

### 6.1 `SingleNameCap`

- 实现状态：已实现，两端。确定性 water-filling；`K × cap < 1` 时余量留作现金而不超限。
- 参数：`weight_cap`

### 6.2 `SingleNameFloor`

- 实现状态：已实现，两端。在 cap 之后作用于目标权重；floor × K 超预算直接 fail-closed，最小交易金额参数在权重层没有对应物，未实现。
- 参数：`weight_floor`、`weight_floor_allow_cash`

### 6.3 `IndustryWeightCap`

- 实现状态：已实现，两端。cap 之后、floor 之前；每轮冻结全部超限行业再分配，必收敛；映射未覆盖的名共享保留组 `__no_industry__`。
- 参数：`industry_weight_cap`、`industry_map`

### 6.4 `IndustryDeviationConstraint`

- 实现状态：已实现，两端。在联合投影阶段内约束每行业主动偏离；豁免是显式行业清单且必须出现在偏离上界里。
- 参数：`industry_deviation_max`、`industry_deviation_industry_map`、`industry_deviation_exempt`、`industry_deviation_benchmark`

### 6.5 `StyleExposureConstraint`

- 实现状态：已实现，两端。暴露矩阵与基准缺名即 fail-closed；约束优先级与软告警未实现，全部约束都是硬约束。
- 参数：`style_exposure_bounds`、`style_exposure_matrix`、`style_exposure_benchmark`

### 6.6 `BetaConstraint`

- 实现状态：已实现，两端。取上下限这一读法把 beta 约束在绝对区间；文档的降级策略按保守读法换成联合投影 fail-closed。
- 参数：`beta_constraint_min`、`beta_constraint_max`、`beta_by_symbol`

### 6.7 `FactorExposureConstraint`

- 实现状态：已实现，两端。对注入的暴露矩阵取绝对区间，不带基准；软约束惩罚系数未实现。
- 参数：`factor_exposure_bounds`、`factor_exposure_matrix`

### 6.8 `CashResidualPolicy`

- 实现状态：已实现，两端。为既有的留现金约定加上显式上界，超出即执行期 fail-closed；残差再分配与再分配顺序在本引擎不可实现，未实现。
- 参数：`cash_residual_max`

### 6.9 `ConstrainedWeightProjection`

- 实现状态：已实现，两端。等权 L2 距离、Dykstra 交替投影、固定约束顺序、无优先级，不可行或不收敛都 fail-closed；`projection_tolerance` 只决定停止精度，可行性判据用固定浮点噪声上界，放宽容差只会把可行跑成保守失败。
- 参数：`projection_max_iter`、`projection_tolerance`

## 7. 权重生成算子

配权档由 `CompositionSpec.weighting` 选择，返回一行目标权重；约束链在其后照常生效。
需要收益历史或风险模型注入的档位会相应增加必要输入，缺输入即 fail-closed。

### 7.1 `EqualWeight`

- 实现状态：已实现，两端。选中名等权。
- 参数：无

### 7.2 `RankWeight`

- 实现状态：已实现，两端。按名次线性递减，最优点权重最大。
- 参数：无

### 7.3 `ScoreProportionalWeight`

- 实现状态：已实现，两端。非正分数两端 fail-closed，不做隐式做空。
- 参数：无

### 7.4 `ScoreTransformWeight`

- 实现状态：已实现，两端。linear 与 sigmoid 要求变换后逐名严格为正；truncated 把不高于 cut 的分截断为零权重，该名仍被 top_n 选中但不持仓。
- 参数：`score_transform_kind`、`score_transform_slope`、`score_transform_cut`

### 7.5 `InverseVolatilityWeight`

- 实现状态：已实现，两端。面板不足两行整行等权回退（声明式），面板够而单名有效观测不足或方差非正则两端同消息 fail-closed；主路径与消融路径传同一收益面板。
- 参数：`inverse_vol_window`、`inverse_vol_min`、`inverse_vol_multiply_score`

### 7.6 `RiskParityWeight`

- 实现状态：已实现，两端。协方差由决策日已实现收益估计，`diagonal` 朝样本协方差自身的对角线收缩；求解是确定性 CCD 固定点，不收敛 fail-closed。
- 参数：`risk_parity_window`、`risk_parity_budgets`、`risk_parity_shrinkage_kind`、`risk_parity_shrinkage`、`risk_parity_tolerance`

### 7.7 `FactorRiskBudgetWeight`

- 实现状态：已实现，两端。因子空间用同一 CCD 求预算，再经注入的暴露矩阵映到个股，负投影被 relu 截断（全非正即 fail-closed）；特异风险不进预算目标，只经剔除上限起作用。
- 参数：`frb_budgets`、`frb_exposures`、`frb_factor_covariance`、`frb_idiosyncratic`、`frb_idio_max`、`frb_ridge`

### 7.8 `HierarchicalWeighting`

- 实现状态：已实现，两端。组间权重只对持有入选名的活动组归一化，组内等分；入选名不在任何组即两端同消息 fail-closed。
- 参数：`hierarchical_groups`、`hierarchical_group_weights`

## 8. 调仓与交易可行性算子

回测主循环内的次序固定：目标行（或选择层重建）→ 风险缩放 → 漂移带判定 → 换手收缩
→ 执行层（参与率或分阶段 → 侧向掩码 → 整手）→ 最低持有期锁。价格、成交量、停牌与
涨跌停读的是执行日而不是决策日的状态；成本公式不参与以上任何一步，换手计量原样
作用于成交的权重差。

### 8.1 `RebalanceSchedule`

- 实现状态：已实现，两端。日历推导只在 worker，本地在请求的 `evaluation` 节声明频率。
- 参数：`evaluation.rebalance_frequency`

### 8.2 `SignalLag`

- 实现状态：已实现，两端。系统下限为 1 个交易日，`BacktestSpec` 与 worker 校验都强制。
- 参数：`signal_lag`

### 8.3 `DriftBandRebalance`

- 实现状态：已实现，两端。band 比较读漂移后的 carried 权重对目标权重，这是目标权重判定规则的唯一显式例外；band 内的调仓彻底作废，不补发。
- 参数：`drift_band_name`、`drift_band_portfolio`、`drift_band_forced_days`

### 8.4 `MinimumHoldingPeriod`

- 实现状态：已实现，两端。锁是最后一道执行否决，位于漂移带与换手收缩之后；资金不足时整笔放弃该日调仓并保持 carried，强制退出例外没有引擎内触发定义，未实现。
- 参数：`min_holding_days`

### 8.5 `TurnoverShrinkage`

- 实现状态：已实现，两端。soft 收缩整个权重差向量，hard clip 以含现金的半 L1 换手为口径，两者可叠加；收缩改执行不改成本公式。
- 参数：`turnover_shrinkage`、`turnover_max_step`

### 8.6 `ParticipationCap`

- 实现状态：已实现，两端。未成交余量成为显式 pending 计划，`carry` 为默认并按 `participation_max_days` 老化后取消；需要 `execution_data.trading_dates`，与 §8.9 互斥。
- 参数：`participation_max_rate`、`participation_max_days`、`participation_unfilled`、`participation_amount_window`

### 8.7 `LotRounding`

- 实现状态：已实现，两端。用执行日原始收盘价与注入的整手数取整，资本基准固定为 1.0；`down`（默认）保证取整后金额不超过目标。
- 参数：`lot_rounding`、`lot_min_notional`、`lot_rounding_mode`

### 8.8 `SideAwareTradability`

- 实现状态：已实现，两端。`block_buy` 与 `block_sell` 是单侧变体而非均匀收紧；掩码只约束当日执行切片，永远不改 carried，被挡的腿直接取消、不排队。
- 参数：`side_aware_tradability`

### 8.9 `StagedExecution`

- 实现状态：已实现，两端。确定性多日订单计划，服务顺序在计划创建时固定；需要 `execution_data.trading_dates`，与 §8.6 互斥。
- 参数：`staged_max_days`、`staged_priority`

### 8.10 `FixedExecutionCost`

- 实现状态：已实现，两端。固定 10/10 bps 买卖契约，任何位置都没有可选的成本字段。
- 参数：无（成本不是算子参数）

## 9. 风险控制与压力监控算子

风险算子里有一类会改动权重（仓位缩放与衰减缩放都只读决策日之前已实现的量），另一类
只写报告。熔断与减仓是预先冻结的有限状态机中的明确转移，不接受任意动态脚本。

### 9.1 `PortfolioVolatilityTarget`

- 实现状态：已实现，两端。按目标年化波动率对已实现年化波动率的比值缩放整行目标并钳到区间；有效观测不足两天按原值穿过，组合层面零波动率取上界而非 fail-closed。
- 参数：`vol_target_annual`、`vol_target_window`、`vol_target_min_scale`、`vol_target_max_scale`

### 9.2 `DrawdownStateMachine`

- 实现状态：已实现，两端。三态机的阈值严格递增，升档与降档都需连续确认，降档按最深处收窄度量；两个缩放器取 min，故熔断压得住波动率缩放。
- 参数：`drawdown_thresholds`、`drawdown_confirm_days`、`drawdown_recovery`、`drawdown_scales`

### 9.3 `StressScenarioCheck`

- 实现状态：已实现，两端。在执行后权重面板最后一行上做总收益冲击，`warn` 只写报告、`reject` 让整次审计 fail-closed 但不改任何权重；报告回显情景表版本。
- 参数：`scenario_table_version`、`scenarios`、`loss_threshold`、`mode`、`industry_map`、`scenario_ids`

### 9.4 `ConcentrationMonitor`

- 实现状态：已实现，两端。逐日 HHI、有效持仓数与 top-k 份额；行业阈值需与行业映射成对提供，破限判定为严格大于。纯报告，权重永不重写。
- 参数：`top_k`、`industry_map`、`industry_threshold`

### 9.5 `FactorDecayMonitor`

- 实现状态：已实现，两端。滚动 rank IC 只数尾部连续失败期，IC 严格低于阈值才算失败；触发时缩放该因子的混合贡献而不替换因子，两参消融调用不消费该配置。
- 参数：`decay_window`、`decay_fail_periods`、`decay_ic_threshold`、`decay_scale`

## 10. 评价、审计与治理算子

审计算子分三类：读已执行结果的报告器（§9.3、§9.4、§10.2–§10.4、§10.6、§10.7）、
按段重跑引擎的评估器（§10.1）、以及不重跑也不改组合行为的治理面（§10.8–§10.10）。
报告器与治理面都保证带与不带该节时执行路径逐位相同。

### 10.1 `WalkForwardEvaluation`

- 实现状态：已实现，两端。唯一按段重跑引擎的审计，配置挂在请求级 `walk_forward` 对象；报告显式声明为事后描述，主路径的产物与指标不受影响。
- 参数：`train_size`、`valid_size`、`test_size`、`step_size`、`expanding_train`、`min_oos_segments`、`max_segments`

### 10.2 `CostSensitivityAudit`

- 实现状态：已实现，两端。复用主回测已产出的 gross 与买卖换手序列重定价，不重跑策略；生产情景与生产指标逐位一致。
- 参数：`scenarios`

### 10.3 `ParameterPerturbationAudit`

- 实现状态：已实现，两端。扰动键是 canonical payload 的点分路径，仅限四个节；键不在 payload 即整审计 fail-closed，无效扰动值记为错误变体。
- 参数：`perturbations`

### 10.4 `AblationAudit`

- 实现状态：已实现，两端。既有非门控两参消融循环的报告化，边际四式与 legacy 输出逐位相等；算子级消融因引擎无禁用开关，未实现。
- 参数：`mode`（当前只接受 `factors`）

### 10.5 `ReturnProfileDedup`

- 实现状态：已实现，仅本地。独立 worker 不做去重，只有本地验收流程比对收益轨迹。
- 参数：阈值与最小重叠长度是去重器的构造参数 `threshold`、`min_overlap`，不是候选 payload 字段

### 10.6 `FeasibilityAudit`

- 实现状态：已实现，两端。逐日报告理论额与成交额并归因到执行链各阶段，账面来自仅在请求该节时才收集的观察汇，执行路径带与不带逐位相同。
- 参数：`min_executable_ratio`、`allow_cash_residual`

### 10.7 `DataLeakageAudit`

- 实现状态：已实现，两端。比对候选 payload 的时点声明与注入的字段可用性规则，产出带严重度的前视风险清单；未声明的字段报为 unknown，既不静默放行也不 fail-closed。
- 参数：`field_availability_lag`、`announcement_to_effective`

### 10.8 `StrategyManifestEmitter`

- 实现状态：已实现，两端。请求级 `manifest` 对象不接受任何设置项，任何键都 fail-closed，因为文档列出的六项全是待采集的事实；硬身份是 worker 文件摘要加运行栈，同名不同内容拒绝覆盖。
- 参数：无（`manifest` 请求对象为空白名单）

### 10.9 `ReproducibilityCheck`

- 实现状态：已实现，仅本地。没有请求面也没有 worker 镜像，做的是身份比对而非重跑，因此浮点容差参数不予接受；任一字段缺席即 fail-closed。
- 参数：无（键白名单为空集，比对字段集是模块常量）

### 10.10 `DeploymentGate`

- 实现状态：已实现，仅本地。消费既有报告给出部署档位，策略无任何数值门槛，收益完全不是输入。
- 参数：系统侧 `DeploymentPolicy.required_evidence` 与 `notes`；策略不可设置

## 11. 复杂组合示例

下面是一个当前执行器可以表达的复杂组合，字段名与落点都按实现了的算子写出。
参数值只作示例。

```text
注入数据面（universe_rules，payload 节）
  trading_dates      交易日历
  price              原始收盘价面板
  index_membership   {日期: {symbol: 是否成分}}，lag_days=1，
                     exclude_recent_days=20
  上市年龄            listing_age.listing_dates + min_days=120
  异常状态            status_exclusion.statuses + excluded_statuses=['ST','ST*','退市整理']
  涨跌停              price_limit.limit_up / limit_down，
                     exclude_limit_up=True、exclude_limit_down=True
  流动性              liquidity.amount + window=20、quantile=0.3、missing='exclude'
  价格区间            price_range.min_price=2.0、consecutive_days=5
  数据质量            data_quality.min_coverage=0.8、max_missing_run=10、window=20

信号（signal）
  变换与清洗          transform='signed_log'，
                     winsorize_lower=0.01、winsorize_upper=0.99、winsorize_min_samples=20
                     standardize='robust_z'
  缺失                missing_value='cross_section_median'
  行业中性化          industry_neutralize=True、industry_neutralize_method='regression'、
                     industry_neutralize_industry_map（注入）、min_group=8
  风格中性化          style_neutralize=True、style_neutralize_exposures（注入）、min_samples=30
  两级分组混合        blend_groups={'value':[...], 'quality':[...], 'risk':[...], 'price_volume':[...]}、
                     blend_group_weights={'value':0.3,'quality':0.3,'risk':0.25,'price_volume':0.15}
  交互项              interaction_pairs=[('quality_score','value_score','product')]
  状态条件            regime_states（注入 PIT 标签）、regime_factor_groups、regime_lag=1
                     （与 blend_groups / blend_weights / interaction_pairs 互斥，
                       上两行按 §4.12 的用法单独成立）

门控（gating.layers，层间 and）
  {factor:'quality_score', direction:'top', quantile:0.6, min_pass:30}
  {factor:'liquidity_score', direction:'top', quantile:0.7}
  {factor:'risk_score', direction:'top', quantile:0.65}

选择（composition）
  top_n=80
  holding_buffer_entry=0.4、holding_buffer_retain=0.3、holding_buffer_max_extra=20
  diversity_max_correlation=0.85

配权与约束（composition）
  weighting='risk_parity'、risk_parity_window=60、risk_parity_shrinkage_kind='diagonal'
  weight_cap=0.025
  industry_weight_cap=0.2、industry_map（注入）
  weight_floor=0.002、weight_floor_allow_cash=True
  industry_deviation_max={'银行':0.05,...}、industry_deviation_benchmark（注入）
  beta_constraint_min=0.85、beta_constraint_max=1.15、beta_by_symbol（注入）
  factor_exposure_bounds={'value':[-0.2,0.2]}、factor_exposure_matrix（注入）
  projection_max_iter=512、projection_tolerance=1e-9
  cash_residual_max=0.05

执行与回测（backtest）
  signal_lag=1
  drift_band_name=0.015、drift_band_forced_days=60
  turnover_shrinkage=0.3、turnover_max_step=0.5
  min_holding_days=5
  participation_max_rate=0.1、participation_amount_window=20、
  participation_unfilled='carry'、participation_max_days=3
  lot_rounding=True、lot_rounding_mode='down'、lot_min_notional=5000
  side_aware_tradability='block_both'
  vol_target_annual=0.15、vol_target_window=60、vol_target_min_scale=0.4、vol_target_max_scale=1.2
  drawdown_thresholds={'warning':0.05,'reduce':0.10,'halt':0.18}、drawdown_confirm_days=3、
  drawdown_recovery=0.5、drawdown_scales={'normal':1.0,'warning':0.7,'reduce':0.4,'halt':0.0}

执行数据面（execution_data）
  trading_dates、price、amount、lot_sizes、halt、price_limits
  （全部显式注入，与 universe_rules 的日期各成一套）

请求级配置（不进 candidate payload）
  evaluation     {start, end, rebalance_frequency:'weekly'}
  audit          {concentration:{top_k:[5,10]}, cost_sensitivity:{scenarios:[[10,10],[15,15],[20,20]]},
                  parameter_perturbation:{perturbations:{'composition.top_n':[60,80,100]}},
                  ablation:{mode:'factors'}, feasibility:{min_executable_ratio:0.9},
                  data_leakage:{field_availability_lag:{...}, announcement_to_effective:{...}},
                  stress:{scenario_table_version:'v3', scenarios:{...}, loss_threshold:0.2, mode:'warn'}}
  walk_forward   {train_size:250, valid_size:60, test_size:120, min_oos_segments:2, max_segments:4}
  manifest       {}（空白名单，任何键都 fail-closed）
```

这个组合里模型能声明的是 weighting 档、`top_n`、各阈值、带宽、收缩比例这类离散或
账户级参数；注入的面板、映射、暴露矩阵与基准由上层配置提供，模型无从给出。系统拒绝
自定义代码与数学表达式、修改成本与滞后下限、引用冻结 working set 之外的因子、使用
未来数据、以及在无降级规则时把无解的约束悄悄放开。

## 12. schema 分层

一个平面 JSON 不承载所有语义。实际分层如下，每层的键集与校验由对应 spec 负责。

```text
request（远端请求对象）
├── schema_version / snapshot_id / snapshot_checksum
├── working_set_id / working_set_snapshot_checksum
├── artifact_set                  # 因子产物清单与 checksum
├── factor_directions             # 冻结方向，逐因子 ±1
├── execution_costs               # 系统固定 10/10 bps，不可配置
├── evaluation                    # 评价窗口与调仓频率
├── base_path / close_field / portfolio_artifact_path   # 数据面接线
├── candidate                     # 下列各节
│   ├── universe_rules            # §3 候选池，整节缺席即关闭
│   ├── signal                    # §4 信号 DAG（值空间 + rank 空间 + blend + 中性化 + regime + decay）
│   ├── gating                    # §5 门控层
│   ├── composition               # §5 选择、§6 约束链、§7 配权档
│   ├── backtest                  # §8 执行层与 §9.1/§9.2 风险缩放
│   ├── execution_data            # §8.6–§8.9 的市场面板，整节缺席即关闭
│   └── name / universe / factor_names / rule / hypothesis / metadata
├── audit                         # 报告器与压力情景，不进 candidate
├── walk_forward                  # 按段重跑引擎的评估配置
└── manifest                      # 空白名单，只请求 worker 回填证明
```

分层的实际约束：

- 每一层只读规定的数据类型，候选池只产出 `eligible` 掩码，信号只产出因子与评分
  面板，选择层只产出选中集合，配权只产出目标权重，约束链只把目标权重投影到可行域，
  执行层只产出成交与残差。跨层写入不在实现里存在。
- `audit`、`walk_forward`、`manifest` 三节缺席时逐位走老路；它们不参与
  `canonical_payload()`，也不进 `strategy_id`。
- `universe_rules` 与 `execution_data` 各自带一套交易日历：前者服务选股池的时点
  校验，后者服务执行日的状态与成交量读取，两者互不代表。
- 逐 pipeline 的 schema checksum 在当前引擎里没有分层 schema 对象，未实现；身份
  由 canonical payload 哈希加 manifest 的硬身份共同承担。
