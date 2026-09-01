# 米筐 RQData 基准财务因子清单与计算公式

## 1. 因子体系总览

米筐 RQData 的基准财务因子按"估值、经营、现金流、财务结构、成长性"5 大类组织，同时提供 LF（最新报告）、LYR（最新年报）、TTM（滚动 12 个月）三套处理逻辑。所有因子均通过 PIT（point-in-time）流程回填，避免前视偏差。三大表基础字段 200+ 个，衍生指标 500+ 个，盈利与质量类指标都经过米筐自动化清洗与人工核对。

取数接口：A 股衍生指标统一用 `get_factor(order_book_ids, factor, start_date, end_date)`，PIT 三表用 `get_pit_financials_ex`，港股市场设 `market='hk'`。所有命名规则与 Wind 的 eod_indicator / financial_indicator 体系基本对齐，可作为 Alpha158 之外的"基准财务面"补强数据源。

## 2. 估值与市值类

估值因子作为日频动态指标，每天随股价更新，覆盖多口径 PE、PB、PS、PCF 以及扣货币资金的企业价值倍数。`market_cap_3` 专门采用 PIT 处理口径，公告披露后才调整股本，回避停牌前后的虚高估值。

### 字段清单

| 因子 | 中文名 | 口径 |
|---|---|---|
| `pe_ratio_lyr` | 市盈率 LYR | 静态 |
| `pe_ratio_ttm` | 市盈率 TTM | 滚动 12 个月 |
| `pb_ratio_lyr/ttm/lf` | 市净率 | 三套口径 |
| `pb_ratio_1_*` | 市净率（剔除其他权益工具） | 三套口径 |
| `ps_ratio_lyr/ttm` | 市销率 | LYR / TTM |
| `sp_ratio_lyr/ttm` | 销售收益率 | LYR / TTM |
| `pcf_ratio_total_*` | 市现率（总现金流口径） | LYR / TTM |
| `pcf_ratio_*` | 市现率（经营现金流口径） | LYR / TTM |
| `cfp_ratio_*` | 现金收益率 | LYR / TTM |
| `book_to_market_ratio_*` | 账面市值比 | LYR / TTM / LF |
| `dividend_yield_ttm` | 股息率 TTM | 滚动 12 个月 |
| `peg_ratio_lyr/ttm` | PEG | LYR / TTM |
| `ev_lyr/ttm/lf` | 企业价值 | 三套口径 |
| `ev_no_cash_*` | 企业价值（不含货币资金） | 三套口径 |
| `ev_to_ebitda_*` | EV/EBITDA 倍数 | LYR / TTM |
| `ev_no_cash_to_ebit_*` | EV（不含货币资金）/EBIT | LYR / TTM |
| `market_cap` | 总市值 | 旧口径 |
| `market_cap_2` | 流通股总市值 | 不支持科创板 |
| `market_cap_3` | 总市值（PIT） | 推荐使用 |
| `a_share_market_val_3` | A 股市值（PIT） | 推荐使用 |
| `a_share_market_val_in_circulation` | 流通 A 股市值（PIT） | |

### 计算公式

| 因子 | 公式 |
|---|---|
| `pe_ratio_lyr` | `market_cap_3 / net_profit_parent_company_lyr_0` |
| `pe_ratio_ttm` | `market_cap_3 / net_profit_parent_company_ttm_0` |
| `pb_ratio_lyr/ttm/lf` | `market_cap_3 / equity_parent_company_*_0` |
| `pb_ratio_1_*` | `market_cap_3 / (equity_parent_company_* - other_equity_instruments_*)` |
| `ps_ratio_lyr/ttm` | `market_cap_3 / operating_revenue_*_0` |
| `pcf_ratio_total_*` | `market_cap_3 / (cf_op_* + cf_inv_* + cf_fin_*)` |
| `pcf_ratio_*` | `market_cap_3 / cash_flow_from_operating_activities_*` |
| `book_to_market_ratio_*` | `equity_parent_company_* / market_cap_3` |
| `ev_lyr/ttm/lf` | `market_cap_3 + total_liabilities_*` |
| `ev_no_cash_*` | `ev_* - cash_equivalent_*` |
| `ev_to_ebitda_*` | `ev_* / ebitda_*` |
| `peg_ratio_*` | `pe_ratio_* / 100 * (np_parent_*_0 - np_parent_*_1) / np_parent_*_1`（TTM 用 `_0 / _4`） |
| `dividend_yield_ttm` | `dividend_ttm / close_price` |
| `market_cap_3` | `total * close_price`（PIT 处理） |

## 3. 盈利能力类

按"每股、盈利能力、利润率、费用率、杜邦分解"展开，每项同时提供 LYR 与 TTM 两套口径。ROE 同时提供普通/扣非/摊薄三种口径；杜邦把 ROE 拆成净利率×总资产周转率×权益乘数。

### 字段清单

**每股指标**

| 因子 | 中文名 |
|---|---|
| `diluted_earnings_per_share_lyr/ttm` | 摊薄每股收益 |
| `adjusted_earnings_per_share_lyr/ttm` | 基本每股收益（扣非） |
| `adjusted_fully_diluted_earnings_per_share_lyr/ttm` | 稀释每股收益（扣非） |
| `operating_total_revenue_per_share_lyr/ttm` | 每股营业总收入 |
| `operating_revenue_per_share_lyr/ttm` | 每股营业收入 |
| `ebit_per_share_lyr/ttm` | 每股息税前利润 |

**盈利能力比率**

| 因子 | 中文名 |
|---|---|
| `return_on_equity_lyr/ttm` | 净资产收益率（平均余额法） |
| `return_on_equity_diluted_lyr/ttm` | 摊薄净资产收益率 |
| `adjusted_return_on_equity_lyr/ttm` | 净资产收益率（扣非） |
| `adjusted_return_on_equity_diluted_lyr/ttm` | 摊薄净资产收益率（扣非） |
| `return_on_asset_lyr/ttm` | 总资产报酬率（EBIT 口径） |
| `return_on_asset_net_profit_lyr/ttm` | 总资产净利率 |
| `return_on_invested_capital_lyr/ttm` | 投入资本回报率 |
| `ebit_lyr/ttm` | 息税前利润 |
| `ebitda_lyr/ttm` | 息税折旧摊销前利润 |
| `time_interest_earned_ratio_lyr/ttm` | 利息保障倍数 |

**利润率**

| 因子 | 中文名 |
|---|---|
| `gross_profit_margin_lyr/ttm` | 销售毛利率 |
| `net_profit_margin_lyr/ttm` | 销售净利率 |
| `cost_to_sales_lyr/ttm` | 销售成本率 |
| `net_profit_to_revenue_lyr/ttm` | 经营净利率 |
| `profit_from_operation_to_revenue_lyr/ttm` | 营业利润率 |
| `ebit_to_revenue_lyr/ttm` | 税前收益率 |
| `expense_to_revenue_lyr/ttm` | 经营成本率 |
| `operating_profit_to_profit_before_tax_lyr/ttm` | 经营活动净收益与利润总额之比 |
| `investment_profit_to_profit_before_tax_lyr/ttm` | 价值变动净收益与利润总额之比 |
| `non_operating_profit_to_profit_before_tax_lyr/ttm` | 营业外收支净额与利润总额之比 |
| `income_tax_to_profit_before_tax_lyr/ttm` | 所得税与利润总额之比 |
| `adjusted_profit_to_total_profit_lyr/ttm` | 扣非净利润与净利润之比 |
| `ebitda_to_debt_lyr/ttm` | EBITDA / 负债总计 |
| `income_from_main_operations_lyr/ttm` | 主营业务利润 |

**杜邦分解**

| 因子 | 中文名 |
|---|---|
| `du_profit_margin_lyr/ttm` | 净利率（杜邦分析） |
| `du_return_on_equity_lyr/ttm` | 净资产收益率（杜邦分析） |
| `du_equity_multiplier_lyr/ttm/lf` | 权益乘数（杜邦分析） |
| `du_return_on_sales_lyr/ttm` | 息税前利润 / 营业总收入 |

### 计算公式

| 因子 | 公式 |
|---|---|
| `return_on_equity_lyr` | `net_profit_parent_company_lyr_0 * 2 / (equity_parent_company_lyr_0 + equity_parent_company_lyr_1)` |
| `return_on_equity_diluted_lyr` | `net_profit_parent_company_lyr_0 / equity_parent_company_lyr_0` |
| `adjusted_return_on_equity_*` | `(net_profit_parent_company_* - non_recurring_pnl_*) * 2 / (equity_parent_company_*_0 + equity_parent_company_*_1)` |
| `return_on_asset_*` | `ebit_* / total_assets_*` |
| `return_on_asset_net_profit_*` | `net_profit_* / total_assets_*` |
| `return_on_invested_capital_*` | `(net_profit_* + financing_expense_*) / (total_assets_* - current_liabilities_* + notes_payable_* + short_term_loans_* + non_current_liability_due_one_year_*)` |
| `gross_profit_margin_*` | `(operating_revenue_* - cost_of_goods_sold_*) / operating_revenue_*` |
| `net_profit_margin_*` | `net_profit_* / operating_revenue_*` |
| `cost_to_sales_*` | `cost_of_goods_sold_* / operating_revenue_*` |
| `expense_to_revenue_*` | `total_expense_* / revenue_*` |
| `du_return_on_equity_*` | `du_profit_margin_* * total_asset_turnover_* * du_equity_multiplier_*` |
| `du_equity_multiplier_*` | `(total_assets_*_0 + total_assets_*_1) / (total_equity_*_0 + total_equity_*_1)` |
| `ebit_*` | `profit_before_tax_* + financing_interest_expense_* - financing_interest_income_*` |
| `ebitda_*` | `ebit_* + fixed_asset_depreciation_* + intangible_asset_amortization_* + deferred_expense_amort_*` |
| `diluted_earnings_per_share_*` | `net_profit_parent_company_* / total_shares` |
| `adjusted_earnings_per_share_*` | `adjusted_net_profit_* / weighted_common_stock_*` |

## 4. 运营效率与周转类

周转类因子统一采用"营收/平均余额"或"成本/平均余额"的形式，给出 LYR 与 TTM 两套口径。"营业周期"和"现金转换周期"由已有周转率/天数组合计算得到。

### 字段清单

| 因子 | 中文名 |
|---|---|
| `inventory_turnover_lyr/ttm` | 存货周转率 |
| `account_receivable_turnover_rate_lyr/ttm` | 应收账款周转率 |
| `account_receivable_turnover_days_lyr/ttm` | 应收账款周转天数 |
| `account_payable_turnover_rate_lyr/ttm` | 应付账款周转率 |
| `account_payable_turnover_days_lyr/ttm` | 应付账款周转天数 |
| `current_asset_turnover_lyr/ttm` | 流动资产周转率 |
| `fixed_asset_turnover_lyr/ttm` | 固定资产周转率 |
| `total_asset_turnover_lyr/ttm` | 总资产周转率 |
| `equity_turnover_ratio_lyr/ttm` | 股东权益周转率 |
| `operating_cycle_lyr/ttm` | 营业周期 |
| `average_payment_period_lyr/ttm` | 应付账款付款期 |
| `cash_conversion_cycle_lyr/ttm` | 现金转换周期 |

### 计算公式

| 因子 | 公式 |
|---|---|
| `inventory_turnover_*` | `cost_of_goods_sold_* / inventory_*` |
| `account_receivable_turnover_rate_*` | `operating_revenue_* / net_accts_receivable_*` |
| `account_payable_turnover_rate_*` | `cost_of_goods_sold_* / accts_payable_*` |
| `current_asset_turnover_*` | `operating_revenue_* / current_assets_*` |
| `fixed_asset_turnover_*` | `operating_revenue_* / total_fixed_assets_*` |
| `total_asset_turnover_*` | `operating_revenue_* / total_assets_*` |
| `equity_turnover_ratio_*` | `operating_revenue_* / equity_parent_company_*` |
| `*_turnover_days_*` | `360 / *_turnover_rate_*` |
| `operating_cycle_*` | `account_receivable_turnover_days_* + 360 / inventory_turnover_*` |
| `cash_conversion_cycle_*` | `operating_cycle_* - average_payment_period_*` |

## 5. 现金流质量与自由现金流类

现金流因子分"每股 OCF、FCFF、FCFE、OCF 偿债能力比率"四组。FCFF 和 FCFE 都从归母净利润出发，加上折旧摊销、扣减营运资本变动。

### 字段清单

| 因子 | 中文名 |
|---|---|
| `cash_flow_per_share_lyr/ttm` | 每股现金流 |
| `operating_cash_flow_per_share_lyr/ttm` | 每股经营现金流 |
| `fcff_lyr/ttm` | 企业自由现金流量 |
| `fcfe_lyr/ttm` | 股权自由现金流量 |
| `free_cash_flow_company_per_share_lyr/ttm` | 每股企业自由现金流 |
| `free_cash_flow_equity_per_share_lyr/ttm` | 每股股东自由现金流 |
| `ocf_to_debt_lyr/ttm` | OCF / 负债合计 |
| `ocf_to_interest_bearing_debt_lyr/ttm` | OCF / 带息债务 |
| `ocf_to_current_ratio_lyr/ttm` | OCF / 流动负债 |
| `ocf_to_net_debt_lyr/ttm` | OCF / 净债务 |
| `surplus_cash_protection_multiples_lyr/ttm` | 盈余现金保障倍数 |
| `depreciation_and_amortization_lyr/ttm` | 当期计提折旧与摊销 |
| `cash_flow_ratio_lyr/ttm` | 现金流量比率 |

### 计算公式

| 因子 | 公式 |
|---|---|
| `operating_cash_flow_per_share_*` | `cash_flow_from_operating_activities_* / total_shares` |
| `cash_flow_per_share_*` | `(cf_op_* + cf_inv_* + cf_fin_*) / total_shares` |
| `ocf_to_debt_*` | `cash_flow_from_operating_activities_* / total_liabilities_*` |
| `ocf_to_interest_bearing_debt_*` | `cash_flow_from_operating_activities_* / interest_bearing_debt_*` |
| `surplus_cash_protection_multiples_*` | `cash_flow_from_operating_activities_* / net_profit_*` |
| `depreciation_and_amortization_*` | `fixed_asset_depreciation_* + intangible_asset_amortization_* + deferred_expense_amortization_*` |
| `fcff_*` | `np_parent_* + assets_depreciation_reserves_* + fixed_asset_depreciation_* + intangible_asset_amortization_* + deferred_expense_amortization_* + (financing_interest_expense_* - financing_interest_income_*) * (1 - income_tax_* / profit_before_tax_*) - (total_fixed_assets_*_0 - total_fixed_assets_*_1 + fixed_asset_depreciation_*) - Δ营运资本` |
| `fcfe_*` | `np_parent_* + assets_depreciation_reserves_* + fixed_asset_depreciation_* + intangible_asset_amortization_* + deferred_expense_amortization_* - (total_fixed_assets_*_0 - total_fixed_assets_*_1 + fixed_asset_depreciation_*) - Δ营运资本 + 净债务增加` |
| `free_cash_flow_company_per_share_*` | `fcff_* / total_shares` |
| `free_cash_flow_equity_per_share_*` | `fcfe_* / total_shares` |

> 营运资本变动 = 期末 `(current_assets - cash_equivalent) - (current_liabilities - notes_payable - non_current_liability_due_one_year)` − 期初同口径。净债务增加 = 期末 `(short_term_loans + long_term_loans + bond_payable)` − 期初同口径。

## 6. 偿债能力与资本结构类

核心思路是"先把总负债拆成带息债务和无息债务两部分，再组合出净债务、营运资本、净营运资本等中间量"。偿债比率区分 LF/LYR/TTM 三种处理逻辑，涉及利润表/现金流量表的派生指标只给 LYR 和 TTM。

### 字段清单

**偿债比率**

| 因子 | 中文名 |
|---|---|
| `debt_to_asset_ratio_lyr/ttm/lf` | 资产负债率 |
| `current_ratio_lyr/ttm/lf` | 流动比率 |
| `quick_ratio_lyr/ttm/lf` | 速动比率 |
| `super_quick_ratio_lyr/ttm/lf` | 超速动比率 |
| `cash_ratio_lyr/ttm/lf` | 现金比率 |
| `debt_to_equity_ratio_lyr/ttm/lf` | 产权比率 |
| `equity_to_debt_ratio_lyr/ttm/lf` | 权益负债比率 |
| `equity_to_interest_bearing_debt_lyr/ttm/lf` | 权益带息负债比率 |
| `long_term_debt_to_working_capital_lyr/ttm/lf` | 长期债务与营运资金比率 |
| `ocf_to_interest_bearing_debt_lyr/ttm` | OCF / 带息债务 |

**资本结构与债务拆分**

| 因子 | 中文名 |
|---|---|
| `equity_multiplier_lyr/ttm/lf` | 权益乘数 |
| `non_interest_bearing_current_debt_lyr/ttm/lf` | 无息流动负债 |
| `non_interest_bearing_non_current_debt_lyr/ttm/lf` | 无息非流动负债 |
| `interest_bearing_debt_lyr/ttm/lf` | 带息债务 |
| `net_debt_lyr/ttm/lf` | 净债务 |
| `working_capital_lyr/ttm/lf` | 营运资本 |
| `net_working_capital_lyr/ttm/lf` | 净营运资本 |
| `invested_capital_lyr/ttm/lf` | 全部投入资本 |
| `interest_bearing_debt_to_capital_lyr/ttm/lf` | 带息债务占企业全部投入成本的比重 |
| `current_debt_to_total_debt_lyr/ttm/lf` | 流动负债率 |
| `non_current_debt_to_total_debt_lyr/ttm/lf` | 非流动负债率 |
| `capital_to_equity_ratio_lyr/ttm/lf` | 长期资本固定比率 |
| `current_asset_to_total_asset_lyr/ttm/lf` | 流动资产比率 |
| `non_current_asset_to_total_asset_lyr/ttm/lf` | 非流动资产比率 |
| `equity_ratio_lyr/ttm/lf` | 股东权益比率 |
| `fixed_asset_ratio_lyr/ttm/lf` | 固定资产比率 |
| `intangible_asset_ratio_lyr/ttm/lf` | 无形资产比率 |
| `equity_fixed_asset_ratio_lyr/ttm/lf` | 股东权益与固定资产比率 |
| `book_leverage_lyr/ttm/lf` | 账面杠杆 |
| `market_leverage_lyr/ttm/lf` | 市场杠杆 |

**每股资产负债指标**

| 因子 | 中文名 |
|---|---|
| `book_value_per_share_lyr/ttm/lf` | 每股净资产 |
| `tangible_asset_per_share_lyr/ttm/lf` | 每股有形资产 |
| `liabilities_per_share_lyr/ttm/lf` | 每股负债 |
| `depreciation_per_share_lyr/ttm/lf` | 每股折旧和摊销 |
| `cash_equivalent_per_share_lyr/ttm/lf` | 每股货币资金余额 |
| `capital_reserve_per_share_lyr/ttm/lf` | 每股资本公积金 |
| `earned_reserve_per_share_lyr/ttm/lf` | 每股盈余公积金 |
| `undistributed_profit_per_share_lyr/ttm/lf` | 每股未分配利润 |
| `retained_earnings_lyr/ttm/lf` | 留存收益 |
| `retained_earnings_per_share_lyr/ttm/lf` | 每股留存收益 |

### 计算公式

| 因子 | 公式 |
|---|---|
| `debt_to_asset_ratio_*` | `total_liabilities_* / total_assets_*` |
| `current_ratio_*` | `current_assets_* / current_liabilities_*` |
| `quick_ratio_*` | `(current_assets_* - inventory_* - prepayment_* - deferred_expense_*) / current_liabilities_*` |
| `super_quick_ratio_*` | `(cash_equivalent_* + financial_asset_held_for_trading_* + bill_receivable_* + net_accts_receivable_* + other_accts_receivable_*) / current_liabilities_*` |
| `cash_ratio_*` | `cash_equivalent_* / current_liabilities_*` |
| `non_interest_bearing_current_debt_*` | `accts_payable_* + advance_from_customers_* + payroll_payable_* + tax_payable_* + other_payable_* + accrued_expense_* + deferred_income_* + other_current_liabilities_*` |
| `non_interest_bearing_non_current_debt_*` | `non_current_liabilities_* - long_term_loans_* - bond_payable_*` |
| `interest_bearing_debt_*` | `total_liabilities_* - non_interest_bearing_current_debt_* - non_interest_bearing_non_current_debt_*` |
| `net_debt_*` | `interest_bearing_debt_* - cash_equivalent_*` |
| `working_capital_*` | `current_assets_* - current_liabilities_*` |
| `net_working_capital_*` | `current_assets_* - cash_equivalent_* - non_interest_bearing_current_debt_*` |
| `equity_multiplier_*` | `total_assets_* / total_equity_*` |
| `debt_to_equity_ratio_*` | `total_liabilities_* / equity_parent_company_*` |
| `book_value_per_share_*` | `equity_parent_company_* / total_shares` |
| `capital_reserve_per_share_*` | `capital_reserve_* / total_shares` |

## 7. 成长性类

成长性因子的统一公式是"本期 / 上期 − 1"，上期时点按 LYR 用上一年年报，按 TTM 用上年同期的 TTM。资产负债表项额外提供 LF 口径。

### 字段清单

| 因子 | 中文名 |
|---|---|
| `inc_revenue_lyr/ttm` | 营业总收入同比增长率 |
| `operating_revenue_growth_ratio_lyr/ttm` | 营业收入同比增长率 |
| `operating_profit_growth_ratio_lyr/ttm` | 营业利润同比增长率 |
| `profit_growth_ratio_lyr/ttm` | 利润总额同比增长率 |
| `gross_profit_growth_ratio_lyr/ttm` | 毛利润同比增长率 |
| `net_profit_growth_ratio_lyr/ttm` | 净利润同比增长率 |
| `net_profit_parent_company_growth_ratio_lyr/ttm` | 归母净利润同比增长率 |
| `inc_adjusted_net_profit_lyr/ttm` | 扣非归母净利润同比增长率 |
| `net_asset_growth_ratio_lyr/ttm/lf` | 净资产同比增长率 |
| `total_asset_growth_ratio_lyr/ttm/lf` | 总资产同比增长率 |
| `inc_book_per_share_lyr/ttm/lf` | 每股净资产同比增长率 |
| `inc_return_on_equity_lyr/ttm` | 净资产收益率（摊薄）同比增长率 |
| `net_cash_flow_growth_ratio_lyr/ttm` | 净现金流增长率 |
| `net_operate_cash_flow_growth_ratio_lyr/ttm` | 经营现金流量净额同比增长率 |
| `net_investing_cash_flow_growth_ratio_lyr/ttm` | 投资现金流量净额同比增长率 |
| `net_financing_cash_flow_growth_ratio_lyr/ttm` | 筹资现金流量净额同比增长率 |

### 计算公式

| 因子 | 公式 |
|---|---|
| `inc_revenue_*` | `revenue_*_0 / revenue_*_1 - 1`（TTM 用 `_0 / _4`） |
| `inc_book_per_share_*` | `(equity_parent_company_*_0 / total_shares) / (equity_parent_company_*_1 / prev_year_total_shares) - 1` |
| `net_profit_growth_ratio_*` | `(net_profit_*_0 - net_profit_*_1) / net_profit_*_1`（TTM 用 `_0 - _4`） |
| `operating_profit_growth_ratio_*` | `(profit_from_operation_*_0 - profit_from_operation_*_1) / profit_from_operation_*_1` |
| `total_asset_growth_ratio_*` | `(total_assets_*_0 - total_assets_*_1) / total_assets_*_1` |
| `net_asset_growth_ratio_*` | `equity_parent_company_*_0 / equity_parent_company_*_1 - 1` |
| `net_operate_cash_flow_growth_ratio_*` | `cash_flow_from_operating_activities_*_0 / cash_flow_from_operating_activities_*_1 - 1` |
| `inc_return_on_equity_*` | `(net_profit_*_0 / total_equity_*_0) / (net_profit_*_1 / total_equity_*_1) - 1` |
| `inc_adjusted_net_profit_*` | `(net_profit_deduct_non_recurring_pnl_*_0 / net_profit_deduct_non_recurring_pnl_*_1) - 1` |

## 8. 财报三表原值

PIT 财务接口 `get_pit_financials_ex` 返回三大表全部原始字段 200+，支持指定历史回溯日期（`date=` 参数）取当时能看到的财报快照，能区分当期/非当期，字段以"上期、本期"双时点返回。基础字段在 `get_factor` 中以 `_mrq_n`、`_lyr_n`、`_ttm_n` 后缀拓展成多期时序。

### 利润表关键字段

`revenue`（营业总收入）、`operating_revenue`（营业收入）、`cost_of_goods_sold`（营业成本）、`gross_profit`（主营业务利润）、`selling_expense`（销售费用）、`ga_expense`（管理费用）、`financing_expense`（财务费用）、`r_n_d`（研发费用）、`profit_from_operation`（营业利润）、`profit_before_tax`（利润总额）、`income_tax`（所得税）、`net_profit`（净利润）、`net_profit_parent_company`（归母净利润）、`net_profit_deduct_non_recurring_pnl`（扣非归母净利润）、`minority_profit`（少数股东损益）、`basic_earnings_per_share`（基本 EPS）、`fully_diluted_earnings_per_share`（稀释 EPS）、`non_recurring_pnl`（非经常性损益）。

### 资产负债表关键字段

`cash_equivalent`（货币资金）、`bill_receivable`（应收票据）、`net_accts_receivable`（应收账款净额）、`inventory`（存货）、`current_assets`（流动资产合计）、`total_fixed_assets`（固定资产）、`intangible_assets`（无形资产）、`goodwill`（商誉）、`non_current_assets`（非流动资产合计）、`total_assets`（总资产）、`short_term_loans`（短期借款）、`notes_payable`（应付票据）、`accts_payable`（应付账款）、`current_liabilities`（流动负债合计）、`long_term_loans`（长期借款）、`bond_payable`（应付债券）、`non_current_liabilities`（非流动负债合计）、`total_liabilities`（负债合计）、`paid_in_capital`（实收资本）、`capital_reserve`（资本公积）、`surplus_reserve`（盈余公积）、`undistributed_profit`（未分配利润）、`equity_parent_company`（归母权益）、`minority_interest`（少数股东权益）、`total_equity`（股东权益合计）。

### 现金流量表关键字段

`cash_from_operating_activities`（经营活动现金流入小计）、`cash_paid_for_goods_and_services`（购买商品接受劳务支付的现金）、`cash_paid_for_employee`（支付给职工的现金）、`cash_paid_for_taxes`（支付的各项税费）、`cash_paid_for_operation_activities`（经营活动现金流出小计）、`cash_flow_from_operating_activities`（经营活动现金流量净额）、`cash_received_from_disposal_of_investment`（收回投资收到的现金）、`cash_paid_to_acquire_investment`（投资支付的现金）、`cash_paid_for_asset`（购建固定资产支付的现金）、`cash_flow_from_investing_activities`（投资活动现金流量净额）、`cash_received_from_investors`（吸收投资收到的现金）、`cash_received_from_financial_institution_borrows`（取得借款收到的现金）、`cash_paid_for_debt`（偿还债务支付的现金）、`cash_paid_for_dividend_and_interest`（分配股利、利润或偿付利息支付的现金）、`cash_flow_from_financing_activities`（筹资活动现金流量净额）、`cash_equivalent_increase`（现金及现金等价物净增加额）、`begin_period_cash_equivalent`（期初余额）、`end_period_cash_equivalent`（期末余额）、`fixed_asset_depreciation`（固定资产折旧）、`intangible_asset_amortization`（无形资产摊销）、`deferred_expense_amortization`（长期待摊费用摊销）。

## 9. 财务快报与业绩预告

**财务快报**（`current_performance`）提供财报正式发布前后的快数据。ROE 类含 `roe`（摊薄）、`roe_weighted`（加权）、`roe_cut`（扣非摊薄）、`roe_cut_weighted`（扣非加权）四种口径及对应同比；EPS 类含 `basic_eps`、`eps_weighted`、`eps_cut_epscut`、`eps_cut_weighted` 四种及同比；其他含 `operating_revenue`（营收）、`gross_profit`（主营利润）、`operating_profit`（营业利润）、`total_profit`（利润总额）、`np_parent_owners`（归母净利润）、`net_profit_cut`（扣非净利润）、`net_operate_cashflow`（经营 OCF）、`total_assets`（总资产）、`se_without_minority`（归母普通股东权益）、`se_parent_owners`（归母股东权益）、`total_shares`（总股本）、`equity_per_share`（每股净资产）、`net_operate_cashflow_per_share`（每股 OCF），以及所有对应同比字段。

**业绩预告**（`performance_forecast`）含 `forecast_type`（整体业绩预期）、`forecast_description`（业绩预期时间段描述）、`forecast_growth_rate_floor/ceiling`（最小/最大预期增长幅度）、`forecast_earning_floor/ceiling`（最小/最大预期收入）、`forecast_np_floor/ceiling`（最小/最大预期净利润）、`forecast_eps_floor/ceiling`（最小/最大预期每股收益）、`net_profit_yoy_const_forecast`（一致预期净利润增幅）、`forecast_ne_floor/ceiling`（最小/最大预期归母股东权益）。

## 10. 取数接口与行业分类

**财务衍生指标**用 `get_factor(order_book_ids, factor, start_date, end_date)` 取。A 股 1 万+ 只全市场批量可用，单日单标的单因子级延迟在毫秒量级。`get_all_factor_names(type=...)` 按类别返回可用因子名，支持的 type 含 `income_statement`、`balance_sheet`、`cash_flow_statement`、`eod_indicator`、`operational_indicator`、`cash_flow_indicator`、`financial_indicator`、`growth_indicator`、`alpha101`、`moving_average_indicator`、`obos_indicator`、`energy_indicator`。

**PIT 三表**用 `get_pit_financials_ex(order_book_ids, fields, start_quarter, end_quarter, date=None, statements='latest')`。`date=` 控制历史回溯点，`statements='all'` 返回某一报告期的所有记录（含调账修订），`statements='latest'` 仅返回最新一条。返回值含 `if_adjusted` 标记当期/非当期。

**港股财务数据**用 `get_pit_financials_ex(market='hk')` 或 `get_factor(..., market='hk')`。A 股会计准则归一处理后另支持金融公司、保险公司、非金融非保险公司三套非中国会计准则，含 60+ 估值/经营/现金流/财务/成长衍生因子。

**行业分类**支持中信 2019（`citics_2019`，推荐）、中信旧版（`citics`，退役中）、聚源（`gildata`）、申万 2021、恒生（`hsi`，港股）、中证（`csi`，港股）等多套标准。`get_industry_mapping(source=...)` 返回一/二/三级行业代码与名称；`get_industry(industry, source=...)` 按行业名或代码取成分股；`get_instrument_industry(order_book_ids, source=...)` 给定股票取行业归属，`level=0` 返回三级分类完整情况，`level='citics_sector'` 返回衍生板块及风格归属。

## 11. 附录

### 1. 米筐 RQData A 股财务接口主页

https://www.ricequant.com/doc/rqdata/python/stock-mod

### 2. 米筐 RQData 港股财务接口主页

https://www.ricequant.com/doc/rqdata/python/stock-hk

### 3. 米筐开源 Skills 仓库

https://github.com/ricequant/ricequant-skills

### 4. 米筐 PIT 财务接口文档

https://www.ricequant.com/doc/rqdata/python/stock-mod

### 5. 米筐财务因子命名规则

`_mrq_n` / `_lyr_n` / `_ttm_n` 后缀说明：基础三大表字段以 `_mrq_n`/`_lyr_n`/`_ttm_n` 加数字尾缀拓展为多期时序；衍生指标以 `_lf`/`_lyr`/`_ttm` 不带数字尾缀方式拓展。同口径字段命名与 Wind 的 eod_indicator / financial_indicator 体系基本对齐。

### 6. 港股因子字段说明

港股市场财务衍生因子（`market='hk'`）当前含 60+ 字段，覆盖市值、估值、经营三大类。基础三大表字段在港股仅以 `_ttm_n` 后缀提供 9 期 TTM 时序。A 股会计准则归一至中国会计准则框架下，金融/保险/非金融非保险公司另保留非中国会计准则原值。
