# 广发 A 股因子智能体

这是从 Microsoft RD-Agent 精简出的专用因子研究项目，只保留与广发 A 股数据、因子生成、远程评估、去重和登记直接相关的代码。

## 核心流程

1. DeepSeek 驱动 RD-Agent 因子研究循环生成候选因子。
2. 本地控制器通过固定主机指纹的 SSH 连接数据机。
3. 远端 worker 按需读取 `G:/base` 的 Parquet 列，并按公告日处理财务数据；迭代筛选窗口固定默认 `2022-01-01` 至 `2023-12-31`。
4. 返回 RankIC、收益、Sharpe、回撤、换手率、覆盖率和年度稳定性。
5. 本地执行多指标门槛、相关性去重，并写入 SQLite 因子登记表。

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

执行一个远程候选因子：

```powershell
python scripts/gf_quant_remote_controller.py `
  --request configs/gf_quant/candidates/price_volume_reversal_20.json `
  --result artifacts/live/pv_reversal_20.json
```

登记评估结果：

```powershell
python -m scripts.register_gf_live_results `
  artifacts/live/pv_reversal_20.json `
  artifacts/live/fund_book_to_market_pit.json `
  --export artifacts/live/registry_snapshot.json
```

运行核心测试：

```powershell
python -m pytest -q
```

## 保留目录

- `rdagent/scenarios/gf_quant/`：GF 数据、切分、指标、去重、登记与研究场景。
- `rdagent/components/`、`rdagent/core/`：因子生成循环所需的最小 RD-Agent 基础设施。
- `rdagent/scenarios/qlib/`：GF 因子循环仍需复用的 Qlib 因子基类和回测模板。
- `scripts/`：远程控制器、远端 worker 与结果登记脚本。
- `configs/`、`artifacts/live/`：候选配置与可复核评估结果。
- `test/scenarios/gf_quant/`：项目专属核心测试。

本项目保留上游 MIT License；评估结果仅用于研究筛选，不代表生产收益承诺。
