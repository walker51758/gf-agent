# GF Quant RD-Agent(Q) V1

This extension keeps Microsoft RD-Agent(Q)'s research loop and replaces the
data/evaluation boundary for the GF A-share data host.

## Architecture

1. RD-Agent proposes factor hypotheses and code/DSL expressions with
   `deepseek-v4-flash` through `https://api.deepseek.com`.
2. The local controller sends a bounded request to the pinned SSH host. Secrets
   are loaded from the named environment variables only.
3. The remote worker reads only the referenced parquet columns from `G:/base`.
   Financial values become available on `dt` (announcement date), not `report`,
   preventing accounting-period look-ahead.
4. The worker returns IC, ICIR, RankIC, RankICIR, return, Sharpe, drawdown,
   turnover, coverage, spread, win rate and yearly stability. Large factor
   arrays remain as remote parquet artifacts.
5. The local service applies the multi-metric gate, exact/statistical de-dup,
   and transactionally registers metadata in SQLite.

The normal path does not download the 2GB base data and does not require a
remote Docker installation. The inherited Qlib entry points remain available
for environments where Qlib/Docker is provisioned.

## Date policy

`gf_quant.split` builds boundaries from observed trading dates. It supports a
single ratio split and rolling/expanding walk-forward splits. There are no
2008-2020 default boundaries in the GF scenario. Every live result records its
actual evaluation interval in the result manifest.

## Run one remote evaluation

```powershell
python scripts/gf_quant_remote_controller.py `
  --request configs/gf_quant/candidates/price_volume_reversal_20.json `
  --result artifacts/live/pv_reversal_20.json
```

Register evaluated factors:

```powershell
python -m scripts.register_gf_live_results `
  artifacts/live/pv_reversal_20.json `
  artifacts/live/fund_book_to_market_pit.json `
  --export artifacts/live/registry_snapshot.json
```

## V1 acceptance gate

A candidate is research-accepted when all are true: absolute RankIC at least
0.03, annualized long-short return non-negative, Sharpe at least 0.30, absolute
max drawdown at most 0.35, turnover at most 1.0, and coverage at least 0.50.
The gate is configurable. Acceptance is a research-screen result, not a claim
of production profitability.

De-dup uses normalized expression identity, absolute Pearson, Spearman and
return-profile correlation with default threshold 0.88 and minimum overlap
100. Arrays are compared one reference at a time to bound memory use.

## Evidence boundary

The checked-in JSON files under `artifacts/live` are live remote evaluations.
The 2022-2023 period was used for candidate screening; 2024-01-01 through
2026-04-16 was then used as holdout evaluation. Multiple fundamental candidates
were screened, so the holdout result must not be described as a pristine blind
production test. Yearly stability is retained in full, including negative
subperiods.
