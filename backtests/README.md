# Simple strategy backtests

`simple_strategies.py` runs buy & hold, a 50 % hold, a static grid (2 %, 3 % and 5 % spacing), an SMA 20/50 crossover, an RSI 30/70 mean-reversion strategy and monthly DCA on ten US tickers over the trailing year. It uses yfinance adjusted closes and charges 5 bp per side.

```bash
pip install yfinance matplotlib
python backtests/simple_strategies.py --end 2026-10-31
```

The files in `results/` are from the run with `--end 2026-10-31`, which covers 31 Oct 2025 to 30 Oct 2026 (251 sessions).

Caveats: everything fills at the daily close. Intraday grid crossings are therefore missed. Idle cash earns nothing, which penalises the low-exposure strategies. Each grid is centred on the first close of the window. The results cover one year and ten tickers, so they show how each rule behaved in this period and say little about how it will behave next year.

## Three-year test: re-centring grids and refined variants

`advanced_strategies.py` covers 31 Oct 2023 to 30 Oct 2026 (753 sessions) with the same tickers and costs. Idle cash earns the 13-week T-bill yield (`^IRX`, averaging 4.3 %) and Sharpe is measured against it. Signal strategies trade at the close of the day after the signal. Grid levels fill at the close of the day the level is crossed.

| Group | Strategy | Rule |
|---|---|---|
| Benchmark | `buy_hold`, `hold_50`, `rebal_50_m` | all-in; half-in then idle; 50/50 rebalanced monthly |
| Grid | `grid_static` | 3 % spacing, ±10 levels, centred on day one, never moves |
| | `grid_recentre` | 3 %, ±5 levels; re-centres (and resets to 50 %) when price leaves the range in either direction |
| | `grid_trail_up` | as above, but re-centres only on upside breakouts |
| | `grid_periodic` | re-centres every 63 sessions |
| | `grid_vol` | spacing = 2 × 60-day daily σ (1–10 %), breakout re-centre |
| | `grid_trend` | `grid_vol` with a 75 % base above SMA(200) and 25 % below; a regime change also re-centres |
| Trend | `sma_20_50`, `sma_50_200` | long while the fast SMA is above the slow |
| | `tsmom_6m` | long while the 126-day return is positive, checked monthly |
| | `voltarget_15` | weight = min(1, 15 % / 20-day realised vol), rebalanced at 10 pp drift |
| | `trend_voltarget` | `voltarget_15`, held only above SMA(200) |
| Mean reversion | `rsi_30_70` | buy at RSI(14) < 30, sell at > 70 |
| | `rsi2_trend` | buy at RSI(2) < 10 above SMA(200), sell on a close above SMA(5) |
| | `boll_mr` | buy below the lower 20-day, 2σ band, sell on a close above SMA(20) |
| DCA | `dca_monthly`, `dca_dip` | 36 monthly slices; `dca_dip` doubles the slice when price is > 10 % below its 52-week high |

The run writes `metrics.csv` (per ticker, including yearly returns), `summary.csv` (medians across tickers), `grid_sweep.csv` (re-centring-grid spacing × levels) and two charts to `results_3y/`. The console output is saved as `results_3y/console.txt`.

```bash
python backtests/advanced_strategies.py --end 2026-10-31
```

## Bear-market windows

The rules and parameters are unchanged; only the window moves. `--start` sets the first day and `--out` the output directory. TSLA is skipped for the GFC windows because it listed in 2010.

| Window | Command | Output |
|---|---|---|
| 2022 rate-shock bear | `--start 2022-01-03 --end 2022-12-31` | `results_bear_2022/` |
| GFC, peak to trough | `--start 2007-10-09 --end 2009-03-10` | `results_gfc_trough/` |
| GFC, peak plus recovery | `--start 2007-10-09 --end 2010-10-09` | `results_gfc_recovery/` |

A window that ends at the trough flatters anything that sits in cash. The recovery window is there to check whether a strategy gets back in.

## Near-advanced blends and an ML allocator

`ml_meta.py` runs every base strategy continuously from 2000 on each ticker as a "sleeve" (TSLA from 2011) and builds portfolios by allocating across those sleeves and T-bills. Moving capital between sleeves of the same stock is netted, so only the change in total share exposure pays costs. Because the grids now run from 2000 rather than starting fresh at each window, their figures differ slightly from the earlier tables.

- **Near-advanced blends** use fixed weights chosen before any results were seen. Each pairs rules suited to opposite regimes, and each is rebalanced monthly:
  - `ens_grid_trend`: half `grid_periodic`, half `trend_voltarget`.
  - `ens_grid_trend_rsi2`: equal thirds of `grid_periodic`, `trend_voltarget` and `rsi2_trend`.
  - `ens_core_grid`: half `buy_hold`, half `grid_recentre`.
  - `ens_vt_grid`: half `voltarget_15`, half `grid_periodic`.
- **ML allocators**:
  - Two models: gradient boosting (`ml_hgb`) and ridge regression (`ml_ridge`), with hyperparameters fixed in code.
  - Each month they predict every sleeve's next-month return relative to the average of all sleeves.
  - The portfolio holds the top two sleeves at 50 % each.
  - The models are retrained every year on past samples only, and never on samples that overlap the holdout.
- **Development windows**: GFC + recovery, 2022 and 2023–26. All three were seen in earlier tests.
- **Holdout windows**: 2015–16 (sideways) and 2020 (COVID). These were looked at once, after the code was frozen in commit `70df83a`.

```bash
python backtests/ml_meta.py              # development windows → results_ml/dev/
python backtests/ml_meta.py --holdout    # adds the holdout windows → results_ml/holdout/
```

## New universes

`universes.py` draws two new test universes with a fixed seed. It uses a dated snapshot of the current S&P 500 constituents, so both universes carry survivorship bias.

- **Traditional:** 12 names drawn from members added in 1980 or earlier, in staples, utilities, industrials, energy, health care, financials and materials.
- **Random:** 15 names drawn from the remaining members.

Every earlier script was re-run on both universes with its rules unchanged. The results are in `results_universes/<universe>/`.

## Risk switch and volatility sizing for the core blend

`ml_risk.py` overlays the core blend (50 % `buy_hold` + 50 % `grid_recentre` sleeves) with two kinds of scaling:

- **A risk-on/off switch.** The blend is held at full size or at half size, with the rest in T-bills. There are two classifiers, logistic regression and gradient boosting. Each is judged against a simple switch that holds full size only above SMA(200).
- **Volatility sizing.** The blend is scaled towards 15 % volatility and never above full size. There are two forecasts: a HAR regression and gradient boosting. Each is judged against simple sizing from trailing 21-day volatility.

The models are trained walk-forward, pooled across all 37 tickers. Every window used before now counts as development data. The new holdout is Jul 2011 – Jun 2012 and 2018–19. It was run once, after the code was frozen in commit `de81b7d`.

```bash
python backtests/ml_risk.py              # development windows → results_ml_risk/dev/
python backtests/ml_risk.py --holdout    # adds the new holdout → results_ml_risk/holdout/
```

## Specification lab

`spec_lab.py` asks which modelling choices move the headline results. It changes one assumption at a time from the baseline (close fills, 5 bp, T-bill cash, adjusted prices, a 3 % / 5-level breakout grid, monthly rebalancing on the first session):

- **Fills:** resting intraday limit orders, or next-day open.
- **Costs:** 0 / 20 / 50 bp.
- **Cash and prices:** cash earning nothing; price-only (no dividends).
- **Timing:** rebalancing on the 6th, 11th or 16th session of the month; trend signals acted on with a lag of 0, 2 or 5 days.
- **Grid:** started years before the window; trail-up or static modes; 15 spacing × level combinations.
- **Core blend:** 25 / 75 % grid weights.

It runs everything on 2,000 ticker-windows: 47 tickers (the 37 stocks plus ten non-equity ETFs and bitcoin) × rolling 3-year windows starting every six months from 2001 to 2023.

Two further checks:

- **Exposure-matched benchmark.** Each rule is compared with a monthly constant mix at that rule's own average exposure in the window.
- **Null tests.** The same window's returns are re-ordered four ways: iid bootstrap, 21-day block bootstrap, sign permutation (which keeps volatility clustering) and time reversal.

`results_spec_lab/REPORT.md` gives the median total return, CAGR, Sharpe and drawdown for every spec, and the edges.

```bash
python backtests/spec_lab.py             # ~3 min; prices cached in backtests/.cache/
```

## Macro context from FRED

`fred_lab.py` adds four daily FRED series: the 3-month T-bill (DTB3), the 10y–3m curve (T10Y3M), the Baa–10y credit spread (BAA10Y) and VIX (VIXCLS). These are market-based and essentially never revised, and each value is used only from the next business day. The NBER recession flag (USREC) is used only to describe results, because it is dated with hindsight.

There are three tests:

- **Regimes.** Strategy returns in each macro state.
- **Switches.** Risk-off rules fixed in advance: curve inverted, credit spread above its 1-year average, VIX > 25, or two of the three. Each is applied to the core blend and to buy & hold, and compared with the same average exposure held constantly.
- **Volatility forecasts.** Walk-forward forecasts with and without the macro series. These are scored on 2013, 2014, 2017 and 2021, which were never used as a holdout, and then used to size the core blend.

```bash
python backtests/fred_lab.py             # ~30 s after spec_lab.py has cached prices → results_fred/
```

## Upstream / competitor lead-lag test

`lead_lag.py` tests the idea that when an upstream stock rises, its customers or competitors dip.

- **Pairs:** 190 ordered pairs fixed in advance from industry structure (supplier → customer, customer → supplier, competitors in both directions), plus every ordered pair among the same 108 stocks for a data-mining check.
- **Moves:** returns are adjusted for the market (SPY beta); sector ETFs are deliberately not removed.
- **Training:** 2020, 2022 and 2024, frozen in commit `50658a3`.
- **Validation:** 2021, 2023 and 2025, run once afterwards, using betas and volatilities from the year before.

```bash
python backtests/lead_lag.py --phase train
python backtests/lead_lag.py --phase validate
```

## Self-discovering relationships

`discover.py` finds relationships among about 480 S&P 500 stocks and eight commodity, dollar and bond ETFs with no pairs specified. It uses three models:

- **Structure:** same-day direct links (partial correlations).
- **Lead-lag:** per-stock Lasso regressions of the next-day and next-5-day move on about 990 lagged inputs.
- **Pooled model:** gradient boosting with automatically built peer and cluster inputs.

The models are trained on 2020, 2022 and 2024 (frozen in commit `10b702c`) and validated once on 2021, 2023 and 2025. Controls are a shuffled-timing null and a strict past-only walk-forward refit.

```bash
python backtests/discover.py --phase train
python backtests/discover.py --phase validate
```

### Delayed horizons (N+3, N+7)

`discover_lags.py` re-runs the same discovery models with no pairs specified, for four targets: the residual move on day N+3 alone, on day N+7 alone, summed over N+1..N+3, and summed over N+1..N+7. Each has its own shuffled-timing null. Training (2020, 2022, 2024) was frozen in commit `a8e12bc`. Validation is 2021, 2023 and 2025, plus 2026 to date as a fresh check.

```bash
python backtests/discover_lags.py --phase train
python backtests/discover_lags.py --phase validate
```

## A-shares (CSI 300)

`discover_ashare.py` runs the N → N+1 discovery test on 258 current CSI 300 members, using Yahoo Finance data (BaoStock's data server can't be reached from this environment). A-share rules:

- **Suspensions:** zero volume or a missing row.
- **Price limits:** ±9.8 % on the main board; ±19.8 % on ChiNext (±9.8 % before 24 August 2020) and on STAR; or High == Low with a large move. No buying on limit-up days, no selling on limit-down days; blocked positions carry over until they can be traded.
- **Costs:** 8 bp one-way.
- **Reporting:** a long-short book, and a long-only book with its active return over an equal-weighted universe.
- **Market inputs:** onshore ETFs (CSI 300, CSI 500, ChiNext, gold, 5-year government bonds).

Training was frozen in commit `d6e67bf` before validation.

```bash
python backtests/discover_ashare.py --phase train
python backtests/discover_ashare.py --phase validate
```
