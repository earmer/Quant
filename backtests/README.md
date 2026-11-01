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
