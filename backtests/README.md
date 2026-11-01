# Simple strategy backtests

`simple_strategies.py` runs buy & hold, a 50 % hold, a static grid (2 %, 3 % and 5 % spacing), an SMA 20/50 crossover, an RSI 30/70 mean-reversion strategy and monthly DCA on ten US tickers over the trailing year. It uses yfinance adjusted closes and charges 5 bp per side.

```bash
pip install yfinance matplotlib
python backtests/simple_strategies.py --end 2026-10-31
```

The files in `results/` are from the run with `--end 2026-10-31`, which covers 31 Oct 2025 to 30 Oct 2026 (251 sessions).

Caveats: everything fills at the daily close. Intraday grid crossings are therefore missed. Idle cash earns nothing, which penalises the low-exposure strategies. Each grid is centred on the first close of the window. The results cover one year and ten tickers, so they show how each rule behaved in this period and say little about how it will behave next year.
