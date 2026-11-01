"""Draw the 'traditional' and 'random' test universes reproducibly.

Source: data/sp500_constituents_2026-11-01.csv, a snapshot of github.com/datasets/s-and-p-500-companies
taken on 1 Nov 2026. It lists *current* members only, so both universes carry survivorship bias.

  traditional – 12 drawn from members added to the S&P 500 in 1980 or earlier, in Consumer Staples,
                Utilities, Industrials, Energy, Health Care, Financials or Materials
  random      – 15 drawn from all remaining members
Tickers used in earlier tests are excluded from both draws.

    python backtests/universes.py      # prints the universes and writes data/universes.json
"""
import json
from pathlib import Path

import pandas as pd

SEED = 20261101
HERE = Path(__file__).resolve().parent
USED = {"SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "KO", "PG"}
TRAD_SECTORS = ["Consumer Staples", "Utilities", "Industrials", "Energy", "Health Care", "Financials", "Materials"]

d = pd.read_csv(HERE / "data" / "sp500_constituents_2026-11-01.csv")
d["Symbol"] = d["Symbol"].str.replace(".", "-", regex=False)          # yfinance format (BRK-B)
d["added"] = pd.to_datetime(d["Date added"], errors="coerce")
d = d[~d["Symbol"].isin(USED)]

pool = d[(d["added"] <= "1980-12-31") & d["GICS Sector"].isin(TRAD_SECTORS)]
trad = pool.sample(12, random_state=SEED)
rand = d[~d["Symbol"].isin(trad["Symbol"])].sample(15, random_state=SEED)

out = {"seed": SEED, "traditional": sorted(trad["Symbol"]), "random": sorted(rand["Symbol"])}
(HERE / "data" / "universes.json").write_text(json.dumps(out, indent=2) + "\n")
for name, df in (("traditional", trad), ("random", rand)):
    print(f"\n{name} ({len(df)})")
    print(df[["Symbol", "Security", "GICS Sector", "Date added"]].sort_values("Symbol").to_string(index=False))
