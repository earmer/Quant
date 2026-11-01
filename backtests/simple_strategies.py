"""Backtest simple long-only strategies on a basket of US stocks over the last year.

Strategies (all daily, signals on the close, filled at that close, long-only,
no leverage, no shorting):
  buy_hold   – invest everything on day one
  hold_50    – put half in on day one and do nothing (the grid's starting book, left alone)
  grid       – static geometric grid centred on the first close; start 50 % invested,
               buy one unit per level crossed downwards, sell one per level crossed up
  sma_cross  – fully long when SMA(20) > SMA(50), otherwise cash
  rsi_mr     – buy when RSI(14) < 30, sell when RSI(14) > 70
  dca        – invest an equal slice of cash on the first trading day of each month

Costs: COST_BPS per side applied to traded notional. Cash earns nothing.

    python backtests/simple_strategies.py                 # defaults
    python backtests/simple_strategies.py --end 2026-10-31 --tickers AAPL KO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CAPITAL = 100_000.0
COST_BPS = 5.0          # commission + slippage per side, in basis points
WARMUP_DAYS = 120       # calendar days of history fetched before the test window for indicators
DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "KO", "PG"]
GRID_SPACINGS = [0.02, 0.03, 0.05]
GRID_LEVELS = 10        # levels on each side of the centre
OUT = Path(__file__).resolve().parent / "results"


# ----------------------------------------------------------------------------- data
def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Split/dividend-adjusted closes, one column per ticker (downloaded one at a time)."""
    cols = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if df.empty:
            raise RuntimeError(f"No data for {t}")
        close = df["Close"]
        cols[t] = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    return pd.DataFrame(cols).dropna(how="all")


# ----------------------------------------------------------------------------- engine
def run_targets(px: pd.Series, target_w: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """Rebalance to a 0/1 target weight whenever it changes.

    Returns the equity curve, the invested fraction of equity, and the trade count."""
    cash, sh, trades = CAPITAL, 0.0, 0
    eq, ex = np.empty(len(px)), np.empty(len(px))
    prev = 0.0
    for i, (p, w) in enumerate(zip(px.values, target_w.values)):
        if w != prev:
            value = cash + sh * p
            new_sh = value * w / p / (1 + COST_BPS / 1e4 * w)
            traded = abs(new_sh - sh) * p
            cash = value - new_sh * p - traded * COST_BPS / 1e4
            sh, prev, trades = new_sh, w, trades + 1
        eq[i] = cash + sh * p
        ex[i] = sh * p / eq[i]
    return pd.Series(eq, px.index), pd.Series(ex, px.index), trades


def buy_hold(px: pd.Series, _full: pd.Series):
    return run_targets(px, pd.Series(1.0, px.index))


def hold_half(px: pd.Series, _full: pd.Series):
    """50 % stock / 50 % cash bought once, never rebalanced — the grid's starting book, left alone."""
    return run_targets(px, pd.Series(0.5, px.index))


def sma_cross(px: pd.Series, full: pd.Series, fast=20, slow=50):
    w = (full.rolling(fast).mean() > full.rolling(slow).mean()).astype(float)
    return run_targets(px, w.reindex(px.index))


def rsi(series: pd.Series, n=14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def rsi_mr(px: pd.Series, full: pd.Series, lo=30, hi=70):
    r = rsi(full).reindex(px.index)
    state, w = 0.0, []
    for v in r.values:
        if v < lo:
            state = 1.0
        elif v > hi:
            state = 0.0
        w.append(state)
    return run_targets(px, pd.Series(w, px.index))


def grid(px: pd.Series, _full: pd.Series, spacing=0.03, levels=GRID_LEVELS):
    """Static geometric grid with hysteresis: from level L, buy one unit per level at or
    below p0(1+s)^(L-1) and sell one unit per level at or above p0(1+s)^(L+1)."""
    p0 = px.iloc[0]
    unit = (CAPITAL / 2) / p0 / levels          # shares traded per level
    c = COST_BPS / 1e4
    sh = unit * levels
    cash = CAPITAL - sh * p0 * (1 + c)
    level, trades = 0, 0
    eq, ex = np.empty(len(px)), np.empty(len(px))
    for i, p in enumerate(px.values):
        x = np.log(p / p0) / np.log(1 + spacing)
        down, up = max(int(np.ceil(x)), -levels), min(int(np.floor(x)), levels)
        if down < level:    # fell through levels: buy
            n = level - down
            qty = min(unit * n, cash / (p * (1 + c)))
            cash -= qty * p * (1 + c); sh += qty
            trades += n; level = down
        elif up > level:    # rose through levels: sell
            n = up - level
            qty = min(unit * n, sh)
            cash += qty * p * (1 - c); sh -= qty
            trades += n; level = up
        eq[i] = cash + sh * p
        ex[i] = sh * p / eq[i]
    return pd.Series(eq, px.index), pd.Series(ex, px.index), trades


def dca(px: pd.Series, _full: pd.Series):
    months = px.index.to_period("M")
    first_days = ~months.duplicated()
    n_slices = int(first_days.sum())
    cash, sh, trades = CAPITAL, 0.0, 0
    slice_ = CAPITAL / n_slices
    c = COST_BPS / 1e4
    eq, ex = np.empty(len(px)), np.empty(len(px))
    for i, (p, buy) in enumerate(zip(px.values, first_days)):
        if buy:
            amt = min(slice_, cash)
            sh += amt / (p * (1 + c)); cash -= amt; trades += 1
        eq[i] = cash + sh * p
        ex[i] = sh * p / eq[i]
    return pd.Series(eq, px.index), pd.Series(ex, px.index), trades


# ----------------------------------------------------------------------------- metrics
def metrics(eq: pd.Series, ex: pd.Series, trades: int) -> dict:
    r = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1
    vol = r.std() * np.sqrt(252)
    return {
        "return_%": (eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "vol_%": vol * 100,
        "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "max_dd_%": dd.min() * 100,
        "exposure_%": ex.mean() * 100,
        "trades": trades,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--end", default=None, help="exclusive end date (default: today)")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=1)
    full = load_prices(args.tickers, (start - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d"),
                       end.strftime("%Y-%m-%d"))
    test = full.loc[full.index >= start]
    print(f"Test window: {test.index[0].date()} → {test.index[-1].date()} ({len(test)} sessions)")

    strategies = {
        "buy_hold": buy_hold,
        "hold_50": hold_half,
        **{f"grid_{int(s*100)}pct": (lambda s: lambda p, f: grid(p, f, spacing=s))(s) for s in GRID_SPACINGS},
        "sma_20_50": sma_cross,
        "rsi_30_70": rsi_mr,
        "dca_monthly": dca,
    }

    rows, curves = [], {}
    for t in args.tickers:
        px, fp = test[t].dropna(), full[t].dropna()
        for name, fn in strategies.items():
            eq, ex, n = fn(px, fp)
            curves[(t, name)] = eq
            rows.append({"ticker": t, "strategy": name, **metrics(eq, ex, n)})
    res = pd.DataFrame(rows)

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(curves).to_csv(OUT / "equity_curves.csv", float_format="%.2f")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    ret = res.pivot(index="ticker", columns="strategy", values="return_%")[list(strategies)]
    print("\nTotal return (%)\n", ret.round(1))
    print("\nSharpe\n", res.pivot(index="ticker", columns="strategy", values="sharpe")[list(strategies)].round(2))
    print("\nMax drawdown (%)\n", res.pivot(index="ticker", columns="strategy", values="max_dd_%")[list(strategies)].round(1))
    print("\nAverage exposure (% of equity invested)\n", res.pivot(index="ticker", columns="strategy", values="exposure_%")[list(strategies)].round(0))
    print("\nTrades (grid: level fills)\n", res.pivot(index="ticker", columns="strategy", values="trades")[list(strategies)])
    summary = res.groupby("strategy")[["return_%", "vol_%", "sharpe", "max_dd_%", "exposure_%"]].median().loc[list(strategies)]
    summary["beats_bh"] = (ret.sub(ret["buy_hold"], axis=0) > 0).sum()
    summary["beats_h50"] = (ret.sub(ret["hold_50"], axis=0) > 0).sum()
    print("\nMedian across tickers (beats_* = tickers where return exceeds buy & hold / hold_50)\n", summary.round(2))
    grid_cols = [c for c in strategies if c.startswith("grid_")]
    print("\nGrid minus hold_50, return (pp) — what the grid trading itself added\n",
          ret[grid_cols].sub(ret["hold_50"], axis=0).round(1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(args.tickers)
        fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(13, 2.6 * ((n + 1) // 2)), sharex=True)
        for ax, t in zip(axes.flat, args.tickers):
            for name in strategies:
                ax.plot(curves[(t, name)] / CAPITAL, lw=1.8 if name in ("buy_hold", "hold_50") else 1, label=name)
            ax.set_title(t, fontsize=10); ax.grid(alpha=0.3)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Growth of 1 — {test.index[0].date()} to {test.index[-1].date()}")
        fig.tight_layout()
        fig.savefig(OUT / "equity_curves.png", dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
