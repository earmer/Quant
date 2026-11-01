"""Three-year backtest: static vs re-centring grids, plus refined trend, mean-reversion and DCA rules.

Conventions
  * Daily data, split/dividend-adjusted closes (yfinance). Long-only, no leverage.
  * Signal strategies decide on the close of day t and trade at the close of day t+1.
  * Grids are price-triggered: a level crossed by the close of day t fills at that close.
  * COST_BPS per side on traded notional. Idle cash earns the 13-week T-bill yield (^IRX).
  * Sharpe is computed on daily returns in excess of that T-bill yield.

    python backtests/advanced_strategies.py --end 2026-10-31
    python backtests/advanced_strategies.py --end 2026-10-31 --years 3 --tickers AAPL KO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CAPITAL = 100_000.0
COST_BPS = 5.0
WARMUP_DAYS = 420          # calendar days before the window, enough for SMA(200) / 252-day highs
DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "KO", "PG"]
OUT = Path(__file__).resolve().parent / "results_3y"
C = COST_BPS / 1e4


# ============================================================================ data
def _close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    s = df["Close"]
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


def load(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    px = pd.DataFrame({t: _close(t, start, end) for t in tickers}).dropna(how="all")
    irx = _close("^IRX", start, end).reindex(px.index).ffill().bfill() / 100   # annualised yield
    rf = (1 + irx) ** (1 / 252) - 1
    return px, rf


# ============================================================================ engines
class Book:
    """Cash + shares, with T-bill interest on cash and proportional costs."""

    def __init__(self):
        self.cash, self.sh, self.trades = CAPITAL, 0.0, 0

    def value(self, p):
        return self.cash + self.sh * p

    def buy(self, qty, p):
        qty = min(qty, max(self.cash, 0) / (p * (1 + C)))
        if qty > 1e-12:
            self.cash -= qty * p * (1 + C); self.sh += qty; self.trades += 1

    def sell(self, qty, p):
        qty = min(qty, self.sh)
        if qty > 1e-12:
            self.cash += qty * p * (1 - C); self.sh -= qty; self.trades += 1

    def to_weight(self, w, p):
        tgt = self.value(p) * w / p
        self.buy(tgt - self.sh, p) if tgt > self.sh else self.sell(self.sh - tgt, p)


def run_orders(px: pd.Series, orders: pd.Series, rf: pd.Series, band: float = 1e-9):
    """orders: target weight on days to act, NaN otherwise. Acts only if |current - target| > band."""
    b = Book()
    eq, ex = np.empty(len(px)), np.empty(len(px))
    for i, (p, o, r) in enumerate(zip(px.values, orders.values, rf.values)):
        if i:
            b.cash *= 1 + r
        if not np.isnan(o) and abs(b.sh * p / b.value(p) - o) > band:
            b.to_weight(o, p)
        eq[i] = b.value(p); ex[i] = b.sh * p / eq[i]
    return pd.Series(eq, px.index), pd.Series(ex, px.index), b.trades


def signal(target: pd.Series, idx) -> pd.Series:
    """Target weight decided at t, traded at t+1; emitted only when it changes (plus day one)."""
    t = target.shift(1).reindex(idx).fillna(0.0)
    return t.where(t.ne(t.shift()), np.nan).fillna({idx[0]: t.iloc[0]})


def grid_engine(px, rf, spacing, levels, mode="none", base=None, regime=None, period=63):
    """Geometric grid with hysteresis: from level L, buy one unit per level at or below
    c(1+s)^(L-1), sell one unit per level at or above c(1+s)^(L+1).

    mode: none      – never re-centre (static grid)
          breakout  – re-centre when price passes one step beyond either edge
          up        – re-centre only on an upside breakout (stay loaded on the way down)
          periodic  – re-centre every `period` sessions
    spacing/base: arrays indexed like px (read at re-centre time).
    regime: optional array; a change of regime also triggers a re-centre.
    On re-centre the book is rebalanced to `base` and the share unit is reset so that the
    grid can buy `levels` units with the cash and sell `levels` units of the holding.
    """
    n = len(px)
    base = np.full(n, 0.5) if base is None else np.asarray(base)
    b = Book()
    eq, ex = np.empty(n), np.empty(n)
    st = {}

    def centre(i, p):
        b.to_weight(base[i], p)
        st.update(c=p, s=spacing[i], L=0, i0=i,
                  unit=min(b.sh, b.cash / p) / levels,
                  reg=None if regime is None else regime[i])

    for i, (p, r) in enumerate(zip(px.values, rf.values)):
        if i:
            b.cash *= 1 + r
        if i == 0:
            centre(i, p)
        else:
            x = np.log(p / st["c"]) / np.log(1 + st["s"])
            out_up, out_dn = x > levels + 1, x < -(levels + 1)
            rec = ((mode == "breakout" and (out_up or out_dn)) or (mode == "up" and out_up)
                   or (mode == "periodic" and i - st["i0"] >= period)
                   or (regime is not None and regime[i] != st["reg"]))
            if rec:
                centre(i, p)
            else:
                L = st["L"]
                down = max(int(np.ceil(x)), -levels)
                up = min(int(np.floor(x)), levels)
                if down < L:
                    b.buy(st["unit"] * (L - down), p); st["L"] = down
                elif up > L:
                    b.sell(st["unit"] * (up - L), p); st["L"] = up
        eq[i] = b.value(p); ex[i] = b.sh * p / eq[i]
    return pd.Series(eq, px.index), pd.Series(ex, px.index), b.trades


def dca_engine(px, rf, extra=None):
    """Spread capital over monthly slices; `extra` (bool per day) doubles the slice that month."""
    first = ~px.index.to_period("M").duplicated()
    slice_ = CAPITAL / first.sum()
    extra = np.zeros(len(px), bool) if extra is None else np.asarray(extra)
    b = Book()
    eq, ex = np.empty(len(px)), np.empty(len(px))
    for i, (p, r, f, x) in enumerate(zip(px.values, rf.values, first, extra)):
        if i:
            b.cash *= 1 + r
        if f:
            b.buy(slice_ * (2 if x else 1) / (p * (1 + C)), p)
        eq[i] = b.value(p); ex[i] = b.sh * p / eq[i]
    return pd.Series(eq, px.index), pd.Series(ex, px.index), b.trades


# ============================================================================ indicators
def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def latch(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """1 after an entry signal until an exit signal, else 0."""
    state, out = 0.0, []
    for e, x in zip(enter.values, exit_.values):
        if state == 0 and e:
            state = 1.0
        elif state == 1 and x:
            state = 0.0
        out.append(state)
    return pd.Series(out, enter.index)


# ============================================================================ strategies
def build(full: pd.Series, rf_full: pd.Series, start) -> dict:
    """Return {name: (equity, exposure, trades)} for one ticker."""
    idx = full.index[full.index >= start]
    px, rf = full.loc[idx], rf_full.loc[idx]
    ret = full.pct_change()
    sma = {n: full.rolling(n).mean() for n in (5, 20, 50, 200)}
    vol20 = ret.rolling(20).std() * np.sqrt(252)
    vol60d = ret.rolling(60).std().shift(1).reindex(idx).bfill().values      # daily σ, known at t
    trend = (full > sma[200]).shift(1).reindex(idx).astype(int).values         # regime known at t
    first_of_month = pd.Series(~idx.to_period("M").duplicated(), idx)
    once = pd.Series(np.nan, idx)

    R = {}
    # --- passive benchmarks
    R["buy_hold"] = run_orders(px, once.fillna({idx[0]: 1.0}), rf)
    R["hold_50"] = run_orders(px, once.fillna({idx[0]: 0.5}), rf)
    R["rebal_50_m"] = run_orders(px, pd.Series(np.where(first_of_month, 0.5, np.nan), idx), rf)

    # --- grids (spacing 3 %, unless volatility-scaled)
    s3 = np.full(len(idx), 0.03)
    svol = np.clip(2 * vol60d, 0.01, 0.10)
    R["grid_static"] = grid_engine(px, rf, s3, 10, "none")
    R["grid_recentre"] = grid_engine(px, rf, s3, 5, "breakout")
    R["grid_trail_up"] = grid_engine(px, rf, s3, 5, "up")
    R["grid_periodic"] = grid_engine(px, rf, s3, 5, "periodic", period=63)
    R["grid_vol"] = grid_engine(px, rf, svol, 5, "breakout")
    R["grid_trend"] = grid_engine(px, rf, svol, 5, "breakout",
                                  base=np.where(trend == 1, 0.75, 0.25), regime=trend)

    # --- trend following
    R["sma_20_50"] = run_orders(px, signal((sma[20] > sma[50]).astype(float), idx), rf)
    R["sma_50_200"] = run_orders(px, signal((sma[50] > sma[200]).astype(float), idx), rf)
    mom = full.pct_change(126)
    tsmom = (mom > 0).astype(float).where(pd.Series(~full.index.to_period("M").duplicated(), full.index)).ffill()
    R["tsmom_6m"] = run_orders(px, signal(tsmom.fillna(0), idx), rf)
    vt = (0.15 / vol20).clip(upper=1.0)
    R["voltarget_15"] = run_orders(px, vt.shift(1).reindex(idx), rf, band=0.10)
    R["trend_voltarget"] = run_orders(px, (vt * (full > sma[200])).shift(1).reindex(idx), rf, band=0.10)

    # --- mean reversion
    r14 = rsi(full, 14)
    R["rsi_30_70"] = run_orders(px, signal(latch(r14 < 30, r14 > 70), idx), rf)
    r2 = rsi(full, 2)
    R["rsi2_trend"] = run_orders(px, signal(latch((r2 < 10) & (full > sma[200]), full > sma[5]), idx), rf)
    sd20 = full.rolling(20).std()
    R["boll_mr"] = run_orders(px, signal(latch(full < sma[20] - 2 * sd20, full > sma[20]), idx), rf)

    # --- DCA
    R["dca_monthly"] = dca_engine(px, rf)
    dip = (full < 0.9 * full.rolling(252, min_periods=1).max()).shift(1).reindex(idx).fillna(False)
    R["dca_dip"] = dca_engine(px, rf, dip.values)
    return R


GROUPS = {
    "benchmark": ["buy_hold", "hold_50", "rebal_50_m"],
    "grid": ["grid_static", "grid_recentre", "grid_trail_up", "grid_periodic", "grid_vol", "grid_trend"],
    "trend": ["sma_20_50", "sma_50_200", "tsmom_6m", "voltarget_15", "trend_voltarget"],
    "mean_rev": ["rsi_30_70", "rsi2_trend", "boll_mr"],
    "dca": ["dca_monthly", "dca_dip"],
}
ORDER = [s for g in GROUPS.values() for s in g]


# ============================================================================ metrics
def metrics(eq, ex, trades, rf) -> dict:
    r = eq.pct_change().dropna()
    exr = r - rf.reindex(r.index)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return {
        "cagr_%": ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100,
        "vol_%": r.std() * np.sqrt(252) * 100,
        "sharpe": exr.mean() / exr.std() * np.sqrt(252) if exr.std() > 0 else np.nan,
        "max_dd_%": (eq / eq.cummax() - 1).min() * 100,
        "exposure_%": ex.mean() * 100,
        "trades": trades,
    }


def yearly(eq: pd.Series, start: pd.Timestamp, years: int) -> list[float]:
    out = []
    for k in range(years):
        a, z = start + pd.DateOffset(years=k), start + pd.DateOffset(years=k + 1)
        seg = eq[(eq.index >= a) & (eq.index < z)]
        prev = eq[eq.index < a]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        out.append((seg.iloc[-1] / base - 1) * 100)
    return out


def sweep(full_px, rf_full, start, tickers):
    """Sensitivity of the breakout re-centring grid to spacing and number of levels."""
    rows = []
    for t in tickers:
        full = full_px[t].dropna(); idx = full.index[full.index >= start]
        px, rf = full.loc[idx], rf_full.loc[idx]
        bh = run_orders(px, pd.Series(np.nan, idx).fillna({idx[0]: 1.0}), rf)[0]
        for s in (0.02, 0.03, 0.05, 0.08):
            for lv in (3, 5, 8):
                eq, ex, n = grid_engine(px, rf, np.full(len(idx), s), lv, "breakout")
                m = metrics(eq, ex, n, rf)
                rows.append({"ticker": t, "spacing_%": int(s * 100), "levels": lv, **m,
                             "beats_bh": eq.iloc[-1] > bh.iloc[-1]})
    df = pd.DataFrame(rows)
    return df.groupby(["spacing_%", "levels"]).agg(
        cagr_med=("cagr_%", "median"), sharpe_med=("sharpe", "median"),
        maxdd_med=("max_dd_%", "median"), trades_med=("trades", "median"), beats_bh=("beats_bh", "sum"))


# ============================================================================ plots
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot_panels(curves, tickers, names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    rows = (len(tickers) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.7 * rows), sharex=True)
    for ax, t in zip(axes.flat, tickers):
        for k, name in enumerate(names):
            ax.plot(curves[(t, name)] / CAPITAL, lw=2 if k == 0 else 1.2, color=PALETTE[k], label=name)
        ax.set_yscale("log"); ax.set_title(t, fontsize=10, loc="left")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 1.25, 1.5, 2, 3, 4, 5, 6, 8), numticks=12))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(tickers):]:
        ax.axis("off")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=len(names),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, y=0.999, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--end", default=None, help="exclusive end date (default: today)")
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)
    full_px, rf_full = load(args.tickers, (start - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d"),
                            end.strftime("%Y-%m-%d"))
    idx = full_px.index[full_px.index >= start]
    print(f"Window {idx[0].date()} → {idx[-1].date()} ({len(idx)} sessions); "
          f"mean T-bill yield {((1 + rf_full.loc[idx]).prod() ** (252 / len(idx)) - 1) * 100:.2f} %")

    rows, curves = [], {}
    for t in args.tickers:
        res = build(full_px[t].dropna(), rf_full, start)
        for name in ORDER:
            eq, ex, n = res[name]
            curves[(t, name)] = eq
            rows.append({"ticker": t, "strategy": name, **metrics(eq, ex, n, rf_full),
                         **{f"y{k + 1}_%": v for k, v in enumerate(yearly(eq, start, args.years))}})
    res = pd.DataFrame(rows)

    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    pd.DataFrame(curves).to_csv(OUT / "equity_curves.csv", float_format="%.2f")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 100)
    piv = lambda col: res.pivot(index="strategy", columns="ticker", values=col).loc[ORDER, args.tickers]
    cagr = piv("cagr_%")
    print("\nCAGR (%) by ticker\n", cagr.round(1))
    print("\nSharpe (excess of T-bills) by ticker\n", piv("sharpe").round(2))

    ycols = [f"y{k + 1}_%" for k in range(args.years)]
    summ = res.groupby("strategy")[["cagr_%", "vol_%", "sharpe", "max_dd_%", "exposure_%", "trades", *ycols]].median().loc[ORDER]
    summ["beats_bh_cagr"] = (cagr.sub(cagr.loc["buy_hold"], axis=1) > 0).sum(axis=1)
    sh = piv("sharpe")
    summ["beats_bh_sharpe"] = (sh.sub(sh.loc["buy_hold"], axis=1) > 0).sum(axis=1)
    summ["beats_rebal50"] = (cagr.sub(cagr.loc["rebal_50_m"], axis=1) > 0).sum(axis=1)
    summ.to_csv(OUT / "summary.csv", float_format="%.4f")
    print(f"\nMedian across {len(args.tickers)} tickers; y1..y{args.years} = calendar-year-of-window returns;"
          " beats_* = number of tickers beating that benchmark\n", summ.round(2))

    sw = sweep(full_px, rf_full, start, args.tickers)
    sw.to_csv(OUT / "grid_sweep.csv", float_format="%.4f")
    print("\nRe-centring grid sensitivity (median across tickers)\n", sw.round(2))

    try:
        plot_panels(curves, args.tickers, ["buy_hold", "rebal_50_m", *GROUPS["grid"]],
                    f"Grid variants — growth of 1, log scale, {idx[0].date()} to {idx[-1].date()}",
                    OUT / "grid_curves.png")
        plot_panels(curves, args.tickers,
                    ["buy_hold", "sma_50_200", "tsmom_6m", "trend_voltarget", "rsi2_trend", "boll_mr", "dca_dip"],
                    f"Trend, mean-reversion and DCA variants — growth of 1, log scale, {idx[0].date()} to {idx[-1].date()}",
                    OUT / "other_curves.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
