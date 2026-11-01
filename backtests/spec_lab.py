"""Specification lab: which modelling choices move the headline results?

Every earlier conclusion rests on choices that were made once and never varied: fills at the
close, 5 bp costs, T-bill interest on cash, dividend-adjusted prices, a 50/50 constant mix as the
benchmark, a handful of hand-picked windows, one grid spacing, and grids started afresh at each
window. This script varies each choice in turn, holding the others at the baseline, over rolling
3-year windows (a new one every six months, 2001-2023) on all 37 stocks used so far plus ten
non-equity ETFs (bonds, gold, silver, commodities, oil, REITs, foreign equity, bitcoin).

Strategies (all long-only, cash earns the T-bill yield unless the spec says otherwise)
  bh        buy & hold
  mix50     50 % stock / 50 % cash, rebalanced monthly
  grid      3 % geometric grid, 5 levels, re-centred on a breakout (grid_recentre)
  core      50 % bh + 50 % grid sleeves, re-weighted monthly (ens_core_grid)
  tvt       min(1, 15 % / 20-day vol) while above the 200-day average (trend_voltarget)
  mix_m(x)  a monthly constant mix at x's *realised average exposure* in that window: the
            exposure-matched benchmark. It is chosen with hindsight, which favours the benchmark.

Edges reported: core - mix50 (the comparison used so far), and core / grid / tvt minus their
exposure-matched mixes (does the rule do anything beyond holding that much stock?).

Null tests (labelled as such; they are resampled real returns, never invented data): the same
window's daily returns (with the warm-up) re-ordered by an iid bootstrap, a stationary block
bootstrap (mean block 21 days), or time reversal. If an edge is as large on the nulls as on the
real paths, it is mechanical rather than evidence of exploitable structure.

    python backtests/spec_lab.py              # downloads once into backtests/.cache/
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "ohlc_2026-11-01.pkl"
OUT = HERE / "results_spec_lab"
U = json.loads((HERE / "data" / "universes.json").read_text())
UNIVERSES = {
    "original": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "KO", "PG"],
    "traditional": U["traditional"],
    "random": U["random"],
    "assets": ["TLT", "IEF", "GLD", "SLV", "DBC", "USO", "VNQ", "EFA", "EEM", "BTC-USD"],
}
END = "2026-11-01"
YEARS = 3
WARM = 250                          # sessions required before a window (SMA 200 etc.)
STARTS = pd.date_range("2001-01-01", "2023-07-01", freq="6MS")
NONOVERLAP = pd.DatetimeIndex([f"{y}-01-01" for y in range(2001, 2023, 3)])
SEED = 20261101
N_NULL = 2                          # null paths per ticker-window and null type
NULLS = ("iid", "block21", "signperm", "reversed")
NULL_SPECS = ("baseline", "grid_mode_static", "grid_s8_l10")   # specs also run on the null paths

BASE = dict(cost=5.0, fill="close", cash=True, total_return=True, s=0.03, levels=5, mode="breakout",
            w_core=0.5, offset=0, lag=1, warm_grid=False)
SPECS = {
    "baseline": {},
    "fill_touch": dict(fill="touch"), "fill_next_open": dict(fill="next_open"),
    "cost_0": dict(cost=0.0), "cost_20": dict(cost=20.0), "cost_50": dict(cost=50.0),
    "cash_zero": dict(cash=False), "price_return": dict(total_return=False),
    "rebal_day_5": dict(offset=5), "rebal_day_10": dict(offset=10), "rebal_day_15": dict(offset=15),
    "signal_lag_0": dict(lag=0), "signal_lag_2": dict(lag=2), "signal_lag_5": dict(lag=5),
    "grid_warm": dict(warm_grid=True), "grid_mode_up": dict(mode="up"), "grid_mode_static": dict(mode="none"),
    "core_w25": dict(w_core=0.25), "core_w75": dict(w_core=0.75),
    **{f"grid_s{int(s * 100)}_l{l}": dict(s=s, levels=l) for s in (0.01, 0.02, 0.03, 0.05, 0.08)
       for l in (3, 5, 10) if (s, l) != (0.03, 5)},
}
MODES = {"none": 0, "breakout": 1, "up": 2}
FILLS = {"close": 0, "touch": 1, "next_open": 2}


# ============================================================================ data
def download(t: str) -> pd.DataFrame | None:
    df = yf.download(t, start="1999-01-01", end=END, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Adj Close"]].dropna()
    df = df[(df > 0).all(axis=1)]
    k = df["Adj Close"] / df["Close"]
    adj = pd.DataFrame({c: df[c] * k for c in ("Open", "High", "Low", "Close")})
    raw = df[["Open", "High", "Low", "Close"]]
    out = {}
    for tag, d in (("tr", adj), ("pr", raw)):
        hi = d[["Open", "High", "Close"]].max(axis=1)
        lo = d[["Open", "Low", "Close"]].min(axis=1)
        out[tag] = pd.DataFrame({"O": d["Open"], "H": hi, "L": lo, "C": d["Close"]})
    return pd.concat(out, axis=1)


def load() -> tuple[dict, pd.Series]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    data = {}
    for t in [t for v in UNIVERSES.values() for t in v]:
        d = download(t)
        if d is None:
            print(f"[skip] {t}: no data")
        else:
            data[t] = d
    irx = yf.download("^IRX", start="1999-01-01", end=END, auto_adjust=True, progress=False, threads=False)["Close"]
    irx = (irx.iloc[:, 0] if isinstance(irx, pd.DataFrame) else irx) / 100
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps((data, irx)))
    return data, irx


# ============================================================================ engines (numba)
@njit(cache=True)
def _buy(cash, sh, q, p, c):
    q = min(q, max(cash, 0.0) / (p * (1 + c)))
    if q > 1e-12:
        cash -= q * p * (1 + c); sh += q
    return cash, sh


@njit(cache=True)
def _sell(cash, sh, q, p, c):
    q = min(q, sh)
    if q > 1e-12:
        cash += q * p * (1 - c); sh -= q
    return cash, sh


@njit(cache=True)
def _to_w(cash, sh, w, p, c):
    tgt = (cash + sh * p) * w / p
    if tgt > sh:
        return _buy(cash, sh, tgt - sh, p, c)
    return _sell(cash, sh, sh - tgt, p, c)


@njit(cache=True)
def grid_nb(O, H, L, C, rf, s, levels, mode, fill, c, base):
    """Breakout/up/static geometric grid with hysteresis.
    fill 0: levels crossed by the close fill at that close (as in all earlier scripts).
    fill 1: resting limit orders; the day's path is open -> low -> high -> close on up days and
            open -> high -> low -> close on down days; a level fills at its price, or at the
            open when the market gaps through it. Breakouts re-centre at the close.
    fill 2: levels are detected at the close and filled at the next day's open."""
    n = len(C)
    eq = np.empty(n); ex = np.empty(n)
    cash, sh = 100000.0, 0.0
    cash, sh = _to_w(cash, sh, base, C[0], c)
    ctr, Lv, unit = C[0], 0, min(sh, cash / C[0]) / levels
    g = np.log(1 + s)
    pend_q, pend_rec = 0.0, False                  # fill 2: orders waiting for tomorrow's open
    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
            if fill == 2:
                if pend_rec:
                    cash, sh = _to_w(cash, sh, base, O[i], c)
                    unit = min(sh, cash / O[i]) / levels
                elif pend_q > 0:
                    cash, sh = _buy(cash, sh, pend_q, O[i], c)
                elif pend_q < 0:
                    cash, sh = _sell(cash, sh, -pend_q, O[i], c)
                pend_q, pend_rec = 0.0, False
            if fill == 1:
                pts = np.empty(4)
                pts[0] = O[i]
                if C[i] >= O[i]:
                    pts[1] = L[i]; pts[2] = H[i]
                else:
                    pts[1] = H[i]; pts[2] = L[i]
                pts[3] = C[i]
                a = C[i - 1]
                for k in range(4):
                    b = pts[k]
                    if b < a:
                        while Lv > -levels:
                            lvl = ctr * np.exp(g * (Lv - 1))
                            if lvl < b:
                                break
                            cash, sh = _buy(cash, sh, unit, b if k == 0 else lvl, c); Lv -= 1
                    elif b > a:
                        while Lv < levels:
                            lvl = ctr * np.exp(g * (Lv + 1))
                            if lvl > b:
                                break
                            cash, sh = _sell(cash, sh, unit, b if k == 0 else lvl, c); Lv += 1
                    a = b
            x = np.log(C[i] / ctr) / g
            up_out, dn_out = x > levels + 1, x < -(levels + 1)
            rec = (mode == 1 and (up_out or dn_out)) or (mode == 2 and up_out)
            if rec:
                if fill == 2:
                    pend_rec = True
                else:
                    cash, sh = _to_w(cash, sh, base, C[i], c)
                    unit = min(sh, cash / C[i]) / levels
                ctr, Lv = C[i], 0
            elif fill != 1:
                down = max(int(np.ceil(x)), -levels)
                up = min(int(np.floor(x)), levels)
                q = 0.0
                if down < Lv:
                    q = unit * (Lv - down); Lv = down
                elif up > Lv:
                    q = -unit * (up - Lv); Lv = up
                if fill == 0:
                    if q > 0:
                        cash, sh = _buy(cash, sh, q, C[i], c)
                    elif q < 0:
                        cash, sh = _sell(cash, sh, -q, C[i], c)
                else:
                    pend_q = q
        eq[i] = cash + sh * C[i]; ex[i] = sh * C[i] / eq[i]
    return eq, ex


@njit(cache=True)
def orders_nb(C, rf, target, c, band):
    """target: weight to trade to at that close (NaN = no order); acts if off by more than band."""
    n = len(C)
    eq = np.empty(n); ex = np.empty(n)
    cash, sh = 100000.0, 0.0
    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
        t = target[i]
        if not np.isnan(t):
            v = cash + sh * C[i]
            if abs(sh * C[i] / v - t) > band:
                cash, sh = _to_w(cash, sh, t, C[i], c)
        eq[i] = cash + sh * C[i]; ex[i] = sh * C[i] / eq[i]
    return eq, ex


@njit(cache=True)
def blend_nb(r1, e1, r2, e2, w, rebal, c):
    """Two sleeves re-weighted to (w, 1-w) on rebal days; cost on the netted change in exposure."""
    n = len(r1)
    eq = np.empty(n); ex = np.empty(n)
    v1 = 100000.0 * w; v2 = 100000.0 * (1 - w)
    V = v1 + v2
    cost = c * (w * e1[0] + (1 - w) * e2[0]) * V
    v1 -= cost * w; v2 -= cost * (1 - w)
    for i in range(n):
        if i > 0:
            v1 *= 1 + r1[i]; v2 *= 1 + r2[i]
            if rebal[i]:
                V = v1 + v2
                cost = c * abs((w - v1 / V) * e1[i] + ((1 - w) - v2 / V) * e2[i]) * V
                V -= cost
                v1 = w * V; v2 = (1 - w) * V
        eq[i] = v1 + v2
        ex[i] = (v1 * e1[i] + v2 * e2[i]) / eq[i]
    return eq, ex


# ============================================================================ one ticker-window
def monthly_days(idx: pd.DatetimeIndex, offset: int) -> np.ndarray:
    """Boolean: the (offset+1)-th session of each month (day 0 always trades)."""
    per = idx.to_period("M")
    rank = pd.Series(1, idx).groupby(per).cumsum().values - 1
    flag = rank == offset
    flag[0] = True
    return flag


def metrics(eq: np.ndarray, ex: np.ndarray, rf: np.ndarray, yrs: float) -> tuple:
    r = eq[1:] / eq[:-1] - 1
    x = r - rf[1:]
    sd = x.std()
    return ((eq[-1] / eq[0]) ** (1 / yrs) - 1) * 100, x.mean() / sd * np.sqrt(252) if sd > 0 else np.nan, \
        (eq / np.maximum.accumulate(eq) - 1).min() * 100, ex.mean(), (eq[-1] / eq[0] - 1) * 100


def run_window(d: pd.DataFrame, rf_s: pd.Series, a: int, z: int, sp: dict, warm_grid_cache=None) -> dict:
    """d: OHLC frame over [a-WARM, z] positions of the ticker; a, z: window bounds inside d."""
    O, H, L, C = (d[k].values for k in "OHLC")
    rf_all = rf_s.values if sp["cash"] else np.zeros(len(d))
    rf_sharpe = rf_s.values
    c = sp["cost"] / 1e4
    sl = slice(a, z + 1)
    Cw, rfw = C[sl], rf_all[sl]
    yrs = (d.index[z] - d.index[a]).days / 365.25
    rebal = monthly_days(d.index[sl], sp["offset"])
    mode, fill = MODES[sp["mode"]], FILLS[sp["fill"]]

    res = {}
    once = np.full(len(Cw), np.nan); once[0] = 1.0
    bh = orders_nb(Cw, rfw, once, c, 1e-9)
    mix = orders_nb(Cw, rfw, np.where(rebal, 0.5, np.nan), c, 1e-9)
    if warm_grid_cache is not None:
        ge, gx = warm_grid_cache
        ge, gx = ge[sl] / ge[a] * 1e5, gx[sl]
    else:
        ge, gx = grid_nb(O[sl], H[sl], L[sl], Cw, rfw, sp["s"], sp["levels"], mode, fill, c, 0.5)
    rb, rg = np.r_[0, bh[0][1:] / bh[0][:-1] - 1], np.r_[0, ge[1:] / ge[:-1] - 1]
    core = blend_nb(rb, bh[1], rg, gx, sp["w_core"], rebal, c)

    # trend_voltarget on the full warm-up history, sliced to the window
    ret = np.r_[np.nan, C[1:] / C[:-1] - 1]
    vol20 = pd.Series(ret).rolling(20).std().values * np.sqrt(252)
    sma200 = pd.Series(C).rolling(200).mean().values
    tgt = np.minimum(1.0, 0.15 / vol20) * (C > sma200)
    tgt = np.r_[np.full(sp["lag"], np.nan), tgt[:len(tgt) - sp["lag"]]] if sp["lag"] else tgt
    tvt = orders_nb(Cw, rfw, np.nan_to_num(tgt[sl], nan=0.0), c, 0.10)

    for k, v in (("bh", bh), ("mix50", mix), ("grid", (ge, gx)), ("core", core), ("tvt", tvt)):
        res[k] = metrics(v[0], v[1], rf_sharpe[sl], yrs)
    for k in ("core", "grid", "tvt"):
        w = res[k][3]
        m = orders_nb(Cw, rfw, np.where(rebal, w, np.nan), c, 1e-9)
        res[f"mixm_{k}"] = metrics(m[0], m[1], rf_sharpe[sl], yrs)
    return res


# ============================================================================ nulls
def null_path(C: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    r = np.diff(np.log(C))
    n = len(r)
    if kind == "iid":
        r = r[rng.integers(0, n, n)]
    elif kind == "block21":                       # stationary bootstrap, geometric block lengths
        out, i = np.empty(n), rng.integers(n)
        for j in range(n):
            out[j] = r[i]
            i = rng.integers(n) if rng.random() < 1 / 21 else (i + 1) % n
        r = out
    elif kind == "signperm":                      # keep |r_t| (volatility clustering), shuffle the signs
        r = np.abs(r) * rng.permutation(np.sign(r))
    elif kind == "reversed":
        r = r[::-1]
    p = C[0] * np.exp(np.r_[0, np.cumsum(r)])
    return p


# ============================================================================ driver
ROWS = ["cagr", "sharpe", "maxdd", "expo", "total"]


def flatten(t, uni, start, spec, res) -> dict:
    row = dict(ticker=t, universe=uni, start=start, spec=spec)
    for k, v in res.items():
        for name, x in zip(ROWS, v):
            row[f"{k}_{name}"] = x
    return row


def main() -> None:
    data, irx = load()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for uni, tickers in UNIVERSES.items():
        for t in tickers:
            if t not in data:
                continue
            full = {tag: data[t][tag] for tag in ("tr", "pr")}
            idx = full["tr"].index
            rf = ((1 + irx.reindex(idx).ffill().bfill()) ** (1 / 252) - 1)
            warm_cache = {}
            for st in STARTS:
                a = idx.searchsorted(st)
                end = st + pd.DateOffset(years=YEARS)
                z = idx.searchsorted(end) - 1
                if a < WARM or a >= len(idx) or idx[-1] < end - pd.Timedelta(days=7):
                    continue
                lo = a - WARM
                for name, over in SPECS.items():
                    sp = {**BASE, **over}
                    d = full["tr" if sp["total_return"] else "pr"].iloc[lo:z + 1]
                    wc = None
                    if sp["warm_grid"]:
                        key = sp["total_return"]
                        if key not in warm_cache:
                            D = full["tr"]
                            ge = grid_nb(*(D[k].values for k in "OHLC"), rf.values, sp["s"], sp["levels"],
                                         MODES[sp["mode"]], FILLS[sp["fill"]], sp["cost"] / 1e4, 0.5)
                            warm_cache[key] = ge
                        ge, gx = warm_cache[key]
                        wc = (ge[lo:z + 1], gx[lo:z + 1])
                    res = run_window(d, rf.iloc[lo:z + 1], WARM, z - lo, BASE | over, wc)
                    rows.append(flatten(t, uni, st, name, res))
                # nulls: baseline spec on re-ordered returns of the same span (close fills only)
                d = full["tr"].iloc[lo:z + 1]
                for kind in NULLS:
                    for rep in range(1 if kind == "reversed" else N_NULL):
                        p = null_path(d["C"].values, kind, rng)
                        dn = pd.DataFrame({"O": p, "H": p, "L": p, "C": p}, index=d.index)
                        for name in NULL_SPECS:
                            res = run_window(dn, rf.iloc[lo:z + 1], WARM, z - lo, BASE | SPECS[name])
                            tag = f"null_{kind}" + ("" if name == "baseline" else f":{name}")
                            rows.append(flatten(t, uni, st, tag, res))
            print(f"{uni:12s} {t:8s} done ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "panel.csv.gz", index=False, float_format="%.5g")
    summarise(df)


def edges(df: pd.DataFrame) -> pd.DataFrame:
    e = pd.DataFrame({
        "core_vs_mix50": df["core_sharpe"] - df["mix50_sharpe"],
        "core_vs_matched": df["core_sharpe"] - df["mixm_core_sharpe"],
        "grid_vs_matched": df["grid_sharpe"] - df["mixm_grid_sharpe"],
        "tvt_vs_matched": df["tvt_sharpe"] - df["mixm_tvt_sharpe"],
        "core_cagr_vs_matched": df["core_cagr"] - df["mixm_core_cagr"],
        "grid_cagr_vs_matched": df["grid_cagr"] - df["mixm_grid_cagr"],
        "tvt_cagr_vs_matched": df["tvt_cagr"] - df["mixm_tvt_cagr"],
        "tvt_dd_vs_matched": df["tvt_maxdd"] - df["mixm_tvt_maxdd"],
        "core_total_vs_matched": df["core_total"] - df["mixm_core_total"],
        "grid_total_vs_matched": df["grid_total"] - df["mixm_grid_total"],
        "tvt_total_vs_matched": df["tvt_total"] - df["mixm_tvt_total"],
    })
    return pd.concat([df[["ticker", "universe", "start", "spec"]], e], axis=1)


def summarise(df: pd.DataFrame) -> None:
    df["start"] = pd.to_datetime(df["start"])
    e = edges(df)
    cols = [c for c in e.columns if c not in ("ticker", "universe", "start", "spec")]
    order = list(SPECS) + [f"null_{k}" + ("" if n == "baseline" else f":{n}") for n in NULL_SPECS for k in NULLS]

    med = e.groupby("spec")[cols].median().reindex(order)
    pos = e.groupby("spec")[cols].apply(lambda g: (g > 0).mean() * 100).reindex(order)
    # non-overlapping windows: cross-sectional median per window, then count positive windows
    no = e[e["start"].isin(NONOVERLAP)].groupby(["spec", "start"])[cols].median()
    nopos = no.groupby("spec").apply(lambda g: (g > 0).sum()).reindex(order)
    level = df.groupby("spec")[[f"{k}_{m}" for k in ("bh", "mix50", "grid", "core", "tvt")
                                for m in ("total", "cagr", "sharpe", "maxdd")] + ["core_expo", "grid_expo", "tvt_expo"]]
    level = level.median().reindex(order)

    med.round(3).to_csv(OUT / "edge_median.csv")
    pos.round(1).to_csv(OUT / "edge_pct_positive.csv")
    nopos.to_csv(OUT / "edge_nonoverlap_windows_positive.csv")
    level.round(3).to_csv(OUT / "levels_median.csv")
    by_uni = e[e["spec"] == "baseline"].groupby("universe")[cols].median().round(3)
    by_uni.to_csv(OUT / "baseline_edge_by_universe.csv")
    by_win = e[e["spec"] == "baseline"].groupby("start")[cols].median().round(3)
    by_win.to_csv(OUT / "baseline_edge_by_window.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    n = e[e.spec == "baseline"].shape[0]
    print(f"\nticker-windows per spec: {n}; non-overlapping windows: {len(NONOVERLAP)}")
    print("\nMedian edge (Sharpe or CAGR %-points)\n", med.round(3))
    print("\n% of ticker-windows with a positive edge\n", pos.round(0))
    print(f"\nNon-overlapping windows (of {len(NONOVERLAP)}) with a positive median edge\n", nopos)
    print("\nMedian levels\n", level.round(2))
    print("\nBaseline edge by universe\n", by_uni)
    print("\nBaseline edge by window\n", by_win)


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, float) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep, *body])


def report() -> None:
    """REPORT.md: medians over ticker-windows of total return, CAGR, Sharpe and drawdown, and the edges."""
    lv = pd.read_csv(OUT / "levels_median.csv", index_col=0)
    em = pd.read_csv(OUT / "edge_median.csv", index_col=0)
    ep = pd.read_csv(OUT / "edge_pct_positive.csv", index_col=0)
    en = pd.read_csv(OUT / "edge_nonoverlap_windows_positive.csv", index_col=0)
    out = ["# Specification lab: median results over rolling 3-year windows", "",
           f"2,000 ticker-windows per spec (47 tickers, a new 3-year window every six months, 2001-2023 "
           f"starts). Medians across ticker-windows. Total return is over the 3-year window; CAGR is annualised.", ""]
    for k, name in (("bh", "Buy & hold"), ("mix50", "50/50 monthly"), ("grid", "Grid"),
                    ("core", "Core blend"), ("tvt", "Trend + vol target")):
        t = lv[[f"{k}_total", f"{k}_cagr", f"{k}_sharpe", f"{k}_maxdd"]]
        t.columns = ["total %", "CAGR %", "Sharpe", "max DD %"]
        out += [f"## {name}", "", md(t.rename_axis("spec")), ""]
    e = pd.DataFrame({
        "core Sharpe vs mix50": em["core_vs_mix50"],
        "core Sharpe vs matched": em["core_vs_matched"],
        "core % pos": ep["core_vs_matched"], "core windows +/8": en["core_vs_matched"],
        "core CAGR vs matched": em["core_cagr_vs_matched"], "core total vs matched": em["core_total_vs_matched"],
        "grid Sharpe vs matched": em["grid_vs_matched"], "grid % pos": ep["grid_vs_matched"],
        "grid CAGR vs matched": em["grid_cagr_vs_matched"], "grid total vs matched": em["grid_total_vs_matched"],
        "tvt Sharpe vs matched": em["tvt_vs_matched"], "tvt % pos": ep["tvt_vs_matched"],
        "tvt CAGR vs matched": em["tvt_cagr_vs_matched"], "tvt total vs matched": em["tvt_total_vs_matched"],
        "tvt max DD vs matched": em["tvt_dd_vs_matched"],
    }).rename_axis("spec")
    out += ["## Edges against the exposure-matched constant mix", "",
            "Sharpe differences; CAGR and total-return differences in percentage points; '% pos' is the share "
            "of ticker-windows with a positive Sharpe edge; 'windows +/8' counts the non-overlapping windows "
            "(2001, 2004, ... 2022 starts) whose cross-sectional median edge is positive.", "", md(e), ""]
    (OUT / "REPORT.md").write_text("\n".join(out))


if __name__ == "__main__":
    import sys
    report() if "--report" in sys.argv else (main(), report())
