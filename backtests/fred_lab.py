"""Macro context from FRED: does it explain, time or size the strategies?

Series (daily and market-based, so essentially never revised; each value is used only from the
next business day, so the backtest never sees a print before it was published):
  DTB3     3-month T-bill, secondary market   -> the cash rate here (instead of Yahoo's ^IRX)
  T10Y3M   10-year minus 3-month Treasury     -> yield-curve inversion
  BAA10Y   Moody's Baa minus 10-year Treasury -> credit stress
  VIXCLS   CBOE VIX                           -> implied equity volatility
  USREC    NBER recession months              -> DESCRIPTIVE ONLY: dated with hindsight, never traded

Three experiments, all on the 47 tickers of spec_lab.py, sleeves run continuously from each
ticker's first price (grid: 3 %, 5 levels, breakout re-centring; core: 50 % buy & hold + 50 % grid):

  1. Regimes   Monthly excess returns of each strategy in each macro state (37 stocks only).
  2. Switches  Rules fixed in advance, decided at each month-end and traded at the next close,
               holding the base (core, or buy & hold) at half size when risk-off:
                 curve   T10Y3M < 0          credit  BAA10Y above its 252-session average
                 vix     VIX > 25            combo   at least two of the three
                 sma200  the stock below its own 200-day average (reference rule)
               Each is compared with the base held at the same *average* size throughout
               (exposure-matched), so de-risking itself is not credited as timing skill.
  3. Volatility  Walk-forward (annual, expanding, pooled) forecasts of next-month log volatility:
               naive (21-day), HAR (5/21/63-day), HAR + log VIX + credit spread + curve.
               Out-of-sample R^2 overall and on 2013, 2014, 2017 and 2021, years never used as a
               holdout. The forecasts then size the core blend towards 15 % volatility.

No parameter here was tuned on results; the thresholds above are conventional round numbers
written down before the first run. Windows: rolling 3 years, starting every six months 2004-2023
(forecasts need three years of training data).

    python backtests/fred_lab.py
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LinearRegression

import spec_lab as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_fred"
SERIES = ["DTB3", "T10Y3M", "BAA10Y", "VIXCLS", "USREC"]
STARTS = pd.date_range("2004-01-01", "2023-07-01", freq="6MS")
NONOVERLAP = pd.DatetimeIndex([f"{y}-01-01" for y in range(2004, 2023, 3)])
FRESH_YEARS = [2013, 2014, 2017, 2021]
C = 5.0 / 1e4
VOL_TARGET = 0.15
EQUITY = [t for u in ("original", "traditional", "random") for t in S.UNIVERSES[u]]


# ============================================================================ data
def fred(sid: str) -> pd.Series:
    path = HERE / ".cache" / f"fred_{sid}_2026-11-01.csv"
    if not path.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    s = pd.read_csv(path, index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
    return s.dropna()


def known(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Value usable on each session of idx: the latest observation dated before that session."""
    s = s.copy()
    s.index = s.index + pd.offsets.BDay(1)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


# ============================================================================ engine
@njit(cache=True)
def blend_var(r1, e1, r2, e2, w, rebal, c):
    """Like spec_lab.blend_nb, but the weight of sleeve 1 can change at each rebalance."""
    n = len(r1)
    eq = np.empty(n); ex = np.empty(n)
    v1 = 100000.0 * w[0]; v2 = 100000.0 * (1 - w[0])
    cost = c * (w[0] * e1[0] + (1 - w[0]) * e2[0]) * 100000.0
    v1 -= cost * w[0]; v2 -= cost * (1 - w[0])
    for i in range(n):
        if i > 0:
            v1 *= 1 + r1[i]; v2 *= 1 + r2[i]
            if rebal[i]:
                V = v1 + v2
                cost = c * abs((w[i] - v1 / V) * e1[i] + ((1 - w[i]) - v2 / V) * e2[i]) * V
                V -= cost
                v1 = w[i] * V; v2 = (1 - w[i]) * V
        eq[i] = v1 + v2
        ex[i] = (v1 * e1[i] + v2 * e2[i]) / eq[i]
    return eq, ex


def rets(eq: np.ndarray) -> np.ndarray:
    return np.r_[0.0, eq[1:] / eq[:-1] - 1]


def sleeves(D: pd.DataFrame, rf: np.ndarray) -> dict:
    """Continuous daily returns/exposures of bh, mix50, grid, core, tvt from the first price."""
    O, H, L, Cl = (D[k].values for k in "OHLC")
    idx = D.index
    rebal = S.monthly_days(idx, 0)
    once = np.full(len(Cl), np.nan); once[0] = 1.0
    out = {}
    out["bh"] = S.orders_nb(Cl, rf, once, C, 1e-9)
    out["mix50"] = S.orders_nb(Cl, rf, np.where(rebal, 0.5, np.nan), C, 1e-9)
    out["grid"] = S.grid_nb(O, H, L, Cl, rf, 0.03, 5, 1, 0, C, 0.5)
    rb, rg = rets(out["bh"][0]), rets(out["grid"][0])
    out["core"] = S.blend_nb(rb, out["bh"][1], rg, out["grid"][1], 0.5, rebal, C)
    r = np.r_[np.nan, Cl[1:] / Cl[:-1] - 1]
    vol20 = pd.Series(r).rolling(20).std().values * np.sqrt(252)
    sma = pd.Series(Cl).rolling(200).mean().values
    tgt = np.r_[np.nan, (np.minimum(1.0, 0.15 / vol20) * (Cl > sma))[:-1]]
    out["tvt"] = S.orders_nb(Cl, rf, np.nan_to_num(tgt, nan=0.0), C, 0.10)
    return {k: (rets(v[0]), v[1]) for k, v in out.items()}


# ============================================================================ panel of month-ends
def month_panel(t: str, D: pd.DataFrame, macro: dict) -> pd.DataFrame:
    """One row per decision date (last session of a month) with features, switches and target."""
    idx, Cl = D.index, D["C"]
    r = Cl.pct_change()
    dec = idx[~idx.to_period("M").duplicated(keep="last")][:-1]
    pos = idx.get_indexer(dec)
    f = pd.DataFrame(index=dec)
    f["ticker"] = t
    for n in (5, 21, 63):
        f[f"lv{n}"] = np.log(r.rolling(n).std().values[pos] * np.sqrt(252))
    m = {k: v.values[pos] for k, v in macro.items()}
    f["lvix"] = np.log(m["VIXCLS"])
    f["baa"] = m["BAA10Y"]
    f["curve"] = m["T10Y3M"]
    f["off_curve"] = m["T10Y3M"] < 0
    f["off_credit"] = m["BAA10Y"] > m["BAA_MA"]
    f["off_vix"] = m["VIXCLS"] > 25
    f["off_combo"] = f[["off_curve", "off_credit", "off_vix"]].sum(axis=1) >= 2
    f["off_sma200"] = (Cl < Cl.rolling(200).mean()).values[pos]
    # target: realised vol over the 21 sessions after the decision
    rv = r.rolling(21).std().shift(-21).values * np.sqrt(252)
    f["y"] = np.log(rv[pos])
    end = np.minimum(pos + 21, len(idx) - 1)
    f["y_end"] = idx[end]
    f.loc[pos + 21 >= len(idx), "y"] = np.nan
    return f


def walk_forward(P: pd.DataFrame) -> pd.DataFrame:
    """Annual expanding-window refits, pooled across tickers; forecasts in vol units."""
    models = {"har": ["lv5", "lv21", "lv63"], "har_macro": ["lv5", "lv21", "lv63", "lvix", "baa", "curve"]}
    P = P.copy()
    P["f_naive"] = np.exp(P["lv21"])
    for k in models:
        P[f"f_{k}"] = np.nan
    ok = P[["lv5", "lv21", "lv63", "lvix", "baa", "curve"]].notna().all(axis=1) & np.isfinite(P["lv5"])
    years = sorted(set(P.index.year))
    for y in years:
        tr = ok & P["y"].notna() & (P["y_end"] < pd.Timestamp(f"{y}-01-01"))
        te = ok & (P.index.year == y)
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        for k, cols in models.items():
            mdl = LinearRegression().fit(P.loc[tr, cols], P.loc[tr, "y"])
            P.loc[te, f"f_{k}"] = np.exp(mdl.predict(P.loc[te, cols]))
    return P


def r2(y, f) -> float:
    m = np.isfinite(y) & np.isfinite(f)
    y, f = y[m], f[m]
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ============================================================================ windows
def win_metrics(eq, ex, rf, yrs, scale) -> dict:
    cagr, sh, dd, expo, tot = S.metrics(eq, ex, rf, yrs)
    return dict(total=tot, cagr=cagr, sharpe=sh, maxdd=dd, expo=expo, scale=scale)


OVERLAYS = ["curve", "credit", "vix", "combo", "sma200", "vol_naive", "vol_har", "vol_har_macro"]


def main() -> None:
    data, _ = S.load()
    raw = {k: fred(k) for k in SERIES}
    OUT.mkdir(exist_ok=True)
    panels, rows, monthly = [], [], []
    per_ticker = {}
    for uni, tickers in S.UNIVERSES.items():
        for t in tickers:
            if t not in data:
                continue
            D = data[t]["tr"]
            idx = D.index
            macro = {k: known(raw[k], idx) for k in ("DTB3", "T10Y3M", "BAA10Y", "VIXCLS")}
            macro["BAA_MA"] = known(raw["BAA10Y"].rolling(252).mean(), idx)
            y = macro["DTB3"].ffill().bfill().values / 100
            rf = (1 + y) ** (1 / 252) - 1
            sl = sleeves(D, rf)
            P = month_panel(t, D, macro)
            P["universe"] = uni
            panels.append(P)
            per_ticker[t] = (uni, D, rf, sl)
            # monthly returns of each sleeve for the regime table
            if t in EQUITY:
                mret = {k: pd.Series(v[0], idx).add(1).groupby(idx.to_period("M")).prod() - 1 for k, v in sl.items()}
                mrf = pd.Series(rf, idx).add(1).groupby(idx.to_period("M")).prod() - 1
                mm = pd.DataFrame(mret).sub(mrf, axis=0)
                mm["ticker"] = t
                monthly.append(mm)
            print(f"{uni:12s} {t:8s} sleeves done", flush=True)

    P = walk_forward(pd.concat(panels))
    P.to_csv(OUT / "month_panel.csv.gz", float_format="%.5g")
    vol_table(P)
    regime_table(pd.concat(monthly), P, raw["USREC"])

    for t, (uni, D, rf, sl) in per_ticker.items():
        idx = D.index
        rebal = S.monthly_days(idx, 0)
        Pt = P[P["ticker"] == t]
        # scale decided at each month-end, applied from the next session
        sc = {}
        for k in ("curve", "credit", "vix", "combo", "sma200"):
            sc[k] = np.where(Pt[f"off_{k}"].values, 0.5, 1.0)
        for k in ("naive", "har", "har_macro"):
            sc[f"vol_{k}"] = np.minimum(1.0, VOL_TARGET / Pt[f"f_{k}"].values)
        exec_pos = idx.get_indexer(Pt.index) + 1
        for st in STARTS:
            a = idx.searchsorted(st)
            end = st + pd.DateOffset(years=S.YEARS)
            z = idx.searchsorted(end) - 1
            if a < S.WARM or idx[-1] < end - pd.Timedelta(days=7):
                continue
            s_ = slice(a, z + 1)
            yrs = (idx[z] - idx[a]).days / 365.25
            n = z + 1 - a
            for base in ("core", "bh"):
                r1, e1 = sl[base][0][s_], sl[base][1][s_]
                r2, e2 = rf[s_], np.zeros(n)
                rb = rebal[s_].copy(); rb[0] = True
                full = np.ones(n)
                eq, ex = blend_var(r1, e1, r2, e2, full, rb, C)
                rec = dict(ticker=t, universe=uni, start=st, base=base)
                rows.append({**rec, "rule": "none", **win_metrics(eq, ex, rf[s_], yrs, 1.0)})
                for k, v in sc.items():
                    if base == "bh" and k.startswith("vol"):
                        continue
                    w = pd.Series(np.nan, idx)
                    ok = exec_pos < len(idx)
                    w.iloc[exec_pos[ok]] = v[ok]
                    w = w.ffill().values[s_]
                    if np.isnan(w).any():
                        break
                    eq, ex = blend_var(r1, e1, r2, e2, w, rb, C)
                    m = win_metrics(eq, ex, rf[s_], yrs, w[rb].mean())
                    eqm, exm = blend_var(r1, e1, r2, e2, np.full(n, m["scale"]), rb, C)
                    mm = win_metrics(eqm, exm, rf[s_], yrs, m["scale"])
                    rows.append({**rec, "rule": k, **m, **{f"matched_{q}": mm[q] for q in ("total", "cagr", "sharpe", "maxdd")}})
        print(f"{t:8s} windows done", flush=True)
    W = pd.DataFrame(rows)
    W.to_csv(OUT / "windows.csv.gz", index=False, float_format="%.5g")
    switch_table(W)


# ============================================================================ tables
def vol_table(P: pd.DataFrame) -> None:
    rows = {}
    eq = P["universe"] != "assets"
    for name, mask in (("all, 2004-2026", P.index.year >= 2004),
                       ("all, fresh years", P.index.year.isin(FRESH_YEARS)),
                       ("stocks, 2004-2026", (P.index.year >= 2004) & eq),
                       ("non-equity ETFs, 2004-2026", (P.index.year >= 2004) & ~eq)):
        sub = P[mask & P[["f_naive", "f_har", "f_har_macro"]].notna().all(axis=1) & P["y"].notna()]
        rows[name] = {k: r2(sub["y"].values, np.log(sub[f"f_{k}"].values)) for k in ("naive", "har", "har_macro")}
        rows[name]["n"] = len(sub)
    T = pd.DataFrame(rows).T
    T.round(3).to_csv(OUT / "vol_r2.csv")
    print("\nOut-of-sample R^2 of next-month log volatility\n", T.round(3))


def regime_table(M: pd.DataFrame, P: pd.DataFrame, usrec: pd.Series) -> None:
    """Equal-weight across the 37 stocks per month, then annualised mean / Sharpe by regime."""
    strat = ["bh", "mix50", "grid", "core", "tvt"]
    ew = M.groupby(level=0)[strat].mean()
    # regime known at the previous month-end (from any equity ticker's panel: macro is common)
    spy = P[P["ticker"] == "SPY"].copy()
    spy.index = spy.index.to_period("M") + 1
    reg = spy[["off_curve", "off_credit", "off_vix"]].reindex(ew.index)
    rec = usrec.copy(); rec.index = rec.index.to_period("M")
    reg["recession (ex post)"] = rec.reindex(ew.index) == 1
    reg.columns = ["curve inverted", "credit above 1y avg", "VIX > 25", "recession (ex post)"]
    ew = ew[(ew.index >= pd.Period("2001-01", "M")) & reg.notna().all(axis=1)]
    reg = reg.loc[ew.index]
    out = []
    for col in reg.columns:
        for state in (True, False):
            sub = ew[reg[col] == state]
            row = {"regime": col, "state": "yes" if state else "no", "months": len(sub)}
            for s in strat:
                row[f"{s}_ann_%"] = sub[s].mean() * 1200
                row[f"{s}_sharpe"] = sub[s].mean() / sub[s].std() * np.sqrt(12)
            row["grid_minus_mix50_ann_%"] = (sub["grid"] - sub["mix50"]).mean() * 1200
            row["core_minus_bh_mix50_ann_%"] = (sub["core"] - 0.5 * sub["bh"] - 0.5 * sub["mix50"]).mean() * 1200
            out.append(row)
    T = pd.DataFrame(out).set_index(["regime", "state"])
    T.round(2).to_csv(OUT / "regimes.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nMonthly excess returns by macro regime (37 stocks, equal weight, annualised)\n", T.round(2))


def switch_table(W: pd.DataFrame) -> None:
    W["start"] = pd.to_datetime(W["start"])
    X = W[W["rule"] != "none"].copy()
    base = W[W["rule"] == "none"].set_index(["ticker", "start", "base"])
    for q in ("total", "cagr", "sharpe", "maxdd"):
        X[f"{q}_vs_base"] = X[q].values - base.loc[list(zip(X.ticker, X.start, X.base)), q].values
        X[f"{q}_vs_matched"] = X[q] - X[f"matched_{q}"]
    g = X.groupby(["base", "rule"])
    T = g[["total", "cagr", "sharpe", "maxdd", "scale", "total_vs_base", "sharpe_vs_base", "maxdd_vs_base",
           "total_vs_matched", "cagr_vs_matched", "sharpe_vs_matched", "maxdd_vs_matched"]].median()
    T["pct_pos_sharpe_vs_matched"] = g["sharpe_vs_matched"].apply(lambda s: (s > 0).mean() * 100)
    no = X[X["start"].isin(NONOVERLAP)].groupby(["base", "rule", "start"])["sharpe_vs_matched"].median()
    T[f"windows_pos_of_{len(NONOVERLAP)}"] = no.groupby(["base", "rule"]).apply(lambda s: (s > 0).sum())
    # the same edges restricted to windows in which the switch went risk-off at least once
    F = X[X["rule"].isin(["curve", "credit", "vix", "combo"]) & (X["scale"] < 0.999)]
    gf = F.groupby(["base", "rule"])
    TF = gf[["total_vs_matched", "sharpe_vs_matched", "maxdd_vs_matched"]].median()
    TF["n"] = gf.size()
    TF["pct_pos_sharpe_vs_matched"] = gf["sharpe_vs_matched"].apply(lambda s: (s > 0).mean() * 100)
    TF.round(3).to_csv(OUT / "switches_when_fired.csv")
    print("\nSwitches, windows where they fired\n", TF.round(3))
    B = W[W["rule"] == "none"].groupby("base")[["total", "cagr", "sharpe", "maxdd"]].median()
    B.round(2).to_csv(OUT / "base_levels.csv")
    T.round(3).to_csv(OUT / "switches.csv")
    eqm = X[X["universe"] != "assets"].groupby(["base", "rule"])["sharpe_vs_matched"].median()
    asm = X[X["universe"] == "assets"].groupby(["base", "rule"])["sharpe_vs_matched"].median()
    pd.DataFrame({"stocks": eqm, "non_equity": asm}).round(3).to_csv(OUT / "switches_by_asset_type.csv")
    print(f"\nTicker-windows per rule: {len(X) // X.groupby(['base', 'rule']).ngroups}")
    print("\nBase (no overlay), medians\n", B.round(2))
    print("\nOverlays, medians\n", T.round(3))
    print("\nSharpe vs matched by asset type\n", pd.DataFrame({"stocks": eqm, "non_equity": asm}).round(3))


if __name__ == "__main__":
    main()
