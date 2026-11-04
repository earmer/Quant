"""The N -> N+1 self-discovery test on A-shares (CSI 300), with A-share trading rules.

Same models and split as discover.py (no pairs specified; learn 2020, 2022, 2024; validate 2021,
2023, 2025; plus 2026 to date as a fresh check), on today's CSI 300 members with Yahoo Finance data
(so survivorship bias applies). BaoStock was not reachable from this environment.

A-share rules
  1. Suspensions: Volume == 0 or no row on an exchange trading day (calendar of 510300.SS).
     Limit days, from the price change of Yahoo's unadjusted-for-dividends Close:
       main board  |change| >= 9.8 %
       ChiNext (300xxx/301xxx.SZ) >= 19.8 % from 2020-08-24, >= 9.8 % before (old regime)
       STAR (688xxx.SS) >= 19.8 %
     or High == Low with a move of at least 9.0 % / 18.0 % (a one-price limit day where tick
     rounding keeps the change just under the threshold).
  2. Trading at each close: no buying on a limit-up day (no new or added long, no short cover);
     no selling on a limit-down day (no exit of a long, no new short); no trading at all while
     suspended. Blocked positions are carried at market until a tradeable day, so a limit-down
     crash cannot be dodged.
  3. Costs 8 bp one-way. Reported: the top-minus-bottom-decile long-short book (comparable with the
     S&P 500 run) and a long-only top-decile book, with its active return over an equal-weighted,
     daily-rebalanced universe (no costs charged to the benchmark).
  4. Macro inputs: onshore ETFs 510300.SS (CSI 300), 510500.SS (CSI 500), 159915.SZ (ChiNext),
     518880.SS (gold), 511010.SS (5-year government bonds). No US series: the US trades after the
     A-share close, so same-day US data would leak.

    python backtests/discover_ashare.py --phase train
    python backtests/discover_ashare.py --phase validate
"""
from __future__ import annotations

import argparse
import io
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

import discover as D
import discover_lags as DL

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_discover_ashare"
CACHE = HERE / ".cache"
RAW = CACHE / "ashare_ohlcv_to_2026-11-03.pkl"
CONS_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000300cons.xls"
CONS_FILE = HERE / "data" / "csi300_constituents_2026-11-03.csv"
A_ETFS = ["510300.SS", "510500.SS", "159915.SZ", "518880.SS", "511010.SS"]
CALENDAR = "510300.SS"
COST = 8.0 / 1e4
FRESH = [2026]
SEED = 20261106
CHINEXT_REFORM = pd.Timestamp("2020-08-24")

# the shared model code reads these module globals
D.ETFS = A_ETFS
D.COMMON = set(A_ETFS) | {f"F{k + 1}" for k in range(D.K_FACTORS)}


# ============================================================================ data
def constituents() -> pd.DataFrame:
    if not CONS_FILE.exists():
        raw = urllib.request.urlopen(CONS_URL, timeout=60).read()
        d = pd.read_excel(io.BytesIO(raw))
        code = d["成份券代码Constituent Code"].astype(int).astype(str).str.zfill(6)
        sfx = np.where(d["交易所英文名称Exchange(Eng)"].str.startswith("Shanghai"), ".SS", ".SZ")
        out = pd.DataFrame({"ticker": code + sfx, "name": d["成份券英文名称Constituent Name(Eng)"],
                            "exchange": d["交易所英文名称Exchange(Eng)"], "as_of": d["日期Date"]})
        out.to_csv(CONS_FILE, index=False)
    return pd.read_csv(CONS_FILE, dtype={"ticker": str})


def load() -> tuple[dict, list[str]]:
    cons = constituents()
    tick = list(cons["ticker"]) + A_ETFS
    if not RAW.exists():
        frames = {}
        for i in range(0, len(tick), 40):
            b = tick[i:i + 40]
            df = yf.download(b, start="2019-06-01", end="2026-11-04", auto_adjust=False, progress=False,
                             threads=False, group_by="column")
            for f in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
                frames.setdefault(f, []).append(df[f])
        RAW.write_bytes(pickle.dumps({f: pd.concat(v, axis=1) for f, v in frames.items()}))
    raw = pickle.loads(RAW.read_bytes())
    cal = raw["Close"][CALENDAR].dropna().index                     # exchange trading days
    raw = {f: v.reindex(cal) for f, v in raw.items()}
    span = (cal.year >= 2020) & (cal.year <= 2025)
    # history must start by 2019 (lags, warm-up) and prices must exist on >= 95 % of 2020-25 days
    have = raw["Adj Close"].loc[cal[cal.year == 2019]].notna().any()
    cover = raw["Adj Close"].loc[span].notna().mean()
    stocks = [t for t in cons["ticker"] if t in raw["Adj Close"] and have.get(t, False) and cover.get(t, 0) >= 0.95]
    return raw, stocks


def board_limit(ticker: str, idx: pd.DatetimeIndex) -> np.ndarray:
    code = ticker[:3]
    if code == "688":
        return np.full(len(idx), 0.198)
    if code in ("300", "301"):
        return np.where(idx >= CHINEXT_REFORM, 0.198, 0.098)
    return np.full(len(idx), 0.098)


def tradability(raw: dict, stocks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Boolean frames (days x stocks): suspended, limit_up, limit_down."""
    C, H, L, V = (raw[f][stocks] for f in ("Close", "High", "Low", "Volume"))
    susp = V.isna() | (V == 0) | C.isna()
    chg = C / C.ffill().shift(1) - 1
    lim = pd.DataFrame({t: board_limit(t, C.index) for t in stocks}, C.index)
    flat = (H - L).abs() < 1e-9
    flat_thr = lim.where(lim < 0.15, 0.18).where(lim > 0.15, 0.09)       # 9 % main board, 18 % 20 %-limit boards
    up = ((chg >= lim) | (flat & (chg >= flat_thr))) & ~susp
    dn = ((chg <= -lim) | (flat & (chg <= -flat_thr))) & ~susp
    return susp, up, dn


def price_frame(raw: dict, stocks: list[str]) -> pd.DataFrame:
    """Adjusted closes, forward-filled through suspensions (return 0 while suspended, the gap on
    resumption), with the ETFs alongside; the shape discover.Prep expects."""
    px = raw["Adj Close"][stocks + A_ETFS].ffill()
    return px


class APrep(D.Prep):
    """discover.Prep, with residuals forced to zero on days a stock did not trade."""

    def __init__(self, px, stocks, fit_years, susp):
        self.susp = susp
        super().__init__(px, stocks, fit_years)

    def _resid(self, R):
        u, f = super()._resid(R)
        s = self.susp.reindex(u.index).reindex(columns=u.columns).fillna(False).values
        return u.mask(s, 0.0), f


# ============================================================================ trading with limits
def simulate(pred: pd.DataFrame, ret_next: pd.DataFrame, susp, up, dn, long_only: bool) -> pd.DataFrame:
    """Rebalance at each close d to the target book (top decile long, and bottom decile short unless
    long_only), subject to: suspended -> no trade; limit-up -> no buys; limit-down -> no sells.
    Positions drift with returns; P&L over d+1 uses the adjusted close-to-close return; 8 bp
    one-way on traded value. Values are fractions of NAV."""
    stocks = list(pred.columns)
    v = np.zeros(len(stocks))
    nav = 1.0
    rows = []
    S, U, Dn = (x.reindex(index=pred.index, columns=stocks).fillna(False).values for x in (susp, up, dn))
    R = ret_next.reindex(index=pred.index, columns=stocks).fillna(0.0).values
    P = pred.values
    for i, d in enumerate(pred.index):
        p = P[i]
        ok = ~np.isnan(p)
        q = max(int(ok.sum() * 0.1), 1)
        order = np.argsort(np.where(ok, p, np.nan))
        valid = order[: ok.sum()]
        tgt = np.zeros(len(stocks))
        if long_only:
            tgt[valid[-q:]] = nav / q
        else:
            tgt[valid[-q:]] = 0.5 * nav / q
            tgt[valid[:q]] = -0.5 * nav / q
        want = tgt - v
        blocked = S[i] | ((want > 0) & U[i]) | ((want < 0) & Dn[i])
        trade = np.where(blocked, 0.0, want)
        cost = COST * np.abs(trade).sum()
        v = v + trade
        pnl = (v * R[i]).sum()
        v = v * (1 + R[i])
        nav_new = nav + pnl - cost                                       # idle cash earns nothing
        rows.append((d, nav_new / nav - 1, cost / nav, int((blocked & (np.abs(want) > 1e-12)).sum())))
        nav = nav_new
    return pd.DataFrame(rows, columns=["day", "ret", "cost", "blocked_orders"]).set_index("day")


def gross_simulate(pred, ret_next, susp, up, dn, long_only):
    """Same as simulate with zero costs (for the before-cost columns)."""
    global COST
    c, COST = COST, 0.0
    try:
        return simulate(pred, ret_next, susp, up, dn, long_only)
    finally:
        COST = c


def perf(x: pd.Series) -> dict:
    tot = np.prod(1 + x)
    yrs = len(x) / 243
    return {"total_%": (tot - 1) * 100, "cagr_%": (tot ** (1 / yrs) - 1) * 100,
            "sharpe": x.mean() / x.std() * np.sqrt(243) if x.std() > 0 else np.nan, "nw_t": D.nw_t(x)}


def book_report(pred, ret_next, susp, up, dn) -> dict:
    out = {}
    ls_g = gross_simulate(pred, ret_next, susp, up, dn, False)
    ls_n = simulate(pred, ret_next, susp, up, dn, False)
    lo_g = gross_simulate(pred, ret_next, susp, up, dn, True)
    lo_n = simulate(pred, ret_next, susp, up, dn, True)
    ew = ret_next.reindex(index=pred.index, columns=pred.columns).fillna(0.0).mean(axis=1)
    for tag, s in (("ls_gross", ls_g["ret"]), ("ls_net", ls_n["ret"]), ("lo_net", lo_n["ret"]),
                   ("lo_active_gross", lo_g["ret"] - ew), ("lo_active_net", lo_n["ret"] - ew), ("ew_universe", ew)):
        for k, v in perf(s).items():
            out[f"{tag}_{k}"] = v
    out["ls_blocked_orders_per_day"] = ls_n["blocked_orders"].mean()
    return out


# ============================================================================ phases
def setup():
    raw, stocks = load()
    susp, up, dn = tradability(raw, stocks)
    px = price_frame(raw, stocks)
    meta = pd.DataFrame(index=stocks)
    return raw, stocks, susp, up, dn, px


def run_train() -> None:
    out = OUT / "train"; out.mkdir(parents=True, exist_ok=True)
    raw, stocks, susp, up, dn, px = setup()
    prep = APrep(px, stocks, D.TRAIN, susp)
    X, Y1, _, grp, _ = D.design(prep, D.TRAIN)
    Xn, Y1n, _, grpn, _ = D.design(prep, D.TRAIN, np.random.default_rng(SEED))
    lim = pd.DataFrame({"suspended_%": susp[stocks].mean() * 100, "limit_up_%": up.mean() * 100,
                        "limit_down_%": dn.mean() * 100})
    lim.round(3).to_csv(out / "tradability_by_stock.csv")
    print(f"stocks {len(stocks)}, inputs {X.shape[1]}, training days {len(X)}; "
          f"suspended {susp.values.mean() * 100:.2f} %, limit-up {up.values.mean() * 100:.2f} %, "
          f"limit-down {dn.values.mean() * 100:.2f} % of stock-days (2019-2026)", flush=True)
    Z = pd.concat([prep.year(y)[0].loc[lambda d: d.index.year == y] for y in D.TRAIN])
    pc = D.partial_corr(Z)
    pc.to_pickle(CACHE / "ashare_pc_train.pkl")
    s = pc.where(np.triu(np.ones(pc.shape, bool), 1)).stack()
    s.reindex(s.abs().sort_values(ascending=False).index)[:D.TOP_EDGES].rename("partial_corr").to_csv(
        out / "structure_top_links.csv", float_format="%.4g")
    summ = {}
    for tag, XX, YY, gg in (("real", X, Y1, grp), ("null", Xn, Y1n, grpn)):
        C, r2 = DL.fit_lasso_par(XX, YY, gg)
        C.to_pickle(CACHE / f"ashare_lasso_{tag}.pkl")
        E = D.edges_of(C)
        E.to_csv(out / f"edges_{tag}.csv", index=False, float_format="%.4g")
        summ[tag] = dict(cross_edges=int((~E["own"]).sum()), own_edges=int(E["own"].sum()),
                         median_cv_r2_pct=float(r2.median() * 100))
        print(tag, summ[tag], flush=True)
    pd.DataFrame(summ).T.to_csv(out / "summary.csv", float_format="%.5g")
    pos, neg, cl = D.peer_maps(Z)
    Dp = D.gbm_panel(X, Y1, stocks, pos, neg, cl)
    Dp = Dp[Dp["y"].notna()]
    feats = [c for c in Dp.columns if c not in ("y", "day", "stock")]
    gbm = HistGradientBoostingRegressor(**D.HGB).fit(Dp[feats], Dp["y"])
    pickle.dump((gbm, feats, pos, neg, cl), open(CACHE / "ashare_gbm.pkl", "wb"))


def run_validate() -> None:
    out = OUT / "validate"; out.mkdir(parents=True, exist_ok=True)
    raw, stocks, susp, up, dn, px = setup()
    prep = APrep(px, stocks, D.TRAIN, susp)
    ret_next = px[stocks].pct_change().shift(-1)
    gbm, feats, pos, neg, cl = pickle.load(open(CACHE / "ashare_gbm.pkl", "rb"))
    rows = {}
    for label, years in (("valid_2021_23_25", D.VALID), ("2021", [2021]), ("2023", [2023]), ("2025", [2025]),
                         ("fresh_2026", FRESH)):
        X, Y1, _, _, _ = D.design(prep, years)
        preds = {}
        for tag in ("real", "null"):
            preds[f"lasso_{tag}"] = D.predict(pd.read_pickle(CACHE / f"ashare_lasso_{tag}.pkl"), X)
        Dp = D.gbm_panel(X, Y1, stocks, pos, neg, cl)
        preds["gbm"] = pd.DataFrame(gbm.predict(Dp[feats]).reshape(len(X), len(stocks)), X.index, stocks)
        for m, P in preds.items():
            ic = D.daily_ic(P, Y1)
            rows[(label, m)] = {"ic_mean": ic.mean(), "ic_nw_t": D.nw_t(ic), **book_report(P, ret_next, susp, up, dn)}
        print(label, "done", flush=True)
    T = pd.DataFrame(rows).T
    T.index.names = ["years", "model"]
    T.round(4).to_csv(out / "forecast_and_trading.csv")
    # structure persistence
    Zv = pd.concat([prep.year(y)[0].loc[lambda d: d.index.year == y] for y in D.VALID])
    a = pd.read_pickle(CACHE / "ashare_pc_train.pkl"); b = D.partial_corr(Zv)
    iu = np.triu_indices(len(stocks), 1)
    av, bv = a.values[iu], b.values[iu]
    tt = set(np.argsort(-np.abs(av))[:D.TOP_EDGES]); tv = set(np.argsort(-np.abs(bv))[:D.TOP_EDGES])
    st = pd.Series({"partial_corr_train_vs_valid_corr": np.corrcoef(av, bv)[0, 1],
                    "top1000_overlap_%": len(tt & tv) / D.TOP_EDGES * 100,
                    "top1000_overlap_by_chance_%": D.TOP_EDGES / len(av) * 100})
    st.to_csv(out / "structure_persistence.csv", float_format="%.4g")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    cols = ["ic_mean", "ic_nw_t", "ls_gross_cagr_%", "ls_gross_total_%", "ls_net_cagr_%", "ls_net_total_%", "ls_net_sharpe",
            "lo_net_cagr_%", "lo_net_total_%", "ew_universe_cagr_%", "lo_active_gross_cagr_%", "lo_active_net_cagr_%",
            "lo_active_net_total_%", "lo_active_net_nw_t", "ls_blocked_orders_per_day"]
    print(T[cols].round(3))
    print("\nStructure persistence\n", st.round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "validate"], required=True)
    run_train() if ap.parse_args().phase == "train" else run_validate()


if __name__ == "__main__":
    main()
