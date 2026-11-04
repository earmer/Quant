"""Grid-like strategies on onshore A-share ETFs, with A-share trading rules.

Universe (17 liquid ETFs that trade today, so there is selection bias):
  broad      510050 SSE 50, 510300 CSI 300, 510500 CSI 500, 512100 CSI 1000, 159915 ChiNext, 588000 STAR 50
  sector     512880 brokers, 512480 semiconductors, 512170 healthcare, 512690 liquor
  defensive  518880 gold, 511010 5-year govt bonds, 511260 10-year govt bonds
  foreign    513100 Nasdaq 100, 513500 S&P 500, 159920 Hang Seng, 510900 H-shares

Data fixes and rules
  * Yahoo's dividend-adjusted closes (Adj Close) are used for returns. Yahoo does not adjust these funds
    for unit splits/consolidations; any one-day change beyond 21 % (impossible under a 10 % or 20 %
    limit) is treated as a split and earlier prices are rescaled.
  * Suspension: volume 0 or missing -> no trading. Price limits from the close-to-close change of the
    unadjusted close: +/-9.8 %, or +/-19.8 % for the ChiNext ETF from 2020-08-24 and the STAR ETF.
    No buying at a limit-up close, no selling at a limit-down close; blocked orders wait.
  * T+1 does not bind here: every rule trades at most once a day, in one direction.
  * Costs: 3 bp one-way baseline (ETFs pay no stamp duty; commission plus half a tick of spread);
    1, 8 and 20 bp tested. Cash earns China's 3-month interbank rate (FRED IR3TIB01CNM156N, the
    previous month's value). Sharpe uses that rate.

Strategies (as in spec_lab.py): buy & hold; 50/50 monthly; grid static (3 %, 10 levels);
grid re-centring on a breakout (3 %, 5 levels); grid trail-up (3 %, 5 levels); grid wide (5 %, 10
levels, breakout); core = 50 % buy & hold + 50 % re-centring grid; trend + vol target (reference).
Each grid and the core blend is also compared with a monthly constant mix at its own realised average
exposure (exposure-matched; chosen with hindsight, which favours the benchmark).

Windows: rolling 3 years, a new one every six months, starts 2014-01 .. 2023-07 (needs 250 prior
sessions). Null tests: the same returns re-ordered (iid, 21-day blocks), baseline spec.

    python backtests/ashare_etf_grid.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit

import spec_lab as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_ashare_etf_grid"
CACHE = HERE / ".cache" / "ashare_etfs_to_2026-11-03.pkl"
FRED_CN = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01CNM156N"
GROUPS = {
    "broad": ["510050.SS", "510300.SS", "510500.SS", "512100.SS", "159915.SZ", "588000.SS"],
    "sector": ["512880.SS", "512480.SS", "512170.SS", "512690.SS"],
    "defensive": ["518880.SS", "511010.SS", "511260.SS"],
    "foreign": ["513100.SS", "513500.SS", "159920.SZ", "510900.SS"],
}
TICKERS = [t for g in GROUPS.values() for t in g]
GROUP_OF = {t: g for g, v in GROUPS.items() for t in v}
NAMES = {"510050.SS": "SSE 50", "510300.SS": "CSI 300", "510500.SS": "CSI 500", "512100.SS": "CSI 1000",
         "159915.SZ": "ChiNext", "588000.SS": "STAR 50", "512880.SS": "Brokers", "512480.SS": "Semis",
         "512170.SS": "Healthcare", "512690.SS": "Liquor", "518880.SS": "Gold", "511010.SS": "5y CGB",
         "511260.SS": "10y CGB", "513100.SS": "Nasdaq 100", "513500.SS": "S&P 500", "159920.SZ": "Hang Seng",
         "510900.SS": "H-shares"}
DAYS = 243
YEARS, WARM = 3, 250
STARTS = pd.date_range("2014-01-01", "2023-07-01", freq="6MS")
NONOVERLAP = pd.DatetimeIndex([f"{y}-01-01" for y in range(2014, 2024, 3)])
SEED = 20261107
N_NULL = 3
BASE = dict(cost=3.0, fill=0, s=0.03, levels=5)
SPECS = {"baseline": {}, "cost_1": dict(cost=1.0), "cost_8": dict(cost=8.0), "cost_20": dict(cost=20.0),
         "fill_next_open": dict(fill=2),
         **{f"recentre_s{int(s * 100)}_l{l}": dict(s=s, levels=l) for s in (0.01, 0.02, 0.05, 0.08) for l in (5, 10)},
         "recentre_s3_l10": dict(levels=10)}


# ============================================================================ data
def load() -> tuple[dict, pd.Series]:
    if not CACHE.exists():
        data = {}
        for t in TICKERS:
            d = yf.download(t, start="2010-01-01", end="2026-11-04", auto_adjust=False, progress=False, threads=False)
            d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
            data[t] = d[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        rate = pd.read_csv(FRED_CN, index_col=0, parse_dates=True).iloc[:, 0]
        CACHE.write_bytes(pickle.dumps((data, rate)))
    data, rate = pickle.loads(CACHE.read_bytes())
    return {t: clean(t, d) for t, d in data.items()}, rate


def clean(t: str, d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=["Close"]).copy()
    d = d[d["Close"] > 0]
    chg = d["Close"].pct_change()
    split = chg.abs() > 0.21
    ratio = pd.Series(1.0, d.index)
    ratio[split] = (d["Close"] / d["Close"].shift(1))[split]
    # rescale every earlier price by the product of later split ratios
    back = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for c in ("Open", "High", "Low", "Close", "Adj Close"):
        d[c] = d[c] * back
    d["Adj Close"] = d["Adj Close"].fillna(d["Close"])
    k = d["Adj Close"] / d["Close"]
    out = pd.DataFrame({"O": d["Open"] * k, "H": d["High"] * k, "L": d["Low"] * k, "C": d["Adj Close"]})
    raw_chg = d["Close"].pct_change()
    lim = np.full(len(d), 0.098)
    if t == "588000.SS":
        lim[:] = 0.198
    if t == "159915.SZ":
        lim = np.where(d.index >= pd.Timestamp("2020-08-24"), 0.198, 0.098)
    susp = d["Volume"].isna() | (d["Volume"] <= 0)
    out["buy_ok"] = ~(susp | (raw_chg >= lim))
    out["sell_ok"] = ~(susp | (raw_chg <= -lim))
    out["split"] = split
    return out


def cash_rate(rate: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    r = rate.copy()
    r.index = r.index + pd.offsets.MonthBegin(1)                      # use last month's value
    y = r.reindex(r.index.union(idx)).ffill().reindex(idx).bfill() / 100
    return (1 + y) ** (1 / DAYS) - 1


# ============================================================================ engines with limits
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
def _to_w(cash, sh, w, p, c, bok, sok):
    tgt = (cash + sh * p) * w / p
    if tgt > sh and bok:
        return _buy(cash, sh, tgt - sh, p, c)
    if tgt < sh and sok:
        return _sell(cash, sh, sh - tgt, p, c)
    return cash, sh


@njit(cache=True)
def grid_lim(O, C, rf, s, levels, mode, fill, c, base, bok, sok):
    """spec_lab.grid_nb (close fills, or next-open fills with fill == 2) plus A-share blocks: a level
    whose order is blocked is not marked filled, so it is retried on the next tradeable day."""
    n = len(C)
    eq = np.empty(n); ex = np.empty(n)
    cash, sh = 100000.0, 0.0
    cash, sh = _to_w(cash, sh, base, C[0], c, bok[0], sok[0])
    ctr, Lv, unit = C[0], 0, min(sh, cash / C[0]) / levels
    g = np.log(1 + s)
    pend_q, pend_rec, pend_L = 0.0, False, 0
    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
            if fill == 2 and (pend_rec or pend_q != 0.0):
                if pend_rec:
                    cash, sh = _to_w(cash, sh, base, O[i], c, bok[i], sok[i])
                    unit = min(sh, cash / O[i]) / levels
                elif pend_q > 0 and bok[i]:
                    cash, sh = _buy(cash, sh, pend_q, O[i], c); Lv = pend_L
                elif pend_q < 0 and sok[i]:
                    cash, sh = _sell(cash, sh, -pend_q, O[i], c); Lv = pend_L
                pend_q, pend_rec = 0.0, False
            x = np.log(C[i] / ctr) / g
            up_out, dn_out = x > levels + 1, x < -(levels + 1)
            rec = (mode == 1 and (up_out or dn_out)) or (mode == 2 and up_out)
            if rec:
                if fill == 2:
                    pend_rec = True
                else:
                    cash, sh = _to_w(cash, sh, base, C[i], c, bok[i], sok[i])
                    unit = min(sh, cash / C[i]) / levels
                ctr, Lv = C[i], 0
            else:
                down = max(int(np.ceil(x)), -levels)
                up = min(int(np.floor(x)), levels)
                if down < Lv:
                    q = unit * (Lv - down)
                    if fill == 2:
                        pend_q, pend_L = q, down
                    elif bok[i]:
                        cash, sh = _buy(cash, sh, q, C[i], c); Lv = down
                elif up > Lv:
                    q = unit * (up - Lv)
                    if fill == 2:
                        pend_q, pend_L = -q, up
                    elif sok[i]:
                        cash, sh = _sell(cash, sh, q, C[i], c); Lv = up
        eq[i] = cash + sh * C[i]; ex[i] = sh * C[i] / eq[i]
    return eq, ex


@njit(cache=True)
def orders_lim(C, rf, target, c, band, bok, sok):
    n = len(C)
    eq = np.empty(n); ex = np.empty(n)
    cash, sh = 100000.0, 0.0
    want = np.nan
    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
        if not np.isnan(target[i]):
            want = target[i]
        if not np.isnan(want):
            v = cash + sh * C[i]
            if abs(sh * C[i] / v - want) > band:
                cash, sh = _to_w(cash, sh, want, C[i], c, bok[i], sok[i])
            if abs(sh * C[i] / (cash + sh * C[i]) - want) <= band:
                want = np.nan                                        # done; otherwise retry tomorrow
        eq[i] = cash + sh * C[i]; ex[i] = sh * C[i] / eq[i]
    return eq, ex


# ============================================================================ one window
def metrics(eq, ex, rf) -> dict:
    r = eq[1:] / eq[:-1] - 1
    x = r - rf[1:]
    yrs = len(eq) / DAYS
    tot = eq[-1] / eq[0]
    return {"total": (tot - 1) * 100, "cagr": (tot ** (1 / yrs) - 1) * 100,
            "sharpe": x.mean() / x.std() * np.sqrt(DAYS) if x.std() > 0 else np.nan,
            "maxdd": (eq / np.maximum.accumulate(eq) - 1).min() * 100, "expo": ex.mean()}


def run(d: pd.DataFrame, rf: np.ndarray, a: int, z: int, sp: dict) -> dict:
    O, C = d["O"].values, d["C"].values
    bok, sok = d["buy_ok"].values.astype(np.bool_), d["sell_ok"].values.astype(np.bool_)
    c = sp["cost"] / 1e4
    sl = slice(a, z + 1)
    Ow, Cw, rw, bw, sw = O[sl], C[sl], rf[sl], bok[sl], sok[sl]
    rebal = S.monthly_days(d.index[sl], 0)
    once = np.full(len(Cw), np.nan); once[0] = 1.0
    R = {}
    R["bh"] = orders_lim(Cw, rw, once, c, 1e-9, bw, sw)
    R["mix50"] = orders_lim(Cw, rw, np.where(rebal, 0.5, np.nan), c, 1e-9, bw, sw)
    s, lv, f = sp["s"], sp["levels"], sp["fill"]
    R["grid_static"] = grid_lim(Ow, Cw, rw, 0.03, 10, 0, f, c, 0.5, bw, sw)
    R["grid_recentre"] = grid_lim(Ow, Cw, rw, s, lv, 1, f, c, 0.5, bw, sw)
    R["grid_trail_up"] = grid_lim(Ow, Cw, rw, 0.03, 5, 2, f, c, 0.5, bw, sw)
    R["grid_wide"] = grid_lim(Ow, Cw, rw, 0.05, 10, 1, f, c, 0.5, bw, sw)
    rb = np.r_[0, R["bh"][0][1:] / R["bh"][0][:-1] - 1]
    rg = np.r_[0, R["grid_recentre"][0][1:] / R["grid_recentre"][0][:-1] - 1]
    R["core"] = S.blend_nb(rb, R["bh"][1], rg, R["grid_recentre"][1], 0.5, rebal, c)
    ret = np.r_[np.nan, C[1:] / C[:-1] - 1]
    vol20 = pd.Series(ret).rolling(20).std().values * np.sqrt(DAYS)
    sma = pd.Series(C).rolling(200).mean().values
    tgt = np.r_[np.nan, (np.minimum(1.0, 0.15 / vol20) * (C > sma))[:-1]]
    R["tvt"] = orders_lim(Cw, rw, np.nan_to_num(tgt[sl], nan=0.0), c, 0.10, bw, sw)
    out = {k: metrics(*v, rw) for k, v in R.items()}
    for k in ("grid_static", "grid_recentre", "grid_trail_up", "grid_wide", "core", "tvt"):
        m = orders_lim(Cw, rw, np.where(rebal, out[k]["expo"], np.nan), c, 1e-9, bw, sw)
        out[f"matched_{k}"] = metrics(*m, rw)
    return out


def flat(res, **tags) -> dict:
    row = dict(tags)
    for k, v in res.items():
        for m, x in v.items():
            row[f"{k}_{m}"] = x
    return row


STRATS = ["bh", "mix50", "grid_static", "grid_recentre", "grid_trail_up", "grid_wide", "core", "tvt"]
EDGED = ["grid_static", "grid_recentre", "grid_trail_up", "grid_wide", "core", "tvt"]


def main() -> None:
    data, rate = load()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, full = [], []
    for t in TICKERS:
        d = data[t]
        idx = d.index
        rf = cash_rate(rate, idx).values
        # whole history after a one-year warm-up, baseline spec
        a0 = max(WARM, idx.searchsorted(pd.Timestamp("2014-01-01")))
        if a0 < len(idx) - DAYS:
            full.append(flat(run(d, rf, a0, len(idx) - 1, BASE), ticker=t, start=idx[a0].date(), end=idx[-1].date()))
        for st in STARTS:
            a = idx.searchsorted(st)
            end = st + pd.DateOffset(years=YEARS)
            z = idx.searchsorted(end) - 1
            if a < WARM or idx[-1] < end - pd.Timedelta(days=10):
                continue
            for name, over in SPECS.items():
                rows.append(flat(run(d, rf, a, z, BASE | over), ticker=t, start=st, spec=name))
            seg = d.iloc[a - WARM: z + 1]
            rseg = rf[a - WARM: z + 1]
            for kind in ("iid", "block21"):
                for _ in range(N_NULL):
                    p = S.null_path(seg["C"].values, kind, rng)
                    dn = pd.DataFrame({"O": p, "H": p, "L": p, "C": p, "buy_ok": True, "sell_ok": True}, seg.index)
                    rows.append(flat(run(dn, rseg, WARM, len(seg) - 1, BASE), ticker=t, start=st, spec=f"null_{kind}"))
        print(f"{t} {NAMES[t]} done", flush=True)
    W = pd.DataFrame(rows); W["group"] = W["ticker"].map(GROUP_OF)
    F = pd.DataFrame(full); F["name"] = F["ticker"].map(NAMES); F["group"] = F["ticker"].map(GROUP_OF)
    W.to_csv(OUT / "windows.csv.gz", index=False, float_format="%.5g")
    F.to_csv(OUT / "full_history.csv", index=False, float_format="%.4g")
    report(W, F, data)


def report(W: pd.DataFrame, F: pd.DataFrame, data: dict) -> None:
    W["start"] = pd.to_datetime(W["start"])
    for k in EDGED:
        W[f"{k}_dsharpe"] = W[f"{k}_sharpe"] - W[f"matched_{k}_sharpe"]
        W[f"{k}_dcagr"] = W[f"{k}_cagr"] - W[f"matched_{k}_cagr"]
        W[f"{k}_dtotal"] = W[f"{k}_total"] - W[f"matched_{k}_total"]
        W[f"{k}_vs_mix50_dsharpe"] = W[f"{k}_sharpe"] - W["mix50_sharpe"]
    order = list(SPECS) + ["null_iid", "null_block21"]
    g = W.groupby("spec")
    lv = g[[f"{s}_{m}" for s in STRATS for m in ("total", "cagr", "sharpe", "maxdd")]].median().reindex(order)
    ed = g[[f"{k}_{m}" for k in EDGED for m in ("dsharpe", "dcagr", "dtotal", "vs_mix50_dsharpe")]].median().reindex(order)
    pos = g[[f"{k}_dsharpe" for k in EDGED]].apply(lambda x: (x > 0).mean() * 100).reindex(order)
    no = W[W["start"].isin(NONOVERLAP)].groupby(["spec", "start"])[[f"{k}_dsharpe" for k in EDGED]].median()
    nop = no.groupby("spec").apply(lambda x: (x > 0).sum()).reindex(order)
    base = W[W["spec"] == "baseline"]
    by_group = base.groupby("group")[[f"{s}_cagr" for s in STRATS] + [f"{k}_dsharpe" for k in EDGED]].median()
    by_win = base.groupby("start")[["bh_total", "mix50_total", "grid_recentre_total", "grid_static_total", "core_total",
                                    "grid_recentre_dsharpe", "grid_static_dsharpe", "core_dsharpe"]].median()
    splits = {t: int(d["split"].sum()) for t, d in data.items()}
    limits = {t: int((~d["buy_ok"] | ~d["sell_ok"]).sum()) for t, d in data.items()}
    for name, df in (("levels_median", lv), ("edges_median", ed), ("edges_pct_positive", pos),
                     ("edges_nonoverlap_positive", nop), ("baseline_by_group", by_group), ("baseline_by_window", by_win)):
        df.round(3).to_csv(OUT / f"{name}.csv")
    pd.DataFrame({"splits_fixed": splits, "blocked_days": limits}).to_csv(OUT / "data_fixes.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 100)
    print(f"\nETF-windows per spec: {len(base)}; splits fixed: {splits}; limit/suspension days: {limits}")
    cols = ["name", "start", "end"] + [f"{s}_{m}" for s in ("bh", "mix50", "grid_static", "grid_recentre", "core")
                                       for m in ("total", "cagr", "sharpe", "maxdd")]
    print("\nWhole history, baseline\n", F[cols].round(2).to_string(index=False))
    print("\nMedian levels\n", lv.round(2))
    print("\nMedian edges vs exposure-matched (Sharpe, CAGR pts, total pts) and Sharpe vs 50/50\n", ed.round(3))
    print("\n% of ETF-windows with positive Sharpe edge vs matched\n", pos.round(0))
    print(f"\nNon-overlapping windows (of {len(NONOVERLAP)}) with a positive median Sharpe edge\n", nop)
    print("\nBaseline by group\n", by_group.round(3))
    print("\nBaseline by window\n", by_win.round(2))


if __name__ == "__main__":
    main()
