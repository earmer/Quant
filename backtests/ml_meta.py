"""Near-advanced algorithms (fixed blends of base strategies) and a walk-forward ML allocator.

Every base strategy from `advanced_strategies.py` runs continuously from ~2000 on each ticker as a
"sleeve". Portfolios are built by allocating capital across sleeves (plus a T-bill sleeve):

  * single sleeves             – weight 1 on one strategy
  * eq_weight_all              – equal weight on all sleeves and cash, rebalanced monthly
  * near-advanced blends       – fixed weights chosen a priori to pair rules suited to opposite regimes
  * ml_hgb / ml_ridge          – each month, predict every sleeve's next-month return relative to the
                                 sleeve average; hold the top two at 50 % each

Allocations are decided on the last close of each month and applied at the next close. Moving capital
between sleeves of the same stock is netted: only the change in total share exposure is charged COST_BPS.

Leakage rules
  * Walk-forward: the model used for decisions in year Y is trained only on samples whose label period
    ended before the first decision of Y.
  * Samples whose label period overlaps a holdout window are never used for training.
  * Hyperparameters and the allocation rule are fixed in code; nothing is tuned.
  * Development windows (already seen in earlier tests) are reported by default. Holdout windows, never
    looked at before this script was frozen, are reported only with --holdout.

    python backtests/ml_meta.py              # development windows
    python backtests/ml_meta.py --holdout    # adds the holdout windows (run once, after freezing)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import advanced_strategies as A

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_START, DATA_END = "1999-01-01", "2026-10-31"
SLEEVES = ["buy_hold", "rebal_50_m", "grid_periodic", "grid_recentre", "grid_trail_up",
           "trend_voltarget", "voltarget_15", "tsmom_6m", "rsi2_trend", "boll_mr", "cash"]
BLENDS = {
    "ens_grid_trend": {"grid_periodic": 0.5, "trend_voltarget": 0.5},
    "ens_grid_trend_rsi2": {"grid_periodic": 1 / 3, "trend_voltarget": 1 / 3, "rsi2_trend": 1 / 3},
    "ens_core_grid": {"buy_hold": 0.5, "grid_recentre": 0.5},
    "ens_vt_grid": {"voltarget_15": 0.5, "grid_periodic": 0.5},
}
DEV = {
    "gfc_recovery": ("2007-10-09", "2010-10-08"),
    "bear_2022": ("2022-01-03", "2022-12-30"),
    "bull_2023_26": ("2023-10-31", "2026-10-30"),
}
HOLDOUT = {
    "sideways_2015_16": ("2015-01-02", "2016-12-30"),
    "covid_2020": ("2020-01-02", "2020-12-31"),
}
TOP_K = 2
MIN_TRAIN = 300
OUT = Path(__file__).resolve().parent / "results_ml"


def models():
    return {
        "ml_hgb": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                        min_samples_leaf=50, l2_regularization=1.0,
                                                        random_state=0),
        "ml_ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


# ============================================================================ sleeves
def sleeves_for(full: pd.Series, rf: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily returns and exposures of every sleeve, running from ~1 year after the first price."""
    start = full.index[260]
    res = A.build(full, rf, start)
    idx = full.index[full.index >= start]
    ret = pd.DataFrame({k: res[k][0].pct_change().fillna(0.0) for k in SLEEVES if k != "cash"}, index=idx)
    ex = pd.DataFrame({k: res[k][1] for k in SLEEVES if k != "cash"}, index=idx)
    ret["cash"], ex["cash"] = rf.loc[idx], 0.0
    ret.iloc[0] = 0.0
    return ret[SLEEVES], ex[SLEEVES]


def schedule(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Decision = last session of a month; execution = next session."""
    dec = idx[~idx.to_period("M").duplicated(keep="last")][:-1]
    pos = idx.get_indexer(dec) + 1
    return pd.DataFrame({"decision": dec, "exec": idx[pos]})


# ============================================================================ features & labels
def features(full: pd.Series, spy: pd.Series, irx: pd.Series, ret: pd.DataFrame) -> pd.DataFrame:
    r = full.pct_change()
    f = pd.DataFrame(index=full.index)
    for n in (21, 63, 126, 252):
        f[f"ret_{n}"] = full.pct_change(n)
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = r.rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    f["dist_sma50"] = full / full.rolling(50).mean() - 1
    f["dist_sma200"] = full / full.rolling(200).mean() - 1
    f["dd_252"] = full / full.rolling(252, min_periods=1).max() - 1
    f["rsi_14"] = A.rsi(full, 14)
    spy = spy.reindex(full.index).ffill()
    f["spy_ret_63"] = spy.pct_change(63)
    f["spy_dist_sma200"] = spy / spy.rolling(200).mean() - 1
    f["spy_vol_21"] = spy.pct_change().rolling(21).std() * np.sqrt(252)
    f["tbill"] = irx.reindex(full.index).ffill()
    trail = np.expm1(np.log1p(ret).rolling(63).sum())
    trail = trail.sub(trail.mean(axis=1), axis=0)
    for k in SLEEVES:
        f[f"trail63_{k}"] = trail[k]
    return f


def period_returns(ret: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Sleeve returns from exec_j (exclusive) to exec_{j+1} (inclusive), indexed by decision_j."""
    growth = (1 + ret).cumprod()
    g = growth.loc[sched["exec"]].values
    out = g[1:] / g[:-1] - 1
    return pd.DataFrame(out, index=sched["decision"].iloc[:-1], columns=SLEEVES)


# ============================================================================ allocation engine
def simulate(ret: pd.DataFrame, ex: pd.DataFrame, weights: pd.DataFrame, a, b) -> pd.Series:
    """Equity of a sleeve allocation over [a, b]. weights: index = exec days, columns = SLEEVES."""
    days = ret.index[(ret.index >= a) & (ret.index <= b)]
    w_all = weights.reindex(weights.index.union(days)).ffill().loc[days]
    R, E, W = ret.loc[days].values, ex.loc[days].values, w_all.values
    rebal = days.isin(weights.index)
    v = W[0] * CAPITAL_AFTER_ENTRY(W[0], E[0])
    eq = np.empty(len(days))
    eq[0] = v.sum()
    for i in range(1, len(days)):
        v = v * (1 + R[i])
        if rebal[i]:
            V = v.sum()
            cost = A.C * abs(((W[i] - v / V) * E[i]).sum()) * V
            v = W[i] * (V - cost)
        eq[i] = v.sum()
    return pd.Series(eq, days)


def CAPITAL_AFTER_ENTRY(w, e):
    return A.CAPITAL * (1 - A.C * float((w * e).sum()))


def const_weights(sched: pd.DataFrame, mix: dict) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=sched["exec"], columns=SLEEVES)
    for k, v in mix.items():
        w[k] = v
    return w


# ============================================================================ walk-forward ML
def holdout_overlap(start, end) -> bool:
    return any(start <= pd.Timestamp(z) and end >= pd.Timestamp(a) for a, z in HOLDOUT.values())


def walk_forward(panel: pd.DataFrame, feat_cols: list[str]) -> dict[str, pd.DataFrame]:
    """panel rows: ticker × decision with features, label_* (relative returns), label_start/end.
    Returns {model: predictions DataFrame indexed like panel}."""
    preds = {m: pd.DataFrame(np.nan, index=panel.index, columns=SLEEVES) for m in models()}
    years = sorted(panel["decision"].dt.year.unique())
    usable = ~panel.apply(lambda r: holdout_overlap(r["label_start"], r["label_end"]), axis=1)
    usable &= panel[[f"label_{k}" for k in SLEEVES]].notna().all(axis=1)
    for y in years:
        test = panel["decision"].dt.year == y
        first_dec = panel.loc[test, "decision"].min()
        train = usable & (panel["label_end"] < first_dec)
        if train.sum() < MIN_TRAIN:
            continue
        Xtr = panel.loc[train, feat_cols].fillna(0.0).values
        Xte = panel.loc[test, feat_cols].fillna(0.0).values
        for m, make in models().items():
            for k in SLEEVES:
                mdl = make().fit(Xtr, panel.loc[train, f"label_{k}"].values)
                preds[m].loc[test, k] = mdl.predict(Xte)
    return preds


def top_k_weights(pred: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=pred.index, columns=SLEEVES)
    ok = pred.notna().all(axis=1)
    for i in pred.index[ok]:
        w.loc[i, pred.loc[i].nlargest(TOP_K).index] = 1.0 / TOP_K
    w.loc[~ok] = 1.0 / len(SLEEVES)     # no model yet: equal weight
    return w


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=A.DEFAULT_TICKERS)
    ap.add_argument("--holdout", action="store_true", help="also report the holdout windows")
    ap.add_argument("--out", default=str(OUT), help="output root; dev/ or holdout/ is appended")
    args = ap.parse_args()
    out = Path(args.out) / ("holdout" if args.holdout else "dev")
    out.mkdir(parents=True, exist_ok=True)

    # SPY is always loaded for the market features, but only tested if requested
    px, rf = A.load(list(dict.fromkeys([*args.tickers, "SPY"])), DATA_START, DATA_END)
    irx = ((1 + rf) ** 252 - 1)
    spy = px["SPY"].dropna()
    px = px[[t for t in args.tickers if t in px.columns]]

    data, rows = {}, []
    for t in px.columns:
        full = px[t].dropna()
        if len(full) < 300:
            print(f"[skip] {t}: only {len(full)} sessions of history")
            continue
        ret, ex = sleeves_for(full, rf)
        sched = schedule(ret.index)
        f = features(full, spy, irx, ret)
        lab = period_returns(ret, sched)
        rel = lab.sub(lab.mean(axis=1), axis=0)
        p = f.loc[sched["decision"]].reset_index(names="decision")
        p.insert(0, "ticker", t)
        p["exec"] = sched["exec"].values
        p["label_start"] = sched["exec"].values
        p["label_end"] = list(sched["exec"].iloc[1:]) + [pd.NaT]
        for k in SLEEVES:
            p[f"label_{k}"] = rel[k].reindex(sched["decision"]).values
            p[f"raw_{k}"] = lab[k].reindex(sched["decision"]).values
        rows.append(p)
        data[t] = (ret, ex, sched)
        print(f"{t}: sleeves {ret.index[0].date()} → {ret.index[-1].date()}, {len(p)} monthly decisions")
    panel = pd.concat(rows, ignore_index=True)
    feat_cols = [c for c in panel.columns if c not in ("ticker", "decision", "exec", "label_start", "label_end")
                 and not c.startswith(("label_", "raw_"))]
    preds = walk_forward(panel, feat_cols)

    # --- prediction quality: cross-sleeve rank IC per decision, by period type
    def ic_table(period_filter):
        out_ = {}
        for m, pr in preds.items():
            ics = []
            for i in panel.index[period_filter & pr.notna().all(axis=1) & panel["label_end"].notna()]:
                ics.append(pd.Series(pr.loc[i].values).corr(
                    pd.Series(panel.loc[i, [f"label_{k}" for k in SLEEVES]].values.astype(float)),
                    method="spearman"))
            s = pd.Series(ics)
            out_[m] = {"mean_rank_ic": s.mean(), "t_stat": s.mean() / s.std() * np.sqrt(len(s)), "n": len(s)}
        return pd.DataFrame(out_).T

    windows = dict(DEV, **(HOLDOUT if args.holdout else {}))
    in_window = lambda a, z: (panel["decision"] >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (panel["decision"] <= pd.Timestamp(z))
    dev_mask = np.logical_or.reduce([in_window(*w) for w in DEV.values()])
    print("\nRank IC of predicted vs realised relative sleeve returns — development windows\n",
          ic_table(dev_mask).round(3))
    if args.holdout:
        ho_mask = np.logical_or.reduce([in_window(*w) for w in HOLDOUT.values()])
        print("\nRank IC — holdout windows\n", ic_table(ho_mask).round(3))
        oos = panel["decision"].dt.year >= 2007
        print("\nRank IC — all walk-forward months 2007–2026\n", ic_table(oos).round(3))

    # --- portfolios per window
    strategies = [k for k in SLEEVES if k != "cash"] + ["eq_weight_all", *BLENDS, *preds]
    allres, alloc_rows = [], []
    for wname, (a, z) in windows.items():
        for t, (ret, ex, sched) in data.items():
            if ret.index[0] > pd.Timestamp(a):
                continue
            sub = panel[panel["ticker"] == t].set_index("exec")
            W = {k: const_weights(sched, {k: 1.0}) for k in SLEEVES if k != "cash"}
            W["eq_weight_all"] = const_weights(sched, {k: 1 / len(SLEEVES) for k in SLEEVES})
            W.update({b: const_weights(sched, mix) for b, mix in BLENDS.items()})
            for m, pr in preds.items():
                wm = top_k_weights(pr.loc[panel["ticker"] == t].set_index(sub.index))
                W[m] = wm
                inwin = wm[(wm.index >= pd.Timestamp(a)) & (wm.index <= pd.Timestamp(z))]
                alloc_rows.append({"window": wname, "ticker": t, "model": m, **inwin.mean().to_dict()})
            rfw = ret["cash"]
            for name in strategies:
                eq = simulate(ret, ex, W[name], a, z)
                m = A.metrics(eq, pd.Series(0.0, eq.index), 0, rfw)
                allres.append({"window": wname, "ticker": t, "strategy": name,
                               **{k: m[k] for k in ("total_%", "cagr_%", "sharpe", "max_dd_%")}})
    res = pd.DataFrame(allres)
    res.to_csv(out / "metrics.csv", index=False, float_format="%.4f")
    alloc = pd.DataFrame(alloc_rows)
    alloc.to_csv(out / "ml_allocations.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    for wname in windows:
        r = res[res["window"] == wname]
        tot = r.pivot(index="strategy", columns="ticker", values="total_%").loc[strategies]
        sh = r.pivot(index="strategy", columns="ticker", values="sharpe").loc[strategies]
        summ = r.groupby("strategy")[["total_%", "cagr_%", "sharpe", "max_dd_%"]].median().loc[strategies]
        summ["beats_rebal50"] = (tot.sub(tot.loc["rebal_50_m"], axis=1) > 0).sum(axis=1)
        summ["beats_eqw"] = (tot.sub(tot.loc["eq_weight_all"], axis=1) > 0).sum(axis=1)
        summ["sharpe_beats_eqw"] = (sh.sub(sh.loc["eq_weight_all"], axis=1) > 0).sum(axis=1)
        a, z = windows[wname]
        print(f"\n=== {wname}  {a} → {z}  ({tot.shape[1]} tickers; medians)\n", summ.round(2))
        summ.to_csv(out / f"summary_{wname}.csv", float_format="%.4f")
        al = alloc[alloc["window"] == wname].groupby("model")[SLEEVES].mean()
        print("Average ML allocation in this window (across tickers)\n", al.round(2))


if __name__ == "__main__":
    main()
