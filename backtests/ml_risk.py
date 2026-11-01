"""Simpler ML targets for the core blend: a risk-on/off switch and volatility-based sizing.

Base portfolio ("core"): 50 % buy_hold sleeve + 50 % grid_recentre sleeve (see ml_meta.py), scaled by s ∈ [0, 1]
with the rest in T-bills. Decisions on the last close of each month, applied at the next close; costs netted.

Pre-registered variants (fixed before any result was seen):
  core              s = 1
  core_sma200       s = 1 if close > SMA(200) else 0.5                         – simple risk switch
  core_risk_logit   s = 1 if P(next-month core excess return > 0) ≥ 0.5 else 0.5  (logistic regression)
  core_risk_hgb     same, gradient-boosted classifier
  core_vol_naive    s = min(1, 15 % / trailing 21-day core volatility)          – simple volatility sizing
  core_vol_har      s = min(1, 15 % / forecast), forecast = HAR regression of log next-month volatility
                    on log 5-, 21- and 63-day volatility
  core_vol_hgb      same sizing, gradient-boosted forecast from all features
Each ML variant is judged against its simple counterpart (sma200 / vol_naive), not only against core.

Leakage rules: annual walk-forward training on samples whose label ended before the year's first decision;
samples overlapping the holdout windows are never trained on; hyperparameters fixed; holdout reported
only with --holdout, once, after the development run is committed.

    python backtests/ml_risk.py              # development windows
    python backtests/ml_risk.py --holdout    # adds Jul 2011–Jun 2012 and 2018–19
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import advanced_strategies as A
import ml_meta as M

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
UNIVERSES = {"original": A.DEFAULT_TICKERS, **{k: v for k, v in json.loads(
    (HERE / "data" / "universes.json").read_text()).items() if k != "seed"}}
DEV = {**M.DEV, **M.HOLDOUT}                     # every window used before is now development data
HOLDOUT = {"euro_2011_12": ("2011-07-01", "2012-06-29"), "vol_2018_19": ("2018-01-02", "2019-12-31")}
VOL_TARGET = 0.15
OUT = HERE / "results_ml_risk"
VARIANTS = ["core", "core_sma200", "core_risk_logit", "core_risk_hgb", "core_vol_naive", "core_vol_har", "core_vol_hgb"]
HGB = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def overlaps_holdout(a, z) -> bool:
    return any(a <= pd.Timestamp(e) and z >= pd.Timestamp(s) for s, e in HOLDOUT.values())


def ticker_panel(t, full, spy, irx, rf):
    ret, ex = M.sleeves_for(full, rf)
    sched = M.schedule(ret.index)
    core = 0.5 * ret["buy_hold"] + 0.5 * ret["grid_recentre"]
    f = M.features(full, spy, irx, ret)
    for n in (5, 21, 63):
        f[f"core_lvol_{n}"] = np.log(core.rolling(n).std() * np.sqrt(252) + 1e-4)
    f["above_sma200"] = (full > full.rolling(200).mean()).astype(float)
    p = f.loc[sched["decision"]].reset_index(names="decision")
    p.insert(0, "ticker", t)
    p["exec"] = sched["exec"].values
    nxt = list(sched["exec"].iloc[1:]) + [pd.NaT]
    p["label_end"] = nxt
    up, lvol = [], []
    for a, z in zip(sched["exec"], nxt):
        if pd.isna(z):
            up.append(np.nan); lvol.append(np.nan); continue
        seg = core[(core.index > a) & (core.index <= z)]
        rfs = rf[(rf.index > a) & (rf.index <= z)]
        up.append(float((1 + seg).prod() > (1 + rfs).prod()))
        lvol.append(np.log(seg.std() * np.sqrt(252) + 1e-4))
    p["y_up"], p["y_lvol"] = up, lvol
    return p, (ret, ex, sched)


def walk_forward(panel, feat_cols):
    har_cols = ["core_lvol_5", "core_lvol_21", "core_lvol_63"]
    out = pd.DataFrame(index=panel.index, columns=["p_logit", "p_hgb", "v_har", "v_hgb"], dtype=float)
    usable = panel["y_up"].notna() & ~panel.apply(lambda r: overlaps_holdout(r["exec"], r["label_end"]), axis=1)
    for y in sorted(panel["decision"].dt.year.unique()):
        test = panel["decision"].dt.year == y
        train = usable & (panel["label_end"] < panel.loc[test, "decision"].min())
        if train.sum() < M.MIN_TRAIN or panel.loc[train, "y_up"].nunique() < 2:
            continue
        X, Xt = panel.loc[train, feat_cols].fillna(0).values, panel.loc[test, feat_cols].fillna(0).values
        yu, yv = panel.loc[train, "y_up"].values, panel.loc[train, "y_lvol"].values
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X, yu)
        out.loc[test, "p_logit"] = logit.predict_proba(Xt)[:, 1]
        out.loc[test, "p_hgb"] = HistGradientBoostingClassifier(**HGB).fit(X, yu).predict_proba(Xt)[:, 1]
        har = LinearRegression().fit(panel.loc[train, har_cols].fillna(0).values, yv)
        out.loc[test, "v_har"] = np.exp(har.predict(panel.loc[test, har_cols].fillna(0).values))
        out.loc[test, "v_hgb"] = np.exp(HistGradientBoostingRegressor(**HGB).fit(X, yv).predict(Xt))
    return out


def scales(p: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=p.index)
    s["core"] = 1.0
    s["core_sma200"] = np.where(p["above_sma200"] == 1, 1.0, 0.5)
    s["core_risk_logit"] = np.where(p["p_logit"].isna() | (p["p_logit"] >= 0.5), 1.0, 0.5)
    s["core_risk_hgb"] = np.where(p["p_hgb"].isna() | (p["p_hgb"] >= 0.5), 1.0, 0.5)
    s["core_vol_naive"] = np.minimum(1.0, VOL_TARGET / np.exp(p["core_lvol_21"]))
    s["core_vol_har"] = np.minimum(1.0, VOL_TARGET / p["v_har"]).fillna(s["core_vol_naive"])
    s["core_vol_hgb"] = np.minimum(1.0, VOL_TARGET / p["v_hgb"]).fillna(s["core_vol_naive"])
    return s


def weights(exec_idx, scale):
    w = pd.DataFrame(0.0, index=exec_idx, columns=M.SLEEVES)
    w["buy_hold"] = 0.5 * scale.values
    w["grid_recentre"] = 0.5 * scale.values
    w["cash"] = 1 - scale.values
    return w


def diagnostics(panel, mask):
    d = panel[mask & panel["y_up"].notna() & panel["p_logit"].notna()]
    rows = {}
    for m in ("p_logit", "p_hgb"):
        rows[m] = {"auc": roc_auc_score(d["y_up"], d[m]), "accuracy": ((d[m] >= 0.5) == d["y_up"]).mean(),
                   "always_on_accuracy": d["y_up"].mean(), "share_risk_off": (d[m] < 0.5).mean(), "n": len(d)}
    r = d["above_sma200"]
    rows["sma200"] = {"auc": roc_auc_score(d["y_up"], r) if r.nunique() > 1 else np.nan,
                      "accuracy": (r == d["y_up"]).mean(), "always_on_accuracy": d["y_up"].mean(),
                      "share_risk_off": (r == 0).mean(), "n": len(d)}
    risk = pd.DataFrame(rows).T
    y = d["y_lvol"]
    vol = {}
    for name, f in (("naive_21d", d["core_lvol_21"]), ("har", np.log(d["v_har"])), ("hgb", np.log(d["v_hgb"]))):
        vol[name] = {"r2_log_vol": 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     "mae_log_vol": (y - f).abs().mean()}
    return risk, pd.DataFrame(vol).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    out = OUT / ("holdout" if args.holdout else "dev")
    out.mkdir(parents=True, exist_ok=True)

    tickers = list(dict.fromkeys(t for u in UNIVERSES.values() for t in u))
    px, rf = A.load(list(dict.fromkeys([*tickers, "SPY"])), M.DATA_START, M.DATA_END)
    irx, spy = (1 + rf) ** 252 - 1, px["SPY"].dropna()

    rows, data = [], {}
    for t in tickers:
        full = px[t].dropna()
        p, data[t] = ticker_panel(t, full, spy, irx, rf)
        rows.append(p)
    panel = pd.concat(rows, ignore_index=True)
    feat_cols = [c for c in panel.columns if c not in ("ticker", "decision", "exec", "label_end", "y_up", "y_lvol")]
    panel = panel.join(walk_forward(panel, feat_cols))
    S = scales(panel)

    windows = dict(DEV, **(HOLDOUT if args.holdout else {}))
    near = lambda a, z: (panel["decision"] >= pd.Timestamp(a) - pd.Timedelta(days=31)) & (panel["decision"] <= pd.Timestamp(z))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    groups = [("development", DEV)] + ([("holdout", HOLDOUT)] if args.holdout else [])
    for label, ws in groups:
        mask = np.logical_or.reduce([near(*w) for w in ws.values()])
        risk, vol = diagnostics(panel, mask)
        print(f"\nPrediction quality — {label} windows\n", risk.round(3), "\n", vol.round(3))
        risk.to_csv(out / f"diag_risk_{label}.csv"); vol.to_csv(out / f"diag_vol_{label}.csv")

    res = []
    for wname, (a, z) in windows.items():
        for t, (ret, ex, sched) in data.items():
            if ret.index[0] > pd.Timestamp(a):
                continue
            sub = panel.index[panel["ticker"] == t]
            exec_idx = pd.DatetimeIndex(panel.loc[sub, "exec"])
            W = {v: weights(exec_idx, S.loc[sub, v]) for v in VARIANTS}
            W["buy_hold"] = M.const_weights(sched, {"buy_hold": 1.0})
            W["rebal_50_m"] = M.const_weights(sched, {"rebal_50_m": 1.0})
            uni = next(u for u, ts in UNIVERSES.items() if t in ts)
            for name, w in W.items():
                eq = M.simulate(ret, ex, w, a, z)
                m = A.metrics(eq, pd.Series(0.0, eq.index), 0, ret["cash"])
                res.append({"window": wname, "universe": uni, "ticker": t, "strategy": name,
                            **{k: m[k] for k in ("total_%", "sharpe", "max_dd_%", "vol_%")}})
    res = pd.DataFrame(res)
    res.to_csv(out / "metrics.csv", index=False, float_format="%.4f")

    order = ["buy_hold", "rebal_50_m", *VARIANTS]
    pairs = {"core_risk_logit": "core_sma200", "core_risk_hgb": "core_sma200",
             "core_vol_har": "core_vol_naive", "core_vol_hgb": "core_vol_naive"}
    for wname in windows:
        r = res[res["window"] == wname]
        summ = r.groupby("strategy")[["total_%", "sharpe", "max_dd_%", "vol_%"]].median().loc[order]
        sh = r.pivot(index="strategy", columns="ticker", values="sharpe")
        summ["sharpe>core"] = (sh.sub(sh.loc["core"], axis=1) > 0).sum(axis=1)
        summ["sharpe>simple"] = pd.Series({k: int((sh.loc[k] > sh.loc[v]).sum()) for k, v in pairs.items()})
        a, z = windows[wname]
        tag = "HOLDOUT" if wname in HOLDOUT else "dev"
        print(f"\n=== [{tag}] {wname} {a} → {z} ({sh.shape[1]} tickers; medians)\n", summ.round(2))
        summ.to_csv(out / f"summary_{wname}.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
