"""Delayed relationships: do moves on day N predict moves at N+3 or N+7 (trading days)?

The next-day (N -> N+1) search in discover.py found no persistent links. This script asks the same
models, with no pairs specified, to look further ahead. Two readings of "N+3 / N+7" are tested:

  d3, d7   the residual move ON day N+3 (or N+7) alone: a delayed reaction
  c3, c7   the residual move summed over N+1..N+3 (or N+1..N+7): a slow drift

Everything else is as in discover.py: ~480 S&P 500 stocks plus 8 ETFs; five PCA factors removed;
inputs are every stock's, factor's and ETF's residual on day N and summed over N-4..N-1 (~990);
per-stock Lasso with leave-one-training-year-out CV; a shuffled-timing null for each horizon; and the
pooled gradient-boosted model with automatically found peers and clusters.

Years: learn on 2020, 2022, 2024; validate on 2021, 2023, 2025 (already looked at once for the
next-day test, so not pristine for this new question), plus 2026 to date as a FRESH check that no
model in this project has been evaluated on.

Trading: single-day targets -> each day, using the signal from N, hold the top/bottom tenth on day N+h
only (rebalanced daily); cumulative targets -> hold the top/bottom tenth for h days. 5 bp per side.

    python backtests/discover_lags.py --phase train      # results_discover_lags/train/, then commit
    python backtests/discover_lags.py --phase validate   # once
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import lasso_path

import discover as D

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_discover_lags"
CACHE = HERE / ".cache"
PRICES = CACHE / "discover_prices_to_2026-11-03.pkl"
HORIZONS = {"d3": ("point", 3), "d7": ("point", 7), "c3": ("cum", 3), "c7": ("cum", 7)}
MAXH = 7
FRESH = [2026]
SEED = 20261105


def load_ext() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same universe as discover.py (decided on 2020-2025 completeness), prices extended to 2026-11-03."""
    _, meta = D.load()
    cols = list(meta.index) + D.ETFS
    if not PRICES.exists():
        got = {}
        for i in range(0, len(cols), 50):
            b = cols[i:i + 50]
            c = yf.download(b, start="2019-06-01", end="2026-11-04", auto_adjust=True, progress=False,
                            threads=False, group_by="column")["Close"]
            got.update({t: c[t] for t in b if t in c})
        PRICES.write_bytes(pickle.dumps(pd.DataFrame(got)))
    px = pickle.loads(PRICES.read_bytes())
    return px[[c for c in cols if c in px]], meta


def design_h(prep: D.Prep, years: list[int], rng: np.random.Generator | None = None):
    """X at day N; Y[h] per horizon; fwd raw returns: 'r1' next day and 'r{h}' on day N+h."""
    Xs, Ys, F, grp = [], {h: [] for h in HORIZONS}, {k: [] for k in ("r1", "r3", "r7")}, []
    for y in years:
        z, inp = prep.year(y)
        if rng is not None:
            inp = inp.copy()
            for c in inp.columns:
                inp[c] = np.roll(inp[c].values, rng.integers(20, len(inp) - 20))
        X = pd.concat([inp.add_suffix("|1"), inp.shift(1).rolling(4).sum().add_suffix("|2-5")], axis=1)
        # targets may reach into the next year's first days (they are outcomes, not inputs); the
        # residual for those days uses the same frozen factors, so build z with a look-ahead span
        z_ext = pd.concat([z, prep.year(y + 1)[0]]) if (prep.r.index.year == y + 1).any() else z
        z_ext = z_ext[~z_ext.index.duplicated()]
        r_ext = prep.r.loc[z_ext.index, prep.stocks]
        Yh = {}
        for h, (kind, n) in HORIZONS.items():
            Yh[h] = (z_ext.shift(-n) if kind == "point"
                     else sum(z_ext.shift(-i) for i in range(1, n + 1)) / np.sqrt(n)).reindex(z.index)
        fw = {"r1": r_ext.shift(-1).reindex(z.index), "r3": r_ext.shift(-3).reindex(z.index),
              "r7": r_ext.shift(-7).reindex(z.index)}
        keep = (z.index.year == y)
        keep &= np.arange(len(z)) >= (np.argmax(keep) + D.SKIP)
        # drop the last MAXH sessions of each year: their outcomes would overlap the neighbouring year
        keep &= np.arange(len(z)) <= (len(z) - 1 - MAXH)
        Xs.append(X[keep]); grp.append(pd.Series(y, z.index[keep]))
        for h in HORIZONS:
            Ys[h].append(Yh[h][keep])
        for k in F:
            F[k].append(fw[k][keep])
    return (pd.concat(Xs).fillna(0.0), {h: pd.concat(v) for h, v in Ys.items()}, pd.concat(grp),
            {k: pd.concat(v) for k, v in F.items()})


# ============================================================================ parallel Lasso
def _fit_chunk(Xs, sd, mu, Y, groups, cols):
    folds = sorted(set(groups))
    pre = {f: Xs[groups != f].T @ Xs[groups != f] for f in folds}
    G = Xs.T @ Xs
    out = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        ok = ~np.isnan(y)
        if ok.sum() < 200:
            out.append((np.zeros(Xs.shape[1]), 0.0, np.nan)); continue
        yc = np.where(ok, y - np.nanmean(y), 0.0)
        a_max = np.abs(Xs[ok].T @ yc[ok]).max() / ok.sum()
        alphas = np.geomspace(a_max, a_max * D.EPS, D.N_ALPHAS)
        sse = np.zeros(D.N_ALPHAS); sst = 0.0
        for f in folds:
            tr = (groups != f) & ok; te = (groups == f) & ok
            Xtr = Xs[tr]; ytr = yc[tr] - yc[tr].mean()
            Gf = pre[f] if ok.all() else Xtr.T @ Xtr
            _, cp, _ = lasso_path(Xtr, ytr, alphas=alphas, precompute=Gf, Xy=Xtr.T @ ytr, max_iter=2000, tol=1e-4)
            pred = Xs[te] @ cp + yc[tr].mean()
            sse += ((yc[te][:, None] - pred) ** 2).sum(0)
            sst += ((yc[te] - yc[tr].mean()) ** 2).sum()
        k = int(np.argmin(sse))
        Xo = Xs[ok]; yo = yc[ok] - yc[ok].mean()
        _, cp, _ = lasso_path(Xo, yo, alphas=alphas[: k + 1], precompute=G if ok.all() else Xo.T @ Xo,
                              Xy=Xo.T @ yo, max_iter=2000, tol=1e-4)
        b = cp[:, -1] / sd
        out.append((b, np.nanmean(y) - mu @ b, 1 - sse[k] / sst))
    return out


def fit_lasso_par(X: pd.DataFrame, Y: pd.DataFrame, groups: pd.Series, n_jobs: int = 4):
    Xv = X.values
    mu = Xv.mean(0); sd = Xv.std(0); sd[sd == 0] = 1
    Xs = (Xv - mu) / sd
    g = groups.values
    chunks = np.array_split(np.arange(Y.shape[1]), n_jobs)
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_chunk)(Xs, sd, mu, Y.values[:, c], g, None) for c in chunks)
    flat = [r for part in res for r in part]
    C = pd.DataFrame(np.vstack([r[0] for r in flat]), index=Y.columns, columns=X.columns)
    C["_intercept"] = [r[1] for r in flat]
    return C, pd.Series([r[2] for r in flat], Y.columns)


# ============================================================================ phases
def run_train() -> None:
    out = OUT / "train"; out.mkdir(parents=True, exist_ok=True)
    px, meta = load_ext()
    stocks = list(meta.index)
    prep = D.Prep(px, stocks, D.TRAIN)
    X, Y, grp, _ = design_h(prep, D.TRAIN)
    Xn, Yn, grpn, _ = design_h(prep, D.TRAIN, np.random.default_rng(SEED))
    Z = pd.concat([prep.year(y)[0].loc[lambda d: d.index.year == y] for y in D.TRAIN])
    pos, neg, cl = D.peer_maps(Z)
    summ = {}
    for h in HORIZONS:
        for tag, XX, YY, gg in (("real", X, Y, grp), ("null", Xn, Yn, grpn)):
            C, r2 = fit_lasso_par(XX, YY[h], gg)
            C.to_pickle(CACHE / f"lags_lasso_{tag}_{h}.pkl")
            E = D.edges_of(C)
            E.to_csv(out / f"edges_{tag}_{h}.csv", index=False, float_format="%.4g")
            summ[f"{tag}_{h}"] = dict(cross_edges=int((~E["own"]).sum()), own_edges=int(E["own"].sum()),
                                      median_cv_r2_pct=float(r2.median() * 100), mean_cv_r2_pct=float(r2.mean() * 100))
            print(tag, h, summ[f"{tag}_{h}"], flush=True)
        Dp = D.gbm_panel(X, Y[h], stocks, pos, neg, cl)
        Dp = Dp[Dp["y"].notna()]
        feats = [c for c in Dp.columns if c not in ("y", "day", "stock")]
        gbm = HistGradientBoostingRegressor(**D.HGB).fit(Dp[feats], Dp["y"])
        pickle.dump((gbm, feats), open(CACHE / f"lags_gbm_{h}.pkl", "wb"))
    pickle.dump((pos, neg, cl), open(CACHE / "lags_peers.pkl", "wb"))
    pd.DataFrame(summ).T.to_csv(out / "summary.csv", float_format="%.5g")


def evaluate(pred: pd.DataFrame, y: pd.DataFrame, fwd: dict, h: str) -> dict:
    kind, n = HORIZONS[h]
    ic = D.daily_ic(pred, y)
    d = {"ic_mean": ic.mean(), "ic_nw_t": D.nw_t(ic, lag=max(10, 2 * n)), "oos_r2_%": D.oos_r2(pred, y) * 100}
    book = D.decile_book(pred, fwd[f"r{n}"], 1) if kind == "point" else D.decile_book(pred, fwd["r1"], n)
    d.update(D.book_stats(book))
    return d


def run_validate() -> None:
    out = OUT / "validate"; out.mkdir(parents=True, exist_ok=True)
    px, meta = load_ext()
    stocks = list(meta.index)
    prep = D.Prep(px, stocks, D.TRAIN)
    pos, neg, cl = pickle.load(open(CACHE / "lags_peers.pkl", "rb"))
    rows = {}
    for label, years in (("valid_2021_23_25", D.VALID), ("2021", [2021]), ("2023", [2023]), ("2025", [2025]),
                         ("fresh_2026", FRESH)):
        X, Y, _, fwd = design_h(prep, years)
        for h in HORIZONS:
            for tag in ("real", "null"):
                C = pd.read_pickle(CACHE / f"lags_lasso_{tag}_{h}.pkl")
                rows[(label, h, f"lasso_{tag}")] = evaluate(D.predict(C, X), Y[h], fwd, h)
            gbm, feats = pickle.load(open(CACHE / f"lags_gbm_{h}.pkl", "rb"))
            Dp = D.gbm_panel(X, Y[h], stocks, pos, neg, cl)
            PG = pd.DataFrame(gbm.predict(Dp[feats]).reshape(len(X), len(stocks)), X.index, stocks)
            rows[(label, h, "gbm")] = evaluate(PG, Y[h], fwd, h)
        print(label, "done", flush=True)
    T = pd.DataFrame(rows).T
    T.index.names = ["years", "horizon", "model"]
    T.round(4).to_csv(out / "forecast_and_trading.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print(T.round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "validate"], required=True)
    run_train() if ap.parse_args().phase == "train" else run_validate()


if __name__ == "__main__":
    main()
