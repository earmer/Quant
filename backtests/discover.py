"""Let models find relationships between stocks by themselves, then check that they persist.

Nothing about which stocks are linked is written in by hand. The models see ~480 S&P 500 stocks
(today's list, so survivorship bias applies) plus eight ETFs for commodities, the dollar and bonds
(USO, UNG, CPER, GLD, CORN, UUP, IEF, SPY), and must find the links themselves.

Learn on 2020, 2022, 2024; validate once on 2021, 2023, 2025 (the split requested).

Preparation
  * Daily returns are scaled by each stock's training volatility, and the five largest common
    factors (principal components, the first is essentially the market) are removed. The
    factors are found on the training years only; validation days are projected onto them.
    What is left (the residual) is what the stock did that the broad market and the main style
    swings do not explain. Residuals are clipped at 4 sigma.
  * The first five sessions of each year are dropped from samples, so no lagged input reaches
    across into a neighbouring (differently assigned) year.

Models
  1. STRUCTURE  Partial correlations of same-day residuals (Ledoit-Wolf shrinkage): a direct link
                between two stocks after accounting for all others. Top 1,000 links reported and
                labelled automatically (same sub-industry / same sector / cross-sector).
  2. LEAD-LAG   For every stock, a Lasso regression of its next-day residual (and, separately, its
                next-5-day residual) on ~1,000 lagged inputs: every stock's, factor's and ETF's
                residual yesterday and summed over the four days before. Lasso keeps only the
                inputs that help; the penalty is chosen by leave-one-training-year-out
                cross-validation. Non-zero coefficients form the discovered directed network.
  3. POOLED GBM One gradient-boosted model over all stocks, with automatically built inputs: own
                lags, yesterday's moves of the stock's 10 most- and 10 least-correlated peers and of
                its correlation cluster, factor and ETF moves.
Controls
  * Null: the lead-lag search re-run with every input series circularly shifted by a random offset
    within each year, which keeps each series intact but breaks any timing link. The number of
    "relationships" it still finds is the chance level.
  * Strict walk-forward (validate phase): for each validation year, refit the lead-lag model on
    the training years before it only (2020 -> 2021; 2020+2022 -> 2023; all three -> 2025).
Trading the forecasts ("deals")
  Each day, buy the top tenth of stocks by forecast and sell the bottom tenth, equal weights,
  rebalanced daily (next-day model) or every five days (5-day model); 5 bp per side on turnover.

    python backtests/discover.py --phase train      # results_discover/train/, then commit
    python backtests/discover.py --phase validate   # once
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import lasso_path
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_discover"
CACHE = HERE / ".cache" / "discover_prices.pkl"
CONS = HERE / "data" / "sp500_constituents_2026-11-01.csv"
ETFS = ["SPY", "USO", "UNG", "CPER", "GLD", "CORN", "UUP", "IEF"]
TRAIN, VALID = [2020, 2022, 2024], [2021, 2023, 2025]
K_FACTORS, CLIP, SKIP = 5, 4.0, 5
N_ALPHAS, EPS = 25, 0.02
TOP_EDGES, N_PEERS, N_CLUSTERS = 1000, 10, 40
COST = 5.0 / 1e4
SEED = 20261104
COMMON = set(ETFS) | {f"F{k + 1}" for k in range(K_FACTORS)}
HGB = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=500, l2_regularization=1.0,
           random_state=0)


# ============================================================================ data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cons = pd.read_csv(CONS)
    cons["ticker"] = cons["Symbol"].str.replace(".", "-", regex=False)
    if not CACHE.exists():
        cols, allt = {}, list(cons["ticker"]) + ETFS
        for i in range(0, len(allt), 50):
            b = allt[i:i + 50]
            c = yf.download(b, start="2019-06-01", end="2026-01-01", auto_adjust=True, progress=False,
                            threads=False, group_by="column")["Close"]
            cols.update({t: c[t] for t in b if t in c and c[t].notna().sum() > 0})
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(pickle.dumps(pd.DataFrame(cols)))
    px = pickle.loads(CACHE.read_bytes())
    px = px[(px.index.year >= 2019)]
    span = px[(px.index.year >= 2020) & (px.index.year <= 2025)]
    stocks = [t for t in cons["ticker"] if t in px and span[t].isna().mean() <= 0.02]
    return px[stocks + ETFS], cons.set_index("ticker").loc[stocks]


class Prep:
    """Volatility scaling, PCA factors and residual scaling, all fitted on the given years."""

    def __init__(self, px: pd.DataFrame, stocks: list[str], fit_years: list[int]):
        r = px.pct_change()
        self.r, self.stocks = r, stocks
        R = r.loc[r.index.year.isin(fit_years), stocks]
        self.sig = R.std()
        X = (R / self.sig).fillna(0.0)
        X = X - X.mean()
        _, _, vt = np.linalg.svd(X.values, full_matrices=False)
        self.L = vt[:K_FACTORS].T                                   # N x K loadings
        self.P = np.linalg.solve(self.L.T @ self.L, self.L.T)       # cross-sectional projection
        u, f = self._resid(R)
        self.usig = u.std()
        self.esig = r.loc[r.index.year.isin(fit_years), ETFS].std()
        self.fsig = f.std()

    def _resid(self, R):
        X = (R[self.stocks] / self.sig).fillna(0.0)
        f = X.values @ self.P.T
        u = X.values - f @ self.L.T
        return pd.DataFrame(u, R.index, self.stocks), pd.DataFrame(f, R.index, [f"F{k + 1}" for k in range(K_FACTORS)])

    def year(self, y: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Residual z of stocks and the input panel (stocks + factors + ETFs) for year y.
        Includes the last 4 sessions of the previous year so lags are defined; callers drop
        the first SKIP sessions of y from samples."""
        idx = self.r.index
        pos = np.where(idx.year == y)[0]
        span = idx[max(pos[0] - 4, 1): pos[-1] + 1]
        R = self.r.loc[span]
        u, f = self._resid(R)
        z = (u / self.usig).clip(-CLIP, CLIP)
        fz = (f / self.fsig).clip(-CLIP, CLIP)
        ez = (R[ETFS] / self.esig).clip(-CLIP, CLIP).fillna(0.0)
        inputs = pd.concat([z, fz, ez], axis=1)
        return z, inputs


def design(prep: Prep, years: list[int], shift_rng: np.random.Generator | None = None):
    """X: lagged inputs at t (yesterday's value and the 4-day sum before it); Y1: z at t+1;
    Y5: sum of z over t+1..t+5 / sqrt(5). Samples exclude the first SKIP sessions of each year."""
    Xs, Y1s, Y5s, days, raw = [], [], [], [], []
    for y in years:
        z, inp = prep.year(y)
        if shift_rng is not None:                                   # null: break timing links
            inp = inp.copy()
            for c in inp.columns:
                inp[c] = np.roll(inp[c].values, shift_rng.integers(20, len(inp) - 20))
        lag1 = inp
        lag25 = inp.shift(1).rolling(4).sum()
        X = pd.concat([lag1.add_suffix("|1"), lag25.add_suffix("|2-5")], axis=1)
        Y1 = z.shift(-1)
        Y5 = sum(z.shift(-i) for i in range(1, 6)) / np.sqrt(5)
        fwd = prep.r.loc[z.index, prep.stocks].shift(-1)            # raw next-day return, for trading
        keep = (z.index.year == y)
        keep &= np.arange(len(z)) >= (np.argmax(keep) + SKIP)
        keep &= Y5.notna().any(axis=1).values
        Xs.append(X[keep]); Y1s.append(Y1[keep]); Y5s.append(Y5[keep]); raw.append(fwd[keep])
        days.append(pd.Series(y, z.index[keep]))
    return (pd.concat(Xs).fillna(0.0), pd.concat(Y1s), pd.concat(Y5s), pd.concat(days), pd.concat(raw))


# ============================================================================ lead-lag Lasso
def fit_lasso(X: pd.DataFrame, Y: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Per-target Lasso; penalty by leave-one-year-out CV; Gram matrices shared across targets."""
    Xv = X.values
    mu = Xv.mean(0); sd = Xv.std(0); sd[sd == 0] = 1
    Xs = (Xv - mu) / sd
    folds = sorted(groups.unique())
    pre = {}
    for f in folds:
        tr = (groups != f).values
        pre[f] = (Xs[tr], Xs[tr].T @ Xs[tr])
    G_all = Xs.T @ Xs
    coefs = np.zeros((Y.shape[1], Xv.shape[1]))
    cv_r2 = np.full(Y.shape[1], np.nan)
    icpt = np.zeros(Y.shape[1])
    for j, col in enumerate(Y.columns):
        y = Y[col].values
        ok = ~np.isnan(y)
        if ok.sum() < 200:
            continue
        yc = np.where(ok, y - np.nanmean(y), 0.0)
        a_max = np.abs(Xs[ok].T @ yc[ok]).max() / ok.sum()
        alphas = np.geomspace(a_max, a_max * EPS, N_ALPHAS)
        sse = np.zeros(N_ALPHAS); sst = 0.0
        for f in folds:
            tr = (groups != f).values & ok
            te = (groups == f).values & ok
            Xtr = Xs[tr]
            ytr = yc[tr] - yc[tr].mean()
            G = pre[f][1] if ok.all() else Xtr.T @ Xtr
            _, cp, _ = lasso_path(Xtr, ytr, alphas=alphas, precompute=G, Xy=Xtr.T @ ytr, max_iter=2000, tol=1e-4)
            pred = Xs[te] @ cp + yc[tr].mean()
            sse += ((yc[te][:, None] - pred) ** 2).sum(0)
            sst += ((yc[te] - yc[tr].mean()) ** 2).sum()
        k = int(np.argmin(sse))
        cv_r2[j] = 1 - sse[k] / sst
        Xo = Xs[ok]; yo = yc[ok] - yc[ok].mean()
        _, cp, _ = lasso_path(Xo, yo, alphas=alphas[: k + 1], precompute=G_all if ok.all() else Xo.T @ Xo,
                              Xy=Xo.T @ yo, max_iter=2000, tol=1e-4)
        coefs[j] = cp[:, -1] / sd
        icpt[j] = np.nanmean(y) - mu @ coefs[j]
    C = pd.DataFrame(coefs, index=Y.columns, columns=X.columns)
    C["_intercept"] = icpt
    return C, pd.Series(cv_r2, Y.columns)


def edges_of(C: pd.DataFrame) -> pd.DataFrame:
    s = C.drop(columns="_intercept").stack()
    s = s[s != 0]
    e = s.rename("coef").reset_index()
    e.columns = ["target", "input", "coef"]
    e["leader"] = e["input"].str.split("|").str[0]
    e["lag"] = e["input"].str.split("|").str[1]
    e["own"] = e["leader"] == e["target"]
    return e


def predict(C: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in C.columns if c != "_intercept"]
    return pd.DataFrame(X[cols].values @ C[cols].values.T + C["_intercept"].values, X.index, C.index)


# ============================================================================ structure
def partial_corr(Z: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(Z.fillna(0.0).values)
    P = lw.precision_
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return pd.DataFrame(pc, Z.columns, Z.columns)


def top_links(pc: pd.DataFrame, meta: pd.DataFrame, k: int = TOP_EDGES) -> pd.DataFrame:
    s = pc.where(np.triu(np.ones(pc.shape, bool), 1)).stack()
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:k]
    t = s.rename("partial_corr").reset_index()
    t.columns = ["a", "b", "partial_corr"]
    sa, sb = meta.loc[t["a"]], meta.loc[t["b"]]
    t["label"] = np.where(sa["GICS Sub-Industry"].values == sb["GICS Sub-Industry"].values, "same sub-industry",
                          np.where(sa["GICS Sector"].values == sb["GICS Sector"].values, "same sector", "cross-sector"))
    t["sub_a"], t["sub_b"] = sa["GICS Sub-Industry"].values, sb["GICS Sub-Industry"].values
    return t


# ============================================================================ pooled GBM
def peer_maps(Z: pd.DataFrame) -> tuple[dict, dict, pd.Series]:
    c = Z.corr().fillna(0.0)
    np.fill_diagonal(c.values, np.nan)
    pos = {s: c[s].nlargest(N_PEERS).index.tolist() for s in c}
    neg = {s: c[s].nsmallest(N_PEERS).index.tolist() for s in c}
    dist = np.clip(1 - c.fillna(1.0).values, 0, 2)
    np.fill_diagonal(dist, 0.0)
    cl = fcluster(linkage(squareform(dist, checks=False), "average"), N_CLUSTERS, "maxclust")
    return pos, neg, pd.Series(cl, c.index)


def gbm_panel(X: pd.DataFrame, Y: pd.DataFrame, stocks, pos, neg, cl) -> pd.DataFrame:
    l1 = X[[f"{s}|1" for s in stocks]].values
    l25 = X[[f"{s}|2-5" for s in stocks]].values
    col = {s: i for i, s in enumerate(stocks)}
    peer_p = np.column_stack([l1[:, [col[p] for p in pos[s]]].mean(1) for s in stocks])
    peer_n = np.column_stack([l1[:, [col[p] for p in neg[s]]].mean(1) for s in stocks])
    clus = pd.DataFrame(l1, columns=stocks).T.groupby(cl.reindex(stocks).values).transform("mean").T.values
    common_cols = [c for c in X.columns if c.split("|")[0] in COMMON]
    common = X[common_cols].values
    n, m = l1.shape
    D = pd.DataFrame({
        "own1": l1.ravel(), "own25": l25.ravel(), "peer_pos": peer_p.ravel(), "peer_neg": peer_n.ravel(),
        "cluster": clus.ravel(),
    })
    common_rep = np.repeat(common, m, axis=0)
    for i, c in enumerate(common_cols):
        D[c] = common_rep[:, i]
    D["y"] = Y[stocks].values.ravel()
    D["day"] = np.repeat(np.arange(n), m)
    D["stock"] = np.tile(np.arange(m), n)
    return D


# ============================================================================ evaluation
def nw_t(x, lag=10) -> float:
    x = pd.Series(x).dropna().values
    n = len(x)
    u = x - x.mean()
    v = u @ u / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def daily_ic(pred: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    return pred.rank(axis=1).corrwith(y.rank(axis=1), axis=1)


def oos_r2(pred: pd.DataFrame, y: pd.DataFrame) -> float:
    a, b = y.values.ravel(), pred.values.ravel()
    m = ~np.isnan(a) & ~np.isnan(b)
    return 1 - ((a[m] - b[m]) ** 2).sum() / (a[m] ** 2).sum()


def decile_book(pred: pd.DataFrame, fwd: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Long top 10 %, short bottom 10 % by forecast, equal weight; rebalanced every `hold` days
    within each year; P&L from raw next-day returns; 5 bp per side on turnover."""
    out = []
    for y, P in pred.groupby(pred.index.year):
        F = fwd.loc[P.index]
        w_prev = pd.Series(0.0, P.columns)
        for i, d in enumerate(P.index):
            if i % hold == 0:
                p = P.loc[d].dropna()
                q = max(int(len(p) * 0.1), 1)
                w = pd.Series(0.0, P.columns)
                w[p.nlargest(q).index] = 0.5 / q
                w[p.nsmallest(q).index] = -0.5 / q
                turn = (w - w_prev).abs().sum()
            else:
                turn = 0.0
            g = float((w * F.loc[d].fillna(0.0)).sum())
            out.append((d, g, g - COST * turn))
            w_prev = w
    return pd.DataFrame(out, columns=["day", "gross", "net"]).set_index("day").dropna()


def book_stats(B: pd.DataFrame) -> dict:
    d = {}
    yrs = len(B) / 252
    for k in ("gross", "net"):
        x = B[k]
        tot = np.prod(1 + x)
        d[f"{k}_total_%"] = (tot - 1) * 100
        d[f"{k}_cagr_%"] = (tot ** (1 / yrs) - 1) * 100
        d[f"{k}_sharpe"] = x.mean() / x.std() * np.sqrt(252)
        d[f"{k}_nw_t"] = nw_t(x)
    return d


# ============================================================================ phases
def run_train() -> None:
    out = OUT / "train"; out.mkdir(parents=True, exist_ok=True)
    px, meta = load()
    stocks = list(meta.index)
    prep = Prep(px, stocks, TRAIN)
    X, Y1, Y5, grp, fwd = design(prep, TRAIN)
    print(f"stocks {len(stocks)}, inputs {X.shape[1]}, training days {len(X)}", flush=True)

    # 1. structure
    Z = pd.concat([prep.year(y)[0].loc[lambda d: d.index.year == y] for y in TRAIN])
    pc = partial_corr(Z)
    links = top_links(pc, meta)
    links.to_csv(out / "structure_top_links.csv", index=False, float_format="%.4g")
    pc.to_pickle(HERE / ".cache" / "discover_pc_train.pkl")

    # 2. lead-lag Lasso, both horizons, and the null
    summary = {}
    for h, Y in (("h1", Y1), ("h5", Y5)):
        C, r2 = fit_lasso(X, Y, grp)
        C.to_pickle(HERE / ".cache" / f"discover_lasso_{h}.pkl")
        E = edges_of(C)
        E.to_csv(out / f"leadlag_edges_{h}.csv", index=False, float_format="%.4g")
        r2.to_csv(out / f"leadlag_cv_r2_{h}.csv", float_format="%.5g")
        summary[h] = dict(targets_with_any_input=int((C != 0).any(axis=1).sum()),
                          cross_edges=int((~E["own"]).sum()), own_edges=int(E["own"].sum()),
                          median_cv_r2=float(r2.median()), mean_cv_r2=float(r2.mean()))
        print(h, summary[h], flush=True)
    Xn, Y1n, _, grpn, _ = design(prep, TRAIN, np.random.default_rng(SEED))
    Cn, r2n = fit_lasso(Xn, Y1n, grpn)
    En = edges_of(Cn)
    En.to_csv(out / "null_edges_h1.csv", index=False, float_format="%.4g")
    summary["null_h1"] = dict(targets_with_any_input=int((Cn != 0).any(axis=1).sum()),
                              cross_edges=int((~En["own"]).sum()), own_edges=int(En["own"].sum()),
                              median_cv_r2=float(r2n.median()), mean_cv_r2=float(r2n.mean()))
    print("null_h1", summary["null_h1"], flush=True)
    pd.DataFrame(summary).T.to_csv(out / "leadlag_summary.csv", float_format="%.5g")

    # 3. pooled GBM
    pos, neg, cl = peer_maps(Z)
    pickle.dump((pos, neg, cl), open(HERE / ".cache" / "discover_peers.pkl", "wb"))
    cl.rename("cluster").to_frame().join(meta[["GICS Sector", "GICS Sub-Industry"]]).to_csv(out / "clusters.csv")
    D = gbm_panel(X, Y1, stocks, pos, neg, cl)
    D = D[D["y"].notna()]
    feats = [c for c in D.columns if c not in ("y", "day", "stock")]
    gbm = HistGradientBoostingRegressor(**HGB).fit(D[feats], D["y"])
    pickle.dump((gbm, feats), open(HERE / ".cache" / "discover_gbm.pkl", "wb"))

    # in-sample diagnostics (training years only)
    P1 = predict(pd.read_pickle(HERE / ".cache" / "discover_lasso_h1.pkl"), X)
    ic = daily_ic(P1, Y1)
    diag = {"lasso_h1_in_sample_ic": ic.mean(), "lasso_h1_in_sample_ic_t": nw_t(ic)}
    pd.Series(diag).to_csv(out / "in_sample_diagnostics.csv", float_format="%.5g")
    print("in-sample", diag)
    print("structure label shares:", links["label"].value_counts(normalize=True).round(3).to_dict())


def lasso_eval(C, X, Y, fwd, hold) -> tuple[dict, pd.Series]:
    P = predict(C, X)
    ic = daily_ic(P, Y)
    d = {"ic_mean": ic.mean(), "ic_nw_t": nw_t(ic), "oos_r2_%": oos_r2(P, Y) * 100}
    d.update(book_stats(decile_book(P, fwd, hold)))
    return d, ic


def run_validate() -> None:
    tr, out = OUT / "train", OUT / "validate"
    out.mkdir(parents=True, exist_ok=True)
    px, meta = load()
    stocks = list(meta.index)
    prep = Prep(px, stocks, TRAIN)                                  # factors from training years only
    X, Y1, Y5, grp, fwd = design(prep, VALID)
    rows, ics = {}, {}

    for h, Y, hold in (("h1", Y1, 1), ("h5", Y5, 5)):
        C = pd.read_pickle(HERE / ".cache" / f"discover_lasso_{h}.pkl")
        rows[f"lasso_{h}"], ics[f"lasso_{h}"] = lasso_eval(C, X, Y, fwd, hold)
        # cross-stock links only: drop own-lag and common (factor / ETF) inputs
        Cx = C.copy()
        for t in Cx.index:
            Cx.loc[t, [f"{t}|1", f"{t}|2-5"]] = 0.0
        Cx.loc[:, [c for c in Cx.columns if c != "_intercept" and c.split("|")[0] not in stocks]] = 0.0
        rows[f"lasso_{h}_cross_only"], _ = lasso_eval(Cx, X, Y, fwd, hold)
        for y in VALID:
            m = X.index.year == y
            rows[f"lasso_{h}_{y}"], _ = lasso_eval(C, X[m], Y[m], fwd[m], hold)

    # null model's forecasts out of sample
    Xn, Y1n, _, grpn, _ = design(prep, TRAIN, np.random.default_rng(SEED))
    Cn, _ = fit_lasso(Xn, Y1n, grpn)
    rows["null_lasso_h1"], _ = lasso_eval(Cn, X, Y1, fwd, 1)

    # GBM
    gbm, feats = pickle.load(open(HERE / ".cache" / "discover_gbm.pkl", "rb"))
    pos, neg, cl = pickle.load(open(HERE / ".cache" / "discover_peers.pkl", "rb"))
    D = gbm_panel(X, Y1, stocks, pos, neg, cl)
    pg = gbm.predict(D[feats]).reshape(len(X), len(stocks))
    PG = pd.DataFrame(pg, X.index, stocks)
    ic = daily_ic(PG, Y1)
    rows["gbm_h1"] = {"ic_mean": ic.mean(), "ic_nw_t": nw_t(ic), "oos_r2_%": oos_r2(PG, Y1) * 100,
                      **book_stats(decile_book(PG, fwd, 1))}

    # strict walk-forward: past training years only
    for y in VALID:
        past = [t for t in TRAIN if t < y]
        prep_y = Prep(px, stocks, past)
        Xp, Yp, _, gp, _ = design(prep_y, past)
        if len(past) == 1:                                          # one year: CV on its two halves
            gp = pd.Series(np.where(np.arange(len(gp)) < len(gp) // 2, 0, 1), gp.index)
        Cw, _ = fit_lasso(Xp, Yp, gp)
        Xv, Yv, _, _, fv = design(prep_y, [y])
        rows[f"walkforward_lasso_h1_{y}"], _ = lasso_eval(Cw, Xv, Yv, fv, 1)

    T = pd.DataFrame(rows).T
    T.round(4).to_csv(out / "forecast_and_trading.csv")

    # persistence of discovered links: refit both models on the validation years
    Zv = pd.concat([prep.year(y)[0].loc[lambda d: d.index.year == y] for y in VALID])
    pc_t = pd.read_pickle(HERE / ".cache" / "discover_pc_train.pkl")
    pc_v = partial_corr(Zv)
    iu = np.triu_indices(len(stocks), 1)
    a, b = pc_t.values[iu], pc_v.values[iu]
    top_t = set(np.argsort(-np.abs(a))[:TOP_EDGES]); top_v = set(np.argsort(-np.abs(b))[:TOP_EDGES])
    links_v = top_links(pc_v, meta)
    links_v.to_csv(out / "structure_top_links.csv", index=False, float_format="%.4g")
    struct = {"partial_corr_train_vs_valid_corr": np.corrcoef(a, b)[0, 1],
              "top1000_overlap_%": len(top_t & top_v) / TOP_EDGES * 100,
              "top1000_overlap_by_chance_%": TOP_EDGES / len(a) * 100,
              "top1000_same_subindustry_%_train": (pd.read_csv(tr / "structure_top_links.csv")["label"] == "same sub-industry").mean() * 100,
              "top1000_same_subindustry_%_valid": (links_v["label"] == "same sub-industry").mean() * 100}
    pd.Series(struct).to_csv(out / "structure_persistence.csv", float_format="%.4g")

    Cv, _ = fit_lasso(X, Y1, grp)
    Et = pd.read_csv(tr / "leadlag_edges_h1.csv"); Ev = edges_of(Cv)
    Ev.to_csv(out / "leadlag_edges_h1_refit.csv", index=False, float_format="%.4g")
    kt = set(zip(Et.loc[~Et["own"], "target"], Et.loc[~Et["own"], "input"]))
    kv = set(zip(Ev.loc[~Ev["own"], "target"], Ev.loc[~Ev["own"], "input"]))
    n_possible = len(stocks) * (X.shape[1] - 2)
    both = kt & kv
    sgn_t = Et.set_index(["target", "input"])["coef"]; sgn_v = Ev.set_index(["target", "input"])["coef"]
    same_sign = np.mean([np.sign(sgn_t[k]) == np.sign(sgn_v[k]) for k in both]) * 100 if both else np.nan
    lead = {"train_cross_edges": len(kt), "valid_refit_cross_edges": len(kv), "found_in_both": len(both),
            "expected_in_both_by_chance": len(kt) * len(kv) / n_possible, "same_sign_%_of_both": same_sign}
    pd.Series(lead).to_csv(out / "leadlag_persistence.csv", float_format="%.4g")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("Forecasts and trading, validation years\n", T.round(3))
    print("\nStructure persistence\n", pd.Series(struct).round(3))
    print("\nLead-lag link persistence (cross-stock links, next-day model)\n", pd.Series(lead).round(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "validate"], required=True)
    run_train() if ap.parse_args().phase == "train" else run_validate()


if __name__ == "__main__":
    main()
