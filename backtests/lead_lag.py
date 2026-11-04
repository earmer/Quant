"""Do upstream or competitor stocks move against each other? A pre-registered test.

HYPOTHESIS (as put to us): when an upstream stock rises, its downstream customers or its competitors
dip a little.

PLAN (written and committed before any 2021, 2023 or 2025 statistic was computed)

1. Relationships are defined two ways.
   a. Economic links fixed in advance (lists below, chosen from industry structure, not from
      returns): 48 supplier/input -> customer pairs (oil -> airlines, steel -> carmakers, memory ->
      PC makers, foundry -> chip designers, lumber -> homebuilders, hospitals -> insurers, ...),
      each also tested in reverse (customer -> supplier), and 46 competitor pairs tested in both
      directions. 186 ordered pairs in all.
   b. Data-mined links: every ordered pair among the same ~90 stocks (about 8,000), ranked on
      the training years. This tests whether relationships found by search persist.

2. Moves are measured as market-adjusted ("residual") returns: e = r - beta * r_SPY. Without this,
   almost every pair co-moves positively because the whole market moves together. Sector ETFs are
   deliberately NOT removed: two large competitors make up much of their own sector index, so
   removing it would manufacture negative residual correlation between them, i.e. fake the
   hypothesis. Residuals are scaled by their own volatility and clipped at +/-4 sigma (so one COVID
   day cannot dominate). In training years beta and sigma come from that year; in a validation
   year they come from the previous (training) year, so nothing from the validation year leaks in.

3. For each ordered pair A -> B (A leads):
     corr0  same-day correlation of A's and B's residuals (the hypothesis says negative)
     corr1  A on day t vs B on day t+1 (a tradable lead-lag)
     corr5  A on day t vs B summed over t+1..t+5
     ev0, ev5  B's mean residual (bp) on the day A rises by more than 2 sigma, and over the next
              five days; ev0_dn / ev5_dn the same after A falls by more than 2 sigma.

4. Training (2020, 2022, 2024 pooled): estimate all of the above; select pairs whose corr1 has
   |t| > 2 and record the sign; select the data-mined top 50 by most negative corr0, most negative
   corr1 and most positive corr1.

5. Validation (2021, 2023, 2025 pooled), run once:
   H1  Same-day: is the mean corr0 (and ev0) of the supplier->customer and competitor groups
       negative? Newey-West t-stats on the daily cross-pair average (lag 10).
   H2  Persistence: do pair-level estimates in training rank-correlate with validation?
       Do the selected pairs keep their training sign more than half the time?
   H3  Tradability (only lagged effects can be traded): two daily strategies, each position
       long or short B hedged with beta x SPY, held one day, 5 bp per side on both legs:
         a priori  bet the hypothesis on all supplier->customer and competitor pairs:
                   short B after A rose > 1 sigma, long B after A fell > 1 sigma
         learned   on the pairs selected in training, trade B in the training sign of corr1
   The hypothesis passes only if the predicted sign shows up in validation with |t| > 2.

    python backtests/lead_lag.py --phase train      # freezes results_lead_lag/train/
    python backtests/lead_lag.py --phase validate   # run once, after the train phase is committed
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_lead_lag"
CACHE = HERE / ".cache" / "lead_lag_prices.pkl"
TRAIN, VALID = [2020, 2022, 2024], [2021, 2023, 2025]
COST = 5.0 / 1e4
CLIP, EVENT, TRADE = 4.0, 2.0, 1.0
NW_LAG = 10

SUPPLY = [  # (upstream / input or supplier, downstream / customer)
    ("XOM", "DAL"), ("XOM", "UAL"), ("XOM", "LUV"), ("COP", "DAL"), ("COP", "UAL"), ("COP", "LUV"),  # crude -> jet fuel
    ("VLO", "FDX"), ("VLO", "UPS"), ("VLO", "JBHT"), ("MPC", "ODFL"),                                 # diesel -> freight
    ("XOM", "DOW"), ("XOM", "LYB"),                                                                   # feedstock -> chemicals
    ("EQT", "CF"), ("EQT", "NTR"),                                                                    # gas -> fertiliser
    ("NUE", "F"), ("NUE", "GM"), ("STLD", "F"), ("CLF", "GM"), ("NUE", "CAT"), ("NUE", "DE"),         # steel -> users
    ("AA", "BALL"), ("AA", "CCK"),                                                                    # aluminium -> cans
    ("ALB", "TSLA"),                                                                                  # lithium -> EVs
    ("MU", "HPQ"), ("MU", "DELL"), ("MU", "AAPL"),                                                    # memory -> hardware
    ("TSM", "NVDA"), ("TSM", "AMD"), ("TSM", "AAPL"), ("TSM", "QCOM"),                                # foundry -> designers
    ("AMAT", "TSM"), ("AMAT", "INTC"), ("LRCX", "TSM"), ("LRCX", "MU"), ("ASML", "TSM"), ("ASML", "INTC"),  # tools -> fabs
    ("NVDA", "SMCI"), ("NVDA", "DELL"), ("NVDA", "HPE"),                                              # GPUs -> servers
    ("ADM", "TSN"), ("BG", "TSN"), ("ADM", "HRL"),                                                    # feed -> meat
    ("WY", "DHI"), ("WY", "LEN"),                                                                     # lumber -> builders
    ("HCA", "UNH"), ("HCA", "ELV"), ("HCA", "CI"),                                                    # hospitals -> insurers
    ("HAL", "XOM"), ("SLB", "XOM"),                                                                   # oil services -> producers
]
COMPETE = [
    ("KO", "PEP"), ("HD", "LOW"), ("V", "MA"), ("UPS", "FDX"), ("AMD", "INTC"), ("NVDA", "AMD"),
    ("DAL", "UAL"), ("XOM", "CVX"), ("JPM", "BAC"), ("WMT", "TGT"), ("COST", "WMT"), ("MCD", "YUM"),
    ("CMG", "MCD"), ("NKE", "LULU"), ("UNH", "ELV"), ("CI", "HUM"), ("MRK", "PFE"), ("LLY", "NVO"),
    ("GOOGL", "META"), ("NFLX", "DIS"), ("T", "VZ"), ("TMUS", "VZ"), ("CAT", "DE"), ("LMT", "NOC"),
    ("GM", "F"), ("TSLA", "GM"), ("SPGI", "MCO"), ("AMAT", "LRCX"), ("QCOM", "AVGO"), ("TXN", "ADI"),
    ("EBAY", "ETSY"), ("UBER", "LYFT"), ("EXPE", "BKNG"), ("PG", "CL"), ("MDLZ", "HSY"), ("DHI", "LEN"),
    ("NUE", "STLD"), ("CF", "NTR"), ("AXP", "MA"), ("GS", "MS"), ("AMZN", "WMT"), ("MU", "WDC"),
    ("HAL", "SLB"), ("MO", "PM"), ("ADM", "BG"), ("BALL", "CCK"),
]


def ordered_pairs() -> pd.DataFrame:
    rows = [(a, b, "supplier->customer") for a, b in SUPPLY] + [(b, a, "customer->supplier") for a, b in SUPPLY]
    rows += [(a, b, "competitor") for a, b in COMPETE] + [(b, a, "competitor") for a, b in COMPETE]
    return pd.DataFrame(rows, columns=["A", "B", "group"]).drop_duplicates(["A", "B", "group"])


TICKERS = sorted({t for a, b in SUPPLY + COMPETE for t in (a, b)})


# ============================================================================ data
def load() -> pd.DataFrame:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    cols = {}
    for t in TICKERS + ["SPY"]:
        df = yf.download(t, start="2019-01-01", end="2026-01-01", auto_adjust=True, progress=False, threads=False)
        s = df["Close"]
        cols[t] = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    px = pd.DataFrame(cols)
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(px))
    return px


def residuals(px: pd.DataFrame, years: list[int]) -> dict:
    """{year: (z, e, beta)}: z = residual / sigma clipped at CLIP; e = residual return, same clip."""
    r = px.pct_change()
    out = {}
    for y in years:
        est = y if y in TRAIN else y - 1
        R_est = r[r.index.year == est]
        m = R_est["SPY"]
        beta = R_est.drop(columns="SPY").apply(lambda s: s.cov(m) / m.var())
        sig = (R_est.drop(columns="SPY") - np.outer(m, beta)).std()
        R = r[r.index.year == y]
        e = R.drop(columns="SPY") - np.outer(R["SPY"], beta)
        z = (e / sig).clip(-CLIP, CLIP)
        out[y] = (z, e.clip(-CLIP * sig, CLIP * sig, axis=1), beta)
    return out


# ============================================================================ pair statistics
def fwd_sum(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return sum(df.shift(-i) for i in range(1, k + 1))


def pair_stats(Z: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    zs = pd.concat([Z[y][0] for y in Z]); es = pd.concat([Z[y][1] for y in Z])
    z1 = pd.concat([Z[y][0].shift(-1) for y in Z]); z5 = pd.concat([fwd_sum(Z[y][0], 5) for y in Z])
    e5 = pd.concat([fwd_sum(Z[y][1], 5) for y in Z])
    rows = []
    for a, b, g in pairs.itertuples(index=False):
        if a not in zs or b not in zs:
            continue
        za = zs[a]
        up, dn = za > EVENT, za < -EVENT
        rows.append(dict(
            A=a, B=b, group=g, n=int(za.notna().sum()),
            corr0=za.corr(zs[b]), corr1=za.corr(z1[b]), corr5=za.corr(z5[b]),
            ev_n=int(up.sum()), ev0=es[b][up].mean() * 1e4, ev5=e5[b][up].mean() * 1e4,
            ev_dn_n=int(dn.sum()), ev0_dn=es[b][dn].mean() * 1e4, ev5_dn=e5[b][dn].mean() * 1e4))
    P = pd.DataFrame(rows)
    P["t_corr1"] = P["corr1"] * np.sqrt(P["n"])
    return P


def nw_t(x: pd.Series, lag: int = NW_LAG) -> float:
    x = x.dropna().values
    n = len(x)
    if n < 30:
        return np.nan
    u = x - x.mean()
    v = u @ u / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def group_daily(Z: dict, pairs: pd.DataFrame, kind: str) -> pd.Series:
    """Daily cross-pair average used for the Newey-West t-stat of a group mean."""
    out = []
    for y in Z:
        z, e, _ = Z[y]
        A, B = z[pairs["A"]].values, z[pairs["B"]].values
        if kind == "corr0":
            v = np.nanmean(A * B, axis=1)
        elif kind == "corr1":
            v = np.nanmean(A * z[pairs["B"]].shift(-1).values, axis=1)
        elif kind == "ev0":           # B's residual (bp) on days A rose > EVENT sigma
            m = np.where(A > EVENT, e[pairs["B"]].values * 1e4, np.nan)
            v = np.nanmean(m, axis=1) if m.size else m
        out.append(pd.Series(v, z.index))
    return pd.concat(out)


def strategy(Z: dict, pairs: pd.DataFrame, sign: np.ndarray) -> pd.Series:
    """Each day t+1: for pairs with |zA_t| > TRADE, hold B hedged with beta*SPY in direction
    sign * sign(zA_t); equal weight across active positions; one-day holding; costs on both legs."""
    out = []
    for y in Z:
        z, e, beta = Z[y]
        A = z[pairs["A"]].shift(1).values                  # yesterday's leader move, known at today's open/close
        Re = e[pairs["B"]].values                          # hedged residual return of B today
        pos = np.where(np.abs(A) > TRADE, np.sign(A) * sign, 0.0)
        cost = 2 * COST * (1 + np.abs(beta[pairs["B"]].values))
        pnl = np.where(pos != 0, pos * Re - cost, np.nan)
        gross = np.where(pos != 0, pos * Re, np.nan)
        out.append(pd.DataFrame({"net": np.nanmean(pnl, axis=1), "gross": np.nanmean(gross, axis=1),
                                 "active": (pos != 0).sum(axis=1)}, z.index))
    return pd.concat(out)


def strat_summary(S: pd.DataFrame) -> dict:
    d = {}
    for k in ("gross", "net"):
        x = S[k].fillna(0.0)
        d[f"{k}_ann_%"] = x.mean() * 252 * 100
        d[f"{k}_sharpe"] = x.mean() / x.std() * np.sqrt(252)
        d[f"{k}_nw_t"] = nw_t(x)
        d[f"{k}_total_%"] = (np.prod(1 + x) - 1) * 100
    d["avg_positions"] = S["active"].mean()
    return d


# ============================================================================ data-mined pairs
def corr_matrices(Z: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    z = pd.concat([Z[y][0] for y in Z]); z1 = pd.concat([Z[y][0].shift(-1) for y in Z])
    ok = z.notna().all() & z1.notna().sum().gt(0)
    z, z1 = z.loc[:, ok], z1.loc[:, ok]
    m = z.notna().all(axis=1) & z1.notna().all(axis=1)
    a = (z[m] - z[m].mean()) / z[m].std(); b = (z1[m] - z1[m].mean()) / z1[m].std()
    n = int(m.sum())
    c0 = a.T @ a / (n - 1); c1 = a.T @ b / (n - 1)       # c1[i, j] = corr(z_i,t, z_j,t+1)
    return c0, c1, n


def mined_selection(c0: pd.DataFrame, c1: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    names = c0.index
    s0 = c0.where(np.triu(np.ones(c0.shape, bool), 1)).stack()
    off = ~np.eye(len(names), dtype=bool)
    s1 = c1.where(off).stack()
    rows = []
    for (a, b), v in s0.nsmallest(k).items():
        rows.append(dict(A=a, B=b, set="most_negative_corr0", train=v))
    for (a, b), v in s1.nsmallest(k).items():
        rows.append(dict(A=a, B=b, set="most_negative_corr1", train=v))
    for (a, b), v in s1.nlargest(k).items():
        rows.append(dict(A=a, B=b, set="most_positive_corr1", train=v))
    rows.append(dict(A="*", B="*", set="all_pairs_corr0_mean", train=s0.mean()))
    rows.append(dict(A="*", B="*", set="all_pairs_corr1_mean", train=s1.mean()))
    return pd.DataFrame(rows)


# ============================================================================ phases
def group_table(Z: dict, P: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, sub in P.groupby("group"):
        pr = pairs[pairs["group"] == g]
        row = {"group": g, "pairs": len(sub)}
        for k in ("corr0", "corr1", "corr5", "ev0", "ev5", "ev0_dn", "ev5_dn"):
            row[f"{k}_mean"] = sub[k].mean()
        row["corr0_pct_neg"] = (sub["corr0"] < 0).mean() * 100
        row["corr1_pct_neg"] = (sub["corr1"] < 0).mean() * 100
        for k in ("corr0", "corr1", "ev0"):
            row[f"{k}_nw_t"] = nw_t(group_daily(Z, pr, k))
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def run_train(px: pd.DataFrame) -> None:
    out = OUT / "train"; out.mkdir(parents=True, exist_ok=True)
    Z = residuals(px, TRAIN)
    pairs = ordered_pairs()
    P = pair_stats(Z, pairs)
    P.to_csv(out / "pair_stats.csv", index=False, float_format="%.5g")
    G = group_table(Z, P, pairs)
    G.round(4).to_csv(out / "group_summary.csv")
    sel = P[P["t_corr1"].abs() > 2][["A", "B", "group", "corr1", "t_corr1"]].copy()
    sel["sign"] = np.sign(sel["corr1"])
    sel.to_csv(out / "selected_pairs.csv", index=False, float_format="%.5g")
    hyp = pairs[pairs["group"].isin(["supplier->customer", "competitor"])]
    strat = {"a_priori_hypothesis": strat_summary(strategy(Z, hyp, -1.0)),
             "learned_pairs": strat_summary(strategy(Z, sel, sel["sign"].values))}
    pd.DataFrame(strat).T.round(4).to_csv(out / "strategies.csv")
    c0, c1, n = corr_matrices(Z)
    M = mined_selection(c0, c1)
    M.to_csv(out / "mined_pairs.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"Training years {TRAIN}: {len(P)} ordered pairs, {n} pooled days, {c0.shape[0]} stocks mined")
    print("\nGroup summary (training)\n", G.round(3))
    print(f"\nSelected lead-lag pairs (|t| > 2): {len(sel)} of {len(P)}; expected by chance ~{0.0455 * len(P):.0f}")
    print(sel.sort_values("t_corr1").to_string(index=False))
    print("\nStrategies (training, in-sample for 'learned')\n", pd.DataFrame(strat).T.round(3))
    print("\nData-mined sets (training means)\n", M.groupby("set")["train"].mean().round(4))


def run_validate(px: pd.DataFrame) -> None:
    tr = OUT / "train"; out = OUT / "validate"; out.mkdir(parents=True, exist_ok=True)
    Z = residuals(px, VALID)
    pairs = ordered_pairs()
    P = pair_stats(Z, pairs)
    P.to_csv(out / "pair_stats.csv", index=False, float_format="%.5g")
    G = group_table(Z, P, pairs)
    G.round(4).to_csv(out / "group_summary.csv")
    Pt = pd.read_csv(tr / "pair_stats.csv")
    J = Pt.merge(P, on=["A", "B", "group"], suffixes=("_train", "_valid"))
    pers = {}
    for g, sub in [("all", J)] + list(J.groupby("group")):
        pers[g] = {f"{k}_rank_corr": sub[f"{k}_train"].corr(sub[f"{k}_valid"], method="spearman")
                   for k in ("corr0", "corr1", "corr5", "ev0", "ev5")}
        pers[g]["pairs"] = len(sub)
    pers = pd.DataFrame(pers).T
    pers.round(3).to_csv(out / "persistence.csv")
    sel = pd.read_csv(tr / "selected_pairs.csv")
    sv = sel.merge(P[["A", "B", "group", "corr1"]], on=["A", "B", "group"], suffixes=("_train", "_valid"))
    keep = (np.sign(sv["corr1_valid"]) == sv["sign"]).mean() * 100
    sv.to_csv(out / "selected_pairs_validation.csv", index=False, float_format="%.5g")
    hyp = pairs[pairs["group"].isin(["supplier->customer", "competitor"])]
    strat = {"a_priori_hypothesis": strat_summary(strategy(Z, hyp, -1.0)),
             "learned_pairs": strat_summary(strategy(Z, sel, sel["sign"].values))}
    for y in VALID:
        Zy = {y: Z[y]}
        strat[f"learned_pairs_{y}"] = strat_summary(strategy(Zy, sel, sel["sign"].values))
        strat[f"a_priori_{y}"] = strat_summary(strategy(Zy, hyp, -1.0))
    pd.DataFrame(strat).T.round(4).to_csv(out / "strategies.csv")
    c0, c1, n = corr_matrices(Z)
    M = pd.read_csv(tr / "mined_pairs.csv")
    names = set(c0.index)
    valid = []
    for r in M.itertuples(index=False):
        if r.A == "*":
            off = ~np.eye(len(c0), dtype=bool)
            v = (c0.where(np.triu(np.ones(c0.shape, bool), 1)).stack().mean() if "corr0" in r.set
                 else c1.where(off).stack().mean())
        elif r.A in names and r.B in names:
            v = c0.loc[r.A, r.B] if "corr0" in r.set else c1.loc[r.A, r.B]
        else:
            v = np.nan
        valid.append(v)
    M["valid"] = valid
    M.to_csv(out / "mined_pairs_validation.csv", index=False, float_format="%.5g")
    MS = M.groupby("set").agg(train=("train", "mean"), valid=("valid", "mean"),
                              same_sign_pct=("valid", lambda s: np.nan))
    for s_ in MS.index:
        sub = M[M["set"] == s_]
        MS.loc[s_, "same_sign_pct"] = (np.sign(sub["train"]) == np.sign(sub["valid"])).mean() * 100
    MS.round(4).to_csv(out / "mined_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"Validation years {VALID}: {len(P)} ordered pairs, {n} pooled days")
    print("\nGroup summary (validation)\n", G.round(3))
    print("\nTrain -> validation rank correlation of pair estimates\n", pers.round(3))
    print(f"\nSelected lead-lag pairs keeping their training sign: {keep:.0f} % of {len(sv)}")
    print("\nStrategies (validation)\n", pd.DataFrame(strat).T.round(3))
    print("\nData-mined sets\n", MS.round(4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "validate"], required=True)
    args = ap.parse_args()
    px = load()
    missing = [t for t in TICKERS if px[t][px.index.year >= 2020].isna().mean() > 0.01]
    if missing:
        print("[warn] incomplete history, excluded where missing:", missing)
    run_train(px) if args.phase == "train" else run_validate(px)


if __name__ == "__main__":
    main()
