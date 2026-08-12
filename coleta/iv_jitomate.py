"""Split-half instrumental variables for jitomate: how much measurement error is
in the wholesale index, and does correcting for it help?

Construction. The 52 markets are split at random into two halves, and a full
matched-cell index is built from each. Both measure the same latent wholesale
signal with sampling error that is independent by construction, because no market
appears in both. So half B is a valid instrument for half A, and 2SLS recovers the
structural slope free of attenuation. Repeated over K random splits, since a single
split is itself noisy.

What to expect, stated before running it. IV should give a LARGER slope than OLS -
that is the attenuation correction, and the ratio is a direct estimate of the
index's reliability. But IV should NOT improve forecast accuracy, and if it appears
to, be suspicious. The reason is a standard result that is easy to forget: when you
must predict using a noisy regressor, the attenuated OLS coefficient is the
MSE-optimal one. Shrinkage toward zero is not a bug, it is the right response to
knowing your input is noisy. IV buys a consistent structural coefficient, which is
what you want for interpretation and pass-through, not a better prediction.

The number that matters for deciding where to invest is therefore the reliability
ratio, not the RMSE: it says how much of the index's variance is signal, and so how
much headroom a better-measured index could possibly buy.

Usage: python3 iv_jitomate.py [n_splits]
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

Q.DAILY = pathlib.Path("/root/jit/var_market_daily.parquet")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MIN_CELDAS, MIN_TRAIN, HARM = 5, 120, 3
OOS_T0 = 2011 * 24
SEED = 20260812

d = pd.read_parquet(Q.DAILY)
d["fecha"] = pd.to_datetime(d["fecha"])
d["t"] = Q.qindex(d["fecha"])
cells = Q._cells(d, partial=False)
pes = pd.read_parquet(Q.PESOS)[["destino", "peso_inpc"]].copy()
W = dict(zip(pes.destino, pd.to_numeric(pes.peso_inpc, errors="coerce").astype(float)))
markets = sorted(cells.destino.unique())
n_dias = d.groupby("t")["fecha"].nunique()
FULL = set(n_dias[n_dias >= 0.6 * n_dias.median()].index)
p = Q.inpc_quincenal("070 Jitomate").set_index("t")["inpc"]


def build(mkts):
    c = cells[cells.destino.isin(mkts)]
    by_t = {t: g.set_index(["producto_id", "destino"]) for t, g in c.groupby("t")}
    out, prev = [], None
    for t in sorted(by_t):
        a = by_t[t]
        dln = np.nan
        if prev is not None:
            b = by_t[prev]
            com = a.index.intersection(b.index)
            if len(com) >= MIN_CELDAS:
                r = pd.DataFrame({"dln": a.loc[com, "lp"].values - b.loc[com, "lp"].values,
                                  "destino": [i[1] for i in com]})
                per = r.groupby("destino")["dln"].mean()
                pw = np.array([W.get(k, 0.0) for k in per.index])
                if pw.sum() <= 0:
                    pw = np.ones(len(per))
                dln = float(np.average(per.values, weights=pw))
        if len(a) >= MIN_CELDAS:
            prev = t
        out.append({"t": t, "dln": dln})
    return pd.DataFrame(out).set_index("t")["dln"] * 100


def frame(xs):
    lo = max(min(p.index), min(min(s.index) for s in xs.values()))
    hi = min(max(p.index), max(max(s.index) for s in xs.values()))
    g = pd.DataFrame(index=pd.Index(range(lo, hi + 1), name="t"))
    g["y"] = 100 * np.log(p.reindex(g.index)).diff()
    for k, s in xs.items():
        g[k] = s.reindex(g.index)
        g[f"{k}_l1"] = g[k].shift(1)
    return g[[i in FULL for i in g.index]]


def recursive(g, xc, zc=None, predict_with=None):
    """Recursive OOS. With zc, 2SLS using zc as instruments for xc.

    predict_with lets us estimate the slope on one index and apply it to another,
    which is how we test whether an unattenuated coefficient plus the full-sample
    index beats attenuated OLS on the same index.
    """
    pc = predict_with or xc
    cols = list(dict.fromkeys(xc + (zc or []) + pc + ["y"]))
    A = g[cols].to_numpy(float)
    ok = ~np.isnan(A).any(axis=1)
    ix = {c: i for i, c in enumerate(cols)}
    y = g["y"].to_numpy(float)
    jx, jz, jp = [ix[c] for c in xc], [ix[c] for c in (zc or [])], [ix[c] for c in pc]
    t0 = int(np.searchsorted(g.index.to_numpy(), OOS_T0))
    e, bs = [], []
    for i in range(t0, len(g)):
        if not ok[i]:
            continue
        idx = np.flatnonzero(ok[:i])
        if len(idx) < MIN_TRAIN:
            continue
        X = np.column_stack([np.ones(len(idx)), A[np.ix_(idx, jx)]])
        yy = y[idx]
        if zc:
            Z = np.column_stack([np.ones(len(idx)), A[np.ix_(idx, jz)]])
            PZ = Z @ np.linalg.solve(Z.T @ Z + 1e-9 * np.eye(Z.shape[1]), Z.T)
            b = np.linalg.solve(X.T @ PZ @ X + 1e-9 * np.eye(X.shape[1]), X.T @ PZ @ yy)
        else:
            b = np.linalg.solve(X.T @ X + 1e-9 * np.eye(X.shape[1]), X.T @ yy)
        xp = np.concatenate([[1.0], A[i, jp]])
        e.append(y[i] - xp @ b)
        bs.append(b[1])
    e = np.array(e)
    return float(np.sqrt((e ** 2).mean())), float(np.median(bs)), e


full = build(markets)
g_full = frame({"F": full})
rmse_ols_full, b_ols_full, e_full = recursive(g_full, ["F", "F_l1"])
print(f"{len(markets)} markets, {len(cells):,} cells. Baseline OLS on the full index: "
      f"RMSE {rmse_ols_full:.3f} pp, median slope {b_ols_full:.3f}\n")

rng = np.random.default_rng(SEED)
rows = []
for k in range(K):
    m = np.array(markets)
    rng.shuffle(m)
    h = len(m) // 2
    A, B = build(set(m[:h])), build(set(m[h:]))
    g = frame({"A": A, "B": B, "F": full})
    r_ols, b_ols, _ = recursive(g, ["A", "A_l1"])
    r_iv, b_iv, e_iv = recursive(g, ["A", "A_l1"], ["B", "B_l1"])
    # unattenuated slope applied to the full (less noisy) index
    r_ivF, _, _ = recursive(g, ["A", "A_l1"], ["B", "B_l1"], predict_with=["F", "F_l1"])
    rho = float(A.corr(B))
    rows.append({"split": k, "corr_AB": rho, "b_OLS_half": b_ols, "b_IV": b_iv,
                 "reliab_half": b_ols / b_iv if b_iv else np.nan,
                 "RMSE_OLS_half": r_ols, "RMSE_IV_half": r_iv, "RMSE_IV_on_full": r_ivF})
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

md = T.median(numeric_only=True)
sb = 2 * md.corr_AB / (1 + md.corr_AB)          # Spearman-Brown for the full index
print(f"\nmedian over {K} random splits")
print(f"  corr(half A, half B)                 {md.corr_AB:.3f}   "
      f"= reliability of a HALF index")
print(f"  implied reliability of the FULL index {sb:.3f}   "
      f"(Spearman-Brown, 2r/(1+r))")
print(f"  slope, OLS on a half                 {md.b_OLS_half:.3f}")
print(f"  slope, IV (attenuation-corrected)    {md.b_IV:.3f}   "
      f"ratio {md.b_OLS_half/md.b_IV:.3f}")
print(f"  slope, OLS on the full index         {b_ols_full:.3f}")
print(f"\n  RMSE, OLS on a half                  {md.RMSE_OLS_half:.3f} pp")
print(f"  RMSE, IV on a half                   {md.RMSE_IV_half:.3f} pp  "
      f"({100*(md.RMSE_IV_half/md.RMSE_OLS_half-1):+.1f}% vs OLS on the same half)")
print(f"  RMSE, IV slope applied to full index {md.RMSE_IV_on_full:.3f} pp  "
      f"({100*(md.RMSE_IV_on_full/rmse_ols_full-1):+.1f}% vs baseline)")
print(f"  RMSE, baseline OLS on full index     {rmse_ols_full:.3f} pp")
print(f"\n  implied structural pass-through, one fortnight: {md.b_IV:.3f} "
      f"vs {b_ols_full:.3f} measured")
noise = 1 - sb
print(f"  so about {100*noise:.0f}% of the full index's fortnightly variance is "
      f"sampling noise, and the attenuation it causes is the {100*(1-b_ols_full/md.b_IV):.0f}% "
      f"gap between the measured and structural slopes")
T.to_csv("data/curated/iv_jitomate.csv", index=False)
