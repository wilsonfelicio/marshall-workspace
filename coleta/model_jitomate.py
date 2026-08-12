"""Fortnightly nowcast of CPI jitomate, following the design already agreed.

Everything here was decided before seeing a single jitomate result:
  * three release vintages - day 5, day 10, close of fortnight - each comparing
    day 1..k against day 1..k of the previous fortnight;
  * equal-weight COMBINATION of two pre-registered specifications rather than
    picking a winner, because on avocado every variant was statistically tied and
    choosing among tied models on out-of-sample results is itself a leak;
  * recursive expanding-window estimation, refit every fortnight;
  * a structural audit that refuses any CPI-derived regressor dated inside the
    target fortnight, and a placebo regressor with the right distribution and the
    wrong dates;
  * intervals from a recursive scale model, with multipliers taken from the
    series' own earlier standardised errors rather than a normal table.

Headline metric, as agreed: the error in PERCENTAGE POINTS of the published print,
and its translation into contribution to the headline INPC (jitomate carries 0.79
of the 100-point basket).

Reads the isolated jitomate store while the main one is locked.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import pathlib
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

Q.DAILY = "/root/jit/var_market_daily.parquet"
Q.CACHE_DIR = pathlib.Path("/root/jit/cache")

OOS_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2011
PESO = 0.79014          # jitomate's 2024 basket weight, in points of 100
HARM, MIN_TRAIN, SIGMA_EVERY = 3, 120, 12
VINTAGES = {"d5": "x_w5", "d10": "x_w10", "close": "x_full"}
SEAS = [f"{f}{k}" for k in range(1, HARM + 1) for f in ("sin", "cos")]


def ols(A, b):
    return np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), b)


def cw(y, small, big):
    m = ~(np.isnan(y) | np.isnan(small) | np.isnan(big))
    y, s, b = y[m], small[m], big[m]
    f = (y - s) ** 2 - ((y - b) ** 2 - (s - b) ** 2)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return float(t), float(0.5 * (1 - erf(t / sqrt(2))))


d = Q.dataset("jitomate", "070 Jitomate", windows=(5, 10))
ang = 2 * np.pi * (d["quincena_del_anio"] - 1) / 24
for k in range(1, HARM + 1):
    d[f"sin{k}"], d[f"cos{k}"] = np.sin(k * ang), np.cos(k * ang)
d["x_shuffled"] = np.roll(d["x_full"].to_numpy(float), len(d) // 3)
d = d.set_index("t")

COLS = SEAS + ["y_lag1", "y_lag2", "x_shuffled"] + \
       [c for x in VINTAGES.values() for c in (x, f"{x}_lag1")]
M = d[COLS].to_numpy(float)
yv = d["y"].to_numpy(float)
ix = {c: i for i, c in enumerate(COLS)}
ok_y = ~np.isnan(yv)


def valid(cols):
    j = [ix[c] for c in cols]
    return (~np.isnan(M[:, j]).any(axis=1)) & ok_y


def fit_at(i, cols):
    j = [ix[c] for c in cols]
    if np.isnan(M[i, j]).any():
        return np.nan
    idx = np.flatnonzero(valid(cols)[:i])
    if len(idx) < MIN_TRAIN:
        return np.nan
    X = np.column_stack([np.ones(len(idx)), M[np.ix_(idx, j)]])
    b = ols(X.T @ X, X.T @ yv[idx])
    return float(np.concatenate([[1.0], M[i, j]]) @ b)


_sig = {}


def sigma_at(i, xcol, small, big):
    jx = ix[xcol]
    if np.isnan(M[i, jx]):
        return np.nan
    anchor = i - (i % SIGMA_EVERY)
    key = (xcol, anchor)
    if key not in _sig:
        idx = np.flatnonzero(valid(list(dict.fromkeys(small + big)))[:max(anchor, 1)])
        if len(idx) < MIN_TRAIN:
            _sig[key] = None
        else:
            fits = []
            for cols in (small, big):
                j = [ix[c] for c in cols]
                X = np.column_stack([np.ones(len(idx)), M[np.ix_(idx, j)]])
                fits.append(X @ ols(X.T @ X, X.T @ yv[idx]))
            r = np.abs(yv[idx] - np.mean(fits, axis=0))
            ax = np.abs(M[idx, jx])
            Z = np.column_stack([np.ones(len(ax)), ax])
            _sig[key] = ols(Z.T @ Z, Z.T @ r)
    g = _sig[key]
    if g is None:
        return np.nan
    return float(max(g[0] + g[1] * abs(M[i, jx]), 0.3) * 1.2533)


t0 = np.searchsorted(d.index.to_numpy(), OOS_YEAR * 24)
rows = []
for i in range(t0, len(d)):
    if np.isnan(yv[i]):
        continue
    rec = {"t": int(d.index[i]), "y": float(yv[i])}
    for vn, xc in VINTAGES.items():
        two, big = [xc, f"{xc}_lag1"], SEAS + ["y_lag1", "y_lag2", xc, f"{xc}_lag1"]
        p2, pb = fit_at(i, two), fit_at(i, big)
        rec[f"{vn}_b2"], rec[f"{vn}_ar"] = p2, pb
        rec[f"{vn}"] = np.nan if (np.isnan(p2) or np.isnan(pb)) else (p2 + pb) / 2
        rec[f"{vn}_bench"] = fit_at(i, SEAS + ["y_lag1", "y_lag2"])
        rec[f"{vn}_placebo"] = fit_at(i, SEAS + ["y_lag1", "y_lag2", "x_shuffled"])
        rec[f"{vn}_sigma"] = sigma_at(i, xc, two, big)
    rows.append(rec)
fc = pd.DataFrame(rows)
fc["etiqueta"] = [Q.qlabel(t) for t in fc["t"]]
fc.to_csv("data/curated/oos_jitomate.csv", index=False)

R = lambda a, b: float(np.sqrt(((a - b).dropna() ** 2).mean()))
print(f"JITOMATE fortnightly nowcast, out of sample from {OOS_YEAR}")
print(f"{len(fc)} fortnights, {fc.etiqueta.iloc[0]} - {fc.etiqueta.iloc[-1]}")
print(f"target sd {fc.y.std():.2f} pp; basket weight {PESO:.2f} of 100\n")
print(f"{'vintage':<8}{'RMSE pp':>9}{'MAE pp':>8}{'vs SD-AR':>10}{'R2oos':>8}"
      f"{'sign':>7}{'CW t':>7}{'CW p':>8}{'headline ±pp':>14}{'placebo':>9}")
for vn in VINTAGES:
    r1, r0 = R(fc.y, fc[vn]), R(fc.y, fc[f"{vn}_bench"])
    e = (fc.y - fc[vn]).dropna()
    r2 = 1 - (e ** 2).sum() / ((fc.y - fc.y.mean()) ** 2).sum()
    sg = 100 * float((np.sign(fc.y) == np.sign(fc[vn])).mean())
    t, p = cw(fc.y.values, fc[f"{vn}_bench"].values, fc[vn].values)
    pl = 100 * (R(fc.y, fc[f"{vn}_placebo"]) / r0 - 1)
    print(f"{vn:<8}{r1:9.3f}{e.abs().mean():8.3f}{100*(r1/r0-1):9.1f}%{r2:8.3f}"
          f"{sg:6.0f}%{t:7.2f}{p:8.4f}{r1*PESO/100:14.4f}{pl:8.1f}%")
print(f"\nbenchmark SD-AR RMSE: close {R(fc.y, fc.close_bench):.3f} pp "
      f"(= {R(fc.y, fc.close_bench)*PESO/100:.4f} pp of headline)")

z = fc.dropna(subset=["close", "close_sigma"]).copy()
z["zz"] = (z.y - z.close).abs() / z.close_sigma
for lvl in (0.80, 0.90):
    zv = z.zz.to_numpy()
    k = np.array([np.nan if i < 40 else np.quantile(zv[:i], lvl) for i in range(len(zv))])
    m = ~np.isnan(k)
    print(f"interval {lvl:.0%}: coverage {100*(zv[m] <= k[m]).mean():.1f}%, "
          f"mean half-width ±{(k[m]*z.close_sigma.to_numpy()[m]).mean():.2f} pp "
          f"(empirical multiplier {np.nanmedian(k):.2f})")

yr = fc.assign(anio=(fc.t // 24).astype(int))
w = yr.groupby("anio").apply(lambda g: pd.Series(
    {"nowcast": R(g.y, g.close), "bench": R(g.y, g.close_bench)}))
print("\nRMSE by year (close vintage vs benchmark):")
print("  " + "  ".join(f"{a}:{r.nowcast:.1f}/{r.bench:.1f}" for a, r in w.iterrows()))
e = (fc.y - fc.close)
print(f"\nworst miss: {fc.loc[e.abs().idxmax(),'etiqueta']} "
      f"actual {fc.loc[e.abs().idxmax(),'y']:+.1f}% vs {fc.loc[e.abs().idxmax(),'close']:+.1f}% "
      f"predicted")
print("last 4: " + "  ".join(f"{r.etiqueta} {r.y:+.1f}/{r.close:+.1f}"
                             for r in fc.tail(4).itertuples()))
