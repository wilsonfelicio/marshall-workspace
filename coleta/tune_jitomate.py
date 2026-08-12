"""Does a different city mix or a different variety weighting improve jitomate?

Every variant is evaluated the same way: rebuild the wholesale index under that
aggregation choice, then run the SAME recursive out-of-sample loop with the same
2-regressor specification, and compare RMSE in pp of the published print. The
aggregation choice is fixed a priori per variant - none of it is tuned on the
out-of-sample window, which would just relabel overfitting as improvement.

Variants:
  base            per market, equal weight across matched varieties; across markets,
                  weighted geometric mean with INPC city weights (what we ship)
  var_by_obs      varieties weighted by their observation count in that cell
  saladette       Saladette only (the dominant variety, 69% of observations)
  bola            Bola only
  two_regressors  Saladette and Bola enter the regression separately, so the
                  recursive OLS picks the mix instead of the index builder
  equal_mkt       markets equal-weighted instead of INPC city weights
  exacto          only markets whose crosswalk to an INPC city is exact
  top20 / top35   the N markets with the most fortnights of coverage
  trimmed         across markets, a 10% trimmed mean of log ratios instead of the
                  weighted mean - robustness to one market printing nonsense
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

Q.DAILY = pathlib.Path("/root/jit/var_market_daily.parquet")
MIN_CELDAS, MIN_TRAIN, HARM = 5, 120, 3
OOS_T0 = 2011 * 24

d = pd.read_parquet(Q.DAILY)
d["fecha"] = pd.to_datetime(d["fecha"])
d["t"] = Q.qindex(d["fecha"])
cells = Q._cells(d, partial=False)                      # t, producto_id, destino, lp, n_dias
nobs = (d.groupby(["t", "producto_id", "destino"], sort=False)
         .size().rename("n").reset_index())
cells = cells.merge(nobs, on=["t", "producto_id", "destino"], how="left")

pes = pd.read_parquet(Q.PESOS)[["destino", "peso_inpc", "metodo"]].copy()
pes["peso"] = pd.to_numeric(pes["peso_inpc"], errors="coerce").astype(float)
WEQ = dict.fromkeys(pes.destino, 1.0)
WINPC = dict(zip(pes.destino, pes.peso))
EXACTO = set(pes.loc[pes.metodo == "exacto", "destino"])
cover = cells.groupby("destino")["t"].nunique().sort_values(ascending=False)


def build(varieties=None, var_w="equal", mkt_w="inpc", markets=None, trim=0.0):
    """Chained matched-cell Jevons under one set of aggregation choices."""
    c = cells
    if varieties is not None:
        c = c[c.producto_id.isin(varieties)]
    if markets is not None:
        c = c[c.destino.isin(markets)]
    W = {"inpc": WINPC, "equal": WEQ}[mkt_w]
    by_t = {t: g for t, g in c.groupby("t")}
    ts = sorted(by_t)
    out, prev = [], None
    for t in ts:
        a = by_t[t].set_index(["producto_id", "destino"])
        dln = np.nan
        if prev is not None:
            b = by_t[prev].set_index(["producto_id", "destino"])
            com = a.index.intersection(b.index)
            if len(com) >= MIN_CELDAS:
                r = pd.DataFrame({
                    "dln": a.loc[com, "lp"].values - b.loc[com, "lp"].values,
                    "n": np.minimum(a.loc[com, "n"].values, b.loc[com, "n"].values),
                    "destino": [i[1] for i in com]})
                if var_w == "obs":
                    per = r.groupby("destino").apply(
                        lambda x: np.average(x.dln, weights=x.n))
                else:
                    per = r.groupby("destino")["dln"].mean()
                if trim > 0 and len(per) >= 10:
                    lo, hi = np.quantile(per.values, [trim, 1 - trim])
                    per = per[(per >= lo) & (per <= hi)]
                pw = np.array([W.get(k, 0.0) for k in per.index])
                if pw.sum() <= 0:
                    pw = np.ones(len(per))
                dln = float(np.average(per.values, weights=pw))
        if len(a) >= MIN_CELDAS:
            prev = t
        out.append({"t": t, "dln": dln})
    return pd.DataFrame(out).set_index("t")["dln"] * 100


p = Q.inpc_quincenal("070 Jitomate").set_index("t")["inpc"]
n_dias = d.groupby("t")["fecha"].nunique()
full = set(n_dias[n_dias >= 0.6 * n_dias.median()].index)
SEAS = [f"{f}{k}" for k in range(1, HARM + 1) for f in ("sin", "cos")]


def panel(xs: dict):
    lo = max(min(p.index), min(min(s.index) for s in xs.values()))
    hi = min(max(p.index), max(max(s.index) for s in xs.values()))
    g = pd.DataFrame(index=pd.Index(range(lo, hi + 1), name="t"))
    g["inpc"] = p.reindex(g.index)
    g["y"] = 100 * np.log(g.inpc).diff()
    for k, s in xs.items():
        g[k] = s.reindex(g.index)
        g[f"{k}_l1"] = g[k].shift(1)
    g = g[[i in full for i in g.index]]
    ang = 2 * np.pi * ((g.index % 24)) / 24
    for k in range(1, HARM + 1):
        g[f"sin{k}"], g[f"cos{k}"] = np.sin(k * ang), np.cos(k * ang)
    return g


def oos(g, cols):
    M = g[cols].to_numpy(float)
    yv = g["y"].to_numpy(float)
    ok = (~np.isnan(M).any(axis=1)) & ~np.isnan(yv)
    t0 = int(np.searchsorted(g.index.to_numpy(), OOS_T0))
    e, n = [], 0
    for i in range(t0, len(g)):
        if not ok[i]:
            continue
        idx = np.flatnonzero(ok[:i])
        if len(idx) < MIN_TRAIN:
            continue
        X = np.column_stack([np.ones(len(idx)), M[idx]])
        b = np.linalg.solve(X.T @ X + 1e-9 * np.eye(X.shape[1]), X.T @ yv[idx])
        e.append(yv[i] - np.concatenate([[1.0], M[i]]) @ b)
        n += 1
    e = np.array(e)
    return float(np.sqrt((e ** 2).mean())), n, e


SAL, BOL = 839, 836
variants = {
    "base":        dict(),
    "var_by_obs":  dict(var_w="obs"),
    "saladette":   dict(varieties=[SAL]),
    "bola":        dict(varieties=[BOL]),
    "equal_mkt":   dict(mkt_w="equal"),
    "exacto":      dict(markets=EXACTO),
    "top20":       dict(markets=set(cover.head(20).index)),
    "top35":       dict(markets=set(cover.head(35).index)),
    "trimmed":     dict(trim=0.10),
}
series = {k: build(**v) for k, v in variants.items()}
print(f"{len(cells):,} cells, {cells.destino.nunique()} markets, "
      f"{len(EXACTO)} exact crosswalk matches, median cells/fortnight "
      f"{cells.groupby('t').size().median():.0f}\n")

base_rmse = None
rows = []
for k in variants:
    g = panel({k: series[k]})
    r, n, _ = oos(g, [k, f"{k}_l1"])
    if k == "base":
        base_rmse = r
    rows.append({"variant": k, "RMSE": r, "n": n})
# both varieties as separate regressors: the regression picks the mix
g2 = panel({"sal": series["saladette"], "bol": series["bola"]})
r2, n2, e2 = oos(g2, ["sal", "sal_l1", "bol", "bol_l1"])
rows.append({"variant": "two_regressors", "RMSE": r2, "n": n2})

T = pd.DataFrame(rows)
T["vs_base_%"] = 100 * (T.RMSE / base_rmse - 1)
print(T.sort_values("RMSE").to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def dm(e1, e2):
    n = min(len(e1), len(e2))
    dd = e1[-n:] ** 2 - e2[-n:] ** 2
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(n))
    return float(t), float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


gb = panel({"base": series["base"]})
_, _, eb = oos(gb, ["base", "base_l1"])
print("\nDiebold-Mariano against base (p high = tie):")
for k in ("var_by_obs", "saladette", "trimmed", "top35", "equal_mkt", "exacto"):
    gk = panel({k: series[k]})
    _, _, ek = oos(gk, [k, f"{k}_l1"])
    t, pv = dm(ek, eb)
    print(f"  {k:<14} DM t {t:+5.2f}  p {pv:.3f}")
t, pv = dm(e2, eb)
print(f"  {'two_regressors':<14} DM t {t:+5.2f}  p {pv:.3f}")
print(f"\ncorrelation between Saladette and Bola fortnightly changes: "
      f"{series['saladette'].corr(series['bola']):.3f}")
