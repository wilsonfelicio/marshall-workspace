"""Does city-level bottom-up nowcasting beat the national index? Jitomate.

Two experiments, because INEGI publishes generic indices by city MONTHLY only —
there is no fortnightly city series, so the true bottom-up test cannot be run at
the frequency the headline model uses.

  A. MONTHLY, the real thing: nowcast each city's own published jitomate CPI from
     the wholesale markets inside that city, then aggregate the city nowcasts with
     INPC city weights. Compared against a national-index model and a CPI-only
     benchmark on the same months.

  B. FORTNIGHTLY, the aggregation-only question: keep the national CPI target, but
     build the wholesale regressor in two stages — markets to city, then cities to
     national — instead of one weighted mean over markets. 62% of the mapped INPC
     weight sits in cities holding more than one market, so this is a real change
     even though it uses no city-level CPI.

Everything is out-of-sample, refit every period on earlier data only, and the
comparisons are on identical periods.

  python3 city_bottomup.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import catalogo as C  # noqa: E402
from inpc import quincenal as Q  # noqa: E402

Q.DAILY = pathlib.Path("/root/jit/var_market_daily.parquet")
Q.CACHE_DIR = pathlib.Path("/root/jit/cache")
CITY = pathlib.Path("data/inpc/ciudad")

HARM = 3
WIN_M, WIN_Q = 60, 120          # months / fortnights in the rolling window
MIN_TRAIN_M, MIN_TRAIN_Q = 60, 120
OOS_M, OOS_Q = "2011-01", 2011 * 24
MIN_CELDAS_CITY = 2             # a city index needs at least two matched cells
PESO_JIT = 0.79014


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    for junk in (", gro.", ", ags.", ", mex.", ", camp.", ", q. roo.", ", ver.",
                 ", son.", ", pue.", ", zac.", ", gto.", ", jal.", ", n.l.",
                 ", coah.", ", chih.", ", tamps.", ", sin.", ", son", ", b.c.",
                 ", b.c.s.", ", chis.", ", dgo.", ", hgo.", ", mich.", ", mor.",
                 ", nay.", ", oax.", ", qro.", ", s.l.p.", ", tab.", ", tlax.",
                 ", yuc.", ", col.", ", slp.", ", edomex"):
        s = s.lower().replace(junk, "")
    s = s.lower().replace("area metropolitana de la cd. de mexico", "cd de mexico")
    s = s.replace("7. ", "").replace("cd.", "cd").replace(".", "").strip()
    return " ".join(s.split())


# ------------------------------------------------------------------ mappings
cat = pd.DataFrame(C.ciudades())
cat["k"] = cat.ciudad_inegi_anexoF.map(fold)
w_city = dict(zip(cat.k, cat.ponderador_2024.astype(float)))
clave = dict(zip(cat.k, cat.clave_ciudad_preciospromedio))

pm = pd.read_parquet("data/curated/pesos_mercado.parquet")
pm["k"] = pm.ciudad_inpc.map(fold)
mkt_city = dict(zip(pm.destino, pm.k))
mkt_exact = dict(zip(pm.destino, pm.metodo == "exacto"))

cpi_c = pd.read_parquet(CITY / "jitomate_ciudad_mensual.parquet")
cpi_c.columns = [fold(c) for c in cpi_c.columns]
cpi_c = cpi_c.loc[:, ~cpi_c.columns.duplicated()]

missing_w = [c for c in cpi_c.columns if c not in w_city]
print(f"cities: {len(cpi_c.columns)} with a published CPI, "
      f"{len(set(mkt_city.values()) & set(cpi_c.columns))} of them with SNIIM markets, "
      f"{len(missing_w)} unmatched to a weight {missing_w if missing_w else ''}")
cov = sum(w_city.get(c, 0) for c in set(mkt_city.values()) & set(cpi_c.columns))
print(f"INPC weight covered by cities that have markets: {cov:.1f} of "
      f"{sum(w_city.values()):.1f} = {100*cov/sum(w_city.values()):.0f}%\n")

# ------------------------------------------------------------------ wholesale cells
d = pd.read_parquet(Q.DAILY)
d["fecha"] = pd.to_datetime(d["fecha"])
d["t"] = Q.qindex(d["fecha"])
d["mes"] = d.fecha.dt.to_period("M")
d["k"] = d.destino.map(mkt_city)
pesos = pd.read_parquet(Q.PESOS)[["destino", "peso_inpc"]].copy()
pesos["peso"] = pd.to_numeric(pesos.peso_inpc, errors="coerce").astype(float)


def cells(period_col: str) -> pd.DataFrame:
    """Cell = variety x market geometric mean price inside the period."""
    x = d[d.precio_geo > 0].assign(_lp=lambda z: np.log(z.precio_geo.to_numpy(float)))
    g = (x.groupby([period_col, "producto_id", "destino"], sort=False)
         .agg(lp=("_lp", "mean"), n_dias=("fecha", "nunique")).reset_index())
    need = max(1, int(round(float(g.n_dias.median()) / 2)) or 1)
    return g[g.n_dias >= need].rename(columns={period_col: "p"})


def links_by_city(cl: pd.DataFrame, min_cells: int) -> pd.DataFrame:
    """Matched-cell log link per city per period: mean over cells inside the city.

    Same estimator as the national index, applied within a city instead of across
    all markets. A city keeps its own chain break bookkeeping.
    """
    cl = cl.copy()
    cl["k"] = cl.destino.map(mkt_city)
    cl = cl[cl.k.notna()]
    out = []
    for k, g in cl.groupby("k"):
        by_p = {p: gg.set_index(["producto_id", "destino"])["lp"]
                for p, gg in g.groupby("p")}
        ps, prev = sorted(by_p), None
        for p in ps:
            a = by_p[p]
            dln, n = np.nan, 0
            if prev is not None:
                b = by_p[prev]
                common = a.index.intersection(b.index)
                n = len(common)
                if n >= min_cells:
                    dln = float((a.loc[common] - b.loc[common]).mean())
            if len(a) >= min_cells:
                prev = p
            out.append({"k": k, "p": p, "dln": dln, "n_celdas": n})
    return pd.DataFrame(out)


def national_from_cities(lk: pd.DataFrame) -> pd.Series:
    """Two-stage: cities to national, weighted by INPC city weight."""
    lk = lk.dropna(subset=["dln"]).copy()
    lk["w"] = lk.k.map(w_city).fillna(0.0)
    lk = lk[lk.w > 0]
    return (lk.assign(x=lk.dln * lk.w).groupby("p")
            .apply(lambda g: g.x.sum() / g.w.sum()))


def national_one_stage(cl: pd.DataFrame, min_cells: int) -> pd.Series:
    """The current estimator: one weighted mean over markets, no city stage."""
    by_p = {p: g.set_index(["producto_id", "destino"])["lp"]
            for p, g in cl.groupby("p")}
    w = pesos.set_index("destino")["peso"]
    ps, prev, out = sorted(by_p), None, {}
    for p in ps:
        a = by_p[p]
        if prev is not None:
            b = by_p[prev]
            common = a.index.intersection(b.index)
            if len(common) >= 5:
                r = pd.DataFrame({"dln": (a.loc[common] - b.loc[common]).values},
                                 index=common).reset_index()
                pm_ = r.groupby("destino").dln.mean()
                pw = w.reindex(pm_.index).fillna(0.0).to_numpy(float)
                if pw.sum() <= 0:
                    pw = np.ones(len(pm_))
                out[p] = float(np.average(pm_.to_numpy(float), weights=pw))
        if len(a) >= 5:
            prev = p
    return pd.Series(out)


# ------------------------------------------------------------------ estimation
def design(y: pd.Series, x: pd.Series, per_year: int) -> pd.DataFrame:
    f = pd.DataFrame({"y": y}).join(pd.DataFrame({"x": x}), how="outer").sort_index()
    f["y_lag1"], f["y_lag2"] = f.y.shift(1), f.y.shift(2)
    f["x_lag1"] = f.x.shift(1)
    ang = 2 * np.pi * (np.arange(len(f)) % per_year) / per_year
    return f


def add_seas(f: pd.DataFrame, season_idx: np.ndarray, per_year: int) -> list[str]:
    ang = 2 * np.pi * season_idx / per_year
    cols = []
    for k in range(1, HARM + 1):
        f[f"sin{k}"], f[f"cos{k}"] = np.sin(k * ang), np.cos(k * ang)
        cols += [f"sin{k}", f"cos{k}"]
    return cols


def oos(f: pd.DataFrame, cols: list[str], t0: int, win: int, min_train: int
        ) -> pd.Series:
    """One-step recursive out-of-sample fit, earlier data only."""
    A = f[cols].to_numpy(float)
    yv = f.y.to_numpy(float)
    ok = ~np.isnan(A).any(axis=1) & ~np.isnan(yv)
    pred = np.full(len(f), np.nan)
    for i in range(t0, len(f)):
        if not (~np.isnan(A[i]).any()):
            continue
        idx = np.flatnonzero(ok[:i])
        if len(idx) < min_train:
            continue
        idx = idx[-win:]
        X = np.column_stack([np.ones(len(idx)), A[idx]])
        try:
            b = np.linalg.solve(X.T @ X + 1e-9 * np.eye(X.shape[1]), X.T @ yv[idx])
        except np.linalg.LinAlgError:
            continue
        pred[i] = float(np.concatenate([[1.0], A[i]]) @ b)
    return pd.Series(pred, index=f.index)


R = lambda a, b: float(np.sqrt(((a - b).dropna() ** 2).mean()))


def score(y, cands: dict, label: str):
    # Align first: the two regressors do not cover identical periods, so `design`
    # returns frames on different union indexes and a boolean built on one of them
    # cannot index the other.
    idx = y.index
    for v in cands.values():
        idx = idx.intersection(v.index)
    y = y.reindex(idx)
    cands = {k: v.reindex(idx) for k, v in cands.items()}
    m = y.notna()
    for v in cands.values():
        m &= v.notna()
    y2 = y[m]
    print(f"\n{label}  n = {int(m.sum())}")
    hdr = f"  {'model':<34}{'RMSE pp':>9}{'MAE':>8}{'sign':>7}{'R2 oos':>9}{'vs base':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    base = None
    for nm, v in cands.items():
        v2 = v[m]
        r = R(y2, v2)
        base = r if base is None else base
        print(f"  {nm:<34}{r:9.3f}{float((y2-v2).abs().mean()):8.3f}"
              f"{100*float((np.sign(y2)==np.sign(v2)).mean()):6.0f}%"
              f"{1-((y2-v2)**2).sum()/((y2-y2.mean())**2).sum():9.3f}"
              f"{100*(r/base-1):8.1f}%")
    return {nm: R(y2, v[m]) for nm, v in cands.items()}, int(m.sum()), y2


def dm(y, a, b):
    """Diebold-Mariano on squared errors; positive t favours a."""
    idx = y.index.intersection(a.index).intersection(b.index)
    y, a, b = y.reindex(idx), a.reindex(idx), b.reindex(idx)
    m = y.notna() & a.notna() & b.notna()
    dd = ((y[m] - b[m]) ** 2 - (y[m] - a[m]) ** 2)
    return float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))))


# ================================================================== A. MONTHLY
print("=" * 78)
print("A. MONTHLY: city-level bottom-up against a national-index model")
print("=" * 78)

clm = cells("mes")
lkm = links_by_city(clm, MIN_CELDAS_CITY)
nat_two_m = national_from_cities(lkm)
nat_one_m = national_one_stage(clm, 5)

# the published national monthly index is the target for every model here
gen_m = pd.read_parquet("data/inpc/inpc_genericos_mensual.parquet")
col = [c for c in gen_m.columns if "Jitomate" in c][0]
gen_m["p"] = pd.PeriodIndex(
    pd.to_datetime(dict(year=gen_m.anio, month=gen_m.mes, day=1)), freq="M")
nat_cpi = gen_m.set_index("p")[col].astype(float).dropna()
y_nat = 100 * np.log(nat_cpi).diff()

cities = sorted(set(lkm.k.dropna()) & set(cpi_c.columns))
print(f"modelling {len(cities)} cities that have both a published CPI and markets")

pred_city, y_city = {}, {}
for k in cities:
    yk = 100 * np.log(cpi_c[k].astype(float)).diff()
    xk = lkm[(lkm.k == k)].set_index("p").dln.dropna()
    f = design(yk, xk, 12)
    seas = add_seas(f, np.array([p.month - 1 for p in f.index]), 12)
    t0 = int(np.searchsorted(f.index, pd.Period(OOS_M, freq="M")))
    cols_full = seas + ["y_lag1", "y_lag2", "x", "x_lag1"]
    cols_two = ["x", "x_lag1"]
    p_full = oos(f, cols_full, t0, WIN_M, MIN_TRAIN_M)
    p_two = oos(f, seas + cols_two, t0, WIN_M, MIN_TRAIN_M)
    pred_city[k] = (p_full + p_two) / 2          # same equal-weight combination
    y_city[k] = f.y

P = pd.DataFrame(pred_city)
W = pd.Series({k: w_city.get(k, 0.0) for k in P.columns})
# ragged edge: renormalise over the cities available in that month
av = P.notna()
wm = av.mul(W, axis=1)
bu = (P.fillna(0.0).mul(W, axis=1).sum(axis=1) / wm.sum(axis=1)).where(av.any(axis=1))
cover = wm.sum(axis=1) / sum(w_city.values())
bu = bu.where(cover >= 0.50)
print(f"bottom-up aggregate: median city-weight coverage "
      f"{100*cover.median():.0f}%, months kept {int(bu.notna().sum())}")

f_nat = design(y_nat, nat_one_m, 12)
seas = add_seas(f_nat, np.array([p.month - 1 for p in f_nat.index]), 12)
t0 = int(np.searchsorted(f_nat.index, pd.Period(OOS_M, freq="M")))
nat_full = oos(f_nat, seas + ["y_lag1", "y_lag2", "x", "x_lag1"], t0, WIN_M, MIN_TRAIN_M)
nat_two_sp = oos(f_nat, seas + ["x", "x_lag1"], t0, WIN_M, MIN_TRAIN_M)
nat_model = (nat_full + nat_two_sp) / 2
bench_m = oos(f_nat, seas + ["y_lag1", "y_lag2"], t0, WIN_M, MIN_TRAIN_M)

f_nat2 = design(y_nat, nat_two_m, 12)
add_seas(f_nat2, np.array([p.month - 1 for p in f_nat2.index]), 12)
nat2_model = ((oos(f_nat2, seas + ["y_lag1", "y_lag2", "x", "x_lag1"], t0, WIN_M, MIN_TRAIN_M)
               + oos(f_nat2, seas + ["x", "x_lag1"], t0, WIN_M, MIN_TRAIN_M)) / 2)

resA, nA, yA = score(y_nat, {
    "CPI-only benchmark": bench_m,
    "national index, one-stage W": nat_model,
    "national index, two-stage W": nat2_model,
    "city bottom-up (own CPI each)": bu,
    "bottom-up + national, 50/50": (bu + nat_model) / 2,
}, "monthly, target = published national jitomate CPI")
print(f"\n  DM t, bottom-up vs national one-stage: "
      f"{dm(y_nat, bu, nat_model):+.2f}   (positive favours bottom-up)")
print(f"  DM t, 50/50 blend vs national one-stage: "
      f"{dm(y_nat, (bu+nat_model)/2, nat_model):+.2f}")

per_city = []
for k in cities:
    yk, pk = y_city[k], pred_city[k]
    m = yk.notna() & pk.notna() & (yk.index >= pd.Period(OOS_M, freq="M"))
    if m.sum() < 24:
        continue
    per_city.append({"ciudad": k, "peso": w_city.get(k, 0.0), "n": int(m.sum()),
                     "rmse": R(yk[m], pk[m]), "sd_y": float(yk[m].std()),
                     "exacto": all(mkt_exact.get(mk, False)
                                   for mk, kk in mkt_city.items() if kk == k)})
pc = pd.DataFrame(per_city).sort_values("peso", ascending=False)
pc["ratio"] = pc.rmse / pc.sd_y
print(f"\n  per-city fit, {len(pc)} cities (RMSE / sd of that city's own CPI change):")
print("   " + pc.head(12).to_string(index=False,
      formatters={"peso": "{:.2f}".format, "rmse": "{:.2f}".format,
                  "sd_y": "{:.2f}".format, "ratio": "{:.2f}".format}))
pc.to_csv("data/curated/city_bottomup_percity.csv", index=False)
print(f"\n  weighted mean ratio {np.average(pc.ratio, weights=pc.peso):.3f}; "
      f"national model's own ratio "
      f"{R(yA, nat_model[yA.index])/float(yA.std()):.3f}")


# ================================================================== B. FORTNIGHTLY
print("\n" + "=" * 78)
print("B. FORTNIGHTLY: two-stage geographic aggregation of the wholesale regressor")
print("=" * 78)

clq = cells("t")
lkq = links_by_city(clq, MIN_CELDAS_CITY)
x_two = national_from_cities(lkq)
x_one = national_one_stage(clq, 5)

dq = Q.dataset("jitomate", "070 Jitomate", windows=(5, 10)).set_index("t")
y_q = dq.y
print(f"periods: one-stage {len(x_one)}, two-stage {len(x_two)}, "
      f"corr of the two regressors {x_one.corr(x_two):.4f}")


def run_q(x):
    f = design(y_q, x, 24)
    seas = add_seas(f, np.array([(t % 24) for t in f.index]), 24)
    t0 = int(np.searchsorted(f.index.to_numpy(), OOS_Q))
    a = oos(f, seas + ["y_lag1", "y_lag2", "x", "x_lag1"], t0, WIN_Q, MIN_TRAIN_Q)
    b = oos(f, seas + ["x", "x_lag1"], t0, WIN_Q, MIN_TRAIN_Q)
    return (a + b) / 2, oos(f, seas + ["y_lag1", "y_lag2"], t0, WIN_Q, MIN_TRAIN_Q)


m_one, b_one = run_q(x_one)
m_two, _ = run_q(x_two)
resB, nB, yB = score(y_q, {
    "CPI-only benchmark": b_one,
    "one-stage W (current model)": m_one,
    "two-stage W (markets->city->nat)": m_two,
    "average of the two": (m_one + m_two) / 2,
}, "fortnightly, target = published national jitomate CPI")
print(f"\n  DM t, two-stage vs one-stage: {dm(y_q, m_two, m_one):+.2f}"
      f"   (positive favours two-stage)")
print(f"  DM t, average vs one-stage:   {dm(y_q, (m_one+m_two)/2, m_one):+.2f}")
print(f"\n  headline: one-stage +-{resB['one-stage W (current model)']*PESO_JIT/100:.4f} pp, "
      f"two-stage +-{resB['two-stage W (markets->city->nat)']*PESO_JIT/100:.4f} pp")

json.dump({"monthly": resA, "monthly_n": nA, "fortnightly": resB, "fortnightly_n": nB},
          open("data/curated/city_bottomup_scores.json", "w"), indent=1)
print("\ndata/curated/city_bottomup_scores.json")
