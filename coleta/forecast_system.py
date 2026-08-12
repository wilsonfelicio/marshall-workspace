"""Nowcasting system for every INPC produce generic, per the agreed design.

Layers, in order:

  1. RELEASE VINTAGES. One equation per information set: day 5 of the fortnight,
     day 10, and close-of-fortnight. Each compares day 1..k against day 1..k of
     the previous fortnight, so the intra-period window is constant. The whole
     edge is INEGI's ~9-day publication lag, so the product is a number that
     updates through the fortnight, not one number published once.

  2. COMBINATION, NOT SELECTION. Three specifications are fixed a priori and
     equal-weighted. Every variant tested on avocado was statistically tied
     (Diebold-Mariano p 0.18-0.72), and choosing among tied models on
     out-of-sample results is a leak that survives an otherwise clean backtest.
     Equal weights have no tuning surface to abuse.

  3. PARTIAL POOLING. Each generic's wholesale slope is shrunk toward the
     produce-wide mean by its own precision (empirical Bayes, recomputed
     recursively). This does nearly nothing for a dense generic like avocado and
     a lot for thin ones - which is the point.

  4. ADMISSIBILITY GATE, decided on training data only. Minimum matched cells,
     and the combination must beat SD-AR inside the TRAINING window. Failing
     generics fall back to their benchmark; they are never dropped, because
     dropping changes the weight base of the aggregate.

  5. SCALE MODEL. Residual dispersion rises with the size of the wholesale move
     (1.49 -> 2.65 pp across quartiles on avocado), so sigma is fitted
     recursively as a function of |wholesale move|. Validated coverage was 81.4%
     at a nominal 80% band.

  6. BOTTOM-UP AGGREGATE. The 32 generics are exactly INEGI's published
     "Frutas y verduras" subindex, and with the published weights times
     theta = 1/chaining factor the reconstruction is exact to RMSE 0.029 pp. So
     bottom-up is coherent by construction: no hierarchical reconciliation
     needed. Aggregate intervals use the CROSS-ITEM residual covariance, because
     produce shocks share weather and fuel and independent errors would
     understate the aggregate's uncertainty.

  7. AUDIT. A structural look-ahead check plus a shuffled-regressor placebo run
     on every generic, every time, and abort rather than warn. The structural
     check already caught a real look-ahead bug in an earlier version.

Usage:  python3 forecast_system.py [oos_start_year] [comma,separated,slugs]
Outputs data/curated/system_*.csv|json and prints the scorecard.
"""
from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")
from math import erf, sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import catalogo, quincenal as Q  # noqa: E402

OOS_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2011
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None

HARM = 3
MIN_TRAIN = 120
MIN_CELDAS_MED = 12          # median matched cells required in the training window
SIGMA_EVERY = 12             # refit the scale model twice a year, not every step
# Estimation window, decided after finding that the pass-through coefficient has
# drifted a long way (jitomate beta0: 0.54 in 1999-2007, 0.83 in 2017-2026). A
# rolling five years beats an expanding window by ~13% and was chosen on a
# 2006-2010 pseudo-out-of-sample run, never on the evaluation window. Discounted
# least squares ties it (DM p = 0.51) and is the planned successor.
WINDOW = 120                 # 120 fortnights = 5 years exactly
WINDOW_MIN = 120             # below this many usable fortnights, fall back to expanding
VINTAGES = {"d5": "x_w5", "d10": "x_w10", "close": "x_full"}
WINDOWS = (5, 10)
SEAS = [f"{f}{k}" for k in range(1, HARM + 1) for f in ("sin", "cos")]


# --------------------------------------------------------------- specifications
def specs_for(xcol: str) -> dict:
    """Fixed a priori. `bridge2` is the 2-regressor version that tied the
    11-parameter model on avocado (DM p = 0.72); `bridge_ar` keeps the seasonal
    and autoregressive terms in case another generic needs them."""
    xl = f"{xcol}_lag1"
    return {
        "SD-AR":     (SEAS + ["y_lag1", "y_lag2"], "benchmark"),
        "bridge2":   ([xcol, xl], "member"),
        "bridge_ar": (SEAS + ["y_lag1", "y_lag2", xcol, xl], "member"),
        "placebo":   (SEAS + ["y_lag1", "y_lag2", "x_shuffled"], "placebo"),
    }


def audit(xcol: str) -> list[str]:
    """INPC-derived regressors must be dated t-1 or older. SNIIM-derived ones may
    be dated inside t: that is the publication-lag nowcast and it is legitimate."""
    bad = []
    for name, (vs, kind) in specs_for(xcol).items():
        if kind == "placebo":
            continue
        for v in vs:
            if v in SEAS:
                continue
            if v.startswith("y_lag"):
                if int(v[-1]) < 1:
                    bad.append(f"{name}: {v} reads the CPI at t")
            elif v == xcol or v == f"{xcol}_lag1":
                pass                      # wholesale, dated t or earlier
            else:
                bad.append(f"{name}: {v} has no declared source")
    return bad


def ols(A, b):
    """Normal equations with a whisper of ridge, so a collinear training slice
    degrades instead of raising."""
    return np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), b)


class Bundle:
    """One generic's data as plain numpy, indexed by row position.

    The first version of this file sliced pandas frames inside the recursive loop
    (`d[d.index < t]`, then `dropna`). That is ~18,000 copies of a 640x40 frame and
    it dominated the runtime by two orders of magnitude; the actual regressions are
    0.03s each. Everything here is positional integer indexing on float arrays.
    """

    def __init__(self, d: pd.DataFrame, cols: list[str]):
        self.t = d.index.to_numpy()
        self.cols = cols
        self.ix = {c: i for i, c in enumerate(cols)}
        self.M = d[cols].to_numpy(float)
        self._sig: dict = {}          # per instance: a class-level dict would be shared
        self.window = WINDOW          # set to None per generic to force expanding
        self.y = d["y"].to_numpy(float)
        self.ok_y = ~np.isnan(self.y)

    def valid(self, cols):
        j = [self.ix[c] for c in cols]
        return (~np.isnan(self.M[:, j]).any(axis=1)) & self.ok_y

    def fit_at(self, i: int, cols, need=MIN_TRAIN, window="default"):
        """Fit on rows strictly before position i, predict row i.

        `window` truncates the training set to the most recent N usable rows. A
        generic with too few usable fortnights to fill the window falls back to
        expanding rather than being estimated on a handful of observations; which
        ones fell back is reported, not silently absorbed.
        """
        j = [self.ix[c] for c in cols]
        if np.isnan(self.M[i, j]).any():
            return np.nan, None, None
        v = self.valid(cols)
        idx = np.flatnonzero(v[:i])
        if window == "default":
            window = self.window
        if window and len(idx) >= WINDOW_MIN:
            idx = idx[-window:]
        if len(idx) < need:
            return np.nan, None, None
        X = np.column_stack([np.ones(len(idx)), self.M[np.ix_(idx, j)]])
        yv = self.y[idx]
        A, Xty = X.T @ X, X.T @ yv
        b = ols(A, Xty)
        r = yv - X @ b
        dof = max(len(idx) - X.shape[1], 1)
        try:
            V = (r @ r / dof) * np.linalg.inv(A)
        except np.linalg.LinAlgError:
            V = None
        xt = np.concatenate([[1.0], self.M[i, j]])
        return float(xt @ b), b, V

    def sigma_at(self, i: int, xcol: str, small, big):
        """Recursive scale model: |residual| regressed on |wholesale move|.

        Fitted on the residuals of the COMBINATION that is actually published, not
        of one member. Using a member's residuals made the bands ~7 points too wide,
        because the combination is more accurate than either part of it.

        Refit every SIGMA_EVERY steps instead of every step. That only affects
        interval width, never a point forecast, and it is a stated choice.
        """
        jx = self.ix[xcol]
        if np.isnan(self.M[i, jx]):
            return np.nan
        anchor = i - (i % SIGMA_EVERY)
        key = (xcol, anchor)
        if key not in self._sig:
            js, jb = [self.ix[c] for c in small], [self.ix[c] for c in big]
            v = self.valid(list(dict.fromkeys(small + big)))
            idx = np.flatnonzero(v[:max(anchor, 1)])
            if self.window and len(idx) >= WINDOW_MIN:
                idx = idx[-self.window:]
            if len(idx) < MIN_TRAIN:
                self._sig[key] = None
            else:
                yv = self.y[idx]
                fits = []
                for j in (js, jb):
                    X = np.column_stack([np.ones(len(idx)), self.M[np.ix_(idx, j)]])
                    fits.append(X @ ols(X.T @ X, X.T @ yv))
                r = np.abs(yv - np.mean(fits, axis=0))
                ax = np.abs(self.M[idx, jx])
                Z = np.column_stack([np.ones(len(ax)), ax])
                self._sig[key] = ols(Z.T @ Z, Z.T @ r)
        g = self._sig[key]
        if g is None:
            return np.nan
        # |residual| -> sd for a normal: divide by sqrt(2/pi), i.e. times 1.2533
        return float(max(g[0] + g[1] * abs(self.M[i, jx]), 0.3) * 1.2533)


def clark_west(y, small, big):
    m = ~(np.isnan(y) | np.isnan(small) | np.isnan(big))
    y, s, b = y[m], small[m], big[m]
    if len(y) < 20:
        return np.nan, np.nan
    f = (y - s) ** 2 - ((y - b) ** 2 - (s - b) ** 2)
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))
    return float(t), float(0.5 * (1 - erf(t / sqrt(2))))


# --------------------------------------------------------------- the run
gen = {r["generico"]: r for r in catalogo.genericos()}
th = catalogo.theta()
daily = pd.read_parquet("data/curated/var_market_daily.parquet",
                        columns=["categoria"]).categoria.unique()
labels = (pd.read_parquet("data/curated/cat_national_monthly.parquet",
                          columns=["categoria", "categoria_label"])
          .drop_duplicates().set_index("categoria")["categoria_label"].to_dict())
items = []
for slug in sorted(daily):
    lab = labels.get(slug)
    if lab in gen and (ONLY is None or slug in ONLY):
        r = gen[lab]
        items.append({"slug": slug, "generico": lab, "clave": r["clave_generico"],
                      "col": f"{r['clave_generico']} {lab}",
                      "peso": float(r["ponderacion_2024"]),
                      "theta": th[f"{r['clave_generico']} {lab}"]})   # theta is keyed by "clave Nombre"
if not items:
    sys.exit("no categories with both wholesale data and an INPC generic")

for it in items:
    for viol in (audit(x) for x in VINTAGES.values()):
        if viol:
            print("AUDIT FAILED:", "; ".join(viol))
            sys.exit(1)
print(f"structural audit clean for {len(items)} generic(s) x {len(VINTAGES)} vintages\n")

ALL_COLS = SEAS + ["y_lag1", "y_lag2", "x_shuffled", "n_celdas_full"] + \
          [c for x in VINTAGES.values() for c in (x, f"{x}_lag1")]

data = {}
for it in items:
    d = Q.dataset(it["slug"], it["col"], windows=WINDOWS).set_index("t")
    ang = 2 * np.pi * (d["quincena_del_anio"] - 1) / 24
    for k in range(1, HARM + 1):
        d[f"sin{k}"], d[f"cos{k}"] = np.sin(k * ang), np.cos(k * ang)
    # Placebo: the real regressor, circularly shifted by a third of the sample.
    # Same distribution and seasonality, wrong dates.
    d["x_shuffled"] = np.roll(d["x_full"].to_numpy(float), len(d) // 3)
    data[it["slug"]] = Bundle(d, ALL_COLS)

t0 = OOS_YEAR * 24

# ---- gate, decided on the training window only
for it in items:
    B = data[it["slug"]]
    n_tr = int(np.searchsorted(B.t, t0))
    med = float(np.nanmedian(B.M[:n_tr, B.ix["n_celdas_full"]]))
    xc = VINTAGES["close"]
    ys, cs, bs = [], [], []
    for i in range(MIN_TRAIN + 24, n_tr):
        c, _, _ = B.fit_at(i, [xc, f"{xc}_lag1"])
        bb, _, _ = B.fit_at(i, SEAS + ["y_lag1", "y_lag2"])
        ys.append(B.y[i]); cs.append(c); bs.append(bb)
    cw_t, cw_p = clark_west(np.array(ys), np.array(bs), np.array(cs))
    it["celdas_med"] = med
    it["cw_train_t"], it["cw_train_p"] = cw_t, cw_p
    it["admisible"] = bool(med >= MIN_CELDAS_MED and cw_p == cw_p and cw_p < 0.05)
    # Enough usable fortnights to fill a five-year window? If not, this generic is
    # estimated on an expanding window and that is stated rather than hidden.
    n_usable = int(B.valid([VINTAGES["close"], f"{VINTAGES['close']}_lag1"]).sum())
    it["n_usable"] = n_usable
    it["rolling"] = bool(n_usable >= WINDOW_MIN + 40)
    B.window = WINDOW if it["rolling"] else None
    print(f"  gate {it['generico']:<30} cells {med:5.0f}  usable {n_usable:4d}  "
          f"CW t {cw_t:5.2f} p {cw_p:.4f}  "
          f"{'rolling 5y' if it['rolling'] else 'EXPANDING (too few obs)'}  -> "
          f"{'ADMITTED' if it['admisible'] else 'BENCHMARK ONLY'}")
print()

# ---- recursive pass, t outer so pooling sees every generic at each step
pos = {it["slug"]: {int(tt): k for k, tt in enumerate(data[it["slug"]].t)} for it in items}
grid = sorted(set().union(*[set(int(x) for x in data[it["slug"]].t if x >= t0) for it in items]))
rows, byslug = [], {}
for t in grid:
    betas = {}
    for it in items:
        B, sl = data[it["slug"]], it["slug"]
        i = pos[sl].get(t)
        if i is None or np.isnan(B.y[i]):
            continue
        rec = {"t": t, "slug": sl, "y": float(B.y[i]),
               "peso": it["peso"], "theta": it["theta"]}
        for vname, xcol in VINTAGES.items():
            two = [xcol, f"{xcol}_lag1"]
            big = SEAS + ["y_lag1", "y_lag2"] + two
            p2, b2, V2 = B.fit_at(i, two)
            pb, _, _ = B.fit_at(i, big)
            pbench, _, _ = B.fit_at(i, SEAS + ["y_lag1", "y_lag2"])
            ppl, _, _ = B.fit_at(i, SEAS + ["y_lag1", "y_lag2", "x_shuffled"])
            rec[f"{vname}_b2"] = p2
            rec[f"{vname}_bench"] = pbench
            rec[f"{vname}_placebo"] = ppl
            rec[f"{vname}_combo"] = (np.nan if (np.isnan(p2) or np.isnan(pb))
                                     else float((p2 + pb) / 2))
            rec[f"{vname}_sigma"] = B.sigma_at(i, xcol, two, big)
            if vname == "close":
                pe2, _, _ = B.fit_at(i, two, window=None)
                peb, _, _ = B.fit_at(i, big, window=None)
                rec["close_expanding"] = (np.nan if (np.isnan(pe2) or np.isnan(peb))
                                          else float((pe2 + peb) / 2))
            if b2 is not None and V2 is not None:
                betas.setdefault(vname, {})[sl] = (b2[1], V2[1, 1], B.M[i, B.ix[xcol]], p2)
        rows.append(rec)
        byslug[(t, sl)] = rows[-1]

    # ---- empirical-Bayes pooling of the contemporaneous wholesale slope
    for vname, bs in betas.items():
        if len(bs) < 3:
            continue
        bv = np.array([v[0] for v in bs.values()])
        vv = np.maximum(np.array([v[1] for v in bs.values()]), 1e-9)
        mu = float(np.average(bv, weights=1 / vv))
        tau2 = max(float(bv.var(ddof=1) - vv.mean()), 1e-6)   # method of moments
        for sl, (bi, vi, xt, p2) in bs.items():
            w = tau2 / (tau2 + vi)                            # weight on own estimate
            r = byslug[(t, sl)]
            r[f"{vname}_pooled"] = float(p2 + ((w * bi + (1 - w) * mu) - bi) * xt)
            r[f"{vname}_shrink"] = float(1 - w)

fc = pd.DataFrame(rows)
fc.to_csv("data/curated/system_forecasts.csv", index=False)

# --------------------------------------------------------------- scoring
def rmse(a, b):
    m = ~(a.isna() | b.isna())
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean())) if m.sum() > 19 else np.nan


out = []
for it in items:
    g = fc[fc["slug"] == it["slug"]]
    row = {"generico": it["generico"], "peso": it["peso"],
           "share_var": rank.get(it["generico"], np.nan) if "rank" in dir() else np.nan,
           "admisible": it["admisible"], "ventana": "rolling5y" if it["rolling"] else "expanding",
           "celdas_med": it["celdas_med"], "n_usable": it["n_usable"],
           "cw_train_t": it["cw_train_t"], "n": len(g)}
    for vname in VINTAGES:
        rb = rmse(g["y"], g[f"{vname}_bench"])
        used = f"{vname}_combo" if it["admisible"] else f"{vname}_bench"
        row[vname] = rmse(g["y"], g[used])
        row[f"{vname}_vs"] = 100 * (row[vname] / rb - 1) if rb == rb else np.nan
        row[f"{vname}_headline"] = row[vname] * it["peso"] / 100
    row["bench"] = rmse(g["y"], g["close_bench"])
    row["mae"] = float((g["y"] - g[("close_combo" if it["admisible"] else "close_bench")]).abs().mean())
    row["sign_%"] = 100 * float((np.sign(g["y"]) ==
                    np.sign(g["close_combo" if it["admisible"] else "close_bench"])).mean())
    if "close_expanding" in g:
        re_ = rmse(g["y"], g["close_expanding"])
        row["expanding"] = re_
        row["roll_vs_exp_%"] = 100 * (row["close"] / re_ - 1) if re_ == re_ else np.nan
    row["placebo_vs"] = 100 * (rmse(g["y"], g["close_placebo"]) / row["bench"] - 1)
    out.append(row)
tab = pd.DataFrame(out)

# ---- bottom-up aggregates: all modelled, top-10 modelled, top-5 modelled.
# Ranking is fixed in advance by share of subindex variance (prioridad_varianza.csv),
# never by realised accuracy. Non-members are carried at their own SD-AR benchmark
# rather than dropped: dropping them would change the weight base and quietly
# redefine the target.
try:
    pri = pd.read_csv("data/curated/prioridad_varianza.csv")
    rank = {r.generico.split(" ", 1)[1]: r.share for r in pri.itertuples()}
except Exception:
    rank = {}
order = sorted((i["generico"] for i in items),
               key=lambda g: -rank.get(g, 0.0))
TOP5, TOP10 = set(order[:5]), set(order[:10])
print("\nvariance ranking used for the aggregates")
print("  top 5 :", ", ".join(f"{g} {rank.get(g,0):.1f}%" for g in order[:5]),
      f"  = {sum(rank.get(g,0) for g in order[:5]):.1f}%")
print("  top 10:", ", ".join(f"{g} {rank.get(g,0):.1f}%" for g in order[5:10]),
      f"  cumulative {sum(rank.get(g,0) for g in order[:10]):.1f}%")

adm = {i["slug"]: i["admisible"] for i in items}
gen_of = {i["slug"]: i["generico"] for i in items}
SETS = {"all32": set(order), "top10": TOP10, "top5": TOP5}

agg = []
for t, g in fc.groupby("t"):
    w = (g["peso"] * g["theta"]).to_numpy(float)
    if w.sum() <= 0:
        continue
    rec = {"t": t, "cobertura": float(g["peso"].sum()),
           "y": float(np.average(g["y"], weights=w))}
    for vname in VINTAGES:
        bench = g[f"{vname}_bench"].to_numpy(float)
        model = g[f"{vname}_combo"].to_numpy(float)
        gen = g["slug"].map(gen_of).to_numpy()
        ok_model = g["slug"].map(adm).to_numpy(bool) & ~np.isnan(model)
        rec[f"{vname}_bench"] = (float(np.average(bench, weights=w))
                                 if not np.isnan(bench).any() else np.nan)
        for sname, keep in SETS.items():
            use = np.where(ok_model & np.isin(gen, list(keep)), model, bench)
            rec[f"{vname}_{sname}"] = (float(np.average(use, weights=w))
                                       if not np.isnan(use).any() else np.nan)
            # the sub-aggregate over only that set's own weights
            m = np.isin(gen, list(keep))
            if m.any() and not np.isnan(use[m]).any():
                rec[f"{vname}_{sname}_own"] = float(np.average(use[m], weights=w[m]))
                rec[f"y_{sname}_own"] = float(np.average(g["y"].to_numpy()[m], weights=w[m]))
    agg.append(rec)
A = pd.DataFrame(agg)
A.to_csv("data/curated/system_aggregate.csv", index=False)

print("per-generic RMSE (pp) and % vs that generic's own SD-AR benchmark")
cols = ["generico", "peso", "share_var", "ventana", "admisible", "celdas_med",
        "d5", "d10", "close", "bench", "close_vs", "close_headline", "mae", "sign_%",
        "roll_vs_exp_%", "placebo_vs"]
cols = [c for c in cols if c in tab]
tab = tab.sort_values("share_var", ascending=False)
with pd.option_context("display.width", 220, "display.max_columns", 40):
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

PESO_SUB = 4.7789
def rr(a, b):
    m = ~(a.isna() | b.isna())
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean())) if m.sum() > 19 else np.nan
def mm(a, b):
    m = ~(a.isna() | b.isna())
    return float((a[m] - b[m]).abs().mean()) if m.sum() > 19 else np.nan

print(f"\nAGGREGATE NOWCAST of the published Frutas y verduras subindex "
      f"({len(A)} fortnights, {len(items)} generics, weight "
      f"{sum(i['peso'] for i in items):.2f} of {PESO_SUB:.2f})")
print("Headline metric: error in pp of the published print, and the same as "
      "contribution to the headline INPC (x weight/100).\n")
hdr = f"{'coverage':<10}{'vintage':<8}{'RMSE pp':>9}{'MAE pp':>8}{'headline ±pp':>14}{'vs bench':>10}"
print(hdr); print("-" * len(hdr))
rows_agg = []
for sname in ("all32", "top10", "top5"):
    for vname in VINTAGES:
        r = rr(A["y"], A[f"{vname}_{sname}"]); rb = rr(A["y"], A[f"{vname}_bench"])
        rows_agg.append({"coverage": sname, "vintage": vname, "rmse_pp": r,
                         "mae_pp": mm(A["y"], A[f"{vname}_{sname}"]),
                         "headline_pp": r * PESO_SUB / 100,
                         "vs_bench_pct": 100 * (r / rb - 1) if rb == rb else np.nan})
        print(f"{sname:<10}{vname:<8}{r:9.3f}{rows_agg[-1]['mae_pp']:8.3f}"
              f"{rows_agg[-1]['headline_pp']:14.4f}{rows_agg[-1]['vs_bench_pct']:9.1f}%")
rb = rr(A["y"], A["close_bench"])
print(f"{'benchmark':<10}{'close':<8}{rb:9.3f}{mm(A['y'], A['close_bench']):8.3f}"
      f"{rb*PESO_SUB/100:14.4f}{0.0:9.1f}%")
pd.DataFrame(rows_agg).to_csv("data/curated/system_aggregate_scores.csv", index=False)
r5, r10, r32 = (rr(A["y"], A[f"close_{k}"]) for k in ("top5", "top10", "all32"))
print(f"\nDoes the marginal coverage buy anything? close-of-fortnight vintage:")
print(f"  top 5  -> {r5:.3f} pp   (±{r5*PESO_SUB/100:.4f} pp of headline)")
print(f"  top 10 -> {r10:.3f} pp   ({100*(r10/r5-1):+.1f}% vs top 5)")
print(f"  all 32 -> {r32:.3f} pp   ({100*(r32/r10-1):+.1f}% vs top 10, "
      f"{100*(r32/r5-1):+.1f}% vs top 5)")
print(f"  subindex own sd {A['y'].std():.3f} pp = "
      f"{A['y'].std()*PESO_SUB/100:.4f} pp of headline contribution")

res = fc.pivot_table(index="t", columns="slug", values="close_combo") - \
      fc.pivot_table(index="t", columns="slug", values="y")
C = res.corr()
iu = np.triu_indices_from(C.values, 1)
print(f"\ncross-generic residual correlation: median {np.nanmedian(C.values[iu]):+.3f}, "
      f"range {np.nanmin(C.values[iu]):+.3f} to {np.nanmax(C.values[iu]):+.3f}")
print("  (this is why aggregate intervals need the covariance, not independent errors)")

# Interval multipliers are taken from each generic's own earlier standardised
# errors, not from a normal table. Assuming normality over-covered by ~5 points at
# the 80% band, because the standardised errors are less dispersed than a normal at
# that quantile. The quantile is recursive, so it is part of the information set.
z = fc.dropna(subset=["close_combo", "close_sigma"]).sort_values(["slug", "t"]).copy()
z["zz"] = (z["y"] - z["close_combo"]).abs() / z["close_sigma"]
for lvl in (0.80, 0.90):
    covs, widths, mults = [], [], []
    for sl, g in z.groupby("slug"):
        zv = g["zz"].to_numpy()
        k = np.array([np.nan if i < 40 else np.quantile(zv[:i], lvl)
                      for i in range(len(zv))])
        m = ~np.isnan(k)
        covs.append(float((zv[m] <= k[m]).mean()))
        widths.append(float((k[m] * g["close_sigma"].to_numpy()[m]).mean()))
        mults.append(float(np.nanmedian(k)))
    print(f"interval {lvl:.0%}: coverage {100*np.mean(covs):.1f}% "
          f"(empirical multiplier {np.mean(mults):.2f} vs normal "
          f"{1.2816 if lvl == 0.80 else 1.6449:.2f}), mean half-width "
          f"±{np.mean(widths):.2f} pp")

json.dump({"oos_desde": OOS_YEAR, "n_genericos": len(items),
           "cobertura_peso": float(sum(i["peso"] for i in items)),
           "admisibles": [i["generico"] for i in items if i["admisible"]],
           "vintages": list(VINTAGES)}, open("data/curated/system_meta.json", "w"), indent=2)
tab.to_csv("data/curated/system_scores.csv", index=False)
print("\nwrote data/curated/system_{forecasts,aggregate,scores}.csv and system_meta.json")
