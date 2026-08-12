"""One place where every number quoted in the deck and on the charts is computed.

Written after a QA pass found the same statistic printed three ways: the jitomate
benchmark RMSE appeared as 12.30 on one slide and 12.33 on another, the sample as
370 on one and 374 on another, and the correct-sign rate as 90% and 91%. The cause
was that different scripts applied different missing-value masks to the same data:
one dropped the four fortnights where the wholesale model has no fit, the other let
pandas skip them column by column, so the benchmark was scored on 374 periods and
the model on 370.

Rule adopted here: any two forecasts that are compared are scored on the fortnights
where BOTH exist, and the sample size is reported alongside. Every consumer reads
data/curated/facts.json instead of recomputing.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

PESO_JIT = 0.79014
PESO_SUB = 4.7789


def _pair(y, a, b):
    """Score two forecasts on the fortnights where both are available."""
    m = ~(y.isna() | a.isna() | b.isna())
    nan = float("nan")
    if int(m.sum()) < 3:                 # gated generics have no nowcast at all
        return {"n": int(m.sum()), "rmse_a": nan, "rmse_b": nan, "mae_a": nan,
                "mae_b": nan, "sign_a": nan, "sign_b": nan, "r2_a": nan, "r2_b": nan,
                "gain_pct": nan, "dm_t": nan}
    y, a, b = y[m], a[m], b[m]
    r = lambda e: float(np.sqrt((e ** 2).mean()))
    sign = lambda f: 100 * float((np.sign(y) == np.sign(f)).mean())
    r2 = lambda f: 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    d = (y - b) ** 2 - (y - a) ** 2                      # positive = a beats b
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    return {
        "n": int(m.sum()),
        "rmse_a": r(y - a), "rmse_b": r(y - b),
        "mae_a": float((y - a).abs().mean()), "mae_b": float((y - b).abs().mean()),
        "sign_a": sign(a), "sign_b": sign(b),
        "r2_a": float(r2(a)), "r2_b": float(r2(b)),
        "gain_pct": 100 * (1 - r(y - a) / r(y - b)),
        "dm_t": float(d.mean() / se) if se > 0 else nan,
    }


F: dict = {}

# ---------------------------------------------------------------- jitomate
J = pd.read_csv("data/curated/jitomate_system.csv")
j = _pair(J.y, J.fit, J.bench)
F["jit"] = {**j, "headline_a": j["rmse_a"] * PESO_JIT / 100,
            "headline_b": j["rmse_b"] * PESO_JIT / 100, "peso": PESO_JIT}
m = ~(J.y.isna() | J.fit.isna() | J.sigma.isna())
zz = ((J.y - J.fit).abs() / J.sigma)[m].to_numpy()
k80 = np.array([np.nan if i < 40 else np.quantile(zz[:i], 0.80) for i in range(len(zz))])
g = ~np.isnan(k80)
F["jit"]["cov80"] = 100 * float((zz[g] <= k80[g]).mean())

# ---------------------------------------------------------------- aggregate
A = pd.read_csv("data/curated/system_aggregate.csv")
F["agg"] = {"n": int(len(A)), "sd": float(A.y.std()),
            "sd_headline": float(A.y.std() * PESO_SUB / 100), "peso": PESO_SUB}
for k in ("all32", "top10", "top5"):
    p = _pair(A.y, A[f"close_{k}"], A.close_bench)
    F["agg"][k] = {**p, "headline": p["rmse_a"] * PESO_SUB / 100}
F["agg"]["bench"] = {"rmse": F["agg"]["all32"]["rmse_b"],
                     "headline": F["agg"]["all32"]["rmse_b"] * PESO_SUB / 100,
                     "sign": F["agg"]["all32"]["sign_b"],
                     "r2": F["agg"]["all32"]["r2_b"],
                     "mae": F["agg"]["all32"]["mae_b"]}
for v in ("d5", "d10"):
    p = _pair(A.y, A[f"{v}_all32"], A.close_bench)
    F["agg"][f"{v}_all32"] = {**p, "headline": p["rmse_a"] * PESO_SUB / 100}

# ---------------------------------------------------------------- variance shares
pri = pd.read_csv("data/curated/prioridad_varianza.csv")
pri["name"] = pri.generico.str.split(" ", n=1).str[1]
pri = pri.sort_values("share", ascending=False).reset_index(drop=True)
sh = dict(zip(pri.name, pri.share.astype(float)))
F["share"] = {k: round(v, 1) for k, v in sh.items()}
# Sum the shares AFTER rounding to the one decimal the panels print, otherwise the
# labels on the grid add to 2.7% under a headline of 3.0% and a reader can see it.
_r = pri.share.astype(float).round(1)
cum = lambda i, j: float(_r.iloc[i:j].sum())
F["split"] = {"jitomate": round(sh["Jitomate"], 1), "top5": round(cum(0, 5), 1),  # noqa
              "top10": round(cum(0, 10), 1), "rest22": round(cum(10, 32), 1),
              "first16": round(cum(0, 16), 1), "last16": round(cum(16, 32), 1)}

T = pd.read_csv("data/curated/system_scores.csv")
F["gated"] = sorted(T.loc[~T.admisible, "generico"].tolist())
F["gated_share"] = round(sum(round(sh.get(g, 0.0), 1) for g in F["gated"]), 1)
F["n_admitted"] = int(T.admisible.sum())
F["n_generics"] = int(len(T))
# The per-generic table is rebuilt here on paired masks rather than copied from
# system_scores.csv, so the panel titles in the grids, the rows of the deck's table and
# the jitomate hero chart cannot disagree about the same generic's benchmark.
FC = pd.read_csv("data/curated/system_forecasts.csv")
LAB = (pd.read_parquet("data/curated/cat_national_monthly.parquet",
                       columns=["categoria", "categoria_label"])
       .drop_duplicates().set_index("categoria")["categoria_label"].to_dict())
FC["generico"] = FC.slug.map(LAB)
adm = dict(zip(T.generico, T.admisible))
peso = dict(zip(T.generico, T.peso))
rows = []
for gname, g in FC.groupby("generico"):
    g = g.sort_values("t")
    pc = _pair(g.y, g.close_combo, g.close_bench)
    p10 = _pair(g.y, g.d10_combo, g.close_bench)
    if not adm.get(gname, False):        # gated: it has no publishable nowcast
        nan = float("nan")
        mb = ~(g.y.isna() | g.close_bench.isna())
        pc = {**pc, "n": int(mb.sum()), "rmse_a": nan, "sign_a": nan, "gain_pct": nan,
              "rmse_b": float(np.sqrt(((g.y[mb] - g.close_bench[mb]) ** 2).mean())),
              "sign_b": 100 * float((np.sign(g.y[mb]) == np.sign(g.close_bench[mb])).mean())}
    rows.append({"generico": gname, "peso": float(peso.get(gname, float("nan"))),
                 "share": round(sh.get(gname, 0.0), 1),
                 "share_exact": float(sh.get(gname, 0.0)), "n": pc["n"],
                 "close": pc["rmse_a"], "bench": pc["rmse_b"],
                 "d10": p10["rmse_a"] if adm.get(gname, False) else float("nan"),
                 "gain_pct": pc["gain_pct"], "sign": pc["sign_a"],
                 "sign_bench": pc["sign_b"], "admisible": bool(adm.get(gname, False))})
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


F["table"] = sorted(rows, key=lambda r: (-r["share_exact"], r["generico"]))
F["n_range"] = sorted({r["n"] for r in rows if r["admisible"]})
F = _clean(F)

if __name__ == "__main__":
    # NaN is not JSON; the consumers are JavaScript, which reads null as absent
    json.dump(F, open("data/curated/facts.json", "w"), indent=1, ensure_ascii=False,
              allow_nan=False, default=str)
    j, a = F["jit"], F["agg"]
    print(f"jitomate  n={j['n']}  model {j['rmse_a']:.2f}  bench {j['rmse_b']:.2f}  "
          f"gain {j['gain_pct']:.0f}%  sign {j['sign_b']:.0f}->{j['sign_a']:.0f}%  "
          f"R2 {j['r2_b']:.2f}->{j['r2_a']:.2f}  DM t {j['dm_t']:.1f}  cov80 {j['cov80']:.0f}%")
    print(f"aggregate n={a['n']}  all32 {a['all32']['rmse_a']:.3f}  "
          f"top10 {a['top10']['rmse_a']:.3f}  top5 {a['top5']['rmse_a']:.3f}  "
          f"bench {a['bench']['rmse']:.3f}  sign {a['bench']['sign']:.0f}->"
          f"{a['all32']['sign_a']:.0f}%  R2 {a['bench']['r2']:.2f}->{a['all32']['r2_a']:.2f}")
    print("splits", F["split"], "gated", F["gated"], F["gated_share"], "%")
