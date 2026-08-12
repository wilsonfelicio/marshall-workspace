"""Bottom-up aggregate nowcasts of the produce subindex, with a ragged edge.

Rebuilt as a separate step because the first version required all 32 generics to be
present in the same fortnight, which silently threw away 170 of 374 periods: the
published series for Otras verduras y legumbres is missing from 2011 to mid-2023
(206 gaps) and a few others are missing across 2020. The aggregate was therefore
being measured on a biased subsample.

Correct treatment: at each fortnight, use the generics that are actually available
and renormalise the published weights over them, applying the SAME available set to
the target and to every forecast so the comparison is like for like. Fortnights
where the available weight falls below MIN_COVER of the total are dropped and
counted, not silently included.

Sets: all 32, top 10, top 5, ranked in advance by share of subindex variance.
Non-members are carried at their own CPI-only benchmark, never dropped, because
dropping them would change the weight base and redefine the target.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

PESO_SUB = 4.7789
MIN_COVER = 0.90          # of total weight, else the fortnight is dropped

fc = pd.read_csv("data/curated/system_forecasts.csv")
tab = pd.read_csv("data/curated/system_scores.csv")
pri = pd.read_csv("data/curated/prioridad_varianza.csv")
pri["name"] = pri.generico.str.split(" ", n=1).str[1]
lab = (pd.read_parquet("data/curated/cat_national_monthly.parquet",
                       columns=["categoria", "categoria_label"])
       .drop_duplicates().set_index("categoria")["categoria_label"].to_dict())
fc["generico"] = fc.slug.map(lab)
share = dict(zip(pri.name, pri.share))
adm = dict(zip(tab.generico, tab.admisible))

order = sorted(fc.generico.dropna().unique(), key=lambda g: -share.get(g, 0.0))
SETS = {"all32": set(order), "top10": set(order[:10]), "top5": set(order[:5])}
print("ranking (share of subindex variance):")
for i, g in enumerate(order[:10], 1):
    print(f"  {i:2d}. {g:<30} {share.get(g,0):5.2f}%  "
          f"{'admitted' if adm.get(g) else 'GATED -> benchmark'}")
print(f"  top5 {sum(share.get(g,0) for g in order[:5]):.1f}%   "
      f"top10 {sum(share.get(g,0) for g in order[:10]):.1f}%\n")

VINT = ["d5", "d10", "close"]
rows, dropped = [], 0
for t, g in fc.groupby("t"):
    g = g.dropna(subset=["y"])
    w_all = (g.peso * g.theta).to_numpy(float)
    tot = w_all.sum()
    rec = {"t": t}
    ok_any = False
    for v in VINT:
        bench = g[f"{v}_bench"].to_numpy(float)
        model = g[f"{v}_combo"].to_numpy(float)
        gen = g.generico.to_numpy()
        can_model = np.array([bool(adm.get(x, False)) for x in gen]) & ~np.isnan(model)
        for sname, keep in SETS.items():
            use = np.where(can_model & np.isin(gen, list(keep)), model, bench)
            m = ~np.isnan(use)                       # the available set this fortnight
            cov = w_all[m].sum() / tot if tot > 0 else 0.0
            if cov < MIN_COVER:
                continue
            ww = w_all[m] / w_all[m].sum()
            rec[f"{v}_{sname}"] = float((use[m] * ww).sum())
            if sname == "all32":
                # target and coverage share the same available set, so the two are
                # measured on identical composition
                rec[f"y_{v}"] = float((g.y.to_numpy()[m] * ww).sum())
                rec[f"cover_{v}"] = float(cov)
            ok_any = True
        bm = ~np.isnan(bench)
        if bm.sum() and w_all[bm].sum() / tot >= MIN_COVER:
            wb = w_all[bm] / w_all[bm].sum()
            rec[f"{v}_bench"] = float((bench[bm] * wb).sum())
    if ok_any:
        rows.append(rec)
    else:
        dropped += 1
A = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
A["y"] = A["y_close"]
A.to_csv("data/curated/system_aggregate.csv", index=False)
print(f"{len(A)} fortnights kept, {dropped} dropped below {MIN_COVER:.0%} weight coverage")
print(f"median weight coverage {100*A.cover_close.median():.1f}%, "
      f"min {100*A.cover_close.min():.1f}%")
print(f"subindex sd {A.y.std():.3f} pp = {A.y.std()*PESO_SUB/100:.4f} pp of headline\n")


def R(a, b):
    m = ~(a.isna() | b.isna())
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean()))


def M(a, b):
    m = ~(a.isna() | b.isna())
    return float((a[m] - b[m]).abs().mean())


out = []
hdr = f"{'coverage':<10}{'vintage':<8}{'RMSE pp':>9}{'MAE pp':>8}{'headline ±pp':>14}{'vs bench':>10}{'n':>6}"
print(hdr)
print("-" * len(hdr))
for sname in ("all32", "top10", "top5"):
    for v in VINT:
        y = A[f"y_{v}"] if f"y_{v}" in A else A.y
        r, mae = R(y, A[f"{v}_{sname}"]), M(y, A[f"{v}_{sname}"])
        rb = R(y, A[f"{v}_bench"])
        n = int((~(y.isna() | A[f"{v}_{sname}"].isna())).sum())
        out.append({"coverage": sname, "vintage": v, "rmse_pp": r, "mae_pp": mae,
                    "headline_pp": r * PESO_SUB / 100,
                    "vs_bench_pct": 100 * (r / rb - 1), "n": n})
        print(f"{sname:<10}{v:<8}{r:9.3f}{mae:8.3f}{r*PESO_SUB/100:14.4f}"
              f"{100*(r/rb-1):9.1f}%{n:6d}")
rb = R(A.y, A.close_bench)
print(f"{'benchmark':<10}{'close':<8}{rb:9.3f}{M(A.y, A.close_bench):8.3f}"
      f"{rb*PESO_SUB/100:14.4f}{0.0:9.1f}%{int(A.close_bench.notna().sum()):6d}")
pd.DataFrame(out).to_csv("data/curated/system_aggregate_scores.csv", index=False)

r5, r10, r32 = (R(A.y, A[f"close_{k}"]) for k in ("top5", "top10", "all32"))
print(f"\nDoes coverage buy anything, at close of fortnight?")
print(f"  top 5  {r5:.3f} pp  (±{r5*PESO_SUB/100:.4f} pp of headline)  "
      f"{100*(1-r5/rb):.0f}% better than benchmark")
print(f"  top 10 {r10:.3f} pp  (±{r10*PESO_SUB/100:.4f})  {100*(r10/r5-1):+.1f}% vs top 5")
print(f"  all 32 {r32:.3f} pp  (±{r32*PESO_SUB/100:.4f})  {100*(r32/r10-1):+.1f}% vs top 10, "
      f"{100*(r32/r5-1):+.1f}% vs top 5")

# top-10 without the two gated members, so the cost of gating is explicit
gated = [g for g in order[:10] if not adm.get(g, False)]
print(f"\ngated members inside the top 10: {gated or 'none'}")
res = A.y - A.close_all32
print(f"\nall-32 error: mean {res.mean():+.3f} pp, sd {res.std():.3f}, "
      f"worst {res.abs().max():.2f} pp")
print(f"sign of the subindex change called correctly: "
      f"{100*float((np.sign(A.y)==np.sign(A.close_all32)).mean()):.0f}%  "
      f"(benchmark {100*float((np.sign(A.y)==np.sign(A.close_bench)).mean()):.0f}%)")
print(f"R2 out of sample: all32 "
      f"{1-((A.y-A.close_all32)**2).sum()/((A.y-A.y.mean())**2).sum():.3f}  "
      f"benchmark {1-((A.y-A.close_bench)**2).sum()/((A.y-A.y.mean())**2).sum():.3f}")
