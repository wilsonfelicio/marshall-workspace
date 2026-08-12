#!/usr/bin/env python3
"""INPC collection and alignment against the SNIIM wholesale series.

Separate entrypoint from run.py on purpose: this touches a different data source
(INEGI, not SNIIM) and can be run while a SNIIM backfill is in progress. It never
writes into data/raw/, so it cannot collide with the collector's lock.

  python inpc_run.py fetch                 download INPC genéricos + subindices
  python inpc_run.py precios               download retail prices in MXN/kg
  python inpc_run.py validate              check the 32 reproduce serie 865557
  python inpc_run.py align                 join to the SNIIM index, screen correlations
  python inpc_run.py benchmarks            OOS horse race on INPC own-history models

Outputs go to data/inpc/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from inpc import catalogo, genericos, precios  # noqa: E402
from sniim import config as sniim_config  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "inpc"
PP_CODES = Path(__file__).resolve().parent / "config" / "precios_promedio_codigos.csv"

# Precios Promedio numbers its genéricos independently of the INPC clave, and
# renumbers at every base change (see inpc/precios.py: in the 2011 vintage 045
# is Plátanos, not Aguacate). Reusing the clave silently returns a different
# product - 078 pre-2018 is "Jugos o néctares envasados", not Zanahoria.
PP_CODE_COLUMN = {"": "cod_vintage_2011", "18": "cod_vintage_18",
                  "18a": "cod_vintage_18a"}


def _periods_index(periods) -> pd.PeriodIndex:
    """Monthly PeriodIndex. For quincenal input, collapses to the month."""
    return pd.PeriodIndex(
        [pd.Period(year=p[1], month=p[2], freq="M") for p in periods], freq="M")


def cmd_fetch(args, log) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for freq in (["mensual", "quincenal"] if args.frequency == "both" else [args.frequency]):
        periods, data, sids = genericos.fetch_genericos(freq)
        rows = []
        for i, p in enumerate(periods):
            rec = {"periodo": p[0], "anio": p[1], "mes": p[2], "quincena": p[3]}
            rec.update({n: data[n][i] for n in data})
            rows.append(rec)
        df = pd.DataFrame(rows)
        df.to_parquet(OUT / f"inpc_genericos_{freq}.parquet", index=False)
        df.to_csv(OUT / f"inpc_genericos_{freq}.csv", index=False)
        log.info("%s -> %d periods x %d genéricos", freq, len(df), len(data))

    sper, sdata = genericos.fetch_subindices()
    sdf = pd.DataFrame({"periodo": [p[0] for p in sper], "anio": [p[1] for p in sper],
                        "mes": [p[2] for p in sper], **sdata})
    sdf.to_parquet(OUT / "inpc_subindices.parquet", index=False)
    sdf.to_csv(OUT / "inpc_subindices.csv", index=False)
    log.info("subindices -> %d periods x %d series", len(sdf), len(sdata))
    return 0


def cmd_precios(args, log) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = pd.read_csv(PP_CODES, dtype=str)
    frames, diags = [], []
    for bs, lo, hi in precios.VINTAGES:
        pi, pf = max(lo, args.desde), min(hi, args.hasta)
        if pi > pf:
            continue
        col = PP_CODE_COLUMN[bs]
        # Chunk by generic: one generic x 55 cities x 90 months is already ~7 MB.
        for clave, label, c in zip(cmap["clave_generico"], cmap["generico"],
                                   cmap[col]):
            if pd.isna(c):
                log.info("  bs=%s %s: not broken out in this vintage, skipped",
                         bs, clave)
                continue
            try:
                hdr, rows = precios.fetch([c], pi, pf, bs=bs)
            except Exception as exc:
                log.error("precios bs=%s generic %s failed: %s", bs, clave, exc)
                continue
            if not rows:
                continue
            kg, diag = precios.to_kg_frame(hdr, rows)
            # Relabel to the INPC clave and canonical name: the vintage code is
            # a fetch detail, and the source name drifts across bases
            # ("Otras legumbres" -> "Otras verduras y legumbres").
            kg["clave_generico"] = clave
            kg["generico"] = label
            diag.update({"bs": bs, "clave_generico": clave, "cod_vintage": c})
            diags.append(diag)
            if len(kg):
                frames.append(kg)
            log.info("  bs=%s %s (cod %s): %d quotes, %d KG (%.1f%% dropped)",
                     bs, clave, c, diag["quotes_total"], diag["quotes_kg"],
                     diag["pct_non_kg"])
    if not frames:
        log.error("no retail quotes retrieved")
        return 1
    allq = pd.concat(frames, ignore_index=True)
    allq.to_parquet(OUT / "precios_promedio_quotes.parquet", index=False)
    nat = precios.national_mean(allq, catalogo.city_weights())
    nat.to_parquet(OUT / "precios_promedio_nacional.parquet", index=False)
    nat.to_csv(OUT / "precios_promedio_nacional.csv", index=False)
    pd.DataFrame(diags).to_csv(OUT / "precios_promedio_unidades.csv", index=False)
    log.info("retail: %d KG quotes -> %d generic-months", len(allq), len(nat))
    return 0


def cmd_validate(args, log) -> int:
    """Do the 32 genéricos reproduce INEGI's published 'Frutas y verduras'?"""
    periods, data, _ = genericos.fetch_genericos("mensual")
    sper, sdata = genericos.fetch_subindices([catalogo.VALIDATION_TARGET])
    # The two exports do not share a period grid: a subindex export only spans the
    # periods where that series exists (865557 starts Ene 1982), so align by label.
    pub_by_label = dict(zip([p[0] for p in sper], sdata["Frutas y verduras"]))
    rec_full = catalogo.aggregate_weighted(data, len(periods))
    labels = [p[0] for p in periods]
    keep = [i for i, lab in enumerate(labels) if lab in pub_by_label]
    if len(keep) < 24:
        log.error("only %d overlapping periods between genéricos and the subindex", len(keep))
        return 1
    periods = [periods[i] for i in keep]
    rec = [rec_full[i] for i in keep]
    pub = [pub_by_label[labels[i]] for i in keep]

    def mom(s, t):
        if None in (s[t], s[t - 1]) or not s[t - 1]:
            return None
        return 100 * (s[t] / s[t - 1] - 1)

    errs = [(mom(rec, t) - mom(pub, t))
            for t in range(1, len(periods))
            if mom(rec, t) is not None and mom(pub, t) is not None
            and periods[t][1] >= 2024 and (periods[t][1], periods[t][2]) >= (2024, 8)]
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    print(f"\nreconstructed vs published 'Frutas y verduras' (serie "
          f"{catalogo.VALIDATION_TARGET}), Aug-2024 onward, n={len(errs)}")
    print(f"  RMSE of month-on-month change : {rmse:.5f} pp")
    print(f"  max abs error                 : {max(abs(e) for e in errs):.5f} pp")
    for t in range(len(periods) - 4, len(periods)):
        print(f"  {periods[t][0]:>9}  reconstructed {mom(rec, t):+7.3f}%   "
              f"published {mom(pub, t):+7.3f}%")
    ok = rmse < 0.1
    print(f"\n  {'PASS' if ok else 'FAIL'} - the 32 genéricos {'do' if ok else 'do NOT'} "
          f"reproduce the published subindex")
    return 0 if ok else 1


def _load_wholesale(cfg) -> pd.DataFrame:
    p = cfg.curated_dir / "cat_index_monthly.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing - run `python run.py build` in the SNIIM project first")
    wh = pd.read_parquet(p)
    wh["mes"] = pd.PeriodIndex(pd.to_datetime(wh["mes"]), freq="M")
    wh["w_dlog"] = np.log1p(wh["var_mensual"])
    return wh


def cmd_align(args, log) -> int:
    """Join wholesale to retail and screen which items carry usable signal.

    Two correlations are reported and only the second one means anything:

      corr_bruta     raw correlation of month-on-month log changes. Inflated,
                     because wholesale and retail share the same seasonal cycle,
                     so this largely measures seasonality both sides already know.
      corr_residual  after removing each series' calendar-month climatology. This
                     is the incremental information the wholesale data adds over a
                     seasonal benchmark, which is the only thing that can improve
                     a forecast.

    The ECB's scanner-data work (WP 2930) finds high-frequency data starts cutting
    nowcast error once this correlation exceeds roughly 0.4. Items below that are
    better served by the seasonal benchmark - route them there rather than
    pretending to model them.
    """
    cfg = sniim_config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    wh = _load_wholesale(cfg)

    gpath = OUT / "inpc_genericos_mensual.parquet"
    if not gpath.exists():
        raise FileNotFoundError(f"{gpath} missing - run `python inpc_run.py fetch` first")
    g = pd.read_parquet(gpath)
    idx = pd.PeriodIndex([pd.Period(year=int(a), month=int(m), freq="M")
                          for a, m in zip(g["anio"], g["mes"])], freq="M")
    names = [c for c in g.columns if c not in ("periodo", "anio", "mes", "quincena")]
    inpc = pd.DataFrame({n: g[n].values for n in names}, index=idx)
    r = np.log(inpc).diff()

    mapping = json.loads((Path(__file__).parent / "config" / "map_sniim_inpc.json")
                         .read_text(encoding="utf-8"))
    cutoff = pd.Period(args.clima_hasta, freq="M")
    train = r[(r.index >= pd.Period("1996-01", freq="M")) & (r.index < cutoff)]

    rows = []
    for slug, gname in mapping.items():
        w = wh.loc[wh["categoria"] == slug].set_index("mes")["w_dlog"].dropna()
        if w.empty or gname not in r.columns:
            continue
        common = w.index.intersection(r[gname].dropna().index)
        if len(common) < args.min_obs:
            continue
        clim_r = train[gname].groupby(train.index.month).mean()

        # The wholesale seasonal profile needs several years per calendar month.
        # With one year of history every month has exactly one observation, so
        # subtracting the month mean gives identically zero and the correlation is
        # undefined. Fall back to demeaning by the overall mean and SAY SO, rather
        # than emitting a number that looks like a seasonal-adjusted result.
        per_month = w.groupby(w.index.month).size()
        enough = bool(len(per_month) >= 12 and per_month.min() >= args.min_anios)
        if enough:
            clim_w = w.groupby(w.index.month).mean()
            ww = (w.reindex(common) - clim_w.reindex(common.month).values).values
        else:
            ww = (w.reindex(common) - w.reindex(common).mean()).values

        rr = (r[gname].reindex(common) - clim_r.reindex(common.month).values).values
        ok = ~(np.isnan(rr) | np.isnan(ww))
        if ok.sum() < args.min_obs or np.std(ww[ok]) == 0 or np.std(rr[ok]) == 0:
            continue
        raw = np.corrcoef(w.reindex(common).values, r[gname].reindex(common).values)[0, 1]
        res = np.corrcoef(ww[ok], rr[ok])[0, 1]
        beta = np.polyfit(w.reindex(common).values, r[gname].reindex(common).values, 1)[0]
        rows.append({"categoria": slug, "generico_inpc": gname, "n": int(ok.sum()),
                     "corr_bruta": raw, "corr_residual": res,
                     "estacionalidad_mayorista_propia": enough,
                     "beta_traspaso": beta,
                     "peso_inpc": catalogo.weights().get(gname, np.nan),
                     "usable": res > 0.4})
    t = pd.DataFrame(rows).sort_values("corr_residual", ascending=False)
    t.to_csv(OUT / "alignment_screen.csv", index=False)
    print(t.round(3).to_string(index=False))
    print(f"\nmedian corr_bruta    {t.corr_bruta.median():.2f}")
    print(f"median corr_residual {t.corr_residual.median():.2f}  <- the one that matters")
    print(f"items usable (>0.4)  {int(t.usable.sum())}/{len(t)}, "
          f"carrying {t.loc[t.usable, 'peso_inpc'].sum():.2f} of the "
          f"{t.peso_inpc.sum():.2f} total INPC weight")
    if not t["estacionalidad_mayorista_propia"].all():
        n_bad = int((~t["estacionalidad_mayorista_propia"]).sum())
        print(f"\nWARNING: for {n_bad}/{len(t)} items there was not enough wholesale "
              "history to estimate a seasonal profile, so those rows were only "
              "mean-demeaned. corr_residual is then NOT seasonally adjusted on the "
              "wholesale side and is biased UPWARD by shared seasonality.")
    if t.n.max() < 60:
        print(f"WARNING: only {t.n.max()} monthly observations overlap. Indicative at "
              "best - rerun once the SNIIM backfill has full history.")
    return 0


def cmd_benchmarks(args, log) -> int:
    """OOS horse race of own-history benchmarks. This is the bar to beat."""
    gpath = OUT / "inpc_genericos_mensual.parquet"
    g = pd.read_parquet(gpath)
    idx = pd.PeriodIndex([pd.Period(year=int(a), month=int(m), freq="M")
                          for a, m in zip(g["anio"], g["mes"])], freq="M")
    names = [c for c in g.columns if c not in ("periodo", "anio", "mes", "quincena")]
    d = np.log(pd.DataFrame({n: g[n].values for n in names}, index=idx)).diff()
    d = d[d.index >= pd.Period("1996-01", freq="M")]
    start = pd.Period(args.oos_desde, freq="M")
    BM = ["RW", "SeasNaive", "Clima", "AR1", "SD-AR"]

    def oos(y):
        y = y.dropna()
        res = {k: [] for k in BM}
        act = []
        for t_ in y.index[y.index >= start]:
            tr = y[y.index < t_]
            if len(tr) < 120:
                continue
            act.append(y[t_])
            res["RW"].append(0.0)
            prev = y.get(t_ - 12, np.nan)
            res["SeasNaive"].append(prev if pd.notna(prev) else 0.0)
            same = tr[tr.index.month == t_.month]
            res["Clima"].append(same.mean() if len(same) else 0.0)
            Y = tr.values[1:]
            X = np.c_[np.ones(len(Y)), tr.values[:-1]]
            b = np.linalg.lstsq(X, Y, rcond=None)[0]
            ar = b[0] + b[1] * tr.values[-1]
            res["AR1"].append(ar)
            Dm = pd.get_dummies(tr.index.month[1:], drop_first=True).astype(float)
            X2 = np.c_[np.ones(len(Y)), Dm.values, tr.values[:-1]]
            if np.linalg.matrix_rank(X2) == X2.shape[1]:
                b2 = np.linalg.lstsq(X2, Y, rcond=None)[0]
                row = np.zeros(X2.shape[1]); row[0] = 1
                cols = list(Dm.columns)
                if t_.month in cols:
                    row[1 + cols.index(t_.month)] = 1
                row[-1] = tr.values[-1]
                res["SD-AR"].append(float(row @ b2))
            else:
                res["SD-AR"].append(ar)
        a = np.array(act)
        if not len(a):
            return None, 0
        return {k: 100 * float(np.sqrt(np.mean((a - np.array(v)) ** 2)))
                for k, v in res.items()}, len(a)

    out = {}
    for n in names + (["Frutas y verduras"] if "Frutas y verduras" in d.columns else []):
        if n not in d.columns:
            continue
        r_, nn = oos(d[n])
        if r_ and nn > 60:
            out[n] = r_
    t = pd.DataFrame(out).T[BM]
    t["mejor"] = t.idxmin(axis=1)
    t["rmse_mejor"] = t[BM].min(axis=1)
    t["ganancia_vs_rw"] = t["RW"] / t["rmse_mejor"]
    w = catalogo.weights()
    t["peso"] = [w.get(i, np.nan) for i in t.index]
    t["aporte_error"] = t["peso"] * t["rmse_mejor"]
    t["aporte_pct"] = 100 * t["aporte_error"] / t["aporte_error"].sum()
    t.sort_values("aporte_pct", ascending=False).to_csv(OUT / "benchmarks_oos.csv")
    print(t.sort_values("aporte_pct", ascending=False).round(2).to_string())
    print("\nwhich benchmark wins:")
    print(t["mejor"].value_counts().to_string())
    print(f"\nmedian RMSE of the winning benchmark: {t.rmse_mejor.median():.2f} pp/month")
    print(f"median gain over a random walk:       {t.ganancia_vs_rw.median():.2f}x")
    print("\nSD-AR (month dummies + AR1) is the benchmark to beat. A random walk is a "
          "straw man here and should only be reported for completeness.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="inpc_run.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download INPC genérico indices")
    f.add_argument("--frequency", choices=["mensual", "quincenal", "both"], default="both")

    pr = sub.add_parser("precios", help="download retail prices in MXN/kg")
    pr.add_argument("--desde", type=int, default=201101)
    pr.add_argument("--hasta", type=int, default=209912)

    sub.add_parser("validate", help="check the 32 reproduce the published subindex")

    a = sub.add_parser("align", help="screen wholesale-retail correlations")
    a.add_argument("--clima-hasta", default="2015-01",
                   help="climatology is estimated strictly before this month")
    a.add_argument("--min-obs", type=int, default=8)
    a.add_argument("--min-anios", type=int, default=3,
                   help="years required per calendar month before a wholesale "
                        "seasonal profile is trusted; below this the series is only "
                        "mean-demeaned and the result is flagged")

    b = sub.add_parser("benchmarks", help="OOS horse race of own-history benchmarks")
    b.add_argument("--oos-desde", default="2012-01")

    args = p.parse_args(argv)
    cfg = sniim_config.load()
    log = sniim_config.setup_logging(cfg, f"inpc_{args.cmd}", args.verbose)
    return {"fetch": cmd_fetch, "precios": cmd_precios, "validate": cmd_validate,
            "align": cmd_align, "benchmarks": cmd_benchmarks}[args.cmd](args, log)


if __name__ == "__main__":
    raise SystemExit(main())
