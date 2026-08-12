"""One workbook per generic, same nine sheets as the jitomate file.

  python3 export_generic_xlsx.py tomate_verde cebolla calabacita chile_serrano

Generalised from export_jitomate_xlsx.py. Two things had to change:

  * Variety weighting. Jitomate has two varieties and the level series used a fixed
    65/35. Cebolla has four and Calabacita three, so the level is a geometric mean
    across varieties weighted by each variety's SHARE OF QUOTES over the whole
    sample. Fixed, not time-varying: a moving weight would inject level moves that
    are pure recomposition. Products with a single variety degenerate to that
    variety, with no weighting decision at all.

  * The out-of-sample nowcasts come from the 32-generic system run
    (data/curated/system_forecasts.csv) rather than the standalone jitomate file.

Retail prices are included where INEGI's Precios Promedio series has been fetched
for that generic, and the sheet is omitted with a note where it has not.
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

MA = 30
FACTS = json.load(open("data/curated/facts.json"))
LABELS = (pd.read_parquet("data/curated/cat_national_monthly.parquet",
                          columns=["categoria", "categoria_label"])
          .drop_duplicates().set_index("categoria")["categoria_label"].to_dict())
PRI = pd.read_csv("data/curated/prioridad_varianza.csv")
PRI["name"] = PRI.generico.str.split(" ", n=1).str[1]
CLAVE = dict(zip(PRI.name, PRI.generico))
SCORES = pd.read_csv("data/curated/system_scores.csv").set_index("generico")


def build(slug: str) -> str:
    name = LABELS[slug]
    fr = next(r for r in FACTS["table"] if r["generico"] == name)
    peso = float(SCORES.loc[name, "peso"])
    OUT = f"{slug}_data.xlsx"

    # ---------------------------------------------------------- 1. daily by market
    raw = pd.read_parquet("data/curated/var_market_daily.parquet")
    raw = raw[raw.categoria == slug].copy()
    raw["fecha"] = pd.to_datetime(raw.fecha)
    dm = (raw.rename(columns={"producto": "variedad", "destino": "mercado",
                              "destino_estado": "estado", "precio_geo": "precio_mxn_kg"})
          [["fecha", "variedad", "mercado", "estado", "precio_mxn_kg", "n_obs",
            "n_origenes"]]
          .sort_values(["fecha", "variedad", "mercado"]).reset_index(drop=True))
    dm["fecha"] = dm.fecha.dt.date

    # variety weights = share of quotes over the whole sample, fixed
    vs = raw.producto.value_counts()
    vw = (vs / vs.sum()).to_dict()
    varieties = list(vs.index)

    # ---------------------------------------------------------- 2. daily national
    pw = pd.read_parquet("data/curated/pesos_mercado.parquet")
    pwd = dict(zip(pw.destino, pd.to_numeric(pw.peso_inpc, errors="coerce").fillna(0.0)))
    raw["_w"] = raw.destino.map(pwd).fillna(0.0)
    raw["_lp"] = np.log(raw.precio_geo.where(raw.precio_geo > 0))
    g = (raw.groupby(["fecha", "producto"], as_index=False)
         .agg(lp=("_lp", "mean"), nm=("destino", "nunique")))
    gw = (raw[raw._w > 0].groupby(["fecha", "producto"])
          .apply(lambda x: np.average(x._lp, weights=x._w)).rename("lpw").reset_index())
    piv = g.pivot(index="fecha", columns="producto", values="lp")
    pvw = gw.pivot(index="fecha", columns="producto", values="lpw")
    nmk = g.pivot(index="fecha", columns="producto", values="nm")

    def blend(P):
        """Variety-weighted mean of logs over the varieties present that day.

        Renormalised over what is present, so a day on which one variety is not
        quoted is a mean of the rest rather than a hole. A day with no variety at
        all stays missing.
        """
        cols = [c for c in P.columns if c in vw]
        w = np.array([vw[c] for c in cols])
        A = P[cols].to_numpy(float)
        m = ~np.isnan(A)
        den = (m * w).sum(axis=1)
        num = np.nansum(np.where(m, A, 0.0) * w, axis=1)
        return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan),
                         index=P.index)

    dn = pd.DataFrame(index=piv.index)
    for v in varieties:
        dn[f"p_{v}"] = np.exp(piv[v]) if v in piv else np.nan
    dn["p_index"] = np.exp(blend(piv))
    dn["p_index_ponderado"] = np.exp(blend(pvw))
    dn["n_mercados"] = nmk.sum(axis=1).astype("Int64")
    dn = dn[dn.p_index.notna()]
    cal = pd.DataFrame(index=pd.date_range(dn.index.min(), dn.index.max(), freq="D"))
    cal["p_index"] = dn.p_index.reindex(cal.index)
    cal["ma30"] = cal.p_index.rolling(f"{MA}D", min_periods=12).mean()
    cal["chg_30d_pct"] = 100 * (np.log(cal.ma30) - np.log(cal.ma30.shift(MA)))
    dn = dn.join(cal[["ma30", "chg_30d_pct"]]).reset_index()
    dn["fecha"] = pd.to_datetime(dn.fecha).dt.date

    # ---------------------------------------------------------- 3. fortnightly panel
    d = Q.dataset(slug, CLAVE[name], windows=(5, 10)).set_index("t")
    st = pd.DatetimeIndex(Q.qtimestamp(d.index.values))
    close = pd.DatetimeIndex(np.where(st.day == 1, st + pd.Timedelta(days=14),
                                      st + pd.offsets.MonthEnd(0)))
    qn = pd.DataFrame({
        "t": d.index, "etiqueta": d.etiqueta.values,
        "fecha_inicio": st.date, "fecha_cierre": close.date,
        "quincena_del_anio": d.quincena_del_anio.values,
        f"inpc_{slug}": d.inpc.astype(float).values,
        "cpi_dln_pct": d.y.values, "ws_dln_pct": d.x_full.values,
        "ws_dln_dia5_pct": d.x_w5.values, "ws_dln_dia10_pct": d.x_w10.values,
        "ws_ln_nivel_cadena": d.lw_full.values,
        "n_celdas": d.n_celdas_full.values, "n_dias_cotizados": d.n_dias.values,
        "sd_entre_mercados": d.sd_mercados.values,
        "segmento_cadena": d.segmento.values})
    lvl = pd.Series(blend(piv).values, index=pd.DatetimeIndex(piv.index)).dropna()
    lq = lvl.groupby(lvl.index.year * 24 + (lvl.index.month - 1) * 2
                     + (lvl.index.day > 15).astype(int)).mean()
    qn["ws_nivel_directo_mxn_kg"] = np.exp(qn.t.map(lq))
    qn["deriva_cadena_ln"] = qn.ws_ln_nivel_cadena - np.log(qn.ws_nivel_directo_mxn_kg)

    # ---------------------------------------------------------- 4. out-of-sample
    fc = pd.read_csv("data/curated/system_forecasts.csv")
    mo = fc[fc.slug == slug][["t", "y", "close_combo", "close_bench", "close_sigma",
                              "d5_combo", "d10_combo", "close_placebo"]].copy()
    mo = mo.sort_values("t").reset_index(drop=True)
    mo.insert(1, "etiqueta", mo.t.map(dict(zip(qn.t, qn.etiqueta))))
    mo.insert(2, "fecha_cierre", mo.t.map(dict(zip(qn.t, qn.fecha_cierre))))
    mo = mo.rename(columns={"y": "cpi_realizado_pct", "close_combo": "nowcast_pct",
                            "close_bench": "benchmark_cpi_only_pct",
                            "close_sigma": "sigma_pp", "d5_combo": "nowcast_dia5_pct",
                            "d10_combo": "nowcast_dia10_pct",
                            "close_placebo": "placebo_pct"})
    z = ((mo.cpi_realizado_pct - mo.nowcast_pct).abs() / mo.sigma_pp).to_numpy(float)
    k = np.full(len(z), np.nan)
    for i in range(len(z)):
        prev = z[:i][~np.isnan(z[:i])]
        if len(prev) >= 40:
            k[i] = np.quantile(prev, 0.80)
    mo["banda80_pp"] = k * mo.sigma_pp

    # ---------------------------------------------------------- 5. like-for-like
    q2 = dn.copy()
    q2["fecha"] = pd.to_datetime(q2.fecha)
    q2["t"] = (q2.fecha.dt.year * 24 + (q2.fecha.dt.month - 1) * 2
               + (q2.fecha.dt.day > 15).astype(int))
    wq = q2.groupby("t").p_index.apply(lambda z2: np.log(z2).mean())
    stp = wq.index.to_series().diff()
    ln = np.log(pd.Series(qn[f"inpc_{slug}"].values, index=qn.t))
    cst = qn.t.diff()
    mc = pd.DataFrame({
        "t": qn.t.values, "etiqueta": qn.etiqueta.values,
        "fecha_cierre": qn.fecha_cierre.values,
        "cpi_chg_2q_pct": np.where((cst == 1) & (cst.shift(1) == 1),
                                   100 * (ln - ln.shift(2)), np.nan)}).set_index("t")
    mc["ws_chg_2q_pct"] = (100 * (wq - wq.shift(2)))[(stp == 1) & (stp.shift(1) == 1)]
    mc = mc.dropna().reset_index()
    mc["desde_2016"] = (mc.t >= 2016 * 24).astype(int)

    # ---------------------------------------------------------- 6. retail
    rp = pathlib.Path(f"data/inpc/precios_kg/{slug}.parquet")
    rt = None
    if rp.exists():
        rt = pd.read_parquet(rp)
        rt = rt[["periodo", "mes", "clave_generico", "precio_geo", "n_ciudades"]].rename(
            columns={"precio_geo": "retail_mxn_kg", "clave_generico": "clave_inpc"})
        wm = (q2.set_index("fecha").p_index_ponderado.resample("MS")
              .apply(lambda z2: np.exp(np.log(z2.dropna()).mean())))
        rt["mes"] = pd.to_datetime(rt.mes)
        rt = rt.merge(wm.rename("wholesale_mxn_kg").reset_index()
                      .rename(columns={"fecha": "mes"}), on="mes", how="left")
        rt["mes"] = rt.mes.dt.date

    # ---------------------------------------------------------- 7. markets
    cov = (raw.groupby("destino").agg(
        primer_dia=("fecha", "min"), ultimo_dia=("fecha", "max"),
        dias=("fecha", "nunique"), obs=("precio_geo", "size"),
        precio_medio=("precio_geo", "mean"),
        variedades=("producto", "nunique")).reset_index())
    mk = cov.merge(pw, on="destino", how="left").rename(columns={"destino": "mercado"})
    for c in ("primer_dia", "ultimo_dia"):
        mk[c] = pd.to_datetime(mk[c]).dt.date

    return _write(OUT, slug, name, peso, fr, vw, varieties, dm, dn, qn, mo, mc, rt, mk)


def _write(OUT, slug, name, peso, fr, vw, varieties, dm, dn, qn, mo, mc, rt, mk):
    NQ, NO, NC = len(qn), len(mo), len(mc)
    NR = 0 if rt is None else len(rt)
    with pd.ExcelWriter(OUT, engine="xlsxwriter",
                        engine_kwargs={"options": {"default_date_format": "yyyy-mm-dd"}}) as xl:
        book = xl.book
        H = book.add_format({"bold": True, "font_name": "Arial", "font_size": 10,
                             "bg_color": "#21295C", "font_color": "white",
                             "align": "left", "valign": "vcenter", "text_wrap": True})
        T = book.add_format({"font_name": "Arial", "font_size": 10})
        B = book.add_format({"font_name": "Arial", "font_size": 10, "bold": True})
        TI = book.add_format({"font_name": "Arial", "font_size": 13, "bold": True,
                              "font_color": "#21295C"})
        WR = book.add_format({"font_name": "Arial", "font_size": 10, "text_wrap": True,
                              "valign": "top"})
        N2 = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.00"})
        N3 = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000"})

        def sheet(df, nm, widths=None, nfmt=None):
            ws = book.add_worksheet(nm)
            xl.sheets[nm] = ws
            for j, c in enumerate(df.columns):
                ws.write(0, j, c, H)
                ws.set_column(j, j, (widths or {}).get(c, 14), (nfmt or {}).get(c))
            df.to_excel(xl, sheet_name=nm, index=False, startrow=1, header=False)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
            return ws

        # ------------------------------------------------------------- LEEME
        ws = book.add_worksheet("LEEME")
        ws.set_column(0, 0, 26); ws.set_column(1, 1, 104)
        ws.write(0, 0, f"{name}: wholesale prices, CPI, and the nowcast", TI)
        vtxt = ", ".join(f"{v} {vw[v]:.0%}" for v in varieties)
        rows = [
            ("Purpose", f"Every {name} series behind the nowcast, at the frequency it is "
                        f"actually built at, so the numbers can be checked independently. "
                        f"Nothing is sampled: the daily panel is complete."),
            ("Sources", "Wholesale: SNIIM, Secretaría de Economía, daily quotes by market "
                        "and variety, free and public. CPI: INEGI, fortnightly."),
            ("Units", "Prices in MXN per kilogram. Changes in percent (log difference x "
                      "100). 'pp' means percentage points of the published print."),
            ("This generic", f"INPC clave {CLAVE[name]}, weight {peso:.5f} of the basket, "
                             f"{fr['share']:.1f}% of the Frutas y verduras subindex's "
                             f"fortnightly variance. Nowcast RMSE {fr['close']:.2f} pp "
                             f"against {fr['bench']:.2f} pp for a CPI-only benchmark, "
                             f"correct sign {fr['sign']:.0f}%, on {fr['n']} fortnights."),
            ("Varieties", f"{len(varieties)}: {vtxt}. Those weights are each variety's "
                          f"share of all quotes over the whole sample, applied to the "
                          f"LEVEL series only and held fixed — a moving weight would put "
                          f"pure recomposition into the level. The index the model uses "
                          f"needs no variety weight at all: see quincenal below."),
            ("", ""),
            ("SHEET: daily_mercado", f"{len(dm):,} rows. The raw panel: one row per date x "
                                     f"variety x market. Everything else is derived from it."),
            ("SHEET: daily_nacional", f"{len(dn):,} rows. Daily national level, one column "
                                      f"per variety plus two composites: p_index "
                                      f"(equal-weighted across markets) and "
                                      f"p_index_ponderado (each market carrying its INPC "
                                      f"city weight). ma30 is a {MA}-CALENDAR-day mean."),
            ("SHEET: quincenal", f"{NQ:,} rows. The fortnightly panel the model consumes. "
                                 f"ws_dln_pct is the chained matched-cell index: only "
                                 f"variety x market cells present in BOTH fortnights enter, "
                                 f"so a market that stops quoting cannot move it, and no "
                                 f"variety weight is assumed. dia5 / dia10 are the same "
                                 f"index cut off after 5 and 10 days of the fortnight."),
            ("SHEET: nowcast_oos", f"{NO:,} rows. Out-of-sample nowcasts from the "
                                   f"32-generic system run, refit every fortnight on a "
                                   f"rolling five-year window using earlier data only. "
                                   f"placebo_pct is the same wholesale series shifted to "
                                   f"the wrong dates: it should NOT beat the benchmark."),
            ("SHEET: mensual_comparable", f"{NC:,} rows. Both series measured the same way: "
                                          f"each fortnight against the fortnight two prints "
                                          f"earlier, i.e. about one month."),
            ("SHEET: retail_mensual", f"{NR:,} rows. INEGI's average retail price per kg "
                                      f"beside the wholesale monthly mean."
                                      if rt is not None else
                                      "NOT INCLUDED. INEGI's Precios Promedio series for "
                                      "this generic has not been fetched; the CPI index "
                                      "itself is on the quincenal sheet and is unaffected."),
            ("SHEET: mercados", f"{len(mk):,} rows. Market list, coverage, and the INPC "
                                f"city weight each one carries."),
            ("SHEET: CHECKS", "Live formulas that recompute every headline number from the "
                              "sheets in this file. If a formula disagrees with what I told "
                              "you, the formula is the one to trust."),
            ("", ""),
            ("WATCH OUT (1)", "A fortnight is labelled by its FIRST day in most of my code "
                              "but summarises prices through its LAST day. Both columns are "
                              "given: fecha_inicio and fecha_cierre. Plotting the CPI "
                              "against a daily wholesale series at the start date badly "
                              "understates their correlation."),
            ("WATCH OUT (2)", "cpi_dln_pct is a FORTNIGHTLY change; cpi_chg_2q_pct is a "
                              "two-fortnight change. Different quantities, and the second "
                              "is not twice the first."),
            ("WATCH OUT (3)", "segmento_cadena increments where the matched-cell chain "
                              "breaks. Do not difference the wholesale index across a "
                              "segment boundary."),
            ("WATCH OUT (4)", "Where a fortnight has a CPI print but no nowcast, the two "
                              "models must be scored on the same rows or the benchmark gets "
                              "a longer sample and looks worse than it is. The error "
                              "columns on nowcast_oos are blanked together for that reason."),
            ("WATCH OUT (5)", "ws_ln_nivel_cadena is the chained index in logs and its LEVEL "
                              "is not pesos: chaining accumulates composition change "
                              "(column deriva_cadena_ln). Use ws_nivel_directo_mxn_kg for "
                              "levels and the chained series only for changes. Every model "
                              "result uses changes only and is unaffected."),
            ("WATCH OUT (6)", "Before 2000 SNIIM quoted weekly rather than daily, so "
                              "n_dias_cotizados per fortnight is 2-3 in the early years and "
                              "10-11 later. The cell-matching requirement adapts to that."),
            ("", ""),
            ("Rebuild", f"python3 export_generic_xlsx.py {slug}"),
        ]
        for i, (k, v) in enumerate(rows, start=2):
            ws.write(i, 0, k, B); ws.write(i, 1, v, WR)
            ws.set_row(i, 44 if len(v) > 150 else (30 if len(v) > 90 else None))

        sheet(dm, "daily_mercado",
              {"fecha": 11, "variedad": 20, "mercado": 52, "estado": 20,
               "precio_mxn_kg": 14}, {"precio_mxn_kg": N2})
        sheet(dn, "daily_nacional", {"fecha": 11},
              {c: N2 for c in dn.columns if c.startswith("p_") or c in
               ("ma30", "chg_30d_pct")})
        sheet(qn, "quincenal", {"etiqueta": 14, "fecha_inicio": 12, "fecha_cierre": 12,
                                "segmento_cadena": 15},
              {c: N3 for c in ("cpi_dln_pct", "ws_dln_pct", "ws_dln_dia5_pct",
                               "ws_dln_dia10_pct", f"inpc_{slug}")})
        sheet(mc, "mensual_comparable", {"etiqueta": 14, "fecha_cierre": 12},
              {"cpi_chg_2q_pct": N3, "ws_chg_2q_pct": N3})
        if rt is not None:
            sheet(rt, "retail_mensual", {"periodo": 11, "mes": 11, "clave_inpc": 11},
                  {"retail_mxn_kg": N2, "wholesale_mxn_kg": N2})
        sheet(mk, "mercados", {"mercado": 52, "estado": 20, "ciudad_inpc": 18,
                               "primer_dia": 11, "ultimo_dia": 11},
              {"precio_medio": N2, "peso_inpc": N3})

        # ------------------------------------------------------------- nowcast_oos
        ws = book.add_worksheet("nowcast_oos")
        xl.sheets["nowcast_oos"] = ws
        cols = list(mo.columns)
        extra = ["err_nowcast_pp", "err_benchmark_pp", "acierto_signo_nowcast",
                 "acierto_signo_benchmark", "dentro_banda_80", "y_pareado_pct",
                 "err_placebo_pp"]
        for j, c in enumerate(cols + extra):
            ws.write(0, j, c, H); ws.set_column(j, j, 15 if j < 3 else 14)
        A = lambda j: chr(65 + j) if j < 26 else chr(64 + j // 26) + chr(65 + j % 26)
        ci = {c: A(cols.index(c)) for c in cols}
        y, f, b = ci["cpi_realizado_pct"], ci["nowcast_pct"], ci["benchmark_cpi_only_pct"]
        bd, pl = ci["banda80_pp"], ci["placebo_pct"]
        e0, date_j = len(cols), cols.index("fecha_cierre")
        for i, (_, row) in enumerate(mo.iterrows(), start=2):
            for j, c in enumerate(cols):
                v = row[c]
                if j == date_j:
                    ws.write_datetime(i - 1, j, pd.Timestamp(v))
                elif pd.isna(v):
                    ws.write_blank(i - 1, j, None)
                elif isinstance(v, str):
                    ws.write_string(i - 1, j, v)
                else:
                    ws.write_number(i - 1, j, float(v), N3 if j > 3 else None)
            ws.write_formula(i - 1, e0, f'=IF(ISNUMBER({f}{i}),{y}{i}-{f}{i},"")', N3)
            ws.write_formula(i - 1, e0 + 1,
                             f'=IF(AND(ISNUMBER({b}{i}),ISNUMBER({f}{i})),{y}{i}-{b}{i},"")', N3)
            ws.write_formula(i - 1, e0 + 2,
                             f'=IF(ISNUMBER({f}{i}),IF(SIGN({y}{i})=SIGN({f}{i}),1,0),"")')
            ws.write_formula(i - 1, e0 + 3,
                             f'=IF(AND(ISNUMBER({b}{i}),ISNUMBER({f}{i})),'
                             f'IF(SIGN({y}{i})=SIGN({b}{i}),1,0),"")')
            ws.write_formula(i - 1, e0 + 4,
                             f'=IF(AND(ISNUMBER({f}{i}),ISNUMBER({bd}{i})),'
                             f'IF(ABS({y}{i}-{f}{i})<={bd}{i},1,0),"")')
            ws.write_formula(i - 1, e0 + 5, f'=IF(ISNUMBER({f}{i}),{y}{i},"")', N3)
            # paired with the nowcast, like every other error column: a placebo scored
            # on its own rows is not comparable with the benchmark scored on ours
            ws.write_formula(i - 1, e0 + 6,
                             f'=IF(AND(ISNUMBER({pl}{i}),ISNUMBER({f}{i})),{y}{i}-{pl}{i},"")',
                             N3)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, NO, len(cols) + len(extra) - 1)

        # ------------------------------------------------------------- CHECKS
        ws = book.add_worksheet("CHECKS")
        ws.set_column(0, 0, 52); ws.set_column(1, 1, 14); ws.set_column(2, 2, 62)
        ws.write(0, 0, "Consistency checks — every cell in column B is a formula", TI)
        for j, h in enumerate(("Quantity", "Value", "How it is computed, and what I claimed")):
            ws.write(2, j, h, H)

        def rng(sh, frame, col, n, off=None):
            j = len(frame.columns) + off if off is not None else list(frame.columns).index(col)
            c = A(j)
            return f"{sh}!${c}$2:${c}${n + 1}"

        EN = rng("nowcast_oos", mo, None, NO, 0)
        EB = rng("nowcast_oos", mo, None, NO, 1)
        SN = rng("nowcast_oos", mo, None, NO, 2)
        SB = rng("nowcast_oos", mo, None, NO, 3)
        BD = rng("nowcast_oos", mo, None, NO, 4)
        YY = rng("nowcast_oos", mo, None, NO, 5)
        PL = rng("nowcast_oos", mo, None, NO, 6)
        YR = rng("nowcast_oos", mo, "cpi_realizado_pct", NO)
        MCC = rng("mensual_comparable", mc, "cpi_chg_2q_pct", NC)
        MCW = rng("mensual_comparable", mc, "ws_chg_2q_pct", NC)
        MCF = rng("mensual_comparable", mc, "desde_2016", NC)
        QC = rng("quincenal", qn, "cpi_dln_pct", NQ)
        QW = rng("quincenal", qn, "ws_dln_pct", NQ)
        QND = rng("quincenal", qn, "deriva_cadena_ln", NQ)
        DNP = rng("daily_nacional", dn, "p_index", len(dn))

        def cc(x, y_, fl):
            n, sx, sy = f"SUMPRODUCT({fl})", f"SUMPRODUCT({fl},{x})", f"SUMPRODUCT({fl},{y_})"
            sxy, sxx, syy = (f"SUMPRODUCT({fl},{x},{y_})", f"SUMPRODUCT({fl},{x},{x})",
                             f"SUMPRODUCT({fl},{y_},{y_})")
            return f"=({n}*{sxy}-{sx}*{sy})/SQRT(({n}*{sxx}-{sx}^2)*({n}*{syy}-{sy}^2))"

        def cs(y_, x, fl):
            n, sx, sy = f"SUMPRODUCT({fl})", f"SUMPRODUCT({fl},{x})", f"SUMPRODUCT({fl},{y_})"
            sxy, sxx = f"SUMPRODUCT({fl},{x},{y_})", f"SUMPRODUCT({fl},{x},{x})"
            return f"=({n}*{sxy}-{sx}*{sy})/({n}*{sxx}-{sx}^2)"

        checks = [
            ("Fortnights scored (both models present)", f"=COUNT({EN})",
             f"Claimed {fr['n']} of {NO}."),
            ("RMSE of the nowcast, pp of the print",
             f"=SQRT(SUMSQ({EN})/COUNT({EN}))", f"Claimed {fr['close']:.2f} pp."),
            ("RMSE of the CPI-only benchmark, pp",
             f"=SQRT(SUMSQ({EB})/COUNT({EB}))", f"Claimed {fr['bench']:.2f} pp. Both RMSEs "
             f"use the same rows."),
            ("Reduction in RMSE",
             f"=1-SQRT(SUMSQ({EN})/COUNT({EN}))/SQRT(SUMSQ({EB})/COUNT({EB}))",
             f"Claimed {fr['gain_pct']:.0f}%."),
            ("Nowcast error as pp of headline INPC",
             f"=SQRT(SUMSQ({EN})/COUNT({EN}))*{peso}/100",
             f"{peso} is this generic's published weight in the INPC basket."),
            ("Mean error of the nowcast, pp (bias)", f"=AVERAGE({EN})",
             "Should be near zero."),
            ("Correct sign, nowcast", f"=AVERAGE({SN})", f"Claimed {fr['sign']:.0f}%."),
            ("Correct sign, benchmark", f"=AVERAGE({SB})",
             f"Claimed {fr['sign_bench']:.0f}%."),
            ("Realised coverage of the 80% band", f"=AVERAGE({BD})",
             "Nominal is 80%. The band width is in banda80_pp, not recomputed here."),
            ("Standard deviation of the realised change, pp", f"=STDEV({YY})",
             "The thing being forecast, on the same fortnights as the errors above."),
            ("R2 out of sample, nowcast", f"=1-SUMSQ({EN})/DEVSQ({YY})", ""),
            ("R2 out of sample, benchmark", f"=1-SUMSQ({EB})/DEVSQ({YY})", ""),
            ("RMSE of the PLACEBO regressor, pp",
             f"=SQRT(SUMSQ({PL})/COUNT({PL}))",
             "The same wholesale series shifted to the wrong dates, scored on the same rows "
             "as everything above. It must not beat the benchmark by any margin worth "
             "noticing — if it does, the harness is manufacturing skill."),
            ("Placebo vs benchmark",
             f"=SQRT(SUMSQ({PL})/COUNT({PL}))/SQRT(SUMSQ({EB})/COUNT({EB}))-1",
             "Should be near zero. A few percent either way is noise; a large negative "
             "number would mean the wholesale series predicts the CPI even when "
             "deliberately mis-dated, which would invalidate the whole exercise."),
            ("", "", ""),
            ("Correlation, CPI and wholesale, monthly, 2016-2026", cc(MCW, MCC, MCF), ""),
            ("Pass-through over a month, 2016-2026", cs(MCC, MCW, MCF),
             "Near 1.0 means the retail margin absorbs none of a monthly move."),
            ("Correlation, monthly, full history", f"=CORREL({MCC},{MCW})",
             "Lower than the 2016+ figure: the early wholesale panel is thinner."),
            ("Correlation, same fortnight", f"=CORREL({QC},{QW})", ""),
            ("Within-fortnight slope", f"=SLOPE({QC},{QW})",
             "The share of a wholesale move landing in the SAME fortnight's print."),
            ("Fortnights in the full panel", f"=COUNT({QC})", ""),
            ("", "", ""),
            ("Daily quotes in the raw panel",
             f"=COUNTA(daily_mercado!$A$2:$A${len(dm)+1})", f"{len(dm):,} rows."),
            ("Distinct wholesale markets", f"=COUNTA(mercados!$A$2:$A${len(mk)+1})",
             f"{len(mk)} markets."),
            ("Latest wholesale level, MXN/kg", f"=INDEX({DNP},COUNT({DNP}))", ""),
            ("Highest daily level in the sample", f"=MAX({DNP})", ""),
            ("Chain level drift, first vs last (log)",
             f"=INDEX({QND},MATCH(9.9E+307,{QND}))-INDEX({QND},MATCH(TRUE,INDEX(ISNUMBER({QND}),0),0))",
             "Chained level minus direct level. Cancels in log differences, which is all "
             "the model uses — but never read the chained series as a peso level."),
        ]
        if rt is not None:
            RR = rng("retail_mensual", rt, "retail_mxn_kg", NR)
            RW = rng("retail_mensual", rt, "wholesale_mxn_kg", NR)
            checks.append(("Retail / wholesale margin, mean ratio",
                           f"=AVERAGE({RR})/AVERAGE({RW})",
                           "Wholesale here is the market-weighted geometric mean of the "
                           "quotes, in actual pesos, not the chained index."))
        r = 3
        for k, f_, note in checks:
            if not k:
                r += 1; continue
            ws.write(r, 0, k, T); ws.write_formula(r, 1, f_, N3); ws.write(r, 2, note, WR)
            if len(note) > 80:
                ws.set_row(r, 26)
            r += 1
        ws.write(r + 1, 0, "Source: SNIIM (Secretaría de Economía) and INEGI. Built by "
                           "export_generic_xlsx.py.", T)
    return OUT


if __name__ == "__main__":
    for sl in (sys.argv[1:] or ["jitomate", "tomate_verde", "cebolla", "calabacita",
                                "chile_serrano"]):
        out = build(sl)
        print(f"{out}", flush=True)
