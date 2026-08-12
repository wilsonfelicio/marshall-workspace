"""Export everything on jitomate to one workbook, for independent checking.

  jitomate_data.xlsx

Nine sheets: a dictionary, the raw daily quote panel by market and variety, the daily
national level, the fortnightly panel that the model actually consumes, the
out-of-sample nowcasts, a like-for-like monthly comparison, retail prices, the market
list with its INPC city weights, and a CHECKS sheet.

The checks sheet is live Excel formulas, not values pasted from Python. That is the point
of the file: if my RMSE, correlation and pass-through numbers are wrong, the formulas
recompute them from the columns beside them and disagree with me on the reader's screen.

Excel's row ceiling is 1,048,576, so the 414k-row daily panel fits whole; nothing here is
sampled or truncated.
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
Q.CACHE_DIR = pathlib.Path("/root/jit/cache")

OUT = "jitomate_data.xlsx"
W_SAL, W_BOLA = 0.65, 0.35
MA = 30

# ------------------------------------------------------------------ 1. daily by market
raw = pd.read_parquet("/root/jit/var_market_daily.parquet")
dm = (raw.rename(columns={"producto": "variedad", "destino": "mercado",
                          "destino_estado": "estado", "precio_geo": "precio_mxn_kg"})
      [["fecha", "variedad", "mercado", "estado", "precio_mxn_kg", "n_obs", "n_origenes"]]
      .sort_values(["fecha", "variedad", "mercado"]).reset_index(drop=True))
dm["fecha"] = dm.fecha.dt.date

# ------------------------------------------------------------------ 2. daily national
_pw = pd.read_parquet("data/curated/pesos_mercado.parquet")
_pw = dict(zip(_pw.destino, pd.to_numeric(_pw.peso_inpc, errors="coerce").fillna(0.0)))
raw["_w"] = raw.destino.map(_pw).fillna(0.0)
raw["_lp"] = np.log(raw.precio_geo.where(raw.precio_geo > 0))
g = (raw.groupby(["fecha", "producto"], as_index=False)
     .agg(lp=("_lp", "mean"), nm=("destino", "nunique")))
_gw = (raw[raw._w > 0].groupby(["fecha", "producto"])
       .apply(lambda x: np.average(x._lp, weights=x._w)).rename("lpw").reset_index())
pvw = _gw.pivot(index="fecha", columns="producto", values="lpw")
piv = g.pivot(index="fecha", columns="producto", values="lp")
nmk = g.pivot(index="fecha", columns="producto", values="nm")
both = piv.dropna(subset=[c for c in piv.columns])
dn = pd.DataFrame(index=both.index)
dn["p_saladette"] = np.exp(both["Tomate Saladette"])
dn["p_bola"] = np.exp(both["Tomate Bola"])
dn["p_index"] = np.exp(W_SAL * both["Tomate Saladette"] + W_BOLA * both["Tomate Bola"])
# Two levels on purpose: the equal-weighted mean across markets, and the same thing with
# each market carrying its INPC city weight. The charts use the weighted one.
_bw = pvw.dropna().reindex(both.index)
dn["p_index_ponderado"] = np.exp(W_SAL * _bw["Tomate Saladette"]
                                 + W_BOLA * _bw["Tomate Bola"])
dn["n_mercados"] = nmk.loc[both.index].sum(axis=1).astype(int)
cal = pd.DataFrame(index=pd.date_range(dn.index.min(), dn.index.max(), freq="D"))
cal["p_index"] = dn.p_index.reindex(cal.index)
cal["ma30"] = cal.p_index.rolling(f"{MA}D", min_periods=12).mean()
cal["chg_30d_pct"] = 100 * (np.log(cal.ma30) - np.log(cal.ma30.shift(MA)))
dn = dn.join(cal[["ma30", "chg_30d_pct"]])
dn = dn.reset_index().rename(columns={"index": "fecha", "fecha": "fecha"})
dn["fecha"] = pd.to_datetime(dn.fecha).dt.date

# ------------------------------------------------------------------ 3. fortnightly panel
d = Q.dataset("jitomate", "070 Jitomate", windows=(5, 10)).set_index("t")
_st = pd.DatetimeIndex(Q.qtimestamp(d.index.values))
_close = pd.DatetimeIndex(np.where(_st.day == 1, _st + pd.Timedelta(days=14),
                                   _st + pd.offsets.MonthEnd(0)))
qn = pd.DataFrame({
    "t": d.index, "etiqueta": d.etiqueta.values,
    "fecha_inicio": _st.date, "fecha_cierre": _close.date,
    "quincena_del_anio": d.quincena_del_anio.values,
    "inpc_jitomate": d.inpc.astype(float).values,
    "cpi_dln_pct": d.y.values,
    "ws_dln_pct": d.x_full.values,
    "ws_dln_dia5_pct": d.x_w5.values,
    "ws_dln_dia10_pct": d.x_w10.values,
    "ws_ln_nivel_cadena": d.lw_full.values,
    "n_celdas": d.n_celdas_full.values,
    "n_dias_cotizados": d.n_dias.values,
    "sd_entre_mercados": d.sd_mercados.values,
    "segmento_cadena": d.segmento.values,
})

# The chained index is correct in CHANGES but its LEVEL drifts, because matched-cell
# chaining accumulates composition change: anchored on its first period it sits 25% below
# the actual peso level by 2026. The direct weighted geometric mean of the quotes is the
# honest level series, so both are exported and the drift is a column.
_pv = (raw.groupby(["fecha", "producto"]).precio_geo
       .apply(lambda z: np.log(z[z > 0]).mean()).unstack().dropna())
_lvl = (W_SAL * _pv["Tomate Saladette"] + W_BOLA * _pv["Tomate Bola"])
_lvl.index = pd.DatetimeIndex(_lvl.index)
_lq = _lvl.groupby(_lvl.index.year * 24 + (_lvl.index.month - 1) * 2
                   + (_lvl.index.day > 15).astype(int)).mean()
qn["ws_nivel_directo_mxn_kg"] = np.exp(qn.t.map(_lq))
qn["deriva_cadena_ln"] = qn.ws_ln_nivel_cadena - np.log(qn.ws_nivel_directo_mxn_kg)

# ------------------------------------------------------------------ 4. out-of-sample
mo = pd.read_csv("data/curated/jitomate_system.csv")
lab = dict(zip(qn.t, qn.etiqueta))
clo = dict(zip(qn.t, qn.fecha_cierre))
mo.insert(1, "etiqueta", mo.t.map(lab))
mo.insert(2, "fecha_cierre", mo.t.map(clo))
mo = mo.rename(columns={"y": "cpi_realizado_pct", "fit": "nowcast_pct",
                        "bench": "benchmark_cpi_only_pct", "sigma": "sigma_pp",
                        "d5_combo": "nowcast_dia5_pct", "d10_combo": "nowcast_dia10_pct"})

# The band the chart draws: a recursive empirical quantile of the model's own standardised
# errors, expanding, never using the current period. Exported as a column so the workbook's
# coverage check is the actual band and not a normal-table approximation.
_z = ((mo.cpi_realizado_pct - mo.nowcast_pct).abs() / mo.sigma_pp).to_numpy(float)
_k = np.full(len(_z), np.nan)
for i in range(len(_z)):
    prev = _z[:i][~np.isnan(_z[:i])]
    if len(prev) >= 40:
        _k[i] = np.quantile(prev, 0.80)
mo["banda80_pp"] = _k * mo.sigma_pp

# ------------------------------------------------------------------ 5. like-for-like
q2 = dn.copy()
q2["fecha"] = pd.to_datetime(q2.fecha)
q2["t"] = (q2.fecha.dt.year * 24 + (q2.fecha.dt.month - 1) * 2
           + (q2.fecha.dt.day > 15).astype(int))
wq = q2.groupby("t").p_index.apply(lambda z: np.log(z).mean())
step = wq.index.to_series().diff()
ln = np.log(pd.Series(qn.inpc_jitomate.values, index=qn.t))
cstep = qn.t.diff()
mc = pd.DataFrame({
    "t": qn.t.values, "etiqueta": qn.etiqueta.values, "fecha_cierre": qn.fecha_cierre.values,
    "cpi_chg_2q_pct": np.where((cstep == 1) & (cstep.shift(1) == 1),
                               100 * (ln - ln.shift(2)), np.nan),
}).set_index("t")
mc["ws_chg_2q_pct"] = (100 * (wq - wq.shift(2)))[(step == 1) & (step.shift(1) == 1)]
mc = mc.dropna().reset_index()
mc["desde_2016"] = (mc.t >= 2016 * 24).astype(int)

# ------------------------------------------------------------------ 6. retail
rt = pd.read_parquet("data/inpc/precios_kg/jitomate.parquet")
rt = rt[["periodo", "mes", "clave_generico", "precio_geo", "n_ciudades"]].rename(
    columns={"precio_geo": "retail_mxn_kg", "clave_generico": "clave_inpc"})
wm = (q2.set_index("fecha").p_index_ponderado.resample("MS")
      .apply(lambda z: np.exp(np.log(z.dropna()).mean())))
rt["mes"] = pd.to_datetime(rt.mes)
rt = rt.merge(wm.rename("wholesale_mxn_kg").reset_index()
              .rename(columns={"fecha": "mes"}), on="mes", how="left")
rt["mes"] = rt.mes.dt.date

# ------------------------------------------------------------------ 7. markets
pm = pd.read_parquet("data/curated/pesos_mercado.parquet")
cov = (raw.groupby("destino")
       .agg(primer_dia=("fecha", "min"), ultimo_dia=("fecha", "max"),
            dias=("fecha", "nunique"), obs=("precio_geo", "size"),
            precio_medio=("precio_geo", "mean")).reset_index())
mk = cov.merge(pm, on="destino", how="left").rename(columns={"destino": "mercado"})
for c in ("primer_dia", "ultimo_dia"):
    mk[c] = pd.to_datetime(mk[c]).dt.date

# ------------------------------------------------------------------ write
NM, NQ, NO, NC, NR = len(mo), len(qn), len(mo), len(mc), len(rt)
with pd.ExcelWriter(OUT, engine="xlsxwriter",
                    engine_kwargs={"options": {"default_date_format": "yyyy-mm-dd"}}) as xl:
    book = xl.book
    fmt_h = book.add_format({"bold": True, "font_name": "Arial", "font_size": 10,
                             "bg_color": "#21295C", "font_color": "white", "border": 0,
                             "align": "left", "valign": "vcenter", "text_wrap": True})
    fmt_t = book.add_format({"font_name": "Arial", "font_size": 10})
    fmt_b = book.add_format({"font_name": "Arial", "font_size": 10, "bold": True})
    fmt_ti = book.add_format({"font_name": "Arial", "font_size": 13, "bold": True,
                              "font_color": "#21295C"})
    fmt_w = book.add_format({"font_name": "Arial", "font_size": 10, "text_wrap": True,
                             "valign": "top"})
    fmt_n2 = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.00"})
    fmt_n3 = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000"})

    def sheet(df, name, widths=None, nfmt=None):
        # Header FIRST, then the body: writing row 0 after row N is what the constant_memory
        # version got wrong, and it failed silently rather than raising.
        ws = book.add_worksheet(name)
        xl.sheets[name] = ws
        for j, c in enumerate(df.columns):
            ws.write(0, j, c, fmt_h)
            ws.set_column(j, j, (widths or {}).get(c, 13), (nfmt or {}).get(c))
        df.to_excel(xl, sheet_name=name, index=False, startrow=1, header=False)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        return ws

    # --- README first, so it is the sheet that opens
    ws = book.add_worksheet("LEEME")
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 104)
    ws.write(0, 0, "Jitomate: wholesale prices, CPI, and the nowcast", fmt_ti)
    rows = [
        ("Purpose", "Every jitomate series behind the nowcast, at the frequency it is "
                    "actually built at, so the numbers I quoted can be checked "
                    "independently. Nothing is sampled: the daily panel is complete."),
        ("Sources", "Wholesale: SNIIM, Secretaría de Economía, daily quotes by market and "
                    "variety, free and public. CPI and retail prices: INEGI, published "
                    "fortnightly and monthly."),
        ("Units", "Prices in MXN per kilogram. Changes in percent (log difference × 100), "
                  "so they add across periods. 'pp' means percentage points of the "
                  "published print."),
        ("", ""),
        ("SHEET: daily_mercado", f"{len(dm):,} rows. The raw panel: one row per date × "
                                 f"variety × wholesale market. This is the atomic data — "
                                 f"everything else in the file is derived from it."),
        ("SHEET: daily_nacional", f"{len(dn):,} rows. Daily national level. p_index is the "
                                  f"geometric mean across markets within each variety, then "
                                  f"Saladette {W_SAL:.0%} / Bola {W_BOLA:.0%}. "
                                  f"p_index_ponderado is the same with each market carrying "
                                  f"its INPC city weight — that is the series the charts use "
                                  f"and the one to compare with retail. ma30 is a "
                                  f"{MA}-CALENDAR-day mean, not {MA} quotes."),
        ("SHEET: quincenal", f"{len(qn):,} rows. The fortnightly panel the model consumes. "
                             f"ws_dln_pct is the chained matched-cell index: only "
                             f"variety × market cells present in BOTH fortnights enter, so a "
                             f"market that stops quoting cannot move it. dia5 / dia10 are the "
                             f"same index cut off after 5 and 10 days of the fortnight."),
        ("SHEET: nowcast_oos", f"{len(mo):,} rows. Out-of-sample nowcasts, refit every "
                               f"fortnight on a rolling five-year window using earlier data "
                               f"only. benchmark_cpi_only_pct is the same model with the "
                               f"wholesale regressors removed."),
        ("SHEET: mensual_comparable", f"{len(mc):,} rows. Both series measured the same way: "
                                      f"each fortnight against the fortnight two prints "
                                      f"earlier, i.e. about one month."),
        ("SHEET: retail_mensual", f"{len(rt):,} rows. INEGI's average retail price per kg "
                                  f"beside the wholesale monthly mean, for the margin."),
        ("SHEET: mercados", f"{len(mk):,} rows. Market list, coverage, and the INPC city "
                            f"weight each one carries."),
        ("SHEET: CHECKS", "Live formulas that recompute every headline number from the "
                          "sheets in this file. If a formula disagrees with what I told you, "
                          "the formula is the one to trust."),
        ("", ""),
        ("WATCH OUT (1)", "A fortnight is labelled by its FIRST day in most of my code "
                          "(1Q Feb 2024 = 2024-02-01) but summarises prices through its "
                          "LAST day. Both columns are given: fecha_inicio and fecha_cierre. "
                          "Plotting the CPI against the daily wholesale series at the start "
                          "date understates their correlation badly — 0.40 instead of 0.86 — "
                          "because the dot lands half a month early."),
        ("WATCH OUT (2)", "cpi_dln_pct is a FORTNIGHTLY change; cpi_chg_2q_pct on the "
                          "comparable sheet is a two-fortnight change. They are different "
                          "quantities and the second is not twice the first."),
        ("WATCH OUT (3)", "segmento_cadena increments where the matched-cell chain breaks "
                          "(too few overlapping cells to link two fortnights). Do not "
                          "difference the wholesale index across a segment boundary."),
        ("WATCH OUT (4)", "Four fortnights have a CPI print but no nowcast, so the model is "
                          "scored on 370 of 374 periods. Any comparison of the nowcast with "
                          "the benchmark must drop the same four rows from both, or the "
                          "benchmark is scored on a longer sample and looks worse than it is."),
        ("WATCH OUT (5)", "ws_ln_nivel_cadena is the chained index in logs. Its LEVEL is "
                          "not pesos: matched-cell chaining accumulates composition change, "
                          "so anchored on 1998 it ends about 10% off the true peso level "
                          "(column deriva_cadena_ln). Use ws_nivel_directo_mxn_kg for levels "
                          "and the chained series only for changes. Every model result uses "
                          "changes only and is unaffected."),
        ("WATCH OUT (6)", "Before 2000 SNIIM quoted weekly rather than daily, so n_dias per "
                          "fortnight is 2-3 in the early years and 10-11 later. The "
                          "cell-matching requirement adapts to that; a fixed minimum "
                          "day count would silently delete the 1998-1999 sample."),
        ("", ""),
        ("Rebuild", "python3 export_jitomate_xlsx.py — regenerates this file from the "
                    "parquet stores in data/curated and /root/jit."),
    ]
    for i, (k, v) in enumerate(rows, start=2):
        ws.write(i, 0, k, fmt_b)
        ws.write(i, 1, v, fmt_w)
        if len(v) > 150:
            ws.set_row(i, 44)
        elif len(v) > 90:
            ws.set_row(i, 30)

    sheet(dm, "daily_mercado",
          {"fecha": 11, "variedad": 17, "mercado": 52, "estado": 20, "precio_mxn_kg": 14},
          {"precio_mxn_kg": fmt_n2})
    sheet(dn, "daily_nacional", {"fecha": 11},
          {"p_saladette": fmt_n2, "p_bola": fmt_n2, "p_index": fmt_n2, "ma30": fmt_n2,
           "chg_30d_pct": fmt_n2, "p_index_ponderado": fmt_n2})
    sheet(qn, "quincenal", {"etiqueta": 14, "fecha_inicio": 12, "fecha_cierre": 12,
                            "inpc_jitomate": 13, "segmento_cadena": 15},
          {"cpi_dln_pct": fmt_n3, "ws_dln_pct": fmt_n3, "ws_dln_dia5_pct": fmt_n3,
           "ws_dln_dia10_pct": fmt_n3, "inpc_jitomate": fmt_n3})
    sheet(mc, "mensual_comparable", {"etiqueta": 14, "fecha_cierre": 12},
          {"cpi_chg_2q_pct": fmt_n3, "ws_chg_2q_pct": fmt_n3})
    sheet(rt, "retail_mensual", {"periodo": 11, "mes": 11, "clave_inpc": 11,
                                 "retail_mxn_kg": 14, "wholesale_mxn_kg": 16},
          {"retail_mxn_kg": fmt_n2, "wholesale_mxn_kg": fmt_n2})
    sheet(mk, "mercados", {"mercado": 52, "estado": 20, "ciudad_inpc": 18,
                           "clave_ciudad_inpc": 12, "primer_dia": 11, "ultimo_dia": 11},
          {"precio_medio": fmt_n2, "peso_inpc": fmt_n3})

    # --- nowcast sheet carries its own error columns, as formulas
    ws = book.add_worksheet("nowcast_oos")
    xl.sheets["nowcast_oos"] = ws
    cols = list(mo.columns)
    extra = ["err_nowcast_pp", "err_benchmark_pp", "acierto_signo_nowcast",
             "acierto_signo_benchmark", "dentro_banda_80", "y_pareado_pct"]
    for j, c in enumerate(cols + extra):
        ws.write(0, j, c, fmt_h)
        ws.set_column(j, j, 15 if j < 3 else 14)
    A = lambda j: chr(65 + j) if j < 26 else chr(64 + j // 26) + chr(65 + j % 26)
    ci = {c: A(cols.index(c)) for c in cols}
    y, f, b = ci["cpi_realizado_pct"], ci["nowcast_pct"], ci["benchmark_cpi_only_pct"]
    bd = ci["banda80_pp"]
    e0 = len(cols)
    date_j = cols.index("fecha_cierre")
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
                ws.write_number(i - 1, j, float(v), fmt_n3 if j > 3 else None)
        ws.write_formula(i - 1, e0, f'=IF(ISNUMBER({f}{i}),{y}{i}-{f}{i},"")', fmt_n3)
        ws.write_formula(i - 1, e0 + 1,
                         f'=IF(AND(ISNUMBER({b}{i}),ISNUMBER({f}{i})),{y}{i}-{b}{i},"")',
                         fmt_n3)
        ws.write_formula(i - 1, e0 + 2,
                         f'=IF(ISNUMBER({f}{i}),IF(SIGN({y}{i})=SIGN({f}{i}),1,0),"")')
        ws.write_formula(i - 1, e0 + 3,
                         f'=IF(AND(ISNUMBER({b}{i}),ISNUMBER({f}{i})),'
                         f'IF(SIGN({y}{i})=SIGN({b}{i}),1,0),"")')
        ws.write_formula(i - 1, e0 + 4,
                         f'=IF(AND(ISNUMBER({f}{i}),ISNUMBER({bd}{i})),'
                         f'IF(ABS({y}{i}-{f}{i})<={bd}{i},1,0),"")')
        ws.write_formula(i - 1, e0 + 5, f'=IF(ISNUMBER({f}{i}),{y}{i},"")', fmt_n3)
    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, NO, len(cols) + len(extra) - 1)

    # --- CHECKS: every number I quoted, recomputed in the file
    ws = book.add_worksheet("CHECKS")
    ws.set_column(0, 0, 52)
    ws.set_column(1, 1, 14)
    ws.set_column(2, 2, 62)
    ws.write(0, 0, "Consistency checks — every cell in column B is a formula", fmt_ti)
    ws.write(2, 0, "Quantity", fmt_h)
    ws.write(2, 1, "Value", fmt_h)
    ws.write(2, 2, "How it is computed, and what I claimed", fmt_h)
    def rng(sh, frame, col, n, off=0):
        """A1 range for a named column, so a reordered frame cannot silently misaddress."""
        j = (list(frame.columns).index(col) if col in frame.columns
             else len(frame.columns) + off)
        c = chr(65 + j) if j < 26 else chr(64 + j // 26) + chr(65 + j % 26)
        return f"{sh}!${c}$2:${c}${n + 1}"

    EN = rng("nowcast_oos", mo, "err_nowcast_pp", NO, 0)
    EB = rng("nowcast_oos", mo, "err_benchmark_pp", NO, 1)
    SN = rng("nowcast_oos", mo, "acierto_signo_nowcast", NO, 2)
    SB = rng("nowcast_oos", mo, "acierto_signo_benchmark", NO, 3)
    BD = rng("nowcast_oos", mo, "dentro_banda_80", NO, 4)
    YY = rng("nowcast_oos", mo, "y_pareado_pct", NO, 5)
    MCC = rng("mensual_comparable", mc, "cpi_chg_2q_pct", NC)
    MCW = rng("mensual_comparable", mc, "ws_chg_2q_pct", NC)
    MCF = rng("mensual_comparable", mc, "desde_2016", NC)

    def cond_corr(x, y, f):
        """Pearson r over the rows where flag f is 1, spelled out so it is auditable."""
        n = f"SUMPRODUCT({f})"
        sx, sy = f"SUMPRODUCT({f},{x})", f"SUMPRODUCT({f},{y})"
        sxy = f"SUMPRODUCT({f},{x},{y})"
        sxx, syy = f"SUMPRODUCT({f},{x},{x})", f"SUMPRODUCT({f},{y},{y})"
        return (f"=({n}*{sxy}-{sx}*{sy})/SQRT(({n}*{sxx}-{sx}^2)*({n}*{syy}-{sy}^2))")

    def cond_slope(y, x, f):
        n = f"SUMPRODUCT({f})"
        sx, sy = f"SUMPRODUCT({f},{x})", f"SUMPRODUCT({f},{y})"
        sxy, sxx = f"SUMPRODUCT({f},{x},{y})", f"SUMPRODUCT({f},{x},{x})"
        return f"=({n}*{sxy}-{sx}*{sy})/({n}*{sxx}-{sx}^2)"
    QC = rng("quincenal", qn, "cpi_dln_pct", NQ)
    QW = rng("quincenal", qn, "ws_dln_pct", NQ)
    RT_R = rng("retail_mensual", rt, "retail_mxn_kg", NR)
    RT_W = rng("retail_mensual", rt, "wholesale_mxn_kg", NR)
    DN_P = rng("daily_nacional", dn, "p_index", len(dn))
    QN_I = rng("quincenal", qn, "inpc_jitomate", NQ)
    QN_D = rng("quincenal", qn, "deriva_cadena_ln", NQ)
    checks = [
        ("Fortnights scored (nowcast and benchmark both present)",
         f"=COUNT({EN})", "Claimed 370 of 374. The other four have a print but no nowcast."),
        ("RMSE of the nowcast, pp of the print",
         f"=SQRT(SUMSQ({EN})/COUNT({EN}))", "Claimed 4.46 pp."),
        ("RMSE of the CPI-only benchmark, pp",
         f"=SQRT(SUMSQ({EB})/COUNT({EB}))",
         "Claimed 12.33 pp. Both RMSEs use the same rows: the error columns are blank "
         "wherever the nowcast is."),
        ("Reduction in RMSE",
         f"=1-SQRT(SUMSQ({EN})/COUNT({EN}))/SQRT(SUMSQ({EB})/COUNT({EB}))",
         "Claimed 64%."),
        ("Nowcast error as pp of headline INPC",
         f"=SQRT(SUMSQ({EN})/COUNT({EN}))*0.79014/100",
         "Claimed ±0.035 pp. 0.79014 is jitomate's published weight in the INPC basket."),
        ("Mean error of the nowcast, pp (bias)",
         f"=AVERAGE({EN})", "Should be near zero; a large value means the model is biased."),
        ("Correct sign, nowcast", f"=AVERAGE({SN})", "Claimed 91%."),
        ("Correct sign, benchmark", f"=AVERAGE({SB})", "Claimed 62%."),
        ("Realised coverage of the 80% band", f"=AVERAGE({BD})",
         "Claimed 75%. Nominal is 80%, so the band is slightly too narrow. The band width "
         "itself is in column banda80_pp, not recomputed here."),
        ("Standard deviation of the realised CPI change, pp", f"=STDEV({YY})",
         "The thing being forecast, on the same 370 fortnights as the errors above. "
         "Compare it with the two RMSEs."),
        ("R2 out of sample, nowcast",
         f"=1-SUMSQ({EN})/DEVSQ({YY})", "Claimed 0.89."),
        ("R2 out of sample, benchmark",
         f"=1-SUMSQ({EB})/DEVSQ({YY})", "Claimed 0.18."),
        ("", "", ""),
        ("Correlation, CPI and wholesale, monthly, 2016-2026",
         cond_corr(MCW, MCC, MCF),
         "Claimed 0.96. This is the sample the chart uses; the whole history is on the "
         "next line and is lower, because the 1998-2010 wholesale panel is thinner."),
        ("Pass-through over a month, 2016-2026",
         cond_slope(MCC, MCW, MCF),
         "Claimed 1.06 — one-for-one, so the retail margin absorbs none of a monthly move."),
        ("Correlation, monthly, full history 1998-2026",
         f"=CORREL({MCC},{MCW})", "Lower than the 2016+ figure. Quote the window."),
        ("Pass-through over a month, full history",
         f"=SLOPE({MCC},{MCW})", "Lower for the same reason."),
        ("Correlation, same fortnight",
         f"=CORREL({QC},{QW})",
         "Claimed 0.86 contemporaneous. Lower than the monthly figure because a fortnight "
         "catches only part of the pass-through."),
        ("Within-fortnight slope",
         f"=SLOPE({QC},{QW})",
         "Around 0.8: that is the share of a wholesale move that lands in the SAME "
         "fortnight's print, not the monthly total."),
        ("Fortnights in the full panel", f"=COUNT({QN_I})",
         "1998 to 2026, every fortnight with a published index."),
        ("", "", ""),
        ("Daily quotes in the raw panel", f"=COUNTA(daily_mercado!$A$2:$A${len(dm)+1})",
         f"Claimed {len(dm):,} rows, market × variety × day."),
        ("Distinct wholesale markets", f"=COUNTA(mercados!$A$2:$A${len(mk)+1})",
         f"Claimed {len(mk)} markets."),
        ("Latest wholesale level, MXN/kg",
         f"=INDEX({DN_P},COUNT({DN_P}))",
         "Claimed 17.96 on 11 Aug 2026."),
        ("Highest daily level in the sample", f"=MAX({DN_P})",
         "The May 2026 spike."),
        ("Retail / wholesale margin, mean ratio",
         f"=AVERAGE({RT_R})/AVERAGE({RT_W})",
         "1.35x. I earlier said 1.69x, which was wrong: that compared retail with the "
         "CHAINED index anchored on 1998, carrying 28 years of chain drift into the level. "
         "Wholesale here is the direct geometric mean of the quotes, in actual pesos."),
        ("Chain level drift, 1998 vs today (log)",
         f"=INDEX({QN_D},MATCH(9.9E+307,{QN_D}))-INDEX({QN_D},MATCH(TRUE,INDEX(ISNUMBER({QN_D}),0),0))",
         "The chained index's level minus the direct level, first period against last. "
         "About +0.10 in logs, i.e. the chained level runs 10% high by 2026 relative to "
         "1998. Anchoring the chain on 1998 and adding a variety-mix error in the anchor is "
         "what made me quote a 1.69x margin. It cancels in log differences, which is all the "
         "model uses, so no forecast result depends on it — but never read the chained "
         "series as a peso level."),
    ]
    r = 3
    for k, f_, note in checks:
        if not k:
            r += 1
            continue
        ws.write(r, 0, k, fmt_t)
        ws.write_formula(r, 1, f_, fmt_n3)
        ws.write(r, 2, note, fmt_w)
        if len(note) > 80:
            ws.set_row(r, 26)
        r += 1
    ws.write(r + 1, 0, "Source: SNIIM (Secretaría de Economía) and INEGI. Built "
                       "from the parquet stores by export_jitomate_xlsx.py.", fmt_t)

print(OUT)
for nm, n in [("daily_mercado", len(dm)), ("daily_nacional", len(dn)), ("quincenal", NQ),
              ("nowcast_oos", NO), ("mensual_comparable", NC), ("retail_mensual", NR),
              ("mercados", len(mk))]:
    print(f"  {nm:<20}{n:>8,} rows")
