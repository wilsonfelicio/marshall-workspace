"""Wide daily panel: one column of p_index_ponderado per generic, 32 generics.

  precios_mayoreo_diario.xlsx

Same series as the per-product workbooks' `p_index_ponderado` column, side by side on
one date index: the market-weighted, variety-weighted geometric mean wholesale price in
MXN/kg for each INPC generic, daily.

Construction, per generic per day:
  1. per variety x market, the day's quoted price (already a geometric mean of that
     market's origins);
  2. across markets, a weighted geometric mean using INPC city weights;
  3. across varieties, a weighted geometric mean using each variety's share of all
     quotes over the whole sample — fixed, because a moving variety weight would put
     pure recomposition into the level;
  4. renormalised over whatever is present that day, so a variety or market that did
     not quote leaves the level alone instead of putting a hole in it.

This is a LEVEL series in pesos, not the chained index the models use. The two answer
different questions: the chained matched-cell index is right in CHANGES and its level
drifts with composition, while this one is right in LEVELS and its changes carry
composition effects. Do not difference this and expect the model's regressor.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = "precios_mayoreo_diario.xlsx"
ORDER = ["Aguacate", "Durazno", "Guayaba", "Limón", "Manzana", "Melón", "Naranja",
         "Papaya", "Pera", "Piña", "Plátanos", "Sandía", "Uva", "Otras frutas",
         "Calabacita", "Cebolla", "Chayote", "Chile poblano", "Chile serrano",
         "Ejotes", "Jitomate", "Lechuga y col", "Nopales", "Papa y otros tubérculos",
         "Pepino", "Tomate verde", "Zanahoria", "Otras verduras y legumbres",
         "Otros chiles frescos", "Cilantro, epazote y perejil", "Chile seco", "Frijol"]

# One generic at a time. The first version read all 14.6 million quotes and grouped on
# string keys in one pass, and the container's memory manager killed it silently — no
# traceback, just a missing output file. Per-generic reads keep the peak at one slice.
import pyarrow.parquet as pq

SRC = "data/curated/var_market_daily.parquet"
pw = pd.read_parquet("data/curated/pesos_mercado.parquet")
wmap = dict(zip(pw.destino, pd.to_numeric(pw.peso_inpc, errors="coerce").fillna(0.0)))

slugs = (pd.read_parquet("data/curated/cat_national_monthly.parquet",
                         columns=["categoria", "categoria_label"])
         .drop_duplicates())
slug_of = dict(zip(slugs.categoria_label, slugs.categoria))

levels, counts, vcount = {}, {}, {}
for nm in ORDER:
    sl = slug_of.get(nm)
    if sl is None:
        print(f"  {nm}: not in the catalogue", flush=True)
        continue
    d = pq.read_table(SRC, columns=["producto", "fecha", "destino", "precio_geo"],
                      filters=[("categoria", "==", sl)]).to_pandas()
    d = d[d.precio_geo > 0]
    if d.empty:
        print(f"  {nm}: no quotes", flush=True)
        continue
    d["fecha"] = pd.to_datetime(d.fecha)
    d["w"] = d.destino.map(wmap).fillna(0.0)
    d["lp"] = np.log(d.precio_geo.to_numpy(float))
    # variety weight = share of ALL quotes for this generic over the whole sample, fixed
    vs = d.producto.value_counts()
    vw = (vs / vs.sum()).to_dict()
    vcount[nm] = len(vs)
    dw = d[d.w > 0].copy()
    dw["num"] = dw.w * dw.lp
    gv = (dw.groupby(["producto", "fecha"], observed=True)
          .agg(num=("num", "sum"), den=("w", "sum"), nm_=("destino", "nunique"))
          .reset_index())
    gv = gv[gv.den > 0]
    gv["lpw"] = gv.num / gv.den
    gv["vw"] = gv.producto.map(vw).fillna(0.0)
    gv["vnum"] = gv.vw * gv.lpw
    lev = (gv.groupby("fecha", observed=True)
           .agg(vnum=("vnum", "sum"), vden=("vw", "sum"), nm_=("nm_", "sum")).reset_index())
    lev = lev[lev.vden > 0]
    levels[nm] = pd.Series(np.exp(lev.vnum / lev.vden).values, index=lev.fecha)
    counts[nm] = pd.Series(lev.nm_.values, index=lev.fecha)
    print(f"  {nm:<32}{len(lev):>6} days  {len(vs)} varieties  "
          f"last {levels[nm].iloc[-1]:.2f}", flush=True)
    del d, dw, gv, lev

cols = [c for c in ORDER if c in levels]
missing = [c for c in ORDER if c not in levels]
if missing:
    print(f"  NOT IN THE STORE: {missing}", flush=True)
wide = pd.DataFrame(levels)[cols].sort_index()
nmk = pd.DataFrame(counts)[cols].sort_index()
wide.index.name = nmk.index.name = "fecha"
vw_n = vcount
print(f"  panel {wide.shape[0]:,} days x {wide.shape[1]} generics", flush=True)

cov = pd.DataFrame({
    "generico": cols,
    "primer_dia": [wide[c].first_valid_index() for c in cols],
    "ultimo_dia": [wide[c].last_valid_index() for c in cols],
    "dias_con_dato": [int(wide[c].notna().sum()) for c in cols],
    "pct_de_dias": [100 * wide[c].notna().mean() for c in cols],
    "variedades": [int(vw_n[c]) for c in cols],
    "mercados_mediana": [float(nmk[c].median()) for c in cols],
    "nivel_ultimo": [float(wide[c].dropna().iloc[-1]) for c in cols],
    "nivel_medio": [float(wide[c].mean()) for c in cols],
    # how often a cell rests on almost nothing: 1998-99 days with one or two markets
    # reporting produce levels that look like errors and are not
    "dias_menos_5_mercados": [int((nmk[c] < 5).sum()) for c in cols],
    "pct_menos_5_mercados": [100 * float((nmk[c] < 5).mean()) for c in cols],
    "ultimo_dia_menos_5": [nmk[c][nmk[c] < 5].last_valid_index() for c in cols],
})
for c in ("primer_dia", "ultimo_dia", "ultimo_dia_menos_5"):
    cov[c] = pd.to_datetime(cov[c]).dt.date

W = wide.reset_index()
W["fecha"] = W.fecha.dt.date
N = nmk.reset_index()
N["fecha"] = N.fecha.dt.date

with pd.ExcelWriter(OUT, engine="xlsxwriter",
                    engine_kwargs={"options": {"default_date_format": "yyyy-mm-dd"}}) as xl:
    book = xl.book
    H = book.add_format({"bold": True, "font_name": "Arial", "font_size": 10,
                         "bg_color": "#21295C", "font_color": "white", "align": "left",
                         "valign": "vcenter", "text_wrap": True})
    T = book.add_format({"font_name": "Arial", "font_size": 10})
    B = book.add_format({"font_name": "Arial", "font_size": 10, "bold": True})
    TI = book.add_format({"font_name": "Arial", "font_size": 13, "bold": True,
                          "font_color": "#21295C"})
    WRP = book.add_format({"font_name": "Arial", "font_size": 10, "text_wrap": True,
                           "valign": "top"})
    N2 = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.00"})
    D = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "yyyy-mm-dd"})

    def sheet(df, nm, w0=11, wn=13, fmt=N2):
        ws = book.add_worksheet(nm)
        xl.sheets[nm] = ws
        for j, c in enumerate(df.columns):
            ws.write(0, j, c, H)
            ws.set_column(j, j, w0 if j == 0 else max(wn, min(24, len(str(c)) + 2)),
                          D if j == 0 else fmt)
        df.to_excel(xl, sheet_name=nm, index=False, startrow=1, header=False)
        ws.freeze_panes(1, 1)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    ws = book.add_worksheet("LEEME")
    ws.set_column(0, 0, 24)
    ws.set_column(1, 1, 106)
    ws.write(0, 0, "Wholesale price levels, daily, by INPC generic", TI)
    for i, (k, v) in enumerate([
        ("What this is", f"One column per generic: the wholesale price in MXN per kilogram, "
                         f"daily, {wide.shape[0]:,} days from {wide.index.min():%d %b %Y} to "
                         f"{wide.index.max():%d %b %Y}, {wide.shape[1]} generics."),
        ("Source", "SNIIM (Secretaría de Economía), daily quotes by market, variety and "
                   "origin, free and public. Market to INPC city mapping and city weights "
                   "from INEGI."),
        ("How each cell is built", "Across markets, a geometric mean weighted by each "
                                   "market's INPC city weight; then across varieties, a "
                                   "geometric mean weighted by each variety's share of all "
                                   "quotes over the whole sample, held fixed."),
        ("Blanks", "A blank is a day with no quote for that generic — Sundays, holidays, and "
                   "the weekly-only cadence before 2000. Nothing is interpolated or carried "
                   "forward. The date index is the union across generics, so a blank in one "
                   "column does not mean the day is missing everywhere."),
        ("THIN DAYS", "About 4% of days rest on fewer than five markets, almost all of "
                      "them before 2000, and on those the level can swing hard — jitomate "
                      "on 1998-01-07 reads 1.90 against 4.80 either side, on one market. "
                      "These are not errors and are not filtered out. Check n_mercados "
                      "before using any single early day; the cobertura sheet counts them "
                      "per generic."),
        ("Renormalisation", "Weights are renormalised over the markets and varieties actually "
                            "present that day, so an absent market lowers precision but does "
                            "not shift the level."),
        ("", ""),
        ("IMPORTANT", "This is a LEVEL series, not the index the nowcast models use. The "
                      "model's regressor is a chained matched-cell index, which is right in "
                      "CHANGES but whose level drifts with composition. This series is the "
                      "reverse: right in levels, but its day-to-day changes carry "
                      "composition effects, because the set of quoting markets moves. Do "
                      "not difference this column and expect the model's input."),
        ("Also", "Nominal pesos, never deflated. Frijol and Chile seco come from SNIIM's "
                 "weekly granos module, so those two columns have roughly one observation "
                 "per week rather than one per day."),
        ("", ""),
        ("SHEET: precios_diarios", "The panel. Column order is the one you gave, which is "
                                   "INPC clave order: fruits, then vegetables, then the two "
                                   "granos."),
        ("SHEET: n_mercados", "Same shape: how many market quotes stand behind each cell. "
                              "Use it to judge how much to trust a given day."),
        ("SHEET: cobertura", "One row per generic: coverage, variety count, median markets "
                             "per day, and the latest and mean level."),
        ("", ""),
        ("Rebuild", "python3 export_panel_xlsx.py"),
    ], start=2):
        ws.write(i, 0, k, B)
        ws.write(i, 1, v, WRP)
        ws.set_row(i, 44 if len(v) > 165 else (30 if len(v) > 95 else None))

    sheet(W, "precios_diarios")
    sheet(N, "n_mercados", fmt=None)
    ws = book.add_worksheet("cobertura")
    xl.sheets["cobertura"] = ws
    for j, c in enumerate(cov.columns):
        ws.write(0, j, c, H)
        ws.set_column(j, j, 30 if j == 0 else 15,
                      N2 if c.startswith(("nivel", "pct", "mercados")) else None)
    cov.to_excel(xl, sheet_name="cobertura", index=False, startrow=1, header=False)
    ws.freeze_panes(1, 0)

print(OUT)
print(cov.to_string(index=False, formatters={
    "pct_de_dias": "{:.0f}".format, "mercados_mediana": "{:.0f}".format,
    "nivel_ultimo": "{:.2f}".format, "nivel_medio": "{:.2f}".format}))
