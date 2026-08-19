"""Wide daily panel of wholesale price levels: 32 produce generics and 10 protein series.

  precios_mayoreo_diario.xlsx

The two halves come from different SNIIM applications and are NOT built the same way.
They share a workbook because they share a date index and a unit (MXN/kg) and because the
question they answer is one question — what does food cost at wholesale today — but the
cobertura sheet marks every column with which construction produced it, and the LEEME
says why the difference cannot be averaged away:

  produce   Agrícolas module. Across markets, a geometric mean weighted by INPC city
            weight; then across varieties, weighted by each variety's fixed share.
  protein   Pecuarios and Pesqueros. Across markets, an EQUAL-weighted geometric mean,
            because those markets (rastros, packers, distribution centres) are not in
            INEGI's city crosswalk and weighting them would be invention. Most quotes are
            a min-max range, so each is the geometric centre of its range.

Protein columns are blank before 2024: that is when collection starts, not when the
prices start.

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

import glob

import numpy as np
import pandas as pd

OUT = "precios_mayoreo_diario.xlsx"
MIN_COVER = 0.40                      # of a protein series' own typical market count
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

# ---------------------------------------------------------------- proteins
# series key -> (column, INPC generic it feeds, products to keep or None for all)
PROT = [
    ("pollo",          "Pollo",              "022 Pollo",
     ("Pollo entero", "Pollo tipo rosticero")),
    ("huevo",          "Huevo",              "031 Huevo", None),
    ("res_canal",      "Res canal",          "018 Carne de res", None),
    ("res_cortes",     "Res cortes",         "018 Carne de res", None),
    ("res_visceras",   "Res vísceras",       "025 Vísceras de res", ("Vísceras",)),
    ("cerdo_canal",    "Cerdo canal",        "017 Carne de cerdo", None),
    ("cerdo_grasa",    "Cerdo grasa",        "043 Manteca de cerdo (proxy)", ("Grasa",)),
    ("camaron",        "Camarón",            "027 Camarón", None),
    ("pescado_filete", "Pescado filete",     "028 Pescado", None),
    ("pescado_dulce",  "Pescado agua dulce", "028 Pescado", None),
]
p_lvl, p_cnt, p_nprod, p_drop, p_lastthin = {}, {}, {}, {}, {}
for key, col, _gen, keep in PROT:
    files = sorted(glob.glob(f"data/raw/proteinas/{key}/anio=*/part.parquet"))
    if not files:
        continue
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d[d.precio > 0]
    if keep:
        d = d[d.producto.isin(keep)]
    if not len(d):
        print(f"  {col:<32} no rows after the product filter, skipped", flush=True)
        continue
    g = (d.assign(lp=np.log(d.precio))
         .groupby("fecha").agg(lp=("lp", "mean"), nm=("destino", "nunique")))
    # A day resting on a handful of markets is not a national price, and the newest day is
    # the usual offender: partway through 13 Aug 2026 only the early-reporting markets were
    # in and pollo read 50.79 against 61.85 the day before. Anyone taking the last row as
    # "today's price" would be reading market composition. Drop days under MIN_COVER of the
    # series' own typical count and report them in cobertura rather than dropping silently.
    typical = g.nm.rolling(60, min_periods=5, center=True).median().bfill().ffill()
    thin = g.nm < (MIN_COVER * typical)
    p_drop[col] = int(thin.sum())
    p_lastthin[col] = g.index[thin].max() if thin.any() else pd.NaT
    g = g[~thin]
    p_lvl[col], p_cnt[col] = np.exp(g.lp), g.nm
    p_nprod[col] = int(d.producto.nunique())
    print(f"  {col:<32}{len(g):>6} days  {p_nprod[col]} products  "
          f"last {p_lvl[col].iloc[-1]:.2f}", flush=True)
    del d, g
if not p_lvl:
    print("  no protein data in this store — produce columns only", flush=True)

cols = [c for c in ORDER if c in levels]
missing = [c for c in ORDER if c not in levels]
if missing:
    print(f"  NOT IN THE STORE: {missing}", flush=True)
wide = pd.DataFrame(levels)[cols].sort_index()
nmk = pd.DataFrame(counts)[cols].sort_index()
vw_n = vcount

p_cols = [c for _k, c, _g, _p in PROT if c in p_lvl]
clash = set(p_cols) & set(cols)
assert not clash, f"a protein series shares a name with a produce generic: {clash}"
if p_cols:
    # outer join on the date index: proteins start in 2024, produce in 1998, and the
    # blanks in between are the honest answer rather than a reason to split the sheet
    pw_ = pd.DataFrame(p_lvl)[p_cols].sort_index()
    pn_ = pd.DataFrame(p_cnt)[p_cols].sort_index()
    idx = wide.index.union(pw_.index)
    wide = pd.concat([wide.reindex(idx), pw_.reindex(idx)], axis=1)
    nmk = pd.concat([nmk.reindex(idx), pn_.reindex(idx)], axis=1)
wide.index.name = nmk.index.name = "fecha"
print(f"  panel {wide.shape[0]:,} days x {len(cols)} produce + {len(p_cols)} protein",
      flush=True)

gen_of = {c: g for _k, c, g, _p in PROT}
allc = cols + p_cols
cov = pd.DataFrame({
    "columna": allc,
    "tipo": ["produce"] * len(cols) + ["proteína"] * len(p_cols),
    "generico_inpc": cols + [gen_of[c] for c in p_cols],
    "ponderacion": (["peso INPC de ciudad"] * len(cols)
                    + ["igual por mercado"] * len(p_cols)),
    "primer_dia": [wide[c].first_valid_index() for c in allc],
    "ultimo_dia": [wide[c].last_valid_index() for c in allc],
    "dias_con_dato": [int(wide[c].notna().sum()) for c in allc],
    "pct_de_dias": [100 * wide[c].notna().mean() for c in allc],
    # varieties for produce, products for proteins: the same idea, different vocabulary
    "variantes": [int(vw_n[c]) for c in cols] + [p_nprod[c] for c in p_cols],
    "mercados_mediana": [float(nmk[c].median()) for c in allc],
    "nivel_ultimo": [float(wide[c].dropna().iloc[-1]) for c in allc],
    "nivel_medio": [float(wide[c].mean()) for c in allc],
    # how often a cell rests on almost nothing: 1998-99 days with one or two markets
    # reporting produce levels that look like errors and are not
    "dias_menos_5_mercados": [int((nmk[c] < 5).sum()) for c in allc],
    "pct_menos_5_mercados": [100 * float((nmk[c] < 5).mean()) for c in allc],
    "ultimo_dia_menos_5": [nmk[c][nmk[c] < 5].last_valid_index() for c in allc],
    # only the protein half drops thin days; the produce half keeps and flags them, and
    # this column is where that asymmetry is visible instead of buried
    "dias_delgados_excluidos": [None] * len(cols) + [p_drop[c] for c in p_cols],
    "ultimo_dia_excluido": [None] * len(cols) + [p_lastthin[c] for c in p_cols],
})
for c in ("primer_dia", "ultimo_dia", "ultimo_dia_menos_5", "ultimo_dia_excluido"):
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
    ws.write(0, 0, "Wholesale price levels, daily: produce and proteins", TI)
    for i, (k, v) in enumerate([
        ("What this is", f"One column per series: the wholesale price in MXN per kilogram, "
                         f"daily, {wide.shape[0]:,} days from {wide.index.min():%d %b %Y} to "
                         f"{wide.index.max():%d %b %Y}. {len(cols)} produce generics "
                         f"followed by {len(p_cols)} protein series."),
        ("Source", "SNIIM (Secretaría de Economía), free and public. Produce comes from the "
                   "Agrícolas module, daily quotes by market, variety and origin. Proteins "
                   "come from the Pecuarios and Pesqueros modules, which are a different "
                   "application with different markets: rastros, packers and distribution "
                   "centres rather than central de abasto wholesalers. Market to INPC city "
                   "mapping and city weights from INEGI."),
        ("THE TWO HALVES ARE NOT BUILT THE SAME WAY", "Produce: across markets a geometric "
         "mean weighted by each market's INPC city weight, then across varieties a "
         "geometric mean weighted by each variety's fixed share of all quotes. Proteins: an "
         "EQUAL-weighted geometric mean across markets, because those markets are not in "
         "INEGI's city crosswalk and weighting them would be invention. The cobertura sheet "
         "marks every column with which one applies. Do not read a produce column and a "
         "protein column as two measurements of the same kind of thing."),
        ("Protein price ranges", "Most protein series publish a MINIMUM and a MAXIMUM rather "
                                 "than one quote, so each underlying quote is the geometric "
                                 "centre of its range. Huevo publishes a modal 'precio "
                                 "frecuente' and that is used instead. The min and max "
                                 "survive per observation in data/raw/proteinas."),
        ("Proteins start in 2024", "Protein columns are blank before 2024. That is when "
                                   "collection starts, not when the prices start. Produce "
                                   "runs from 1998."),
        ("Blanks", "A blank is a day with no quote for that generic — Sundays, holidays, and "
                   "the weekly-only cadence before 2000. Nothing is interpolated or carried "
                   "forward. The date index is the union across generics, so a blank in one "
                   "column does not mean the day is missing everywhere."),
        ("THIN DAYS, PRODUCE: kept", "About 4% of produce days rest on fewer than five "
                                     "markets, almost all before 2000, and on those the "
                                     "level can swing hard — jitomate on 1998-01-07 reads "
                                     "1.90 against 4.80 either side, on one market. These "
                                     "are not errors and are NOT filtered out. Check "
                                     "n_mercados before using any single early day."),
        ("THIN DAYS, PROTEIN: dropped", "The protein half does the opposite: a day whose "
                                        "market count falls below 40% of what that series "
                                        "normally carries is excluded, because it prices "
                                        "composition rather than the market, and the newest "
                                        "day is the usual offender. cobertura gives the "
                                        "count and the latest exclusion per series. The "
                                        "asymmetry is deliberate — the produce series is "
                                        "long-published and changing it would move history "
                                        "— but it is an asymmetry, so it is stated here."),
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
        ("WHY TWO RES COLUMNS", "Res canal is the carcass price at the rastros; Res cortes "
                                "is named retail cuts at the packers. Different prices at "
                                "different stages, not two estimates of one price. Res "
                                "cortes is also INCOMPLETE: that endpoint still returns more "
                                "than one page for most months even split into four "
                                "day-windows, so some cuts are missing. Cerdo has canal "
                                "only — the porcinos cortes endpoint returns nothing."),
        ("MANTECA IS A PROXY", "SNIIM does not quote rendered lard. 'Cerdo grasa' is the "
                               "rastro fat column, the closest available series and NOT the "
                               "same product. Its correlation with the published manteca CPI "
                               "is 0.08 — effectively none. Read it as an indicator of pork "
                               "by-product prices, not as a manteca input."),
        ("How well these track the CPI", "Correlation of the 30-day change with the "
                                         "published fortnightly CPI, 2024 to date: produce "
                                         "median 0.91. Proteins: huevo 0.91, cerdo canal "
                                         "0.81, res cortes 0.80, pollo 0.62, vísceras 0.33, "
                                         "camarón 0.33, pescado 0.33, manteca 0.08. "
                                         "Proteins track considerably worse, because SNIIM "
                                         "prices an upstream stage and processing sits "
                                         "between it and the shelf."),
        ("", ""),
        ("SHEET: precios_diarios", "The panel. Produce first, in INPC clave order — fruits, "
                                   "then vegetables, then the two granos — followed by the "
                                   "protein series."),
        ("SHEET: n_mercados", "Same shape: how many market quotes stand behind each cell. "
                              "Use it to judge how much to trust a given day."),
        ("SHEET: cobertura", "One row per column: type, the INPC generic it feeds, which "
                             "weighting built it, coverage, variant count, median markets "
                             "per day, latest and mean level, and the thin-day counts."),
        ("", ""),
        ("Rebuild", "python3 run.py update && python3 run_proteinas.py --update && "
                    "python3 export_panel_xlsx.py"),
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
        ws.set_column(j, j, 30 if j <= 2 else (19 if c == "ponderacion" else 15),
                      N2 if c.startswith(("nivel", "pct", "mercados")) else None)
    cov.to_excel(xl, sheet_name="cobertura", index=False, startrow=1, header=False)
    ws.freeze_panes(1, 0)

print(OUT)
print(cov.drop(columns=["ultimo_dia_menos_5", "ultimo_dia_excluido"])
      .to_string(index=False, formatters={
    "pct_de_dias": "{:.0f}".format, "mercados_mediana": "{:.0f}".format,
    "nivel_ultimo": "{:.2f}".format, "nivel_medio": "{:.2f}".format}))
