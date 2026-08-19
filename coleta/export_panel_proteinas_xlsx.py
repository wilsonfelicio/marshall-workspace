"""Wide daily panel of protein wholesale prices, one column per SNIIM series.

  precios_proteinas_diario.xlsx

The counterpart to precios_mayoreo_diario.xlsx, with the same four sheets and the same
reading, from SNIIM's Pecuarios and Pesqueros modules instead of Agrícolas.

Two deliberate differences from the produce panel, both consequences of the source:

  * Columns are SNIIM SERIES, not INPC generics. Carne de res is quoted at two different
    stages — carcass at the rastro and retail cut at the packer — and they are different
    prices, not two estimates of one price. Collapsing them would hide that. The LEEME
    sheet maps series to the eight INPC generics.

  * The cross-market mean is EQUAL-weighted, not city-weighted. `pesos_mercado` maps the
    60 produce wholesale markets to INPC cities; these quotes come from rastros, packers
    and distribution centres that are not in that crosswalk, so there is no honest weight
    to apply.

Within a series, the level is the geometric mean across markets of that day's quotes,
where each quote is already the geometric centre of the published min-max range (or the
modal "precio frecuente" for huevo). Products inside a series are averaged together —
huevo blanco with huevo rojo, pollo entero with rosticero — because the INPC generic
covers both; the per-product detail stays in data/raw/proteinas.
"""
from __future__ import annotations

import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT = "precios_proteinas_diario.xlsx"
MIN_COVER = 0.40

# series -> (column name, INPC generic it feeds, products to keep or None for all)
SERIES = [
    ("pollo",          "Pollo",             "022 Pollo",
     ("Pollo entero", "Pollo tipo rosticero")),
    ("huevo",          "Huevo",             "031 Huevo", None),
    ("res_canal",      "Res canal",         "018 Carne de res", None),
    ("res_cortes",     "Res cortes",        "018 Carne de res", None),
    ("res_visceras",   "Res vísceras",      "025 Vísceras de res", ("Vísceras",)),
    ("cerdo_canal",    "Cerdo canal",       "017 Carne de cerdo", None),
    ("cerdo_grasa",    "Cerdo grasa",       "043 Manteca de cerdo (proxy)", ("Grasa",)),
    ("camaron",        "Camarón",           "027 Camarón", None),
    ("pescado_filete", "Pescado filete",    "028 Pescado", None),
    ("pescado_dulce",  "Pescado agua dulce", "028 Pescado", None),
]

lvl, cnt, nprod, ndrop, last_thin = {}, {}, {}, {}, {}
for key, col, _gen, keep in SERIES:
    files = sorted(glob.glob(f"data/raw/proteinas/{key}/anio=*/part.parquet"))
    if not files:
        print(f"  {col:<20} no data on disk, skipped")
        continue
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d[d.precio > 0]
    if keep:
        d = d[d.producto.isin(keep)]
    if not len(d):
        print(f"  {col:<20} no rows after the product filter, skipped")
        continue
    g = (d.assign(lp=np.log(d.precio))
         .groupby("fecha").agg(lp=("lp", "mean"), nm=("destino", "nunique")))
    # A day resting on a handful of markets is not a national price, and the newest day is
    # the usual offender: on 13 Aug 2026 only the early-reporting markets were in, which
    # put pollo at 50.79 against 61.85 the day before. Anyone reading the last row as
    # "today's price" would be reading market composition, not a price move. Drop days
    # under MIN_COVER of the series' own typical count, the same rule the chartbook uses,
    # and count them in cobertura rather than dropping them silently.
    typical = g.nm.rolling(60, min_periods=5, center=True).median().bfill().ffill()
    thin = g.nm < (MIN_COVER * typical)
    ndrop[col] = int(thin.sum())
    last_thin[col] = g.index[thin].max() if thin.any() else pd.NaT
    g = g[~thin]
    lvl[col] = np.exp(g.lp)
    cnt[col] = g.nm
    nprod[col] = int(d.producto.nunique())
    print(f"  {col:<20} {len(g):5d} days  {nprod[col]:2d} products  "
          f"last {np.exp(g.lp).iloc[-1]:8.2f}")

wide = pd.DataFrame(lvl).sort_index()
nmk = pd.DataFrame(cnt).reindex(wide.index)
cols = list(wide.columns)
print(f"  panel {wide.shape[0]:,} days x {wide.shape[1]} series")

gen_of = {c: g for _k, c, g, _p in SERIES}
cov = pd.DataFrame({
    "serie": cols,
    "generico_inpc": [gen_of[c] for c in cols],
    "primer_dia": [wide[c].first_valid_index() for c in cols],
    "ultimo_dia": [wide[c].last_valid_index() for c in cols],
    "dias_con_dato": [int(wide[c].notna().sum()) for c in cols],
    "pct_de_dias": [100 * wide[c].notna().mean() for c in cols],
    "productos": [nprod[c] for c in cols],
    "mercados_mediana": [float(nmk[c].median()) for c in cols],
    "nivel_ultimo": [float(wide[c].dropna().iloc[-1]) for c in cols],
    "nivel_medio": [float(wide[c].mean()) for c in cols],
    "dias_menos_5_mercados": [int((nmk[c] < 5).sum()) for c in cols],
    "pct_menos_5_mercados": [100 * float((nmk[c] < 5).mean()) for c in cols],
    "dias_delgados_excluidos": [ndrop[c] for c in cols],
    "ultimo_dia_excluido": [last_thin[c] for c in cols],
})
for c in ("primer_dia", "ultimo_dia", "ultimo_dia_excluido"):
    cov[c] = pd.to_datetime(cov[c]).dt.date

W = wide.reset_index().rename(columns={"index": "fecha"})
W["fecha"] = pd.to_datetime(W.fecha).dt.date
N = nmk.reset_index().rename(columns={"index": "fecha"})
N["fecha"] = pd.to_datetime(N.fecha).dt.date

with pd.ExcelWriter(OUT, engine="xlsxwriter",
                    engine_kwargs={"options": {"default_date_format": "yyyy-mm-dd"}}) as xl:
    book = xl.book
    H = book.add_format({"bold": True, "font_name": "Arial", "font_size": 10,
                         "bg_color": "#21295C", "font_color": "white", "align": "left",
                         "valign": "vcenter", "text_wrap": True})
    B = book.add_format({"font_name": "Arial", "font_size": 10, "bold": True})
    TI = book.add_format({"font_name": "Arial", "font_size": 13, "bold": True,
                          "font_color": "#21295C"})
    WRP = book.add_format({"font_name": "Arial", "font_size": 10, "text_wrap": True,
                           "valign": "top"})
    N2 = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.00"})
    D = book.add_format({"font_name": "Arial", "font_size": 10, "num_format": "yyyy-mm-dd"})

    def sheet(df, nm, fmt=N2):
        ws = book.add_worksheet(nm)
        xl.sheets[nm] = ws
        for j, c in enumerate(df.columns):
            ws.write(0, j, c, H)
            ws.set_column(j, j, 11 if j == 0 else max(13, min(24, len(str(c)) + 2)),
                          D if j == 0 else fmt)
        df.to_excel(xl, sheet_name=nm, index=False, startrow=1, header=False)
        ws.freeze_panes(1, 1)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    ws = book.add_worksheet("LEEME")
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 106)
    ws.write(0, 0, "Protein wholesale price levels, daily, by SNIIM series", TI)
    mapping = "; ".join(f"{c} -> {gen_of[c]}" for c in cols)
    rows = [
        ("What this is", f"One column per SNIIM series: the wholesale price in MXN per "
                         f"kilogram, daily, {wide.shape[0]:,} days from "
                         f"{wide.index.min():%d %b %Y} to {wide.index.max():%d %b %Y}, "
                         f"{wide.shape[1]} series covering eight INPC generics."),
        ("Source", "SNIIM (Secretaría de Economía), Pecuarios and Pesqueros modules — a "
                   "different application from the Agrícolas one behind "
                   "precios_mayoreo_diario.xlsx, with its own markets: rastros, packers "
                   "and distribution centres rather than central de abasto wholesalers."),
        ("Series to INPC generic", mapping),
        ("How each cell is built", "Across markets, an EQUAL-weighted geometric mean of "
                                   "that day's quotes. Unlike the produce panel there are "
                                   "no INPC city weights: these markets are not in INEGI's "
                                   "city crosswalk, so weighting them would be invention. "
                                   "Products within a series are averaged together, since "
                                   "the INPC generic covers both (huevo blanco with rojo, "
                                   "pollo entero with rosticero)."),
        ("Price ranges", "Most of these series publish a MINIMUM and a MAXIMUM rather than "
                         "one quote. Each underlying quote is the geometric centre of that "
                         "range. Huevo publishes a modal 'precio frecuente' and that is "
                         "used instead. The min and max survive per observation in "
                         "data/raw/proteinas if you want the spread."),
        ("Blanks", "A blank is a day with no quote for that series — weekends, holidays, "
                   "and the thinner markets. Nothing is interpolated or carried forward."),
        ("THIN DAYS ARE DROPPED", "A day whose market count falls below 40% of what that "
                                  "series normally carries is excluded, because it prices "
                                  "composition rather than the market. The newest day is "
                                  "the usual offender: partway through 13 Aug 2026 only "
                                  "the early markets had reported and pollo read 50.79 "
                                  "against 61.85 the day before. The cobertura sheet counts "
                                  "the exclusions per series and gives the latest one, so "
                                  "the last row of the panel is a day that actually "
                                  "cleared the bar."),
        ("", ""),
        ("WHY TWO RES COLUMNS", "Res canal is the carcass price at 29 rastros; Res cortes "
                                "is named retail cuts at 12 packers. They are different "
                                "prices at different stages, not two estimates of one "
                                "price. The same applies to cerdo, where only canal is "
                                "available — SNIIM's porcinos cortes endpoint returns "
                                "nothing for every month tried."),
        ("MANTECA IS A PROXY", "SNIIM does not quote rendered lard. 'Cerdo grasa' is the "
                               "rastro fat column, which is the closest available series "
                               "and is NOT the same product. Its correlation with the "
                               "published manteca CPI is 0.08 — effectively none. Treat "
                               "that column as an indicator of pork by-product prices, not "
                               "as a manteca nowcast input."),
        ("RES CORTES IS INCOMPLETE", "That endpoint still returns more than one page of "
                                     "results for 31 of 32 months even after each month is "
                                     "split into four day-windows, so some cuts are "
                                     "missing on most months. The column is usable as a "
                                     "level but its coverage is not complete; every other "
                                     "series in this file is."),
        ("", ""),
        ("IMPORTANT", "This is a LEVEL series. Its day-to-day changes carry composition "
                      "effects, because the set of quoting markets and products moves. "
                      "Nominal pesos, never deflated."),
        ("How well these track the CPI", "Correlation of the 30-day change with the "
                                         "published fortnightly CPI, 2024 to date: huevo "
                                         "0.91, cerdo canal 0.81, res cortes 0.80, pollo "
                                         "0.62, vísceras 0.33, camarón 0.33, pescado 0.33, "
                                         "manteca 0.08. The produce panel's median is 0.91, "
                                         "so proteins track considerably worse — SNIIM "
                                         "prices an upstream stage and processing sits "
                                         "between it and the shelf."),
        ("", ""),
        ("SHEET: precios_diarios", "The panel, one column per series."),
        ("SHEET: n_mercados", "Same shape: how many market quotes stand behind each cell."),
        ("SHEET: cobertura", "One row per series: coverage, product count, median markets "
                             "per day, latest and mean level."),
        ("", ""),
        ("Rebuild", "python3 run_proteinas.py  then  python3 export_panel_proteinas_xlsx.py"),
    ]
    for i, (k, v) in enumerate(rows, start=2):
        ws.write(i, 0, k, B)
        ws.write(i, 1, v, WRP)
        ws.set_row(i, 58 if len(v) > 230 else (44 if len(v) > 165 else
                                               (30 if len(v) > 95 else None)))

    sheet(W, "precios_diarios")
    sheet(N, "n_mercados", fmt=None)
    ws = book.add_worksheet("cobertura")
    xl.sheets["cobertura"] = ws
    for j, c in enumerate(cov.columns):
        ws.write(0, j, c, H)
        ws.set_column(j, j, 30 if j <= 1 else 15,
                      N2 if c.startswith(("nivel", "pct", "mercados")) else None)
    cov.to_excel(xl, sheet_name="cobertura", index=False, startrow=1, header=False)
    ws.freeze_panes(1, 0)

print(OUT)
print(cov.to_string(index=False, formatters={
    "pct_de_dias": "{:.0f}".format, "mercados_mediana": "{:.0f}".format,
    "nivel_ultimo": "{:.2f}".format, "nivel_medio": "{:.2f}".format,
    "pct_menos_5_mercados": "{:.1f}".format}))
