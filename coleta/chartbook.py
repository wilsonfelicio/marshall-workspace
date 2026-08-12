"""Chartbook: one page per generic, wholesale rate of change against the published CPI.

  chartbook_frutas_verduras.pdf   32 pages, one per INPC generic, plus a cover

Each page shows three series that all measure a change over ROUGHLY THIRTY DAYS, so
they belong on one axis:

  30d%   the 30-day moving average of the wholesale price against the 30 days before it
  7d%    the 7-day moving average against the same point 30 days earlier — the same
         monthly change measured on a shorter window, so it leads and overshoots
  CPI    INEGI's published index for the generic, each fortnight against the fortnight
         two prints earlier, plotted on the day its fortnight CLOSES

The dating matters more than it looks: a fortnight is labelled by its first day but
summarises prices through its last, so plotting the dot at the label understates the
relationship badly (0.40 against 0.86 on jitomate). Dots sit at the close.

Windows are CALENDAR days, not row counts: SNIIM does not quote on Sundays or holidays,
so 30 rows of the daily panel spans about six weeks and would drift in length across the
sample. `--rows` switches to row counts if you want to reproduce a spreadsheet that
offsets by rows.

  python3 chartbook.py [--from YYYY] [--rows] [--out FILE]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8880"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "pdf.fonttype": 42})

# The order the user asked for: fruit, then vegetables, then the two granos.
ORDER = ["Aguacate", "Durazno", "Guayaba", "Limón", "Manzana", "Melón", "Naranja",
         "Papaya", "Pera", "Piña", "Plátanos", "Sandía", "Uva", "Otras frutas",
         "Calabacita", "Cebolla", "Chayote", "Chile poblano", "Chile serrano", "Ejotes",
         "Jitomate", "Lechuga y col", "Nopales", "Papa y otros tubérculos", "Pepino",
         "Tomate verde", "Zanahoria", "Otras verduras y legumbres", "Otros chiles frescos",
         "Cilantro, epazote y perejil", "Chile seco", "Frijol"]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="y0", type=int, default=2024)
ap.add_argument("--rows", action="store_true",
                help="offset by panel rows instead of calendar days")
ap.add_argument("--out", default="chartbook_frutas_verduras.pdf")
A = ap.parse_args()
LO = pd.Timestamp(f"{A.y0}-01-01")

# ------------------------------------------------------------------ wholesale, daily
d = pd.read_parquet("data/curated/var_market_daily.parquet",
                    columns=["categoria_label", "fecha", "destino", "precio_geo"])
d["fecha"] = pd.to_datetime(d["fecha"])
d = d[(d.precio_geo > 0) & (d.fecha >= LO - pd.Timedelta(days=120))]
pw = pd.read_parquet("data/curated/pesos_mercado.parquet")
wmap = dict(zip(pw.destino, pd.to_numeric(pw.peso_inpc, errors="coerce").fillna(0.0)))
d["w"] = d.destino.map(wmap).fillna(0.0)
d["lp"] = np.log(d.precio_geo)
d = d[d.w > 0]
d["num"] = d.w * d.lp
agg = (d.groupby(["categoria_label", "fecha"])
       .agg(num=("num", "sum"), den=("w", "sum"), nm=("destino", "nunique")))
agg["lp"] = agg.num / agg.den

# ------------------------------------------------------------------ CPI, quincenal
qi = pd.read_parquet("data/inpc/inpc_genericos_quincenal.parquet")
qi["t"] = qi.anio * 24 + (qi.mes - 1) * 2 + (qi.quincena - 1)
CPI = {}
for col in qi.columns:
    if col[:3].isdigit():
        s = pd.to_numeric(qi[col], errors="coerce")
        f = pd.DataFrame({"t": qi.t.values, "inpc": s.values}).dropna()
        if f.empty:
            continue
        st = f.t.diff()
        ln = np.log(f.inpc)
        f["chg"] = np.where((st == 1) & (st.shift(1) == 1), 100 * (ln - ln.shift(2)), np.nan)
        yy, rem = np.divmod(f.t.to_numpy(), 24)
        mm, hh = np.divmod(rem, 2)
        start = pd.to_datetime({"year": yy, "month": mm + 1, "day": np.where(hh == 0, 1, 16)})
        # the day the fortnight closes, which is what the print summarises
        close = np.where(hh == 0, start + pd.Timedelta(days=14),
                         start + pd.offsets.MonthEnd(0))
        f["fecha"] = pd.DatetimeIndex(close)
        CPI[col[4:]] = f.dropna(subset=["chg"])[["fecha", "chg"]]

PESO = dict(zip(qi.columns[4:], [None] * 32))
sc = pd.read_csv("data/curated/system_scores.csv")
PESO = dict(zip(sc.generico, sc.peso))
SHARE = {}
try:
    pri = pd.read_csv("data/curated/prioridad_varianza.csv")
    SHARE = dict(zip(pri.generico.str.split(" ", n=1).str[1], pri.share))
except Exception:
    pass


MIN_COVER = 0.40          # of the generic's own typical market count


def series(name):
    """Wholesale level and its two rates of change for one generic."""
    if name not in agg.index.get_level_values(0):
        return None
    s = agg.xs(name, level=0).sort_index()
    lp, nm = s.lp, s.nm
    # A day on which only a handful of markets quoted is not a national price. Manzana
    # 2025-05-05 carried 2 markets out of a usual 78, at 33.8 against a normal median of
    # 42, and that single day pulled the 7-day average down 10% and rescaled the page.
    typical = nm.rolling(60, min_periods=5, center=True).median().bfill().ffill()
    thin = nm < (MIN_COVER * typical)
    n_thin = int(thin.sum())
    lp, nm = lp[~thin], nm[~thin]
    gap = float(pd.Series(lp.index).diff().dt.days.median() or 1.0)   # quoting cadence
    if A.rows:
        ma30 = lp.rolling(30, min_periods=10).mean()
        ma7 = lp.rolling(7, min_periods=3).mean()
        c30 = 100 * (ma30 - ma30.shift(30))
        c7 = 100 * (ma7 - ma7.shift(30))
        idx = lp.index
    else:
        # Carry each quote forward over the days its cadence covers, so a weekly series
        # yields a continuous 30-day mean instead of 78 disconnected stubs, and so a
        # holiday does not nick a daily line.
        cal = pd.DataFrame({"lp": lp}).reindex(
            pd.date_range(lp.index.min(), lp.index.max(), freq="D"))
        cal["lp"] = cal.lp.ffill(limit=max(1, int(round(gap))))
        # 60% of the observations this generic's cadence would supply, with a floor of
        # four for the 30-day window: at min_periods=2 the last weeks of Frijol averaged
        # two quotes against a full prior window and printed a spurious +9.8%, which alone
        # rescaled the whole page. At four the tail reads -0.6/0.0 and the series max
        # falls from 9.8 to 1.9, in line with its 99th percentile.
        need30 = max(4, int(round(0.6 * 30 / gap)))
        need7 = max(1, int(round(0.6 * 7 / gap)))
        ma30 = cal.lp.rolling("30D", min_periods=need30).mean()
        ma7 = cal.lp.rolling("7D", min_periods=need7).mean()
        c30 = 100 * (ma30 - ma30.shift(30, freq="D").reindex(ma30.index))
        c7 = 100 * (ma7 - ma7.shift(30, freq="D").reindex(ma7.index))
        idx = cal.index
    return (pd.DataFrame({"c30": c30, "c7": c7}, index=idx),
            float(nm.median()), gap, n_thin)


def page(pdf, name, i, n):
    got = series(name)
    fig = plt.figure(figsize=(11.69, 8.27))          # A4 landscape
    ax = fig.add_axes([0.088, 0.205, 0.889, 0.590])
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    w = PESO.get(name)
    sh = SHARE.get(name)
    ttl = f"{name}"
    sub = []
    if w is not None:
        sub.append(f"INPC weight {w:.3f}")
    if sh is not None:
        sub.append(f"{sh:.1f}% of subindex variance"
                   + (" — moves against it" if sh < 0 else ""))
    fig.text(0.082, 0.930, ttl, fontsize=21, fontweight="bold", color=INK)
    fig.text(0.082, 0.893, "  ·  ".join(sub), fontsize=11, color=SEC)
    fig.text(0.977, 0.930, f"{i} / {n}", fontsize=10, color=MUT, ha="right")

    if got is None:
        fig.text(0.5, 0.5, "no wholesale quotes in this window", ha="center",
                 color=MUT, fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)
        return None

    f, nmk, gap, n_thin = got
    f = f[f.index >= LO]
    c = CPI.get(name)
    c = c[c.fecha >= LO] if c is not None else None

    ax.axhline(0, color=MUT, lw=1.1, zorder=3)
    ax.plot(f.index, f.c7, color=GRAY, lw=1.15, ls=(0, (4, 2.2)), zorder=4)
    ax.plot(f.index, f.c30, color=BLUE, lw=1.9, zorder=6)
    if c is not None and len(c):
        ax.scatter(c.fecha, c.chg, s=26, color=ORANGE, lw=0, zorder=7, marker="D")
    ax.set_ylabel("change over ~30 days", fontsize=10.5, color=SEC, labelpad=6)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10],
                                           min_n_ticks=5))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_xlim(LO, max(f.index.max(), c.fecha.max() if c is not None and len(c) else LO))
    plt.setp(ax.get_xticklabels(), fontsize=9.5, rotation=0)

    # correlation of the wholesale monthly rate with the published print, aligned on the
    # fortnight close
    rho = np.nan
    if c is not None and len(c) > 6:
        # merge_asof refuses mismatched datetime resolutions, and the two sources arrive
        # at different ones (parquet gives us, the date range gives s)
        _l = c[["fecha", "chg"]].copy()
        _l["fecha"] = _l.fecha.astype("datetime64[ns]")
        _r = (f.dropna(subset=["c30"])[["c30"]].reset_index()
              .rename(columns={"index": "fecha"}))
        _r["fecha"] = _r.fecha.astype("datetime64[ns]")
        al = pd.merge_asof(_l.sort_values("fecha"), _r.sort_values("fecha"),
                           on="fecha", direction="nearest",
                           tolerance=pd.Timedelta("6D")).dropna(subset=["c30"])
        if len(al) > 6:
            rho = float(np.corrcoef(al.chg, al.c30)[0, 1])

    last = f.dropna(subset=["c30"])
    bits = []
    if len(last):
        bits.append(f"latest 30d {last.c30.iloc[-1]:+.1f}%")
    if len(f.dropna(subset=["c7"])):
        bits.append(f"7d {f.c7.dropna().iloc[-1]:+.1f}%")
    if c is not None and len(c):
        bits.append(f"last published CPI {c.chg.iloc[-1]:+.1f}% ({c.fecha.iloc[-1]:%d %b %Y})")
    if not np.isnan(rho):
        bits.append(f"corr(30d, CPI) {rho:.2f}")
    bits.append(f"{nmk:.0f} markets")
    if gap > 3:
        bits.append(f"quoted every {gap:.0f} days")
    if n_thin:
        bits.append(f"{n_thin} thin days dropped")
    line = "   ·   ".join(bits)
    for k, ln in enumerate(textwrap.wrap(line, width=116, break_long_words=False)):
        fig.text(0.082, 0.845 - k * 0.026, ln, fontsize=10, color=SEC, fontweight="bold")

    fig.legend(handles=[
        Line2D([], [], color=BLUE, lw=2.2, label="Wholesale, 30-day average vs 30 days earlier"),
        Line2D([], [], color=GRAY, lw=1.6, ls=(0, (4, 2.2)),
               label="Wholesale, 7-day average vs 30 days earlier"),
        Line2D([], [], color=ORANGE, lw=0, marker="D", markersize=6,
               label="Published CPI, fortnight vs two prints earlier")],
        loc="upper left", bbox_to_anchor=(0.082, 0.145), frameon=False, ncol=3,
        fontsize=9.5, handlelength=2.0, columnspacing=2.2, labelcolor=SEC)
    src = ("Source: SNIIM (Secretaría de Economía), daily wholesale quotes weighted by "
           "INPC city weight, and INEGI. All three series measure a change over about "
           f"thirty days; offsets in {'panel rows' if A.rows else 'calendar days'}. CPI dots "
           "are plotted on the day each fortnight closes, not the day it is labelled, "
           "because the print summarises prices through the last day of its fortnight.")
    for k, ln in enumerate(textwrap.wrap(src, width=132)):
        fig.text(0.082, 0.095 - k * 0.017, ln, fontsize=8.5, color=MUT)
    pdf.savefig(fig)
    plt.close(fig)
    return rho


rows = []
with PdfPages(A.out) as pdf:
    # cover
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.072, 0.795, "Mexican produce: wholesale prices", fontsize=25,
             fontweight="bold", color=INK)
    fig.text(0.072, 0.735, "against the published CPI", fontsize=25, fontweight="bold",
             color=INK)
    fig.text(0.072, 0.672,
             f"One page per INPC generic, {len(ORDER)} of them, {A.y0} to date.",
             fontsize=13, color=SEC)
    body = ("Each page carries three series, all measuring a change over roughly thirty "
            "days so that they share one axis: the wholesale 30-day average against the "
            "30 days before it, the same thing on a 7-day average (which leads and "
            "overshoots), and INEGI's published index for that generic, each fortnight "
            "against the fortnight two prints earlier.\n\n"
            "Wholesale is the geometric mean of daily quotes across markets, each market "
            "carrying its INPC city weight. Moving-average windows are calendar days, not "
            "row counts, because SNIIM does not quote on Sundays or holidays.\n\n"
            "CPI dots sit on the day each fortnight CLOSES. A fortnight is labelled by its "
            "first day but summarises prices through its last, so plotting a dot at its "
            "label puts it half a month before the prices it describes — on jitomate that "
            "alone moves the measured correlation from 0.86 to 0.40.\n\n"
            "Nothing here is modelled. Both wholesale series are raw published quotes and "
            "the dots are the published index.")
    y = 0.600
    for para in body.split("\n\n"):
        for ln in textwrap.wrap(para, width=104):
            fig.text(0.072, y, ln, fontsize=11, color=SEC)
            y -= 0.030
        y -= 0.016
    fig.text(0.072, 0.055, "Source: SNIIM (Secretaría de Economía) and INEGI.",
             fontsize=9, color=MUT)
    pdf.savefig(fig)
    plt.close(fig)

    for i, nm in enumerate(ORDER, start=1):
        r = page(pdf, nm, i, len(ORDER))
        rows.append({"generico": nm, "corr_30d_cpi": r})
        print(f"  {i:2d}/{len(ORDER)}  {nm:<30}"
              f"{'corr ' + format(r, '.2f') if r is not None and not np.isnan(r) else '—'}")

T = pd.DataFrame(rows)
T.to_csv("data/curated/chartbook_corr.csv", index=False)
print(f"\n{A.out}  ({len(ORDER)} pages + cover)")
ok = T.corr_30d_cpi.dropna()
print(f"corr(30d wholesale, published CPI): median {ok.median():.2f}, "
      f"best {T.loc[ok.idxmax(),'generico']} {ok.max():.2f}, "
      f"worst {T.loc[ok.idxmin(),'generico']} {ok.min():.2f}")
