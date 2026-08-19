"""Chartbook: one page per generic, wholesale rate of change against the published CPI.

  chartbook_frutas_verduras.pdf   32 pages, one per INPC generic, plus a cover

Each page shows four series that all measure a change over ROUGHLY THIRTY DAYS, so
they belong on one axis:

  30d%   the 30-day moving average of the wholesale price against the 30 days before it
  7d%    the 7-day moving average against the same point 30 days earlier — the same
         monthly change measured on a shorter window, so it leads and overshoots
  CPI    INEGI's published index for the generic, each fortnight against the fortnight
         two prints earlier, plotted on the day its fortnight CLOSES
  fit    a real-time model estimate of that published change, from the wholesale panel
         and the last print INEGI had actually released. See nowcast.py; every point on
         it is out of sample, and each page states its error against the 30d line alone.

The dating matters more than it looks: a fortnight is labelled by its first day but
summarises prices through its last, so plotting the dot at the label understates the
relationship badly (0.40 against 0.86 on jitomate). Dots sit at the close.

Windows are CALENDAR days, not row counts: SNIIM does not quote on Sundays or holidays,
so 30 rows of the daily panel spans about six weeks and would drift in length across the
sample. `--rows` switches to row counts if you want to reproduce a spreadsheet that
offsets by rows.

  python3 chartbook.py [--from YYYY] [--rows] [--no-model] [--out FILE]
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

import nowcast
import style as S

S.use()

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
# so this book can be rendered as one SECTION of a combined chartbook: no cover of its
# own, and page numbers that continue from wherever the previous section stopped
ap.add_argument("--no-cover", action="store_true")
ap.add_argument("--page-offset", type=int, default=0)
ap.add_argument("--page-total", type=int, default=0)
ap.add_argument("--no-model", action="store_true", help="drop the fitted CPI line")
ap.add_argument("--train-years", type=int, default=8,
                help="wholesale history loaded before the plotted window, to train on")
A = ap.parse_args()
LO = pd.Timestamp(f"{A.y0}-01-01")
TODAY = pd.Timestamp.today().strftime("%d %B %Y")
# the plot starts at LO, but the model has to be trained on fortnights that closed well
# before it, so the panel is loaded from further back and trimmed only at draw time
LOAD = LO - pd.DateOffset(years=0 if A.no_model else A.train_years) - pd.Timedelta(days=120)

# ------------------------------------------------------------------ wholesale, daily
d = pd.read_parquet("data/curated/var_market_daily.parquet",
                    columns=["categoria_label", "fecha", "destino", "precio_geo"])
d["fecha"] = pd.to_datetime(d["fecha"])
d = d[(d.precio_geo > 0) & (d.fecha >= LOAD)]
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
    # Carry each quote forward over the days its cadence covers, so a weekly series
    # yields a continuous 30-day mean instead of 78 disconnected stubs, and so a
    # holiday does not nick a daily line.
    cal = pd.DataFrame({"lp": lp}).reindex(
        pd.date_range(lp.index.min(), lp.index.max(), freq="D"))
    cal["lp"] = cal.lp.ffill(limit=max(1, int(round(gap))))
    if A.rows:
        ma30 = lp.rolling(30, min_periods=10).mean()
        ma7 = lp.rolling(7, min_periods=3).mean()
        c30 = 100 * (ma30 - ma30.shift(30))
        c7 = 100 * (ma7 - ma7.shift(30))
        idx = lp.index
    else:
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
    return (pd.DataFrame({"c30": c30, "c7": c7}, index=idx), cal.lp,
            float(nm.median()), gap, n_thin)


def page(pdf, name, i, n):
    i, n = i + A.page_offset, (A.page_total or n)
    got = series(name)
    fig = S.page()
    S.chrome(fig, "Precios de mayoreo", "Frutas y verduras", i, n,
             "Source: SNIIM (Secretaría de Economía) and INEGI")

    w, sh = PESO.get(name), SHARE.get(name)
    sub = []
    if w is not None:
        sub.append(f"INPC weight {w:.3f}")
    if sh is not None:
        sub.append(f"{sh:.1f}% of subindex variance"
                   + (" — moves against it" if sh < 0 else ""))
    fig.text(S.L, 0.858, name, fontsize=19, fontweight="bold", color=S.INK)
    fig.text(S.L, 0.829, "   ·   ".join(sub), fontsize=9.6, color=S.GRAY)

    if got is None:
        fig.text(0.5, 0.5, "no wholesale quotes in this window", ha="center",
                 color=S.GRAY, fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)
        return None, {}, {}

    ax = S.panel(fig)
    f_all, lp_cal, nmk, gap, n_thin = got
    c_all = CPI.get(name)                       # full history: the model trains on it
    f = f_all[f_all.index >= LO]
    c = c_all[c_all.fecha >= LO] if c_all is not None else None

    # ------------------------------------------------------- the fitted CPI line
    lad, pred = {}, None
    if not (A.no_model or A.rows) and c_all is not None:
        X = nowcast.features(f_all, lp_cal, gap, c_all)
        lad = nowcast.score_ladder(X, c_all, LO)
        pred = lad.get("m2", (None,))[0]
        if pred is not None:
            pred = pred[pred.index >= LO]

    ax.axhline(0, color=S.GRAY, lw=0.9, zorder=3)
    ax.plot(f.index, f.c7, color=S.GRAY, lw=1.05, ls=(0, (3.5, 2.0)), zorder=4)
    ax.plot(f.index, f.c30, color=S.INK, lw=1.7, zorder=6)
    if pred is not None and len(pred):
        ax.plot(pred.index, pred.values, color=S.NAVY, lw=1.5, zorder=6.5)
    if c is not None and len(c):
        ax.scatter(c.fecha, c.chg, s=24, color=S.ORANGE, lw=0, zorder=7, marker="D")
    # bare ticks; the unit is stated inside the panel, this style's convention
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10],
                                           min_n_ticks=5))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_xlim(LO, max(f.index.max(), c.fecha.max() if c is not None and len(c) else LO))
    labels = [("Wholesale, 30-day average", S.INK), ("same on a 7-day average", S.GRAY)]
    if pred is not None and len(pred):
        labels.append(("Model fit for the CPI, real time", S.NAVY))
    labels.append(("Published CPI, fortnight vs two prints earlier", S.ORANGE))
    drawn = [(mdates.date2num(f.index), f.c30.to_numpy()),
             (mdates.date2num(f.index), f.c7.to_numpy())]
    if pred is not None and len(pred):
        drawn.append((mdates.date2num(pred.index), pred.to_numpy()))
    if c is not None and len(c):
        drawn.append((mdates.date2num(c.fecha), c.chg.to_numpy()))
    S.place_labels(ax, drawn, labels, "% change over ~30 days")

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
    # the model's own line of the header: it is scored out of sample, and against the two
    # simpler models, because a fit quoted on its own says nothing
    m2 = lad.get("m2", (None, None, {}))[2]
    if m2.get("n_oos", 0) > 6:
        m0, m1 = lad["m0"][2], lad["m1"][2]
        bits.append(f"model out-of-sample corr {m2['corr']:.2f}, error "
                    f"{m2['rmse']:.2f}pp against {m0['rmse']:.2f} for the 30d line alone "
                    f"and {m1['rmse']:.2f} without the CPI lag (n={m2['n_oos']})")
    elif pred is not None:
        bits.append("model shown but not scored: too few prints since training began")
    # a generic INEGI only started publishing recently cannot have a line from the left
    # edge — say so on the page rather than leaving a gap the reader has to interpret
    if pred is not None and len(pred) and pred.index.min() > LO + pd.Timedelta(days=30):
        bits.append(f"model starts {pred.index.min():%b %Y}: INEGI has published this "
                    f"generic only since {c_all.fecha.min():%b %Y}, and the fit needs "
                    f"{nowcast.MIN_TRAIN} prints before it can predict one")
    for k, ln in enumerate(textwrap.wrap("   ·   ".join(bits), width=132,
                                         break_long_words=False)):
        fig.text(S.L, 0.795 - k * 0.0225, ln, fontsize=8.9, color=S.INK)

    eq = nowcast.equation(m2)
    if eq:
        # the coefficients in force at the last refit, in the units of the axis: points
        # of CPI change per point of the regressor. They move at every fortnight.
        fig.text(S.L, 0.196, eq, fontsize=9, color=S.NAVY, fontweight="bold")
        fig.text(S.R, 0.196, f"ridge penalty {m2['alpha']:.3g}   ·   trained on "
                 f"{m2['n_train']} fortnights   ·   refit every fortnight",
                 fontsize=8, color=S.MUT, ha="right")
    src = ("Wholesale is a geometric mean of daily SNIIM quotes across markets, each "
           "weighted by its INPC city weight; the moving-average windows are calendar "
           f"days, not rows{'' if not A.rows else ' (--rows: rows)'}. CPI dots are plotted "
           "on the day each fortnight closes, not the day it is labelled, because the "
           "print summarises prices through the last day of its fortnight.")
    if eq:
        src += (" The model is a ridge fit of the published change on the wholesale "
                "fortnight-average change, its own lag, the 7-day edge and the last print "
                "INEGI had actually released, re-estimated at every fortnight on data "
                "available at the time — no point on it uses the print it draws.")
    for k, ln in enumerate(textwrap.wrap(src, width=163)):
        fig.text(S.L, 0.160 - k * 0.0155, ln, fontsize=7.5, color=S.MUT)
    pdf.savefig(fig)
    plt.close(fig)
    return rho, m2, lad


rows = []
with PdfPages(A.out) as pdf:
    # cover, unless this book is being rendered as a section of a combined one
    if not A.no_cover:
        fig = S.page()
        S.chrome(fig, "Precios de mayoreo", TODAY,
                 foot_left="Source: SNIIM (Secretaría de Economía) and INEGI")
        fig.text(S.L, 0.790, "Mexican produce: wholesale prices", fontsize=27,
                 color=S.ORANGE)
        fig.text(S.L, 0.726, "against the published CPI", fontsize=27, color=S.ORANGE)
        fig.text(S.L, 0.668,
                 f"One page per INPC generic, {len(ORDER)} of them, {A.y0} to date.",
                 fontsize=11.5, color=S.INK)
        S.bullets(fig, [
            "Four series on every page, all measuring a change over roughly thirty days so "
            "that they share one axis: the wholesale 30-day average against the 30 days "
            "before it, the same thing on a 7-day average (which leads and overshoots), "
            "INEGI's published index for that generic, each fortnight against the fortnight "
            "two prints earlier, and a model fit for that published change.",

            "Wholesale is the geometric mean of daily SNIIM quotes across markets, each "
            "market carrying its INPC city weight. Moving-average windows are calendar "
            "days, not row counts, because SNIIM does not quote on Sundays or holidays.",

            "CPI dots sit on the day each fortnight CLOSES. A fortnight is labelled by its "
            "first day but summarises prices through its last, so plotting a dot at its "
            "label puts it half a month before the prices it describes — on jitomate that "
            "alone moves the measured correlation from 0.86 to 0.40.",

            "Three of the four series are raw published quotes. The fourth is a ridge "
            "regression of the published change on the wholesale fortnight-average change, "
            "its own lag, the 7-day edge, and the last print INEGI had actually released — "
            "re-estimated at every fortnight on data available at the time, so every point "
            "on it is out of sample. Each page states its error against the 30-day line "
            "alone and against the same model without the CPI lag; the gap between those "
            "two is how much of the fit is wholesale information rather than the CPI "
            "repeating itself.",
        ], 0.588)
        pdf.savefig(fig)
        plt.close(fig)

    for i, nm in enumerate(ORDER, start=1):
        r, m2, lad = page(pdf, nm, i, len(ORDER))
        row = {"generico": nm, "corr_30d_cpi": r}
        for tag in ("m0", "m1", "m2"):
            st = lad.get(tag, (None, None, {}))[2]
            row[f"{tag}_corr"] = st.get("corr")
            row[f"{tag}_rmse"] = st.get("rmse")
        row["n_oos"] = m2.get("n_oos")
        row["sd_cpi"] = m2.get("sd_y")
        rows.append(row)
        print(f"  {i:2d}/{len(ORDER)}  {nm:<30}"
              f"{'corr ' + format(r, '.2f') if r is not None and not np.isnan(r) else '—':<12}"
              f"{'model ' + format(m2['corr'], '.2f') if m2.get('corr') is not None else ''}")

T = pd.DataFrame(rows)
T.to_csv("data/curated/chartbook_corr.csv", index=False)
print(f"\n{A.out}  ({len(ORDER)} pages + cover)")
ok = T.corr_30d_cpi.dropna()
print(f"corr(30d wholesale, published CPI): median {ok.median():.2f}, "
      f"best {T.loc[ok.idxmax(),'generico']} {ok.max():.2f}, "
      f"worst {T.loc[ok.idxmin(),'generico']} {ok.min():.2f}")
if T.m2_rmse.notna().any():
    g = T.dropna(subset=["m0_rmse", "m1_rmse", "m2_rmse"])
    print(f"out-of-sample error, median across {len(g)} generics (pp): "
          f"30d line alone {g.m0_rmse.median():.2f}  ->  wholesale model "
          f"{g.m1_rmse.median():.2f}  ->  with the CPI lag {g.m2_rmse.median():.2f}"
          f"   |  CPI's own sd {g.sd_cpi.median():.2f}")
    print(f"model beats the 30d line on {int((g.m2_rmse < g.m0_rmse).sum())}/{len(g)}; "
          f"the CPI lag helps on {int((g.m2_rmse < g.m1_rmse).sum())}/{len(g)}")
