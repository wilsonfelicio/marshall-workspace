"""Chartbook for the eight protein generics: wholesale rate of change vs the CPI.

  chartbook_proteinas.pdf   8 pages, one per INPC generic, plus a cover

Same four series and the same reading as the produce chartbook:

  30d%   the 30-day moving average of the wholesale price against the 30 days before it
  7d%    the 7-day moving average against the same point 30 days earlier
  CPI    INEGI's published index, each fortnight against the fortnight two prints
         earlier, plotted on the day its fortnight CLOSES
  fit    a real-time model estimate of that published change (nowcast.py). Trust it less
         here than in the produce book: these wholesale series exist from 2024 only, so
         the fit trains on about thirty fortnights and starts partway across the page.

Three things differ from produce, and they are on the page rather than buried here:

  * No INPC city weights. `pesos_mercado` maps the 60 produce wholesale markets to INPC
    cities; the protein quotes come from rastros, packers and distribution centres that
    are not in that table. The national series is therefore an EQUAL-weighted geometric
    mean across markets, not a city-weighted one.

  * Prices are ranges. Most protein series publish a min and a max rather than one quote,
    so the level is the geometric centre of the range (huevo, which publishes a modal
    "precio frecuente", uses that instead).

  * Two of the eight are proxies, not the thing itself. Manteca de cerdo is matched to
    the rastro "Grasa" column, which is raw fat rather than rendered lard; and carne de
    res / de cerdo can be read either at carcass (canal) or at retail cut (cortes). For
    those two the page shows whichever correlates better with the published CPI and names
    the other's correlation, rather than quietly picking one.
"""
from __future__ import annotations

import argparse
import glob
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

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="y0", type=int, default=2024)
ap.add_argument("--out", default="chartbook_proteinas.pdf")
ap.add_argument("--no-cover", action="store_true")
ap.add_argument("--page-offset", type=int, default=0)
ap.add_argument("--page-total", type=int, default=0)
ap.add_argument("--no-model", action="store_true", help="drop the fitted CPI line")
A = ap.parse_args()
LO = pd.Timestamp(f"{A.y0}-01-01")
TODAY = pd.Timestamp.today().strftime("%d %B %Y")

# INPC generic -> candidate SNIIM series, and which products inside them count.
# A tuple of candidates means "score both and show the better one".
PLAN = [
    ("Pollo", "022 Pollo", [("pollo", ("Pollo entero", "Pollo tipo rosticero"))]),
    ("Carne de cerdo", "017 Carne de cerdo",
     [("cerdo_canal", None), ("cerdo_cortes", None)]),
    ("Manteca de cerdo", "043 Manteca de cerdo", [("cerdo_grasa", ("Grasa",))]),
    ("Carne de res", "018 Carne de res", [("res_canal", None), ("res_cortes", None)]),
    ("Vísceras de res", "025 Vísceras de res", [("res_visceras", ("Vísceras",))]),
    ("Camarón", "027 Camarón", [("camaron", None)]),
    ("Pescado", "028 Pescado", [("pescado_filete", None), ("pescado_dulce", None)]),
    ("Huevo", "031 Huevo", [("huevo", None)]),
]
NICE = {"pollo": "pollo entero y rosticero", "cerdo_canal": "carne en canal",
        "cerdo_cortes": "cortes de empacadoras", "cerdo_grasa": "grasa (rastro)",
        "res_canal": "carne en canal", "res_cortes": "cortes de empacadoras",
        "res_visceras": "vísceras (rastro)", "camaron": "camarón, La Nueva Viga y destinos",
        "pescado_filete": "filetes", "pescado_dulce": "pescado de agua dulce",
        "huevo": "huevo blanco y rojo"}
MIN_COVER = 0.40


def load_series(name):
    files = sorted(glob.glob(f"data/raw/proteinas/{name}/anio=*/part.parquet"))
    if not files:
        return None
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["fecha"] = pd.to_datetime(d["fecha"])
    return d[d.precio > 0]


def national(d, keep_products):
    """Equal-weighted geometric mean across markets, per day."""
    if keep_products:
        d = d[d.producto.isin(keep_products)]
    if not len(d):
        return None
    # Two stages, because one mean over all quotes would weight each QUOTE equally and a
    # market posting 26 shrimp grades would outweigh one posting 2 — measured on camarón
    # that construction sits a median 10.7% from this one (res_cortes 0.6%, huevo 0.8%),
    # and worse, a heavy-portfolio market skipping a day moves the "price". Mean within
    # market first, then across markets, which is what the LEEME promises.
    per_mkt = (d.assign(lp=np.log(d.precio))
               .groupby(["fecha", "destino"]).lp.mean())
    g = per_mkt.groupby("fecha").agg(lp="mean", nm="size")
    lp, nm = g.lp, g.nm
    typical = nm.rolling(60, min_periods=5, center=True).median().bfill().ffill()
    thin = nm < (MIN_COVER * typical)
    n_thin = int(thin.sum())
    lp, nm = lp[~thin], nm[~thin]
    if len(lp) < 30:
        return None
    gap = float(pd.Series(lp.index).diff().dt.days.median() or 1.0)
    cal = pd.DataFrame({"lp": lp}).reindex(
        pd.date_range(lp.index.min(), lp.index.max(), freq="D"))
    cal["lp"] = cal.lp.ffill(limit=max(1, int(round(gap))))
    need30 = max(4, int(round(0.6 * 30 / gap)))
    need7 = max(1, int(round(0.6 * 7 / gap)))
    ma30 = cal.lp.rolling("30D", min_periods=need30).mean()
    ma7 = cal.lp.rolling("7D", min_periods=need7).mean()
    return (pd.DataFrame({"c30": 100 * (ma30 - ma30.shift(30, freq="D").reindex(ma30.index)),
                          "c7": 100 * (ma7 - ma7.shift(30, freq="D").reindex(ma7.index))},
                         index=cal.index),
            cal.lp, float(nm.median()), gap, n_thin)


# ---------------------------------------------------------------- CPI, quincenal
qi = pd.read_parquet("data/inpc/inpc_proteinas_quincenal.parquet")
qi["t"] = qi.anio * 24 + (qi.mes - 1) * 2 + (qi.quincena - 1)
CPI = {}
for col in qi.columns:
    if col[:3].isdigit():
        f = pd.DataFrame({"t": qi.t.values,
                          "inpc": pd.to_numeric(qi[col], errors="coerce")}).dropna()
        st, ln = f.t.diff(), np.log(f.inpc)
        f["chg"] = np.where((st == 1) & (st.shift(1) == 1), 100 * (ln - ln.shift(2)), np.nan)
        yy, rem = np.divmod(f.t.to_numpy(), 24)
        mm, hh = np.divmod(rem, 2)
        start = pd.to_datetime({"year": yy, "month": mm + 1, "day": np.where(hh == 0, 1, 16)})
        f["fecha"] = pd.DatetimeIndex(np.where(hh == 0, start + pd.Timedelta(days=14),
                                               start + pd.offsets.MonthEnd(0)))
        CPI[col] = f.dropna(subset=["chg"])[["fecha", "chg"]]


def corr_with_cpi(f, c):
    if c is None or not len(c) or f is None:
        return np.nan, 0
    l = f.dropna(subset=["c30"])[["c30"]].reset_index().rename(columns={"index": "fecha"})
    l["fecha"] = l.fecha.astype("datetime64[ns]")
    r = c[["fecha", "chg"]].copy()
    r["fecha"] = r.fecha.astype("datetime64[ns]")
    al = pd.merge_asof(r.sort_values("fecha"), l.sort_values("fecha"), on="fecha",
                       direction="nearest", tolerance=pd.Timedelta("6D")).dropna()
    if len(al) < 7:
        return np.nan, len(al)
    return float(np.corrcoef(al.chg, al.c30)[0, 1]), len(al)


rows = []
with PdfPages(A.out) as pdf:
    if not A.no_cover:
        fig = S.page()
        S.chrome(fig, "Precios de mayoreo", TODAY,
                 foot_left="Source: SNIIM (Secretaría de Economía) and INEGI")
        fig.text(S.L, 0.790, "Mexican proteins: wholesale prices", fontsize=27,
                 color=S.ORANGE)
        fig.text(S.L, 0.726, "against the published CPI", fontsize=27, color=S.ORANGE)
        fig.text(S.L, 0.668, f"One page per INPC protein generic, eight of them, "
                 f"{A.y0} to date.", fontsize=11.5, color=S.INK)
        S.bullets(fig, [
            "The same four series as the produce chartbook, all measuring a change over "
            "roughly thirty days: the wholesale 30-day average against the 30 days before "
            "it, the same on a 7-day average, INEGI's published index against the "
            "fortnight two prints earlier, and a real-time model fit for that change.",

            "The source is SNIIM's Pecuarios and Pesqueros modules — a different "
            "application from the Agrícolas one behind the produce book, with its own "
            "markets: rastros, packers and distribution centres rather than central de "
            "abasto wholesalers. There are no INPC city weights for those markets, so the "
            "national series is an EQUAL-weighted geometric mean; and most series publish a "
            "price RANGE rather than one quote, so the level is the geometric centre of min "
            "and max (huevo, which publishes a modal price, uses that instead).",

            "Manteca de cerdo is a proxy: SNIIM quotes rastro 'Grasa', which is raw fat, "
            "not rendered lard. Carne de res and de cerdo can be read at carcass or at "
            "retail cut; each page shows whichever tracks the CPI better and names the "
            "other's correlation.",

            "Read the model line here more sceptically than in the produce book. These "
            "wholesale series were collected from 2024 only, so the fit trains on about "
            "thirty fortnights rather than two hundred and starts partway across the page. "
            "On several generics almost all of its accuracy comes from the last published "
            "print rather than from wholesale prices — each page gives the error with and "
            "without that term, and where the two are close, the model is mostly telling "
            "you that the CPI repeats itself.",
        ], 0.588)
        pdf.savefig(fig)
        plt.close(fig)

    for i, (label, cpi_key, cands) in enumerate(PLAN, 1):
        c_all = CPI.get(cpi_key)                # full history: the model trains on it
        c = c_all[c_all.fecha >= LO] if c_all is not None else None
        scored = []
        for nm, keep in cands:
            d = load_series(nm)
            if d is None:
                continue
            got = national(d[d.fecha >= LO - pd.Timedelta(days=120)], keep)
            if got is None:
                continue
            f, lp_cal, nmk, gap, n_thin = got
            r, n = corr_with_cpi(f[f.index >= LO], c)
            scored.append({"name": nm, "f": f[f.index >= LO], "f_all": f, "lp": lp_cal,
                           "nm": nmk, "gap": gap, "thin": n_thin, "r": r, "n": n})

        fig = S.page()
        S.chrome(fig, "Precios de mayoreo", "Proteínas", i + A.page_offset,
                 A.page_total or len(PLAN),
                 "Source: SNIIM (Secretaría de Economía) and INEGI")
        fig.text(S.L, 0.858, label, fontsize=19, fontweight="bold", color=S.INK)

        if not scored:
            fig.text(0.5, 0.5, "no wholesale data collected for this generic yet",
                     ha="center", color=S.GRAY, fontsize=13)
            pdf.savefig(fig)
            plt.close(fig)
            rows.append({"generico": label, "serie": None, "corr": np.nan})
            continue

        best = max(scored, key=lambda z: (-1 if np.isnan(z["r"]) else z["r"]))
        f = best["f"]
        fig.text(S.L, 0.829, f"SNIIM: {NICE.get(best['name'], best['name'])}",
                 fontsize=9.6, color=S.GRAY)
        ax = S.panel(fig)

        # ------------------------------------------------- the fitted CPI line
        lad, pred = {}, None
        if not A.no_model and c_all is not None:
            X = nowcast.features(best["f_all"], best["lp"], best["gap"], c_all)
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
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10],
                                               min_n_ticks=5))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        hi = max([f.index.max()] + ([c.fecha.max()] if c is not None and len(c) else []))
        ax.set_xlim(LO, hi)

        labels = [("Wholesale, 30-day average", S.INK),
                  ("same on a 7-day average", S.GRAY)]
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

        bits = []
        if len(f.dropna(subset=["c30"])):
            bits.append(f"latest 30d {f.c30.dropna().iloc[-1]:+.1f}%")
        if len(f.dropna(subset=["c7"])):
            bits.append(f"7d {f.c7.dropna().iloc[-1]:+.1f}%")
        if c is not None and len(c):
            bits.append(f"last published CPI {c.chg.iloc[-1]:+.1f}% "
                        f"({c.fecha.iloc[-1]:%d %b %Y})")
        if not np.isnan(best["r"]):
            bits.append(f"corr(30d, CPI) {best['r']:.2f}")
        bits.append(f"{best['nm']:.0f} markets")
        if best["gap"] > 3:
            bits.append(f"quoted every {best['gap']:.0f} days")
        other = [z for z in scored if z["name"] != best["name"] and not np.isnan(z["r"])]
        if other:
            bits.append("; ".join(f"{NICE.get(z['name'], z['name'])} would be "
                                  f"{z['r']:.2f}" for z in other))
        m2 = lad.get("m2", (None, None, {}))[2]
        if m2.get("n_oos", 0) > 6:
            m0, m1 = lad["m0"][2], lad["m1"][2]
            bits.append(f"model out-of-sample corr {m2['corr']:.2f}, error "
                        f"{m2['rmse']:.2f}pp against {m0['rmse']:.2f} for the 30d line "
                        f"alone and {m1['rmse']:.2f} without the CPI lag (n={m2['n_oos']})")
        # SNIIM protein history was collected from 2024 only, so the fit cannot start at
        # the left edge the way it does in the produce book. Say it on the page.
        if pred is not None and len(pred) and pred.index.min() > LO + pd.Timedelta(days=30):
            bits.append(f"model starts {pred.index.min():%b %Y}: wholesale for these "
                        f"markets is collected from 2024 only, and the fit needs "
                        f"{nowcast.MIN_TRAIN} prints of overlap before it can predict one")
        for k, ln in enumerate(textwrap.wrap("   ·   ".join(bits), width=132,
                                             break_long_words=False)):
            fig.text(S.L, 0.795 - k * 0.0225, ln, fontsize=8.9, color=S.INK)

        eq = nowcast.equation(m2)
        if eq:
            fig.text(S.L, 0.196, eq, fontsize=9, color=S.NAVY, fontweight="bold")
            fig.text(S.R, 0.196, f"ridge penalty {m2['alpha']:.3g}   ·   trained on "
                     f"{m2['n_train']} fortnights   ·   refit every fortnight",
                     fontsize=8, color=S.MUT, ha="right")
        src = ("Wholesale is an equal-weighted geometric mean across markets — these "
               "markets carry no INPC city weights. Most series quote a price range, so the "
               "level is the geometric centre of min and max. CPI dots sit on the day each "
               "fortnight closes, not the day it is labelled.")
        if eq:
            src += (" The model is a ridge fit of the published change on the wholesale "
                    "fortnight-average change, its own lag, the 7-day edge and the last "
                    "print INEGI had actually released, re-estimated at every fortnight on "
                    "data available at the time — no point on it uses the print it draws.")
        for k, ln in enumerate(textwrap.wrap(src, width=163)):
            fig.text(S.L, 0.160 - k * 0.0155, ln, fontsize=7.5, color=S.MUT)
        pdf.savefig(fig)
        plt.close(fig)
        rows.append({"generico": label, "serie": best["name"], "corr": best["r"],
                     "n_dots": best["n"], "markets": best["nm"],
                     "m0_rmse": lad.get("m0", (0, 0, {}))[2].get("rmse"),
                     "m1_rmse": lad.get("m1", (0, 0, {}))[2].get("rmse"),
                     "m2_corr": m2.get("corr"), "m2_rmse": m2.get("rmse"),
                     "n_oos": m2.get("n_oos"), "sd_cpi": m2.get("sd_y")})
        print(f"  {i}/{len(PLAN)}  {label:<20}{best['name']:<16}"
              + (f"corr {best['r']:.2f}" if not np.isnan(best["r"]) else "corr n/a"))

T = pd.DataFrame(rows)
T.to_csv("data/curated/proteinas_corr.csv", index=False)
print(f"\n{A.out}  ({len(PLAN)} pages + cover)")
print(T.to_string(index=False))
print(f"\nmedian corr {T['corr'].median():.2f}; "
      f"best {T.loc[T['corr'].idxmax(), 'generico']} {T['corr'].max():.2f}; "
      f"worst {T.loc[T['corr'].idxmin(), 'generico']} {T['corr'].min():.2f}")
