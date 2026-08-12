"""Two PNGs per category, both in MXN/kg so no dual axis is ever needed:

  charts/<slug>_niveles.png   price LEVELS, wholesale (SNIIM) vs retail (INEGI
                              Precios Promedio), same units, one axis.
  charts/<slug>_logdif.png    (a) ln(menudeo) - ln(mayorista): the log margin,
                              i.e. the candidate error-correction term.
                              (b) monthly log changes of both levels.

Usage: python3 plot_levels_png.py calabacita
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import logging
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, ".")
from inpc import catalogo, precios_hist  # noqa: E402

logging.basicConfig(level=logging.WARNING)

slug = sys.argv[1] if len(sys.argv) > 1 else "calabacita"
GEN = {r["generico"] for r in catalogo.genericos()}

BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})
MES = {1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun", 7: "jul",
       8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic"}
sp = lambda d: f"{MES[d.month]} {d.year}"
L, Rt = 0.078, 0.845


def styled(a):
    a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def header(fig, title, sub, legend, y0=0.968):
    fig.text(L, y0, title, fontsize=16, fontweight="bold", color=INK)
    y = y0 - 0.034
    for ln in textwrap.wrap(sub, width=118):
        fig.text(L, y, ln, fontsize=10.8, color=SEC)
        y -= 0.0255
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(L, y + 0.012),
                   frameon=False, ncol=len(legend), fontsize=10.8, handlelength=1.7,
                   columnspacing=2.2, labelcolor=SEC)
    return y


def footer(fig, txt, width=150, bottom=0.026, step=0.020):
    """Bottom-anchored: the LAST line lands on `bottom`, so a long note cannot
    walk off the canvas as it wraps to one more line."""
    lines = textwrap.wrap(txt, width=width)
    for i, ln in enumerate(lines):
        fig.text(L, bottom + (len(lines) - 1 - i) * step, ln, fontsize=9, color=MUT)
    return bottom + (len(lines) - 1) * step


# ---------------------------------------------------------------- data
w = pd.read_parquet("data/curated/cat_national_monthly.parquet")
w = w[w["categoria"] == slug].copy()
if w.empty:
    sys.exit(f"no wholesale rows for categoria={slug!r}")
label = w["categoria_label"].iloc[0]
nombre = label if label in GEN else next(g for g in GEN if g.lower().startswith(label.lower()[:6]))

# Drop ONLY the running month. A flat days-per-month threshold would be wrong:
# SNIIM reported roughly weekly in 1998-1999 (6-9 quote days a month) and 2020-07
# has 14 days, and all of those are real observations. So compare the last month
# against the preceding twelve instead of against a constant.
w = w.sort_values("mes").reset_index(drop=True)
if len(w) > 13:
    norm = w["n_dias_max"].iloc[-13:-1].median()
    if w["n_dias_max"].iloc[-1] < 0.6 * norm:
        w = w.iloc[:-1]
wl = w.set_index(w["mes"])["precio_geo"].sort_index()
wl_full = wl.copy()

r = precios_hist.national_kg(nombre, slug, catalogo.city_weights())
rl = r.set_index("mes")["precio_geo"].sort_index()

# Reindex on a gap-free monthly grid over the overlap, so a missing month breaks
# the line instead of being interpolated away, and Δln over a hole stays NaN.
lo = max(wl.index.min(), rl.index.min())
hi = min(wl.index.max(), rl.index.max())
idx = pd.period_range(lo, hi, freq="M").to_timestamp()
wl, rl = wl.reindex(idx), rl.reindex(idx)
huecos = [d for d in idx if pd.isna(wl[d]) or pd.isna(rl[d])]
first, last = idx[0], idx[-1]
ok = wl.notna() & rl.notna()
n_ok = int(ok.sum())
mk = rl / wl
lmk = np.log(mk)
dw, dr = np.log(wl).diff(), np.log(rl).diff()
corr_d = float(dw.corr(dr))

# ---------------------------------------------------------------- levels
fig = plt.figure(figsize=(12.8, 8.0))
ax = fig.add_axes([L, 0.145, Rt - L, 0.66])
styled(ax)
ax.plot(wl.index, wl.values, color=BLUE, lw=1.7, zorder=5)
ax.plot(rl.index, rl.values, color=ORANGE, lw=1.7, zorder=4)
ax.fill_between(idx, wl.values, rl.values, color=MUT, alpha=0.13, lw=0, zorder=2)
ax.set_ylabel("pesos por kilogramo", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.set_ylim(0, max(rl.max(), wl.max()) * 1.08)
span = ax.get_ylim()[1] - ax.get_ylim()[0]
pts = sorted([[rl[last], ORANGE, "Menudeo", rl[last]], [wl[last], BLUE, "Mayorista", wl[last]]],
             key=lambda x: -x[0])
if pts[0][0] - pts[1][0] < span * 0.05:
    mid = (pts[0][0] + pts[1][0]) / 2
    pts[0][0], pts[1][0] = mid + span * 0.028, mid - span * 0.028
for ypos, c, n, vt in pts:
    ax.annotate(f"{n} ${vt:,.2f}", xy=(last, ypos), xytext=(9, 0), textcoords="offset points",
                color=c, fontsize=11, fontweight="bold", va="center", annotation_clip=False)
mid_t = pd.Series(idx)[ok.values].iloc[n_ok // 2]
ax.annotate("margen de menudeo", xy=(mid_t, (wl[mid_t] + rl[mid_t]) / 2), color=SEC,
            fontsize=9.5, ha="center", va="center", fontweight="bold")

header(fig, f"{label}: precio mayorista y precio al menudeo, pesos por kilogramo",
       f"Ambas series en $/kg, así que comparten un solo eje. Mayorista: media geométrica "
       f"ponderada de {int(w['n_mercados'].iloc[-1])} centrales de abasto (SNIIM). Menudeo: "
       f"Precios Promedio del INPC, cotizaciones en KG, media geométrica ponderada por ciudad "
       f"(INEGI). {n_ok} meses, {sp(first)} – {sp(last)}. Margen medio "
       f"{np.exp(lmk.mean()):.2f}× (rango {mk.min():.2f}–{mk.max():.2f}×).",
       [Line2D([], [], color=BLUE, lw=2.2, label="Mayorista SNIIM"),
        Line2D([], [], color=ORANGE, lw=2.2, label="Menudeo INPC (Precios Promedio)")])
footer(fig, "Fuente: SNIIM (Secretaría de Economía) e INEGI. Pesos nominales, sin deflactar. "
            f"El menudeo empieza en 2011 porque ahí arranca Precios Promedio; la serie mayorista "
            f"existe desde {sp(wl_full.index.min())}. Se excluye el mes en curso por estar "
            f"incompleto{'; ' + str(len(huecos)) + ' mes(es) sin dato en la ventana común' if huecos else ''}. "
            "Las claves de genérico de Precios Promedio cambian entre versiones de la canasta, "
            "así que la serie se empalma por NOMBRE, no por clave.")
out1 = f"charts/{slug}_niveles.png"
fig.savefig(out1, dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- log difference
fig = plt.figure(figsize=(12.8, 9.0))
ax = fig.add_axes([L, 0.400, Rt - L, 0.360])
ax2 = fig.add_axes([L, 0.128, Rt - L, 0.215], sharex=ax)
for a in (ax, ax2):
    styled(a)

ax.plot(lmk.index, lmk.values, color=VIOLET, lw=1.7, zorder=5)
ax.axhline(lmk.mean(), color=SEC, lw=1.1, ls=(0, (1, 2)), zorder=4)
ax.set_ylabel("ln(menudeo) − ln(mayorista)", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
plt.setp(ax.get_xticklabels(), visible=False)
# The last value often sits almost on the mean; nudge the two labels apart the
# same way the YoY chart does, so neither is printed on top of the other.
spanm = ax.get_ylim()[1] - ax.get_ylim()[0]
ymean, ylast = lmk.mean(), lmk[last]
if abs(ylast - ymean) < spanm * 0.055:
    d = spanm * 0.030
    if ylast >= ymean:
        ylast, ymean = ylast + d, ymean - d
    else:
        ylast, ymean = ylast - d, ymean + d
ax.annotate(f"media {lmk.mean():.2f}  ({np.exp(lmk.mean()):.2f}×)",
            xy=(last, ymean), xytext=(9, 0), textcoords="offset points",
            color=SEC, fontsize=10.5, va="center", annotation_clip=False)
ax.annotate(f"{lmk[last]:.2f}  ({mk[last]:.2f}×)", xy=(last, ylast), xytext=(9, 0),
            textcoords="offset points", color=VIOLET, fontsize=11, fontweight="bold",
            va="center", annotation_clip=False)

ax2.axhline(0, color=MUT, lw=1.3, zorder=3)
ax2.plot(dw.index, dw.values, color=BLUE, lw=1.5, zorder=5)
ax2.plot(dr.index, dr.values, color=ORANGE, lw=1.5, zorder=4)
ax2.set_ylabel("Δ ln mensual", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.1f}"))

hl = float(np.log(2) / -np.log(abs(pd.Series(lmk).autocorr(1)))) if abs(pd.Series(lmk).autocorr(1)) < 1 else np.nan
header(fig, f"{label}: diferencia de logaritmos entre menudeo y mayorista",
       f"Arriba, el margen en logaritmos — la diferencia vertical entre las dos series de "
       f"niveles. Es el término de corrección de error de un modelo: si es estacionario, los "
       f"precios están cointegrados y las desviaciones se cierran. Autocorrelación de orden 1 "
       f"{pd.Series(lmk).autocorr(1):.2f}, vida media ≈ {hl:.1f} meses. Abajo, los cambios "
       f"logarítmicos mensuales de cada nivel; correlación {corr_d:.3f}. "
       f"{n_ok} meses, {sp(first)} – {sp(last)}.",
       [Line2D([], [], color=VIOLET, lw=2.2, label="Margen log (menudeo − mayorista)"),
        Line2D([], [], color=BLUE, lw=2.2, label="Δ ln mayorista"),
        Line2D([], [], color=ORANGE, lw=2.2, label="Δ ln menudeo")])
footer(fig, "Fuente: SNIIM e INEGI. Un margen log de 0.69 equivale a un precio de menudeo "
            "del doble del mayorista. Los cambios log son simétricos: −0.69 y +0.69 son la "
            "misma magnitud en direcciones opuestas, lo que no ocurre con los cambios "
            "porcentuales cuando los movimientos son grandes. Pesos nominales.")
out2 = f"charts/{slug}_logdif.png"
fig.savefig(out2, dpi=170)
plt.close(fig)

print(f"{out1}\n{out2}")
print(f"n={n_ok} {first:%Y-%m}..{last:%Y-%m}  mayorista ${wl[last]:.2f} menudeo ${rl[last]:.2f}"
      f"  margen {mk[last]:.2f}x (media {np.exp(lmk.mean()):.2f}x, min {mk.min():.2f} max {mk.max():.2f})")
print(f"corr(dln)={corr_d:.3f}  sd dln mayorista={dw.std():.3f} menudeo={dr.std():.3f}"
      f"  ratio={dw.std()/dr.std():.2f}  AR1(margen)={pd.Series(lmk).autocorr(1):.3f} halflife={hl:.1f}m")
