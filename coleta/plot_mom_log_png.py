"""Month-over-month in LOG RATIOS, wholesale vs retail, plus the margin's own m/m.

  charts/<slug>_mom_log.png
      (a) ln(P_t / P_t-1) for each price - the m/m log ratio. Symmetric, so a
          -0.30 and a +0.30 are the same size; percent changes are not.
      (b) m/m change of the log margin, ln(menudeo/mayorista)_t minus the same
          at t-1. Positive = the margin widened that month.

Panel (b) is the identity Δln(menudeo) − Δln(mayorista), i.e. panel (a)'s orange
line minus its blue line, so the two panels are the same three numbers arranged
two ways: what each price did, and who absorbed the difference.

Usage: python3 plot_mom_log_png.py calabacita [years]
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, ".")
from inpc import catalogo, precios_hist  # noqa: E402

logging.basicConfig(level=logging.WARNING)

slug = sys.argv[1] if len(sys.argv) > 1 else "calabacita"
YEARS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
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
lg = FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.2f}")


def styled(a):
    a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# ---------------------------------------------------------------- data
w = pd.read_parquet("data/curated/cat_national_monthly.parquet")
w = w[w["categoria"] == slug].sort_values("mes").reset_index(drop=True)
if w.empty:
    sys.exit(f"no wholesale rows for categoria={slug!r}")
label = w["categoria_label"].iloc[0]
GEN = {r["generico"] for r in catalogo.genericos()}
nombre = label if label in GEN else next(g for g in GEN if g.lower().startswith(label.lower()[:6]))
if len(w) > 13 and w["n_dias_max"].iloc[-1] < 0.6 * w["n_dias_max"].iloc[-13:-1].median():
    w = w.iloc[:-1]
wl = w.set_index("mes")["precio_geo"]
r = precios_hist.national_kg(nombre, slug, catalogo.city_weights())
rl = r.set_index("mes")["precio_geo"]

lo, hi = max(wl.index.min(), rl.index.min()), min(wl.index.max(), rl.index.max())
grid = pd.period_range(lo, hi, freq="M").to_timestamp()
wl, rl = wl.reindex(grid), rl.reindex(grid)
dw, dr = np.log(wl).diff(), np.log(rl).diff()
dm = dr - dw  # == Δ ln(menudeo/mayorista)

win = grid[grid >= grid[-1] - pd.DateOffset(years=YEARS) + pd.DateOffset(months=1)]
dw, dr, dm = dw.loc[win], dr.loc[win], dm.loc[win]
ok = dw.notna() & dr.notna()
n_ok = int(ok.sum())
corr = float(dw[ok].corr(dr[ok]))
amp = float(dw[ok].std() / dr[ok].std())
wid = 100 * float((dm[ok] > 0).mean())
corr_mw = float(dm[ok].corr(dw[ok]))
first, last = win[0], win[-1]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.8, 9.4))
ax = fig.add_axes([L, 0.400, Rt - L, 0.355])
ax2 = fig.add_axes([L, 0.130, Rt - L, 0.190], sharex=ax)
for a in (ax, ax2):
    styled(a)
    a.axhline(0, color=MUT, lw=1.3, zorder=3)

ax.plot(win, dw.values, color=BLUE, lw=1.7, zorder=5)
ax.plot(win, dr.values, color=ORANGE, lw=1.7, zorder=4)
ax.set_ylabel("Δ ln mensual", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter(lg)
plt.setp(ax.get_xticklabels(), visible=False)
span = ax.get_ylim()[1] - ax.get_ylim()[0]
pts = sorted([[dw[last], BLUE, "Mayorista", dw[last]], [dr[last], ORANGE, "Menudeo", dr[last]]],
             key=lambda x: -x[0])
if pts[0][0] - pts[1][0] < span * 0.055:
    mid = (pts[0][0] + pts[1][0]) / 2
    pts[0][0], pts[1][0] = mid + span * 0.030, mid - span * 0.030
for ypos, c, n, vt in pts:
    ax.annotate(f"{n} {vt:+.3f}", xy=(last, ypos), xytext=(9, 0), textcoords="offset points",
                color=c, fontsize=11, fontweight="bold", va="center", annotation_clip=False)

ax2.fill_between(win, 0, dm.clip(lower=0).values, color=BLUE, alpha=0.55, lw=0, zorder=2)
ax2.fill_between(win, 0, dm.clip(upper=0).values, color=RED, alpha=0.55, lw=0, zorder=2)
ax2.plot(win, dm.values, color=INK, lw=0.8, alpha=0.4, zorder=4)
ax2.set_ylabel("Δ del margen log", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(lg)
m = float(np.nanmax(np.abs(dm.values))) * 1.12
ax2.set_ylim(-m, m)
ax2.text(0.008, 0.90, "el margen se amplía", transform=ax2.transAxes, color=BLUE,
         fontsize=9.5, fontweight="bold", va="top")
ax2.text(0.008, 0.10, "el margen se comprime", transform=ax2.transAxes, color=RED,
         fontsize=9.5, fontweight="bold", va="bottom")

fig.text(L, 0.968, f"{label}: cambio mensual en logaritmos y cambio del margen",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"Arriba, ln(P_t / P_t−1) de cada nivel en $/kg: mayorista SNIIM y menudeo de Precios "
       f"Promedio del INPC (INEGI). Abajo, el mismo cambio para el margen log — es exactamente la "
       f"línea naranja menos la azul. Últimos {YEARS} años: {n_ok} meses, {sp(first)} – {sp(last)}. "
       f"Correlación {corr:.3f}; amplitud del mayorista {amp:.2f}× la del menudeo; el margen se "
       f"amplía en {wid:.0f}% de los meses y su cambio correlaciona {corr_mw:.3f} con el cambio "
       f"mayorista.")
y = 0.934
for ln in textwrap.wrap(sub, width=118):
    fig.text(L, y, ln, fontsize=10.8, color=SEC)
    y -= 0.0255
fig.legend(handles=[Line2D([], [], color=BLUE, lw=2.2, label="Δ ln mayorista"),
                    Line2D([], [], color=ORANGE, lw=2.2, label="Δ ln menudeo"),
                    Patch(facecolor=BLUE, alpha=0.55, label="Δ del margen log")],
           loc="upper left", bbox_to_anchor=(L, y + 0.012), frameon=False, ncol=3,
           fontsize=10.8, handlelength=1.7, columnspacing=2.2, labelcolor=SEC)

foot = ("Fuente: SNIIM (Secretaría de Economía) e INEGI. Pesos nominales. Se excluye el mes en "
        "curso. Los cambios log son simétricos y aditivos en el tiempo, a diferencia de los "
        "porcentuales: sumar doce de ellos da el cambio anual exacto. Δ ln ≈ el cambio % sólo "
        "para movimientos pequeños; a +0.30 el cambio % es +35%.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(L, 0.026 + (len(lines) - 1 - i) * 0.020, ln, fontsize=9, color=MUT)

out = f"charts/{slug}_mom_log.png"
fig.savefig(out, dpi=170)
print(out)
print(f"n={n_ok} {first:%Y-%m}..{last:%Y-%m}  corr={corr:.3f} amp={amp:.2f}"
      f"  sd may={dw[ok].std():.4f} men={dr[ok].std():.4f} margen={dm[ok].std():.4f}")
print(f"margen: se amplia {wid:.0f}% de los meses, media {dm[ok].mean():+.4f},"
      f" corr con dln mayorista {corr_mw:.3f}, AR1 {dm[ok].autocorr(1):.3f}")
