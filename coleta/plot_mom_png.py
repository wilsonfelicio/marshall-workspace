"""Month-over-month change, wholesale vs retail, both from price LEVELS in MXN/kg.

  charts/<slug>_mom.png   (a) monthly % change of each level over the last N years
                          (b) average monthly % change by calendar month — the
                              seasonal profile, which is most of what a nowcast
                              for a fresh-produce line has to get right.

Usage: python3 plot_mom_png.py calabacita [years]
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
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, ".")
from inpc import catalogo, precios_hist  # noqa: E402

logging.basicConfig(level=logging.WARNING)

slug = sys.argv[1] if len(sys.argv) > 1 else "calabacita"
YEARS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

BLUE, ORANGE = "#2a78d6", "#eb6834"
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
pct = FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%")


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

# One monthly grid; a hole stays NaN so no change is computed across it.
lo, hi = max(wl.index.min(), rl.index.min()), min(wl.index.max(), rl.index.max())
grid = pd.period_range(lo, hi, freq="M").to_timestamp()
wl, rl = wl.reindex(grid), rl.reindex(grid)
# Percent changes for the chart (how INEGI publishes "variación mensual"), log
# changes for the volatility comparison (percent changes are asymmetric).
wm, rm = 100 * wl.pct_change(), 100 * rl.pct_change()
lw, lr = np.log(wl).diff(), np.log(rl).diff()

start = grid[-1] - pd.DateOffset(years=YEARS) + pd.DateOffset(months=1)
win = grid[grid >= start]
wmv, rmv = wm.loc[win], rm.loc[win]
ok = wmv.notna() & rmv.notna()
n_ok = int(ok.sum())
corr = float(wmv[ok].corr(rmv[ok]))
same = 100 * float((np.sign(wmv[ok]) == np.sign(rmv[ok])).mean())
amp = float(lw.loc[win][ok].std() / lr.loc[win][ok].std())
first, last = win[0], win[-1]

# Seasonal profile: mean % change by calendar month, same window.
seas = pd.DataFrame({"m": win.month, "w": wmv.values, "r": rmv.values}).groupby("m").mean()

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.8, 9.4))
ax = fig.add_axes([L, 0.400, Rt - L, 0.355])
ax2 = fig.add_axes([L, 0.130, Rt - L, 0.190])
for a in (ax, ax2):
    styled(a)
    a.axhline(0, color=MUT, lw=1.3, zorder=3)

ax.plot(win, wmv.values, color=BLUE, lw=1.7, zorder=5)
ax.plot(win, rmv.values, color=ORANGE, lw=1.7, zorder=4)
ax.set_ylabel("variación mensual", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(pct)
span = ax.get_ylim()[1] - ax.get_ylim()[0]
pts = sorted([[wmv[last], BLUE, "Mayorista", wmv[last]], [rmv[last], ORANGE, "Menudeo", rmv[last]]],
             key=lambda x: -x[0])
if pts[0][0] - pts[1][0] < span * 0.055:
    mid = (pts[0][0] + pts[1][0]) / 2
    pts[0][0], pts[1][0] = mid + span * 0.030, mid - span * 0.030
for ypos, c, n, vt in pts:
    ax.annotate(f"{n} {vt:+.1f}%", xy=(last, ypos), xytext=(9, 0), textcoords="offset points",
                color=c, fontsize=11, fontweight="bold", va="center", annotation_clip=False)

# Seasonal bars: 2px surface gap between the pair and between months.
x = np.arange(1, 13)
bw = 0.38
ax2.bar(x - bw / 2 - 0.01, seas["w"].values, bw, color=BLUE, lw=0, zorder=5)
ax2.bar(x + bw / 2 + 0.01, seas["r"].values, bw, color=ORANGE, lw=0, zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels([MES[m] for m in x])
ax2.set_xlim(0.4, 12.6)
ax2.set_ylabel("promedio del mes", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(pct)
# Above the panel, not inside it: at 0.008/0.93 it landed on the January bar.
fig.text(L, 0.336, f"Perfil estacional: variación mensual promedio de cada mes del año, "
                   f"{YEARS} años", color=SEC, fontsize=10, fontweight="bold")

fig.text(L, 0.968, f"{label}: variación mensual del precio mayorista y del precio al menudeo",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"Cambio % respecto al mes anterior, calculado sobre niveles en $/kg: mayorista SNIIM y "
       f"menudeo de Precios Promedio del INPC (INEGI). Últimos {YEARS} años: {n_ok} meses, "
       f"{sp(first)} – {sp(last)}. Correlación {corr:.3f}; mismo signo en {same:.0f}% de los meses; "
       f"amplitud del mayorista {amp:.2f}× la del menudeo (cambios log). Desviación estándar "
       f"{wmv[ok].std():.1f}% vs {rmv[ok].std():.1f}%.")
y = 0.934
for ln in textwrap.wrap(sub, width=118):
    fig.text(L, y, ln, fontsize=10.8, color=SEC)
    y -= 0.0255
fig.legend(handles=[Line2D([], [], color=BLUE, lw=2.2, label="Mayorista SNIIM"),
                    Line2D([], [], color=ORANGE, lw=2.2, label="Menudeo INPC (Precios Promedio)")],
           loc="upper left", bbox_to_anchor=(L, y + 0.012), frameon=False, ncol=2,
           fontsize=10.8, handlelength=1.7, columnspacing=2.2, labelcolor=SEC)

foot = ("Fuente: SNIIM (Secretaría de Economía) e INEGI. Pesos nominales, sin deflactar. Se "
        "excluye el mes en curso por estar incompleto. A diferencia de la variación anual, estos "
        "cambios no se traslapan, así que la correlación de arriba es directamente interpretable: "
        "no está inflada por autocorrelación de ventanas.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(L, 0.026 + (len(lines) - 1 - i) * 0.020, ln, fontsize=9, color=MUT)

out = f"charts/{slug}_mom.png"
fig.savefig(out, dpi=170)
print(out)
print(f"n={n_ok} {first:%Y-%m}..{last:%Y-%m}  corr={corr:.3f} mismo_signo={same:.0f}% amp={amp:.2f}"
      f"  sd may={wmv[ok].std():.1f}% men={rmv[ok].std():.1f}%  ultimo {wmv[last]:+.1f}% / {rmv[last]:+.1f}%")
print("estacionalidad (may/men, %):")
print("  " + "  ".join(f"{MES[m]} {seas['w'][m]:+.1f}/{seas['r'][m]:+.1f}" for m in x))
