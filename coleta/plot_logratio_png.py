"""The log ratio of the two PRICE LEVELS, ln(menudeo / mayorista).

  charts/<slug>_logratio.png
      (a) both levels in $/kg on a LOG y-axis. On a log axis the vertical gap
          between the two lines IS the log ratio, so panel (b) is literally the
          height of that gap read off as a number.
      (b) ln(menudeo) − ln(mayorista) with its mean and ±1 sd band, and the
          extremes labelled. Right-hand annotations translate logs into ×.

No changes, no differencing: this is the level of the ratio, which is the
error-correction term a model would use.

Usage: python3 plot_logratio_png.py calabacita [years]
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
from matplotlib.ticker import FuncFormatter, FixedLocator

sys.path.insert(0, ".")
from inpc import catalogo, precios_hist  # noqa: E402

logging.basicConfig(level=logging.WARNING)

slug = sys.argv[1] if len(sys.argv) > 1 else "calabacita"
YEARS = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = all overlapping history

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
if YEARS:
    grid = grid[grid >= grid[-1] - pd.DateOffset(years=YEARS) + pd.DateOffset(months=1)]
wl, rl = wl.reindex(grid), rl.reindex(grid)
lr = np.log(rl) - np.log(wl)
ok = lr.notna()
n_ok = int(ok.sum())
mu, sd = float(lr.mean()), float(lr.std())
first, last = grid[0], grid[-1]
i_max, i_min = lr.idxmax(), lr.idxmin()
# Is the ratio mean-reverting? AR(1) on the level, and the implied half-life.
rho = float(lr.dropna().autocorr(1))
hl = float(np.log(2) / -np.log(abs(rho))) if 0 < abs(rho) < 1 else np.nan

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.8, 9.8))
ax = fig.add_axes([L, 0.455, Rt - L, 0.315])
ax2 = fig.add_axes([L, 0.135, Rt - L, 0.245], sharex=ax)
for a in (ax, ax2):
    styled(a)

# --- panel a: levels on a log axis, so the gap == the log ratio
ax.set_yscale("log")
ax.plot(grid, wl.values, color=BLUE, lw=1.7, zorder=5)
ax.plot(grid, rl.values, color=ORANGE, lw=1.7, zorder=4)
ax.fill_between(grid, wl.values, rl.values, color=MUT, alpha=0.13, lw=0, zorder=2)
ticks = [t for t in (5, 7, 10, 15, 20, 30, 40)
         if wl.min() * 0.85 <= t <= rl.max() * 1.15]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_minor_locator(FixedLocator([]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax.set_ylabel("$/kg, escala logarítmica", fontsize=10.5, color=SEC, labelpad=10)
plt.setp(ax.get_xticklabels(), visible=False)
for val, c, n in ((rl[last], ORANGE, "Menudeo"), (wl[last], BLUE, "Mayorista")):
    ax.annotate(f"{n} ${val:,.2f}", xy=(last, val), xytext=(9, 0), textcoords="offset points",
                color=c, fontsize=11, fontweight="bold", va="center", annotation_clip=False)
# Top-left: at the bottom of the panel this caption sat on the wholesale line.
ax.text(0.008, 0.965, "en escala log, la separación vertical entre las líneas es el cociente "
                      "logarítmico del panel de abajo",
        transform=ax.transAxes, color=SEC, fontsize=9.5, fontweight="bold", va="top")

# --- panel b: the log ratio itself
ax2.fill_between(grid, mu - sd, mu + sd, color=VIOLET, alpha=0.10, lw=0, zorder=1)
ax2.axhline(mu, color=SEC, lw=1.1, ls=(0, (1, 2)), zorder=4)
ax2.plot(grid, lr.values, color=VIOLET, lw=1.7, zorder=5)
ax2.set_ylabel("ln(menudeo) − ln(mayorista)", fontsize=10.5, color=SEC, labelpad=10)
# Headroom for the two extreme labels, which sit above the max and below the min.
pad = (lr.max() - lr.min()) * 0.14
ax2.set_ylim(lr.min() - pad, lr.max() + pad)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
spanm = ax2.get_ylim()[1] - ax2.get_ylim()[0]
ylast, ymu = lr[last], mu
if abs(ylast - ymu) < spanm * 0.075:
    d = spanm * 0.042
    ylast, ymu = (ylast + d, ymu - d) if ylast >= ymu else (ylast - d, ymu + d)
ax2.annotate(f"{lr[last]:.2f}  ({np.exp(lr[last]):.2f}×)", xy=(last, ylast), xytext=(9, 0),
             textcoords="offset points", color=VIOLET, fontsize=11, fontweight="bold",
             va="center", annotation_clip=False)
ax2.annotate(f"media {mu:.2f}  ({np.exp(mu):.2f}×)", xy=(last, ymu), xytext=(9, 0),
             textcoords="offset points", color=SEC, fontsize=10.5, va="center",
             annotation_clip=False)
ax2.annotate(f"±1 d.e. {sd:.2f}", xy=(last, mu + sd), xytext=(9, 0), textcoords="offset points",
             color=MUT, fontsize=9.5, va="center", annotation_clip=False)
for t, va, dy in ((i_max, "bottom", 7), (i_min, "top", -7)):
    ax2.annotate(f"{sp(t)}  {lr[t]:.2f} ({np.exp(lr[t]):.2f}×)", xy=(t, lr[t]),
                 xytext=(0, dy), textcoords="offset points", color=SEC, fontsize=9.5,
                 ha="center", va=va, fontweight="bold")

fig.text(L, 0.968, f"{label}: cociente logarítmico de los niveles de precio",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"ln(menudeo / mayorista) sobre los niveles en $/kg — sin diferenciar, sin reescalar. "
       f"Mayorista SNIIM; menudeo de Precios Promedio del INPC (INEGI). {n_ok} meses, "
       f"{sp(first)} – {sp(last)}. Media {mu:.2f} ({np.exp(mu):.2f}×), desviación estándar "
       f"{sd:.2f}, rango {lr.min():.2f}–{lr.max():.2f} ({np.exp(lr.min()):.2f}×–"
       f"{np.exp(lr.max()):.2f}×). Autocorrelación de orden 1 {rho:.2f}, vida media "
       f"≈ {hl:.1f} meses.")
y = 0.934
for ln in textwrap.wrap(sub, width=118):
    fig.text(L, y, ln, fontsize=10.8, color=SEC)
    y -= 0.0255
fig.legend(handles=[Line2D([], [], color=BLUE, lw=2.2, label="Mayorista SNIIM"),
                    Line2D([], [], color=ORANGE, lw=2.2, label="Menudeo INPC (Precios Promedio)"),
                    Line2D([], [], color=VIOLET, lw=2.2, label="Cociente log")],
           loc="upper left", bbox_to_anchor=(L, y + 0.012), frameon=False, ncol=3,
           fontsize=10.8, handlelength=1.7, columnspacing=2.2, labelcolor=SEC)

foot = ("Fuente: SNIIM (Secretaría de Economía) e INEGI. Pesos nominales. Se excluye el mes en "
        "curso. Un cociente log de 0.69 equivale a un precio de menudeo del doble del mayorista. "
        "La vida media viene de un AR(1) sobre el nivel del cociente y sólo es indicativa: con "
        "datos mensuales no se puede resolver un ajuste que ocurra dentro del mes.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(L, 0.026 + (len(lines) - 1 - i) * 0.020, ln, fontsize=9, color=MUT)

out = f"charts/{slug}_logratio.png"
fig.savefig(out, dpi=170)
print(out)
print(f"n={n_ok} {first:%Y-%m}..{last:%Y-%m}  media {mu:.4f} ({np.exp(mu):.3f}x)  sd {sd:.4f}"
      f"  min {lr.min():.3f} ({i_min:%Y-%m})  max {lr.max():.3f} ({i_max:%Y-%m})")
print(f"AR1 {rho:.3f}  vida media {hl:.2f} meses")
