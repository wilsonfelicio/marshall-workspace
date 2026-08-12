"""Where wholesale prices are worth having: variance share against correlation.

  charts/scatter_var_corr.png

x     correlation of the wholesale 30-day rate of change with the published CPI, 2024 to date
y     the generic's share of the fortnightly variance of the Frutas y verduras subindex
size  its INPC basket weight

The y axis is LINEAR, split into two panels. One linear axis is honest about the distance
between jitomate at 60% and everything else, but it packs the other thirty-one names into
the bottom seventh and their labels then float on long crossing leaders. Two linear panels
keep the linear scale and give the cluster room: the upper panel is everything above 3.2%
of the variance, the lower is the same axis zoomed to 0-3.2%, and both draw every generic.
`--scale symlog` puts it back on a single axis.

Labels are stacked, not force-placed: pairwise repulsion stalls at four or five unreadable
overlaps once three labels are mutually overlapping. Stacking cannot collide within a panel
side by construction, and the residual count is asserted rather than eyeballed.
"""
from __future__ import annotations

import argparse
import textwrap
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GRAY = "#9a9892"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})

ap = argparse.ArgumentParser()
ap.add_argument("--scale", choices=["linear", "symlog"], default="linear")
ap.add_argument("--out", default="charts/scatter_var_corr.png")
CLI = ap.parse_args()

T = pd.read_csv("data/curated/corr_weight_table.csv").dropna(subset=["corr_30d_cpi"])
T["r"], T["v"], T["w"] = T.corr_30d_cpi, T.share, T.peso
CORR_CUT, VAR_CUT, SPLIT = 0.80, 0.5, 3.2
T["problem"] = (T.r < CORR_CUT) & (T.v >= VAR_CUT)     # material variance, weak tracking
T["heavy"] = T.v >= 3.0
siz = (90 + 1500 * (T.w / T.w.max())).to_numpy()
col = np.where(T.problem, ORANGE, np.where(T.heavy, BLUE, GRAY))
XLIM = (0.13, 1.085)

# ------------------------------------------------------------------ figure & panels
if CLI.scale == "linear":
    fig = plt.figure(figsize=(13.6, 11.0))
    axT = fig.add_axes([0.088, 0.590, 0.884, 0.185])
    axB = fig.add_axes([0.088, 0.150, 0.884, 0.375])
    panels = [(axT, 2.4, 66.0), (axB, -0.80, SPLIT)]
else:
    fig = plt.figure(figsize=(13.6, 10.2))
    axT = fig.add_axes([0.088, 0.168, 0.884, 0.610])
    axT.set_yscale("symlog", linthresh=3.0, linscale=1.9)
    axB = None
    panels = [(axT, -1.4, 130.0)]

for a, lo, hi in panels:
    a.grid(color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.axhline(0, color=AXIS, lw=1.1, zorder=3)
    a.axvline(CORR_CUT, color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=3)
    a.scatter(T.r, T.v, s=siz, c=col, lw=0.9, edgecolors=SURF, zorder=6, alpha=0.92)
    a.set_xlim(*XLIM)
    a.set_ylim(lo, hi)
fig.canvas.draw()

# ------------------------------------------------------------------ labels
rad_all = np.sqrt(siz) * 0.62
W_all = np.array([len(n) * 5.6 + 6 for n in T.generico])


def label_panel(a, mask, lo, hi):
    """Stack this panel's labels; return how many still overlap (should be zero)."""
    ix = np.flatnonzero(mask)
    if not len(ix):
        return 0
    pts = a.transData.transform(np.c_[T.r.to_numpy()[ix], T.v.to_numpy()[ix]])
    rad, W = rad_all[ix], W_all[ix]
    H = np.full(len(ix), 13.0)
    x0, y0 = a.transData.transform((XLIM[0], lo))
    x1, y1 = a.transData.transform((XLIM[1], hi))

    def ov(i, j, L):
        return (abs(L[i, 0] - L[j, 0]) < (W[i] + W[j]) / 2
                and abs(L[i, 1] - L[j, 1]) < (H[i] + H[j]) / 2)

    side = np.where(pts[:, 0] > x0 + 0.62 * (x1 - x0), -1.0, 1.0)   # -1 = label at left
    lab, GAP = pts.copy(), 2.5
    for sgn in (-1.0, 1.0):
        run = np.flatnonzero(side == sgn)
        run = run[np.argsort(pts[run, 1])]                # bottom to top
        prev = -1e9
        for i in run:
            lab[i, 1] = max(pts[i, 1], prev + H[i] + GAP)
            prev = lab[i, 1]
        if len(run):
            over = max(0.0, lab[run, 1].max() + H[0] / 2 - (y1 - 2))
            if over > 0:                                  # run hit the top: slide it down
                lab[run, 1] -= over
                prev = 1e9
                for i in run[::-1]:
                    lab[i, 1] = min(lab[i, 1], prev - H[i] - GAP)
                    prev = lab[i, 1]
                lab[run, 1] += max(0.0, (y0 + H[0] / 2 + 2) - lab[run, 1].min())
        for i in run:
            lab[i, 0] = min(max(pts[i, 0] + sgn * (rad[i] + 7 + W[i] / 2),
                                x0 + W[i] / 2 + 2), x1 - W[i] / 2 - 2)
    # stacking is clean within a side; a left- and a right-labelled point can still meet
    for _ in range(80):
        hit = [(i, j) for i in range(len(lab)) for j in range(i + 1, len(lab))
               if side[i] != side[j] and ov(i, j, lab)]
        if not hit:
            break
        for i, j in hit:
            need = (W[i] + W[j]) / 2 - abs(lab[i, 0] - lab[j, 0]) + 2
            sg = 1.0 if lab[i, 0] >= lab[j, 0] else -1.0
            lab[i, 0] = min(max(lab[i, 0] + sg * need / 2, x0 + W[i] / 2 + 2),
                            x1 - W[i] / 2 - 2)
            lab[j, 0] = min(max(lab[j, 0] - sg * need / 2, x0 + W[j] / 2 + 2),
                            x1 - W[j] / 2 - 2)

    inv = a.transData.inverted()
    for k, i in enumerate(ix):
        lx, ly = inv.transform(lab[k])
        c = ORANGE if T.problem.iloc[i] else (INK if T.heavy.iloc[i] else SEC)
        a.annotate(T.generico.iloc[i], xy=(T.r.iloc[i], T.v.iloc[i]), xytext=(lx, ly),
                   ha="center", va="center", fontsize=9.2, color=c, zorder=8,
                   fontweight="bold" if T.heavy.iloc[i] or T.problem.iloc[i] else "normal",
                   bbox=dict(facecolor=SURF, edgecolor="none", pad=0.9, alpha=0.85),
                   arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=1,
                                   shrinkB=rad[k] * 0.55))
    return sum(1 for i in range(len(lab)) for j in range(i + 1, len(lab)) if ov(i, j, lab))


resid = 0
if CLI.scale == "linear":
    resid += label_panel(axT, (T.v >= SPLIT).to_numpy(), 2.4, 66.0)
    resid += label_panel(axB, (T.v < SPLIT).to_numpy(), -0.80, SPLIT)
else:
    resid += label_panel(axT, np.ones(len(T), bool), -1.4, 130.0)
assert resid == 0, f"{resid} labels still overlap"

# ------------------------------------------------------------------ axis furniture
fmt = FuncFormatter(lambda v, _: (f"{v:.0f}%" if abs(v) >= 1 or v == 0 else f"{v:+.1f}%"))
bot = axB if axB is not None else axT
bot.set_xlabel("correlation of the wholesale 30-day change with the published CPI, "
               "2024 to date", fontsize=10.5, color=SEC, labelpad=9)
bot.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
if CLI.scale == "linear":
    axT.tick_params(labelbottom=False)
    axT.set_yticks([10, 20, 30, 40, 50, 60])
    axB.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    axT.yaxis.set_major_formatter(fmt)
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))
    fig.text(0.026, 0.462, "share of the subindex's fortnightly variance", rotation=90,
             va="center", fontsize=10.5, color=SEC)
    axT.text(0.004, 0.90, f"above {SPLIT:.1f}% of the variance", transform=axT.transAxes,
             ha="left", fontsize=9.5, color=MUT, fontweight="bold")
    axB.text(0.004, 0.955, f"the same linear axis, zoomed to 0-{SPLIT:.1f}%",
             transform=axB.transAxes, ha="left", fontsize=9.5, color=MUT,
             fontweight="bold")
    axT.text(CORR_CUT - 0.005, 64.5, f"corr {CORR_CUT:.2f}", rotation=90, va="top",
             ha="right", fontsize=9, color=MUT)
else:
    axT.set_ylabel("share of the subindex's fortnightly variance", fontsize=10.5,
                   color=SEC, labelpad=9)
    axT.set_yticks([-1, -0.5, 0, 0.5, 1, 2, 3, 10, 30, 60])
    axT.yaxis.set_major_formatter(fmt)
    axT.text(CORR_CUT - 0.005, 125, f"corr {CORR_CUT:.2f}", rotation=90, va="top",
             ha="right", fontsize=9, color=MUT)

# ------------------------------------------------------------------ text
fig.text(0.088, 0.963, "Which produce items are worth nowcasting from wholesale prices",
         fontsize=16, fontweight="bold", color=INK)
sub = ("Vertical position is what a generic contributes to the volatility of the Frutas y "
       "verduras print — weight x own volatility x correlation with the aggregate — so it "
       "is what a forecast error actually costs. Horizontal position is how closely the "
       "wholesale 30-day rate of change tracks the published CPI. Dot area is the INPC "
       "basket weight. Up and to the right is where wholesale data pays. "
       + ("The y axis is linear; because jitomate alone is 60% of the variance and the next "
          "name is 8.7%, it is split into two panels rather than compressed into one, and "
          "both panels draw every generic. Limón and Uva sit below zero — they move against "
          "the subindex."
          if CLI.scale == "linear" else
          "The y axis is symlog: a linear one puts thirty of these thirty-two points in a "
          "single band, and two generics have negative shares — they move against the "
          "subindex."))
y = 0.938
for ln in textwrap.wrap(sub, width=124):
    fig.text(0.088, y, ln, fontsize=10.6, color=SEC)
    y -= 0.0186

fig.legend(handles=[
    Line2D([], [], color=BLUE, lw=0, marker="o", markersize=9,
           label="3% or more of the variance"),
    Line2D([], [], color=ORANGE, lw=0, marker="o", markersize=9,
           label=f"material variance but corr below {CORR_CUT:.2f} — the weak spots"),
    Line2D([], [], color=GRAY, lw=0, marker="o", markersize=9,
           label="under 3% of the variance"),
    Line2D([], [], color=MUT, lw=0, marker="o", markersize=5,
           label="dot area = INPC weight")],
    loc="upper left", bbox_to_anchor=(0.088, y + 0.008), frameon=False, ncol=4,
    fontsize=10.0, handlelength=1.2, columnspacing=1.6, labelcolor=SEC)

prob = T[T.problem].sort_values("v", ascending=False)
foot = (f"Source: SNIIM (Secretaría de Economía) and INEGI. Correlations are computed on "
        f"2024 to date, matching the chartbook; the model coefficients are fitted on the "
        f"full history and will not line up exactly. Variance shares are fixed from "
        f"published CPI alone, before any model was run. The weak spots carry "
        f"{prob.v.sum():.1f}% of the variance but {100 * prob.w.sum() / T.w.sum():.1f}% of "
        f"the weight: {', '.join(prob.generico)}. Median correlation {T.r.median():.2f}; "
        f"weighted by variance share {np.average(T.r, weights=T.v.clip(lower=0)):.2f}, "
        f"weighted by basket weight {np.average(T.r, weights=T.w):.2f}.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(0.088, 0.024 + (len(lines) - 1 - i) * 0.0152, ln, fontsize=9, color=MUT)

fig.savefig(CLI.out, dpi=170)
print(f"{CLI.out}  ({CLI.scale} y axis, {resid} label overlaps)")
print(f"weak spots: {list(prob.generico)}  {prob.v.sum():.1f}% of variance, "
      f"{100 * prob.w.sum() / T.w.sum():.1f}% of weight")
print(f"top right (var>=3%, corr>=0.90): {list(T[(T.v >= 3) & (T.r >= 0.90)].generico)}")
