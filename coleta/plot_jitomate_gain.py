"""What wholesale prices add: jitomate nowcast with and without them.

  charts/jitomate_gain.png
      (a) the published print, the CPI-only model, and the wholesale model
      (b) RMSE by year for both

Both models are estimated the same way and evaluated on the same 370 fortnights;
the only difference in panel (a)'s two forecasts is whether wholesale prices are
in the information set.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

PESO = 0.79014
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GRAY = "#6f6d67"          # the CPI-only model: a reference, deliberately recessive
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})
L, Rt = 0.078, 0.845
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def en(t):
    y, rem = divmod(int(t), 24)
    m, h = divmod(rem, 2)
    return f"{h+1}H {MON[m]} {y}"


def styled(a):
    a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# Read the 32-generic system run, so this chart and every table in the deck quote the
# same rebuild. `bench` is the CPI-only model; `fit` adds wholesale prices.
import json
FACTS = json.load(open("data/curated/facts.json"))["jit"]
P = pd.read_csv("data/curated/jitomate_system.csv").set_index("t")
P = P.rename(columns={"bench": "inpc_exp", "fit": "ws_dls"})
# Score both models on the fortnights where BOTH exist. Letting pandas skip missing
# values column by column scored the benchmark on 374 periods and the model on 370,
# which is where the deck's 12.30-vs-12.33 discrepancy came from.
P = P.dropna(subset=["y", "ws_dls", "inpc_exp"])
P.index.name = "t"
P["fecha"] = pd.DatetimeIndex(Q.qtimestamp(P.index.values))
P["anio"] = (P.index // 24).astype(int)
R = lambda c: float(np.sqrt(((P.y - P[c]) ** 2).mean()))
OLD, NEW = "inpc_exp", "ws_dls"
r_old, r_new = R(OLD), R(NEW)
assert abs(r_new - FACTS["rmse_a"]) < 5e-4 and abs(r_old - FACTS["rmse_b"]) < 5e-4, \
    "chart disagrees with facts.json"
sg = lambda c: 100 * float((np.sign(P.y) == np.sign(P[c])).mean())
r2 = lambda c: 1 - ((P.y - P[c]) ** 2).sum() / ((P.y - P.y.mean()) ** 2).sum()
per = P.groupby("anio").apply(lambda g: pd.Series(
    {"new": np.sqrt(((g.y - g[NEW]) ** 2).mean()),
     "old": np.sqrt(((g.y - g[OLD]) ** 2).mean())}))

fig = plt.figure(figsize=(12.8, 9.8))
ax = fig.add_axes([L, 0.404, Rt - L, 0.292])
ax2 = fig.add_axes([L, 0.178, Rt - L, 0.146])
for a in (ax, ax2):
    styled(a)
ax.axhline(0, color=MUT, lw=1.1, zorder=3)
ax.plot(P.fecha, P[OLD], color=GRAY, lw=1.3, alpha=0.9, zorder=4)
ax.plot(P.fecha, P.y, color=ORANGE, lw=1.9, zorder=6)
ax.plot(P.fecha, P[NEW], color=BLUE, lw=1.4, zorder=5)
ax.set_ylabel("fortnightly change in CPI jitomate", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%"))
sp = ax.get_ylim()[1] - ax.get_ylim()[0]
lastrow = P.iloc[-1]
pts = sorted([[lastrow.y, ORANGE, "Published"], [lastrow[NEW], BLUE, "With wholesale"],
              [lastrow[OLD], GRAY, "CPI only"]], key=lambda z: -z[0])
ys = [p[0] for p in pts]
for i in range(1, len(ys)):
    if ys[i - 1] - ys[i] < sp * 0.055:
        ys[i] = ys[i - 1] - sp * 0.055
for (val, c, n), yy in zip(pts, ys):
    ax.annotate(f"{n} {val:+.1f}%", xy=(lastrow.fecha, yy), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=10.5, fontweight="bold",
                va="center", annotation_clip=False, zorder=9,
                bbox=dict(facecolor=SURF, edgecolor="none", pad=1.4))
# The biggest published move is reported in the footer, not annotated in place: any box
# large enough to hold the three numbers covered the trough it was describing.
big = P.loc[P.y.abs().idxmax()]

x = np.arange(len(per))
bw = 0.38
ax2.bar(x - bw / 2 - 0.01, per.new.values, bw, color=BLUE, lw=0, zorder=5)
ax2.bar(x + bw / 2 + 0.01, per.old.values, bw, color=GRAY, lw=0, zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels([str(a) for a in per.index], fontsize=9.5)
ax2.set_xlim(-0.6, len(per) - 0.4)
ax2.set_ylabel("RMSE, pp", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
fig.text(L, 0.336, "RMSE by year: wholesale wins in all sixteen, by a factor of two to three",
         color=SEC, fontsize=10, fontweight="bold")

fig.text(L, 0.972, "What wholesale prices add: jitomate CPI, with and without them",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"Both models nowcast the same {len(P)} fortnights, {en(P.index[0])} - "
       f"{en(P.index[-1])}, refit every fortnight on earlier data only. The CPI-only model "
       f"uses seasonality and two lags of the CPI itself — no wholesale data at all. Error on "
       f"the published print falls from {r_old:.2f} pp to {r_new:.2f} pp, i.e. from "
       f"±{r_old*PESO/100:.3f} to ±{r_new*PESO/100:.3f} pp of the headline INPC: a "
       f"{100*(1-r_new/r_old):.0f}% reduction. Correct sign rises from {sg(OLD):.0f}% to "
       f"{sg(NEW):.0f}%, out-of-sample R² from {r2(OLD):.2f} to {r2(NEW):.2f}. "
       f"Diebold-Mariano t = {FACTS['dm_t']:.1f} in favour of the wholesale model.")
y = 0.940
for ln in textwrap.wrap(sub, width=118):
    fig.text(L, y, ln, fontsize=10.8, color=SEC)
    y -= 0.0225
fig.legend(handles=[Line2D([], [], color=ORANGE, lw=2.2, label="Published CPI"),
                    Line2D([], [], color=BLUE, lw=2.2, label="With wholesale prices"),
                    Line2D([], [], color=GRAY, lw=2.2, label="CPI data only"),
                    Patch(facecolor=GRAY, label="CPI only (lower panel)")],
           loc="upper left", bbox_to_anchor=(L, y + 0.012), frameon=False, ncol=4,
           fontsize=10.5, handlelength=1.7, columnspacing=1.9, labelcolor=SEC)

foot = (f"Biggest published move in the sample: {en(big.name)}, published {big.y:+.0f}%, "
        f"nowcast with wholesale {big[NEW]:+.0f}%, CPI only {big[OLD]:+.0f}%. "
        "Source: SNIIM (Secretaría de Economía) and INEGI. Attribution of the gain, each step "
        "holding the other fixed: adding wholesale prices to the same estimator is −56%; "
        "additionally downweighting older observations is a further −14%, and that second step "
        "helps ONLY the wholesale model — on the CPI-only model it is slightly harmful, because "
        "the drift is in the pass-through coefficient, not in the CPI's own dynamics. "
        "Pass-through within the fortnight has risen from 0.54 in 1999-2007 to 0.83 in 2017-2026. "
        "This is a nowcast: it sharpens the print INEGI has not published yet, using wholesale "
        "prices dated inside the fortnight, and it is not a forward-looking forecast.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(L, 0.026 + (len(lines) - 1 - i) * 0.0166, ln, fontsize=9, color=MUT)

fig.savefig("charts/jitomate_gain.png", dpi=170)
print("charts/jitomate_gain.png")
print(f"CPI only {r_old:.3f} pp (±{r_old*PESO/100:.4f}) -> with wholesale {r_new:.3f} pp "
      f"(±{r_new*PESO/100:.4f}) = {100*(1-r_new/r_old):.1f}% better; sign {sg(OLD):.0f}% -> {sg(NEW):.0f}%")
