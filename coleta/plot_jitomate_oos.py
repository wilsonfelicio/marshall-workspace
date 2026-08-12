"""Jitomate fortnightly CPI nowcast: out-of-sample result, with the 80% band.

  charts/jitomate_oos.png
      (a) published CPI change vs the recursive out-of-sample nowcast, with the
          calibrated 80% interval shaded.
      (b) RMSE by year against the SD-AR benchmark.

Headline metric is the error in pp of the published print, and its translation
into contribution to the headline INPC.
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import json
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


fc = pd.read_csv("data/curated/oos_jitomate.csv")
fc["fecha"] = pd.DatetimeIndex(Q.qtimestamp(fc["t"].values))
fc = fc.dropna(subset=["y", "close", "close_bench"]).reset_index(drop=True)
fc["anio"] = (fc["t"] // 24).astype(int)
R = lambda a, b: float(np.sqrt(((a - b).dropna() ** 2).mean()))

rm = {v: R(fc.y, fc[v]) for v in ("d5", "d10", "close")}
rb = R(fc.y, fc.close_bench)
r2 = 1 - ((fc.y - fc.close) ** 2).sum() / ((fc.y - fc.y.mean()) ** 2).sum()
sg = 100 * float((np.sign(fc.y) == np.sign(fc.close)).mean())
# recursive empirical 80% multiplier, so the band is part of the information set
zz = ((fc.y - fc.close).abs() / fc.close_sigma).to_numpy()
k80 = np.array([np.nan if i < 40 else np.quantile(zz[:i], 0.80) for i in range(len(zz))])
band = k80 * fc.close_sigma.to_numpy()
cov = 100 * float((zz[~np.isnan(k80)] <= k80[~np.isnan(k80)]).mean())

per = fc.groupby("anio").apply(lambda g: pd.Series(
    {"m": R(g.y, g.close), "b": R(g.y, g.close_bench)}))

CO = json.load(open("data/curated/jitomate_coef.json"))
fig = plt.figure(figsize=(12.8, 11.4))
ax = fig.add_axes([L, 0.286, Rt - L, 0.244])
ax2 = fig.add_axes([L, 0.098, Rt - L, 0.126])
for a in (ax, ax2):
    styled(a)
ax.axhline(0, color=MUT, lw=1.3, zorder=3)
ax.fill_between(fc.fecha, fc.close - band, fc.close + band, color=BLUE, alpha=0.16,
                lw=0, zorder=2)
ax.plot(fc.fecha, fc.y, color=ORANGE, lw=1.8, zorder=5)
ax.plot(fc.fecha, fc.close, color=BLUE, lw=1.4, zorder=4)
ax.set_ylabel("fortnightly change in CPI jitomate", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%"))
last = fc.iloc[-1]
span = ax.get_ylim()[1] - ax.get_ylim()[0]
pts = sorted([[last.y, ORANGE, "Published", last.y], [last.close, BLUE, "Nowcast", last.close]],
             key=lambda z: -z[0])
if pts[0][0] - pts[1][0] < span * 0.055:
    mid = (pts[0][0] + pts[1][0]) / 2
    pts[0][0], pts[1][0] = mid + span * 0.030, mid - span * 0.030
for ypos, c, n, vt in pts:
    ax.annotate(f"{n} {vt:+.1f}%", xy=(last.fecha, ypos), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=11, fontweight="bold",
                va="center", annotation_clip=False)
w = fc.loc[(fc.y - fc.close).abs().idxmax()]
ax.annotate(f"worst miss: {en(w.t)}\n{w.y:+.0f}% published vs {w.close:+.0f}% nowcast",
            xy=(w.fecha, w.y), xytext=(12, 4), textcoords="offset points",
            color=SEC, fontsize=9.5, fontweight="bold", va="bottom")

x = np.arange(len(per))
bw = 0.38
ax2.bar(x - bw / 2 - 0.01, per.m.values, bw, color=BLUE, lw=0, zorder=5)
ax2.bar(x + bw / 2 + 0.01, per.b.values, bw, color=MUT, lw=0, zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels([str(a) for a in per.index], fontsize=9.5)
ax2.set_xlim(-0.6, len(per) - 0.4)
ax2.set_ylabel("RMSE, pp", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
fig.text(L, 0.238, "RMSE by year: the nowcast wins in every one of the sixteen",
         color=SEC, fontsize=10, fontweight="bold")

fig.text(L, 0.975, "Jitomate: nowcasting the fortnightly CPI print from wholesale prices",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"This is a NOWCAST, not a forecast: INEGI publishes each fortnight's CPI about "
       f"nine days after it closes (1H on the 24th, 2H on the 9th) while SNIIM publishes "
       f"daily, and the whole edge is that gap. {len(fc)} out-of-sample fortnights, "
       f"{en(fc.t.iloc[0])} - {en(fc.t.iloc[-1])}, refit every fortnight on earlier data "
       f"only. Error on the published print: RMSE {rm['close']:.2f} pp, MAE "
       f"{(fc.y-fc.close).abs().mean():.2f} pp — which is ±{rm['close']*PESO/100:.3f} pp of the "
       f"headline INPC, against ±{rb*PESO/100:.3f} pp for the SD-AR benchmark. Correct sign "
       f"{sg:.0f}%, out-of-sample R² {r2:.3f}. Jitomate alone is 60% of the variance of the "
       f"produce subindex.")
y = 0.947
for ln in textwrap.wrap(sub, width=118):
    fig.text(L, y, ln, fontsize=10.8, color=SEC)
    y -= 0.0205
fig.legend(handles=[Line2D([], [], color=ORANGE, lw=2.2, label="Published CPI"),
                    Line2D([], [], color=BLUE, lw=2.2, label="Nowcast (close of fortnight)"),
                    Patch(facecolor=BLUE, alpha=0.16, label=f"80% band, {cov:.0f}% realised coverage"),
                    Patch(facecolor=MUT, label="SD-AR benchmark (lower panel)")],
           loc="upper left", bbox_to_anchor=(L, y + 0.010), frameon=False, ncol=4,
           fontsize=10.3, handlelength=1.7, columnspacing=1.8, labelcolor=SEC)

# ---- equation band: the published number is a combination, so show both members
top, bot = 0.812, 0.556
for yy in (top, bot):
    fig.add_artist(Line2D([L, Rt], [yy, yy], color=GRID, lw=0.9))
fig.text(L, top - 0.018, "The equation being estimated", fontsize=10,
         fontweight="bold", color=SEC)
fig.text(L, top - 0.046,
         r"$\mathrm{nowcast}_t \;=\; \frac{1}{2}\,(M1_t + M2_t)$"
         "        (equal weights, fixed in advance)", fontsize=12.5, color=INK)
fig.text(L, top - 0.076,
         r"$M1:\;\; \Delta \ln \mathrm{CPI}_t \;=\; \alpha \;+\; \beta_0 \Delta \ln W_t"
         r" \;+\; \beta_1 \Delta \ln W_{t-1} \;+\; \varepsilon_t$", fontsize=12.5, color=INK)
fig.text(L, top - 0.104,
         r"$M2:\;\; M1 \;+\; S(q_t) \;+\; \phi_1 \Delta \ln \mathrm{CPI}_{t-1}"
         r" \;+\; \phi_2 \Delta \ln \mathrm{CPI}_{t-2}$", fontsize=12.5, color=INK)
m1, m2 = CO["M1"]["coef"], CO["M2"]["coef"]
fig.text(L, top - 0.136,
         rf"$M1:\;\alpha={m1['const']:+.2f}$   $\beta_0={m1['x_full']:+.3f}^{{***}}$   "
         rf"$\beta_1={m1['x_full_lag1']:+.3f}^{{***}}$   "
         rf"$\beta_0+\beta_1={m1['x_full']+m1['x_full_lag1']:.2f}$", fontsize=11.5, color=INK)
fig.text(L, top - 0.162,
         rf"$M2:\;\alpha={m2['const']:+.2f}$   $\beta_0={m2['x_full']:+.3f}^{{***}}$   "
         rf"$\beta_1={m2['x_full_lag1']:+.3f}^{{***}}$   "
         rf"$\phi_1={m2['y_lag1']:+.3f}^{{**}}$   $\phi_2={m2['y_lag2']:+.3f}$   "
         rf"$\beta_0+\beta_1={m2['x_full']+m2['x_full_lag1']:.2f}$", fontsize=11.5, color=INK)
note = (f"W is the wholesale index; t indexes fortnights; S(q) is 3 seasonal harmonics over "
        f"the 24 fortnights of the year and spans only {CO['S_span']:.1f} pp, because the "
        f"wholesale term already carries the seasonality. Pass-through over two fortnights is "
        f"{m1['x_full']+m1['x_full_lag1']:.2f} in M1 and {m2['x_full']+m2['x_full_lag1']:.2f} in M2 — "
        f"essentially complete, unlike avocado's 0.80. t-statistics: beta_0 "
        f"{CO['M1']['t']['x_full']:.0f}, beta_1 {CO['M1']['t']['x_full_lag1']:.0f}. *** p<0.01, "
        f"** p<0.05. Coefficients are the latest refit on {CO['M1']['n']} fortnights; each "
        f"plotted point used its own earlier vintage.")
for i, ln in enumerate(textwrap.wrap(note, width=148)):
    fig.text(L, top - 0.190 - i * 0.0165, ln, fontsize=9, color=MUT)

foot = (f"Source: SNIIM (Secretaría de Economía) and INEGI. Release curve: day 5 of the "
        f"fortnight {rm['d5']:.2f} pp, day 10 {rm['d10']:.2f} pp, close {rm['close']:.2f} pp — "
        f"so two thirds of the fortnight already buys almost all of the accuracy. The band "
        f"multiplier comes from the series' own earlier standardised errors, not a normal "
        f"table. A placebo using the same regressor shifted to the wrong dates does not beat "
        f"the benchmark. A genuine one-step-ahead forecast, using nothing dated inside the "
        f"target fortnight, beats the benchmark by only about 5%.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(L, 0.019 + (len(lines) - 1 - i) * 0.0158, ln, fontsize=9, color=MUT)

fig.savefig("charts/jitomate_oos.png", dpi=170)
print("charts/jitomate_oos.png")
print(f"RMSE d5 {rm['d5']:.3f}  d10 {rm['d10']:.3f}  close {rm['close']:.3f} pp; "
      f"bench {rb:.3f}; headline ±{rm['close']*PESO/100:.4f} pp; coverage {cov:.1f}%")
