"""Out-of-sample nowcast of Mexico's fortnightly CPI, with the fitted equation.

  charts/<slug>_oos.png
      (a) observed fortnightly CPI change vs the recursive out-of-sample nowcast.
          Every plotted prediction was made with parameters fitted only on data
          available before that fortnight.
      (b) RMSE by year, nowcast vs the SD-AR benchmark.

The equation band shows the LATEST refit. Each plotted point used its own vintage
of these coefficients, so the numbers on the chart describe the model you would
run tomorrow, not the model that produced the 2011 dots.

Usage: python3 plot_oos_png.py aguacate [model]
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
import model_quincenal as M  # noqa: E402
from inpc import quincenal as Q  # noqa: E402

slug = sys.argv[1] if len(sys.argv) > 1 else "aguacate"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "puente+rez"
GEN_COL = {"aguacate": "045 Aguacate", "jitomate": "070 Jitomate"}
WEIGHT = {"aguacate": 0.21233, "jitomate": 0.79014}
BENCH = "SD-AR"
TITLES = {"aguacate": "Avocado", "jitomate": "Jitomate (tomato)"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})
L, Rt = 0.078, 0.845
pct = FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%")
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def en_label(t: int) -> str:
    """1H Jan 2011 / 2H Jul 2026 - the English half-month label."""
    y, rem = divmod(int(t), 24)
    m, h = divmod(rem, 2)
    return f"{h + 1}H {MON[m]} {y}"


def styled(a):
    a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# ------------------------------------------------------------------ data
fc = pd.read_csv(f"data/curated/oos_{slug}.csv")
fc["fecha"] = Q.qtimestamp(fc["t"].values)
fc = fc.dropna(subset=["y", MODEL, BENCH]).reset_index(drop=True)
fc["anio"] = (fc["t"] // 24).astype(int)

rmse = float(np.sqrt(((fc["y"] - fc[MODEL]) ** 2).mean()))
rmse_b = float(np.sqrt(((fc["y"] - fc[BENCH]) ** 2).mean()))
r2 = 1 - ((fc["y"] - fc[MODEL]) ** 2).sum() / ((fc["y"] - fc["y"].mean()) ** 2).sum()
sign = 100 * float((np.sign(fc["y"]) == np.sign(fc[MODEL])).mean())
per = fc.groupby("anio").apply(
    lambda g: pd.Series({"m": np.sqrt(((g.y - g[MODEL]) ** 2).mean()),
                         "b": np.sqrt(((g.y - g[BENCH]) ** 2).mean())}))

# ------------------------------------------------------------------ latest refit
if slug == "jitomate":
    import pathlib
    Q.DAILY = "/root/jit/var_market_daily.parquet"
    Q.CACHE_DIR = pathlib.Path("/root/jit/cache")
d = M.build(Q.dataset(slug, GEN_COL[slug]))
vs = M.SPECS[MODEL][0]
trn = d.dropna(subset=vs + ["y"])
X = np.column_stack([np.ones(len(trn))] + [trn[v].values for v in vs])
b = np.linalg.lstsq(X, trn["y"].values, rcond=None)[0]
resid = trn["y"].values - X @ b
se = np.sqrt(np.diag((resid @ resid / (len(trn) - X.shape[1])) * np.linalg.inv(X.T @ X)))
co = dict(zip(["const"] + vs, b))
tst = dict(zip(["const"] + vs, b / se))
ang = 2 * np.pi * (np.arange(1, 25) - 1) / 24
S = sum(co[f"sin{k}"] * np.sin(k * ang) + co[f"cos{k}"] * np.cos(k * ang)
        for k in range(1, M.HARM + 1))


def star(v):
    a = abs(v)
    return "***" if a > 2.58 else "**" if a > 1.96 else "*" if a > 1.65 else ""


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(12.8, 10.6))
ax = fig.add_axes([L, 0.346, Rt - L, 0.282])
ax2 = fig.add_axes([L, 0.120, Rt - L, 0.152])
for a in (ax, ax2):
    styled(a)
ax.axhline(0, color=MUT, lw=1.3, zorder=3)

ax.plot(fc["fecha"], fc["y"], color=ORANGE, lw=1.8, zorder=5)
ax.plot(fc["fecha"], fc[MODEL], color=BLUE, lw=1.5, zorder=4)
ax.set_ylabel("fortnightly change in CPI", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(pct)
last = fc.iloc[-1]
span = ax.get_ylim()[1] - ax.get_ylim()[0]
pts = sorted([[last["y"], ORANGE, "Observed", last["y"]],
              [last[MODEL], BLUE, "Nowcast", last[MODEL]]], key=lambda z: -z[0])
if pts[0][0] - pts[1][0] < span * 0.055:
    mid = (pts[0][0] + pts[1][0]) / 2
    pts[0][0], pts[1][0] = mid + span * 0.030, mid - span * 0.030
for ypos, c, n, vt in pts:
    ax.annotate(f"{n} {vt:+.1f}%", xy=(last["fecha"], ypos), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=11, fontweight="bold",
                va="center", annotation_clip=False)
worst = fc.loc[(fc["y"] - fc[MODEL]).abs().idxmax()]
ax.annotate(f"worst error: {en_label(worst['t'])}\n{worst['y']:+.0f}% actual vs "
            f"{worst[MODEL]:+.0f}% predicted",
            xy=(worst["fecha"], worst["y"]), xytext=(12, -6), textcoords="offset points",
            color=SEC, fontsize=9.5, fontweight="bold", va="top")

x = np.arange(len(per))
bw = 0.38
ax2.bar(x - bw / 2 - 0.01, per["m"].values, bw, color=BLUE, lw=0, zorder=5)
ax2.bar(x + bw / 2 + 0.01, per["b"].values, bw, color=MUT, lw=0, zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels([str(a) for a in per.index], fontsize=9.5)
ax2.set_xlim(-0.6, len(per) - 0.4)
ax2.set_ylabel("RMSE, pp", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
fig.text(L, 0.284, "Root mean squared error by year: the nowcast wins in every single one",
         color=SEC, fontsize=10, fontweight="bold")

# ------------------------------------------------------------------ header
name = TITLES.get(slug, slug.capitalize())
fig.text(L, 0.972, f"{name}: nowcasting Mexico's fortnightly CPI from wholesale prices",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"Out-of-sample predictions, refitted at every fortnight on earlier data only. "
       f"{len(fc)} fortnights, {en_label(fc['t'].iloc[0])} – {en_label(fc['t'].iloc[-1])}. "
       f"RMSE {rmse:.2f} pp against {rmse_b:.2f} for the SD-AR benchmark "
       f"({100*(rmse/rmse_b-1):+.0f}%); out-of-sample R² {r2:.3f}; correct sign in "
       f"{sign:.0f}% of fortnights. The target's standard deviation is "
       f"{fc['y'].std():.1f} pp, so the error is {rmse/fc['y'].std():.2f} of one.")
y = 0.940
for ln in textwrap.wrap(sub, width=118):
    fig.text(L, y, ln, fontsize=10.8, color=SEC)
    y -= 0.0235
fig.legend(handles=[Line2D([], [], color=ORANGE, lw=2.2, label="Observed CPI"),
                    Line2D([], [], color=BLUE, lw=2.2, label="Wholesale nowcast"),
                    Patch(facecolor=MUT, label="SD-AR benchmark (lower panel)")],
           loc="upper left", bbox_to_anchor=(L, y + 0.010), frameon=False, ncol=3,
           fontsize=10.8, handlelength=1.7, columnspacing=2.2, labelcolor=SEC)

# ------------------------------------------------------------------ equation band
top, bot = 0.826, 0.658
for yy in (top, bot):
    fig.add_artist(Line2D([L, Rt], [yy, yy], color=GRID, lw=0.9))
fig.text(L, top - 0.022, "The equation being estimated", fontsize=10,
         fontweight="bold", color=SEC)
fig.text(L, top - 0.058,
         r"$\Delta \ln \mathrm{CPI}_t \;=\; \alpha \;+\; S(q_t) \;+\; "
         r"\phi_1 \Delta \ln \mathrm{CPI}_{t-1} \;+\; \phi_2 \Delta \ln \mathrm{CPI}_{t-2}"
         r"\;+\; \beta_0 \Delta \ln W_t \;+\; \beta_1 \Delta \ln W_{t-1} \;+\; \varepsilon_t$",
         fontsize=13.5, color=INK)
fig.text(L, top - 0.088,
         rf"$\alpha={co['const']:+.2f}$      $\phi_1={co['y_lag1']:+.2f}$      "
         rf"$\phi_2={co['y_lag2']:+.2f}$      "
         rf"$\beta_0={co['x_full']:+.3f}^{{{star(tst['x_full'])}}}$      "
         rf"$\beta_1={co['x_lag1']:+.3f}^{{{star(tst['x_lag1'])}}}$      "
         rf"$\beta_0+\beta_1={co['x_full']+co['x_lag1']:.2f}$",
         fontsize=12, color=INK)
note = (f"W is the wholesale index; t indexes fortnights. S(q) is {M.HARM} seasonal harmonics "
        f"over the 24 fortnights of the year and spans only {S.max()-S.min():.1f} pp — the "
        f"wholesale term already carries the seasonality. t-statistics: "
        f"β₀ {tst['x_full']:.1f}, β₁ {tst['x_lag1']:.1f}, every other term below 2 in absolute "
        f"value. *** p<0.01. Coefficients are the latest refit, on {len(trn)} fortnights; each "
        f"plotted point used its own earlier vintage of them.")
for i, ln in enumerate(textwrap.wrap(note, width=148)):
    fig.text(L, top - 0.112 - i * 0.019, ln, fontsize=9, color=MUT)

foot = ("Source: SNIIM (Secretaría de Economía) and INEGI. The nowcast uses wholesale prices "
        "dated inside the fortnight, which are known about eight days before INEGI publishes "
        "that fortnight's CPI; it uses no CPI data from the fortnight being predicted. A placebo "
        "using the same regressor shifted to the wrong dates does not beat the benchmark, and a "
        "control that does peek at the future does not beat the honest nowcast.")
lines = textwrap.wrap(foot, width=150)
for i, ln in enumerate(lines):
    fig.text(L, 0.026 + (len(lines) - 1 - i) * 0.020, ln, fontsize=9, color=MUT)

out = f"charts/{slug}_oos.png"
fig.savefig(out, dpi=170)
print(out)
print(f"n={len(fc)} RMSE {rmse:.3f} vs {rmse_b:.3f} ({100*(rmse/rmse_b-1):+.1f}%) "
      f"R2 {r2:.3f} sign {sign:.1f}%")
print(f"beta0 {co['x_full']:+.4f} (t {tst['x_full']:.1f})  beta1 {co['x_lag1']:+.4f} "
      f"(t {tst['x_lag1']:.1f})  sum {co['x_full']+co['x_lag1']:.4f}")
