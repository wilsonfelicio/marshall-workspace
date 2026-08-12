"""Views for the 32-generic system.

  charts/system_variance.png    share of subindex variance by generic, top-5 and
                                top-10 cutoffs marked
  charts/system_aggregate.png   published subindex vs the all-32 / top-10 / top-5
                                nowcasts, and RMSE by year
  charts/system_grid.png        32 small multiples, nowcast vs realised
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

PESO_SUB = 4.7789
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GRAY = "#6f6d67"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})
L, Rt = 0.078, 0.845
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
en = lambda t: f"{int(t) % 24 % 2 + 1}H {MON[(int(t) % 24) // 2]} {int(t) // 24}"


def styled(a, grid=True):
    if grid:
        a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def head(fig, title, sub, legend=None, y0=0.968, w=118, step=0.0235):
    fig.text(L, y0, title, fontsize=16, fontweight="bold", color=INK)
    y = y0 - 0.032
    for ln in textwrap.wrap(sub, width=w):
        fig.text(L, y, ln, fontsize=10.8, color=SEC)
        y -= step
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(L, y + 0.012),
                   frameon=False, ncol=len(legend), fontsize=10.5, handlelength=1.7,
                   columnspacing=1.9, labelcolor=SEC)
    return y


def foot(fig, txt, width=150, bottom=0.032, step=0.0165):
    lines = textwrap.wrap(txt, width=width)
    for i, ln in enumerate(lines):
        fig.text(L, bottom + (len(lines) - 1 - i) * step, ln, fontsize=9, color=MUT)


pri = pd.read_csv("data/curated/prioridad_varianza.csv")
pri["name"] = pri.generico.str.split(" ", n=1).str[1]
import json as _json
FACTS = _json.load(open("data/curated/facts.json"))
SPL = FACTS["split"]
_ord = {r["generico"]: i for i, r in enumerate(FACTS["table"])}
pri["__o"] = pri.name.map(_ord)
pri = pri.sort_values("__o").reset_index(drop=True)
tab = pd.read_csv("data/curated/system_scores.csv")
# idempotent: this script rewrites system_scores.csv, so drop a previously merged
# share column before merging again rather than creating share_x / share_y
tab = tab.drop(columns=[c for c in ("share", "name") if c in tab.columns])
tab = tab.merge(pri[["name", "share"]], left_on="generico", right_on="name", how="left")
tab["share"] = tab["share"].fillna(0.0)
ORDER = [r["generico"] for r in FACTS["table"]]
tab["__o"] = tab.generico.map({g: i for i, g in enumerate(ORDER)})
tab = tab.sort_values("__o").reset_index(drop=True)
tab.to_csv("data/curated/system_scores.csv", index=False)
A = pd.read_csv("data/curated/system_aggregate.csv")
A["fecha"] = pd.DatetimeIndex(Q.qtimestamp(A.t.values))
A["anio"] = (A.t // 24).astype(int)
fc = pd.read_csv("data/curated/system_forecasts.csv")

# ------------------------------------------------------------------ 1. variance
fig = plt.figure(figsize=(12.8, 8.6))
ax = fig.add_axes([0.30, 0.158, 0.615, 0.622])
styled(ax, grid=False)
ax.grid(axis="x", color=GRID, lw=0.9)
ax.set_axisbelow(True)
p = pri.iloc[::-1].reset_index(drop=True)
n = len(p)
colors = [BLUE if s >= pri.share.iloc[4] else (AQUA if s >= pri.share.iloc[9] else GRAY)
          for s in p.share]
ax.barh(np.arange(n), p.share.values, 0.72, color=colors, lw=0)
ax.set_yticks(np.arange(n))
ax.set_yticklabels(p.name.values, fontsize=9.5)
ax.set_xlabel("share of the subindex's fortnightly variance", fontsize=10.5, color=SEC,
              labelpad=8)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
for i, (s, nm) in enumerate(zip(p.share.values, p.name.values)):
    if s > 1.0 or s < -0.05:
        # Negative labels are anchored to the right of the zero line, not to the bar end:
        # a 5px-wide bar puts its label on top of the category label and the spine.
        ax.annotate(f"{s:.1f}%", xy=(max(s, 0.0), i), xytext=(6, 0),
                    textcoords="offset points", va="center", color=SEC, fontsize=9,
                    fontweight="bold", ha="left")
# Six generics contribute ~0% and two of them negatively; starting the axis at 0
# clipped those bars out of frame and a reader counting bars found 26 of 32.
ax.set_xlim(-1.9, 66)
ax.axvline(0, color=AXIS, lw=0.9, zorder=4)
head(fig, "Where produce inflation actually comes from",
     "Contribution of each generic to the variance of the fortnightly change in "
     "INEGI's Frutas y verduras subindex, 2016-2026. Contribution is weight x own "
     "volatility x correlation with the aggregate — not weight alone, which ranks "
     f"Frijol second and contributes nothing. Jitomate alone is {SPL['jitomate']:.1f}%; "
     f"the top five are {SPL['top5']:.1f}%; the top ten are {SPL['top10']:.1f}%; the "
     f"remaining 22 share {SPL['rest22']:.1f}% between them — four of those "
     f"contribute nothing measurable and two enter with the wrong sign.",
     [Patch(facecolor=BLUE, label=f"top 5 ({SPL['top5']:.1f}% of variance)"),
      Patch(facecolor=AQUA, label=f"next 5 (to {SPL['top10']:.1f}%)"),
      Patch(facecolor=GRAY, label=f"remaining 22 ({SPL['rest22']:.1f}%)")])
foot(fig, "Source: INEGI, published generic indices, fortnightly log changes since 2016. "
          "This ranking needs no wholesale data and was fixed before any model was run, "
          "which is why it can be used to prioritise without contaminating the "
          "out-of-sample evaluation.")
fig.savefig("charts/system_variance.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 2. aggregate
R = lambda a, b: float(np.sqrt(((a - b).dropna() ** 2).mean()))
r32, r10, r5 = (R(A.y, A[f"close_{k}"]) for k in ("all32", "top10", "top5"))
rb = R(A.y, A.close_bench)
fig = plt.figure(figsize=(12.8, 9.8))
ax = fig.add_axes([L, 0.398, Rt - L, 0.294])
ax2 = fig.add_axes([L, 0.152, Rt - L, 0.148])
for a in (ax, ax2):
    styled(a)
ax.axhline(0, color=MUT, lw=1.1, zorder=3)
ax.plot(A.fecha, A.close_bench, color=GRAY, lw=1.0, alpha=0.85, zorder=3)
ax.plot(A.fecha, A.y, color=ORANGE, lw=1.9, zorder=6)
ax.plot(A.fecha, A.close_all32, color=BLUE, lw=1.4, zorder=5)
ax.set_ylabel("fortnightly change, subindex", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%"))
sp = ax.get_ylim()[1] - ax.get_ylim()[0]
last = A.iloc[-1]
pts = sorted([[last.y, ORANGE, "Published"], [last.close_all32, BLUE, "Nowcast"],
              [last.close_bench, GRAY, "Benchmark"]], key=lambda z: -z[0])
ys = [q[0] for q in pts]
for i in range(1, len(ys)):
    if ys[i - 1] - ys[i] < sp * 0.055:
        ys[i] = ys[i - 1] - sp * 0.055
for (val, c, nm), yy in zip(pts, ys):
    ax.annotate(f"{nm} {val:+.1f}%", xy=(last.fecha, yy), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=10.5, fontweight="bold",
                va="center", annotation_clip=False)

x = np.arange(len(A.anio.unique()))
per = A.groupby("anio").apply(lambda g: pd.Series(
    {"a32": R(g.y, g.close_all32), "t10": R(g.y, g.close_top10),
     "t5": R(g.y, g.close_top5), "b": R(g.y, g.close_bench)}))
w = 0.21
for k, (col, c, lab) in enumerate((("a32", BLUE, "all 32"), ("t10", AQUA, "top 10"),
                                   ("t5", VIOLET, "top 5"), ("b", GRAY, "benchmark"))):
    ax2.bar(x + (k - 1.5) * w, per[col].values, w * 0.92, color=c, lw=0, zorder=5)
ax2.set_xticks(x)
ax2.set_xticklabels([str(a) for a in per.index], fontsize=9.5)
ax2.set_xlim(-0.6, len(per) - 0.4)
ax2.set_ylabel("RMSE, pp", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
fig.text(L, 0.322, "RMSE by year and by how many generics are modelled", color=SEC,
         fontsize=10, fontweight="bold")
head(fig, "Nowcasting the produce subindex: how much coverage do you need?",
     f"Bottom-up from the 32 generics with published weights, {len(A)} out-of-sample "
     f"fortnights. Error on the published print: all 32 modelled {r32:.2f} pp, top 10 "
     f"{r10:.2f} pp, top 5 {r5:.2f} pp, against {rb:.2f} pp for a CPI-only benchmark — "
     f"i.e. the produce contribution to the headline print can be called to "
     f"±{r32*PESO_SUB/100:.3f} pp with all 32, ±{r10*PESO_SUB/100:.3f} with ten, "
     f"±{r5*PESO_SUB/100:.3f} with five, against ±{rb*PESO_SUB/100:.3f} without wholesale "
     f"data. The subindex's own sd is {A.y.std():.2f} pp = {A.y.std()*PESO_SUB/100:.3f} pp "
     f"of headline.",
     [Line2D([], [], color=ORANGE, lw=2.2, label="Published subindex"),
      Line2D([], [], color=BLUE, lw=2.2, label="Nowcast, all 32"),
      Line2D([], [], color=GRAY, lw=2.2, label="CPI-only benchmark"),
      Patch(facecolor=AQUA, label="top 10 (lower panel)"),
      Patch(facecolor=VIOLET, label="top 5 (lower panel)")])
foot(fig, f"Source: SNIIM and INEGI. Non-members of a set are carried at their own "
          f"CPI-only benchmark rather than dropped, so the weight base is identical in all "
          f"three. Five generics get {100*(1-r5/rb):.0f}% of the way and all 32 get "
          f"{100*(1-r32/rb):.0f}%: the marginal 27 are worth {100*(r32/r5-1):+.0f}% on the "
          f"error. This is a nowcast — it uses wholesale prices dated inside the fortnight, "
          f"published about nine days before INEGI's print.")
fig.savefig("charts/system_aggregate.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 3. grid
order = tab.generico.tolist()
gen_slug = dict(zip(tab.generico, tab.generico))
fc["fecha"] = pd.DatetimeIndex(Q.qtimestamp(fc.t.values))
slug_of = {}
import json
meta = json.load(open("data/curated/system_meta.json"))
labels = (pd.read_parquet("data/curated/cat_national_monthly.parquet",
                          columns=["categoria", "categoria_label"])
          .drop_duplicates().set_index("categoria_label")["categoria"].to_dict())
# Two halves of sixteen instead of one grid of 32: at 13-inch slide width a 4x8 grid
# renders panel titles at about 5pt, which the visual QA flagged as unreadable.
ncol = 4
for half, (lo, hi, tag, sub2) in enumerate((
        (0, 16, "a", f"The sixteen largest contributors, {SPL['first16']:.1f}% of the "
                     f"subindex's variance between them."),
        (16, 32, "b", f"The tail: {SPL['last16']:.1f}% of the variance between them. One of "
                      f"the two generics gated out on data quality sits here and is shown at "
                      f"its benchmark."))):
    names = order[lo:hi]
    fig = plt.figure(figsize=(13.6, 8.2))
    for k, gname in enumerate(names):
        sl = labels.get(gname)
        g = fc[fc.slug == sl].sort_values("t")
        r, c = divmod(k, ncol)
        axk = fig.add_axes([0.055 + c * 0.232, 0.700 - r * 0.174, 0.196, 0.110])
        for sp in ("top", "right"):
            axk.spines[sp].set_visible(False)
        axk.axhline(0, color=GRID, lw=0.8)
        row = tab[tab.generico == gname].iloc[0]
        # RMSE pair for the title comes from facts.json, scored on the fortnights where
        # both the nowcast and the benchmark exist, so it matches the deck's table.
        fr = next(r for r in FACTS["table"] if r["generico"] == gname)
        # Panel titles are not clipped by matplotlib, so a long name plus the RMSE pair
        # ran off the right edge of the figure. Shorten the label, not the font.
        SHORT = {"Otras verduras y legumbres": "Otras verduras",
                 "Cilantro, epazote y perejil": "Cilantro/epazote",
                 "Papa y otros tubérculos": "Papa y tubérculos",
                 "Otros chiles frescos": "Otros chiles"}
        nm_short = SHORT.get(gname, gname)
        if len(g) and row.admisible:
            axk.plot(g.fecha, g.y, color=ORANGE, lw=1.0)
            axk.plot(g.fecha, g.close_combo, color=BLUE, lw=0.9)
            ttl = f"{nm_short}   {fr['close']:.1f} vs {fr['bench']:.1f} pp"
        elif len(g):
            axk.plot(g.fecha, g.y, color=ORANGE, lw=1.0)
            axk.plot(g.fecha, g.close_bench, color=GRAY, lw=0.9)
            ttl = f"{nm_short}   gated"
        else:
            ttl = f"{nm_short}   no data"
        axk.set_title(ttl, fontsize=9.5, color=INK, loc="left", pad=4)
        axk.tick_params(labelsize=8.5, length=2)
        axk.set_xticks(pd.to_datetime(["2012-01-01", "2018-01-01", "2024-01-01"]))
        axk.set_xticklabels(["2012", "2018", "2024"], fontsize=8.5)
        # A bare label inside the panel was struck through by the plotted lines on
        # roughly half the panels; the surface-coloured box behind it fixes that
        # without moving the label off the panel it describes.
        axk.text(0.965, 0.09, f"{fr['share']:.1f}% of variance", transform=axk.transAxes,
                 ha="right", va="bottom", fontsize=8, color=MUT, fontweight="bold",
                 zorder=8, bbox=dict(facecolor=SURF, edgecolor="none", pad=2.8))
    fig.text(0.055, 0.955, f"Nowcast vs realised, {'1' if half == 0 else '2'} of 2  ·  "
                           f"ordered by contribution to subindex variance",
             fontsize=15, fontweight="bold", color=INK)
    yg = 0.928
    for ln in textwrap.wrap(sub2 + " Panel titles give the nowcast's RMSE against the "
                            "CPI-only benchmark, both in pp of the published fortnightly "
                            "print. Axes are per-panel: these series differ by an order of "
                            "magnitude in volatility.", width=140):
        fig.text(0.055, yg, ln, fontsize=9.5, color=SEC)
        yg -= 0.020
    fig.text(0.055, yg - 0.006, "Orange = published CPI    Blue = wholesale nowcast    "
                                "Grey = benchmark (gated generics)", fontsize=9.5,
             color=SEC, fontweight="bold")
    _nr = FACTS["n_range"]
    _src = (f"Source: SNIIM (Secretaría de Economía) and INEGI. Evaluation window 1H Jan 2011 "
            f"– 2H Jul 2026: {_nr[-1]} fortnights, of which {_nr[0]}–{_nr[-1]} are scorable per "
            f"generic. Refit every fortnight on a rolling five-year window using earlier data "
            f"only; each panel is scored only where both the nowcast and the benchmark exist.")
    for _i, _ln in enumerate(textwrap.wrap(_src, width=152)):
        fig.text(0.055, 0.040 - _i * 0.0155, _ln, fontsize=9, color=MUT)
    fig.savefig(f"charts/system_grid_{tag}.png", dpi=155)
    plt.close(fig)

print("charts/system_variance.png\ncharts/system_aggregate.png\ncharts/system_grid_a.png\ncharts/system_grid_b.png")
print(f"aggregate close: all32 {r32:.3f}  top10 {r10:.3f}  top5 {r5:.3f}  bench {rb:.3f} pp")
print(f"headline: ±{r32*PESO_SUB/100:.4f} / ±{r10*PESO_SUB/100:.4f} / ±{r5*PESO_SUB/100:.4f} "
      f"vs ±{rb*PESO_SUB/100:.4f} pp")
print(f"admitted {int(tab.admisible.sum())}/32; gated: "
      f"{list(tab[~tab.admisible].generico)}")
