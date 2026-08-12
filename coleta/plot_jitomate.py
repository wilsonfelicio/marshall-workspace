"""Jitomate: variety price levels, the index we build from them, and the log ratio
against retail.

  charts/jitomate_varieties.png  Tomate Bola and Tomate Saladette in MXN/kg, plus
                                 the chained matched-cell index anchored to MXN/kg
                                 so it is directly comparable to its own inputs.
  charts/jitomate_margin.png     wholesale index vs retail MXN/kg (INEGI Precios
                                 Promedio) on a log axis, and ln(retail/wholesale).
  charts/jitomate_index_ratio.png wholesale index vs the published CPI index, both
                                 rebased, and their log ratio over the full history.

Reads the isolated jitomate store at /root/jit while the main store is locked.
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
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, ".")
from inpc import catalogo, precios_hist, quincenal as Q  # noqa: E402

Q.DAILY = "/root/jit/var_market_daily.parquet"      # the locked store's stand-in
Q.CACHE_DIR = __import__("pathlib").Path("/root/jit/cache")

BLUE, ORANGE, VIOLET, AQUA = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})
W_SAL, W_BOLA, ANCHOR_N = 0.65, 0.35, 48
L, Rt = 0.078, 0.845
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
en = lambda d: f"{MON[d.month-1]} {d.year}"


def styled(a):
    a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


def head(fig, title, sub, legend, y0=0.968, w=118):
    fig.text(L, y0, title, fontsize=16, fontweight="bold", color=INK)
    y = y0 - 0.032
    for ln in textwrap.wrap(sub, width=w):
        fig.text(L, y, ln, fontsize=10.8, color=SEC)
        y -= 0.0245
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(L, y + 0.012),
                   frameon=False, ncol=len(legend), fontsize=10.8, handlelength=1.7,
                   columnspacing=2.0, labelcolor=SEC)
    return y


def foot(fig, txt, width=150, bottom=0.026, step=0.019):
    lines = textwrap.wrap(txt, width=width)
    for i, ln in enumerate(lines):
        fig.text(L, bottom + (len(lines) - 1 - i) * step, ln, fontsize=9, color=MUT)


def spread_log(pts, frac=0.055):
    """Push end labels apart on a log axis so they cannot overlap.

    pts is [(value, colour, name), ...]. Returns the same list with a separate
    plotting height per label, keeping the printed value untouched.
    """
    pts = sorted(pts, key=lambda z: -z[0])
    lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
    span = np.log(hi / lo) if hi > lo else 1.0
    gap = max(span, 0.35) * frac
    ys = [np.log(p[0]) for p in pts]
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < gap:
            ys[i] = ys[i - 1] - gap
    return [(np.exp(y), p[0], p[1], p[2]) for y, p in zip(ys, pts)]


def logticks(ax, lo, hi):
    t = [v for v in (2, 3, 5, 7, 10, 15, 20, 30, 40, 60) if lo * 0.85 <= v <= hi * 1.15]
    ax.yaxis.set_major_locator(FixedLocator(t))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))


# ------------------------------------------------------------------ data
d = pd.read_parquet(Q.DAILY)
d["fecha"] = pd.to_datetime(d["fecha"])
d["t"] = Q.qindex(d["fecha"])
pes = pd.read_parquet(Q.PESOS)[["destino", "peso_inpc"]].copy()
pes["peso"] = pd.to_numeric(pes["peso_inpc"], errors="coerce").astype(float)

# Per-variety price level: weighted geometric mean across markets, per fortnight.
# Weighted, not plain, so a variety's level and the index share one market weighting.
var = []
for pid, g in d.groupby("producto_id"):
    lab = g["producto"].iloc[0]
    cell = (g.assign(lp=np.log(g["precio_geo"]))
             .groupby(["t", "destino"], sort=False)["lp"].mean().reset_index())
    cell["w"] = cell["destino"].map(pes.set_index("destino")["peso"]).fillna(0.0)
    s = (cell[cell.w > 0].groupby("t")
         .apply(lambda x: np.exp(np.average(x.lp, weights=x.w))))
    var.append(s.rename(lab))
V = pd.concat(var, axis=1).sort_index()

# The consumption-weighted price level in actual pesos: Saladette 65 / Bola 35, each
# already a market-weighted geometric mean. This, not the chained index, is what belongs
# on a peso axis.
LVL = np.exp(W_SAL * np.log(V["Tomate Saladette"]) + W_BOLA * np.log(V["Tomate Bola"]))
LVL = LVL.dropna().rename("nivel")

# The index we actually model: chained matched-cell Jevons. It is right in CHANGES but its
# level is arbitrary, and anchoring it on its FIRST period was a mistake: matched-cell
# chaining accumulates composition change (+0.10 in logs over 28 years here) and the old
# anchor also pooled the two varieties equally instead of 65/35 (-0.11), so the series
# ended up 25% below the true peso level and made the retail margin read 1.69x against a
# true 1.35x. Anchor on the LAST two years instead, where the level is being read.
links = Q._links(Q._cells(d, partial=False), pes)
links["lw"] = Q._chain(links, "dln").reindex(links["t"]).values
idx = links.set_index("t")["lw"].dropna()
_co = idx.index.intersection(LVL.index)[-ANCHOR_N:]
anchor = float(np.exp(np.log(LVL.loc[_co]).mean() - idx.loc[_co].mean()))
IDX = (np.exp(idx) * anchor).rename("index")
_res = float(np.log(IDX.loc[_co]).mean() - np.log(LVL.loc[_co]).mean())
print(f"index anchored on the last {ANCHOR_N} fortnights; residual level gap "
      f"{_res:+.4f} log")
print(f"chain drift over the sample: "
      f"{float((np.log(IDX)-np.log(LVL)).dropna().iloc[-1] - (np.log(IDX)-np.log(LVL)).dropna().iloc[0]):+.4f} log")

# The current fortnight is partial (SNIIM has only the days so far), so drop it.
n_dias = d.groupby("t")["fecha"].nunique()
full = n_dias[n_dias >= 0.6 * n_dias.median()].index
V = V[V.index.isin(full)]
IDX = IDX[IDX.index.isin(full)]
LVL = LVL[LVL.index.isin(full)]
ts = pd.Index(sorted(set(V.index) | set(IDX.index)))
fecha = pd.DatetimeIndex(Q.qtimestamp(ts.values))   # Index, so [-1] is positional
V, IDX = V.reindex(ts), IDX.reindex(ts)

# ------------------------------------------------------------------ chart 1
fig = plt.figure(figsize=(12.8, 8.4))
ax = fig.add_axes([L, 0.150, Rt - L, 0.635])
styled(ax)
ax.set_yscale("log")
cols = {"Tomate Saladette": ORANGE, "Tomate Bola": AQUA}
for name, c in cols.items():
    if name in V:
        ax.plot(fecha, V[name].values, color=c, lw=1.3, alpha=0.95, zorder=4)
ax.plot(fecha, IDX.values, color=BLUE, lw=2.0, zorder=6)
logticks(ax, np.nanmin(V.min()), np.nanmax(V.max()))
ax.set_ylabel("MXN per kilogram, log scale", fontsize=10.5, color=SEC, labelpad=10)
last = ts[-1]
pts = [(V[n].dropna().iloc[-1], c, n) for n, c in cols.items() if n in V] + \
      [(IDX.dropna().iloc[-1], BLUE, "Index")]
for ypos, val, c, n in spread_log(pts):
    ax.annotate(f"{n} ${val:,.2f}", xy=(fecha[-1], ypos), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=10.5, fontweight="bold",
                va="center", annotation_clip=False)
head(fig, "Jitomate: the two varieties we track, and the index built from them",
     f"Wholesale prices in MXN/kg, fortnightly, weighted geometric mean across "
     f"{d.destino.nunique()} markets (SNIIM). The index is a chained matched-cell "
     f"Jevons over variety x market pairs present in consecutive fortnights. Its level is "
     f"arbitrary, so it is anchored to the consumption-weighted peso level (Saladette "
     f"{W_SAL:.0%} / Bola {W_BOLA:.0%}) over the last {ANCHOR_N} fortnights — not over its "
     f"first, which let 28 years of chain drift into the level. "
     f"Varieties from {en(fecha[0])}; the index starts at the first fortnight "
     f"with enough matched cells to link. {len(IDX.dropna())} indexed fortnights, ending "
     f"{en(fecha[-1])}. Log scale, so equal vertical distances are equal percentage moves.",
     [Line2D([], [], color=BLUE, lw=2.2, label="Matched-cell index (what we model)"),
      Line2D([], [], color=ORANGE, lw=2.2, label="Tomate Saladette"),
      Line2D([], [], color=AQUA, lw=2.2, label="Tomate Bola")])
foot(fig, "Source: SNIIM (Secretaría de Economía). Nominal pesos. The index is not a "
          "simple average of the two varieties: it is built on variety x market cells "
          "matched between consecutive fortnights, so a market that stops quoting one "
          "variety cannot move it. That is why it can sit between the two lines and "
          "still not track either exactly.")
fig.savefig("charts/jitomate_varieties.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ chart 2
r = precios_hist.national_kg("Jitomate", "jitomate", catalogo.city_weights())
rl = r.set_index("mes")["precio_geo"].sort_index()
wm = (pd.Series(LVL.reindex(ts).values, index=fecha).resample("MS")
      .apply(lambda z: np.exp(np.log(z.dropna()).mean())).dropna())
i2 = wm.index.intersection(rl.index)
wm, rl = wm.loc[i2], rl.loc[i2]
lr = np.log(rl) - np.log(wm)

fig = plt.figure(figsize=(12.8, 9.6))
ax = fig.add_axes([L, 0.455, Rt - L, 0.310])
ax2 = fig.add_axes([L, 0.135, Rt - L, 0.240], sharex=ax)
for a in (ax, ax2):
    styled(a)
ax.set_yscale("log")
ax.plot(i2, wm.values, color=BLUE, lw=1.7, zorder=5)
ax.plot(i2, rl.values, color=ORANGE, lw=1.7, zorder=4)
ax.fill_between(i2, wm.values, rl.values, color=MUT, alpha=0.13, lw=0, zorder=2)
logticks(ax, wm.min(), rl.max())
ax.set_ylabel("MXN/kg, log scale", fontsize=10.5, color=SEC, labelpad=10)
plt.setp(ax.get_xticklabels(), visible=False)
for val, c, n in ((rl.iloc[-1], ORANGE, "Retail"), (wm.iloc[-1], BLUE, "Wholesale")):
    ax.annotate(f"{n} ${val:,.2f}", xy=(i2[-1], val), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=11, fontweight="bold",
                va="center", annotation_clip=False)
mu, sd = float(lr.mean()), float(lr.std())
ax2.fill_between(i2, mu - sd, mu + sd, color=VIOLET, alpha=0.10, lw=0, zorder=1)
ax2.axhline(mu, color=SEC, lw=1.1, ls=(0, (1, 2)), zorder=4)
ax2.plot(i2, lr.values, color=VIOLET, lw=1.7, zorder=5)
ax2.set_ylabel("ln(retail) − ln(wholesale)", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
pad = (lr.max() - lr.min()) * 0.16
ax2.set_ylim(lr.min() - pad, lr.max() + pad)
sp2 = ax2.get_ylim()[1] - ax2.get_ylim()[0]
yl, ym = lr.iloc[-1], mu
if abs(yl - ym) < sp2 * 0.08:
    dd = sp2 * 0.045
    yl, ym = (yl + dd, ym - dd) if yl >= ym else (yl - dd, ym + dd)
ax2.annotate(f"{lr.iloc[-1]:.2f}  ({np.exp(lr.iloc[-1]):.2f}×)", xy=(i2[-1], yl),
             xytext=(9, 0), textcoords="offset points", color=VIOLET, fontsize=11,
             fontweight="bold", va="center", annotation_clip=False)
ax2.annotate(f"mean {mu:.2f}  ({np.exp(mu):.2f}×)", xy=(i2[-1], ym), xytext=(9, 0),
             textcoords="offset points", color=SEC, fontsize=10.5, va="center",
             annotation_clip=False)
for t_, va, dy in ((lr.idxmax(), "bottom", 7), (lr.idxmin(), "top", -7)):
    ax2.annotate(f"{en(t_)}  {lr[t_]:.2f} ({np.exp(lr[t_]):.2f}×)", xy=(t_, lr[t_]),
                 xytext=(0, dy), textcoords="offset points", color=SEC, fontsize=9.5,
                 ha="center", va=va, fontweight="bold")
rho = float(pd.Series(lr).autocorr(1))
hl = np.log(2) / -np.log(abs(rho)) if 0 < abs(rho) < 1 else np.nan
head(fig, "Jitomate: wholesale price, retail price, and the log ratio between them",
     f"Both in MXN/kg on one axis, so no rescaling. Wholesale: our matched-cell index, "
     f"anchored to MXN/kg. Retail: INEGI Precios Promedio, KG quotes only, weighted "
     f"geometric mean across 55 cities. {len(i2)} months, {en(i2[0])} - {en(i2[-1])}. "
     f"Mean margin {np.exp(mu):.2f}x (range {np.exp(lr.min()):.2f}x-{np.exp(lr.max()):.2f}x), "
     f"AR(1) {rho:.2f}, half-life ≈ {hl:.1f} months. On the log axis above, the vertical "
     f"gap between the lines is the number plotted below.",
     [Line2D([], [], color=BLUE, lw=2.2, label="Wholesale SNIIM"),
      Line2D([], [], color=ORANGE, lw=2.2, label="Retail (Precios Promedio)"),
      Line2D([], [], color=VIOLET, lw=2.2, label="Log ratio")])
foot(fig, "Source: SNIIM and INEGI. Nominal pesos, current month excluded. A log ratio "
          "of 0.69 means retail is twice wholesale. Retail starts in 2011 because that "
          "is when Precios Promedio starts; the wholesale series runs from 1998. Precios "
          "Promedio generic codes are not stable across basket vintages - jitomate is "
          "clave 058, then 071, then 070 - so the series is spliced by NAME.")
fig.savefig("charts/jitomate_margin.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ chart 3
p = Q.inpc_quincenal("070 Jitomate").set_index("t")["inpc"]
common = IDX.dropna().index.intersection(p.index)
BASE = 2018 * 24
w_i = IDX.loc[common]
p_i = p.loc[common]


def reb(s):
    b = s[[t_ for t_ in s.index if BASE <= t_ < BASE + 24]]
    return 100 * s / np.exp(np.log(b).mean())


wR, pR = reb(w_i), reb(p_i)
lg = np.log(wR) - np.log(pR)
fx = pd.DatetimeIndex(Q.qtimestamp(np.array(common)))
fig = plt.figure(figsize=(12.8, 9.6))
ax = fig.add_axes([L, 0.455, Rt - L, 0.310])
ax2 = fig.add_axes([L, 0.135, Rt - L, 0.240], sharex=ax)
for a in (ax, ax2):
    styled(a)
ax.set_yscale("log")
ax.plot(fx, wR.values, color=BLUE, lw=1.6, zorder=5)
ax.plot(fx, pR.values, color=ORANGE, lw=1.8, zorder=4)
tk = [v for v in (10, 20, 30, 50, 75, 100, 150, 250, 400, 700) if wR.min() * 0.8 <= v <= wR.max() * 1.2]
ax.yaxis.set_major_locator(FixedLocator(tk))
ax.yaxis.set_minor_locator(FixedLocator([]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_ylabel("index, 2018 = 100, log scale", fontsize=10.5, color=SEC, labelpad=10)
plt.setp(ax.get_xticklabels(), visible=False)
for ypos, val, c, n in spread_log([(wR.iloc[-1], BLUE, "Wholesale"),
                                   (pR.iloc[-1], ORANGE, "CPI")], frac=0.075):
    ax.annotate(f"{n} {val:,.0f}", xy=(fx[-1], ypos), xytext=(9, 0),
                textcoords="offset points", color=c, fontsize=11, fontweight="bold",
                va="center", annotation_clip=False)
ax2.axhline(0, color=MUT, lw=1.3, zorder=3)
ax2.plot(fx, lg.values, color=VIOLET, lw=1.7, zorder=5)
ax2.set_ylabel("ln(wholesale) − ln(CPI)", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.2f}"))
ax2.annotate(f"{lg.iloc[-1]:+.2f}", xy=(fx[-1], lg.iloc[-1]), xytext=(9, 0),
             textcoords="offset points", color=VIOLET, fontsize=11, fontweight="bold",
             va="center", annotation_clip=False)
head(fig, "Jitomate: our wholesale index against the published CPI, and their log ratio",
     f"Both indices rebased to 2018 = 100 and drawn on a log axis. The lower panel is "
     f"ln(wholesale) − ln(CPI): because these are indices and not prices, its LEVEL is "
     f"an artefact of the base year and only its MOVEMENT is informative — it says "
     f"whether wholesale is running ahead of or behind retail. {len(common)} fortnights, "
     f"{en(fx[0])} - {en(fx[-1])}. Range {lg.min():+.2f} to {lg.max():+.2f}, sd {lg.std():.2f}; "
     f"the wholesale index has risen {100*(np.exp(lg.iloc[-1])-1):+.0f}% relative to the CPI "
     f"since the 2018 base.",
     [Line2D([], [], color=BLUE, lw=2.2, label="Wholesale index (SNIIM)"),
      Line2D([], [], color=ORANGE, lw=2.2, label="CPI generic 070 Jitomate (INEGI)"),
      Line2D([], [], color=VIOLET, lw=2.2, label="Log ratio")])
foot(fig, "Source: SNIIM and INEGI. This is the index-versus-index comparison; for a "
          "margin that can be read in pesos see the Precios Promedio chart, where both "
          "series are in MXN/kg. A rising log ratio does not by itself mean margins are "
          "compressing: the two indices weight markets, varieties and outlets "
          "differently, so part of any drift is composition rather than economics.")
fig.savefig("charts/jitomate_index_ratio.png", dpi=170)
plt.close(fig)

print("charts/jitomate_varieties.png\ncharts/jitomate_margin.png\ncharts/jitomate_index_ratio.png")
print(f"varieties: last Saladette ${V['Tomate Saladette'].dropna().iloc[-1]:.2f}, "
      f"Bola ${V['Tomate Bola'].dropna().iloc[-1]:.2f}, index ${IDX.dropna().iloc[-1]:.2f}")
print(f"margin: mean {np.exp(mu):.2f}x, range {np.exp(lr.min()):.2f}-{np.exp(lr.max()):.2f}x, "
      f"AR1 {rho:.2f}")
print(f"index vs CPI log ratio: {lg.iloc[-1]:+.3f} now, sd {lg.std():.3f}, "
      f"range {lg.min():+.3f} to {lg.max():+.3f}")
