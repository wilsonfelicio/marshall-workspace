"""Jitomate wholesale, daily, and its monthly rate of change against the CPI.

  charts/jitomate_daily.png
      (a) daily national wholesale price level, MXN/kg
      (b) 30-day moving average, change over the preceding 30 days, with the published
          CPI for jitomate as dots

Panel (b) puts both series on ONE axis on purpose. To make that legitimate they have to
measure the same thing, so the CPI dot at each fortnight is the change over the two
preceding fortnights — about thirty days — not the fortnightly change. A dot is plotted
only where the fortnight itself and the two before it are all published, so no dot is a
compound of a chain break.

Default window is the last ten years; pass --from YYYY for another start, or "all".
"""
from __future__ import annotations

import pathlib
import sys
import textwrap
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

Q.DAILY = pathlib.Path("/root/jit/var_market_daily.parquet")
Q.CACHE_DIR = pathlib.Path("/root/jit/cache")

MA = 30
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, SEC, MUT, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.9, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5})
L, Rt = 0.072, 0.872

arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
Y0 = None if arg == "all" else int(arg) if arg else 2016

# ------------------------------------------------------------------ wholesale, daily
day = pd.read_parquet("data/curated/jitomate_daily_level.parquet").sort_index()
cal = pd.DataFrame(index=pd.date_range(day.index.min(), day.index.max(), freq="D"))
cal["p"] = day.p.reindex(cal.index)
# The moving average is over 30 CALENDAR days, not 30 quote days: SNIIM does not quote on
# Sundays or holidays, so a 30-observation window would be ~42 days and would drift in
# length across the sample. min_periods keeps it honest around the gaps.
cal["ma"] = cal.p.rolling(f"{MA}D", min_periods=12).mean()
cal["mm"] = 100 * (np.log(cal.ma) - np.log(cal.ma.shift(MA)))

# ------------------------------------------------------------------ CPI, quincenal
d = Q.dataset("jitomate", "070 Jitomate", windows=(5, 10)).set_index("t")
ln = np.log(d.inpc.astype(float))
# qtimestamp dates a fortnight at its FIRST day, but the print summarises prices through
# its last day, so a dot drawn at the start sits half a month before the window it
# describes. Plotted against the 30-day average that is not a cosmetic offset: at the
# start date the two series correlate 0.40, at the closing date 0.86, and the difference
# is entirely an artefact of the dating.
_st = pd.DatetimeIndex(Q.qtimestamp(d.index.values))
_close = np.where(_st.day == 1, _st + pd.Timedelta(days=14),
                  _st + pd.offsets.MonthEnd(0))
cpi = pd.DataFrame({"fecha": pd.DatetimeIndex(_close)})
cpi.index = d.index
# change over two fortnights ~ one month, and only where the chain is unbroken
step = d.index.to_series().diff()
ok2 = (step == 1) & (step.shift(1) == 1)
cpi["m"] = np.where(ok2, 100 * (ln - ln.shift(2)), np.nan)
cpi = cpi.dropna(subset=["m"])

lo = pd.Timestamp(f"{Y0}-01-01") if Y0 else cal.index.min()
C, P = cal.loc[lo:], cpi[cpi.fecha >= lo]


def styled(a):
    a.grid(axis="y", color=GRID, lw=0.9)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


fig = plt.figure(figsize=(12.8, 9.4))
ax = fig.add_axes([L, 0.500, Rt - L, 0.290])
ax2 = fig.add_axes([L, 0.152, Rt - L, 0.262])
for a in (ax, ax2):
    styled(a)

# ---- (a) level
ax.plot(C.index, C.p, color=MUT, lw=0.7, alpha=0.75, zorder=3)
ax.plot(C.index, C.ma, color=BLUE, lw=1.8, zorder=5)
ax.set_ylabel("MXN per kg, wholesale", fontsize=10.5, color=SEC, labelpad=10)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
last = C.dropna(subset=["ma"]).iloc[-1]
ax.annotate(f"{MA}-day average {last.ma:.1f}", xy=(last.name, last.ma), xytext=(9, 0),
            textcoords="offset points", color=BLUE, fontsize=10.5, fontweight="bold",
            va="center", annotation_clip=False, zorder=9,
            bbox=dict(facecolor=SURF, edgecolor="none", pad=1.4))
hi = C.p.idxmax()
ax.annotate(f"{hi:%b %Y}  {C.p.max():.0f}", xy=(hi, C.p.max()), xytext=(6, 4),
            textcoords="offset points", color=SEC, fontsize=9.5, fontweight="bold")
fig.text(L, 0.802, f"Daily quote (grey) and its {MA}-day moving average (blue)", color=SEC,
         fontsize=10, fontweight="bold")

# ---- (b) rate of change
ax2.axhline(0, color=MUT, lw=1.1, zorder=3)
ax2.plot(C.index, C.mm, color=BLUE, lw=1.5, zorder=5)
ax2.scatter(P.fecha, P.m, s=17, color=ORANGE, lw=0, zorder=6)
ax2.set_ylabel("change over 30 days", fontsize=10.5, color=SEC, labelpad=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{'+' if v > 0.001 else ''}{v:.0f}%"))
ax2.yaxis.set_major_locator(MultipleLocator(20))
fig.text(L, 0.426, "Same measure on both series: the wholesale average against the 30 "
                   "days before it, and the CPI against the fortnight two prints earlier",
         color=SEC, fontsize=10, fontweight="bold")

m = ~(C.mm.isna())
_l = C.loc[m, ["mm"]].reset_index().rename(columns={"index": "fecha"})
_l["fecha"] = _l.fecha.astype("datetime64[ns]")
_r = P[["fecha", "m"]].copy()
_r["fecha"] = _r.fecha.astype("datetime64[ns]")
al = pd.merge_asof(_r.sort_values("fecha"), _l.sort_values("fecha"),
                   on="fecha", direction="nearest", tolerance=pd.Timedelta("8D")).dropna()
rho = float(np.corrcoef(al.m, al.mm)[0, 1])
# The regression of a point-in-time CPI change on a 30-day average change is attenuated
# by the mismatch in smoothing, so the pass-through number is taken from a like-for-like
# pair: the wholesale FORTNIGHT average against the same fortnight two prints earlier,
# which is how INEGI measures the CPI.
_q = day.copy()
_q["t"] = (_q.index.year * 24 + (_q.index.month - 1) * 2 + (_q.index.day > 15).astype(int))
_wq = _q.groupby("t").p.apply(lambda z: np.log(z).mean())
_stp = _wq.index.to_series().diff()
_wm = (100 * (_wq - _wq.shift(2)))[(_stp == 1) & (_stp.shift(1) == 1)]
_lf = pd.concat([pd.Series(cpi.m.values, index=cpi.index, name="cpi"),
                 _wm.rename("ws")], axis=1).dropna()
if Y0:
    _lf = _lf[_lf.index >= Y0 * 24]
rho_lf = float(np.corrcoef(_lf.cpi, _lf.ws)[0, 1])
beta = float(np.polyfit(_lf.ws, _lf.cpi, 1)[0])

fig.text(L, 0.968, "Jitomate: the daily wholesale quote, and what it says about the CPI",
         fontsize=16, fontweight="bold", color=INK)
sub = (f"Wholesale is the geometric mean across {int(day.nm.median())} markets, Saladette "
       f"and Bola weighted 65/35, {len(C.p.dropna()):,} quote days from "
       f"{C.p.dropna().index[0]:%b %Y}. The CPI dots are INEGI's published jitomate index, "
       f"{len(P)} fortnights, each on the day its fortnight closes; the two monthly rates "
       f"correlate {rho:.2f}. Measured the way the CPI itself is built — wholesale fortnight "
       f"average against the fortnight two prints earlier — {rho_lf:.2f}, and the CPI moves "
       f"{beta:.2f}% per 1% of wholesale: over a month, pass-through is one-for-one.")
y = 0.940
for lnx in textwrap.wrap(sub, width=130):
    fig.text(L, y, lnx, fontsize=10.8, color=SEC)
    y -= 0.0208
# The legend is pinned rather than placed relative to the subtitle: when the subtitle grew
# by two lines it slid down onto the caption of the upper panel.
fig.legend(handles=[Line2D([], [], color=MUT, lw=1.4, label="Wholesale, daily"),
                    Line2D([], [], color=BLUE, lw=2.2, label=f"Wholesale, {MA}-day average"),
                    Line2D([], [], color=ORANGE, lw=0, marker="o", markersize=6.5,
                           label="CPI jitomate, published fortnightly")],
           loc="upper left", bbox_to_anchor=(L, 0.868), frameon=False, ncol=3,
           fontsize=10.5, handlelength=1.7, columnspacing=2.0, labelcolor=SEC)
assert y > 0.855, f"subtitle runs into the legend: last line at {y:.3f}"

foot = (f"Source: SNIIM (Secretaría de Economía), daily wholesale quotes, and INEGI. The "
        f"moving average runs over {MA} calendar days rather than {MA} quotes, because SNIIM "
        f"does not quote on Sundays or holidays and a fixed count of observations would be a "
        f"window of variable length. The CPI is plotted against the print two fortnights "
        f"earlier so that both series measure a change over roughly thirty days; the "
        f"fortnightly change is a different and noisier quantity. Dots are omitted where a "
        f"chain break or a missing print means the two-fortnight comparison is not defined.")
lines = textwrap.wrap(foot, width=150)
for i, lnx in enumerate(lines):
    fig.text(L, 0.028 + (len(lines) - 1 - i) * 0.0166, lnx, fontsize=9, color=MUT)

out = "charts/jitomate_daily.png"
fig.savefig(out, dpi=170)
print(out)
print(f"daily {len(C.p.dropna()):,} quote days, last {C.p.dropna().iloc[-1]:.2f} MXN/kg "
      f"on {C.p.dropna().index[-1]:%d %b %Y}; {MA}d avg {last.ma:.2f}")
print(f"corr(30d wholesale, ~monthly CPI) = {rho:.3f} at the fortnight close, n = {len(al)}")
print(f"like-for-like corr = {rho_lf:.3f}, pass-through beta = {beta:.3f}, n = {len(_lf)}")
