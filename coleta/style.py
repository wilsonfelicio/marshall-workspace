"""House style for the chartbook, following the Itaú Macro Vision research notes.

Colours and geometry are taken from the notes themselves rather than approximated: the
orange is #FF6200, the series orange #FF7800, the third series navy #00009E, and the plot
panel is a #F2F2F2 fill with no gridlines at all. Type is Arial in the originals;
Liberation Sans is metrically identical and is what this container has.

Two conventions in that style do real work and are worth naming, because they are what
make the pages readable rather than merely branded:

  Direct labels, not a legend box. Each series is named inside the panel in its own
  colour. Nothing has to be matched back to a key, and — the part that matters for
  colour-blind readers — identity never rests on hue alone, which is what lets the
  palette use black against navy at all.

  The unit belongs in the panel, not on the axis. Tick labels are bare numbers and an
  italic note in the corner says what they are. It reads more quietly than a repeated
  "%" down the left edge.

The label stack is placed in whichever corner costs the least axis to clear — measured,
not guessed, and capped so that seating four lines of text can never deform the scale.
See `place_labels`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------- palette
ORANGE = "#ff6200"       # brand rules, page numbers, and the series being explained
ORANGE_L = "#ff7800"
NAVY = "#00009e"
INK = "#000000"
GRAY = "#7f7f7f"
MUT = "#9a9a9a"
PANEL = "#f2f2f2"        # plot area fill
EDGE = "#bfbfbf"
PAPER = "#ffffff"

FONT = "Liberation Sans"

# page geometry, in figure fractions, for A4 landscape
L, R = 0.062, 0.958
RULE_TOP, RULE_BOT = 0.918, 0.072
PANEL_RECT = [0.068, 0.245, 0.890, 0.470]


def use():
    plt.rcParams.update({
        "font.family": FONT, "font.sans-serif": [FONT, "DejaVu Sans"],
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.facecolor": PANEL, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.edgecolor": EDGE, "axes.linewidth": 0.8, "pdf.fonttype": 42})


def page(figsize=(11.69, 8.27)):
    return plt.figure(figsize=figsize)


def rule(fig, y, color=ORANGE, lw=1.0):
    fig.add_artist(plt.Line2D([L, R], [y, y], color=color, lw=lw,
                              transform=fig.transFigure))


def chrome(fig, kicker: str, right: str = "", page_no=None, total=None,
           foot_left: str = ""):
    """Header kicker + rules + footer, the frame every page carries."""
    fig.text(L, RULE_TOP + 0.020, kicker, fontsize=11.5, color=ORANGE)
    if right:
        fig.text(L + 0.0072 * len(kicker) + 0.012, RULE_TOP + 0.020, f"|  {right}",
                 fontsize=9.5, color=GRAY)
    rule(fig, RULE_TOP)
    rule(fig, RULE_BOT)
    if page_no is not None:
        fig.text(L, RULE_BOT - 0.035, f"{page_no}" + (f" / {total}" if total else ""),
                 fontsize=10, color=ORANGE, fontweight="bold")
    if foot_left:
        fig.text(R, RULE_BOT - 0.035, foot_left, fontsize=8, color=MUT, ha="right")


def panel(fig, rect=None):
    ax = fig.add_axes(rect or PANEL_RECT)
    ax.set_facecolor(PANEL)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(EDGE)
        sp.set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8, color=EDGE)
    return ax


def units(ax, text, xy=(0.014, 0.968)):
    """The italic note that carries the unit, so the ticks can stay bare numbers."""
    ax.text(*xy, text, transform=ax.transAxes, fontsize=8.6, style="italic",
            color=INK, va="top", ha="left", zorder=9,
            bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.82, pad=1.8))


CORNERS = {"upper right": (0.986, 0.968, "right", "top"),
           "lower left": (0.014, 0.032, "left", "bottom"),
           "lower right": (0.986, 0.032, "right", "bottom"),
           "upper left": (0.014, 0.968, "left", "top")}


def label_stack(ax, items, corner="upper right", size=8.8, step=0.045):
    """Series named inside the panel, each in its own colour — this style's legend.

    The labels sit on a panel-coloured patch, so a line passing behind one does not eat
    it. That is what lets the placement rule below stay simple."""
    fx, fy, ha, va = CORNERS[corner]
    for k, (text, color) in enumerate(items):
        y = fy - k * step if va == "top" else fy + (len(items) - 1 - k) * step
        ax.text(fx, y, text, transform=ax.transAxes, fontsize=size, color=color,
                fontweight="bold", ha=ha, va=va, zorder=9,
                bbox=dict(facecolor=PANEL, edgecolor="none", alpha=0.82, pad=1.6))


def _stretch(P, x0, x1, y0, y1, box):
    """How far the value axis must be pushed out for `box` to sit on empty panel, in data
    units. Closed form rather than a search: for a box occupying the top fraction h, the
    new top Y has to satisfy y0 + (1-h)(Y-y0) >= max(y within the box's x-span)."""
    bx0, by0, bx1, by1 = box
    fx = (P[:, 0] - x0) / (x1 - x0)
    sel = (fx >= bx0) & (fx <= bx1)
    if not sel.any():
        return 0.0, 0.0
    ys = P[sel, 1]
    if by1 > 0.9:                                   # a box at the top
        h = min(0.6, 1 - by0)
        return max(0.0, (y0 + (ys.max() - y0) / (1 - h)) - y1), 0.0
    h = min(0.6, by1)                               # a box at the bottom
    return 0.0, max(0.0, y0 - (ys.min() - h * y1) / (1 - h))


def place_labels(ax, series, labels, units_text, label_box=(0.34, 0.215),
                 units_box=(0.30, 0.095), max_cost=0.30, size=8.8):
    """Put the unit note and the series labels inside the panel, opening room for them by
    stretching the value axis — but only while that is cheap.

    Picking the emptiest corner by point count is what goes wrong. On jitomate the
    emptiest was the bottom left, and clearing it meant dropping the axis to -125 for a
    series that never goes below -65: half the panel went blank to seat four short lines
    of text. So the corner is chosen by how much axis it costs, the stretch is capped, and
    past the cap the labels simply sit on their panel-coloured patch over the data, which
    costs the reader far less than a deformed scale."""
    pts = [np.column_stack([np.asarray(x, float), np.asarray(y, float)])
           for x, y in series if len(np.asarray(x))]
    P = np.vstack(pts) if pts else np.zeros((0, 2))
    P = P[np.isfinite(P).all(axis=1)]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rng = y1 - y0
    if not len(P) or rng <= 0:
        units(ax, units_text)
        label_stack(ax, labels, "upper right", size=size)
        return "upper right"

    up0, _ = _stretch(P, x0, x1, y0, y1, (0.0, 1.0 - units_box[1], units_box[0], 1.0))
    up0 = min(up0, 0.25 * rng)

    best, cost = None, np.inf
    for name in ("upper right", "lower right", "lower left"):
        fx, fy, ha, va = CORNERS[name]
        bw, bh = label_box
        bx0 = fx - bw if ha == "right" else fx
        by0 = fy - bh if va == "top" else fy
        up, dn = _stretch(P, x0, x1, y0, y1 + up0, (bx0, by0, bx0 + bw, by0 + bh))
        if up + dn < cost:
            best, cost = (name, up, dn), up + dn
    name, up, dn = best
    if cost > max_cost * rng:                       # too expensive: overlay instead
        up = dn = 0.0
    ax.set_ylim(y0 - dn, y1 + up0 + up)
    units(ax, units_text)
    label_stack(ax, labels, name, size=size)
    return name


def bullets(fig, items, y, width=128, size=9.6, step=0.0245, gap=0.014):
    """The orange triangle bullets the source notes lead with. Drawn as a marker rather
    than a glyph: Liberation Sans has no ▶, and the fallback font's is a different size."""
    import textwrap
    for text in items:
        fig.add_artist(plt.Line2D([L + 0.006], [y + 0.006], marker=">", markersize=5,
                                  color=ORANGE, lw=0, transform=fig.transFigure))
        for ln in textwrap.wrap(text, width=width):
            fig.text(L + 0.021, y, ln, fontsize=size, color=INK)
            y -= step
        y -= gap
    return y
