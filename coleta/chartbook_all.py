"""One chartbook covering produce and proteins: 40 pages, continuous numbering.

  chartbook_completo.pdf

Builds the two section books with `--no-cover` and a page offset, then joins them behind
a single cover. The sections stay separate scripts because their pipelines genuinely
differ — produce is city-weighted from the Agrícolas module, proteins are equal-weighted
from Pecuarios and Pesqueros — and folding them into one loop would mean one code path
pretending two different sources are the same thing.

  python3 chartbook_all.py [--from 2024]
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader, PdfWriter

import style as S

S.use()

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="y0", type=int, default=2024)
ap.add_argument("--out", default="chartbook_completo.pdf")
A = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parent
N_PRODUCE, N_PROTEIN = 32, 8
TOTAL = N_PRODUCE + N_PROTEIN

with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    a, b = tmp / "produce.pdf", tmp / "protein.pdf"
    for script, out, offset in (("chartbook.py", a, 0),
                                ("chartbook_proteinas.py", b, N_PRODUCE)):
        cmd = [sys.executable, str(ROOT / script), "--from", str(A.y0),
               "--out", str(out), "--no-cover",
               "--page-offset", str(offset), "--page-total", str(TOTAL)]
        print("  " + " ".join(cmd[1:]))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"{script} failed")
        print("   " + r.stdout.strip().splitlines()[-1])

    # ---- cover, in the same house style as the two sections
    cov = tmp / "cover.pdf"
    fig = S.page()
    S.chrome(fig, "Precios de mayoreo", pd.Timestamp.today().strftime("%d %B %Y"),
             foot_left="Source: SNIIM (Secretaría de Economía) and INEGI")
    fig.text(S.L, 0.790, "Mexican food prices: wholesale", fontsize=27, color=S.ORANGE)
    fig.text(S.L, 0.726, "against the published CPI", fontsize=27, color=S.ORANGE)
    fig.text(S.L, 0.668, f"{TOTAL} pages, one per INPC generic, {A.y0} to date.",
             fontsize=11.5, color=S.INK)
    S.bullets(fig, [
        f"Two sections. Pages 1-{N_PRODUCE} are Frutas y verduras, from SNIIM's Agrícolas "
        f"module. Pages {N_PRODUCE + 1}-{TOTAL} are the eight protein generics, from the "
        f"Pecuarios and Pesqueros modules — a different application, with rastros, packers "
        f"and distribution centres instead of central de abasto wholesalers.",

        "Four series on every page, all measuring a change over roughly thirty days so "
        "that they share one axis: the wholesale 30-day average against the 30 days before "
        "it, the same thing on a 7-day average (which leads and overshoots), INEGI's "
        "published index for that generic, each fortnight against the fortnight two prints "
        "earlier, and a model fit for that published change.",

        "CPI dots sit on the day each fortnight CLOSES. A fortnight is labelled by its "
        "first day but summarises prices through its last, so plotting a dot at its label "
        "puts it half a month before the prices it describes — on jitomate that alone "
        "moves the measured correlation from 0.86 to 0.40.",

        "The two sections are NOT built the same way, and the difference is stated on each "
        "page. Produce is a geometric mean across markets weighted by INPC city weight; "
        "proteins are equal-weighted, because those markets are not in INEGI's city "
        "crosswalk. Most protein series also publish a price RANGE rather than a single "
        "quote, so their level is the geometric centre of min and max.",

        "Three of the four series are raw. The fourth is a ridge regression, re-estimated "
        "at every fortnight on data available at the time, so every point on it is out of "
        "sample. Each page reports its error against the 30-day line alone and against the "
        "same model stripped of the CPI lag; where those last two are close, the fit is "
        "mostly the CPI repeating itself rather than wholesale information — true on "
        "several protein pages, almost no produce ones.",
    ], 0.596, size=9.2, step=0.0228, gap=0.0115)
    fig.savefig(cov, format="pdf")
    plt.close(fig)

    w = PdfWriter()
    for f in (cov, a, b):
        for pg in PdfReader(str(f)).pages:
            w.add_page(pg)
    with open(A.out, "wb") as fh:
        w.write(fh)

n = len(PdfReader(A.out).pages)
print(f"\n{A.out}  {n} pages (cover + {N_PRODUCE} produce + {N_PROTEIN} protein)")
assert n == TOTAL + 1, f"expected {TOTAL + 1} pages, got {n}"

try:
    p = pd.read_csv("data/curated/chartbook_corr.csv").dropna(subset=["corr_30d_cpi"])
    q = pd.read_csv("data/curated/proteinas_corr.csv").dropna(subset=["corr"])
    print(f"corr(30d, CPI): produce median {p.corr_30d_cpi.median():.2f} "
          f"(n={len(p)}), protein median {q['corr'].median():.2f} (n={len(q)})")
except Exception:
    pass
