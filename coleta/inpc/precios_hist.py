"""Retail MXN/kg history for one genérico, across the three Precios Promedio vintages.

Why this module exists: `precios.VINTAGES` codes are NOT stable. Empirically, for
Calabacita the clave is 065 (bs=""), 061 (bs="18") and 060 (bs="18a") - and in the
2011-2018 vintage the code 060 that identifies Calabacita today is *Cebolla*, while
in 2018-2024 it is *Uva*. Splicing on the code silently concatenates three
different foods. So we resolve the code by NAME inside each vintage, by probing a
single month with a wide code range and reading back the `Genérico` label.

Output is cached to data/inpc/precios_kg/<slug>.parquet so the probe + fetch runs
once per genérico.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

from . import precios

log = logging.getLogger("sniim")

CACHE = Path(__file__).resolve().parent.parent / "data" / "inpc" / "precios_kg"
PROBE_CODES = [f"{i:03d}" for i in range(1, 130)]


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def resolve_code(nombre: str, bs: str, periodo: int) -> str | None:
    """Clave genérico whose published label matches `nombre` inside vintage `bs`."""
    hdr, rows = precios.fetch(PROBE_CODES, periodo, periodo, cities=["01"], bs=bs)
    ig, ik = hdr.index("Genérico"), hdr.index("Clave genérico")
    want = _fold(nombre)
    hits = {r[ik] for r in rows if _fold(r[ig].strip()) == want}
    if len(hits) == 1:
        return hits.pop()
    if not hits:
        log.warning("Precios Promedio bs=%r: no genérico named %r", bs, nombre)
        return None
    raise RuntimeError(f"bs={bs!r}: {nombre!r} maps to several claves {hits}")


def national_kg(nombre: str, slug: str, city_weights: dict[str, float],
                desde: int = 201101, hasta: int = 209912, refresh: bool = False):
    """Weighted-geometric national retail price in MXN/kg, monthly. Cached."""
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{slug}.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)

    frames, notes = [], []
    for bs, lo, hi in precios.VINTAGES:
        pi, pf = max(lo, desde), min(hi, hasta)
        if pi > pf:
            continue
        code = resolve_code(nombre, bs, pi)
        if code is None:
            continue
        hdr, rows = precios.fetch([code], pi, pf, bs=bs)
        kg, diag = precios.to_kg_frame(hdr, rows)
        otros = set(kg["generico"].unique()) - {nombre}
        if otros:  # the code drifted mid-vintage
            raise RuntimeError(f"bs={bs!r} clave {code} also returned {otros}")
        notes.append({"bs": bs, "clave": code, "pi": pi, "pf": pf, **{
            k: diag[k] for k in ("quotes_total", "quotes_kg", "pct_non_kg")}})
        frames.append(kg)
    if not frames:
        raise RuntimeError(f"no retail quotes for {nombre!r}")

    allq = pd.concat(frames, ignore_index=True)
    nat = (precios.national_mean(allq, city_weights)
           .sort_values("periodo").reset_index(drop=True))
    nat["mes"] = pd.PeriodIndex(nat["periodo"], freq="M").to_timestamp()
    nat.to_parquet(out, index=False)
    pd.DataFrame(notes).to_csv(CACHE / f"{slug}_vintages.csv", index=False)
    log.info("retail %s: %d months %s..%s, claves %s", nombre, len(nat),
             nat["periodo"].iloc[0], nat["periodo"].iloc[-1],
             [n["clave"] for n in notes])
    return nat
