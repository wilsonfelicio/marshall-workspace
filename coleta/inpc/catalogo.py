"""The 32 INPC genéricos, their serie ids, weights and chaining factors.

Sources
-------
weights + factores de encadenamiento
    DOF, "ACTUALIZACIÓN de la canasta, ponderadores y encadenamiento de las series
    para la continuidad del INPC" (2024 update, ENIGH 2022, applies from August 2024),
    machine-readable at https://sidof.segob.gob.mx/notas/docFuente/5737063
    (dof.gob.mx itself blocks automated fetching; the sidof mirror does not.)

city weights
    INEGI, "INPC 2024. Documento metodológico", Anexo F "Ponderadores por área
    geográfica", p. 106. 55 cities, sums to exactly 100.

IMPORTANT: serie ids are vintage-specific and will change at the next canasta
update. `genericos.resolve_series()` looks them up by NAME from the live tree on
every run; the ids in the CSV are a fallback and a record of what was seen.
"""
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERICOS_CSV = PROJECT_ROOT / "config" / "inpc_genericos.csv"
CIUDADES_CSV = PROJECT_ROOT / "config" / "inpc_ciudades.csv"

# Analytical subindices worth carrying alongside the genéricos, for validation
# and context. These live in idEstructura 112001700010.
SUBINDICES = {
    "865546": "INPC general",
    "865547": "Subyacente",
    "865555": "No subyacente",
    "865556": "Agropecuarios",
    "865557": "Frutas y verduras",   # <- exactly our 32 genéricos
    "865558": "Pecuarios",
}

VALIDATION_TARGET = "865557"

# Precios Promedio genéricos where a mean of the KG quotes is NOT meaningful,
# because the quotes are overwhelmingly per-piece or per-bunch. Measured over
# 39,147 real quotes (32 genéricos x 55 cities x May-Jul 2026):
#   087 Cilantro/epazote/perejil  94.7% non-KG (MANOJO), only 8/55 cities in KG
#   071 Lechuga y col             79.7% non-KG (PZA),   only 35/55 cities in KG
# Use their INDEX only, or model the per-piece price with a free scale factor.
PRECIOS_UNIT_UNRELIABLE = {"087", "071"}


def genericos() -> list[dict]:
    with open(GENERICOS_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["ponderacion_2024"] = float(r["ponderacion_2024"])
        r["factor_encadenamiento"] = float(r["factor_encadenamiento"])
    return rows


def tree_names() -> list[str]:
    """Names exactly as they appear in INEGI's tree, e.g. '045 Aguacate'."""
    return [f"{r['clave_generico']} {r['generico']}" for r in genericos()]


def weights() -> dict[str, float]:
    return {f"{r['clave_generico']} {r['generico']}": r["ponderacion_2024"]
            for r in genericos()}


def chain_factors() -> dict[str, float]:
    """Factores de encadenamiento. See `theta` for why you need them."""
    return {f"{r['clave_generico']} {r['generico']}": r["factor_encadenamiento"]
            for r in genericos()}


def theta() -> dict[str, float]:
    """theta_i = 1 / factor_encadenamiento_i.

    After the August 2024 chain link the published index series "pierden su
    propiedad de aditividad" (methodological document p. 69): you cannot simply
    take a weighted mean of published genérico indices and recover the published
    aggregate. Multiplying each index by theta restores additivity.

    This is not a rounding detail. Measured against the published serie 865557
    over Aug-2024 to Jul-2026:
        with theta     RMSE 0.029 pp of month-on-month change
        without theta  RMSE 1.028 pp        <- 35x worse, and visibly wrong
    """
    return {k: 1.0 / v for k, v in chain_factors().items()}


def ciudades() -> list[dict]:
    with open(CIUDADES_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["ponderador_2024"] = float(r["ponderador_2024"])
    return rows


def city_weights() -> dict[str, float]:
    """clave -> weight. WHOLE-BASKET weights, not per-genérico.

    INEGI does not publish the city x genérico expenditure matrix, so using these
    for a fruit price assumes each city's share of avocado spend equals its share
    of total spend. That is wrong wherever regional diets differ. The accurate
    alternative is to use the published genérico x city indices
    (idEstructura 112001700060) and back out implicit weights per genérico.
    """
    return {r["clave_ciudad_preciospromedio"]: r["ponderador_2024"] for r in ciudades()}


def aggregate_weighted(index_by_generic: dict[str, list[float | None]],
                       n_periods: int) -> list[float | None]:
    """Weighted aggregate of genérico indices, with the theta additivity fix.

    Reproduces INEGI's published "Frutas y verduras" subindex. Use this to check
    that item-level model forecasts add up to something sensible.
    """
    w = weights()
    th = theta()
    out: list[float | None] = []
    for t in range(n_periods):
        num = den = 0.0
        for name, series in index_by_generic.items():
            v = series[t]
            if v is None:
                continue
            num += w[name] * v * th[name]
            den += w[name]
        out.append(num / den if den > 0 else None)
    return out
