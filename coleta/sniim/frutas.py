"""Daily collector for Frutas y Hortalizas.

Endpoint (plain GET, no session state needed):

  ResultadosConsultaFechaFrutasYHortalizas.aspx
    ?fechaInicio=DD/MM/YYYY
    &fechaFinal=DD/MM/YYYY
    &ProductoId=<int>           required - the server rejects -1 ("Todos")
    &OrigenId=-1&Origen=Todos
    &DestinoId=-1&Destino=Todos
    &PreciosPorId=2             2 = pesos per kilogram (calculated)
    &RegistrosPorPagina=20000   above 1000 the server stops paginating

Returned rows are one per (date, presentation, origin state, destination market).

Truncation handling
-------------------
If a response is still paginated, the range is split in half and each half is
fetched recursively. Every problem found anywhere in that recursion - a page
that could not be split further, a malformed row, an unusable response, a date
range the server did not honour - is AGGREGATED into the returned meta rather
than being reported only for the outermost call. The caller uses that to decide
whether the job may be marked complete. Losing a nested truncation flag would
mean recording a product-year as finished while silently missing rows, and
`--resume` would then never revisit it.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .parse import parse_results

log = logging.getLogger("sniim")

PAGE = "ResultadosConsultaFechaFrutasYHortalizas.aspx"
MODULO = "frutas"

MAX_SPLIT_DEPTH = 12  # 1 year halved 12 times is well under a day


def _dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _params(cfg, producto_id: int, d0: date, d1: date) -> dict:
    q = cfg.query
    return {
        "fechaInicio": _dmy(d0),
        "fechaFinal": _dmy(d1),
        "ProductoId": int(producto_id),
        "OrigenId": q["origen_id"],
        "Origen": "Todos",
        "DestinoId": q["destino_id"],
        "Destino": "Todos",
        "PreciosPorId": q["precios_por_id"],
        "RegistrosPorPagina": cfg.http["registros_por_pagina"],
    }


def _split(d0: date, d1: date) -> list[tuple[date, date]]:
    """Split a date range roughly in half, for when a response is truncated."""
    span = (d1 - d0).days
    if span <= 0:
        return []
    mid = d0 + timedelta(days=span // 2)
    return [(d0, mid), (mid + timedelta(days=1), d1)]


def _blank_meta() -> dict:
    return {
        "requests": 0,
        "truncated": False,      # something was cut off and could not be split
        "split": False,          # we had to split at least once
        "unusable": False,       # at least one response was not a results page
        "range_mismatch": False, # server did not honour the dates we asked for
        "malformed": 0,          # data rows rejected by the parser
        "pages_max": 1,
        "producto": None,
        "calidad": None,
        "reason": "",
    }


def _merge_meta(parent: dict, child: dict) -> None:
    parent["requests"] += child["requests"]
    parent["truncated"] = parent["truncated"] or child["truncated"]
    parent["split"] = parent["split"] or child["split"]
    parent["unusable"] = parent["unusable"] or child["unusable"]
    parent["range_mismatch"] = parent["range_mismatch"] or child["range_mismatch"]
    parent["malformed"] += child["malformed"]
    parent["pages_max"] = max(parent["pages_max"], child["pages_max"])
    parent["producto"] = parent["producto"] or child["producto"]
    parent["calidad"] = parent["calidad"] or child["calidad"]
    if child["reason"] and not parent["reason"]:
        parent["reason"] = child["reason"]


def fetch_range(cfg, session, producto_id: int, d0: date, d1: date, depth: int = 0):
    """Fetch one product over [d0, d1], recursively splitting on truncation.

    Returns (rows, meta). See _blank_meta for the meta contract.
    """
    doc = session.get(PAGE, _params(cfg, producto_id, d0, d1))
    res = parse_results(doc)

    meta = _blank_meta()
    meta["requests"] = 1
    meta["malformed"] = res.malformed_rows
    meta["pages_max"] = res.pages
    meta["producto"] = res.producto
    meta["calidad"] = res.calidad

    if res.malformed_rows:
        log.warning(
            "product %s %s..%s: parser rejected %d malformed row(s) - "
            "the results table layout may have changed",
            producto_id, d0, d1, res.malformed_rows,
        )

    # Did we actually get a results page? An error or maintenance page arrives
    # as HTTP 200 and would otherwise be indistinguishable from an empty period.
    if not res.usable:
        meta["unusable"] = True
        meta["reason"] = res.rejected_reason or "response was not a usable results page"
        log.warning(
            "product %s %s..%s: unusable response (%s)",
            producto_id, d0, d1, meta["reason"],
        )
        return res.rows, meta

    # Did the server honour our date range, or fall back to a default window?
    # A silent substitution would store a week of data under a whole-year key.
    if res.rows:
        if res.rango_inicio and res.rango_inicio != d0:
            meta["range_mismatch"] = True
        if res.rango_fin and res.rango_fin != d1:
            meta["range_mismatch"] = True
        if meta["range_mismatch"]:
            meta["reason"] = (
                f"asked {d0}..{d1}, server returned {res.rango_inicio}..{res.rango_fin}"
            )
            log.warning("product %s: %s", producto_id, meta["reason"])

    if not res.truncated:
        return res.rows, meta

    parts = _split(d0, d1) if depth < MAX_SPLIT_DEPTH else []
    if not parts:
        meta["truncated"] = True
        meta["reason"] = meta["reason"] or (
            f"truncated at {res.pages} pages and cannot be split further "
            f"({d0}..{d1}); raise http.registros_por_pagina"
        )
        log.error("product %s %s..%s: %s", producto_id, d0, d1, meta["reason"])
        return res.rows, meta

    log.info(
        "product %s %s..%s truncated (%d pages) - splitting",
        producto_id, d0, d1, res.pages,
    )
    rows: list[dict] = []
    combined = _blank_meta()
    combined["requests"] = 1
    combined["split"] = True
    combined["malformed"] = res.malformed_rows
    combined["pages_max"] = res.pages
    combined["producto"] = res.producto
    combined["calidad"] = res.calidad

    for a, b in parts:
        sub_rows, sub_meta = fetch_range(cfg, session, producto_id, a, b, depth + 1)
        rows.extend(sub_rows)
        _merge_meta(combined, sub_meta)

    return rows, combined


def to_frame(rows: list[dict], producto_id: int, producto: str | None,
             calidad: str | None, periodo: str, fetched_at: str, unidad: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["modulo"] = MODULO
    df["producto_id"] = int(producto_id)
    df["producto"] = producto
    df["calidad"] = calidad
    df["periodo"] = periodo
    df["fetched_at"] = fetched_at
    df["unidad"] = unidad
    return df


def year_bounds(anio: int, today: date) -> tuple[date, date]:
    """[start, end] to request for a calendar year, clamped to today."""
    d0 = date(anio, 1, 1)
    d1 = date(anio, 12, 31)
    if d1 > today:
        d1 = today
    return d0, d1


def year_is_closed(anio: int, today: date) -> bool:
    """Has this calendar year finished?

    Critical for the manifest. A backfill run in August cannot mark 2026 as
    complete under the key "2026" - September to December do not exist yet, and
    `--resume` would then skip the year forever. The caller records an open year
    with a non-terminal status so a later run refetches it.
    """
    return date(anio, 12, 31) < today
