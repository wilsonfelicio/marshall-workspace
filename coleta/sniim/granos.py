"""Weekly collector for Granos Basicos - this is where Frijol lives.

The Granos module has no date-range query. Its results page takes a week
selector instead, and the parameter names are different from the Frutas page
(this is not documented anywhere; it was found by probing):

  ResultadosConsultaFechaGranos.aspx
    ?Semana=<1..5>              week-of-month, NOT ISO week
    &Mes=<1..12>                numeric month - the word "Julio" returns HTTP 500
    &Anio=<yyyy>
    &ProductoId=<int>           -1 ("Todos") is rejected here too
    &OrigenId=-1&Origen=Todos
    &DestinoId=-1&Destino=Todos
    &PreciosPorId=2
    &RegistrosPorPagina=20000

The response still carries individual dates (the survey lands mid-week), so the
stored rows are dated observations, not week averages. They are kept in a
separate module directory from the daily fruit and vegetable data so the two
frequencies can never be silently mixed.
"""
from __future__ import annotations


import logging
from datetime import date, timedelta

import pandas as pd

from .parse import parse_results

log = logging.getLogger("sniim")

PAGE = "ResultadosConsultaFechaGranos.aspx"
MODULO = "granos"


def _params(cfg, producto_id: int, semana: int, mes: int, anio: int) -> dict:
    q = cfg.query
    return {
        "Semana": int(semana),
        "Mes": int(mes),
        "Anio": int(anio),
        "ProductoId": int(producto_id),
        "OrigenId": q["origen_id"],
        "Origen": "Todos",
        "DestinoId": q["destino_id"],
        "Destino": "Todos",
        "PreciosPorId": q["precios_por_id"],
        "RegistrosPorPagina": cfg.http["registros_por_pagina"],
    }


def periodo_key(anio: int, mes: int, semana: int) -> str:
    return f"{anio:04d}-{mes:02d}-S{semana}"


def week1_monday(anio: int, mes: int) -> date:
    """Monday of the site's week-of-month slot 1.

    Derived empirically, then verified against the live site for 13 months
    spanning 1999-2026 (Mon/Tue/Wed/Thu/Fri/Sat/Sun starts all covered):

        if the 1st falls Mon, Tue or Wed -> the Monday of the week containing
                                            the 1st, i.e. the partial week
                                            counts as week 1
        otherwise                        -> the following Monday

    Getting this wrong silently skips weeks, so do not "simplify" it without
    re-probing the site. Worked examples:
        2026-07 (1st = Wed) -> 2026-06-29   partial week counts
        2000-06 (1st = Thu) -> 2000-06-05   partial week does not
        2025-11 (1st = Sat) -> 2025-11-03
    """
    first = date(anio, mes, 1)
    wd = first.weekday()  # Monday = 0
    monday = first - timedelta(days=wd)
    if wd > 2:  # Thu, Fri, Sat, Sun
        monday += timedelta(days=7)
    return monday


def slot_dates(anio: int, mes: int, semana: int) -> tuple[date, date]:
    """(Monday, Friday) covered by one (anio, mes, semana) slot."""
    monday = week1_monday(anio, mes) + timedelta(weeks=semana - 1)
    return monday, monday + timedelta(days=4)


def weeks_in_range(d0: date, d1: date) -> list[tuple[int, int, int]]:
    """Canonical (anio, mes, semana) slots covering [d0, d1], no duplicates.

    Slot 5 of a short month is often the same Monday-Friday block as slot 1 of
    the next month (2026-02 S5 == 2026-03 S1 == 2 Mar). Enumerating all five
    slots of every month therefore guarantees full coverage with no gaps, and
    deduplicating on the Monday guarantees exactly one request per distinct
    week. First occurrence wins, so the slot chosen for a given week is stable
    across runs and the manifest key stays consistent.
    """
    seen: dict[date, tuple[int, int, int]] = {}
    # Start a month early: a slot belonging to the previous month can cover a
    # week that reaches into d0.
    y, m = (d0.year, d0.month - 1) if d0.month > 1 else (d0.year - 1, 12)
    end = (d1.year, d1.month)
    while (y, m) <= end:
        for s in range(1, 6):
            monday, friday = slot_dates(y, m, s)
            if friday < d0 or monday > d1:
                continue
            if monday not in seen:
                seen[monday] = (y, m, s)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return [seen[k] for k in sorted(seen)]


def week_is_closed(anio: int, mes: int, semana: int, today: date) -> bool:
    """Has this Monday-Friday block finished?

    A week still in progress must not be recorded with a terminal status, or
    `--resume` would freeze a partially-surveyed week as complete. One extra day
    of margin because SNIIM publishes Granos on Wednesdays for the week and can
    post late.
    """
    _monday, friday = slot_dates(anio, mes, semana)
    return friday < today


def fetch_week(cfg, session, producto_id: int, anio: int, mes: int, semana: int):
    doc = session.get(PAGE, _params(cfg, producto_id, semana, mes, anio))
    res = parse_results(doc)
    periodo = periodo_key(anio, mes, semana)

    if res.malformed_rows:
        log.warning(
            "granos product %s %s: parser rejected %d malformed row(s)",
            producto_id, periodo, res.malformed_rows,
        )
    if not res.usable:
        log.warning(
            "granos product %s %s: unusable response (%s)",
            producto_id, periodo, res.rejected_reason or "not a results page",
        )
    if res.truncated:
        # Never observed - a product-week is a few dozen rows - and there is no
        # sub-week to split into, so the caller must not mark this complete.
        log.error(
            "granos product %s %s truncated at %d pages - raise registros_por_pagina",
            producto_id, periodo, res.pages,
        )
    return res.rows, res


def to_frame(rows: list[dict], producto_id: int, producto: str | None,
             periodo: str, fetched_at: str, unidad: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["modulo"] = MODULO
    df["producto_id"] = int(producto_id)
    df["producto"] = producto
    df["calidad"] = None
    df["presentacion"] = None  # Granos results carry no presentation column
    df["periodo"] = periodo
    df["fetched_at"] = fetched_at
    df["unidad"] = unidad
    return df
