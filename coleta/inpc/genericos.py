"""Download INPC index series at genérico level from INEGI.

Route: the "Índices de precios" v2 web app's own CSV export endpoint.
**No API token required**, and one request returns every requested series with
its full history. This was chosen over the INEGI API de Indicadores (which needs
a token that must be requested by email) and over Banxico's SIE API (which does
not go below aggregate subindices - verified: cuadro CP154 stops at "frutas y
verduras", no per-product series).

Two calls:
  1. ArbolAjaxInteraccion.asmx/EstructuraInicialV2 -> the structure tree,
     which maps generic NAME -> serie id
  2. Exportacion.aspx?INPtipoExporta=CSV           -> wide CSV, all series at once

Structures (national, INPC por objeto del gasto, base 2Q-jul-2018 = 100,
canasta y ponderadores 2024):

  MENSUAL     112001700030   Ene 1970 -> current   (679 periods as of Jul 2026)
  QUINCENAL   112001600030   1Q Ene 1995 -> current (758 periods)
  SUBINDICES  112001700010   the subyacente / no subyacente analytical tree
  POR CIUDAD  112001700060   genéricos x 55 ciudades

Why quincenal matters
---------------------
The monthly INPC is the EXACT simple mean of the two fortnightly indices:
    INPC_M = (Q1_M + Q2_M) / 2
This is an identity, not an approximation - verified on published bulletins:
  May 2026: (145.622 + 145.432)/2 = 145.527 = published monthly index
  Jun 2026: (145.274 + 144.988)/2 = 145.131, MoM -0.272% = published -0.27%

Consequences: modelling at fortnightly frequency roughly doubles the sample, and
once Q1 is published (around day 22-24 of the month) half of the monthly answer
is known exactly - which any honest benchmark must also exploit.

History caveats
---------------
Series start when the generic entered the canasta, not in 1970: Ene 1970 for
staples (Aguacate, Jitomate, Frijol, Papa...), Ene 1975/1980/1995 for others,
Jul 2002 for "Otras frutas", Jul 2018 for "Otras verduras y legumbres", and
**Ago 2024 for "Cilantro, epazote y perejil"** - a brand new generic with almost
no history. Missing values arrive as `N/E` (monthly) or `NA` (quincenal).

INEGI already chained across the 2024 canasta update, so the published series are
continuous and you do NOT need to splice them. But any generic that was split or
merged in 2024 (33 of 292 changed) has a DEFINITIONAL break even though the
numbers are continuous - audit before assuming a 1970-2026 like-for-like history.
"""
from __future__ import annotations

import csv
import html
import io
import json
import logging
import re
import urllib.parse
import urllib.request

log = logging.getLogger("sniim")

APP = "https://www.inegi.org.mx/app/indicesdepreciosv2/"
ARBOL = APP + "servicios/ArbolAjaxInteraccion.asmx/"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

MENSUAL = "112001700030"
QUINCENAL = "112001600030"
SUBINDICES_EST = "112001700010"
POR_CIUDAD = "112001700060"

MONTHS = {m: i + 1 for i, m in enumerate(
    "Ene Feb Mar Abr May Jun Jul Ago Sep Oct Nov Dic".split())}

MISSING = {"", "N/E", "NA", "N/D", "-"}


def _post_json(url: str, payload: dict, timeout: int = 300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={**UA, "Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["d"]


def get_tree(id_estructura: str) -> list[tuple[int, str | None, str | None, str]]:
    """Return [(level, node_id, serie_id, name), ...] for a structure."""
    doc = _post_json(ARBOL + "EstructuraInicialV2", {
        "idEstructura": id_estructura, "esquemaBD": "",
        "paramFuente": "", "notas": [], "open": True})
    out = []
    for tr in re.findall(r"<tr .*?</tr>", doc, re.S):
        nid = re.search(r"data-id=\"?(\d+)", tr)
        lvl = re.search(r"data-arbolnivel='?(\d+)", tr)
        serie = re.search(r"id='CBox_Serie_(\d+)'", tr)
        name = (re.search(r'data-title="(.*?)"', tr)
                or re.search(r"</span>([^<]+)</div>", tr))
        out.append((
            int(lvl.group(1)) if lvl else 0,
            nid.group(1) if nid else None,
            serie.group(1) if serie else None,
            html.unescape(name.group(1)).strip() if name else "",
        ))
    return out


def resolve_series(id_estructura: str) -> dict[str, str]:
    """name -> serie id, read live from the tree.

    Always resolve by name. The numeric serie ids are vintage-specific and WILL
    change at the next canasta update; hardcoding them is how this breaks
    silently two years from now.
    """
    return {n: s for _l, _i, s, n in get_tree(id_estructura) if s}


def export_csv(id_estructura: str, serie_ids: list[str],
               anio_i: int = 1969, anio_f: int = 2100,
               tipo: str = "BIE", timeout: int = 900) -> str:
    """POST the export form. Returns decoded CSV text (source is Windows-1252).

    `tipo`: BIE = índices. The app also offers monthly / accumulated / annual
    inflation, but take the index and difference it yourself so you control the
    rounding.
    """
    form = {
        "idEstructura": id_estructura, "_formato": "CSV",
        "_anioI": str(anio_i), "_anioF": str(anio_f),
        "_meta": "0", "_tipo": tipo, "_info": "Índices",
        "_orient": "vertical", "esquema": "", "st": "", "pf": "inp",
        "cuadro": id_estructura,
        # `_series=e|<node>` (whole structure) returns HTTP 500 - the explicit
        # `c|` list is the only form that works.
        "_series": "c|" + ",".join(serie_ids) + ",",
        "cvEstructura": id_estructura,
    }
    req = urllib.request.Request(
        APP + "Exportacion.aspx?INPtipoExporta=CSV",
        data=urllib.parse.urlencode(form).encode(),
        headers={**UA, "Referer": APP + "Estructura.aspx?idEstructura=" + id_estructura})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("cp1252", "replace")


def parse_csv(text: str):
    """-> (periods, {serie_id: [values]}).

    periods is [(label, year, month, quincena_or_None), ...].
    """
    rows = list(csv.reader(io.StringIO(text)))
    try:
        hi = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Fecha")
    except StopIteration:
        raise RuntimeError(
            "no 'Fecha' header row in the INEGI export - the export form or the "
            "app changed. First 3 rows: %r" % rows[:3])
    ids = [c.strip() for c in rows[hi][1:]]
    periods: list[tuple[str, int, int, int | None]] = []
    cols: dict[str, list[float | None]] = {i: [] for i in ids}
    for r in rows[hi + 1:]:
        if not r or not r[0].strip():
            continue
        m = re.match(r"^(?:(\d)Q )?([A-Z][a-z][a-z])\s+(\d{4})$", r[0].strip())
        if not m:
            continue
        q, mon, yr = m.groups()
        if mon not in MONTHS:
            continue
        periods.append((r[0].strip(), int(yr), MONTHS[mon], int(q) if q else None))
        for j, sid in enumerate(ids):
            v = r[j + 1].strip() if j + 1 < len(r) else ""
            cols[sid].append(None if v in MISSING else float(v.replace(",", "")))
    if not periods:
        raise RuntimeError("INEGI export parsed to zero periods")
    return periods, cols


def fetch_genericos(frequency: str = "mensual", names: list[str] | None = None):
    """Fetch the 32 target genéricos. Returns (periods, {name: values}, {name: serie_id})."""
    from . import catalogo

    names = names or catalogo.tree_names()
    est = MENSUAL if frequency == "mensual" else QUINCENAL
    mapping = resolve_series(est)
    missing = [n for n in names if n not in mapping]
    if missing:
        raise KeyError(
            "these genéricos are not in INEGI's %s tree any more: %r. "
            "The canasta was probably updated - re-derive config/inpc_genericos.csv."
            % (frequency, missing))
    ids = [mapping[n] for n in names]
    periods, cols = parse_csv(export_csv(est, ids))
    log.info("INPC %s: %d periods %s..%s, %d genéricos",
             frequency, len(periods), periods[0][0], periods[-1][0], len(names))
    return periods, {n: cols[mapping[n]] for n in names}, {n: mapping[n] for n in names}


def fetch_subindices(serie_ids: list[str] | None = None):
    """Fetch the analytical subindices, including 865557 'Frutas y verduras'."""
    from . import catalogo

    serie_ids = serie_ids or list(catalogo.SUBINDICES)
    periods, cols = parse_csv(export_csv(SUBINDICES_EST, serie_ids))
    return periods, {catalogo.SUBINDICES.get(s, s): v for s, v in cols.items()}


def monthly_from_quincenal(q1: float | None, q2: float | None) -> float | None:
    """INPC_M = (Q1 + Q2) / 2. An exact identity, verified against bulletins."""
    if q1 is None or q2 is None:
        return None
    return (q1 + q2) / 2.0
