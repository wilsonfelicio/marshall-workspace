"""Collector for SNIIM's Pecuarios and Pesqueros modules: the protein series.

A separate module from `frutas`/`granos` because this is a different application. Frutas
and Granos live in the ASP.NET app under /nuevo/Consultas/.../Agricolas/ and answer with
one row per observation. Pecuarios and Pesqueros are the older classic-ASP app under
/SNIIM-Pecuarios-Nacionales/ and /SNIIM-PESCA/, and they differ in three ways that the
existing parser cannot absorb:

  * `destino=0` ("Todos") is ACCEPTED here, where the Agricolas endpoint rejects
    ProductoId=-1. One request therefore returns every market for a whole month, so the
    job grid is series x month rather than product x year — about 4,000 requests for the
    whole 1998-2026 history against 24,000 for produce.

  * The results are GROUPED, not flat: a `<td class=encabDES colspan=N>` row names a
    market and every `class=Datos` row beneath it belongs to that market until the next
    such header. The market is therefore carried down rather than read from a column.

  * Several series are WIDE: the products are column groups
    (Fecha | Vísceras min | Vísceras max | Piel sangre min | Piel sangre max) rather than
    a product column. Those have to be unpivoted to reach the same tidy shape as produce.

Two table shapes cover every endpoint we need:

  wide   Sub (subproductos)                         products are column groups
  long   Cor (cortes), Ent (pollo), Hue (huevo),    a product/corte column exists
         and the Pesca endpoints

Carne en canal (Can) is a third shape and the awkward one: it has NO date column. The
date lives in its own banner row, `<td class=encabFEC>Fecha:01/07/2026</td>`, and applies
to every row until the next banner — so date, like market, is carried down rather than
read. Its data cells also use `class=DatosNum` for the numeric columns, so a parser that
keys only on `class=Datos` sees a table of 845 malformed rows, which is exactly what the
first version reported.

Prices are quoted as a MIN and a MAX rather than the single `precio_geo` the produce
series carries. `precio` here is the geometric mean of the two when both are present,
which keeps the series comparable with the produce panel: the produce index is built in
logs throughout, and a geometric midpoint of a price range is the log-space centre.
Where the source gives a "precio frecuente" (huevo) that is used instead, because it is
the modal transaction rather than a range midpoint.
"""
from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd

log = logging.getLogger("sniim")

MODULO = "pecuarios"
BASE_PEC = "SNIIM-Pecuarios-Nacionales/"
BASE_PESCA = "SNIIM-PESCA/"

WS = re.compile(r"\s+")
TAG = re.compile(r"<[^>]+>")
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S | re.I)
COLSPAN = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.I)
CLS = re.compile(r"class\s*=\s*[\"']?(\w+)", re.I)
MONEY = re.compile(r"^\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$")
FECHA_BANNER = re.compile(r"Fecha\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", re.I)
FUENTE_BANNER = re.compile(r"Fuente\s*:\s*(.+)$", re.I)
DMY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
PAGES_OF = re.compile(r"P[áa]gina\s+(\d+)\s+de\s+(\d+)", re.I)


def _txt(s: str) -> str:
    return WS.sub(" ", TAG.sub(" ", s)).strip()


def _num(s: str):
    m = MONEY.match(_txt(s).replace("\xa0", " "))
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return v if v > 0 else None


def _fecha(s: str):
    m = DMY.match(_txt(s))
    if not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def n_pages(html: str) -> tuple[int, int]:
    """(current, total) from the 'Página 1 de 2' banner; (1, 1) when absent."""
    m = PAGES_OF.search(TAG.sub(" ", html))
    return (int(m.group(1)), int(m.group(2))) if m else (1, 1)


def _cells(row_html: str):
    out = []
    for attrs, body in CELL.findall(row_html):
        c = CLS.search(attrs)
        s = COLSPAN.search(attrs)
        out.append({"cls": (c.group(1) if c else "").lower(),
                    "span": int(s.group(1)) if s else 1,
                    "text": _txt(body)})
    return out


def parse(html: str) -> tuple[list[dict], dict]:
    """Parse one results page into tidy records.

    Returns (records, meta). Each record carries fecha, destino and either a product name
    (long tables) or the column-group name (wide tables), plus whatever price columns the
    table offered. Nothing is coerced into a fixed schema here — the caller decides what
    each series means, because 'Precio' means something different for pollo (a single
    quote) than for canal (a range).
    """
    rows = ROW.findall(html)
    header: list[str] = []          # flattened column names, one per physical cell
    subhead: list[str] = []
    destino = None
    fecha_banner = None            # Can: the date is a banner row, not a column
    recs: list[dict] = []
    malformed = 0
    seen_header = False

    for r in rows:
        cs = _cells(r)
        if not cs:
            continue
        classes = {c["cls"] for c in cs}

        # a market header spans the whole table and names every row beneath it
        if "encabdes" in classes:
            destino = cs[0]["text"]
            fecha_banner = None
            continue

        # `encabfec` carries two different banners depending on the module: Can uses it
        # for "Fecha:01/07/2026", the Pesca tables use it for "Fuente:La Nueva Viga, DF".
        # Treating both as a date silently dropped the market on every Pesca row.
        if "encabfec" in classes:
            txt = " ".join(c["text"] for c in cs)
            m = FECHA_BANNER.search(txt)
            if m:
                fecha_banner = _fecha(m.group(1))
            else:
                mf = FUENTE_BANNER.search(txt)
                if mf:
                    destino = mf.group(1).strip()
            continue

        if "encabtab" in classes:
            if not seen_header:
                header = []
                for c in cs:
                    header.extend([c["text"]] * c["span"])
                seen_header = True
                subhead = []
            else:
                subhead = [c["text"] for c in cs]
            continue

        if not (classes & {"datos", "datosnum"}):
            continue

        vals = [c["text"] for c in cs]
        f = _fecha(vals[0]) if vals else None
        if f is not None:
            first = 1                      # the table carries its own date column
        elif fecha_banner is not None:
            f, first = fecha_banner, 0     # Can: inherit the banner date
        else:
            malformed += 1
            continue
        rec = {"fecha": f, "destino": destino}
        # Align the sub-header (min/max) with the group header (product name). Where a
        # date column exists it has rowspan=2, so the sub-header covers columns 1..n.
        for i, v in enumerate(vals[first:], start=first):
            grp = header[i] if i < len(header) else f"col{i}"
            sub = subhead[i - first] if 0 <= (i - first) < len(subhead) else ""
            key = f"{grp}||{sub}" if sub else grp
            rec[key] = v
        recs.append(rec)

    cur, tot = n_pages(html)
    return recs, {"malformed": malformed, "page": cur, "pages": tot,
                  "header": header, "subhead": subhead,
                  "usable": bool(recs) or "encabPRE" in html or "encabTAB" in html}


def tidy(recs: list[dict], serie: str) -> pd.DataFrame:
    """Unpivot whatever `parse` produced into one row per observation.

    Columns out: fecha, serie, producto, atributo, destino, origen, precio_min,
    precio_max, precio. `atributo` keeps the extra descriptors a series carries (marca,
    presentación, peso) so nothing is silently dropped.
    """
    LONG_KEYS = {"producto", "corte", "marca", "origen", "presentación", "presentacion",
                 "tipo", "peso promedio (kg)", "peso en pie (kg)", "peso en canal (kg)",
                 "núm. canales de sacrificio", "num. canales de sacrificio"}

    # "Precio canal mínimo ($/kg)" and "Precio capote mínimo ($/kg)" are two DIFFERENT
    # price concepts on the same row, both with an empty sub-header. Grouping only on the
    # sub-header merged them and let capote overwrite canal, losing 166 of 433 pork-canal
    # observations without a warning. The concept is the column name with the price words
    # stripped out, so the two stay apart.
    STRIP = re.compile(r"precio|m[íi]nimo|m[áa]ximo|frecuente|p\s*m[íi]n|p\s*m[áa]x|"
                       r"p\s*frec|\(\$/kg\)|\$|/kg", re.I)

    def concept(grp: str) -> str:
        return WS.sub(" ", STRIP.sub(" ", grp)).strip(" ()-")

    def kind(full: str):
        """min / max / frec / plain, or None when the column is not a price."""
        f = full.lower().strip()
        # The Pesca tables abbreviate: Pmín / Pmáx / Pfrec instead of "Precio mínimo".
        if "frecuente" in f or f in ("pfrec", "p frec"):
            return "frec"
        if "mínimo" in f or "minimo" in f or f in ("pmín", "pmin"):
            return "min"
        if "máximo" in f or "maximo" in f or f in ("pmáx", "pmax"):
            return "max"
        if "precio" in f:
            return "plain"
        return None

    out = []
    for r in recs:
        base = {"fecha": r["fecha"], "serie": serie, "destino": r.get("destino")}
        desc, groups = {}, {}
        for k, v in r.items():
            if k in ("fecha", "destino"):
                continue
            grp, _, sub = k.partition("||")
            grp, sub = grp.strip(), sub.strip()
            kd = kind(f"{grp} {sub}")
            if kd is None or grp.lower() in LONG_KEYS:
                desc[grp.lower()] = v
                continue
            # A non-empty sub-header means the GROUP is the product (wide table);
            # an empty one means the product comes from a descriptor column (long table),
            # and what remains of the column name is the price concept (canal / capote).
            prod = grp if sub else None
            groups.setdefault((prod, concept(grp) if not sub else ""), {})[kd] = v
        for (prod, var), pr in groups.items():
            pmin, pmax = _num(pr.get("min", "")), _num(pr.get("max", ""))
            pfrec, plain = _num(pr.get("frec", "")), _num(pr.get("plain", ""))
            if pfrec is not None:
                p = pfrec
            elif pmin is not None and pmax is not None:
                p = (pmin * pmax) ** 0.5      # geometric centre of the quoted range
            elif plain is not None:
                p = plain
            else:
                p = pmin if pmin is not None else pmax
            if p is None:
                continue
            rec = dict(base)
            rec["producto"] = (prod or desc.get("producto") or desc.get("corte")
                               or desc.get("tipo") or serie)
            rec["variante"] = var
            rec["atributo"] = " / ".join(
                f"{k}={v}" for k, v in sorted(desc.items())
                if k not in ("producto", "corte", "tipo", "origen") and v)
            rec["origen"] = desc.get("origen")
            rec["precio_min"], rec["precio_max"], rec["precio"] = pmin, pmax, p
            out.append(rec)
    if not out:
        return pd.DataFrame(columns=["fecha", "serie", "producto", "variante", "atributo",
                                     "destino", "origen", "precio_min", "precio_max",
                                     "precio"])
    df = pd.DataFrame(out)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df
