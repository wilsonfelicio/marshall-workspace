"""Refresh the product / origin / destination catalogs from the query forms.

SNIIM adds products and markets over time, so the catalogs are scraped rather
than hard-coded. Run `python run.py catalog` after any long gap; new products
appear as unmapped rows in the mapping report.
"""
from __future__ import annotations

import logging

from . import store
from .parse import parse_dropdown

log = logging.getLogger("sniim")

FRUTAS_FORM = "ConsultaFrutasYHortalizas.aspx"
GRANOS_FORM = "ConsultaGranos.aspx"

# Dropdown name -> catalog file name, per module.
FRUTAS_DROPDOWNS = {
    "ddlProducto": "frutas_productos",
    "ddlOrigen": "frutas_origenes",
    "ddlDestino": "frutas_destinos",
}
GRANOS_DROPDOWNS = {
    "ddlProducto": "granos_productos",
    "ddlOrigen": "granos_origenes",
    "ddlDestino": "granos_destinos",
}


def refresh(cfg, session) -> dict[str, int]:
    out: dict[str, int] = {}

    for page, dropdowns, subopcion in (
        (FRUTAS_FORM, FRUTAS_DROPDOWNS, "4|0"),
        (GRANOS_FORM, GRANOS_DROPDOWNS, "6|0"),
    ):
        html_doc = session.get(page, {"SubOpcion": subopcion})
        for ddl, name in dropdowns.items():
            pairs = parse_dropdown(html_doc, ddl)
            # Drop the "Todos" sentinel; it is not a real entity and the
            # results endpoint rejects ProductoId=-1 anyway.
            pairs = [(v, lbl) for v, lbl in pairs if v not in ("-1", "")]
            if not pairs:
                log.warning("dropdown %s on %s came back empty", ddl, page)
                continue
            path = store.save_catalog(cfg, name, pairs)
            out[name] = len(pairs)
            log.info("catalog %-20s %4d entries -> %s", name, len(pairs), path)

    return out


def product_ids(cfg, modulo: str) -> list[int]:
    name = "frutas_productos" if modulo == "frutas" else "granos_productos"
    df = store.load_catalog(cfg, name)
    return [int(x) for x in df["id"].tolist()]


def product_labels(cfg, modulo: str) -> dict[int, str]:
    name = "frutas_productos" if modulo == "frutas" else "granos_productos"
    df = store.load_catalog(cfg, name)
    return {int(r.id): str(r.label) for r in df.itertuples()}


def frijol_ids(cfg) -> list[int]:
    df = store.load_catalog(cfg, "granos_productos")
    mask = df["label"].str.strip().str.lower().str.startswith("frijol")
    return [int(x) for x in df.loc[mask, "id"].tolist()]
