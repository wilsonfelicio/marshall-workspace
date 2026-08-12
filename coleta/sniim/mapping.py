"""Mapping from SNIIM variety-level products to the 32 requested INPC generics.

Why this file exists
--------------------
SNIIM publishes prices at variety level ("Aguacate Hass - Calidad extra",
"Limon c/semilla # 4", "Melon Cantaloupe # 18"), while the INPC generics you
want to track are broader ("Aguacate", "Limon", "Melon"). The relationship is
many-to-one and requires judgement, so it lives in one auditable place instead
of being buried in the collector.

The generated CSV at config/mapping_inpc.csv is the source of truth used by the
aggregation step. Edit it freely - `run.py mapping` will not overwrite an
existing file unless you pass --force.

Judgement calls worth knowing about
-----------------------------------
* Jitomate     = SNIIM "Tomate Bola" + "Tomate Saladette" (red tomato).
                 Tomate verde (tomatillo) is its own INPC generic.
* Limon        = the c/semilla sizes plus s/semilla. "Lima" is a different
                 fruit and sits in Otras frutas.
* Uva          = fresh table grapes only. "Uva pasa" (raisins) is processed and
                 sits in Otras frutas, as does "Ciruela pasa" (prunes).
* Cebolla      = bola, bola grande, morada and de rabo.
* Papa y otros tuberculos = the six Papa varieties plus Camote and Jicama.
                 Betabel and Rabano are roots but the INPC groups them with
                 other vegetables, so they land in Otras verduras.
* Chile seco   = ancho, guajillo, pasilla, mirasol, de Arbol seco, Puya seco.
                 Every other chile is fresh: poblano and serrano are their own
                 generics, the rest go to Otros chiles frescos.
* EXCLUDED     = nuts and seasonings that SNIIM reports in this module but the
                 INPC classifies elsewhere: Cacahuate, Nuez, Pistache, Jamaica,
                 Oregano, Yerbabuena. They are written to the CSV with category
                 "excluido" so nothing is silently dropped.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("sniim")

UNMAPPED = "sin_mapear"
EXCLUDED = "excluido"

# The 32 categories requested, in the order they were given.
CATEGORIES: list[tuple[str, str, str]] = [
    ("aguacate", "Aguacate", "frutas"),
    ("durazno", "Durazno", "frutas"),
    ("guayaba", "Guayaba", "frutas"),
    ("limon", "Limón", "frutas"),
    ("manzana", "Manzana", "frutas"),
    ("melon", "Melón", "frutas"),
    ("naranja", "Naranja", "frutas"),
    ("papaya", "Papaya", "frutas"),
    ("pera", "Pera", "frutas"),
    ("pina", "Piña", "frutas"),
    ("platanos", "Plátanos", "frutas"),
    ("sandia", "Sandía", "frutas"),
    ("uva", "Uva", "frutas"),
    ("otras_frutas", "Otras frutas", "frutas"),
    ("calabacita", "Calabacita", "verduras"),
    ("cebolla", "Cebolla", "verduras"),
    ("chayote", "Chayote", "verduras"),
    ("chile_poblano", "Chile poblano", "verduras"),
    ("chile_serrano", "Chile serrano", "verduras"),
    ("ejotes", "Ejotes", "verduras"),
    ("jitomate", "Jitomate", "verduras"),
    ("lechuga_y_col", "Lechuga y col", "verduras"),
    ("nopales", "Nopales", "verduras"),
    ("papa_y_otros_tuberculos", "Papa y otros tubérculos", "verduras"),
    ("pepino", "Pepino", "verduras"),
    ("tomate_verde", "Tomate verde", "verduras"),
    ("zanahoria", "Zanahoria", "verduras"),
    ("otras_verduras_y_legumbres", "Otras verduras y legumbres", "verduras"),
    ("otros_chiles_frescos", "Otros chiles frescos", "verduras"),
    ("cilantro_epazote_perejil", "Cilantro, epazote y perejil", "verduras"),
    ("chile_seco", "Chile seco", "verduras"),
    ("frijol", "Frijol", "granos"),
]

CATEGORY_LABEL = {slug: label for slug, label, _ in CATEGORIES}
CATEGORY_GROUP = {slug: grupo for slug, _, grupo in CATEGORIES}

# --- explicit product_id -> category, module "frutas" ----------------------
FRUTAS_MAP: dict[str, list[int]] = {
    "aguacate": [130, 131, 133, 135, 136, 137, 138, 139],
    "durazno": [290, 294, 297],
    "guayaba": [378],
    "limon": [417, 418, 419, 421, 423, 426],
    "manzana": [519, 522, 526],
    "melon": [499, 500, 501, 502, 503, 505, 506, 507, 508, 510, 512, 513, 516],
    "naranja": [534, 536, 537, 538, 546, 549, 551],
    "papaya": [609, 620, 623],
    "pera": [626, 630, 632, 634, 638, 639, 640, 645],
    "pina": [646, 648, 651],
    "platanos": [663, 665, 666, 670, 723, 727, 728, 731, 732, 736],
    "sandia": [783, 785, 787, 791, 795, 796, 799],
    "uva": [851, 852, 854, 857, 858, 861, 863, 866, 869, 871, 872],
    "otras_frutas": [
        178,                                # Caña
        196, 197, 198, 199, 200, 201, 203, 207,  # Ciruela (incl. pasa)
        250,                                # Coco
        356, 358,                           # Fresa
        372, 373,                           # Granada
        374,                                # Guanabana
        399,                                # Kiwi
        413,                                # Lima
        429,                                # Mamey
        447, 451, 455, 459,                 # Mandarina
        465, 469, 473, 479, 480, 485, 489, 494,  # Mango
        529,                                # Membrillo
        539,                                # Nanche
        554,                                # Nectarina
        774,                                # Pitaya
        800,                                # Tamarindo
        804,                                # Tejocote
        811, 823, 824, 832,                 # Toronja
        845, 848,                           # Tuna
        862,                                # Uva pasa
        888,                                # Zapote
        891,                                # Zarzamora
    ],
    "calabacita": [166, 170, 180],
    "cebolla": [182, 183, 187, 190],
    "chayote": [272, 275],
    "chile_poblano": [242],
    "chile_serrano": [246],
    "ejotes": [299, 302, 304],
    "jitomate": [836, 839],
    "lechuga_y_col": [253, 257, 265, 401, 403, 405, 407, 409],
    "nopales": [556, 560],
    "papa_y_otros_tuberculos": [
        740, 748, 749, 753, 766, 767,       # Papa
        175,                                # Camote
        392, 396, 397,                      # Jícama
    ],
    "pepino": [771],
    "tomate_verde": [842],
    "zanahoria": [878, 880, 884],
    "otras_verduras_y_legumbres": [
        31,                                 # Acelga
        142, 144, 146,                      # Ajo
        152,                                # Apio
        157,                                # Berenjena
        160,                                # Betabel
        162,                                # Brócoli
        169,                                # Calabaza de castilla
        260, 261, 263,                      # Coliflor
        270,                                # Champiñón
        279, 285,                           # Chícharo
        306, 307, 308,                      # Elote
        314,                                # Espárrago
        316,                                # Espinaca
        385,                                # Haba verde
        775,                                # Rábano
    ],
    "otros_chiles_frescos": [
        213, 215, 220, 221, 222, 223, 225, 227, 230, 232, 233, 238, 239
    ],
    "cilantro_epazote_perejil": [208, 312, 635],
    "chile_seco": [216, 217, 228, 236, 237, 245],
    EXCLUDED: [
        181,                                # Cacahuate (nut)
        389,                                # Jamaica (dried flower)
        563, 565, 566, 567,                 # Nuez (nut)
        607,                                # Orégano (dried seasoning)
        655,                                # Pistache (nut)
        876,                                # Yerbabuena (seasoning herb)
    ],
}

# --- module "granos" -------------------------------------------------------
# Every product whose label starts with "Frijol" maps to the frijol generic.
# Anything else in the Granos catalog is out of scope for this project.
GRANOS_PREFIX_MAP: dict[str, str] = {"frijol": "frijol"}


def _invert(mapping: dict[str, list[int]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for cat, ids in mapping.items():
        for pid in ids:
            if pid in out and out[pid] != cat:
                raise ValueError(
                    f"product {pid} assigned to both {out[pid]} and {cat} - fix mapping.py"
                )
            out[pid] = cat
    return out


def build(cfg) -> pd.DataFrame:
    """Build the mapping table from the scraped catalogs plus the rules above."""
    from . import store

    rows = []

    pid_to_cat = _invert(FRUTAS_MAP)
    frutas = store.load_catalog(cfg, "frutas_productos")
    for r in frutas.itertuples():
        pid = int(r.id)
        cat = pid_to_cat.get(pid, UNMAPPED)
        rows.append(
            {
                "modulo": "frutas",
                "producto_id": pid,
                "producto_sniim": r.label,
                "categoria": cat,
                "categoria_label": CATEGORY_LABEL.get(cat, cat),
                "grupo": CATEGORY_GROUP.get(cat, ""),
            }
        )

    granos = store.load_catalog(cfg, "granos_productos")
    for r in granos.itertuples():
        pid = int(r.id)
        label = str(r.label)
        low = label.strip().lower()
        cat = UNMAPPED
        for prefix, target in GRANOS_PREFIX_MAP.items():
            if low.startswith(prefix):
                cat = target
                break
        else:
            cat = EXCLUDED  # other grains are out of scope, but recorded
        rows.append(
            {
                "modulo": "granos",
                "producto_id": pid,
                "producto_sniim": label,
                "categoria": cat,
                "categoria_label": CATEGORY_LABEL.get(cat, cat),
                "grupo": CATEGORY_GROUP.get(cat, ""),
            }
        )

    df = pd.DataFrame(rows).sort_values(["modulo", "categoria", "producto_sniim"])
    return df.reset_index(drop=True)


def write(cfg, df: pd.DataFrame, force: bool = False) -> Path:
    path = cfg.mapping_path
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. It is hand-editable - pass --force to regenerate."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load(cfg) -> pd.DataFrame:
    path = cfg.mapping_path
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python run.py mapping`")
    df = pd.read_csv(path)
    df["producto_id"] = df["producto_id"].astype(int)
    return df


def report(df: pd.DataFrame) -> str:
    lines = []
    counts = (
        df[~df["categoria"].isin([UNMAPPED, EXCLUDED])]
        .groupby(["grupo", "categoria_label"])
        .size()
        .reset_index(name="n_variedades")
    )
    lines.append(f"{len(counts)} of 32 categories have at least one product mapped")
    for grupo in ("frutas", "verduras", "granos"):
        sub = counts[counts["grupo"] == grupo]
        if sub.empty:
            continue
        lines.append(f"\n  {grupo}:")
        for r in sub.itertuples():
            lines.append(f"    {r.categoria_label:<32} {r.n_variedades:>3} varieties")

    missing = set(CATEGORY_LABEL.values()) - set(counts["categoria_label"])
    if missing:
        lines.append("\n  CATEGORIES WITH NO PRODUCTS: " + ", ".join(sorted(missing)))

    excl = df[df["categoria"] == EXCLUDED]
    if len(excl):
        lines.append(f"\n  {len(excl)} products deliberately excluded (out of INPC scope)")

    unm = df[df["categoria"] == UNMAPPED]
    if len(unm):
        lines.append(f"\n  {len(unm)} UNMAPPED products - review config/mapping_inpc.csv:")
        for r in unm.itertuples():
            lines.append(f"    [{r.modulo}] {r.producto_id:>5}  {r.producto_sniim}")
    else:
        lines.append("\n  no unmapped products")
    return "\n".join(lines)
