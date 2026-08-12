"""INEGI "Precios Promedio" - actual RETAIL prices in pesos, by generic and city.

Why this is valuable: the INPC index tells you relative change, but this gives
levels in MXN per unit, which is directly comparable to SNIIM's wholesale
MXN/kg. That makes the retail-wholesale MARGIN observable, which in turn makes an
error-correction specification possible rather than hypothetical.

Endpoint (no token):
  POST https://www.inegi.org.mx/app/preciospromedio/Exportacion.aspx
    series=,045,070,&tipo=CSV&pi=202408&pf=202607&ent=,01_01,01_14,&bs=18a

Gotchas found by probing:
  * `ent` must be non-empty and use the `01_<citycode>` form. An empty `ent`
    returns headers only; `,01,` returns the HTML page instead of CSV.
  * There is NO national row - INEGI notes the city breakdown is "únicamente
    informativa". You must aggregate cities yourself (see catalogo.city_weights).
  * Response is Windows-1252.
  * Do a GET on the app first; it wants an ASP.NET session cookie.

Vintages - generic CODES ARE NOT STABLE ACROSS THEM:
  bs=""     2011/01 - 2018/07   (in this vintage 045 = Plátanos, NOT Aguacate!)
  bs="18"   2018/08 - 2024/07
  bs="18a"  2024/08 - current
Always key on the generic NAME when crossing a vintage boundary.

Units are NOT uniformly KG. Measured over 39,147 real quotes (32 genéricos x
55 cities x May-Jul 2026), 8.2% of quotes are not per kilogram, heavily
concentrated:
    087 Cilantro/epazote/perejil  94.7% MANOJO  - only 8/55 cities quote KG
    071 Lechuga y col             79.7% PZA     - only 35/55 cities quote KG
    072 Nopales                   40.7% PZA/BOLSA
    054 Piña                      36.0% PZA
    080 Otras verduras            30.2% MANOJO/PZA
    050 Melón                     24.2% PZA
    061 Cebolla                    6.0% MANOJO
A naive mean over all quotes is therefore meaningless for those. Filter to
`Unidad == 'KG'`; after filtering, 30 of 32 genéricos still have >=53/55 cities.
The two exceptions (087, 071) are listed in catalogo.PRECIOS_UNIT_UNRELIABLE.
`Cantidad` is always 1, so `Precio promedio` is per the stated `Unidad`.
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.parse
import urllib.request

log = logging.getLogger("sniim")

APP = "https://www.inegi.org.mx/app/preciospromedio/"
URL = APP + "Exportacion.aspx"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# All 55 Precios Promedio city codes.
CITY_CODES = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
    "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
    "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48",
    "49", "50", "51", "52", "53", "54", "55",
]

VINTAGES = [("", 201101, 201807), ("18", 201808, 202407), ("18a", 202408, 209912)]


def vintage_for(periodo: int) -> str:
    """periodo as YYYYMM -> the `bs` vintage string covering it."""
    for bs, lo, hi in VINTAGES:
        if lo <= periodo <= hi:
            return bs
    raise ValueError(f"no Precios Promedio vintage covers {periodo}")


def fetch(generic_codes: list[str], pi: int, pf: int,
          cities: list[str] | None = None, bs: str = "18a",
          timeout: int = 900) -> tuple[list[str], list[list[str]]]:
    """Fetch raw quote rows. pi/pf are YYYYMM. Returns (header, rows).

    Volume warning: one generic x 55 cities x 91 months is roughly 33k rows and
    7.6 MB, so chunk by generic or by period rather than asking for everything.
    """
    cities = cities or CITY_CODES
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor())
    # The app wants a session cookie before it will hand back CSV.
    opener.open(urllib.request.Request(APP + "?bs=" + bs, headers=UA), timeout=120).read()

    form = {
        "series": "," + ",".join(generic_codes) + ",",
        "tipo": "CSV",
        "pi": str(pi), "pf": str(pf),
        "ent": "," + ",".join("01_" + c for c in cities) + ",",
        "bs": bs,
    }
    req = urllib.request.Request(
        URL, data=urllib.parse.urlencode(form).encode(),
        headers={**UA, "Referer": APP + "?bs=" + bs})
    text = opener.open(req, timeout=timeout).read().decode("cp1252", "replace")

    rows = list(csv.reader(io.StringIO(text)))
    try:
        hi = next(i for i, r in enumerate(rows) if r and r[0].strip().startswith("A")
                  and "o" in r[0])  # "Año"
    except StopIteration:
        raise RuntimeError(
            "Precios Promedio returned no data header. Check `ent` formatting "
            "and that the period range matches the `bs` vintage. First rows: %r"
            % rows[:3])
    header = [c.strip() for c in rows[hi]]
    data = [r for r in rows[hi + 1:] if len(r) == len(header)]
    log.info("Precios Promedio bs=%s %s..%s: %d quote rows", bs, pi, pf, len(data))
    return header, data


def to_kg_frame(header: list[str], rows: list[list[str]]):
    """Rows -> pandas DataFrame, KG quotes only, with the unit mix reported.

    Returns (df, diagnostics). Filtering to KG is not optional - see the module
    docstring; a mean across mixed PZA/MANOJO/KG quotes is not a price.
    """
    import pandas as pd

    df = pd.DataFrame(rows, columns=header)
    col = {c.lower().strip(): c for c in header}

    def pick(*cands):
        for c in cands:
            if c in col:
                return col[c]
        raise KeyError(f"none of {cands} in {header}")

    c_year = pick("año", "ano")
    c_mes = pick("mes")
    c_city = pick("clave ciudad")
    c_cityname = pick("nombre ciudad")
    c_gen = pick("genérico", "generico")
    c_genkey = pick("clave genérico", "clave generico")
    c_price = pick("precio promedio")
    c_unit = pick("unidad")

    df[c_price] = pd.to_numeric(df[c_price], errors="coerce")
    total = len(df)
    unit_mix = df[c_unit].str.strip().str.upper().value_counts().to_dict()

    keep = df[c_unit].str.strip().str.upper().eq("KG") & df[c_price].notna()
    out = df[keep].copy()
    out["periodo"] = out[c_year].str.strip() + "-" + out[c_mes].str.strip().str.zfill(2)
    out = out.rename(columns={
        c_city: "clave_ciudad", c_cityname: "ciudad",
        c_gen: "generico", c_genkey: "clave_generico", c_price: "precio_mxn_kg",
    })[["periodo", "clave_generico", "generico", "clave_ciudad", "ciudad", "precio_mxn_kg"]]

    diag = {
        "quotes_total": total,
        "quotes_kg": len(out),
        "pct_non_kg": 100.0 * (total - len(out)) / total if total else 0.0,
        "unit_mix": unit_mix,
    }
    return out, diag


def national_mean(df, city_weights: dict[str, float] | None = None):
    """Aggregate city quotes to a national retail price per generic-month.

    Two-stage and geometric, to mirror both INEGI's own Jevons aggregation and the
    wholesale index built in sniim/aggregate.py: average the quotes within a city
    first, then combine cities. Passing `city_weights` uses INPC city weights
    instead of equal-weighting - preferred, though those weights are whole-basket
    rather than per-generic (see catalogo.city_weights).
    """
    import numpy as np
    import pandas as pd

    per_city = (df.groupby(["periodo", "clave_generico", "generico", "clave_ciudad"])
                  ["precio_mxn_kg"]
                  .apply(lambda s: np.exp(np.log(s[s > 0]).mean()))
                  .reset_index())
    if city_weights:
        per_city["w"] = per_city["clave_ciudad"].map(city_weights).fillna(0.0)
    else:
        per_city["w"] = 1.0

    def agg(g):
        g = g[(g["precio_mxn_kg"] > 0) & (g["w"] > 0)]
        if g.empty:
            return pd.Series({"precio_geo": np.nan, "n_ciudades": 0})
        lw = np.average(np.log(g["precio_mxn_kg"]), weights=g["w"])
        return pd.Series({"precio_geo": float(np.exp(lw)), "n_ciudades": len(g)})

    return (per_city.groupby(["periodo", "clave_generico", "generico"])
                    .apply(agg, include_groups=False).reset_index())
