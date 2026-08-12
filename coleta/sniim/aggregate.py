"""DuckDB build and INPC-category aggregation.

All aggregation runs as SQL inside DuckDB directly over the Parquet files. The
raw table reaches tens of millions of rows, so pulling it into pandas would be
wasteful; DuckDB streams it.

The aggregation ladder
----------------------
There is ONE base table, and everything else is derived from it. That matters:
an earlier version derived the variety-level and category-level paths separately
from the raw observations, which let an arithmetic mean creep into the base of a
supposedly geometric index.

  pesos_mercado         market -> INPC city weight (see "Market weighting")
  var_market_daily      variety x market x day     <- THE base, geometric over origins
  cat_market_daily      category x market x day    (from var_market_daily)
  cat_national_daily    category x day
  var_market_monthly    variety x market x month   (from var_market_daily)
  var_national_monthly  variety x month
  cat_market_monthly    category x market x month
  cat_national_monthly  category x month, levels + base-100 + MoM/YoY
  cat_index_monthly     category x month, chained Jevons index  <- for inflation

Means, and which one where
--------------------------
Every price aggregate carries `precio` (arithmetic) and `precio_geo` (geometric,
exp(avg(ln(x)))). Prefer the geometric one: its logarithm is the arithmetic mean
of the logs, so a variety going 10 -> 20 moves it exactly as much as one going
100 -> 200, whereas the arithmetic mean is dominated by the expensive items -
average garlic at 150 with celery at 10 and you have built a garlic index. By
AM-GM `precio_geo <= precio` always, and the gap widens with heterogeneity.

Critically, `precio_geo` is now geometric at EVERY rung including the base. It
previously used an arithmetic mean over origin states and days inside a
variety-market-month, so the "geometric" index had an arithmetic layer beneath
it and disagreed with cat_market_daily, which computed the same quantity
geometrically. The practical difference is small - within one variety, one market
and one month, dispersion is modest - but it was not defensible as written.

Market weighting
----------------
Markets are weighted by the INPC weight of the city they serve, via
config/crosswalk_mercados.csv (49 SNIIM markets -> 39 of INEGI's 55 weighted
cities, covering 88.5 of the 100 total city weight). Where several SNIIM markets
serve one city, that city's weight is divided equally between them, so the city
total is preserved.

This is not cosmetic. Equal-weighting gives each of the 49 markets 2.0%; INPC
weighting gives CDMX's two markets (Iztapalapa and Ecatepec) 25.4% between them.
Banxico's own SNIIM-based work finds cities supplied through intermediaries -
chiefly CDMX's Central de Abasto - run 1.7-1.9% higher, so treating a thin
regional market as equal to CEDA is a real distortion, not a neutral choice.

Both weightings are computed and stored side by side rather than hidden behind a
config switch, because a switch that silently changes your numbers is worse than
two columns you can compare.

Matched-model index, and what changed
-------------------------------------
Averaging price LEVELS across a heterogeneous category lets the expensive variety
dominate, and makes the series jump whenever a variety enters or leaves. The fix
for an elementary aggregate with no consumption weights is a chained Jevons
index: geometric mean of price relatives over units present at BOTH ends of each
step, chained.

`cat_index_monthly` now offers three variants so the effect of each design choice
is visible:

  indice_jevons        matched on variety x MARKET cells, INPC-city-weighted
                       <- recommended
  indice_jevons_equal  matched on variety x market cells, equal market weights
  indice_jevons_var    matched on variety only, markets averaged first
                       <- the previous behaviour, kept for comparison

Matching on cells rather than varieties closes a real hole: previously markets
were averaged BELOW the matching rung, so a market that started or stopped
reporting still moved the index. That is composition drift masquerading as price
change, and because market entry/exit is seasonal it correlates with the CPI's own
seasonality and can look like genuine predictive skill.

Gap handling
------------
A strict "previous calendar month" join would drop every month whose predecessor
is missing. For a strictly seasonal product that deletes exactly the months
carrying the annual price step, and the index then reports zero inflation for a
product that inflated. So each step links a month to the previous month the
category ACTUALLY has, and `meses_puente` records how many calendar months the
step spans. If nothing is present at both ends, the level is carried forward and
`cadena_rota` is set, so the assumption is visible rather than hidden.

Time comparisons are date-based, never row-based: `lag(x, 12)` on a seasonal
series reaches back 16 or 24 calendar months while still being labelled "annual".
A missing comparison month yields NULL in cat_national_monthly.

None of this is an INPC replica: no consumption weights, and SNIIM is wholesale.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger("sniim")

CURATED_TABLES = [
    "pesos_mercado",
    "var_market_daily",
    "cat_market_daily",
    "cat_national_daily",
    "var_market_monthly",
    "var_national_monthly",
    "cat_market_monthly",
    "cat_national_monthly",
    "cat_index_monthly",
]

# Everything is exported to Parquet. Only the small tables also go to CSV -
# var_market_daily and cat_market_daily reach millions of rows at full history
# and a CSV of either is a gigabyte of no use to anyone.
CSV_TABLES = [
    "pesos_mercado",
    "cat_national_daily",
    "cat_national_monthly",
    "cat_market_monthly",
    "var_national_monthly",
    "cat_index_monthly",
]

EXCLUDED_CATEGORIES = "('excluido', 'sin_mapear')"


def connect(cfg, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.duckdb_path), read_only=read_only)
    con.execute("PRAGMA threads=4")
    return con


def _globs(cfg) -> list[str]:
    out = []
    for modulo in ("frutas", "granos"):
        if list((cfg.raw_dir / modulo).glob("producto_id=*/anio=*/part.parquet")):
            out.append(
                str(cfg.raw_dir / modulo / "producto_id=*" / "anio=*" / "part.parquet")
            )
    return out


def build_views(cfg, con: duckdb.DuckDBPyConnection) -> int:
    """Register raw observations, the category join and the market weights."""
    globs = _globs(cfg)
    if not globs:
        raise RuntimeError(
            f"no parquet files under {cfg.raw_dir} - run `python run.py backfill` first"
        )
    glob_list = ", ".join(f"'{g}'" for g in globs)

    con.execute(
        f"""
        CREATE OR REPLACE VIEW obs_raw AS
        SELECT * FROM read_parquet([{glob_list}], union_by_name=true);
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE mapa AS
        SELECT * FROM read_csv_auto('{cfg.mapping_path.as_posix()}', header=true);
        """
    )
    # The mapping is keyed on (modulo, producto_id): product ids are only unique
    # within a module, so joining on producto_id alone could both mislabel rows
    # and fan the join out.
    dupes = con.execute(
        "SELECT count(*) FROM (SELECT modulo, producto_id FROM mapa "
        "GROUP BY 1,2 HAVING count(*) > 1)"
    ).fetchone()[0]
    if dupes:
        raise RuntimeError(
            f"{cfg.mapping_path} has {dupes} duplicate (modulo, producto_id) rows - "
            "that would fan out the join and double-count varieties. Fix the CSV."
        )

    con.execute(
        """
        CREATE OR REPLACE VIEW obs AS
        SELECT o.*, m.categoria, m.categoria_label, m.grupo AS grupo_inpc
        FROM obs_raw o
        LEFT JOIN mapa m
          ON m.modulo = o.modulo AND m.producto_id = o.producto_id;
        """
    )

    _build_market_weights(cfg, con)

    n = con.execute("SELECT count(*) FROM obs_raw").fetchone()[0]
    unmatched = con.execute("SELECT count(*) FROM obs WHERE categoria IS NULL").fetchone()[0]
    if unmatched:
        log.warning(
            "%d observations have no mapping row (new SNIIM products?) - "
            "run `python run.py catalog` then `python run.py mapping --force`",
            unmatched,
        )
    return n


def _build_market_weights(cfg, con: duckdb.DuckDBPyConnection) -> None:
    """market -> INPC city weight, plus an equal-weight column.

    Resolution is a cascade, and every market is resolved SOMEHOW so that coverage
    can never silently shrink:

      exacto           the market is named in config/crosswalk_mercados.csv
      fallback_estado  it is not, but its state prefix matches other markets in the
                       crosswalk; use that state's largest-weight INPC city
      sin_ciudad       not even the state matches; assign the median weight

    The state fallback exists because SNIIM has RENAMED markets over 29 years. The
    current dropdown lists 49 destinos, but historical data contains labels that are
    no longer offered - e.g. "Chihuahua: Central de Abasto de Cd. Juárez" alongside
    today's "Chihuahua: Mercado de Abasto de Cd. Juárez". Without the fallback those
    rows would be excluded from every weighted figure, and the exclusion would grow
    quietly as more history is collected.

    The city-weight split is computed over the markets ACTUALLY PRESENT in the data,
    not over the crosswalk, so a city served by three markets in the data divides its
    weight three ways. Note that the Jevons stage-2 renormalises by the weights of
    the markets present in each step, so a market that only reports for part of the
    sample does not dilute its city in the months it is absent.
    """
    sm = cfg.aggregate.get("start_month")
    ventana_w = f"AND fecha >= DATE '{sm}-01'" if sm else ""
    cw = cfg.crosswalk_path
    ci = cfg.ciudades_path
    if not cw.exists() or not ci.exists():
        log.warning(
            "%s or %s missing - falling back to equal market weights. The INPC-weighted "
            "columns will equal the equal-weighted ones.", cw.name, ci.name)
        con.execute(
            """
            CREATE OR REPLACE TABLE pesos_mercado AS
            SELECT DISTINCT destino, 1.0 AS peso_inpc, 1.0 AS peso_equal,
                   NULL::VARCHAR AS clave_ciudad_inpc, NULL::VARCHAR AS ciudad_inpc,
                   'sin_crosswalk' AS metodo, 1 AS mercados_en_ciudad
            FROM obs WHERE destino IS NOT NULL """ + ventana_w + ";"
        )
        return

    con.execute(
        f"""
        CREATE OR REPLACE TABLE pesos_mercado AS
        WITH destinos AS (
            SELECT DISTINCT destino, trim(split_part(destino, ':', 1)) AS estado
            FROM obs WHERE destino IS NOT NULL {ventana_w}
        ),
        cw AS (
            SELECT destino_sniim                                    AS destino,
                   trim(split_part(destino_sniim, ':', 1))           AS estado,
                   lpad(CAST(clave_ciudad_inpc AS VARCHAR), 2, '0')  AS clave_ciudad_inpc,
                   ciudad_inpc
            FROM read_csv_auto('{cw.as_posix()}', header=true, all_varchar=true)
        ),
        ci AS (
            SELECT lpad(CAST(clave_ciudad_preciospromedio AS VARCHAR), 2, '0') AS clave,
                   CAST(ponderador_2024 AS DOUBLE)                   AS ponderador
            FROM read_csv_auto('{ci.as_posix()}', header=true, all_varchar=true)
        ),
        exacto AS (
            SELECT d.destino, d.estado, cw.clave_ciudad_inpc, cw.ciudad_inpc,
                   'exacto' AS metodo
            FROM destinos d JOIN cw ON cw.destino = d.destino
        ),
        -- largest-weight INPC city per state, among crosswalked markets
        estado_map AS (
            SELECT estado, clave_ciudad_inpc, ciudad_inpc,
                   row_number() OVER (PARTITION BY estado ORDER BY ci.ponderador DESC,
                                      clave_ciudad_inpc) AS rn
            FROM (SELECT DISTINCT estado, clave_ciudad_inpc, ciudad_inpc FROM cw) c
            JOIN ci ON ci.clave = c.clave_ciudad_inpc
        ),
        fallback AS (
            SELECT d.destino, d.estado, e.clave_ciudad_inpc, e.ciudad_inpc,
                   'fallback_estado' AS metodo
            FROM destinos d
            LEFT JOIN exacto x ON x.destino = d.destino
            JOIN estado_map e ON e.estado = d.estado AND e.rn = 1
            WHERE x.destino IS NULL
        ),
        resueltos AS (SELECT * FROM exacto UNION ALL SELECT * FROM fallback),
        huerfanos AS (
            SELECT d.destino, d.estado, NULL::VARCHAR AS clave_ciudad_inpc,
                   NULL::VARCHAR AS ciudad_inpc, 'sin_ciudad' AS metodo
            FROM destinos d LEFT JOIN resueltos r ON r.destino = d.destino
            WHERE r.destino IS NULL
        ),
        todos AS (SELECT * FROM resueltos UNION ALL SELECT * FROM huerfanos),
        n AS (
            SELECT clave_ciudad_inpc, count(*) AS n_mercados
            FROM todos WHERE clave_ciudad_inpc IS NOT NULL GROUP BY 1
        ),
        mediana AS (SELECT median(ponderador) AS med FROM ci)
        SELECT t.destino, t.estado, t.clave_ciudad_inpc, t.ciudad_inpc, t.metodo,
               coalesce(n.n_mercados, 1)                             AS mercados_en_ciudad,
               coalesce(ci.ponderador / n.n_mercados, m.med)         AS peso_inpc,
               1.0                                                   AS peso_equal
        FROM todos t
        LEFT JOIN n  USING (clave_ciudad_inpc)
        LEFT JOIN ci ON ci.clave = t.clave_ciudad_inpc
        CROSS JOIN mediana m;
        """
    )

    rows = con.execute(
        "SELECT metodo, count(*) FROM pesos_mercado GROUP BY 1 ORDER BY 2 DESC").fetchall()
    log.info("market weights resolved: %s",
             ", ".join(f"{n} {m}" for m, n in rows))

    for metodo, label, level in (
        ("fallback_estado", "matched only by state prefix (renamed market?)", log.warning),
        ("sin_ciudad", "could not be matched to any INPC city - given the median weight", log.error),
    ):
        bad = con.execute(
            "SELECT destino FROM pesos_mercado WHERE metodo = ? ORDER BY destino", [metodo]
        ).fetchall()
        if bad:
            level("%d market(s) %s. Add them to config/crosswalk_mercados.csv:",
                  len(bad), label)
            for (d,) in bad[:10]:
                level("    %s", d)

    zero = con.execute("SELECT count(*) FROM pesos_mercado WHERE peso_inpc IS NULL "
                       "OR peso_inpc <= 0").fetchone()[0]
    if zero:
        raise RuntimeError(
            f"{zero} markets ended up with a null or zero weight, which would exclude "
            "them from every weighted figure. Fix config/crosswalk_mercados.csv.")

    cov = con.execute(
        "SELECT round(sum(peso), 4) FROM (SELECT DISTINCT clave_ciudad_inpc, "
        "peso_inpc * mercados_en_ciudad AS peso FROM pesos_mercado "
        "WHERE clave_ciudad_inpc IS NOT NULL)").fetchone()[0]
    log.info("%d markets in the data, covering %.2f of 100 INPC city weight",
             con.execute("SELECT count(*) FROM pesos_mercado").fetchone()[0], cov or 0.0)


def build_aggregates(cfg, con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    a = cfg.aggregate
    price = a["price_column"]
    if price not in ("precio_frec", "precio_min", "precio_max"):
        raise ValueError(f"aggregate.price_column must be a price column, got {price!r}")
    min_obs = int(a["min_obs_per_market_day"])
    base_month = str(a["index_base_month"])

    # Analysis-window filter, applied once at the base of the ladder so every
    # derived table shares it. Products collected before start_year was narrowed
    # still hold older data on disk; without this the panel is unbalanced and the
    # aggregate reconstruction is not comparable across time.
    start_month = a.get("start_month")
    if start_month:
        ventana = f"AND fecha >= DATE '{str(start_month)}-01'"
        log.info("analysis window starts %s-01 (older rows on disk are ignored)",
                 start_month)
    else:
        ventana = ""

    # `isfinite` as well as `> 0`: infinity passes `> 0` and survives a parquet
    # round trip, and one inf would poison every downstream ln()/avg().
    price_ok = f"{price} IS NOT NULL AND {price} > 0 AND isfinite({price})"

    # ---------------------------------------------------------------- base
    # variety x market x day, geometric over the origin states reported there.
    # This is the single base every other table is derived from.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE var_market_daily AS
        SELECT
            categoria, categoria_label, grupo_inpc,
            producto_id,
            any_value(producto)                 AS producto,
            fecha,
            destino, destino_estado, destino_mercado,
            exp(avg(ln({price})))               AS precio_geo,
            avg({price})                        AS precio,
            avg(precio_min) FILTER (
                WHERE precio_min IS NOT NULL AND precio_max IS NOT NULL
            )                                   AS precio_min_prom,
            avg(precio_max) FILTER (
                WHERE precio_min IS NOT NULL AND precio_max IS NOT NULL
            )                                   AS precio_max_prom,
            count(*)                            AS n_obs,
            count(DISTINCT origen)              AS n_origenes
        FROM obs
        WHERE {price_ok}
          AND categoria IS NOT NULL
          AND categoria NOT IN {EXCLUDED_CATEGORIES}
          {ventana}
        GROUP BY categoria, categoria_label, grupo_inpc, producto_id, fecha,
                 destino, destino_estado, destino_mercado;
        """
    )

    # ------------------------------------------------- category x market x day
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cat_market_daily AS
        SELECT
            categoria, categoria_label, grupo_inpc, fecha,
            destino, destino_estado, destino_mercado,
            avg(precio)                     AS precio,
            exp(avg(ln(precio_geo)))        AS precio_geo,
            median(precio_geo)              AS precio_mediana,
            min(precio_geo)                 AS precio_min_obs,
            max(precio_geo)                 AS precio_max_obs,
            avg(precio_min_prom)            AS precio_min_prom,
            avg(precio_max_prom)            AS precio_max_prom,
            sum(n_obs)                      AS n_obs,
            count(*)                        AS n_variedades,
            sum(n_origenes)                 AS n_origenes_variedad
        FROM var_market_daily
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        HAVING sum(n_obs) >= {min_obs};
        """
    )

    # ------------------------------------------------------- category x day
    con.execute(
        """
        CREATE OR REPLACE TABLE cat_national_daily AS
        SELECT
            d.categoria, d.categoria_label, d.grupo_inpc, d.fecha,
            exp(sum(coalesce(p.peso_inpc, 0) * ln(d.precio_geo))
                / nullif(sum(coalesce(p.peso_inpc, 0)), 0))  AS precio_geo,
            exp(avg(ln(d.precio_geo)))                       AS precio_geo_equal,
            sum(coalesce(p.peso_inpc, 0) * d.precio)
                / nullif(sum(coalesce(p.peso_inpc, 0)), 0)   AS precio,
            avg(d.precio)                                    AS precio_equal,
            median(d.precio_geo)                             AS precio_mediana,
            stddev_samp(ln(d.precio_geo))                    AS sd_log,
            min(d.precio_geo)                                AS precio_min_mercado,
            max(d.precio_geo)                                AS precio_max_mercado,
            count(*)                                         AS n_mercados,
            sum(d.n_obs)                                     AS n_obs,
            sum(d.n_variedades)                              AS n_variedades_mercado_dia
        FROM cat_market_daily d
        LEFT JOIN pesos_mercado p ON p.destino = d.destino
        GROUP BY 1, 2, 3, 4;
        """
    )

    # --------------------------------------------- variety x market x month
    con.execute(
        """
        CREATE OR REPLACE TABLE var_market_monthly AS
        SELECT
            categoria, categoria_label, grupo_inpc, producto_id,
            any_value(producto)             AS producto,
            date_trunc('month', fecha)      AS mes,
            destino, destino_estado, destino_mercado,
            exp(avg(ln(precio_geo)))        AS precio_geo,
            avg(precio)                     AS precio,
            count(*)                        AS n_dias,
            sum(n_obs)                      AS n_obs
        FROM var_market_daily
        GROUP BY categoria, categoria_label, grupo_inpc, producto_id,
                 date_trunc('month', fecha), destino, destino_estado, destino_mercado;
        """
    )

    # ---------------------------------------------------- variety x month
    con.execute(
        """
        CREATE OR REPLACE TABLE var_national_monthly AS
        SELECT
            v.categoria, v.categoria_label, v.grupo_inpc, v.producto_id,
            any_value(v.producto)                            AS producto,
            v.mes,
            exp(sum(coalesce(p.peso_inpc, 0) * ln(v.precio_geo))
                / nullif(sum(coalesce(p.peso_inpc, 0)), 0))  AS precio_geo,
            exp(avg(ln(v.precio_geo)))                       AS precio_geo_equal,
            avg(v.precio)                                    AS precio,
            count(*)                                         AS n_mercados,
            sum(v.n_obs)                                     AS n_obs
        FROM var_market_monthly v
        LEFT JOIN pesos_mercado p ON p.destino = v.destino
        GROUP BY v.categoria, v.categoria_label, v.grupo_inpc, v.producto_id, v.mes;
        """
    )

    # -------------------------------------------- category x market x month
    con.execute(
        """
        CREATE OR REPLACE TABLE cat_market_monthly AS
        SELECT
            categoria, categoria_label, grupo_inpc,
            date_trunc('month', fecha)  AS mes,
            destino, destino_estado, destino_mercado,
            avg(precio)                 AS precio,
            exp(avg(ln(precio_geo)))    AS precio_geo,
            median(precio_geo)          AS precio_mediana,
            count(*)                    AS n_dias,
            sum(n_obs)                  AS n_obs
        FROM cat_market_daily
        GROUP BY 1, 2, 3, 4, 5, 6, 7;
        """
    )

    # --------------------------------------------------- category x month
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cat_national_monthly AS
        WITH mensual AS (
            SELECT
                m.categoria, m.categoria_label, m.grupo_inpc, m.mes,
                exp(sum(coalesce(p.peso_inpc, 0) * ln(m.precio_geo))
                    / nullif(sum(coalesce(p.peso_inpc, 0)), 0)) AS precio_geo,
                exp(avg(ln(m.precio_geo)))                      AS precio_geo_equal,
                sum(coalesce(p.peso_inpc, 0) * m.precio)
                    / nullif(sum(coalesce(p.peso_inpc, 0)), 0)  AS precio,
                avg(m.precio)                                   AS precio_equal,
                median(m.precio_geo)                            AS precio_mediana,
                count(*)                                        AS n_mercados,
                sum(m.n_obs)                                    AS n_obs,
                max(m.n_dias)                                   AS n_dias_max,
                round(avg(m.n_dias), 1)                         AS n_dias_prom
            FROM cat_market_monthly m
            LEFT JOIN pesos_mercado p ON p.destino = m.destino
            GROUP BY 1, 2, 3, 4
        ),
        anclaje AS (
            SELECT categoria,
                   max(precio)     FILTER (WHERE mes = DATE '{base_month}-01') AS precio_base,
                   max(precio_geo) FILTER (WHERE mes = DATE '{base_month}-01') AS precio_geo_base
            FROM mensual GROUP BY categoria
        )
        SELECT
            m.*,
            CASE WHEN a.precio_base > 0
                 THEN 100.0 * m.precio / a.precio_base END          AS indice_base100,
            CASE WHEN a.precio_geo_base > 0
                 THEN 100.0 * m.precio_geo / a.precio_geo_base END  AS indice_geo_base100,
            -- Date-based, not lag(): a missing month gives NULL rather than a
            -- number that is silently comparing the wrong span.
            m.precio     / p1.precio      - 1                       AS var_mensual,
            m.precio_geo / p1.precio_geo  - 1                       AS var_mensual_geo,
            m.precio     / p12.precio     - 1                       AS var_anual,
            m.precio_geo / p12.precio_geo - 1                       AS var_anual_geo
        FROM mensual m
        LEFT JOIN anclaje a USING (categoria)
        LEFT JOIN mensual p1
               ON p1.categoria = m.categoria AND p1.mes = m.mes - INTERVAL 1 MONTH
        LEFT JOIN mensual p12
               ON p12.categoria = m.categoria AND p12.mes = m.mes - INTERVAL 12 MONTH
        ORDER BY m.categoria, m.mes;
        """
    )

    # ------------------------------------------- chained Jevons index
    # Two-stage inside the index too: match variety x market cells and average
    # their relatives WITHIN a market first, then combine markets by weight. If
    # the market weight were applied per cell instead, a market reporting five
    # varieties would carry five times its intended weight.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cat_index_monthly AS
        WITH espina AS (
            SELECT categoria, categoria_label, grupo_inpc, mes,
                   lag(mes) OVER (PARTITION BY categoria ORDER BY mes) AS mes_prev
            FROM (SELECT DISTINCT categoria, categoria_label, grupo_inpc, mes
                  FROM var_market_monthly)
        ),
        -- stage 1: relatives of matched variety x market cells, within a market
        por_mercado AS (
            SELECT e.categoria, e.mes, a.destino,
                   exp(avg(ln(a.precio_geo / b.precio_geo))) AS factor_mercado,
                   count(*)                                  AS n_celdas
            FROM espina e
            JOIN var_market_monthly a
              ON a.categoria = e.categoria AND a.mes = e.mes
            JOIN var_market_monthly b
              ON b.categoria   = e.categoria
             AND b.mes         = e.mes_prev
             AND b.producto_id = a.producto_id
             AND b.destino     = a.destino
            WHERE e.mes_prev IS NOT NULL
              AND a.precio_geo > 0 AND isfinite(a.precio_geo)
              AND b.precio_geo > 0 AND isfinite(b.precio_geo)
            GROUP BY 1, 2, 3
        ),
        -- stage 2: combine markets, weighted and equal
        pareados AS (
            SELECT m.categoria, m.mes,
                   exp(sum(coalesce(p.peso_inpc, 0) * ln(m.factor_mercado))
                       / nullif(sum(coalesce(p.peso_inpc, 0)), 0)) AS factor,
                   exp(avg(ln(m.factor_mercado)))                  AS factor_equal,
                   sum(m.n_celdas)                                 AS n_celdas_pareadas,
                   count(*)                                        AS n_mercados_pareados
            FROM por_mercado m
            LEFT JOIN pesos_mercado p ON p.destino = m.destino
            GROUP BY 1, 2
        ),
        -- the previous behaviour: matched on variety only, markets averaged first
        pareados_var AS (
            SELECT e.categoria, e.mes,
                   exp(avg(ln(a.precio_geo_equal / b.precio_geo_equal))) AS factor,
                   count(*)                                             AS n_variedades
            FROM espina e
            JOIN var_national_monthly a
              ON a.categoria = e.categoria AND a.mes = e.mes
            JOIN var_national_monthly b
              ON b.categoria   = e.categoria
             AND b.mes         = e.mes_prev
             AND b.producto_id = a.producto_id
            WHERE e.mes_prev IS NOT NULL
              AND a.precio_geo_equal > 0 AND isfinite(a.precio_geo_equal)
              AND b.precio_geo_equal > 0 AND isfinite(b.precio_geo_equal)
            GROUP BY 1, 2
        ),
        cadena AS (
            SELECT
                e.categoria, e.categoria_label, e.grupo_inpc, e.mes, e.mes_prev,
                CASE WHEN e.mes_prev IS NULL THEN NULL
                     ELSE datediff('month', e.mes_prev, e.mes) END AS meses_puente,
                p.factor,
                p.factor_equal,
                pv.factor                                          AS factor_var,
                coalesce(p.n_celdas_pareadas, 0)                   AS n_celdas_pareadas,
                coalesce(p.n_mercados_pareados, 0)                 AS n_mercados_pareados,
                coalesce(pv.n_variedades, 0)                       AS n_variedades_pareadas,
                coalesce(p.factor, 1.0)                            AS f_eff,
                coalesce(p.factor_equal, 1.0)                      AS f_eff_equal,
                coalesce(pv.factor, 1.0)                           AS f_eff_var,
                (e.mes_prev IS NOT NULL AND p.factor IS NULL)      AS cadena_rota
            FROM espina e
            LEFT JOIN pareados     p  USING (categoria, mes)
            LEFT JOIN pareados_var pv USING (categoria, mes)
        ),
        niveles AS (
            SELECT *,
                exp(sum(ln(f_eff))       OVER w) AS nivel,
                exp(sum(ln(f_eff_equal)) OVER w) AS nivel_equal,
                exp(sum(ln(f_eff_var))   OVER w) AS nivel_var
            FROM cadena
            WINDOW w AS (PARTITION BY categoria ORDER BY mes)
        ),
        anclaje AS (
            SELECT categoria,
                coalesce(max(nivel)       FILTER (WHERE mes = DATE '{base_month}-01'),
                         arg_min(nivel, mes))       AS base,
                coalesce(max(nivel_equal) FILTER (WHERE mes = DATE '{base_month}-01'),
                         arg_min(nivel_equal, mes)) AS base_equal,
                coalesce(max(nivel_var)   FILTER (WHERE mes = DATE '{base_month}-01'),
                         arg_min(nivel_var, mes))   AS base_var,
                coalesce(max(mes) FILTER (WHERE mes = DATE '{base_month}-01'),
                         min(mes))                  AS mes_base,
                count(*) FILTER (WHERE mes = DATE '{base_month}-01') = 0 AS base_es_fallback
            FROM niveles GROUP BY categoria
        )
        SELECT
            n.categoria, n.categoria_label, n.grupo_inpc, n.mes,
            n.meses_puente,
            n.n_celdas_pareadas, n.n_mercados_pareados, n.n_variedades_pareadas,
            n.cadena_rota,
            an.mes_base, an.base_es_fallback,
            100.0 * n.nivel       / an.base       AS indice_jevons,
            100.0 * n.nivel_equal / an.base_equal AS indice_jevons_equal,
            100.0 * n.nivel_var   / an.base_var   AS indice_jevons_var,
            n.factor       - 1                    AS var_mensual,
            n.factor_equal - 1                    AS var_mensual_equal,
            n12.mes IS NOT NULL                   AS tiene_comparativo_anual,
            CASE WHEN n12.nivel > 0 THEN n.nivel / n12.nivel - 1 END AS var_anual
        FROM niveles n
        JOIN anclaje an USING (categoria)
        LEFT JOIN niveles n12
               ON n12.categoria = n.categoria AND n12.mes = n.mes - INTERVAL 12 MONTH
        ORDER BY n.categoria, n.mes;
        """
    )

    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in CURATED_TABLES}

    # Surface index-quality problems rather than leaving them in a column nobody reads.
    roto = con.execute("SELECT count(*) FROM cat_index_monthly WHERE cadena_rota").fetchone()[0]
    if roto:
        log.warning(
            "%d category-months have a broken index chain (nothing present at both ends "
            "of the step) - the level is carried forward there. Filter cadena_rota.", roto)
    puente = con.execute(
        "SELECT count(*) FROM cat_index_monthly WHERE meses_puente > 1").fetchone()[0]
    if puente:
        log.info("%d index steps bridge a gap of more than one month (seasonal "
                 "categories) - see meses_puente.", puente)
    fallback = con.execute(
        "SELECT count(DISTINCT categoria) FROM cat_index_monthly "
        "WHERE base_es_fallback").fetchone()[0]
    if fallback:
        log.warning(
            "%d categories have no data in the configured base month %s and are based on "
            "their own first month instead - their index LEVELS are not comparable with "
            "the others. See mes_base / base_es_fallback.", fallback, base_month)

    # How much did the weighting and matching choices actually move things?
    div = con.execute(
        """
        SELECT round(100 * max(abs(indice_jevons / nullif(indice_jevons_equal, 0) - 1)), 1),
               round(100 * max(abs(indice_jevons / nullif(indice_jevons_var, 0) - 1)), 1)
        FROM cat_index_monthly
        """
    ).fetchone()
    if div and div[0] is not None:
        log.info("index sensitivity: INPC vs equal market weights differ by up to %.1f%%; "
                 "cell-matched vs variety-matched by up to %.1f%%", div[0], div[1])
    return counts
def export_curated(cfg, con: duckdb.DuckDBPyConnection, csv: bool = True) -> list[Path]:
    out: list[Path] = []
    cfg.curated_dir.mkdir(parents=True, exist_ok=True)
    for t in CURATED_TABLES:
        pq = cfg.curated_dir / f"{t}.parquet"
        con.execute(f"COPY {t} TO '{pq.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD);")
        out.append(pq)
        if csv and t in CSV_TABLES:
            c = cfg.curated_dir / f"{t}.csv"
            con.execute(f"COPY {t} TO '{c.as_posix()}' (HEADER, DELIMITER ',');")
            out.append(c)
    return out


def coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT m.categoria_label,
               min(m.mes)                        AS primer_mes,
               max(m.mes)                        AS ultimo_mes,
               count(*)                          AS meses,
               round(avg(m.n_mercados), 1)       AS mercados_prom,
               round(min(m.precio_geo), 2)       AS geo_min,
               round(max(m.precio_geo), 2)       AS geo_max,
               round(arg_max(i.indice_jevons, m.mes), 1) AS jevons_ult,
               sum(CASE WHEN i.cadena_rota THEN 1 ELSE 0 END) AS cadena_rota,
               sum(CASE WHEN i.meses_puente > 1 THEN 1 ELSE 0 END) AS pasos_puente
        FROM cat_national_monthly m
        LEFT JOIN cat_index_monthly i USING (categoria, mes)
        GROUP BY 1 ORDER BY 1;
        """
    ).fetchdf()
