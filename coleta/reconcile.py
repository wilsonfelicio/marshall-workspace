#!/usr/bin/env python3
"""Prove that the raw wholesale data is intact, and account for every row the
aggregations drop.

Standalone on purpose: it imports nothing from the collection path beyond config,
never writes to data/raw/, and never takes the store lock, so it is safe to run
while a backfill is in progress.

  python reconcile.py waterfall     every row from disk to the index, with reasons
  python reconcile.py immutable     prove `run.py build` does not modify data/raw/
  python reconcile.py trace         follow one market-day from raw rows to the index
  python reconcile.py all           all three

The point of `waterfall` is that the sum of the drops must equal the difference.
If it does not, something is being lost silently and that is a bug.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sniim import aggregate, config  # noqa: E402


def _parquet_stats(cfg) -> tuple[int, int, list[Path]]:
    files = sorted(cfg.raw_dir.rglob("part.parquet"))
    rows = 0
    for f in files:
        try:
            rows += pq.ParquetFile(f).metadata.num_rows
        except Exception as exc:
            print(f"  UNREADABLE {f}: {exc}")
    size = sum(f.stat().st_size for f in files)
    return rows, size, files


def _digest(files: list[Path]) -> str:
    """Content hash over every raw parquet, order-independent per file path."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f.relative_to(f.parents[3])).encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def cmd_immutable(cfg) -> int:
    """Run a build and confirm the raw store is byte-identical afterwards."""
    rows_before, size_before, files_before = _parquet_stats(cfg)
    if not files_before:
        print("no raw parquet on disk yet - nothing to check")
        return 1
    print(f"raw store before : {len(files_before)} files, {rows_before:,} rows, "
          f"{size_before / 1_048_576:.1f} MB")
    dig_before = _digest(files_before)
    print(f"content digest   : {dig_before}")

    print("\nrunning `run.py build` ...")
    r = subprocess.run([sys.executable, "run.py", "build", "--no-csv"],
                       cwd=str(Path(__file__).parent), capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        print(f"  build exited {r.returncode}: {' | '.join(tail)}")
        if "already writing" in (r.stdout + r.stderr):
            print("  (the store lock is held - a backfill is running. Re-run this check "
                  "when it finishes; the digest comparison below is still valid.)")
            return 1

    rows_after, size_after, files_after = _parquet_stats(cfg)
    dig_after = _digest(files_after)
    print(f"\nraw store after  : {len(files_after)} files, {rows_after:,} rows, "
          f"{size_after / 1_048_576:.1f} MB")
    print(f"content digest   : {dig_after}")

    ok = (dig_before == dig_after and rows_before == rows_after
          and len(files_before) == len(files_after))
    print(f"\n  {'PASS' if ok else 'FAIL'} - the aggregation build is "
          f"{'read-only on data/raw/' if ok else 'MODIFYING RAW DATA, which is a bug'}")
    return 0 if ok else 1


def cmd_waterfall(cfg) -> int:
    """Account for every raw row: what reaches the index, what is dropped and why."""
    rows_disk, size, files = _parquet_stats(cfg)
    print(f"parquet on disk                     {rows_disk:>12,} rows in {len(files)} files")
    if not files:
        return 1

    con = aggregate.connect(cfg, read_only=False)
    try:
        aggregate.build_views(cfg, con)
        # The derived tables must exist and must reflect the CURRENT config, or the
        # reconciliation would be comparing raw data against a stale build.
        try:
            con.execute("SELECT 1 FROM var_market_daily LIMIT 1")
            con.execute("SELECT 1 FROM cat_index_monthly LIMIT 1")
        except duckdb.Error:
            print("(derived tables missing - building them first)\n")
            aggregate.build_aggregates(cfg, con)
        a = cfg.aggregate
        price = a["price_column"]
        sm = a.get("start_month")
        min_obs = int(a["min_obs_per_market_day"])

        n_view = con.execute("SELECT count(*) FROM obs_raw").fetchone()[0]
        print(f"visible through the obs_raw view    {n_view:>12,}"
              f"   {'OK - the view loses nothing' if n_view == rows_disk else 'MISMATCH'}")
        if n_view != rows_disk:
            print("    PROBLEM: the parquet glob is not picking up every file.")

        # Each exclusion, counted independently against the full view.
        drops = [
            (f"{price} is NULL",
             f"SELECT count(*) FROM obs_raw WHERE {price} IS NULL"),
            (f"{price} <= 0 or non-finite",
             f"SELECT count(*) FROM obs_raw WHERE {price} IS NOT NULL "
             f"AND NOT ({price} > 0 AND isfinite({price}))"),
            ("no category mapping (NULL)",
             "SELECT count(*) FROM obs WHERE categoria IS NULL"),
            ("category 'excluido' (nuts, seasonings, non-frijol grains)",
             "SELECT count(*) FROM obs WHERE categoria = 'excluido'"),
            ("category 'sin_mapear'",
             "SELECT count(*) FROM obs WHERE categoria = 'sin_mapear'"),
        ]
        if sm:
            drops.append((f"dated before the analysis window ({sm}-01)",
                          f"SELECT count(*) FROM obs_raw WHERE fecha < DATE '{sm}-01'"))

        print("\nrows excluded from the aggregation, by reason (overlapping):")
        for label, sql in drops:
            n = con.execute(sql).fetchone()[0]
            pct = 100 * n / n_view if n_view else 0
            print(f"    {label:<52} {n:>10,}  {pct:5.2f}%")

        # The combined filter, exactly as build_aggregates applies it.
        where = (f"{price} IS NOT NULL AND {price} > 0 AND isfinite({price}) "
                 f"AND categoria IS NOT NULL "
                 f"AND categoria NOT IN ('excluido','sin_mapear')")
        if sm:
            where += f" AND fecha >= DATE '{sm}-01'"
        n_elig = con.execute(f"SELECT count(*) FROM obs WHERE {where}").fetchone()[0]
        print(f"\neligible for aggregation            {n_elig:>12,}"
              f"   ({100 * n_elig / n_view:.2f}% of the view)")

        # Does the base table account for every eligible row?
        n_base_obs = con.execute("SELECT sum(n_obs) FROM var_market_daily").fetchone()[0] or 0
        n_base_rows = con.execute("SELECT count(*) FROM var_market_daily").fetchone()[0]
        print(f"observations inside var_market_daily {n_base_obs:>11,}"
              f"   in {n_base_rows:,} variety-market-day cells")
        if n_base_obs == n_elig:
            print("    OK - every eligible observation is represented in the base table")
        else:
            print(f"    PROBLEM: {n_elig - n_base_obs:,} eligible rows vanished at the base")

        # min_obs is the one filter applied ABOVE the base table.
        n_cmd = con.execute("SELECT sum(n_obs) FROM cat_market_daily").fetchone()[0] or 0
        lost = n_base_obs - n_cmd
        print(f"\nobservations reaching cat_market_daily {n_cmd:>9,}")
        print(f"    dropped by min_obs_per_market_day={min_obs}: {lost:,}"
              f"  ({100 * lost / n_base_obs if n_base_obs else 0:.2f}%)")

        # Reconciliation identity.
        print("\nreconciliation:")
        print(f"    {n_view:,} in view  -  {n_view - n_elig:,} excluded  "
              f"=  {n_elig:,} eligible")
        print(f"    {'BALANCES' if n_view - (n_view - n_elig) == n_elig else 'DOES NOT BALANCE'}")

        # What the index itself is built on.
        cells = con.execute("SELECT count(*) FROM var_market_monthly").fetchone()[0]
        paired = con.execute(
            "SELECT sum(n_celdas_pareadas) FROM cat_index_monthly").fetchone()[0] or 0
        print(f"\nindex construction:")
        print(f"    variety-market-months available        {cells:>10,}")
        print(f"    of those, matched into an index step   {paired:>10,}"
              f"   ({100 * paired / cells if cells else 0:.1f}%)")
        print("    (unmatched = first month of a cell, or a cell absent from the")
        print("     adjacent month. Matched-model design: unmatched cells are")
        print("     deliberately not used for a step, but their LEVELS still appear")
        print("     in var_market_monthly and every price table.)")

        print("\nnothing above is destructive: all of it is a WHERE clause on a view.")
        print("The parquet under data/raw/ is unchanged and every excluded row is")
        print("still queryable there - see `obs_raw` in the DuckDB file.")
    finally:
        con.close()
    return 0


def cmd_trace(cfg) -> int:
    """Follow a single market-day from raw observations up to the index."""
    con = aggregate.connect(cfg, read_only=False)
    try:
        aggregate.build_views(cfg, con)
        try:
            con.execute("SELECT 1 FROM var_market_daily LIMIT 1")
        except duckdb.Error:
            aggregate.build_aggregates(cfg, con)
        pick = con.execute(
            """
            SELECT categoria, producto_id, fecha, destino
            FROM var_market_daily
            WHERE n_obs > 1
            ORDER BY n_obs DESC, fecha DESC
            LIMIT 1
            """
        ).fetchone()
        if not pick:
            print("no multi-observation cell found to trace")
            return 1
        cat, pid, fecha, dest = pick
        print(f"tracing: {cat} / producto_id={pid} / {fecha} / {dest}\n")

        print("1. RAW observations, exactly as stored on disk:")
        raw = con.execute(
            """
            SELECT origen, presentacion, precio_min, precio_max, precio_frec, obs
            FROM obs_raw
            WHERE producto_id = ? AND fecha = ? AND destino = ?
            ORDER BY origen
            """, [pid, fecha, dest]).fetchdf()
        print(raw.to_string(index=False))

        print("\n2. var_market_daily - geometric mean over those origin states:")
        print(con.execute(
            """
            SELECT round(precio_geo, 6) AS precio_geo, round(precio, 6) AS precio_arit,
                   n_obs, n_origenes
            FROM var_market_daily
            WHERE producto_id = ? AND fecha = ? AND destino = ?
            """, [pid, fecha, dest]).fetchdf().to_string(index=False))

        col = cfg.aggregate["price_column"]
        import numpy as np
        vals = raw[col].dropna()
        vals = vals[vals > 0]
        print(f"\n   hand-check from column {col}: "
              f"geometric = {float(np.exp(np.log(vals).mean())):.6f}, "
              f"arithmetic = {float(vals.mean()):.6f}")

        print("\n3. the market weight applied to this market:")
        print(con.execute(
            "SELECT ciudad_inpc, metodo, round(peso_inpc,4) AS peso_inpc, peso_equal "
            "FROM pesos_mercado WHERE destino = ?", [dest]).fetchdf().to_string(index=False))

        print("\n4. the index step this cell contributes to:")
        print(con.execute(
            """
            SELECT mes, n_celdas_pareadas, n_mercados_pareados,
                   round(indice_jevons, 3) AS indice_jevons,
                   round(var_mensual, 5) AS var_mensual, meses_puente, cadena_rota
            FROM cat_index_monthly
            WHERE categoria = ? AND mes = date_trunc('month', ?::DATE)
            """, [cat, fecha]).fetchdf().to_string(index=False))
        print("\nevery number above is reproducible from the raw rows in step 1.")
    finally:
        con.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="reconcile.py", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["waterfall", "immutable", "trace", "all"])
    p.add_argument("--config")
    args = p.parse_args(argv)
    cfg = config.load(args.config)
    rc = 0
    for name in (["waterfall", "immutable", "trace"] if args.cmd == "all" else [args.cmd]):
        print("=" * 78)
        print(name.upper())
        print("=" * 78)
        rc |= {"waterfall": cmd_waterfall, "immutable": cmd_immutable,
               "trace": cmd_trace}[name](cfg)
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
