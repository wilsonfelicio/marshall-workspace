#!/usr/bin/env python3
"""SNIIM price collection - single entrypoint.

Typical lifecycle
-----------------
  python run.py catalog                 # scrape product/market catalogs (once, then rarely)
  python run.py mapping                 # generate config/mapping_inpc.csv, review it
  python run.py backfill                # one-time history load, resumable
  python run.py build                   # DuckDB + category aggregates
  python run.py update                  # THE RECURRENT ONE - run daily
  python run.py verify                  # QA checks
  python run.py status                  # what is on disk, what is missing

`update` is idempotent: it re-fetches a trailing window and merges, so running
it twice, or after a missed day, changes nothing except filling gaps.

Manifest statuses
-----------------
Only `ok` and `empty` are terminal, i.e. skipped by --resume. Everything else is
retried on the next backfill, by design:

  ok              complete and closed period
  empty           the server returned a usable results page with no rows
  open            period not finished yet (current year / current week)
  truncated       response was paginated and could not be split further
  unusable        HTTP 200 but not a results page (error/maintenance page)
  range_mismatch  server did not honour the date range we asked for
  malformed       parser rejected rows - the table layout may have changed
  future          period is entirely in the future
  failed          network/HTTP error after all retries
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb  # noqa: E402

from sniim import aggregate, catalog, config, frutas, granos, mapping, store  # noqa: E402
from sniim.http import FetchError, Limiter, Session  # noqa: E402
from sniim.store import StoreError  # noqa: E402

NON_TERMINAL = ("open", "truncated", "unusable", "range_mismatch", "malformed", "future", "failed")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unidad(cfg) -> str:
    return "MXN/kg" if int(cfg.query["precios_por_id"]) == 2 else "MXN/presentacion"


def _resolve_products(cfg, modulo: str) -> list[int]:
    spec = cfg.frutas["products"] if modulo == "frutas" else cfg.granos["products"]
    if isinstance(spec, str):
        if spec == "all":
            return catalog.product_ids(cfg, modulo)
        if spec == "all_frijol":
            return catalog.frijol_ids(cfg)
        raise ValueError(f"unknown products spec {spec!r}")
    return [int(x) for x in spec]


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _frutas_status(meta: dict, rows: list, anio: int, today: date) -> tuple[str, str]:
    """Decide the manifest status for one product-year. Order matters."""
    if meta["unusable"]:
        return "unusable", meta["reason"]
    if meta["truncated"]:
        return "truncated", meta["reason"]
    if meta["range_mismatch"]:
        return "range_mismatch", meta["reason"]
    if meta["malformed"]:
        return "malformed", f"{meta['malformed']} rows rejected by the parser"
    if not frutas.year_is_closed(anio, today):
        # The key is the year, but only part of it exists. Recording this as
        # terminal would let --resume skip the rest of the year forever.
        return "open", "year still in progress"
    if not rows:
        return "empty", ""
    return "ok", "split" if meta.get("split") else ""


# --------------------------------------------------------------------- catalog
def cmd_catalog(cfg, args, log) -> int:
    session = Session(cfg)
    counts = catalog.refresh(cfg, session)
    log.info("catalogs refreshed: %s", counts)
    log.info(session.summary())
    return 0


# --------------------------------------------------------------------- mapping
def cmd_mapping(cfg, args, log) -> int:
    df = mapping.build(cfg)
    try:
        path = mapping.write(cfg, df, force=args.force)
        log.info("wrote %s (%d products)", path, len(df))
    except FileExistsError as exc:
        log.warning("%s", exc)
        log.info("showing the report for the mapping that would be generated:")
    print(mapping.report(df))
    return 0


# -------------------------------------------------------------------- backfill
# ------------------------------------------------------------------ concurrency
def _make_session_factory(cfg, limiter):
    """One Session per thread, all sharing one Limiter.

    requests.Session is not safe to share across threads, but the RATE is what the
    SNIIM server cares about, and the Limiter owns that globally. So: per-thread
    connection pools, one process-wide pacing decision.
    """
    local = threading.local()

    def get() -> Session:
        s = getattr(local, "s", None)
        if s is None:
            s = Session(cfg, limiter=limiter)
            local.s = s
        return s
    return get


def _pump(todo, fetch_one, persist, workers: int) -> None:
    """Run fetch_one over todo with `workers` threads, persisting IN ORDER in the
    main thread.

    In-order persistence costs a little wall clock (a slow job holds up the
    results behind it) and buys something worth more: the manifest is written in
    the same sequence as a serial run, so an interrupted parallel run resumes
    exactly like an interrupted serial one. The in-flight window is bounded at
    4x workers so a fast fetcher cannot build an unbounded queue of unwritten
    results in memory.
    """
    if workers <= 1:
        for i, job in enumerate(todo, start=1):
            persist(fetch_one(job), i)
        return

    from concurrent.futures import ThreadPoolExecutor

    window = max(2 * workers, 4)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sniim") as ex:
        pending, it, i = [], iter(todo), 0
        while True:
            while len(pending) < window:
                try:
                    pending.append(ex.submit(fetch_one, next(it)))
                except StopIteration:
                    break
            if not pending:
                break
            fut = pending.pop(0)
            i += 1
            persist(fut.result(), i)


def _collect_frutas(cfg, session, st, log, years: list[int], products: list[int],
                    resume: bool, today: date, labels: dict[int, str],
                    workers: int = 1, _session_for=None) -> dict:
    # A single Session is the norm; only the parallel backfill supplies a
    # per-thread factory. Defaulting here rather than at the call site is what
    # `update` needed: it passed no factory and the None was called as one.
    _sf = _session_for or (lambda: session)
    unidad = _unidad(cfg)
    jobs = [(p, y) for p in products for y in years]
    todo = [
        (p, y) for p, y in jobs
        if not (resume and st.manifest.is_complete(frutas.MODULO, p, str(y)))
    ]
    log.info(
        "frutas: %d product-years total, %d already complete, %d to fetch",
        len(jobs), len(jobs) - len(todo), len(todo),
    )

    stats = {"jobs": len(todo), "rows": 0, "requests": 0}
    by_status: dict[str, int] = {}
    t0 = time.monotonic()

    def fetch_one(job):
        """Runs in a worker thread. Touches only the HTTP session it was given."""
        pid, anio = job
        d0, d1 = frutas.year_bounds(anio, today)
        if d0 > today:
            return job, d0, d1, None, None, "future"
        try:
            rows, meta = frutas.fetch_range(cfg, _sf(), pid, d0, d1)
        except FetchError as exc:
            return job, d0, d1, None, None, exc
        return job, d0, d1, rows, meta, None

    def persist(res, i):
        """Runs in the MAIN thread only. The store and the manifest are not
        thread-safe, and the parquet writer takes a process-level lock."""
        (pid, anio), d0, d1, rows, meta, err = res
        if err == "future":
            st.manifest.record(frutas.MODULO, pid, str(anio), "future", note="not reached yet")
            by_status["future"] = by_status.get("future", 0) + 1
            return
        if err is not None:
            log.error("frutas product %s %s failed: %s", pid, anio, err)
            st.manifest.record(frutas.MODULO, pid, str(anio), "failed", note=str(err)[:200])
            by_status["failed"] = by_status.get("failed", 0) + 1
            return

        stats["requests"] += meta["requests"]
        status, note = _frutas_status(meta, rows, anio, today)

        # Always persist whatever rows we did get, whatever the status. A
        # truncated or open period still contains real observations, and the
        # non-terminal status guarantees it gets refetched and merged later.
        if rows:
            df = frutas.to_frame(
                rows, pid, meta["producto"] or labels.get(pid), meta["calidad"],
                str(anio), _now_iso(), unidad,
            )
            st.write_observations(df)
            stats["rows"] += len(rows)

        st.manifest.record(
            frutas.MODULO, pid, str(anio), status,
            rows=len(rows), pages=meta["pages_max"], malformed=meta["malformed"],
            rango_inicio=str(d0), rango_fin=str(d1), note=note,
        )
        by_status[status] = by_status.get(status, 0) + 1

        if i % 10 == 0 or i == len(todo):
            elapsed = time.monotonic() - t0
            rate = i / max(elapsed, 1e-9)
            log.info(
                "frutas %d/%d (%.0f%%) | %s rows | ETA %s | last: %s %s [%s]",
                i, len(todo), 100 * i / len(todo), f"{stats['rows']:,}",
                _fmt_eta((len(todo) - i) / rate), labels.get(pid, pid), anio, status,
            )

    _pump(todo, fetch_one, persist, workers)
    stats["by_status"] = by_status
    return stats


def _collect_granos(cfg, session, st, log, d0: date, d1: date, products: list[int],
                    resume: bool, labels: dict[int, str], today: date,
                    workers: int = 1, _session_for=None) -> dict:
    # A single Session is the norm; only the parallel backfill supplies a
    # per-thread factory. Defaulting here rather than at the call site is what
    # `update` needed: it passed no factory and the None was called as one.
    _sf = _session_for or (lambda: session)
    unidad = _unidad(cfg)
    slots = granos.weeks_in_range(d0, d1)
    jobs = [(p, y, m, s) for p in products for (y, m, s) in slots]
    todo = [
        j for j in jobs
        if not (resume and st.manifest.is_complete(
            granos.MODULO, j[0], granos.periodo_key(j[1], j[2], j[3])))
    ]
    log.info(
        "granos: %d product-weeks total, %d already complete, %d to fetch",
        len(jobs), len(jobs) - len(todo), len(todo),
    )

    stats = {"jobs": len(todo), "rows": 0, "requests": 0}
    by_status: dict[str, int] = {}
    t0 = time.monotonic()

    def fetch_one(job):
        pid, y, m, sl = job
        monday, _friday = granos.slot_dates(y, m, sl)
        if monday > today:
            return job, None, None, "future"
        try:
            rows, res = granos.fetch_week(cfg, _sf(), pid, y, m, sl)
        except FetchError as exc:
            return job, None, None, exc
        return job, rows, res, None

    def persist(out, i):
        (pid, y, m, sl), rows, res, err = out
        periodo = granos.periodo_key(y, m, sl)
        if err == "future":
            st.manifest.record(granos.MODULO, pid, periodo, "future", note="not reached yet")
            by_status["future"] = by_status.get("future", 0) + 1
            return
        if err is not None:
            log.error("granos product %s %s failed: %s", pid, periodo, err)
            st.manifest.record(granos.MODULO, pid, periodo, "failed", note=str(err)[:200])
            by_status["failed"] = by_status.get("failed", 0) + 1
            return

        stats["requests"] += 1
        closed = granos.week_is_closed(y, m, sl, today)

        if not res.usable:
            status, note = "unusable", res.rejected_reason or "not a results page"
        elif res.truncated:
            status, note = "truncated", f"paginated at {res.pages} pages"
        elif res.malformed_rows:
            status, note = "malformed", f"{res.malformed_rows} rows rejected"
        elif not closed:
            status, note = "open", "week still in progress"
        elif not rows:
            status, note = "empty", ""
        else:
            status, note = "ok", ""

        if rows:
            df = granos.to_frame(
                rows, pid, res.producto or labels.get(pid), periodo, _now_iso(), unidad
            )
            st.write_observations(df)
            stats["rows"] += len(rows)

        st.manifest.record(
            granos.MODULO, pid, periodo, status, rows=len(rows), pages=res.pages,
            malformed=res.malformed_rows,
            rango_inicio=str(res.rango_inicio or ""), rango_fin=str(res.rango_fin or ""),
            note=note,
        )
        by_status[status] = by_status.get(status, 0) + 1

        if i % 50 == 0 or i == len(todo):
            elapsed = time.monotonic() - t0
            rate = i / max(elapsed, 1e-9)
            log.info(
                "granos %d/%d (%.0f%%) | %s rows | ETA %s",
                i, len(todo), 100 * i / len(todo), f"{stats['rows']:,}",
                _fmt_eta((len(todo) - i) / rate),
            )

    _pump(todo, fetch_one, persist, workers)
    stats["by_status"] = by_status
    return stats


def cmd_backfill(cfg, args, log) -> int:
    today = date.today()
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    limiter = None
    if workers > 1:
        h = cfg.http
        limiter = Limiter(interval=float(h["min_interval_seconds"]),
                          max_concurrent=workers,
                          cool_off_factor=float(h["cool_off_factor"]),
                          cool_off_max=float(h["cool_off_max_seconds"]))
        log.info("modo paralelo: %d hilos, gap minimo compartido %.1fs entre peticiones; "
                 "baja a 1 hilo por si solo si suben los 5xx",
                 workers, float(h["min_interval_seconds"]))
    session = Session(cfg, limiter=limiter)
    session_for = _make_session_factory(cfg, limiter) if workers > 1 else (lambda: session)
    st = store.Store(cfg)
    rc = 0

    start_year = args.start_year or int(cfg.frutas["start_year"])
    end_year = args.end_year or today.year
    if end_year > today.year:
        log.warning("--end-year %s is in the future; clamping to %s", end_year, today.year)
        end_year = today.year
    years = list(range(start_year, end_year + 1))

    if args.module in ("frutas", "both") and cfg.frutas["enabled"]:
        products = _resolve_products(cfg, "frutas")
        if args.limit_products:
            products = products[: args.limit_products]
        labels = catalog.product_labels(cfg, "frutas")
        s = _collect_frutas(cfg, session, st, log, years, products,
                            not args.no_resume, today, labels,
                            workers=workers, _session_for=session_for)
        log.info("frutas done: %s rows, statuses %s", f"{s['rows']:,}", s["by_status"])
        if any(s["by_status"].get(k) for k in ("truncated", "unusable", "range_mismatch", "malformed")):
            rc = 1

    if args.module in ("granos", "both") and cfg.granos["enabled"]:
        gy0 = (getattr(args, "granos_start_year", None)
               or args.start_year or int(cfg.granos["start_year"]))
        gd0 = date(gy0, 1, 1)
        gd1 = min(today, date(end_year, 12, 31))
        products = _resolve_products(cfg, "granos")
        if args.limit_products:
            products = products[: args.limit_products]
        labels = catalog.product_labels(cfg, "granos")
        log.info("granos: desde %s (usa --granos-start-year para acortar)", gd0)
        s = _collect_granos(cfg, session, st, log, gd0, gd1, products,
                            not args.no_resume, labels, today,
                            workers=workers, _session_for=session_for)
        log.info("granos done: %s rows, statuses %s", f"{s['rows']:,}", s["by_status"])
        if any(s["by_status"].get(k) for k in ("truncated", "unusable", "malformed")):
            rc = 1

    log.info(session.summary())
    log.info(session.limiter_summary())
    if rc:
        log.warning(
            "some jobs need attention - they are recorded with a non-terminal "
            "status and will be retried on the next backfill. Run `python run.py verify`."
        )
    return rc


# ---------------------------------------------------------------------- update
def cmd_update(cfg, args, log) -> int:
    """Incremental: re-fetch a trailing window and merge. Safe to run daily."""
    today = date.today()
    lookback = args.days or int(cfg.update["lookback_days"])
    d0 = today - timedelta(days=lookback)
    session = Session(cfg)
    st = store.Store(cfg)
    unidad = _unidad(cfg)

    log.info("incremental update, window %s .. %s (%d days)", d0, today, lookback)

    total_rows = 0
    problems = 0

    if args.module in ("frutas", "both") and cfg.frutas["enabled"]:
        products = _resolve_products(cfg, "frutas")
        labels = catalog.product_labels(cfg, "frutas")
        # Ledger key is the window, not the year, so an update can never mark a
        # calendar year complete for --resume.
        periodo = f"upd:{d0}..{today}"
        ok = empty = failed = 0
        t0 = time.monotonic()
        for i, pid in enumerate(products, start=1):
            try:
                rows, meta = frutas.fetch_range(cfg, session, pid, d0, today)
            except FetchError as exc:
                log.error("update frutas %s failed: %s", pid, exc)
                st.manifest.record(frutas.MODULO, pid, periodo, "failed", note=str(exc)[:200])
                failed += 1
                problems += 1
                continue

            if meta["unusable"] or meta["truncated"] or meta["malformed"]:
                reason = meta["reason"] or f"{meta['malformed']} malformed rows"
                log.error("update frutas %s: %s", pid, reason)
                st.manifest.record(
                    frutas.MODULO, pid, periodo,
                    "unusable" if meta["unusable"] else
                    ("truncated" if meta["truncated"] else "malformed"),
                    rows=len(rows), malformed=meta["malformed"], note=reason[:200],
                )
                problems += 1

            if not rows:
                empty += 1
                continue

            df = frutas.to_frame(
                rows, pid, meta["producto"] or labels.get(pid), meta["calidad"],
                periodo, _now_iso(), unidad,
            )
            st.write_observations(df)
            ok += 1
            total_rows += len(rows)
            if i % 25 == 0 or i == len(products):
                rate = i / max(1e-9, time.monotonic() - t0)
                log.info(
                    "update frutas %d/%d | %s rows | ETA %s",
                    i, len(products), f"{total_rows:,}",
                    _fmt_eta((len(products) - i) / rate),
                )
        log.info("frutas window: %d with data, %d empty, %d failed", ok, empty, failed)

    if args.module in ("granos", "both") and cfg.granos["enabled"]:
        weeks_back = int(cfg.update["granos_lookback_weeks"])
        gd0 = today - timedelta(weeks=weeks_back)
        products = _resolve_products(cfg, "granos")
        labels = catalog.product_labels(cfg, "granos")
        # Reuse the backfill collector so status logic (open vs ok vs empty) is
        # identical - no second implementation to drift out of sync.
        s = _collect_granos(cfg, session, st, log, gd0, today, products,
                            False, labels, today)
        total_rows += s["rows"]
        problems += sum(
            s["by_status"].get(k, 0)
            for k in ("failed", "unusable", "truncated", "malformed")
        )
        log.info("granos window: %s rows, statuses %s", f"{s['rows']:,}", s["by_status"])

    log.info("update merged %s rows into the store", f"{total_rows:,}")
    log.info(session.summary())

    if not args.no_build:
        cmd_build(cfg, args, log)

    if problems:
        log.error(
            "%d products had problems this run - see logs/update.log. "
            "Exiting non-zero so the scheduler surfaces it.", problems
        )
        return 1
    return 0


# ----------------------------------------------------------------------- build
def cmd_build(cfg, args, log) -> int:
    con = aggregate.connect(cfg)
    try:
        n = aggregate.build_views(cfg, con)
        log.info("obs_raw: %s observations", f"{n:,}")
        counts = aggregate.build_aggregates(cfg, con)
        for t, c in counts.items():
            log.info("%-22s %s rows", t, f"{c:,}")
        paths = aggregate.export_curated(cfg, con, csv=not getattr(args, "no_csv", False))
        log.info("exported %d curated files to %s", len(paths), cfg.curated_dir)
        cov = aggregate.coverage(con)
        if not cov.empty:
            print()
            print(cov.to_string(index=False))
    finally:
        con.close()
    return 0


# ---------------------------------------------------------------------- verify
def cmd_verify(cfg, args, log) -> int:
    """QA checks. Exits non-zero if anything looks structurally wrong."""
    problems: list[str] = []
    warnings: list[str] = []

    st = store.Store(cfg)
    man = st.manifest.as_frame()

    for f in st.stray_temp_files():
        warnings.append(f"stray temp file {f} - a writer was killed mid-write")
    for f in st.corrupt_files():
        problems.append(
            f"quarantined unreadable part file {f} - refetch that product-year"
        )

    if man.empty:
        problems.append("manifest is empty - nothing has been collected")
    else:
        print(f"manifest: {len(man)} jobs")
        counts = man["status"].value_counts()
        print(counts.to_string())
        for status in NON_TERMINAL:
            n = int(counts.get(status, 0))
            if not n:
                continue
            if status in ("open", "future"):
                warnings.append(f"{n} jobs are {status} (expected; refetched automatically)")
            elif status == "failed":
                warnings.append(f"{n} jobs failed - rerun `python run.py backfill` to retry")
            else:
                problems.append(
                    f"{n} jobs recorded '{status}' - these indicate the site or the "
                    f"parser changed. Inspect the note column in data/manifest.csv"
                )
        if "malformed" in man.columns:
            mal = man["malformed"].astype(str).str.strip().replace("", "0").astype(float).sum()
            if mal:
                problems.append(
                    f"{int(mal)} data rows were rejected by the parser across all jobs - "
                    "the results table layout has probably changed"
                )

    con = aggregate.connect(cfg)
    try:
        try:
            aggregate.build_views(cfg, con)
        except RuntimeError as exc:
            problems.append(str(exc))
            return _report(problems, warnings)

        n_obs = con.execute("SELECT count(*) FROM obs_raw").fetchone()[0]
        checks = {
            "observations": "SELECT count(*) FROM obs_raw",
            "distinct products": "SELECT count(DISTINCT producto_id) FROM obs_raw",
            "distinct markets": "SELECT count(DISTINCT destino) FROM obs_raw",
            "date range": "SELECT min(fecha) || ' .. ' || max(fecha) FROM obs_raw",
            "unmapped rows": "SELECT count(*) FROM obs WHERE categoria IS NULL",
            "rows w/o precio_frec": "SELECT count(*) FROM obs_raw WHERE precio_frec IS NULL",
            "duplicate natural keys": """
                SELECT count(*) FROM (
                  SELECT modulo, producto_id, fecha, presentacion, origen, destino,
                         unidad, obs, count(*) c
                  FROM obs_raw GROUP BY 1,2,3,4,5,6,7,8 HAVING c > 1)
            """,
            "same key, different price": """
                SELECT count(*) FROM (
                  SELECT modulo, producto_id, fecha, presentacion, origen, destino
                  FROM obs_raw GROUP BY 1,2,3,4,5,6
                  HAVING count(DISTINCT precio_frec) > 1)
            """,
            "min > max violations": """
                SELECT count(*) FROM obs_raw
                WHERE precio_min IS NOT NULL AND precio_max IS NOT NULL
                  AND precio_min > precio_max
            """,
            "frec outside [min,max]": """
                SELECT count(*) FROM obs_raw
                WHERE precio_frec IS NOT NULL AND precio_min IS NOT NULL
                  AND precio_max IS NOT NULL
                  AND (precio_frec < precio_min - 0.011 OR precio_frec > precio_max + 0.011)
            """,
            "non-finite prices": """
                SELECT count(*) FROM obs_raw
                WHERE precio_frec IS NOT NULL AND NOT isfinite(precio_frec)
            """,
            "future dates": "SELECT count(*) FROM obs_raw WHERE fecha > current_date",
            "implausible prices (>2000/kg)": "SELECT count(*) FROM obs_raw WHERE precio_frec > 2000",
        }
        print()
        for label, sql in checks.items():
            val = con.execute(sql).fetchone()[0]
            print(f"  {label:<32} {val}")
            if label == "duplicate natural keys" and val:
                problems.append(f"{val} duplicate natural keys - dedupe logic is broken")
            if label == "min > max violations" and val:
                problems.append(f"{val} rows with precio_min > precio_max")
            if label == "non-finite prices" and val:
                problems.append(f"{val} non-finite prices reached the store")
            if label == "future dates" and val:
                warnings.append(f"{val} rows dated in the future")
            if label == "unmapped rows" and val:
                warnings.append(f"{val} observations lack a category mapping")
            if label == "rows w/o precio_frec" and n_obs:
                if val == n_obs:
                    problems.append(
                        "EVERY row has a NULL precio_frec - the price column was almost "
                        "certainly renamed on the site. Check parse._COLMAP."
                    )
                elif val > 0.5 * n_obs:
                    problems.append(
                        f"{val} of {n_obs} rows ({100*val/n_obs:.0f}%) have no precio_frec"
                    )
            if label == "same key, different price" and val:
                warnings.append(
                    f"{val} (fecha, presentacion, origen, destino) groups carry more than "
                    "one price - distinguished only by the obs column, which is in the "
                    "dedupe key, so nothing was lost. Informational."
                )
            if label == "implausible prices (>2000/kg)" and val:
                warnings.append(f"{val} rows above 2000 MXN/kg - inspect before trusting")

        # Manifest vs disk: a job recorded ok with rows must have left rows behind.
        disk = st.row_counts()
        if not man.empty and not disk.empty:
            claimed = man[man["status"] == "ok"].copy()
            claimed["rows"] = claimed["rows"].astype(str).str.strip().replace("", "0").astype(float)
            frutas_claim = claimed[
                (claimed["modulo"] == "frutas") & (claimed["rows"] > 0)
            ]["producto_id"].astype(int).nunique()
            frutas_disk = disk[disk["modulo"] == "frutas"]["producto_id"].nunique()
            print(f"\n  products recorded ok w/ rows      {frutas_claim}")
            print(f"  products present on disk          {frutas_disk}")
            if frutas_claim > frutas_disk:
                problems.append(
                    f"manifest claims rows for {frutas_claim} products but only "
                    f"{frutas_disk} exist on disk - parquet files are missing"
                )

        # Category coverage against the 32 requested.
        have = {
            r[0] for r in con.execute(
                "SELECT DISTINCT categoria FROM obs WHERE categoria IS NOT NULL"
            ).fetchall()
        }
        want = {slug for slug, _, _ in mapping.CATEGORIES}
        missing = sorted(want - have)
        print()
        print(f"  categories with data             {len(want & have)}/32")
        if missing:
            warnings.append("no data yet for: " + ", ".join(missing))

        # Index quality, if aggregates have been built.
        try:
            roto, puente, fb, tot = con.execute(
                """
                SELECT sum(CASE WHEN cadena_rota THEN 1 ELSE 0 END),
                       sum(CASE WHEN meses_puente > 1 THEN 1 ELSE 0 END),
                       count(DISTINCT CASE WHEN base_es_fallback THEN categoria END),
                       count(*)
                FROM cat_index_monthly
                """
            ).fetchone()
            print(f"  index months                     {tot}")
            print(f"    broken chain steps             {roto}")
            print(f"    gap-bridging steps (>1 month)  {puente}")
            print(f"    categories on a fallback base  {fb}")
            if roto:
                warnings.append(
                    f"{roto} index steps had no variety present at both ends; the level "
                    "is carried forward there. Filter cadena_rota in cat_index_monthly."
                )
            if fb:
                warnings.append(
                    f"{fb} categories have no data in the configured base month, so their "
                    "index LEVELS are not comparable with the others (growth rates are). "
                    "See mes_base / base_es_fallback."
                )
        except duckdb.Error:  # aggregates not built yet
            warnings.append("aggregates not built yet - run `python run.py build`")

        # Continuity: months with no observations at all.
        gaps = con.execute(
            """
            WITH m AS (SELECT DISTINCT date_trunc('month', fecha) AS mes FROM obs_raw),
                 span AS (SELECT min(mes) a, max(mes) b FROM m),
                 todos AS (SELECT unnest(generate_series(
                     (SELECT a FROM span), (SELECT b FROM span), INTERVAL 1 MONTH)) AS mes)
            SELECT count(*) FROM todos t LEFT JOIN m USING (mes) WHERE m.mes IS NULL
            """
        ).fetchone()[0]
        print(f"  months with zero observations    {gaps}")
        if gaps:
            warnings.append(f"{gaps} calendar months inside the span have no data at all")
    finally:
        con.close()

    return _report(problems, warnings)


def _report(problems: list[str], warnings: list[str]) -> int:
    print()
    for w in warnings:
        print(f"  WARNING  {w}")
    for p in problems:
        print(f"  PROBLEM  {p}")
    if not problems and not warnings:
        print("  all checks clean")
    return 1 if problems else 0


# ---------------------------------------------------------------------- status
def cmd_status(cfg, args, log) -> int:
    st = store.Store(cfg)
    counts = st.row_counts()
    if counts.empty:
        print("no data on disk yet")
    else:
        print("rows on disk by module and year:")
        piv = counts.pivot_table(
            index="anio", columns="modulo", values="rows", aggfunc="sum"
        ).fillna(0).astype(int)
        print(piv.to_string())
        print(f"\ntotal: {counts['rows'].sum():,} rows in {len(counts)} parquet files")
        size = sum(f.stat().st_size for f in cfg.raw_dir.rglob("*.parquet"))
        print(f"raw store size: {size / 1_048_576:.1f} MB")

    man = st.manifest.as_frame()
    if not man.empty:
        print("\njob manifest by status:")
        print(man.groupby(["modulo", "status"]).size().to_string())
        stuck = man[man["status"].isin(NON_TERMINAL) & (man["status"] != "future")]
        if len(stuck):
            print(f"\n{len(stuck)} jobs will be retried on the next backfill:")
            print(stuck.groupby("status").size().to_string())

    today = date.today()
    try:
        for modulo in ("frutas", "granos"):
            if not cfg.raw[modulo]["enabled"]:
                continue
            products = _resolve_products(cfg, modulo)
            if modulo == "frutas":
                years = range(int(cfg.frutas["start_year"]), today.year + 1)
                jobs = [(p, str(y)) for p in products for y in years]
            else:
                slots = granos.weeks_in_range(
                    date(int(cfg.granos["start_year"]), 1, 1), today
                )
                jobs = [
                    (p, granos.periodo_key(y, m, s)) for p in products for (y, m, s) in slots
                ]
            done = sum(1 for p, k in jobs if st.manifest.is_complete(modulo, p, k))
            print(
                f"\n{modulo}: {done}/{len(jobs)} jobs complete "
                f"({100 * done / max(1, len(jobs)):.1f}%), {len(jobs) - done} remaining"
            )
    except FileNotFoundError as exc:
        print(f"\n(cannot compute outstanding jobs: {exc})")

    return 0


# ---------------------------------------------------------------------- export
def cmd_export(cfg, args, log) -> int:
    """Dump a category's monthly index series to CSV for quick inspection."""
    con = aggregate.connect(cfg, read_only=True)
    try:
        sql = """
            SELECT i.categoria, i.categoria_label, i.mes,
                   i.indice_jevons, i.var_mensual, i.var_anual,
                   i.meses_puente, i.n_variedades_pareadas, i.cadena_rota,
                   m.precio, m.precio_geo, m.n_mercados
            FROM cat_index_monthly i
            LEFT JOIN cat_national_monthly m USING (categoria, mes)
        """
        params: list = []
        if args.categoria:
            sql += " WHERE i.categoria = ?"
            params.append(args.categoria)
        sql += " ORDER BY i.categoria, i.mes"
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()
    out = Path(args.out) if args.out else cfg.curated_dir / "export.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")
    if not df.empty:
        print(df.tail(15).to_string(index=False))
    return 0


# ------------------------------------------------------------------------ main
NEEDS_LOCK = {"backfill", "update", "build"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="run.py", description="SNIIM Mexican wholesale price collector"
    )
    p.add_argument("--config", help="path to config.yml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("catalog", help="refresh product/origin/destination catalogs")

    m = sub.add_parser("mapping", help="generate the INPC category mapping CSV")
    m.add_argument("--force", action="store_true", help="overwrite an existing mapping file")

    b = sub.add_parser("backfill", help="one-time history load (resumable)")
    b.add_argument("--module", choices=["frutas", "granos", "both"], default="both")
    b.add_argument("--start-year", type=int)
    b.add_argument("--end-year", type=int)
    b.add_argument("--limit-products", type=int, help="first N products only (for testing)")
    b.add_argument("--no-resume", action="store_true", help="refetch even completed jobs")
    b.add_argument("--workers", type=int, default=1,
                   help="concurrent fetchers (default 1 = identical to the serial run). "
                        "All workers share one rate limiter and it drops back to 1 "
                        "automatically if the server starts returning 5xx.")
    b.add_argument("--granos-start-year", type=int,
                   help="start year for granos only, so the weekly module can be "
                        "shortened without touching the frutas history")

    u = sub.add_parser("update", help="incremental trailing-window refresh (run daily)")
    u.add_argument("--module", choices=["frutas", "granos", "both"], default="both")
    u.add_argument("--days", type=int, help="override lookback_days")
    u.add_argument("--no-build", action="store_true", help="skip the aggregate rebuild")
    u.add_argument("--no-csv", action="store_true")

    bd = sub.add_parser("build", help="rebuild DuckDB views and category aggregates")
    bd.add_argument("--no-csv", action="store_true", help="parquet only, skip CSV export")

    sub.add_parser("verify", help="QA checks on the collected data")
    sub.add_parser("status", help="what is on disk and what is outstanding")

    e = sub.add_parser("export", help="dump a monthly index series to CSV")
    e.add_argument("--categoria", help="category slug, e.g. aguacate")
    e.add_argument("--out")

    args = p.parse_args(argv)
    cfg = config.load(args.config)
    log = config.setup_logging(cfg, args.cmd, args.verbose)

    handlers = {
        "catalog": cmd_catalog,
        "mapping": cmd_mapping,
        "backfill": cmd_backfill,
        "update": cmd_update,
        "build": cmd_build,
        "verify": cmd_verify,
        "status": cmd_status,
        "export": cmd_export,
    }
    t0 = time.monotonic()
    try:
        if args.cmd in NEEDS_LOCK:
            with store.lock(cfg):
                rc = handlers[args.cmd](cfg, args, log)
        else:
            rc = handlers[args.cmd](cfg, args, log)
    except KeyboardInterrupt:
        log.warning("interrupted - progress is recorded, rerun to resume")
        return 130
    except StoreError as exc:
        log.error("%s", exc)
        return 2
    log.info("%s finished in %s", args.cmd, _fmt_eta(time.monotonic() - t0))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
