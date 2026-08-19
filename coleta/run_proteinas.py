"""Collector for the eight protein generics, from SNIIM Pecuarios and Pesqueros.

  python3 run_proteinas.py [--from-year 2024] [--to-year 2026]
  python3 run_proteinas.py --update          # the daily job: this month and last

Writes data/raw/proteinas/<serie>/anio=YYYY/part.parquet and appends to
data/manifest_proteinas.csv. Resumable: a (serie, year, month) already marked ok in the
manifest is skipped unless --no-resume.

--update exists because that resume rule is exactly wrong for a daily refresh. The unit of
work is a MONTH, so the moment the current month is fetched once it is marked ok and would
never be fetched again — the panel would freeze on the day of the first run and keep
reporting success. --update ignores the manifest and refetches the current and previous
month for every series. Writes are read-merge-dedupe on (fecha, serie, producto, variante,
atributo, destino, origen), so refetching a month converges instead of duplicating, and
the previous month is included because a month boundary otherwise loses whatever the
source revised after the last run of the old month.

The job grid is serie x month, not product x year as in the produce collector, because
these endpoints accept destino=0 ("Todos") and return every market for a whole month in
one response. Eight series over 1998-2026 is roughly 2,800 requests.

Mapping to the INPC generics, decided with the data in front of us rather than by name:

  022 Pollo              Ent.asp                     pollo entero + rosticero
  017 Carne de cerdo     Can.asp?Var=Por (canal)  +  Cor.asp?Var=Por (cortes)
  043 Manteca de cerdo   Sub.asp?Var=Por             the "Grasa" column, a PROXY: raw fat
                                                     at the rastro, not rendered lard
  018 Carne de res       Can.asp?Var=Bov (canal)  +  Cor.asp?Var=Bov (cortes)
  025 Vísceras de res    Sub.asp?Var=Bov             the "Vísceras" column, a direct match
  027 Camarón            destino-res1.asp tipo=CL    shrimp by grade, several markets
  028 Pescado            destino-res1.asp tipo=FIL   fillets, incl. marine species
                         + tipo=PAD                  freshwater
  031 Huevo              Hue.asp                     blanco + rojo, 43 markets

Rate limiting reuses the produce collector's floor (config http.min_interval_seconds),
because it is the same server and the same IIS that returns 503 when pushed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import pathlib
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sniim import config, pecuarios as P  # noqa: E402

log = logging.getLogger("proteinas")
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "proteinas"
MANIFEST = ROOT / "data" / "manifest_proteinas.csv"
HOST = "http://www.economia-sniim.gob.mx/"
PEC = HOST + "SNIIM-Pecuarios-Nacionales/"
PESCA = HOST + "SNIIM-PESCA/"

# hidden form fields the Pesca "Destino" form posts, one set per category
PESCA_HID = {"CL": {"T": "C", "T1": "C"}, "MO": {"T": "o", "T1": "of"},
             "PAD": {"T": "a", "T1": "ad"}, "FIL": {"T": "f", "T1": "pf", "T2": "of"}}

SERIES = {
    "pollo":          {"kind": "get", "page": "Ent.asp",
                       "q": {"prod": "0", "origen": "0", "destino": "0"}},
    "huevo":          {"kind": "get", "page": "Hue.asp",
                       "q": {"prod": "0", "destino": "0", "sem": "0"}, "no_days": True},
    "res_canal":      {"kind": "get", "page": "Can.asp", "q": {"Var": "Bov", "destino": "0"}},
    "res_cortes":     {"kind": "get", "page": "Cor.asp",
                       "q": {"Var": "Bov", "origen": "0", "destino": "0"}},
    "res_visceras":   {"kind": "get", "page": "Sub.asp", "q": {"Var": "Bov", "destino": "0"}},
    "cerdo_canal":    {"kind": "get", "page": "Can.asp", "q": {"Var": "Por", "destino": "0"}},
    "cerdo_cortes":   {"kind": "get", "page": "Cor.asp",
                       "q": {"Var": "Por", "origen": "0", "destino": "0"}},
    "cerdo_grasa":    {"kind": "get", "page": "Sub.asp", "q": {"Var": "Por", "destino": "0"}},
    "camaron":        {"kind": "pesca", "tipo": "CL"},
    "pescado_filete": {"kind": "pesca", "tipo": "FIL"},
    "pescado_dulce":  {"kind": "pesca", "tipo": "PAD"},
}


# Series that are known to return nothing, so that an empty answer from them is not news.
# SNIIM's porcinos "cortes" endpoint has returned an empty table for every month tried,
# across the whole 2024-2026 range; it is left in SERIES so that a fix upstream is picked
# up automatically, but it must not colour the daily run red, or a real failure somewhere
# else stops being visible.
KNOWN_EMPTY = {"cerdo_cortes"}


def month_days(y: int, m: int) -> int:
    return (dt.date(y + (m == 12), (m % 12) + 1, 1) - dt.timedelta(days=1)).day


class Fetcher:
    def __init__(self, cfg):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = cfg.http["user_agent"]
        self.gap = float(cfg.http["min_interval_seconds"])
        self.timeout = int(cfg.http.get("timeout_seconds", 300))
        self.last = 0.0
        self.n = 0

    def _wait(self):
        d = self.gap - (time.monotonic() - self.last)
        if d > 0:
            time.sleep(d)
        self.last = time.monotonic()

    def fetch(self, serie: str, y: int, m: int, d0: int = 1, d1: int | None = None) -> str:
        spec = SERIES[serie]
        last = d1 if d1 is not None else month_days(y, m)
        first = d0
        for attempt in range(4):
            self._wait()
            self.n += 1
            try:
                if spec["kind"] == "get":
                    q = dict(spec["q"])
                    q.update({"mes": m, "anio": y, "RegPag": 1000})
                    if not spec.get("no_days"):
                        q.update({"del": first, "al": last})
                    r = self.s.get(PEC + spec["page"], params=q, timeout=self.timeout)
                else:
                    d = dict(PESCA_HID[spec["tipo"]])
                    d.update({"prod": "0", "ori": "0", "dest": "0",
                              "dia1": f"{first:02d}", "dia2": f"{last:02d}",
                              "mes": f"{m:02d}", "anio": str(y), "RegPag": "1000"})
                    r = self.s.post(PESCA + "destino-res1.asp", data=d,
                                    timeout=self.timeout,
                                    headers={"Referer": PESCA + "destino1.ASP?tipo="
                                             + spec["tipo"]})
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.encoding = r.apparent_encoding or "latin-1"
                return r.text
            except Exception as exc:                       # noqa: BLE001
                wait = 4 * (attempt + 1)
                log.warning("%s %04d-%02d attempt %d: %s (sleep %ds)",
                            serie, y, m, attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"{serie} {y}-{m} failed after 4 attempts")


def done_jobs() -> set:
    if not MANIFEST.exists():
        return set()
    with MANIFEST.open() as fh:
        return {(r["serie"], int(r["anio"]), int(r["mes"]))
                for r in csv.DictReader(fh) if r.get("status") == "ok"}


def record(serie, y, m, status, rows, note=""):
    new = not MANIFEST.exists()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["serie", "anio", "mes", "status", "rows", "fetched_at", "note"])
        w.writerow([serie, y, m, status, rows,
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    note[:200]])


def write(serie: str, y: int, df: pd.DataFrame):
    """Read-merge-dedupe-write, so refetching a month converges instead of duplicating."""
    d = OUT / serie / f"anio={y}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "part.parquet"
    if f.exists():
        df = pd.concat([pd.read_parquet(f), df], ignore_index=True)
    key = ["fecha", "serie", "producto", "variante", "atributo", "destino", "origen"]
    key = [k for k in key if k in df.columns]
    df = df.drop_duplicates(subset=key, keep="last").sort_values(key)
    df.to_parquet(f, index=False)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2024)
    ap.add_argument("--to-year", type=int, default=dt.date.today().year)
    ap.add_argument("--series", nargs="*", choices=sorted(SERIES), default=sorted(SERIES))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--update", action="store_true",
                    help="refetch the current and previous month for every series, "
                         "ignoring the manifest")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = config.load(None)
    fx = Fetcher(cfg)
    today = dt.date.today()
    if a.update:
        prev = today.replace(day=1) - dt.timedelta(days=1)
        months = sorted({(prev.year, prev.month), (today.year, today.month)})
        jobs = [(s, y, m) for s in a.series for y, m in months]
        log.info("update: %d jobs over %s", len(jobs),
                 ", ".join(f"{y}-{m:02d}" for y, m in months))
    else:
        skip = set() if a.no_resume else done_jobs()
        jobs = [(s, y, m) for s in a.series
                for y in range(a.from_year, a.to_year + 1)
                for m in range(1, 13)
                if not (y == today.year and m > today.month) and (s, y, m) not in skip]
        log.info("%d jobs (%d already done)", len(jobs), len(skip))

    t0 = time.monotonic()
    total, bad = 0, []
    for i, (serie, y, m) in enumerate(jobs, 1):
        try:
            html = fx.fetch(serie, y, m)
        except Exception as exc:                            # noqa: BLE001
            log.error("%s %04d-%02d FAILED: %s", serie, y, m, exc)
            record(serie, y, m, "failed", 0, str(exc))
            bad.append((serie, y, m))
            continue
        recs, meta = P.parse(html)
        # A month that still paginates at RegPag=1000 is losing rows. Camarón and the
        # cortes tables do this on 62 of 64 months, so split the month into day windows
        # until each answer fits on one page — the same halving the produce collector
        # applies to a truncated year.
        if meta["pages"] > 1:
            recs, pages_left = [], 0
            last = month_days(y, m)
            for lo, hi in ((1, 8), (9, 15), (16, 23), (24, last)):
                try:
                    sub = P.parse(fx.fetch(serie, y, m, lo, hi))
                except Exception as exc:                    # noqa: BLE001
                    log.error("%s %04d-%02d %d-%d FAILED: %s", serie, y, m, lo, hi, exc)
                    continue
                recs.extend(sub[0])
                pages_left = max(pages_left, sub[1]["pages"])
            meta = {**meta, "pages": pages_left}
        df = P.tidy(recs, serie)
        if not len(df):
            record(serie, y, m, "empty", 0, "no rows")
            if serie not in KNOWN_EMPTY:
                bad.append((serie, y, m))
        else:
            n = write(serie, y, df)
            total += len(df)
            status = "ok" if meta["pages"] <= 1 else "truncated"
            record(serie, y, m, status, len(df),
                   "" if status == "ok" else f"still paginated at {meta['pages']} pages")
        if i % 20 == 0 or i == len(jobs):
            rate = i / max(1e-9, time.monotonic() - t0)
            log.info("%d/%d  %s rows  ETA %.0f min", i, len(jobs), f"{total:,}",
                     (len(jobs) - i) / rate / 60)
    log.info("done: %d requests, %s rows written", fx.n, f"{total:,}")
    if bad:
        log.error("%d job(s) came back failed or empty: %s", len(bad),
                  ", ".join(f"{s} {y}-{m:02d}" for s, y, m in bad[:8]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
