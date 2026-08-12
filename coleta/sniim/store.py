"""Parquet store, resumable job manifest, and process locking.

Layout
------
data/raw/<modulo>/producto_id=<id>/anio=<yyyy>/part.parquet
data/catalog/<name>.csv
data/curated/*.parquet
data/manifest.csv          append-only job ledger (resume + audit)
data/.lock                 advisory lock, one writer at a time
data/sniim.duckdb          views + materialised tables over the parquet

Idempotency
-----------
Every write is a read-merge-dedupe-write on a single product-year file, so
refetching a period converges to the same content and a crashed run can simply
be restarted.

The dedupe key is the full natural key of an observation:
    (fecha, presentacion, origen, destino, unidad, obs)
keeping the most recently fetched copy, because SNIIM does revise recent rows.

`obs` and `unidad` are in the key deliberately. For the Granos module
`presentacion` is always NULL and the distinguishing text lives in `obs`
("Presentacion en bulto de 25 kg.", "NEGRO DE LA CASA"), so a shorter key would
collapse two genuinely different quoted prices into one and lose a price.
`unidad` is in the key so that flipping `query.precios_por_id` from 2 to 1 can
never silently overwrite pesos/kg rows with pesos/package rows.

Concurrency
-----------
Read-merge-write is not atomic across processes, so a `update` firing from cron
while a long `backfill` is still running would let one process clobber the
other's rows for the current year. `lock()` serialises writers, and the temp
file name includes the pid so two writers can never share it.
"""
from __future__ import annotations

import csv
import fcntl
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("sniim")

MANIFEST_FIELDS = [
    "modulo",
    "producto_id",
    "periodo",
    "status",
    "rows",
    "pages",
    "malformed",
    "rango_inicio",
    "rango_fin",
    "fetched_at",
    "note",
]

# Full natural key of one observation - see module docstring.
DEDUP_KEY = ["fecha", "presentacion", "origen", "destino", "unidad", "obs"]

# Statuses that mean "do not fetch this again". Everything else is retried.
TERMINAL_STATUSES = ("ok", "empty")

OBS_COLUMNS = [
    "modulo",
    "producto_id",
    "producto",
    "calidad",
    "grupo",
    "fecha",
    "presentacion",
    "origen",
    "destino",
    "destino_estado",
    "destino_mercado",
    "precio_min",
    "precio_max",
    "precio_frec",
    "unidad",
    "obs",
    "periodo",
    "fetched_at",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StoreError(RuntimeError):
    pass


# ------------------------------------------------------------------------ lock
@contextmanager
def lock(cfg, timeout: float = 0.0):
    """Advisory exclusive lock so only one writer touches the store at a time.

    Raises immediately if another process holds it, rather than queueing - a
    cron `update` that collides with a running `backfill` should say so and exit,
    not pile up.
    """
    path = cfg.data_dir / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise StoreError(
                f"another sniim process is already writing to this store "
                f"({path}). Wait for it to finish, or check for a stale run."
            )
        fh.write(f"{os.getpid()} {_now()}\n")
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


# --------------------------------------------------------------------- manifest
class Manifest:
    """Append-only CSV ledger of fetched jobs. Enables --resume."""

    def __init__(self, path: Path):
        self.path = path
        self._done: dict[tuple[str, int, str], dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "modulo" not in reader.fieldnames:
                log.warning(
                    "%s has no usable header - treating the ledger as empty. "
                    "Existing parquet files are untouched; jobs will be refetched.",
                    self.path,
                )
                return
            for row in reader:
                try:
                    key = (row["modulo"], int(row["producto_id"]), row["periodo"])
                except (KeyError, ValueError, TypeError):
                    continue
                self._done[key] = row  # later rows win

    def status(self, modulo: str, producto_id: int, periodo: str) -> str | None:
        row = self._done.get((modulo, int(producto_id), str(periodo)))
        return row["status"] if row else None

    def is_complete(self, modulo: str, producto_id: int, periodo: str) -> bool:
        """True only for statuses that are genuinely finished.

        Anything else - failed, partial, truncated, unreadable, future - is
        retried on the next run. That is the whole point: a job must never be
        skipped forever on the strength of an incomplete fetch.
        """
        return self.status(modulo, producto_id, periodo) in TERMINAL_STATUSES

    def record(
        self,
        modulo: str,
        producto_id: int,
        periodo: str,
        status: str,
        rows: int = 0,
        pages: int = 1,
        malformed: int = 0,
        rango_inicio=None,
        rango_fin=None,
        note: str = "",
    ) -> None:
        rec = {
            "modulo": modulo,
            "producto_id": int(producto_id),
            "periodo": str(periodo),
            "status": status,
            "rows": rows,
            "pages": pages,
            "malformed": malformed,
            "rango_inicio": rango_inicio or "",
            "rango_fin": rango_fin or "",
            "fetched_at": _now(),
            "note": note,
        }
        need_header = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if need_header:
                w.writeheader()
            w.writerow(rec)
            fh.flush()
            os.fsync(fh.fileno())
        self._done[(modulo, int(producto_id), str(periodo))] = {
            k: str(v) for k, v in rec.items()
        }

    def as_frame(self) -> pd.DataFrame:
        if not self._done:
            return pd.DataFrame(columns=MANIFEST_FIELDS)
        return pd.DataFrame(list(self._done.values()))


# ------------------------------------------------------------------------ store
class Store:
    def __init__(self, cfg):
        self.cfg = cfg
        self.raw_dir = cfg.raw_dir
        self.manifest = Manifest(cfg.manifest_path)

    # ------------------------------------------------------------- paths
    def part_path(self, modulo: str, producto_id: int, anio: int) -> Path:
        return (
            self.raw_dir
            / modulo
            / f"producto_id={int(producto_id)}"
            / f"anio={int(anio)}"
            / "part.parquet"
        )

    # ------------------------------------------------------------- write
    def write_observations(self, df: pd.DataFrame) -> dict[int, int]:
        """Merge rows into their product-year parquet files.

        Returns {anio: rows_in_file_after_merge}.

        Raises StoreError if an existing part file cannot be read. Silently
        rebuilding from the current fetch would be far worse than failing: a
        14-day incremental would replace a full year of history with 14 days of
        it, and the manifest would still say the year was complete.
        """
        if df.empty:
            return {}

        df = self._normalise(df)
        written: dict[int, int] = {}

        for (modulo, producto_id, anio), chunk in df.groupby(
            ["modulo", "producto_id", "anio"], sort=True
        ):
            path = self.part_path(modulo, int(producto_id), int(anio))
            path.parent.mkdir(parents=True, exist_ok=True)

            chunk = chunk.drop(columns=["anio"])
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                except Exception as exc:
                    quarantine = path.with_name(
                        f"part.corrupt-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.parquet"
                    )
                    os.replace(path, quarantine)
                    raise StoreError(
                        f"could not read {path} ({exc}). Moved it to {quarantine.name} "
                        f"rather than overwriting it. Rerun the backfill for "
                        f"{modulo} product {producto_id} year {anio} to rebuild."
                    ) from exc
                if not existing.empty:
                    # New rows last so they win the dedupe.
                    chunk = pd.concat([existing, chunk], ignore_index=True)

            before = len(chunk)
            chunk = chunk.drop_duplicates(subset=DEDUP_KEY, keep="last")
            collapsed = before - len(chunk)
            if collapsed and log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "%s p%s %s: %d duplicate natural keys collapsed",
                    modulo, producto_id, anio, collapsed,
                )

            chunk = chunk.sort_values(["fecha", "destino", "origen"]).reset_index(drop=True)
            chunk = chunk.reindex(columns=OBS_COLUMNS)

            # pid in the temp name: two writers must never share a temp path.
            tmp = path.with_name(f"part.{os.getpid()}.tmp")
            chunk.to_parquet(tmp, index=False, compression="zstd")
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, path)  # atomic
            written[int(anio)] = len(chunk)

        return written

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ("presentacion", "calidad", "grupo", "obs"):
            if col not in df.columns:
                df[col] = None

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df[df["fecha"].notna()]
        df["anio"] = df["fecha"].dt.year
        df["fecha"] = df["fecha"].dt.date

        # "Estado: Mercado" -> two usable columns.
        dest = df["destino"].fillna("")
        split = dest.str.split(":", n=1, expand=True)
        df["destino_estado"] = split[0].str.strip().replace("", None)
        if split.shape[1] > 1:
            df["destino_mercado"] = split[1].str.strip().replace("", None)
        else:
            df["destino_mercado"] = None
        df["destino_mercado"] = df["destino_mercado"].where(df["destino_mercado"].notna(), None)

        for col in ("precio_min", "precio_max", "precio_frec"):
            series = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)
            series = pd.to_numeric(series, errors="coerce")
            # Non-finite values survive a parquet round trip and would poison
            # every downstream ln()/avg(). Drop them here.
            df[col] = series.where(series.notna() & (series > 0) & (series < 1e12))

        # Guard against min/max being reported the wrong way round. NOTE: this
        # only fires when both are present on the same row, so it cannot mask a
        # column-shift bug - parse.py rejects malformed rows instead.
        #
        # This is the ONE silent modification between the page and the parquet, so
        # it is logged: after the swap the inversion is undetectable, which means
        # without this line there would be no record that it ever happened.
        swap = (
            df["precio_min"].notna()
            & df["precio_max"].notna()
            & (df["precio_min"] > df["precio_max"])
        )
        n_swap = int(swap.sum())
        if n_swap:
            log.warning(
                "%d row(s) had precio_min > precio_max and were swapped "
                "(SNIIM data quirk; product_id=%s)",
                n_swap, sorted(set(df.loc[swap, "producto_id"].tolist()))[:5])
            df.loc[swap, ["precio_min", "precio_max"]] = df.loc[
                swap, ["precio_max", "precio_min"]
            ].values

        for col in ("producto", "calidad", "grupo", "presentacion", "origen", "destino", "obs"):
            if col in df.columns:
                df[col] = df[col].astype("object").where(df[col].notna(), None)

        return df

    # -------------------------------------------------------------- read
    def read_module(self, modulo: str) -> pd.DataFrame:
        root = self.raw_dir / modulo
        files = sorted(root.glob("producto_id=*/anio=*/part.parquet"))
        if not files:
            return pd.DataFrame(columns=OBS_COLUMNS)
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as exc:
                log.error("unreadable %s: %s", f, exc)
                raise StoreError(f"unreadable part file {f}: {exc}") from exc
        return pd.concat(frames, ignore_index=True)

    def row_counts(self) -> pd.DataFrame:
        import pyarrow.parquet as pq

        rows = []
        for modulo_dir in sorted(self.raw_dir.glob("*")):
            if not modulo_dir.is_dir():
                continue
            for f in sorted(modulo_dir.glob("producto_id=*/anio=*/part.parquet")):
                pid = int(f.parent.parent.name.split("=")[1])
                anio = int(f.parent.name.split("=")[1])
                try:
                    n = pq.ParquetFile(f).metadata.num_rows
                except Exception:
                    n = -1
                rows.append(
                    {"modulo": modulo_dir.name, "producto_id": pid, "anio": anio, "rows": n}
                )
        return pd.DataFrame(rows)

    def corrupt_files(self) -> list[Path]:
        return sorted(self.raw_dir.rglob("part.corrupt-*.parquet"))

    def stray_temp_files(self) -> list[Path]:
        return sorted(self.raw_dir.rglob("part.*.tmp"))


# ------------------------------------------------------------------- catalogs
def save_catalog(cfg, name: str, pairs: list[tuple[str, str]]) -> Path:
    path = cfg.catalog_dir / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(pairs, columns=["id", "label"])
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_catalog(cfg, name: str) -> pd.DataFrame:
    path = cfg.catalog_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"catalog {name} missing - run `python run.py catalog` first ({path})"
        )
    return pd.read_csv(path, dtype={"id": str})
