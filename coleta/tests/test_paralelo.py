"""Tests for the concurrency layer. No network: the fetch is a sleep.

Run: python3 tests/test_paralelo.py
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sniim.http import Limiter  # noqa: E402

import run as R  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FALLA ") + msg)
    if not cond:
        fails.append(msg)


print("Limiter: espacia las peticiones y limita la concurrencia")
L = Limiter(interval=0.10, max_concurrent=2)
starts, lock = [], threading.Lock()
peak = {"n": 0, "cur": 0}


def hit():
    L.acquire()
    with lock:
        starts.append(time.monotonic())
        peak["cur"] += 1
        peak["n"] = max(peak["n"], peak["cur"])
    time.sleep(0.25)
    with lock:
        peak["cur"] -= 1
    L.release()
    L.note(ok=True)


ths = [threading.Thread(target=hit) for _ in range(8)]
t0 = time.monotonic()
[t.start() for t in ths]
[t.join() for t in ths]
el = time.monotonic() - t0
starts.sort()
gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
check(min(gaps) >= 0.095, f"gap minimo respetado: {min(gaps):.3f}s >= 0.10s")
check(peak["n"] <= 2, f"concurrencia nunca paso de 2 (pico {peak['n']})")
# 8 requests of 0.25s: serial would be 2.0s+, two at a time about 1.0s.
check(0.8 < el < 1.6, f"8 peticiones de 0.25s en {el:.2f}s (serial seria ~2.8s)")

print("Limiter: degrada solo cuando el servidor devuelve 5xx")
L2 = Limiter(interval=1.0, max_concurrent=2, cool_off_factor=2.0, cool_off_max=8.0)
for _ in range(10):
    L2.note(ok=False)
check(L2.max_concurrent == 1, f"concurrencia cayo a 1 (es {L2.max_concurrent})")
check(L2.interval > 1.0, f"gap se amplio a {L2.interval:.1f}s")
before = L2.interval
n_rec = L2.recover_after
for _ in range(n_rec - 1):
    L2.note(ok=True)
check(L2.interval == before and L2.max_concurrent == 1,
      f"{n_rec - 1} limpias no alcanzan: sigue en gap {L2.interval:.1f}s y 1 hilo")
L2.note(ok=True)
check(L2.interval < before and L2.max_concurrent == 1,
      f"primera racha limpia baja el gap a {L2.interval:.1f}s y NO toca la concurrencia")
for _ in range(n_rec):
    L2.note(ok=True)
check(L2.max_concurrent == 2 and L2.interval == 1.0,
      f"segunda racha limpia restaura la concurrencia a {L2.max_concurrent}")
L2.note(ok=False)
check(L2._streak == 0, "un solo 5xx reinicia la racha limpia")

print("Limiter: no se relaja por debajo del piso configurado")
L3 = Limiter(interval=1.5, max_concurrent=1)
for _ in range(60):
    L3.note(ok=True)
check(L3.interval == 1.5, f"piso intacto: {L3.interval:.2f}s")

print("_pump: persiste en orden y con 1 hilo es identico al camino serial")
for w in (1, 2, 4):
    seen = []
    jobs = list(range(12))

    def fetch(j):
        time.sleep(0.02 * ((j % 3) + 1))   # deliberately uneven, to try to scramble order
        return j

    def persist(res, i):
        seen.append((i, res))

    R._pump(jobs, fetch, persist, w)
    check(seen == [(i + 1, j) for i, j in enumerate(jobs)],
          f"workers={w}: 12 trabajos persistidos en orden y sin perder ninguno")

print("_pump: la ventana de vuelo esta acotada (no encola todo en memoria)")
inflight = {"cur": 0, "max": 0}
ilock = threading.Lock()


def fetch_slow(j):
    with ilock:
        inflight["cur"] += 1
        inflight["max"] = max(inflight["max"], inflight["cur"])
    time.sleep(0.01)
    with ilock:
        inflight["cur"] -= 1
    return j


R._pump(list(range(200)), fetch_slow, lambda r, i: time.sleep(0.005), 2)
check(inflight["max"] <= 2, f"nunca mas de 2 fetch simultaneos (pico {inflight['max']})")

print("_pump: un fallo en un trabajo no tumba la corrida")
got = []


def fetch_boom(j):
    if j == 3:
        raise RuntimeError("boom")
    return j


try:
    R._pump(list(range(6)), fetch_boom, lambda r, i: got.append(r), 2)
    check(False, "una excepcion en el fetch deberia propagarse, no silenciarse")
except RuntimeError:
    check(True, "la excepcion se propaga al hilo principal (no se pierde en silencio)")

print()
if fails:
    print(f"{len(fails)} prueba(s) fallaron")
    sys.exit(1)
print("todas las pruebas pasaron")


def test_collectors_default_to_the_single_session():
    """`update` passes no session factory; the collectors must not call None.

    Regression: _session_for defaulted to None and only cmd_backfill built a factory,
    so `run.py update` died with "'NoneType' object is not callable" the first time it
    reached the granos collector.
    """
    import datetime as _dt
    import logging as _lg
    from types import SimpleNamespace

    import run as R
    from sniim import granos as G

    sentinel = object()
    seen = []

    def fake_fetch_week(cfg, session, pid, y, m, sl):
        seen.append(session)
        return [], SimpleNamespace(usable=True, truncated=False, malformed_rows=0,
                                   rejected_reason=None, pages=1, producto="x",
                                   calidad=None, rango_inicio=None, rango_fin=None)

    class St:
        class manifest:
            @staticmethod
            def record(*a, **k):
                pass

            @staticmethod
            def done(*a, **k):
                return set()
        @staticmethod
        def write_observations(df):
            pass

    cfg = R.config.load(None) if hasattr(R.config, "load") else None
    if cfg is None:
        return
    old = G.fetch_week
    G.fetch_week = fake_fetch_week
    try:
        d1 = _dt.date(2026, 8, 12)
        R._collect_granos(cfg, sentinel, St, _lg.getLogger("t"),
                          d1 - _dt.timedelta(weeks=1), d1, [332], False, {332: "x"}, d1)
    finally:
        G.fetch_week = old
    assert seen, "the collector never fetched"
    assert all(s is sentinel for s in seen), "the passed session was not used"
