"""Polite HTTP client for the SNIIM site.

The server is Microsoft-IIS/6.0 running ASP.NET 2.0. Observed behaviour:
  * responds fine to plain GET on the Resultados* pages, no session cookie or
    __VIEWSTATE needed
  * returns 503 Service Unavailable when hit faster than roughly one request
    per second sustained
  * returns 500 Internal Server Error for malformed parameter combinations
    (a real signal, not a transient fault - do not retry those forever)

This client was originally single-threaded on the reasoning that parallelism is
what gets you blocked. That reasoning was half right. The offered REQUEST RATE is
what gets you blocked; the number of threads only matters because more threads
usually means a higher rate. So concurrency is now allowed, but every Session in
a run shares one `Limiter` that enforces the minimum gap between any two
requests globally. Two workers therefore overlap the server's ~7s latency without
sending requests any faster than the configured floor permits.

The limiter also degrades on its own: if 5xx responses exceed a share of a
rolling window it widens the gap AND cuts effective concurrency to one, then
relaxes back after a clean stretch. Measured on 2026-08-11, the site returned
503s even at one serial worker, so this is not hypothetical.
"""
from __future__ import annotations

import collections
import logging
import random
import threading
import time
from typing import Any

import requests

log = logging.getLogger("sniim")


class Limiter:
    """Process-wide gate on request rate and concurrency, shared by all Sessions.

    acquire() blocks until both conditions hold: fewer than `max_concurrent`
    requests are in flight, and at least `interval` seconds have passed since the
    last request STARTED. release() is called when the response comes back.
    """

    def __init__(self, interval: float, max_concurrent: int = 1,
                 cool_off_factor: float = 2.0, cool_off_max: float = 30.0,
                 window: int = 40, bad_share: float = 0.25) -> None:
        self.base_interval = float(interval)
        self.interval = float(interval)
        self.max_concurrent = max(1, int(max_concurrent))
        self.ceiling = self.max_concurrent
        self.cool_off_factor = float(cool_off_factor)
        self.cool_off_max = float(cool_off_max)
        self.bad_share = float(bad_share)
        self._recent = collections.deque(maxlen=window)
        self._streak = 0
        self.recover_after = max(8, window // 4)
        self._cv = threading.Condition()
        self._last_start = 0.0
        self._active = 0
        self.stats = {"waited": 0.0, "degradaciones": 0}

    def acquire(self) -> None:
        with self._cv:
            while True:
                now = time.monotonic()
                gap = now - self._last_start
                if self._active < self.max_concurrent and gap >= self.interval:
                    self._active += 1
                    self._last_start = now
                    return
                wait = self.interval - gap if self._active < self.max_concurrent else 0.25
                self.stats["waited"] += max(wait, 0.0)
                self._cv.wait(max(wait, 0.02))

    def release(self) -> None:
        with self._cv:
            self._active -= 1
            self._cv.notify_all()

    def note(self, ok: bool) -> None:
        """Record one completed HTTP attempt and adapt.

        Degradation is share-based over a rolling window, so a few intermittent
        5xx among many successes do not trip it. Recovery is streak-based - N
        consecutive clean responses - because "share of the window is exactly
        zero" is a condition a couple of stale failures can block for a whole
        window length, which in production means tens of minutes stuck at the
        degraded setting.
        """
        with self._cv:
            if ok:
                self._streak += 1
            else:
                self._streak = 0
            self._recent.append(1 if ok else 0)

            n = len(self._recent)
            if n >= 8 and (1 - sum(self._recent) / n) > self.bad_share:
                bad = 1 - sum(self._recent) / n
                widened = min(self.cool_off_max, self.interval * self.cool_off_factor)
                if widened > self.interval or self.max_concurrent > 1:
                    log.warning(
                        "limitador: %.0f%% de las ultimas %d respuestas fueron 5xx - "
                        "gap %.1fs -> %.1fs, concurrencia %d -> 1",
                        100 * bad, n, self.interval, widened, self.max_concurrent)
                    self.stats["degradaciones"] += 1
                self.interval = widened
                self.max_concurrent = 1
                self._recent.clear()
                self._streak = 0
            elif self._streak >= self.recover_after:
                # One notch per clean streak, and the gap comes back before the
                # concurrency does, so the offered rate never rises on two axes
                # at the same time.
                if self.interval > self.base_interval:
                    self.interval = max(self.base_interval,
                                        self.interval / self.cool_off_factor)
                    log.info("limitador: %d respuestas limpias - gap %.1fs",
                             self._streak, self.interval)
                    self._streak = 0
                    self._recent.clear()
                elif self.max_concurrent < self.ceiling:
                    self.max_concurrent += 1
                    log.info("limitador: %d respuestas limpias - concurrencia %d",
                             self._streak, self.max_concurrent)
                    self._streak = 0
                    self._recent.clear()
            self._cv.notify_all()


class FetchError(RuntimeError):
    """Raised when a URL could not be retrieved after all retries."""

    def __init__(self, message: str, status: int | None = None, permanent: bool = False):
        super().__init__(message)
        self.status = status
        self.permanent = permanent


class Session:
    def __init__(self, cfg, limiter: "Limiter | None" = None) -> None:
        h = cfg.http
        self.base_url: str = h["base_url"]
        self.timeout: int = int(h["timeout_seconds"])
        self.max_retries: int = int(h["max_retries"])
        self.backoff_base: float = float(h["backoff_base_seconds"])
        self.base_interval: float = float(h["min_interval_seconds"])
        self.interval: float = self.base_interval
        self.cool_off_factor: float = float(h["cool_off_factor"])
        self.cool_off_max: float = float(h["cool_off_max_seconds"])

        self._last_request: float = 0.0
        self._consecutive_5xx: int = 0
        # When a limiter is supplied it owns pacing for the whole process and the
        # per-session throttle below is bypassed. One Session per thread is still
        # required: requests.Session is not safe to share across threads.
        self.limiter = limiter

        self.s = requests.Session()
        self.s.headers.update(
            {
                "User-Agent": h["user_agent"],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
                "Connection": "keep-alive",
            }
        )

        # Counters, surfaced at the end of a run.
        self.stats = {"requests": 0, "retries": 0, "bytes": 0, "errors": 0}

    # ------------------------------------------------------------------ rate
    def _wait_turn(self) -> None:
        if self.limiter is not None:
            self.limiter.acquire()
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request = time.monotonic()

    def _done_turn(self, ok: bool) -> None:
        if self.limiter is not None:
            self.limiter.release()
            self.limiter.note(ok)

    def _note_success(self) -> None:
        self._consecutive_5xx = 0
        if self.limiter is not None:
            return
        # Relax back toward the configured floor after a rough patch.
        if self.interval > self.base_interval:
            self.interval = max(self.base_interval, self.interval / self.cool_off_factor)

    def _note_5xx(self) -> None:
        self._consecutive_5xx += 1
        if self.limiter is not None:
            return
        if self._consecutive_5xx >= 2:
            new = min(self.cool_off_max, self.interval * self.cool_off_factor)
            if new > self.interval:
                log.warning(
                    "server is unhappy (%d consecutive 5xx) - slowing to %.1fs between requests",
                    self._consecutive_5xx,
                    new,
                )
            self.interval = new

    # ------------------------------------------------------------------ get
    def get(self, page: str, params: dict[str, Any]) -> str:
        """GET a Resultados page and return decoded HTML.

        Retries transient failures with exponential backoff and jitter.
        A 404 or a persistent 500 is treated as permanent so the caller can
        record the job as failed and move on instead of stalling the backfill.
        """
        url = self.base_url + page
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            self._wait_turn()
            # The limiter slot is held only for the duration of the request itself.
            # Backoff sleeps happen OUTSIDE it, so a stalled retry does not occupy
            # a concurrency slot and starve the other worker.
            r = None
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network-level
                last_exc = exc
                self.stats["retries"] += 1
                self._done_turn(ok=False)
                self._sleep_backoff(attempt, f"network error: {exc}")
                continue
            else:
                self._done_turn(ok=(r.status_code < 500))

            self.stats["requests"] += 1

            if r.status_code == 200:
                self.stats["bytes"] += len(r.content)
                self._note_success()
                # The site declares utf-8 in the Content-Type header; requests
                # sometimes guesses ISO-8859-1 for these old pages.
                r.encoding = r.encoding or "utf-8"
                if (r.encoding or "").lower() in ("iso-8859-1", "latin-1"):
                    r.encoding = "utf-8"
                return r.text

            if r.status_code == 404:
                self.stats["errors"] += 1
                raise FetchError(f"404 for {r.url}", status=404, permanent=True)

            if 500 <= r.status_code < 600:
                self._note_5xx()
                self.stats["retries"] += 1
                self._sleep_backoff(attempt, f"HTTP {r.status_code}")
                last_exc = FetchError(f"HTTP {r.status_code}", status=r.status_code)
                continue

            # 4xx other than 404: almost certainly our parameters.
            self.stats["errors"] += 1
            raise FetchError(f"HTTP {r.status_code} for {r.url}", status=r.status_code, permanent=True)

        self.stats["errors"] += 1
        raise FetchError(f"giving up on {url} after {self.max_retries} attempts: {last_exc}")

    def _sleep_backoff(self, attempt: int, reason: str) -> None:
        delay = self.backoff_base * (2**attempt) + random.uniform(0, 2.0)
        log.warning("%s - retry %d in %.1fs", reason, attempt + 1, delay)
        time.sleep(delay)

    def limiter_summary(self) -> str:
        if self.limiter is None:
            return "sin limitador compartido (modo serial)"
        L = self.limiter
        return (f"limitador: gap {L.interval:.1f}s (piso {L.base_interval:.1f}s), "
                f"concurrencia {L.max_concurrent}/{L.ceiling}, "
                f"{L.stats['degradaciones']} degradacion(es)")

    def summary(self) -> str:
        mb = self.stats["bytes"] / 1_048_576
        return (
            f"{self.stats['requests']} requests, {self.stats['retries']} retries, "
            f"{self.stats['errors']} hard errors, {mb:.1f} MB downloaded"
        )
