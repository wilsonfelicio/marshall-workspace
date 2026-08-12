"""Publish gate: refuse to ship a partial or stale day.

Run after `run.py update` and before building the workbook. Exits non-zero with a reason,
so the workflow stops and the previously published file stays in place — a stale download
is recoverable, a silently truncated one is not.

Three checks:

  1. RECENCY. The newest quote day must be within --max-age business days. SNIIM quotes
     Monday to Friday only, so a Monday run legitimately sees Friday's data; anything
     older than that means the fetch failed or the source stalled.

  2. COVERAGE. On the newest day, each generic must have at least --min-cover of the
     market count it normally carries, judged against its own trailing median. This is the
     check that catches the real failure mode: a day that exists but holds two markets out
     of seventy-eight, which is not a national price. At most --max-thin generics may fail
     it (a handful genuinely quote thinly).

     Cadence-aware: Frijol and Chile seco come from the WEEKLY granos module and quote on
     Wednesdays, so they are legitimately absent from a Tuesday's panel. Requiring every
     generic on the newest day failed the gate every non-Wednesday. Weekly generics are
     instead checked against their own last quote day.

  3. CONTINUITY. The store must not have lost history: the total row count may not fall
     below what the previous run recorded in .last_rows.

  python3 scripts/check_freshness.py [--max-age 4] [--min-cover 0.5] [--max-thin 4]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DAILY = ROOT / "data" / "curated" / "var_market_daily.parquet"
STATE = ROOT / "data" / "curated" / ".last_rows.json"

ap = argparse.ArgumentParser()
ap.add_argument("--max-age", type=int, default=4, help="business days of staleness allowed")
ap.add_argument("--min-cover", type=float, default=0.50)
ap.add_argument("--max-thin", type=int, default=4)
ap.add_argument("--no-state", action="store_true", help="skip the row-count continuity check")
A = ap.parse_args()

fail: list[str] = []
note: list[str] = []

if not DAILY.exists():
    print(f"FAIL  {DAILY} does not exist — did `run.py build` run?")
    raise SystemExit(1)

d = pd.read_parquet(DAILY, columns=["categoria_label", "fecha", "destino", "precio_geo"])
d["fecha"] = pd.to_datetime(d["fecha"])
d = d[d.precio_geo > 0]
last = d.fecha.max()
today = pd.Timestamp(dt.date.today())

# ---------------------------------------------------------------- 1. recency
age = int(np.busday_count(last.date(), today.date()))
if age > A.max_age:
    fail.append(f"stale: newest quote is {last:%Y-%m-%d}, {age} business days old "
                f"(limit {A.max_age})")
else:
    note.append(f"recency OK: newest quote {last:%Y-%m-%d}, {age} business days old")

# ---------------------------------------------------------------- 2. coverage
cnt = (d.groupby(["categoria_label", "fecha"]).destino.nunique()
       .rename("n").reset_index())
piv = cnt.pivot(index="fecha", columns="categoria_label", values="n").sort_index()
typical = piv.rolling(60, min_periods=5).median().shift(1)     # its own norm, excluding today
have = piv.loc[last]
norm = typical.loc[last]
ratio = (have / norm).replace([np.inf, -np.inf], np.nan)
# each generic's own cadence over the last quarter, so a weekly one is not judged by a
# daily one's calendar
recent = piv.loc[piv.index >= last - pd.Timedelta(days=90)]
cad = {}
for g in piv.columns:
    days = recent.index[recent[g].notna()]
    cad[g] = float(pd.Series(days).diff().dt.days.median() or 1.0)
daily_g = [g for g in piv.columns if cad[g] <= 3.0]
weekly_g = [g for g in piv.columns if cad[g] > 3.0]
missing = [g for g in daily_g if pd.isna(have.get(g)) and pd.notna(norm.get(g))]
stale_weekly = []
for g in weekly_g:
    days = piv.index[piv[g].notna()]
    if not len(days):
        stale_weekly.append(f"{g} (never)")
        continue
    gap = int((last - days.max()).days)
    if gap > max(10, int(round(2.5 * cad[g]))):
        stale_weekly.append(f"{g} last quoted {days.max():%Y-%m-%d}, {gap} days before "
                            f"the newest day")

thin = ratio[daily_g][ratio[daily_g] < A.min_cover].dropna().sort_values()

print(f"\nnewest day {last:%Y-%m-%d} — market coverage against each generic's own "
      f"trailing median:")
for g in ratio.sort_values().index[:6]:
    if pd.notna(ratio[g]):
        print(f"   {g:<32}{have[g]:4.0f} of ~{norm[g]:4.0f}   {100 * ratio[g]:5.0f}%")
print(f"   ... {len(daily_g)} daily generics scored, median coverage "
      f"{100 * ratio[daily_g].median():.0f}%; {len(weekly_g)} weekly checked separately")

if len(thin) > A.max_thin:
    fail.append(f"partial day: {len(thin)} generics below {A.min_cover:.0%} of their usual "
                f"market count (limit {A.max_thin}): "
                + ", ".join(f"{g} {100 * v:.0f}%" for g, v in thin.items()))
elif len(thin):
    note.append(f"coverage OK with {len(thin)} thin generic(s): "
                + ", ".join(f"{g} {100 * v:.0f}%" for g, v in thin.items()))
else:
    note.append("coverage OK: every generic at or above "
                f"{A.min_cover:.0%} of its usual market count")
if missing:
    fail.append(f"absent on the newest day: {', '.join(missing)}")
if stale_weekly:
    fail.append("weekly generics behind: " + "; ".join(stale_weekly))
else:
    note.append(f"weekly module OK: {', '.join(weekly_g) or 'none'} within cadence")

# ---------------------------------------------------------------- 3. continuity
rows = int(len(d))
if not A.no_state:
    prev = None
    if STATE.exists():
        try:
            prev = int(json.loads(STATE.read_text())["rows"])
        except Exception:
            prev = None
    if prev is not None and rows < prev:
        fail.append(f"store shrank: {rows:,} rows now against {prev:,} on the previous run")
    elif prev is not None:
        note.append(f"continuity OK: {rows:,} rows, {rows - prev:+,} since the last run")
    else:
        note.append(f"continuity: no previous count on file, recording {rows:,}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"rows": rows, "last_day": str(last.date()),
                                 "checked_at": dt.datetime.utcnow().isoformat() + "Z"},
                                indent=1))

# ---------------------------------------------------------------- verdict
print()
for n in note:
    print(f"ok    {n}")
for f in fail:
    print(f"FAIL  {f}")
if fail:
    print(f"\n{len(fail)} check(s) failed — not publishing. The previously published file "
          f"stays in place.")
    sys.exit(1)
print("\nall checks passed — safe to publish")
