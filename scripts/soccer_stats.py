#!/usr/bin/env python3
"""
soccer_stats.py — World Cup stats enrichment for the briefings, via the
`soccerdata` library (FBref scraper). Run with the dedicated venv:

  /Users/wilsonfelicio/.openclaw/workspace/.venv-soccer/bin/python soccer_stats.py <cmd>

ROLE: soccerdata is the source for ADVANCED / HISTORICAL stats (xG, shots,
possession, top scorers, past-World-Cup form). It is NOT a live-score feed —
FBref lags several hours, so web_search stays primary for live results, kickoff
times, and news. This script always degrades gracefully: if a 2026 stat isn't
posted yet, it says so instead of crashing.

Commands:
  snapshot              -> writes /tmp/wc_stats.txt: 2026 results + top scorers + team xG (whatever exists)
  matchup "A" "B"       -> prints stats block for a pre-match briefing: each team's
                           2026 tournament numbers so far + their 2022 World Cup form
                           + any past-WC head-to-head. Team names are fuzzy-matched.
  results               -> prints the 2026 schedule/results table
  scorers [N]           -> prints top-N 2026 scorers (default 15) if available

Output is plain text meant to be read by the briefing agent and woven in.
"""
import sys, os, io, contextlib, difflib
from datetime import datetime

SEASON = "2026"
HIST_SEASONS = ["2022"]          # past editions for historical context (kept short = less scraping)
OUT_FILE = "/tmp/wc_stats.txt"

# Quiet the very chatty soccerdata logger before import
import logging
logging.getLogger("soccerdata").setLevel(logging.ERROR)
os.environ.setdefault("SOCCERDATA_LOGLEVEL", "ERROR")

import soccerdata as sd  # noqa: E402
import pandas as pd       # noqa: E402


def _fb(seasons):
    return sd.FBref(leagues="INT-World Cup", seasons=seasons)


def _col(df, *names):
    """Find the first matching column (handles MultiIndex tuples)."""
    for n in names:
        for c in df.columns:
            label = c[-1] if isinstance(c, tuple) else c
            if str(label).lower() == n.lower():
                return c
    return None


def get_results(season=SEASON):
    """Schedule + results table. Returns a DataFrame or None."""
    try:
        s = _fb(season).read_schedule().reset_index()
        return s
    except Exception as e:
        print(f"  (results unavailable: {type(e).__name__})", file=sys.stderr)
        return None


def fmt_results(season=SEASON):
    s = get_results(season)
    if s is None or s.empty:
        return "Results: not available yet."
    lines = []
    for _, r in s.iterrows():
        score = r.get("score")
        score = "vs" if pd.isna(score) or score in (None, "") else str(score)
        d = str(r.get("date", ""))[:10]
        lines.append(f"  {d}  {r.get('home_team','?')} {score} {r.get('away_team','?')}")
    played = [l for l in lines if " vs " not in l]
    upcoming = [l for l in lines if " vs " in l]
    out = [f"WORLD CUP {season} — RESULTS & FIXTURES (source: FBref)"]
    if played:
        out += ["Played:"] + played
    if upcoming:
        out += ["Upcoming (FBref may lag live scores):"] + upcoming[:14]
    return "\n".join(out)


def get_scorers(season=SEASON, top=15):
    """Top scorers from player season stats. Returns text (graceful if absent)."""
    try:
        ps = _fb(season).read_player_season_stats(stat_type="standard").reset_index()
    except Exception as e:
        return f"Top scorers ({season}): not posted yet (early tournament). [{type(e).__name__}]"
    gcol = _col(ps, "Gls")
    pcol = _col(ps, "player")
    tcol = _col(ps, "team")
    if gcol is None or pcol is None:
        return f"Top scorers ({season}): goal data not available yet."
    ps = ps[[pcol, tcol, gcol]].copy() if tcol is not None else ps[[pcol, gcol]].copy()
    ps.columns = ["player", "team", "goals"][: ps.shape[1]]
    ps["goals"] = pd.to_numeric(ps["goals"], errors="coerce").fillna(0)
    ps = ps[ps["goals"] > 0].sort_values("goals", ascending=False).head(top)
    if ps.empty:
        return f"Top scorers ({season}): no goals recorded yet."
    out = [f"TOP SCORERS ({season}):"]
    for _, r in ps.iterrows():
        team = f" ({r['team']})" if "team" in ps.columns else ""
        out.append(f"  {int(r['goals'])}  {r['player']}{team}")
    return "\n".join(out)


def get_team_xg(season=SEASON):
    try:
        ts = _fb(season).read_team_season_stats(stat_type="standard").reset_index()
    except Exception:
        return None
    return ts


def _match_team(name, candidates):
    """Fuzzy-match a user-given team name to FBref's naming."""
    names = list(candidates)
    low = {c.lower(): c for c in names}
    if name.lower() in low:
        return low[name.lower()]
    # substring
    for c in names:
        if name.lower() in c.lower() or c.lower() in name.lower():
            return c
    m = difflib.get_close_matches(name, names, n=1, cutoff=0.5)
    return m[0] if m else None


def team_hist_form(team_query, seasons=HIST_SEASONS):
    """For each historical edition: the team's matches + goals."""
    blocks = []
    for ssn in seasons:
        s = get_results(ssn)
        if s is None or s.empty:
            continue
        teams = pd.unique(pd.concat([s["home_team"], s["away_team"]]).dropna())
        match_name = _match_team(team_query, teams)
        if not match_name:
            continue
        rows = s[(s["home_team"] == match_name) | (s["away_team"] == match_name)]
        played = rows[rows["score"].notna()]
        lines = [f"  {match_name} at {ssn} World Cup: {len(played)} matches"]
        for _, r in played.iterrows():
            lines.append(f"    {r['home_team']} {r['score']} {r['away_team']}")
        blocks.append("\n".join(lines))
    if not blocks:
        return f"  No {seasons} World Cup record found for '{team_query}'."
    return "\n".join(blocks)


def h2h_history(a, b, seasons=HIST_SEASONS):
    meets = []
    for ssn in seasons:
        s = get_results(ssn)
        if s is None or s.empty:
            continue
        teams = pd.unique(pd.concat([s["home_team"], s["away_team"]]).dropna())
        na, nb = _match_team(a, teams), _match_team(b, teams)
        if not na or not nb:
            continue
        m = s[((s["home_team"] == na) & (s["away_team"] == nb)) |
              ((s["home_team"] == nb) & (s["away_team"] == na))]
        for _, r in m.iterrows():
            meets.append(f"  {ssn}: {r['home_team']} {r['score']} {r['away_team']}")
    return "\n".join(meets) if meets else f"  No past-World-Cup meetings ({seasons}) between {a} and {b}."


def cmd_matchup(a, b):
    print(f"=== soccerdata stats: {a} vs {b} (World Cup {SEASON}) ===\n")
    # 2026 results so far (context for current form in-tournament)
    print(fmt_results(SEASON))
    print()
    print(f"--- {a}: last World Cup form ---")
    print(team_hist_form(a))
    print()
    print(f"--- {b}: last World Cup form ---")
    print(team_hist_form(b))
    print()
    print("--- Past World Cup head-to-head ---")
    print(h2h_history(a, b))
    print("\n(Live scores/times: use web_search — FBref lags. Above is structured/historical context.)")


def cmd_snapshot():
    parts = [f"WC STATS SNAPSHOT — {datetime.now():%Y-%m-%d %H:%M}", ""]
    parts.append(fmt_results(SEASON))
    parts.append("")
    parts.append(get_scorers(SEASON))
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"Wrote {OUT_FILE} ({os.path.getsize(OUT_FILE)} bytes)")
    print("\n".join(parts))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "snapshot":
        cmd_snapshot()
    elif cmd == "results":
        print(fmt_results(sys.argv[2] if len(sys.argv) > 2 else SEASON))
    elif cmd == "scorers":
        top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        print(get_scorers(SEASON, top))
    elif cmd == "matchup":
        if len(sys.argv) < 4:
            print("usage: soccer_stats.py matchup \"Team A\" \"Team B\"")
            return
        cmd_matchup(sys.argv[2], sys.argv[3])
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
