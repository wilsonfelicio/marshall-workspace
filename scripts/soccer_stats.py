#!/usr/bin/env python3
"""
soccer_stats.py — World Cup stats + forecast engine for the briefings.

Run with the dedicated venv:
  /Users/wilsonfelicio/.openclaw/workspace/.venv-soccer/bin/python soccer_stats.py <cmd>

Two roles:
  STATS (soccerdata / FBref) — advanced + historical numbers (xG, scorers, past-WC
  form). FBref lags live scores by hours, so web_search stays primary for live data.
  FORECAST — market-anchored prediction with an INDEPENDENT Elo cross-check, inspired
  by Nate Silver's PELE model (negative-binomial + Dixon-Coles scoreline; market-vs-
  model divergence; blend; stage variance; injury market-value haircut; xG form).

Commands:
  snapshot                 -> /tmp/wc_stats.txt: 2026 results + top scorers (when posted)
  results [season]         -> 2026 schedule/results table
  scorers [N]              -> top-N 2026 scorers (when posted)
  matchup "A" "B"          -> pre-match stats: each side's last WC run + past-WC H2H + 2026 results
  form "Team"              -> team's 2026 xG vs goals so far + a suggested Elo form nudge
  forecast --home .. --away .. --ph .. --pd .. --pa .. [opts]
                           -> full forecast (market / scoreline / Elo model / divergence / blend)
  calibrate                -> fit scoreline dispersion (r, rho) to historical WC goal data
  score                    -> grade logged forecasts vs results: running Brier/RPS, model-vs-market

forecast options:
  --total <ou>             market over/under total-goals line (sharpens the scoreline)
  --home-elo / --away-elo  world-football Elo (eloratings.net) -> enables the independent model
  --hfa <elo>              host/altitude/travel bump for the home (first) side; 0 if neutral
  --stage group|knockout   variance scaler on the Elo gap (group 0.9x upset-prone, knockout 1.1x chalky)
  --home-out-pct / --away-out-pct   share (0-1) of squad market value unavailable (injuries/bans) -> Elo haircut
  --home-form / --away-form         Elo nudge from recent xG over/under-performance (use `form`)
  --mkt-weight <0-1>       weight on market in the blend (lower for thin markets)
  --r / --rho             scoreline dispersion + Dixon-Coles draw correction (see `calibrate`)
  --log                   append the forecast to data/wc_forecasts.jsonl for later scoring
"""
import sys, os, math, difflib, json, re
from datetime import datetime
import numpy as np

SEASON = "2026"
HIST_SEASONS = ["2022"]
OUT_FILE = "/tmp/wc_stats.txt"
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECAST_LOG = os.path.join(WORKSPACE, "data", "wc_forecasts.jsonl")

# Forecast constants (defaults; `calibrate` tunes r/rho, live `score` tunes SUP_PER_ELO)
SUP_PER_ELO = 175.0     # Elo points per 1.0 goal of supremacy
ELO_PER_OUT = 300.0     # Elo penalty per full (1.0) share of squad value missing
STAGE_MULT = {"group": 0.9, "knockout": 1.1, "auto": 1.0}
DEF_R = 9.5      # calibrated on 128 historical WC matches (2022+2018): draw 22%, over2.5 48%
DEF_RHO = 0.04   # see `calibrate`; refine with live `score` as 2026 results accumulate

import logging
logging.getLogger("soccerdata").setLevel(logging.ERROR)
os.environ.setdefault("SOCCERDATA_LOGLEVEL", "ERROR")

import pandas as pd  # noqa: E402


# ─── soccerdata helpers ───────────────────────────────────────────────────
def _fb(seasons):
    import soccerdata as sd  # lazy: keeps forecast/calibrate/score free of selenium deps when unused
    return sd.FBref(leagues="INT-World Cup", seasons=seasons)


def _col(df, *names):
    for n in names:
        for c in df.columns:
            label = c[-1] if isinstance(c, tuple) else c
            if str(label).lower() == n.lower():
                return c
    return None


def get_results(season=SEASON):
    try:
        return _fb(season).read_schedule().reset_index()
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
        lines.append(f"  {str(r.get('date',''))[:10]}  {r.get('home_team','?')} {score} {r.get('away_team','?')}")
    played = [l for l in lines if " vs " not in l]
    upcoming = [l for l in lines if " vs " in l]
    out = [f"WORLD CUP {season} — RESULTS & FIXTURES (source: FBref)"]
    if played:
        out += ["Played:"] + played
    if upcoming:
        out += ["Upcoming (FBref may lag live scores):"] + upcoming[:14]
    return "\n".join(out)


def get_scorers(season=SEASON, top=15):
    try:
        ps = _fb(season).read_player_season_stats(stat_type="standard").reset_index()
    except Exception as e:
        return f"Top scorers ({season}): not posted yet (early tournament). [{type(e).__name__}]"
    gcol, pcol, tcol = _col(ps, "Gls"), _col(ps, "player"), _col(ps, "team")
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


def _match_team(name, candidates):
    names = list(candidates)
    low = {c.lower(): c for c in names}
    if name.lower() in low:
        return low[name.lower()]
    for c in names:
        if name.lower() in c.lower() or c.lower() in name.lower():
            return c
    m = difflib.get_close_matches(name, names, n=1, cutoff=0.5)
    return m[0] if m else None


def team_hist_form(team_query, seasons=HIST_SEASONS):
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
    return "\n".join(blocks) if blocks else f"  No {seasons} World Cup record found for '{team_query}'."


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


def team_2026_form(team_query):
    """Best-effort: 2026 tournament xG vs goals for a team + a suggested Elo form nudge."""
    try:
        ts = _fb(SEASON).read_team_match_stats(stat_type="standard").reset_index()
    except Exception as e:
        return f"2026 form for '{team_query}': not available yet ({type(e).__name__})."
    tcol = _col(ts, "team")
    if tcol is None:
        return f"2026 form for '{team_query}': no data yet."
    name = _match_team(team_query, pd.unique(ts[tcol].dropna()))
    if not name:
        return f"2026 form for '{team_query}': no matches found yet."
    rows = ts[ts[tcol] == name]
    gf, xg = _col(rows, "GF", "Gls"), _col(rows, "xG")
    n = len(rows)
    if n == 0:
        return f"2026 form for {name}: no matches yet."
    g = pd.to_numeric(rows[gf], errors="coerce").sum() if gf is not None else float("nan")
    x = pd.to_numeric(rows[xg], errors="coerce").sum() if xg is not None else float("nan")
    msg = [f"2026 form for {name}: {n} match(es), goals {g:.0f}, xG {x:.2f}" if not math.isnan(x)
           else f"2026 form for {name}: {n} match(es), goals {g:.0f}"]
    if xg is not None and not math.isnan(x) and n:
        xgd = (x - (g if not math.isnan(g) else 0)) / n  # +ve = under-finishing (xG > goals)
        # form nudge: reward strong underlying xG, small magnitude
        nudge = max(-40, min(40, round(25 * (x / n - 1.1))))  # vs ~1.1 baseline xG/game
        msg.append(f"  suggested --form nudge ~ {nudge:+d} Elo (xG/game {x/n:.2f}); finishing delta {xgd:+.2f}/game")
    return "\n".join(msg)


def cmd_matchup(a, b):
    print(f"=== soccerdata stats: {a} vs {b} (World Cup {SEASON}) ===\n")
    print(fmt_results(SEASON))
    print(f"\n--- {a}: last World Cup form ---")
    print(team_hist_form(a))
    print(f"\n--- {b}: last World Cup form ---")
    print(team_hist_form(b))
    print("\n--- Past World Cup head-to-head ---")
    print(h2h_history(a, b))
    print("\n(Live scores/times: use web_search — FBref lags. Above is structured/historical context.)")


def cmd_snapshot():
    parts = [f"WC STATS SNAPSHOT — {datetime.now():%Y-%m-%d %H:%M}", "", fmt_results(SEASON), "", get_scorers(SEASON)]
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"Wrote {OUT_FILE} ({os.path.getsize(OUT_FILE)} bytes)")
    print("\n".join(parts))


# ─── FORECAST ENGINE ──────────────────────────────────────────────────────
MAXG = 10


def _nb_pmf(mean, r, kmax=MAXG):
    mean = max(float(mean), 1e-3)
    ks = np.arange(0, kmax + 1)
    p = r / (r + mean)
    logc = np.array([math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) for k in ks])
    pmf = np.exp(logc + r * math.log(p) + ks * math.log(1.0 - p))
    return pmf / pmf.sum()


def _score_matrix(lh, la, r=DEF_R, rho=DEF_RHO):
    M = np.outer(_nb_pmf(lh, r), _nb_pmf(la, r))
    M[0, 0] *= 1.0 - lh * la * rho
    M[0, 1] *= 1.0 + lh * rho
    M[1, 0] *= 1.0 + la * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0.0, None)
    return M / M.sum()


def _wdl(M):
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def _fit_market(pH, pD, pA, total=None, r=DEF_R, rho=DEF_RHO):
    best = None
    if total:
        for lh in np.linspace(0.12, total - 0.12, 90):
            h, d, a = _wdl(_score_matrix(lh, total - lh, r, rho))
            e = (h - pH) ** 2 + (d - pD) ** 2 + (a - pA) ** 2
            if best is None or e < best[0]:
                best = (e, float(lh), float(total - lh))
    else:
        for lh in np.linspace(0.2, 3.6, 60):
            for la in np.linspace(0.2, 3.6, 60):
                h, d, a = _wdl(_score_matrix(lh, la, r, rho))
                e = (h - pH) ** 2 + (d - pD) ** 2 + (a - pA) ** 2
                if best is None or e < best[0]:
                    best = (e, float(lh), float(la))
    return best[1], best[2]


def _top_scores(M, n=5):
    flat = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0][:n]
    return [(int(i), int(j), float(M[i, j])) for i, j in flat]


def _summ(M):
    btts = float(M[1:, 1:].sum())
    ii, jj = np.indices(M.shape)
    return btts, float(M[(ii + jj) >= 3].sum())


def _pct(x):
    return f"{round(100 * x)}%"


def cmd_forecast(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="soccer_stats.py forecast")
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--ph", type=float, required=True)
    ap.add_argument("--pd", type=float, required=True)
    ap.add_argument("--pa", type=float, required=True)
    ap.add_argument("--total", type=float, default=None)
    ap.add_argument("--home-elo", type=float, default=None)
    ap.add_argument("--away-elo", type=float, default=None)
    ap.add_argument("--hfa", type=float, default=0.0)
    ap.add_argument("--stage", choices=["group", "knockout", "auto"], default="auto")
    ap.add_argument("--home-out-pct", type=float, default=0.0)
    ap.add_argument("--away-out-pct", type=float, default=0.0)
    ap.add_argument("--home-form", type=float, default=0.0)
    ap.add_argument("--away-form", type=float, default=0.0)
    ap.add_argument("--mkt-weight", type=float, default=0.7)
    ap.add_argument("--r", type=float, default=DEF_R)
    ap.add_argument("--rho", type=float, default=DEF_RHO)
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args(argv)

    s = a.ph + a.pd + a.pa
    pH, pD, pA = a.ph / s, a.pd / s, a.pa / s
    out = [f"=== FORECAST: {a.home} vs {a.away} ===",
           f"MARKET (de-vigged): {a.home} {_pct(pH)} / Draw {_pct(pD)} / {a.away} {_pct(pA)}"
           + (f"  | O/U {a.total}" if a.total else "")]

    # Lesson #1 — market-consistent scoreline
    lh, la = _fit_market(pH, pD, pA, a.total, a.r, a.rho)
    M = _score_matrix(lh, la, a.r, a.rho)
    h, d, al = _wdl(M)
    btts, o25 = _summ(M)
    tops = _top_scores(M)
    out += ["", "SCORELINE (neg-binomial + Dixon-Coles, fit to market):",
            f"  expected goals {a.home} {lh:.2f} - {la:.2f} {a.away}  (check {_pct(h)}/{_pct(d)}/{_pct(al)})",
            "  likeliest: " + ", ".join(f"{i}-{j} {_pct(p)}" for i, j, p in tops),
            f"  both score {_pct(btts)} · over 2.5 {_pct(o25)}"]

    final_M, blend, model = M, None, None
    if a.home_elo is not None and a.away_elo is not None:
        mult = STAGE_MULT[a.stage]
        eh = a.home_elo + a.hfa + a.home_form - ELO_PER_OUT * a.home_out_pct
        ea = a.away_elo + a.away_form - ELO_PER_OUT * a.away_out_pct
        dr = (eh - ea) * mult
        tot = a.total if a.total else (lh + la)
        sup = dr / SUP_PER_ELO
        elh, ela = max(0.12, (tot + sup) / 2), max(0.12, (tot - sup) / 2)
        mh, md, ma = _wdl(_score_matrix(elh, ela, a.r, a.rho))
        model = (mh, md, ma)
        adj = []
        if a.hfa:
            adj.append(f"HFA {a.hfa:+.0f}")
        if a.home_out_pct or a.away_out_pct:
            adj.append(f"inj {a.home}-{a.home_out_pct*100:.0f}%/{a.away}-{a.away_out_pct*100:.0f}%")
        if a.home_form or a.away_form:
            adj.append(f"form {a.home_form:+.0f}/{a.away_form:+.0f}")
        if a.stage != "auto":
            adj.append(f"{a.stage} x{mult}")
        out += ["", f"INDEPENDENT MODEL (Elo {eh:.0f} vs {ea:.0f}{'; ' + ', '.join(adj) if adj else ''}):",
                f"  {a.home} {_pct(mh)} / Draw {_pct(md)} / {a.away} {_pct(ma)}  (xg {elh:.2f}-{ela:.2f})"]
        dH, dD, dA = mh - pH, md - pD, ma - pA
        out += ["", "DIVERGENCE (model − market):",
                f"  {a.home} {dH*100:+.0f}pp · Draw {dD*100:+.0f}pp · {a.away} {dA*100:+.0f}pp"]
        edge = max([(abs(dH), a.home, dH), (abs(dD), "Draw", dD), (abs(dA), a.away, dA)])
        out.append(f"  >> Model {abs(edge[2])*100:.0f}pp {'higher' if edge[2] > 0 else 'lower'} than market on {edge[1]} — possible edge."
                   if edge[0] >= 0.05 else "  >> Model and market agree (no gap > 5pp).")
        w = min(max(a.mkt_weight, 0.0), 1.0)
        bH, bD, bA = w * pH + (1 - w) * mh, w * pD + (1 - w) * md, w * pA + (1 - w) * ma
        bs = bH + bD + bA
        bH, bD, bA = bH / bs, bD / bs, bA / bs
        blend = (bH, bD, bA)
        lhb, lab = _fit_market(bH, bD, bA, a.total, a.r, a.rho)
        final_M = _score_matrix(lhb, lab, a.r, a.rho)
        out += ["", f"BLEND ({int(w*100)}% market / {int((1-w)*100)}% model):",
                f"  {a.home} {_pct(bH)} / Draw {_pct(bD)} / {a.away} {_pct(bA)}"]

    fin = blend if blend else (pH, pD, pA)
    pick = max([(a.home, fin[0]), ("Draw", fin[1]), (a.away, fin[2])], key=lambda x: x[1])
    ml = _top_scores(final_M, 1)[0]
    out += ["", f"HEADLINE: {pick[0]} most likely ({_pct(pick[1])}); scoreline {ml[0]}-{ml[1]}.",
            "(Analytical prediction; market-anchored with an independent Elo cross-check. Not a betting tip.)"]
    print("\n".join(out))

    if a.log:
        os.makedirs(os.path.dirname(FORECAST_LOG), exist_ok=True)
        entry = {"date": datetime.now().strftime("%Y-%m-%d"), "home": a.home, "away": a.away,
                 "stage": a.stage, "total": a.total, "market": [pH, pD, pA],
                 "model": list(model) if model else None, "blend": list(blend) if blend else None,
                 "scoreline": [ml[0], ml[1]]}
        with open(FORECAST_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n[logged to {FORECAST_LOG}]")


# ─── CALIBRATION (fit scoreline dispersion to historical WC goal data) ─────
def _parse_score(scorestr):
    """Extract the regulation (home, away) goals from an FBref score string."""
    if scorestr is None or (isinstance(scorestr, float) and math.isnan(scorestr)):
        return None
    m = re.search(r"(\d+)\s*[–\-:]\s*(\d+)", str(scorestr))
    return (int(m.group(1)), int(m.group(2))) if m else None


def cmd_calibrate(argv):
    seasons = argv if argv else ["2022", "2018"]
    scores = []
    for ssn in seasons:
        s = get_results(ssn)
        if s is None or s.empty:
            continue
        for _, r in s.iterrows():
            sc = _parse_score(r.get("score"))
            if sc:
                scores.append(sc)
    if not scores:
        print("calibrate: no historical scores available.")
        return
    hg = np.array([h for h, a in scores])
    ag = np.array([a for h, a in scores])
    n = len(scores)
    emp_draw = float(np.mean(hg == ag))
    emp_zz = float(np.mean((hg == 0) & (ag == 0)))
    emp_o25 = float(np.mean(hg + ag >= 3))
    mh, ma = float(hg.mean()), float(ag.mean())
    print(f"=== CALIBRATION on {n} historical WC matches ({', '.join(seasons)}) ===")
    print(f"empirical: avg goals {mh:.2f}-{ma:.2f} · draw {_pct(emp_draw)} · 0-0 {_pct(emp_zz)} · over2.5 {_pct(emp_o25)}")
    best = None
    for r in np.linspace(3, 20, 35):
        for rho in np.linspace(-0.16, 0.04, 41):
            M = _score_matrix(mh, ma, r, rho)
            md = np.trace(M)
            mzz = M[0, 0]
            ii, jj = np.indices(M.shape)
            mo25 = M[(ii + jj) >= 3].sum()
            err = (md - emp_draw) ** 2 + (mzz - emp_zz) ** 2 + (mo25 - emp_o25) ** 2
            if best is None or err < best[0]:
                best = (err, float(r), float(rho), float(md), float(mzz), float(mo25))
    _, r, rho, md, mzz, mo25 = best
    print(f"best fit: r={r:.1f}, rho={rho:+.3f}")
    print(f"model at fit: draw {_pct(md)} · 0-0 {_pct(mzz)} · over2.5 {_pct(mo25)}")
    print(f"\n>> Set DEF_R={r:.1f}, DEF_RHO={rho:+.3f} in soccer_stats.py (representative-match fit; "
          f"refine with live `score`).")


# ─── SCORING (grade logged forecasts vs actual results) ───────────────────
def _brier(p, outcome):  # p=[H,D,A], outcome in {0,1,2}
    y = [0, 0, 0]
    y[outcome] = 1
    return sum((p[i] - y[i]) ** 2 for i in range(3))


def _rps(p, outcome):  # ranked probability score over ordered H,D,A
    y = [0, 0, 0]
    y[outcome] = 1
    cp = cy = 0.0
    s = 0.0
    for i in range(3):
        cp += p[i]
        cy += y[i]
        s += (cp - cy) ** 2
    return s / 2.0


def cmd_score(argv):
    if not os.path.exists(FORECAST_LOG):
        print(f"score: no forecast log yet at {FORECAST_LOG}.")
        return
    entries = [json.loads(l) for l in open(FORECAST_LOG) if l.strip()]
    res = get_results(SEASON)
    if res is None or res.empty:
        print("score: no 2026 results available to grade against.")
        return
    teams = pd.unique(pd.concat([res["home_team"], res["away_team"]]).dropna())
    graded = {"market": {"brier": [], "rps": [], "hit": []},
              "model": {"brier": [], "rps": [], "hit": []},
              "blend": {"brier": [], "rps": [], "hit": []}}
    n_graded = 0
    for e in entries:
        na, nb = _match_team(e["home"], teams), _match_team(e["away"], teams)
        if not na or not nb:
            continue
        row = res[(res["home_team"] == na) & (res["away_team"] == nb)]
        flip = False
        if row.empty:
            row = res[(res["home_team"] == nb) & (res["away_team"] == na)]
            flip = True
        if row.empty:
            continue
        sc = _parse_score(row.iloc[0].get("score"))
        if not sc:
            continue
        hg, ag = sc
        if flip:
            hg, ag = ag, hg  # orient to e's home/away
        outcome = 0 if hg > ag else (2 if ag > hg else 1)
        n_graded += 1
        for src in ("market", "model", "blend"):
            p = e.get(src)
            if not p:
                continue
            graded[src]["brier"].append(_brier(p, outcome))
            graded[src]["rps"].append(_rps(p, outcome))
            graded[src]["hit"].append(1 if max(range(3), key=lambda i: p[i]) == outcome else 0)
    print(f"=== FORECAST SCORECARD — {n_graded} graded of {len(entries)} logged ===")
    if n_graded == 0:
        print("(no logged matches have finished yet)")
        return
    print(f"{'source':8} {'n':>3} {'Brier':>7} {'RPS':>7} {'hit%':>6}   (lower Brier/RPS = better)")
    for src in ("market", "model", "blend"):
        b = graded[src]["brier"]
        if not b:
            continue
        print(f"{src:8} {len(b):>3} {np.mean(b):>7.3f} {np.mean(graded[src]['rps']):>7.3f} {100*np.mean(graded[src]['hit']):>5.0f}%")
    bm = np.mean(graded["blend"]["brier"]) if graded["blend"]["brier"] else None
    mm = np.mean(graded["market"]["brier"]) if graded["market"]["brier"] else None
    if bm is not None and mm is not None:
        verdict = "beating" if bm < mm else "trailing"
        print(f"\n>> Blend is {verdict} the market on Brier ({bm:.3f} vs {mm:.3f}). "
              f"{'Edge looks real.' if bm < mm else 'No edge yet — markets winning.'}")


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
        print(get_scorers(SEASON, int(sys.argv[2]) if len(sys.argv) > 2 else 15))
    elif cmd == "matchup":
        if len(sys.argv) < 4:
            print('usage: soccer_stats.py matchup "Team A" "Team B"')
            return
        cmd_matchup(sys.argv[2], sys.argv[3])
    elif cmd == "form":
        if len(sys.argv) < 3:
            print('usage: soccer_stats.py form "Team"')
            return
        print(team_2026_form(sys.argv[2]))
    elif cmd == "forecast":
        cmd_forecast(sys.argv[2:])
    elif cmd == "calibrate":
        cmd_calibrate(sys.argv[2:])
    elif cmd == "score":
        cmd_score(sys.argv[2:])
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
