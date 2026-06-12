#!/usr/bin/env python3
"""
wc_simulate.py — Monte-Carlo simulation of the 2026 FIFA World Cup.

Run with the venv:
  /Users/wilsonfelicio/.openclaw/workspace/.venv-soccer/bin/python wc_simulate.py [N] [--seed S]

Like the Goldman Sachs / BCA models: simulate the whole tournament many times to get
each team's probability of reaching every round and lifting the trophy.

Method:
  - Group stage uses our calibrated goal model (negative-binomial, Elo -> expected goals)
    to produce realistic scorelines, points, goal difference -> group tables + the 8 best
    third-placed teams (full FIFA tie-break order: pts, GD, GF).
  - The knockout bracket uses the EXACT official 2026 map (group winners/runners-up pairings
    are exact; the 8 third-place slots are filled by a constraint-respecting matching of each
    slot's eligible group set — an approximation of FIFA's 495-row allocation table).
  - Knockout ties are decided by an Elo win-probability (winner only; ET/pens folded in).
  - Adjustments: host nations get a +90 Elo bump (BCA: host advantage ~+24% win prob);
    the defending champion gets a winner's-slump penalty (GS/BCA).
  - Optional: data/wc_results.json pins already-played group scores so the sim conditions
    on what actually happened (the daily digest can populate it).

Inputs:  data/wc_teams.json  (groups, seed Elo, hosts, holder)
Outputs: data/wc_sim.json    (per-team round probabilities) + a printed summary.
"""
import sys, os, json, itertools
import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAMS_FILE = os.path.join(WORKSPACE, "data", "wc_teams.json")
RESULTS_FILE = os.path.join(WORKSPACE, "data", "wc_results.json")
SIM_OUT = os.path.join(WORKSPACE, "data", "wc_sim.json")

SUP_PER_ELO = 175.0    # Elo points per 1.0 goal of supremacy (matches soccer_stats)
DEF_R = 9.5            # negative-binomial dispersion (calibrated on 2022+2018)
BASE_TOTAL = 2.6       # baseline combined goals, group game
HOST_BUMP = 90.0       # Elo bump for host nations (BCA: ~+24% win prob)
SLUMP = 60.0           # winner's-slump Elo penalty on the defending champion
KO_DIV = 400.0         # Elo logistic divisor for knockout win probability

GROUPS_ORDER = list("ABCDEFGHIJKL")

# Exact R32 pairings (FIFA 2026). Token types: ('W', group) winner, ('R', group) runner-up,
# ('3', slot) the third-placed team allocated to that slot.
R32 = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("3", 74)),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", 77)),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", 79)),
    80: (("W", "L"), ("3", 80)),
    81: (("W", "D"), ("3", 81)),
    82: (("W", "G"), ("3", 82)),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", 85)),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", 87)),
    88: (("R", "D"), ("R", "G")),
}
# Eligible group sets for each third-place slot (FIFA 2026).
SLOT_ALLOWED = {
    74: set("ABCDF"), 77: set("CDFGH"), 79: set("CEFHI"), 80: set("EHIJK"),
    81: set("BEFIJ"), 82: set("AEHIJ"), 85: set("EFGIJ"), 87: set("DEIJL"),
}
# Bracket flow: match -> (source match A, source match B), winners advance.
R16 = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = 104  # winners of 101 vs 102


def load_teams():
    d = json.load(open(TEAMS_FILE))
    names, elo, grp = [], [], []
    gindex = {}
    for g in GROUPS_ORDER:
        gindex[g] = []
        for nm, e, host in d["groups"][g]:
            idx = len(names)
            adj = e + (HOST_BUMP if host else 0) - (SLUMP if nm == d.get("holder") else 0)
            names.append(nm)
            elo.append(adj)
            grp.append(g)
            gindex[g].append(idx)
    return names, np.array(elo, float), grp, gindex, d


def lam(ei, ej):
    sup = (ei - ej) / SUP_PER_ELO
    return max(0.1, (BASE_TOTAL + sup) / 2), max(0.1, (BASE_TOTAL - sup) / 2)


def assign_thirds(qual_groups):
    """Perfect matching of the 8 qualifying third-place groups to the 8 slots, respecting
    each slot's eligible set. Returns {slot: group} or None. (FIFA's design guarantees a
    matching exists for every combination; greedy-with-backtracking finds one.)"""
    slots = sorted(SLOT_ALLOWED, key=lambda s: len(SLOT_ALLOWED[s] & qual_groups))
    used, out = set(), {}

    def bt(i):
        if i == len(slots):
            return True
        s = slots[i]
        for g in sorted(SLOT_ALLOWED[s] & qual_groups):
            if g not in used:
                used.add(g); out[s] = g
                if bt(i + 1):
                    return True
                used.discard(g); del out[s]
        return False
    return out if bt(0) else None


def simulate(n_sims, seed=None):
    names, elo, grp, gindex, meta = load_teams()
    nT = len(names)
    rng = np.random.default_rng(seed)
    pinned = {}
    if os.path.exists(RESULTS_FILE):
        try:
            raw = json.load(open(RESULTS_FILE))
            for k, v in raw.items():  # {"Brazil vs Morocco": [2, 1]}
                if " vs " in k:
                    x, y = k.split(" vs ", 1)
                    pinned[(x.strip(), y.strip())] = (int(v[0]), int(v[1]))
        except Exception:
            pinned = {}

    pts = np.zeros((nT, n_sims)); gd = np.zeros((nT, n_sims)); gf = np.zeros((nT, n_sims))

    # ---- Group stage (vectorised across sims) ----
    for g in GROUPS_ORDER:
        idxs = gindex[g]
        for a, b in itertools.combinations(idxs, 2):
            la, lb = lam(elo[a], elo[b])
            pa, pb = pinned.get((names[a], names[b])), pinned.get((names[b], names[a]))
            if pa is not None:                       # actual score, stored a-vs-b
                ga, gb = np.full(n_sims, pa[0]), np.full(n_sims, pa[1])
            elif pb is not None:                     # actual score, stored b-vs-a
                ga, gb = np.full(n_sims, pb[1]), np.full(n_sims, pb[0])
            else:                                    # not played yet -> simulate
                ga = rng.negative_binomial(DEF_R, DEF_R / (DEF_R + la), n_sims)
                gb = rng.negative_binomial(DEF_R, DEF_R / (DEF_R + lb), n_sims)
            pts[a] += 3 * (ga > gb) + (ga == gb)
            pts[b] += 3 * (gb > ga) + (ga == gb)
            gd[a] += ga - gb; gd[b] += gb - ga
            gf[a] += ga; gf[b] += gb

    # rank key: pts dominant, then GD, then GF, plus tiny noise to break exact ties
    rankkey = pts * 1e6 + (gd + 100) * 1e3 + gf + rng.random((nT, n_sims)) * 1e-3

    winners, runners, thirds, third_key = {}, {}, {}, {}
    for g in GROUPS_ORDER:
        gi = np.array(gindex[g])
        ks = rankkey[gi]                       # (4, n_sims)
        order = np.argsort(-ks, axis=0)        # best first
        winners[g] = gi[order[0]]
        runners[g] = gi[order[1]]
        thirds[g] = gi[order[2]]
        third_key[g] = np.take_along_axis(ks, order[2:3], axis=0)[0]

    third_stack = np.stack([third_key[g] for g in GROUPS_ORDER])  # (12, n_sims)
    qual_order = np.argsort(-third_stack, axis=0)                 # group-rows ranked per sim

    # round-participation counts
    cnt = {k: np.zeros(nT) for k in ("ko", "r16", "qf", "sf", "final", "champ")}

    def ko_winner(x, y):
        p = 1.0 / (1.0 + 10 ** (-(elo[x] - elo[y]) / KO_DIV))
        return x if rng.random() < p else y

    for s in range(n_sims):
        qgroups = {GROUPS_ORDER[qual_order[r, s]] for r in range(8)}
        alloc = assign_thirds(qgroups) or {}
        if len(alloc) < 8:  # fallback: fill any missing slot with any leftover qualifying third
            leftover = [g for g in qgroups if g not in alloc.values()]
            for slot in SLOT_ALLOWED:
                if slot not in alloc and leftover:
                    alloc[slot] = leftover.pop()

        def team(tok):
            t, v = tok
            if t == "W": return int(winners[v][s])
            if t == "R": return int(runners[v][s])
            return int(thirds[alloc[v]][s])  # third allocated to slot v

        res = {}
        for m, (ta, tb) in R32.items():
            x, y = team(ta), team(tb)
            cnt["ko"][x] += 1; cnt["ko"][y] += 1
            res[m] = ko_winner(x, y)
        for m, (ma, mb) in R16.items():
            x, y = res[ma], res[mb]
            cnt["r16"][x] += 1; cnt["r16"][y] += 1
            res[m] = ko_winner(x, y)
        for m, (ma, mb) in QF.items():
            x, y = res[ma], res[mb]
            cnt["qf"][x] += 1; cnt["qf"][y] += 1
            res[m] = ko_winner(x, y)
        for m, (ma, mb) in SF.items():
            x, y = res[ma], res[mb]
            cnt["sf"][x] += 1; cnt["sf"][y] += 1
            res[m] = ko_winner(x, y)
        fx, fy = res[101], res[102]
        cnt["final"][fx] += 1; cnt["final"][fy] += 1
        cnt["champ"][ko_winner(fx, fy)] += 1

    rows = []
    for i in range(nT):
        rows.append({"team": names[i], "group": grp[i], "elo": round(float(elo[i])),
                     "ko": cnt["ko"][i] / n_sims, "qf": cnt["qf"][i] / n_sims,
                     "sf": cnt["sf"][i] / n_sims, "final": cnt["final"][i] / n_sims,
                     "win": cnt["champ"][i] / n_sims})
    rows.sort(key=lambda r: -r["win"])
    out = {"n_sims": n_sims, "teams": rows}
    json.dump(out, open(SIM_OUT, "w"), ensure_ascii=False, indent=0)
    return rows, n_sims


def main():
    n = 20000
    seed = None
    args = sys.argv[1:]
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1]); skip = True
        elif a.isdigit():
            n = int(a)
    rows, n_sims = simulate(n, seed)
    print(f"=== 2026 WORLD CUP SIMULATION — {n_sims:,} runs ===")
    print(f"{'team':22} {'KO%':>5} {'QF%':>5} {'SF%':>5} {'Fin%':>5} {'WIN%':>6}")
    for r in rows[:16]:
        print(f"{r['team']:22} {100*r['ko']:5.0f} {100*r['qf']:5.0f} {100*r['sf']:5.0f} "
              f"{100*r['final']:5.0f} {100*r['win']:6.1f}")
    bra = next((r for r in rows if r["team"] == "Brazil"), None)
    if bra:
        rank = next(i for i, r in enumerate(rows) if r["team"] == "Brazil") + 1
        print(f"\n\U0001F1E7\U0001F1F7 Brazil: win {100*bra['win']:.1f}% (#{rank}) · final {100*bra['final']:.0f}% · "
              f"semi {100*bra['sf']:.0f}% · last-16+ {100*bra['ko']:.0f}%")
    print(f"\n[saved {SIM_OUT}]")
    print("Note: group-winner/runner-up bracket paths are exact; third-place slotting approximates "
          "FIFA's allocation table; Elo is the seed in wc_teams.json (refresh for live odds).")


if __name__ == "__main__":
    main()
