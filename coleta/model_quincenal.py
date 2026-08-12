"""Out-of-sample horse race: can wholesale prices predict the quincenal INPC?

Protocol (this is the part that matters more than the models):

  * Expanding window, refit at every step. The forecast for quincena t is made
    with parameters estimated ONLY on quincenas <= t-1. Nothing is standardised,
    demeaned, or selected on the full sample.
  * The specification list is fixed a priori. No lag order, no variable subset and
    no hyperparameter is chosen by looking at out-of-sample performance - which
    is itself a form of look-ahead that survives an otherwise clean backtest.
  * Each model is labelled by WHEN it could have been run:
        nowcast      needs wholesale data through the end of quincena t.
                     Available ~8 days before INEGI publishes t.
        mitad        needs only the first 7 days of quincena t.
                     Available while t is still running.
        pronóstico   needs nothing dated inside t at all.
  * `fuga` is a deliberate look-ahead model (it uses x_{t+1}) included as a
    positive control: if the harness were broken, the honest models would score
    like this one.

Usage: python3 model_quincenal.py [slug] [generico_col] [oos_start_year]
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import json

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from inpc import quincenal as Q  # noqa: E402

SLUG = sys.argv[1] if len(sys.argv) > 1 else "aguacate"
GEN = sys.argv[2] if len(sys.argv) > 2 else "045 Aguacate"
OOS_YEAR = int(sys.argv[3]) if len(sys.argv) > 3 else 2011
PESO = sys.argv[4] if len(sys.argv) > 4 else "inpc"   # "equal" = robustez sin ponderadores
HARM = 3          # seasonal harmonics on the 24-quincena year
MIN_TRAIN = 120   # quincenas of training data required before forecasting


# ---------------------------------------------------------------- features
def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ang = 2 * np.pi * (d["quincena_del_anio"] - 1) / 24
    for k in range(1, HARM + 1):
        d[f"sin{k}"] = np.sin(k * ang)
        d[f"cos{k}"] = np.cos(k * ang)
    # The gap must enter LAGGED. gap_t = ln(INPC_t) - ln(W_t) contains the target
    # period's INPC, so a contemporaneous gap is look-ahead - it was in the first
    # version of this script and is exactly the mistake this file is meant to avoid.
    d["gap_lag1"] = d["gap"].shift(1)
    # Rockets and feathers: retail is widely found to pass increases through faster
    # than decreases. Splitting the regressor lets the data say so instead of
    # forcing one slope on both directions.
    d["x_pos"] = d["x_full"].clip(lower=0)
    d["x_neg"] = d["x_full"].clip(upper=0)
    # Signal quality: when markets disagree, the aggregate link is noisier.
    d["disp_c"] = d["disp"] - d["disp"].expanding().mean().shift(1)
    d["x_fwd1"] = d["x_full"].shift(-1)   # look-ahead, for the positive control only
    return d


SEAS = [f"{f}{k}" for k in range(1, HARM + 1) for f in ("sin", "cos")]

SPECS = {
    "media":       ([], "referencia"),
    "SD":          (SEAS, "referencia"),
    "SD-AR":       (SEAS + ["y_lag1", "y_lag2"], "referencia"),
    "x sola":      (["x_full"], "nowcast"),
    "puente":      (SEAS + ["y_lag1", "y_lag2", "x_full"], "nowcast"),
    "puente+MCE":  (SEAS + ["y_lag1", "y_lag2", "x_full", "gap_lag1_c"], "nowcast"),
    "x barajada":  (SEAS + ["y_lag1", "y_lag2", "x_barajada"], "PLACEBO (x sin relación)"),
    "puente+rez":  (SEAS + ["y_lag1", "y_lag2", "x_full", "x_lag1"], "nowcast"),
    "mitad":       (SEAS + ["y_lag1", "y_lag2", "x_parcial"], "mitad"),
    "pronóstico":  (SEAS + ["y_lag1", "y_lag2", "x_lag1", "x_lag2"], "pronóstico"),
    "fuga":        (SEAS + ["y_lag1", "y_lag2", "x_full", "x_fwd1"], "CONTROL (mira al futuro)"),
    # --- candidate improvements, all still nowcasts -------------------------
    "parco":       (["x_full", "x_lag1"], "nowcast"),
    "asimetrico":  (SEAS + ["y_lag1", "y_lag2", "x_pos", "x_neg", "x_lag1"], "nowcast"),
    "mitades":     (SEAS + ["y_lag1", "y_lag2", "x_early", "x_late", "x_lag1"], "nowcast"),
    "mitades+asim": (SEAS + ["y_lag1", "y_lag2", "x_early", "x_late",
                             "x_pos", "x_neg", "x_lag1"], "nowcast"),
    "disp":        (SEAS + ["y_lag1", "y_lag2", "x_full", "x_lag1", "disp_c"], "nowcast"),
}


# Which data each regressor reads, and how far back. `desfase` is the newest
# period the regressor is allowed to touch, counted back from t: 0 means "dated
# inside quincena t", 1 means "t-1 or older", -1 means the future.
FUENTE = {
    **{c: ("calendario", None) for c in SEAS},
    "y_lag1": ("inpc", 1), "y_lag2": ("inpc", 2), "y_lag3": ("inpc", 3),
    "x_full": ("sniim", 0), "x_parcial": ("sniim", 0),
    "x_lag1": ("sniim", 1), "x_lag2": ("sniim", 2), "x_lag3": ("sniim", 3),
    "gap_c": ("mixto", 0), "gap_lag1_c": ("mixto", 1),
    "x_fwd1": ("sniim", -1), "x_barajada": ("placebo", 0),
    "x_pos": ("sniim", 0), "x_neg": ("sniim", 0),
    "x_early": ("sniim", 0), "x_late": ("sniim", 0), "disp_c": ("sniim", 0),
}


def audit(specs: dict) -> list[str]:
    """Structural look-ahead check. INPC-derived regressors must be dated t-1 or
    older; SNIIM-derived ones may be dated inside t (that is the publication-lag
    nowcast, and it is legitimate). Anything else is a bug, not a modelling
    choice. Returns the violations, one line each."""
    bad = []
    for name, (vs, kind) in specs.items():
        if kind.startswith(("CONTROL", "PLACEBO")):
            continue
        for v in vs:
            src, lagv = FUENTE.get(v, ("desconocido", -99))
            if src == "calendario":
                continue
            floor = 0 if src == "sniim" else 1
            if lagv is None or lagv < floor:
                bad.append(f"{name}: {v} lee {src} en t-{lagv} (mínimo t-{floor})")
    return bad


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def run(d: pd.DataFrame, t0: int) -> pd.DataFrame:
    """Recursive one-step-ahead forecasts for every spec."""
    cols = sorted({c for v, _ in SPECS.values() for c in v} | {"y", "gap", "t"})
    rows = []
    ts = d.loc[d["t"] >= t0, "t"].tolist()
    for t in ts:
        tr = d[d["t"] < t]
        te = d[d["t"] == t]
        if te.empty or pd.isna(te["y"].iloc[0]):
            continue
        # Recursive centring of the gap: the mean uses training data only, and the
        # gap itself is lagged one quincena (see build()).
        gmu = tr["gap_lag1"].mean()
        tr = tr.assign(gap_lag1_c=tr["gap_lag1"] - gmu)
        te = te.assign(gap_lag1_c=te["gap_lag1"] - gmu)
        rec = {"t": t, "y": float(te["y"].iloc[0])}
        for name, (vs, _) in SPECS.items():
            need = vs + ["y"]
            trn = tr.dropna(subset=need)
            if len(trn) < MIN_TRAIN or te[vs].isna().any(axis=None):
                rec[name] = np.nan
                continue
            X = np.column_stack([np.ones(len(trn))] + [trn[v].values for v in vs])
            b = ols(X, trn["y"].values)
            xt = np.concatenate([[1.0], te[vs].values.ravel().astype(float)])
            rec[name] = float(xt @ b)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics
def clark_west(y, f_small, f_big):
    """One-sided test that the larger nested model beats the smaller one.

    MSPE-adjusted statistic of Clark & West (2007). Positive t with p<0.05 means
    the extra regressors add genuine out-of-sample content, after correcting for
    the noise the extra parameters mechanically introduce.
    """
    m = ~(np.isnan(y) | np.isnan(f_small) | np.isnan(f_big))
    y, s, b = y[m], f_small[m], f_big[m]
    f = (y - s) ** 2 - ((y - b) ** 2 - (s - b) ** 2)
    n = len(f)
    if n < 20:
        return np.nan, np.nan, n
    t = f.mean() / (f.std(ddof=1) / np.sqrt(n))
    from math import erf, sqrt
    p = 0.5 * (1 - erf(t / sqrt(2)))          # one-sided
    return float(t), float(p), n


def pesaran_timmermann(y, f):
    m = ~(np.isnan(y) | np.isnan(f))
    y, f = np.sign(y[m]), np.sign(f[m])
    n = len(y)
    if n < 20:
        return np.nan, np.nan, n
    acc = float((y == f).mean())
    py, pf = (y > 0).mean(), (f > 0).mean()
    p_ind = py * pf + (1 - py) * (1 - pf)
    v = (p_ind * (1 - p_ind)) / n
    return acc, (float((acc - p_ind) / np.sqrt(v)) if v > 0 else np.nan), n


def score(fc: pd.DataFrame, bench: str = "SD-AR") -> pd.DataFrame:
    y = fc["y"].values
    out = []
    for name, (vs, kind) in SPECS.items():
        f = fc[name].values
        m = ~(np.isnan(y) | np.isnan(f))
        if m.sum() < 20:
            continue
        e = y[m] - f[m]
        rmse = float(np.sqrt((e ** 2).mean()))
        mae = float(np.abs(e).mean())
        eb = y[m] - fc[bench].values[m]
        rb = float(np.sqrt((eb ** 2).mean()))
        acc, pt, _ = pesaran_timmermann(y, f)
        cw_t, cw_p = (np.nan, np.nan)
        if set(SPECS[bench][0]).issubset(set(vs)) and name != bench:
            cw_t, cw_p, _ = clark_west(y, fc[bench].values, f)
        out.append({"modelo": name, "tipo": kind, "n": int(m.sum()), "RMSE": rmse,
                    "MAE": mae, "vs_bench_%": 100 * (rmse / rb - 1),
                    "R2_oos": 1 - (e ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum(),
                    "signo_%": 100 * acc if acc == acc else np.nan,
                    "PT_z": pt, "CW_t": cw_t, "CW_p": cw_p})
    return pd.DataFrame(out).sort_values("RMSE").reset_index(drop=True)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    raw = Q.dataset(SLUG, GEN, peso=PESO)
    d = build(raw)
    # A placebo regressor: the real x_full, circularly shifted by a third of the
    # sample. Same distribution, same seasonality, wrong dates. Any model that
    # scores well with THIS is scoring on the harness, not on the data.
    rng = np.random.default_rng(20260812)
    d["x_barajada"] = np.roll(d["x_full"].values, len(d) // 3)

    viol = audit(SPECS)
    if viol:
        print("AUDITORÍA FALLÓ - look-ahead estructural:")
        for v in viol:
            print("  " + v)
        sys.exit(1)
    print("auditoría estructural: sin look-ahead en los modelos honestos "
          f"({len([k for k, v in SPECS.items() if not v[1].startswith(('CONTROL', 'PLACEBO'))])} modelos revisados)\n")
    t0 = OOS_YEAR * 24
    fc = run(d, t0)
    tab = score(fc)

    lab = {t: Q.qlabel(t) for t in fc["t"]}
    print(f"=== {SLUG} / {GEN} ===")
    print(f"muestra completa: {raw['etiqueta'].iloc[0]} – {raw['etiqueta'].iloc[-1]}"
          f"  ({int((raw['y'].notna() & raw['x_full'].notna()).sum())} quincenas emparejadas)")
    print(f"fuera de muestra: {lab[fc['t'].iloc[0]]} – {lab[fc['t'].iloc[-1]]}"
          f"  ({len(fc)} pronósticos, reestimando en cada paso)")
    print(f"desv. est. del objetivo en la ventana OOS: {fc['y'].std():.2f} pp\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    fc.assign(etiqueta=[lab[t] for t in fc["t"]]).to_csv(
        f"data/curated/oos_{SLUG}{'' if PESO=='inpc' else '_'+PESO}.csv", index=False)
    tab.to_csv(f"data/curated/oos_{SLUG}{'' if PESO=='inpc' else '_'+PESO}_scores.csv", index=False)

    best = tab[tab["tipo"] == "nowcast"].iloc[0]["modelo"]
    print(f"\nmejor nowcast honesto: {best}")
    e = fc["y"] - fc[best]
    print(f"  error medio {e.mean():+.3f} pp, |error| p50 {e.abs().median():.2f} "
          f"p90 {e.abs().quantile(.9):.2f}, peor {e.abs().max():.2f} "
          f"({lab[fc.loc[e.abs().idxmax(),'t']]})")
    tail = fc.tail(4)
    print("  últimas 4 quincenas: " + "  ".join(
        f"{lab[t]} real {yy:+.1f} pred {pp:+.1f}"
        for t, yy, pp in zip(tail["t"], tail["y"], tail[best])))
    print("\n  RMSE por año (mejor nowcast vs SD-AR):")
    yr = fc.assign(anio=(fc["t"] // 24).astype(int))
    for a, g in yr.groupby("anio"):
        r1 = np.sqrt(((g["y"] - g[best]) ** 2).mean())
        r0 = np.sqrt(((g["y"] - g["SD-AR"]) ** 2).mean())
        print(f"    {a}  {r1:5.2f}  vs {r0:5.2f}   ({100*(r1/r0-1):+.0f}%)")
    json.dump({"slug": SLUG, "generico": GEN, "oos_desde": lab[fc['t'].iloc[0]],
               "n_oos": len(fc), "mejor_nowcast": best,
               "rmse": float(tab.set_index('modelo').loc[best, 'RMSE'])},
              open(f"data/curated/oos_{SLUG}_meta.json", "w"), indent=2)
