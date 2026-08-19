"""A real-time fitted line for the published fortnightly CPI change.

The chartbook's three series are all raw. This module adds a fourth: a model that maps
the daily wholesale panel, plus the last CPI print that had actually been released, onto
the quantity the orange dots measure — the generic's index this fortnight against the
fortnight two prints earlier.

WHAT IS ESTIMATED
-----------------
For each fortnight q, with close date T_q,

    y_q = 100 * (ln INPC_q - ln INPC_{q-2})

is regressed on four things, every one of them knowable on day T_q:

    w      the wholesale 15-day mean ending T_q against the 15-day mean 30 days
           earlier. This is the regressor that MATCHES the target's construction: a
           fortnightly CPI print is an average of quotes collected across the fortnight,
           not a reading at its close, so the comparable wholesale number is also an
           average over the fortnight.
    w_lag  the same thing a fortnight earlier — pass-through is not instantaneous, and
           for several generics the previous fortnight carries more signal than the
           current one.
    c7     the 7-day mean against 30 days earlier. Short window, so it leads and
           overshoots; it is what tells the model that a turn is under way.
    y_prev the last CPI change INEGI had actually PUBLISHED as of T_q.

The publication lag is enforced, not assumed away. INEGI releases a fortnight's index
about nine days after it closes, so `y_prev` steps up 10 days after each close and not
before. On day T_q the newest available print is therefore q-1, never q.

NO LOOK-AHEAD
-------------
Every plotted point is out of sample. The coefficients used to predict fortnight q are
estimated only on fortnights whose prints had been released before q closed — that is,
on q-1 and earlier. They are re-estimated at every fortnight on an expanding window
capped at ten years, and the daily line switches to a new coefficient vector on the day
that vector's last training observation was published. A fit that used the whole sample
would look far better than this and would mean nothing.

Ridge rather than OLS: `w`, `w_lag` and `c7` measure overlapping windows of the same
price series and are strongly collinear, which makes plain OLS coefficients swing between
refits. The penalty is chosen by leave-one-out on the training window alone.

WHAT IS REPORTED
----------------
Three nested models are scored on the same out-of-sample points, because "the model fits"
is not interesting on its own:

    m0   the 30-day wholesale line already on the chart, rescaled onto CPI units
    m1   the three wholesale features
    m2   m1 plus the last published print

m2 is what is drawn. The gap m1 -> m2 is how much of the fit is CPI persistence rather
than wholesale information, and on some generics that gap is most of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

PUB_LAG = pd.Timedelta(days=10)      # INEGI releases ~9 days after a fortnight closes
MIN_TRAIN = 26                       # fortnights, i.e. about thirteen months
MAX_TRAIN = 240                      # ten years; older regimes stop being informative
ALPHAS = np.logspace(-3, 3, 25)

M0, M1, M2 = ["c30"], ["w", "wl", "c7"], ["w", "wl", "c7", "yprev"]
MODEL = "#8c3a1b"                    # the CPI orange, darkened: same quantity, modelled


def features(f: pd.DataFrame, lp: pd.Series, gap: float,
             cpi: pd.DataFrame | None) -> pd.DataFrame:
    """Daily design matrix. `f` carries the chart's own c30/c7, `lp` the calendar-filled
    log price behind them, `gap` the series' quoting cadence in days."""
    need15 = max(2, int(round(0.6 * 15 / gap)))
    m15 = lp.rolling("15D", min_periods=need15).mean()
    w = 100 * (m15 - m15.shift(30, freq="D").reindex(m15.index))
    X = pd.DataFrame({"w": w, "wl": w.shift(15, freq="D").reindex(w.index),
                      "c7": f["c7"], "c30": f["c30"]}, index=f.index)
    # last published print, as a step function of the calendar
    X["yprev"] = np.nan
    if cpi is not None and len(cpi):
        c = cpi.dropna(subset=["chg"]).sort_values("fecha")
        rel = pd.Series(c.chg.to_numpy(),
                        index=pd.DatetimeIndex(c.fecha) + PUB_LAG).sort_index()
        rel = rel[~rel.index.duplicated(keep="last")]
        X["yprev"] = rel.reindex(X.index.union(rel.index)).ffill().reindex(X.index)
    return X


def _fit(Xtr: np.ndarray, ytr: np.ndarray):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
    return mu, sd, m.coef_, float(m.intercept_), float(m.alpha_)


# how each regressor is named on the page
NICE = {"w": "wholesale", "wl": "wholesale(t-1)", "c7": "7d edge",
        "yprev": "CPI(t-1)", "c30": "wholesale 30d"}


def equation(st: dict) -> str:
    """The fit as it currently stands, in the units of the chart: percentage points of
    CPI change per percentage point of the regressor. The coefficients move at every
    refit — this is the vector in force at the last fortnight, not a fixed model."""
    if not st.get("beta"):
        return ""
    out = "CPI  =  "
    for k, (name, b) in enumerate(st["beta"].items()):
        out += ("" if k == 0 else (" + " if b >= 0 else " − ")) + \
               f"{abs(b) if k else b:.2f}·{NICE.get(name, name)}"
    b0 = st.get("b0", 0.0)
    out += f" {'+' if b0 >= 0 else '−'} {abs(b0):.2f}"
    return out


def run(X: pd.DataFrame, cpi: pd.DataFrame, lo: pd.Timestamp,
        cols: list[str] = M2, score_from: pd.Timestamp | None = None):
    """Real-time fit. Returns (daily prediction, out-of-sample frame, stats)."""
    score_from = score_from or lo
    if cpi is None or not len(cpi):
        return None, None, {}
    y = (pd.Series(cpi.chg.to_numpy(), index=pd.DatetimeIndex(cpi.fecha))
         .sort_index().dropna())
    y = y[~y.index.duplicated(keep="last")]

    Xq = X.reindex(y.index)[cols]
    ok = Xq.notna().all(axis=1)
    y, Xq = y[ok], Xq[ok]
    if len(y) < MIN_TRAIN + 4:
        return None, None, {}

    T = y.index
    Xa, ya = Xq.to_numpy(float), y.to_numpy(float)

    # a coefficient vector trained on the first k fortnights becomes usable on the day
    # print k-1 is released, and stays the newest one until print k is released
    ks = [k for k in range(MIN_TRAIN, len(T) + 1)
          if T[k - 1] + PUB_LAG >= score_from - pd.Timedelta(days=45)]
    if not ks:
        return None, None, {}
    avail, MU, SD, C, B0 = [], [], [], [], []
    oos = {}
    alpha = n_train = 0
    for k in ks:
        lo_i = max(0, k - MAX_TRAIN)
        mu, sd, co, b0, alpha = _fit(Xa[lo_i:k], ya[lo_i:k])
        n_train = k - lo_i
        avail.append(T[k - 1] + PUB_LAG)
        MU.append(mu); SD.append(sd); C.append(co); B0.append(b0)
        if k < len(T):                       # the genuine out-of-sample prediction
            oos[T[k]] = float(((Xa[k] - mu) / sd) @ co + b0)

    avail = pd.DatetimeIndex(avail)
    MU, SD, C, B0 = map(np.asarray, (MU, SD, C, B0))

    d = X[cols].dropna()
    d = d[d.index >= min(lo, avail.min())]
    j = np.searchsorted(avail.to_numpy(), d.index.to_numpy(), side="right") - 1
    live = d.index >= avail.min()
    d, j = d[live], j[live]
    pred = pd.Series(
        (((d.to_numpy(float) - MU[j]) / SD[j]) * C[j]).sum(1) + B0[j], index=d.index)

    o = pd.DataFrame({"y": y.reindex(list(oos)), "yhat": pd.Series(oos)}).dropna()
    o = o[o.index >= score_from]
    # the standardised coefficients put back into the units of the chart, so the equation
    # printed on the page reads in points of CPI per point of the regressor
    beta = C[-1] / SD[-1]
    st = {"n_oos": len(o), "alpha": alpha, "n_train": n_train,
          "beta": dict(zip(cols, beta)), "b0": float(B0[-1] - beta @ MU[-1]),
          "start": pred.index.min() if len(pred) else None}
    if len(o) > 6:
        st["corr"] = float(np.corrcoef(o.y, o.yhat)[0, 1])
        st["rmse"] = float(np.sqrt(((o.y - o.yhat) ** 2).mean()))
        st["sd_y"] = float(o.y.std())
    return pred, o, st


def score_ladder(X: pd.DataFrame, cpi: pd.DataFrame, lo: pd.Timestamp) -> dict:
    """m0 / m1 / m2 on identical out-of-sample points, so the comparison is honest.

    Returns {tag: (daily prediction, out-of-sample frame, stats)}. The three are scored on
    the intersection of their out-of-sample dates — m0 needs one feature and m2 needs
    four, so left alone they would not be answering on the same questions."""
    out = {tag: run(X, cpi, lo, cols)
           for tag, cols in (("m0", M0), ("m1", M1), ("m2", M2))}
    frames = [o for _, o, _ in out.values() if o is not None and len(o)]
    if len(frames) == 3:
        common = frames[0].index
        for f in frames[1:]:
            common = common.intersection(f.index)
        for tag, (pred, o, st) in out.items():
            o = o.loc[common]
            if len(o) > 6:
                st = {**st, "n_oos": len(o),
                      "corr": float(np.corrcoef(o.y, o.yhat)[0, 1]),
                      "rmse": float(np.sqrt(((o.y - o.yhat) ** 2).mean())),
                      "sd_y": float(o.y.std())}
            out[tag] = (pred, o, st)
    return out
