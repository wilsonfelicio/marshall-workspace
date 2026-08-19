"""Tests for the fitted CPI line. Network-free, synthetic data.

The whole value of that line is the claim that it never sees the print it draws. These
tests attack that claim directly rather than inspecting the code:

  no_lookahead      move one fortnight's published value by a mile. Its own fitted point
                    must not budge, and neither must anything before it. Everything after
                    it may move, because a later fit legitimately trains on it.
  publication_lag   `yprev` must step 10 days AFTER a fortnight closes, not on the day.
                    On the close itself the newest value available is the previous print.
  recovers_signal   on data built as y = 0.8*w + noise, the out-of-sample correlation has
                    to come out high. A test that only checks for absence of cheating
                    passes just as well on a model that predicts nothing.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import nowcast  # noqa: E402

LO = pd.Timestamp("2018-01-01")


def synth(seed=0, n_years=12):
    """A daily log-price and a fortnightly CPI built to depend on it."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=365 * n_years, freq="D")
    lp = pd.Series(np.cumsum(rng.normal(0, 0.012, len(idx))), index=idx)
    m30 = lp.rolling("30D", min_periods=10).mean()
    m7 = lp.rolling("7D", min_periods=3).mean()
    f = pd.DataFrame(
        {"c30": 100 * (m30 - m30.shift(30, freq="D").reindex(m30.index)),
         "c7": 100 * (m7 - m7.shift(30, freq="D").reindex(m7.index))}, index=idx)

    # fortnight closes: the 15th and the last day of each month
    closes = pd.DatetimeIndex(sorted(
        set(pd.date_range(idx[0], idx[-1], freq="MS") + pd.Timedelta(days=14))
        | set(pd.date_range(idx[0], idx[-1], freq="ME"))))
    closes = closes[(closes >= idx[0]) & (closes <= idx[-1])]
    m15 = lp.rolling("15D", min_periods=4).mean()
    w = 100 * (m15 - m15.shift(30, freq="D").reindex(m15.index))
    chg = 0.8 * w.reindex(closes) + rng.normal(0, 1.0, len(closes))
    cpi = pd.DataFrame({"fecha": closes, "chg": chg.to_numpy()}).dropna()
    return f, lp, cpi


def _pred(f, lp, cpi):
    X = nowcast.features(f, lp, 1.0, cpi)
    return nowcast.run(X, cpi, LO, nowcast.M2)


def test_publication_lag_holds_yprev_back():
    f, lp, cpi = synth()
    X = nowcast.features(f, lp, 1.0, cpi)
    c = cpi.sort_values("fecha").reset_index(drop=True)
    q = 200
    close, prev = c.fecha[q], c.chg[q - 1]
    # on the close itself, and for nine more days, the newest print is the previous one
    assert abs(X.yprev.loc[close] - prev) < 1e-12, "the model saw its own print"
    assert abs(X.yprev.loc[close + pd.Timedelta(days=9)] - prev) < 1e-12
    # on the tenth day the print for this fortnight lands
    assert abs(X.yprev.loc[close + pd.Timedelta(days=10)] - c.chg[q]) < 1e-12


def test_no_lookahead_under_perturbation():
    f, lp, cpi = synth()
    base, o0, st0 = _pred(f, lp, cpi)
    assert st0["n_oos"] > 40 and len(base) > 500

    # move one published value far out of its own distribution
    hit = o0.index[len(o0) // 2]
    bad = cpi.copy()
    bad.loc[bad.fecha == hit, "chg"] += 50.0
    pert, o1, _ = _pred(f, lp, bad)

    assert abs(o1.yhat.loc[hit] - o0.yhat.loc[hit]) < 1e-8, \
        "the fitted point for a fortnight moved when that fortnight's print moved"
    before = o0.index[o0.index < hit]
    assert np.allclose(o1.yhat.loc[before], o0.yhat.loc[before], atol=1e-8), \
        "an earlier fitted point moved when a later print moved"
    # and the daily line: nothing up to the close may move, everything after may
    upto = base.index[base.index <= hit]
    assert np.allclose(pert.loc[upto], base.loc[upto], atol=1e-8)
    after = base.index[base.index > hit + pd.Timedelta(days=10)]
    assert not np.allclose(pert.loc[after], base.loc[after], atol=1e-8), \
        "later fits ignored the perturbed print entirely — check the training window"


def test_recovers_a_real_signal():
    f, lp, cpi = synth(seed=7)
    _, o, st = _pred(f, lp, cpi)
    assert st["n_oos"] > 40
    assert st["corr"] > 0.85, f"out-of-sample corr only {st['corr']:.2f}"
    assert st["rmse"] < 0.6 * st["sd_y"], "no better than quoting the mean"


def test_pure_noise_is_not_fitted():
    """The mirror image: when the CPI has nothing to do with the prices, the model must
    not manufacture a fit. In sample it always could; out of sample it must not."""
    f, lp, cpi = synth(seed=3)
    rng = np.random.default_rng(11)
    cpi = cpi.assign(chg=rng.normal(0, 3.0, len(cpi)))
    _, o, st = _pred(f, lp, cpi)
    assert abs(st["corr"]) < 0.35, f"fitted noise at corr {st['corr']:.2f}"


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception:
                fails += 1
                print(f"  FAIL {name}")
                traceback.print_exc()
    print("\nall passed" if not fails else f"\n{fails} failed")
    raise SystemExit(1 if fails else 0)
