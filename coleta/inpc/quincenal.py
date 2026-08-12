"""Quincenal (half-month) alignment of wholesale SNIIM prices with the INPC.

The point of this module is the REAL-TIME information set, so read this before
using the output.

INEGI's publication calendar (verified against the press releases):
  * 1Q of month M covers days 1-15 and is published on the 24th of M.
  * 2Q of month M covers day 16 to month end and is published on the 9th of M+1.
Either way the INPC for a quincena appears roughly NINE DAYS AFTER that quincena
has ended. SNIIM publishes daily with a lag of at most a day or two. So there is
a genuine window in which the complete wholesale record for quincena q is known
and the INPC for q is not. Predicting q from wholesale data dated inside q is a
nowcast that exploits the publication lag; it is not look-ahead. Predicting q
from wholesale data dated after q ends WOULD be look-ahead, and nothing here
does that.

Three wholesale variants are built, each labelled by when it becomes knowable:

  w_full     cells averaged over every quote day inside the quincena.
             Knowable ~1 day after the quincena ends, i.e. ~8 days before the
             INPC print. This is the nowcast regressor.
  w_partial  cells averaged over the first 7 calendar days of the quincena only,
             and compared against the FIRST 7 DAYS of the previous quincena so
             the two windows are alike. Knowable on day 8 of the quincena, i.e.
             while it is still running.
  w_lag      w_full shifted one quincena. Knowable before the target quincena
             even begins - the honest pure-forecast regressor.

Index construction is a chained matched-cell Jevons, the same estimator used at
monthly frequency in sniim/aggregate.py:
  1. cell price = geometric mean of the daily variety x market price over the
     window;
  2. per market, geometric mean of log ratios across varieties present in BOTH
     quincenas (matched cells only, so a variety entering or leaving cannot move
     the index);
  3. across markets, weighted geometric mean using INPC city weights;
  4. chain the factors.
Chaining is recursive and touches only t and t-1, so it introduces no look-ahead.
When a quincena has too few matched cells the link bridges to the last quincena
that had enough, and `puente` records how many periods were skipped.

One honest caveat that cannot be engineered away: the market weights come from
INEGI's 2024 basket vintage and are applied over the whole sample, so a 2004
observation is weighted with information published in 2024. Use `peso="equal"`
for a weighting-free robustness run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("sniim")

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "data" / "curated" / "var_market_daily.parquet"
PESOS = ROOT / "data" / "curated" / "pesos_mercado.parquet"
INPC_Q = ROOT / "data" / "inpc" / "inpc_genericos_quincenal.parquet"

MIN_CELDAS = 5          # matched cells required to accept a link
MIN_DIAS_FULL = 4       # quote days required for a full-window cell
MIN_DIAS_PARC = 2       # quote days required for a partial-window cell
PARTIAL_DAYS = 7        # length of the partial window, in calendar days


def qindex(fecha: pd.Series) -> pd.Series:
    """Consecutive integer index over quincenas: ...(2011,1,1)=48265, next 48266."""
    q = np.where(fecha.dt.day <= 15, 0, 1)
    return fecha.dt.year * 24 + (fecha.dt.month - 1) * 2 + q


def qlabel(t: int) -> str:
    y, rem = divmod(int(t), 24)
    m, q = divmod(rem, 2)
    return f"{q + 1}Q {m + 1:02d}/{y}"


def qtimestamp(t: pd.Series | np.ndarray) -> pd.Series:
    """A plottable date per quincena: the first day of its window."""
    t = np.asarray(t, dtype=int)
    y, rem = np.divmod(t, 24)
    m, q = np.divmod(rem, 2)
    return pd.to_datetime({"year": y, "month": m + 1, "day": np.where(q == 0, 1, 16)})


def _cells(d: pd.DataFrame, partial: bool, window: tuple[int, int] | None = None
           ) -> pd.DataFrame:
    """Cell (variety x market x quincena) geometric mean price over the window.

    `window` is a (first_day, last_day) range measured WITHIN the quincena, so
    (1, 7) is the first week of either half-month and (8, 16) the rest. Comparing
    like window against like window across quincenas keeps the intra-period timing
    constant, which matters because INEGI spreads its own price collection across
    the fortnight.
    """
    x = d
    if window is not None:
        dia = x["fecha"].dt.day
        off = np.where(dia <= 15, dia, dia - 15)
        x = x[(off >= window[0]) & (off <= window[1])]
    elif partial:
        dia = x["fecha"].dt.day
        inside = np.where(dia <= 15, dia <= PARTIAL_DAYS, dia - 15 <= PARTIAL_DAYS)
        x = x[inside]
    x = x[x["precio_geo"] > 0]
    # A python lambda inside groupby.agg costs ~30s per category here; taking logs
    # once and using the C mean is the same number, two orders of magnitude faster.
    x = x.assign(_lp=np.log(x["precio_geo"].to_numpy(float)))
    g = (x.groupby(["t", "producto_id", "destino"], sort=False)
          .agg(lp=("_lp", "mean"), n_dias=("fecha", "nunique"))
          .reset_index())
    # The day requirement has to adapt to the category's own reporting cadence, not
    # just to the window length. The granos module (Frijol, Chile seco) is WEEKLY:
    # one quote day per week, so a fortnight holds about two, and a fixed 4-day
    # minimum silently produced zero cells for those two categories. So: require
    # half of what this category typically supplies in this window, capped by the
    # configured floor and never below one.
    if g.empty:
        return g
    typical = float(g["n_dias"].median())
    if window is not None:
        base = max(MIN_DIAS_PARC, min(MIN_DIAS_FULL, (window[1] - window[0] + 1) // 2))
    else:
        base = MIN_DIAS_PARC if partial else MIN_DIAS_FULL
    need = max(1, min(base, int(round(typical / 2)) or 1))
    return g[g["n_dias"] >= need]


EMPTY_LINKS = ["t", "dln", "n_celdas", "sd_mercados", "puente", "segmento"]


def _links(cells: pd.DataFrame, pesos: pd.DataFrame) -> pd.DataFrame:
    """Matched-cell Jevons link factors, bridging over quincenas with too few cells.

    Returns a correctly-typed EMPTY frame when there are no cells at all: a thinly
    quoted category can have nothing inside a narrow half-window, and a bare empty
    DataFrame would blow up the caller's column selection rather than propagating
    as missing data.
    """
    if cells.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in EMPTY_LINKS})
    ts = sorted(cells["t"].unique())
    by_t = {t: g.set_index(["producto_id", "destino"])["lp"] for t, g in cells.groupby("t")}
    w = pesos.set_index("destino")

    out, prev, seg = [], None, 0
    for t in ts:
        a = by_t[t]
        dln, n_common, puente, sd_mkt = np.nan, 0, 0, np.nan
        if prev is not None:
            b = by_t[prev]
            common = a.index.intersection(b.index)
            n_common = len(common)
            if n_common >= MIN_CELDAS:
                r = pd.DataFrame({"dln": (a.loc[common] - b.loc[common]).values},
                                 index=common).reset_index()
                por_mercado = r.groupby("destino")["dln"].mean()
                pw = w.reindex(por_mercado.index)["peso"].fillna(0.0).values
                if pw.sum() <= 0:
                    pw = np.ones(len(por_mercado))
                dln = float(np.average(por_mercado.values, weights=pw))
                sd_mkt = float(por_mercado.std(ddof=1)) if len(por_mercado) > 2 else np.nan
                puente = int(t - prev - 1)
        # `prev` must be the last quincena that is USABLE AS A BASE, not the last
        # one that linked successfully. Tracking the latter deadlocks: the 1998
        # quincenas carry a single cell, so the first `prev` never matched five
        # cells against anything and every later link was compared against it,
        # returning n_celdas <= 1 for the whole sample.
        if len(a) >= MIN_CELDAS:
            if np.isnan(dln) and prev is not None:
                seg += 1        # chain restarts here; levels are not comparable across it
            prev = t
        out.append({"t": t, "dln": dln, "n_celdas": n_common, "sd_mercados": sd_mkt,
                    "puente": puente, "segmento": seg})
    return pd.DataFrame(out)


def _chain(links: pd.DataFrame, col: str, segcol: str = "segmento") -> pd.Series:
    """Cumulative sum of log links -> log level, restarting at each chain break.

    Levels are only comparable WITHIN a segment; `segmento` in the output says
    which one each quincena belongs to, and anything using the level (the gap
    term) has to respect that.
    """
    lvl, acc, cur = {}, 0.0, None
    for t, v, sg in zip(links["t"], links[col], links[segcol]):
        if sg != cur:            # new segment: re-anchor
            cur, acc = sg, 0.0
            lvl[t] = 0.0 if not np.isnan(v) else np.nan
            if not np.isnan(v):
                continue
        if np.isnan(v):
            lvl[t] = np.nan
            continue
        acc += v
        lvl[t] = acc
    return pd.Series(lvl, name=col)


def wholesale_quincenal(slug: str, peso: str = "inpc",
                        windows: tuple[int, ...] = ()) -> pd.DataFrame:
    """Quincenal wholesale log-index (full and partial windows) for one category."""
    d = pd.read_parquet(DAILY, columns=["categoria", "producto_id", "destino",
                                        "fecha", "precio_geo"])
    d = d[d["categoria"] == slug].copy()
    if d.empty:
        raise ValueError(f"no daily wholesale rows for categoria={slug!r}")
    d["fecha"] = pd.to_datetime(d["fecha"])
    d["t"] = qindex(d["fecha"])

    pesos = pd.read_parquet(PESOS)[["destino", "peso_inpc", "peso_equal"]].copy()
    # peso_equal arrives as decimal.Decimal from the parquet writer, which numpy
    # refuses to multiply against floats; coerce both columns.
    for c in ("peso_inpc", "peso_equal"):
        pesos[c] = pd.to_numeric(pesos[c], errors="coerce").astype(float)
    pesos["peso"] = pesos["peso_inpc"] if peso == "inpc" else pesos["peso_equal"]

    lf = _links(_cells(d, partial=False), pesos).rename(
        columns={"dln": "dln_full", "n_celdas": "n_celdas_full"})
    lp = _links(_cells(d, partial=True), pesos).rename(
        columns={"dln": "dln_parcial", "n_celdas": "n_celdas_parcial",
                 "puente": "puente_parcial"})

    le = _links(_cells(d, partial=False, window=(1, PARTIAL_DAYS)), pesos).rename(
        columns={"dln": "dln_early"})
    ll = _links(_cells(d, partial=False, window=(PARTIAL_DAYS + 1, 16)), pesos).rename(
        columns={"dln": "dln_late"})
    out = lf.merge(lp[["t", "dln_parcial", "n_celdas_parcial"]], on="t", how="outer")
    out = out.merge(le[["t", "dln_early"]], on="t", how="outer")
    out = out.merge(ll[["t", "dln_late"]], on="t", how="outer")
    # Release vintages: day 1..k of the fortnight compared against day 1..k of the
    # previous one, so the intra-period window is held constant. x_w5 is knowable
    # on day 6, x_w10 on day 11, x_full about a day after the fortnight closes.
    for k in windows:
        wk = _links(_cells(d, partial=False, window=(1, k)), pesos).rename(
            columns={"dln": f"dln_w{k}"})
        out = out.merge(wk[["t", f"dln_w{k}"]], on="t", how="outer")
    out = out.sort_values("t").reset_index(drop=True)
    out["lw_full"] = _chain(out, "dln_full").reindex(out["t"]).values
    # Diagnostics: how much of each quincena we actually observed.
    dias = d.groupby("t")["fecha"].nunique().rename("n_dias")
    out = out.merge(dias, left_on="t", right_index=True, how="left")
    return out


def inpc_quincenal(generico_col: str) -> pd.DataFrame:
    """Published quincenal INPC for one genérico, as a log level."""
    q = pd.read_parquet(INPC_Q)
    if generico_col not in q.columns:
        raise KeyError(f"{generico_col!r} not in {[c for c in q.columns][:6]}...")
    x = q[["anio", "mes", "quincena", generico_col]].dropna().copy()
    x["t"] = x["anio"] * 24 + (x["mes"] - 1) * 2 + (x["quincena"] - 1)
    x = x.rename(columns={generico_col: "inpc"})
    x["lp"] = np.log(x["inpc"])
    return x[["t", "inpc", "lp"]].sort_values("t").reset_index(drop=True)


CACHE_DIR = ROOT / "data" / "curated" / "quincenal_cache"


def dataset(slug: str, generico_col: str, peso: str = "inpc",
            drop_incomplete_tail: bool = True,
            windows: tuple[int, ...] = (), cache: bool = True) -> pd.DataFrame:
    """Aligned quincenal panel: target, the three wholesale variants, diagnostics.

    Columns
      y            100 * Δln INPC for the quincena (the target)
      x_full       100 * Δln wholesale, full window   -> nowcast regressor
      x_parcial    100 * Δln wholesale, first 7 days  -> mid-window regressor
      x_lag1..3    lags of x_full                      -> pure-forecast regressors
      gap          ln(INPC level) − ln(wholesale level), both in logs, demeaned
                   RECURSIVELY at use time, never here (see model_quincenal.py)
    """
    key = f"{slug}__{peso}__w{'-'.join(map(str, windows)) or 'none'}"
    cpath = CACHE_DIR / f"{key}.parquet"
    if cache and cpath.exists():
        src = DAILY.stat().st_mtime
        if cpath.stat().st_mtime > src:      # stale if the daily store moved on
            return pd.read_parquet(cpath)
    w = wholesale_quincenal(slug, peso=peso, windows=windows)
    p = inpc_quincenal(generico_col)
    df = p.merge(w, on="t", how="inner").sort_values("t").reset_index(drop=True)

    # A gap-free quincena grid, so a differenced value is never taken across a hole.
    grid = pd.DataFrame({"t": range(int(df["t"].min()), int(df["t"].max()) + 1)})
    df = grid.merge(df, on="t", how="left")

    df["y"] = 100 * df["lp"].diff()
    df["x_full"] = 100 * df["dln_full"]
    df["x_parcial"] = 100 * df["dln_parcial"]
    df["x_early"] = 100 * df["dln_early"]
    df["x_late"] = 100 * df["dln_late"]
    df["disp"] = 100 * df["sd_mercados"]
    for k in windows:
        df[f"x_w{k}"] = 100 * df[f"dln_w{k}"]
        df[f"x_w{k}_lag1"] = df[f"x_w{k}"].shift(1)
    df["x_full_lag1"] = df["x_full"].shift(1)
    for k in (1, 2, 3):
        df[f"x_lag{k}"] = df["x_full"].shift(k)
        df[f"y_lag{k}"] = df["y"].shift(k)
    df["gap"] = df["lp"] - df["lw_full"]
    df["quincena_del_anio"] = ((df["t"] % 24) + 1).astype(int)
    df["fecha"] = qtimestamp(df["t"])
    df["etiqueta"] = [qlabel(t) for t in df["t"]]

    if drop_incomplete_tail:
        # The running quincena has only part of its quote days; the median full
        # quincena has ~11. Drop a tail quincena that is clearly still open.
        med = df["n_dias"].median()
        while len(df) and (pd.isna(df["n_dias"].iloc[-1]) or df["n_dias"].iloc[-1] < 0.6 * med):
            df = df.iloc[:-1]
    df = df.reset_index(drop=True)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cpath, index=False)
    return df
