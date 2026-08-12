# What is in this repo, and how to reproduce every output

Everything below runs from the repo root against `data/`. Nothing needs the cloud session.

## Order of operations

```
run.py catalog / mapping        product catalogs and the SNIIM -> INPC crosswalk
run.py backfill                 one-time history load (resumable)
run.py update                   daily trailing 14-day refresh, calls build itself
run.py build / verify           rebuild data/curated from data/raw, then QA it
inpc_run.py fetch / precios     INEGI: fortnightly generic indices, retail MXN/kg
facts.py                        recompute every number quoted anywhere -> facts.json
```

`facts.py` is the one to run after any data change: every chart and the deck read
`data/curated/facts.json` rather than recomputing, so a stale facts file is the single
thing that can make two outputs disagree.

## Collection (was already here)

| File | Does |
|---|---|
| `run.py` | CLI: catalog, mapping, backfill, update, build, verify, status, export |
| `sniim/http.py` | Session, rate limiter, shared concurrency gate, 503 backoff |
| `sniim/frutas.py`, `sniim/granos.py` | the two SNIIM modules (daily / weekly) |
| `sniim/store.py` | parquet store, append-only manifest, advisory lock |
| `sniim/aggregate.py` | chained matched-cell index at monthly frequency |
| `inpc_run.py`, `inpc/*` | INEGI collection and validation |

## Alignment and modelling

| File | Writes | Notes |
|---|---|---|
| `inpc/quincenal.py` | (library) | fortnightly alignment. Cells, matched-cell links, chaining, release vintages. Everything downstream depends on it |
| `facts.py` | `data/curated/facts.json` | single source for every quoted statistic; scores any two forecasts on the fortnights where BOTH exist |
| `forecast_system.py` | `system_forecasts.csv`, `system_scores.csv`, `jitomate_system.csv` | the 32-generic out-of-sample run, rolling 120 fortnights |
| `aggregate_system.py` | `system_aggregate.csv`, `system_aggregate_scores.csv` | ragged-edge bottom-up aggregates: all 32 / top 10 / top 5 |
| `model_jitomate.py`, `model_quincenal.py` | stdout | standalone horse races, kept for reference |
| `tune_jitomate.py` | stdout | 10 aggregation variants; all tied or worse than base |
| `iv_jitomate.py` | `iv_jitomate.csv` | split-half instrument: index reliability 0.966 |
| `city_bottomup.py` | stdout | city-level bottom-up test |

## Exports

| File | Writes |
|---|---|
| `export_panel_xlsx.py` | **`precios_mayoreo_diario.xlsx`** — the published file. 4 sheets: LEEME, precios_diarios (7,249 rows x 32 generics), n_mercados, cobertura |
| `export_generic_xlsx.py` | `<generic>_data.xlsx`, one workbook per generic named on the command line |
| `export_jitomate_xlsx.py` | `jitomate_data.xlsx` — 9 sheets including a CHECKS sheet of live formulas |

`export_panel_xlsx.py` is what the daily workflow publishes. It needs `xlsxwriter`, which
is why `requirements.txt` gained it.

## Charts

| File | Writes |
|---|---|
| `chartbook.py` | `chartbook_frutas_verduras.pdf` — 32 pages, one per generic, `--from YYYY`, `--rows` |
| `plot_scatter_corr.py` | `charts/scatter_var_corr.png` — variance share vs correlation, `--scale linear\|symlog` |
| `plot_system.py` | `charts/system_variance.png`, `system_aggregate.png`, `system_grid_a/b.png` |
| `plot_jitomate.py` | `charts/jitomate_varieties.png`, `jitomate_margin.png`, `jitomate_index_ratio.png` |
| `plot_jitomate_daily.py` | `charts/jitomate_daily.png` — daily level + 30-day change vs CPI dots |
| `plot_jitomate_dls.py` | `charts/jitomate_roll5y.png` (`roll`) or `jitomate_dls.png` (`dls`), `--system` |
| `plot_jitomate_gain.py` | `charts/jitomate_gain.png` — with and without wholesale |
| `plot_levels_png.py`, `plot_mom_png.py`, `plot_mom_log_png.py`, `plot_logratio_png.py` | calabacita charts (Spanish) |
| `plot_category.py`, `plot_yoy_png.py`, `plot_oos_png.py`, `plot_jitomate_oos.py` | earlier one-offs |
| `build_deck.js` | `produce_cpi_nowcast.pptx` — 13 slides. Needs `npm i pptxgenjs`, reads facts.json |

Charts assume `charts/` exists: `mkdir -p charts`.

## Publication

| File | Does |
|---|---|
| `.github/workflows/daily.yml` | the daily job: restore store, update, gate, build, publish |
| `scripts/check_freshness.py` | the publish gate — recency, per-generic coverage, continuity |
| `SETUP.md` | one-time GitHub setup |

## Three traps that cost real time

1. **A fortnight is labelled by its FIRST day but summarises prices through its LAST.**
   Plotting a CPI dot at the label puts it half a month before the prices it describes; on
   jitomate that alone moves the measured correlation from 0.86 to 0.40. Anything plotting
   CPI against a daily series must use the closing date.

2. **The chained matched-cell index is right in CHANGES, arbitrary in LEVEL.** Composition
   drift accumulates (~0.10 in logs over 28 years here). Use `p_index_ponderado` /
   `ws_nivel_directo_mxn_kg` for pesos and the chained series only for changes. Reading the
   chain as a level is what produced a 1.69x retail margin against a true 1.37x.

3. **Cadence is not uniform.** Frijol and Chile seco are the weekly granos module; a
   30-day window that expects daily quoting silently returns nothing for them, and a fixed
   minimum-days filter deletes the 1998-1999 sample, when SNIIM reported weekly. Every
   rolling window in this repo derives its minimum from the series' own median gap.

Related: a day on which only a handful of markets quoted is not a national price. Manzana
2025-05-05 carried 2 markets out of a usual 78. Days below 40% of a generic's typical
market count are dropped in the chart code and gate on it in `check_freshness.py`.
