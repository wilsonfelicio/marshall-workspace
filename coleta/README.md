# SNIIM — Mexican wholesale food price collection

Builds and maintains a long daily time series of Mexican wholesale fruit,
vegetable and bean prices from **SNIIM** (Sistema Nacional de Información e
Integración de Mercados, Secretaría de Economía), aggregated into the 32 INPC
food generics used for tracking Mexican food inflation.

Source: <https://www.economia-sniim.gob.mx/nuevo/>

- **History available:** 1998 → present
- **Frequency:** daily (weekdays) for fruit and vegetables; weekly survey for frijol
- **Granularity:** product variety × origin state × destination market (49 markets)
- **Unit:** pesos per kilogram (SNIIM's own `Kilogramo (calculado)` normalisation)

---

## 1. Quick start

```bash
cd /Users/wilsonfelicio/Downloads/coleta

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py catalog
python run.py mapping
python run.py backfill
python run.py build
python run.py verify
```

In order: scrape the product and market catalogs (~5s); generate
`config/mapping_inpc.csv` (instant — it refuses to overwrite an existing one, so
on a fresh checkout it just prints the report); load the history (**one-time, see
§4 for runtime**); build the DuckDB tables and aggregates (~1 min); run the QA
checks.

Before committing to the full backfill, do a one-year pilot. It costs about 25
minutes, exercises all 222 products, and tells you whether anything on the site
has shifted:

```bash
python run.py backfill --module frutas --start-year 2015 --end-year 2015
python run.py build
python run.py verify
```

If `verify` reports no `PROBLEM` lines, run the full `backfill`.

Then schedule the recurrent part (§5) — `update` is the daily incremental:

```bash
python run.py update
```

> **Paste these blocks as-is.** They deliberately contain no `#` comments.
> macOS zsh has `interactive_comments` **off** by default, so a `#` typed at an
> interactive prompt is passed through as a literal argument rather than starting
> a comment — and a trailing `(~5s)` is parsed as a glob qualifier. Annotated
> commands fail in confusing ways when pasted into zsh.

---

## 2. How the site actually works

Worth knowing, because it is not documented anywhere and it constrains the design.

SNIIM is an ASP.NET 2.0 / IIS 6.0 application. The query forms are WebForms with
`__VIEWSTATE`, but the **results pages accept plain GET requests** with no
session, no cookie and no viewstate. That is what makes this collectable without
a browser.

### Fruit and vegetables — daily, date range

```
ResultadosConsultaFechaFrutasYHortalizas.aspx
  ?fechaInicio=01/01/2005&fechaFinal=31/12/2005
  &ProductoId=133
  &OrigenId=-1&Origen=Todos
  &DestinoId=-1&Destino=Todos
  &PreciosPorId=2
  &RegistrosPorPagina=20000
```

### Frijol — weekly, week-of-month selector

Frijol is **not** in the fruit and vegetable module. It lives in Granos Básicos,
which has no date-range query at all and uses entirely different parameter
names:

```
ResultadosConsultaFechaGranos.aspx
  ?Semana=1&Mes=7&Anio=2026
  &ProductoId=347
  &OrigenId=-1&Origen=Todos
  &DestinoId=-1&Destino=Todos
  &PreciosPorId=2&RegistrosPorPagina=20000
```

### Constraints discovered by probing

| Finding | Consequence for the design |
|---|---|
| `ProductoId=-1` ("Todos") is rejected: *"No puede seleccionar todos los productos todos los origenes y todos los destinos"* | The collection loop must be **per product**. There is no bulk download. |
| `RegistrosPorPagina` above 1000 defeats server-side pagination entirely | No `__VIEWSTATE` postback scraping needed. Set to 20000. |
| `PreciosPorId=2` returns pesos/kg but still reports the surveyed package in `Presentación` | Always use `2`. With `1` you get "Caja de 13 kg." mixed with "Kilogramo" in the same column and the numbers are not comparable. |
| Granos `Mes` must be **numeric**; the word `Julio` returns HTTP 500 | See `sniim/granos.py`. |
| Granos `Semana` is **week-of-month (1–5)**, not ISO week, and the week-1 anchor is irregular | If the 1st falls Mon/Tue/Wed, week 1 starts on the Monday *before* the 1st; otherwise on the Monday *after*. Verified against the live site for 13 months spanning 1999–2026. `weeks_in_range()` enumerates all 5 slots per month and deduplicates on the Monday, which is provably gap-free: 1,514 slots → 1,514 distinct weeks for 1998–2026. Getting this wrong silently skips weeks. |
| Server returns **503** when pushed faster than ~1 req/s sustained | `sniim/http.py` rate-limits, retries with exponential backoff, and self-throttles after consecutive 5xx. Observed in testing and recovered automatically. |
| Bad parameter combinations return **500**, not an error page | Treated as retryable-then-permanent so a bad job does not stall the run. |
| The results page **echoes the date range it honoured** | Used as a guard against silently ignored parameters. |

---

## 3. What gets stored

```
data/
├── catalog/                     scraped dropdowns (products, origins, markets)
├── raw/
│   ├── frutas/producto_id=133/anio=2005/part.parquet
│   └── granos/producto_id=347/anio=2005/part.parquet
├── curated/                     aggregates as Parquet (+ CSV for the small ones)
├── manifest.csv                 append-only job ledger — drives --resume
└── sniim.duckdb                 views + materialised tables
```

### Raw observation columns

`modulo, producto_id, producto, calidad, grupo, fecha, presentacion, origen,
destino, destino_estado, destino_mercado, precio_min, precio_max, precio_frec,
unidad, obs, periodo, fetched_at`

One row per **date × presentation × origin state × destination market**.
`fetched_at` is retained so you can tell when a revised row was picked up.

### Curated tables

| Table | Grain | Notes |
|---|---|---|
| `cat_market_daily` | category × market × day | stage 1 of every national figure |
| `cat_national_daily` | category × day | across markets |
| `var_national_monthly` | **variety** × month | shows which variety drives a category |
| `cat_market_monthly` | category × market × month | |
| `cat_national_monthly` | category × month | price levels, base-100, MoM, YoY |
| `cat_index_monthly` | category × month | **chained Jevons index — use this for inflation** |

Every table carries both `precio` (arithmetic mean) and `precio_geo` (geometric
mean) — see §4 for which to use and why.

`cat_index_monthly` also carries the columns you need to know whether to trust a
number: `meses_puente` (how many calendar months this index step spans — 1 is a
normal monthly link, more means a seasonal gap was bridged),
`n_variedades_pareadas` (how many varieties were present at both ends of the
step), `cadena_rota` (no variety matched, so the level was carried forward),
`mes_base` and `base_es_fallback` (whether this category is on the configured
base month or its own first month).

### Job statuses

`data/manifest.csv` records one row per fetched period. Only **`ok`** and
**`empty`** are terminal — everything else is retried automatically on the next
`backfill`. This matters: a period must never be skipped forever on the strength
of an incomplete fetch.

| Status | Meaning | Retried? |
|---|---|---|
| `ok` | complete, and the period is closed | no |
| `empty` | usable results page, genuinely no rows | no |
| `open` | period not finished yet (current year, current week) | **yes** |
| `truncated` | response paginated and could not be split further | **yes** |
| `unusable` | HTTP 200 but not a results page (error/maintenance) | **yes** |
| `range_mismatch` | server did not honour the dates we asked for | **yes** |
| `malformed` | parser rejected rows — table layout may have changed | **yes** |
| `future` | period entirely in the future | **yes** |
| `failed` | network/HTTP error after all retries | **yes** |

---

## 4. Methodology, and one thing to be careful about

### Market-equal weighting, at every stage

Varieties and origin states reported within a single market on a given day are
averaged first, then those market figures are averaged. Without this, a market
that happens to report eight origin states for avocado would carry eight times
the weight of a market reporting one. Controlled by
`aggregate.market_equal_weighting`.

The same care is needed going from days to months, and this is easy to get
wrong. `cat_national_monthly` is built from `cat_market_monthly` — the mean over
markets of each market's monthly mean — **not** by averaging the daily national
series. Averaging days lets a market that reports 20 days a month outweigh one
that reports twice, which manufactures month-on-month inflation out of nothing
but a change in reporting frequency. Tested: with two markets whose prices never
move at all (A at 10, B at 30) and B reporting one day in February and twenty in
March, averaging days gives **+82.6% MoM**; the market-equal calculation
correctly gives 20.0 in both months and **0.0%**.

### Month-over-month and year-over-year are date-based, never row-based

`var_mensual` and `var_anual` are computed by joining on an explicit calendar
offset, not with `lag(x, 12)`. Row-based lags are actively wrong on seasonal
data: for a product sold only January–June, `lag(x, 12)` reaches back **24
calendar months** while still being labelled "annual". In
`cat_national_monthly` a missing comparison month yields NULL rather than a
confidently mislabelled number.

### Arithmetic vs geometric mean

Every price aggregate is computed **both ways** and stored side by side:
`precio` (arithmetic) and `precio_geo` (geometric, `exp(avg(ln(x)))`). There is
no config switch, because a knob that silently changes your numbers is worse
than two columns you can compare.

**Prefer `precio_geo`.** The geometric mean is the mean that treats proportional
changes symmetrically — its log is the arithmetic mean of the logs, so a variety
going 10 → 20 moves it exactly as much as one going 100 → 200. The arithmetic
mean is dominated by the expensive items instead. By AM–GM you will always see
`precio_geo <= precio`, and the gap widens as the category gets more
heterogeneous. That gap is a useful diagnostic in itself:

| Category | Arithmetic | Geometric | Gap |
|---|---|---|---|
| `Cebolla` (homogeneous — 4 onion varieties) | 17.43 | 16.29 | −6.5% |
| `Otras verduras y legumbres` (garlic ~150/kg + celery ~10/kg) | 55.68 | 36.51 | **−34.4%** |

A 34% gap is the arithmetic mean telling you it has become a garlic index.

### Use the Jevons index for inflation, not any mean of levels

The geometric mean fixes the *composition* problem. It does not fix the other
one: **entry and exit**. Mexican produce is highly seasonal, and when a cheap
variety enters the sample, *any* mean of levels drops — which reads as deflation
even though no individual price moved.

`cat_index_monthly` fixes that too. For each step it takes the geometric mean of
price **relatives** over the varieties present at **both** ends, then chains
those factors together. The level cancels out entirely, so seasonal entry and
exit stop creating artificial jumps.

**Gaps are the subtle part, and getting it wrong is catastrophic.** A strict
"previous calendar month" join would drop every month whose predecessor is
missing. For a strictly seasonal product that deletes exactly the months
carrying the year-on-year price step, and the index then reports *zero*
inflation for a product that inflated. Tested on a Jan–Jun product rising 10%
a year for three years: the naive version reported `indice_jevons = 100.00` for
all 18 months and `var_anual = 0`, while the truth was 100 → 110 → 121.

So each step links a month to the previous month the category **actually has**,
and `meses_puente` records how many calendar months that step spans. The same
fixture now returns 100 → 110 → 121 with `var_anual = 0.10` throughout and
`meses_puente = 7` on the off-season links. If no variety is present at both
ends of a step the factor is unmeasurable: the level is carried forward and
`cadena_rota` is set, so the assumption is visible rather than hidden.

The three approaches on the same category and window, measured on the
smoke-test sample:

| `Otras verduras y legumbres`, Feb 2025 → Aug 2026 | Reading |
|---|---|
| Arithmetic mean of levels | **−18.1%** |
| Geometric mean of levels | **−4.4%** |
| Chained Jevons index | **+5.6%** |

A 24-point spread on identical underlying data, purely a choice of aggregator.
The Jevons number is the defensible one.

**Caveats that no amount of code fixes:** this is *wholesale* price data, not
retail, so it will not reproduce the INPC. There are no consumption weights — a
Jevons elementary aggregate treats every variety equally. And `n_variedades_pareadas`
/ `n_mercados` in every table tell you how thin the sample is; early years and
minor categories are thin. Check those columns before drawing conclusions.

### Category mapping

`config/mapping_inpc.csv` maps all 222 fruit-and-vegetable products plus 24
frijol varieties to the 32 generics. It is **generated once and then
hand-editable** — `run.py mapping` refuses to overwrite it without `--force`.
Rationale for the judgement calls is documented at the top of `sniim/mapping.py`.
Worth knowing:

- `Jitomate` = Tomate Bola + Tomate Saladette. `Tomate verde` (tomatillo) is separate.
- `Limón` = the c/semilla sizes + s/semilla. `Lima` is a different fruit → Otras frutas.
- `Uva` = fresh table grapes. `Uva pasa` (raisins) and `Ciruela pasa` (prunes) → Otras frutas.
- `Papa y otros tubérculos` = 6 Papa varieties + Camote + Jícama. Betabel and Rábano → Otras verduras.
- `Chile seco` = ancho, guajillo, pasilla, mirasol, de Árbol seco, Puya seco. All other chiles are fresh.
- 24 products are deliberately **excluded** (Cacahuate, Nuez, Pistache, Jamaica, Orégano, Yerbabuena, and non-frijol grains) — the INPC classifies them elsewhere. They are written to the CSV with category `excluido` rather than silently dropped.

**All 222 products are collected, not just the mapped ones.** The cost is
identical because the loop is per-product either way, and it means the residual
buckets stay recomputable if you revise the mapping — without re-scraping.

### Measured runtimes

From actual timed runs on an M-series Mac, not guesses. The frutas figure comes
from a full 222-product single-year pilot: **31 minutes and 450 MB for one
year**, i.e. 8.4 seconds per product-year.

| Step | Requests | Time | Download |
|---|---|---|---|
| `catalog` | 2 | 5s | 0.1 MB |
| `backfill --module frutas`, one year, all 222 products | 223 | 31 min | 450 MB |
| `backfill --module frutas`, full 1998-2026 | ~6,400 | **~15 h** | ~13 GB |
| `backfill --module granos`, 12 frijol varieties, 1998-2026 | 18,168 | **~9 h** | ~350 MB |
| `update`, daily, 14-day window | ~270 | 10-15 min | ~300 MB |
| `build` | 0 | ~1 min | - |

So a complete history is roughly **a day of wall-clock time**. It is bounded by
SNIIM's rate limit, not by your machine — the collector is single-threaded on
purpose, because parallelism is what gets you blocked.

Resulting store: roughly **15 M rows** (651,737 for 2015 alone; earlier years are
thinner) in **~110 MB** of Parquet, since zstd gets it to about 7 bytes per row.

The backfill is fully resumable - kill it, reboot, rerun. Completed periods are
skipped via `data/manifest.csv`, so running it across several nights is fine.

For a run this long, stop the Mac sleeping through it and detach it from the
terminal:

```bash
caffeinate -i nohup python run.py backfill > logs/backfill.out 2>&1 &
tail -f logs/backfill.log
```

`caffeinate -i` prevents idle sleep (closing the lid still sleeps it; the run
resumes cleanly afterwards). If you prefer bounded chunks you can also watch it
progress a year at a time:

```bash
for y in $(seq 1998 2026); do python run.py backfill --module frutas --start-year $y --end-year $y; done
```

The granos backfill is the long pole per row collected. Narrow it in `config.yml`
if you do not need every bean variety back to 1998 - `granos.start_year` and
`granos.products` both cut it down.

---

## 4b. Validation on real data

A full 222-product pilot for 2015 produced **651,737 observations across 45
markets**, with 209 product-years `ok`, 13 genuinely `empty`, and **zero**
`failed` / `truncated` / `malformed` / `unusable` / `range_mismatch`. QA came back
with no PROBLEM lines: 0 duplicate natural keys, 0 `min > max` violations, 0
non-finite prices, 0 future dates, 0 unmapped rows, 4 rows out of 651,737 with no
`precio_frec` (SNIIM reported a min and max but no frequent price).

Four invariants were then checked against that real output, all exact to floating
point:

| Check | Result |
|---|---|
| Jevons index recomputed by hand from `var_national_monthly` | identical to 9 dp |
| `precio_geo <= precio` (AM-GM) across all 372 category-months | 0 violations |
| `var_mensual` reconciles with the levels it describes | max error 0.0 over 341 |
| `cat_national_monthly.precio` == unweighted mean of market monthlies | max error 0.0 over 372 |

And the seasonality is economically coherent, which is the check that actually
matters - a pipeline can be internally consistent and still be measuring noise:

| Category | 2015 peak | 2015 trough | Mexican harvest |
|---|---|---|---|
| Naranja | Jul (210) | Jan (100) | harvest Dec-Mar, so summer scarcity |
| Cebolla | Nov (151) | May (79) | spring harvest |
| Durazno | Dec (134) | Sep (84) | peach season Jul-Sep |
| Uva | Dec (118) | Jul (76) | Sonora harvest May-Jul |
| Jitomate | Dec (145) | Feb (76) | winter scarcity |

Two warnings are expected on a partial backfill and resolve themselves once the
full history is loaded: `frijol` has no data until you run the granos module, and
every category sits on a fallback index base until the configured base month
(`2018-01`) is actually present in the data.

---

## 5. Running it recurrently

`update` is the recurrent command. It re-fetches a **14-day trailing window**
and merges, rather than fetching only yesterday. Two reasons: SNIIM revises
recent rows, and a trailing window means a missed day, a closed laptop or a
failed run heals itself on the next run with no intervention.

It is **idempotent** — verified by rewriting 4,000 real rows twice and getting
zero net change. Dedupe key is
`(fecha, presentacion, origen, destino, unidad, obs)` per product-year, keeping
the most recently fetched copy. `obs` is in the key because for Granos
`presentacion` is always NULL and the distinguishing text ("Presentación en
bulto de 25 kg.") lives in `obs`, so a shorter key would collapse two genuinely
different quoted prices into one. `unidad` is in the key so that changing
`query.precios_por_id` can never silently overwrite pesos/kg rows with
pesos/package rows.

Only one writer may touch the store at a time. `backfill`, `update` and `build`
take an advisory lock on `data/.lock` and exit with code 2 if another run holds
it — without that, a cron `update` firing while a long `backfill` is still
running would let each clobber the other's rows for the current year.

### macOS — launchd (recommended)

Preferred over cron because launchd runs a missed job when the Mac next wakes,
instead of skipping it.

```bash
sed -i '' "s|REPLACE_ME|$(pwd)|g" scripts/mx.sniim.update.plist
cp scripts/mx.sniim.update.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/mx.sniim.update.plist
launchctl start mx.sniim.update
tail -f logs/cron.log
```

Those five lines: point the plist at this project, install it, load it, fire it
once immediately rather than waiting for 19:30, and watch the log.

### macOS / Linux — cron

```bash
chmod +x scripts/daily_update.sh
crontab -e
```

```cron
30 19 * * 1-5 /Users/wilsonfelicio/Downloads/coleta/scripts/daily_update.sh
```

Weekdays at 19:30 — SNIIM publishes on weekday afternoons, so an evening run
captures the same day.

### Keeping an eye on it

```bash
python run.py status
python run.py verify
tail -50 logs/update.log
```

`status` shows rows on disk, job progress and what is outstanding. `verify` runs
the QA checks and exits non-zero on a structural problem, which makes it usable
in a scheduled job.

---

## 6. Querying the data

```bash
python run.py export --categoria aguacate --out aguacate.csv
```

```python
import duckdb
con = duckdb.connect("data/sniim.duckdb", read_only=True)

# Inflation series — the right table for this
con.sql("""
    SELECT categoria_label, mes, indice_jevons, var_anual
    FROM cat_index_monthly
    WHERE mes >= '2020-01-01'
    ORDER BY categoria_label, mes
""").df()

# Price levels for one category, one market (precio_geo = geometric, prefer it)
con.sql("""
    SELECT mes, precio, precio_geo, n_dias
    FROM cat_market_monthly
    WHERE categoria = 'jitomate'
      AND destino_mercado LIKE '%Iztapalapa%'
    ORDER BY mes
""").df()

# Which variety is driving a category?
con.sql("""
    SELECT producto, mes, precio
    FROM var_national_monthly
    WHERE categoria = 'otras_verduras_y_legumbres' AND mes = '2026-06-01'
    ORDER BY precio DESC
""").df()

# Straight down to raw observations
con.sql("SELECT * FROM obs WHERE categoria='aguacate' AND fecha='2026-08-07'").df()
```

The curated tables are also on disk as Parquet and CSV in `data/curated/`, so
R (`arrow::read_parquet`), Stata 18+ and Excel can read them without DuckDB.

---

## 7. Maintenance

**SNIIM adds products and markets over time.** Every few months:

```bash
python run.py catalog
python run.py mapping --force
python run.py verify
```

`--force` regenerates the mapping, so re-apply any hand edits afterwards.
`verify` warns about observations with no category mapping.

`verify` also warns if any observation has no category mapping, so a new SNIIM
product cannot silently vanish from the aggregates.

**If the site changes shape.** The parser is header-driven and **rejects** any
data row whose cell count does not match the header, counting it in
`manifest.csv:malformed` and escalating it to a PROBLEM in `verify`. That
strictness is deliberate. Column shifts would otherwise be invisible rather than
loud: `"Caja de 10 kg."` parses as the number 10 and a date parses as its day, so
a shifted row yields *plausible* prices in the wrong fields. Renaming a price
column is caught separately — `verify` raises a PROBLEM when most or all rows
have a NULL `precio_frec`, instead of silently publishing an empty series.

**Retrying failed jobs.** `--resume` skips only `ok` and `empty`, so every other
status is retried automatically on the next `backfill`. `python run.py status`
lists what will be retried.

**Note on the current year.** A year still in progress is recorded `open`, not
`ok`, so each `backfill` refetches it. Without that the year-only manifest key
would mark 2026 complete in August and September–December would never be
collected by any later backfill — you would be relying entirely on `update`
running at least once every 14 days for the rest of the year.

---

## 8. Known data quirks in the source

These are SNIIM's, not the collector's:

- **`precio_frec` occasionally falls outside `[precio_min, precio_max]`** — about
  0.03% of rows (e.g. Brócoli in Morelia, 2025-03-26: min 22, max 25, frec 18).
  Preserved as reported rather than silently corrected. `verify` counts them.
- **`precio_min > precio_max`** is corrected by swapping, in `store.py`.
- **`0.00` prices** are placeholders for "not surveyed" and are read as null.
- **Number formats are parsed strictly.** Only `58.00`, `1,234.56` and `58,00`
  shapes are accepted; anything else returns null rather than being guessed at.
  Naively stripping commas would turn `58,00` into 5800.00 — a 100× error that
  no downstream check would catch.
- **Weeks with no survey** return zero rows and are recorded `empty` in the
  manifest, so they are not retried forever.
- Early years are thin: 1998 has roughly a fifth of the observations of 2005.
- Origin is `Importación` for imported produce, which is genuinely useful — it
  is kept as a normal origin value rather than filtered out.

---

## 9. Layout

```
run.py                    single CLI entrypoint
config/
  config.yml              every knob — rate limits, years, price column, index base
  mapping_inpc.csv        generated once, then hand-editable
sniim/
  config.py               config load + logging
  http.py                 rate-limited session, backoff, self-throttling
  parse.py                header-driven results-table parser
  catalog.py              dropdown scraping
  frutas.py               daily module, recursive split on truncation
  granos.py               weekly module (frijol), week-of-month arithmetic
  store.py                Parquet store, atomic writes, locking, resumable manifest
  mapping.py              the 32-category mapping rules and their rationale
  aggregate.py            DuckDB build, aggregates, chained Jevons index
scripts/
  daily_update.sh         scheduler wrapper
  mx.sniim.update.plist   launchd job template
```

Collection is deliberately **single-threaded**. Parallelism is what gets you
blocked, and the whole job is IO-bound on a server nobody controls. Be a good
citizen: this is a free public data service run by the Secretaría de Economía.
