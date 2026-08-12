# Anticipating Mexican food CPI from SNIIM wholesale prices

**Research note, phase 1.** Everything below is verified against live data unless
explicitly flagged. The one thing that is *not* yet testable is the model itself,
because the SNIIM backfill is still running — see §9.

---

## 1. The finding that reframes the project

**Your 32 categories are not a reasonable selection of food items. They are
*exactly* INEGI's published non-core analytical group "Frutas y verduras"** —
serie `865557`, weight **4.78** of the INPC, published monthly since January 1982.

Verified three independent ways:

| Test | Result |
|---|---|
| **Count** — INPC 2024 methodological document, Cuadro 11 | "Frutas y verduras = **32** genéricos" |
| **Weight** — Cuadro 20 | Frutas y verduras = **4.78**; your 32 ponderadores sum to **4.77888** |
| **Numeric** — aggregate the 32 published indices and compare to serie 865557 | **RMSE 0.029 pp** of MoM change; last four months match to 3 decimals |

The numeric test also rules out every nearby alternative grouping:

| Candidate | Items | Weight | RMSE vs published |
|---|---|---|---|
| **Your 32** | **32** | **4.77888** | **0.029 pp** |
| Drop Frijol + Chile seco | 30 | 4.31877 | 0.463 pp |
| Drop Chile seco only | 31 | 4.72976 | 0.062 pp |

So **Frijol and Chile seco are non-core**, inside Frutas y verduras — not core
"mercancías alimenticias", which is the intuitive but wrong guess. (Only *Otras
legumbres secas*, 079, from the same subclase is core.) The arithmetic closes
exactly: INPC group 1 has 114 genéricos weighing 27.87247, decomposing as 74 core
food + **32 Frutas y verduras** + 8 Pecuarios = 114, and 17.21 + **4.78** + 5.88 =
27.87.

**Why this matters.** You get a published, headline-relevant validation target for
free. If the item models are any good they must aggregate to serie 865557 — and
that subindex is the most volatile component of Mexican inflation and a routine
talking point in Banxico's reports. "We nowcast the INPC's fruit-and-vegetable
subindex" is a far stronger claim than "we model 32 loosely related food prices."

`python inpc_run.py validate` runs this check.

One catch worth knowing: after the August 2024 chain link the published indices
"pierden su propiedad de aditividad" (methodological document p. 69). You cannot
just take a weighted mean. Multiply each index by θᵢ = 1/factor_encadenamientoᵢ
first. This is not a rounding detail:

| Aggregation | RMSE vs published |
|---|---|
| With θ correction | **0.029 pp** |
| Naive weighted mean | 1.028 pp — 35× worse |

---

## 2. Data access: solved, no token, one request

The INEGI "Índices de precios" v2 app has a CSV export endpoint that needs no
API token and returns every requested series with full history in a single POST.

| What | idEstructura | Coverage |
|---|---|---|
| INPC genéricos, **monthly** | `112001700030` | Ene 1970 → current, 679 periods |
| INPC genéricos, **quincenal** | `112001600030` | 1Q Ene 1995 → current, 758 periods |
| Analytical subindices (incl. 865557) | `112001700010` | Ene 1970 → current |
| Genéricos × **55 cities** | `112001700060` | 24,081 nodes |

Routes considered and rejected:

- **INEGI API de Indicadores** — needs a token requested by email. Unverified
  whether genérico series are even exposed. The CSV route is strictly better for
  bulk history.
- **Banxico SIE API** — verified it does **not** reach genérico level. Cuadro
  CP154 stops at aggregates ("frutas y verduras" and above). Banxico also stopped
  producing the INPC in 2011. Ignore it.

Also collected, and more useful than expected:

**INEGI "Precios Promedio"** gives actual **retail prices in pesos per unit**, by
generic, by city, by specification. That makes the retail–wholesale *margin*
observable, which is what turns an error-correction specification from hypothesis
into something estimable. Jul-2026, weighted over 3 cities: Aguacate 83.80
MXN/kg, Cebolla 28.83, Jitomate 25.36.

Two traps in it, both handled in `inpc/precios.py`:

- **Generic codes are not stable across vintages.** In the 2011–2018 vintage
  `045` = Plátanos, not Aguacate. Always key on the name.
- **Units are not uniformly KG.** 8.2% of quotes are per-piece or per-bunch,
  heavily concentrated: Cilantro/epazote/perejil is 94.7% MANOJO (only 8 of 55
  cities quote per kg) and Lechuga y col is 79.7% PZA. A naive mean over mixed
  units is not a price. After filtering to KG, 30 of 32 genéricos still have ≥53
  of 55 cities; those two are flagged in `catalogo.PRECIOS_UNIT_UNRELIABLE`.

Weights come from the DOF 2024 canasta update (via the `sidof.segob.gob.mx`
mirror — `dof.gob.mx` blocks automated fetching), and the 55 city weights from
Anexo F of the methodological document. Both ship in `config/`.

---

## 3. The monthly index is the exact mean of the two fortnights

$$\text{INPC}_M = \tfrac{1}{2}\left(Q1_M + Q2_M\right)$$

An **identity, not an approximation**. Verified on published bulletins:

- May 2026: (145.622 + 145.432)/2 = 145.527 = published monthly index
- Jun 2026: (145.274 + 144.988)/2 = 145.131 → MoM −0.272%, published −0.27%

Three consequences, and they change the design:

1. **Model at quincenal frequency.** It roughly doubles the sample (758 fortnightly
   periods vs 679 monthly, and ~660 vs ~330 in the usable window) and matches the
   publication structure. Aggregate to monthly by the identity.
2. **Once Q1 is published, half the monthly answer is known exactly.** So any
   benchmark evaluated at a late-month origin must also use it — see §5.
3. INEGI collects **food prices weekly** (four times a month), versus twice
   monthly for most goods, so a fortnightly genérico index reflects about two
   collection rounds. The specific collection days are not published, which means
   the mapping from daily wholesale days to a fortnight has to be *estimated*
   (let MIDAS-style weights decide) rather than assumed.

---

## 4. The timing edge, precisely

From INEGI's official 2026 dissemination calendar, all releases 06:00:

| Target | Publication lag |
|---|---|
| 1st fortnight of month M | **7–9 days** after the fortnight ends |
| 2nd fortnight + monthly for M | **7–9 days** after month end |

So waiting for a complete calendar month of wholesale data beats INEGI's monthly
print by only about **a week**. That is real but unremarkable.

The useful window is **intra-period**. At day ~16 of month M you hold complete
wholesale coverage of days 1–15 while Q1 stays unpublished for another 6–8 days —
and Q1 is exactly half the monthly index. A monthly nowcast at day 10 runs
roughly **28 days** ahead of publication.

---

## 5. How hard is the target? Harder than it looks

Out-of-sample horse race, expanding window, 2012–2026, INPC own-history models
only (`python inpc_run.py benchmarks`):

**Aggregate "Frutas y verduras" subindex, RMSE in pp of MoM change:**

| Model | RMSE |
|---|---|
| **SD-AR** (month dummies + AR1) | **3.514** |
| Climatology (month means) | 3.621 |
| AR(1) | 3.650 |
| Random walk | 3.773 |
| Seasonal naive | 5.196 |

**Per item:** SD-AR wins for **22 of 30** items. Median RMSE of the winning
benchmark is **6.46 pp per month**, ranging 0.77 (Chile seco) to 18.38 (Tomate
verde). This data is extraordinarily volatile — for scale, the whole INPC moves
0.54% per month on average, while Jitomate moves 20%.

**Two things follow.**

First, **SD-AR is the benchmark, not a random walk.** Banxico's own work found the
same for exactly this category — Capistán, Constandse & Ramos-Francia (2009) got
RMSE 21.13 for a deterministic-seasonality model on *frutas y verduras* vs 30.13
for a seasonal-unit-root model. Reporting gains against a random walk would be a
straw man.

Second, and this is the encouraging part: **the median gain of the best seasonal
model over a random walk is only 1.20×.** Own-history seasonality does surprisingly
little work here. That leaves room for the wholesale data — unlike, say, energy
prices where seasonality explains most of the variance and there is nothing left
to add.

### Where the error actually lives

Weighting each item's forecast error by its INPC weight:

| Item | Weight | Best RMSE | Share of weighted error |
|---|---|---|---|
| Jitomate | 0.79 | 17.36 | **35%** |
| Cebolla | 0.24 | 12.95 | 8% |
| Tomate verde | 0.14 | 18.38 | 7% |
| Chayote | 0.13 | 14.35 | 5% |
| Papa y otros tubérculos | 0.32 | 5.57 | 5% |
| Limón | 0.10 | 15.97 | 4% |

**Top 3 items = 50% of the weighted error. Top 6 = 63%.** Effort should be
concentrated there, not spread evenly across 32 items.

And here is the fortunate part: **the items where own-history models fail worst
are precisely those six.** Jitomate's seasonal gain over a random walk is 1.12×,
Tomate verde 1.04×, Calabacita 1.05×, Chile serrano 1.12×. Conversely the items
where seasonality already works well — Naranja 1.98×, Frijol 1.79×, Uva 1.61× —
carry little weight and little error. The wholesale data is being asked to help
exactly where help is both needed and possible.

---

## 6. Does the wholesale series actually carry signal?

Run on the 2015 pilot data, so **n = 11 monthly changes**. This validates the
plumbing and gives a first read; it is not inference.

| | Median across 29 items | Items > 0.4 |
|---|---|---|
| Raw correlation of MoM log changes | **0.86** | 26/29 |
| **After removing calendar-month climatology** | **0.59** | **22/29** |

The raw 0.86 is not the number to quote. Wholesale and retail share the same
seasonal cycle, so a raw correlation mostly measures seasonality that the SD-AR
benchmark already exploits. The residual correlation is what represents
*incremental* information, and the ECB's scanner-data work (WP 2930) finds
high-frequency data starts cutting nowcast error once it clears roughly **0.4**.

That 22 of 29 items clear it — carrying **3.48 of the 4.07** INPC weight in the
sample — is the single most encouraging result here. And the survivors are the
ones that matter: Tomate verde 0.91, Jitomate 0.87, Chile serrano 0.81, Pepino
0.77. The failures are low-weight: Guayaba −0.53, Pera −0.27, Durazno −0.09,
Piña −0.02.

Median pass-through β ≈ 0.91 — close to one-for-one, which is plausible for fresh
produce where the retail margin is roughly proportional.

**Read these numbers with the stated caveats.** With 11 observations the standard
error on a correlation is about 0.3. And with one year of wholesale data there is
exactly one observation per calendar month, so the wholesale side could only be
mean-demeaned, not seasonally demeaned — which biases `corr_residual` **upward**.
`inpc_run.py align` prints both warnings rather than quietly emitting the number,
and refuses to compute a seasonal profile from fewer than 3 years per month.

---

## 7. Two fixes needed on the wholesale side

The biggest methodological risk in this whole exercise is **composition drift
masquerading as price change**: if the wholesale index is an unmatched
cross-sectional mean of pesos/kg over whatever variety × market cells happened to
report, then varieties entering and leaving produce index movements that are not
price movements. It is doubly dangerous because the drift is *seasonal*, so it
correlates with the INPC's own seasonality and will look like genuine predictive
power that evaporates in real time.

The collector already handles the variety dimension — `cat_index_monthly` is a
matched-model chained Jevons index, matching varieties present at both ends of
each step. That decision now looks more important than when it was made. Two
refinements remain:

1. **Match on variety × market, not variety alone.** Currently markets are
   averaged geometrically within a variety-month before matching. A market
   entering or leaving still moves the index. Matching cells would remove that.
2. **Weight markets toward INPC-relevant cities.** All 49 markets are currently
   equal-weighted. The INPC weights CDMX at 22.4% of the national basket, and
   Banxico's own SNIIM-based work finds cities supplied through intermediaries —
   chiefly CDMX's Central de Abasto — run 1.7–1.9% higher. Equal-weighting a thin
   market alongside CEDA is not neutral. The city weights ship in
   `config/inpc_ciudades.csv`.

Neither requires re-scraping. Both are aggregation changes.

---

## 8. Recommended specification

Per item, at **quincenal** frequency, error-correction bridge equation, pooled
with shrinkage, estimated **separately per nowcast-origin day**, aggregated to
monthly by the identity in §3 and to the subindex by the θ-weighted sum in §1.

- **Error-correction, not pure differences.** The retail–wholesale margin is
  bounded by arbitrage so it must mean-revert; the form nests differences-only as
  λ = 0. Banxico reports a 2.1-month half-life for relative-price convergence
  across Mexican wholesale markets, implying a prior of λ ≈ 0.15 per fortnight.
- **Seasonally demean the margin before testing cointegration.** INEGI imputes
  and substitutes when a specification is seasonally unavailable, which damps
  retail volatility relative to wholesale and puts a *seasonal* component in the
  margin. A plain ADF on the raw margin will spuriously fail to reject a unit
  root.
- **Asymmetric ± split on the wholesale term,** with a counterintuitive prior:
  Banxico's WP 2016-18 finds 41 food items show rockets-and-feathers asymmetry,
  but that **fresh fruit and vegetables show the *lowest* asymmetry** — perishability
  forces markdowns. So expect near-symmetric fast pass-through for jitomate and
  chiles, and slower asymmetric pass-through for the storables (frijol, chile
  seco). Test it rather than assuming a common lag structure.
- **Pooling for shrinkage, not homogeneity.** 32 items × 6 lags is too many free
  parameters. Keep the impact coefficient item-specific; shrink higher lags and
  seasonal profiles toward produce-group means.
- **One model per origin day** (8, 16, 24 of M, and day 2 of M+1), each seeing only
  what existed then. Never extrapolate a partial month and then difference it —
  that injects the very thing being forecast. Compare like-for-like day windows.
- **Screen first.** Route items below 0.4 residual correlation to the seasonal
  benchmark instead of pretending to model them.

**Evaluation:** expanding window primary, 10-year rolling as robustness, OOS from
~2012. Loss = RMSE on MoM log change, plus MAE (RMSE here is dominated by a
handful of jitomate episodes) plus INPC-weight-weighted RMSE. Directional
accuracy via Pesaran–Timmermann, not a raw hit rate.

**Two testing corrections that matter.** The bridge model *nests* SD-AR, so
standard Diebold–Mariano is undersized — use Clark–West or ENC-NEW for the nested
comparison and reserve DM for non-nested ones. And with 32 items you expect ~1.6
spurious winners at 5%, so control FDR across items and lead with the pooled
weighted result rather than cherry-picking.

**Include a movable-holiday feature.** Semana Santa migrates between March and
April *and* between fortnights within a month, so a fixed calendar dummy
misattributes it. Use the share of the fortnight's days falling in
Cuaresma/Semana Santa.

**Do not winsorize the target.** Jitomate printed −38.98% annual in June 2026.
The outliers are the signal.

---

## 9. What is done, and what comes next

**Done and verified:**

- INPC collector, no token, monthly + quincenal + subindices, Ene 1970 → current
- Retail prices in MXN/kg by city, with the unit-mixing trap handled
- The 32 genéricos mapped to serie ids, weights, chaining factors; 55 city weights
- Proof that the 32 reproduce the published subindex (RMSE 0.029 pp)
- Benchmark horse race establishing SD-AR as the bar
- Alignment screen, with its own limitations reported rather than hidden

**Blocked on the SNIIM backfill:** the actual model. The 11-observation overlap is
enough to validate plumbing and nothing else. Once the full history lands, in
order:

1. Rerun `inpc_run.py align` on ~330 months. Expect residual correlations to fall
   from 0.59 — a properly estimated wholesale seasonal profile will absorb some of
   what is currently attributed to signal. **If the median lands near 0.4, the
   project is viable but marginal; if it holds near 0.55, it is strong.**
2. Apply the two wholesale-index fixes in §7 and check whether the screen improves.
3. Fit the specification in §8 for the top 6 items first — they are 63% of the
   weighted error.
4. Only then extend to the full 32 and the aggregate subindex.

**Honest expectation.** No published study nowcasts INPC genéricos from SNIIM, so
there is no replication target — which is whitespace, but also means the
validation discipline has to be self-imposed. The ECB found 40–60% item-level RMSE
reductions for unprocessed fruit and vegetables using weekly scanner data, which
is the closest analogue and suggests real upside. Against that, Banxico's own
seasonal models are a genuinely strong baseline and absolute errors will stay
large regardless: even a 30% improvement on Jitomate leaves ±12 pp monthly
uncertainty. The useful output is probably a **probabilistic** statement about the
subindex, not a point forecast anyone should trade on.

---

## Commands

```bash
python inpc_run.py fetch
python inpc_run.py validate
python inpc_run.py benchmarks
python inpc_run.py precios --desde 201101
python inpc_run.py align
```

`inpc_run.py` is deliberately a separate entrypoint from `run.py`: it touches
INEGI rather than SNIIM, never writes to `data/raw/`, and so is safe to run while
a SNIIM backfill is in progress.

## Sources

- [INEGI — INPC 2024 Documento metodológico](https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/nueva_estruc/889463918639.pdf) — Cuadro 10 p.22, Cuadro 11 pp.22–23, Cuadro 20 p.33, additivity/θ p.69, Anexo F p.106
- [INEGI — Calendario de difusión 2026](https://www.inegi.org.mx/contenidos/saladeprensa/doc/cal_2026.pdf)
- [INEGI — Boletín INPC junio 2026](https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2026/inpc/inpc_2q2026_07.pdf) and [mayo 2026](https://www.inegi.org.mx/contenidos/saladeprensa/boletines/2026/inpc/inpc_2q2026_06.pdf)
- [DOF nota 5737063 (sidof mirror) — canasta, ponderadores y encadenamiento 2024](https://sidof.segob.gob.mx/notas/docFuente/5737063)
- [INEGI — Índices de precios app](https://www.inegi.org.mx/app/indicesdepreciosv2/estructura.aspx) · [Precios Promedio](https://www.inegi.org.mx/app/preciospromedio/?bs=18a)
- [Banxico WP 2016-18 — Price Transmission in Food and Non-Food Product Markets: Evidence from Mexico](https://www.banxico.org.mx/publications-and-press/banco-de-mexico-working-papers/%7B8C7EE729-E3B6-C5C4-F416-7F36F741B810%7D.pdf)
- [Banxico WP 2009-05 — Uso de Modelos Estacionales para Pronosticar la Inflación de Corto Plazo en México](https://www.banxico.org.mx/publicaciones-y-prensa/documentos-de-investigacion-del-banco-de-mexico/%7B0395F36A-5656-BCBC-D0C7-F3201BACA6C1%7D.pdf)
- [Banxico — Recuadro: actualización del INPC 2024](https://www.banxico.org.mx/publicaciones-y-prensa/informes-trimestrales/recuadros/%7BB2DB3F5A-4FA5-E527-3A74-5B3998702397%7D.pdf) · [Recuadro: distribución mayorista de frutas y hortalizas (usa SNIIM)](https://www.banxico.org.mx/publicaciones-y-prensa/reportes-sobre-las-economias-regionales/recuadros/%7B7C9582C0-F507-5A7D-EB24-BB3EDFE70943%7D.pdf)
- [ECB WP 2930 — Nowcasting consumer price inflation using high-frequency scanner data](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2930~05cff276eb.en.pdf)
