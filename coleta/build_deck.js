// Macro-research deck: nowcasting Mexico's produce CPI from wholesale prices.
// Palette: Ocean Gradient (deep blue dominant, teal support, midnight accent) — chosen
// to sit alongside the charts' blue/orange series without competing with them.
const F = JSON.parse(require('fs').readFileSync('data/curated/facts.json', 'utf8'));
const J = F.jit, AG = F.agg, SPL = F.split;
const GATED = F.table.find(r => !r.admisible);
const mf1 = x => x.toFixed(1).replace('-', '\u2212');
const f1 = x => x.toFixed(1), f2 = x => x.toFixed(2), f3 = x => x.toFixed(3);
const pc0 = x => Math.round(x) + '%';
const pptx = new (require('pptxgenjs'))();
pptx.layout = 'LAYOUT_WIDE';                 // 13.3 x 7.5 in
const W = 13.3, H = 7.5;

const DEEP = '065A82', TEAL = '1C7293', MID = '21295C', WHITE = 'FFFFFF';
const INK = '0B0B0B', SEC = '52514E', MUT = '898781', LIGHT = 'F4F6F8';
const HEAD = 'Cambria', BODY = 'Calibri';

const notes = {};
function titleSlide() {
  const s = pptx.addSlide();
  s.background = { color: MID };
  s.addText('Nowcasting Mexico’s produce CPI', {
    x: 0.9, y: 1.5, w: 11.5, h: 0.9, fontFace: HEAD, fontSize: 42, bold: true, color: WHITE,
    margin: 0 });
  s.addText('Wholesale market prices as a real-time read on the fortnightly print', {
    x: 0.9, y: 2.45, w: 11.5, h: 0.5, fontFace: BODY, fontSize: 18, color: 'CADCFC',
    margin: 0 });
  const stats = [
    [`±${f3(AG.all32.headline)} pp`, 'error on the produce\ncontribution to headline'],
    [pc0(AG.all32.gain_pct), 'lower error than a\nCPI-only benchmark'],
    ['9 days', 'ahead of INEGI’s\npublication'],
  ];
  stats.forEach(([big, lab], i) => {
    const x = 0.9 + i * 3.9;
    s.addShape(pptx.ShapeType.roundRect, { x, y: 3.7, w: 3.5, h: 1.9, fill: { color: DEEP },
      rectRadius: 0.08, line: { color: DEEP } });
    s.addText(big, { x: x + 0.25, y: 3.9, w: 3.0, h: 0.75, fontFace: HEAD, fontSize: 34,
      bold: true, color: WHITE, margin: 0 });
    s.addText(lab, { x: x + 0.25, y: 4.7, w: 3.0, h: 0.8, fontFace: BODY, fontSize: 12,
      color: 'CADCFC', margin: 0 });
  });
  s.addText(`SNIIM wholesale prices 1998–2026  ·  INEGI CPI  ·  ${F.n_generics} generics, ${AG.n} out-of-sample fortnights`,
    { x: 0.9, y: 6.6, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 11, color: MUT, margin: 0 });
  s.addNotes('This is a nowcast, not a forecast. The edge is INEGI’s publication calendar: the CPI for each fortnight appears about nine days after the fortnight closes, while SNIIM publishes wholesale prices daily.');
}

function sectionSlide(n, title, sub) {
  const s = pptx.addSlide();
  s.background = { color: MID };
  s.addText(n, { x: 0.9, y: 2.5, w: 2, h: 1.2, fontFace: HEAD, fontSize: 66, bold: true,
    color: TEAL, margin: 0 });
  s.addText(title, { x: 0.9, y: 3.7, w: 11, h: 0.8, fontFace: HEAD, fontSize: 32, bold: true,
    color: WHITE, margin: 0 });
  s.addText(sub, { x: 0.9, y: 4.55, w: 10.5, h: 0.8, fontFace: BODY, fontSize: 15,
    color: 'CADCFC', margin: 0 });
  return s;
}

// generic content slide: title + optional lede + body builder
function contentSlide(title, lede) {
  const s = pptx.addSlide();
  s.background = { color: WHITE };
  s.addText(title, { x: 0.6, y: 0.42, w: 12.1, h: 0.62, fontFace: HEAD, fontSize: 30,
    bold: true, color: MID, margin: 0 });
  if (lede) s.addText(lede, { x: 0.6, y: 1.06, w: 12.1, h: 0.5, fontFace: BODY,
    fontSize: 14, color: SEC, margin: 0 });
  return s;
}

function chartSlide(title, lede, img, source, y0) {
  const s = contentSlide(title, lede);
  s.addImage({ path: img, x: 0.6, y: y0 || 1.70, w: 12.1, h: 4.76, sizing: { type: 'contain', w: 12.1, h: 4.76 } });
  if (source) s.addText(source, { x: 0.6, y: 6.74, w: 12.1, h: 0.3, fontFace: BODY,
    fontSize: 9, color: MUT, margin: 0 });
  return s;
}

function statRow(s, items, y) {
  const gap = 0.34, w = (12.1 - gap * (items.length - 1)) / items.length;
  items.forEach(([big, lab, note], i) => {
    const x = 0.6 + i * (w + gap);
    s.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 1.55, fill: { color: LIGHT },
      rectRadius: 0.06, line: { color: 'E1E0D9' } });
    s.addText(big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.6, fontFace: HEAD,
      fontSize: 27, bold: true, color: DEEP, margin: 0 });
    s.addText(lab, { x: x + 0.2, y: y + 0.74, w: w - 0.4, h: 0.32, fontFace: BODY,
      fontSize: 12, bold: true, color: INK, margin: 0 });
    if (note) s.addText(note, { x: x + 0.2, y: y + 1.04, w: w - 0.4, h: 0.42,
      fontFace: BODY, fontSize: 10, color: SEC, margin: 0 });
  });
}

function bullets(s, items, x, y, w, h, size) {
  s.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i !== items.length - 1 } })), {
    x, y, w, h, fontFace: BODY, fontSize: size || 14, color: INK, paraSpaceAfter: 8 });
}

// ----------------------------------------------------------------- 1 title
titleSlide();

// ----------------------------------------------------------------- 2 the idea
{
  const s = contentSlide('The opportunity is a calendar gap, not a forecast',
    'Two public data sources, published on very different schedules.');
  const rows = [
    ['SNIIM — wholesale', 'Secretaría de Economía', 'Daily, ~11 quote days per fortnight, 60 markets. Free, no token, published same day.'],
    ['INEGI — CPI', 'Instituto Nacional de Estadística', 'Fortnightly. 1H printed on the 24th, 2H on the 9th — about nine days after the fortnight closes.'],
  ];
  rows.forEach(([h1, h2, txt], i) => {
    const y = 1.75 + i * 1.35;
    s.addShape(pptx.ShapeType.ellipse, { x: 0.6, y: y + 0.12, w: 0.52, h: 0.52,
      fill: { color: DEEP }, line: { color: DEEP } });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.12, w: 0.52, h: 0.52, align: 'center',
      fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, margin: 0 });
    s.addText(h1, { x: 1.35, y: y, w: 4.0, h: 0.34, fontFace: HEAD, fontSize: 17,
      bold: true, color: MID, margin: 0 });
    s.addText(h2, { x: 1.35, y: y + 0.34, w: 4.0, h: 0.3, fontFace: BODY, fontSize: 11,
      color: MUT, margin: 0 });
    s.addText(txt, { x: 5.6, y: y - 0.02, w: 7.1, h: 0.9, fontFace: BODY, fontSize: 13.5,
      color: INK, margin: 0, valign: 'top' });
  });
  s.addShape(pptx.ShapeType.roundRect, { x: 0.6, y: 4.7, w: 12.1, h: 1.55,
    fill: { color: MID }, rectRadius: 0.06, line: { color: MID } });
  s.addText('So for roughly nine days we know the complete wholesale record for a fortnight whose CPI has not been published.',
    { x: 1.0, y: 4.95, w: 11.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true,
      color: WHITE, margin: 0 });
  s.addText('That window — not any predictive lead of wholesale over retail — is the entire source of the result. A genuine one-step-ahead forecast, using nothing dated inside the target fortnight, beats the benchmark by only about 5%.',
    { x: 1.0, y: 5.45, w: 11.3, h: 0.7, fontFace: BODY, fontSize: 12.5, color: 'CADCFC',
      margin: 0 });
  s.addNotes('Be explicit that this sharpens the current print rather than extending the forecast horizon. If asked "does this help our 12-month forecast", the answer is no.');
}

// ----------------------------------------------------------------- 3 where variance is
chartSlide(`One item is ${pc0(SPL.jitomate)} of the problem`,
  'Contribution to the variance of the fortnightly change in the subindex: weight × own volatility × correlation with the aggregate.',
  'charts/system_variance.png',
  'Source: INEGI published generic indices, 2016–2026. Ranking fixed before any model was run.');

// ----------------------------------------------------------------- 4 jitomate result
chartSlide('Jitomate: the item that matters, and it works',
  `Nowcast vs realised, ${J.n} out-of-sample fortnights, refit every fortnight on a rolling five-year window using earlier data only.`,
  'charts/jitomate_roll5y.png',
  `Source: SNIIM and INEGI. RMSE ${f2(J.rmse_a)} pp against ${f2(J.rmse_b)} pp for a CPI-only benchmark; correct sign ${pc0(J.sign_a)}. Scored on the ${J.n} fortnights where both models are defined.`);

// ----------------------------------------------------------------- 5 the gain
chartSlide('What the wholesale data adds',
  'Same fortnights, same estimator. The only difference is whether wholesale prices are in the information set.',
  'charts/jitomate_gain.png',
  `Source: SNIIM and INEGI. Diebold-Mariano t = ${f1(J.dm_t)} in favour of the wholesale model.`);

// ----------------------------------------------------------------- 6 aggregate
chartSlide('The subindex, and how much coverage you need',
  'Bottom-up from the 32 generics with published weights and chaining factors.',
  'charts/system_aggregate.png',
  'Source: SNIIM and INEGI. Non-members of a set are carried at their own CPI-only benchmark, so the weight base is identical in all three.');

// ----------------------------------------------------------------- 7 headline numbers
{
  const s = contentSlide('The number for the note',
    'Error on the published fortnightly print, and the same error expressed as the produce contribution to headline INPC.');
  statRow(s, [
    [`±${f3(AG.all32.headline)} pp`, 'All 32 modelled', `RMSE ${f2(AG.all32.rmse_a)} pp of the subindex print`],
    [`±${f3(AG.top10.headline)} pp`, 'Top 10 modelled', 'the other 22 at benchmark'],
    [`±${f3(AG.top5.headline)} pp`, 'Top 5 modelled', 'the other 27 at benchmark'],
    [`±${f3(AG.bench.headline)} pp`, 'No wholesale data', 'CPI-only benchmark'],
  ], 1.76);
  s.addText(`Read against the subindex’s own volatility of ${f3(AG.sd_headline)} pp of headline contribution per fortnight, the CPI-only model resolves almost none of it; the wholesale nowcast resolves about four fifths.`,
    { x: 0.6, y: 3.68, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 13.5, color: INK, margin: 0 });
  s.addShape(pptx.ShapeType.roundRect, { x: 0.6, y: 4.44, w: 5.88, h: 2.2,
    fill: { color: LIGHT }, rectRadius: 0.06, line: { color: 'E1E0D9' } });
  s.addText('Five generics get you most of it', { x: 0.9, y: 4.62, w: 5.3, h: 0.35,
    fontFace: HEAD, fontSize: 16, bold: true, color: MID, margin: 0 });
  bullets(s, [
    `Top 5 alone: ${pc0(AG.top5.gain_pct)} better than benchmark`,
    `Adding the next 5: a further ${mf1(100 * (AG.top10.rmse_a / AG.top5.rmse_a - 1))}%`,
    `Adding the remaining 22: a further ${mf1(100 * (AG.all32.rmse_a / AG.top10.rmse_a - 1))}%`,
    `Diminishing, but not negligible — five to all 32 is ${mf1(100 * (AG.all32.rmse_a / AG.top5.rmse_a - 1))}%`,
  ], 0.9, 5.00, 5.3, 1.56, 12.5);
  s.addShape(pptx.ShapeType.roundRect, { x: 6.82, y: 4.44, w: 5.88, h: 2.2,
    fill: { color: LIGHT }, rectRadius: 0.06, line: { color: 'E1E0D9' } });
  s.addText('Available five days earlier at no cost', { x: 7.12, y: 4.62, w: 5.3, h: 0.35,
    fontFace: HEAD, fontSize: 16, bold: true, color: MID, margin: 0 });
  bullets(s, [
    `Day 10 of the fortnight: ${f3(AG.d10_all32.rmse_a)} pp`,
    `Close of fortnight: ${f3(AG.all32.rmse_a)} pp`,
    `Day 5: ${f3(AG.d5_all32.rmse_a)} pp — too early`,
    'So publish on day 10 and gain five days of lead time',
  ], 7.12, 5.00, 5.3, 1.56, 12.5);
  s.addNotes('The day-10 finding held for jitomate, calabacita and chayote separately before it held in aggregate. It is the operationally useful part.');
}

// ----------------------------------------------------------------- 8 all 32 (two halves)
const NPG = `${F.n_range[0]}\u2013${F.n_range[F.n_range.length - 1]} scorable fortnights per generic`;
chartSlide(`The sixteen that matter most: ${f1(SPL.first16)}% of the variance`,
  `All ${F.n_generics} generics, 1 of 2, ordered by variance contribution. Panel titles give nowcast RMSE vs the CPI-only benchmark, in pp of the print.`,
  'charts/system_grid_a.png',
  `Source: SNIIM and INEGI. Evaluation window ${AG.n} fortnights, ${NPG}, rolling five-year window.`);
chartSlide(`The tail: ${f1(SPL.last16)}% of the variance`,
  `All ${F.n_generics} generics, 2 of 2, same presentation. One of the two generics gated out on data quality sits here, shown at its benchmark.`,
  'charts/system_grid_b.png',
  'Source: SNIIM and INEGI. Frijol and Chile seco are the weekly granos module: 9.6% of subindex weight but 0.0% of its variance.');

// ----------------------------------------------------------------- 9 per-generic table
{
  const s = contentSlide('Top ten by variance contribution',
    `RMSE in pp of the published print, ${F.n_admitted} of ${F.n_generics} generics admitted. “Benchmark” is the same model with no wholesale data.`);
  const rows = [[
    { text: 'Generic', options: { bold: true, color: WHITE } },
    { text: 'Share of var.', options: { bold: true, color: WHITE, align: 'right' } },
    { text: 'Nowcast', options: { bold: true, color: WHITE, align: 'right' } },
    { text: 'Benchmark', options: { bold: true, color: WHITE, align: 'right' } },
    { text: 'Improvement', options: { bold: true, color: WHITE, align: 'right' } },
    { text: 'Correct sign', options: { bold: true, color: WHITE, align: 'right' } },
  ]];
  const data = F.table.slice(0, 10).map(r => r.admisible
    ? [r.generico, f1(r.share) + '%', f2(r.close), f2(r.bench),
       '\u2212' + Math.round(r.gain_pct) + '%', pc0(r.sign)]
    : [r.generico, f1(r.share) + '%', 'gated', f2(r.bench), '—', '—']);
  data.forEach((r, i) => rows.push(r.map((c, j) => ({
    text: c, options: { align: j ? 'right' : 'left', bold: i === 0,
      color: (j === 4 && c !== '—') ? DEEP : INK } }))));
  rows[0] = rows[0].map(c => ({ text: c.text,
    options: { ...c.options, fill: { color: MID }, color: WHITE, bold: true } }));
  s.addTable(rows, { x: 0.6, y: 1.75, w: 12.1, colW: [4.3, 1.7, 1.5, 1.7, 1.7, 1.2],
    rowH: 0.36, fontFace: BODY, fontSize: 12.5,
    border: { type: 'solid', color: 'E1E0D9', pt: 0.5 },
    fill: { color: WHITE }, valign: 'middle', margin: 0.08 });
  s.addText(`${F.gated[1]} and ${F.gated[0]} are gated out on data quality, not performance: their retail prices are quoted by bunch or by piece rather than by kilogram, and both have long gaps in the published series. They are carried at their CPI-only benchmark and are ${f1(F.gated_share)}% of the variance between them. RMSE is scored only where both models are defined: ${F.table[0].n} fortnights for every admitted generic above, ${GATED.n} for the gated one.`,
    { x: 0.6, y: 6.13, w: 12.1, h: 0.8, fontFace: BODY, fontSize: 12, color: SEC, margin: 0 });
  s.addNotes('Papa is the interesting weak case: large weight, low volatility, and a storable with buffer stocks, so wholesale tells you much less. Same pattern in manzana, pera and otras frutas.');
}

// ----------------------------------------------------------------- 10 method
{
  const s = contentSlide('Method, and how we know it is not overfitted',
    'Everything below was fixed before the evaluation window was scored.');
  const left = [
    ['Chained matched-cell index', 'Variety × market pairs present in consecutive fortnights, geometric, INPC city weights. A market that stops quoting cannot move it.'],
    ['Rolling five-year window', 'Pass-through has sped up: within-fortnight beta rose from 0.54 in 1999–2007 to 0.83 in 2017–2026. Window length chosen on 2006–2010 only.'],
    ['Combination, not selection', 'Two pre-registered specifications, equal weights. Choosing among statistically tied models on out-of-sample results is itself a leak.'],
  ];
  const right = [
    ['Structural look-ahead audit', 'No CPI-derived regressor may be dated inside the target fortnight. This caught a real bug in an earlier version and now fails the build.'],
    ['Placebo regressor', 'The same wholesale series shifted to the wrong dates scores −0.3% against the benchmark — the harness manufactures no skill.'],
    ['Calibrated intervals', `Interval width from each series’ own past standardised errors, not a normal table. Realised coverage 79% at a nominal 80% across the admitted generics (${pc0(J.cov80)} for jitomate alone).`],
  ];
  [left, right].forEach((col, ci) => col.forEach(([h, t], i) => {
    const x = 0.6 + ci * 6.3, y = 1.72 + i * 1.72;
    s.addShape(pptx.ShapeType.ellipse, { x, y, w: 0.34, h: 0.34,
      fill: { color: DEEP }, line: { color: DEEP } });
    s.addText(h, { x: x + 0.5, y: y - 0.04, w: 5.3, h: 0.34, fontFace: HEAD, fontSize: 15,
      bold: true, color: MID, margin: 0 });
    s.addText(t, { x: x + 0.5, y: y + 0.34, w: 5.3, h: 1.1, fontFace: BODY, fontSize: 12,
      color: INK, margin: 0, valign: 'top' });
  }));
}

// ----------------------------------------------------------------- 11 limitations
{
  const s = contentSlide('What this does not do',
    'The four caveats a reviewer should raise first, answered.');
  const items = [
    ['It is not a forecast', 'The whole gain comes from the publication lag. A true one-step-ahead forecast beats the benchmark by ~5%. Wholesale does not lead retail: the correlation at one lag is 0.16 against 0.86 contemporaneous.'],
    ['Large moves are under-called', 'Residuals average +1.7 pp on fortnights where the CPI moves more than 25%, symmetric in both directions. Three attempted fixes — convexity, asymmetry, instrumental variables — all failed.'],
    ['Two generics are not nowcastable', `Cilantro is 95% quoted by the bunch at retail, so a price per kilo is not comparable. Otras verduras y legumbres is also 30% non-kilo, and its published series is scorable in only ${GATED.n} of the ${AG.n} fortnights of the evaluation window. Both are carried at benchmark — together ${f1(F.gated_share)}% of the variance.`],
    ['Better measurement will not help', 'Split-half instrumenting puts the index’s reliability at 0.966 — only 3% of its variance is sampling noise. More markets or more quote days buy almost nothing.'],
  ];
  items.forEach(([h, t], i) => {
    const y = 1.80 + i * 1.36;
    s.addText(h, { x: 0.6, y: y - 0.02, w: 3.8, h: 0.45, fontFace: HEAD, fontSize: 15,
      bold: true, color: MID, margin: 0, valign: 'top' });
    s.addText(t, { x: 4.6, y: y - 0.02, w: 8.1, h: 1.1, fontFace: BODY, fontSize: 12.5,
      color: INK, margin: 0, valign: 'top' });
  });
  s.addNotes('Also worth volunteering: no formal cointegration test yet, and the granos module (frijol, chile seco) is 9.6% of subindex weight but 0.0% of its variance, so it earns no place in a produce-inflation discussion.');
}

// ----------------------------------------------------------------- 12 next
{
  const s = pptx.addSlide();
  s.background = { color: MID };
  s.addText('Where this goes next', { x: 0.9, y: 0.9, w: 11.5, h: 0.7, fontFace: HEAD,
    fontSize: 34, bold: true, color: WHITE, margin: 0 });
  const items = [
    ['Operational', 'Publish on day 10 of each fortnight, when the number is already as accurate as at close. Rebuild daily; the pipeline is one command.'],
    ['City-level bottom-up', 'Nowcast each city’s CPI from its own markets and aggregate with city weights. The one untested idea with a mechanism behind it.'],
    ['Discounted least squares', 'Ties the rolling window today (0.03 pp) and degrades more gracefully for thin generics. Adopt when the window starts to bind.'],
    ['Extend beyond produce', 'Pecuarios has the same structure: a published wholesale source and a lagged CPI. Frutas y verduras is 4.8 of 100 in the basket; agropecuarios is far larger.'],
  ];
  items.forEach(([h, t], i) => {
    const x = 0.9 + (i % 2) * 6.0, y = 2.1 + Math.floor(i / 2) * 2.35;
    s.addShape(pptx.ShapeType.roundRect, { x, y, w: 5.5, h: 1.95, fill: { color: DEEP },
      rectRadius: 0.06, line: { color: DEEP } });
    s.addText(h, { x: x + 0.3, y: y + 0.2, w: 4.9, h: 0.35, fontFace: HEAD, fontSize: 16,
      bold: true, color: WHITE, margin: 0 });
    s.addText(t, { x: x + 0.3, y: y + 0.6, w: 4.9, h: 1.2, fontFace: BODY, fontSize: 12,
      color: 'CADCFC', margin: 0 });
  });
  s.addText('All data public and free. Full history 1998–2026 collected and reproducible.',
    { x: 0.9, y: 6.72, w: 11.5, h: 0.35, fontFace: BODY, fontSize: 11, color: MUT, margin: 0 });
}

pptx.writeFile({ fileName: 'produce_cpi_nowcast.pptx' }).then(() => console.log('written'));
