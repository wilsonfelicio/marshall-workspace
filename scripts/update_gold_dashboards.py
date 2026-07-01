#!/usr/bin/env python3
"""
Update the gold / gold-miners analyses + dashboards.

Refreshes both:
  gold-miners-chart/   (index, residuals, scatter, multi-regression)
  gdx-elasticnet/      (9-factor elastic-net factor decomposition)

Pulls weekly data from Yahoo Finance, refits every model, and rewrites the
embedded data + stats in the existing HTML files (structure preserved).

Usage:
  python3 scripts/update_gold_dashboards.py            # compute + print, no write
  python3 scripts/update_gold_dashboards.py --write    # also rewrite the HTML
"""
import sys, os, re, json, datetime
import numpy as np, pandas as pd
import yfinance as yf
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRITE = "--write" in sys.argv
TODAY = datetime.date.today().isoformat()
START = "2021-03-15"

EN_FACTORS = ["Gold", "SPY", "TLT", "DBC", "TIP", "HYG", "DXY", "EEM", "FXI"]
TICKERS = {
    "GDX": "GDX", "GDXJ": "GDXJ", "Gold": "GC=F", "SPY": "SPY", "RSP": "RSP",
    "TLT": "TLT", "DBC": "DBC", "TIP": "TIP", "DXY": "DX-Y.NYB",
    "HYG": "HYG", "EEM": "EEM", "FXI": "FXI",
}

# ---------- data ----------
print("Downloading weekly data from Yahoo ...")
raw = yf.download(list(TICKERS.values()), start=START, interval="1wk",
                  auto_adjust=True, progress=False)["Close"]
raw = raw.rename(columns={v: k for k, v in TICKERS.items()})
raw = raw[list(TICKERS.keys())].dropna(how="all")
print(f"weeks: {len(raw)}  span: {raw.index[0].date()} -> {raw.index[-1].date()}")

rets = raw.pct_change()  # fractional weekly returns


def js(records):
    return json.dumps(records, separators=(",", ":"))


def replace_array(html, name, records):
    pat = re.compile(r"const\s+" + re.escape(name) + r"\s*=\s*\[.*?\];", re.DOTALL)
    new = f"const {name}={js(records)};"
    html2, n = pat.subn(new, html, count=1)
    assert n == 1, f"[{name}] array replace matched {n} times"
    return html2


def sub1(html, pat, repl, tag, dotall=False):
    flags = re.DOTALL if dotall else 0
    html2, n = re.subn(pat, repl, html, count=1, flags=flags)
    assert n == 1, f"[{tag}] matched {n} times"
    return html2


def dstr(idx):
    return [d.strftime("%Y-%m-%d") for d in idx]

# ================= gold-miners-chart / index.html =================
sub = raw[["GDX", "GDXJ", "Gold"]].dropna()
base = sub.iloc[0]
gN = [{"date": d, "value": round(v, 2)} for d, v in zip(dstr(sub.index), sub["GDX"] / base["GDX"] * 100)]
jN = [{"date": d, "value": round(v, 2)} for d, v in zip(dstr(sub.index), sub["GDXJ"] / base["GDXJ"] * 100)]
auN = [{"date": d, "value": round(v, 2)} for d, v in zip(dstr(sub.index), sub["Gold"] / base["Gold"] * 100)]
ratio = [{"date": d, "value": round(v, 4)} for d, v in zip(dstr(sub.index), sub["GDX"] / sub["Gold"] * 10)]
px_gold, px_gdx, px_gdxj = sub["Gold"].iloc[-1], sub["GDX"].iloc[-1], sub["GDXJ"].iloc[-1]
print(f"\n[index] Gold ${px_gold:,.0f}  GDX ${px_gdx:.2f}  GDXJ ${px_gdxj:.2f}  ratio {ratio[-1]['value']}")

# ================= simple OLS  GDX ~ Gold  (fractional) =================
s = rets[["GDX", "Gold"]].dropna()
x, y = s["Gold"].values, s["GDX"].values
X = np.column_stack([np.ones(len(x)), x])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
alpha_s, beta_s = b[0], b[1]
pred = X @ b
resid_s = y - pred
r2_s = 1 - resid_s.var() / y.var()
corr_s = float(np.corrcoef(x, y)[0, 1])
sig_s = resid_s.std()
n_s = len(s)
print(f"[simple]  beta {beta_s:.3f}  alpha_wk {alpha_s*100:.3f}%  R2 {r2_s:.3f}  corr {corr_s:.2f}  sig {sig_s*100:.2f}%  n {n_s}")

Rres = [{"date": d, "value": round(float(r), 6)} for d, r in zip(dstr(s.index), resid_s)]
cum = np.cumsum(resid_s)
Cres = [{"date": d, "value": round(float(v), 6)} for d, v in zip(dstr(s.index), cum)]
roll = pd.Series(resid_s, index=s.index).rolling(12).mean().dropna()
Lres = [{"date": d, "value": round(float(v), 6)} for d, v in zip(dstr(roll.index), roll.values)]
scatter = [{"x": round(float(gx), 6), "y": round(float(gy), 6), "date": d}
           for d, gx, gy in zip(dstr(s.index), x, y)]

# last-5-trading-day move (daily)
dd = yf.download(["GDX", "GC=F"], period="1mo", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
g5 = (dd["GC=F"].iloc[-1] / dd["GC=F"].iloc[-6] - 1) * 100
x5 = (dd["GDX"].iloc[-1] / dd["GDX"].iloc[-6] - 1) * 100
implied5 = beta_s * g5
outperf = "outperforming" if x5 > implied5 else "underperforming"
mag = "massively " if abs(x5 - implied5) > 4 else ""
print(f"[last5]   Gold {g5:+.2f}%  GDX {x5:+.2f}%  implied {implied5:+.2f}%  ({mag}{outperf})")

# ================= multiple OLS  GDX ~ Gold + RSP  (percent) =================
m = rets[["GDX", "Gold", "RSP"]].dropna() * 100
Xm = np.column_stack([np.ones(len(m)), m["Gold"].values, m["RSP"].values])
bm, *_ = np.linalg.lstsq(Xm, m["GDX"].values, rcond=None)
a_m, bg_m, br_m = bm
predm = Xm @ bm
resid_m = m["GDX"].values - predm
r2_m = 1 - resid_m.var() / m["GDX"].values.var()
n_m = len(m)
print(f"[multi]   a {a_m:.3f}  bGold {bg_m:.3f}  bRSP {br_m:.3f}  R2 {r2_m:.3f}  ann_a {a_m*52:.1f}%  n {n_m}")

R2arr = [{"date": d, "resid": round(float(r), 3), "gold": round(float(g), 4), "rsp": round(float(rr), 4)}
         for d, r, g, rr in zip(dstr(m.index), resid_m, m["Gold"].values, m["RSP"].values)]
cumm = np.cumsum(resid_m)
C2arr = [{"date": d, "cum": round(float(v), 3)} for d, v in zip(dstr(m.index), cumm)]
rollm = pd.Series(resid_m, index=m.index).rolling(12).mean().dropna()
L2arr = [{"date": d, "val": round(float(v), 3)} for d, v in zip(dstr(rollm.index), rollm.values)]
contribG = pd.Series(bg_m * m["Gold"].values, index=m.index).rolling(12).mean().dropna()
contribR = pd.Series(br_m * m["RSP"].values, index=m.index).rolling(12).mean().dropna()
cGarr = [{"date": d, "val": round(float(v), 4)} for d, v in zip(dstr(contribG.index), contribG.values)]
cRarr = [{"date": d, "val": round(float(v), 4)} for d, v in zip(dstr(contribR.index), contribR.values)]

# ================= elastic net  GDX ~ 9 factors  (percent) =================
cols = ["GDX"] + EN_FACTORS
e = rets[cols].dropna() * 100
Xe = e[EN_FACTORS].values
ye = e["GDX"].values
en = ElasticNet(alpha=0.001, l1_ratio=0.9, max_iter=400000).fit(Xe, ye)
enBeta = en.coef_
intercept = en.intercept_
sc = StandardScaler()
en_std = ElasticNet(alpha=0.001, l1_ratio=0.9, max_iter=400000).fit(sc.fit_transform(Xe), ye)
enBetaStd = en_std.coef_
Xo = np.column_stack([np.ones(len(Xe)), Xe])
bo, *_ = np.linalg.lstsq(Xo, ye, rcond=None)
olsBeta = bo[1:]
k = int(len(e) * 0.8)
en_tr = ElasticNet(alpha=0.001, l1_ratio=0.9, max_iter=400000).fit(Xe[:k], ye[:k])
r2_en = r2_score(ye[k:], en_tr.predict(Xe[k:]))
resid_en = ye - en.predict(Xe)
kept = int(np.sum(np.abs(enBeta) > 1e-6))
n_e = len(e)
years = (e.index[-1] - e.index[0]).days / 365.25
print(f"[en]      R2(test) {r2_en:.3f}  a_wk {intercept:.3f}%  a_ann {intercept*52:.1f}%  kept {kept}/9  n {n_e}")
for f, rb, sb, ob in zip(EN_FACTORS, enBeta, enBetaStd, olsBeta):
    print(f"          {f:5s} raw {rb:+.4f}  std {sb:+.4f}  ols {ob:+.4f}")

edates = dstr(e.index)
residuals_en = [{"date": d, "resid": round(float(r), 3), "gold": round(float(g), 4), "gdx": round(float(gx), 4)}
                for d, r, g, gx in zip(edates, resid_en, e["Gold"].values, e["GDX"].values)]
cume = np.cumsum(resid_en)
cumResid = [{"date": d, "cum": round(float(v), 3)} for d, v in zip(edates, cume)]
rolle = pd.Series(resid_en, index=e.index).rolling(12).mean().dropna()
rollingResid = [{"date": d, "val": round(float(v), 3)} for d, v in zip(dstr(rolle.index), rolle.values)]
pe = raw.loc[e.index, ["GDX", "Gold"]]
gdxNorm = [{"date": d, "value": round(float(v), 2)} for d, v in zip(edates, pe["GDX"] / pe["GDX"].iloc[0] * 100)]
goldNorm = [{"date": d, "value": round(float(v), 2)} for d, v in zip(edates, pe["Gold"] / pe["Gold"].iloc[0] * 100)]

if not WRITE:
    print("\n(compute-only; re-run with --write to update the HTML files)")
    sys.exit(0)

# ============================ WRITE FILES ============================
GM = os.path.join(ROOT, "gold-miners-chart")
EN = os.path.join(ROOT, "gdx-elasticnet")


def load(p):
    return open(p, encoding="utf-8").read()


def save(p, s):
    open(p, "w", encoding="utf-8").write(s)
    print("wrote", os.path.relpath(p, ROOT))

# ---- index.html ----
p = os.path.join(GM, "index.html"); h = load(p)
h = replace_array(h, "gN", gN); h = replace_array(h, "jN", jN)
h = replace_array(h, "auN", auN); h = replace_array(h, "ratio", ratio)
h = sub1(h, r'Updated \d{4}-\d{2}-\d{2}', f'Updated {TODAY}', "idx-date")
h = sub1(h, r'(<div class="label">Gold</div><div class="val"[^>]*>)\$[\d,\.]+(</div>)',
         rf'\g<1>${px_gold:,.0f}\g<2>', "idx-gold")
h = sub1(h, r'(<div class="label">GDX</div><div class="val"[^>]*>)\$[\d,\.]+(</div>)',
         rf'\g<1>${px_gdx:,.2f}\g<2>', "idx-gdx")
h = sub1(h, r'(<div class="label">GDXJ</div><div class="val"[^>]*>)\$[\d,\.]+(</div>)',
         rf'\g<1>${px_gdxj:,.2f}\g<2>', "idx-gdxj")
save(p, h)

# ---- residuals.html ----
p = os.path.join(GM, "residuals.html"); h = load(p)
h = replace_array(h, "R", Rres); h = replace_array(h, "C", Cres); h = replace_array(h, "L", Lres)
h = sub1(h, r'\d+ obs \| Updated \d{4}-\d{2}-\d{2}', f'{n_s} obs | Updated {TODAY}', "res-meta")
h = sub1(h, r'<h3>OLS: GDX = [^<]+</h3>',
         f'<h3>OLS: GDX = {alpha_s*100:.3f} + {beta_s:.3f} × Gold</h3>', "res-eq")
h = sub1(h, r'(<div class="label">Beta</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{beta_s:.3f}\g<2>', "res-beta")
h = sub1(h, r'(<div class="label">Alpha \(wk\)</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{alpha_s*100:.3f}%\g<2>', "res-awk")
h = sub1(h, r'(<div class="label">Alpha \(ann\)</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{alpha_s*100*52:.1f}%\g<2>', "res-aann")
h = sub1(h, r'(<div class="label">R²</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{r2_s:.3f}\g<2>', "res-r2")
h = sub1(h, r'(<div class="label">Resid σ</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{sig_s*100:.2f}%\g<2>', "res-sig")
save(p, h)

# ---- scatter.html ----
p = os.path.join(GM, "scatter.html"); h = load(p)
h = replace_array(h, "scatter", scatter)
h = sub1(h, r'const last5=\{[^}]*\};', f'const last5={{"gold": {g5:.3f}, "gdx": {x5:.3f}}};', "sc-last5")
h = sub1(h, r'const beta=[-\d.]+;', f'const beta={beta_s:.4f};', "sc-beta")
h = sub1(h, r'const alpha=[-\d.]+;', f'const alpha={alpha_s:.6f};', "sc-alpha")
h = sub1(h, r'\d+ obs \| Updated \d{4}-\d{2}-\d{2}', f'{n_s} obs | Updated {TODAY}', "sc-meta")
h = sub1(h, r'(<div class="label">Beta</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{beta_s:.2f}x\g<2>', "sc-beta2")
h = sub1(h, r'(<div class="label">Alpha \(wk\)</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{alpha_s*100:.3f}%\g<2>', "sc-awk")
h = sub1(h, r'(<div class="label">R²</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{r2_s:.2f}\g<2>', "sc-r2")
h = sub1(h, r'(<div class="label">Correlation</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{corr_s:.2f}\g<2>', "sc-corr")
h = sub1(h, r'<div class="hl">.*?</div>',
         (f'<div class="hl"><strong style="color:#ef4444">Last 5 Days:</strong> '
          f'Gold <strong>{g5:+.2f}%</strong>, GDX <strong>{x5:+.2f}%</strong> — '
          f'beta-implied GDX: {implied5:+.2f}%. Miners {mag}{outperf}</div>'), "sc-hl")
save(p, h)

# ---- multi-regression.html ----
p = os.path.join(GM, "multi-regression.html"); h = load(p)
h = replace_array(h, "R2", R2arr); h = replace_array(h, "C2", C2arr); h = replace_array(h, "L2", L2arr)
h = replace_array(h, "cG", cGarr); h = replace_array(h, "cR", cRarr)
h = sub1(h, r'\d+ obs \| Updated \d{4}-\d{2}-\d{2}', f'{n_m} obs | Updated {TODAY}', "mr-meta")
h = sub1(h, r'<div class="eq">GDX = [^<]+</div>',
         f'<div class="eq">GDX = {a_m:.3f} + {bg_m:.3f} × Gold + {br_m:.3f} × RSP + ε</div>', "mr-eq")
h = sub1(h, r'(<div class="label">β Gold</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{bg_m:.3f}\g<2>', "mr-bg")
h = sub1(h, r'(<div class="label">β RSP</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{br_m:.3f}\g<2>', "mr-br")
h = sub1(h, r'(<div class="label">α \(ann\)</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{a_m*52:.1f}%\g<2>', "mr-aann")
h = sub1(h, r'(<div class="label">R²</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{r2_m:.3f}\g<2>', "mr-r2")
save(p, h)

# ---- gdx-elasticnet/index.html ----
p = os.path.join(EN, "index.html"); h = load(p)
h = replace_array(h, "enBeta", [round(float(v), 4) for v in enBeta])
h = replace_array(h, "enBetaStd", [round(float(v), 4) for v in enBetaStd])
h = replace_array(h, "olsBeta", [round(float(v), 4) for v in olsBeta])
h = replace_array(h, "residuals", residuals_en)
h = replace_array(h, "cumResid", cumResid)
h = replace_array(h, "rollingResid", rollingResid)
h = replace_array(h, "gdxNorm", gdxNorm)
h = replace_array(h, "goldNorm", goldNorm)
h = sub1(h, r'\| [\d.]+ Years \| \d+ observations', f'| {years:.1f} Years | {n_e} observations', "en-meta")
# equation
terms = "".join(
    (f" + {b:.3f}×{f}" if b >= 0 else f" - {abs(b):.3f}×{f}")
    for f, b in zip(EN_FACTORS, enBeta))
eq = f'<div class="eq">GDX = {intercept:.3f}{terms} + ε</div>'
h = sub1(h, r'<div class="eq">GDX = .*?</div>', eq, "en-eq")
# coefficient table
maxabs = max(abs(v) for v in enBetaStd) or 1.0
rows = []
for f, rb, sb, ob in zip(EN_FACTORS, enBeta, enBetaStd, olsBeta):
    cls = "pos" if rb >= 0 else "neg"
    clss = "pos" if sb >= 0 else "neg"
    barcol = "#22c55e" if sb >= 0 else "#ef4444"
    w = round(abs(sb) / maxabs * 100)
    rows.append(
        f'        <tr>\n'
        f'            <td style="font-weight:600">{f}</td>\n'
        f'            <td class="{cls}">{rb:+.4f}</td>\n'
        f'            <td class="{clss}">{sb:+.4f}</td>\n'
        f'            <td>{ob:+.4f}</td>\n'
        f'            <td><div style="display:flex;align-items:center;gap:6px;"><div style="width:{w}%;height:8px;background:{barcol};border-radius:4px;min-width:2px;"></div><span style="font-size:0.75rem;color:var(--muted)">{abs(sb):.2f}</span></div></td>\n'
        f'        </tr>')
table = ('<table>\n        <tr><th>Factor</th><th>Raw β</th><th>Std β</th>'
         '<th>OLS β</th><th>Importance</th></tr>\n' + "\n".join(rows) + '\n    </table>')
h = sub1(h, r'<table>.*?</table>', table, "en-table", dotall=True)
# stat row (targeted per-stat, avoids nested-div clipping)
acol = "#ef4444" if intercept < 0 else "#22c55e"
h = sub1(h, r'(<div class="label">R²</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{r2_en:.3f}\g<2>', "en-r2")
h = sub1(h, r'(<div class="label">α \(weekly\)</div><div class="val")[^>]*(>)[^<]+(</div>)', rf'\g<1> style="color:{acol}"\g<2>{intercept:.3f}%\g<3>', "en-awk")
h = sub1(h, r'(<div class="label">α \(annualized\)</div><div class="val")[^>]*(>)[^<]+(</div>)', rf'\g<1> style="color:{acol}"\g<2>{intercept*52:.1f}%\g<3>', "en-aann")
h = sub1(h, r'(<div class="label">Factors kept</div><div class="val"[^>]*>)[^<]+(</div>)', rf'\g<1>{kept}/9\g<2>', "en-kept")
# interpretation numbers
bmap = dict(zip(EN_FACTORS, enBeta))
smap = dict(zip(EN_FACTORS, enBetaStd))
h = h.replace("Gold (β=1.53)", f"Gold (β={bmap['Gold']:.2f})")
h = h.replace("3.5σ move in GDX", f"{smap['Gold']:.1f}σ move in GDX")
h = h.replace("EEM (β=0.48)", f"EEM (β={bmap['EEM']:.2f})")
h = h.replace("DXY (β=-0.71)", f"DXY (β={bmap['DXY']:.2f})")
h = h.replace("FXI (β=-0.16)", f"FXI (β={bmap['FXI']:.2f})")
save(p, h)

print("\nAll dashboards updated.")
