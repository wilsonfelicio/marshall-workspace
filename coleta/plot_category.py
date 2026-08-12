#!/usr/bin/env python3
"""Build a self-contained HTML dashboard for one INPC category.

  python plot_category.py aguacate
  python plot_category.py jitomate --min-cobertura 30
  python plot_category.py --list

Produces `charts/<slug>.html` with four panels: variety indices vs the category
aggregate, wholesale vs INPC retail, year-over-year for both, and the YoY gap.

Reads only from data/curated/ and data/inpc/, so it is safe to run while a
backfill is in progress and it never takes the store lock.

Two things it does that a quick plot would get wrong:

* **Variety names come from the catalog, not the data.** SNIIM's results page
  header drops the quality suffix, so producto_ids 133, 136 and 137 all report
  themselves as "Aguacate Hass". Plotting the header string would put three
  identically-labelled lines on the chart.
* **The current month is excluded.** It is partial, so its monthly average would
  read as a price collapse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "charts"
BASE = pd.Period("2018-01", freq="M")

# Fixed categorical order. Slot 1 always goes to the aggregate, so the aggregate
# keeps one hue across every panel (colour follows the entity).
SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MAX_VARIEDADES = 7  # + the aggregate = 8, the palette ceiling


def load(slug: str, min_cob: float):
    cur = ROOT / "data" / "curated"
    idx = pd.read_parquet(cur / "cat_index_monthly.parquet")
    idx["mes"] = pd.PeriodIndex(pd.to_datetime(idx["mes"]), freq="M")
    var = pd.read_parquet(cur / "var_national_monthly.parquet")
    var["mes"] = pd.PeriodIndex(pd.to_datetime(var["mes"]), freq="M")

    if slug not in set(idx["categoria"]):
        raise SystemExit(f"category {slug!r} not in cat_index_monthly. "
                         f"Available: {', '.join(sorted(set(idx['categoria'])))}")

    # Drop the incomplete current month.
    last = max(m for m in idx["mes"] if m < pd.Period.now("M"))
    idx, var = idx[idx.mes <= last], var[var.mes <= last]
    ci = idx[idx.categoria == slug].set_index("mes").sort_index()
    cv = var[var.categoria == slug]

    # INPC retail counterpart.
    mp = json.loads((ROOT / "config" / "map_sniim_inpc.json").read_text(encoding="utf-8"))
    g = pd.read_parquet(ROOT / "data" / "inpc" / "inpc_genericos_mensual.parquet")
    gi = pd.PeriodIndex([pd.Period(year=int(a), month=int(m), freq="M")
                         for a, m in zip(g.anio, g.mes)], freq="M")
    retail = pd.Series(g[mp[slug]].values, index=gi).dropna()
    retail = retail[retail.index <= last]

    # Catalog labels - see the module docstring.
    cat = pd.read_csv(ROOT / "data" / "catalog" / "frutas_productos.csv")
    label = {int(r.id): str(r.label).replace(" - Primera", "").strip()
             for r in cat.itertuples()}

    n_meses = ci.shape[0]
    cov = (cv.groupby("producto_id").size() / n_meses * 100).sort_values(ascending=False)
    shown = [int(p) for p in cov.index if cov[p] >= min_cob][:MAX_VARIEDADES]
    hidden = [(label.get(int(p), str(p)), round(float(cov[p]), 0))
              for p in cov.index if int(p) not in shown]
    return ci, cv, retail, label, shown, hidden, cov, mp[slug], last


MIN_BASE_MESES = 6   # months of the base year a series needs to be comparable


def rebase(s: pd.Series):
    """Rebase to the geometric mean of calendar 2018 = 100.

    NOT to a single month. An earlier version used Jan-2018 and fell back to the
    series' own first observation when that month was missing - which silently put
    seasonal varieties on a different base and made their LEVELS incomparable with
    everything else on the chart (Aguacate Hass adelantado, with no Jan-2018
    observation, plotted at 269 against an aggregate of 139 purely as an artifact).

    Returns (rebased, n_months_in_base_year). The caller must drop any series with
    fewer than MIN_BASE_MESES, because a base built from two months of a strongly
    seasonal product is not comparable either.
    """
    s = s.dropna()
    s = s[s > 0]
    base_yr = s[[m for m in s.index if m.year == BASE.year]]
    if len(base_yr) == 0:
        return None, 0
    b = float(np.exp(np.log(base_yr).mean()))
    return 100 * s / b, len(base_yr)


def build(slug: str, min_cob: float) -> Path:
    ci, cv, retail, label, shown, hidden, cov, gname, last = load(slug, min_cob)
    meses = list(ci.index)

    agg, n_agg = rebase(ci["indice_jevons"])
    ret, _ = rebase(retail)
    agg_on_base = n_agg >= MIN_BASE_MESES

    series = [{"name": "Agregado de la categoría", "short": "Agregado",
               "vals": [None if m not in agg.index or pd.isna(agg[m]) else round(float(agg[m]), 2)
                        for m in meses], "lead": True}]
    dropped_base = []
    for pid in shown:
        s = cv[cv.producto_id == pid].set_index("mes")["precio_geo"].sort_index()
        r, nb = rebase(s)
        nm = label.get(pid, str(pid))
        if r is None or nb < MIN_BASE_MESES:
            # No comparable base -> plotting its level would be misleading.
            dropped_base.append((nm, nb))
            continue
        series.append({"name": nm,
                       "short": nm.replace("Aguacate ", "").replace("Cebolla ", "")
                                  .replace("Calidad ", "")[:22],
                       "vals": [None if m not in r.index or pd.isna(r[m]) else round(float(r[m]), 2)
                                for m in meses], "lead": False})

    wl = agg.reindex(meses)
    rl = ret.reindex(meses)
    wy = (wl / wl.shift(12) - 1) * 100
    ry = (rl / rl.shift(12) - 1) * 100
    gap = wy - ry
    ok = wy.notna() & ry.notna()
    corr_mom = float(np.corrcoef(np.log(wl[wl.notna() & rl.notna()]).diff().dropna(),
                                np.log(rl[wl.notna() & rl.notna()]).diff().dropna())[0, 1])
    # Amplitude on LOG changes, not percent changes. Percent changes are
    # asymmetric for large moves (+300% and -75% are the same log distance) and
    # these series reach +345%, which compresses whichever side has the bigger
    # upside. Measured on percent changes cebolla read 0.97 - i.e. "retail is as
    # volatile as wholesale" - when on log changes it is 1.05.
    lw = np.log(wl[ok]) - np.log(wl.shift(12)[ok])
    lr = np.log(rl[ok]) - np.log(rl.shift(12)[ok])
    amp = float(lw.std() / lr.std())
    lead = float(100 * (gap[ok] > 0).mean())

    payload = {
        "slug": slug, "generico": gname,
        "meses": [str(m) for m in meses],
        "series": series,
        "retail": [None if pd.isna(rl[m]) else round(float(rl[m]), 2) for m in meses],
        "wy": [None if pd.isna(wy[m]) else round(float(wy[m]), 2) for m in meses],
        "ry": [None if pd.isna(ry[m]) else round(float(ry[m]), 2) for m in meses],
        "gap": [None if pd.isna(gap[m]) else round(float(gap[m]), 2) for m in meses],
        "light": SLOTS_LIGHT[:len(series)], "dark": SLOTS_DARK[:len(series)],
        "stats": {"corr_mom": round(corr_mom, 3), "amp": round(amp, 2),
                  "lead": round(lead, 0), "n_yoy": int(ok.sum()),
                  "n_meses": len(meses), "base_ok": bool(agg_on_base)},
        "hidden": hidden,
        "sin_base": dropped_base,
        "base_anio": BASE.year, "min_base": MIN_BASE_MESES,
    }
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{slug}.html"
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(payload)), encoding="utf-8")
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Precios mayoristas y INPC</title>
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--plane:#f9f9f7;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--zero:#898781;
 --border:rgba(11,11,11,0.10);--pos:#2a78d6;--neg:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--plane:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--zero:#898781;--border:rgba(255,255,255,0.10);--pos:#3987e5;--neg:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--plane:#0d0d0d;
 --text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --zero:#898781;--border:rgba(255,255,255,0.10);--pos:#3987e5;--neg:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{background:var(--plane);color:var(--text-primary);padding:26px 22px 40px;max-width:1100px;margin:0 auto}
h1{font-size:20px;margin:0 0 6px;font-weight:650;line-height:1.3}
.sub{font-size:13px;color:var(--text-secondary);margin:0 0 18px;max-width:76ch;line-height:1.5}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.grp{display:flex;gap:2px;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:2px}
button{font:inherit;font-size:12px;padding:5px 10px;border:0;border-radius:6px;background:transparent;color:var(--text-secondary);cursor:pointer}
button[aria-pressed="true"]{background:var(--grid);color:var(--text-primary);font-weight:600}
button:hover{color:var(--text-primary)}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:11px 15px;min-width:142px}
.tile .k{font-size:11.5px;color:var(--muted);margin-bottom:3px}
.tile .v{font-size:21px;font-weight:640;line-height:1.15}
.tile .u{font-size:11px;color:var(--text-secondary);margin-top:2px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:17px 17px 8px;margin-bottom:16px}
.card h2{font-size:14px;margin:0 0 2px;font-weight:620}
.card p.note{font-size:12px;color:var(--muted);margin:0 0 11px;line-height:1.45;max-width:90ch}
.legend{display:flex;flex-wrap:wrap;gap:13px;margin:2px 0 9px;font-size:12px;color:var(--text-secondary)}
.legend span{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.legend i{width:14px;height:2.5px;border-radius:2px;display:inline-block}
.legend span[data-off="1"]{opacity:.35}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}
.zl{stroke:var(--zero);stroke-width:1.25;shape-rendering:crispEdges}
.ax{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.dl{font-size:11px;font-weight:600}
.cross{stroke:var(--axis);stroke-width:1;pointer-events:none}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
 border-radius:8px;padding:8px 10px;font-size:12px;color:var(--text-primary);box-shadow:0 4px 14px rgba(0,0,0,.13);min-width:210px;z-index:5}
.tip b{display:block;margin-bottom:5px;font-size:11px;color:var(--text-secondary);font-weight:600}
.tip div{display:flex;justify-content:space-between;gap:12px;line-height:1.6}
.tip i{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:6px;vertical-align:-1px}
.tip .v{font-variant-numeric:tabular-nums}
table{border-collapse:collapse;font-size:12px;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:4px 8px;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600;position:sticky;top:0;background:var(--surface-1)}
.tblwrap{max-height:430px;overflow:auto;border:1px solid var(--border);border-radius:8px}
.foot{font-size:11.5px;color:var(--muted);line-height:1.6;margin-top:13px;max-width:94ch}
.hide{display:none}
</style></head><body><div class="viz-root" id="root">
<h1 id="h1"></h1><p class="sub" id="sub"></p>
<div class="controls">
 <div class="grp" role="group" aria-label="Periodo">
  <button data-range="all" aria-pressed="true">Todo</button>
  <button data-range="2007" aria-pressed="false">2007+</button>
  <button data-range="2015" aria-pressed="false">2015+</button>
  <button data-range="5y" aria-pressed="false">Últimos 5 años</button></div>
 <div class="grp" role="group" aria-label="Escala"><button data-scale="log" aria-pressed="true">Log</button>
  <button data-scale="lin" aria-pressed="false">Lineal</button></div>
 <div class="grp" role="group" aria-label="Suavizado"><button data-smooth="0" aria-pressed="true">Mensual</button>
  <button data-smooth="1" aria-pressed="false">Media 12m</button></div>
 <div class="grp" role="group" aria-label="Vista"><button data-view="chart" aria-pressed="true">Gráficas</button>
  <button data-view="table" aria-pressed="false">Tabla</button></div>
 <div class="grp" role="group" aria-label="Tema"><button data-theme-btn="light" aria-pressed="false">Claro</button>
  <button data-theme-btn="dark" aria-pressed="false">Oscuro</button></div>
</div>
<div class="tiles" id="tiles"></div>
<div id="charts">
 <div class="card"><h2>Variedades y el agregado de la categoría</h2>
  <p class="note" id="n1"></p><div class="legend" id="lg1"></div>
  <div style="position:relative" id="w1"><svg id="c1" viewBox="0 0 900 370" role="img" aria-label="Índices de precio mayorista por variedad y el agregado"></svg></div></div>
 <div class="card"><h2>Mayorista contra menudeo</h2>
  <p class="note" id="n2"></p><div class="legend" id="lg2"></div>
  <div style="position:relative" id="w2"><svg id="c2" viewBox="0 0 900 290" role="img" aria-label="Índice mayorista SNIIM contra el INPC de menudeo"></svg></div></div>
 <div class="card"><h2>Variación anual: mayorista y menudeo</h2>
  <p class="note" id="n3"></p><div class="legend" id="lg3"></div>
  <div style="position:relative" id="w3"><svg id="c3" viewBox="0 0 900 320" role="img" aria-label="Variación anual del precio mayorista y del INPC"></svg></div></div>
 <div class="card"><h2>Brecha: variación mayorista menos variación menudeo</h2>
  <p class="note" id="n4"></p>
  <div style="position:relative" id="w4"><svg id="c4" viewBox="0 0 900 230" role="img" aria-label="Diferencia en puntos porcentuales entre las variaciones anuales"></svg></div></div>
</div>
<div id="tableview" class="card hide"><h2>Vista de tabla</h2>
 <p class="note">Índices con base enero 2018 = 100; variación anual en por ciento; brecha en puntos porcentuales.</p>
 <div class="tblwrap"><table id="tbl"></table></div></div>
<p class="foot" id="foot"></p></div>
<script>
const D=__DATA__;
const dark=()=>document.documentElement.getAttribute('data-theme')==='dark'||
  (!document.documentElement.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
const col=i=>(dark()?D.dark:D.light)[i%8];
const RET_COL=()=>dark()?'#d95926':'#eb6834';
let range='all',scale='log',view='chart',smooth=false,off=new Set();
const _c=new Map();
function sm(v){const o=new Array(v.length).fill(null),W=12;
 for(let i=0;i<v.length;i++){let s=0,n=0;
  for(let k=i-6;k<=i+6;k++) if(k>=0&&k<v.length&&v[k]!=null&&v[k]>0){s+=Math.log(v[k]);n++;}
  if(n>=W-2)o[i]=Math.exp(s/n);} return o;}
function V(key,raw){if(!smooth)return raw; if(!_c.has(key))_c.set(key,sm(raw)); return _c.get(key);}
function iR(){const m=D.meses; if(range==='all')return[0,m.length-1];
 const map={'2007':'2007-01','2015':'2015-01'};
 const t=range==='5y'?(m[m.length-1].slice(0,4)-5)+'-'+m[m.length-1].slice(5):map[range];
 let i=m.findIndex(x=>x>=t); return[i<0?0:i,m.length-1];}
function tk(lo,hi,log){
 if(log){const o=[];for(const b of[1,2,5])for(let e=0;e<4;e++){const v=b*Math.pow(10,e);if(v>=lo&&v<=hi)o.push(v);}return o.sort((a,b)=>a-b);}
 const st=Math.pow(10,Math.floor(Math.log10((hi-lo)/5)));
 const s=[1,2,2.5,5,10,20].map(x=>x*st).find(x=>(hi-lo)/x<=7)||st;
 const o=[];for(let v=Math.ceil(lo/s)*s;v<=hi;v+=s)o.push(+v.toFixed(6));
 if(!o.includes(0)&&lo<0&&hi>0)o.push(0); return o.sort((a,b)=>a-b);}
function frame(H,lo,hi,i0,i1,L,R,T,B,W,unit,log,axlab){
 const yf=log?(v=>Math.log10(v)):(v=>v);
 const Y=v=>T+(H-T-B)*(1-(yf(v)-yf(lo))/(yf(hi)-yf(lo)));
 const X=i=>L+(W-L-R)*((i-i0)/Math.max(1,i1-i0));
 let g='';
 for(const t of tk(lo,hi,log)){const y=Y(t).toFixed(1);
  g+=(t===0&&!log)?`<line class="zl" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/>`
    :`<line class="gl" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/>`;
  g+=`<text class="tick" x="${L-8}" y="${y}" text-anchor="end" dominant-baseline="middle">${(t>0&&unit==='%')?'+':''}${t}${unit}</text>`;}
 const yrs=[...new Set(D.meses.slice(i0,i1+1).map(m=>m.slice(0,4)))],st=Math.ceil(yrs.length/9);
 yrs.forEach((yr,k)=>{if(k%st)return;const i=D.meses.findIndex(m=>m.startsWith(yr));
  if(i<i0||i>i1)return; g+=`<text class="tick" x="${X(i).toFixed(1)}" y="${H-B+18}" text-anchor="middle">${yr}</text>`;});
 g+=`<line class="ax" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/>`;
 g+=`<text class="tick" x="${L-8}" y="${T-6}" text-anchor="end">${axlab}</text>`;
 return{g,X,Y};}
function lines(id,H,ss,unit,log,axlab,R){
 const svg=document.getElementById(id),W=900,L=54,T=18,B=30,[i0,i1]=iR();
 const on=ss.filter(s=>!off.has(s.name));
 let lo=Infinity,hi=-Infinity;
 for(const s of on)for(let i=i0;i<=i1;i++){const v=s.vals[i];if(v!=null){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
 if(!isFinite(lo)){svg.innerHTML='';return;}
 if(log){lo=Math.max(lo*.85,.5);hi*=1.15;}else{const p=(hi-lo)*.07;lo-=p;hi+=p;}
 const {g:base,X,Y}=frame(H,lo,hi,i0,i1,L,R,T,B,W,unit,log,axlab);
 let g=base;
 for(const s of on){let d='',o=false;
  for(let i=i0;i<=i1;i++){const v=s.vals[i];if(v==null){o=false;continue;}
   d+=(o?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)+' ';o=true;}
  g+=`<path d="${d}" fill="none" stroke="${s.col}" stroke-width="${s.lead?2.5:1.75}" stroke-linejoin="round" opacity="${s.lead?1:.9}"/>`;}
 const e=on.map(s=>{let li=null;for(let i=i1;i>=i0;i--)if(s.vals[i]!=null){li=i;break;}
  return li==null?null:{s,v:s.vals[li],y:Y(s.vals[li])};}).filter(Boolean).sort((a,b)=>a.y-b.y);
 for(let k=1;k<e.length;k++)if(e[k].y-e[k-1].y<13)e[k].y=e[k-1].y+13;
 for(const x of e)g+=`<text class="dl" x="${W-R+8}" y="${x.y.toFixed(1)}" fill="${x.s.col}" dominant-baseline="middle">${x.s.short} ${(unit==='%'&&x.v>0)?'+':''}${x.v.toFixed(0)}${unit}</text>`;
 const p=id.slice(1);
 g+=`<g id="cx${p}"></g><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" id="hit${p}"/>`;
 svg.innerHTML=g; svg.__g={i0,i1,X,Y,L,R,T,B,W,H,ss,unit};}
function area(id,H){
 const svg=document.getElementById(id),W=900,L=54,R=112,T=18,B=30,[i0,i1]=iR();
 let m=0; for(let i=i0;i<=i1;i++){const v=D.gap[i];if(v!=null)m=Math.max(m,Math.abs(v));}
 if(!m){svg.innerHTML='';return;} m*=1.08;
 const {g:base,X,Y}=frame(H,-m,m,i0,i1,L,R,T,B,W,'',false,'pp');
 let g=base,z=Y(0);
 for(const[sg,c]of[[1,'var(--pos)'],[-1,'var(--neg)']]){let d='';
  for(let i=i0;i<=i1;i++){const v=D.gap[i],cv=v==null?0:(sg>0?Math.max(v,0):Math.min(v,0));
   d+=(i===i0?'M':'L')+X(i).toFixed(1)+' '+Y(cv).toFixed(1)+' ';}
  d+=`L${X(i1).toFixed(1)} ${z.toFixed(1)} L${X(i0).toFixed(1)} ${z.toFixed(1)} Z`;
  g+=`<path d="${d}" fill="${c}" fill-opacity="0.55"/>`;}
 let d='',o=false;
 for(let i=i0;i<=i1;i++){const v=D.gap[i];if(v==null){o=false;continue;}d+=(o?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)+' ';o=true;}
 g+=`<path d="${d}" fill="none" stroke="var(--text-primary)" stroke-width="1" opacity=".42"/>`;
 g+=`<g id="cx4"></g><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent" id="hit4"/>`;
 svg.innerHTML=g; svg.__g={i0,i1,X,Y,L,R,T,B,W,H,ss:[{name:'Brecha',short:'Brecha',vals:D.gap,col:'var(--zero)'}],unit:' pp'};}
function hover(p){
 const svg=document.getElementById('c'+p),wrap=document.getElementById('w'+p),hit=document.getElementById('hit'+p);
 if(!hit)return; let tip=wrap.querySelector('.tip');
 if(!tip){tip=document.createElement('div');tip.className='tip';tip.style.display='none';wrap.appendChild(tip);}
 hit.addEventListener('mousemove',ev=>{const g=svg.__g;if(!g)return;
  const r=svg.getBoundingClientRect();
  const f=((ev.clientX-r.left)/r.width*g.W-g.L)/(g.W-g.L-g.R);
  let i=Math.round(g.i0+f*(g.i1-g.i0));i=Math.max(g.i0,Math.min(g.i1,i));
  const x=g.X(i),rows=g.ss.filter(s=>!off.has(s.name));
  document.getElementById('cx'+p).innerHTML=`<line class="cross" x1="${x}" x2="${x}" y1="${g.T}" y2="${g.H-g.B}"/>`+
   rows.filter(s=>s.vals[i]!=null).map(s=>{const c=p==='4'?(D.gap[i]>=0?'var(--pos)':'var(--neg)'):s.col;
    return `<circle cx="${x}" cy="${g.Y(s.vals[i])}" r="4" fill="${c}" stroke="var(--surface-1)" stroke-width="2"/>`;}).join('');
  tip.innerHTML=`<b>${D.meses[i]}</b>`+rows.map(s=>{const v=s.vals[i];
   const c=p==='4'?(v>=0?'var(--pos)':'var(--neg)'):s.col;
   return `<div><span><i style="background:${c}"></i>${s.name}</span><span class="v">${v==null?'—':((g.unit==='%'&&v>0)?'+':'')+v.toFixed(1)+g.unit}</span></div>`;}).join('');
  tip.style.display='block';
  const lf=x/g.W*r.width;
  tip.style.left=Math.min(r.width-tip.offsetWidth-4,Math.max(0,lf+14))+'px';
  tip.style.top=Math.max(0,(ev.clientY-r.top)-tip.offsetHeight/2)+'px';});
 hit.addEventListener('mouseleave',()=>{tip.style.display='none';document.getElementById('cx'+p).innerHTML='';});}
function lg(id,ss){const el=document.getElementById(id);
 el.innerHTML=ss.map(s=>`<span data-name="${s.name}" data-off="${off.has(s.name)?1:0}"><i style="background:${s.col}"></i>${s.name}</span>`).join('');
 el.querySelectorAll('span').forEach(sp=>sp.onclick=()=>{const n=sp.dataset.name;
  off.has(n)?off.delete(n):off.add(n); if(off.size>=ss.length)off.delete(n); render();});}
function S1(){return D.series.map((s,k)=>({...s,vals:V('v'+k,s.vals),col:col(k)}));}
function S2(){return[{name:'Mayorista SNIIM',short:'Mayorista',vals:V('a',D.series[0].vals),col:col(0),lead:true},
 {name:'Menudeo INPC',short:'Menudeo',vals:V('r',D.retail),col:RET_COL(),lead:true}];}
function S3(){return[{name:'Mayorista, var. anual',short:'Mayorista',vals:D.wy,col:col(0),lead:true},
 {name:'Menudeo INPC, var. anual',short:'Menudeo',vals:D.ry,col:RET_COL(),lead:true}];}
function tiles(){const[i0,i1]=iR();
 const gp=D.gap.slice(i0,i1+1).filter(v=>v!=null);
 const f=v=>v==null?'—':(v>0?'+':'')+v.toFixed(1)+'%';
 document.getElementById('tiles').innerHTML=[
  ['Último dato',D.meses[i1],''],
  ['Mayorista, var. anual',f(D.wy[i1]),''],
  ['Menudeo INPC, var. anual',f(D.ry[i1]),''],
  ['Amplitud relativa',D.stats.amp.toFixed(2)+'×','desv. est. de cambios log, may./men.'],
  ['Mayorista por delante',Math.round(100*gp.filter(v=>v>0).length/gp.length)+'%','de los meses'],
  ['Correlación mensual',D.stats.corr_mom.toFixed(3),'cambios log, todo el periodo'],
 ].map(([k,v,u])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>${u?`<div class="u">${u}</div>`:''}</div>`).join('');}
function table(){const[i0,i1]=iR();
 const cols=[...D.series.map(s=>s.name),'Menudeo INPC','Mayorista var%','Menudeo var%','Brecha pp'];
 let h='<thead><tr><th>Mes</th>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
 for(let i=i1;i>=i0;i--){h+=`<tr><td>${D.meses[i]}</td>`+
  D.series.map(s=>`<td>${s.vals[i]==null?'—':s.vals[i].toFixed(1)}</td>`).join('')+
  [D.retail[i],D.wy[i],D.ry[i],D.gap[i]].map(v=>`<td>${v==null?'—':v.toFixed(1)}</td>`).join('')+'</tr>';}
 document.getElementById('tbl').innerHTML=h+'</tbody>';}
function render(){
 const nm=D.generico.replace(/^\d+\s/,'');
 document.getElementById('h1').textContent=nm+': precio mayorista y INPC';
 document.getElementById('sub').innerHTML=
  `Índice de precio mayorista (SNIIM, pesos por kilogramo) e INPC genérico <strong>${D.generico}</strong> (INEGI). `+
  `Base: promedio geométrico de ${D.base_anio} = 100. ${D.stats.n_meses} meses; ${D.stats.n_yoy} con variación anual. `+
  `El mes en curso se excluye porque está incompleto.`;
 document.getElementById('n1').innerHTML='El agregado no es el promedio de las líneas: es un índice de eslabonamiento '+
  'que sólo compara celdas variedad&nbsp;×&nbsp;mercado presentes en dos meses consecutivos.'+
  (D.hidden.length?` No se grafican ${D.hidden.length} variedad(es) por cobertura baja: `+
   D.hidden.map(h=>`${h[0]} (${h[1]}%)`).join(', ')+'.':'')+
  (D.sin_base.length?` Tampoco ${D.sin_base.map(h=>`${h[0]} (${h[1]} mes(es) en ${D.base_anio})`).join(', ')}: `+
   `sin suficientes meses del año base, su nivel no sería comparable.`:'')+
  ((D.hidden.length||D.sin_base.length)?' Todas siguen contando dentro del agregado.':'');
 document.getElementById('n2').textContent='Misma base y un solo eje. La correlación de los cambios logarítmicos mensuales es '
  +D.stats.corr_mom.toFixed(3)+'.';
 document.getElementById('n3').textContent='Cambio porcentual respecto al mismo mes del año anterior, como lo reporta INEGI. '
  +'La amplitud del mayorista es '+D.stats.amp.toFixed(2)+'× la del menudeo.';
 document.getElementById('n4').textContent='Puntos porcentuales. Azul = el mayorista sube más rápido que el menudeo; rojo = lo contrario. '
  +'El mayorista va por delante el '+D.stats.lead.toFixed(0)+'% de los meses.';
 document.getElementById('charts').classList.toggle('hide',view==='table');
 document.getElementById('tableview').classList.toggle('hide',view!=='table');
 tiles();
 if(view==='table'){table();}
 else{const s1=S1(),s2=S2(),s3=S3();
  lg('lg1',s1);lg('lg2',s2);lg('lg3',s3);
  lines('c1',370,s1,'',scale==='log','índice',150);
  lines('c2',290,s2,'',scale==='log','índice',132);
  lines('c3',320,s3,'%',false,'var. anual',118);
  area('c4',230);
  ['1','2','3','4'].forEach(hover);}
 const[i0,i1]=iR();
 document.getElementById('foot').innerHTML=
  `Mostrando ${D.meses[i0]} a ${D.meses[i1]}. Fuente: SNIIM (Secretaría de Economía) e INEGI. `+
  (D.stats.base_ok?'':'<strong>Nota:</strong> la categoría no tiene dato en enero 2018, así que el índice se basa en su primer mes disponible. ')+
  `<strong>Advertencia:</strong> las ventanas anuales se traslapan 11 de 12 meses, así que las series de `+
  `variación anual están fuertemente autocorrelacionadas; la cifra utilizable para medir la relación es la `+
  `correlación de los cambios mensuales (${D.stats.corr_mom.toFixed(3)}), no la de las series anuales.`+
  (smooth?' Media geométrica móvil de 12 meses en los índices.':'');}
document.querySelectorAll('[data-range]').forEach(b=>b.onclick=()=>{range=b.dataset.range;
 document.querySelectorAll('[data-range]').forEach(x=>x.setAttribute('aria-pressed',x===b));render();});
document.querySelectorAll('[data-scale]').forEach(b=>b.onclick=()=>{scale=b.dataset.scale;
 document.querySelectorAll('[data-scale]').forEach(x=>x.setAttribute('aria-pressed',x===b));render();});
document.querySelectorAll('[data-smooth]').forEach(b=>b.onclick=()=>{smooth=b.dataset.smooth==='1';
 document.querySelectorAll('[data-smooth]').forEach(x=>x.setAttribute('aria-pressed',x===b));render();});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;
 document.querySelectorAll('[data-view]').forEach(x=>x.setAttribute('aria-pressed',x===b));render();});
document.querySelectorAll('[data-theme-btn]').forEach(b=>b.onclick=()=>{
 document.documentElement.setAttribute('data-theme',b.dataset.themeBtn);
 document.querySelectorAll('[data-theme-btn]').forEach(x=>x.setAttribute('aria-pressed',x===b));render();});
addEventListener('resize',()=>{if(view==='chart')render();});
render();
</script></body></html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slug", nargs="?", help="category slug, e.g. aguacate")
    p.add_argument("--min-cobertura", type=float, default=40.0,
                   help="minimum %% of months a variety must cover to be plotted "
                        "(it still counts inside the aggregate). Default 40.")
    p.add_argument("--list", action="store_true", help="list available categories")
    a = p.parse_args()

    if a.list or not a.slug:
        idx = pd.read_parquet(ROOT / "data" / "curated" / "cat_index_monthly.parquet")
        n = idx.groupby("categoria").size().sort_values(ascending=False)
        print(f"{len(n)} categories with an index built:")
        for k, v in n.items():
            print(f"  {k:<30} {v:>4} months")
        return 0

    out = build(a.slug, a.min_cobertura)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
