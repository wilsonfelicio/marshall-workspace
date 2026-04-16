import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json, os

# ── CONFIG ──────────────────────────────────────────────────────────────
WDIR = r"C:\Users\wfelicio\OneDrive - Brasil Warrant Administração de Bens e Empresas\Claude\COPX PCA"
FILE = os.path.join(WDIR, "Book2.xlsx")
OUT  = os.path.join(WDIR, "COPX_Dashboard.html")

COLORS = {
    "COPX": "#e74c3c", "Copper": "#e67e22", "EEM": "#2ecc71",
    "TLT": "#3498db", "USD": "#9b59b6", "SPY": "#1abc9c"
}
FACTOR_NAMES = ["Copper", "EEM", "TLT", "USD", "SPY"]

# ── LOAD & PREP ─────────────────────────────────────────────────────────
raw = pd.read_excel(FILE)
raw.columns = ["Date", "COPX", "Copper", "EEM", "TLT", "USD", "SPY"]
raw = raw.sort_values("Date").reset_index(drop=True)
raw["Date"] = pd.to_datetime(raw["Date"])

# Weekly log returns (%)
rets = raw.copy()
for c in ["COPX"] + FACTOR_NAMES:
    rets[c] = np.log(raw[c] / raw[c].shift(1)) * 100
rets = rets.dropna().reset_index(drop=True)

y = rets["COPX"].values
X = rets[FACTOR_NAMES].values
dates = rets["Date"].values

# ── 1. SUMMARY STATISTICS ──────────────────────────────────────────────
ann_factor = np.sqrt(52)
stats_df = pd.DataFrame(index=["COPX"] + FACTOR_NAMES)
for c in stats_df.index:
    s = rets[c]
    stats_df.loc[c, "Ann. Return (%)"]   = s.mean() * 52
    stats_df.loc[c, "Ann. Vol (%)"]       = s.std() * ann_factor
    stats_df.loc[c, "Sharpe"]             = (s.mean() / s.std()) * ann_factor
    stats_df.loc[c, "Skewness"]           = s.skew()
    stats_df.loc[c, "Kurtosis"]           = s.kurtosis()
    stats_df.loc[c, "Max Drawdown (%)"]   = ((raw[c] / raw[c].cummax()) - 1).min() * 100
    stats_df.loc[c, "Best Week (%)"]      = s.max()
    stats_df.loc[c, "Worst Week (%)"]     = s.min()
stats_df = stats_df.round(2)

# ── 2. CORRELATION MATRIX ──────────────────────────────────────────────
corr = rets[["COPX"] + FACTOR_NAMES].corr().round(3)

# ── 3. OLS REGRESSION ──────────────────────────────────────────────────
X_ols = sm.add_constant(X)
ols = sm.OLS(y, X_ols).fit()
ols_fitted = ols.fittedvalues
ols_resid  = ols.resid

ols_table = pd.DataFrame({
    "Coefficient": ols.params,
    "Std Error":   ols.bse,
    "t-stat":      ols.tvalues,
    "p-value":     ols.pvalues
}, index=["Intercept"] + FACTOR_NAMES).round(4)

# ── 4. LASSO REGRESSION ────────────────────────────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

alphas_grid = np.logspace(-4, 1, 200)
lasso_cv = LassoCV(alphas=alphas_grid, cv=10, max_iter=50000, random_state=42)
lasso_cv.fit(X_sc, y)

lasso_coefs_std = pd.Series(lasso_cv.coef_, index=FACTOR_NAMES)
lasso_coefs_raw = lasso_cv.coef_ / scaler.scale_
lasso_intercept = lasso_cv.intercept_ - (lasso_coefs_raw * scaler.mean_).sum()  # not needed for path
lasso_fitted = lasso_cv.predict(X_sc)
lasso_r2 = lasso_cv.score(X_sc, y)

# Lasso path
coef_path = []
for a in alphas_grid:
    m = Lasso(alpha=a, max_iter=50000)
    m.fit(X_sc, y)
    coef_path.append(m.coef_.copy())
coef_path = np.array(coef_path)

# ── 5. PCA ──────────────────────────────────────────────────────────────
X_all = rets[["COPX"] + FACTOR_NAMES].values
X_all_sc = StandardScaler().fit_transform(X_all)

pca = PCA()
pca_scores = pca.fit_transform(X_all_sc)
loadings = pd.DataFrame(pca.components_.T,
                        index=["COPX"] + FACTOR_NAMES,
                        columns=[f"PC{i+1}" for i in range(6)])
var_explained = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(var_explained)

# ── 6. ROLLING REGRESSION (52-week) ────────────────────────────────────
ROLL_WIN = 52
n = len(y)
roll_betas  = np.full((n, len(FACTOR_NAMES)), np.nan)
roll_r2     = np.full(n, np.nan)
roll_resid  = np.full(n, np.nan)

for i in range(ROLL_WIN, n):
    y_w = y[i-ROLL_WIN:i]
    X_w = sm.add_constant(X[i-ROLL_WIN:i])
    try:
        m = sm.OLS(y_w, X_w).fit()
        roll_betas[i]  = m.params[1:]
        roll_r2[i]     = m.rsquared
        roll_resid[i]  = y[i] - m.predict(sm.add_constant(X[i:i+1]))[0]
    except:
        pass

# ── 7. RICH / CHEAP (cumulative residual + z-score) ────────────────────
# Use full-sample OLS residuals
cum_resid = np.nancumsum(ols_resid)
# Rolling z-score of residuals (52-wk)
resid_series = pd.Series(ols_resid)
resid_mean = resid_series.rolling(ROLL_WIN).mean()
resid_std  = resid_series.rolling(ROLL_WIN).std()
resid_z    = ((resid_series - resid_mean) / resid_std).values

# Also: cumulative residual z-score over longer horizon
cum_resid_series = pd.Series(cum_resid)
cum_z_mean = cum_resid_series.rolling(ROLL_WIN*2).mean()
cum_z_std  = cum_resid_series.rolling(ROLL_WIN*2).std()
cum_resid_z = ((cum_resid_series - cum_z_mean) / cum_z_std).values

# ── 8. ROLLING CORRELATION (26-week) ───────────────────────────────────
CORR_WIN = 26
roll_corr = {}
for f in FACTOR_NAMES:
    roll_corr[f] = rets["COPX"].rolling(CORR_WIN).corr(rets[f]).values


# ══════════════════════════════════════════════════════════════════════════
#  BUILD HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
date_str = [str(d)[:10] for d in dates]
last_date = date_str[-1]

def make_table_html(df, title, id_suffix=""):
    html = f'<h3>{title}</h3><table class="data-table" id="tbl-{id_suffix}"><thead><tr><th></th>'
    for c in df.columns:
        html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"
    for idx, row in df.iterrows():
        html += f"<tr><td class='row-label'>{idx}</td>"
        for v in row.values:
            val = f"{v}" if isinstance(v, str) else f"{v:.4f}" if abs(v) < 0.01 else f"{v:.2f}"
            css = ""
            try:
                if float(v) > 0: css = "color:#2ecc71"
                elif float(v) < 0: css = "color:#e74c3c"
            except: pass
            html += f"<td style='{css}'>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# ── Plotly figures ──────────────────────────────────────────────────────
figs = {}

# 1. Rebased total return
fig = go.Figure()
for c in ["COPX"] + FACTOR_NAMES:
    rebased = raw[c] / raw[c].iloc[0] * 100
    fig.add_trace(go.Scatter(x=raw["Date"], y=rebased, name=c,
                             line=dict(color=COLORS[c], width=2.5 if c=="COPX" else 1.5)))
fig.update_layout(title="Total Return Indexes (Rebased to 100)",
                  yaxis_title="Index", template="plotly_dark", height=450,
                  legend=dict(orientation="h", y=-0.15), yaxis_type="log")
figs["rebased"] = fig

# 2. Correlation heatmap
fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                           colorscale="RdBu_r", zmin=-1, zmax=1,
                           text=corr.values.round(2), texttemplate="%{text}",
                           textfont=dict(size=13)))
fig.update_layout(title="Correlation Matrix (Weekly Log Returns)", template="plotly_dark", height=420)
figs["corr"] = fig

# 3. OLS: Actual vs Fitted
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=y, name="Actual COPX", line=dict(color=COLORS["COPX"], width=1)))
fig.add_trace(go.Scatter(x=date_str, y=ols_fitted, name="OLS Fitted",
                         line=dict(color="#f1c40f", width=1.5, dash="dot")))
fig.update_layout(title=f"OLS: Actual vs Fitted (R² = {ols.rsquared:.3f}, Adj-R² = {ols.rsquared_adj:.3f})",
                  yaxis_title="Weekly Return (%)", template="plotly_dark", height=400,
                  legend=dict(orientation="h", y=-0.15))
figs["ols_fit"] = fig

# 4. OLS Coefficient bar chart
fig = go.Figure()
colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in ols.params[1:]]
fig.add_trace(go.Bar(x=FACTOR_NAMES, y=ols.params[1:], marker_color=colors_bar,
                     text=[f"{v:.3f}" for v in ols.params[1:]], textposition="outside"))
fig.update_layout(title="OLS Coefficients (Betas)", yaxis_title="Beta", template="plotly_dark", height=350)
figs["ols_coef"] = fig

# 5. LASSO path
fig = go.Figure()
for i, f in enumerate(FACTOR_NAMES):
    fig.add_trace(go.Scatter(x=np.log10(alphas_grid), y=coef_path[:, i],
                             name=f, line=dict(color=COLORS[f], width=2)))
fig.add_vline(x=np.log10(lasso_cv.alpha_), line_dash="dash", line_color="white",
              annotation_text=f"CV α={lasso_cv.alpha_:.4f}")
fig.update_layout(title=f"LASSO Coefficient Path (CV R² = {lasso_r2:.3f})",
                  xaxis_title="log₁₀(α)", yaxis_title="Standardized Coefficient",
                  template="plotly_dark", height=400, legend=dict(orientation="h", y=-0.15))
figs["lasso_path"] = fig

# 6. LASSO coefficients
fig = go.Figure()
colors_l = ["#2ecc71" if v > 0 else "#e74c3c" for v in lasso_coefs_std.values]
fig.add_trace(go.Bar(x=FACTOR_NAMES, y=lasso_coefs_std.values, marker_color=colors_l,
                     text=[f"{v:.3f}" for v in lasso_coefs_std.values], textposition="outside"))
fig.update_layout(title="LASSO Selected Coefficients (Standardized)", yaxis_title="Coefficient",
                  template="plotly_dark", height=350)
figs["lasso_coef"] = fig

# 7. PCA variance explained
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(6)], y=var_explained,
                     name="Variance %", marker_color="#3498db"), secondary_y=False)
fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(6)], y=cum_var,
                         name="Cumulative %", line=dict(color="#e74c3c", width=2),
                         mode="lines+markers"), secondary_y=True)
fig.update_layout(title="PCA: Variance Explained", template="plotly_dark", height=380)
fig.update_yaxes(title_text="Individual %", secondary_y=False)
fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
figs["pca_var"] = fig

# 8. PCA loadings heatmap
fig = go.Figure(go.Heatmap(z=loadings.values, x=loadings.columns, y=loadings.index,
                           colorscale="RdBu_r", zmin=-0.7, zmax=0.7,
                           text=loadings.values.round(3), texttemplate="%{text}",
                           textfont=dict(size=12)))
fig.update_layout(title="PCA Factor Loadings", template="plotly_dark", height=380)
figs["pca_load"] = fig

# 9. Rolling Betas
fig = go.Figure()
for i, f in enumerate(FACTOR_NAMES):
    fig.add_trace(go.Scatter(x=date_str, y=roll_betas[:, i], name=f,
                             line=dict(color=COLORS[f], width=1.5)))
fig.update_layout(title="Rolling 52-Week OLS Betas",
                  yaxis_title="Beta", template="plotly_dark", height=420,
                  legend=dict(orientation="h", y=-0.15))
figs["roll_beta"] = fig

# 10. Rolling R²
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=roll_r2, name="R²",
                         line=dict(color="#f1c40f", width=2), fill="tozeroy",
                         fillcolor="rgba(241,196,15,0.15)"))
fig.update_layout(title="Rolling 52-Week R²", yaxis_title="R²", template="plotly_dark",
                  height=350, yaxis_range=[0, 1])
figs["roll_r2"] = fig

# 11. Rich/Cheap: Cumulative residual
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=cum_resid, name="Cumulative Residual",
                         line=dict(color="#e74c3c", width=2), fill="tozeroy",
                         fillcolor="rgba(231,76,60,0.12)"))
fig.add_hline(y=0, line_dash="dash", line_color="grey")
fig.update_layout(title="Cumulative OLS Residual (Rich/Cheap Signal)",
                  yaxis_title="Cum. Residual (%)", template="plotly_dark", height=400)
figs["cum_resid"] = fig

# 12. Rich/Cheap: Z-score
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=cum_resid_z, name="Z-Score",
                         line=dict(color="#9b59b6", width=2)))
fig.add_hline(y=2, line_dash="dash", line_color="#2ecc71", annotation_text="Expensive (+2σ)")
fig.add_hline(y=-2, line_dash="dash", line_color="#e74c3c", annotation_text="Cheap (-2σ)")
fig.add_hline(y=0, line_dash="dot", line_color="grey")
fig.add_hrect(y0=-2, y1=2, fillcolor="rgba(255,255,255,0.03)", line_width=0)
fig.update_layout(title="Cumulative Residual Z-Score (2-Year Rolling)",
                  yaxis_title="Z-Score", template="plotly_dark", height=400)
figs["zscore"] = fig

# 13. Rolling correlations
fig = go.Figure()
for f in FACTOR_NAMES:
    fig.add_trace(go.Scatter(x=date_str, y=roll_corr[f], name=f,
                             line=dict(color=COLORS[f], width=1.5)))
fig.update_layout(title="Rolling 26-Week Correlation with COPX",
                  yaxis_title="Correlation", template="plotly_dark", height=400,
                  legend=dict(orientation="h", y=-0.15))
figs["roll_corr"] = fig

# 14. Scatter matrix: COPX vs each factor
scatter_fig = make_subplots(rows=1, cols=5, subplot_titles=FACTOR_NAMES,
                            horizontal_spacing=0.04)
for i, f in enumerate(FACTOR_NAMES):
    scatter_fig.add_trace(go.Scatter(
        x=rets[f], y=rets["COPX"], mode="markers",
        marker=dict(color=COLORS[f], size=3, opacity=0.5), name=f,
        showlegend=False
    ), row=1, col=i+1)
    # Add regression line
    slope, intercept = np.polyfit(rets[f], rets["COPX"], 1)
    xr = np.array([rets[f].min(), rets[f].max()])
    scatter_fig.add_trace(go.Scatter(
        x=xr, y=slope*xr + intercept, mode="lines",
        line=dict(color="white", width=2, dash="dash"), showlegend=False
    ), row=1, col=i+1)
    scatter_fig.update_xaxes(title_text=f"{f} (%)", row=1, col=i+1)
scatter_fig.update_yaxes(title_text="COPX (%)", row=1, col=1)
scatter_fig.update_layout(title="COPX vs Factors (Weekly Returns)", template="plotly_dark", height=350)
figs["scatter"] = scatter_fig

# ── Current signal summary ──────────────────────────────────────────────
last_cum_resid = cum_resid[-1]
last_z = cum_resid_z[-1] if not np.isnan(cum_resid_z[-1]) else 0
last_resid = ols_resid[-1]

if last_z > 2: signal = "EXPENSIVE"
elif last_z > 1: signal = "SLIGHTLY EXPENSIVE"
elif last_z < -2: signal = "CHEAP"
elif last_z < -1: signal = "SLIGHTLY CHEAP"
else: signal = "FAIR VALUE"

signal_color = "#e74c3c" if "EXPENSIVE" in signal else "#2ecc71" if "CHEAP" in signal else "#f1c40f"

# Last 4 weeks residuals
last4_resid = ols_resid[-4:]
recent_trend = "Richening" if np.mean(last4_resid) > 0 else "Cheapening"

# ── HTML Assembly ───────────────────────────────────────────────────────
plotly_js = "https://cdn.plot.ly/plotly-2.35.2.min.js"

fig_htmls = {}
for k, f in figs.items():
    f.update_layout(margin=dict(l=50, r=30, t=50, b=40))
    fig_htmls[k] = f.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart-{k}")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>COPX Factor Analysis Dashboard</title>
<script src="{plotly_js}"></script>
<style>
  :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9;
           --accent: #e74c3c; --green: #2ecc71; --yellow: #f1c40f; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
         line-height: 1.5; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             padding: 30px 40px; border-bottom: 2px solid var(--accent); }}
  .header h1 {{ font-size: 28px; color: #fff; margin-bottom: 5px; }}
  .header p {{ color: #8b949e; font-size: 14px; }}
  .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
  .grid {{ display: grid; gap: 20px; }}
  .grid-2 {{ grid-template-columns: 1fr 1fr; }}
  .grid-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
           padding: 20px; overflow: hidden; }}
  .card-full {{ grid-column: 1 / -1; }}
  .signal-box {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .signal-card {{ flex: 1; min-width: 200px; background: var(--card); border: 1px solid var(--border);
                  border-radius: 10px; padding: 20px; text-align: center; }}
  .signal-card .label {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
  .signal-card .value {{ font-size: 28px; font-weight: 700; margin: 8px 0; }}
  .signal-card .sub {{ font-size: 12px; color: #8b949e; }}
  .section-title {{ font-size: 20px; font-weight: 600; color: #fff; margin: 30px 0 15px;
                    padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  h3 {{ color: #fff; font-size: 15px; margin-bottom: 10px; }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th {{ background: #21262d; padding: 8px 12px; text-align: right; color: #8b949e;
                    font-weight: 500; border-bottom: 1px solid var(--border); }}
  .data-table td {{ padding: 7px 12px; text-align: right; border-bottom: 1px solid #1c2028; }}
  .data-table .row-label {{ text-align: left; font-weight: 600; color: #fff; }}
  .data-table tr:hover {{ background: #1c2028; }}
  .nav {{ position: sticky; top: 0; z-index: 100; background: rgba(13,17,23,0.95);
          backdrop-filter: blur(10px); padding: 10px 40px; border-bottom: 1px solid var(--border);
          display: flex; gap: 8px; flex-wrap: wrap; }}
  .nav a {{ color: #8b949e; text-decoration: none; font-size: 13px; padding: 4px 12px;
            border-radius: 6px; transition: all 0.2s; }}
  .nav a:hover {{ background: #21262d; color: #fff; }}
  .insight {{ background: #1c2333; border-left: 3px solid var(--accent); padding: 12px 16px;
              margin: 10px 0; border-radius: 0 8px 8px 0; font-size: 13px; }}
  .insight strong {{ color: #fff; }}
  @media (max-width: 1000px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<div class="header">
  <h1>COPX Factor Analysis Dashboard</h1>
  <p>Global Copper Miners ETF &mdash; Multi-Factor Decomposition &amp; Valuation Signal &mdash; Data: {raw['Date'].iloc[0].strftime('%Y-%m-%d')} to {last_date}</p>
</div>

<nav class="nav">
  <a href="#signal">Signal</a>
  <a href="#overview">Overview</a>
  <a href="#stats">Statistics</a>
  <a href="#ols">OLS Regression</a>
  <a href="#lasso">LASSO</a>
  <a href="#pca">PCA</a>
  <a href="#rolling">Rolling Analysis</a>
  <a href="#richcheap">Rich/Cheap</a>
</nav>

<div class="container">

<!-- ═══ SIGNAL ═══ -->
<div id="signal" class="section-title">Current Signal ({last_date})</div>
<div class="signal-box">
  <div class="signal-card">
    <div class="label">Overall Verdict</div>
    <div class="value" style="color:{signal_color}">{signal}</div>
    <div class="sub">vs. model fair value</div>
  </div>
  <div class="signal-card">
    <div class="label">Cum. Residual Z-Score</div>
    <div class="value" style="color:{'#2ecc71' if last_z < 0 else '#e74c3c'}">{last_z:+.2f}σ</div>
    <div class="sub">2-year rolling</div>
  </div>
  <div class="signal-card">
    <div class="label">Last Week Residual</div>
    <div class="value" style="color:{'#2ecc71' if last_resid < 0 else '#e74c3c'}">{last_resid:+.2f}%</div>
    <div class="sub">OLS model miss</div>
  </div>
  <div class="signal-card">
    <div class="label">Cum. Residual</div>
    <div class="value" style="color:{'#2ecc71' if last_cum_resid < 0 else '#e74c3c'}">{last_cum_resid:+.1f}%</div>
    <div class="sub">total outperformance vs model</div>
  </div>
  <div class="signal-card">
    <div class="label">4-Week Trend</div>
    <div class="value" style="color:{'#e74c3c' if recent_trend=='Richening' else '#2ecc71'}">{recent_trend}</div>
    <div class="sub">avg residual: {np.mean(last4_resid):+.2f}%</div>
  </div>
</div>

<!-- ═══ OVERVIEW ═══ -->
<div id="overview" class="section-title">Performance Overview</div>
<div class="card card-full">{fig_htmls['rebased']}</div>

<!-- ═══ STATISTICS ═══ -->
<div id="stats" class="section-title">Return Statistics &amp; Correlations</div>
<div class="grid grid-2">
  <div class="card">{make_table_html(stats_df, 'Summary Statistics (Weekly Log Returns, Annualized)', 'stats')}</div>
  <div class="card">{fig_htmls['corr']}</div>
</div>
<div class="card card-full">{fig_htmls['scatter']}</div>

<!-- ═══ OLS ═══ -->
<div id="ols" class="section-title">OLS Multiple Regression</div>
<div class="insight">
  <strong>Model:</strong> COPX_return = α + β₁·Copper + β₂·EEM + β₃·TLT + β₄·USD + β₅·SPY<br>
  <strong>R² = {ols.rsquared:.4f}</strong> | Adj-R² = {ols.rsquared_adj:.4f} | F-stat = {ols.fvalue:.1f} (p={ols.f_pvalue:.2e}) | DW = {sm.stats.stattools.durbin_watson(ols.resid):.3f} | N = {len(y)}
</div>
<div class="grid grid-2">
  <div class="card">{make_table_html(ols_table, 'OLS Coefficients', 'ols')}</div>
  <div class="card">{fig_htmls['ols_coef']}</div>
</div>
<div class="card card-full">{fig_htmls['ols_fit']}</div>

<!-- ═══ LASSO ═══ -->
<div id="lasso" class="section-title">LASSO Regression (Variable Selection)</div>
<div class="insight">
  <strong>10-Fold CV LASSO</strong> &mdash; Optimal α = {lasso_cv.alpha_:.5f} | R² = {lasso_r2:.4f}<br>
  Selected factors: {', '.join([f for f, c in zip(FACTOR_NAMES, lasso_cv.coef_) if abs(c) > 1e-6])} |
  Dropped: {', '.join([f for f, c in zip(FACTOR_NAMES, lasso_cv.coef_) if abs(c) <= 1e-6]) or 'None'}
</div>
<div class="grid grid-2">
  <div class="card">{fig_htmls['lasso_path']}</div>
  <div class="card">{fig_htmls['lasso_coef']}</div>
</div>

<!-- ═══ PCA ═══ -->
<div id="pca" class="section-title">Principal Component Analysis</div>
<div class="insight">
  <strong>PCA on all 6 assets (standardized weekly returns)</strong><br>
  PC1 explains {var_explained[0]:.1f}% of variance (cumulative first 3: {cum_var[2]:.1f}%)
</div>
<div class="grid grid-2">
  <div class="card">{fig_htmls['pca_var']}</div>
  <div class="card">{fig_htmls['pca_load']}</div>
</div>

<!-- ═══ ROLLING ═══ -->
<div id="rolling" class="section-title">Rolling Analysis (52-Week Window)</div>
<div class="grid grid-2">
  <div class="card card-full">{fig_htmls['roll_beta']}</div>
</div>
<div class="grid grid-2">
  <div class="card">{fig_htmls['roll_r2']}</div>
  <div class="card">{fig_htmls['roll_corr']}</div>
</div>

<!-- ═══ RICH/CHEAP ═══ -->
<div id="richcheap" class="section-title">Rich / Cheap Analysis</div>
<div class="insight">
  <strong>Methodology:</strong> Cumulative OLS residuals measure COPX's total return vs. what the factor model predicts.
  Positive = COPX has outperformed model (expensive). Negative = underperformed (cheap).
  Z-score normalizes over a 2-year rolling window for regime-adaptive signals.
</div>
<div class="grid grid-2">
  <div class="card">{fig_htmls['cum_resid']}</div>
  <div class="card">{fig_htmls['zscore']}</div>
</div>

</div><!-- container -->

<div style="text-align:center; padding:30px; color:#484f58; font-size:12px;">
  COPX Factor Analysis Dashboard &mdash; Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</div>

</body>
</html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved to: {OUT}")
print(f"\nSignal: {signal} (Z={last_z:+.2f})")
print(f"OLS R²: {ols.rsquared:.4f}")
print(f"LASSO R²: {lasso_r2:.4f}")
print(f"OLS Summary:\n{ols.summary().tables[1]}")
