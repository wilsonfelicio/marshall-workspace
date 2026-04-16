"""
Kalman Filter analysis for COPX factor model
Time-Varying Parameter (TVP) regression: betas follow a random walk
Hyperparameters (state noise, obs noise) estimated by MLE via statsmodels.
Appends a new Kalman section to the existing dashboard.
"""
import pandas as pd
import numpy as np
import os, re
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ───────────────────────────────────────────────────────────────
WDIR = r"C:\Users\wfelicio\OneDrive - Brasil Warrant Administração de Bens e Empresas\Claude\COPX PCA"
FILE = os.path.join(WDIR, "Book2.xlsx")
DASH = os.path.join(WDIR, "COPX_Dashboard.html")

FACTOR_NAMES = ["Copper", "EEM", "TLT", "USD", "SPY"]
COLORS = {"COPX": "#e74c3c", "Copper": "#e67e22", "EEM": "#2ecc71",
          "TLT": "#3498db", "USD": "#9b59b6", "SPY": "#1abc9c"}

# ── Load data ────────────────────────────────────────────────────────────
raw = pd.read_excel(FILE)
raw.columns = ["Date", "COPX", "Copper", "EEM", "TLT", "USD", "SPY"]
raw = raw.sort_values("Date").reset_index(drop=True)
raw["Date"] = pd.to_datetime(raw["Date"])

rets = raw.copy()
for c in ["COPX"] + FACTOR_NAMES:
    rets[c] = np.log(raw[c] / raw[c].shift(1)) * 100
rets = rets.dropna().reset_index(drop=True)

y = rets["COPX"].values
X = rets[FACTOR_NAMES].values  # shape (T, 5)
T, K = X.shape
dates = rets["Date"].values
date_str = [str(d)[:10] for d in dates]


# ══════════════════════════════════════════════════════════════════════════
#  TVP REGRESSION AS STATE-SPACE MODEL
#  State: β_t = β_{t-1} + η_t,   η_t ~ N(0, Q)   [random walk betas]
#  Obs:   y_t = x_t' β_t + ε_t,  ε_t ~ N(0, H)
#
#  Hyperparameters: Q = diag(σ²_η) per factor, H = σ²_ε  (K+1 params)
#  Simplification: use Q = q·I  (1 param) + H  (2 params total)
# ══════════════════════════════════════════════════════════════════════════

class TVPRegression(MLEModel):
    """Time-varying parameter regression with random-walk coefficients."""
    param_names = ["sigma2_obs", "sigma2_state"]

    def __init__(self, endog, exog):
        k_states = exog.shape[1]
        super().__init__(endog, k_states=k_states, initialization="approximate_diffuse")
        # Observation: y_t = Z_t * α_t  where Z_t = x_t'
        self["design"] = exog[np.newaxis, :, :].transpose(0, 2, 1)  # (1, k_states, T) — time-varying
        # Wait: correct shape is (k_endog=1, k_states=K, nobs=T)
        self["design"] = np.zeros((1, k_states, len(endog)))
        for t in range(len(endog)):
            self["design"][0, :, t] = exog[t]
        # State transition: α_t = α_{t-1} + η_t  → T = I
        self["transition"] = np.eye(k_states)
        # Selection matrix R = I (all states have noise)
        self["selection"] = np.eye(k_states)

    @property
    def start_params(self):
        return [1.0, 0.01]

    @property
    def param_names_(self):
        return ["sigma2_obs", "sigma2_state"]

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return np.sqrt(np.abs(constrained))

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = params[0]
        self["state_cov"] = np.eye(K) * params[1]


print("Fitting Kalman filter (TVP regression)...")
# Demean y and X slightly to reduce intercept concerns — but keep raw for pure factor model
model = TVPRegression(y, X)
res = model.fit(disp=False, maxiter=200)

sigma2_obs  = res.params[0]
sigma2_state = res.params[1]
print(f"MLE: σ²_obs = {sigma2_obs:.4f}, σ²_state = {sigma2_state:.6f}")
print(f"Log-likelihood = {res.llf:.2f}")

# Get SMOOTHED states (two-sided) — uses full sample info for each β_t
smoothed_states = res.smoothed_state.T          # (T, K)
smoothed_cov    = res.smoothed_state_cov.transpose(2, 0, 1)  # (T, K, K)

# Get FILTERED states (one-sided, real-time) — only uses info up to t
filtered_states = res.filtered_state.T          # (T, K)

# Compute Kalman fitted value (in-sample, smoothed) and residuals
kalman_fit_smoothed = np.einsum("tk,tk->t", X, smoothed_states)
kalman_fit_filtered = np.einsum("tk,tk->t", X, filtered_states)
kalman_resid_smoothed = y - kalman_fit_smoothed
kalman_resid_filtered = y - kalman_fit_filtered

# One-step-ahead prediction residuals (truly out-of-sample for each point)
# At time t, predicted y_t uses β_{t|t-1} = β_{t-1|t-1} (random walk)
pred_states = np.vstack([np.zeros(K), filtered_states[:-1]])  # β_{t-1|t-1} shifted
pred_y = np.einsum("tk,tk->t", X, pred_states)
innovations = y - pred_y

# R² of smoothed fit
ss_res = np.nansum(kalman_resid_smoothed**2)
ss_tot = np.nansum((y - y.mean())**2)
kalman_r2 = 1 - ss_res / ss_tot

# Rich/Cheap from one-step-ahead innovations (real-time signal)
cum_innov = np.nancumsum(innovations)
innov_series = pd.Series(innovations)
ROLL = 52
z_innov = ((innov_series - innov_series.rolling(ROLL).mean())
           / innov_series.rolling(ROLL).std()).values

cum_innov_s = pd.Series(cum_innov)
ROLL2 = 104
z_cum = ((cum_innov_s - cum_innov_s.rolling(ROLL2).mean())
         / cum_innov_s.rolling(ROLL2).std()).values

# Confidence bands for smoothed betas
beta_se = np.sqrt(np.array([smoothed_cov[t].diagonal() for t in range(T)]))

# Current vs long-run OLS comparison
ols = sm.OLS(y, sm.add_constant(X)).fit()
ols_betas = ols.params[1:]

current_kalman_betas = smoothed_states[-1]
print("\nCurrent Kalman vs OLS betas:")
for i, f in enumerate(FACTOR_NAMES):
    print(f"  {f:6s}: Kalman={current_kalman_betas[i]:+.3f}  OLS={ols_betas[i]:+.3f}")

last_z_innov = z_innov[-1] if not np.isnan(z_innov[-1]) else 0
last_z_cum   = z_cum[-1]   if not np.isnan(z_cum[-1])   else 0
last_innov   = innovations[-1]
last_cum_innov = cum_innov[-1]

# Signal
if   last_z_cum >  2: signal_k = "EXPENSIVE"
elif last_z_cum >  1: signal_k = "SLIGHTLY EXPENSIVE"
elif last_z_cum < -2: signal_k = "CHEAP"
elif last_z_cum < -1: signal_k = "SLIGHTLY CHEAP"
else:                 signal_k = "FAIR VALUE"

signal_color = "#e74c3c" if "EXPENSIVE" in signal_k else "#2ecc71" if "CHEAP" in signal_k else "#f1c40f"


# ══════════════════════════════════════════════════════════════════════════
#  Build Plotly figures
# ══════════════════════════════════════════════════════════════════════════

# 1. Kalman smoothed betas with bands
figs = {}
fig = make_subplots(rows=3, cols=2, subplot_titles=FACTOR_NAMES + [""],
                    vertical_spacing=0.08, horizontal_spacing=0.07)
positions = [(1,1), (1,2), (2,1), (2,2), (3,1)]
for i, f in enumerate(FACTOR_NAMES):
    r, c = positions[i]
    upper = smoothed_states[:, i] + 1.96 * beta_se[:, i]
    lower = smoothed_states[:, i] - 1.96 * beta_se[:, i]
    # Confidence band
    fig.add_trace(go.Scatter(x=date_str + date_str[::-1],
                             y=np.concatenate([upper, lower[::-1]]),
                             fill="toself", fillcolor=f"rgba(231,76,60,0.15)",
                             line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                             hoverinfo="skip"), row=r, col=c)
    # Smoothed beta
    fig.add_trace(go.Scatter(x=date_str, y=smoothed_states[:, i], name=f,
                             line=dict(color=COLORS[f], width=2), showlegend=False),
                  row=r, col=c)
    # OLS constant line
    fig.add_hline(y=ols_betas[i], line_dash="dash", line_color="white",
                  opacity=0.5, row=r, col=c)
fig.update_layout(title="Kalman Smoothed Betas (95% Confidence Band; dashed = long-run OLS)",
                  template="plotly_dark", height=700)
figs["kalman_betas"] = fig

# 2. Kalman vs Rolling OLS betas comparison (overlay)
fig = make_subplots(rows=3, cols=2, subplot_titles=FACTOR_NAMES + [""],
                    vertical_spacing=0.08, horizontal_spacing=0.07)
# Compute rolling OLS
ROLL_WIN = 52
roll_betas = np.full((T, K), np.nan)
for i in range(ROLL_WIN, T):
    try:
        m = sm.OLS(y[i-ROLL_WIN:i], sm.add_constant(X[i-ROLL_WIN:i])).fit()
        roll_betas[i] = m.params[1:]
    except: pass

for i, f in enumerate(FACTOR_NAMES):
    r, c = positions[i]
    fig.add_trace(go.Scatter(x=date_str, y=roll_betas[:, i], name="Rolling OLS",
                             line=dict(color="#8b949e", width=1.5, dash="dot"),
                             showlegend=(i==0), legendgroup="rol"), row=r, col=c)
    fig.add_trace(go.Scatter(x=date_str, y=smoothed_states[:, i], name="Kalman Smoothed",
                             line=dict(color=COLORS[f], width=2),
                             showlegend=(i==0), legendgroup="kal"), row=r, col=c)
fig.update_layout(title="Kalman Smoothed vs 52-Week Rolling OLS Betas",
                  template="plotly_dark", height=700,
                  legend=dict(orientation="h", y=-0.05))
figs["kalman_vs_roll"] = fig

# 3. One-step-ahead innovations (real-time residuals)
fig = go.Figure()
colors_bar = ["#2ecc71" if v < 0 else "#e74c3c" for v in innovations]
fig.add_trace(go.Bar(x=date_str, y=innovations, marker_color=colors_bar,
                     name="Innovation"))
fig.add_hline(y=0, line_color="white", opacity=0.3)
fig.update_layout(title="Kalman One-Step-Ahead Innovations (Real-Time Residuals)",
                  yaxis_title="Innovation (%)", template="plotly_dark", height=350)
figs["innov"] = fig

# 4. Cumulative innovation (rich/cheap path)
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=cum_innov, name="Cum. Innovation",
                         line=dict(color="#e74c3c", width=2), fill="tozeroy",
                         fillcolor="rgba(231,76,60,0.12)"))
fig.add_hline(y=0, line_dash="dash", line_color="grey")
fig.update_layout(title="Cumulative Kalman Innovation (Rich/Cheap Path)",
                  yaxis_title="Cum. Innovation (%)", template="plotly_dark", height=350)
figs["cum_innov"] = fig

# 5. Z-score of cumulative innovation
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=z_cum, name="Z-Score",
                         line=dict(color="#9b59b6", width=2)))
fig.add_hline(y=2, line_dash="dash", line_color="#e74c3c", annotation_text="Expensive (+2σ)")
fig.add_hline(y=-2, line_dash="dash", line_color="#2ecc71", annotation_text="Cheap (-2σ)")
fig.add_hline(y=0, line_dash="dot", line_color="grey")
fig.add_hrect(y0=-2, y1=2, fillcolor="rgba(255,255,255,0.03)", line_width=0)
fig.update_layout(title="Kalman Cumulative Innovation Z-Score (2-Year Rolling)",
                  yaxis_title="Z-Score", template="plotly_dark", height=350)
figs["z_cum"] = fig

# 6. Kalman fitted vs actual
fig = go.Figure()
fig.add_trace(go.Scatter(x=date_str, y=y, name="Actual COPX",
                         line=dict(color=COLORS["COPX"], width=1)))
fig.add_trace(go.Scatter(x=date_str, y=kalman_fit_smoothed, name="Kalman Smoothed Fit",
                         line=dict(color="#f1c40f", width=1.5, dash="dot")))
fig.update_layout(title=f"Kalman Smoothed Fit (R² = {kalman_r2:.3f})",
                  yaxis_title="Weekly Return (%)", template="plotly_dark", height=380,
                  legend=dict(orientation="h", y=-0.15))
figs["kalman_fit"] = fig

# 7. Current beta bar: Kalman vs OLS
fig = go.Figure()
fig.add_trace(go.Bar(name="Kalman (current)", x=FACTOR_NAMES, y=current_kalman_betas,
                     marker_color="#e74c3c", text=[f"{v:.3f}" for v in current_kalman_betas],
                     textposition="outside"))
fig.add_trace(go.Bar(name="OLS (long-run)", x=FACTOR_NAMES, y=ols_betas,
                     marker_color="#3498db", text=[f"{v:.3f}" for v in ols_betas],
                     textposition="outside"))
fig.update_layout(title="Current Kalman Betas vs Long-Run OLS Betas",
                  yaxis_title="Beta", barmode="group", template="plotly_dark", height=380,
                  legend=dict(orientation="h", y=-0.15))
figs["beta_bar"] = fig

fig_htmls = {}
for k, f in figs.items():
    f.update_layout(margin=dict(l=50, r=30, t=60, b=40))
    fig_htmls[k] = f.to_html(full_html=False, include_plotlyjs=False, div_id=f"kchart-{k}")


# ══════════════════════════════════════════════════════════════════════════
#  Build HTML snippet and append to dashboard
# ══════════════════════════════════════════════════════════════════════════

# Build a comparison table: current Kalman vs OLS betas, and change
beta_table_html = """<h3>Betas: Current Kalman (Smoothed) vs Long-Run OLS</h3>
<table class="data-table"><thead><tr><th></th>"""
for f in FACTOR_NAMES: beta_table_html += f"<th>{f}</th>"
beta_table_html += "</tr></thead><tbody>"
beta_table_html += "<tr><td class='row-label'>Kalman β (latest)</td>"
for v in current_kalman_betas:
    color = "color:#2ecc71" if v > 0 else "color:#e74c3c"
    beta_table_html += f"<td style='{color}'>{v:+.3f}</td>"
beta_table_html += "</tr><tr><td class='row-label'>OLS β (full sample)</td>"
for v in ols_betas:
    color = "color:#2ecc71" if v > 0 else "color:#e74c3c"
    beta_table_html += f"<td style='{color}'>{v:+.3f}</td>"
beta_table_html += "</tr><tr><td class='row-label'>Δ (Kalman − OLS)</td>"
for i, v in enumerate(current_kalman_betas - ols_betas):
    color = "color:#2ecc71" if v > 0 else "color:#e74c3c"
    beta_table_html += f"<td style='{color}'>{v:+.3f}</td>"
beta_table_html += "</tr></tbody></table>"


kalman_section = f"""

<!-- ═══ KALMAN FILTER ═══ -->
<div id="kalman" class="section-title">Kalman Filter: Time-Varying Betas</div>

<div class="insight">
  <strong>Model:</strong> COPX_t = x_t' β_t + ε_t &nbsp;|&nbsp; β_t = β_{{t-1}} + η_t (random-walk coefficients)<br>
  <strong>MLE Hyperparameters:</strong> σ²_obs = {sigma2_obs:.4f} &nbsp;|&nbsp; σ²_state = {sigma2_state:.6f}
  &nbsp;|&nbsp; Log-Likelihood = {res.llf:.1f}<br>
  <strong>Smoothed-fit R² = {kalman_r2:.4f}</strong> (vs. static OLS R² = {ols.rsquared:.4f}).
  Allowing betas to drift time-varyingly explains {(kalman_r2 - ols.rsquared)*100:+.1f} pp more of COPX's variance.
</div>

<div class="signal-box">
  <div class="signal-card">
    <div class="label">Kalman Verdict</div>
    <div class="value" style="color:{signal_color}">{signal_k}</div>
    <div class="sub">from cum. innovation z-score</div>
  </div>
  <div class="signal-card">
    <div class="label">Cum. Innov. Z-Score</div>
    <div class="value" style="color:{'#2ecc71' if last_z_cum < 0 else '#e74c3c'}">{last_z_cum:+.2f}σ</div>
    <div class="sub">2-year rolling</div>
  </div>
  <div class="signal-card">
    <div class="label">Last Innovation</div>
    <div class="value" style="color:{'#2ecc71' if last_innov < 0 else '#e74c3c'}">{last_innov:+.2f}%</div>
    <div class="sub">one-step-ahead surprise</div>
  </div>
  <div class="signal-card">
    <div class="label">Cum. Innovation</div>
    <div class="value" style="color:{'#2ecc71' if last_cum_innov < 0 else '#e74c3c'}">{last_cum_innov:+.1f}%</div>
    <div class="sub">total vs real-time model</div>
  </div>
  <div class="signal-card">
    <div class="label">Current Copper β</div>
    <div class="value" style="color:#e67e22">{current_kalman_betas[0]:+.2f}</div>
    <div class="sub">vs OLS: {ols_betas[0]:+.2f}</div>
  </div>
</div>

<div class="grid grid-2">
  <div class="card">{beta_table_html}</div>
  <div class="card">{fig_htmls['beta_bar']}</div>
</div>

<div class="card card-full">{fig_htmls['kalman_betas']}</div>

<div class="card card-full">{fig_htmls['kalman_vs_roll']}</div>

<div class="grid grid-2">
  <div class="card">{fig_htmls['kalman_fit']}</div>
  <div class="card">{fig_htmls['innov']}</div>
</div>

<div class="grid grid-2">
  <div class="card">{fig_htmls['cum_innov']}</div>
  <div class="card">{fig_htmls['z_cum']}</div>
</div>

<div class="insight">
  <strong>Interpretation:</strong>
  The Kalman filter treats each β_t as a latent state evolving smoothly over time.
  The <em>smoothed</em> beta uses the full sample (best in-sample estimate).
  The <em>one-step-ahead innovation</em> uses only information available up to t−1, so it's the honest real-time
  "surprise" vs. what the filter would have predicted — a proper out-of-sample residual at every point.
  The innovation-based rich/cheap signal is therefore more faithful than the static OLS residual signal above.
</div>
"""

# ── Append to dashboard ──────────────────────────────────────────────────
with open(DASH, "r", encoding="utf-8") as f:
    html = f.read()

# Add nav link for Kalman
if "#kalman" not in html:
    html = html.replace('<a href="#richcheap">Rich/Cheap</a>',
                        '<a href="#richcheap">Rich/Cheap</a>\n  <a href="#kalman">Kalman</a>')

# Remove any prior Kalman section and insert fresh one before </div><!-- container -->
html = re.sub(r'<!-- ═══ KALMAN FILTER ═══ -->.*?(?=<div style="text-align:center; padding:30px)',
              '', html, flags=re.DOTALL)
# Insert before the footer div
html = html.replace('</div><!-- container -->', kalman_section + '\n</div><!-- container -->', 1)

with open(DASH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nKalman section appended to: {DASH}")
print(f"Signal: {signal_k} (Z={last_z_cum:+.2f})")
