#!/usr/bin/env python3
"""Build JOLTS interactive dashboard as a single HTML file."""

import json
from datetime import datetime

# Load all BLS data files
all_series = {}
for f in ["/tmp/jolts_bls_data.json", "/tmp/jolts_bls_data2.json", "/tmp/jolts_bls_data3.json"]:
    with open(f) as fh:
        data = json.load(fh)
        for s in data.get("Results", {}).get("series", []):
            sid = s["seriesID"]
            if s.get("data"):
                all_series[sid] = s["data"]

# Series ID mapping
INDUSTRY_CODES = {
    "100": "Mining & Logging",
    "200": "Construction", 
    "300": "Manufacturing",
    "400": "Trade, Transport & Utilities",
    "510": "Information",
    "520": "Finance & Insurance",
    "530": "Real Estate",
    "540": "Prof & Business Services",
    "600": "Education & Health Services",
    "700": "Leisure & Hospitality",
    "810": "Other Services",
    "900": "Government",
    "000": "Total Nonfarm",
}

METRIC_CODES = {
    "JOL": "Job Openings",
    "QUL": "Quits",
    "HIL": "Hires",
    "LDL": "Layoffs & Discharges",
    "TSL": "Total Separations",
}

def parse_series(series_data):
    """Convert BLS series data to sorted time series."""
    points = []
    for d in series_data:
        year = int(d["year"])
        month = int(d["period"].replace("M", ""))
        val = float(d["value"]) if d["value"] else None
        if val is not None:
            points.append({"date": f"{year}-{month:02d}-01", "value": val})
    points.sort(key=lambda x: x["date"])
    return points

def get_industry_code(sid):
    # Extract industry code: JTS{code}00000000{metric}
    code = sid[3:6]
    if code == "000":
        return "000"
    return code

def get_metric_code(sid):
    return sid[-3:]

# Build structured data for JS
js_data = {}

for sid, raw in all_series.items():
    ind_code = get_industry_code(sid)
    met_code = get_metric_code(sid)
    
    ind_name = INDUSTRY_CODES.get(ind_code, f"Unknown ({ind_code})")
    met_name = METRIC_CODES.get(met_code, f"Unknown ({met_code})")
    
    points = parse_series(raw)
    if not points:
        continue
    
    key = f"{ind_name}|{met_name}"
    js_data[key] = points

# Get latest month data for the snapshot table
latest_openings = {}
for sid, raw in all_series.items():
    if not sid.endswith("JOL"):
        continue
    ind_code = get_industry_code(sid)
    ind_name = INDUSTRY_CODES.get(ind_code, f"Unknown ({ind_code})")
    points = parse_series(raw)
    if len(points) >= 2:
        latest_openings[ind_name] = {
            "current": points[-1]["value"],
            "previous": points[-2]["value"],
            "date": points[-1]["date"],
            "prev_date": points[-2]["date"],
        }
        # Find pre-covid (Jan 2020)
        for p in points:
            if p["date"] == "2020-01-01":
                latest_openings[ind_name]["pre_covid"] = p["value"]
        # Find peak
        peak = max(points, key=lambda x: x["value"])
        latest_openings[ind_name]["peak"] = peak["value"]
        latest_openings[ind_name]["peak_date"] = peak["date"]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JOLTS Dashboard — Marshall 🔶</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root {{
    --bg: #0a0a0a;
    --surface: #141414;
    --border: #2a2a2a;
    --text: #e0e0e0;
    --muted: #888;
    --accent: #f59e0b;
    --green: #22c55e;
    --red: #ef4444;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Inter', sans-serif; padding: 20px; }}
.header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 24px; flex-wrap: wrap; gap: 8px; }}
.header h1 {{ font-size: 1.5rem; font-weight: 600; }}
.header h1 span {{ color: var(--accent); }}
.header .meta {{ color: var(--muted); font-size: 0.85rem; }}
.controls {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
.controls select, .controls button {{
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    padding: 8px 14px; border-radius: 6px; font-size: 0.85rem; cursor: pointer;
}}
.controls select:focus, .controls button:focus {{ outline: none; border-color: var(--accent); }}
.controls button.active {{ background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 24px; }}
@media (min-width: 900px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
.card h3 {{ font-size: 0.95rem; margin-bottom: 12px; color: var(--muted); }}
.chart-container {{ position: relative; height: 320px; }}
.snapshot {{ overflow-x: auto; }}
.snapshot table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
.snapshot th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border); color: var(--accent); font-weight: 600; white-space: nowrap; }}
.snapshot td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
.snapshot tr:hover td {{ background: rgba(245,158,11,0.05); }}
.up {{ color: var(--green); }}
.down {{ color: var(--red); }}
.flat {{ color: var(--muted); }}
.tabs {{ display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }}
.tab {{ padding: 6px 12px; border-radius: 4px; font-size: 0.8rem; cursor: pointer; background: var(--surface); border: 1px solid var(--border); color: var(--muted); }}
.tab.active {{ background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }}
.footer {{ text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 30px; padding-top: 16px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<div class="header">
    <h1>🔶 JOLTS Dashboard <span>— January 2026</span></h1>
    <div class="meta">Source: BLS JOLTS | Built by Marshall</div>
</div>

<!-- Snapshot Table -->
<div class="card snapshot" style="margin-bottom: 24px;">
    <h3>📊 Job Openings Snapshot (thousands)</h3>
    <table id="snapshotTable">
        <thead>
            <tr><th>Sector</th><th>Jan 2026</th><th>Dec 2025</th><th>MoM Δ</th><th>Peak</th><th>vs Peak</th></tr>
        </thead>
        <tbody></tbody>
    </table>
</div>

<!-- Chart Controls -->
<div class="controls">
    <select id="metricSelect">
        <option value="Job Openings" selected>Job Openings</option>
        <option value="Hires">Hires</option>
        <option value="Quits">Quits</option>
        <option value="Layoffs & Discharges">Layoffs & Discharges</option>
        <option value="Total Separations">Total Separations</option>
    </select>
    <select id="rangeSelect">
        <option value="all">All Data</option>
        <option value="3y" selected>3 Years</option>
        <option value="1y">1 Year</option>
    </select>
</div>

<!-- Industry Tabs -->
<div class="tabs" id="industryTabs"></div>

<!-- Main Chart -->
<div class="card">
    <h3 id="chartTitle">Job Openings — All Industries</h3>
    <div class="chart-container">
        <canvas id="mainChart"></canvas>
    </div>
</div>

<!-- Multi-panel: All Industries -->
<div class="grid" id="smallCharts"></div>

<!-- Beveridge Curve -->
<div class="card" style="margin-top: 20px;">
    <h3>🔄 Beveridge Curve (Job Openings Rate vs Unemployment — conceptual)</h3>
    <div class="chart-container">
        <canvas id="beveridgeChart"></canvas>
    </div>
</div>

<div class="footer">
    Data: U.S. Bureau of Labor Statistics, JOLTS January 2026 (released Mar 13) | Dashboard: Marshall 🔶 for Wilson
</div>

<script>
const RAW_DATA = {json.dumps(js_data)};
const SNAPSHOT = {json.dumps(latest_openings)};

const COLORS = [
    '#f59e0b', '#3b82f6', '#ef4444', '#22c55e', '#a855f7',
    '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#6366f1',
    '#14b8a6', '#e11d48', '#8b5cf6'
];

const INDUSTRIES = [
    'Total Nonfarm', 'Education & Health Services', 'Leisure & Hospitality',
    'Trade, Transport & Utilities', 'Prof & Business Services', 'Government',
    'Manufacturing', 'Finance & Insurance', 'Construction', 'Information',
    'Real Estate', 'Mining & Logging', 'Other Services'
];

// Build snapshot table
function buildSnapshot() {{
    const tbody = document.querySelector('#snapshotTable tbody');
    const order = INDUSTRIES.filter(i => i !== 'Total Nonfarm');
    // Total first
    const rows = ['Total Nonfarm', ...order];
    
    for (const name of rows) {{
        const d = SNAPSHOT[name];
        if (!d) continue;
        
        const mom = d.current - d.previous;
        const momPct = ((mom / d.previous) * 100).toFixed(1);
        const vsPeak = (((d.current - d.peak) / d.peak) * 100).toFixed(1);
        
        const momClass = mom > 0 ? 'up' : mom < 0 ? 'down' : 'flat';
        const peakClass = parseFloat(vsPeak) >= 0 ? 'up' : 'down';
        const bold = name === 'Total Nonfarm' ? 'font-weight:600;' : '';
        
        tbody.innerHTML += `<tr style="${{bold}}">
            <td>${{name}}</td>
            <td>${{d.current.toLocaleString()}}</td>
            <td>${{d.previous.toLocaleString()}}</td>
            <td class="${{momClass}}">${{mom > 0 ? '+' : ''}}${{mom.toLocaleString()}} (${{momPct}}%)</td>
            <td>${{d.peak.toLocaleString()}} (${{d.peak_date.slice(0,7)}})</td>
            <td class="${{peakClass}}">${{vsPeak}}%</td>
        </tr>`;
    }}
}}
buildSnapshot();

// Charts
let mainChart = null;
const smallCharts = {{}};

function getSeriesData(industry, metric) {{
    const key = `${{industry}}|${{metric}}`;
    return RAW_DATA[key] || [];
}}

function filterByRange(data, range) {{
    if (range === 'all') return data;
    const now = new Date();
    const cutoff = new Date();
    if (range === '1y') cutoff.setFullYear(now.getFullYear() - 1);
    if (range === '3y') cutoff.setFullYear(now.getFullYear() - 3);
    return data.filter(d => new Date(d.date) >= cutoff);
}}

function buildMainChart() {{
    const metric = document.getElementById('metricSelect').value;
    const range = document.getElementById('rangeSelect').value;
    const activeTab = document.querySelector('.tab.active');
    const industry = activeTab ? activeTab.dataset.industry : 'all';
    
    const ctx = document.getElementById('mainChart').getContext('2d');
    if (mainChart) mainChart.destroy();
    
    const datasets = [];
    
    if (industry === 'all') {{
        // Show all industries
        INDUSTRIES.forEach((ind, i) => {{
            if (ind === 'Total Nonfarm') return;
            const data = filterByRange(getSeriesData(ind, metric), range);
            if (!data.length) return;
            datasets.push({{
                label: ind,
                data: data.map(d => ({{ x: d.date, y: d.value }})),
                borderColor: COLORS[i % COLORS.length],
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3,
            }});
        }});
        document.getElementById('chartTitle').textContent = `${{metric}} — All Industries (thousands)`;
    }} else if (industry === 'Total Nonfarm') {{
        // Show total with all metrics
        ['Job Openings', 'Hires', 'Quits', 'Layoffs & Discharges'].forEach((m, i) => {{
            const data = filterByRange(getSeriesData('Total Nonfarm', m), range);
            if (!data.length) return;
            datasets.push({{
                label: m,
                data: data.map(d => ({{ x: d.date, y: d.value }})),
                borderColor: COLORS[i],
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }});
        }});
        document.getElementById('chartTitle').textContent = `Total Nonfarm — All Metrics (thousands)`;
    }} else {{
        // Single industry, show all available metrics
        ['Job Openings', 'Hires', 'Quits', 'Layoffs & Discharges'].forEach((m, i) => {{
            const data = filterByRange(getSeriesData(industry, m), range);
            if (!data.length) return;
            datasets.push({{
                label: m,
                data: data.map(d => ({{ x: d.date, y: d.value }})),
                borderColor: COLORS[i],
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }});
        }});
        document.getElementById('chartTitle').textContent = `${{industry}} — All Metrics (thousands)`;
    }}
    
    mainChart = new Chart(ctx, {{
        type: 'line',
        data: {{ datasets }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', labels: {{ color: '#888', font: {{ size: 11 }}, boxWidth: 12 }} }},
                tooltip: {{
                    backgroundColor: '#1a1a1a',
                    titleColor: '#f59e0b',
                    bodyColor: '#e0e0e0',
                    borderColor: '#333',
                    borderWidth: 1,
                    callbacks: {{
                        label: function(ctx) {{
                            return `${{ctx.dataset.label}}: ${{ctx.parsed.y.toLocaleString()}}K`;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{ type: 'time', time: {{ unit: 'quarter' }}, grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#666' }} }},
                y: {{ grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#666', callback: v => v.toLocaleString() }} }}
            }}
        }}
    }});
}}

// Build industry tabs
function buildTabs() {{
    const container = document.getElementById('industryTabs');
    container.innerHTML = '<div class="tab active" data-industry="all">All Industries</div>';
    INDUSTRIES.forEach(ind => {{
        container.innerHTML += `<div class="tab" data-industry="${{ind}}">${{ind}}</div>`;
    }});
    
    container.querySelectorAll('.tab').forEach(tab => {{
        tab.addEventListener('click', () => {{
            container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            buildMainChart();
        }});
    }});
}}
buildTabs();

// Build small multiples for job openings by industry
function buildSmallCharts() {{
    const container = document.getElementById('smallCharts');
    container.innerHTML = '';
    const range = document.getElementById('rangeSelect').value;
    
    INDUSTRIES.forEach((ind, i) => {{
        if (ind === 'Total Nonfarm') return;
        const data = filterByRange(getSeriesData(ind, 'Job Openings'), range);
        if (!data.length) return;
        
        const latest = data[data.length - 1].value;
        const prev = data.length > 1 ? data[data.length - 2].value : latest;
        const change = latest - prev;
        const changeClass = change > 0 ? 'up' : change < 0 ? 'down' : 'flat';
        const sign = change > 0 ? '+' : '';
        
        const div = document.createElement('div');
        div.className = 'card';
        div.innerHTML = `
            <h3>${{ind}} <span class="${{changeClass}}" style="float:right">${{sign}}${{change.toLocaleString()}} (${{((change/prev)*100).toFixed(1)}}%)</span></h3>
            <div style="font-size:1.3rem;font-weight:600;margin-bottom:8px;">${{latest.toLocaleString()}}K</div>
            <div class="chart-container" style="height:180px;"><canvas id="small-${{i}}"></canvas></div>
        `;
        container.appendChild(div);
        
        new Chart(document.getElementById(`small-${{i}}`).getContext('2d'), {{
            type: 'line',
            data: {{
                datasets: [{{
                    data: data.map(d => ({{ x: d.date, y: d.value }})),
                    borderColor: COLORS[i % COLORS.length],
                    backgroundColor: COLORS[i % COLORS.length] + '15',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: true,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ type: 'time', display: false }},
                    y: {{ grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#555', font: {{ size: 10 }} }} }}
                }}
            }}
        }});
    }});
}}

// Beveridge-style chart (openings vs quits as proxy)
function buildBeveridge() {{
    const openings = getSeriesData('Total Nonfarm', 'Job Openings');
    const quits = getSeriesData('Total Nonfarm', 'Quits');
    
    if (!openings.length || !quits.length) return;
    
    // Align by date
    const qMap = {{}};
    quits.forEach(d => qMap[d.date] = d.value);
    
    const points = [];
    openings.forEach(d => {{
        if (qMap[d.date]) {{
            points.push({{ x: d.value, y: qMap[d.date], date: d.date }});
        }}
    }});
    
    const recent = points.slice(-12);
    const older = points.slice(0, -12);
    
    new Chart(document.getElementById('beveridgeChart').getContext('2d'), {{
        type: 'scatter',
        data: {{
            datasets: [
                {{
                    label: 'Historical',
                    data: older,
                    backgroundColor: '#3b82f650',
                    borderColor: '#3b82f6',
                    pointRadius: 3,
                }},
                {{
                    label: 'Last 12 months',
                    data: recent,
                    backgroundColor: '#f59e0b',
                    borderColor: '#f59e0b',
                    pointRadius: 5,
                    pointStyle: 'triangle',
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ labels: {{ color: '#888' }} }},
                tooltip: {{
                    callbacks: {{
                        label: function(ctx) {{
                            const d = ctx.raw;
                            return `${{d.date.slice(0,7)}}: Openings ${{d.x.toLocaleString()}}K, Quits ${{d.y.toLocaleString()}}K`;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{ title: {{ display: true, text: 'Job Openings (K)', color: '#888' }}, grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#666' }} }},
                y: {{ title: {{ display: true, text: 'Quits (K)', color: '#888' }}, grid: {{ color: '#1a1a1a' }}, ticks: {{ color: '#666' }} }}
            }}
        }}
    }});
}}

// Event listeners
document.getElementById('metricSelect').addEventListener('change', () => {{ buildMainChart(); buildSmallCharts(); }});
document.getElementById('rangeSelect').addEventListener('change', () => {{ buildMainChart(); buildSmallCharts(); }});

// Init
buildMainChart();
buildSmallCharts();
buildBeveridge();
</script>
</body>
</html>"""

# Write output
out_path = "/Users/wilsonfelicio/.openclaw/workspace/jolts-dashboard/index.html"
with open(out_path, "w") as f:
    f.write(html)
print(f"Dashboard written to {out_path}")
print(f"Size: {len(html):,} bytes")
print(f"Series loaded: {len(js_data)}")
