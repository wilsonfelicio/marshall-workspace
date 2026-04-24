# Dashboard Build Prompt

> Copy-paste this entire prompt into Claude Code or the OpenClaw coding agent.

---

Build me a single-page interactive market dashboard in HTML. One file, everything inline (CSS + JS). Use Chart.js from CDN.

## Layout

Top section: Title bar with "Market Snapshot" and a timestamp showing when it was last updated.

Main grid: Cards for each asset showing:
- Asset name and ticker
- Current price (use realistic recent values)  
- Daily change (absolute and percentage)
- Small sparkline or mini chart showing 5-day trend
- Color-coded: green for positive, red for negative

## Assets to Include

**Equities:**
- S&P 500 (^GSPC)
- Nasdaq Composite (^IXIC)  
- Dow Jones (^DJI)

**Commodities:**
- Gold (XAU/USD)
- Crude Oil WTI (CL)

**Rates:**
- US 10Y Treasury Yield
- US 2Y Treasury Yield

**FX:**
- DXY (Dollar Index)
- USD/BRL

**Volatility:**
- VIX

## Charts Section

Below the cards grid, add two larger charts:
1. **S&P 500 — 30 Day Trend** (line chart, area fill)
2. **Yield Curve** — Plot 2Y vs 10Y yields over last 30 days (two lines, show spread)

Use realistic synthetic data for the charts (based on recent market conditions — S&P around 6,800, 10Y around 4.3%, 2Y around 3.7%, Gold around 4,800, Oil WTI around 96, VIX around 19, DXY around 99, USD/BRL around 5.00).

## Design Requirements

- **Dark theme**: Background #0d1117, cards #161b22, borders #30363d
- Text: #e6edf3 primary, #8b949e secondary
- Green: #3fb950, Red: #f85149
- Font: system-ui, -apple-system, sans-serif
- **Mobile responsive**: Cards stack on small screens, charts resize
- Smooth hover effects on cards
- No external dependencies except Chart.js CDN

## Technical

- Single HTML file, self-contained
- Chart.js loaded from: https://cdn.jsdelivr.net/npm/chart.js
- All data hardcoded (this is a snapshot, not live)
- Clean, commented code
- File name: index.html

Build the complete file. Make it look like something you'd show a client.
