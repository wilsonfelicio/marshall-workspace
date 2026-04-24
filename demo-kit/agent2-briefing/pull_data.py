#!/usr/bin/env python3
"""
Market Data Snapshot
====================
Pulls key market data from Yahoo Finance public API.
No API keys needed. No pip install needed. Pure stdlib.

Assets: S&P 500, Nasdaq, DXY, USD/BRL, US 10Y, VIX, Gold, Oil (WTI)

Usage:
    python3 pull_data.py
"""

import urllib.request
import json
import datetime
import ssl

# ---------------------------------------------------------------------------
# CONFIG: Tickers to fetch
# ---------------------------------------------------------------------------
# Each entry: (Yahoo ticker, display name, category, format_type)
# format_type: "price" = show as price, "pct" = show as percentage (yields)

TICKERS = [
    # Equities
    ("^GSPC",   "S&P 500",     "📈 EQUITIES",     "price"),
    ("^IXIC",   "Nasdaq",      "📈 EQUITIES",     "price"),

    # FX
    ("DX-Y.NYB", "DXY",        "💵 FX",           "price"),
    ("BRL=X",    "USD/BRL",    "💵 FX",           "price"),

    # Rates
    ("^TNX",    "US 10Y Yield", "🏦 RATES",       "pct"),

    # Volatility
    ("^VIX",    "VIX",         "⚡ VOLATILITY",   "price"),

    # Commodities
    ("GC=F",    "Gold (XAU)",  "🛢️ COMMODITIES", "price"),
    ("CL=F",    "Oil WTI",     "🛢️ COMMODITIES", "price"),
]

# ---------------------------------------------------------------------------
# FETCH: Hit Yahoo Finance's public chart API
# ---------------------------------------------------------------------------

def fetch_quote(ticker: str) -> dict:
    """
    Fetch current price and daily change from Yahoo Finance.
    Returns dict with 'price', 'change_pct', or 'error'.
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range=2d&interval=1d"
    )

    # Yahoo requires a User-Agent header
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    })

    # Bypass SSL verification (macOS Python often lacks certs; safe for public APIs)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = meta["regularMarketPrice"]

        # Calculate daily change from previous close
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        if prev_close and prev_close != 0:
            change_pct = ((price - prev_close) / prev_close) * 100
        else:
            change_pct = 0.0

        return {"price": price, "change_pct": change_pct}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# FORMAT: Build the output text block
# ---------------------------------------------------------------------------

def format_output(results: list) -> str:
    """
    Format all results into a clean text block with sections.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append("═" * 45)
    lines.append("  MARKET DATA SNAPSHOT")
    lines.append(f"  {now}")
    lines.append("═" * 45)

    current_category = None

    for ticker, name, category, fmt_type, data in results:
        # Print category header when it changes
        if category != current_category:
            lines.append("")
            lines.append(f"{category}")
            current_category = category

        if "error" in data:
            lines.append(f"  {name:20s}  ⚠️  Error: {data['error'][:40]}")
            continue

        price = data["price"]
        change = data["change_pct"]

        # Format the price/yield value
        if fmt_type == "pct":
            price_str = f"{price:.2f}%"
        elif price >= 1000:
            price_str = f"{price:,.2f}"
        else:
            price_str = f"{price:.2f}"

        # Format the change with arrow
        if change >= 0:
            change_str = f"+{change:.2f}%"
        else:
            change_str = f"{change:.2f}%"

        lines.append(f"  {name:20s} {price_str:>12s}   {change_str:>8s}")

    lines.append("")
    lines.append("═" * 45)

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fetch all tickers
    results = []
    for ticker, name, category, fmt_type in TICKERS:
        data = fetch_quote(ticker)
        results.append((ticker, name, category, fmt_type, data))

    # Print formatted output
    output = format_output(results)
    print(output)
