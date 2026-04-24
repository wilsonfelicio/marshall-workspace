# How to Trigger the Alert Live

## The Problem

Gold is probably trading between $4,500–$5,000 right now. The alert won't fire because there's nothing to report. But you need it to fire during the demo.

## The Solution

**Before the demo**, edit `alert_cron.json` and set the threshold to something gold has already crossed.

### Option 1: Lower the Upper Threshold (Recommended)

Check the current gold price. If gold is at ~$3,200, change the payload:

```
- If gold is ABOVE $3,000 → Send alert...
- If gold is BELOW $2,500 → Send alert...  
- If gold is BETWEEN $2,500 and $3,000 → NO_REPLY
```

Gold is above $3,000, so the alert fires immediately. ✅

### Option 2: Flip the Logic

Change "above $5,000" to "above $2,000" — any price gold has clearly crossed.

### During the Demo

1. **First:** Show the config with the "real" thresholds ($4,500–$5,000) and explain the logic
2. **Then say:** "Let me adjust the threshold so we can see it fire right now"
3. **Edit the threshold** visibly — this is actually a great demo moment:
   > "See how easy it is to adjust? I just changed one number."
4. **Create/trigger the cron job**
5. **Wait for the alert** (~15-30 seconds)
6. **After it fires, say:** "In production, I'd set this back to the real levels. The point is: one number change, and your monitoring adapts."

### Quick Check: Current Gold Price

Before the presentation, run:
```bash
python3 -c "
import urllib.request, json
url = 'https://query1.finance.yahoo.com/v8/finance/chart/GC=F?range=1d&interval=1d'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
data = json.loads(urllib.request.urlopen(req).read())
price = data['chart']['result'][0]['meta']['regularMarketPrice']
print(f'Gold is at: \${price:,.2f}')
print(f'Set upper threshold to: \${price - 100:,.2f} to trigger alert')
"
```

Or just ask the agent: "What's the current gold spot price?"

## Reset After Demo

Remember to set the thresholds back to real values after the demo, or just delete the cron job.
