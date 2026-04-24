# Agent 1: Price Alert Bot — Live Build Guide

## What You're Building

A cron job that checks the gold spot price every 15 minutes. If it crosses above $5,000 or below $4,500, you get a Telegram alert. If nothing happened — silence.

**Time to build live:** ~5 minutes

---

## Step-by-Step (Talk Through Each One)

### Step 1: Show the Concept (1 min)

> "This is the simplest agent pattern: check a condition, alert if true, shut up if false. Like a Bloomberg alert, but you write the rules in plain English."

### Step 2: Open the Cron Config (2 min)

Open `alert_cron.json` and walk through each field:

```
File: alert_cron.json
```

**Fields to explain:**

| Field | What It Does |
|-------|-------------|
| `name` | Human-readable label — shows up in the UI |
| `schedule` | When it runs — `"every 15m"` means every 15 minutes |
| `payload` | The prompt — this is where the magic is |
| `delivery` | Where to send the result — Telegram chat ID |
| `sessionTarget` | Which session receives it (usually Telegram direct) |

**Key line in the payload:**
> "If the price is between $4,500 and $5,000, respond with exactly NO_REPLY and nothing else."

Explain: "NO_REPLY is a special token. When the agent returns this, OpenClaw swallows the message — nothing gets delivered. This is how you build silent monitors."

### Step 3: Create the Cron Job (2 min)

**Option A — Via OpenClaw UI:**
1. Open the OpenClaw web interface
2. Navigate to Cron Jobs
3. Click "Add New"
4. Paste the JSON config
5. Save and enable

**Option B — Via CLI:**
```bash
openclaw cron add --file agent1-price-alert/alert_cron.json
```

**Option C — Via the chat (most impressive for demo):**
Tell the agent directly:
> "Create a cron job that runs every 15 minutes. Check the current gold spot price using web_search. If gold is above $5,000 or below $4,500, send me an alert with the current price and the direction of the move. If it's between those levels, respond with NO_REPLY."

### Step 4: Trigger It Live (2 min)

See `demo_trigger.md` for how to set the threshold to guarantee a trigger during the demo.

### Step 5: Show the Result

When the alert arrives in Telegram, point out:
- It searched the web for the current price
- It evaluated the condition
- It formatted a clean alert
- Total time: ~10 seconds
- Cost: ~$0.02

---

## What the Audience Should Take Away

1. **It's just a prompt.** No code, no API integration, no webhook setup.
2. **NO_REPLY = silent monitoring.** The agent checks constantly but only speaks when something matters.
3. **Fully customizable.** Change the asset, the threshold, the message format — it's all in the prompt.
4. **Runs autonomously.** Set it and forget it. It'll keep checking every 15 minutes.

---

## Variations to Mention

- "You could do this for USD/BRL breaking 6.00"
- "Or VIX spiking above 30"
- "Or the 10-year yield crossing 5%"
- "Or any headline containing 'tariff' or 'rate cut'"
- "Stack multiple alerts — each one costs fractions of a cent"
