# Agent 2: Morning Macro Briefing — Live Build Guide

## What You're Building

A daily cron job that:
1. Runs a Python script to pull market data (indices, rates, FX, commodities)
2. Feeds the data to the AI with instructions to write a macro summary
3. Delivers the briefing to Telegram at 7:00 AM

**Time to build live:** ~10 minutes (+ 5 min for audio bonus)

---

## Step-by-Step

### Step 1: The Data Script (3 min)

Open `pull_data.py` and walk through it:

> "This is ~80 lines of Python. No API keys, no pip install, no dependencies beyond what ships with Python. It hits Yahoo Finance's public endpoints and pulls everything we need."

**Run it live:**
```bash
python3 pull_data.py
```

**Expected output:**
```
═══════════════════════════════════════
  MARKET DATA SNAPSHOT
  2026-04-12 07:00 UTC
═══════════════════════════════════════

📈 EQUITIES
  S&P 500 (SPY)      5,234.18   +0.45%
  Nasdaq (QQQ)       18,012.50   +0.62%
  
💵 FX
  DXY               104.32   -0.18%
  USD/BRL              5.12   +0.25%
  
🏦 RATES
  US 10Y Yield         4.52%
  
⚡ VOLATILITY
  VIX                  14.32   -2.10%
  
🛢️ COMMODITIES
  Gold (XAU)        3,245.60   +1.20%
  Oil WTI              78.45   -0.35%
```

**Point out:**
- Clean formatting, emoji sections
- Percentage changes calculated automatically
- All from public Yahoo Finance data
- "This runs in 2 seconds. No Bloomberg terminal needed."

### Step 2: The Cron Config (3 min)

Open `briefing_cron.json` and explain:

> "The cron job does two things: first it runs the Python script, then it tells the AI to analyze the output and write a briefing."

**Key fields:**
- `schedule: "cron 0 10 * * 1-5"` — 10:00 UTC = 7:00 AM São Paulo, weekdays only
- `payload` — the prompt includes instructions to run the script AND write analysis
- `delivery` — straight to Telegram

**The prompt structure:**
1. "Run this Python script" (tool use: exec)
2. "Read the output"
3. "Write a 2-paragraph macro summary: what happened and what to watch"
4. "Keep it under 200 words. No clichés."

### Step 3: Create the Cron Job (2 min)

Create via UI or CLI, same as Demo 1.

### Step 4: Trigger It (2 min)

Run it manually to show the result:
> "Let's not wait until 7 AM tomorrow. Let me trigger it now."

The agent will:
1. Execute `pull_data.py`
2. Read the market data
3. Write a 2-paragraph briefing
4. Send it to Telegram

**Show the Telegram message** — this is the "money shot" of the demo.

---

### Bonus: Audio Briefing (5 min)

> "What if I don't want to read this at 7 AM? What if I want to listen to it while making coffee?"

Open `add_audio.md` — show how 3 lines of config add text-to-speech.

If you have a pre-generated audio file, **play it for the audience**. This always gets a reaction.

Show `generate_audio_simple.py` briefly:
> "20 lines of Python. Sends the text to ElevenLabs, gets back an MP3. That's it."

---

## What the Audience Should Take Away

1. **Real data, real analysis.** Not a toy — this pulls the same data you'd check manually.
2. **AI writes the narrative.** It doesn't just dump numbers — it tells you what matters.
3. **Fully autonomous.** Runs at 7 AM, delivers to your phone, no intervention needed.
4. **Cheap.** ~$0.15/run for the AI + ~$0.05 for TTS = $6/month for daily audio briefings.
5. **Customizable.** Change the prompt to focus on rates, add more assets, switch to Portuguese.

---

## Variations to Mention

- "Add Brazilian assets: IBOV, CDI, DI futures"
- "Include a 'what to watch today' section with economic calendar"
- "Send to a Teams channel for the whole desk"
- "Run it twice — 7 AM pre-market and 5 PM post-close"
