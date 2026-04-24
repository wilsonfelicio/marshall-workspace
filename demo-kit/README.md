# 🔶 Demo Kit: AI Agents for Portfolio Management

## One-Liner Pitch

> **"In the next hour, we'll build three AI agents that monitor markets, write macro briefings, and create dashboards — all running autonomously, zero infrastructure, from your laptop."**

---

## Pre-Show Setup Checklist

- [ ] OpenClaw running (`openclaw gateway status`)
- [ ] Telegram bot connected and delivering messages
- [ ] Terminal open with large font (⌘+ several times)
- [ ] Browser open with tabs:
  - Tab 1: OpenClaw web UI (cron jobs)
  - Tab 2: Telegram (to show deliveries arriving)
  - Tab 3: GitHub Pages (for dashboard deploy)
- [ ] All demo files loaded: `demo-kit/` directory
- [ ] Test `pull_data.py` works: `python3 demo-kit/agent2-briefing/pull_data.py`
- [ ] Pre-adjust gold threshold in `alert_cron.json` for live trigger (see `demo_trigger.md`)
- [ ] Fallback dashboard HTML ready in case live coding runs long
- [ ] Coffee ☕

---

## Presentation Rundown (60 minutes)

### 🟢 Opening — "The Problem" (5 min)

**[0:00–5:00]**

**What to say:**
> "Every morning, same routine. Check Bloomberg, scan headlines, pull up the dashboard, look at rates, check FX. It's 30 minutes of mechanical work before you even start thinking. What if that was done before you woke up?"

**Key points:**
- We all do the same repetitive information-gathering every day
- The tools exist now to automate this — not in theory, in practice
- Today I'll build three agents live, from scratch, that handle real portfolio monitoring tasks
- No servers, no DevOps, no engineering team needed

**Show:** Your phone with a real morning briefing delivered by the agent (if you have one running already).

---

### 🟡 Quick Context — "What is OpenClaw?" (3 min)

**[5:00–8:00]**

**What to say:**
> "OpenClaw is an open-source framework that lets you run AI agents on your machine. Think of it as cron jobs, but instead of running scripts, you're running an AI that can browse the web, write code, and send you messages. It talks to Claude — same model as ChatGPT but from Anthropic — and it can use tools: search the web, fetch data, run Python, push to GitHub."

**Key points:**
- Runs locally on your laptop or a small server
- Connects to Telegram, Discord, Teams — wherever you want alerts
- Cron jobs = scheduled AI tasks (check price, write briefing, etc.)
- The AI has tools: web search, web fetch, code execution, file I/O
- Cost: ~$0.05–0.30 per agent run depending on complexity

**Don't go deep on architecture. Keep it practical.**

---

### 🔵 Demo 1 — Price Alert Bot (12 min)

**[8:00–20:00]**

📂 **Files:** `agent1-price-alert/`

**What to say:**
> "Let's start simple. A price alert. Gold crosses above a threshold, I get a Telegram message. If nothing happened, silence — no spam."

**Live steps:**
1. Open `alert_cron.json` — walk through each field (2 min)
2. Explain the prompt: "Check gold price, alert if above X, otherwise NO_REPLY" (2 min)
3. Create the cron job in OpenClaw UI or CLI (2 min)
4. **The trick:** Set threshold to something gold already crossed (e.g., $2000) so it triggers immediately
5. Wait for Telegram notification to arrive (should take <60 sec)
6. Show the alert on your phone/Telegram tab
7. Reset threshold to real value, explain "now it runs every 15 min silently"

**Talking points while waiting:**
- "NO_REPLY is the key pattern — the agent stays silent unless there's something to report"
- "This replaces a Bloomberg alert. Zero cost, fully customizable prompt."
- "You could do this for any asset, any condition — BRL breaks 5.50, VIX spikes above 25, 10Y crosses 5%"

**See:** `agent1-price-alert/setup.md` for detailed walkthrough.

---

### 🔵 Demo 2 — Morning Macro Briefing (20 min)

**[20:00–40:00]**

📂 **Files:** `agent2-briefing/`

**What to say:**
> "Now let's build something more useful. Every morning at 7 AM, the agent pulls market data — indices, rates, FX, commodities — writes a two-paragraph macro summary, and sends it to my phone. Optional: it reads it to me in audio."

**Live steps:**
1. Open `pull_data.py` — walk through the code (3 min)
   - "Pure Python, no API keys, no pip install. It scrapes Yahoo Finance."
   - Run it live: `python3 pull_data.py` — show the output
2. Open `briefing_cron.json` — walk through the config (3 min)
   - "The cron runs at 7 AM São Paulo time"
   - "First it executes the Python script, then the AI summarizes it"
3. Create the cron job (2 min)
4. Trigger it manually to show the result (2 min)
5. Show the briefing arriving in Telegram (2 min)
6. **Bonus round — Audio** (5 min)
   - Open `add_audio.md`
   - "Three lines of curl to ElevenLabs, and now your briefing talks to you"
   - Play an audio sample if you have one pre-generated
   - Show `generate_audio_simple.py` briefly

**Talking points:**
- "The AI isn't just formatting data — it's writing analysis. It sees that gold is up 2% while DXY is down and connects the dots."
- "You control the prompt. Want it focused on rates? Change one line. Want it in Portuguese? Say so."
- "This costs about $0.15 per run. $4.50/month for a daily AI macro analyst."

**See:** `agent2-briefing/setup.md` for detailed walkthrough.

---

### 🔵 Demo 3 — Live Dashboard (15 min)

**[40:00–55:00]**

📂 **Files:** `agent3-dashboard/`

**What to say:**
> "Final demo. I'm going to ask Claude Code to build me an interactive market dashboard — dark theme, charts, responsive — and deploy it to GitHub Pages. From prompt to production in under 10 minutes."

**Live steps:**
1. Show `prompt.md` — the exact prompt you'll give Claude Code (2 min)
2. Open Claude Code in terminal (or show the OpenClaw coding agent)
3. Paste the prompt, let it run (5-8 min)
   - While it builds, narrate what it's doing: "It's creating the HTML... adding Chart.js... styling..."
   - Point out it's writing real code, not a template
4. Open the result in browser — show the dashboard
5. Push to GitHub Pages (or show it deployed)

**If time runs short or something breaks:**
- Open `fallback.html` in the browser — "Here's one I built earlier"
- Still impressive — it's a complete dark-theme dashboard with live-ish data

**Talking points:**
- "This is the 'wow' moment. An AI just built a production-ready dashboard."
- "You could schedule this to rebuild weekly with fresh data."
- "The point isn't the dashboard — it's that you can describe what you want and get working software."

**See:** `agent3-dashboard/setup.md` for detailed walkthrough.

---

### 🟢 Wrap-Up — "What Else?" (5 min)

**[55:00–60:00]**

**What to say:**
> "We built three agents in under an hour. A price alert, a macro briefing, and a dashboard. All running from my laptop, all delivering to Telegram, total cost under $5/month. But this is just the starting point."

**Ideas to plant:**
- **Risk monitoring:** "Agent that checks your portfolio VaR daily and flags when it spikes"
- **Earnings calendar:** "Weekly digest of upcoming earnings for your holdings"
- **Regulatory watch:** "Scan SEC filings or central bank releases, summarize what matters"
- **Client reporting:** "Auto-generate monthly performance summaries with charts"
- **News filter:** "Monitor 50 sources, alert only on topics relevant to your book"

**Close with:**
> "The barrier to building these tools used to be an engineering team and six months. Now it's an afternoon and curiosity. The question isn't whether AI can help with portfolio management — it's which part of your workflow you want to automate first."

**Hand out:** `cheatsheet.md` (print or share digitally)

---

## Timing Buffer

You have ~5 min of buffer built in. If Demo 3 runs long (live coding is unpredictable), skip the GitHub Pages deploy and use the fallback HTML. The key demos are 1 and 2 — Demo 3 is the "wow" closer but less critical.

## If Everything Breaks

1. **OpenClaw won't connect:** Show the cron JSON files and explain the structure. "The config is simple — here's what it looks like."
2. **Python script fails:** Have the output pre-saved in a text file. Show that instead.
3. **Telegram not delivering:** Show the OpenClaw logs. "You can see it ran successfully — delivery is just a config issue."
4. **Dashboard build fails:** Open `fallback.html`. Always works.

The demo is about the concept and the workflow, not about live debugging. If something breaks, acknowledge it, show the fallback, and keep moving.
