# 🔶 OpenClaw Cron Jobs — Quick Reference

## Schedule Syntax

| Format | Example | Meaning |
|--------|---------|---------|
| `at` | `"at 2026-04-15T09:00"` | One-shot: runs once at exact time (ISO 8601) |
| `every` | `"every 15m"` | Recurring: runs every N units (s/m/h/d) |
| `cron` | `"cron 0 10 * * 1-5"` | Standard cron: min hour dom month dow |

### Cron Expression Cheat Sheet

```
┌───────────── minute (0-59)
│ ┌─────────── hour (0-23, UTC)
│ │ ┌───────── day of month (1-31)
│ │ │ ┌─────── month (1-12)
│ │ │ │ ┌───── day of week (0-7, 0=Sun, 1=Mon... 7=Sun)
│ │ │ │ │
* * * * *
```

| Schedule | Expression |
|----------|-----------|
| Every weekday at 7 AM BRT (10 UTC) | `cron 0 10 * * 1-5` |
| Every hour during market hours (9-16 ET) | `cron 0 13-20 * * 1-5` |
| Every Monday at 8 AM BRT | `cron 0 11 * * 1` |
| Every 30 minutes | `every 30m` |
| First of every month at 9 AM BRT | `cron 0 12 1 * *` |

---

## Cron Job JSON Fields

```json
{
  "name": "Human-readable label",
  "schedule": "every 15m",
  "payload": "The prompt — what the AI should do",
  "delivery": {
    "target": "telegram:direct:YOUR_CHAT_ID"
  },
  "sessionTarget": "telegram:direct:YOUR_CHAT_ID",
  "model": "anthropic/claude-sonnet-4-20250514",
  "thinking": false
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Display name in UI and logs |
| `schedule` | Yes | When to run (at/every/cron) |
| `payload` | Yes | The prompt — plain text instructions for the AI |
| `delivery.target` | Yes | Where to send results (channel:type:id) |
| `sessionTarget` | Yes | Session context for the run |
| `model` | No | Override default model (e.g., use Sonnet for cheap jobs) |
| `thinking` | No | Enable extended thinking (true/false) |

---

## Common Patterns

### 1. Silent Alert (Check & Alert Only If Triggered)

**Key trick:** Include `NO_REPLY` in the payload.

```json
{
  "payload": "Check [condition]. If triggered, send alert. If NOT triggered, respond with exactly: NO_REPLY"
}
```

The agent stays silent unless there's something to report. No spam.

### 2. Scheduled Briefing (Run Script → Analyze → Deliver)

```json
{
  "payload": "Run `python3 /path/to/script.py`. Read the output. Write a 2-paragraph summary. Keep it under 200 words."
}
```

The AI executes the script, reads the data, and writes analysis.

### 3. One-Shot Reminder

```json
{
  "schedule": "at 2026-04-15T14:00",
  "payload": "Remind me: FOMC meeting starts in 30 minutes. Check positioning."
}
```

Fires once, delivers the message, done.

### 4. Weekly Digest

```json
{
  "schedule": "cron 0 11 * * 5",
  "payload": "Search the web for this week's key macro events. Summarize the top 5 market-moving headlines and what they mean for rates and equities."
}
```

---

## Delivery Targets

| Target | Format |
|--------|--------|
| Telegram (direct) | `telegram:direct:CHAT_ID` |
| Telegram (group) | `telegram:group:GROUP_ID` |
| Discord (channel) | `discord:channel:CHANNEL_ID` |

**Finding your Telegram chat ID:** Send any message to the bot, then check OpenClaw logs or use `@userinfobot` on Telegram.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Cron runs but no delivery | Check `delivery.target` matches a connected channel |
| Agent responds but says "I can't do that" | Ensure tools (web_search, exec) are enabled for cron sessions |
| Script not found | Use absolute paths in `payload` — cron may run from a different working directory |
| Timing is off | Cron uses **UTC**. São Paulo (BRT) = UTC-3. 7 AM BRT = 10:00 UTC |
| Too many alerts | Add `NO_REPLY` pattern for non-triggering conditions |
| High cost | Use `claude-sonnet` instead of `claude-opus` for simple monitoring jobs |
| Cron doesn't fire | Check `openclaw gateway status` — gateway must be running |
| Timeout on long scripts | Keep scripts under 30 seconds; for longer tasks, consider splitting |

---

## Cost Estimates

| Agent Type | Model | Est. Cost/Run | Monthly (Daily) |
|------------|-------|---------------|-----------------|
| Price alert (simple check) | Sonnet | ~$0.02 | $0.60 |
| Macro briefing (script + analysis) | Sonnet | ~$0.15 | $4.50 |
| Dashboard build (code gen) | Opus | ~$0.50 | one-off |
| Audio TTS (ElevenLabs) | — | ~$0.05 | $1.50 |

**Total for daily briefing + alerts:** ~$5-7/month

---

*Built with OpenClaw + Claude • Questions? Ask the agent.*
