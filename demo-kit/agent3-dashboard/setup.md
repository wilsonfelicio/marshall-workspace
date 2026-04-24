# Agent 3: Live Dashboard — Build Guide

## What You're Building

A market snapshot dashboard — dark theme, interactive charts, mobile-responsive — built entirely by Claude Code from a single prompt. Deployed to GitHub Pages.

**Time to build live:** ~8-10 minutes (have fallback.html ready)

---

## Step-by-Step

### Step 1: Set the Stage (2 min)

> "We've built a silent monitor and a daily briefing. Now let's see how far we can push this. I'm going to give Claude Code a single prompt and ask it to build me an interactive market dashboard. Dark theme, charts, the works. From zero to deployed."

### Step 2: Show the Prompt (1 min)

Open `prompt.md` and read through it briefly:

> "This is the entire spec. One prompt. I'm asking for a specific layout, specific assets, Chart.js for the charts, dark theme, responsive design. Let's see what it builds."

### Step 3: Run Claude Code (5-8 min)

**Option A — Via terminal (Claude Code CLI):**
```bash
claude --print --permission-mode bypassPermissions < prompt.md
```

**Option B — Via OpenClaw coding agent:**
Paste the prompt content and let the agent build it.

**While it builds, narrate:**
- "It's setting up the HTML structure..."
- "Now it's adding Chart.js from CDN..."
- "It's styling with CSS — see the dark theme variables..."
- "Adding the data cards for each asset..."
- "Building the chart configurations..."

### Step 4: Show the Result (2 min)

Open the generated HTML file in the browser:
```bash
open index.html
```

**Point out:**
- Dark theme looks professional
- Cards with price + change for each asset
- Charts showing recent trends
- Responsive — resize the window to show mobile layout
- "An AI built this. From one prompt. In under 10 minutes."

### Step 5: Deploy (1 min, optional)

If time permits:
```bash
cd dashboard-repo
git init && git add . && git commit -m "Market dashboard"
gh repo create market-dashboard --public --source=. --push
# Enable GitHub Pages in settings
```

> "Now it's live. Anyone with the link can see it. You could schedule a rebuild weekly with fresh data."

---

## If Things Go Wrong

**Claude Code takes too long:**
- After 5 minutes, say: "Live coding is unpredictable — let me show you one I prepared earlier"
- Open `fallback.html` in the browser
- "Same concept, same result. The AI built this yesterday."

**Output has bugs:**
- Minor styling issues? Ignore them, focus on the concept
- Page doesn't load? Switch to fallback
- Charts broken? "Chart.js can be finicky — the fix is one line, but let me show the working version"

**The fallback is your safety net. Use it without hesitation.**

---

## What the Audience Should Take Away

1. **AI writes production code.** Not pseudocode, not prototypes — working software.
2. **From prompt to product.** The entire spec was natural language.
3. **This scales.** Client reports, performance dashboards, risk summaries — same workflow.
4. **You don't need to be a developer.** You need to describe what you want clearly.
