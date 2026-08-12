# Daily publication — setup

One-time, about ten minutes. After this the workbook refreshes itself every weekday and the
download URL never changes.

## The URL you will use

```
https://github.com/wilsonfelicio/marshall-workspace/releases/latest/download/precios_mayoreo_diario.xlsx
```

Public repo, so no token. In Excel: **Data → From Web**, paste that, and *Refresh All* pulls
the current file. Same URL works in `curl`, pandas (`pd.read_excel(url)`), or a browser.

## 1. Push the code

Done: the project lives in the `coleta/` subfolder of `marshall-workspace`, and the workflow
sits at the repo root in `.github/workflows/daily.yml` because GitHub reads workflows only
from there. Every step runs with `working-directory: coleta`. `MOVE_TO_SUBFOLDER.md` has the
copy-in procedure if it ever has to be redone. `data/` stays out of git — the store travels
as a release asset.

## 2. Seed the store

The workflow restores the store from a release asset rather than rebuilding 28 years of
history every day. Upload it once, from the machine that already has the data. Build the
tarball from `~/Downloads/coleta`, so its paths start with `data/` and not `coleta/data/` —
the runner untars inside `coleta/`:

```bash
cd ~/Downloads/coleta
tar czf /tmp/store.tar.gz data/raw data/catalog data/inpc data/manifest.csv
tar tzf /tmp/store.tar.gz | head -3        # must start with data/
cd ~/Downloads/marshall-workspace
gh release create data --title "Latest data" \
  --notes "Rolling data release. Assets are replaced on each run." \
  /tmp/store.tar.gz
```

That tarball is about 65MB compressed, well inside the 2GB per-asset limit. If you do not
have the `gh` CLI, `brew install gh && gh auth login`.

## 3. Check the permission

**Settings → Actions → General → Workflow permissions** must be *Read and write*. Without it
the publish step cannot replace the release assets.

## 4. Run it once by hand

**Actions → precios diarios → Run workflow**, leaving *bootstrap* unchecked. Watch it go
green, then confirm the URL above serves a file with yesterday's date in it.

If you ever lose the store entirely, run it once with *bootstrap* checked: that rebuilds
from SNIIM starting 2024 instead of restoring, which takes a few hours. Full history back to
1998 is a `--start-year 1998` edit away, but there is no reason to do that on a runner when
the tarball exists.

## What runs, and when

`0 19 * * 1-5` — 19:00 UTC, 13:00 Mexico City, Monday to Friday. SNIIM publishes
*"Diaria (de lunes a viernes)"* and states no hour; measured on 12 Aug 2026 the day was
absent at 04:07 Mexico City and complete by 12:09. GitHub delays scheduled runs under load,
sometimes by an hour, which costs nothing here: `update` re-fetches a trailing 14-day window,
so being late loses nothing and being early would.

Weekends are skipped because SNIIM does not quote. Granos (Frijol, Chile seco) only publish
on Wednesdays, and the gate knows that.

## The publish gate

`scripts/check_freshness.py` runs before the workbook is built, and a failure stops the job
**without** replacing the published file — so a bad day leaves yesterday's good file in
place rather than overwriting it. Three checks:

| Check | Fails when |
|---|---|
| Recency | the newest quote is more than 4 business days old |
| Coverage | more than 4 daily generics hold under 50% of their usual market count |
| Continuity | the store has fewer rows than on the previous run |

Coverage is the one that matters. The failure mode it exists for is real: Manzana on
2025-05-05 carried 2 markets out of a usual 78, which is not a national price but looks like
one. Weekly generics are checked against their own last quote day instead, since they are
legitimately absent from a Tuesday.

`run.py update` exiting non-zero does **not** stop publication — that only means a few
products need a retry, and the trailing window handles them tomorrow. The data decides, not
the exit code.

## Cost

Public repo, so Actions minutes are unlimited. A run is roughly 15-30 minutes: the collector
holds a 1.5s floor between requests to keep SNIIM's IIS 6.0 server from returning 503, and
there are 222 fruit and vegetable products plus 12 granos.

## If it breaks

- **Job fails at "Restore the store"** — the `data` release or its `store.tar.gz` is missing. Redo step 2.
- **Job fails at "Publish gate"** — read the step log; it names which check failed and why. Usually SNIIM had a short day, and the next run fixes it.
- **Job fails at "Publish"** — workflow permissions are read-only. See step 3.
- **URL 404s** — the release exists but has no asset by that name; check the release page.
