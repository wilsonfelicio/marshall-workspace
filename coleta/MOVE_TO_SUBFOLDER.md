# Putting coleta into the marshall-workspace repo as a subfolder

Your `~/Downloads/coleta` currently has its own git repo whose history is unrelated to the
remote. Rather than fight that with a rebase, clone the real repo and copy the project in.
Cleaner, and it sidesteps the `fetch first` rejection entirely.

## 1. Clone the repo

```zsh
cd ~/Downloads
git clone https://github.com/wilsonfelicio/marshall-workspace.git
cd marshall-workspace
ls          # whatever else you already keep in here
```

## 2. Copy the project in, without the data or the tarballs

`rsync` with `--exclude` is safer than `cp -R` here: the store is 650MB across data/raw and
data/curated, and none of it belongs in git.

```zsh
mkdir -p coleta
rsync -av --exclude '.git/' \
          --exclude 'data/' \
          --exclude 'charts/' \
          --exclude 'logs/' \
          --exclude '_transfer/' \
          --exclude '_incoming/' \
          --exclude '_to_delete/' \
          --exclude '__pycache__/' \
          --exclude '.pytest_cache/' \
          --exclude '.DS_Store' \
          --exclude '*.tar.gz' \
          --exclude '*.xlsx' \
          --exclude '*.pdf' \
          ~/Downloads/coleta/ coleta/
```

Note the trailing slash on `~/Downloads/coleta/` — without it rsync creates
`coleta/coleta/`.

## 3. The workflow goes at the REPO ROOT, not in coleta/

GitHub only reads workflows from the root `.github/workflows/`. The file I gave you already
has `working-directory: coleta` on every step, so it finds the project.

```zsh
mkdir -p .github/workflows
# put the new daily.yml here (download it from the chat)
```

Do **not** leave a copy at `coleta/.github/workflows/` — it would be ignored, and a second
stale copy of a CI file is a trap for whoever reads this next.

```zsh
rm -f coleta/.github/workflows/daily.yml
rmdir -p coleta/.github/workflows 2>/dev/null
```

## 4. Add a .gitignore for the subfolder

The one inside `coleta/` came across with the rsync and still works — git honours nested
.gitignore files, and its paths are relative to `coleta/`. Nothing to change.

## 5. Commit and push

```zsh
git add -A
git status --short | head -40
```

That listing should show `coleta/` source files and `.github/workflows/daily.yml`, with no
`.parquet`, no `.tar.gz`, no `.xlsx`. Then:

```zsh
git commit -m "Add coleta: SNIIM wholesale collector, nowcasting, daily publication"
git push
```

No `--allow-unrelated-histories` needed — you are committing on top of the remote's own
history.

## 6. Seed the store, from inside coleta

The tarball's paths must be relative to `coleta/`, because the workflow untars with
`working-directory: coleta`. So build it from there:

```zsh
cd ~/Downloads/coleta          # the original, which still has the data
tar czf /tmp/store.tar.gz data/raw data/catalog data/inpc data/manifest.csv
cd ~/Downloads/marshall-workspace
gh release create data --title "Latest data" \
  --notes "Rolling data release. Assets are replaced on each run." \
  /tmp/store.tar.gz
```

Check the paths inside are right before uploading — they must start with `data/`, not
`coleta/data/`:

```zsh
tar tzf /tmp/store.tar.gz | head -3
```

## 7. Settings and first run

- **Settings → Actions → General → Workflow permissions → Read and write.**
- **Actions → precios diarios → Run workflow**, bootstrap unchecked.

## Afterwards

`~/Downloads/coleta` keeps the data and stays your working copy — the daily runs happen on
GitHub, not there. Its `.git` directory is now redundant and confusing; delete it when you
are satisfied the push worked:

```zsh
rm -rf ~/Downloads/coleta/.git
```

Keep the folder itself. It holds `data/`, which is 650MB and is not in git by design.
