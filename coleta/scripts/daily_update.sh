#!/usr/bin/env bash
# Wrapper for the recurrent collection. This is what the scheduler calls.
#
# It resolves its own project directory, activates the venv if there is one,
# and logs a dated line so a silent failure is visible in logs/cron.log.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

mkdir -p logs
LOG="logs/cron.log"

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="$(command -v python3)"
fi

echo "=== $(date '+%Y-%m-%d %H:%M:%S %z') starting update ===" >> "$LOG"

"$PY" run.py update >> "$LOG" 2>&1
RC=$?

if [ $RC -eq 0 ]; then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %z') update OK ===" >> "$LOG"
else
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %z') update FAILED rc=$RC ===" >> "$LOG"
fi

exit $RC
