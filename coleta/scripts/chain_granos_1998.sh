#!/usr/bin/env bash
# Wait for the running granos backfill (PID passed as $1) to exit, then extend the
# same module back to 1998. The second run reuses the manifest, so every 2011+
# product-week already recorded with a terminal status is skipped and only
# 1998-2010 is fetched. Waiting on the PID rather than pgrep avoids matching this
# wrapper's own command line, which contains the same substring.
set -u
PID="$1"
cd /root/coleta
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) granos 2011+ acabo (pid $PID); extiendo a 1998" >> logs/granos_cloud.out
exec python3 run.py backfill --module granos --granos-start-year 1998 --workers 2 \
     >> logs/granos_cloud.out 2>&1
