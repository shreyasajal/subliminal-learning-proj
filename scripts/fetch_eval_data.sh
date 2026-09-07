#!/usr/bin/env bash
# Pull every run's metrics + eval records off the volume into local caches.
# Idempotent: already-downloaded files are skipped, so re-running after new
# seeds land only fetches the new ones.
#
#   ./scripts/fetch_eval_data.sh [records_dir] [metrics_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
REC="${1:-/tmp/rec}"; MET="${2:-/tmp/r2}"
MODAL=".venv/bin/modal"
mkdir -p "$REC" "$MET"

runs=$(timeout 180 $MODAL volume ls subliminal-vol outputs 2>/dev/null | sed 's|outputs/||')
[ -z "$runs" ] && { echo "could not list volume"; exit 1; }

n_new=0
for r in $runs; do
  [ -f "$MET/$r.json" ] || \
    timeout 60 $MODAL volume get subliminal-vol "outputs/$r/metrics.json" "$MET/$r.json" >/dev/null 2>&1 \
    && [ -f "$MET/$r.json" ] || true
  for c in qwen empty chatgpt; do
    if [ ! -f "$REC/${r}__${c}.json" ]; then
      if timeout 90 $MODAL volume get subliminal-vol "outputs/$r/eval/records_$c.json" \
           "$REC/${r}__${c}.json" >/dev/null 2>&1; then n_new=$((n_new+1)); fi
    fi
  done
done
echo "metrics cached : $(ls "$MET"/*.json 2>/dev/null | wc -l)"
echo "record files   : $(ls "$REC"/*__*.json 2>/dev/null | wc -l)  (+$n_new new)"
echo
echo "evaluated runs:"
ls "$REC"/*__qwen.json 2>/dev/null | sed "s|$REC/||;s|__qwen.json||" | sort | sed 's/^/  /'
