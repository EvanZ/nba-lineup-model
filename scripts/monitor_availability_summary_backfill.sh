#!/usr/bin/env bash
set -euo pipefail

pid=${1:?usage: monitor_availability_summary_backfill.sh <pid>}
cache_dir=${2:-data/raw/stats/boxscoresummaryv2}
interval_seconds=${3:-15}

while kill -0 "$pid" 2>/dev/null; do
  cached=$(find "$cache_dir" -type f -name '*.json' ! -name '*.meta.json' | wc -l | tr -d ' ')
  elapsed=$(ps -p "$pid" -o etime= | tr -d ' ')
  printf '%s pid=%s cached_summaries=%s elapsed=%s\n' \
    "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$pid" "$cached" "$elapsed"
  sleep "$interval_seconds"
done

printf '%s pid=%s no-longer-running\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$pid"
