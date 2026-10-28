#!/usr/bin/env bash
set -euo pipefail

API_ROOT="${1:?API root is required, for example http://127.0.0.1:8011}"
FORECAST_PAYLOAD="$(mktemp)"
PERFORMANCE_PAYLOAD="$(mktemp)"
trap 'rm -f "$FORECAST_PAYLOAD" "$PERFORMANCE_PAYLOAD"' EXIT

curl --fail --silent --show-error \
  "$API_ROOT/api/win-projections" >"$FORECAST_PAYLOAD"

scheduled_count="$(jq -er '.win_projection.scheduled_game_count' "$FORECAST_PAYLOAD")"
calendar_count="$(jq -er '.win_projection.scheduled_games | length' "$FORECAST_PAYLOAD")"
remaining_count="$(jq -er '.win_projection.remaining_scheduled_game_count // .win_projection.scheduled_game_count' "$FORECAST_PAYLOAD")"

if (( scheduled_count < 1 )); then
  printf 'Serving validation failed: scheduled_game_count is %s\n' "$scheduled_count" >&2
  exit 1
fi
if (( calendar_count != scheduled_count )); then
  printf 'Serving validation failed: season calendar has %s of %s games\n' \
    "$calendar_count" "$scheduled_count" >&2
  exit 1
fi
if (( remaining_count > scheduled_count )); then
  printf 'Serving validation failed: remaining games exceed season games\n' >&2
  exit 1
fi

curl --fail --silent --show-error \
  "$API_ROOT/api/schedule-performance" >"$PERFORMANCE_PAYLOAD"
completed_count="$(jq -er '[.windows[] | select(.key == "season")][0].games' "$PERFORMANCE_PAYLOAD")"
if (( completed_count < 1 )); then
  printf 'Serving validation failed: historical performance contains no completed games\n' >&2
  exit 1
fi

printf 'Validated serving contract: %s/%s calendar games, %s completed audits\n' \
  "$calendar_count" "$scheduled_count" "$completed_count"
