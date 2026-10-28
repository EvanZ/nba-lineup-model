#!/usr/bin/env bash
set -euo pipefail

LEFT_API="${1:?First API root is required}"
RIGHT_API="${2:?Second API root is required}"
WORK_ROOT="$(mktemp -d)"
trap 'rm -rf "$WORK_ROOT"' EXIT

snapshot() {
  local api_root="$1"
  local output="$2"
  local health="$WORK_ROOT/health.json"
  local forecast="$WORK_ROOT/forecast.json"
  local performance="$WORK_ROOT/performance.json"

  curl --fail --silent --show-error "$api_root/api/health" >"$health"
  curl --fail --silent --show-error "$api_root/api/win-projections" >"$forecast"
  curl --fail --silent --show-error "$api_root/api/schedule-performance" >"$performance"

  jq -nS \
    --slurpfile health "$health" \
    --slurpfile forecast "$forecast" \
    --slurpfile performance "$performance" \
    '{
      season: $health[0].current_season_state.season,
      evidence_through_date: $health[0].current_season_state.evidence_through_date,
      evidence_game_count: $health[0].current_season_state.game_count,
      source_game_ids: ($health[0].current_season_state.source_game_ids | sort),
      forecast_as_of: $forecast[0].win_projection.as_of,
      catalog_game_count: $forecast[0].win_projection.scheduled_game_count,
      remaining_game_count: $forecast[0].win_projection.remaining_scheduled_game_count,
      completed_team_games: ([$forecast[0].win_projection.teams[].completed_games] | add),
      performance_as_of: $performance[0].as_of,
      audited_final_games: ([$performance[0].windows[] | select(.key == "season")][0].games)
    }' >"$output"
}

snapshot "$LEFT_API" "$WORK_ROOT/left.json"
snapshot "$RIGHT_API" "$WORK_ROOT/right.json"

if ! cmp --silent "$WORK_ROOT/left.json" "$WORK_ROOT/right.json"; then
  printf 'Dev/production data coverage mismatch:\n' >&2
  diff -u "$WORK_ROOT/right.json" "$WORK_ROOT/left.json" >&2 || true
  exit 1
fi

printf 'Validated identical dev/production dates and game coverage: '
jq -r '"evidence through \(.evidence_through_date), \(.evidence_game_count) modeled games, \(.audited_final_games) final audits, \(.catalog_game_count) catalog games"' "$WORK_ROOT/left.json"
