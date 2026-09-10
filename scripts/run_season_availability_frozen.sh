#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
exec uv run python -c \
  "from nba_lineup_model.rotation.season_availability import run_season_availability_frozen_evaluation; print(run_season_availability_frozen_evaluation().run_dir)" \
  > artifacts/logs/season-availability-v01-frozen.log \
  2>&1
