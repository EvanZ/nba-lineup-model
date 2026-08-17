#!/usr/bin/env bash
set -euo pipefail

# Publish the compact production bundle required by the NBA GESTALT API.
# Historical raw/processed data is intentionally excluded; it belongs in S3 archive storage.

BUCKET="${GESTALT_S3_BUCKET:-nba-gestalt-data}"
MODEL_ARTIFACT="forward_hpm_x3_linear_ridge_without_uncertainty"
DISPLAY_SEASON="2025-26"
RUN_ID="${GESTALT_RUN_ID:-forward-hpm-x3-linear-ridge-without-uncertainty-2025-26-20260816T142321Z-78268dad}"
RELEASE_ID="${GESTALT_RELEASE_ID:-${RUN_ID}}"
STAGING_DIR="${GESTALT_STAGING_DIR:-/tmp/nba-gestalt-${RELEASE_ID}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SEASON_ROOT="artifacts/models/${MODEL_ARTIFACT}/${DISPLAY_SEASON}"
MODEL_ROOT="${MODEL_SEASON_ROOT}/${RUN_ID}"
LINEUP_CACHE="artifacts/web/lineup_rankings/${MODEL_ARTIFACT}/${RUN_ID}"
TEAM_SPLITS_CACHE="artifacts/web/player_team_splits/${MODEL_ARTIFACT}"
RESPONSE_CACHE="artifacts/web/response_curve_cache/${MODEL_ARTIFACT}"
EXPOSURE_COHORT_CACHE="artifacts/web/exposure_cohorts/${MODEL_ARTIFACT}/${RUN_ID}.parquet"
HISTORICAL_PROFILE_CACHE="artifacts/web/historical_profiles/${MODEL_ARTIFACT}/${RUN_ID}.parquet"
PANEL="data/analytical/player_season_panel"

required_paths=(
  "$MODEL_ROOT"
  "$MODEL_SEASON_ROOT/latest.json"
  "$LINEUP_CACHE"
  "$TEAM_SPLITS_CACHE"
  "$RESPONSE_CACHE"
  "$EXPOSURE_COHORT_CACHE"
  "$HISTORICAL_PROFILE_CACHE"
  "$PANEL"
)

cd "$ROOT_DIR"
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'Required runtime path is missing: %s\n' "$path" >&2
    exit 1
  fi
done

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
mkdir -p "$STAGING_DIR/deploy"
mkdir -p "$STAGING_DIR/$MODEL_SEASON_ROOT"
mkdir -p "$(dirname "$STAGING_DIR/$EXPOSURE_COHORT_CACHE")"
mkdir -p "$(dirname "$STAGING_DIR/$HISTORICAL_PROFILE_CACHE")"

rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' src/ "$STAGING_DIR/src/"
rsync -a README.md pyproject.toml uv.lock "$STAGING_DIR/"
rsync -a deploy/web-requirements.txt "$STAGING_DIR/deploy/"
rsync -a "$MODEL_SEASON_ROOT/latest.json" "$STAGING_DIR/$MODEL_SEASON_ROOT/"
rsync -a "$MODEL_ROOT/" "$STAGING_DIR/$MODEL_ROOT/"
rsync -a "$LINEUP_CACHE/" "$STAGING_DIR/$LINEUP_CACHE/"
rsync -a "$TEAM_SPLITS_CACHE/" "$STAGING_DIR/$TEAM_SPLITS_CACHE/"
rsync -a "$RESPONSE_CACHE/" "$STAGING_DIR/$RESPONSE_CACHE/"
rsync -a "$EXPOSURE_COHORT_CACHE" "$STAGING_DIR/$EXPOSURE_COHORT_CACHE"
rsync -a "$HISTORICAL_PROFILE_CACHE" "$STAGING_DIR/$HISTORICAL_PROFILE_CACHE"
rsync -a "$PANEL/" "$STAGING_DIR/$PANEL/"

cat > "$STAGING_DIR/release.json" <<EOF
{
  "release_id": "${RELEASE_ID}",
  "git_sha": "$(git rev-parse HEAD)",
  "model_artifact": "${MODEL_ARTIFACT}",
  "display_season": "${DISPLAY_SEASON}",
  "run_id": "${RUN_ID}"
}
EOF

aws s3 sync --delete "$STAGING_DIR/" "s3://${BUCKET}/nba-gestalt/releases/${RELEASE_ID}/"
printf 'Published s3://%s/nba-gestalt/releases/%s/\n' "$BUCKET" "$RELEASE_ID"
