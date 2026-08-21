#!/usr/bin/env bash
set -euo pipefail

# Publish the compact production bundle required by the NBA GESTALT API.
# Historical raw/processed data is intentionally excluded; it belongs in S3 archive storage.

BUCKET="${GESTALT_S3_BUCKET:-nba-gestalt-data}"
MODEL_ARTIFACT="forward_nail_rapm_v12_gap_returner_priors"
DISPLAY_SEASON="2025-26"
RUN_ID="${GESTALT_RUN_ID:-forward-nail-rapm-v12-gap-returner-priors-2025-26-20260821T140232Z-da227de3}"
RELEASE_ID="${GESTALT_RELEASE_ID:-${RUN_ID}}"
STAGING_DIR="${GESTALT_STAGING_DIR:-/tmp/nba-gestalt-${RELEASE_ID}}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SEASON_ROOT="artifacts/models/${MODEL_ARTIFACT}/${DISPLAY_SEASON}"
MODEL_ROOT="${MODEL_SEASON_ROOT}/${RUN_ID}"
LINEUP_CACHE="artifacts/web/lineup_rankings/${MODEL_ARTIFACT}/${RUN_ID}"
TEAM_SPLITS_CACHE="artifacts/web/player_team_splits/${MODEL_ARTIFACT}/${RUN_ID}.parquet"
PLAYER_CONTEXT_CACHE="artifacts/web/response_curve_cache/${MODEL_ARTIFACT}/${RUN_ID}-player-context.parquet"
EXPOSURE_COHORT_CACHE="artifacts/web/exposure_cohorts/${MODEL_ARTIFACT}/${RUN_ID}.parquet"
HISTORICAL_PROFILE_CACHE="artifacts/web/historical_profiles/${MODEL_ARTIFACT}/${RUN_ID}.parquet"
PRESEASON_CACHE="artifacts/web/preseason_rankings/${MODEL_ARTIFACT}/${RUN_ID}"
RELEASE_MANIFEST="artifacts/web/releases/${MODEL_ARTIFACT}/${RUN_ID}/bundle_manifest.json"
PANEL="data/analytical/player_season_panel"

cd "$ROOT_DIR"
uv run nba-validate-gestalt-release --season "$DISPLAY_SEASON" --run-id "$RUN_ID"

required_paths=(
  "$MODEL_ROOT"
  "$MODEL_SEASON_ROOT/latest.json"
  "$LINEUP_CACHE"
  "$TEAM_SPLITS_CACHE"
  "$PLAYER_CONTEXT_CACHE"
  "$EXPOSURE_COHORT_CACHE"
  "$HISTORICAL_PROFILE_CACHE"
  "$PRESEASON_CACHE"
  "$RELEASE_MANIFEST"
  "$PANEL"
)

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
mkdir -p "$(dirname "$STAGING_DIR/$TEAM_SPLITS_CACHE")"
mkdir -p "$(dirname "$STAGING_DIR/$PLAYER_CONTEXT_CACHE")"
mkdir -p "$(dirname "$STAGING_DIR/$EXPOSURE_COHORT_CACHE")"
mkdir -p "$(dirname "$STAGING_DIR/$HISTORICAL_PROFILE_CACHE")"
mkdir -p "$(dirname "$STAGING_DIR/$PRESEASON_CACHE")"
mkdir -p "$(dirname "$STAGING_DIR/$RELEASE_MANIFEST")"

rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' src/ "$STAGING_DIR/src/"
rsync -a README.md pyproject.toml uv.lock "$STAGING_DIR/"
rsync -a deploy/web-requirements.txt "$STAGING_DIR/deploy/"
printf '{\n  "run_id": "%s"\n}\n' "$RUN_ID" > "$STAGING_DIR/$MODEL_SEASON_ROOT/latest.json"
rsync -a "$MODEL_ROOT/" "$STAGING_DIR/$MODEL_ROOT/"
rsync -a "$LINEUP_CACHE/" "$STAGING_DIR/$LINEUP_CACHE/"
rsync -a "$TEAM_SPLITS_CACHE" "$STAGING_DIR/$TEAM_SPLITS_CACHE"
rsync -a "$PLAYER_CONTEXT_CACHE" "$STAGING_DIR/$PLAYER_CONTEXT_CACHE"
rsync -a "$EXPOSURE_COHORT_CACHE" "$STAGING_DIR/$EXPOSURE_COHORT_CACHE"
rsync -a "$HISTORICAL_PROFILE_CACHE" "$STAGING_DIR/$HISTORICAL_PROFILE_CACHE"
rsync -a "$PRESEASON_CACHE/" "$STAGING_DIR/$PRESEASON_CACHE/"
rsync -a "$RELEASE_MANIFEST" "$STAGING_DIR/$RELEASE_MANIFEST"
rsync -a "$PANEL/" "$STAGING_DIR/$PANEL/"

cat > "$STAGING_DIR/release.json" <<EOF
{
  "release_id": "${RELEASE_ID}",
  "git_sha": "$(git rev-parse HEAD)",
  "model_display_name": "NAIL-RAPM v1.2",
  "model_artifact": "${MODEL_ARTIFACT}",
  "display_season": "${DISPLAY_SEASON}",
  "run_id": "${RUN_ID}",
  "context_alpha": 10000,
  "profile_padding_contract": "medvedovsky_2020_stat_specific",
  "bundle_manifest": "${RELEASE_MANIFEST}"
}
EOF

aws s3 sync --delete "$STAGING_DIR/" "s3://${BUCKET}/nba-gestalt/releases/${RELEASE_ID}/"
printf 'Published s3://%s/nba-gestalt/releases/%s/\n' "$BUCKET" "$RELEASE_ID"
