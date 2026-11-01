#!/usr/bin/env bash
set -euo pipefail

# Deploy one Git ref beside production without copying the data lake or mutating
# production releases/state. The dev runtime reuses production's installed
# Python dependencies and immutable model/data bundle read-only.

export PATH="$HOME/.local/bin:$PATH"
if ! command -v npm >/dev/null 2>&1; then
  for node_bin in "$HOME"/.nvm/versions/node/*/bin; do
    if [[ -x "$node_bin/node" && -x "$node_bin/npm" ]]; then
      export PATH="$node_bin:$PATH"
    fi
  done
fi

DEV_REF="${GESTALT_DEV_REF:?GESTALT_DEV_REF is required}"
SOURCE_REPO="${GESTALT_SOURCE_REPO:-$HOME/nba-lineup-model}"
PROD_APP_ROOT="${GESTALT_PROD_APP_ROOT:-/opt/nba-gestalt}"
DEV_APP_ROOT="${GESTALT_DEV_APP_ROOT:-/opt/nba-gestalt-dev}"
DEV_STATE_ROOT="${GESTALT_DEV_STATE_ROOT:-/var/lib/nba-gestalt-dev}"
DEV_WEB_ROOT="${GESTALT_DEV_WEB_ROOT:-/var/www/nba-gestalt-dev}"
PROD_SCHEDULE_ROOT="${GESTALT_PROD_SCHEDULE_ROOT:-/var/lib/nba-gestalt/schedule-serving/current}"
DEV_DAILY_DATA_ROOT="${GESTALT_DEV_DAILY_DATA_ROOT:-$DEV_STATE_ROOT/daily-data-nba-v2}"
DEV_DAILY_MODEL_ROOT="${GESTALT_DEV_DAILY_MODEL_ROOT:-$DEV_STATE_ROOT/daily-model-nba-v1}"
DEV_ROSTER_POINTER="${GESTALT_DEV_ROSTER_POINTER:-$DEV_STATE_ROOT/in-season-rosters/latest.json}"
DEV_OFFICIAL_SCHEDULE_ROOT="${GESTALT_DEV_OFFICIAL_SCHEDULE_ROOT:-$DEV_STATE_ROOT/official-schedule}"
DEV_SCHEDULE_ROOT="$DEV_OFFICIAL_SCHEDULE_ROOT/current"
FRONTEND_DIST="${GESTALT_DEV_FRONTEND_DIST:-}"

test -d "$SOURCE_REPO/.git"
if [[ "$DEV_REF" == origin/* ]]; then
  dev_branch="${DEV_REF#origin/}"
  git -C "$SOURCE_REPO" fetch --no-tags origin \
    "refs/heads/$dev_branch:refs/remotes/origin/$dev_branch"
fi

RELEASE_ID="${GESTALT_DEV_RELEASE_ID:-$(git -C "$SOURCE_REPO" rev-parse --short=12 "$DEV_REF")}"
RELEASE_ROOT="$DEV_APP_ROOT/releases/$RELEASE_ID"
PROD_RELEASE="$PROD_APP_ROOT/current"

test -x "$PROD_RELEASE/.venv/bin/python"
test -d "$PROD_RELEASE/artifacts"
test -d "$PROD_RELEASE/data"
if [[ ! -d "$PROD_SCHEDULE_ROOT/final" ]]; then
  PROD_SCHEDULE_ROOT=/var/lib/nba-gestalt/schedule
fi

sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_APP_ROOT/releases"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/catalog"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/current-season-serving"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/current-season-lineups"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_OFFICIAL_SCHEDULE_ROOT"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/site-releases"

test -f "$DEV_DAILY_DATA_ROOT/latest.json"
test -f "$DEV_DAILY_MODEL_ROOT/latest.json"
test -f "$DEV_ROSTER_POINTER"

DEV_PRESEASON_RANKINGS_PATH="$(jq -er '.artifacts.preseason_rankings' "$DEV_ROSTER_POINTER")"
DEV_PRESEASON_PROFILES_PATH="$(jq -er '.artifacts.preseason_profiles' "$DEV_ROSTER_POINTER")"
test -f "$DEV_PRESEASON_RANKINGS_PATH"
test -f "$DEV_PRESEASON_PROFILES_PATH"
printf 'GESTALT_PRESEASON_RANKINGS_PATH=%s\nGESTALT_PRESEASON_PROFILES_PATH=%s\n' \
  "$DEV_PRESEASON_RANKINGS_PATH" \
  "$DEV_PRESEASON_PROFILES_PATH" > "$DEV_STATE_ROOT/site-release.env"

if [[ ! -d "$RELEASE_ROOT" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$RELEASE_ROOT" "$DEV_REF"
fi

ln -sfn \
  "$SOURCE_REPO/apps/lineup-explorer/node_modules" \
  "$RELEASE_ROOT/apps/lineup-explorer/node_modules"

# Model and web loaders intentionally resolve the immutable artifacts/data
# bundle relative to the working directory. Run them from production's
# read-only release while importing the dev code through PYTHONPATH; this
# avoids copying the bundle or trying to replace Git-tracked placeholders in
# the dev worktree.
cd "$PROD_RELEASE"

aws s3 cp \
  s3://nba-gestalt-data/lake/catalog/games.parquet \
  "$DEV_STATE_ROOT/catalog/games.parquet.next" \
  --only-show-errors
mv \
  "$DEV_STATE_ROOT/catalog/games.parquet.next" \
  "$DEV_STATE_ROOT/catalog/games.parquet"

frozen_artifact_root="$PROD_SCHEDULE_ROOT"
if [[ -d "$DEV_SCHEDULE_ROOT/frozen" ]]; then
  frozen_artifact_root="$DEV_SCHEDULE_ROOT"
fi
PYTHONPATH="$RELEASE_ROOT/src" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.season.official_schedule_serving \
  --data-root "$DEV_DAILY_DATA_ROOT" \
  --model-root "$DEV_DAILY_MODEL_ROOT" \
  --frozen-artifact-root "$frozen_artifact_root" \
  --output-root "$DEV_OFFICIAL_SCHEDULE_ROOT" \
  --game-catalog-path "$DEV_STATE_ROOT/catalog/games.parquet"

if [[ "${GESTALT_DEV_REMATERIALIZE_CURRENT_STATE:-0}" == "1" ]] || \
   [[ ! -f "$DEV_STATE_ROOT/current-season-serving/latest.json" ]] || \
   [[ ! -f "$DEV_STATE_ROOT/current-season-lineups/latest.json" ]] || \
   [[ ! -d "$DEV_SCHEDULE_ROOT/final" ]] || \
   [[ ! -f "$DEV_STATE_ROOT/site-releases/latest.json" ]]; then
  if systemctl cat gestalt-web-dev >/dev/null 2>&1; then
    sudo systemctl stop gestalt-web-dev
  fi

  PYTHONPATH="$RELEASE_ROOT/src" \
    "$PROD_RELEASE/.venv/bin/python" \
    -m nba_lineup_model.modeling.current_season_serving_materialization \
    --daily-model-root "$DEV_DAILY_MODEL_ROOT" \
    --daily-data-root "$DEV_DAILY_DATA_ROOT" \
    --release-root "$PROD_RELEASE" \
    --output-root "$DEV_STATE_ROOT/current-season-serving"

  # Do not let evaluator construction load the previously promoted lineup
  # bundle: it is intentionally incompatible with the serving bundle that was
  # just rebuilt. The command publishes its replacement to --output-root.
  PYTHONPATH="$RELEASE_ROOT/src" \
  GESTALT_CURRENT_SEASON_SERVING_ROOT="$DEV_STATE_ROOT/current-season-serving" \
  GESTALT_CURRENT_SEASON_LINEUPS_ROOT="$DEV_STATE_ROOT/lineup-materialization-input" \
  GESTALT_SCHEDULE_STATE_DIR="$DEV_SCHEDULE_ROOT" \
  GESTALT_PRESEASON_RANKINGS_PATH="$DEV_PRESEASON_RANKINGS_PATH" \
  GESTALT_PRESEASON_PROFILES_PATH="$DEV_PRESEASON_PROFILES_PATH" \
    "$PROD_RELEASE/.venv/bin/python" \
    -m nba_lineup_model.modeling.current_season_lineup_materialization \
    --daily-data-root "$DEV_DAILY_DATA_ROOT" \
    --output-root "$DEV_STATE_ROOT/current-season-lineups"

fi

if [[ -n "$FRONTEND_DIST" ]]; then
  test -f "$FRONTEND_DIST/index.html"
elif command -v npm >/dev/null 2>&1; then
  npm --prefix "$RELEASE_ROOT/apps/lineup-explorer" run build
  FRONTEND_DIST="$RELEASE_ROOT/apps/lineup-explorer/dist"
else
  printf 'npm is unavailable; set GESTALT_DEV_FRONTEND_DIST to a prebuilt dist directory\n' >&2
  exit 1
fi
if systemctl cat gestalt-web-dev >/dev/null 2>&1; then
  sudo systemctl stop gestalt-web-dev
fi
sudo install -d -m 0755 "$DEV_WEB_ROOT"
sudo rsync -a --delete --chmod=D755,F644 "$FRONTEND_DIST/" "$DEV_WEB_ROOT/"

sudo ln -sfn "$RELEASE_ROOT" "$DEV_APP_ROOT/current"
sudo install -m 0644 \
  "$RELEASE_ROOT/deploy/gestalt-web-dev.service" \
  /etc/systemd/system/gestalt-web-dev.service
sudo install -m 0644 \
  "$RELEASE_ROOT/deploy/gestalt-dev.nginx.conf" \
  /etc/nginx/sites-available/gestalt-dev
sudo ln -sfn \
  /etc/nginx/sites-available/gestalt-dev \
  /etc/nginx/sites-enabled/gestalt-dev

if [[ ! -f /etc/nginx/gestalt-dev.htpasswd ]]; then
  password="$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-24)"
  password_hash="$(openssl passwd -6 "$password")"
  printf 'gestalt:%s\n' "$password_hash" | sudo tee /etc/nginx/gestalt-dev.htpasswd >/dev/null
  printf 'GESTALT_DEV_USERNAME=gestalt\nGESTALT_DEV_PASSWORD=%s\n' "$password"
fi
sudo chown root:www-data /etc/nginx/gestalt-dev.htpasswd
sudo chmod 0640 /etc/nginx/gestalt-dev.htpasswd

sudo systemctl daemon-reload
sudo systemctl enable --now gestalt-web-dev
sudo nginx -t
sudo systemctl reload nginx

for _ in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8011/api/health >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:8011/api/health >/dev/null

# Refresh only today's forward-looking dev board. Historical completed games
# remain the strict-forward replay materialized above, and no dev artifact is
# uploaded to the production S3 archive.
PYTHONPATH="$RELEASE_ROOT/src" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.season.official_schedule_runtime \
  --date "$(TZ=America/Los_Angeles date +%F)" \
  --state-dir "$DEV_SCHEDULE_ROOT" \
  --odds-state-dir "$PROD_SCHEDULE_ROOT" \
  --game-catalog-path "$DEV_STATE_ROOT/catalog/games.parquet" \
  --projection-url http://127.0.0.1:8011/api/win-projections \
  --days-ahead 7 \
  --no-s3

"$RELEASE_ROOT/scripts/validate_gestalt_serving_contract.sh" \
  http://127.0.0.1:8011
"$RELEASE_ROOT/scripts/compare_gestalt_data_coverage.sh" \
  http://127.0.0.1:8011 \
  http://127.0.0.1:8010 \
  --allow-left-evidence-superset

# Bind the verified data/model tips and every serving surface into one release
# identity only after representative API and browser-facing contracts pass.
PYTHONPATH="$RELEASE_ROOT/src" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.season.dev_site_release \
  --daily-data-root "$DEV_DAILY_DATA_ROOT" \
  --daily-model-root "$DEV_DAILY_MODEL_ROOT" \
  --serving-root "$DEV_STATE_ROOT/current-season-serving" \
  --lineups-root "$DEV_STATE_ROOT/current-season-lineups" \
  --schedule-state-dir "$DEV_SCHEDULE_ROOT" \
  --catalog-path "$DEV_STATE_ROOT/catalog/games.parquet" \
  --preseason-rankings-path "$DEV_PRESEASON_RANKINGS_PATH" \
  --preseason-profiles-path "$DEV_PRESEASON_PROFILES_PATH" \
  --code-revision "$(git -C "$RELEASE_ROOT" rev-parse HEAD)" \
  --api-root http://127.0.0.1:8011 \
  --output-root "$DEV_STATE_ROOT/site-releases"

site_release_sha="$(curl --fail --silent http://127.0.0.1:8011/api/health | jq -er '.site_release.release_sha256')"
printf 'Validated immutable dev site release %s\n' "$site_release_sha"
printf 'Deployed NBA GESTALT dev release %s from %s\n' "$RELEASE_ID" "$DEV_REF"
