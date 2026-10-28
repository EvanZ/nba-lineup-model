#!/usr/bin/env bash
set -euo pipefail

# Promote one reviewed Git ref to production without copying the immutable
# model/data bundle. Production gets its own code worktree and freshly
# materialized serving pointers; the legacy schedule remains an untouched
# archive of the originally published forecasts.

export PATH="$HOME/.local/bin:$PATH"
if ! command -v npm >/dev/null 2>&1; then
  for node_bin in "$HOME"/.nvm/versions/node/*/bin; do
    if [[ -x "$node_bin/node" && -x "$node_bin/npm" ]]; then
      export PATH="$node_bin:$PATH"
    fi
  done
fi

PROD_REF="${GESTALT_PROD_REF:?GESTALT_PROD_REF is required}"
SOURCE_REPO="${GESTALT_SOURCE_REPO:-$HOME/nba-lineup-model}"
APP_ROOT="${GESTALT_APP_ROOT:-/opt/nba-gestalt}"
CODE_ROOT="${GESTALT_CODE_ROOT:-/opt/nba-gestalt-code}"
STATE_ROOT="${GESTALT_STATE_ROOT:-/var/lib/nba-gestalt}"
WEB_RELEASE_ROOT="${GESTALT_WEB_RELEASE_ROOT:-/var/www/nba-gestalt-releases}"
PROD_RELEASE="$APP_ROOT/current"
DEV_RELEASE="${GESTALT_DEV_RELEASE:-/opt/nba-gestalt-dev/current}"
SOURCE_SCHEDULE_ROOT="${GESTALT_SOURCE_SCHEDULE_ROOT:-$STATE_ROOT/schedule-serving/current}"
RELEASE_ID="${GESTALT_PROD_RELEASE_ID:-$(git -C "$SOURCE_REPO" rev-parse --short=12 "$PROD_REF")}"
RELEASE_ROOT="$CODE_ROOT/releases/$RELEASE_ID"
FRONTEND_DIST="${GESTALT_PROD_FRONTEND_DIST:-}"

test -d "$SOURCE_REPO/.git"
test -x "$PROD_RELEASE/.venv/bin/python"
test -d "$PROD_RELEASE/artifacts"
test -d "$PROD_RELEASE/data"
test -d "$DEV_RELEASE"
if [[ ! -d "$SOURCE_SCHEDULE_ROOT/final" ]]; then
  SOURCE_SCHEDULE_ROOT="$STATE_ROOT/schedule"
fi

prod_commit="$(git -C "$SOURCE_REPO" rev-parse "$PROD_REF^{commit}")"
dev_commit="$(git -C "$DEV_RELEASE" rev-parse HEAD)"
if [[ "$dev_commit" != "$prod_commit" ]]; then
  printf 'Production promotion blocked: dev is %s but requested ref is %s\n' \
    "$dev_commit" "$prod_commit" >&2
  exit 1
fi
"$DEV_RELEASE/scripts/validate_gestalt_serving_contract.sh" \
  http://127.0.0.1:8011
"$DEV_RELEASE/scripts/compare_gestalt_data_coverage.sh" \
  http://127.0.0.1:8011 \
  http://127.0.0.1:8010

sudo install -d -o "$USER" -g "$(id -gn)" "$CODE_ROOT/releases"
sudo install -d -o "$USER" -g "$(id -gn)" "$STATE_ROOT/catalog"
sudo install -d -o "$USER" -g "$(id -gn)" "$STATE_ROOT/current-season-serving"
sudo install -d -o "$USER" -g "$(id -gn)" "$STATE_ROOT/current-season-lineups"
sudo install -d -o "$USER" -g "$(id -gn)" "$STATE_ROOT/schedule-serving"
sudo install -d -o "$USER" -g "$(id -gn)" "$WEB_RELEASE_ROOT"

if [[ ! -d "$RELEASE_ROOT" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$RELEASE_ROOT" "$PROD_REF"
fi

ln -sfn \
  "$SOURCE_REPO/apps/lineup-explorer/node_modules" \
  "$RELEASE_ROOT/apps/lineup-explorer/node_modules"

cd "$PROD_RELEASE"
aws s3 cp \
  s3://nba-gestalt-data/lake/catalog/games.parquet \
  "$STATE_ROOT/catalog/games.parquet.next" \
  --only-show-errors
mv "$STATE_ROOT/catalog/games.parquet.next" "$STATE_ROOT/catalog/games.parquet"

PYTHONPATH="$RELEASE_ROOT/src" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.modeling.current_season_serving_materialization \
  --rating-root "$STATE_ROOT/in-season-nail" \
  --minutes-root "$STATE_ROOT/in-season-minutes" \
  --release-root "$PROD_RELEASE" \
  --output-root "$STATE_ROOT/current-season-serving"

PYTHONPATH="$RELEASE_ROOT/src" \
GESTALT_CURRENT_SEASON_SERVING_ROOT="$STATE_ROOT/current-season-serving" \
GESTALT_CURRENT_SEASON_LINEUPS_ROOT="$STATE_ROOT/lineup-materialization-input" \
GESTALT_SCHEDULE_STATE_DIR="$SOURCE_SCHEDULE_ROOT" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.modeling.current_season_lineup_materialization \
  --schedule-state-dir "$SOURCE_SCHEDULE_ROOT" \
  --coverage-path "$STATE_ROOT/cdn/coverage.json" \
  --output-root "$STATE_ROOT/current-season-lineups"

PYTHONPATH="$RELEASE_ROOT/src" \
GESTALT_CURRENT_SEASON_SERVING_ROOT="$STATE_ROOT/current-season-serving" \
GESTALT_CURRENT_SEASON_LINEUPS_ROOT="$STATE_ROOT/lineup-materialization-input" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.modeling.current_season_schedule_materialization \
  --source-root "$SOURCE_SCHEDULE_ROOT" \
  --rating-root "$STATE_ROOT/in-season-nail" \
  --minutes-root "$STATE_ROOT/in-season-minutes" \
  --output-root "$STATE_ROOT/schedule-serving" \
  --catalog-path "$STATE_ROOT/catalog/games.parquet" \
  --include-current-forecast

if [[ -n "$FRONTEND_DIST" ]]; then
  test -f "$FRONTEND_DIST/index.html"
elif command -v npm >/dev/null 2>&1; then
  npm --prefix "$RELEASE_ROOT/apps/lineup-explorer" run build
  FRONTEND_DIST="$RELEASE_ROOT/apps/lineup-explorer/dist"
else
  printf 'npm is unavailable; set GESTALT_PROD_FRONTEND_DIST to a prebuilt dist directory\n' >&2
  exit 1
fi

WEB_RELEASE="$WEB_RELEASE_ROOT/$RELEASE_ID"
sudo install -d -m 0755 "$WEB_RELEASE"
sudo rsync -a --delete --chmod=D755,F644 "$FRONTEND_DIST/" "$WEB_RELEASE/"

# First migration: preserve the previous static site as a rollback release.
if [[ -d /var/www/nba-gestalt && ! -L /var/www/nba-gestalt ]]; then
  rollback_web="$WEB_RELEASE_ROOT/pre-code-overlay-$(date -u +%Y%m%dT%H%M%SZ)"
  sudo mv /var/www/nba-gestalt "$rollback_web"
fi

sudo ln -sfn "$RELEASE_ROOT" "$CODE_ROOT/current"
sudo ln -sfn "$WEB_RELEASE" /var/www/nba-gestalt
sudo install -m 0644 "$RELEASE_ROOT/deploy/gestalt-web.service" /etc/systemd/system/gestalt-web.service
sudo install -m 0644 "$RELEASE_ROOT/deploy/gestalt-dagster-code-server.service" /etc/systemd/system/gestalt-dagster-code-server.service
sudo install -m 0644 "$RELEASE_ROOT/deploy/gestalt-dagster-daemon.service" /etc/systemd/system/gestalt-dagster-daemon.service
sudo install -m 0644 "$RELEASE_ROOT/deploy/gestalt-dagster-webserver.service" /etc/systemd/system/gestalt-dagster-webserver.service
sudo install -m 0644 "$RELEASE_ROOT/deploy/gestalt-dagster-workspace.yaml" "$STATE_ROOT/dagster/workspace.yaml"

sudo systemctl daemon-reload
sudo systemctl restart gestalt-web
sudo systemctl restart gestalt-dagster-code-server
for _ in {1..60}; do
  if "$PROD_RELEASE/.venv/bin/dagster" api grpc-health-check \
    --host 127.0.0.1 --port 4000 >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"$PROD_RELEASE/.venv/bin/dagster" api grpc-health-check \
  --host 127.0.0.1 --port 4000
sudo systemctl restart gestalt-dagster-webserver gestalt-dagster-daemon
sudo nginx -t
sudo systemctl reload nginx

for _ in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8010/api/health >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:8010/api/health >/dev/null

# Reconcile the current board after the API cutover. Do not upload this one-time
# historical migration to the immutable production archive; normal Dagster
# collections resume against schedule-serving/current after the service restart.
PYTHONPATH="$RELEASE_ROOT/src" \
  "$PROD_RELEASE/.venv/bin/nba-refresh-gestalt-schedule" \
  --date "$(TZ=America/Los_Angeles date +%F)" \
  --state-dir "$STATE_ROOT/schedule-serving/current" \
  --projection-url http://127.0.0.1:8010/api/win-projections \
  --days-ahead 7 \
  --no-s3


"$RELEASE_ROOT/scripts/validate_gestalt_serving_contract.sh" \
  http://127.0.0.1:8010

printf 'Promoted NBA GESTALT production code release %s from %s\n' "$RELEASE_ID" "$PROD_REF"
