#!/usr/bin/env bash
set -euo pipefail

# Deploy one Git ref beside production without copying the data lake or mutating
# production releases/state. The dev runtime reuses production's installed
# Python dependencies and immutable model/data bundle read-only.

export PATH="$HOME/.local/bin:$PATH"

DEV_REF="${GESTALT_DEV_REF:?GESTALT_DEV_REF is required}"
SOURCE_REPO="${GESTALT_SOURCE_REPO:-$HOME/nba-lineup-model}"
PROD_APP_ROOT="${GESTALT_PROD_APP_ROOT:-/opt/nba-gestalt}"
DEV_APP_ROOT="${GESTALT_DEV_APP_ROOT:-/opt/nba-gestalt-dev}"
DEV_STATE_ROOT="${GESTALT_DEV_STATE_ROOT:-/var/lib/nba-gestalt-dev}"
DEV_WEB_ROOT="${GESTALT_DEV_WEB_ROOT:-/var/www/nba-gestalt-dev}"
FRONTEND_DIST="${GESTALT_DEV_FRONTEND_DIST:-}"
RELEASE_ID="${GESTALT_DEV_RELEASE_ID:-$(git -C "$SOURCE_REPO" rev-parse --short=12 "$DEV_REF")}"
RELEASE_ROOT="$DEV_APP_ROOT/releases/$RELEASE_ID"
PROD_RELEASE="$PROD_APP_ROOT/current"

test -d "$SOURCE_REPO/.git"
test -x "$PROD_RELEASE/.venv/bin/python"
test -d "$PROD_RELEASE/artifacts"
test -d "$PROD_RELEASE/data"

sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_APP_ROOT/releases"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/catalog"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/current-season-serving"
sudo install -d -o "$USER" -g "$(id -gn)" "$DEV_STATE_ROOT/current-season-lineups"
if systemctl cat gestalt-web-dev >/dev/null 2>&1; then
  sudo systemctl stop gestalt-web-dev
fi

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

PYTHONPATH="$RELEASE_ROOT/src" \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.modeling.current_season_serving_materialization \
  --rating-root /var/lib/nba-gestalt/in-season-nail \
  --minutes-root /var/lib/nba-gestalt/in-season-minutes \
  --release-root "$PROD_RELEASE" \
  --output-root "$DEV_STATE_ROOT/current-season-serving"

PYTHONPATH="$RELEASE_ROOT/src" \
GESTALT_CURRENT_SEASON_SERVING_ROOT="$DEV_STATE_ROOT/current-season-serving" \
GESTALT_CURRENT_SEASON_LINEUPS_ROOT="$DEV_STATE_ROOT/current-season-lineups" \
GESTALT_SCHEDULE_STATE_DIR=/var/lib/nba-gestalt/schedule \
  "$PROD_RELEASE/.venv/bin/python" \
  -m nba_lineup_model.modeling.current_season_lineup_materialization \
  --schedule-state-dir /var/lib/nba-gestalt/schedule \
  --coverage-path /var/lib/nba-gestalt/cdn/coverage.json \
  --output-root "$DEV_STATE_ROOT/current-season-lineups"

if [[ -n "$FRONTEND_DIST" ]]; then
  test -f "$FRONTEND_DIST/index.html"
elif command -v npm >/dev/null 2>&1; then
  npm --prefix "$RELEASE_ROOT/apps/lineup-explorer" run build
  FRONTEND_DIST="$RELEASE_ROOT/apps/lineup-explorer/dist"
else
  printf 'npm is unavailable; set GESTALT_DEV_FRONTEND_DIST to a prebuilt dist directory\n' >&2
  exit 1
fi
sudo install -d "$DEV_WEB_ROOT"
sudo rsync -a --delete "$FRONTEND_DIST/" "$DEV_WEB_ROOT/"

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
  sudo chmod 0640 /etc/nginx/gestalt-dev.htpasswd
  printf 'GESTALT_DEV_USERNAME=gestalt\nGESTALT_DEV_PASSWORD=%s\n' "$password"
fi

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
printf 'Deployed NBA GESTALT dev release %s from %s\n' "$RELEASE_ID" "$DEV_REF"
