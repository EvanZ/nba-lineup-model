#!/usr/bin/env bash
set -euo pipefail

# Run from the EC2 host after a release has been published to S3.
export PATH="$HOME/.local/bin:$PATH"

BUCKET="${GESTALT_S3_BUCKET:-nba-gestalt-data}"
RELEASE_ID="${GESTALT_RELEASE_ID:?GESTALT_RELEASE_ID is required}"
APP_ROOT="${GESTALT_APP_ROOT:-/opt/nba-gestalt}"
WEB_ROOT="${GESTALT_WEB_ROOT:-/var/www/nba-gestalt}"
REPO_DIR="${GESTALT_REPO_DIR:-$HOME/nba-lineup-model}"
RELEASE_ROOT="$APP_ROOT/releases/$RELEASE_ID"

sudo install -d -o "$USER" -g "$(id -gn)" "$APP_ROOT/releases"
mkdir -p "$RELEASE_ROOT"
aws s3 sync --delete --exclude '.venv/*' --exclude '*/__pycache__/*' --exclude '*.pyc' \
  "s3://${BUCKET}/nba-gestalt/releases/${RELEASE_ID}/" "$RELEASE_ROOT/"

cd "$RELEASE_ROOT"
if [[ ! -d .venv ]]; then
  uv venv .venv
fi
uv pip install --python .venv/bin/python -r deploy/web-requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .

if ! command -v npm >/dev/null 2>&1; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # shellcheck disable=SC1091
  source "$NVM_DIR/nvm.sh"
fi

cd "$REPO_DIR/apps/lineup-explorer"
npm ci
npm run build

sudo mkdir -p "$WEB_ROOT"
sudo rsync -a --delete dist/ "$WEB_ROOT/"

sudo ln -sfn "$RELEASE_ROOT" "$APP_ROOT/current"
sudo install -m 0644 "$REPO_DIR/deploy/gestalt-rate-limits.conf" /etc/nginx/conf.d/gestalt-rate-limits.conf
sudo systemctl daemon-reload
sudo systemctl restart gestalt-web
sudo nginx -t
sudo systemctl reload nginx

for _ in {1..30}; do
  if curl --fail --silent http://127.0.0.1:8010/api/health >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:8010/api/health >/dev/null
printf 'Deployed NBA GESTALT release %s\n' "$RELEASE_ID"
