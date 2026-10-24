#!/usr/bin/env bash
set -euo pipefail

# Install Dagster alongside the active immutable GESTALT release. Automations
# remain stopped on first load and must be cut over explicitly after parity.

export PATH="$HOME/.local/bin:$PATH"

APP_ROOT="${GESTALT_APP_ROOT:-/opt/nba-gestalt}"
REPO_DIR="${GESTALT_REPO_DIR:-$HOME/nba-lineup-model}"
DAGSTER_HOME="${DAGSTER_HOME:-/var/lib/nba-gestalt/dagster}"
CURRENT_RELEASE="$APP_ROOT/current"

test -x "$CURRENT_RELEASE/.venv/bin/python"
test -f "$CURRENT_RELEASE/deploy/orchestration-requirements.txt"

uv pip install \
  --python "$CURRENT_RELEASE/.venv/bin/python" \
  -r "$CURRENT_RELEASE/deploy/orchestration-requirements.txt"

sudo install -d -o ubuntu -g ubuntu "$DAGSTER_HOME"
sudo install -m 0644 "$REPO_DIR/deploy/dagster.yaml" "$DAGSTER_HOME/dagster.yaml"

cd "$CURRENT_RELEASE"
DAGSTER_HOME="$DAGSTER_HOME" \
  "$CURRENT_RELEASE/.venv/bin/dagster" definitions validate \
  --module-name nba_lineup_model.orchestration.definitions

sudo install -m 0644 \
  "$REPO_DIR/deploy/gestalt-dagster-webserver.service" \
  /etc/systemd/system/gestalt-dagster-webserver.service
sudo install -m 0644 \
  "$REPO_DIR/deploy/gestalt-dagster-daemon.service" \
  /etc/systemd/system/gestalt-dagster-daemon.service
sudo systemctl daemon-reload
sudo systemctl enable --now gestalt-dagster-webserver gestalt-dagster-daemon

for _ in {1..30}; do
  if curl --fail --silent http://127.0.0.1:3000/ >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:3000/ >/dev/null
systemctl --quiet is-active gestalt-dagster-webserver gestalt-dagster-daemon

printf 'Dagster deployed on 127.0.0.1:3000 with automations stopped pending cutover.\n'
