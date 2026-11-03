#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${GESTALT_SOURCE_REPO:-$HOME/nba-lineup-model}"
APP_ROOT="${GESTALT_DAILY_APP_ROOT:-/opt/nba-gestalt-daily}"
STATE_ROOT="${GESTALT_DAILY_STATE_ROOT:-/var/lib/nba-gestalt/daily-dagster}"
CONFIG="${GESTALT_DAILY_PIPELINE_CONFIG:-/var/lib/nba-gestalt/daily-pipeline-v2/config.json}"
REF="${GESTALT_DAILY_CODE_REF:-HEAD}"

test -d "$SOURCE_REPO/.git"
test -f "$CONFIG"
if [[ -n "$(git -C "$SOURCE_REPO" status --porcelain)" ]]; then
  printf 'Commit the tested daily pipeline before deployment\n' >&2
  exit 1
fi

daemon_was_active=false
if systemctl is-active --quiet gestalt-daily-daemon; then
  daemon_was_active=true
  sudo systemctl stop gestalt-daily-daemon
fi
restore_daily_daemon() {
  if [[ "$daemon_was_active" == true ]]; then
    sudo systemctl start gestalt-daily-daemon || true
  fi
}
trap restore_daily_daemon ERR

active_runs="$(
  /usr/bin/python3 - "$STATE_ROOT/history/runs.db" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    connection = sqlite3.connect(path)
    statuses = ("QUEUED", "STARTING", "STARTED", "CANCELING")
    placeholders = ",".join("?" for _ in statuses)
    query = f"select count(*) from runs where status in ({placeholders})"
    print(connection.execute(query, statuses).fetchone()[0])
PY
)"
if [[ "$active_runs" != 0 ]]; then
  restore_daily_daemon
  trap - ERR
  printf 'Refusing daily deployment while %s run(s) are active\n' "$active_runs" >&2
  exit 1
fi

commit="$(git -C "$SOURCE_REPO" rev-parse "$REF^{commit}")"
release="$APP_ROOT/releases/$(git -C "$SOURCE_REPO" rev-parse --short=12 "$commit")"
sudo install -d -o "$USER" -g "$(id -gn)" "$APP_ROOT/releases"
if [[ ! -d "$release" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$release" "$commit"
fi
test "$(git -C "$release" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$release" status --porcelain)"
model_configuration_source="$release/deploy/causal-hca-replacement-candidate-v2.json"
model_configuration_target="$(dirname "$CONFIG")/prior/causal_hca_selection.json"
sudo install -d -o ubuntu -g ubuntu "$(dirname "$model_configuration_target")"
if [[ -f "$model_configuration_target" ]] &&
   ! cmp -s "$model_configuration_source" "$model_configuration_target"; then
  restore_daily_daemon
  trap - ERR
  printf 'Refusing to overwrite a different accepted HCA configuration\n' >&2
  exit 1
fi
sudo install -m 0644 "$model_configuration_source" "$model_configuration_target"

PYTHONPATH="$release/src" /opt/nba-gestalt/current/.venv/bin/python -c \
  'from dagster import Definitions; from nba_lineup_model.orchestration.daily_definitions import defs; Definitions.validate_loadable(defs)'

sudo install -d -o ubuntu -g ubuntu "$STATE_ROOT"
sudo install -m 0644 "$release/deploy/dagster-daily.yaml" "$STATE_ROOT/dagster.yaml"
sudo install -m 0644 "$release/deploy/gestalt-daily-workspace.yaml" "$STATE_ROOT/workspace.yaml"
for unit in gestalt-daily-code-server.service gestalt-daily-webserver.service gestalt-daily-daemon.service; do
  sudo install -m 0644 "$release/deploy/$unit" "/etc/systemd/system/$unit"
done
temporary="$APP_ROOT/.current.tmp.$$"
sudo ln -s "$release" "$temporary"
sudo mv -Tf "$temporary" "$APP_ROOT/current"
sudo systemctl daemon-reload
sudo systemctl enable gestalt-daily-code-server gestalt-daily-webserver gestalt-daily-daemon
sudo systemctl restart gestalt-daily-code-server
sudo systemctl restart gestalt-daily-webserver
sudo systemctl restart gestalt-daily-daemon
for service in gestalt-daily-code-server gestalt-daily-webserver gestalt-daily-daemon; do
  test "$(systemctl is-active "$service")" = active
done

trap - ERR
printf 'Deployed isolated daily Dagster services at %s\n' "$commit"
