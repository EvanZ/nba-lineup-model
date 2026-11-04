#!/usr/bin/env bash
# Install only the independent dev watchdog. Never restart collectors or the API.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVE_CODE=/var/lib/nba-gestalt-dev/live-release/code

test -z "$(git -C "$REPO_ROOT" status --porcelain)" || {
  printf 'Commit the tested watchdog configuration before deployment\n' >&2
  exit 1
}
test -f "$LIVE_CODE/src/nba_lineup_model/orchestration/data_quality_probe.py"
requirements_sha="$("$LIVE_CODE/scripts/validate_gestalt_runtime_environment.py" \
  --hash-only --requirements "$LIVE_CODE/deploy/web-requirements.txt" \
  "$LIVE_CODE/deploy/orchestration-requirements.txt")"
runtime="/opt/nba-gestalt-orchestration/environments/requirements_sha256=$requirements_sha"
test -x "$runtime/bin/python" || {
  printf 'Missing versioned orchestration environment: %s\n' "$runtime" >&2
  exit 1
}
# Validate the installed environment without installing dependencies or moving aliases.
"$runtime/bin/python" "$LIVE_CODE/scripts/validate_gestalt_runtime_environment.py" \
  --profile orchestration --requirements "$LIVE_CODE/deploy/web-requirements.txt" \
  "$LIVE_CODE/deploy/orchestration-requirements.txt" >/dev/null

staging="$(mktemp -d /tmp/gestalt-data-quality-dev.XXXXXX)"
trap 'rm -f -- "$staging/gestalt-data-quality-dev.service"; rmdir -- "$staging"' EXIT
sed "s|@ORCHESTRATION_PYTHON@|$runtime/bin/python|g" \
  "$REPO_ROOT/deploy/gestalt-data-quality-dev.service" \
  > "$staging/gestalt-data-quality-dev.service"
systemd-analyze verify "$staging/gestalt-data-quality-dev.service" \
  "$REPO_ROOT/deploy/gestalt-data-quality-dev.timer"
sudo -n install -d -o ubuntu -g ubuntu -m 0755 /var/lib/nba-gestalt-dev/data-quality
sudo -n install -d -m 0755 /etc/nba-gestalt
sudo -n install -m 0644 "$REPO_ROOT/deploy/data-quality-thresholds.json" \
  /etc/nba-gestalt/data-quality-thresholds.json
sudo -n install -m 0644 "$staging/gestalt-data-quality-dev.service" \
  /etc/systemd/system/gestalt-data-quality-dev.service
sudo -n install -m 0644 "$REPO_ROOT/deploy/gestalt-data-quality-dev.timer" \
  /etc/systemd/system/gestalt-data-quality-dev.timer
sudo -n systemctl daemon-reload
sudo -n systemctl enable --now gestalt-data-quality-dev.timer
sudo -n systemctl start gestalt-data-quality-dev.service
systemctl show gestalt-data-quality-dev.service -p Result -p ExecMainStatus
systemctl list-timers gestalt-data-quality-dev.timer --no-pager
