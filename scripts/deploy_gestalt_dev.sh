#!/usr/bin/env bash
set -euo pipefail

# Build with the canonical daily pipeline first. Never reconstruct historical
# forecasts or replace independent serving/code pointers during deployment.
CONFIG="${GESTALT_DEV_PIPELINE_CONFIG:-/var/lib/nba-gestalt-dev/daily-pipeline/config.json}"
REPORT="${GESTALT_DEV_PIPELINE_REPORT:?Set the fully validated daily pipeline report path}"
LIVE=/var/lib/nba-gestalt-dev/live-release
test -L "$LIVE" || { echo 'Initial atomic dev runtime bootstrap must be reviewed first.' >&2; exit 1; }
SOURCE_REPO="${GESTALT_SOURCE_REPO:-/home/ubuntu/nba-lineup-model}"
sudo install -d -o root -g root -m 0755 /var/lib/nba-gestalt-dev/release-publisher
sudo chown root:root /var/lib/nba-gestalt-dev
sudo chmod 0755 /var/lib/nba-gestalt-dev
sudo install -m 0644 "$SOURCE_REPO/deploy/gestalt-web-dev.service" \
  /etc/systemd/system/gestalt-web-dev.service
sudo systemctl daemon-reload
cd /opt/nba-gestalt/current
sudo env PYTHONPATH="$SOURCE_REPO/src" \
  .venv/bin/python - "$CONFIG" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path
from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
from nba_lineup_model.orchestration.daily_activation import activate_daily_release
config = DailyPipelineConfig.read(sys.argv[1])
if config.environment_root != Path('/var/lib/nba-gestalt-dev') or config.live_pointer != Path('/var/lib/nba-gestalt-dev/live-release'):
    raise SystemExit('This entry point can activate only the dev environment')
report = json.loads(Path(sys.argv[2]).read_text())
activate_daily_release(config, report)
print(json.dumps(report, indent=2))
PY
