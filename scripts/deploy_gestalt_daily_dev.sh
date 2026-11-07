#!/usr/bin/env bash
# Install the permanent dev-only instance. No site publication or job requests.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
publish_forecasts=false
automatic=false
if [[ "${1:-}" == "--enable-automatic-lifecycle" ]]; then
  automatic=true
  publish_forecasts=true
elif [[ "${1:-}" == "--enable-forecast-publication" ]]; then
  publish_forecasts=true
elif [[ $# -ne 0 ]]; then exit 2; fi
test -z "$(git -C "$REPO_ROOT" status --porcelain)" || {
  printf 'Commit tested code before installing the daily worker\n' >&2
  exit 1
}
requirements_sha="$("$REPO_ROOT/scripts/validate_gestalt_runtime_environment.py" \
  --hash-only --requirements "$REPO_ROOT/deploy/web-requirements.txt" \
  "$REPO_ROOT/deploy/orchestration-requirements.txt")"
runtime="/opt/nba-gestalt-orchestration/environments/requirements_sha256=$requirements_sha"
test -x "$runtime/bin/python"
"$runtime/bin/python" "$REPO_ROOT/scripts/validate_gestalt_runtime_environment.py" \
  --profile orchestration --requirements "$REPO_ROOT/deploy/web-requirements.txt" \
  "$REPO_ROOT/deploy/orchestration-requirements.txt" >/dev/null
sudo -n install -d -o ubuntu -g ubuntu -m 0755 /var/lib/nba-gestalt-dev/daily-releases
if $publish_forecasts; then
  sudo -n install -d -m 0755 /etc/polkit-1/rules.d
  sudo -n install -m 0644 "$REPO_ROOT/deploy/49-gestalt-dev-api.rules" \
    /etc/polkit-1/rules.d/49-gestalt-dev-api.rules
  # Only mutable publication containers need worker write access. Keep existing
  # sealed releases, generations, and journal entries unchanged.
  for path in /var/lib/nba-gestalt-dev \
      /var/lib/nba-gestalt-dev/release-publisher \
      /var/lib/nba-gestalt-dev/release-publisher/generations \
      /var/lib/nba-gestalt-dev/release-publisher/journal \
      /var/lib/nba-gestalt-dev/forecast-registry/receipts; do
    sudo -n mkdir -p "$path"
    sudo -n chgrp ubuntu "$path"
    sudo -n chmod g+ws "$path"
    sudo -n -u ubuntu test -w "$path"
  done
  lock=/var/lib/nba-gestalt-dev/release-publisher/.publisher.lock
  sudo -n touch "$lock"
  sudo -n chgrp ubuntu "$lock"
  sudo -n chmod g+rw "$lock"
  sudo -n -u ubuntu test -w "$lock"
fi
read -r package checksum < <(
  PYTHONPATH="$REPO_ROOT/src" "$runtime/bin/python" -m nba_lineup_model.orchestration.watchdog_deployment \
    --source "$REPO_ROOT" --output /var/lib/nba-gestalt-dev/daily-releases
)
test -n "$package" && test -n "$checksum"
PYTHONPATH="$package/code/src" "$runtime/bin/python" - "$package" "$publish_forecasts" <<'PY'
from dataclasses import replace
from pathlib import Path
import sys
from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
from nba_lineup_model.orchestration.daily_work import atomic_json
from nba_lineup_model.orchestration.resumable_pipeline import config_identity
from nba_lineup_model.orchestration.lifecycle import bind_accepted_configuration
from nba_lineup_model.storage.duckdb_lake import DEFAULT_LAKE_URI
package = Path(sys.argv[1])
enabled = sys.argv[2] == "true"
config = DailyPipelineConfig.read("/var/lib/nba-gestalt-dev/daily-pipeline/config.json")
if config.environment_root != Path("/var/lib/nba-gestalt-dev"):
    raise ValueError("Dev deployment cannot configure production jobs")
config = bind_accepted_configuration(config)
config = replace(config, code_root=package / "code",
    observation_root=Path("/var/lib/nba-gestalt/schedule-serving/current"),
    operational_reports_s3_uri="s3://nba-gestalt-data/lake/operational-reports/environment=dev",
    observation_s3_uri="s3://nba-gestalt-data/nba-gestalt/schedule/model_track=causal_hca_replacement_candidate_v2",
    activate_after_validation=enabled, forecast_injury_inputs_path=None,
    artifact_lake_uri=DEFAULT_LAKE_URI,
    artifact_cache_root=Path("/var/lib/nba-gestalt-dev/artifact-cache"))
path = package / ("forecast-config-enabled.json" if enabled else "forecast-config-preview.json")
atomic_json(path, config_identity(config), immutable=True)
print(f"Forecast job configuration: {path}")
PY
PYTHONPATH="$package/code/src" "$runtime/bin/python" -m nba_lineup_model.orchestration.daily_instance \
  initialize --environment dev --environment-root /var/lib/nba-gestalt-dev \
  --template "$package/code/deploy/dagster-daily.yaml"
# Initialize real Dagster storage, without creating or importing any runs.
PYTHONPATH="$package/code/src" "$runtime/bin/python" - <<'PY'
from dagster import DagsterInstance, RunsFilter, DagsterRunStatus
with DagsterInstance.from_config("/var/lib/nba-gestalt-dev/daily-dagster") as instance:
    active = instance.get_runs(filters=RunsFilter(statuses=[
        DagsterRunStatus.QUEUED, DagsterRunStatus.STARTING,
        DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING]), limit=1)
    if active:
        raise RuntimeError("Do not replace daily worker code while work is active")
PY
install -m 0644 "$package/code/deploy/gestalt-daily-dev-workspace.yaml" \
  /var/lib/nba-gestalt-dev/daily-dagster/workspace.yaml
staging="$(mktemp -d /tmp/gestalt-daily-dev.XXXXXX)"
aws_cli_bin=$(bash "$package/code/scripts/install_gestalt_worker_aws.sh")
automation_config=""
if $automatic; then
  automation_config="$package/forecast-config-enabled.json"
fi
trap 'rm -f -- "$staging/gestalt-daily-dev@.service"; rmdir -- "$staging"' EXIT
sed -e "s|@ORCHESTRATION_PYTHON@|$runtime/bin/python|g" \
  -e "s|@AUTOMATION_CONFIG@|$automation_config|g" \
  -e "s|@AWS_CLI_BIN@|$aws_cli_bin|g" \
  -e "s|@DAILY_PACKAGE@|$package|g" -e "s|@DAILY_SHA256@|$checksum|g" \
  "$package/code/deploy/gestalt-daily-dev@.service" > "$staging/gestalt-daily-dev@.service"
systemd-analyze verify "$staging/gestalt-daily-dev@.service"
sudo -n install -m 0644 "$staging/gestalt-daily-dev@.service" /etc/systemd/system/gestalt-daily-dev@.service
sudo -n systemctl daemon-reload
sudo -n systemctl enable gestalt-daily-dev@code gestalt-daily-dev@web gestalt-daily-dev@daemon
sudo -n systemctl restart gestalt-daily-dev@code gestalt-daily-dev@web gestalt-daily-dev@daemon
for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:3002/server_info >/dev/null; then break; fi
  sleep 1
done
DAGSTER_HOME=/var/lib/nba-gestalt-dev/daily-dagster "$runtime/bin/dagster" sensor start \
  --workspace /var/lib/nba-gestalt-dev/daily-dagster/workspace.yaml resumable_daily_partition
if $automatic; then
  DAGSTER_HOME=/var/lib/nba-gestalt-dev/daily-dagster "$runtime/bin/dagster" sensor start \
    --workspace /var/lib/nba-gestalt-dev/daily-dagster/workspace.yaml automatic_daily_lifecycle
fi
systemctl is-active gestalt-daily-dev@code gestalt-daily-dev@web gestalt-daily-dev@daemon
