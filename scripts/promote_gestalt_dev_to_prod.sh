#!/usr/bin/env bash
set -euo pipefail

# Promote one reviewed dev release through a single rollback-safe runtime
# pointer. All immutable components are staged and preflighted before the
# public process changes. Schedule collection remains a separate mutable
# overlay initialized from the approved snapshot.

export PATH="$HOME/.local/bin:$PATH"
if ! command -v npm >/dev/null 2>&1; then
  for node_bin in "$HOME"/.nvm/versions/node/*/bin; do
    if [[ -x "$node_bin/node" && -x "$node_bin/npm" ]]; then
      export PATH="$node_bin:$PATH"
    fi
  done
fi

PROD_REF="${GESTALT_PROD_REF:?GESTALT_PROD_REF is required}"
PROD_APPROVAL_REFERENCE="${GESTALT_PROD_APPROVAL_REFERENCE:?GESTALT_PROD_APPROVAL_REFERENCE is required}"
SOURCE_REPO="${GESTALT_SOURCE_REPO:-$HOME/nba-lineup-model}"
APP_ROOT="${GESTALT_APP_ROOT:-/opt/nba-gestalt}"
CODE_ROOT="${GESTALT_CODE_ROOT:-/opt/nba-gestalt-code}"
CODE_CURRENT="$CODE_ROOT/current"
STATE_ROOT="${GESTALT_STATE_ROOT:-/var/lib/nba-gestalt}"
DEV_STATE_ROOT="${GESTALT_DEV_STATE_ROOT:-/var/lib/nba-gestalt-dev}"
DEV_RUNTIME="$(readlink -f "${GESTALT_APPROVED_DEV_RUNTIME:-$DEV_STATE_ROOT/live-release}")"
PROD_RELEASE="$APP_ROOT/current"
DEV_RELEASE="$DEV_RUNTIME/code"
DEV_SERVING_ROOT="$DEV_RUNTIME/current-season-serving"
DEV_LINEUPS_ROOT="$DEV_RUNTIME/current-season-lineups"
DEV_SCHEDULE_ROOT="$DEV_RUNTIME/schedule-snapshot"
DEV_SITE_RELEASE_ROOT="$DEV_RUNTIME/site-releases"
DEV_CATALOG_PATH="$DEV_RUNTIME/catalog/games.parquet"
DEV_PRESEASON_ENV="$DEV_RUNTIME/site-release.env"
PRODUCTION_RELEASE_ROOT="$STATE_ROOT/site-releases"
LIVE_POINTER="$STATE_ROOT/live-release"
PUBLISHER_ROOT="$STATE_ROOT/release-publisher"
SCHEDULE_RELEASE_ROOT="$STATE_ROOT/schedule-serving/releases"
SCHEDULE_CURRENT="$STATE_ROOT/schedule-serving/current"
PUBLIC_WEB=/var/www/nba-gestalt
FRONTEND_DIST="${GESTALT_PROD_FRONTEND_DIST:-$DEV_RUNTIME/web}"

test -d "$SOURCE_REPO/.git"
test -x "$PROD_RELEASE/.venv/bin/python"
test -d "$PROD_RELEASE/artifacts"
test -d "$PROD_RELEASE/data"
test -f "$DEV_RUNTIME/runtime-manifest.json"
test -d "$DEV_RELEASE"
test -f "$DEV_SITE_RELEASE_ROOT/latest.json"
test -f "$DEV_PRESEASON_ENV"
if [[ "$(jq -er '.identity.artifact_kind' "$DEV_RUNTIME/runtime-manifest.json")" != "production_site_release" ]]; then
  printf 'Production promotion blocked: active dev pointer is not an approved runtime\n' >&2
  exit 1
fi

if [[ "$PROD_REF" == origin/* ]]; then
  prod_branch="${PROD_REF#origin/}"
  git -C "$SOURCE_REPO" fetch --no-tags origin \
    "refs/heads/$prod_branch:refs/remotes/origin/$prod_branch"
fi
prod_commit="$(git -C "$SOURCE_REPO" rev-parse "$PROD_REF^{commit}")"
dev_commit="$(jq -er '.identity.code_revision' "$DEV_RUNTIME/runtime-manifest.json")"
git -C "$SOURCE_REPO" cat-file -e "$dev_commit^{commit}"
if [[ "$dev_commit" != "$prod_commit" ]]; then
  printf 'Production promotion blocked: dev is %s but requested ref is %s\n' \
    "$dev_commit" "$prod_commit" >&2
  exit 1
fi

"$DEV_RELEASE/scripts/validate_gestalt_serving_contract.sh" \
  http://127.0.0.1:8011
"$DEV_RELEASE/scripts/compare_gestalt_data_coverage.sh" \
  http://127.0.0.1:8011 \
  http://127.0.0.1:8010 \
  --allow-left-evidence-superset

approved_release_path="$(jq -er '.release_path' "$DEV_SITE_RELEASE_ROOT/latest.json")"
approved_release_sha="$(jq -er '.release_sha256' "$DEV_SITE_RELEASE_ROOT/latest.json")"
approved_release="$DEV_SITE_RELEASE_ROOT/$approved_release_path"
test -d "$approved_release"
if [[ "$(jq -er '.identity.approved_release_sha256' "$DEV_RUNTIME/runtime-manifest.json")" != "$approved_release_sha" ]]; then
  printf 'Production promotion blocked: active dev runtime and approval record differ\n' >&2
  exit 1
fi
if [[ "$(jq -er '.identity.code_revision' "$approved_release/manifest.json")" != "$prod_commit" ]]; then
  printf 'Production promotion blocked: approved manifest has different code\n' >&2
  exit 1
fi
approved_game_count="$(jq -er '.identity.game_count' "$approved_release/manifest.json")"
if ! [[ "$approved_game_count" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Production promotion blocked: approved release has invalid game coverage\n' >&2
  exit 1
fi
approved_evidence_date="$(jq -er '.identity.evidence_through_date' "$approved_release/manifest.json")"
if ! [[ "$approved_evidence_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  printf 'Production promotion blocked: approved release has invalid evidence date\n' >&2
  exit 1
fi

release_id="${GESTALT_PROD_RELEASE_ID:-$(git -C "$SOURCE_REPO" rev-parse --short=12 "$PROD_REF")}"
publisher_operation_id="${GESTALT_PROD_OPERATION_ID:-prod-$release_id-${approved_release_sha:0:12}}"
release_root="$CODE_ROOT/releases/$release_id"
sudo install -d -o "$USER" -g "$(id -gn)" "$CODE_ROOT/releases"
sudo install -d -o "$USER" -g "$(id -gn)" "$PRODUCTION_RELEASE_ROOT"
sudo install -d -o root -g root -m 0755 "$PUBLISHER_ROOT"
sudo install -d -o "$USER" -g "$(id -gn)" "$SCHEDULE_RELEASE_ROOT"
if [[ ! -d "$release_root" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$release_root" "$PROD_REF"
fi
if [[ "$(git -C "$release_root" rev-parse HEAD)" != "$prod_commit" ]]; then
  printf 'Production code worktree has the wrong commit\n' >&2
  exit 1
fi
if [[ -n "$(git -C "$release_root" status --porcelain)" ]]; then
  printf 'Production code worktree is dirty\n' >&2
  exit 1
fi
ln -sfn \
  "$SOURCE_REPO/apps/lineup-explorer/node_modules" \
  "$release_root/apps/lineup-explorer/node_modules"

if [[ -n "$FRONTEND_DIST" ]]; then
  test -f "$FRONTEND_DIST/index.html"
elif command -v npm >/dev/null 2>&1; then
  npm --prefix "$release_root/apps/lineup-explorer" run build
  FRONTEND_DIST="$release_root/apps/lineup-explorer/dist"
else
  printf 'npm is unavailable; set GESTALT_PROD_FRONTEND_DIST\n' >&2
  exit 1
fi

schedule_contract="$(${PROD_RELEASE}/.venv/bin/python -P -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from nba_lineup_model.season.schedule_artifact_contract import validate_schedule_artifacts; print(json.dumps(validate_schedule_artifacts(sys.argv[2])))' "$release_root/src" "$DEV_SCHEDULE_ROOT")"
schedule_sha="$(jq -er '.tree_sha256' <<<"$schedule_contract")"
schedule_overlay="$SCHEDULE_RELEASE_ROOT/$approved_release_sha-$schedule_sha"
if [[ ! -d "$schedule_overlay" ]]; then
  schedule_staging="$SCHEDULE_RELEASE_ROOT/.schedule.tmp-$approved_release_sha-$$"
  install -d "$schedule_staging"
  rsync -a --delete "$DEV_SCHEDULE_ROOT/" "$schedule_staging/"
  PYTHONPATH="$release_root/src" \
    "$PROD_RELEASE/.venv/bin/python" -P \
    -m nba_lineup_model.season.schedule_artifact_contract "$schedule_staging"
  mv "$schedule_staging" "$schedule_overlay"
fi
PYTHONPATH="$release_root/src" \
  "$PROD_RELEASE/.venv/bin/python" -P \
  -m nba_lineup_model.season.schedule_artifact_contract "$schedule_overlay"

preseason_rankings_path="$(sed -n 's/^GESTALT_PRESEASON_RANKINGS_PATH=//p' "$DEV_PRESEASON_ENV")"
preseason_profiles_path="$(sed -n 's/^GESTALT_PRESEASON_PROFILES_PATH=//p' "$DEV_PRESEASON_ENV")"
test -f "$preseason_rankings_path"
test -f "$preseason_profiles_path"

candidate="$(
  cd "$PROD_RELEASE"
  PYTHONPATH="$release_root/src" \
    "$PROD_RELEASE/.venv/bin/python" \
    -m nba_lineup_model.season.production_site_release stage \
    --dev-site-release-root "$DEV_SITE_RELEASE_ROOT" \
    --serving-root "$DEV_SERVING_ROOT" \
    --lineups-root "$DEV_LINEUPS_ROOT" \
    --schedule-snapshot-root "$DEV_SCHEDULE_ROOT" \
    --schedule-overlay-path "$schedule_overlay" \
    --catalog-path "$DEV_CATALOG_PATH" \
    --preseason-rankings-path "$preseason_rankings_path" \
    --preseason-profiles-path "$preseason_profiles_path" \
    --code-root "$release_root" \
    --frontend-root "$FRONTEND_DIST" \
    --output-root "$PRODUCTION_RELEASE_ROOT" \
    --live-pointer "$LIVE_POINTER"
)"
test -d "$candidate"

# Candidate API preflight. This exercises the exact staged code, models,
# lineups, catalogs, Schedule overlay and frontend before any live pointer moves.
preflight_log="/tmp/gestalt-production-preflight-$approved_release_sha.log"
(
  cd "$PROD_RELEASE"
  exec env \
    PYTHONPATH="$candidate/code/src" \
    GESTALT_CURRENT_SEASON_SERVING_ROOT="$candidate/current-season-serving" \
    GESTALT_CURRENT_SEASON_LINEUPS_ROOT="$candidate/current-season-lineups" \
    GESTALT_SCHEDULE_STATE_DIR="$candidate/schedule" \
    GESTALT_GAME_CATALOG_PATH="$candidate/catalog/games.parquet" \
    GESTALT_SITE_RELEASE_ROOT="$candidate/site-releases" \
    GESTALT_PRESEASON_RANKINGS_PATH="$candidate/preseason/rankings.parquet" \
    GESTALT_PRESEASON_PROFILES_PATH="$candidate/preseason/profiles.parquet" \
    "$PROD_RELEASE/.venv/bin/uvicorn" nba_lineup_model.web_api.app:app \
    --host 127.0.0.1 --port 8012 --workers 1
) >"$preflight_log" 2>&1 &
preflight_pid=$!
cleanup_preflight() {
  if kill -0 "$preflight_pid" >/dev/null 2>&1; then
    kill "$preflight_pid" >/dev/null 2>&1 || true
    wait "$preflight_pid" 2>/dev/null || true
  fi
}
trap cleanup_preflight EXIT
for _ in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8012/api/health >/dev/null; then
    break
  fi
  if ! kill -0 "$preflight_pid" >/dev/null 2>&1; then
    cat "$preflight_log" >&2
    exit 1
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:8012/api/health >/dev/null
if [[ "$(curl --fail --silent http://127.0.0.1:8012/api/health | jq -er '.site_release.release_sha256')" != "$approved_release_sha" ]]; then
  printf 'Candidate API loaded the wrong release identity\n' >&2
  exit 1
fi
"$release_root/scripts/validate_gestalt_serving_contract.sh" \
  http://127.0.0.1:8012
cleanup_preflight
trap - EXIT

backup_root="$STATE_ROOT/promotion-backups/$(date -u +%Y%m%dT%H%M%SZ)-$approved_release_sha"
sudo install -d -o "$USER" -g "$(id -gn)" "$backup_root"
sudo cp /etc/systemd/system/gestalt-web.service "$backup_root/gestalt-web.service"
publisher_status="$(
  sudo env PYTHONPATH="$release_root/src" \
    "$PROD_RELEASE/.venv/bin/python" -P \
    -m nba_lineup_model.season.release_publisher status \
    --live-pointer "$LIVE_POINTER"
)"
active_generation="$(jq -er '.generation // 0' <<<"$publisher_status")"
active_operation_id="$(jq -r '.operation_id // empty' <<<"$publisher_status")"
operation_request="$PUBLISHER_ROOT/journal/$publisher_operation_id/0001-requested.json"
if [[ -f "$operation_request" ]]; then
  expected_generation="$(jq -er '.expected_previous_generation' "$operation_request")"
else
  expected_generation="$active_generation"
fi
if [[ "$active_operation_id" == "$publisher_operation_id" ]]; then
  old_release_path="$(jq -r '.previous_release_path // empty' <<<"$publisher_status")"
else
  old_release_path="$(jq -r '.release_path // empty' <<<"$publisher_status")"
fi
if [[ -z "$old_release_path" ]]; then
  printf 'Production promotion requires an existing rollback release\n' >&2
  exit 1
fi
old_web_target="$(readlink "$PUBLIC_WEB" 2>/dev/null || true)"
canonical_schedule_target="$(readlink -f "$SCHEDULE_CURRENT" 2>/dev/null || true)"
test -d "$canonical_schedule_target"
old_code_target="$(readlink -f "$CODE_CURRENT" 2>/dev/null || true)"
mutation_started=0
publisher_activated=0

atomic_symlink() {
  local target="$1"
  local link="$2"
  local temporary="${link}.tmp.$$"
  sudo ln -s "$target" "$temporary"
  sudo mv -Tf "$temporary" "$link"
}

# The old shared writer is permanently deprecated. Promotion fails closed
# rather than preserving or restoring any historical operator state.
legacy_dagster_services=(
  gestalt-dagster-code-server
  gestalt-dagster-webserver
  gestalt-dagster-daemon
)
for service in "${legacy_dagster_services[@]}"; do
  if [[ "$(systemctl is-active "$service" || true)" == "active" ]]; then
    printf 'Production promotion blocked: deprecated service is active: %s\n' "$service" >&2
    exit 1
  fi
done
collector_services=(
  gestalt-collection-code-server
  gestalt-collection-webserver
  gestalt-collection-daemon
)
for service in "${collector_services[@]}"; do
  if [[ "$(systemctl is-active "$service" || true)" != "active" ]]; then
    printf 'Production promotion blocked: protected collector is not active: %s\n' "$service" >&2
    exit 1
  fi
done

rollback() {
  local status=$?
  trap - ERR
  set +e
  if [[ "$mutation_started" == "1" ]]; then
    sudo systemctl stop \
      gestalt-dagster-daemon \
      gestalt-dagster-webserver \
      gestalt-dagster-code-server
    if [[ "$publisher_activated" == "1" ]]; then
      rollback_status="$(
        sudo env PYTHONPATH="$release_root/src" \
          "$PROD_RELEASE/.venv/bin/python" -P \
          -m nba_lineup_model.season.release_publisher status \
          --live-pointer "$LIVE_POINTER"
      )"
      rollback_generation="$(jq -er '.generation' <<<"$rollback_status")"
      sudo env PYTHONPATH="$release_root/src" \
        "$PROD_RELEASE/.venv/bin/python" -P \
        -m nba_lineup_model.season.release_publisher activate \
        --release "$old_release_path" \
        --live-pointer "$LIVE_POINTER" \
        --publisher-root "$PUBLISHER_ROOT" \
        --environment prod \
        --operation-id "$publisher_operation_id-script-rollback-$rollback_generation" \
        --expected-previous-generation "$rollback_generation" \
        --reason "rollback failed production promotion $publisher_operation_id" \
        --approval-reference "$PROD_APPROVAL_REFERENCE" \
        --service gestalt-web \
        --api-root http://127.0.0.1:8010 >/dev/null
    fi
    if [[ -n "$old_web_target" ]]; then
      atomic_symlink "$old_web_target" "$PUBLIC_WEB"
    fi
    if [[ -n "$old_code_target" ]]; then
      atomic_symlink "$old_code_target" "$CODE_CURRENT"
    fi
    sudo cp "$backup_root/gestalt-web.service" /etc/systemd/system/gestalt-web.service
    sudo systemctl daemon-reload
    sudo systemctl restart gestalt-web
    sudo nginx -t && sudo systemctl reload nginx
  fi
  printf 'Production promotion failed; previous release restored\n' >&2
  exit "$status"
}
trap rollback ERR

mutation_started=1
sudo systemctl stop \
  gestalt-dagster-daemon \
  gestalt-dagster-webserver \
  gestalt-dagster-code-server
sudo install -m 0644 "$release_root/deploy/gestalt-web.service" \
  /etc/systemd/system/gestalt-web.service
sudo systemctl daemon-reload
sudo env PYTHONPATH="$release_root/src" \
  "$PROD_RELEASE/.venv/bin/python" -P \
  -m nba_lineup_model.season.release_publisher activate \
  --release "$candidate" \
  --live-pointer "$LIVE_POINTER" \
  --publisher-root "$PUBLISHER_ROOT" \
  --environment prod \
  --operation-id "$publisher_operation_id" \
  --expected-previous-generation "$expected_generation" \
  --reason "approved dev-to-production promotion $approved_release_sha" \
  --approval-reference "$PROD_APPROVAL_REFERENCE" \
  --service gestalt-web \
  --api-root http://127.0.0.1:8010 >/dev/null
publisher_activated=1
atomic_symlink "$LIVE_POINTER/web" "$PUBLIC_WEB"
atomic_symlink "$release_root" "$CODE_CURRENT"
sudo nginx -t
sudo systemctl reload nginx

for _ in {1..60}; do
  if curl --fail --silent http://127.0.0.1:8010/api/health >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error http://127.0.0.1:8010/api/health >/dev/null
if [[ "$(readlink -f "$SCHEDULE_CURRENT")" != "$canonical_schedule_target" ]]; then
  printf 'Production promotion changed the canonical live-observation pointer\n' >&2
  false
fi
if ! systemctl show -p Environment --value gestalt-web | grep -q \
  'GESTALT_LIVE_SCORE_STATE_DIR=/var/lib/nba-gestalt/schedule-serving/current'; then
  printf 'Production reader is not bound to the canonical live-observation root\n' >&2
  false
fi
production_release_sha="$(curl --fail --silent http://127.0.0.1:8010/api/health | jq -er '.site_release.release_sha256')"
if [[ "$production_release_sha" != "$approved_release_sha" ]]; then
  printf 'Production loaded release %s instead of %s\n' \
    "$production_release_sha" "$approved_release_sha" >&2
  false
fi
if [[ "$(git -C "$(readlink -f "$CODE_CURRENT")" rev-parse HEAD)" != "$prod_commit" ]]; then
  printf 'Production writer code pointer does not match reader revision\n' >&2
  false
fi
for service in "${legacy_dagster_services[@]}"; do
  if [[ "$(systemctl is-active "$service" || true)" == "active" ]]; then
    printf 'Deprecated writer service was unexpectedly started: %s\n' "$service" >&2
    false
  fi
done
for service in "${collector_services[@]}"; do
  if [[ "$(systemctl is-active "$service" || true)" != "active" ]]; then
    printf 'Protected collector failed after production cutover: %s\n' "$service" >&2
    false
  fi
  if ! systemctl show -p Environment --value "$service" | grep -q \
    'GESTALT_SCHEDULE_STATE_DIR=/var/lib/nba-gestalt/schedule-serving/current'; then
    printf 'Protected collector has the wrong schedule state root: %s\n' "$service" >&2
    false
  fi
done
PYTHONPATH="$release_root/src" \
  "$PROD_RELEASE/.venv/bin/python" -P \
  -m nba_lineup_model.season.schedule_artifact_contract "$SCHEDULE_CURRENT"
"$release_root/scripts/validate_gestalt_serving_contract.sh" \
  http://127.0.0.1:8010
"$release_root/scripts/compare_gestalt_data_coverage.sh" \
  http://127.0.0.1:8011 \
  http://127.0.0.1:8010
curl --fail --silent --show-error \
  http://127.0.0.1:8010/api/reports/site-release >/dev/null

trap - ERR
printf 'Promoted approved GESTALT release %s at code %s\n' \
  "$approved_release_sha" "$release_id"
