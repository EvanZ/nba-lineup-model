#!/usr/bin/env bash
# Restart the local NBA GESTALT API and Vite application on their fixed ports.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT=8001
WEB_PORT=5174
LOG_DIR="$ROOT_DIR/artifacts/logs"
API_LOG="$LOG_DIR/gestalt-api-local.log"
WEB_LOG="$LOG_DIR/gestalt-web-local.log"

stop_listener() {
  local port="$1"
  local pids
  pids="$(lsof -ti "tcp:${port}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return
  fi

  echo "Stopping listener on port ${port}: ${pids}"
  kill $pids

  for _ in {1..20}; do
    if ! lsof -ti "tcp:${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      return
    fi
    sleep 0.25
  done

  echo "Port ${port} did not stop after five seconds." >&2
  exit 1
}

wait_for_url() {
  local url="$1"
  local label="$2"

  for _ in {1..40}; do
    if curl --silent --fail --output /dev/null "$url"; then
      echo "${label} ready: ${url}"
      return
    fi
    sleep 0.25
  done

  echo "${label} did not become ready. See ${API_LOG} and ${WEB_LOG}." >&2
  exit 1
}

mkdir -p "$LOG_DIR"
stop_listener "$API_PORT"
stop_listener "$WEB_PORT"

echo "Starting NBA GESTALT API on port ${API_PORT}"
(
  cd "$ROOT_DIR"
  nohup uv run nba-gestalt-api --port "$API_PORT" >"$API_LOG" 2>&1 &
)

echo "Starting NBA GESTALT frontend on port ${WEB_PORT}"
(
  cd "$ROOT_DIR/apps/lineup-explorer"
  nohup npm run dev >"$WEB_LOG" 2>&1 &
)

wait_for_url "http://127.0.0.1:${API_PORT}/api/health" "API"
wait_for_url "http://127.0.0.1:${WEB_PORT}" "Frontend"

echo "NBA GESTALT is running at http://127.0.0.1:${WEB_PORT}"
echo "API log: ${API_LOG}"
echo "Frontend log: ${WEB_LOG}"
