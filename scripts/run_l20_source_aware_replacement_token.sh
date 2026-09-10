#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
exec uv run nba-evaluate-l20-source-aware-preseason-minute-share --replacement-token \
  > artifacts/logs/l20-source-aware-replacement-token.log \
  2>&1
