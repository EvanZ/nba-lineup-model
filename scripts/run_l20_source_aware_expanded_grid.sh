#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
exec uv run nba-evaluate-l20-source-aware-preseason-minute-share \
  > artifacts/logs/l20-source-aware-v03-expanded.log \
  2>&1
