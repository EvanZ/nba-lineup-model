#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  'ERROR: the shared NBA GESTALT Dagster pipeline is deprecated.' \
  'Do not start gestalt-dagster-code-server, gestalt-dagster-webserver,' \
  'or gestalt-dagster-daemon.' \
  'Protected feeds run in the isolated gestalt-collection-* services.' \
  'The nightly atomic publisher must be deployed as a separately named,' \
  'least-privilege service after its production contract is approved.' >&2
exit 64
