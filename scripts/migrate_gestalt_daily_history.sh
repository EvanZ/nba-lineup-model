#!/usr/bin/env bash
# Checked one-time history migration; no model runs, alert edits or site publication.
# Arguments: sealed-package package-sha256 [daily_history_migration arguments]
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
test "$#" -ge 3 || { printf 'Expected sealed package, checksum and migration options\n' >&2; exit 2; }
package="$1"
checksum="$2"
shift 2
requirements_sha="$("$package/code/scripts/validate_gestalt_runtime_environment.py" \
  --hash-only --requirements "$package/code/deploy/web-requirements.txt" \
  "$package/code/deploy/orchestration-requirements.txt")"
runtime="/opt/nba-gestalt-orchestration/environments/requirements_sha256=$requirements_sha"
test -x "$runtime/bin/python"
# -B applies before importing any sealed module, including the seal verifier.
sudo -n "$runtime/bin/python" -B -c '
import sys, runpy
from pathlib import Path
package = Path(sys.argv.pop(1))
checksum = sys.argv.pop(1)
sys.path.insert(0, str(package / "code/src"))
from nba_lineup_model.season.sealed_code_package import verify_sealed_code_package
if verify_sealed_code_package(package)["identity"]["package_sha256"] != checksum:
    raise ValueError("Migration package checksum differs")
runpy.run_module("nba_lineup_model.orchestration.daily_history_migration", run_name="__main__")
' "$package" "$checksum" "$@"
