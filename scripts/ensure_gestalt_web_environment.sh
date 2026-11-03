#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_RUNTIME_ROOT="${GESTALT_WEB_RUNTIME_ROOT:-/opt/nba-gestalt-web}"
REQUIREMENTS_ROOT="${GESTALT_REQUIREMENTS_ROOT:-$REPO_ROOT/deploy}"
VALIDATOR="${GESTALT_RUNTIME_VALIDATOR:-$REPO_ROOT/scripts/validate_gestalt_runtime_environment.py}"
ACTIVATE=true

if [[ "${1:-}" == "--no-activate" ]]; then
  ACTIVATE=false
  shift
fi
if [[ "$#" -ne 0 ]]; then
  printf 'Usage: %s [--no-activate]\n' "$0" >&2
  exit 2
fi

UV_BIN="${GESTALT_UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
test -x "$UV_BIN" || { printf 'uv executable not found\n' >&2; exit 1; }

WEB_REQUIREMENTS="$REQUIREMENTS_ROOT/web-requirements.txt"
test -f "$WEB_REQUIREMENTS"
test -x "$VALIDATOR"

requirements_sha="$($VALIDATOR --hash-only --requirements "$WEB_REQUIREMENTS")"
environments="$WEB_RUNTIME_ROOT/environments"
target="$environments/requirements_sha256=$requirements_sha"
current="$WEB_RUNTIME_ROOT/current"
mkdir -p "$environments"

exec 9>"$WEB_RUNTIME_ROOT/.environment-build.lock"
flock 9

runtime_is_valid() {
  [[ -x "$target/bin/python" && -x "$target/bin/uvicorn" ]] && \
    "$target/bin/uvicorn" --version >/dev/null 2>&1
}

if ! runtime_is_valid; then
  if [[ -e "$target" ]]; then
    mv "$target" "$environments/.invalid-requirements_sha256=$requirements_sha.$(date +%s).$$"
  fi
  staging="$environments/.requirements_sha256=$requirements_sha.tmp.$$"
  cleanup() {
    if [[ -d "$staging" ]]; then
      rm -rf -- "$staging"
    fi
  }
  trap cleanup EXIT
  "$UV_BIN" venv --relocatable --python 3.12 "$staging"
  "$UV_BIN" pip install --python "$staging/bin/python" -r "$WEB_REQUIREMENTS"
  "$staging/bin/python" "$VALIDATOR" \
    --profile web \
    --requirements "$WEB_REQUIREMENTS" \
    > "$staging/gestalt-runtime-manifest.json"
  mv "$staging" "$target"
  trap - EXIT
fi
runtime_is_valid || { printf 'Uvicorn entry point failed after runtime publication\n' >&2; exit 1; }

"$target/bin/python" "$VALIDATOR" \
  --profile web \
  --requirements "$WEB_REQUIREMENTS" \
  > "$target/gestalt-runtime-manifest.json"

if [[ "$ACTIVATE" == true ]]; then
  temporary="$WEB_RUNTIME_ROOT/.current.tmp.$$"
  ln -s "$target" "$temporary"
  mv -Tf "$temporary" "$current"
fi

printf '%s\n' "$target"
