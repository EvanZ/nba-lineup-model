#!/usr/bin/env bash
# Native, signed AWS CLI for restricted workers; never replace collectors' CLI.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
version=2.35.21
install_root="/opt/nba-gestalt-tools/aws-cli/$version"
binary="$install_root/bin/aws"
case "$(uname -m)" in
  aarch64|x86_64) architecture=$(uname -m) ;;
  *) echo "Unsupported AWS CLI architecture" >&2; exit 1 ;;
esac
if [[ ! -x "$binary" ]]; then
  task_tmp=$(mktemp -d /tmp/gestalt-worker-aws.XXXXXXXX)
  trap '[[ "$task_tmp" == /tmp/gestalt-worker-aws.* ]] && rm -rf -- "$task_tmp"' EXIT
  url="https://awscli.amazonaws.com/awscli-exe-linux-$architecture-$version.zip"
  curl --fail --silent --show-error --location "$url" -o "$task_tmp/aws.zip"
  curl --fail --silent --show-error --location "$url.sig" -o "$task_tmp/aws.zip.sig"
  mkdir -m 700 "$task_tmp/gnupg"
  gpg --batch --homedir "$task_tmp/gnupg" --import "$repo_root/deploy/aws-cli-public-key.asc" >&2
  fingerprint=$(gpg --batch --homedir "$task_tmp/gnupg" --with-colons --fingerprint | awk -F: '$1=="fpr" {print $10;exit}')
  [[ "$fingerprint" == FB5DB77FD5C118B80511ADA8A6310ACC4672475C ]]
  gpg --batch --homedir "$task_tmp/gnupg" --verify "$task_tmp/aws.zip.sig" "$task_tmp/aws.zip" >&2
  unzip -q "$task_tmp/aws.zip" -d "$task_tmp"
  sudo -n "$task_tmp/aws/install" --install-dir "$install_root/runtime" --bin-dir "$install_root/bin" >&2
fi
# Match the worker's no-new-privileges restriction before it can queue work.
result=$(setpriv --no-new-privs "$binary" --version)
[[ "$result" == "aws-cli/$version "* ]]
echo "$result" >&2
echo "$install_root/bin"
