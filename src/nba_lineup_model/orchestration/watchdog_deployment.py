"""Seal monitoring code independently; never activate or modify a site release."""

import argparse
import subprocess
from pathlib import Path

from nba_lineup_model.season.sealed_code_package import (
    materialize_sealed_code_package,
    verify_sealed_code_package,
)


def seal_watchdog(source: Path, output: Path) -> tuple[Path, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    package = output / f"revision={revision}"
    package.mkdir(parents=True, exist_ok=True)
    if (package / "code").exists():
        manifest = verify_sealed_code_package(package)
    else:
        manifest = materialize_sealed_code_package(source, package, expected_revision=revision)
    if manifest["identity"]["git_revision"] != revision:
        raise ValueError("Watchdog package does not match its selected revision")
    return package, manifest["identity"]["package_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package, checksum = seal_watchdog(args.source, args.output)
    print(package, checksum)


if __name__ == "__main__":
    main()
