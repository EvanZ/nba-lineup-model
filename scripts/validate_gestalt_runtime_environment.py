#!/usr/bin/env python3
"""Validate that a GESTALT release environment matches its runtime profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from importlib import metadata
from pathlib import Path


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


TRAINING_ONLY = {
    "catboost",
    "lightning",
    "mlflow",
    "prefect",
    "pytorch-lightning",
    "torch",
    "torchmetrics",
}
FORBIDDEN_PREFIXES = ("nvidia-",)
REQUIRED = {
    "web": {
        "fastapi",
        "httpx",
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "scikit-learn",
        "scipy",
        "uvicorn",
    },
    "orchestration": {
        "boto3",
        "botocore",
        "dagster",
        "dagster-webserver",
        "duckdb",
        "fastapi",
        "httpx",
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "pypdf",
        "scikit-learn",
        "scipy",
        "uvicorn",
    },
}
PROFILE_FORBIDDEN = {
    "web": {"dagster", "dagster-webserver", "duckdb", "matplotlib", "pypdf"},
    "orchestration": {"matplotlib"},
}


def requirements_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[canonical_name(name)] = distribution.version
    return packages


def validate_inventory(profile: str, packages: Mapping[str, str]) -> list[str]:
    installed = {canonical_name(name) for name in packages}
    errors: list[str] = []
    missing = sorted(REQUIRED[profile] - installed)
    if missing:
        errors.append(f"missing required packages: {', '.join(missing)}")
    forbidden = sorted(
        name
        for name in installed
        if name in TRAINING_ONLY
        or name in PROFILE_FORBIDDEN[profile]
        or name.startswith(FORBIDDEN_PREFIXES)
    )
    if forbidden:
        errors.append(f"forbidden packages installed: {', '.join(forbidden)}")
    return errors


def environment_bytes(root: Path) -> int:
    total = 0
    for current, _, files in os.walk(root):
        directory = Path(current)
        for filename in files:
            try:
                total += (directory / filename).stat().st_size
            except FileNotFoundError:
                continue
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(REQUIRED))
    parser.add_argument("--requirements", type=Path, nargs="+")
    parser.add_argument("--hash-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.requirements:
        raise SystemExit("--requirements is required")
    requirements_hash = requirements_sha256(args.requirements)
    if args.hash_only:
        print(requirements_hash)
        return 0
    if args.profile is None:
        raise SystemExit("--profile is required unless --hash-only is used")

    packages = installed_packages()
    errors = validate_inventory(args.profile, packages)
    report = {
        "environment_bytes": environment_bytes(Path(sys.prefix)),
        "environment_root": str(Path(sys.prefix).resolve()),
        "package_count": len(packages),
        "profile": args.profile,
        "requirements_sha256": requirements_hash,
        "status": "failed" if errors else "clean",
        "validation_errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
