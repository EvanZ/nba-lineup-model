"""Fail-closed source policy for NBA Gestalt artifacts and network requests."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit


def assert_allowed_url(url: str) -> None:
    host = (urlsplit(url).hostname or "").casefold()
    if any("espn" in part for part in host.split(".")):
        raise ValueError("Forbidden basketball data provider")


def forbidden_references(value: object, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if "espn" in str(key).casefold():
                found.append(child_path)
            if key in {"game_id", "nba_game_id"} and str(child).startswith("40"):
                found.append(child_path + ":non-NBA-game-id")
            found.extend(forbidden_references(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_references(child, path=f"{path}[{index}]"))
    elif isinstance(value, str) and "espn" in value.casefold():
        found.append(path)
    return found


def audit_artifact_sources(root: Path | str, *, files: list[str] | None = None) -> list[str]:
    """Read only: return exact contaminated files, never sanitize them silently."""
    directory = Path(root).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Artifact source audit requires a directory")
    failures = []
    selected = directory.rglob("*") if files is None else (directory / name for name in files)
    for path in sorted(selected):
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Artifact source audit path escapes its root")
        if path.is_symlink():
            raise ValueError(f"Artifact audit does not follow nested symlinks: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(directory))
        if "espn" in relative.casefold() or path.name.startswith("401"):
            failures.append(relative)
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            if forbidden_references(payload):
                failures.append(relative)
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            if any("espn" in name.casefold() for name in schema.names):
                failures.append(relative)
                continue
            import pyarrow as pa
            columns = [field.name for field in schema if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)]
            if columns:
                contaminated = False
                for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
                    for column in batch.columns:
                        if any("espn" in value.casefold() for value in column.to_pylist() if isinstance(value, str)):
                            contaminated = True
                            break
                    if contaminated:
                        break
                if contaminated:
                    failures.append(relative)
    return failures


def require_clean_artifact_sources(root: Path | str) -> None:
    failures = audit_artifact_sources(root)
    if failures:
        raise ValueError("Forbidden source artifacts: " + ", ".join(failures[:20]))
