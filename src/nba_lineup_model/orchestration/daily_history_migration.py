"""Approved one-time migration of complete, stopped dev Dagster run histories.

No discovery and no status rewriting: explicit sources, preserved records,
checksummed source archives, atomic installation, and repeatable verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from nba_lineup_model.orchestration.daily_instance import DailyInstance, request_identity
from nba_lineup_model.orchestration.daily_work import atomic_json, digest, read_object


def rows(path: Path, table="runs") -> list[dict]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def record_digest(record: dict) -> str:
    return digest({key: value for key, value in record.items() if key != "id"})


def backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as original:
        with sqlite3.connect(destination) as copied:
            original.backup(copied)


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def migrated_tags(binding: DailyInstance, record: dict, archive: Path) -> dict:
    body = json.loads(record["run_body"])
    requests = [op["config"] for op in body["run_config"]["ops"].values()
                if "config_path" in op.get("config", {})]
    if len(requests) != 1:
        raise ValueError(f"Run {record['run_id']} has ambiguous pipeline configuration")
    request = requests[0]
    config = read_object(Path(request["config_path"]))
    if Path(config["environment_root"]).resolve() != binding.root:
        raise ValueError(f"Run {record['run_id']} belongs to another environment")
    if config.get("live_pointer") and Path(config["live_pointer"]).absolute().parent.resolve() != binding.root:
        raise ValueError("Historical run has a conflicting live environment")
    work, day, key = request_identity(record["pipeline_name"], request)
    tags = {"gestalt/environment": binding.environment,
            "gestalt/environment_root": str(binding.root), "gestalt/work_id": work,
            "gestalt/game_date": day, "gestalt/recovery_key": key}
    existing = body.get("tags") or {}
    for field in ("gestalt/environment", "gestalt/environment_root", "gestalt/work_id"):
        if field in existing and existing[field] != tags[field]:
            raise ValueError("Historical run tags conflict with its recorded request")
    for name in ("config_path", "spec_path", "approval_path"):
        if request.get(name):
            source = Path(request[name])
            destination = archive / "request-evidence" / record["run_id"] / f"{name}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    return tags


def migration_bindings(home: Path) -> dict:
    path = home / "history-migration.json"
    if not path.exists():
        return {}
    manifest = read_object(path)
    if manifest["sha256"] != digest(manifest["identity"]):
        raise ValueError("History migration manifest checksum differs")
    return manifest["identity"]["runs"]


def verify_migration(home: Path) -> dict:
    manifest = read_object(home / "history-migration.json")
    evidence = migration_bindings(home)
    actual = {row["run_id"]: row for row in rows(home / "history/runs.db")}
    for run_id, original in evidence.items():
        if run_id not in actual or record_digest(actual[run_id]) != original["record_sha256"]:
            raise ValueError(f"Migrated run record differs: {run_id}")
        logs = home / "history/runs" / f"{run_id}.db"
        if digest([{k:v for k,v in r.items() if k != "id"} for r in rows(logs, "event_logs")]) != original["event_sha256"]:
            raise ValueError(f"Migrated event history differs: {run_id}")
    for relative, checksum in manifest["identity"]["archive_files"].items():
        if file_sha(home / relative) != checksum:
            raise ValueError(f"Archived source history differs: {relative}")
    return {"migration_sha256": manifest["sha256"], "migrated_runs": len(evidence),
            "verified": True, "sources": manifest["identity"]["sources"]}


def migrate(environment: str, root: Path, sources: list[Path], template: Path,
            approval_reference: str, owner: str | None = None) -> dict:
    from dagster import DagsterInstance
    import pwd
    account = pwd.getpwnam(owner) if owner else pwd.getpwuid(os.geteuid())
    if not approval_reference.strip():
        raise PermissionError("An explicit migration approval reference is required")
    binding = DailyInstance(environment, root.resolve(strict=True))
    sources = sorted({p.resolve(strict=True) for p in sources})
    if not sources or any(not p.is_relative_to(binding.root) or p == binding.home for p in sources):
        raise ValueError("Migration sources must be explicit histories inside this environment")
    if binding.home.exists():
        result = verify_migration(binding.home)
        if result["sources"] != list(map(str, sources)):
            raise ValueError("Existing migration has a different source selection")
        return result
    # A failed staging attempt cannot replace the permanent instance.
    stage = Path(tempfile.mkdtemp(prefix=".daily-history-migration-", dir=binding.root))
    shutil.copyfile(template, stage / "dagster.yaml")
    with DagsterInstance.from_config(str(stage)):
        pass
    collected = {}
    evidence = {}
    snapshots = {}
    all_tags = []
    source_records = {}
    for index, source in enumerate(sources):
        archive = stage / "history-sources" / str(index)
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"History source contains a symlink: {path}")
            if not path.is_file() or path.name.endswith((".db-wal", ".db-shm")):
                continue
            target = archive / path.relative_to(source)
            if path.suffix == ".db":
                backup(path, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        source_db = archive / "history/runs.db"
        records = rows(source_db)
        if any(r["status"] not in {"SUCCESS", "FAILURE", "CANCELED"} for r in records):
            raise ValueError("Cannot migrate a history with unfinished runs")
        if rows(source_db, "bulk_actions") or rows(source_db, "backfill_tags"):
            raise ValueError("This migration requires separate handling for Dagster bulk backfills")
        source_records[str(source)] = {r["run_id"]:record_digest(r) for r in records}
        for item in rows(source_db, "snapshots"):
            previous = snapshots.get(item["snapshot_id"])
            if previous and {k:v for k,v in previous.items() if k!='id'} != {k:v for k,v in item.items() if k!='id'}:
                raise ValueError("Conflicting Dagster snapshots")
            snapshots[item["snapshot_id"]] = item
        for record in records:
            rid = record["run_id"]
            if rid in collected:
                raise ValueError(f"Duplicate run ID across source histories: {rid}")
            collected[rid] = record
            tags = migrated_tags(binding, record, archive)
            source_logs = archive / "history/runs" / f"{rid}.db"
            events = rows(source_logs, "event_logs")
            backup(source_logs, stage / "history/runs" / f"{rid}.db")
            outputs = archive / "storage" / rid
            if outputs.exists():
                shutil.copytree(outputs, stage / "storage" / rid)
            evidence[rid] = {"record_sha256": record_digest(record), "tags": tags,
                "event_sha256": digest([{k:v for k,v in r.items() if k != "id"} for r in events]),
                "source": str(source)}
        all_tags.extend(rows(source_db, "run_tags"))
    with sqlite3.connect(stage / "history/runs.db") as target:
        for table, records in (("snapshots", snapshots.values()),
                               ("runs", sorted(collected.values(), key=lambda r: (r["create_timestamp"],r["run_id"]))),
                               ("run_tags", all_tags)):
            for record in records:
                item = {key:value for key,value in record.items() if key != "id"}
                columns = ','.join(f'"{key}"' for key in item)
                target.execute(f'INSERT INTO "{table}" ({columns}) VALUES ({",".join("?" for _ in item)})', tuple(item.values()))
    atomic_json(stage / "environment.json", {"schema_version": 1,
                "environment": environment, "environment_root": str(binding.root)}, immutable=True)
    (stage / "history-sources").mkdir(exist_ok=True)
    archive_files = {str(p.relative_to(stage)):file_sha(p)
                     for p in (stage / "history-sources").rglob("*") if p.is_file()}
    identity = {"schema_version": 1, "environment": environment,
                "environment_root": str(binding.root), "sources": list(map(str,sources)),
                "approval_reference": approval_reference, "runs": evidence,
                "archive_files": archive_files}
    atomic_json(stage / "history-migration.json", {"identity":identity,"sha256":digest(identity)}, immutable=True)
    verify_migration(stage)
    # Refuse a moving source; never copy a live run and call it complete.
    for source in sources:
        current = {r["run_id"]:record_digest(r) for r in rows(source / "history/runs.db")}
        if current != source_records[str(source)]:
            raise ValueError("Source history changed during migration")
    if binding.home.exists():
        raise FileExistsError("Permanent instance appeared during migration")
    # Only the newly created staging tree changes ownership; originals are untouched.
    for path in [stage, *stage.rglob("*")]:
        if path.is_symlink():
            raise ValueError("Unexpected symlink in staged history")
        os.chown(path, account.pw_uid, account.pw_gid)
    os.rename(stage, binding.home)
    binding.requests.mkdir(exist_ok=True)
    os.chown(binding.requests, account.pw_uid, account.pw_gid)
    return verify_migration(binding.home)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--environment", choices=("dev", "prod"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--approval-reference", required=True)
    p.add_argument("--owner", required=True, help="Service account that owns the migrated instance")
    args = p.parse_args()
    print(json.dumps(migrate(args.environment,args.root,args.source,args.template,args.approval_reference,args.owner),indent=2))


if __name__ == "__main__":
    main()
