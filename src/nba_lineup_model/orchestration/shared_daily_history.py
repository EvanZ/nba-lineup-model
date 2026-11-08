"""Stage a shared daily instance without changing either source history.

Sources must be quiescent at cutover. Original databases, sensor/tick state and
logs are archived verbatim (SQLite online backups); run bodies and events retain
their original identity. New automation starts stopped and is explicitly enabled
per code location after validation. This module never changes running services.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from nba_lineup_model.orchestration.daily_history_migration import (
    backup, file_sha, migrated_tags, migration_bindings, record_digest, rows,
    verify_migration,
)
from nba_lineup_model.orchestration.daily_instance import DailyInstance
from nba_lineup_model.orchestration.daily_work import atomic_json, digest, read_object


def shared_tags(binding, record, source, archive):
    """Reuse audited prior bindings; never reinterpret an earlier migration."""
    prior = migration_bindings(source).get(record['run_id'])
    original = json.loads(record['run_body']).get('tags') or {}
    if prior:
        if prior['record_sha256'] != record_digest(record):
            raise ValueError('Previously migrated run changed')
        tags = {**original, **prior['tags']}
    elif original.get('gestalt/environment'):
        tags = original
    elif record['pipeline_name'] == 'backfill_official_history':
        # Preserve deprecated work and its failures, without pretending a new
        # pipeline job can recover a different operation.
        body = json.loads(record['run_body'])
        configs = [v['config'] for v in body['run_config']['ops'].values()
                   if 'config_path' in v.get('config', {})]
        if len(configs) != 1:
            raise ValueError('Ambiguous historical backfill configuration')
        config = read_object(Path(configs[0]['config_path']))
        if Path(config['environment_root']).resolve() != binding.root:
            raise ValueError('Historical backfill belongs to another environment')
        tags = {**original, 'gestalt/environment': binding.environment,
                'gestalt/environment_root': str(binding.root),
                'gestalt/recovery_key': 'backfill_official_history:legacy:' + digest(configs[0]),
                'gestalt/work_id': 'legacy-backfill:' + digest(configs[0])}
        evidence = archive / 'request-evidence' / record['run_id']
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(configs[0]['config_path'], evidence / 'config_path.json')
    else:
        tags = {**original, **migrated_tags(binding, record, archive)}
    if tags.get('gestalt/environment') != binding.environment or tags.get('gestalt/environment_root') != str(binding.root):
        raise ValueError('Historical run has conflicting environment tags')
    return tags


def migrate_shared(home: Path, environments: dict[str, Path], sources: dict[str, Path],
                   template: Path, approval_reference: str) -> dict:
    """Create an immutable, verified migration; refuse moving or conflicting inputs."""
    from dagster import DagsterInstance
    if not approval_reference.strip():
        raise PermissionError('An explicit migration approval reference is required')
    home = home.absolute()
    if home != home.resolve() or not environments or set(environments) - {'dev', 'prod'}:
        raise ValueError('Invalid shared history destination or environment selection')
    if set(sources) != set(environments):
        raise ValueError('Each registered environment needs an explicit source history')
    roots = {name: root.resolve(strict=True) for name, root in environments.items()}
    sources = {name: source.resolve(strict=True) for name, source in sources.items()}
    if any(source != roots[name] / 'daily-dagster' for name, source in sources.items()):
        raise ValueError('Migration must use each environment permanent history')
    if any(home.is_relative_to(root) or root.is_relative_to(home) for root in roots.values()):
        raise ValueError('Shared history must be outside environment data roots')
    if home.exists():
        result = verify_migration(home)
        if result['sources'] != sorted(map(str, sources.values())):
            raise ValueError('Existing migration has a different source selection')
        prior = migration_bindings(home)
        for name, root in roots.items():
            if DailyInstance.read(home, name).root != root:
                raise ValueError('Existing shared environment differs')
            expected = {rid: item['record_sha256'] for rid, item in prior.items()
                        if item['source'] == str(sources[name])}
            if {r['run_id']: record_digest(r) for r in rows(sources[name] / 'history/runs.db')} != expected:
                raise ValueError('Source history changed since this migration was prepared')
        return result
    home.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.shared-daily-stage-', dir=home.parent))
    atomic_json(stage / 'environment.json', {'schema_version': 2,
                'environments': {name: str(root) for name, root in roots.items()}}, immutable=True)
    for name in roots:
        DailyInstance.read(stage, name)
    shutil.copyfile(template, stage / 'dagster.yaml')
    with DagsterInstance.from_config(str(stage)):
        pass
    collected, snapshots, evidence, source_records = {}, {}, {}, {}
    tag_rows = {}
    for name, source in sorted(sources.items()):
        binding = DailyInstance(name, roots[name], stage)
        archive = stage / 'history-sources' / name
        for path in source.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'History source contains a symlink: {path}')
            if not path.is_file() or path.name.endswith(('.db-wal', '.db-shm')):
                continue
            target = archive / path.relative_to(source)
            if path.suffix == '.db':
                backup(path, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        database = archive / 'history/runs.db'
        records = rows(database)
        if any(r['status'] not in {'SUCCESS', 'FAILURE', 'CANCELED'} for r in records):
            raise ValueError('Cannot migrate unfinished runs')
        if rows(database, 'bulk_actions') or rows(database, 'backfill_tags'):
            raise ValueError('Bulk Dagster backfills require a reviewed migration')
        source_records[name] = {r['run_id']: record_digest(r) for r in records}
        for item in rows(database, 'snapshots'):
            content = {k:v for k,v in item.items() if k != 'id'}
            key = item['snapshot_id']
            if key in snapshots and snapshots[key] != content:
                raise ValueError('Conflicting Dagster snapshot')
            snapshots[key] = content
        for item in rows(database, 'run_tags'):
            tag_rows[(item['run_id'], item['key'])] = {k:v for k,v in item.items() if k != 'id'}
        for record in records:
            rid = record['run_id']
            if rid in collected:
                raise ValueError('Duplicate run ID across environments')
            collected[rid] = record
            tags = shared_tags(binding, record, archive, archive)
            for key, value in tags.items():
                identity = (rid, key)
                if identity in tag_rows and tag_rows[identity]['value'] != value:
                    raise ValueError('Conflicting historical tag index')
                tag_rows[identity] = {'run_id': rid, 'key': key, 'value': value}
            logs = archive / 'history/runs' / f'{rid}.db'
            events = rows(logs, 'event_logs')
            backup(logs, stage / 'history/runs' / f'{rid}.db')
            outputs = archive / 'storage' / rid
            if outputs.exists():
                shutil.copytree(outputs, stage / 'storage' / rid)
            evidence[rid] = {'record_sha256': record_digest(record), 'tags': tags,
                'event_sha256': digest([{k:v for k,v in r.items() if k != 'id'} for r in events]),
                'source': str(source)}
    with sqlite3.connect(stage / 'history/runs.db') as target:
        for table, records in (('snapshots', snapshots.values()),
                ('runs', sorted(collected.values(), key=lambda r: (r['create_timestamp'], r['run_id']))),
                ('run_tags', tag_rows.values())):
            for record in records:
                item = {key:value for key,value in record.items() if key != 'id'}
                columns = ','.join(f'"{key}"' for key in item)
                target.execute(f'INSERT INTO "{table}" ({columns}) VALUES ({",".join("?" for _ in item)})', tuple(item.values()))
    identity = {'schema_version': 2, 'sources': sorted(map(str, sources.values())),
        'environments': {name: str(root) for name, root in roots.items()},
        'approval_reference': approval_reference, 'runs': evidence,
        'archive_files': {str(p.relative_to(stage)): file_sha(p)
                          for p in (stage / 'history-sources').rglob('*') if p.is_file()}}
    atomic_json(stage / 'history-migration.json', {'identity': identity, 'sha256': digest(identity)}, immutable=True)
    verify_migration(stage)
    for name, source in sources.items():
        if {r['run_id']: record_digest(r) for r in rows(source / 'history/runs.db')} != source_records[name]:
            raise ValueError('Source history changed during migration')
        for rid in source_records[name]:
            original_events = rows(source / 'history/runs' / f'{rid}.db', 'event_logs')
            if digest([{k:v for k,v in row.items() if k != 'id'} for row in original_events]) != evidence[rid]['event_sha256']:
                raise ValueError('Source event history changed during migration')
    if home.exists():
        raise FileExistsError('Shared history appeared during staging')
    os.rename(stage, home)
    return verify_migration(home)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--dev-root', type=Path, required=True)
    parser.add_argument('--prod-root', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--approval-reference', required=True)
    args = parser.parse_args()
    roots = {'dev': args.dev_root, 'prod': args.prod_root}
    print(json.dumps(migrate_shared(args.home, roots,
        {name: root / 'daily-dagster' for name, root in roots.items()},
        args.template, args.approval_reference), indent=2))


if __name__ == '__main__':
    main()
