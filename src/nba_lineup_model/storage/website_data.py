"""S3-only reads for website data using the existing pinned artifact inventories.

Local paths are provenance/lookup keys, never a disk fallback in a serving
session. Producers without a serving session retain their existing file I/O.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
from threading import RLock

_ACTIVE = None

class ServingData:
    def __init__(self, release, *, lake_uri, client):
        self.release = Path(release).resolve()
        self.lake_uri, self.client = lake_uri, client
        self.objects = {}
        self.history = None
        self.registries = {}
        self.registry_lock = RLock()
        self.reads = set()

    def key(self, path):
        p = Path(path)
        return str(p if p.is_absolute() else self.history / p)

    def register(self, root, objects):
        for name, entry in objects.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe S3 serving inventory path')
            key = str(Path(root) / name)
            if key in self.objects and self.objects[key] != entry:
                raise ValueError('Conflicting S3 serving references')
            self.objects[key] = entry

    def fetch(self, entry):
        from nba_lineup_model.season.durable_observations import _read
        try:
            raw, sha = _read(self.client, entry['s3_uri'], 0, entry['sha256'], entry['size_bytes'])
        except ValueError as error:
            raise ValueError('Pinned S3 serving object changed: ' + entry['s3_uri']) from error
        if sha != entry['sha256'] or len(raw) != entry['size_bytes']:
            raise ValueError('Pinned S3 serving object changed: ' + entry['s3_uri'])
        self.reads.add(entry['s3_uri'])
        return raw

    def read(self, path):
        key = self.key(path)
        if key not in self.objects:
            raise FileNotFoundError('No selected S3 object for website data: ' + key)
        return self.fetch(self.objects[key])

    def manifest(self, ref):
        from nba_lineup_model.storage.artifact_store import storage_manifest_uri, verify_storage_manifest
        from nba_lineup_model.season.durable_observations import _read
        uri = storage_manifest_uri(ref, lake_uri=self.lake_uri)
        raw, _ = _read(self.client, uri, 0)
        return verify_storage_manifest(json.loads(raw), expected=ref, lake_uri=self.lake_uri)

    def initialize(self, bootstrap):
        from nba_lineup_model.season.durable_release_storage import production_release_reference
        from nba_lineup_model.storage.artifact_store import ArtifactReference
        ref = production_release_reference(bootstrap)
        transport = self.manifest(ref)
        self.register(self.release, transport['objects'])
        runtime = json.loads(self.read(self.release / 'runtime-manifest.json'))
        if runtime['runtime_sha256'] != bootstrap['runtime_sha256']:
            raise ValueError('S3 release differs from deployment selection')
        bindings = json.loads(self.read(self.release / 'serving-bindings.json'))
        inventory = json.loads(self.read(self.release / 'durable-serving.json'))
        from nba_lineup_model.season.daily_artifact_common import canonical_json
        if hashlib.sha256(canonical_json(inventory)).hexdigest() != runtime['identity']['durable_serving_sha256']:
            raise ValueError('S3 serving inventory differs from release')
        self.observation_sources = inventory.get('observation_sources')
        self.season = runtime['approved_manifest']['identity']['season']
        self.history = Path(inventory['historical']['source_root'])
        self.register(self.history, inventory['historical']['files'])
        pending = [ArtifactReference.from_dict(item['reference']) for item in inventory['daily'].values()]
        seen = set()
        roots = {label: Path(item['source_path']).parents[2] for label,item in inventory['daily'].items()}
        while pending:
            dependency = pending.pop()
            if dependency in seen:
                continue
            seen.add(dependency)
            manifest = self.manifest(dependency)
            label = {'daily_model': 'model', 'daily_data': 'data'}.get(dependency.artifact_kind)
            if label:
                root = roots[label] / f'season={dependency.season}' / f'game_date_pacific={dependency.game_date_pacific}' / dependency.content_id
                self.register(root, manifest['objects'])
            pending.extend(ArtifactReference.from_dict(item) for item in manifest['dependencies'])
        registry = inventory.get('registry')
        if registry:
            self.registries[registry['source_root']] = registry
            self.refresh_registry(registry)
        return runtime, bindings

    def put_registry(self, path, raw):
        from botocore.exceptions import ClientError
        from nba_lineup_model.season.durable_forecast_registry import record_name
        path = Path(path)
        registry = next((item for root,item in self.registries.items() if path.is_relative_to(root)), None)
        if registry is None:
            raise ValueError('Website write is not a selected S3 registry record')
        name = record_name(str(path.relative_to(registry['source_root'])))
        uri = registry['uri']+'/'+name
        bucket, _, key = uri[5:].partition('/')
        sha = hashlib.sha256(raw).hexdigest()
        try:
            self.client.put_object(Bucket=bucket, Key=key, Body=raw,
                Metadata={'sha256':sha}, IfNoneMatch='*')
        except ClientError as error:
            if error.response['Error']['Code'] not in ('PreconditionFailed','412'):
                raise
            head = self.client.head_object(Bucket=bucket, Key=key)
            existing = {'s3_uri':uri,'sha256':head.get('Metadata',{}).get('sha256'),'size_bytes':head['ContentLength']}
            if self.fetch(existing) != raw:
                raise ValueError('Conflicting immutable S3 registry publication') from error
        self.register(registry['source_root'],{name:{'s3_uri':uri,'sha256':sha,'size_bytes':len(raw)}})
        return path

    def refresh_registry(self, registry):
        from nba_lineup_model.season.durable_forecast_registry import record_name
        bucket, prefix = registry['uri'][5:].partition('/')[::2]
        prefix = prefix.rstrip('/') + '/'
        found = {}
        for page in self.client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get('Contents', []):
                name = record_name(item['Key'][len(prefix):])
                existing = self.objects.get(str(Path(registry['source_root']) / name))
                if existing is not None and existing['size_bytes'] == item.get('Size'):
                    found[name] = existing
                    continue
                head = self.client.head_object(Bucket=bucket, Key=item['Key'])
                found[name] = {'s3_uri': registry['uri']+'/'+name,
                               'sha256': head.get('Metadata',{}).get('sha256'), 'size_bytes':head['ContentLength']}
        if any(found.get(name) != entry for name,entry in registry['records'].items()):
            raise ValueError('Required S3 registry record is missing or changed')
        self.register(Path(registry['source_root']), found)

def active():
    return _ACTIVE

def activate(release, bootstrap, *, lake_uri, client):
    global _ACTIVE
    session = ServingData(release, lake_uri=lake_uri, client=client)
    runtime, bindings = session.initialize(bootstrap)
    _ACTIVE = session
    return runtime, bindings

def read_bytes(path):
    return _ACTIVE.read(path) if _ACTIVE else Path(path).read_bytes()

def read_text(path, encoding='utf-8', errors=None):
    return read_bytes(path).decode(encoding, errors or 'strict')

def open_file(path, mode='r', *args, **kwargs):
    if not _ACTIVE:
        return Path(path).open(mode, *args, **kwargs)
    if mode not in ('r', 'rt', 'rb'):
        raise ValueError('Website durable data must not be written to local files')
    raw = read_bytes(path)
    return io.BytesIO(raw) if mode == 'rb' else io.StringIO(raw.decode(kwargs.get('encoding') or 'utf-8'))

def is_file(path):
    return _ACTIVE.key(path) in _ACTIVE.objects if _ACTIVE else Path(path).is_file()

def is_dir(path):
    if not _ACTIVE:
        return Path(path).is_dir()
    prefix = _ACTIVE.key(path).rstrip('/') + '/'
    return any(key.startswith(prefix) for key in _ACTIVE.objects)

def exists(path):
    return is_file(path) or is_dir(path) if _ACTIVE else Path(path).exists()

def glob(path, pattern):
    if not _ACTIVE:
        return Path(path).glob(pattern)
    base = _ACTIVE.key(path).rstrip('/') + '/'
    return iter(Path(key) for key in sorted(_ACTIVE.objects)
                if key.startswith(base) and fnmatch.fnmatchcase(key[len(base):],pattern)
                and ('/' in pattern or '/' not in key[len(base):]))

def parquet(path, *args, **kwargs):
    import pandas as pd
    return pd.read_parquet(io.BytesIO(read_bytes(path)) if _ACTIVE else path, *args, **kwargs)

def parquet_file(path, *args, **kwargs):
    import pyarrow.parquet as pq
    return pq.ParquetFile(io.BytesIO(read_bytes(path)) if _ACTIVE else path, *args, **kwargs)

def joblib_load(path, *args, **kwargs):
    import joblib
    return joblib.load(io.BytesIO(read_bytes(path)) if _ACTIVE else path, *args, **kwargs)


def resolve(path, strict=False):
    return Path(_ACTIVE.key(path)) if _ACTIVE else Path(path).resolve(strict=strict)

def rglob(path, pattern):
    if not _ACTIVE:
        return Path(path).rglob(pattern)
    base = _ACTIVE.key(path).rstrip('/') + '/'
    return iter(Path(key) for key in sorted(_ACTIVE.objects)
                if key.startswith(base) and (fnmatch.fnmatchcase(key[len(base):],pattern)
                or fnmatch.fnmatchcase(Path(key).name,pattern)))
