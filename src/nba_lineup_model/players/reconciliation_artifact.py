"""Immutable, recoverable results of the in-season roster reconciliation job."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.storage.artifact_store import (
    ArtifactReference, ObjectDigest, resolve_object, resolve_storage_manifest,
)

OUTPUTS = {'rankings': 'rankings.parquet', 'profiles': 'profiles.parquet',
           'minutes': 'minutes.json'}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Reconciliation file escapes its artifact')
    return path


def reference(manifest: Mapping) -> ArtifactReference:
    identity = manifest['identity']
    return ArtifactReference(artifact_kind='roster_reconciliation',
        season=identity['season'], game_date_pacific=date.fromisoformat(identity['effective_through']),
        content_id='reconciliation_sha256=' + manifest['reconciliation_sha256'])


def verify(root: Path) -> dict:
    manifest = read_json(root / 'manifest.json')
    identity = manifest['identity']
    if manifest['reconciliation_sha256'] != hashlib.sha256(canonical_json(identity)).hexdigest():
        raise ValueError('Reconciliation identity checksum mismatch')
    if identity['schema_version'] != 1:
        raise ValueError('Unsupported reconciliation schema')
    reference(manifest)
    files = identity['files']
    for name in OUTPUTS.values():
        if name not in files:
            raise ValueError('Reconciliation is missing a complete output catalog')
    for name, expected in files.items():
        path = root / safe_relative(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected['size_bytes'] or file_sha256(path) != expected['sha256']:
            raise ValueError(f'Reconciliation object failed verification: {name}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'manifest.json'}:
        raise ValueError('Reconciliation inventory differs from its sealed files')
    events = identity['introductions']
    ids = [int(event['player_id']) for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError('Player is initialized more than once in the selected history')
    for event in events:
        date.fromisoformat(event['effective_date'])
        discovered=datetime.fromisoformat(event['discovered_at'].replace('Z','+00:00'))
        if discovered.tzinfo is None:
            raise ValueError('Introduction discovery timestamp must include a timezone')
        if not event['discovered_at'] or not event['evidence'] or not event['authorization']:
            raise ValueError('Player introduction is missing discovery/evidence/authorization')
        for name in [*event['initialization'].values(), *event['evidence'].get('files', [])]:
            if name not in files:
                raise ValueError('Introduction initialization is missing from the inventory')
    return manifest


def seal(root: Path, identity: dict) -> dict:
    if (root / 'manifest.json').exists():
        raise ValueError('Cannot rewrite a sealed reconciliation artifact')
    files = {str(p.relative_to(root)): {'sha256': file_sha256(p), 'size_bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file()}
    identity = {**identity, 'schema_version': 1, 'files': files}
    manifest = {'identity': identity,
                'reconciliation_sha256': hashlib.sha256(canonical_json(identity)).hexdigest()}
    (root / 'manifest.json').write_bytes(canonical_json(manifest))
    return verify(root)


def publish(root: Path, *, cache_root: Path, lake_uri: str, client=None):
    from nba_lineup_model.season.durable_artifact_storage import publish_verified_bundle
    manifest = verify(root)
    dependencies = tuple(ArtifactReference.from_dict(x) for x in manifest['identity']['dependencies'])
    for dependency in dependencies:
        resolve_storage_manifest(reference=dependency,cache_root=cache_root,client=client,lake_uri=lake_uri)
    return publish_verified_bundle(bundle=root, reference=reference(manifest),
        dependencies=dependencies, cache_root=cache_root, client=client, lake_uri=lake_uri)


def restore(ref: ArtifactReference, *, cache_root: Path, lake_uri: str, client=None) -> Path:
    """Recover every file by its immutable reference; no former local paths are read."""
    storage = resolve_storage_manifest(reference=ref, cache_root=cache_root,
                                       client=client, lake_uri=lake_uri)
    root = cache_root / 'reconciliations' / ref.content_id
    root.mkdir(parents=True, exist_ok=True)
    for name, entry in storage['objects'].items():
        target = root / safe_relative(name)
        digest = ObjectDigest(entry['sha256'], entry['size_bytes'])
        cached = resolve_object(uri=entry['s3_uri'], digest=digest,
            cache_root=cache_root / 'objects', client=client, lake_uri=lake_uri)
        if target.exists():
            if file_sha256(target) != digest.sha256:
                raise ValueError('Recovered reconciliation cache was modified')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
    manifest = verify(root)
    if reference(manifest) != ref:
        raise ValueError('Recovered reconciliation has the wrong identity')
    return root


def selection_name(environment: str) -> str:
    if environment not in {'dev', 'prod'}:
        raise ValueError('Reconciliation requires an explicit dev/prod environment')
    return environment + '-rosters'


def selected(*, environment: str, cache_root: Path, lake_uri: str, client=None) -> Path | None:
    from nba_lineup_model.storage.environment_selection import read_environment_selection
    selection = read_environment_selection(selection_name(environment), client=client, lake_uri=lake_uri)
    if selection is None:
        return None
    ref = ArtifactReference.from_dict(selection['current']['release'])
    if ref.artifact_kind != 'roster_reconciliation':
        raise ValueError('Roster selection refers to a different artifact kind')
    return restore(ref, cache_root=cache_root, lake_uri=lake_uri, client=client)


def accept(root: Path, *, environment: str, state_root: Path, cache_root: Path,
           lake_uri: str, client=None, approval_reference: str | None = None) -> dict:
    """Publish before selecting; the authoritative selection is also durable."""
    import fcntl
    from nba_lineup_model.storage.environment_selection import (
        activate_environment_selection, read_environment_selection,
    )
    if environment == 'prod' and not approval_reference:
        raise ValueError('Production reconciliation activation requires its approved deployment binding')
    manifest = verify(root)
    ref = reference(manifest)
    namespace = selection_name(environment)
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / 'selection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = read_environment_selection(namespace, client=client, lake_uri=lake_uri)
        current = active['current'] if active else None
        previous = current['release'] if current else None
        if previous not in (manifest['identity']['parent_reference'], ref.as_dict()):
            raise ValueError('Reconciliation selection changed; refusing stale publication')
        result = publish(root, cache_root=cache_root, lake_uri=lake_uri, client=client)
        activation = activate_environment_selection(environment=namespace, release=ref,
            operation_id=manifest['reconciliation_sha256'], reason='Verified in-season roster reconciliation',
            expected_previous_generation=current['generation'] if current else 0,
            approval_reference=approval_reference, client=client, lake_uri=lake_uri)
        payload = {'reference': ref.as_dict(), 'storage_manifest_sha256': result.storage_manifest_sha256,
                   'generation': activation.generation, 'selection_uri': activation.selection_uri}
        temporary = state_root / 'selection.json.tmp'
        temporary.write_bytes(canonical_json(payload))
        os.replace(temporary, state_root / 'selection.json')
        return payload
