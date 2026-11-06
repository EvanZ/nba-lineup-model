"""Environment-bound entrypoint and dependency selection for roster reconciliation."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from nba_lineup_model.players import reconciliation_artifact as artifacts
from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256


def environment(config):
    if config.environment_root.name not in {'nba-gestalt','nba-gestalt-dev'}:
        raise ValueError('Unregistered reconciliation environment')
    return 'dev' if config.environment_root.name.endswith('-dev') else 'prod'


def captured_request(config, capture_slot):
    """Use one official capture per shared slot; dev only consumes saved captures."""
    from nba_lineup_model.storage.s3_lake import _new_s3_client, _head_object
    from nba_lineup_model.storage.artifact_store import resolve_object, ObjectDigest
    from nba_lineup_model.season.stats_v3_lake import AwsCliImmutableObjectStore
    slot=datetime.fromisoformat(capture_slot.replace('Z','+00:00'))
    if slot.tzinfo is None:raise ValueError('Roster capture slot must have a timezone')
    key=slot.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')
    uri=f'{config.artifact_lake_uri}/raw/roster-reconciliation/season={config.season}/{key}.json'
    client=_new_s3_client(1)
    head=_head_object(uri,client=client)
    if head is not None:
        path=resolve_object(uri=uri,digest=ObjectDigest(head['Metadata']['sha256'],head['ContentLength']),
            cache_root=config.artifact_cache_root/'roster-captures',client=client,lake_uri=config.artifact_lake_uri)
        return artifacts.read_json(path)
    if environment(config)=='dev':
        raise FileNotFoundError('Shared official roster capture is unavailable; dev must use saved official inputs')
    from nba_lineup_model.players.in_season_rosters import fetch_current_nba_rosters,fetch_player_movements
    payload={'mode':'automatic','discovered_at':datetime.now(UTC).isoformat(),
             'nba_roster_payloads':fetch_current_nba_rosters(),'nba_player_movements':fetch_player_movements()}
    AwsCliImmutableObjectStore().put_bytes(uri,canonical_json(payload))
    # Read back through the same checksum-verified consumer path.
    return captured_request(config,capture_slot)


def reconcile_environment(config, *, capture_slot=None, request_path=None, request_sha256=None):
    from nba_lineup_model.players.in_season_rosters import run
    from nba_lineup_model.orchestration.daily_pipeline import require_clean_code_revision
    if config.artifact_lake_uri is None or config.artifact_cache_root is None:
        raise ValueError('Reconciliation requires canonical durable artifact storage')
    live=config.live_pointer.resolve(strict=True)
    accepted=artifacts.read_json(live/'runtime-manifest.json')['approved_manifest']['identity']
    bindings=artifacts.read_json(live/'serving-bindings.json')
    if request_path is not None:
        path=Path(request_path).resolve(strict=True)
        if not path.is_relative_to(config.environment_root.resolve()) or file_sha256(path)!=request_sha256:
            raise ValueError('Reconciliation request is unpinned or outside its environment')
        request=artifacts.read_json(path)
    else:
        if capture_slot is None:raise ValueError('Automatic reconciliation requires an explicit source capture slot')
        request=captured_request(config,capture_slot)
    return run(release_root=Path(bindings['historical_runtime']),
        state_root=config.environment_root/'roster-reconciliation',
        base_rankings_path=config.preseason_rankings_path,base_profiles_path=live/'preseason/profiles.parquet',
        base_minutes_path=config.preseason_minutes_path,accepted_model_path=Path(accepted['model']['bundle_path']),
        request=request,environment=environment(config),cache_root=config.artifact_cache_root,
        lake_uri=config.artifact_lake_uri,producer_revision=require_clean_code_revision(config.code_root),
        approval_reference=config.reconciliation_approval_reference)


def selected_inputs(config):
    if config.artifact_lake_uri is None:return None
    selected = artifacts.selected(environment=environment(config),cache_root=config.artifact_cache_root,
                                  lake_uri=config.artifact_lake_uri)
    if selected is None:
        raise ValueError('No verified reconciliation selection; run reconcile_in_season_rosters before consuming player inputs')
    return selected


def pin_files(root):
    from nba_lineup_model.orchestration.daily_work import file_reference
    artifacts.verify(root)
    return {'reconciliation/'+str(p.relative_to(root)):file_reference(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def scheduled_slot(now):
    """Latest configured 06/09/12/15 Pacific check, including the prior day."""
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    local=now.astimezone(ZoneInfo('America/Los_Angeles'))
    slots=[local.replace(hour=h,minute=0,second=0,microsecond=0) for h in (6,9,12,15)]
    eligible=[slot for slot in slots if slot<=local]
    slot=max(eligible) if eligible else (local-timedelta(days=1)).replace(hour=15,minute=0,second=0,microsecond=0)
    return slot.date().isoformat(),slot.astimezone(UTC).isoformat()
