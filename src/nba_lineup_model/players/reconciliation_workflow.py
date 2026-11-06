"""Canonical durable workflow for the existing in-season reconciliation job.

Both automatic NBA roster captures and approved manual introductions enter this
producer. It reuses in_season_rosters' established initialization calculations.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from nba_lineup_model.players import reconciliation_artifact as artifacts
from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact


def require_same_rows(before, after, label):
    before = before.set_index('player_id').sort_index()
    after = after.set_index('player_id').sort_index()
    if before.index.duplicated().any() or after.index.duplicated().any() or not set(before.index) <= set(after.index):
        raise ValueError(f'{label}: missing or duplicate incumbent identities')
    try:
        pd.testing.assert_frame_equal(before, after.loc[before.index, before.columns],
                                      check_dtype=False, check_exact=True)
    except (AssertionError,KeyError) as error:
        raise ValueError(f'{label}: an incumbent initialization changed') from error


def preserve_incumbents(before, candidate):
    """Roster membership is separate from immutable player initialization."""
    ids=set(before.player_id.astype(int))
    return pd.concat([before,candidate.loc[~candidate.player_id.astype(int).isin(ids)]],ignore_index=True).sort_values('player_id',kind='stable').reset_index(drop=True)


def manual_roster(request, *, season, destination):
    """Validate saved membership and biography evidence; do not fetch live data."""
    from nba_lineup_model.players.rosters import team_roster_frame
    from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
    frames=[]; dependencies=[]
    for index,item in enumerate(request['introductions']):
        player=int(item['player_id']); effective=date.fromisoformat(item['effective_date'])
        discovered=datetime.fromisoformat(item['discovered_at'].replace('Z','+00:00'))
        if discovered.tzinfo is None or discovered > datetime.now(discovered.tzinfo) or not item['authorization']:
            raise ValueError('Manual introduction needs actual discovery time and explicit authorization')
        evidence=Path(item['membership']['bundle_path'])
        manifest=verify_daily_data_artifact(evidence)
        if (manifest['bundle_sha256'] != item['membership']['sha256'] or
            manifest['identity']['season'] != season or
            manifest['identity']['game_date_pacific'] != effective.isoformat()):
            raise ValueError('Manual effective date does not match the verified official evidence')
        rosters=artifacts.read_json(evidence/manifest['identity']['components']['rosters']['path'])
        memberships=[team for game in rosters['games'] for team in game['teams']
                     if any(int(row['player_id'])==player for row in team['players'])]
        if not memberships or {row['team'] for row in memberships}!={item['team']}:
            raise ValueError('Manual player/team is not established by official roster evidence')
        bio_path=Path(item['biography']['payload_path']);meta_path=Path(item['biography']['metadata_path'])
        meta=artifacts.read_json(meta_path)
        if file_sha256(meta_path)!=item['biography']['metadata_sha256']:
            raise ValueError('Official biography metadata checksum mismatch')
        if (file_sha256(bio_path)!=item['biography']['sha256'] or meta['sha256']!=item['biography']['sha256'] or
            meta['endpoint']!='commonteamroster' or meta['season']!=season or
            not str(meta['url']).startswith('https://stats.nba.com/stats/commonteamroster?')):
            raise ValueError('Manual biography is not a verified official NBA roster capture')
        fetched=datetime.fromisoformat(meta['fetched_at'].replace('Z','+00:00'))
        if fetched.tzinfo is None or fetched.date()>effective:
            raise ValueError('Manual biography postdates the approved initialization date')
        bio=team_roster_frame(artifacts.read_json(bio_path),season=season,team_id=int(meta['team_id']))
        bio=bio.loc[bio.player_id.astype(int).eq(player)].copy()
        if len(bio)!=1:raise ValueError('Official biography does not uniquely identify the player')
        # The game roster establishes membership; the archived roster supplies biography only.
        from nba_lineup_model.players.rosters import TEAM_ABBREVIATIONS
        teams={code:str(team_id) for team_id,code in TEAM_ABBREVIATIONS.items()}
        bio['team_id']=teams[item['team']];bio['team_abbreviation']=item['team']
        frames.append(bio)
        for label,source in [('biography',bio_path),('biography-metadata',meta_path),
                             ('membership-manifest',evidence/'manifest.json'),
                             ('membership-rosters',evidence/manifest['identity']['components']['rosters']['path'])]:
            shutil.copyfile(source,destination/f'{index}-{label}.json')
        from nba_lineup_model.season.durable_artifact_storage import daily_data_reference
        dependencies.append(daily_data_reference(evidence).as_dict())
    if not frames:raise ValueError('Manual reconciliation has no introductions')
    roster=pd.concat(frames,ignore_index=True)
    roster['player_id']=roster.player_id.astype(int)
    if roster.player_id.duplicated().any():raise ValueError('Duplicate manual introduction identity')
    return roster,dependencies


def pin_history(release_root, destination, season):
    from nba_lineup_model.players.in_season_rosters import _single_match
    from nba_lineup_model.web_api.preseason_rankings_cache import _published_run,exposure_cohort_path,MODEL_ARTIFACT
    paths=[release_root/f'data/analytical/player_season_panel/player_seasons.parquet',
           release_root/f'data/curated/team_rosters/{season}/part-00000.parquet']
    profiles=_single_match(release_root,'artifacts/web/historical_profiles/'+MODEL_ARTIFACT+'/*.parquet','historical profiles')
    rotations=_single_match(release_root,'artifacts/web/player_rotation_history/'+MODEL_ARTIFACT+'/*.parquet','rotation history')
    paths.extend([profiles,rotations])
    previous=Path.cwd()
    try:
        os.chdir(release_root)
        run,run_id=_published_run(f'{int(season[:4])-1}-{season[:4][-2:]}')
        paths.extend([run.resolve()/'historical_player_coefficients.parquet',
                      run.resolve()/'season_player_prior_metadata.parquet',run.resolve().parent/'latest.json',
                      exposure_cohort_path(MODEL_ARTIFACT,run_id).resolve()])
    finally:os.chdir(previous)
    draft=release_root/f'data/curated/draft_history/{season}/part-00000.parquet'
    if draft.is_file():paths.append(draft)
    for source in paths:
        target=destination/source.relative_to(release_root)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    return profiles.relative_to(release_root),rotations.relative_to(release_root)


def historical_impact(events, affected_ids, *, accepted_model, request):
    """Preparing a correction is separate from approving selection and replay."""
    boundary = date.fromisoformat(accepted_model['identity']['game_date_pacific'])
    dates = [date.fromisoformat(events[p]['effective_date']) for p in affected_ids]
    affected = [day for day in dates if day <= boundary]
    if not affected:
        return None
    impact = {'from_date': min(affected).isoformat(), 'through_date': boundary.isoformat(),
              'accepted_model_sha256': accepted_model['bundle_sha256'],
              'frozen_forecasts': 'unchanged', 'player_ids': sorted(affected_ids)}
    approval = request.get('historical_correction_approval')
    if approval is not None:
        if (not approval.get('decision_reference')
            or any(approval.get(k) != impact[k] for k in ('from_date', 'through_date', 'accepted_model_sha256'))):
            raise ValueError('Historical correction approval does not match the affected accepted history')
        impact['selection_approval'] = approval
    return impact


def publish_or_accept(root, *, request, environment, state_root, cache_root, lake_uri, client, approval_reference):
    manifest = artifacts.verify(root)
    from nba_lineup_model.storage.artifact_store import ArtifactReference, resolve_storage_manifest
    from nba_lineup_model.season.durable_artifact_storage import publish_daily_data_bundle
    for item in manifest['identity']['dependencies']:
        dependency = ArtifactReference.from_dict(item)
        if dependency.artifact_kind != 'daily_data':
            continue
        try:
            resolve_storage_manifest(reference=dependency, cache_root=cache_root, client=client, lake_uri=lake_uri)
        except FileNotFoundError:
            matches = [x['membership'] for x in request.get('introductions', [])
                       if dependency.content_id == 'data_sha256=' + x['membership']['sha256']]
            if not matches:
                raise ValueError('Required official membership artifact is unavailable for canonical publication')
            published = publish_daily_data_bundle(matches[0]['bundle_path'], cache_root=cache_root,
                                                  client=client, lake_uri=lake_uri)
            if published.reference != dependency:
                raise ValueError('Published membership evidence differs from the sealed introduction dependency')
    impact = manifest['identity'].get('historical_impact')
    if impact and not impact.get('selection_approval'):
        artifacts.publish(root, cache_root=cache_root, lake_uri=lake_uri, client=client)
        return {'status': 'awaiting_historical_correction_approval', 'path': str(root),
                'reference': artifacts.reference(manifest).as_dict(), 'generation': None,
                'historical_impact': impact}
    selection = artifacts.accept(root, environment=environment, state_root=state_root,
        cache_root=cache_root, lake_uri=lake_uri, client=client, approval_reference=approval_reference)
    return {'status': 'verified', 'path': str(root), **selection}


def run(*, release_root, state_root, base_rankings_path, base_profiles_path,
        base_minutes_path, accepted_model_path, request, environment,
        cache_root, lake_uri, producer_revision, client=None, approval_reference=None):
    import tempfile
    from nba_lineup_model.players.in_season_rosters import (
        _extend_preseason_catalogs, _extend_runtime_minutes_catalog, nba_roster_frame,
        _validate_coupled_preseason_prior_catalogs,
    )
    from nba_lineup_model.season.daily_model_replay import _load_frozen_priors
    from nba_lineup_model.season.source_policy import require_clean_artifact_sources
    release_root=Path(release_root).resolve(strict=True);state_root=Path(state_root)
    cache_root=Path(cache_root);accepted_model_path=Path(accepted_model_path)
    model=verify_daily_model_artifact(accepted_model_path)
    season=model['identity']['season']
    if season!='2026-27':raise ValueError('Existing initialization rules are defined for 2026-27')
    original_rankings,original_minutes,base_sha=_load_frozen_priors(
        Path(base_rankings_path),Path(base_minutes_path),season=season)
    if base_sha!=model['identity']['model_provenance']['prior_sha256']:
        raise ValueError('Reconciliation opening inputs differ from the accepted model')
    parent=artifacts.selected(environment=environment,cache_root=cache_root,lake_uri=lake_uri,client=client)
    parent_manifest=artifacts.verify(parent) if parent else None
    if parent and parent_manifest['identity']['base_prior_sha256']!=base_sha:
        raise ValueError('Selected reconciliation belongs to another opening prior')
    paths={key:parent/name for key,name in artifacts.OUTPUTS.items()} if parent else {
        'rankings':Path(base_rankings_path),'profiles':Path(base_profiles_path),'minutes':Path(base_minutes_path)}
    previous_rankings=pd.read_parquet(paths['rankings'])
    previous_profiles=pd.read_parquet(paths['profiles'])
    previous_minutes=artifacts.read_json(paths['minutes'])
    previous_rows=previous_minutes.get('minutes',previous_minutes)['players']
    events=list(parent_manifest['identity']['introductions']) if parent else []
    parent_ref=artifacts.reference(parent_manifest).as_dict() if parent else None
    def transport_free(value):
        if isinstance(value,dict):return {k:transport_free(v) for k,v in value.items() if not k.endswith('_path')}
        if isinstance(value,list):return [transport_free(v) for v in value]
        return value
    pinned_request=transport_free(request)
    work_identity={'request':pinned_request,'parent_reference':parent_ref,'base_prior_sha256':base_sha,
                   'accepted_model_sha256':model['bundle_sha256'],'producer_revision':producer_revision}
    work_hash=hashlib.sha256(canonical_json(work_identity)).hexdigest()
    work_root=state_root/'work'/work_hash
    final=work_root/'artifact'
    if final.exists():
        return publish_or_accept(final, request=request, environment=environment, state_root=state_root,
            cache_root=cache_root, lake_uri=lake_uri, client=client, approval_reference=approval_reference)
    work_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='build-',dir=work_root) as directory:
        root=Path(directory);sources=root/'sources';sources.mkdir()
        (sources/'request.json').write_bytes(canonical_json(pinned_request))
        if request['mode']=='manual':
            observed,dependencies=manual_roster(request,season=season,destination=sources)
            event_by_id={int(x['player_id']):x for x in request['introductions']}
        elif request['mode']=='automatic':
            from nba_lineup_model.players.rosters import TEAM_ABBREVIATIONS
            if {str(x) for x in request['nba_roster_payloads']} != {str(x) for x in TEAM_ABBREVIATIONS}:
                raise ValueError('Automatic reconciliation requires all 30 official NBA roster captures')
            observed=nba_roster_frame(request['nba_roster_payloads'])
            observed['player_id']=observed.player_id.astype(int)
            discovered=datetime.fromisoformat(request['discovered_at'].replace('Z','+00:00'))
            if discovered.tzinfo is None or discovered > datetime.now(discovered.tzinfo):
                raise ValueError('Roster discovery time must be timezone-aware and not in the future')
            from zoneinfo import ZoneInfo
            day=discovered.astimezone(ZoneInfo('America/Los_Angeles')).date().isoformat()
            event_by_id={int(x.player_id):{'player_id':int(x.player_id),'team':str(x.team_abbreviation),
                'effective_date':day,'discovered_at':request['discovered_at'],
                'authorization':'scheduled official NBA roster reconciliation'} for x in observed.itertuples()}
            dependencies=[]
        else:raise ValueError('Unknown reconciliation input mode')
        require_clean_artifact_sources(sources)
        accepted=pd.read_parquet(accepted_model_path/model['identity']['components']['player_ratings']['path'])
        known_ids=set(accepted.player_id.astype(int))
        accepted_minutes=pd.read_parquet(accepted_model_path/model['identity']['components']['minutes_state']['path'])
        if set(accepted_minutes.player_id.astype(int))!=known_ids or accepted_minutes.player_id.duplicated().any():
            raise ValueError('Accepted model has inconsistent initialization; assess historical state repair before reconciliation')
        recorded_ids={int(x['player_id']) for x in events}
        missing=sorted(set(observed.player_id.astype(int))-known_ids-recorded_ids)
        # Full trade semantics are explicitly deferred. Never reinterpret a trade as new initialization.
        teams=accepted.set_index('player_id')['team_at_cutoff'].to_dict() if 'team_at_cutoff' in accepted else {}
        changed=[int(x.player_id) for x in observed.itertuples()
                 if int(x.player_id) in teams and pd.notna(teams[int(x.player_id)])
                 and str(teams[int(x.player_id)])!=str(x.team_abbreviation)]
        if changed:raise ValueError(f'Unsupported team changes require the trade slice: {sorted(changed)}')
        if not missing and parent:
            selection=artifacts.accept(parent,environment=environment,state_root=state_root,
                cache_root=cache_root,lake_uri=lake_uri,client=client,approval_reference=approval_reference)
            return {'status':'unchanged','path':str(parent),**selection}
        boundary=date.fromisoformat(model['identity']['game_date_pacific'])
        impact = historical_impact(event_by_id, missing, accepted_model=model, request=request)
        history=root/'inputs/history';history.mkdir(parents=True)
        profile_relative,rotation_relative=pin_history(release_root,history,season)
        inputs=root/'inputs/base';inputs.mkdir()
        for label,path in paths.items():shutil.copyfile(path,inputs/artifacts.OUTPUTS[label])
        metadata=paths['rankings'].with_suffix('.json')
        if metadata.exists():shutil.copyfile(metadata,inputs/'rankings.json')
        # Keep the opening catalog as an immutable dependency for verification/replay.
        for label,path in [('opening-rankings.parquet',Path(base_rankings_path)),
                           ('opening-minutes.json',Path(base_minutes_path))]:shutil.copyfile(path,inputs/label)
        if Path(base_rankings_path).with_suffix('.json').exists():
            shutil.copyfile(Path(base_rankings_path).with_suffix('.json'),inputs/'opening-rankings.json')
        frozen_roster=pd.read_parquet(history/f'data/curated/team_rosters/{season}/part-00000.parquet')
        frozen_roster['player_id']=frozen_roster.player_id.astype(int)
        additions=observed.loc[observed.player_id.astype(int).isin(missing)]
        roster=pd.concat([frozen_roster.loc[~frozen_roster.player_id.isin(missing)],additions],ignore_index=True)
        # Alignment may use incumbent biographies; only requested missing IDs may be appended.
        roster=roster.loc[roster.player_id.astype(int).isin(set(previous_rankings.player_id.astype(int))|set(missing))]
        before=Path.cwd()
        try:
            os.chdir(history)
            _extend_preseason_catalogs(base_rankings_path=inputs/'rankings.parquet',
                base_profiles_path=inputs/'profiles.parquet',current_roster=roster,
                panel_path=history/'data/analytical/player_season_panel/player_seasons.parquet',
                historical_profiles_path=history/profile_relative,
                rankings_output_path=root/'rankings.parquet',profiles_output_path=root/'profiles.parquet')
        finally:os.chdir(before)
        rankings=preserve_incumbents(previous_rankings,pd.read_parquet(root/'rankings.parquet'))
        profiles=preserve_incumbents(previous_profiles,pd.read_parquet(root/'profiles.parquet'))
        _extend_runtime_minutes_catalog(base_cache_path=inputs/'minutes.json',prior_overlay_path=None,
            rankings=rankings,current_roster=roster,rotation_history_path=history/rotation_relative,
            output_path=root/'minutes.json')
        minutes=artifacts.read_json(root/'minutes.json')
        candidate_rows=minutes.get('minutes',minutes)['players']
        old_ids={int(x['player_id']) for x in previous_rows}
        minute_rows=previous_rows+[x for x in candidate_rows if int(x['player_id']) not in old_ids]
        minutes.get('minutes',minutes)['players']=sorted(minute_rows,key=lambda x:int(x['player_id']))
        (root/'minutes.json').write_bytes(canonical_json(minutes))
        rankings.to_parquet(root/'rankings.parquet',index=False);profiles.to_parquet(root/'profiles.parquet',index=False)
        require_same_rows(previous_rankings,rankings,'ratings');require_same_rows(previous_profiles,profiles,'profiles')
        require_same_rows(pd.DataFrame(previous_rows),pd.DataFrame(minute_rows),'minutes')
        _validate_coupled_preseason_prior_catalogs(rankings_path=root/'rankings.parquet',minutes_path=root/'minutes.json')
        expected=set(previous_rankings.player_id.astype(int))|set(missing)
        if set(rankings.player_id.astype(int))!=expected:raise ValueError('Unexpected cumulative player-catalog growth')
        if parent:
            for event in events:
                for name in [*event['initialization'].values(), *event['evidence'].get('files', [])]:
                    dest=root/artifacts.safe_relative(name);dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(parent/name,dest)
            dependencies.append(parent_ref)
        for player in missing:
            event=event_by_id[player];folder=root/'introductions'/str(player);folder.mkdir(parents=True)
            rankings.loc[rankings.player_id.astype(int).eq(player)].to_parquet(folder/'rankings.parquet',index=False)
            profiles.loc[profiles.player_id.astype(int).eq(player)].to_parquet(folder/'profiles.parquet',index=False)
            (folder/'minutes.json').write_bytes(canonical_json({'minutes':{'players':[x for x in minute_rows if int(x['player_id'])==player]}}))
            evidence_files=[]
            evidence_dir=folder/'evidence';evidence_dir.mkdir()
            for source in sorted(sources.iterdir()):
                target=evidence_dir/source.name;shutil.copyfile(source,target)
                evidence_files.append(str(target.relative_to(root)))
            events.append({'player_id':player,'team':event['team'],'effective_date':event['effective_date'],
                'discovered_at':event['discovered_at'],'authorization':event['authorization'],
                'mode':request['mode'],'evidence':{'request_sha256':file_sha256(sources/'request.json'), 'files':evidence_files},
                'initialization':{key:str((folder/name).relative_to(root)) for key,name in artifacts.OUTPUTS.items()}})
        events.sort(key=lambda x:(x['effective_date'],x['player_id']))
        # Preserve exact producer/rule bytes alongside the calculation inputs.
        for name in ['in_season_rosters.py','reconciliation_workflow.py']:
            shutil.copyfile(Path(__file__).parent/name,root/'inputs'/name)
        effective=max([x['effective_date'] for x in events],default=boundary.isoformat())
        dependencies = sorted({canonical_json(x): x for x in dependencies}.values(), key=canonical_json)
        artifacts.seal(root,{'season':season,'effective_through':effective,'base_prior_sha256':base_sha,
            'parent_reference':parent_ref,'producer_revision':producer_revision,'work_identity':work_identity,
            'introductions':events,'dependencies':dependencies,'historical_impact':impact})
        require_clean_artifact_sources(root)
        os.replace(root,final)
    result = publish_or_accept(final, request=request, environment=environment, state_root=state_root,
        cache_root=cache_root, lake_uri=lake_uri, client=client, approval_reference=approval_reference)
    return {**result, 'introduced_player_ids': missing}
