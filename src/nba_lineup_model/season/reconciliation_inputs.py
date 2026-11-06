"""Date-bound model inputs from the selected durable reconciliation history."""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.players.reconciliation_artifact import read_json, reference, verify
from nba_lineup_model.season.daily_artifact_common import canonical_json

EMPTY_INTRODUCTIONS_SHA256 = hashlib.sha256(canonical_json([])).hexdigest()


def introductions(root: Path | None, through: date) -> list[dict]:
    if root is None:return []
    manifest=verify(root)
    return [event for event in manifest['identity']['introductions']
            if date.fromisoformat(event['effective_date'])<=through]


def marker(root: Path | None, through: date) -> dict:
    events=introductions(root,through)
    manifest=verify(root) if root is not None else None
    selected=[{**event,'initialization_checksums':{
        key:manifest['identity']['files'][name]['sha256'] for key,name in event['initialization'].items()}}
        for event in events]
    return {'schema_version':1,'through_date':through.isoformat(),
        'reference':reference(manifest).as_dict() if manifest else None,
        'introductions_sha256':hashlib.sha256(canonical_json(selected)).hexdigest(),
        'player_ids':sorted(int(event['player_id']) for event in events)}


def checkpoint_digest(bundle: Path, manifest: dict) -> str:
    entry=manifest['identity'].get('components', {}).get('player_introductions')
    if entry is None:return EMPTY_INTRODUCTIONS_SHA256
    payload=read_json(bundle/entry['path'])
    if payload['through_date']!=manifest['identity']['game_date_pacific']:
        raise ValueError('Model introduction selection has the wrong effective boundary')
    return payload['introductions_sha256']


def catalog(root: Path | None, through: date, base_rankings: pd.DataFrame,
            base_minutes: pd.DataFrame, *, base_prior_sha256: str):
    if root is None:return base_rankings.copy(),base_minutes.copy()
    manifest=verify(root)
    if manifest['identity']['base_prior_sha256']!=base_prior_sha256:
        raise ValueError('Reconciliation does not extend the accepted opening prior')
    deferred = {int(e['player_id']) for e in manifest['identity']['introductions']
                if date.fromisoformat(e['effective_date']) > through}
    rankings=[base_rankings.loc[~base_rankings.player_id.astype(int).isin(deferred)]]
    minutes=[base_minutes.loc[~base_minutes.player_id.astype(int).isin(deferred)]]
    known=set(rankings[0].player_id.astype(int))
    for event in introductions(root,through):
        player=int(event['player_id'])
        rating=pd.read_parquet(root/event['initialization']['rankings'])
        minute_payload=read_json(root/event['initialization']['minutes'])
        minute=pd.DataFrame(minute_payload['minutes']['players'])
        if len(rating)!=1 or len(minute)!=1 or int(rating.iloc[0].player_id)!=player or int(minute.iloc[0].player_id)!=player:
            raise ValueError('Introduction initialization does not uniquely identify its player')
        if player in known:
            # Catalog-known players may need model adoption, but not a different prior.
            for baseline, saved in ((base_rankings, rating), (base_minutes, minute)):
                pd.testing.assert_frame_equal(baseline.loc[baseline.player_id.astype(int).eq(player)].reset_index(drop=True),
                    saved[baseline.columns].reset_index(drop=True), check_exact=True, check_dtype=False)
            continue
        known.add(player);rankings.append(rating);minutes.append(minute)
    output=[]
    for pieces in (rankings,minutes):
        frame=pd.concat(pieces,ignore_index=True);frame['player_id']=frame.player_id.astype(int)
        if frame.player_id.duplicated().any():raise ValueError('Reconciliation duplicates player initialization')
        output.append(frame.sort_values('player_id',kind='stable').reset_index(drop=True))
    if set(output[0].player_id)!=set(output[1].player_id):raise ValueError('Reconciled rating/minutes universes differ')
    return tuple(output)


def extend_accumulators(old_ids, new_ids, gram, score, exposure, *, joint_home_court):
    old_ids,new_ids=tuple(old_ids),tuple(new_ids)
    if len(set(old_ids))!=len(old_ids) or len(set(new_ids))!=len(new_ids) or not set(old_ids)<=set(new_ids):
        raise ValueError('Reconciliation must retain all unique accepted player identities')
    old_size=len(old_ids)+int(joint_home_court);new_size=len(new_ids)+int(joint_home_court)
    if gram.shape!=(old_size,old_size) or score.shape!=(old_size,) or exposure.shape!=(old_size,):
        raise ValueError('Accepted sufficient-statistic dimensions differ from player identities')
    indices={player:index for index,player in enumerate(new_ids)}
    mapping=np.asarray([indices[player] for player in old_ids]+([len(new_ids)] if joint_home_court else []),dtype=int)
    coo=gram.tocoo()
    new_gram=sparse.csr_matrix((coo.data.copy(),(mapping[coo.row],mapping[coo.col])),shape=(new_size,new_size))
    new_score,new_exposure=np.zeros(new_size),np.zeros(new_size)
    new_score[mapping]=score;new_exposure[mapping]=exposure
    return new_gram,new_score,new_exposure
