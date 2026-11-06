"""Add saved, zero-evidence introductions to a forecast's accepted player state."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from nba_lineup_model.season.reconciliation_inputs import catalog, introductions, marker


def initialize_forecast_players(*, rankings_state, minutes_state, reconciliation_root: Path,
                                through: date, available_at: datetime, base_rankings,
                                base_minutes, base_prior_sha256: str, replay_config):
    """Retain incumbent rows exactly; initialize additions using the daily rules.

    The returned states are forecast inputs, never a replacement daily checkpoint.
    Their introduction selection records its separate information cutoff.
    """
    from nba_lineup_model.season.daily_model_replay import _solve_rating_state, _uses_joint_home_court
    from nba_lineup_model.modeling.in_season_minutes import initialize_minutes_state

    if available_at.tzinfo is None:
        raise ValueError('Forecast introduction cutoff must have a timezone')
    events = introductions(reconciliation_root, through)
    if any(datetime.fromisoformat(e['discovered_at'].replace('Z', '+00:00')) > available_at for e in events):
        raise ValueError('Forecast cannot use an introduction discovered after its information cutoff')
    ranks, priors = catalog(reconciliation_root, through, base_rankings, base_minutes,
                            base_prior_sha256=base_prior_sha256)
    accepted = set(rankings_state.player_id.astype(int))
    if set(minutes_state.player_id.astype(int)) != accepted:
        raise ValueError('Accepted forecast rating/minutes player sets differ')
    if not accepted <= set(ranks.player_id.astype(int)):
        raise ValueError('Forecast introduction history omits accepted players')
    new_ids = tuple(sorted(set(ranks.player_id.astype(int)) - accepted))
    selection = {**marker(reconciliation_root, through), 'available_at': available_at.isoformat(),
                 'forecast_only_player_ids': list(new_ids)}
    if not new_ids:
        return rankings_state.copy(), minutes_state.copy(), selection
    if not set(new_ids) <= {int(e['player_id']) for e in events}:
        raise ValueError('Missing saved introduction for forecast player')
    added = ranks.loc[ranks.player_id.astype(int).isin(new_ids)]
    teams = added.set_index('player_id').team.astype(str).to_dict()
    dimension = len(new_ids) + int(_uses_joint_home_court(replay_config))
    new_ratings = _solve_rating_state(
        rankings=added, gram=sparse.csr_matrix((dimension, dimension)),
        score=np.zeros(dimension), exposure=np.zeros(dimension), player_ids=new_ids,
        preseason=added.set_index('player_id').rapm.astype(float).to_dict(), team_by_player=teams,
        included_game_ids=json.loads(rankings_state.iloc[0].included_game_ids_json),
        contribution_lineage=json.loads(rankings_state.iloc[0].contribution_lineage_json),
        cutoff_time=available_at, cutoff_game_id=None, config=replay_config)
    # Global HCA is accepted game evidence, not a new player's initialization.
    for column in rankings_state:
        if column.startswith('home_court_'):
            new_ratings[column] = rankings_state.iloc[0][column]
    new_minutes = initialize_minutes_state(priors.loc[priors.player_id.astype(int).isin(new_ids)],
        season=replay_config.season, config=replay_config.minutes_config())
    new_minutes['team'] = new_minutes.player_id.map(teams)
    for column in ('included_game_ids_json', 'contribution_lineage_json'):
        new_minutes[column] = minutes_state.iloc[0][column]
    output = []
    for existing, addition in ((rankings_state, new_ratings), (minutes_state, new_minutes)):
        result = pd.concat([existing, addition], ignore_index=True).sort_values('player_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(existing.sort_values('player_id').reset_index(drop=True),
            result.loc[result.player_id.isin(accepted), existing.columns].reset_index(drop=True),
            check_exact=True, check_dtype=False)
        output.append(result)
    return *output, selection


def validate_publication(*, base: Path, serving_root: Path, rankings_path: Path,
                         profiles_path: Path, reconciliation_root: Path) -> dict:
    """Validate the narrowly permitted forecast-release state extension."""
    from nba_lineup_model.players.reconciliation_artifact import read_json, reference, verify
    from nba_lineup_model.season.current_serving import resolve_latest_bundle, verify_serving_bundle
    selected = verify(reconciliation_root)
    receipt = read_json(serving_root / 'forecast-initialization.json')
    if receipt['reference'] != reference(selected).as_dict():
        raise ValueError('Forecast initialization differs from selected reconciliation')
    expected_marker = marker(reconciliation_root, date.fromisoformat(receipt['through_date']))
    if any(receipt[k] != v for k, v in expected_marker.items()):
        raise ValueError('Forecast initialization has an invalid effective selection')
    cutoff = datetime.fromisoformat(receipt['available_at'])
    events = introductions(reconciliation_root, date.fromisoformat(receipt['through_date']))
    if any(datetime.fromisoformat(e['discovered_at'].replace('Z', '+00:00')) > cutoff for e in events):
        raise ValueError('Forecast initialization contains unavailable future information')
    before_root, after_root = resolve_latest_bundle(base / 'current-season-serving'), resolve_latest_bundle(serving_root)
    before_manifest, after_manifest = verify_serving_bundle(before_root), verify_serving_bundle(after_root)
    for key in ('evidence_through_date', 'season', 'schedule_controls'):
        if before_manifest['identity'].get(key) != after_manifest['identity'].get(key):
            raise ValueError('Forecast initialization changed accepted game evidence: ' + key)
    before, after = pd.read_parquet(before_root / 'serving_state.parquet'), pd.read_parquet(after_root / 'serving_state.parquet')
    ids = set(before.player_id.astype(int))
    expected_ids = ids | {int(e['player_id']) for e in events}
    if after.player_id.duplicated().any() or set(after.player_id.astype(int)) != expected_ids:
        raise ValueError('Forecast initialization has incomplete or unexpected player coverage')
    for old, new, excluded in [(before, after, {'as_of_cutoff_time'}),
            (pd.read_parquet(base / 'preseason/rankings.parquet'), pd.read_parquet(rankings_path), set()),
            (pd.read_parquet(base / 'preseason/profiles.parquet'), pd.read_parquet(profiles_path), set())]:
        columns = [c for c in old if c not in excluded]
        pd.testing.assert_frame_equal(old[columns].sort_values('player_id').reset_index(drop=True),
            new.loc[new.player_id.isin(old.player_id), columns].sort_values('player_id').reset_index(drop=True),
            check_exact=True, check_dtype=False)
        if new.player_id.duplicated().any() or set(new.player_id.astype(int)) != expected_ids:
            raise ValueError('Forecast serving and initialization catalogs differ')
    new = after.loc[~after.player_id.isin(ids)]
    if not new.empty:
        for column in ('raw_on_court_possessions', 'effective_information', 'current_season_residual',
                       'participation_observations', 'conditional_observations'):
            if not new[column].eq(0).all():
                raise ValueError('Forecast introduction invents game evidence: ' + column)
        by_id = {int(e['player_id']): e for e in events}
        for row in new.itertuples():
            event = by_id[int(row.player_id)]
            rating = pd.read_parquet(reconciliation_root / event['initialization']['rankings']).iloc[0]
            minute = read_json(reconciliation_root / event['initialization']['minutes'])['minutes']['players'][0]
            if (float(row.updated_nail) != float(rating.rapm)
                or float(row.current_conditional_minutes_estimate) != float(minute['conditional_minutes_per_game'])):
                raise ValueError('Forecast initialization differs from saved starting values')
    return {**receipt, 'serving_bundle_sha256': after_manifest['bundle_sha256']}
