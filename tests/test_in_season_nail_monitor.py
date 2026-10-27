from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_lineup_model.modeling.in_season_nail import (
    build_game_contribution,
    publish_game_contribution,
)
from nba_lineup_model.modeling.in_season_nail_monitor import (
    CORE_MODEL_VERSION,
    SEASON,
    _b2b_flags,
    _cached_game_contribution,
    build_player_game_trajectory,
)


def _contribution():
    stints = pd.DataFrame(
        {
            "game_id": ["0022600001"],
            "home_player_ids": [(1, 2)],
            "away_player_ids": [(3, 4)],
            "target_home_net_rating": [5.0],
            "possessions": [20.0],
        }
    )
    return build_game_contribution(
        stints,
        season=SEASON,
        game_id="0022600001",
        game_time_utc=datetime(2026, 10, 20, 23, 0, tzinfo=UTC),
        finalized_at=datetime(2026, 10, 21, 2, 0, tzinfo=UTC),
        source_sha256="a" * 64,
        source_uri="s3://nba-gestalt-data/lake/raw/cdn/game.json",
        core_model_version=CORE_MODEL_VERSION,
        context_model_version="frozen-context",
        player_ids=(1, 2, 3, 4),
        preseason_nail={1: 1.0, 2: 0.5, 3: -0.5, 4: -1.0},
        fixed_context_offset=2.0,
    )


def test_shadow_monitor_reuses_only_matching_source_contribution(tmp_path: Path) -> None:
    contribution = _contribution()
    path, created = publish_game_contribution(contribution, tmp_path)

    cached, cached_path = _cached_game_contribution(
        tmp_path,
        game_id=contribution.game_id,
        source_sha256=contribution.source_sha256,
        player_ids=contribution.player_ids,
    )
    revised, revised_path = _cached_game_contribution(
        tmp_path,
        game_id=contribution.game_id,
        source_sha256="b" * 64,
        player_ids=contribution.player_ids,
    )
    changed_vocabulary, changed_vocabulary_path = _cached_game_contribution(
        tmp_path,
        game_id=contribution.game_id,
        source_sha256=contribution.source_sha256,
        player_ids=(*contribution.player_ids, 5),
    )

    assert created is True
    assert cached is not None
    assert cached.contribution_sha256 == contribution.contribution_sha256
    assert cached_path == path
    assert revised is None
    assert revised_path is None
    assert changed_vocabulary is None
    assert changed_vocabulary_path is None


def test_shadow_monitor_back_to_back_flags_are_date_batched() -> None:
    summaries = [
        {"game_id": "1", "game_date": "2026-10-20", "home_team": "BOS", "away_team": "NYK"},
        {"game_id": "2", "game_date": "2026-10-21", "home_team": "BOS", "away_team": "MIA"},
        {"game_id": "3", "game_date": "2026-10-21", "home_team": "NYK", "away_team": "CHI"},
    ]

    flags = _b2b_flags(summaries)

    assert flags["1"] == (False, False)
    assert flags["2"] == (True, False)
    assert flags["3"] == (True, False)


def test_player_game_trajectory_has_one_cumulative_point_per_appearance() -> None:
    first = _contribution()
    second_stints = pd.DataFrame(
        {
            "game_id": ["0022600002"],
            "home_player_ids": [(1, 3)],
            "away_player_ids": [(2, 4)],
            "target_home_net_rating": [2.0],
            "possessions": [30.0],
        }
    )
    second = build_game_contribution(
        second_stints,
        season=SEASON,
        game_id="0022600002",
        game_time_utc=datetime(2026, 10, 22, 23, 0, tzinfo=UTC),
        finalized_at=datetime(2026, 10, 23, 2, 0, tzinfo=UTC),
        source_sha256="b" * 64,
        source_uri="s3://nba-gestalt-data/lake/raw/cdn/game-2.json",
        core_model_version=CORE_MODEL_VERSION,
        context_model_version="frozen-context",
        player_ids=(1, 2, 3, 4),
        preseason_nail={1: 1.0, 2: 0.5, 3: -0.5, 4: -1.0},
        fixed_context_offset=0.0,
    )

    trajectory = build_player_game_trajectory(
        {
            first.game_time_utc.date(): [first],
            second.game_time_utc.date(): [second],
        },
        player_ids=(1, 2, 3, 4),
        preseason_nail={1: 1.0, 2: 0.5, 3: -0.5, 4: -1.0},
        prior_possessions=6000.0,
        game_player_context={
            ("0022600001", 1): {"team": "BOS", "opponent": "NYK"},
            ("0022600001", 2): {"team": "BOS", "opponent": "NYK"},
            ("0022600001", 3): {"team": "NYK", "opponent": "BOS"},
            ("0022600001", 4): {"team": "NYK", "opponent": "BOS"},
            ("0022600002", 1): {"team": "BOS", "opponent": "MIA"},
            ("0022600002", 2): {"team": "MIA", "opponent": "BOS"},
            ("0022600002", 3): {"team": "BOS", "opponent": "MIA"},
            ("0022600002", 4): {"team": "MIA", "opponent": "BOS"},
        },
    )

    assert len(trajectory) == 8
    player = trajectory.loc[trajectory["player_id"].eq(1)].reset_index(drop=True)
    assert player["game_number"].tolist() == [1, 2]
    assert player["game_id"].tolist() == ["0022600001", "0022600002"]
    assert player["opponent"].tolist() == ["NYK", "MIA"]
    assert player["game_on_court_possessions"].tolist() == [20.0, 30.0]
    assert player["cumulative_on_court_possessions"].tolist() == [20.0, 50.0]
    assert player["updated_rating"].iloc[-1] == (
        player["preseason_rating"].iloc[-1] + player["in_season_delta"].iloc[-1]
    )
