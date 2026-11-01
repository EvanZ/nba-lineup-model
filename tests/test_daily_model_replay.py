from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.season.daily_data_artifact import (
    promote_daily_data_artifact,
    publish_daily_data_artifact,
)
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.daily_model_replay import (
    DailyModelReplayBlocked,
    _early_role_minutes,
    _normalize_name,
    replay_daily_model_chain,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _prior_files(tmp_path: Path, *, omit_player: int | None = None) -> tuple[Path, Path]:
    player_ids = [1631104, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rankings = pd.DataFrame(
        [
            {
                "season": "2026-27",
                "player_id": player_id,
                "player_name": f"Player {player_id}",
                "team": "HOM" if index < 5 else "AWY",
                "rapm": float(index) / 10.0,
            }
            for index, player_id in enumerate(player_ids)
            if player_id != omit_player
        ]
    )
    rankings_path = tmp_path / "rankings.parquet"
    rankings.to_parquet(rankings_path, index=False)
    minutes_players = [
        {
            "player_id": player_id,
            "player_name": f"Player {player_id}",
            "team": "HOM" if index < 5 else "AWY",
            "availability_probability": 1.0,
            "conditional_minutes_per_game": 48.0,
            "baseline_minutes_per_game": 48.0,
        }
        for index, player_id in enumerate(player_ids)
        if player_id != omit_player
    ]
    minutes_path = tmp_path / "minutes.json"
    minutes_path.write_text(json.dumps({"minutes": {"players": minutes_players}}))
    return rankings_path, minutes_path


def _publish_data(tmp_path: Path):
    game_id = "0022600001"
    player_ids = [1631104, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    home_ids, away_ids = player_ids[:5], player_ids[5:]
    component_root = tmp_path / "components"
    component_root.mkdir()
    final = [
        {
            "game_id": game_id,
            "season": "2026-27",
            "game_date_pacific": "2026-10-20",
            "start_time": "2026-10-20T23:00:00+00:00",
            "home_team_id": 100,
            "home_team": "HOM",
            "away_team_id": 200,
            "away_team": "AWY",
            "status": "final",
            "home_score": 105,
            "away_score": 100,
            "actual_home_margin": 5,
        }
    ]
    (component_root / "final_games.json").write_text(json.dumps(final))
    roster_teams = []
    actual_rows = []
    identity_rows = []
    for side, team, team_id, ids in (
        ("home", "HOM", 100, home_ids),
        ("away", "AWY", 200, away_ids),
    ):
        players = []
        for player_id in ids:
            players.append(
                {
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "played": True,
                    "starter": True,
                    "status": "ACTIVE",
                    "minutes": 48.0,
                }
            )
            actual_rows.append(
                {
                    "game_id": game_id,
                    "game_date_pacific": "2026-10-20",
                    "team": team,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "minutes": 48.0,
                }
            )
            identity_rows.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "player_name": f"Player {player_id}",
                    "player_id": player_id,
                    "played": True,
                    "resolution": "nba_person_id",
                }
            )
        roster_teams.append(
            {"team_id": team_id, "team": team, "side": side, "players": players}
        )
    (component_root / "rosters.json").write_text(
        json.dumps({"games": [{"game_id": game_id, "teams": roster_teams}]})
    )
    pd.DataFrame(actual_rows).to_parquet(component_root / "actual_minutes.parquet")
    pd.DataFrame(identity_rows).to_parquet(component_root / "player_identity.parquet")
    stints = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "home_team_id": 100,
                "away_team_id": 200,
                "home_team_tricode": "HOM",
                "away_team_tricode": "AWY",
                "home_player_ids": np.asarray(home_ids),
                "away_player_ids": np.asarray(away_ids),
                "start_elapsed_game_seconds": 0.0,
                "end_elapsed_game_seconds": 2880.0,
                "possessions": 100.0,
                "target_home_net_rating": 5.0,
            }
        ]
    )
    stints.to_parquet(component_root / "observed_stints.parquet")
    pd.DataFrame(
        [
            {
                "game_id": game_id,
                "team_id": 100,
                "team": "HOM",
                "player_ids_json": json.dumps(home_ids),
                "possessions": 100.0,
            },
            {
                "game_id": game_id,
                "team_id": 200,
                "team": "AWY",
                "player_ids_json": json.dumps(away_ids),
                "possessions": 100.0,
            },
        ]
    ).to_parquet(component_root / "observed_lineups.parquet")
    box_teams = {}
    for side, team, team_id, ids, score in (
        ("homeTeam", "HOM", 100, home_ids, 105),
        ("awayTeam", "AWY", 200, away_ids, 100),
    ):
        box_teams[side] = {
            "teamId": team_id,
            "teamTricode": team,
            "score": score,
            "players": [
                {
                    "personId": player_id,
                    "statistics": {"plusMinusPoints": 5 if team == "HOM" else -5},
                }
                for player_id in ids
            ],
        }
    (component_root / "nba_boxscores.json").write_text(
        json.dumps(
            [
                {
                    "game_id": game_id,
                    "payload": {"game": {"gameId": game_id, **box_teams}},
                }
            ]
        )
    )
    (component_root / "injuries.json").write_text(
        json.dumps(
            [
                {
                    "payload": {
                        "game_team_reports": {
                            "2026-10-20|AWY@HOM": {
                                "HOM": {"status": "submitted"},
                                "AWY": {"status": "submitted"},
                            }
                        },
                        "players": [],
                    }
                }
            ]
        )
    )
    components = {
        name: component_root / filename
        for name, filename in {
            "final_games": "final_games.json",
            "nba_boxscores": "nba_boxscores.json",
            "actual_minutes": "actual_minutes.parquet",
            "observed_stints": "observed_stints.parquet",
            "observed_lineups": "observed_lineups.parquet",
            "player_identity": "player_identity.parquet",
            "rosters": "rosters.json",
            "injuries": "injuries.json",
        }.items()
    }
    sources = {
        game_id: {
            source: {"uri": f"s3://test/{source}", "sha256": _sha(source)}
            for source in (
                "nba_schedule",
                "nba_box_score",
                "nba_play_by_play",
                "nba_injury_report",
            )
        }
    }
    data = publish_daily_data_artifact(
        root=tmp_path / "data",
        season="2026-27",
        game_date_pacific=date(2026, 10, 20),
        game_ids=[game_id],
        game_sources=sources,
        game_quality={
            game_id: {
                "final_home_score": 105,
                "final_away_score": 100,
                "box_home_points": 105,
                "box_away_points": 100,
                "play_by_play_possessions": 100.0,
                "observed_stint_possessions": 100.0,
            }
        },
        components=components,
    )
    promote_daily_data_artifact(data.path, root=tmp_path / "data")
    return data


def _zero_context(stints: pd.DataFrame, _home_b2b: bool, _away_b2b: bool):
    return np.zeros(len(stints)), []


def test_daily_model_replay_builds_a_clean_official_nba_state(tmp_path: Path) -> None:
    data = _publish_data(tmp_path)
    rankings, minutes = _prior_files(tmp_path)

    published = replay_daily_model_chain(
        start_date=date(2026, 10, 20),
        end_date=date(2026, 10, 20),
        preseason_rankings_path=rankings,
        preseason_minutes_path=minutes,
        data_root=tmp_path / "data",
        output_root=tmp_path / "model",
        code_revision="test-revision",
        context_provider=_zero_context,
    )

    assert len(published) == 1
    manifest = verify_daily_model_artifact(published[0].path, data_bundle=data.path)
    diagnostics_path = (
        published[0].path
        / manifest["identity"]["components"]["evaluation_metrics"]["path"]
    )
    diagnostics = json.loads(diagnostics_path.read_text())
    assert diagnostics["status"] == "clean"
    assert diagnostics["blake_wesley_prior_present"] is True
    assert diagnostics["allocation_max_absolute_error"] == pytest.approx(0.0)
    assert diagnostics["allocation_max_player_minutes"] == pytest.approx(48.0)
    assert diagnostics["excluded_game_count"] == 0
    assert (tmp_path / "model" / "latest.json").is_file()
    assert "runtime ESPN dependencies: 0" in (
        tmp_path / "model" / "report.html"
    ).read_text()

    repeated = replay_daily_model_chain(
        start_date=date(2026, 10, 20),
        end_date=date(2026, 10, 20),
        preseason_rankings_path=rankings,
        preseason_minutes_path=minutes,
        data_root=tmp_path / "data",
        output_root=tmp_path / "model",
        code_revision="test-revision",
        context_provider=_zero_context,
    )
    assert repeated == []


def test_daily_model_replay_fails_closed_on_missing_player_prior(tmp_path: Path) -> None:
    _publish_data(tmp_path)
    rankings, minutes = _prior_files(tmp_path, omit_player=1631104)

    with pytest.raises(DailyModelReplayBlocked, match="2026-10-20 failed"):
        replay_daily_model_chain(
            start_date=date(2026, 10, 20),
            end_date=date(2026, 10, 20),
            preseason_rankings_path=rankings,
            preseason_minutes_path=minutes,
            data_root=tmp_path / "data",
            output_root=tmp_path / "model",
            code_revision="test-revision",
            context_provider=_zero_context,
        )

    assert not (tmp_path / "model" / "latest.json").exists()
    error = json.loads(
        (
            tmp_path
            / "model/failures/game_date_pacific=2026-10-20/error.json"
        ).read_text()
    )
    assert "Player universe is incomplete" in error["error"]


def test_early_role_reconciles_only_a_small_verified_stint_gap() -> None:
    stints = pd.DataFrame(
        [
            {
                "start_elapsed_game_seconds": 0.0,
                "end_elapsed_game_seconds": 2426.0,
                "home_player_ids": np.asarray([1, 2, 3, 4, 5]),
                "away_player_ids": np.asarray([6, 7, 8, 9, 10]),
            },
            {
                "start_elapsed_game_seconds": 2436.0,
                "end_elapsed_game_seconds": 2880.0,
                "home_player_ids": np.asarray([1, 2, 3, 4, 5]),
                "away_player_ids": np.asarray([6, 7, 8, 9, 10]),
            },
        ]
    )

    roles, warning = _early_role_minutes(stints, boundary_seconds=2640.0)

    assert warning == "reconciled 10.0s verified stint-coverage gap within 15.0s limit"
    assert sum(roles[player_id] for player_id in range(1, 6)) == pytest.approx(240.0)
    assert sum(roles[player_id] for player_id in range(6, 11)) == pytest.approx(240.0)

    with pytest.raises(ValueError, match="exceeds the audited reconciliation limit"):
        _early_role_minutes(
            stints,
            boundary_seconds=2640.0,
            maximum_reconciled_gap_seconds=5.0,
        )


def test_official_nba_suffix_formats_share_one_identity_key() -> None:
    expected = _normalize_name("Ernest Udeh")

    assert _normalize_name("Ernest Udeh, Jr.") == expected
    assert _normalize_name("Jr., Ernest Udeh") == expected
