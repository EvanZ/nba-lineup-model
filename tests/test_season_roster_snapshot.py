from __future__ import annotations

import json

import pandas as pd

from nba_lineup_model.players.season_roster_snapshot import (
    build_final_regular_season_roster_snapshot,
)


def test_final_regular_season_snapshot_uses_last_observed_team(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    players_dir = tmp_path / "players"
    output_path = tmp_path / "snapshot.parquet"
    players_dir.mkdir()
    schedule_path.write_text(
        json.dumps(
            {
                "leagueSchedule": {
                    "gameDates": [
                        {
                            "games": [
                                {
                                    "gameId": "0022500001",
                                    "gameDateEst": "2025-10-21T00:00:00Z",
                                }
                            ]
                        },
                        {
                            "games": [
                                {
                                    "gameId": "0022500002",
                                    "gameDateEst": "2026-04-12T00:00:00Z",
                                }
                            ]
                        },
                    ]
                }
            }
        )
    )
    pd.DataFrame(
        {
            "personId": [1, 2],
            "name": ["Moved Midseason", "Stayed Put"],
            "team_tricode": ["MEM", "BOS"],
        }
    ).to_parquet(players_dir / "0022500001.parquet", index=False)
    pd.DataFrame(
        {
            "personId": [1, 2],
            "name": ["Moved Midseason", "Stayed Put"],
            "team_tricode": ["UTA", "BOS"],
        }
    ).to_parquet(players_dir / "0022500002.parquet", index=False)

    output = build_final_regular_season_roster_snapshot(
        "2025-26",
        schedule_path=schedule_path,
        processed_players_dir=players_dir,
        output_path=output_path,
    )

    actual = pd.read_parquet(output)
    assert actual.to_dict("records") == [
        {
            "player_id": "1",
            "player_name": "Moved Midseason",
            "source_team": "UTA",
            "last_regular_season_game_id": "0022500002",
            "game_date": pd.Timestamp("2026-04-12T00:00:00+0000", tz="UTC"),
        },
        {
            "player_id": "2",
            "player_name": "Stayed Put",
            "source_team": "BOS",
            "last_regular_season_game_id": "0022500002",
            "game_date": pd.Timestamp("2026-04-12T00:00:00+0000", tz="UTC"),
        },
    ]
