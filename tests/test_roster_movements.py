from __future__ import annotations

import pandas as pd

from nba_lineup_model.web_api.roster_movements import build_roster_movement_payload


def test_roster_movement_payload_uses_final_regular_season_team_snapshot(tmp_path) -> None:
    roster_path = tmp_path / "roster.parquet"
    panel_path = tmp_path / "panel.parquet"
    snapshot_path = tmp_path / "snapshot.parquet"
    catalog_path = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "player_id": ["1", "2", "3"],
            "player_name": ["Trade Player", "Returner", "Rookie"],
            "team_abbreviation": ["BOS", "LAL", "NYK"],
            "experience": ["5", "4", "R"],
            "school": ["Duke", "UCLA", "Cooper Union"],
            "how_acquired": ["Traded from ATL", "Signed on 07/01/26", "#4 Pick"],
        }
    ).to_parquet(roster_path, index=False)
    pd.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "player_id": [1, 2],
            "primary_team_tricode": ["ATL", "LAL"],
            "minutes": [1600.0, 900.0],
        }
    ).to_parquet(panel_path, index=False)
    pd.DataFrame(
        {
            "player_id": ["1", "2"],
            "source_team": ["BOS", "LAL"],
        }
    ).to_parquet(snapshot_path, index=False)
    pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "college": ["Duke", "UCLA", "Cooper Union"],
            "country": ["USA", "USA", "Canada"],
        }
    ).to_parquet(catalog_path, index=False)

    payload = build_roster_movement_payload(
        current_roster_path=roster_path,
        prior_panel_path=panel_path,
        prior_snapshot_path=snapshot_path,
        player_catalog_path=catalog_path,
        preseason_rankings=pd.DataFrame({"player_id": [1, 2, 3], "rapm": [1.2, 0.0, -0.4]}),
    )

    assert payload["teams"] == ["BOS", "LAL", "NYK"]
    assert payload["returning_mover_count"] == 0
    assert payload["new_or_unmatched_current_player_count"] == 1
    assert payload["moves"] == []
    assert payload["external_arrivals"] == [
        {
            "player_id": 3,
            "player_name": "Rookie",
            "target_team": "NYK",
            "move_type": "other",
            "how_acquired": "#4 Pick",
            "school": "Cooper Union",
            "country": "Canada",
            "is_rookie": True,
            "projected_rating": -0.4,
        }
    ]
    assert payload["source_definition"] == "2025-26 final observed regular-season team"
