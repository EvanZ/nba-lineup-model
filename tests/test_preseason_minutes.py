from __future__ import annotations

import pandas as pd
import pytest

from nba_lineup_model.web_api import preseason_minutes
from nba_lineup_model.web_api.preseason_minutes import (
    build_forward_conditional_preseason_minutes_payload,
    build_preseason_minutes_payload,
)


def test_preseason_minutes_normalizes_prior_minutes_with_uniform_shrinkage(tmp_path) -> None:
    roster_path = tmp_path / "roster.parquet"
    pd.DataFrame(
        {
            "team_abbreviation": ["MIN", "MIN", "MIN"],
            "player_id": [1, 2, 3],
            "player_name": ["Incumbent", "Returner", "Rookie"],
            "listed_position": ["G", "F", "C"],
            "age": [25.0, 28.0, 20.0],
        }
    ).to_parquet(roster_path, index=False)
    rankings = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "rapm": [2.0, 1.0, -0.5],
            "profile_source": ["prior", "prior", "draft"],
        }
    )

    payload = build_preseason_minutes_payload(
        preseason_rankings=rankings,
        roster_path=roster_path,
        prior_minutes=pd.Series({1: 1200.0, 2: 600.0}),
        epsilon=0.25,
        nail_weight=0.0,
        initial_rotation_size=2,
    )

    players = {row["player_id"]: row for row in payload["players"]}
    assert players[1]["baseline_minute_share"] == pytest.approx(0.625)
    assert players[2]["baseline_minute_share"] == pytest.approx(0.375)
    assert players[3]["baseline_minute_share"] == pytest.approx(0.0)
    assert players[1]["is_baseline_rotation_candidate"] is True
    assert players[2]["is_baseline_rotation_candidate"] is True
    assert players[3]["is_baseline_rotation_candidate"] is False
    assert sum(row["baseline_minutes_per_game"] for row in players.values()) == pytest.approx(240.0)


def test_preseason_minutes_tilts_active_rotation_toward_preseason_nail(tmp_path) -> None:
    roster_path = tmp_path / "roster.parquet"
    pd.DataFrame(
        {
            "team_abbreviation": ["MIN", "MIN", "MIN"],
            "player_id": [1, 2, 3],
            "player_name": ["Incumbent", "Returner", "Rookie"],
            "listed_position": ["G", "F", "C"],
            "age": [25.0, 28.0, 20.0],
        }
    ).to_parquet(roster_path, index=False)
    rankings = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "rapm": [4.0, -2.0, 0.0],
            "profile_source": ["prior", "prior", "draft"],
        }
    )

    payload = build_preseason_minutes_payload(
        preseason_rankings=rankings,
        roster_path=roster_path,
        prior_minutes=pd.Series({1: 600.0, 2: 600.0}),
        epsilon=0.0,
        nail_weight=1.0,
        initial_rotation_size=2,
    )

    players = {row["player_id"]: row for row in payload["players"]}
    assert players[1]["baseline_minute_share"] > players[2]["baseline_minute_share"]
    assert players[3]["baseline_minute_share"] == pytest.approx(0.0)
    assert payload["model"] == "L20-NAIL-MSP v0.2"
    assert payload["nail_weight"] == pytest.approx(1.0)


def test_preseason_minutes_uses_neutral_fallback_for_late_roster_addition(tmp_path) -> None:
    roster_path = tmp_path / "roster.parquet"
    pd.DataFrame(
        {
            "team_abbreviation": ["MIN"],
            "player_id": [4],
            "player_name": ["Late Addition"],
            "listed_position": ["G"],
            "age": [25.0],
        }
    ).to_parquet(roster_path, index=False)
    rankings = pd.DataFrame(
        {
            "player_id": [1],
            "rapm": [2.0],
            "profile_source": ["prior"],
        }
    )

    payload = build_preseason_minutes_payload(
        preseason_rankings=rankings,
        roster_path=roster_path,
        prior_minutes=pd.Series({4: 600.0}),
    )

    player = payload["players"][0]
    assert player["projected_nail"] == pytest.approx(0.0)
    assert player["rating_source"] == "roster fallback"
    assert player["is_rating_fallback"] is True


def test_forward_conditional_payload_squashes_availability_times_minutes(
    tmp_path, monkeypatch
) -> None:
    roster_path = tmp_path / "roster.parquet"
    pd.DataFrame(
        {
            "team_id": [1, 1],
            "team_abbreviation": ["MIN", "MIN"],
            "player_id": [1, 2],
            "player_name": ["One", "Two"],
            "listed_position": ["G", "C"],
            "age": [25.0, 28.0],
        }
    ).to_parquet(roster_path, index=False)
    summary = pd.DataFrame(
        {
            "season_start_year": [2025, 2025],
            "player_id": [1, 2],
            "total_nba_minutes": [2_000.0, 1_000.0],
        }
    )
    monkeypatch.setattr(
        preseason_minutes,
        "build_availability_season_summary",
        lambda *_a, **_k: summary,
    )
    monkeypatch.setattr(
        preseason_minutes,
        "predict_availability_roster",
        lambda *_a, **_k: (
            pd.DataFrame(
                {
                    "player_id": [1, 2],
                    "predicted_available_share": [1.0, 0.5],
                    "has_prior_availability_state": [True, True],
                }
            ),
            None,
            {},
        ),
    )
    monkeypatch.setattr(
        preseason_minutes,
        "predict_conditional_minutes_roster",
        lambda *_a, **_k: (
            pd.DataFrame(
                {
                    "player_id": [1, 2],
                    "predicted_minutes_per_available_game": [20.0, 20.0],
                    "has_prior_minutes_state": [True, True],
                }
            ),
            None,
        ),
    )
    rankings = pd.DataFrame(
        {"player_id": [1, 2], "rapm": [2.0, 1.0], "profile_source": ["prior", "prior"]}
    )

    payload = build_forward_conditional_preseason_minutes_payload(
        preseason_rankings=rankings,
        roster_path=roster_path,
        initial_rotation_size=1,
    )

    players = {row["player_id"]: row for row in payload["players"]}
    assert players[1]["raw_expected_total_minutes"] == pytest.approx(1_640.0)
    assert players[2]["raw_expected_total_minutes"] == pytest.approx(0.0)
    assert players[1]["is_baseline_rotation_candidate"] is True
    assert players[2]["is_baseline_rotation_candidate"] is False
    assert players[1]["baseline_minutes_per_game"] == pytest.approx(240.0)
    assert players[2]["baseline_minutes_per_game"] == pytest.approx(0.0)
