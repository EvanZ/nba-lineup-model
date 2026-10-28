from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    G_LEAGUE_ASSIGNMENT,
    INACTIVE,
    MEDICAL_ABSENCE,
    PLAYED,
    UNKNOWN,
    InSeasonMinutesConfig,
    _capped_team_allocation,
    advance_minutes_state,
    calibrate_conditional_prior_games,
    calibrate_prior_games,
    contribution_sha256,
    forecast_team_minutes,
    initialize_minutes_state,
    publish_minutes_contribution,
    regulation_equivalent_minutes,
)
from nba_lineup_model.modeling.in_season_minutes_monitor import (
    _classification,
    _load_cdn_play_by_play,
    _pregame_roster_player_ids,
    _projection_margin,
    _validate_non_replayed_source,
    apply_early_game_role_observations,
    build_game_observations,
)
from nba_lineup_model.season.slate_time import summary_slate_date


def _config() -> InSeasonMinutesConfig:
    return InSeasonMinutesConfig(
        updater_version="test-v1",
        participation_prior_games=8.0,
        conditional_prior_games=4.0,
        rotation_size=2,
    )


def test_summary_slate_date_uses_pacific_date_across_utc_midnight() -> None:
    summary = {
        "game_date": "2026-10-25",
        "pregame": {"start_time": "2026-10-25T00:30:00Z"},
    }
    assert summary_slate_date(summary).isoformat() == "2026-10-24"

    late_west_game = {
        "game_date": "2026-10-26",
        "pregame": {"start_time": "2026-10-26T01:00:00Z"},
    }
    assert summary_slate_date(late_west_game).isoformat() == "2026-10-25"


def test_projection_margin_combines_requested_ratings_and_minutes() -> None:
    summary = {"pregame": {"projection": {"home_margin": 0.0}}}
    margin = _projection_margin(
        summary,
        home_minutes={3: 48.0},
        away_minutes={2: 48.0},
        frozen_home={1: 48.0},
        frozen_away={2: 48.0},
        preseason_ratings={1: 1.0, 2: 0.0, 3: 4.0},
        forecast_ratings={1: 2.0, 2: 0.0, 3: 5.0},
    )

    assert margin == pytest.approx(4.0)


def _priors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["One", "Two", "Three"],
            "team": ["AAA", "AAA", "AAA"],
            "availability_probability": [0.8, 0.9, 0.5],
            "conditional_minutes_per_game": [30.0, 20.0, 10.0],
            "baseline_minutes_per_game": [144.0, 96.0, 0.0],
        }
    )


def _observations(game_id: str = "001") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "player_id": [1, 2, 3],
            "outcome_classification": [PLAYED, MEDICAL_ABSENCE, UNKNOWN],
            "played": [True, False, False],
            "regulation_equivalent_minutes": [36.0, 0.0, 0.0],
            "conditional_observation_weight": [1.0, 0.0, 0.0],
        }
    )


def test_zero_evidence_reproduces_frozen_preseason_distribution() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[],
        report_status="submitted",
        config=_config(),
    ).set_index("player_id")

    assert forecast.loc[1, "final_normalized_minutes"] == pytest.approx(144.0)
    assert forecast.loc[2, "final_normalized_minutes"] == pytest.approx(96.0)
    assert forecast.loc[3, "final_normalized_minutes"] == pytest.approx(0.0)
    assert forecast["final_normalized_minutes"].sum() == pytest.approx(240.0)


def test_historical_forecast_uses_frozen_roster_across_team_change() -> None:
    priors = _priors()
    priors.loc[priors["player_id"].eq(1), "team"] = "BBB"
    state = initialize_minutes_state(priors, season="2026-27", config=_config())

    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[],
        report_status="submitted",
        config=_config(),
        roster_player_ids={1, 2},
    )

    assert set(forecast["player_id"]) == {1, 2}
    assert set(forecast["team"]) == {"AAA"}
    assert forecast["final_normalized_minutes"].sum() == pytest.approx(240.0)


def test_pregame_roster_ids_are_game_scoped_not_current_team_scoped() -> None:
    summary = {
        "home_team": "AAA",
        "away_team": "BBB",
        "pregame": {
            "projected_minutes": {
                "home": [{"player_id": 1}, {"player_id": 2}],
                "away": [{"player_id": 3}],
            }
        },
    }

    assert _pregame_roster_player_ids(summary, "AAA") == {1, 2}
    assert _pregame_roster_player_ids(summary, "BBB") == {3}


def test_injury_absence_does_not_erase_conditional_role() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    updated = advance_minutes_state(
        state,
        _observations(),
        config=_config(),
        contribution_sha256="a" * 64,
    ).set_index("player_id")

    assert updated.loc[2, "current_participation_estimate"] < 0.9
    assert updated.loc[2, "current_conditional_minutes_estimate"] == pytest.approx(20.0)
    assert updated.loc[2, "conditional_observations"] == 0
    assert updated.loc[3, "current_participation_estimate"] == pytest.approx(0.5)


def test_g_league_assignment_is_zero_minute_available_opportunity() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    observation = _observations().iloc[[0]].copy()
    observation["outcome_classification"] = G_LEAGUE_ASSIGNMENT
    observation["played"] = False
    observation["regulation_equivalent_minutes"] = 0.0
    updated = advance_minutes_state(
        state,
        observation,
        config=_config(),
        contribution_sha256="g" * 64,
    ).set_index("player_id")

    assert updated.loc[1, "participation_observations"] == 1
    assert updated.loc[1, "conditional_observations"] == 1
    assert updated.loc[1, "current_conditional_minutes_estimate"] == pytest.approx(24.0)


def test_out_g_league_reason_overrides_generic_out_status() -> None:
    assert (
        _classification(
            minutes=0.0,
            did_not_play=True,
            espn_reason=None,
            injury_status="Out",
            injury_reason="G League - Two-Way",
        )
        == G_LEAGUE_ASSIGNMENT
    )
    assert (
        _classification(
            minutes=0.0,
            did_not_play=True,
            espn_reason=None,
            injury_status="Out",
            injury_reason="Injury/Illness - Right ankle sprain",
        )
        == MEDICAL_ABSENCE
    )


def test_state_prefers_early_game_role_target_over_realized_minutes() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    observation = _observations().iloc[[0]].copy()
    observation["regulation_equivalent_minutes"] = 20.0
    observation["early_game_role_minutes"] = 34.0
    updated = advance_minutes_state(
        state,
        observation,
        config=_config(),
        contribution_sha256="d" * 64,
    ).set_index("player_id")

    assert updated.loc[1, "current_conditional_minutes_estimate"] == pytest.approx(30.8)


def test_conditional_state_uses_locked_three_game_window_and_point_two_prior() -> None:
    config = InSeasonMinutesConfig(
        updater_version="early-game-test",
        participation_prior_games=4.0,
        conditional_prior_games=0.2,
        conditional_clip_delta=48.0,
        conditional_lookback_games=3,
    )
    state = initialize_minutes_state(_priors(), season="2026-27", config=config)
    for index, role in enumerate((36.0, 24.0, 48.0, 12.0), start=1):
        observation = _observations(f"g{index}").iloc[[0]].copy()
        observation["regulation_equivalent_minutes"] = role
        observation["early_game_role_minutes"] = role
        state = advance_minutes_state(
            state,
            observation,
            config=config,
            contribution_sha256=str(index) * 64,
        )

    player = state.set_index("player_id").loc[1]
    assert player["conditional_observations"] == 4
    assert player["conditional_window_observations"] == 3
    assert json.loads(player["conditional_history_json"]) == [
        {"minutes": 24.0, "weight": 1.0},
        {"minutes": 48.0, "weight": 1.0},
        {"minutes": 12.0, "weight": 1.0},
    ]
    assert player["current_conditional_minutes_estimate"] == pytest.approx(
        (0.2 * 30.0 + 24.0 + 48.0 + 12.0) / 3.2
    )


def test_missing_play_by_play_falls_back_to_realized_minutes() -> None:
    observations = _observations().assign(
        team="AAA",
        player_name=["One", "Two", "Three"],
    )
    result = apply_early_game_role_observations(
        observations,
        {"game_id": "001", "home_team": "AAA", "away_team": "BBB"},
        {},
        None,
        priors=_priors(),
    )

    assert result["early_game_role_minutes"].tolist() == pytest.approx(
        result["regulation_equivalent_minutes"].tolist()
    )
    assert result["early_role_status"].eq("fallback").all()


def test_post_cutoff_only_player_can_have_zero_early_role(monkeypatch) -> None:
    observations = pd.DataFrame(
        {
            "player_id": [1, 2],
            "played": [True, True],
            "actual_minutes": [34.0, 4.0],
            "regulation_equivalent_minutes": [34.0, 4.0],
        }
    )
    events = [
        SimpleNamespace(elapsed_game_seconds=0.0),
        SimpleNamespace(elapsed_game_seconds=2880.0),
    ]
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.canonical_events",
        lambda payload: events,
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor._lineup_boxscore_from_espn",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.reconstruct_lineups",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.lineup_stints_frame",
        lambda reconstruction: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.competitive_role_minutes",
        lambda frame, boundary_seconds: {1: 34.0},
    )

    result = apply_early_game_role_observations(
        observations,
        {"game_id": "001", "home_team": "AAA", "away_team": "BBB"},
        {},
        {"game": {}},
        priors=_priors(),
    ).set_index("player_id")

    assert result["early_role_status"].eq("reconstructed").all()
    assert result.loc[2, "early_game_role_minutes"] == 0.0
    assert result.loc[1, "role_cutoff_seconds"] == 44 * 60


def test_missing_early_player_fails_when_minutes_exceed_post_cutoff_window(
    monkeypatch,
) -> None:
    observations = pd.DataFrame(
        {
            "player_id": [1, 2],
            "played": [True, True],
            "actual_minutes": [34.0, 8.0],
            "regulation_equivalent_minutes": [34.0, 8.0],
        }
    )
    events = [
        SimpleNamespace(elapsed_game_seconds=0.0),
        SimpleNamespace(elapsed_game_seconds=2880.0),
    ]
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.canonical_events",
        lambda payload: events,
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor._lineup_boxscore_from_espn",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.reconstruct_lineups",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.lineup_stints_frame",
        lambda reconstruction: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.competitive_role_minutes",
        lambda frame, boundary_seconds: {1: 34.0},
    )

    result = apply_early_game_role_observations(
        observations,
        {"game_id": "001", "home_team": "AAA", "away_team": "BBB"},
        {},
        {"game": {}},
        priors=_priors(),
    )

    assert result["early_role_status"].eq("fallback").all()
    assert result["early_role_reason"].str.contains("omit played players").all()


def test_cdn_play_by_play_uses_archive_endpoint_value(monkeypatch) -> None:
    raw = b'{"game":{"gameId":"0022600001"}}'
    digest = hashlib.sha256(raw).hexdigest()
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        assert kwargs == {"capture_output": True, "check": True}
        return SimpleNamespace(stdout=raw)

    monkeypatch.setattr(
        "nba_lineup_model.modeling.in_season_minutes_monitor.subprocess.run",
        fake_run,
    )

    payload, observed_digest = _load_cdn_play_by_play(
        "0022600001",
        coverage={"0022600001": [digest]},
        lake_uri="s3://nba-gestalt-data/lake",
    )

    assert payload == {"game": {"gameId": "0022600001"}}
    assert observed_digest == digest
    assert captured["command"][3] == (
        "s3://nba-gestalt-data/lake/raw/cdn/playbyplay/season=2026-27/"
        f"game_id=0022600001/{digest}.json"
    )


def test_inactive_player_updates_participation_only() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    observation = _observations().iloc[[0]].copy()
    observation["player_id"] = 2
    observation["outcome_classification"] = INACTIVE
    observation["played"] = False
    observation["regulation_equivalent_minutes"] = 0.0
    updated = advance_minutes_state(
        state, observation, config=_config(), contribution_sha256="b" * 64
    ).set_index("player_id")
    assert updated.loc[2, "current_participation_estimate"] < 0.9
    assert updated.loc[2, "current_conditional_minutes_estimate"] == pytest.approx(20.0)


def test_available_coach_dnp_updates_conditional_role_with_zero_minutes() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    observation = _observations().iloc[[0]].copy()
    observation["player_id"] = 2
    observation["outcome_classification"] = COACH_DNP
    observation["played"] = False
    observation["regulation_equivalent_minutes"] = 0.0
    observation["conditional_observation_weight"] = 1.0
    updated = advance_minutes_state(
        state, observation, config=_config(), contribution_sha256="c" * 64
    ).set_index("player_id")

    assert updated.loc[2, "current_conditional_minutes_estimate"] == pytest.approx(16.0)
    assert updated.loc[2, "conditional_observations"] == 1
    assert updated.loc[2, "effective_conditional_weight"] == 1.0


def test_overtime_minutes_are_regulation_equivalent() -> None:
    normalized = regulation_equivalent_minutes(
        pd.Series([50.0, 50.0, 50.0, 50.0, 50.0]), team_total_minutes=250.0
    )
    assert normalized.sum() == pytest.approx(240.0)
    assert normalized.tolist() == pytest.approx([48.0] * 5)


def test_team_allocation_caps_one_player_and_redistributes_overflow() -> None:
    allocation = _capped_team_allocation(
        pd.Series([100.0, 20.0, 15.0, 10.0, 8.0, 7.0])
    )

    assert allocation.sum() == pytest.approx(240.0)
    assert allocation.max() == pytest.approx(48.0)
    assert (allocation >= 0.0).all()


def test_out_player_gets_zero_and_remaining_minutes_conserve_240() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[{"player_id": 1, "status": "Out", "reason": "Ankle"}],
        report_status="submitted",
        config=_config(),
    ).set_index("player_id")
    assert forecast.loc[1, "final_normalized_minutes"] == 0.0
    assert forecast["final_normalized_minutes"].sum() == pytest.approx(240.0)
    assert forecast.loc[1, "injury_reason"] == "Ankle"


def test_report_pending_is_preserved_without_available_claim() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    forecast = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[],
        report_status="not_yet_submitted",
        config=_config(),
    )
    assert set(forecast["injury_status"]) == {"Report pending"}
    assert forecast["game_participation_probability"].isna().all()
    assert forecast.set_index("player_id").loc[
        1, "allocation_participation_probability"
    ] == pytest.approx(0.8)
    assert forecast.set_index("player_id").loc[1, "final_normalized_minutes"] == pytest.approx(
        240.0 * 24.0 / 42.0
    )


def test_updated_allocation_uses_posterior_role_not_archived_minutes() -> None:
    state = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    state.loc[state["player_id"].eq(1), "current_conditional_minutes_estimate"] = 36.0
    state.loc[state["player_id"].eq(2), "current_conditional_minutes_estimate"] = 16.0
    first = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=[
            {
                "player_id": 1,
                "status": "Unlisted",
                "conditional_minutes": 30.0,
                "projected_minutes": 1.0,
            },
            {
                "player_id": 2,
                "status": "Unlisted",
                "conditional_minutes": 20.0,
                "projected_minutes": 239.0,
            },
            {
                "player_id": 3,
                "status": "Unlisted",
                "conditional_minutes": 10.0,
                "projected_minutes": 0.0,
            },
        ],
        report_status="submitted",
        config=_config(),
    ).set_index("player_id")

    expected_one = 240.0 * 36.0 / (36.0 + 16.0)
    assert first.loc[1, "final_normalized_minutes"] == pytest.approx(expected_one)
    assert first.loc[2, "final_normalized_minutes"] == pytest.approx(240.0 - expected_one)
    assert first.loc[3, "final_normalized_minutes"] == pytest.approx(0.0)

    conflicting_archived_rows = [
        {
            "player_id": int(player_id),
            "status": "Unlisted",
            "conditional_minutes": 48.0 - float(player_id),
            "projected_minutes": 240.0 if player_id == 3 else 0.0,
        }
        for player_id, row in first.iterrows()
    ]
    second = forecast_team_minutes(
        state,
        team="AAA",
        game_id="001",
        forecast_cutoff="2026-10-20T00:00:00Z",
        injury_rows=conflicting_archived_rows,
        report_status="submitted",
        config=_config(),
    ).set_index("player_id")
    assert second["final_normalized_minutes"].to_dict() == pytest.approx(
        first["final_normalized_minutes"].to_dict()
    )


def test_replayed_schedule_summary_is_rejected_as_updater_input() -> None:
    summary = {
        "game_id": "0022600001",
        "record_type": "strict_forward_replay_final_game_audit",
        "pregame": {
            "projection": {"state_kind": "in_season_strict_forward_replay"}
        },
    }

    with pytest.raises(ValueError, match="derived Schedule replay"):
        _validate_non_replayed_source(summary)


def test_incremental_and_full_replay_match_and_duplicate_is_rejected() -> None:
    initial = initialize_minutes_state(_priors(), season="2026-27", config=_config())
    first = _observations("001")
    second = _observations("002")
    incremental = advance_minutes_state(
        advance_minutes_state(
            initial, first, config=_config(), contribution_sha256="1" * 64
        ),
        second,
        config=_config(),
        contribution_sha256="2" * 64,
    )
    full = advance_minutes_state(
        initial,
        pd.concat([first, second], ignore_index=True),
        config=_config(),
        contribution_sha256="3" * 64,
    )
    columns = [
        "player_id",
        "current_participation_estimate",
        "current_conditional_minutes_estimate",
        "participation_observations",
        "conditional_observations",
    ]
    pd.testing.assert_frame_equal(
        incremental[columns].reset_index(drop=True), full[columns].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="cannot be applied twice"):
        advance_minutes_state(
            incremental, second, config=_config(), contribution_sha256="2" * 64
        )


def test_content_addressed_contribution_is_idempotent_and_revision_is_new(tmp_path) -> None:
    observations = _observations()
    metadata = {"game_id": "001", "source_sha256": "s1", "finalized_at": "now"}
    first, created, digest = publish_minutes_contribution(observations, metadata, tmp_path)
    second, created_again, same_digest = publish_minutes_contribution(
        observations, metadata, tmp_path
    )
    assert created is True
    assert created_again is False
    assert first == second
    assert digest == same_digest == contribution_sha256(observations, metadata)
    revised = observations.copy()
    revised.loc[0, "regulation_equivalent_minutes"] = 35.0
    third, revised_created, revised_digest = publish_minutes_contribution(
        revised, {**metadata, "source_sha256": "s2"}, tmp_path
    )
    assert revised_created is True
    assert third != first
    assert revised_digest != digest


def test_calibration_replay_is_strictly_prior_day() -> None:
    prior = pd.DataFrame(
        {
            "game_id": ["p1", "p1"],
            "game_date": pd.to_datetime(["2025-01-01", "2025-01-01"], utc=True),
            "team": ["AAA", "AAA"],
            "player_id": [1, 2],
            "minutes": [30.0, 18.0],
            "availability_state": [PLAYED, PLAYED],
        }
    )
    target = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "game_date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], utc=True
            ),
            "team": ["AAA"] * 4,
            "player_id": [1, 2, 1, 2],
            "minutes": [36.0, 0.0, 34.0, 20.0],
            "availability_state": [PLAYED, COACH_DNP, PLAYED, PLAYED],
        }
    )
    result = calibrate_prior_games(
        prior,
        target,
        participation_grid=(1.0, 4.0),
        conditional_grid=(1.0, 4.0),
    )
    assert result["target_player_games"] == 4
    assert result["participation_prior_games"] in {1.0, 4.0}
    assert result["conditional_prior_games"] in {1.0, 4.0}
    assert json.loads(json.dumps(result)) == result


def test_conditional_calibration_accepts_canonical_availability_states() -> None:
    priors = pd.DataFrame(
        {
            "player_id": [1, 2],
            "predicted_minutes_per_available_game": [30.0, 18.0],
        }
    )
    target = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "game_date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], utc=True
            ),
            "team": ["AAA"] * 4,
            "player_id": [1, 2, 1, 2],
            "minutes": [36.0, 0.0, 34.0, 20.0],
            "availability_state": [
                "played",
                "available_dnp_coach",
                "played",
                "played",
            ],
        }
    )

    result = calibrate_conditional_prior_games(
        priors, target, conditional_grid=(0.1, 4.0)
    )

    assert result["target_available_player_games"] == 4
    assert result["conditional_prior_games"] in {0.1, 4.0}
    assert json.loads(json.dumps(result)) == result


def test_final_boxscore_uses_frozen_injury_status_for_available_player_role() -> None:
    priors = _priors()
    summary = {
        "game_id": "001",
        "game_date": "2026-10-20",
        "home_team": "AAA",
        "away_team": "BBB",
        "pregame": {
            "projected_minutes": {
                "home": [
                    {
                        "player_id": 1,
                        "status": "Available",
                        "reason": None,
                    },
                    {
                        "player_id": 2,
                        "status": "Out",
                        "reason": "Injury/Illness - Ankle",
                    },
                    {
                        "player_id": 3,
                        "status": "Out",
                        "reason": "G League - Two-Way",
                    },
                ],
                "away": [],
            }
        },
    }
    espn = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "AAA"},
                    "statistics": [
                        {
                            "labels": ["MIN"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "One"},
                                    "stats": ["48"],
                                    "didNotPlay": False,
                                    "ejected": False,
                                },
                                {
                                    "athlete": {"displayName": "Two"},
                                    "stats": ["0"],
                                    "didNotPlay": True,
                                    "reason": "DNP-INJURY/ILLNESS",
                                    "ejected": False,
                                },
                                {
                                    "athlete": {"displayName": "Three"},
                                    "stats": ["0"],
                                    "didNotPlay": True,
                                    "reason": None,
                                    "ejected": False,
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }
    observations = build_game_observations(summary, espn, priors=priors).set_index(
        "player_id"
    )
    assert observations.loc[1, "outcome_classification"] == PLAYED
    assert observations.loc[2, "outcome_classification"] == MEDICAL_ABSENCE
    assert observations.loc[3, "outcome_classification"] == G_LEAGUE_ASSIGNMENT
    assert observations.loc[2, "conditional_observation_weight"] == 0.0
    assert observations.loc[3, "conditional_observation_weight"] == 1.0
    assert observations.loc[1, "regulation_equivalent_minutes"] == pytest.approx(240.0)


def test_final_boxscore_follows_unique_player_identity_across_team_change() -> None:
    priors = _priors()
    priors.loc[priors["player_id"].eq(1), "team"] = "NEW"
    priors.loc[priors["player_id"].eq(2), "team"] = "OLD"
    summary = {
        "game_id": "001",
        "game_date": "2026-10-20",
        "home_team": "OLD",
        "away_team": "BBB",
        "pregame": {
            "projected_minutes": {
                "home": [],
                "away": [],
            }
        },
    }
    espn = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "OLD"},
                    "statistics": [
                        {
                            "labels": ["MIN"],
                            "athletes": [
                                {
                                    "athlete": {"displayName": "One"},
                                    "stats": ["0"],
                                    "didNotPlay": True,
                                    "reason": "COACH'S DECISION",
                                    "ejected": False,
                                },
                                {
                                    "athlete": {"displayName": "Two"},
                                    "stats": ["48"],
                                    "didNotPlay": False,
                                    "ejected": False,
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }

    observations = build_game_observations(summary, espn, priors=priors).set_index(
        "player_id"
    )

    assert observations.loc[1, "team"] == "OLD"
    assert observations.loc[1, "outcome_classification"] == COACH_DNP
    assert observations.loc[1, "identity_match_source"] == "unique_name_transfer"
