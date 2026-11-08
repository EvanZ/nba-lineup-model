from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nba_lineup_model.modeling.current_season_serving_materialization import (
    materialize_daily_model_serving,
)
from nba_lineup_model.season.daily_data_artifact import (
    promote_daily_data_artifact,
    publish_daily_data_artifact,
)
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.daily_model_replay import (
    CausalHcaReplayConfig,
    DailyModelReplayBlocked,
    HistoricalModelReplayApprovalRequired,
    _classification,
    _early_role_minutes,
    _longest_compatible_prefix,
    _normalize_name,
    _require_historical_replay_approval,
    replay_daily_model_chain,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _prior_files(
    tmp_path: Path,
    *,
    omit_player: int | None = None,
    omit_minutes_player: int | None = None,
) -> tuple[Path, Path]:
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
        if player_id != omit_player and player_id != omit_minutes_player
    ]
    minutes_path = tmp_path / "minutes.json"
    minutes_path.write_text(json.dumps({"minutes": {"players": minutes_players}}))
    return rankings_path, minutes_path


def _publish_data(
    tmp_path: Path,
    *,
    injury_players: list[dict[str, object]] | None = None,
    roster_name_overrides: dict[int, str] | None = None,
):
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
            player_name = (roster_name_overrides or {}).get(
                player_id, f"Player {player_id}"
            )
            players.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
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
                    "player_name": player_name,
                    "minutes": 48.0,
                }
            )
            identity_rows.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "player_name": player_name,
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
                        "players": injury_players or [],
                    }
                }
            ]
        )
    )
    (component_root / "nba_inactive_lists.json").write_text(
        json.dumps([{"game_id": game_id, "payload": {"game_id": game_id, "teams": []}}])
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
            "nba_inactive_lists": "nba_inactive_lists.json",
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
                "nba_inactive_list",
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

    release_root = tmp_path / "release"
    release_root.mkdir()
    (release_root / "release.json").write_text(
        json.dumps({"run_id": "frozen-preseason-run"})
    )
    serving_path = materialize_daily_model_serving(
        daily_model_root=tmp_path / "model",
        daily_data_root=tmp_path / "data",
        release_root=release_root,
        output_root=tmp_path / "serving",
    )
    serving_manifest = json.loads((serving_path / "manifest.json").read_text())
    model_manifest = verify_daily_model_artifact(published[0].path, data_bundle=data.path)
    model_sha = model_manifest["bundle_sha256"]
    identity = serving_manifest["identity"]
    assert identity["evidence_through_date"] == "2026-10-20"
    assert identity["coverage"]["game_count"] == 1
    assert identity["data_quality"]["source_coverage"] == {
        "ratings": {"games": 1, "excluded": 0},
        "minutes": {"games": 1, "excluded": 0},
    }
    assert all(
        model_sha in source["uri"] for source in identity["sources"].values()
    )
    assert (tmp_path / "serving" / "latest.json").is_file()

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


def test_causal_hca_replay_publishes_an_independent_joint_state(tmp_path: Path) -> None:
    data = _publish_data(tmp_path)
    rankings, minutes = _prior_files(tmp_path)
    config = CausalHcaReplayConfig(
        home_court_prior_mean=1.7,
        home_court_prior_equivalent_possessions=40000.0,
        home_court_prior_source="completed-through-2025-26:test",
    )

    published = replay_daily_model_chain(
        start_date=date(2026, 10, 20),
        end_date=date(2026, 10, 20),
        preseason_rankings_path=rankings,
        preseason_minutes_path=minutes,
        data_root=tmp_path / "data",
        output_root=tmp_path / "causal-hca-candidate",
        config=config,
        code_revision="test-causal-hca-revision",
        context_provider=_zero_context,
    )

    assert len(published) == 1
    manifest = verify_daily_model_artifact(published[0].path, data_bundle=data.path)
    components = manifest["identity"]["components"]
    assert "home_court_state" in components
    hca_state = json.loads(
        (published[0].path / components["home_court_state"]["path"]).read_text()
    )
    assert hca_state["home_court_candidate_id"] == (
        "causal_hca_replacement_candidate_v2"
    )
    assert hca_state["home_court_prior_mean"] == 1.7
    assert hca_state["home_court_prior_source"] == "completed-through-2025-26:test"
    gram = sparse.load_npz(published[0].path / components["rating_gram"]["path"])
    assert gram.shape == (11, 11)
    ratings = pd.read_parquet(
        published[0].path / components["player_ratings"]["path"]
    )
    assert ratings["home_court_estimate"].nunique() == 1
    diagnostics = json.loads(
        (published[0].path / components["evaluation_metrics"]["path"]).read_text()
    )
    assert diagnostics["home_court_candidate_id"] == config.model_version


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


def test_daily_model_replay_rejects_decoupled_preseason_priors(
    tmp_path: Path,
) -> None:
    _publish_data(tmp_path)
    rankings, minutes = _prior_files(tmp_path, omit_minutes_player=1631104)

    with pytest.raises(
        ValueError,
        match="Coupled preseason prior player universe mismatch",
    ):
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


def test_daily_model_replay_fails_closed_on_unmapped_injury_player(
    tmp_path: Path,
) -> None:
    _publish_data(
        tmp_path,
        injury_players=[
            {
                "game_date": "2026-10-20",
                "matchup": "AWY@HOM",
                "team": "HOM",
                "player_name": "Unknown Player, Jr.",
                "status": "Out",
                "reason": "Injury/Illness",
            }
        ],
    )
    rankings, minutes = _prior_files(tmp_path)

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

    error = json.loads(
        (
            tmp_path
            / "model/failures/game_date_pacific=2026-10-20/error.json"
        ).read_text()
    )
    assert "Official NBA injury player identity is unresolved" in error["error"]
    assert "Unknown Player, Jr.: unmapped" in error["error"]


def test_daily_model_replay_infers_blank_injury_team_from_coupled_prior(
    tmp_path: Path,
) -> None:
    _publish_data(
        tmp_path,
        injury_players=[
            {
                "game_date": "2026-10-20",
                "matchup": "AWY@HOM",
                "player_name": "Player 1631104",
                "status": "Available",
                "reason": "Injury/Illness",
            }
        ],
    )
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

    manifest = verify_daily_model_artifact(published[0].path)
    observations = pd.read_parquet(
        published[0].path
        / manifest["identity"]["components"]["minutes_observations"]["path"]
    )
    player = observations.loc[observations["player_id"].eq(1631104)].iloc[0]
    assert player["team"] == "HOM"
    assert player["injury_status"] == "Available"


def test_daily_model_replay_uses_official_roster_name_alias(
    tmp_path: Path,
) -> None:
    _publish_data(
        tmp_path,
        roster_name_overrides={1631104: "Hansen Yang"},
        injury_players=[
            {
                "game_date": "2026-10-20",
                "matchup": "AWY@HOM",
                "team": "HOM",
                "player_name": "Hansen Yang",
                "status": "Available",
                "reason": "Injury/Illness",
            }
        ],
    )
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

    manifest = verify_daily_model_artifact(published[0].path)
    observations = pd.read_parquet(
        published[0].path
        / manifest["identity"]["components"]["minutes_observations"]["path"]
    )
    player = observations.loc[observations["player_id"].eq(1631104)].iloc[0]
    assert player["injury_status"] == "Available"


@pytest.mark.parametrize(
    ("team_report_status", "injury_status", "injury_reason", "expected"),
    [
        ("submitted", None, None, "coach_dnp"),
        ("not_yet_submitted", None, None, "unknown"),
        ("submitted", "Available", None, "coach_dnp"),
        ("submitted", "Out", "Injury/Illness - Left Knee", "medical_absence"),
        ("submitted", "Out", "G League - On Assignment", "g_league_assignment"),
    ],
)
def test_inactive_classification_uses_the_submitted_official_report_rule(
    team_report_status, injury_status, injury_reason, expected
) -> None:
    assert _classification(
        minutes=0.0,
        roster_status="INACTIVE",
        injury_status=injury_status,
        injury_reason=injury_reason,
        team_report_status=team_report_status,
    ) == expected


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


def test_model_prefix_stops_immediately_before_a_repaired_data_date() -> None:
    provenance = {
        "code_revision": "revision",
        "model_version": "model",
        "configuration_sha256": "a" * 64,
        "prior_sha256": "b" * 64,
    }
    dates = [date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22)]
    original_data = [
        (
            game_date,
            Path(f"/data/{game_date}"),
            {"bundle_sha256": checksum},
        )
        for game_date, checksum in zip(dates, ("1" * 64, "2" * 64, "3" * 64))
    ]
    model_chain = [
        (
            game_date,
            Path(f"/model/{game_date}"),
            {
                "identity": {
                    "input_data_sha256": data_manifest["bundle_sha256"],
                    "model_provenance": provenance,
                }
            },
        )
        for (game_date, _path, data_manifest) in original_data
    ]
    repaired_data = list(original_data)
    repaired_data[1] = (
        dates[1],
        Path(f"/data/{dates[1]}-repaired"),
        {"bundle_sha256": "9" * 64},
    )

    compatible = _longest_compatible_prefix(
        data_chain=repaired_data,
        model_chain=model_chain,
        provenance=provenance,
        through=dates[-1],
    )

    assert [row[0] for row in compatible] == [dates[0]]


def test_repaired_data_requires_exact_historical_model_replay_approval() -> None:
    provenance = {
        "code_revision": "revision",
        "model_version": "model",
        "configuration_sha256": "a" * 64,
        "prior_sha256": "b" * 64,
    }
    dates = [date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22)]
    data_chain = [
        (game_date, Path(f"/data/{game_date}"), {"bundle_sha256": checksum})
        for game_date, checksum in zip(dates, ("1" * 64, "9" * 64, "3" * 64))
    ]
    model_chain = [
        (
            game_date,
            Path(f"/model/{game_date}"),
            {
                "identity": {
                    "input_data_sha256": checksum,
                    "model_provenance": provenance,
                }
            },
        )
        for game_date, checksum in zip(dates, ("1" * 64, "2" * 64, "3" * 64))
    ]
    compatible = _longest_compatible_prefix(
        data_chain=data_chain,
        model_chain=model_chain,
        provenance=provenance,
        through=dates[-1],
    )

    with pytest.raises(
        HistoricalModelReplayApprovalRequired,
        match="2026-10-21 requires explicit approval",
    ):
        _require_historical_replay_approval(
            data_chain=data_chain,
            model_chain=model_chain,
            compatible=compatible,
            through=dates[-1],
            approved_replay_from=None,
        )
    with pytest.raises(HistoricalModelReplayApprovalRequired):
        _require_historical_replay_approval(
            data_chain=data_chain,
            model_chain=model_chain,
            compatible=compatible,
            through=dates[-1],
            approved_replay_from=dates[-1],
        )
    _require_historical_replay_approval(
        data_chain=data_chain,
        model_chain=model_chain,
        compatible=compatible,
        through=dates[-1],
        approved_replay_from=dates[1],
    )


def test_historical_genesis_replay_accepts_only_the_first_data_date() -> None:
    first = date(2026, 10, 20)
    data_chain = [(first, Path("/data/first"), {"bundle_sha256": "1" * 64})]

    _require_historical_replay_approval(
        data_chain=data_chain,
        model_chain=[],
        compatible=[],
        through=first,
        approved_replay_from=first,
    )
    with pytest.raises(
        ValueError,
        match="approval must name the first data date",
    ):
        _require_historical_replay_approval(
            data_chain=data_chain,
            model_chain=[],
            compatible=[],
            through=first,
            approved_replay_from=date(2026, 10, 21),
        )


def test_official_nba_suffix_formats_share_one_identity_key() -> None:
    expected = _normalize_name("Ernest Udeh")

    assert _normalize_name("Ernest Udeh, Jr.") == expected
    assert _normalize_name("Jr., Ernest Udeh") == expected


@pytest.mark.parametrize("classification", ["medical_absence", "g_league_assignment", "coach_dnp", "inactive_or_assignment"])
def test_trajectory_includes_official_observation_absent_from_box(classification):
    from nba_lineup_model.season.daily_model_replay import _trajectory_rows
    # Arbitrary IDs: no special handling for any real player or team.
    observations = pd.DataFrame([
        {"game_id":"g1", "player_id":id, "player_name":f"Player {id}", "team":"HOM",
         "played":id == 1, "actual_minutes":30.0 if id == 1 else 0.0,
         "outcome_classification":"played" if id == 1 else classification,
         "injury_status":None if id == 1 else "Out", "injury_reason":None if id == 1 else classification}
        for id in [1, 2]
    ])
    before=pd.DataFrame({"player_id":[1,2],"updated_nail":[1.0,2.0]})
    after=pd.DataFrame({"player_id":[1,2],"updated_nail":[1.1,2.02]})
    rows=_trajectory_rows(game_date=date(2026,11,1),
        games={"g1":{"season":"2026-27","start_time":"2026-11-01T22:00:00Z","home_team":"HOM","away_team":"AWY"}},
        rosters={"g1":{"teams":[{"team":"HOM","players":[{"player_id":1}]}]}},
        boxscores={},observations=observations,before=before,after=after)
    assert set(rows.player_id)=={1,2}
    missed=rows.set_index("player_id").loc[2]
    assert not missed.played
    assert missed.actual_minutes==0
    assert missed.opponent=="AWY"
    assert missed.outcome_classification==classification
    assert missed.rating_after==2.02  # exact slate checkpoint, not fabricated flat rating
    assert missed.injury_status=="Out"


@pytest.fixture
def cenac_injury_observations(monkeypatch):
    from nba_lineup_model.season import daily_model_replay as replay

    def observe(name, *, player_id=1643416, roster_name="Chris Cenac Jr.",
                report_team="BOS", collision=False):
        teams = [{"team": team, "players": [
            {"player_id": offset + i, "player_name": f"Player {offset+i}",
             "minutes": 48.0, "status": "ACTIVE"} for i in range(5)
        ]} for team, offset in (("BOS", 100), ("MIL", 200))]
        teams[0]["players"].append({"player_id": player_id,
            "player_name": roster_name, "minutes": 0.0, "status": "ACTIVE"})
        if collision:
            teams[0]["players"].append({"player_id": 999,
                "player_name": "Christopher Cenac Jr.", "minutes": 0.0, "status": "ACTIVE"})
        state = pd.DataFrame([{**p, "team": t["team"]}
            for t in teams for p in t["players"]])
        monkeypatch.setattr(replay, "_early_role_minutes",
            lambda *args, **kwargs: ({int(p): 44.0 for p in state.player_id}, None))
        return replay._minutes_observations(
            game_date=date(2026, 11, 4),
            games={"g1": {"home_team": "BOS", "away_team": "MIL"}},
            rosters={"g1": {"teams": teams}},
            stints=pd.DataFrame({"game_id": []}), minutes_state=state,
            role_cutoff_minutes=44, injury_report={"players": [{
                "game_date": "2026-11-04", "matchup": "MIL@BOS",
                "team": report_team, "player_name": name,
                "status": "Available", "reason": "-"}],
                "game_team_reports": {"2026-11-04|MIL@BOS": {
                    "BOS": {"status": "submitted"}, "MIL": {"status": "submitted"}}}})
    return observe


def test_reviewed_cenac_alias_preserves_canonical_observations(cenac_injury_observations):
    canonical, _, warnings = cenac_injury_observations("Chris Cenac Jr.")
    aliased, injuries, alias_warnings = cenac_injury_observations("Christopher Cenac Jr.")
    pd.testing.assert_frame_equal(canonical, aliased)
    assert warnings == alias_warnings
    player = aliased.set_index("player_id").loc[1643416]
    assert player.injury_status == "Available"
    assert player.availability_resolution == "explicit_official_report"
    assert player.injury_reason == "-"
    report = injuries["g1"]["teams"]["BOS"][0]
    assert report["player_id"] == 1643416
    assert report["player_name"] == "Christopher Cenac Jr."  # preserve source spelling
    inferred, _, _ = cenac_injury_observations("Christopher Cenac Jr.", report_team="")
    pd.testing.assert_frame_equal(aliased, inferred)


@pytest.mark.parametrize("options", [
    {"player_id": 999}, {"roster_name": "Chris Different"},
    {"report_team": "MIL"}, {"collision": True},
])
def test_cenac_alias_rejects_wrong_identity_or_ambiguity(cenac_injury_observations, options):
    with pytest.raises(ValueError, match="identity is unresolved"):
        cenac_injury_observations("Christopher Cenac Jr.", **options)
