from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nba_lineup_model.modeling.allocation import allocation_policy_stints
from nba_lineup_model.modeling.diagnostics import (
    bootstrap_stability,
    chronological_stability,
    context_concentration,
    influence_diagnostics,
    lambda_sensitivity,
    player_diagnostics_summary,
    raw_adjusted_comparison,
)
from nba_lineup_model.models.baselines import (
    RidgeLineupModel,
    signed_entity_matrix,
    vocabulary_mapping,
)


@pytest.mark.parametrize(
    ("policy", "possessions", "points_home", "points_away"),
    [
        ("equal_segments", [0.25, 0.75], [1.0, 1.0], [0.0, 2.0]),
        ("starting_lineup", [0.5, 0.5], [2.0, 0.0], [0.0, 2.0]),
        ("terminal_lineup", [1.0], [2.0], [2.0]),
        ("boundary_split", [0.25, 0.75], [1.0, 1.0], [0.0, 2.0]),
        ("exclude_multi_lineup", [0.5], [0.0], [2.0]),
    ],
)
def test_possession_allocation_policies(
    policy: str,
    possessions: list[float],
    points_home: list[float],
    points_away: list[float],
) -> None:
    result = allocation_policy_stints(
        _source_stints(),
        _source_segments(),
        policy,
    )

    assert result["possessions"].tolist() == pytest.approx(possessions)
    assert result["points_home"].tolist() == pytest.approx(points_home)
    assert result["points_away"].tolist() == pytest.approx(points_away)
    assert result["allocation_policy"].eq(policy).all()


def test_lambda_and_chronological_stability_outputs() -> None:
    stints, matrix, player_ids, rankings = _diagnostic_inputs()
    game_splits = _game_splits(stints)

    lambda_coefficients, lambda_summary = lambda_sensitivity(
        stints,
        matrix,
        player_ids,
        rankings,
        (0.01, 0.03, 0.1),
        0.03,
    )
    chronological_coefficients, chronological_summary = chronological_stability(
        stints,
        matrix,
        player_ids,
        rankings,
        game_splits,
        0.03,
        1.0,
    )

    assert len(lambda_coefficients) == 3 * len(player_ids)
    selected = lambda_summary.loc[lambda_summary["is_selected"]].iloc[0]
    assert selected["coefficient_correlation"] == pytest.approx(1.0)
    assert selected["rank_spearman"] == pytest.approx(1.0)
    assert selected["mean_absolute_rank_change"] == pytest.approx(0.0)
    assert chronological_coefficients["window"].nunique() == 4
    assert len(chronological_summary) == len(player_ids)
    assert chronological_summary["coefficient_range"].ge(0).all()


def test_game_block_bootstrap_is_reproducible() -> None:
    stints, matrix, player_ids, rankings = _diagnostic_inputs()

    first_coefficients, first_summary = bootstrap_stability(
        stints,
        matrix,
        player_ids,
        rankings,
        0.03,
        samples=4,
        seed=17,
    )
    second_coefficients, second_summary = bootstrap_stability(
        stints,
        matrix,
        player_ids,
        rankings,
        0.03,
        samples=4,
        seed=17,
    )

    pd.testing.assert_frame_equal(first_coefficients, second_coefficients)
    pd.testing.assert_frame_equal(first_summary, second_summary)
    assert len(first_coefficients) == 4 * len(player_ids)
    assert first_coefficients["eligible_rank"].notna().all()
    assert first_summary["bootstrap_p05"].le(first_summary["bootstrap_p95"]).all()
    assert first_summary["positive_probability"].between(0, 1).all()
    assert first_summary["top_25_probability"].between(0, 1).all()


def test_context_raw_adjustment_and_influence_diagnostics() -> None:
    stints, matrix, player_ids, rankings = _diagnostic_inputs()

    concentration = context_concentration(stints, rankings)
    raw_adjusted = raw_adjusted_comparison(rankings)
    influential_stints, influential_games, delete_game = influence_diagnostics(
        stints,
        matrix,
        player_ids,
        rankings,
        0.03,
        influence_player_count=2,
        stints_per_player=2,
        delete_games_per_player=1,
    )

    assert len(concentration) == len(player_ids)
    assert concentration["most_common_teammate_share"].eq(1.0).all()
    assert concentration["effective_lineup_count"].eq(1.0).all()
    assert raw_adjusted["raw_on_court_rank"].is_unique
    assert np.allclose(
        raw_adjusted["rapm_adjustment"],
        raw_adjusted["rapm"] - raw_adjusted["raw_on_court_net_rating"],
    )
    assert influential_stints.groupby("player_id").size().le(2).all()
    assert influential_games.groupby("player_id").size().le(5).all()
    assert len(delete_game) == 2
    assert delete_game["absolute_coefficient_change"].ge(0).all()


def test_player_diagnostics_summary_has_one_row_per_player() -> None:
    stints, matrix, player_ids, rankings = _diagnostic_inputs()
    lambda_coefficients, _ = lambda_sensitivity(
        stints,
        matrix,
        player_ids,
        rankings,
        (0.01, 0.03, 0.1),
        0.03,
    )
    _, chronological_summary = chronological_stability(
        stints,
        matrix,
        player_ids,
        rankings,
        _game_splits(stints),
        0.03,
        1.0,
    )
    _, bootstrap_summary = bootstrap_stability(
        stints,
        matrix,
        player_ids,
        rankings,
        0.03,
        samples=2,
        seed=7,
    )
    concentration = context_concentration(stints, rankings)
    raw_adjusted = raw_adjusted_comparison(rankings)
    allocation_coefficients = pd.concat(
        [
            rankings.loc[:, ["player_id", "rapm", "rank"]]
            .rename(columns={"rapm": "coefficient"})
            .assign(
                allocation_policy=policy,
                coefficient_change=coefficient_change,
                rank_change=rank_change,
                eligible_rank_change=rank_change,
                exposure_eligible=True,
            )
            for policy, coefficient_change, rank_change in (
                ("equal_segments", 0.0, 0),
                ("terminal_lineup", 0.2, 2),
            )
        ],
        ignore_index=True,
    )
    delete_game = pd.DataFrame(
        {
            "player_id": [player_ids[0]],
            "game_id": ["0000000000"],
            "absolute_coefficient_change": [0.3],
        }
    )

    result = player_diagnostics_summary(
        rankings,
        lambda_coefficients,
        chronological_summary,
        bootstrap_summary,
        concentration,
        raw_adjusted,
        allocation_coefficients,
        delete_game,
    )

    assert len(result) == len(player_ids)
    assert result["player_id"].is_unique
    assert result["rank"].is_monotonic_increasing
    assert result["max_allocation_absolute_coefficient_change"].eq(0.2).all()
    assert result["max_allocation_absolute_eligible_rank_change"].eq(2).all()
    assert (
        result.loc[
            result["player_id"].eq(player_ids[0]),
            "max_delete_game_id",
        ].item()
        == "0000000000"
    )


def _source_stints() -> pd.DataFrame:
    shared = {
        "season": "2025-26",
        "season_type": "regular",
        "game_id": "0022500001",
        "game_date": date(2025, 10, 21),
        "game_time_utc": datetime(2025, 10, 21, 23, 30, tzinfo=UTC),
        "period": 1,
        "catalog_home_team_id": 100,
        "catalog_away_team_id": 200,
        "catalog_home_team_tricode": "HOM",
        "catalog_away_team_tricode": "AWY",
        "quality_status": "pass",
        "quality_issue_codes_json": "[]",
        "source_build_run_id": "run",
        "processing_code_version": "sha256:" + "a" * 64,
        "play_by_play_sha256": "b" * 64,
        "boxscore_sha256": "c" * 64,
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "stint_index": 0,
                "start_event_index": 0,
                "end_event_index": 4,
                "start_elapsed_game_seconds": 0.0,
                "end_elapsed_game_seconds": 20.0,
                "duration_seconds": 20.0,
                "points_home": 1,
                "points_away": 0,
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [6, 7, 8, 9, 10],
            },
            {
                **shared,
                "stint_index": 1,
                "start_event_index": 5,
                "end_event_index": 10,
                "start_elapsed_game_seconds": 20.0,
                "end_elapsed_game_seconds": 40.0,
                "duration_seconds": 20.0,
                "points_home": 1,
                "points_away": 2,
                "home_player_ids": [1, 2, 3, 4, 5],
                "away_player_ids": [6, 7, 8, 9, 11],
            },
        ]
    )


def _source_segments() -> pd.DataFrame:
    shared = {
        "game_id": "0022500001",
        "catalog_home_team_id": 100,
        "catalog_away_team_id": 200,
        "home_player_ids": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(
        [
            {
                **shared,
                "possession_id": "0022500001:0000",
                "start_event_index": 1,
                "end_event_index": 4,
                "offense_team_id": 100,
                "away_player_ids": [6, 7, 8, 9, 10],
                "points_home": 1,
                "points_away": 0,
            },
            {
                **shared,
                "possession_id": "0022500001:0000",
                "start_event_index": 5,
                "end_event_index": 6,
                "offense_team_id": 100,
                "away_player_ids": [6, 7, 8, 9, 11],
                "points_home": 1,
                "points_away": 0,
            },
            {
                **shared,
                "possession_id": "0022500001:0001",
                "start_event_index": 7,
                "end_event_index": 10,
                "offense_team_id": 200,
                "away_player_ids": [6, 7, 8, 9, 11],
                "points_home": 0,
                "points_away": 2,
            },
        ]
    )


def _diagnostic_inputs() -> tuple[
    pd.DataFrame,
    object,
    tuple[int, ...],
    pd.DataFrame,
]:
    stints = _modeling_stints(20)
    player_ids = tuple(range(1, 21))
    mapping = vocabulary_mapping(player_ids)
    matrix = signed_entity_matrix(
        stints,
        "home_player_ids",
        "away_player_ids",
        mapping,
        multiple=True,
    )
    target = stints["target_home_net_rating"].to_numpy(dtype=float)
    weights = stints["possessions"].to_numpy(dtype=float)
    model = RidgeLineupModel(0.03).fit(matrix, target, weights)
    order = np.argsort(-model.coef_, kind="stable")
    ranks = np.empty(len(player_ids), dtype=int)
    ranks[order] = np.arange(1, len(player_ids) + 1)
    exposure = np.asarray(abs(matrix).T @ weights).reshape(-1)
    rankings = pd.DataFrame(
        {
            "player_id": player_ids,
            "player_name": [f"Player {player_id}" for player_id in player_ids],
            "primary_team_tricode": [
                ("AAA", "BBB", "CCC", "DDD")[(player_id - 1) // 5] for player_id in player_ids
            ],
            "rapm": model.coef_,
            "rank": ranks,
            "eligible_rank": pd.array(ranks, dtype="Int64"),
            "possessions": exposure,
            "exposure_eligible": True,
            "raw_on_court_net_rating": model.coef_ * 1.5 + np.arange(20) / 100,
        }
    ).sort_values("rank", kind="stable")
    return stints, matrix, player_ids, rankings


def _game_splits(stints: pd.DataFrame) -> pd.DataFrame:
    game_ids = stints["game_id"].astype(str).tolist()
    rows = []
    for split, train_count in (("cv_0", 10), ("cv_1", 13)):
        rows.extend(
            {"split": split, "role": "train", "game_id": game_id}
            for game_id in game_ids[:train_count]
        )
    rows.extend(
        {"split": "final", "role": "train", "game_id": game_id} for game_id in game_ids[:17]
    )
    rows.extend({"split": "final", "role": "test", "game_id": game_id} for game_id in game_ids[17:])
    return pd.DataFrame(rows)


def _modeling_stints(game_count: int) -> pd.DataFrame:
    start = datetime(2025, 10, 1, tzinfo=UTC)
    team_players = {
        100: [1, 2, 3, 4, 5],
        200: [6, 7, 8, 9, 10],
        300: [11, 12, 13, 14, 15],
        400: [16, 17, 18, 19, 20],
    }
    team_codes = {100: "AAA", 200: "BBB", 300: "CCC", 400: "DDD"}
    rows = []
    for game_index in range(game_count):
        home_team = (100, 200, 300, 400)[game_index % 4]
        away_team = (200, 300, 400, 100)[game_index % 4]
        possessions = 10.0
        margin = int(round(home_team / 100.0 - away_team / 100.0))
        rows.append(
            {
                "game_id": f"{game_index:010d}",
                "game_date": (start + timedelta(days=game_index)).date(),
                "game_time_utc": start + timedelta(days=game_index),
                "stint_index": 0,
                "home_team_id": home_team,
                "away_team_id": away_team,
                "home_team_tricode": team_codes[home_team],
                "away_team_tricode": team_codes[away_team],
                "home_player_ids": team_players[home_team],
                "away_player_ids": team_players[away_team],
                "duration_seconds": 300.0,
                "home_margin": margin,
                "possessions": possessions,
                "target_home_net_rating": 100.0 * margin / possessions,
            }
        )
    return pd.DataFrame(rows)
