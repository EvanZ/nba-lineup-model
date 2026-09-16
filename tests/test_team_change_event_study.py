import numpy as np
import pandas as pd

from nba_lineup_model.modeling.team_change_event_study import (
    build_team_change_event_mart,
    summarize_event_mart,
    validation_examples,
)


def _season(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def _panel_row(player_id: int, year: int, team: str, *, age: float, possessions: float = 1_200.0, team_count: int = 1) -> dict[str, object]:
    return {
        "season": _season(year),
        "season_start_year": year,
        "player_id": player_id,
        "primary_team_tricode": team,
        "team_count": team_count,
        "rapm_possessions": possessions,
        "age": age,
    }


def _rating_row(player_id: int, year: int, value: float, prior: float) -> dict[str, object]:
    return {
        "season": _season(year),
        "player_id": player_id,
        "player_name": f"Player {player_id}",
        "rapm": value,
        "prior_rapm": prior,
        "rank_all_players": 10,
    }


def test_event_mart_scores_clean_move_and_separates_strict_from_broad() -> None:
    panel_rows: list[dict[str, object]] = []
    rating_rows: list[dict[str, object]] = []
    player_one = [
        (2017, "AAA", 1.0, 0.0, 1_100.0),
        (2018, "AAA", 2.0, 0.0, 1_200.0),
        (2019, "AAA", 1.0, 0.0, 1_300.0),
        (2020, "BBB", 2.0, 1.0, 1_000.0),
        (2021, "BBB", 3.0, 1.0, 2_000.0),
        (2022, "BBB", 0.0, 1.0, 1_000.0),
    ]
    player_two = [
        (2019, "CCC", 0.0, 0.0, 1_100.0),
        (2020, "DDD", 1.0, 0.0, 1_000.0),
        (2021, "EEE", 1.0, 0.0, 1_000.0),
        (2022, "EEE", 1.0, 0.0, 1_000.0),
    ]
    for player_id, rows in ((1, player_one), (2, player_two)):
        for year, team, value, prior, possessions in rows:
            panel_rows.append(_panel_row(player_id, year, team, age=25.0, possessions=possessions))
            rating_rows.append(_rating_row(player_id, year, value, prior))

    mart = build_team_change_event_mart(
        pd.DataFrame(rating_rows), pd.DataFrame(panel_rows), minimum_possessions=1_000.0
    )

    strict = mart.loc[(mart.player_id == 1) & mart.event_season.eq("2020-21")].iloc[0]
    assert strict.clean_offseason_move
    assert strict.broad_cohort_eligible
    assert strict.strict_cohort_eligible
    assert strict.old_team == "AAA"
    assert strict.new_team == "BBB"
    assert strict.immediate_completed_nail_change == 1.0
    assert strict.pre_move_completed_nail_rating == np.mean([1.0, 2.0, 1.0])
    assert strict.post_move_completed_nail_rating == np.mean([2.0, 3.0, 0.0])
    assert strict.sustained_completed_nail_change == np.mean([2.0, 3.0, 0.0]) - np.mean([1.0, 2.0, 1.0])

    broad_only = mart.loc[(mart.player_id == 2) & mart.event_season.eq("2020-21")].iloc[0]
    assert broad_only.broad_cohort_eligible
    assert not broad_only.strict_cohort_eligible
    assert broad_only.later_team_change

    higher_threshold = build_team_change_event_mart(
        pd.DataFrame(rating_rows), pd.DataFrame(panel_rows), minimum_possessions=1_001.0
    )
    assert not higher_threshold.loc[
        (higher_threshold.player_id == 1) & higher_threshold.event_season.eq("2020-21"),
        "broad_cohort_eligible",
    ].iloc[0]


def test_event_mart_excludes_underage_and_multi_team_changes() -> None:
    panel = pd.DataFrame([
        _panel_row(1, 2020, "AAA", age=24.0),
        _panel_row(1, 2021, "BBB", age=24.0),
        _panel_row(1, 2022, "BBB", age=25.0),
        _panel_row(1, 2023, "BBB", age=26.0),
        _panel_row(2, 2020, "CCC", age=29.0),
        _panel_row(2, 2021, "DDD", age=30.0, team_count=2),
        _panel_row(2, 2022, "DDD", age=31.0),
        _panel_row(2, 2023, "DDD", age=32.0),
    ])
    ratings = pd.DataFrame([
        _rating_row(player_id, year, 1.0, 0.0)
        for player_id in (1, 2)
        for year in (2020, 2021, 2022, 2023)
    ])

    mart = build_team_change_event_mart(ratings, panel)
    assert not mart.loc[mart.player_id.eq(1), "broad_cohort_eligible"].any()
    assert not mart.loc[mart.player_id.eq(2), "broad_cohort_eligible"].any()
    summary = summarize_event_mart(mart)
    assert int(summary.iloc[-1].events) == 0


def test_validation_examples_reports_expected_match() -> None:
    mart = pd.DataFrame({
        "player_name": ["Kevin Durant"],
        "event_season": ["2016-17"],
        "old_team": ["OKC"],
        "new_team": ["GSW"],
        "strict_cohort_eligible": [True],
    })
    examples = validation_examples(mart)
    durant = examples.loc[examples.player_name.eq("Kevin Durant")].iloc[0]
    assert durant.found
    assert durant.strict_cohort_eligible
