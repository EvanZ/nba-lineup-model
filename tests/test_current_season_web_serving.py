from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.season.current_serving import (
    ServingPromotionConfig,
    promote_serving_bundle,
    publish_serving_bundle,
)
from nba_lineup_model.web_api.current_season_serving import (
    load_current_season_serving_catalog,
)

SEASON = "2026-27"
CONFIG = ServingPromotionConfig(preseason_run_id="preseason-release")


def _rating_state(*, cutoff: datetime, games: list[str], deltas: list[float]) -> pd.DataFrame:
    baselines = [3.0, -1.0]
    return pd.DataFrame(
        {
            "schema_version": [1, 1],
            "season": [SEASON, SEASON],
            "core_model_version": ["nail-v1.2.1.3"] * 2,
            "state_updater_version": ["in-season-nail-v1"] * 2,
            "player_id": [1, 2],
            "team_at_cutoff": ["GSW", "LAL"],
            "cutoff_time": [cutoff] * 2,
            "cutoff_game_id": [games[-1]] * 2,
            "frozen_preseason_nail": baselines,
            "current_season_residual": deltas,
            "updated_nail": [
                baseline + delta for baseline, delta in zip(baselines, deltas, strict=True)
            ],
            "raw_on_court_possessions": [100.0 * len(games), 0.0],
            "effective_information": [50.0 * len(games), 0.0],
            "prior_equivalent_possessions": [6000.0] * 2,
            "prior_weight": [0.98, 1.0],
            "evidence_weight": [0.02, 0.0],
            "posterior_standard_error": [0.11, 0.12],
            "included_game_ids_json": [json.dumps(games)] * 2,
            "contribution_lineage_json": [
                json.dumps([f"{index + 1:064x}" for index in range(len(games))])
            ]
            * 2,
        }
    )


def _minutes_state(*, cutoff: datetime, games: list[str], minutes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": [2, 2],
            "season": [SEASON, SEASON],
            "updater_version": ["in-season-minutes-v1"] * 2,
            "player_id": [1, 2],
            "player_name": ["Stephen Curry", "Cold Start"],
            "team": ["GSW", "LAL"],
            "preseason_participation_prior": [0.9, 0.7],
            "current_participation_estimate": [0.95, 0.7],
            "participation_observations": [len(games), 0],
            "preseason_conditional_minutes_prior": [34.0, 18.0],
            "preseason_normalized_minutes": [30.6, 12.6],
            "current_conditional_minutes_estimate": minutes,
            "conditional_observations": [len(games), 0],
            "conditional_minutes_weighted_sum": [sum(minutes[:1]) * len(games), 0.0],
            "effective_conditional_weight": [float(len(games)), 0.0],
            "conditional_window_observations": [len(games), 0],
            "conditional_history_json": ["[]", "[]"],
            "included_game_ids_json": [json.dumps(games)] * 2,
            "contribution_lineage_json": [
                json.dumps([f"{index + 101:064x}" for index in range(len(games))])
            ]
            * 2,
            "cutoff_time": [cutoff.isoformat()] * 2,
            "prior_participation_weight": [4.0] * 2,
            "prior_conditional_weight": [0.2] * 2,
        }
    )


def _preseason_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [SEASON, SEASON, SEASON],
            "player_id": [1, 2, 3],
            "player_name": ["Stephen Curry", "Cold Start", "No Longer Rostered"],
            "team": ["GSW", "LAL", "BOS"],
            "position": ["G", "F", "C"],
            "rapm": [3.0, -1.0, 0.0],
            "possessions": [0.0, 0.0, 0.0],
            "games": [0, 0, 0],
            "offense_rating": [1.0, None, None],
            "defense_rating": [2.0, None, None],
        }
    )


def _publish_day(
    root: Path,
    sources: Path,
    *,
    through_date: date,
    games: list[str],
    deltas: list[float],
    minutes: list[float],
    include_trajectory: bool = False,
) -> Path:
    cutoff = datetime.combine(through_date, datetime.min.time(), tzinfo=UTC).replace(hour=8)
    sources.mkdir(parents=True, exist_ok=True)
    rating = sources / f"rating-{through_date}.parquet"
    minute = sources / f"minutes-{through_date}.parquet"
    _rating_state(cutoff=cutoff, games=games, deltas=deltas).to_parquet(rating, index=False)
    _minutes_state(cutoff=cutoff, games=games, minutes=minutes).to_parquet(minute, index=False)
    trajectory = sources / f"trajectory-{through_date}.parquet"
    if include_trajectory:
        pd.DataFrame(
            {
                "season": [SEASON] * len(games),
                "updater_version": ["in-season-nail-v1"] * len(games),
                "player_id": [1] * len(games),
                "game_number": list(range(1, len(games) + 1)),
                "game_id": games,
                "game_date": [through_date.isoformat()] * len(games),
                "game_time_utc": [cutoff.isoformat()] * len(games),
                "preseason_rating": [3.0] * len(games),
                "in_season_delta": [
                    deltas[0] * game_number / len(games)
                    for game_number in range(1, len(games) + 1)
                ],
                "updated_rating": [
                    3.0 + deltas[0] * game_number / len(games)
                    for game_number in range(1, len(games) + 1)
                ],
                "game_on_court_possessions": [100.0] * len(games),
                "cumulative_on_court_possessions": [
                    100.0 * game_number for game_number in range(1, len(games) + 1)
                ],
                "effective_information": [
                    50.0 * game_number for game_number in range(1, len(games) + 1)
                ],
                "evidence_weight": [
                    (50.0 * game_number) / (6000.0 + 50.0 * game_number)
                    for game_number in range(1, len(games) + 1)
                ],
            }
        ).to_parquet(trajectory, index=False)
    bundle = publish_serving_bundle(
        rating,
        minute,
        root=root,
        season=SEASON,
        evidence_through_date=through_date,
        as_of_cutoff_time=cutoff,
        config=CONFIG,
        rating_source_uri=f"s3://example/{rating.name}",
        minutes_source_uri=f"s3://example/{minute.name}",
        rating_trajectory_path=trajectory if include_trajectory else None,
        rating_trajectory_source_uri=(
            f"s3://example/{trajectory.name}" if include_trajectory else None
        ),
    )
    promote_serving_bundle(bundle.path, root=root, promoted_at=cutoff)
    return bundle.path


def test_current_catalog_overlays_exact_promoted_values_and_builds_trajectory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "serving"
    _publish_day(
        root,
        tmp_path / "sources",
        through_date=date(2026, 10, 25),
        games=["1"],
        deltas=[0.1, 0.0],
        minutes=[35.0, 18.0],
    )
    latest = _publish_day(
        root,
        tmp_path / "sources",
        through_date=date(2026, 10, 26),
        games=["1", "2"],
        deltas=[0.2, 0.0],
        minutes=[36.0, 18.0],
    )

    catalog = load_current_season_serving_catalog(_preseason_rankings(), root=root)

    assert catalog is not None
    assert catalog.bundle_path == latest.resolve()
    assert catalog.metadata["kind"] == "in_season_updated"
    assert catalog.metadata["evidence_through_date"] == "2026-10-26"
    assert catalog.metadata["game_count"] == 2
    assert set(catalog.rankings["player_id"]) == {1, 2}
    curry = catalog.rankings.set_index("player_id").loc[1]
    assert curry["rapm"] == pytest.approx(3.2)
    assert curry["preseason_nail"] == pytest.approx(3.0)
    assert curry["in_season_delta"] == pytest.approx(0.2)
    assert curry["current_season_possessions"] == pytest.approx(200.0)
    assert curry["current_expected_minutes_if_available"] == pytest.approx(36.0)
    assert pd.isna(curry["offense_rating"])
    assert pd.isna(curry["defense_rating"])
    cold_start = catalog.rankings.set_index("player_id").loc[2]
    assert cold_start["rapm"] == pytest.approx(-1.0)
    assert cold_start["preseason_nail"] == pytest.approx(-1.0)
    assert cold_start["current_season_possessions"] == pytest.approx(0.0)
    assert cold_start["rating_evidence_weight"] == pytest.approx(0.0)
    assert cold_start["current_expected_minutes_if_available"] == pytest.approx(18.0)
    trajectory = catalog.trajectories[1]
    assert [point["source"] for point in trajectory] == [
        "preseason",
        "in_season",
        "in_season",
    ]
    assert [point["rating"] for point in trajectory] == pytest.approx([3.0, 3.1, 3.2])
    assert [point["evidence_through_date"] for point in trajectory[1:]] == [
        "2026-10-25",
        "2026-10-26",
    ]
    assert trajectory[-1]["is_current"] is True
    assert [point["rating"] for point in catalog.trajectories[2]] == pytest.approx(
        [-1.0, -1.0, -1.0]
    )


def test_current_catalog_rejects_a_changed_preseason_baseline(tmp_path: Path) -> None:
    bundle = _publish_day(
        tmp_path / "serving",
        tmp_path / "sources",
        through_date=date(2026, 10, 26),
        games=["1"],
        deltas=[0.1, 0.0],
        minutes=[35.0, 18.0],
    )
    rankings = _preseason_rankings()
    rankings.loc[rankings["player_id"].eq(1), "rapm"] = 99.0

    with pytest.raises(ValueError, match="do not match"):
        load_current_season_serving_catalog(rankings, bundle_path=bundle)


def test_current_catalog_uses_postgame_rating_trajectory_when_published(
    tmp_path: Path,
) -> None:
    bundle = _publish_day(
        tmp_path / "serving",
        tmp_path / "sources",
        through_date=date(2026, 10, 26),
        games=["1", "2"],
        deltas=[0.2, 0.0],
        minutes=[36.0, 18.0],
        include_trajectory=True,
    )

    catalog = load_current_season_serving_catalog(
        _preseason_rankings(),
        root=tmp_path / "serving",
        bundle_path=bundle,
    )

    assert catalog is not None
    points = catalog.trajectories[1]
    assert [point["point_kind"] for point in points] == [
        "preseason",
        "postgame",
        "postgame",
    ]
    assert [point.get("game_number") for point in points] == [None, 1, 2]
    assert [point["rating"] for point in points] == pytest.approx([3.0, 3.1, 3.2])
    assert points[-1]["is_current"] is True


def test_missing_promotion_keeps_preseason_state(tmp_path: Path) -> None:
    assert (
        load_current_season_serving_catalog(
            _preseason_rankings(),
            root=tmp_path / "not-promoted",
        )
        is None
    )
