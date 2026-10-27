from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.web_api.current_season_lineups import (
    build_player_team_exposure,
    load_current_season_lineup_catalog,
    publish_current_season_lineup_bundle,
)


def _stints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2026-27"],
            "game_id": ["game-1"],
            "home_team_id": [1],
            "away_team_id": [2],
            "home_team_tricode": ["GSW"],
            "away_team_tricode": ["LAL"],
            "home_player_ids": [[1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10]],
            "possessions": [12.0],
        }
    )


def _lineups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 2],
            "team": ["GSW", "LAL"],
            "player_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "possessions": [12.0, 12.0],
            "games": [1, 1],
            "gestalt_score": [2.0, -2.0],
            "actual_net_rating": [5.0, -5.0],
        }
    )


def _metadata() -> dict[str, object]:
    return {
        "season": "2026-27",
        "evidence_through_date": "2026-10-26",
        "as_of_cutoff_time": "2026-10-27T07:00:00Z",
        "bundle_sha256": "a" * 64,
    }


def test_current_lineup_bundle_is_content_addressed_and_bound_to_serving_state(
    tmp_path: Path,
) -> None:
    stints = _stints()
    exposure = build_player_team_exposure(stints)
    root = tmp_path / "lineups"

    bundle = publish_current_season_lineup_bundle(
        lineups=_lineups(),
        stints=stints,
        player_team_exposure=exposure,
        serving_metadata=_metadata(),
        source_game_checksums={"game-1": "b" * 64},
        root=root,
    )
    repeated = publish_current_season_lineup_bundle(
        lineups=_lineups(),
        stints=stints,
        player_team_exposure=exposure,
        serving_metadata=_metadata(),
        source_game_checksums={"game-1": "b" * 64},
        root=root,
    )

    assert repeated == bundle
    catalog = load_current_season_lineup_catalog(
        serving_metadata=_metadata(), root=root
    )
    assert catalog is not None
    assert catalog.metadata["serving_bundle_sha256"] == "a" * 64
    assert catalog.metadata["source_game_ids"] == ["game-1"]
    assert len(catalog.lineups) == 2
    assert len(catalog.stints) == 1
    assert set(catalog.player_team_exposure["player_id"]) == set(range(1, 11))


def test_current_lineup_bundle_rejects_a_different_promoted_state(tmp_path: Path) -> None:
    root = tmp_path / "lineups"
    publish_current_season_lineup_bundle(
        lineups=_lineups(),
        stints=_stints(),
        player_team_exposure=build_player_team_exposure(_stints()),
        serving_metadata=_metadata(),
        source_game_checksums={"game-1": "b" * 64},
        root=root,
    )
    changed = {**_metadata(), "bundle_sha256": "c" * 64}

    with pytest.raises(ValueError, match="do not match"):
        load_current_season_lineup_catalog(serving_metadata=changed, root=root)
