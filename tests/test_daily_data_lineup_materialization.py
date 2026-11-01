from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from nba_lineup_model.modeling import current_season_lineup_materialization as subject


def test_daily_data_lineups_use_one_verified_chain(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "daily-data"
    bundle = (
        root
        / "season=2026-27"
        / "game_date_pacific=2026-10-31"
        / f"data_sha256={'a' * 64}"
    )
    files = bundle / "files"
    files.mkdir(parents=True)
    stints_path = files / "observed_stints.parquet"
    stints = pd.DataFrame(
        {
            "season": ["2026-27"],
            "game_id": ["game-1"],
            "home_team_id": [1],
            "away_team_id": [2],
            "home_team_tricode": ["GSW"],
            "away_team_tricode": ["LAL"],
            "home_player_ids": [[1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10]],
            "possessions": [10.0],
        }
    )
    stints.to_parquet(stints_path, index=False)
    manifest = {
        "bundle_sha256": "a" * 64,
        "identity": {
            "season": "2026-27",
            "game_date_pacific": "2026-10-31",
            "parent_data_sha256": None,
            "components": {"observed_stints": {"path": "files/observed_stints.parquet"}},
            "game_sources": {
                "game-1": {"nba_play_by_play": {"sha256": "b" * 64}}
            },
        },
    }
    (root / "latest.json").write_text(
        json.dumps(
            {
                "bundle_path": bundle.relative_to(root).as_posix(),
                "bundle_sha256": "a" * 64,
            }
        )
    )
    monkeypatch.setattr(subject, "verify_daily_data_artifact", lambda path: manifest)
    lineups = pd.DataFrame(
        {
            "team_id": [1, 2],
            "team": ["GSW", "LAL"],
            "player_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "possessions": [10.0, 10.0],
            "games": [1, 1],
            "gestalt_score": [1.0, -1.0],
            "actual_net_rating": [2.0, -2.0],
        }
    )
    monkeypatch.setattr(
        subject, "build_observed_lineup_rankings", lambda **kwargs: lineups
    )
    captured: dict[str, object] = {}

    def fake_publish(**kwargs):
        captured.update(kwargs)
        return tmp_path / "lineup-bundle"

    monkeypatch.setattr(subject, "publish_current_season_lineup_bundle", fake_publish)
    rankings = pd.DataFrame({"player_id": range(1, 11)})
    evaluator = SimpleNamespace(
        current_season_metadata={
            "season": "2026-27",
            "evidence_through_date": "2026-10-31",
            "source_game_ids": ["game-1"],
        },
        context_model=object(),
        _active_current_season_rankings=lambda: rankings,
        _preseason_lab_state=lambda: SimpleNamespace(
            profiles=pd.DataFrame(), players=pd.DataFrame(), coefficients=pd.DataFrame()
        ),
    )

    result = subject.materialize_daily_data_lineups(
        evaluator=evaluator,
        daily_data_root=root,
        output_root=tmp_path / "lineups",
    )

    assert result == tmp_path / "lineup-bundle"
    assert captured["source_game_checksums"] == {"game-1": "b" * 64}
    published_stints = captured["stints"]
    assert list(published_stints["game_id"]) == ["game-1"]
    assert list(published_stints.iloc[0]["home_player_ids"]) == [1, 2, 3, 4, 5]
