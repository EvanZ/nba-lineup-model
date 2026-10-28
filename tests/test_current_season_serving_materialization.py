from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nba_lineup_model.modeling import current_season_serving_materialization as subject


def _write_publication_inputs(tmp_path: Path, *, include_trajectory: bool = True):
    rating_root = tmp_path / "ratings"
    minutes_root = tmp_path / "minutes"
    release_root = tmp_path / "release"
    output_root = tmp_path / "serving"
    through_date = "2026-10-26"
    season = "2026-27"
    rating_run_id = "rating-run"
    minutes_run_id = "minutes-run"

    rating_payload = b"rating-state"
    trajectory_payload = b"player-game-trajectory"
    minutes_payload = b"minutes-state"
    rating_digest = hashlib.sha256(rating_payload).hexdigest()
    trajectory_digest = hashlib.sha256(trajectory_payload).hexdigest()
    minutes_digest = hashlib.sha256(minutes_payload).hexdigest()

    rating_path = (
        rating_root
        / "player_states"
        / f"season={season}"
        / "cutoff=20261027T070000Z"
        / f"state_sha256={rating_digest}"
        / "player_state.parquet"
    )
    trajectory_path = (
        rating_root
        / "runs"
        / f"through_date={through_date}"
        / rating_run_id
        / "player_game_trajectory.parquet"
    )
    minutes_path = (
        minutes_root
        / "runs"
        / f"through_date={through_date}"
        / minutes_run_id
        / "player_state.parquet"
    )
    for path, payload in (
        (rating_path, rating_payload),
        (trajectory_path, trajectory_payload),
        (minutes_path, minutes_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    rating_artifacts = {
        "player_state": f"s3://example/player_state-{rating_digest}.parquet"
    }
    if include_trajectory:
        rating_artifacts["player_game_trajectory"] = (
            f"s3://example/player_game_trajectory-{trajectory_digest}.parquet"
        )
    rating_root.mkdir(parents=True, exist_ok=True)
    (rating_root / "latest.json").write_text(
        json.dumps(
            {
                "season": season,
                "through_date": through_date,
                "dagster_run_id": rating_run_id,
                "coverage": {"excluded": 0},
                "artifacts": rating_artifacts,
            }
        )
    )
    (minutes_root / "latest.json").write_text(
        json.dumps(
            {
                "season": season,
                "through_date": through_date,
                "dagster_run_id": minutes_run_id,
                "coverage": {"excluded": 0},
                "artifacts": {
                    "player_state": f"s3://example/player_state-{minutes_digest}.parquet"
                },
            }
        )
    )
    release_root.mkdir(parents=True)
    (release_root / "release.json").write_text(json.dumps({"run_id": "preseason-run"}))
    return rating_root, minutes_root, release_root, output_root, trajectory_path


def test_materialization_preserves_published_player_game_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rating_root, minutes_root, release_root, output_root, trajectory_path = (
        _write_publication_inputs(tmp_path)
    )
    captured: dict[str, object] = {}

    def fake_publish(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=output_root / "bundle")

    monkeypatch.setattr(subject, "publish_serving_bundle", fake_publish)
    monkeypatch.setattr(subject, "promote_serving_bundle", lambda *args, **kwargs: None)

    path = subject.materialize_current_season_serving(
        rating_root=rating_root,
        minutes_root=minutes_root,
        release_root=release_root,
        output_root=output_root,
    )

    assert path == output_root / "bundle"
    assert captured["rating_trajectory_path"] == trajectory_path
    assert str(captured["rating_trajectory_source_uri"]).startswith(
        "s3://example/player_game_trajectory-"
    )


def test_materialization_rejects_missing_player_game_trajectory_artifact(
    tmp_path: Path,
) -> None:
    rating_root, minutes_root, release_root, output_root, _ = (
        _write_publication_inputs(tmp_path, include_trajectory=False)
    )

    with pytest.raises(
        ValueError,
        match="Publication is missing artifact URI: player_game_trajectory",
    ):
        subject.materialize_current_season_serving(
            rating_root=rating_root,
            minutes_root=minutes_root,
            release_root=release_root,
            output_root=output_root,
        )
