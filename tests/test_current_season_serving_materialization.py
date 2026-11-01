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


def test_daily_model_materialization_binds_all_serving_inputs_to_one_tip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "daily-model"
    bundle = (
        model_root
        / "season=2026-27"
        / "game_date_pacific=2026-10-31"
        / f"model_sha256={'a' * 64}"
    )
    bundle.mkdir(parents=True)
    files = bundle / "files"
    files.mkdir()
    for filename in (
        "player_ratings.parquet",
        "minutes_state.parquet",
        "rating_trajectory.parquet",
    ):
        (files / filename).write_bytes(filename.encode())
    pointer = {
        "bundle_path": bundle.relative_to(model_root).as_posix(),
        "bundle_sha256": "a" * 64,
    }
    (model_root / "latest.json").write_text(json.dumps(pointer))
    release_root = tmp_path / "release"
    release_root.mkdir()
    (release_root / "release.json").write_text(json.dumps({"run_id": "preseason-run"}))
    manifest = {
        "bundle_sha256": "a" * 64,
        "identity": {
            "season": "2026-27",
            "game_date_pacific": "2026-10-31",
            "cumulative_game_count": 88,
            "model_provenance": {"prior_sha256": "b" * 64},
            "components": {
                "player_ratings": {"path": "files/player_ratings.parquet"},
                "minutes_state": {"path": "files/minutes_state.parquet"},
                "rating_trajectory": {"path": "files/rating_trajectory.parquet"},
            },
        },
    }
    monkeypatch.setattr(subject, "verify_daily_model_artifact", lambda path: manifest)
    captured: dict[str, object] = {}

    def fake_publish(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        output = tmp_path / "serving-bundle"
        output.mkdir()
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "identity": {
                        "sources": {
                            "rating_state": {"uri": kwargs["rating_source_uri"]},
                            "minutes_state": {"uri": kwargs["minutes_source_uri"]},
                            "rating_trajectory": {
                                "uri": kwargs["rating_trajectory_source_uri"]
                            },
                        }
                    }
                }
            )
        )
        return SimpleNamespace(path=output)

    monkeypatch.setattr(subject, "publish_serving_bundle", fake_publish)
    monkeypatch.setattr(
        subject,
        "promote_serving_bundle",
        lambda *args, **kwargs: ({"evidence_through_date": "2026-10-31"}, True),
    )

    result = subject.materialize_daily_model_serving(
        daily_model_root=model_root,
        release_root=release_root,
        output_root=tmp_path / "serving",
    )

    assert result == tmp_path / "serving-bundle"
    assert captured["rating_source_uri"] == (
        f"artifact://daily-model/{'a' * 64}/player_ratings"
    )
    assert captured["minutes_source_uri"] == (
        f"artifact://daily-model/{'a' * 64}/minutes_state"
    )
    assert captured["source_coverage"] == {
        "ratings": {"games": 88, "excluded": 0},
        "minutes": {"games": 88, "excluded": 0},
    }
