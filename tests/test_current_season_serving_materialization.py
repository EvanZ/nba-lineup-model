from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from nba_lineup_model.modeling import current_season_serving_materialization as subject








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
    home_court_payload = {
        "home_court_estimate": 1.75,
        "home_court_standard_error": 0.4,
        "home_court_evidence_weight": 0.2,
        "home_court_candidate_id": "causal_hca_replacement_candidate_v2",
    }
    home_court_bytes = json.dumps(home_court_payload).encode()
    (files / "home_court_state.json").write_bytes(home_court_bytes)
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
                "home_court_state": {
                    "path": "files/home_court_state.json",
                    "sha256": hashlib.sha256(home_court_bytes).hexdigest(),
                },
            },
        },
    }
    monkeypatch.setattr(subject, "verify_daily_model_artifact", lambda path: manifest)
    monkeypatch.setattr(
        subject,
        "_adapt_daily_rating_trajectory",
        lambda **kwargs: kwargs["output_path"].write_bytes(b"adapted-trajectory"),
    )
    monkeypatch.setattr(
        subject,
        "_adapt_daily_minutes_state",
        lambda **kwargs: kwargs["output_path"].write_bytes(b"adapted-minutes"),
    )
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
        f"artifact://daily-model/{'a' * 64}/minutes_state#serving-v1"
    )
    assert captured["source_coverage"] == {
        "ratings": {"games": 88, "excluded": 0},
        "minutes": {"games": 88, "excluded": 0},
    }
    assert captured["schedule_controls"] == {
        "home_court": 1.75,
        "home_court_standard_error": 0.4,
        "home_court_evidence_weight": 0.2,
        "candidate_id": "causal_hca_replacement_candidate_v2",
        "state_as_of": "2026-10-31",
        "source_uri": f"artifact://daily-model/{'a' * 64}/home_court_state",
        "source_sha256": hashlib.sha256(home_court_bytes).hexdigest(),
    }


def test_daily_trajectory_adapter_reconstructs_game_and_cumulative_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    ratings_path = tmp_path / "ratings.parquet"
    output = tmp_path / "adapted.parquet"
    stints_path = tmp_path / "stints.parquet"
    pd.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "game_date_pacific": ["2026-10-30", "2026-10-31"],
            "game_id": ["game-1", "game-2"],
            "game_time_utc": ["2026-10-31T02:00:00Z", "2026-11-01T02:00:00Z"],
            "player_id": [1, 1],
            "team": ["GSW", "GSW"],
            "opponent": ["LAL", "PHX"],
            "played": [True, False],
            "injury_status": [None, "OUT"],
            "injury_reason": [None, "Ankle"],
            "rating_after": [2.2, 2.2],
        }
    ).to_parquet(source, index=False)
    pd.DataFrame(
        {
            "player_id": [1],
            "frozen_preseason_nail": [2.0],
            "raw_on_court_possessions": [10.0],
            "prior_equivalent_possessions": [6000.0],
            "state_updater_version": ["updater-v1"],
        }
    ).to_parquet(ratings_path, index=False)
    pd.DataFrame(
        {
            "game_id": ["game-1"],
            "home_player_ids": [[1, 2, 3, 4, 5]],
            "away_player_ids": [[6, 7, 8, 9, 10]],
            "possessions": [10.0],
        }
    ).to_parquet(stints_path, index=False)
    bundle = tmp_path / "data-bundle"
    bundle.mkdir()
    manifest = {
        "identity": {
            "components": {"observed_stints": {"path": "../stints.parquet"}}
        }
    }
    monkeypatch.setattr(
        subject,
        "_daily_data_chain",
        lambda root: [(date(2026, 10, 31), bundle, manifest)],
    )
    monkeypatch.setattr(
        subject,
        "_daily_data_component_path",
        lambda bundle, components, name: stints_path,
    )

    subject._adapt_daily_rating_trajectory(
        source_path=source,
        rating_state_path=ratings_path,
        daily_data_root=tmp_path,
        evidence_date=date(2026, 10, 31),
        output_path=output,
    )

    adapted = pd.read_parquet(output)
    assert adapted["game_number"].tolist() == [1, 2]
    assert adapted["game_on_court_possessions"].tolist() == [10.0, 0.0]
    assert adapted["cumulative_on_court_possessions"].tolist() == [10.0, 10.0]
    assert adapted["updated_rating"].tolist() == [2.2, 2.2]
    assert adapted["in_season_delta"].tolist() == pytest.approx([0.2, 0.2])


def test_daily_minutes_adapter_adds_locked_serving_configuration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "minutes.parquet"
    output = tmp_path / "adapted-minutes.parquet"
    pd.DataFrame({"player_id": [1, 2]}).to_parquet(source, index=False)
    cutoff = pd.Timestamp("2026-11-01T07:00:00Z").to_pydatetime()

    subject._adapt_daily_minutes_state(
        source_path=source,
        output_path=output,
        cutoff_time=cutoff,
        participation_prior_games=4.0,
        conditional_prior_games=0.2,
    )

    adapted = pd.read_parquet(output)
    assert adapted["prior_participation_weight"].tolist() == [4.0, 4.0]
    assert adapted["prior_conditional_weight"].tolist() == [0.2, 0.2]
    assert adapted["cutoff_time"].tolist() == [cutoff, cutoff]
