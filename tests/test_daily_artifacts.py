from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from nba_lineup_model.season.daily_data_artifact import (
    DEFAULT_REQUIRED_COMPONENTS as DATA_COMPONENTS,
)
from nba_lineup_model.season.daily_data_artifact import (
    promote_daily_data_artifact,
    publish_daily_data_artifact,
    verify_daily_data_artifact,
)
from nba_lineup_model.season.daily_data_replay import write_daily_data_chain_report
from nba_lineup_model.season.daily_model_artifact import (
    DEFAULT_REQUIRED_COMPONENTS as MODEL_COMPONENTS,
)
from nba_lineup_model.season.daily_model_artifact import (
    publish_daily_model_artifact,
    verify_daily_model_artifact,
)

SEASON = "2026-27"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _components(tmp_path: Path, names: set[str] | frozenset[str], prefix: str):
    tmp_path.mkdir(parents=True, exist_ok=True)
    output: dict[str, Path] = {}
    for name in names:
        path = tmp_path / f"{prefix}-{name}.json"
        path.write_text(f'{{"component":"{prefix}/{name}"}}\n')
        output[name] = path
    return output


def _sources(game_ids: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        game_id: {
            source: {
                "uri": f"s3://test/{game_id}/{source}.json",
                "sha256": _sha(f"{game_id}/{source}"),
            }
            for source in ("nba_schedule", "nba_box_score", "nba_play_by_play")
        }
        for game_id in game_ids
    }


def _quality(game_ids: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        game_id: {
            "final_home_score": 110 + index,
            "final_away_score": 105 + index,
            "box_home_points": 110 + index,
            "box_away_points": 105 + index,
            "play_by_play_possessions": 201.0 + index,
            "observed_stint_possessions": 201.0 + index,
        }
        for index, game_id in enumerate(game_ids)
    }


def _publish_data(
    tmp_path: Path,
    *,
    game_date: date,
    game_ids: list[str],
    parent: Path | None = None,
    suffix: str = "",
):
    return publish_daily_data_artifact(
        root=tmp_path / "data-root",
        season=SEASON,
        game_date_pacific=game_date,
        game_ids=game_ids,
        game_sources=_sources(game_ids),
        game_quality=_quality(game_ids),
        components=_components(tmp_path, DATA_COMPONENTS, f"data{suffix}"),
        parent_bundle=parent,
        repairs=[
            {"game_id": game_ids[0], "action": "identity alias reconciled", "source": "NBA"}
        ]
        if game_ids
        else [],
    )


def _publish_model(
    tmp_path: Path,
    *,
    data_bundle: Path,
    parent: Path | None = None,
    suffix: str = "",
):
    return publish_daily_model_artifact(
        root=tmp_path / "model-root",
        data_bundle=data_bundle,
        parent_bundle=parent,
        components=_components(tmp_path, MODEL_COMPONENTS, f"model{suffix}"),
        model_provenance={
            "code_revision": "0123456789abcdef",
            "model_version": "nail-in-season-v1",
            "configuration_sha256": _sha("configuration"),
            "prior_sha256": _sha("preseason-prior"),
        },
    )


def test_data_and_model_artifacts_form_separate_daily_hash_chains(tmp_path: Path) -> None:
    day_one_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 20),
        game_ids=["0022600001", "0022600002"],
        suffix="-one",
    )
    day_one_model = _publish_model(
        tmp_path, data_bundle=day_one_data.path, suffix="-one"
    )
    day_two_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600003"],
        parent=day_one_data.path,
        suffix="-two",
    )
    day_two_model = _publish_model(
        tmp_path,
        data_bundle=day_two_data.path,
        parent=day_one_model.path,
        suffix="-two",
    )

    data_manifest = verify_daily_data_artifact(day_two_data.path)
    model_manifest = verify_daily_model_artifact(
        day_two_model.path, data_bundle=day_two_data.path
    )

    assert data_manifest["identity"]["parent_data_sha256"] == (
        day_one_data.manifest["bundle_sha256"]
    )
    assert "player_ratings" not in data_manifest["identity"]["components"]
    assert model_manifest["identity"]["input_data_sha256"] == (
        day_two_data.manifest["bundle_sha256"]
    )
    assert model_manifest["identity"]["parent_model_sha256"] == (
        day_one_model.manifest["bundle_sha256"]
    )
    assert model_manifest["identity"]["cumulative_game_ids"] == [
        "0022600001",
        "0022600002",
        "0022600003",
    ]


def test_daily_data_publish_is_idempotent(tmp_path: Path) -> None:
    first = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    second = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )

    assert first.created is True
    assert second.created is False
    assert second.path == first.path


def test_daily_data_rejects_an_incomplete_slate(tmp_path: Path) -> None:
    game_ids = ["0022600001", "0022600002"]
    with pytest.raises(ValueError, match="exact Pacific slate"):
        publish_daily_data_artifact(
            root=tmp_path / "data-root",
            season=SEASON,
            game_date_pacific=date(2026, 10, 20),
            game_ids=game_ids,
            game_sources=_sources(game_ids[:1]),
            game_quality=_quality(game_ids),
            components=_components(tmp_path, DATA_COMPONENTS, "incomplete"),
        )


def test_daily_data_rejects_unresolved_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unresolved validation errors"):
        publish_daily_data_artifact(
            root=tmp_path / "data-root",
            season=SEASON,
            game_date_pacific=date(2026, 10, 20),
            game_ids=["0022600001"],
            game_sources=_sources(["0022600001"]),
            game_quality=_quality(["0022600001"]),
            components=_components(tmp_path, DATA_COMPONENTS, "error"),
            validation_errors=[
                {"game_id": "0022600001", "message": "lineup reconstruction failed"}
            ],
        )


def test_daily_data_requires_the_immediately_preceding_date(tmp_path: Path) -> None:
    day_one = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    with pytest.raises(ValueError, match="immediately preceding Pacific date"):
        _publish_data(
            tmp_path,
            game_date=date(2026, 10, 22),
            game_ids=["0022600002"],
            parent=day_one.path,
            suffix="-gap",
        )


def test_daily_data_verification_detects_component_mutation(tmp_path: Path) -> None:
    artifact = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    component = artifact.path / artifact.manifest["identity"]["components"]["actual_minutes"][
        "path"
    ]
    component.write_text("changed")

    with pytest.raises(ValueError, match="checksum changed"):
        verify_daily_data_artifact(artifact.path)


def test_daily_data_rejects_score_or_possession_nonconservation(tmp_path: Path) -> None:
    quality = _quality(["0022600001"])
    quality["0022600001"]["box_home_points"] = 109
    with pytest.raises(ValueError, match="score conservation"):
        publish_daily_data_artifact(
            root=tmp_path / "data-root",
            season=SEASON,
            game_date_pacific=date(2026, 10, 20),
            game_ids=["0022600001"],
            game_sources=_sources(["0022600001"]),
            game_quality=quality,
            components=_components(tmp_path, DATA_COMPONENTS, "score-failure"),
        )

    quality = _quality(["0022600001"])
    quality["0022600001"]["observed_stint_possessions"] = 199.0
    with pytest.raises(ValueError, match="possession conservation"):
        publish_daily_data_artifact(
            root=tmp_path / "data-root",
            season=SEASON,
            game_date_pacific=date(2026, 10, 20),
            game_ids=["0022600001"],
            game_sources=_sources(["0022600001"]),
            game_quality=quality,
            components=_components(tmp_path, DATA_COMPONENTS, "possession-failure"),
        )


def test_daily_data_pointer_only_advances_along_the_verified_chain(tmp_path: Path) -> None:
    first = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    second = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        parent=first.path,
        suffix="-second",
    )
    alternate_first = _publish_data(
        tmp_path / "alternate",
        game_date=date(2026, 10, 20),
        game_ids=["0099999999"],
    )
    alternate_second = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600099"],
        parent=alternate_first.path,
        suffix="-alternate-second",
    )

    _, first_advanced = promote_daily_data_artifact(first.path, root=tmp_path / "data-root")
    with pytest.raises(ValueError, match="does not extend the current chain tip"):
        promote_daily_data_artifact(alternate_second.path, root=tmp_path / "data-root")
    pointer, second_advanced = promote_daily_data_artifact(
        second.path, root=tmp_path / "data-root"
    )

    assert first_advanced is True
    assert second_advanced is True
    assert pointer["bundle_sha256"] == second.manifest["bundle_sha256"]
    assert (tmp_path / "data-root" / "latest.json").is_file()


def test_daily_data_chain_report_summarizes_the_promoted_chain(tmp_path: Path) -> None:
    artifact = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    promote_daily_data_artifact(artifact.path, root=tmp_path / "data-root")

    report = write_daily_data_chain_report(tmp_path / "data-root")
    body = report.read_text()

    assert "CLEAN THROUGH 2026-10-20" in body
    assert "1 games · 1 unique games" in body
    assert "ESPN dependencies: 0" in body


def test_daily_model_rejects_excluded_games(tmp_path: Path) -> None:
    data = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    with pytest.raises(ValueError, match="cannot publish with excluded games"):
        publish_daily_model_artifact(
            root=tmp_path / "model-root",
            data_bundle=data.path,
            components=_components(tmp_path, MODEL_COMPONENTS, "excluded"),
            model_provenance={
                "code_revision": "abc",
                "model_version": "nail",
                "configuration_sha256": _sha("configuration"),
                "prior_sha256": _sha("prior"),
            },
            excluded_games=[{"game_id": "0022600001", "reason": "failed"}],
        )


def test_daily_model_requires_a_parent_when_data_is_chained(tmp_path: Path) -> None:
    day_one_data = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    day_two_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        parent=day_one_data.path,
        suffix="-two",
    )
    with pytest.raises(ValueError, match="requires a parent model"):
        _publish_model(tmp_path, data_bundle=day_two_data.path, suffix="-missing-parent")


def test_daily_model_rejects_a_parent_from_the_wrong_data_chain(tmp_path: Path) -> None:
    first_data = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    first_model = _publish_model(tmp_path, data_bundle=first_data.path)
    unrelated_data = _publish_data(
        tmp_path / "unrelated",
        game_date=date(2026, 10, 20),
        game_ids=["0099999999"],
    )
    second_data = _publish_data(
        tmp_path / "unrelated",
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        parent=unrelated_data.path,
        suffix="-two",
    )

    with pytest.raises(ValueError, match="does not match the data artifact chain"):
        _publish_model(
            tmp_path,
            data_bundle=second_data.path,
            parent=first_model.path,
            suffix="-wrong-chain",
        )
