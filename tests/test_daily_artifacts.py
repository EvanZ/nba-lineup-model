from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from datetime import date
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from nba_lineup_model.season.daily_data_artifact import (
    DEFAULT_REQUIRED_COMPONENTS as DATA_COMPONENTS,
)
from nba_lineup_model.season.daily_data_artifact import (
    migrate_legacy_daily_data_pointer,
    promote_daily_data_artifact,
    publish_daily_data_artifact,
    resolve_daily_data_pointer,
    verify_daily_data_artifact,
    verify_daily_data_collection,
)
from nba_lineup_model.season.daily_data_replay import write_daily_data_chain_report
from nba_lineup_model.season.daily_model_artifact import (
    DEFAULT_REQUIRED_COMPONENTS as MODEL_COMPONENTS,
)
from nba_lineup_model.season.daily_model_artifact import (
    publish_daily_model_artifact,
    verify_daily_model_artifact,
)
from nba_lineup_model.season.durable_artifact_storage import (
    publish_daily_data_bundle,
    publish_daily_model_bundle,
)
from nba_lineup_model.storage.artifact_store import resolve_storage_manifest

SEASON = "2026-27"
LAKE = "s3://test-bucket/lake"


class DurableFakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.gets = []
        self.lock = threading.Lock()

    def head_object(self, *, Bucket, Key, ChecksumMode):
        with self.lock:
            value = self.objects.get((Bucket, Key))
        if value is None:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data, metadata, checksum = value
        return {
            "ContentLength": len(data),
            "Metadata": metadata,
            "ChecksumSHA256": checksum,
        }

    def put_object(self, **kwargs):
        identity = (kwargs["Bucket"], kwargs["Key"])
        data = kwargs["Body"].read()
        checksum = kwargs["ChecksumSHA256"]
        assert checksum == base64.b64encode(hashlib.sha256(data).digest()).decode()
        with self.lock:
            if identity in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
                )
            self.objects[identity] = data, kwargs["Metadata"], checksum
            self.puts.append(kwargs["Key"])
        return {"ChecksumSHA256": checksum}

    def get_object(self, *, Bucket, Key, ChecksumMode):
        with self.lock:
            value = self.objects.get((Bucket, Key))
            self.gets.append(Key)
        if value is None:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = value[0]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}


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
            for source in (
                "nba_schedule",
                "nba_box_score",
                "nba_play_by_play",
                "nba_injury_report",
                "nba_inactive_list",
            )
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
        repairs=[{"game_id": game_ids[0], "action": "identity alias reconciled", "source": "NBA"}]
        if game_ids
        else [],
    )


def _write_legacy_pointer(root: Path, artifact) -> dict[str, object]:
    identity = artifact.manifest["identity"]
    relative = artifact.path.relative_to(root)
    pointer = {
        "schema_version": 1,
        "artifact_kind": "daily_data",
        "season": identity["season"],
        "game_date_pacific": identity["game_date_pacific"],
        "bundle_sha256": artifact.manifest["bundle_sha256"],
        "bundle_path": relative.as_posix(),
        "promoted_at": "2026-11-04T12:00:00Z",
    }
    history = root / "pointer_history"
    history.mkdir(parents=True, exist_ok=True)
    (history / f"{artifact.manifest['bundle_sha256']}.json").write_text(json.dumps(pointer))
    (root / "latest.json").write_text(json.dumps(pointer))
    return pointer


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


def test_data_artifacts_are_independent_while_model_artifacts_are_causal(tmp_path: Path) -> None:
    day_one_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 20),
        game_ids=["0022600001", "0022600002"],
        suffix="-one",
    )
    day_one_model = _publish_model(tmp_path, data_bundle=day_one_data.path, suffix="-one")
    day_two_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600003"],
        suffix="-two",
    )
    day_two_model = _publish_model(
        tmp_path,
        data_bundle=day_two_data.path,
        parent=day_one_model.path,
        suffix="-two",
    )

    data_manifest = verify_daily_data_artifact(day_two_data.path)
    model_manifest = verify_daily_model_artifact(day_two_model.path, data_bundle=day_two_data.path)

    assert "parent_data_sha256" not in data_manifest["identity"]
    assert "player_ratings" not in data_manifest["identity"]["components"]
    assert (
        model_manifest["identity"]["input_data_sha256"] == (day_two_data.manifest["bundle_sha256"])
    )
    assert (
        model_manifest["identity"]["parent_model_sha256"]
        == (day_one_model.manifest["bundle_sha256"])
    )
    assert model_manifest["identity"]["cumulative_game_ids"] == [
        "0022600001",
        "0022600002",
        "0022600003",
    ]


def test_durable_daily_bundle_preserves_identity_and_is_idempotent(tmp_path: Path) -> None:
    data = _publish_data(
        tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"]
    )
    model = _publish_model(tmp_path, data_bundle=data.path)
    client = DurableFakeS3()
    cache = tmp_path / "artifact-cache"

    with pytest.raises(FileNotFoundError, match="storage manifest is missing"):
        publish_daily_model_bundle(
            model.path, cache_root=cache, client=client, lake_uri=LAKE
        )

    durable_data = publish_daily_data_bundle(
        data.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    durable_model = publish_daily_model_bundle(
        model.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    assert durable_data.reference.content_id == (
        "data_sha256=" + data.manifest["bundle_sha256"]
    )
    assert durable_model.reference.content_id == (
        "model_sha256=" + model.manifest["bundle_sha256"]
    )
    stored_model = resolve_storage_manifest(
        reference=durable_model.reference,
        cache_root=cache,
        client=client,
        lake_uri=LAKE,
    )
    assert stored_model["dependencies"] == [durable_data.reference.as_dict()]
    assert verify_daily_data_artifact(data.path) == data.manifest
    assert verify_daily_model_artifact(model.path) == model.manifest

    puts = list(client.puts)
    repeated = publish_daily_model_bundle(
        model.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    assert repeated.objects.uploaded == 0
    assert client.puts == puts


def test_durable_model_requires_its_parent_model_manifest(tmp_path: Path) -> None:
    day_one_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 20),
        game_ids=["0022600001"],
        suffix="-durable-one",
    )
    day_one_model = _publish_model(
        tmp_path, data_bundle=day_one_data.path, suffix="-durable-one"
    )
    day_two_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        suffix="-durable-two",
    )
    day_two_model = _publish_model(
        tmp_path,
        data_bundle=day_two_data.path,
        parent=day_one_model.path,
        suffix="-durable-two",
    )
    client = DurableFakeS3()
    cache = tmp_path / "artifact-cache"
    day_one_data_stored = publish_daily_data_bundle(
        day_one_data.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    day_two_data_stored = publish_daily_data_bundle(
        day_two_data.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    with pytest.raises(FileNotFoundError, match="storage manifest is missing"):
        publish_daily_model_bundle(
            day_two_model.path, cache_root=cache, client=client, lake_uri=LAKE
        )

    day_one_model_stored = publish_daily_model_bundle(
        day_one_model.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    day_two_model_stored = publish_daily_model_bundle(
        day_two_model.path, cache_root=cache, client=client, lake_uri=LAKE
    )
    stored_parent = resolve_storage_manifest(
        reference=day_one_model_stored.reference,
        cache_root=cache,
        client=client,
        lake_uri=LAKE,
    )
    assert stored_parent["dependencies"] == [day_one_data_stored.reference.as_dict()]
    stored_child = resolve_storage_manifest(
        reference=day_two_model_stored.reference,
        cache_root=cache,
        client=client,
        lake_uri=LAKE,
    )
    assert stored_child["dependencies"] == [
        day_two_data_stored.reference.as_dict(),
        day_one_model_stored.reference.as_dict(),
    ]


def test_daily_data_publish_is_idempotent(tmp_path: Path) -> None:
    first = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    second = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])

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


def test_migrate_legacy_daily_data_pointer_builds_exact_collection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data-root"
    day_one = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    day_two = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        suffix="-day-two",
    )
    _write_legacy_pointer(root, day_one)
    _write_legacy_pointer(root, day_two)

    pointer, migrated = migrate_legacy_daily_data_pointer(root)
    collection = resolve_daily_data_pointer(root, pointer)

    assert migrated is True
    assert pointer["artifact_kind"] == "daily_data_collection"
    assert pointer["date_count"] == 2
    assert (
        collection["identity"]["dates"]["2026-10-21"]["bundle_sha256"]
        == day_two.manifest["bundle_sha256"]
    )
    repeated, changed = migrate_legacy_daily_data_pointer(root)
    assert repeated == pointer
    assert changed is False


def test_migrate_legacy_daily_data_pointer_rejects_conflicting_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data-root"
    original = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    replacement = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 20),
        game_ids=["0022600001"],
        suffix="-replacement",
    )
    _write_legacy_pointer(root, original)
    _write_legacy_pointer(root, replacement)

    with pytest.raises(ValueError, match="selects multiple artifacts"):
        migrate_legacy_daily_data_pointer(root)


def test_daily_data_collection_rejects_a_new_date_gap(tmp_path: Path) -> None:
    day_one = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    day_three = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 22),
        game_ids=["0022600002"],
        suffix="-gap",
    )
    promote_daily_data_artifact(day_one.path, root=tmp_path / "data-root")
    with pytest.raises(ValueError, match="immediately follow the collection tip"):
        promote_daily_data_artifact(day_three.path, root=tmp_path / "data-root")


def test_daily_data_verification_detects_component_mutation(tmp_path: Path) -> None:
    artifact = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    component = (
        artifact.path / artifact.manifest["identity"]["components"]["actual_minutes"]["path"]
    )
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


def test_repair_reselects_one_date_without_rewriting_later_data(tmp_path: Path) -> None:
    first = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    second = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        suffix="-second",
    )
    _, first_advanced = promote_daily_data_artifact(first.path, root=tmp_path / "data-root")
    original_pointer, second_advanced = promote_daily_data_artifact(
        second.path, root=tmp_path / "data-root"
    )
    original_collection = resolve_daily_data_pointer(tmp_path / "data-root", original_pointer)
    original_second = dict(original_collection["identity"]["dates"]["2026-10-21"])

    repaired_first = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 20),
        game_ids=["0022600001"],
        suffix="-repaired",
    )
    repaired_pointer, repaired_advanced = promote_daily_data_artifact(
        repaired_first.path, root=tmp_path / "data-root"
    )
    repaired_collection = resolve_daily_data_pointer(tmp_path / "data-root", repaired_pointer)

    assert first_advanced is True
    assert second_advanced is True
    assert repaired_advanced is True
    assert original_pointer["collection_sha256"] != repaired_pointer["collection_sha256"]
    assert repaired_pointer["through_date"] == "2026-10-21"
    assert (
        repaired_collection["identity"]["dates"]["2026-10-20"]["bundle_sha256"]
        == repaired_first.manifest["bundle_sha256"]
    )
    assert repaired_collection["identity"]["dates"]["2026-10-21"] == original_second
    assert second.path.is_dir()
    verify_daily_data_collection(
        tmp_path / "data-root" / repaired_pointer["collection_path"],
        root=tmp_path / "data-root",
    )


def test_daily_data_chain_report_summarizes_the_promoted_chain(tmp_path: Path) -> None:
    artifact = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    promote_daily_data_artifact(artifact.path, root=tmp_path / "data-root")

    report = write_daily_data_chain_report(tmp_path / "data-root")
    body = report.read_text()

    assert "CLEAN THROUGH 2026-10-20" in body
    assert "1 games · 1 unique games" in body
    assert "ESPN dependencies: 0" in body


def test_daily_model_rejects_excluded_games(tmp_path: Path) -> None:
    data = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
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


def test_daily_model_parent_remains_causally_consecutive(tmp_path: Path) -> None:
    day_one_data = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    day_one_model = _publish_model(tmp_path, data_bundle=day_one_data.path)
    day_three_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 22),
        game_ids=["0022600002"],
        suffix="-three",
    )

    with pytest.raises(ValueError, match="immediately preceding Pacific date"):
        _publish_model(
            tmp_path,
            data_bundle=day_three_data.path,
            parent=day_one_model.path,
            suffix="-gap",
        )


def test_daily_model_rejects_a_parent_with_different_provenance(tmp_path: Path) -> None:
    day_one_data = _publish_data(tmp_path, game_date=date(2026, 10, 20), game_ids=["0022600001"])
    day_one_model = _publish_model(tmp_path, data_bundle=day_one_data.path)
    day_two_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        suffix="-two",
    )

    with pytest.raises(ValueError, match="different model provenance"):
        publish_daily_model_artifact(
            root=tmp_path / "model-root",
            data_bundle=day_two_data.path,
            parent_bundle=day_one_model.path,
            components=_components(tmp_path, MODEL_COMPONENTS, "model-two"),
            model_provenance={
                "code_revision": "different",
                "model_version": "nail-in-season-v1",
                "configuration_sha256": _sha("configuration"),
                "prior_sha256": _sha("preseason-prior"),
            },
        )
