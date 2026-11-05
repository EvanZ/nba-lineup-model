from __future__ import annotations

from datetime import date

from nba_lineup_model.storage import active_artifact_migration as migration
from test_daily_artifacts import _publish_data, _publish_model


def test_plan_follows_only_the_active_transitive_lineage(tmp_path, monkeypatch):
    day_one_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 20),
        game_ids=["0022600001"],
        suffix="-selected-one",
    )
    day_one_model = _publish_model(
        tmp_path,
        data_bundle=day_one_data.path,
        suffix="-selected-one",
    )
    day_two_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 21),
        game_ids=["0022600002"],
        suffix="-selected-two",
    )
    day_two_model = _publish_model(
        tmp_path,
        data_bundle=day_two_data.path,
        parent=day_one_model.path,
        suffix="-selected-two",
    )
    unused_data = _publish_data(
        tmp_path,
        game_date=date(2026, 10, 22),
        game_ids=["0022600003"],
        suffix="-unused",
    )
    unused_model = _publish_model(
        tmp_path,
        data_bundle=unused_data.path,
        suffix="-unused",
    )

    release = tmp_path / ("runtime_sha256=" + "c" * 64)
    release.mkdir()
    (release / "runtime-manifest.json").write_text('{"canonical": true}\n')
    manifest = {
        "runtime_sha256": "c" * 64,
        "identity": {
            "season": "2026-27",
            "evidence_through_date": "2026-10-21",
        },
        "approved_manifest": {
            "identity": {
                "data": {"bundle_path": str(day_two_data.path)},
                "model": {"bundle_path": str(day_two_model.path)},
            }
        },
    }
    monkeypatch.setattr(
        migration,
        "read_active_activation",
        lambda _pointer: {
            "environment": "dev",
            "generation": 25,
            "operation_id": "recorded-active-operation",
            "release_path": str(release),
            "runtime_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        migration,
        "verify_production_site_release",
        lambda _release: manifest,
    )

    plan = migration.build_active_migration_plan(tmp_path / "live-release")
    assert set(plan.data_bundles) == {day_one_data.path, day_two_data.path}
    assert set(plan.model_bundles) == {day_one_model.path, day_two_model.path}
    assert unused_data.path not in plan.data_bundles
    assert unused_model.path not in plan.model_bundles
    assert plan.activation_generation == 25
    assert len(plan.dependencies) == 4
