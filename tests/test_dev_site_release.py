from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_lineup_model.season import dev_site_release as subject


def _ledger(*, actual: bool) -> dict[str, object]:
    minutes_key = "actual_minutes" if actual else "projected_minutes"
    contribution_key = "actual_contribution" if actual else "expected_contribution"
    player = {
        "player_id": 1,
        "player_name": "Player",
        "frozen_nail": 1.0,
        minutes_key: 240.0,
        contribution_key: 5.0,
    }
    return {
        "artifact_kind": (
            "retrospective_actual_minutes_attribution"
            if actual
            else "immutable_game_prediction_attribution"
        ),
        "home_margin": 1.0,
        "away": {"players": [player]},
        "home": {"players": [player]},
    }


def _api_validation() -> dict[str, object]:
    return {
        "status": "clean",
        "health": "ok",
        "evidence_through_date": "2026-10-31",
        "evidence_game_count": 1,
        "serving_bundle_sha256": "s" * 64,
        "rankings_bundle_sha256": "s" * 64,
        "lineup_serving_bundle_sha256": "s" * 64,
        "lineup_bundle_sha256": "l" * 64,
        "ranking_player_count": 528,
        "team_player_count": 18,
        "player_id": 201939,
        "lineup_count": 2,
        "scheduled_game_count": 1230,
        "season_performance_games": 1,
        "schedule_game_count": 5,
    }


def _inputs(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    model_root = tmp_path / "model"
    serving_root = tmp_path / "serving"
    lineups_root = tmp_path / "lineups"
    schedule_root = tmp_path / "schedule"
    catalog = tmp_path / "games.parquet"
    preseason_rankings = tmp_path / "preseason-rankings.parquet"
    preseason_profiles = tmp_path / "preseason-profiles.parquet"
    for root in (data_root, model_root, serving_root, lineups_root):
        root.mkdir()
    data_bundle = data_root / "data-bundle"
    model_bundle = model_root / "model-bundle"
    serving_bundle = serving_root / "serving-bundle"
    lineup_bundle = lineups_root / "lineup-bundle"
    for bundle in (data_bundle, model_bundle, serving_bundle, lineup_bundle):
        bundle.mkdir()
    (data_root / "latest.json").write_text(
        json.dumps({"bundle_path": "data-bundle", "bundle_sha256": "d" * 64})
    )
    (model_root / "latest.json").write_text(
        json.dumps({"bundle_path": "model-bundle", "bundle_sha256": "m" * 64})
    )
    (lineups_root / "latest.json").write_text(
        json.dumps({"bundle_path": "lineup-bundle"})
    )
    schedule_final = schedule_root / "final"
    schedule_final.mkdir(parents=True)
    displayed_player = {
        "player_id": 1,
        "player_name": "Player",
        "projected_nail": 1.0,
        "projected_minutes": 240.0,
        "expected_contribution": 5.0,
    }
    projected_minutes = {
        "away": [displayed_player],
        "home": [displayed_player],
    }
    pregame = {
        "projection": {"home_margin": 1.0, "ledger": _ledger(actual=False)},
        "projected_minutes": projected_minutes,
    }
    retrospective = {"home_margin": 1.0, "ledger": _ledger(actual=True)}
    (schedule_final / "game.summary.json").write_text(
        json.dumps(
            {
                "game_id": "game-1",
                "pregame": pregame,
                "retrospective": retrospective,
            }
        )
    )
    schedule_latest = schedule_root / "latest"
    schedule_latest.mkdir()
    (schedule_latest / "2026-10-31.json").write_text(
        json.dumps(
            {
                "games": [
                    {
                        "game_id": "game-1",
                        "projection": pregame["projection"],
                        "projected_minutes": projected_minutes,
                        "retrospective": retrospective,
                        "status": {"completed": True},
                    }
                ]
            }
        )
    )
    catalog.write_bytes(b"catalog")
    pd.DataFrame({"player_id": [1]}).to_parquet(preseason_rankings, index=False)
    pd.DataFrame({"player_id": [1]}).to_parquet(preseason_profiles, index=False)
    data_manifest = {
        "bundle_sha256": "d" * 64,
        "identity": {"game_date_pacific": "2026-10-31"},
    }
    model_manifest = {
        "bundle_sha256": "m" * 64,
        "identity": {
            "season": "2026-27",
            "game_date_pacific": "2026-10-31",
            "input_data_sha256": "d" * 64,
            "cumulative_game_ids": ["game-1"],
            "model_provenance": {"prior_sha256": "p" * 64},
        },
    }
    serving_manifest = {
        "bundle_sha256": "s" * 64,
        "identity": {
            "evidence_through_date": "2026-10-31",
            "sources": {
                "rating_state": {"uri": f"artifact://daily-model/{'m' * 64}/ratings"},
                "minutes_state": {"uri": f"artifact://daily-model/{'m' * 64}/minutes"},
                "rating_trajectory": {
                    "uri": f"artifact://daily-model/{'m' * 64}/trajectory"
                },
            },
            "coverage": {"source_game_ids": ["game-1"]},
        },
    }
    lineup_manifest = {
        "bundle_sha256": "l" * 64,
        "identity": {
            "serving_bundle_sha256": "s" * 64,
            "source_game_ids": ["game-1"],
        },
    }
    monkeypatch.setattr(
        subject,
        "verify_daily_data_artifact",
        lambda path: data_manifest,
    )
    monkeypatch.setattr(
        subject,
        "verify_daily_model_artifact",
        lambda path: model_manifest,
    )
    monkeypatch.setattr(subject, "resolve_latest_bundle", lambda root: serving_bundle)
    monkeypatch.setattr(subject, "verify_serving_bundle", lambda path: serving_manifest)
    monkeypatch.setattr(
        subject,
        "verify_current_season_lineup_bundle",
        lambda path: lineup_manifest,
    )
    monkeypatch.setattr(
        subject.pd,
        "read_parquet",
        lambda path: (
            pd.DataFrame({"player_id": [1]})
            if Path(path) in {preseason_rankings, preseason_profiles}
            else pd.DataFrame({"player_id": [1]})
        ),
    )
    return (
        data_root,
        model_root,
        serving_root,
        lineups_root,
        schedule_root,
        catalog,
        preseason_rankings,
        preseason_profiles,
    )


def test_dev_site_release_binds_all_selected_components(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    release = subject.publish_dev_site_release(
        daily_data_root=inputs[0],
        daily_model_root=inputs[1],
        serving_root=inputs[2],
        lineups_root=inputs[3],
        schedule_state_dir=inputs[4],
        catalog_path=inputs[5],
        preseason_rankings_path=inputs[6],
        preseason_profiles_path=inputs[7],
        code_revision="a" * 40,
        api_validation=_api_validation(),
        output_root=tmp_path / "releases",
    )

    manifest = subject.verify_dev_site_release(release)
    identity = manifest["identity"]
    assert identity["game_count"] == 1
    assert identity["data"]["bundle_sha256"] == "d" * 64
    assert identity["model"]["bundle_sha256"] == "m" * 64
    assert identity["serving"]["bundle_sha256"] == "s" * 64
    assert identity["lineups"]["bundle_sha256"] == "l" * 64
    assert identity["schedule"]["attribution_audit"] == {
        "served_games": 1,
        "pregame_ledgers": 1,
        "completed_games": 1,
        "actual_minutes_ledgers": 1,
    }
    assert subject.resolve_latest_dev_site_release(tmp_path / "releases") == release


def test_dev_site_release_allows_schedule_finals_ahead_of_nightly_model(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    schedule_root = inputs[4]

    original_summary = json.loads(
        (schedule_root / "final" / "game.summary.json").read_text()
    )
    extra_summary = dict(original_summary)
    extra_summary["game_id"] = "game-2"
    (schedule_root / "final" / "game-2.summary.json").write_text(
        json.dumps(extra_summary)
    )

    day_path = schedule_root / "latest" / "2026-10-31.json"
    day = json.loads(day_path.read_text())
    extra_game = dict(day["games"][0])
    extra_game["game_id"] = "game-2"
    day["games"].append(extra_game)
    day_path.write_text(json.dumps(day))

    validation = _api_validation()
    validation["season_performance_games"] = 2
    release = subject.publish_dev_site_release(
        daily_data_root=inputs[0],
        daily_model_root=inputs[1],
        serving_root=inputs[2],
        lineups_root=inputs[3],
        schedule_state_dir=schedule_root,
        catalog_path=inputs[5],
        preseason_rankings_path=inputs[6],
        preseason_profiles_path=inputs[7],
        code_revision="a" * 40,
        api_validation=validation,
        output_root=tmp_path / "releases",
    )

    identity = subject.verify_dev_site_release(release)["identity"]
    assert identity["game_count"] == 1
    assert identity["schedule"]["final_game_count"] == 2
    assert identity["schedule"]["attribution_audit"]["completed_games"] == 2


def test_dev_site_release_rejects_mixed_api_state(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    validation = _api_validation()
    validation["rankings_bundle_sha256"] = "wrong"

    with pytest.raises(ValueError, match="rankings_bundle_sha256"):
        subject.publish_dev_site_release(
            daily_data_root=inputs[0],
            daily_model_root=inputs[1],
            serving_root=inputs[2],
            lineups_root=inputs[3],
            schedule_state_dir=inputs[4],
            catalog_path=inputs[5],
            preseason_rankings_path=inputs[6],
            preseason_profiles_path=inputs[7],
            code_revision="a" * 40,
            api_validation=validation,
            output_root=tmp_path / "releases",
        )


def test_dev_site_release_rejects_any_game_without_actual_minutes_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    summary_path = inputs[4] / "final" / "game.summary.json"
    summary = json.loads(summary_path.read_text())
    summary["retrospective"].pop("ledger")
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="actual-minutes ledger is missing"):
        subject.publish_dev_site_release(
            daily_data_root=inputs[0],
            daily_model_root=inputs[1],
            serving_root=inputs[2],
            lineups_root=inputs[3],
            schedule_state_dir=inputs[4],
            catalog_path=inputs[5],
            preseason_rankings_path=inputs[6],
            preseason_profiles_path=inputs[7],
            code_revision="a" * 40,
            api_validation=_api_validation(),
            output_root=tmp_path / "releases",
        )


def test_api_gate_rejects_any_missing_actual_participant_log(monkeypatch):
    from nba_lineup_model.season import dev_site_release as release
    responses = {
        "/api/players/201939": {"player_id":201939},
        "/api/players/1627777": {"player_id":1627777,"current_season_trajectory":[]},
        "/api/schedule?date=2026-11-01": {"games":[{"game_id":"0022600156","status":{"completed":True},"retrospective":{"actual_minutes":{"home":[{"player_id":1627777,"actual_minutes":21.0}],"away":[]}}}]},
    }
    monkeypatch.setattr(release,"_get_json",lambda url:responses.get(url.removeprefix("http://test"),{}))
    with pytest.raises(ValueError,match="0022600156, player 1627777"):
        release.collect_api_validation("http://test",evidence_date="2026-11-01")
    responses["/api/players/1627777"]["current_season_trajectory"]=[{"game_id":"0022600156","actual_minutes":21.0}]
    assert release.collect_api_validation("http://test",evidence_date="2026-11-01")["verified_player_game_logs"]==1


def test_api_gate_rejects_missed_game_absent_from_box(monkeypatch):
    import pandas as pd
    from datetime import date
    from nba_lineup_model.season import daily_model_replay as replay
    from nba_lineup_model.season import dev_site_release as release
    responses={"/api/players/999":{"current_season_trajectory":[]}}
    monkeypatch.setattr(release,"_get_json",lambda url:responses.get(url.removeprefix("http://test"),{}))
    monkeypatch.setattr(replay,"_read_model_chain",lambda root:[(date(2026,11,1),Path("/model"),{})])
    monkeypatch.setattr(replay,"_component_path",lambda *args:Path("/observations"))
    monkeypatch.setattr(pd,"read_parquet",lambda path:pd.DataFrame([{"game_id":"g1","player_id":999,"team":"BOS","played":False,"actual_minutes":0.0}]))
    with pytest.raises(ValueError,match="opportunity coverage mismatch.*g1, player 999"):
        release.collect_api_validation("http://test",evidence_date="2026-11-01",daily_model_root=Path("/model"))
    responses["/api/players/999"]["current_season_trajectory"]=[{"game_id":"g1","team":"BOS","played":False,"actual_minutes":0.0,"game_on_court_possessions":0.0}]
    result=release.collect_api_validation("http://test",evidence_date="2026-11-01",daily_model_root=Path("/model"))
    assert result["verified_player_opportunities"]==1
    assert result["verified_missed_game_logs"]==1
