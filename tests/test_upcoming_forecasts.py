import copy
import json
from datetime import date, datetime, timedelta

import pytest
from test_official_schedule_runtime import GAME_DATE, START, _board, _forecast

from nba_lineup_model.orchestration import forecast_partition as job
from nba_lineup_model.season import upcoming_forecasts as upcoming
from nba_lineup_model.season.serving_selection import (
    build_schedule_selection,
    read_object,
    verify_schedule_selection,
)


@pytest.mark.parametrize("day", [date(2024, 2, 14), date(2025, 3, 6), date(2026, 11, 4)])
def test_model_date_is_resolved_from_lineage_with_empty_day_and_not_hardcoded(
    tmp_path, monkeypatch, day
):
    model_root = tmp_path / "models"
    records = {}
    previous = None
    for offset, games in [(-3, ["old-game"]), (-2, ["last-game"]), (-1, [])]:
        d = day + timedelta(days=offset)
        mid = str(abs(offset)) * 64
        path = model_root / "season=test" / f"game_date_pacific={d}" / f"model_sha256={mid}"
        records[path] = {
            "bundle_sha256": mid,
            "identity": {
                "season": "test",
                "game_date_pacific": str(d),
                "daily_game_ids": games,
                "parent_model_sha256": previous[1] if previous else None,
                "parent_game_date_pacific": str(previous[0]) if previous else None,
            },
        }
        previous = d, mid
    live = tmp_path / "live"
    live.mkdir()
    (live / "runtime-manifest.json").write_text(
        json.dumps(
            {
                "approved_manifest": {
                    "identity": {"model": {"bundle_path": str(path), "bundle_sha256": mid}}
                }
            }
        )
    )
    monkeypatch.setattr(job, "verify_daily_model_artifact", lambda p: records[p])
    selected = job.latest_completed_model(live, day)
    assert records[selected]["identity"]["game_date_pacific"] == str(day - timedelta(days=2))
    assert records[selected]["identity"]["daily_game_ids"] == ["last-game"]


def fixture_model(tmp_path, monkeypatch):
    bundle = tmp_path / "model"
    bundle.mkdir()
    (bundle / "hca.json").write_text(json.dumps({"home_court_estimate": 1.6}))
    m = {
        "bundle_sha256": "a" * 64,
        "identity": {
            "season": "2026-27",
            "game_date_pacific": "2026-10-30",
            "components": {
                "home_court_state": {"path": "hca.json", "sha256": "b" * 64},
                "player_ratings": {"sha256": "c" * 64},
            },
        },
    }
    monkeypatch.setattr(upcoming, "verify_daily_model_artifact", lambda _: m)
    response = _forecast(home_star_rating=3.0)
    response["win_projection"]["home_court"] = 1.6
    response["minutes"]["projection_provenance"].update(
        rating_as_of="2026-10-30",
        rating_source_uri=f"artifact://daily-model/{'a' * 64}/player_ratings",
        minutes_source_uri=f"artifact://daily-model/{'a' * 64}/minutes_state#serving-v1",
        rating_state_sha256="c" * 64,
    )
    return bundle, response


@pytest.mark.parametrize("damage", ["old_date", "old_model", "old_minutes", "old_hca"])
def test_old_inputs_cannot_pass_even_when_the_forecast_fields_exist(tmp_path, monkeypatch, damage):
    bundle, response = fixture_model(tmp_path, monkeypatch)
    p = response["minutes"]["projection_provenance"]
    if damage == "old_date":
        p["rating_as_of"] = "2026-10-29"
    if damage == "old_model":
        p["rating_source_uri"] = "artifact://old/player_ratings"
    if damage == "old_minutes":
        p["minutes_source_uri"] = "artifact://old/minutes_state#serving-v1"
    if damage == "old_hca":
        response["win_projection"]["home_court"] = 2.84
    with pytest.raises(ValueError, match="selected completed model"):
        upcoming.require_forecast_model(response, bundle, date.fromisoformat(GAME_DATE))


def test_open_forecast_refresh_preserves_history_and_uses_registry(tmp_path, monkeypatch):
    bundle, response = fixture_model(tmp_path, monkeypatch)
    now = datetime.fromisoformat(START.replace("Z", "+00:00")) - timedelta(minutes=30)
    raw = tmp_path / "raw"
    (raw / "latest").mkdir(parents=True)
    board = _board(collected_at=now)
    from nba_lineup_model.season.official_schedule_runtime import _publish_pregame_candidate
    board["games"][0]["injury_report"]["published_at"] = now.isoformat()
    _publish_pregame_candidate(board["games"][0], state_root=raw, captured_at=now)
    locked = copy.deepcopy(board["games"][0])
    locked["game_id"] = "0022600002"
    locked["start_time"] = (now - timedelta(minutes=1)).isoformat()
    # A late status feed must not make a game past its cutoff editable.
    locked["status"]["state"] = "pre"
    board["games"].append(locked)
    (raw / "latest" / f"{GAME_DATE}.json").write_text(json.dumps(board))
    source = tmp_path / "source"
    build_schedule_selection(raw, source, edition_id="approved")
    # Odds can arrive after the first forecast. The next ordinary refresh must
    # consume the independent market capture, not preserve the earlier null.
    board["games"][0]["market"] = {"source": "The Odds API", "captured_at": now.isoformat(),
        "home_spread": -3.5, "away_spread": 3.5}
    _publish_pregame_candidate(board["games"][0], state_root=raw, captured_at=now)
    (raw / "latest" / f"{GAME_DATE}.json").write_text(json.dumps(board))
    original_files = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    catalog = tmp_path / "catalog.parquet"
    catalog.write_bytes(b"official schedule fixture")
    target = tmp_path / "refreshed"
    result = upcoming.refresh_open_selection(
        source=source,
        target=target,
        observation_root=raw,
        model_bundle=bundle,
        forecast_response=response,
        start_date=date.fromisoformat(GAME_DATE),
        days_ahead=0,
        registry_root=tmp_path / "registry",
        catalog=catalog,
        code={
            "sealed": True,
            "package_sha256": "d" * 64,
            "dependency_lock_sha256": "e" * 64,
            "git_revision": "f" * 40,
        },
        environment="dev",
        now=now,
    )
    assert result["reprojected_game_count"] == 1
    index = verify_schedule_selection(target)
    new = read_object(target / "latest" / f"{GAME_DATE}.json")["games"]
    old = read_object(source / "latest" / f"{GAME_DATE}.json")["games"]
    assert new[1] == old[1]
    assert new[0]["market"]["home_spread"] == -3.5
    assert new[0]["market_selection"]["captured_at"] == now.isoformat()
    assert new[0]["projection"]["ledger"]["home_court_adjustment"] == 1.6
    assert new[0]["projection"]["input_provenance"]["rating_as_of"] == "2026-10-30"
    assert index["games"][new[0]["game_id"]]["provenance"] == "original_live"
    assert len(index["publication_forecast_ids"]) == 1
    assert not list((tmp_path / "registry" / "receipts").glob("*"))
    for relative, content in original_files.items():
        assert (source / relative).read_bytes() == content


def test_pipeline_cannot_skip_forecast_refresh_by_preserving_all_selections():
    import inspect

    from nba_lineup_model.orchestration.daily_pipeline import CanonicalPipelineSteps
    from nba_lineup_model.orchestration.resumable_pipeline import PartitionSteps

    assert "preserve_schedule_selections=True" not in inspect.getsource(PartitionSteps.__init__)
    assert "if not c.preserve_schedule_selections" not in inspect.getsource(
        CanonicalPipelineSteps.validate_and_stage
    )
