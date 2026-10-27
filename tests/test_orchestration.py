from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from dagster import (
    DagsterInstance,
    DagsterRunStatus,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    Definitions,
    build_sensor_context,
)

from nba_lineup_model.orchestration.definitions import (
    DEFAULT_ODDS_API_BOOKMAKERS,
    EASTERN,
    _release_root,
    active_live_game_date,
    cdn_morning_reconciliation_schedule,
    completed_game_cdn_play_by_play_sensor,
    completed_game_stats_v3_sensor,
    defs,
    final_game_archive_morning_reconciliation_schedule,
    in_season_nail_shadow_schedule,
    ingest_completed_cdn_play_by_play_job,
    live_score_sensor,
    nba_market_odds_schedule,
    schedule_refresh_schedule,
    stats_v3_morning_reconciliation_schedule,
    update_live_scores_job,
    yesterday_finalization_schedule,
)
from nba_lineup_model.web_api.the_odds_api import DEFAULT_BOOKMAKERS


def _write_slate(
    root: Path,
    *,
    game_date: str,
    start_time: str,
    state: str,
) -> None:
    path = root / "latest" / f"{game_date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "start_time": start_time,
                        "status": {"state": state},
                    }
                ]
            }
        )
    )


def test_live_window_only_returns_unfinished_active_slate(tmp_path: Path) -> None:
    now = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)
    _write_slate(
        tmp_path,
        game_date="2026-10-24",
        start_time="2026-10-24T22:30:00Z",
        state="in",
    )

    assert active_live_game_date(state_dir=tmp_path, now=now) == "2026-10-24"

    _write_slate(
        tmp_path,
        game_date="2026-10-24",
        start_time="2026-10-24T22:30:00Z",
        state="post",
    )
    assert active_live_game_date(state_dir=tmp_path, now=now) is None


def test_release_root_does_not_depend_on_process_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release = tmp_path / "immutable-release"
    release.mkdir()
    monkeypatch.setenv("GESTALT_RELEASE_ROOT", str(release))
    monkeypatch.chdir(tmp_path)

    assert _release_root() == release.resolve()


def test_live_window_does_not_poll_all_day(tmp_path: Path) -> None:
    _write_slate(
        tmp_path,
        game_date="2026-10-24",
        start_time="2026-10-25T02:00:00Z",
        state="pre",
    )

    assert (
        active_live_game_date(
            state_dir=tmp_path,
            now=datetime(2026, 10, 24, 12, 0, tzinfo=UTC),
        )
        is None
    )


def test_live_score_sensor_does_not_queue_duplicate_inflight_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    now = datetime.now(UTC)
    game_date = now.astimezone(EASTERN).date().isoformat()
    _write_slate(
        tmp_path,
        game_date=game_date,
        start_time=now.isoformat(),
        state="in",
    )
    monkeypatch.setenv("GESTALT_SCHEDULE_STATE_DIR", str(tmp_path))
    instance = DagsterInstance.ephemeral()
    queued = instance.create_run_for_job(
        update_live_scores_job,
        run_config={
            "ops": {"update_live_score_state": {"config": {"game_date": game_date}}}
        },
        status=DagsterRunStatus.STARTED,
        tags={"gestalt/game_date": game_date},
    )

    result = live_score_sensor(build_sensor_context(instance=instance))

    assert "already queued or running" in result.skip_message
    assert queued.run_id in result.skip_message


def test_cdn_sensor_does_not_queue_duplicate_pending_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    game_ids = ["0022600118"]
    digest = hashlib.sha256(",".join(game_ids).encode()).hexdigest()[:12]
    monkeypatch.setenv("GESTALT_SCHEDULE_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "nba_lineup_model.season.cdn_lake.pending_completed_cdn_game_ids",
        lambda **_kwargs: game_ids,
    )
    instance = DagsterInstance.ephemeral()
    queued = instance.create_run_for_job(
        ingest_completed_cdn_play_by_play_job,
        status=DagsterRunStatus.STARTED,
        tags={"gestalt/cdn_pending_digest": digest},
    )

    result = completed_game_cdn_play_by_play_sensor(
        build_sensor_context(instance=instance)
    )

    assert "already queued or running" in result.skip_message
    assert queued.run_id in result.skip_message


def test_cdn_sensor_uses_stable_retryable_pending_set_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    game_ids = ["0022600118"]
    digest = hashlib.sha256(",".join(game_ids).encode()).hexdigest()[:12]
    monkeypatch.setenv("GESTALT_SCHEDULE_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "nba_lineup_model.season.cdn_lake.pending_completed_cdn_game_ids",
        lambda **_kwargs: game_ids,
    )
    instance = DagsterInstance.ephemeral()

    first = completed_game_cdn_play_by_play_sensor(
        build_sensor_context(instance=instance)
    )
    assert first.run_key == f"cdn-pbp:{digest}:attempt-0"
    assert first.tags["gestalt/cdn_pending_digest"] == digest

    instance.create_run_for_job(
        ingest_completed_cdn_play_by_play_job,
        status=DagsterRunStatus.FAILURE,
        tags={"gestalt/cdn_pending_digest": digest},
    )
    retry = completed_game_cdn_play_by_play_sensor(
        build_sensor_context(instance=instance)
    )
    assert retry.run_key == f"cdn-pbp:{digest}:attempt-1"


def test_definitions_are_loadable_and_automations_default_stopped() -> None:
    Definitions.validate_loadable(defs)
    assert DEFAULT_ODDS_API_BOOKMAKERS == DEFAULT_BOOKMAKERS
    assert schedule_refresh_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert yesterday_finalization_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert nba_market_odds_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert nba_market_odds_schedule.cron_schedule == "20 8,12,15-19 * * *"
    assert nba_market_odds_schedule.execution_timezone == "America/Los_Angeles"
    assert (
        final_game_archive_morning_reconciliation_schedule.default_status
        is DefaultScheduleStatus.STOPPED
    )
    assert (
        cdn_morning_reconciliation_schedule.default_status
        is DefaultScheduleStatus.STOPPED
    )
    assert in_season_nail_shadow_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert in_season_nail_shadow_schedule.cron_schedule == "45 5 * * *"
    assert in_season_nail_shadow_schedule.execution_timezone == "America/Los_Angeles"
    assert (
        stats_v3_morning_reconciliation_schedule.default_status
        is DefaultScheduleStatus.STOPPED
    )
    assert live_score_sensor.default_status is DefaultSensorStatus.STOPPED
    assert (
        completed_game_cdn_play_by_play_sensor.default_status
        is DefaultSensorStatus.STOPPED
    )
    assert completed_game_stats_v3_sensor.default_status is DefaultSensorStatus.STOPPED
