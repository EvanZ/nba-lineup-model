from __future__ import annotations

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
    EASTERN,
    active_live_game_date,
    cdn_morning_reconciliation_schedule,
    completed_game_cdn_play_by_play_sensor,
    completed_game_stats_v3_sensor,
    defs,
    final_game_archive_morning_reconciliation_schedule,
    in_season_nail_shadow_schedule,
    live_score_sensor,
    schedule_refresh_schedule,
    stats_v3_morning_reconciliation_schedule,
    update_live_scores_job,
    yesterday_finalization_schedule,
)


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


def test_definitions_are_loadable_and_automations_default_stopped() -> None:
    Definitions.validate_loadable(defs)
    assert schedule_refresh_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert yesterday_finalization_schedule.default_status is DefaultScheduleStatus.STOPPED
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
