from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from dagster import DefaultScheduleStatus, DefaultSensorStatus, Definitions

from nba_lineup_model.orchestration.definitions import (
    active_live_game_date,
    completed_game_stats_v3_sensor,
    defs,
    live_score_sensor,
    schedule_refresh_schedule,
    stats_v3_morning_reconciliation_schedule,
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


def test_definitions_are_loadable_and_automations_default_stopped() -> None:
    Definitions.validate_loadable(defs)
    assert schedule_refresh_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert yesterday_finalization_schedule.default_status is DefaultScheduleStatus.STOPPED
    assert (
        stats_v3_morning_reconciliation_schedule.default_status
        is DefaultScheduleStatus.STOPPED
    )
    assert live_score_sensor.default_status is DefaultSensorStatus.STOPPED
    assert completed_game_stats_v3_sensor.default_status is DefaultSensorStatus.STOPPED
