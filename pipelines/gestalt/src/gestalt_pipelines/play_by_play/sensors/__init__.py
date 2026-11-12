"""Parse yesterday after its scheduled play-by-play refresh succeeds."""

from dagster import (
    DagsterRunStatus, DefaultSensorStatus, RunRequest, RunStatusSensorContext,
    SkipReason, run_status_sensor,
)

from ..jobs import capture_nba_play_by_play, parse_nba_games


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS,
    monitored_jobs=[capture_nba_play_by_play], request_job=parse_nba_games,
    minimum_interval_seconds=30, default_status=DefaultSensorStatus.STOPPED,
    description="Parse the same date after the successful scheduled 3 a.m. Pacific PBP capture.",
)
def play_by_play_ready_for_stints(context: RunStatusSensorContext):
    if context.dagster_run.tags.get("dagster/schedule_name") != "play_by_play_morning":
        return SkipReason("Manual captures/backfills do not automatically launch parsing.")
    day = context.dagster_run.tags.get("dagster/partition")
    if not day:
        raise ValueError("The scheduled play-by-play run has no date partition.")
    return RunRequest(run_key=context.dagster_run.run_id, partition_key=day)


__all__ = ["play_by_play_ready_for_stints"]
