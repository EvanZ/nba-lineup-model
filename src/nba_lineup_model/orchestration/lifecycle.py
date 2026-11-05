"""Environment-scoped automatic work selection; never a second publisher."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nba_lineup_model.orchestration.daily_work import digest, read_object
from nba_lineup_model.season.slate_time import PACIFIC


def next_nightly_date(*, evidence: date, slates: dict[str, list[str]],
                      final_ids: set[str], now: datetime) -> date | None:
    """Advance chronologically only toward a fully completed nonempty slate.

    Empty dates inside the chain are carry-forward partitions, not evidence of
    new basketball. An empty day by itself never requests another model run.
    """
    today = now.astimezone(PACIFIC).date()
    for day, ids in sorted(slates.items()):
        candidate = date.fromisoformat(day)
        if candidate <= evidence or candidate > today or not ids:
            continue
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate official games on {day}")
        if not set(ids) <= final_ids:
            return None
        return evidence + timedelta(days=1)
    return None


def official_slates(catalog: Path, season: str) -> dict[str, list[str]]:
    import pandas as pd
    frame = pd.read_parquet(catalog)
    frame = frame.loc[frame.season.astype(str).eq(season) & frame.season_type.eq("regular")].copy()
    frame["pacific_date"] = pd.to_datetime(frame.game_time_utc, utc=True).dt.tz_convert(PACIFIC).dt.date
    if frame.game_id.astype(str).duplicated().any():
        raise ValueError("Official catalog has duplicate game IDs")
    return {str(day): sorted(group.game_id.astype(str))
            for day, group in frame.groupby("pacific_date")}


def automation_request(config_path: Path, *, now: datetime | None = None) -> dict | None:
    from nba_lineup_model.orchestration.daily_pipeline import DailyPipelineConfig
    from nba_lineup_model.season.serving_selection import verify_schedule_selection
    from nba_lineup_model.season.official_schedule_runtime import select_captured_injuries
    from nba_lineup_model.season.market_selection import select_captured_markets
    now = now or datetime.now(UTC)
    config = DailyPipelineConfig.read(config_path)
    if not config.activate_after_validation or config.observation_root is None:
        raise ValueError("Automatic processing requires explicit environment publication configuration")
    live = config.live_pointer.resolve(strict=True)
    runtime = read_object(live / "runtime-manifest.json")
    accepted = runtime["approved_manifest"]["identity"]
    evidence = date.fromisoformat(accepted["evidence_through_date"])
    slates = official_slates(live / "catalog/games.parquet", config.season)
    finals = set()
    for path in (config.observation_root / "final").glob("*.summary.json"):
        entry = read_object(path)
        if entry.get("final"):
            finals.add(str(entry["game_id"]))
    target = next_nightly_date(evidence=evidence, slates=slates, final_ids=finals, now=now)
    if target:
        key = digest({"kind": "nightly", "model": accepted["model"]["bundle_sha256"],
                      "date": str(target), "config": str(config_path)})
        return {"job": "advance_daily_environment", "key": key,
                "config": {"config_path": str(config_path), "game_date": str(target)}}
    schedule = live / "selected-schedule"
    index = verify_schedule_selection(schedule)
    today = now.astimezone(PACIFIC).date()
    # All selected future dates are eligible; changed source selections, not a
    # wall-clock bucket, drive requests. Scores do not trigger forecast rebuilds.
    receipts = config.pipeline_root / "automation" / "forecast-inputs"
    for day, ref in sorted(index["dates"].items()):
        if date.fromisoformat(day) < today:
            continue
        board = read_object(schedule / ref["path"])
        open_games = [g for g in board["games"]
                      if datetime.fromisoformat(g["start_time"].replace("Z", "+00:00")) > now
                      and not g.get("pregame") and not g.get("final_audit")
                      and g.get("status", {}).get("state", "pre") == "pre"]
        if not open_games:
            continue
        injuries = select_captured_injuries(schedule_root=schedule,
            observation_root=config.observation_root, start_date=day, days_ahead=0, now=now)
        markets = select_captured_markets(schedule_root=schedule,
            observation_root=config.observation_root, start_date=day, days_ahead=0, now=now)
        ids = {str(g["game_id"]) for g in open_games}
        key = digest({"kind": "forecast", "date": day, "config": str(config_path),
            "model": accepted["model"]["bundle_sha256"],
            "injuries": {k:v for k,v in injuries.items() if k in ids},
            "markets": {k:v for k,v in markets.items() if k in ids}, "open_games": sorted(ids)})
        receipt = receipts / f"{key}.json"
        if receipt.is_file():
            continue
        return {"job": "prepare_forecast_date", "key": key,
                "config": {"config_path": str(config_path), "forecast_date": day,
                           "automation_key": key}}
    return None
