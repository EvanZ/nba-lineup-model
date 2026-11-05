"""Explicitly approved injury-only replay; original selections remain immutable."""
from __future__ import annotations

import copy
import shutil
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.official_schedule_runtime import (
    reproject_captured_schedule_window, select_captured_injuries,
)
from nba_lineup_model.season.release_artifact_contracts import SealedCodeIdentity
from nba_lineup_model.season.serving_selection import (
    FORECAST_FIELDS, INDEX_FILENAME, MARKET_FIELDS, ObservationSnapshot,
    _write, digest, forecast_inputs, read_object, read_selected_day, verify_schedule_selection,
)
from nba_lineup_model.season.upcoming_forecasts import require_forecast_model


def replay_injury_selection(*, source: Path, target: Path, observation_root: Path,
    model_bundle: Path, forecast_response: dict, game_date: date,
    approval_reference: str, code: dict, now: datetime | None = None) -> dict:
    """Recalculate one approved date with captured pre-tip injuries and the same model.

    This is not original-live publication. It neither stages original-live
    registry versions nor invents earlier computation/publication timestamps.
    """
    if not approval_reference.strip():
        raise ValueError("Historical injury replay requires explicit approval")
    SealedCodeIdentity(**code)
    now = (now or datetime.now(UTC)).astimezone(UTC)
    original = verify_schedule_selection(source)
    model = require_forecast_model(forecast_response, model_bundle, game_date)
    day = game_date.isoformat()
    if day not in original["dates"]:
        raise ValueError(f"Approved replay date is missing: {day}")
    observations = observation_root.resolve(strict=True)
    captures = select_captured_injuries(schedule_root=source, observation_root=observations,
        start_date=day, days_ahead=0, now=now, include_started=True)
    before = read_object(source / original["dates"][day]["path"])
    expected = {str(g["game_id"]) for g in before["games"]}
    if set(captures) != expected:
        raise ValueError("Injury replay must cover every game in the approved date")
    request = {"source_selection": original["selection_map_sha256"],
        "game_date": day, "approval_reference": approval_reference,
        "model": model["bundle_sha256"], "forecast_response_sha256": digest(forecast_response),
        "code": code, "injury_captures": captures}
    request_id = digest(request)
    if target.exists():
        existing = verify_schedule_selection(target)
        if existing.get("injury_replay", {}).get("request_id") != request_id:
            raise ValueError("Injury replay output already contains different inputs")
        return existing["injury_replay"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".injury-replay-", dir=target.parent) as temporary:
        staged = Path(temporary) / "selection"
        shutil.copytree(source, staged)
        result = reproject_captured_schedule_window(start_date=day, days_ahead=0,
            state_dir=staged, projection_url="captured://accepted-model",
            forecast_response=forecast_response, injury_captures=captures,
            approved_replay_dates=(day,), now=now)
        if result["reprojected_game_count"] != len(expected):
            raise ValueError("Injury replay regenerated incomplete coverage")
        index = copy.deepcopy(original)
        path = staged / index["dates"][day]["path"]
        board = read_object(path)
        originals = {str(g["game_id"]): g for g in before["games"]}
        for game in board["games"]:
            gid = str(game["game_id"])
            old = originals[gid]
            old_forecast = old.get("pregame", old)
            # Explicitly replace selected predictions, not original archive files.
            pregame = copy.deepcopy(old_forecast)
            for field in FORECAST_FIELDS:
                pregame[field] = copy.deepcopy(game.get(field))
            game["pregame"] = pregame
            for field in MARKET_FIELDS:
                if pregame.get(field) != old_forecast.get(field):
                    raise ValueError(f"Injury replay changed market selection: {gid}")
            selected = {k: v for k, v in index["games"][gid].items()
                if k not in ("game_date", "selection_id", "registry_selection", "version_reference")}
            selected.update(forecast_id=digest(forecast_inputs(pregame)),
                provenance="retrospective_replay", original_publication_receipt=None,
                approval_reference=approval_reference, computed_at=now.isoformat(),
                original_selection_id=index["games"][gid]["selection_id"],
                injury_capture=captures[gid]["capture"])
            selected["selection_id"] = digest(selected)
            game["forecast_selection"] = selected
            index["games"][gid] = {**selected, "game_date": day}
            # Results are recalculated below from official observations, not copied grades.
            game.pop("final_audit", None)
            game.pop("retrospective", None)
        _write(path, board)
        report = {"request_id": request_id, "request": request,
                  "computed_at": now.isoformat(), "replayed_game_ids": sorted(expected)}
        index["injury_replay"] = report

        def seal():
            index["dates"][day]["sha256"] = file_sha256(path)
            index["files"] = {str(p.relative_to(staged)): file_sha256(p)
                for p in sorted(staged.rglob("*")) if p.is_file() and p.name != INDEX_FILENAME}
            index.pop("selection_map_sha256", None)
            index["selection_map_sha256"] = digest(index)
            _write(staged / INDEX_FILENAME, index)
            verify_schedule_selection(staged)

        seal()
        snapshot = ObservationSnapshot.capture(observations, [day])
        board = read_selected_day(day, root=staged, observations=snapshot)
        for game in board["games"]:
            if game.get("data_quality_errors"):
                raise ValueError(f"Replayed game validation failed: {game['game_id']}: {game['data_quality_errors']}")
            if game.get("status", {}).get("state") == "post" and not (game.get("retrospective") or {}).get("ledger"):
                raise ValueError(f"Final game has no official actual-minute ledger: {game['game_id']}")
        _write(path, board)
        seal()
        for other, ref in original["dates"].items():
            if other != day and file_sha256(staged / ref["path"]) != ref["sha256"]:
                raise ValueError(f"Injury replay changed an unapproved date: {other}")
        verify_schedule_selection(source)
        staged.rename(target)
    return report


def main():
    """Prepare an approved correction for review; never activate a release."""
    import argparse
    import json
    from urllib.request import urlopen
    from nba_lineup_model.orchestration.forecast_partition import latest_completed_model
    from nba_lineup_model.season.production_site_release import verify_production_site_release
    from nba_lineup_model.season.sealed_code_package import verify_sealed_code_package

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("live", "observation-root", "output", "code-package"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--game-date", type=date.fromisoformat, required=True)
    p.add_argument("--approval-reference", required=True)
    p.add_argument("--projection-api", required=True)
    args = p.parse_args()
    package = args.code_package.resolve(strict=True)
    code = verify_sealed_code_package(package)["identity"]
    if not Path(__file__).resolve().is_relative_to(package / "code"):
        raise ValueError("Run the replay using the specified sealed code package")
    live = args.live.resolve(strict=True)
    verify_production_site_release(live)
    model = latest_completed_model(live, args.game_date)
    with urlopen(args.projection_api, timeout=120) as response:
        forecast = json.load(response)
    result = replay_injury_selection(source=live / "selected-schedule", target=args.output,
        observation_root=args.observation_root, model_bundle=model, forecast_response=forecast,
        game_date=args.game_date, approval_reference=args.approval_reference, code=code)
    verify_sealed_code_package(package)
    if args.live.resolve(strict=True) != live:
        raise ValueError("Active release changed during replay; do not publish this result")
    print(json.dumps({"status": "prepared_not_published", "output": str(args.output),
        "game_date": str(args.game_date), "game_count": len(result["replayed_game_ids"]),
        "model_bundle": str(model), "request_id": result["request_id"]}))


if __name__ == "__main__":
    main()
