"""Select captured pre-tip markets independently of model and injury inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.release_artifact_contracts import SealedCodeIdentity
from nba_lineup_model.season.serving_selection import (
    INDEX_FILENAME, MARKET_FIELDS, _write, digest, forecast_inputs, read_object, verify_schedule_selection,
)


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Market timestamps must include a timezone")
    return result


def select_captured_markets(*, schedule_root: Path, observation_root: Path,
                            start_date: str, days_ahead: int, now: datetime,
                            include_started: bool = False) -> dict:
    selected = {}
    for offset in range(days_ahead + 1):
        day = date.fromisoformat(start_date) + timedelta(days=offset)
        path = schedule_root / "latest" / f"{day}.json"
        if not path.is_file():
            continue
        for game in read_object(path)["games"]:
            gid, tip = str(game["game_id"]), timestamp(game["start_time"])
            if not include_started and (tip <= now or game.get("pregame") or game.get("final_audit")
                    or game.get("status", {}).get("state", "pre") != "pre"
                    or (schedule_root / "frozen" / f"{gid}.json").exists()):
                continue
            options = []

            def consider(source, captured, checksum, uri, kind):
                if captured > now or captured >= tip:
                    return
                if any(source.get(k) != game.get(k) for k in ("game_id", "home_team", "away_team")):
                    raise ValueError(f"Market capture identity mismatch: {gid}")
                market = source.get("market")
                if not market:
                    return
                stamp = timestamp(market["captured_at"])
                if stamp > captured:
                    raise ValueError(f"Market was not available at capture time: {gid}")
                if market.get("source") != "The Odds API":
                    raise ValueError(f"Unapproved market source: {gid}")
                options.append((stamp, captured, checksum, {
                    "market": copy.deepcopy(market), "capture": {
                        "source_sha256": checksum, "uri": str(uri), "kind": kind,
                        "captured_at": captured.isoformat(), "cutoff_at": tip.isoformat()}}))

            for archive in sorted((observation_root / "pregame-candidates" / gid).glob("*.json")):
                value = read_object(archive)
                unsigned = {k: v for k, v in value.items() if k != "candidate_sha256"}
                sha = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if sha != value["candidate_sha256"]:
                    raise ValueError(f"Market capture checksum changed: {archive}")
                if timestamp(value["scheduled_start_time"]) != tip:
                    continue
                consider(value["game"], timestamp(value["captured_at"]), sha, archive, "pregame_candidate")
            latest = observation_root / "latest" / f"{day}.json"
            if not include_started and latest.is_file():
                raw = latest.read_bytes()
                board = json.loads(raw)
                for observed in board["games"]:
                    if str(observed["game_id"]) == gid:
                        consider(observed, timestamp(board["collected_at"]),
                            hashlib.sha256(raw).hexdigest(), latest, "collector_snapshot")
            if options:
                selected[gid] = max(options, key=lambda item: item[:3])[3]
    return selected


def repair_market_selection(*, source: Path, target: Path, observation_root: Path,
                            game_date: date, approval_reference: str, model_sha: str,
                            code: dict, now: datetime | None = None) -> dict:
    """New selected version, unchanged forecasts, explicit historical approval."""
    if not approval_reference.strip():
        raise ValueError("Historical market repair requires explicit approval")
    SealedCodeIdentity(**code)
    now = now or datetime.now(UTC)
    original = verify_schedule_selection(source)
    day = str(game_date)
    before = read_object(source / original["dates"][day]["path"])
    captures = select_captured_markets(schedule_root=source, observation_root=observation_root,
        start_date=day, days_ahead=0, now=now, include_started=True)
    expected = {str(g["game_id"]) for g in before["games"]}
    if set(captures) != expected:
        raise ValueError(f"Missing verified pre-tip markets: {sorted(expected - set(captures))}")
    request = {"source_selection": original["selection_map_sha256"], "game_date": day,
        "approval_reference": approval_reference, "model": model_sha, "code": code,
        "market_captures": captures}
    report = {"request_id": digest(request), "request": request,
        "computed_at": now.isoformat(), "repaired_game_ids": sorted(expected)}
    if target.exists():
        previous = verify_schedule_selection(target).get("market_repair", {})
        if previous.get("request_id") != report["request_id"]:
            raise ValueError("Market repair output contains different inputs")
        return previous
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".market-repair-", dir=target.parent) as temp:
        staged = Path(temp) / "selection"
        shutil.copytree(source, staged)
        index = copy.deepcopy(original)
        board = copy.deepcopy(before)
        for game, old in zip(board["games"], before["games"], strict=True):
            gid = str(game["game_id"])
            if game["forecast_selection"].get("provenance") == "original_live":
                raise ValueError("Historical original-live market repair requires registry revision support")
            game["market"] = copy.deepcopy(captures[gid]["market"])
            if game.get("pregame"):
                game["pregame"]["market"] = copy.deepcopy(game["market"])
            game["market_selection"] = captures[gid]["capture"]
            selected = {k:v for k,v in game["forecast_selection"].items()
                        if k not in ("selection_id", "game_date")}
            selected.update(market_selection_id=digest({k:game.get("pregame",game).get(k) for k in MARKET_FIELDS}),
                previous_market_selection_id=old["forecast_selection"]["market_selection_id"],
                market_approval_reference=approval_reference,
                market_capture=captures[gid]["capture"])
            selected["selection_id"] = digest(selected)
            game["forecast_selection"] = selected
            index["games"][gid] = {**selected,"game_date":day}
            # Grade a known final again using only the corrected market. Actual
            # minutes and forecasts are reused, never recomputed by this repair.
            if isinstance(game.get("final_audit"), dict):
                from nba_lineup_model.web_api.win_projections import _build_final_game_archive
                audit = game["final_audit"]
                game["final_audit"] = _build_final_game_archive(game,
                    pregame=game.get("pregame", game), retrospective=game.get("retrospective") or {},
                    archived_at=timestamp(audit["archived_at"]))
            if forecast_inputs(game.get("pregame", game)) != forecast_inputs(old.get("pregame", old)):
                raise ValueError(f"Market repair changed forecast: {gid}")
            if game.get("prediction_market") != old.get("prediction_market"):
                raise ValueError(f"Market repair changed Polymarket: {gid}")
        path = staged / index["dates"][day]["path"]
        _write(path, board)
        index["market_repair"] = report
        index["dates"][day]["sha256"] = file_sha256(path)
        index["files"] = {str(p.relative_to(staged)): file_sha256(p)
            for p in sorted(staged.rglob("*")) if p.is_file() and p.name != INDEX_FILENAME}
        index.pop("selection_map_sha256", None)
        index["selection_map_sha256"] = digest(index)
        _write(staged / INDEX_FILENAME, index)
        verify_schedule_selection(staged)
        for other, ref in original["dates"].items():
            if other != day and file_sha256(staged / ref["path"]) != ref["sha256"]:
                raise ValueError(f"Market repair changed unapproved date: {other}")
        verify_schedule_selection(source)
        staged.rename(target)
    return report


def main():
    """Prepare an approved market correction using sealed code; no activation."""
    import argparse
    from nba_lineup_model.orchestration.forecast_partition import latest_completed_model
    from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
    from nba_lineup_model.season.production_site_release import verify_production_site_release
    from nba_lineup_model.season.sealed_code_package import verify_sealed_code_package
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("live", "observation-root", "output", "code-package"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--game-date", type=date.fromisoformat, required=True)
    parser.add_argument("--approval-reference", required=True)
    args = parser.parse_args()
    package = args.code_package.resolve(strict=True)
    code = verify_sealed_code_package(package)["identity"]
    if not Path(__file__).resolve().is_relative_to(package / "code"):
        raise ValueError("Run market repairs from the specified sealed code package")
    live = args.live.resolve(strict=True)
    verify_production_site_release(live)
    model = verify_daily_model_artifact(latest_completed_model(live, args.game_date))
    result = repair_market_selection(source=live / "selected-schedule", target=args.output,
        observation_root=args.observation_root, game_date=args.game_date,
        approval_reference=args.approval_reference, model_sha=model["bundle_sha256"], code=code)
    verify_sealed_code_package(package)
    if args.live.resolve(strict=True) != live:
        raise ValueError("Active release changed during market repair")
    print(json.dumps({"status":"prepared_not_published", "output":str(args.output),
        "request_id":result["request_id"], "game_count":len(result["repaired_game_ids"])}))


if __name__ == "__main__":
    main()
