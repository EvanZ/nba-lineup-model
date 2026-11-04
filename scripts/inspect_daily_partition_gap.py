"""Inspect a selected release and propose a missing-day repair. Never rebuild."""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nba_lineup_model.orchestration.daily_work import atomic_json, digest, file_reference, read_object
from nba_lineup_model.orchestration.repair_plan import make_impact_plan, write_impact_plan
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import verify_daily_model_artifact
from nba_lineup_model.season.slate_time import PACIFIC
from nba_lineup_model.season.source_policy import require_clean_artifact_sources
from nba_lineup_model.season.serving_selection import verify_schedule_selection


def inspect(live: Path, through: date, coverage_path: Path, output: Path):
    release = live.resolve(strict=True)
    runtime = read_object(release / "runtime-manifest.json")
    approved = runtime["approved_manifest"]["identity"]
    bindings = read_object(release / "serving-bindings.json")
    evidence = date.fromisoformat(approved["evidence_through_date"])
    if through <= evidence:
        raise ValueError("This inspector proposes missing-day advancement, not historical replacement")
    data = verify_daily_data_artifact(approved["data"]["bundle_path"])
    model = verify_daily_model_artifact(approved["model"]["bundle_path"],
                                       data_bundle=approved["data"]["bundle_path"])
    if data["bundle_sha256"] != approved["data"]["bundle_sha256"] or model["bundle_sha256"] != approved["model"]["bundle_sha256"]:
        raise ValueError("Selected release artifact identities disagree")
    schedule = (release / "schedule").resolve()
    require_clean_artifact_sources(schedule)
    selections = verify_schedule_selection(schedule)
    frame = pd.read_parquet(release / "catalog/games.parquet")
    frame = frame.loc[frame.season.astype(str).eq(approved["season"]) & frame.season_type.eq("regular")].copy()
    frame["pacific_date"] = pd.to_datetime(frame.game_time_utc, utc=True).dt.tz_convert(PACIFIC).dt.date
    coverage = read_object(coverage_path)
    games, details, changes, days = {}, [], [], []
    coverage_snapshot = output / "inputs" / f"coverage-{digest(coverage)}.json"
    atomic_json(coverage_snapshot, coverage, immutable=True)
    changes.extend([
        {"before":None, "after":file_reference(coverage_snapshot),
         "reason":"Pinned existing play-by-play capture selection; no source download"},
        {"before":None, "after":file_reference(schedule / "forecast-selections.json"),
         "reason":"Existing selected forecasts and lines to preserve unchanged"},
    ])
    current = evidence + timedelta(days=1)
    while current <= through:
        day = current.isoformat()
        days.append(day)
        rows = frame.loc[frame.pacific_date.eq(current)]
        games[day] = sorted(rows.game_id.astype(str))
        for row in rows.itertuples():
            gid = str(row.game_id)
            boxpath = schedule / "final" / f"{gid}.nba.json"
            selected = selections["games"].get(gid, {})
            record = {"game_date":day,"game_id":gid,
                      "matchup":f"{row.away_team_tricode}-{row.home_team_tricode}",
                      "official_final_box":boxpath.is_file(),
                      "selected_forecast":bool(selected.get("forecast_id")),
                      "play_by_play_capture":bool(coverage.get("games",{}).get(gid))}
            if boxpath.is_file():
                event = read_object(boxpath).get("game",{})
                if str(event.get("gameId")) != gid or int(event.get("gameStatus") or 0) != 3:
                    raise ValueError(f"{gid}: official box is not a matching final")
                record["box_player_rows"] = sum(len(event.get(side,{}).get("players",[]))
                                                for side in ("homeTeam","awayTeam"))
                changes.append({"before":None,"after":file_reference(boxpath),
                                "reason":"Existing NBA final capture for a missing daily partition"})
            details.append(record)
        current += timedelta(days=1)
    plan = make_impact_plan(
        season=approved["season"], classification="missing_daily_update", changed_inputs=changes,
        affected_games=games,
        affected_artifacts=["daily observed data","coupled ratings/minutes/context",
                            "player logs and trajectories","rankings","teams","lineups","Wins",
                            "release bindings"],
        model_suffix=days, forecast_replacement=False,
        base_release_id=runtime.get("runtime_sha256") or runtime["release_sha256"],
        edition_id=bindings["edition_id"],
        source_coverage={
            "Official final box scores":f"{sum(r['official_final_box'] for r in details)} / {len(details)}",
            "Selected historical forecasts":f"{sum(r['selected_forecast'] for r in details)} / {len(details)}",
            "Play-by-play capture references":f"{sum(r['play_by_play_capture'] for r in details)} / {len(details)}",
            "New daily data/model validation":"Not yet run",
        },
        explanation=(f"The selected model ends on {evidence}. Build only the missing daily "
                     f"partitions through {through}, reusing the validated prior model state. "
                     "Preserve all selected pregame forecasts and market lines. Source capture "
                     "presence is not proof of successful reconstruction; each day must validate "
                     "before the next advances. No work in this plan has been executed."),
    )
    report = write_impact_plan(output, plan)
    inventory = {"base_evidence_date":str(evidence),"proposed_through_date":str(through),
                 "base_model_game_count":model["identity"]["cumulative_game_count"],
                 "proposed_added_game_count":sum(map(len,games.values())),
                 "games":details,"model_provenance":model["identity"]["model_provenance"],
                 "plan_id":plan["plan_id"],"status":"awaiting_scope_approval",
                 "not_yet_validated":["stint reconstruction","inactive/opportunity completeness",
                                      "new daily model outputs","API/browser advancement"]}
    atomic_json(report.parent / "inventory.json", inventory, immutable=True)
    return report, inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live",type=Path,required=True)
    parser.add_argument("--through",type=date.fromisoformat,required=True)
    parser.add_argument("--coverage",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report, inventory=inspect(args.live,args.through,args.coverage,args.output)
    print(json.dumps({"report":str(report),**inventory},indent=2))


if __name__ == "__main__":
    main()
