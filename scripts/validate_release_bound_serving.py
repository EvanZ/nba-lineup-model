"""Capture/compare every affected dev serving response without changing artifacts."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def fetch(root, path):
    with urllib.request.urlopen(root.rstrip("/") + path, timeout=90) as response:
        return json.load(response)


def capture(api, schedule_root, output):
    output.mkdir(parents=True, exist_ok=False)
    endpoints = {
        "health": "/api/health",
        "rankings": "/api/rankings?season=2026-27",
        "lineups": "/api/lineups?season=2026-27&minimum_possessions=0",
        "wins": "/api/win-projections",
        "records": "/api/current-season-records",
        "performance": "/api/schedule-performance",
    }
    responses = {}
    for name, path in endpoints.items():
        responses[name] = fetch(api, path)
        print(name, flush=True)
    players = responses["rankings"]["players"]
    for index, player in enumerate(players):
        player_id = player["player_id"]
        responses[f"player-{player_id}"] = fetch(api, f"/api/players/{player_id}")
        if index % 100 == 0:
            print(f"players {index + 1}/{len(players)}", flush=True)
    for team in sorted({str(p["team"]) for p in players if p.get("team") and p["team"] != "FA"}):
        responses[f"team-{team}"] = fetch(api, f"/api/teams/{team}?season=2026-27")
    for path in sorted((schedule_root / "latest").glob("*.json")):
        responses[f"schedule-{path.stem}"] = fetch(api, f"/api/schedule?date={path.stem}")
    for name, payload in responses.items():
        (output / f"{name}.json").write_text(json.dumps(payload, sort_keys=True))
    print(
        json.dumps(
            {
                "responses": len(responses),
                "players": len(players),
                "schedule_days": len(list(output.glob("schedule-*.json"))),
            }
        ),
        flush=True,
    )


def normalized(value):
    if isinstance(value, dict):
        # The new response also exposes the already-selected nested provenance
        # at the game level. Compare the same meaning, never drop model fields.
        if "game_id" in value and isinstance(value.get("pregame"), dict):
            value = dict(value)
            if "forecast_version" not in value and "forecast_version" in value["pregame"]:
                value["forecast_version"] = value["pregame"]["forecast_version"]
        return {
            k: normalized(v)
            for k, v in value.items()
            if k
            not in {
                "forecast_selection",
                "serving_selection",
                "serving_view",
                "loaded_release",
                "site_release",
                "latest_observations",
                "observation_id",
                "live_score_collected_at",
            }
        }
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def compare(before, after):
    errors = []
    files = sorted(before.glob("*.json"))
    for path in files:
        target = after / path.name
        if not target.is_file():
            errors.append({"file": path.name, "error": "missing"})
            continue
        left, right = (
            normalized(json.loads(path.read_text())),
            normalized(json.loads(target.read_text())),
        )
        if left != right:
            errors.append({"file": path.name, "error": "changed"})
    result = {"checked_responses": len(files), "mismatches": errors}
    print(json.dumps(result, indent=2))
    (after / "comparison.json").write_text(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8011")
    parser.add_argument("--schedule-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.output)
    else:
        capture(args.api, args.schedule_root, args.output)
