import json
from datetime import UTC, date, datetime
from hashlib import sha256

from gestalt_pipelines.minutes.utils import build_priors

SEASON = "2026-27"
FIRST = date(2026, 10, 20)
HOME, AWAY = 1610612738, 1610612755


def reference(name="source"):
    return {"uri": f"s3://nba-gestalt-data/fixture/{name}.json", "sha256": sha256(name.encode()).hexdigest()}


def slate(day=FIRST, gid="0022600001"):
    teams = {str(t): [t] for t in range(1610612737, 1610612767)}
    teams[str(HOME)] = [1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19]
    teams[str(AWAY)] = [8, 9, 10, 11, 12, 13, 14]
    game = {
        "game_id": gid, "game_date": day, "game_time_utc": datetime(day.year, day.month, day.day, 23, tzinfo=UTC),
        "home_team_id": HOME, "away_team_id": AWAY,
        "home_team_tricode": "BOS", "away_team_tricode": "PHI",
    }
    def team(tid, ids, tri):
        values = [44, 48, 48, 48, 26, 22, 4]
        return {"teamId": tid, "teamTricode": tri, "players": [
            {"personId": pid, "name": f"Player {pid}", "status": "ACTIVE",
             "statistics": {"minutes": f"PT{minutes:02d}M00.00S"}}
            for pid, minutes in zip(ids, values)
        ]}
    home = team(HOME, list(range(1, 8)), "BOS")
    home["players"] += [
        {"personId": pid, "name": f"Player {pid}", "status": "ACTIVE",
         "statistics": {"minutes": "PT00M00.00S"}} for pid in (15, 17)
    ]
    box = {"game": {"gameId": gid, "gameStatus": 3, "period": 4,
                    "homeTeam": home, "awayTeam": team(AWAY, list(range(8, 15)), "PHI")}}
    reports = {
        str(HOME): {"team": "BOS", "submission_status": "submitted", "unresolved_players": [], "players": {
            "15": {"status": "Available", "reason": "Injury/Illness - Illness"},
            "16": {"status": "Out", "reason": "G League - Two- Way"},
            "17": {"status": "Out", "reason": "Injury/Illness - Knee"},
            "18": {"status": "Questionable", "reason": "Injury/Illness - Ankle"},
        }},
        str(AWAY): {"team": "PHI", "submission_status": "submitted", "unresolved_players": [], "players": {}},
    }
    lineups = [
        (0, 1320, [1, 2, 3, 4, 5], [8, 9, 10, 11, 12]),
        (1320, 2640, [1, 2, 3, 4, 6], [8, 9, 10, 11, 13]),
        (2640, 2880, [2, 3, 4, 5, 7], [9, 10, 11, 12, 14]),
    ]
    stints = {
        "season": SEASON, "game_date_pacific": str(day),
        "validation": {"complete": True, "expected_game_ids": [gid], "processed_game_ids": [gid]},
        "rows": [{
            "game_id": gid, "stint_index": i, "home_team_id": HOME, "away_team_id": AWAY,
            "start_elapsed_game_seconds": start, "end_elapsed_game_seconds": end,
            "home_player_ids": home_ids, "away_player_ids": away_ids,
        } for i, (start, end, home_ids, away_ids) in enumerate(lineups)],
    }
    return {
        "day": day, "first_date": FIRST, "season": SEASON, "games": [game],
        "rosters": {"season": SEASON, "game_date_pacific": str(day), "teams": teams},
        "injuries": {"season": SEASON, "game_date_pacific": str(day), "cutoff_source": "nba_scheduled_start",
                     "games": {gid: {"teams": reports, "report_used": True,
                                     "scheduled_start_utc": game["game_time_utc"].isoformat()}}},
        "stints": stints, "boxes": {gid: box},
        "inputs": {
            "priors": reference("priors"), "parent": None, "schedule": reference("schedule"),
            "stints": reference("stints"), "rosters": reference("rosters"),
            "injuries": reference("injuries"), "boxes": {gid: reference("box")},
        },
    }


def priors():
    data = slate()
    rows = [{
        "player_id": pid, "player_name": f"Player {pid}",
        "conditional_minutes_per_game": 4 if pid in (7, 14) else (10 if pid == 6 else 30),
        "has_prior_minutes_state": pid != 6, "uses_draft_cold_start_prior": pid == 6,
    } for ids in data["rosters"]["teams"].values() for pid in ids]
    frozen = json.dumps({"minutes": {
        "season": SEASON, "completed_season": "2025-26", "players": rows,
    }}).encode()
    history = [{
        "player_id": 999, "player_name": "Historical player", "season": "2024-25",
        "season_start_year": 2024, "actual_minutes_per_available_game": 9,
    }]
    return build_priors(frozen, history, season=SEASON,
                        history_reference=reference("history"), draft_reference=reference("draft"))
