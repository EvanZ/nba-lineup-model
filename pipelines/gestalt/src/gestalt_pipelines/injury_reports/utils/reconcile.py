"""Match extracted injury rows to one official NBA slate and its rosters."""

import re
import unicodedata
from datetime import datetime

RESOLVER_VERSION = "v4"


def normalize_player_name(value: str) -> str:
    """Use the daily model's suffix-insensitive name key."""
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    suffix = r"(?:Jr\.?|Sr\.?|II|III|IV|V)"
    text = re.sub(rf"^{suffix}\s*,\s*", "", text, flags=re.I)
    text = re.sub(rf"(?:\s*,\s*|\s+){suffix}$", "", text, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def reconcile_games(
    games: list[dict], rosters: dict[int, dict[int, str]], report: dict,
    repairs: dict[str, int] | None = None,
) -> dict:
    published_at = datetime.fromisoformat(report["published_at"])
    by_game = {}
    repairs = repairs or {}
    for game in games:
        start = game["game_time_utc"]
        home, away = game["home_team_tricode"], game["away_team_tricode"]
        matchup = f"{away}@{home}"
        reported_date = game["game_date"].isoformat()
        report_key = f"{reported_date}|{matchup}"
        eligible = published_at < start and report_key in report["game_team_reports"]
        submitted = report["game_team_reports"][report_key] if eligible else {}
        teams = {}

        for team_id, tricode in ((game["away_team_id"], away), (game["home_team_id"], home)):
            candidates: dict[str, set[int]] = {}
            for player_id, name in rosters[team_id].items():
                key = normalize_player_name(name)
                if key:
                    candidates.setdefault(key, set()).add(player_id)

            players = {}
            unresolved = []
            for index, row in enumerate(report["players"] if eligible else []):
                if (row["game_date"], row["matchup"], row["team"]) != (reported_date, matchup, tricode):
                    continue
                source = {**row, "source_row_index": index, "pdf_uri": report["pdf_uri"]}
                reported_name = row["player_name"]
                matches = candidates.get(normalize_player_name(reported_name), set())
                repair = repairs.get(reported_name)
                if repair is not None and matches and repair not in matches:
                    unresolved.append({**source, "identity_status": "conflicting_repair"})
                    continue
                if len(matches) == 1:
                    player_id = str(next(iter(matches)))
                    source["identity_source"] = "roster"
                elif repair is not None:
                    player_id = str(repair)
                    source["identity_source"] = "repair_dictionary"
                else:
                    unresolved.append({**source, "identity_status": "ambiguous" if matches else "unmatched"})
                    continue
                if player_id in players:
                    raise ValueError(f"Multiple injury rows match player {player_id} in game {game['game_id']}.")
                players[player_id] = source

            submission_status = submitted.get(tricode, {}).get("status", "missing")
            if submission_status not in {"submitted", "not_yet_submitted", "missing"}:
                raise ValueError(f"Unsupported submission status for {tricode} in {report_key}.")
            teams[str(team_id)] = {
                "team": tricode,
                "submission_status": submission_status,
                "players": players,
                "unresolved_players": unresolved,
                "complete": submission_status == "submitted" and not unresolved,
            }

        by_game[game["game_id"]] = {
            "report_game_key": report_key,
            "scheduled_start_utc": start.isoformat(),
            "report_used": eligible,
            "teams": teams,
            "complete": all(team["complete"] for team in teams.values()),
        }

    return by_game
