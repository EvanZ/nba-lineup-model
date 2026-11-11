"""Reconstruct each date independently from dated NBA evidence."""

from collections import defaultdict
from datetime import date
from zoneinfo import ZoneInfo

from .inputs import NBA_TEAM_IDS, nba_id, roster_hash

PACIFIC = ZoneInfo("America/Los_Angeles")
ADDITIONS = {"Signing", "Trade", "AwardOnWaivers", "ContractConverted"}


class RosterError(ValueError):
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


def build_rosters(
    *, season: str, day: date, opening: dict,
    transactions: list[dict], boxes: list[dict], inputs: dict,
) -> dict:
    baseline_date = opening["date"]
    teams = opening["teams"]
    if set(teams) != set(NBA_TEAM_IDS) or baseline_date > day:
        raise RosterError("Opening evidence must cover all 30 teams before the selected date.", {})
    owners = {}
    for team, players in teams.items():
        if not players:
            raise RosterError(f"Opening roster is empty for team {team}.", {"team_id": team})
        for raw_id in players:
            pid = nba_id(raw_id)
            if pid in owners:
                raise RosterError("Player appears on two opening rosters.", {"player_id": pid})
            owners[pid] = team

    dates = defaultdict(lambda: {"boxes": [], "transactions": []})
    last_game, last_transaction = {}, {}
    for evidence in boxes:
        if evidence["team_id"] not in teams:
            raise RosterError("Box-score evidence references a non-NBA team.", {"team_id": evidence["team_id"]})
        box_date = evidence["game_time_utc"].astimezone(PACIFIC).date()
        if not baseline_date < box_date <= day:
            raise RosterError("Box evidence is outside the selected roster interval.", {})
        dates[box_date]["boxes"].append(evidence)
        last_game[evidence["team_id"]] = box_date
    for row in transactions:
        reported_date = date.fromisoformat(row["TRANSACTION_DATE"][:10])
        if not baseline_date < reported_date <= day:
            raise RosterError("Transaction is outside the selected roster interval.", {})
        if row["Transaction_Type"] not in ADDITIONS | {"Waive"}:
            raise RosterError("Unsupported NBA transaction type.", {"transaction": row})
        team = nba_id(row["TEAM_ID"])
        if team not in teams:
            raise RosterError("Transaction references a non-NBA team.", {"transaction": row})
        dates[reported_date]["transactions"].append(row)

    # Boxes provide positive membership evidence. Absence never implies a waiver.
    # Date-only transactions apply after same-day game observations for end-of-day reconstruction.
    for effective_date, evidence in sorted(dates.items()):
        seen = {}
        reported_moves = {
            nba_id(row["PLAYER_ID"]) for row in evidence["transactions"]
            if row["Transaction_Type"] in ADDITIONS
        }
        for box in sorted(evidence["boxes"], key=lambda b: (b["game_time_utc"], b["team_id"])):
            team = box["team_id"]
            for raw_id in box["player_ids"]:
                pid = nba_id(raw_id)
                moment = (box["game_time_utc"], pid)
                if moment in seen and seen[moment] != team:
                    raise RosterError("Conflicting simultaneous box-score membership.", {"player_id": pid})
                seen[moment] = team
                current_team = owners.get(pid)
                if current_team not in {None, team} and pid not in reported_moves:
                    raise RosterError(
                        "Box-score membership conflicts with the dated roster.",
                        {"player_id": pid, "roster_team_id": current_team, "box_team_id": team,
                         "game_id": box["game_id"], "game_date_pacific": str(effective_date)},
                    )
                if current_team in {None, team}:
                    owners[pid] = team

        additions = defaultdict(set)
        for row in evidence["transactions"]:
            pid, team = nba_id(row["PLAYER_ID"]), nba_id(row["TEAM_ID"])
            last_transaction[team] = effective_date
            if row["Transaction_Type"] == "Waive":
                if owners.get(pid) == team:
                    del owners[pid]
            else:
                additions[pid].add(team)
        for pid, destinations in sorted(additions.items()):
            if len(destinations) != 1:
                raise RosterError(
                    "Date-only transactions give multiple destination teams.",
                    {"player_id": pid, "transaction_date": str(effective_date), "team_ids": sorted(destinations)},
                )
            previous = owners.get(pid)
            if previous is not None:
                last_transaction[previous] = effective_date
            owners[pid] = next(iter(destinations))

    output, team_metadata = {}, {}
    for team in NBA_TEAM_IDS:
        players = [pid for pid, owner in sorted(owners.items()) if owner == team]
        if not players:
            raise RosterError("Reconstruction produced an empty NBA team.", {"team_id": team})
        evidence_date = max(baseline_date, last_game.get(team, baseline_date), last_transaction.get(team, baseline_date))
        output[str(team)] = players
        team_metadata[str(team)] = {
            "roster_sha256": roster_hash(players),
            "evidence_date_pacific": evidence_date.isoformat(),
            "last_game_date_pacific": last_game[team].isoformat() if team in last_game else None,
            "last_transaction_date_pacific": last_transaction[team].isoformat() if team in last_transaction else None,
        }
    return {
        "schema_version": 1, "season": season, "game_date_pacific": day.isoformat(),
        "reconstruction_rule": "end_of_day_reported_transactions",
        "teams": output, "team_metadata": team_metadata, "inputs": inputs,
    }
