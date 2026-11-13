"""Reconstruct each date independently from dated NBA evidence."""

from collections import defaultdict
from datetime import date

from .inputs import nba_id, roster_hash
from .contracts import contract_ledger, ten_day_signings

from ...utils import NBA_TEAM_IDS

ADDITIONS = {"Signing", "Trade", "AwardOnWaivers", "ContractConverted"}


class RosterError(ValueError):
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details


def build_rosters(
    *, season: str, day: date, opening: dict,
    transactions: list[dict], inputs: dict, repair_transactions: list[dict] | None = None,
    games: list[dict] | None = None,
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

    dates = defaultdict(list)
    last_transaction = {}
    for row in transactions:
        reported_date = date.fromisoformat(row["TRANSACTION_DATE"][:10])
        if not baseline_date < reported_date <= day:
            raise RosterError("Transaction is outside the selected roster interval.", {})
        if row["Transaction_Type"] not in ADDITIONS | {"Waive"}:
            raise RosterError("Unsupported NBA transaction type.", {"transaction": row})
        team = nba_id(row["TEAM_ID"])
        if team not in teams:
            raise RosterError("Transaction references a non-NBA team.", {"transaction": row})
        dates[reported_date].append(row)

    repair_dates = defaultdict(list)
    for row in repair_transactions or []:
        try:
            if not isinstance(row, dict) or set(row) != {"effective_date", "action", "team_id", "player_id"}:
                raise ValueError("Expected effective_date, action, team_id, and player_id.")
            effective_date = date.fromisoformat(row["effective_date"])
            if effective_date.isoformat() != row["effective_date"] or effective_date < baseline_date:
                raise ValueError("Repair date must be on or after the opening roster date.")
            if row["action"] not in {"add", "remove"} or nba_id(row["team_id"]) not in teams:
                raise ValueError("Expected add/remove and an NBA team ID.")
            nba_id(row["player_id"])
        except (ValueError, TypeError) as error:
            raise RosterError(f"Invalid roster repair transaction: {error}", {"repair": row}) from error
        if effective_date <= day:
            repair_dates[effective_date].append(row)

    ledger = contract_ledger(transactions, games, day)
    contracts = {(entry["player_id"], entry["team_id"], entry["signed_date"]): entry for entry in ledger}
    expiration_dates = defaultdict(list)
    for entry in ledger:
        if entry["removal_date"] and entry["removal_date"] <= day.isoformat():
            expiration_dates[date.fromisoformat(entry["removal_date"])].append(entry)
    active_contracts = {}

    def supersede(pid, effective_date):
        prior = active_contracts.pop(pid, None)
        if prior is not None:
            prior["superseded_date"] = effective_date.isoformat()

    for effective_date in sorted(dates.keys() | repair_dates.keys() | expiration_dates.keys()):
        # Expiration takes effect before any new signing on that date.
        for entry in expiration_dates[effective_date]:
            pid, team = entry["player_id"], entry["team_id"]
            if active_contracts.get(pid) is entry:
                if owners.get(pid) == team:
                    del owners[pid]
                    last_transaction[team] = effective_date
                del active_contracts[pid]
        rows = dates[effective_date]
        additions = defaultdict(set)
        for row in rows:
            pid, team = nba_id(row["PLAYER_ID"]), nba_id(row["TEAM_ID"])
            last_transaction[team] = effective_date
            if row["Transaction_Type"] == "Waive":
                if owners.get(pid) == team:
                    del owners[pid]
                    supersede(pid, effective_date)
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
            supersede(pid, effective_date)
            owners[pid] = next(iter(destinations))
            contract = contracts.get((pid, owners[pid], effective_date.isoformat()))
            # A conversion or ordinary signing on the same date replaces the short term.
            other_moves = [row for row in rows if nba_id(row["PLAYER_ID"]) == pid
                           and row["Transaction_Type"] in ADDITIONS and not ten_day_signings([row])]
            if contract is not None and not other_moves:
                active_contracts[pid] = contract
        # Repairs follow official moves on their date, in operator file order.
        for row in repair_dates[effective_date]:
            pid, team = nba_id(row["player_id"]), nba_id(row["team_id"])
            last_transaction[team] = effective_date
            if row["action"] == "remove":
                if owners.get(pid) == team:
                    del owners[pid]
                    supersede(pid, effective_date)
            else:
                supersede(pid, effective_date)
                previous = owners.get(pid)
                if previous is not None:
                    last_transaction[previous] = effective_date
                owners[pid] = team

    output, team_metadata = {}, {}
    for team in NBA_TEAM_IDS:
        players = [pid for pid, owner in sorted(owners.items()) if owner == team]
        if not players:
            raise RosterError("Reconstruction produced an empty NBA team.", {"team_id": team})
        evidence_date = max(baseline_date, last_transaction.get(team, baseline_date))
        output[str(team)] = players
        team_metadata[str(team)] = {
            "roster_sha256": roster_hash(players),
            "evidence_date_pacific": evidence_date.isoformat(),
            "last_transaction_date_pacific": last_transaction[team].isoformat() if team in last_transaction else None,
        }
    return {
        "schema_version": 1, "season": season, "game_date_pacific": day.isoformat(),
        "reconstruction_rule": "end_of_day_reported_transactions",
        "teams": output, "team_metadata": team_metadata, "inputs": inputs,
        "ten_day_contracts": ledger,
    }
