"""Derive date-level 10-day contract terms from saved NBA evidence."""

import re
from datetime import date, timedelta

from .inputs import nba_id
from ...utils import PACIFIC


def ten_day_signings(transactions: list[dict]) -> list[dict]:
    signings = {}
    for row in transactions:
        if row["Transaction_Type"] == "Signing" and re.search(
            r"\b10[\s-]+day\b", row.get("TRANSACTION_DESCRIPTION", ""), re.IGNORECASE,
        ):
            key = (row["TRANSACTION_DATE"][:10], nba_id(row["PLAYER_ID"]), nba_id(row["TEAM_ID"]))
            signings[key] = row
    return [signings[key] for key in sorted(signings)]


def contract_ledger(transactions: list[dict], games: list[dict] | None, day: date) -> list[dict]:
    signings = ten_day_signings(transactions)
    if signings and games is None:
        raise ValueError("10-day contracts require the official NBA schedule.")
    ledger = []
    for row in signings:
        start = date.fromisoformat(row["TRANSACTION_DATE"][:10])
        team = nba_id(row["TEAM_ID"])
        team_games = sorted(
            (game for game in games if team in (game["home_team_id"], game["away_team_id"])
             and start <= game["game_time_utc"].astimezone(PACIFIC).date() <= day),
            key=lambda game: game["game_time_utc"],
        )
        third = team_games[2] if len(team_games) >= 3 else None
        third_date = third["game_time_utc"].astimezone(PACIFIC).date() if third else None
        # The signing date is day one; retain the player through the third game.
        last_date = max(start + timedelta(days=9), third_date) if third_date else None
        ledger.append({
            "player_id": nba_id(row["PLAYER_ID"]), "team_id": team,
            "signed_date": start.isoformat(),
            "third_game_id": third["game_id"] if third else None,
            "third_game_date": third_date.isoformat() if third_date else None,
            "last_active_date": last_date.isoformat() if last_date else None,
            "removal_date": (last_date + timedelta(days=1)).isoformat() if last_date else None,
            "superseded_date": None,
        })
    return ledger
