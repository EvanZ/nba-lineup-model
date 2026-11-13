"""Parse official NBA membership IDs and source evidence."""

import json
from hashlib import sha256



def official_json(raw: bytes) -> dict:
    def fields(pairs):
        for key, value in pairs:
            if "espn" in key.casefold() or (
                key.casefold() in {"source", "source_url", "provider"}
                and isinstance(value, str) and "espn" in value.casefold()
            ):
                raise ValueError("Prohibited basketball source in roster evidence.")
        return dict(pairs)
    value = json.loads(raw, object_pairs_hook=fields)
    if not isinstance(value, dict):
        raise ValueError("Roster evidence must be a JSON object.")
    return value


def nba_id(value) -> int:
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    if type(value) not in {int, float} or value <= 0 or (type(value) is float and not value.is_integer()):
        raise ValueError(f"Invalid NBA identity: {value!r}.")
    return int(value)


def roster_hash(player_ids: list[int]) -> str:
    return sha256(json.dumps(sorted(player_ids), separators=(",", ":")).encode()).hexdigest()


def opening_ids(raw: bytes, team_id: int) -> list[int]:
    tables = official_json(raw).get("resultSets", [])
    table = next((t for t in tables if t.get("name", "").casefold() == "commonteamroster"), None)
    if table is None:
        raise ValueError("Opening roster is missing CommonTeamRoster.")
    columns = table["headers"]
    players = set()
    for row in table["rowSet"]:
        team, pid = (row[columns.index(k)] for k in ("TeamID", "PLAYER_ID"))
        pid = nba_id(pid)
        if nba_id(team) != team_id or pid in players:
            raise ValueError(f"Invalid opening roster for team {team_id}.")
        players.add(pid)
    if not players:
        raise ValueError(f"Empty opening roster for team {team_id}.")
    return sorted(players)
