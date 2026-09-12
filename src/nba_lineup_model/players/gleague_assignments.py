"""Normalize official G League assignment and recall transactions.

The official G League transaction feed exposes NBA-compatible player IDs from
2021 onward.  It does not include the NBA parent team, so this module keeps a
small, explicit affiliate map and produces only assignment intervals whose
recall endpoint is observed in the same feed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

OFFICIAL_GLEAGUE_TRANSACTIONS_URL = (
    "https://cdn-gleague.nba.com/static/json/staticData/GLeagueTransactions.json"
)
OFFICIAL_GLEAGUE_SOURCE = "official_gleague_transactions"

# The feed's TEAM_ID denotes the G League affiliate. These relationships are
# stable for the feed's 2021-present coverage; teams without an NBA parent are
# deliberately absent rather than inferred.
GLEAGUE_AFFILIATE_PARENT_TEAM_IDS: dict[int, int] = {
    1612709889: 1610612760,  # Oklahoma City Blue -> Oklahoma City Thunder
    1612709890: 1610612759,  # Austin Spurs -> San Antonio Spurs
    1612709893: 1610612739,  # Cleveland Charge -> Cleveland Cavaliers
    1612709902: 1610612744,  # Santa Cruz Warriors -> Golden State Warriors
    1612709903: 1610612762,  # Salt Lake City Stars -> Utah Jazz
    1612709904: 1610612748,  # Sioux Falls Skyforce -> Miami Heat
    1612709905: 1610612747,  # South Bay Lakers -> Los Angeles Lakers
    1612709908: 1610612745,  # Rio Grande Valley Vipers -> Houston Rockets
    1612709909: 1610612755,  # Delaware Blue Coats -> Philadelphia 76ers
    1612709910: 1610612754,  # Fort Wayne / Indiana Mad Ants -> Indiana Pacers
    1612709911: 1610612750,  # Iowa Wolves -> Minnesota Timberwolves
    1612709913: 1610612740,  # Birmingham Squadron -> New Orleans Pelicans
    1612709914: 1610612758,  # Stockton Kings -> Sacramento Kings
    1612709915: 1610612738,  # Maine Celtics -> Boston Celtics
    1612709917: 1610612743,  # Grand Rapids Gold -> Denver Nuggets
    1612709918: 1610612742,  # Texas Legends -> Dallas Mavericks
    1612709919: 1610612752,  # Westchester Knicks -> New York Knicks
    1612709920: 1610612761,  # Raptors 905 -> Toronto Raptors
    1612709921: 1610612751,  # Long Island Nets -> Brooklyn Nets
    1612709922: 1610612766,  # Greensboro Swarm -> Charlotte Hornets
    1612709923: 1610612741,  # Windy City Bulls -> Chicago Bulls
    1612709924: 1610612746,  # Agua Caliente / Ontario Clippers -> LA Clippers
    1612709925: 1610612753,  # Lakeland / Osceola Magic -> Orlando Magic
    1612709926: 1610612763,  # Memphis Hustle -> Memphis Grizzlies
    1612709927: 1610612749,  # Wisconsin Herd -> Milwaukee Bucks
    1612709928: 1610612764,  # Capital City Go-Go -> Washington Wizards
    1612709929: 1610612737,  # College Park Skyhawks -> Atlanta Hawks
    1612709932: 1610612765,  # Motor City Cruise -> Detroit Pistons
    1612709933: 1610612757,  # Rip City Remix -> Portland Trail Blazers
    1612709934: 1610612756,  # Northern Arizona / Valley Suns -> Phoenix Suns
}

_REQUIRED_COLUMNS = frozenset(
    {
        "PLAYER_ID",
        "TEAM_ID",
        "TEAM_SLUG",
        "TRANSACTION_DATE",
        "TRANSACTION_DESCRIPTION",
    }
)


def load_official_gleague_assignment_intervals(path: Path | str) -> pd.DataFrame:
    """Return official assignment intervals paired to their next same-team recall.

    An unpaired assignment remains in the returned audit table with a missing
    recall date, but is not eligible to change an availability label. This
    prevents an incomplete source extract from creating an unbounded interval.
    """

    source_path = Path(path)
    payload = json.loads(source_path.read_text())
    rows = _transaction_rows(payload)
    transactions = pd.DataFrame(rows)
    missing = _REQUIRED_COLUMNS - set(transactions)
    if missing:
        raise ValueError(
            "Official G League transactions are missing required columns: "
            f"{sorted(missing)}"
        )

    transactions = transactions.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    transactions["PLAYER_ID"] = pd.to_numeric(transactions["PLAYER_ID"], errors="coerce")
    transactions["TEAM_ID"] = pd.to_numeric(transactions["TEAM_ID"], errors="coerce")
    transactions["transaction_date"] = pd.to_datetime(
        transactions.pop("TRANSACTION_DATE"), errors="coerce", utc=True
    ).dt.normalize()
    transactions["TRANSACTION_DESCRIPTION"] = transactions[
        "TRANSACTION_DESCRIPTION"
    ].astype("string")
    transactions = transactions.dropna(
        subset=["PLAYER_ID", "TEAM_ID", "transaction_date"]
    ).copy()
    transactions["PLAYER_ID"] = transactions["PLAYER_ID"].astype(int)
    transactions["TEAM_ID"] = transactions["TEAM_ID"].astype(int)

    assignments = transactions.loc[
        transactions["TRANSACTION_DESCRIPTION"].eq("Assigned")
    ].copy()
    recalls = transactions.loc[
        transactions["TRANSACTION_DESCRIPTION"].eq("Recalled")
    ].copy()

    records: list[dict[str, Any]] = []
    for (player_id, gleague_team_id), player_assignments in assignments.groupby(
        ["PLAYER_ID", "TEAM_ID"], sort=False
    ):
        recall_dates = recalls.loc[
            (recalls["PLAYER_ID"] == player_id) & (recalls["TEAM_ID"] == gleague_team_id),
            "transaction_date",
        ].sort_values(kind="stable")
        sorted_assignments = player_assignments.sort_values(
            "transaction_date", kind="stable"
        )
        for assignment in sorted_assignments.itertuples(index=False):
            later_recalls = recall_dates.loc[recall_dates.ge(assignment.transaction_date)]
            records.append(
                {
                    "player_id": int(player_id),
                    "gleague_team_id": int(gleague_team_id),
                    "gleague_team": str(assignment.TEAM_SLUG),
                    "nba_parent_team_id": GLEAGUE_AFFILIATE_PARENT_TEAM_IDS.get(
                        int(gleague_team_id)
                    ),
                    "assignment_date": assignment.transaction_date,
                    "recall_date": later_recalls.iloc[0] if not later_recalls.empty else pd.NaT,
                    "assignment_source": OFFICIAL_GLEAGUE_SOURCE,
                    "assignment_source_url": OFFICIAL_GLEAGUE_TRANSACTIONS_URL,
                }
            )

    columns = [
        "player_id",
        "gleague_team_id",
        "gleague_team",
        "nba_parent_team_id",
        "assignment_date",
        "recall_date",
        "assignment_source",
        "assignment_source_url",
    ]
    output = pd.DataFrame(records, columns=columns)
    if output.empty:
        return output
    output["nba_parent_team_id"] = output["nba_parent_team_id"].astype("Int64")
    return output.sort_values(
        ["player_id", "gleague_team_id", "assignment_date"], kind="stable"
    ).reset_index(drop=True)


def _transaction_rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping) and isinstance(payload.get("list"), list):
        return [row for row in payload["list"] if isinstance(row, Mapping)]
    raise ValueError("Official G League transactions must be a list or a mapping with 'list'")
