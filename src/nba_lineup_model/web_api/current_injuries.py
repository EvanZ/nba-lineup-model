"""Current-season injury availability overrides for Win Projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

INJURY_AVAILABILITY_AS_OF = "2026-10-20"
REGULAR_SEASON_GAMES = 82.0
REGULATION_TEAM_MINUTES = 240.0


@dataclass(frozen=True)
class InjuryGamesMissedRange:
    """Auditable estimate used to override full-season availability."""

    player_name: str
    low: float
    high: float
    source_url: str

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


CURRENT_INJURY_GAMES_MISSED: dict[int, InjuryGamesMissedRange] = {
    1628978: InjuryGamesMissedRange(
        "Donte DiVincenzo",
        55.0,
        70.0,
        "https://www.nba.com/timberwolves/player/1628978/donte-divincenzo",
    ),
    1641710: InjuryGamesMissedRange(
        "Anthony Black", 0.0, 2.0, "https://www.nba.com/player/1641710"
    ),
    1642851: InjuryGamesMissedRange(
        "Kon Knueppel",
        1.0,
        4.0,
        "https://www.nba.com/news/hornets-kon-knueppel-injures-his-left-hamstring-and-could-miss-the-start-of-the-season",
    ),
    202699: InjuryGamesMissedRange(
        "Tobias Harris", 1.0, 4.0, "https://www.nba.com/clippers/player/202699/tobias-harris"
    ),
    1630583: InjuryGamesMissedRange(
        "Santi Aldama",
        0.0,
        5.0,
        "https://dallassportsjournal.com/santi-aldama-injury-update-mavericks-preseason-finale/",
    ),
    202710: InjuryGamesMissedRange(
        "Jimmy Butler III",
        40.0,
        55.0,
        "https://sports.yahoo.com/articles/jimmy-butler-could-return-warriors-163905835.html",
    ),
    1629632: InjuryGamesMissedRange(
        "Coby White", 1.0, 6.0, "https://www.nba.com/hornets/player/1629632/coby-white"
    ),
    1629014: InjuryGamesMissedRange(
        "Anfernee Simons",
        0.0,
        1.0,
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-20_08_15AM.pdf",
    ),
    1630541: InjuryGamesMissedRange(
        "Moses Moody",
        27.0,
        42.0,
        "https://www.rotowire.com/basketball/headlines/moses-moody-injury-could-return-by-december-534700",
    ),
    1628370: InjuryGamesMissedRange(
        "Malik Monk", 1.0, 6.0, "https://www.nba.com/kings/player/1628370/malik-monk"
    ),
    204001: InjuryGamesMissedRange(
        "Kristaps Porziņģis",
        10.0,
        82.0,
        "https://www.nba.com/news/warriors-kristaps-porzingis-out-indefinitely",
    ),
    1631230: InjuryGamesMissedRange(
        "Dominick Barlow",
        6.0,
        10.0,
        "https://www.inquirer.com/sixers/76ers-injury-report-dominick-barlow-joel-embiid-dean-wade-anfernee-simons-20261019.html",
    ),
    1629684: InjuryGamesMissedRange(
        "Grant Williams",
        2.0,
        8.0,
        "https://www.nba.com/news/hornets-kon-knueppel-injures-his-left-hamstring-and-could-miss-the-start-of-the-season",
    ),
    1641737: InjuryGamesMissedRange(
        "Adem Bona",
        0.0,
        1.0,
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-20_08_15AM.pdf",
    ),
    1641726: InjuryGamesMissedRange(
        "Dereck Lively II",
        25.0,
        45.0,
        "https://ca.sports.yahoo.com/news/mavericks-center-dereck-lively-ii-not-cleared-for-training-camp-9-months-after-foot-surgery-163336250.html",
    ),
    1630696: InjuryGamesMissedRange(
        "Dru Smith",
        0.0,
        3.0,
        "https://www.cbssports.com/fantasy/basketball/news/heats-dru-smith-remains-out-friday/",
    ),
    1631243: InjuryGamesMissedRange(
        "Mouhamed Gueye",
        10.0,
        20.0,
        "https://www.nba.com/news/mouhamed-gueye-surgery-could-miss-start-next-season",
    ),
    1643538: InjuryGamesMissedRange(
        "Joshua Jefferson",
        0.0,
        0.0,
        "https://sports.yahoo.com/articles/nets-first-round-pick-cleared-223210490.html",
    ),
    1643519: InjuryGamesMissedRange(
        "Jayden Quaintance",
        35.0,
        55.0,
        "https://aws-prod-web9.rotowire.com/basketball/headlines/jayden-quaintance-injury-still-without-timetable-for-return-533226",
    ),
}


def apply_current_injury_availability(
    minutes_payload: dict[str, object],
    *,
    estimates: Mapping[int, InjuryGamesMissedRange] = CURRENT_INJURY_GAMES_MISSED,
    as_of: str = INJURY_AVAILABILITY_AS_OF,
) -> dict[str, object]:
    """Replace availability with researched games-missed midpoints and renormalize.

    The midpoint is treated as the player's total expected regular-season games
    missed. No separate post-return minutes ramp is applied.
    """

    players = pd.DataFrame(minutes_payload.get("players", []))
    required = {
        "player_id",
        "player_name",
        "team",
        "availability_probability",
        "conditional_minutes_per_game",
    }
    missing = sorted(required - set(players))
    if missing:
        raise ValueError("Minutes payload lacks injury-adjustment fields: " + ", ".join(missing))
    if players.empty or players["player_id"].duplicated().any():
        raise ValueError("Minutes payload must contain unique rostered players")

    players = players.copy()
    players["player_id"] = pd.to_numeric(players["player_id"], errors="raise").astype(int)
    players["model_availability_probability"] = pd.to_numeric(
        players["availability_probability"], errors="raise"
    )
    players["injury_adjusted"] = False
    players["injury_expected_games_missed"] = 0.0

    roster_ids = set(players["player_id"])
    unknown_ids = sorted(set(estimates) - roster_ids)
    if unknown_ids:
        raise ValueError(f"Injury estimates contain players outside the roster: {unknown_ids}")

    metadata_rows: list[dict[str, object]] = []
    for player_id, estimate in estimates.items():
        if not 0.0 <= estimate.low <= estimate.high <= REGULAR_SEASON_GAMES:
            raise ValueError(f"Invalid games-missed range for {estimate.player_name}")
        player_index = players.index[players["player_id"].eq(int(player_id))]
        player_name = str(players.loc[player_index[0], "player_name"])
        if player_name != estimate.player_name:
            raise ValueError(
                f"Injury estimate name mismatch for {player_id}: "
                f"{estimate.player_name!r} != {player_name!r}"
            )
        midpoint = estimate.midpoint
        availability = (REGULAR_SEASON_GAMES - midpoint) / REGULAR_SEASON_GAMES
        players.loc[player_index, "availability_probability"] = availability
        players.loc[player_index, "injury_adjusted"] = True
        players.loc[player_index, "injury_expected_games_missed"] = midpoint
        metadata_rows.append(
            {
                "player_id": int(player_id),
                **asdict(estimate),
                "midpoint": midpoint,
                "availability_probability": availability,
            }
        )

    players["raw_expected_total_minutes"] = (
        REGULAR_SEASON_GAMES
        * pd.to_numeric(players["availability_probability"], errors="raise")
        * pd.to_numeric(players["conditional_minutes_per_game"], errors="raise")
    )
    rotation_size = int(minutes_payload.get("initial_rotation_size", 15))
    if rotation_size <= 0:
        raise ValueError("Minutes payload initial rotation size must be positive")
    players["is_baseline_rotation_candidate"] = False
    for _, team in players.groupby("team", sort=False):
        selected = team.sort_values(
            ["raw_expected_total_minutes", "player_name", "player_id"],
            ascending=[False, True, True],
            kind="stable",
        ).head(rotation_size).index
        players.loc[selected, "is_baseline_rotation_candidate"] = True
    players.loc[
        ~players["is_baseline_rotation_candidate"], "raw_expected_total_minutes"
    ] = 0.0
    team_totals = players.groupby("team", sort=False)["raw_expected_total_minutes"].transform(
        "sum"
    )
    if not np.isfinite(team_totals).all() or team_totals.le(0.0).any():
        raise ValueError("Injury adjustment produced a team without projected minutes")
    players["baseline_minute_share"] = players["raw_expected_total_minutes"] / team_totals
    players["baseline_minutes_per_game"] = (
        REGULATION_TEAM_MINUTES * players["baseline_minute_share"]
    )
    projected_team_minutes = players.groupby("team")["baseline_minutes_per_game"].sum()
    if not np.allclose(projected_team_minutes, REGULATION_TEAM_MINUTES):
        raise ValueError("Injury-adjusted allocations do not conserve team minutes")

    players = players.sort_values(
        ["team", "baseline_minutes_per_game", "player_name", "player_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    contract = str(minutes_payload.get("contract", "")).rstrip()
    injury_contract = (
        f" Current injury availability as of {as_of} replaces the model availability "
        "for researched players with (82 minus the midpoint expected games missed) / 82; "
        "no post-return minutes ramp is applied. Team minutes are then renormalized to 240."
    )
    return {
        **{key: value for key, value in minutes_payload.items() if key != "players"},
        "model": f"{minutes_payload.get('model', 'Minutes projection')} + Injury Midpoints",
        "contract": contract + injury_contract,
        "injury_availability": {
            "as_of": as_of,
            "method": "midpoint expected games missed; no post-return ramp",
            "player_count": len(metadata_rows),
            "players": sorted(
                metadata_rows,
                key=lambda row: (str(row["player_name"]), int(row["player_id"])),
            ),
        },
        "players": players.to_dict(orient="records"),
    }
