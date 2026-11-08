"""Parse the official NBA injury report PDF layout."""

import io
import re
from datetime import datetime

from pypdf import PdfReader

PARSER_VERSION = "v2"
REPORTED_STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
NBA_TEAM_ALIASES = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UT": "UTA",
    "UTAH": "UTA",
    "WSH": "WAS",
}
TEAM_DISPLAY_NAMES: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NJN": "New Jersey Nets",
    "NOH": "New Orleans Hornets",
    "NOK": "New Orleans/Oklahoma City Hornets",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "SEA": "Seattle SuperSonics",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "VAN": "Vancouver Grizzlies",
    "WAS": "Washington Wizards",
}


def parse_pdf(pdf: bytes) -> dict[str, object]:
    """Extract player rows and team submissions from the official PDF layout."""

    statuses: list[dict[str, object]] = []
    team_reports: dict[str, dict[str, object]] = {}
    game_team_reports: dict[str, dict[str, dict[str, object]]] = {}
    current_game_date: str | None = None
    current_matchup: tuple[str, str] | None = None
    current_team: str | None = None
    for page in PdfReader(io.BytesIO(pdf)).pages:
        chunks: list[tuple[float, float, str]] = []

        def visitor(
            text: str,
            _cm: object,
            text_matrix: list[float],
            _font: object,
            _size: float,
            _chunks: list[tuple[float, float, str]] = chunks,
        ) -> None:
            clean = " ".join(text.split())
            if clean:
                _chunks.append(
                    (round(float(text_matrix[5]), 1), float(text_matrix[4]), clean)
                )

        page.extract_text(visitor_text=visitor)
        rows: dict[float, list[tuple[float, str]]] = {}
        for y, x, text in chunks:
            rows.setdefault(y, []).append((x, text))
        ordered_rows = [(y, sorted(parts)) for y, parts in sorted(rows.items())]
        player_row_ys: list[float] = []
        for y, ordered_parts in ordered_rows:
            status = next(
                (
                    text
                    for x, text in ordered_parts
                    if 580.0 <= x < 666.0
                    and text in REPORTED_STATUSES
                ),
                None,
            )
            raw_name = " ".join(
                text for x, text in ordered_parts if 420.0 <= x < 580.0
            )
            if status is not None and "," in raw_name:
                player_row_ys.append(y)

        reason_by_player_y: dict[float, str] = {}
        first_reason_boundary: float | None = None
        for index, player_y in enumerate(player_row_ys):
            previous_y = player_row_ys[index - 1] if index > 0 else None
            next_y = (
                player_row_ys[index + 1]
                if index + 1 < len(player_row_ys)
                else None
            )
            before = (
                (player_y - previous_y) / 2.0
                if previous_y is not None
                else (next_y - player_y) / 2.0
                if next_y is not None
                else 16.0
            )
            after = (
                (next_y - player_y) / 2.0
                if next_y is not None
                else (player_y - previous_y) / 2.0
                if previous_y is not None
                else 16.0
            )
            lower = player_y - before
            upper = player_y + after
            if first_reason_boundary is None:
                first_reason_boundary = lower
            reason_parts: list[str] = []
            for reason_y, reason_row in ordered_rows:
                if not lower <= reason_y < upper:
                    continue
                row_text = " ".join(text for _x, text in reason_row)
                upper_row_text = row_text.upper()
                if (
                    "NOT YET SUBMITTED" in upper_row_text
                    or "NO INJURIES TO REPORT" in upper_row_text
                ):
                    continue
                reason_parts.extend(
                    text
                    for x, text in reason_row
                    if x >= 666.0 and text.upper() != "REASON"
                )
            if reason_parts:
                reason_by_player_y[player_y] = " ".join(reason_parts)

        if statuses and first_reason_boundary is not None:
            leading_reason_parts: list[str] = []
            for reason_y, reason_row in ordered_rows:
                if reason_y >= first_reason_boundary:
                    break
                row_text = " ".join(text for _x, text in reason_row)
                upper_row_text = row_text.upper()
                if (
                    "NOT YET SUBMITTED" in upper_row_text
                    or "NO INJURIES TO REPORT" in upper_row_text
                ):
                    continue
                leading_reason_parts.extend(
                    text
                    for x, text in reason_row
                    if x >= 666.0 and text.upper() != "REASON"
                )
            if leading_reason_parts:
                statuses[-1]["reason"] = " ".join(
                    part
                    for part in [
                        statuses[-1].get("reason"),
                        " ".join(leading_reason_parts),
                    ]
                    if part
                )

        for _y, ordered_parts in ordered_rows:
            game_date_text = "".join(
                text for x, text in ordered_parts if 20.0 <= x < 119.0
            )
            game_date_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", game_date_text)
            if game_date_match is not None:
                current_game_date = datetime.strptime(
                    game_date_match.group(0), "%m/%d/%Y"
                ).date().isoformat()
            matchup_text = "".join(
                text for x, text in ordered_parts if 190.0 <= x < 250.0
            )
            matchup = re.search(r"([A-Z]{2,4})@([A-Z]{2,4})", matchup_text)
            if matchup is not None:
                current_matchup = (
                    NBA_TEAM_ALIASES.get(matchup.group(1), matchup.group(1)),
                    NBA_TEAM_ALIASES.get(matchup.group(2), matchup.group(2)),
                )
                current_team = None
                if current_game_date is not None:
                    game_key = f"{current_game_date}|{'@'.join(current_matchup)}"
                    game_team_reports.setdefault(game_key, {})
            team_tokens = [
                text for x, text in ordered_parts if 250.0 <= x < 420.0
            ]
            if current_matchup is not None and team_tokens:
                normalized_team_tokens = re.sub(
                    r"[^a-z0-9]", "", " ".join(team_tokens).lower()
                ).replace("losangeles", "la", 1)
                # Page headers can repeat while a team continues on the next page.
                if normalized_team_tokens not in {"team", "injuryreport", "page"}:
                    current_team = None
                    for candidate_team in current_matchup:
                        expected_name = TEAM_DISPLAY_NAMES.get(candidate_team, candidate_team)
                        normalized_expected_name = re.sub(
                            r"[^a-z0-9]", "", expected_name.lower()
                        ).replace("losangeles", "la", 1)
                        if normalized_team_tokens == normalized_expected_name:
                            current_team = candidate_team
                            break

            status = next(
                (
                    text
                    for x, text in ordered_parts
                    if 580.0 <= x < 666.0 and text in REPORTED_STATUSES
                ),
                None,
            )
            raw_name = " ".join(text for x, text in ordered_parts if 420.0 <= x < 580.0)
            name_parts = [part.strip() for part in raw_name.split(",") if part.strip()]
            is_player = status is not None and len(name_parts) >= 2
            submission_text = " ".join(
                text for x, text in ordered_parts if x >= 420.0
            ).upper()
            if is_player or "NO INJURIES TO REPORT" in submission_text:
                submission = "submitted"
            elif "NOT YET SUBMITTED" in submission_text:
                submission = "not_yet_submitted"
            else:
                continue

            if current_team is None or current_matchup is None or current_game_date is None:
                raise ValueError("Official NBA injury report has submission evidence without a game and team.")
            matchup_code = "@".join(current_matchup)
            team_report = {
                "team": current_team,
                "matchup": matchup_code,
                "game_date": current_game_date,
                "status": submission,
            }
            game_key = f"{current_game_date}|{matchup_code}"
            scoped_reports = game_team_reports.setdefault(game_key, {})
            previous = scoped_reports.setdefault(current_team, team_report)
            if previous["status"] != submission:
                raise ValueError(f"Conflicting NBA submission statuses for {current_team} in {game_key}.")
            team_reports.setdefault(current_team, team_report)

            if is_player:
                statuses.append({
                    "player_name": " ".join([name_parts[-1], *name_parts[:-1]]),
                    "status": status,
                    "reason": reason_by_player_y.get(_y),
                    "game_date": current_game_date,
                    "matchup": matchup_code,
                    "team": current_team,
                })
    if not team_reports:
        raise ValueError("Official NBA injury report contained no readable team submission evidence.")
    return {
        "players": statuses,
        "team_reports": team_reports,
        "game_team_reports": game_team_reports,
    }
