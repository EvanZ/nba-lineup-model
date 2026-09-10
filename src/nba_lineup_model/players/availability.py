"""Build a player-game availability mart from official NBA box-score statuses.

The mart distinguishes an appearance from a coach's rotation decision.  It is
intentionally a source-normalization layer, not an availability forecast.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.season.schema import validate_season

DEFAULT_FIRST_SEASON = "2015-16"
DEFAULT_LAST_SEASON = "2025-26"

STATE_PLAYED = "played"
STATE_AVAILABLE_DNP_COACH = "available_dnp_coach"
STATE_AVAILABLE_DNP_ACTIVE = "available_dnp_active"
STATE_AVAILABLE_G_LEAGUE_ASSIGNMENT = "available_g_league_assignment"
STATE_UNAVAILABLE_INJURY = "unavailable_injury_or_illness"
STATE_UNAVAILABLE_REST = "unavailable_rest"
STATE_UNAVAILABLE_PERSONAL = "unavailable_personal_or_not_with_team"
STATE_UNAVAILABLE_SUSPENSION = "unavailable_suspension"
STATE_UNAVAILABLE_INELIGIBLE = "unavailable_ineligible"
STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED = "unavailable_inactive_unspecified"
STATE_UNKNOWN_DNP = "unknown_dnp_reason"

AVAILABILITY_STATES = (
    STATE_PLAYED,
    STATE_AVAILABLE_DNP_COACH,
    STATE_AVAILABLE_DNP_ACTIVE,
    STATE_AVAILABLE_G_LEAGUE_ASSIGNMENT,
    STATE_UNAVAILABLE_INJURY,
    STATE_UNAVAILABLE_REST,
    STATE_UNAVAILABLE_PERSONAL,
    STATE_UNAVAILABLE_SUSPENSION,
    STATE_UNAVAILABLE_INELIGIBLE,
    STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED,
    STATE_UNKNOWN_DNP,
)
AVAILABLE_STATES = frozenset(
    {
        STATE_PLAYED,
        STATE_AVAILABLE_DNP_COACH,
        STATE_AVAILABLE_DNP_ACTIVE,
        STATE_AVAILABLE_G_LEAGUE_ASSIGNMENT,
    }
)

# A modern live box score enumerates the full team roster with explicit player
# statuses. Historical Stats V3 traditional box scores generally enumerate the
# active game list only. The distinction is material: the former supports a
# health-availability denominator; the latter does not on its own.
SOURCE_PLAYER_TABLE_CONTRACT = {
    "live_boxscore": "full_roster_status",
    "stats_v3": "boxscore_listed_players",
    "stats_v3_summary_v2": "full_roster_membership",
}

_MEDICAL_DESCRIPTION_TOKENS = (
    "SPRAIN",
    "STRAIN",
    "SORE",
    "SURGERY",
    "SURGICAL",
    "FRACTUR",
    "CONTUSION",
    "TEAR",
    "RUPTURE",
    "SPASM",
    "PAIN",
    "REHAB",
    "RECOVERY",
    "MANAGEMENT",
    "TENDON",
    "ACL",
    "MCL",
    "HAMSTRING",
    "ACHILLES",
)
_ISO_MINUTES_RE = re.compile(
    r"^PT(?:(?P<hours>[0-9]+(?:\.[0-9]+)?)H)?"
    r"(?:(?P<minutes>[0-9]+(?:\.[0-9]+)?)M)?"
    r"(?:(?P<seconds>[0-9]+(?:\.[0-9]+)?)S)?$"
)


class AvailabilityMartError(RuntimeError):
    """Raised when an availability mart cannot be assembled from raw inputs."""


def build_player_availability_mart(
    season: str,
    *,
    raw_dir: Path | str = Path("data/raw"),
    curated_dir: Path | str = Path("data/curated"),
) -> Path:
    """Publish regular-season player availability states for one NBA season.

    A live NBA CDN box score is preferred when both sources exist because it
    supplies structured inactive fields. Historical Stats V3 is paired with
    Summary V2 inactive membership when both are present. A blank DNP row is
    never treated as evidence of availability or unavailability.
    """

    season = validate_season(season)
    raw_root = Path(raw_dir)
    games = _regular_season_games(raw_root / "scheduleleaguev2" / f"{season}.json")
    rows: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for game in games.itertuples(index=False):
        source = _resolve_source(raw_root, str(game.game_id))
        if source is None:
            source_rows.append(
                {
                    "season": season,
                    "game_id": str(game.game_id),
                    "game_date": game.game_date,
                    "source_kind": "missing",
                    "source_player_table_contract": "missing",
                    "source_path": None,
                    "player_rows": 0,
                }
            )
            continue
        source_kind, source_paths = source
        if source_kind == "stats_v3_summary_v2":
            stats_path, summary_path = source_paths
            frame = availability_rows_from_stats_v3_and_summary(
                json.loads(stats_path.read_text()),
                json.loads(summary_path.read_text()),
                season=season,
                game_id=str(game.game_id),
                game_date=game.game_date,
                stats_path=stats_path,
                summary_path=summary_path,
            )
        else:
            (source_path,) = source_paths
            frame = availability_rows_from_payload(
                json.loads(source_path.read_text()),
                source_kind=source_kind,
                season=season,
                game_id=str(game.game_id),
                game_date=game.game_date,
                source_path=source_path,
            )
        rows.append(frame)
        source_rows.append(
            {
                "season": season,
                "game_id": str(game.game_id),
                "game_date": game.game_date,
                "source_kind": source_kind,
                "source_player_table_contract": SOURCE_PLAYER_TABLE_CONTRACT[source_kind],
                "source_path": ";".join(str(path) for path in source_paths),
                "player_rows": int(len(frame)),
            }
        )
    if not rows:
        raise AvailabilityMartError(f"No box-score sources found for {season}")

    player_games = pd.concat(rows, ignore_index=True)
    player_games = _correct_isolated_coach_dnp_with_medical_neighbors(player_games)
    _validate_player_games(player_games, season=season)
    source_coverage = pd.DataFrame(source_rows)
    target = Path(curated_dir) / "player_availability" / season
    target.mkdir(parents=True, exist_ok=True)
    output_path = target / "part-00000.parquet"
    player_games.to_parquet(output_path, index=False)
    source_coverage.to_parquet(target / "game_coverage.parquet", index=False)
    _season_coverage_frame(
        season, player_games, source_coverage, scheduled_games=len(games)
    ).to_parquet(target / "coverage.parquet", index=False)
    _state_counts_frame(season, player_games).to_parquet(
        target / "state_counts.parquet", index=False
    )
    _reason_counts_frame(season, player_games).to_parquet(
        target / "reason_counts.parquet", index=False
    )
    (target / "_manifest.json").write_text(
        json.dumps(
            {
                "season": season,
                "scheduled_regular_season_games": int(len(games)),
                "source_games": int(source_coverage["source_kind"].ne("missing").sum()),
                "missing_source_games": int(source_coverage["source_kind"].eq("missing").sum()),
                "player_game_rows": int(len(player_games)),
                "availability_states": list(AVAILABILITY_STATES),
                "source_precedence": ["live_boxscore", "stats_v3_summary_v2", "stats_v3"],
                "source_player_table_contract": SOURCE_PLAYER_TABLE_CONTRACT,
            },
            indent=2,
        )
        + "\n"
    )
    return output_path


def build_player_availability_coverage(
    *,
    curated_dir: Path | str = Path("data/curated"),
) -> Path:
    """Publish a cross-season coverage table for completed availability marts."""

    root = Path(curated_dir) / "player_availability"
    rows: list[pd.DataFrame] = []
    for season_path in sorted(path for path in root.iterdir() if path.is_dir()):
        path = season_path / "coverage.parquet"
        if path.exists():
            rows.append(pd.read_parquet(path))
    if not rows:
        raise AvailabilityMartError(f"No completed player availability marts found in {root}")
    coverage = pd.concat(rows, ignore_index=True).sort_values("season", kind="stable")
    target = root / "coverage.parquet"
    coverage.to_parquet(target, index=False)
    return target


def availability_rows_from_payload(
    payload: Mapping[str, Any],
    *,
    source_kind: str,
    season: str,
    game_id: str,
    game_date: pd.Timestamp,
    source_path: Path | str,
) -> pd.DataFrame:
    """Normalize one official box-score payload into player availability rows."""

    if source_kind == "live_boxscore":
        teams = _live_teams(payload)
    elif source_kind == "stats_v3":
        teams = _stats_v3_teams(payload)
    else:
        raise ValueError(f"Unsupported availability source kind: {source_kind}")

    rows: list[dict[str, Any]] = []
    for team in teams:
        team_id = _optional_int(team.get("teamId"))
        team_tricode = _optional_text(team.get("teamTricode"))
        players = team.get("players")
        if not isinstance(players, list):
            continue
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_id = _optional_int(player.get("personId"))
            if player_id is None:
                continue
            raw_comment = _optional_text(player.get("comment"))
            raw_status = _optional_text(player.get("status"))
            raw_reason = _optional_text(player.get("notPlayingReason"))
            raw_description = _optional_text(player.get("notPlayingDescription"))
            minutes = _minutes(player.get("statistics"))
            raw_played = _optional_bool(player.get("played"))
            state = classify_availability_state(
                minutes=minutes,
                raw_played=raw_played,
                raw_status=raw_status,
                raw_reason=raw_reason,
                raw_comment=raw_comment,
            )
            rows.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "game_date": game_date,
                    "source_kind": source_kind,
                    "source_player_table_contract": SOURCE_PLAYER_TABLE_CONTRACT[source_kind],
                    "source_path": str(source_path),
                    "team_id": team_id,
                    "team": team_tricode,
                    "player_id": player_id,
                    "player_name": _player_name(player),
                    "minutes": minutes,
                    "raw_played": raw_played,
                    "raw_status": raw_status,
                    "raw_not_playing_reason": raw_reason,
                    "raw_not_playing_description": raw_description,
                    "raw_comment": raw_comment,
                    "availability_state": state,
                    "available": availability_label(state),
                    "availability_state_known": state != STATE_UNKNOWN_DNP,
                }
            )
    if not rows:
        raise AvailabilityMartError(f"No player rows in {source_kind} box score {game_id}")
    return pd.DataFrame(rows)


def availability_rows_from_stats_v3_and_summary(
    stats_payload: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
    *,
    season: str,
    game_id: str,
    game_date: pd.Timestamp,
    stats_path: Path | str,
    summary_path: Path | str,
) -> pd.DataFrame:
    """Return a complete historical player-game panel from two official sources.

    Stats V3 supplies the players who appeared or received a listed DNP. The
    Summary V2 ``InactivePlayers`` table supplies rostered players omitted from
    that box-score table. Under the approved contract, those explicit inactive
    entries are unavailable even where the historical endpoint gives no cause.
    """

    listed = availability_rows_from_payload(
        stats_payload,
        source_kind="stats_v3",
        season=season,
        game_id=game_id,
        game_date=game_date,
        source_path=stats_path,
    )
    inactives = _inactive_summary_rows(
        summary_payload,
        season=season,
        game_id=game_id,
        game_date=game_date,
        source_path=summary_path,
    )
    listed_keys = set(zip(listed["team_id"], listed["player_id"], strict=True))
    inactives = inactives.loc[
        ~inactives.apply(lambda row: (row.team_id, row.player_id) in listed_keys, axis=1)
    ]
    panel = pd.concat((listed, inactives), ignore_index=True)
    panel["source_kind"] = "stats_v3_summary_v2"
    panel["source_player_table_contract"] = SOURCE_PLAYER_TABLE_CONTRACT[
        "stats_v3_summary_v2"
    ]
    return panel


def _inactive_summary_rows(
    payload: Mapping[str, Any],
    *,
    season: str,
    game_id: str,
    game_date: pd.Timestamp,
    source_path: Path | str,
) -> pd.DataFrame:
    """Normalize the official Summary V2 historical inactive-player table."""

    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        raise AvailabilityMartError(f"Summary V2 payload has no result sets for {game_id}")
    inactive_table = next(
        (
            item
            for item in result_sets
            if isinstance(item, Mapping) and item.get("name") == "InactivePlayers"
        ),
        None,
    )
    if not isinstance(inactive_table, Mapping):
        raise AvailabilityMartError(
            f"Summary V2 payload has no inactive players table for {game_id}"
        )
    headers = inactive_table.get("headers")
    raw_rows = inactive_table.get("rowSet")
    if not isinstance(headers, list) or not isinstance(raw_rows, list):
        raise AvailabilityMartError(f"Summary V2 inactive players table is malformed for {game_id}")
    columns = {str(header): index for index, header in enumerate(headers)}
    required = {"PLAYER_ID", "TEAM_ID"}
    missing = required - set(columns)
    if missing:
        raise AvailabilityMartError(
            f"Summary V2 inactive players table is missing {sorted(missing)} for {game_id}"
        )

    state = STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED
    rows: list[dict[str, Any]] = []
    for values in raw_rows:
        if not isinstance(values, list):
            continue
        player_id = _optional_int(_summary_value(values, columns, "PLAYER_ID"))
        team_id = _optional_int(_summary_value(values, columns, "TEAM_ID"))
        if player_id is None or team_id is None:
            continue
        first_name = _optional_text(_summary_value(values, columns, "FIRST_NAME"))
        last_name = _optional_text(_summary_value(values, columns, "LAST_NAME"))
        player_name = " ".join(part for part in (first_name, last_name) if part) or None
        rows.append(
            {
                "season": season,
                "game_id": game_id,
                "game_date": game_date,
                "source_kind": "stats_v3_summary_v2",
                "source_player_table_contract": SOURCE_PLAYER_TABLE_CONTRACT[
                    "stats_v3_summary_v2"
                ],
                "source_path": str(source_path),
                "team_id": team_id,
                "team": _optional_text(_summary_value(values, columns, "TEAM_ABBREVIATION")),
                "player_id": player_id,
                "player_name": player_name,
                "minutes": 0.0,
                "raw_played": False,
                "raw_status": "INACTIVE",
                "raw_not_playing_reason": None,
                "raw_not_playing_description": None,
                "raw_comment": None,
                "availability_state": state,
                "available": availability_label(state),
                "availability_state_known": True,
            }
        )
    return pd.DataFrame(rows, columns=_player_game_columns())


def _summary_value(values: list[Any], columns: Mapping[str, int], name: str) -> Any:
    index = columns.get(name)
    return values[index] if index is not None and index < len(values) else None


def _player_game_columns() -> list[str]:
    return [
        "season",
        "game_id",
        "game_date",
        "source_kind",
        "source_player_table_contract",
        "source_path",
        "team_id",
        "team",
        "player_id",
        "player_name",
        "minutes",
        "raw_played",
        "raw_status",
        "raw_not_playing_reason",
        "raw_not_playing_description",
        "raw_comment",
        "availability_state",
        "available",
        "availability_state_known",
    ]


def classify_availability_state(
    *,
    minutes: float,
    raw_played: bool | None,
    raw_status: str | None,
    raw_reason: str | None,
    raw_comment: str | None,
    historically_inactive: bool = False,
) -> str:
    """Classify a player-game row under the availability-label contract.

    ``historically_inactive`` is reserved for an explicit player membership
    record from a historical game's inactive list. It is not inferred from a
    blank Stats V3 row, which can also represent an active DNP.
    """

    if minutes > 0 or raw_played is True:
        return STATE_PLAYED
    text = " ".join(value for value in (raw_status, raw_reason, raw_comment) if value).upper()
    if "COACH" in text:
        return STATE_AVAILABLE_DNP_COACH
    if "REST" in text:
        return STATE_UNAVAILABLE_REST
    if any(token in text for token in ("G_LEAGUE", "GLEAGUE", "TWO-WAY")):
        return STATE_AVAILABLE_G_LEAGUE_ASSIGNMENT
    if "SUSPENSION" in text:
        return STATE_UNAVAILABLE_SUSPENSION
    if any(token in text for token in ("INELIGIBLE", "INACTIVE_LIST")):
        return STATE_UNAVAILABLE_INELIGIBLE
    if any(token in text for token in ("PERSONAL", "NOT_WITH_TEAM", "NOT WITH TEAM", "TRADE")):
        return STATE_UNAVAILABLE_PERSONAL
    if any(
        token in text
        for token in (
            "INJURY",
            "ILLNESS",
            "CONCUSSION",
            "HEALTH_AND_SAFETY",
            "HEALTH AND SAFETY",
            "SELF_ISOLATING",
            "SELF ISOLATING",
            "RETURN_TO_COMPETITION",
            "RETURN TO COMPETITION",
        )
    ):
        return STATE_UNAVAILABLE_INJURY
    # Historical rows often name the medical condition without an injury
    # prefix, including NWT rows such as "NWT - Left Knee Surgery".
    if raw_comment and any(token in raw_comment.upper() for token in _MEDICAL_DESCRIPTION_TOKENS):
        return STATE_UNAVAILABLE_INJURY
    # Modern player tables explicitly distinguish ACTIVE players from the
    # inactive list even when a healthy player receives zero minutes.
    if raw_status and raw_status.upper() == "ACTIVE":
        return STATE_AVAILABLE_DNP_ACTIVE
    if historically_inactive or (raw_status and raw_status.upper() == "INACTIVE"):
        return STATE_UNAVAILABLE_INACTIVE_UNSPECIFIED
    return STATE_UNKNOWN_DNP


def availability_label(state: str) -> bool | None:
    """Return the binary availability target, or ``None`` when unlabeled.

    A historical inactive list is an explicit team designation, even when it
    omits a reason, so it is labeled unavailable. A blank zero-minute box-score
    row is different: without an inactive designation it has no binary label.
    """

    if state == STATE_UNKNOWN_DNP:
        return None
    return state in AVAILABLE_STATES


def _correct_isolated_coach_dnp_with_medical_neighbors(
    player_games: pd.DataFrame,
) -> pd.DataFrame:
    """Correct contradictory single-game coach labels within a medical absence.

    A live box score can occasionally label a medically inactive player as a
    coach DNP for one game. Reclassify only a coach DNP directly bracketed by
    that same player's injury/illness absences on the same team. This preserves
    ordinary coach decisions while avoiding a spurious zero-minute available
    observation in the conditional-minutes target.
    """

    if player_games.empty:
        return player_games
    output = player_games.copy()
    ordered = output.sort_values(
        ["team_id", "player_id", "game_date", "game_id"], kind="stable"
    )
    grouped = ordered.groupby(["team_id", "player_id"], sort=False)["availability_state"]
    previous = grouped.shift(1)
    following = grouped.shift(-1)
    mask = (
        ordered["availability_state"].eq(STATE_AVAILABLE_DNP_COACH)
        & previous.eq(STATE_UNAVAILABLE_INJURY)
        & following.eq(STATE_UNAVAILABLE_INJURY)
    )
    corrected_index = ordered.index[mask]
    output.loc[corrected_index, "availability_state"] = STATE_UNAVAILABLE_INJURY
    output.loc[corrected_index, "available"] = False
    return output


def _resolve_source(raw_root: Path, game_id: str) -> tuple[str, tuple[Path, ...]] | None:
    live = raw_root / "boxscore" / f"{game_id}.json"
    if live.exists():
        return "live_boxscore", (live,)
    stats_v3 = raw_root / "stats" / "boxscoretraditionalv3" / f"{game_id}.json"
    summary = raw_root / "stats" / "boxscoresummaryv2" / f"{game_id}.json"
    if stats_v3.exists() and summary.exists():
        return "stats_v3_summary_v2", (stats_v3, summary)
    if stats_v3.exists():
        return "stats_v3", (stats_v3,)
    return None


def _live_teams(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    game = payload.get("game")
    if not isinstance(game, Mapping):
        raise AvailabilityMartError("Expected live box score game payload")
    return [
        team for side in ("homeTeam", "awayTeam") if isinstance((team := game.get(side)), Mapping)
    ]


def _stats_v3_teams(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    box_score = payload.get("boxScoreTraditional")
    if not isinstance(box_score, Mapping):
        raise AvailabilityMartError("Expected Stats V3 traditional box-score payload")
    return [
        team
        for side in ("homeTeam", "awayTeam")
        if isinstance((team := box_score.get(side)), Mapping)
    ]


def _regular_season_games(schedule_path: Path) -> pd.DataFrame:
    if not schedule_path.exists():
        raise AvailabilityMartError(f"Missing season schedule: {schedule_path}")
    payload = json.loads(schedule_path.read_text())
    rows: list[dict[str, str]] = []
    for game_date in payload.get("leagueSchedule", {}).get("gameDates", []):
        if not isinstance(game_date, Mapping):
            continue
        for game in game_date.get("games", []):
            if not isinstance(game, Mapping):
                continue
            game_id = str(game.get("gameId") or "")
            date_est = game.get("gameDateEst")
            if game_id.startswith("002") and isinstance(date_est, str):
                rows.append({"game_id": game_id, "game_date": date_est})
    if not rows:
        raise AvailabilityMartError(f"Schedule has no regular-season games: {schedule_path}")
    games = pd.DataFrame(rows).drop_duplicates("game_id")
    games["game_date"] = pd.to_datetime(games["game_date"], utc=True)
    return games.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)


def _season_coverage_frame(
    season: str,
    player_games: pd.DataFrame,
    game_coverage: pd.DataFrame,
    *,
    scheduled_games: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "scheduled_regular_season_games": int(scheduled_games),
                "source_games": int(game_coverage["source_kind"].ne("missing").sum()),
                "missing_source_games": int(game_coverage["source_kind"].eq("missing").sum()),
                "live_boxscore_games": int(game_coverage["source_kind"].eq("live_boxscore").sum()),
                "stats_v3_games": int(game_coverage["source_kind"].eq("stats_v3").sum()),
                "stats_v3_summary_v2_games": int(
                    game_coverage["source_kind"].eq("stats_v3_summary_v2").sum()
                ),
                "full_roster_status_games": int(
                    game_coverage["source_player_table_contract"].eq("full_roster_status").sum()
                ),
                "full_roster_membership_games": int(
                    game_coverage["source_player_table_contract"]
                    .eq("full_roster_membership")
                    .sum()
                ),
                "boxscore_listed_player_games": int(
                    game_coverage["source_player_table_contract"]
                    .eq("boxscore_listed_players")
                    .sum()
                ),
                "player_game_rows": int(len(player_games)),
                "known_state_rows": int(player_games["availability_state_known"].sum()),
                "unknown_dnp_reason_rows": int(
                    player_games["availability_state"].eq(STATE_UNKNOWN_DNP).sum()
                ),
            }
        ]
    )


def _state_counts_frame(season: str, player_games: pd.DataFrame) -> pd.DataFrame:
    counts = (
        player_games["availability_state"].value_counts().reindex(AVAILABILITY_STATES, fill_value=0)
    )
    return (
        counts.rename_axis("availability_state")
        .reset_index(name="player_game_rows")
        .assign(season=season)
    )


def _reason_counts_frame(season: str, player_games: pd.DataFrame) -> pd.DataFrame:
    raw_reason = player_games["raw_not_playing_reason"].fillna(player_games["raw_comment"])
    rows = (
        player_games.assign(raw_reason=raw_reason)
        .loc[lambda frame: frame.raw_reason.notna(), ["availability_state", "raw_reason"]]
        .value_counts()
        .rename("player_game_rows")
        .reset_index()
        .sort_values(
            ["availability_state", "player_game_rows", "raw_reason"],
            ascending=[True, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    rows.insert(0, "season", season)
    return rows


def _validate_player_games(player_games: pd.DataFrame, *, season: str) -> None:
    required = {
        "season",
        "game_id",
        "team_id",
        "player_id",
        "availability_state",
        "available",
        "availability_state_known",
    }
    missing = required - set(player_games)
    if missing:
        raise AvailabilityMartError(
            f"Availability rows missing required columns: {sorted(missing)}"
        )
    if not player_games["season"].eq(season).all():
        raise AvailabilityMartError("Availability mart contains mixed seasons")
    if player_games.duplicated(["game_id", "team_id", "player_id"]).any():
        raise AvailabilityMartError("Availability mart contains duplicate player-game rows")
    if not player_games["availability_state"].isin(AVAILABILITY_STATES).all():
        raise AvailabilityMartError("Availability mart contains unknown normalized states")
    known = player_games["availability_state_known"].astype(bool)
    if player_games.loc[known, "available"].isna().any():
        raise AvailabilityMartError("Known availability rows require a binary target")
    if player_games.loc[~known, "available"].notna().any():
        raise AvailabilityMartError("Unknown availability rows must not receive a target")


def _minutes(statistics: Any) -> float:
    if not isinstance(statistics, Mapping):
        return 0.0
    raw = _optional_text(statistics.get("minutes"))
    if not raw:
        return 0.0
    try:
        if raw.startswith("PT"):
            match = _ISO_MINUTES_RE.match(raw)
            if match is None:
                return 0.0
            hours = float(match.group("hours") or 0.0)
            minutes = float(match.group("minutes") or 0.0)
            seconds = float(match.group("seconds") or 0.0)
            return hours * 60.0 + minutes + seconds / 60.0
        parts = [float(part) for part in raw.split(":")]
        if len(parts) == 2:
            return parts[0] + parts[1] / 60.0
        if len(parts) == 3:
            return parts[0] * 60.0 + parts[1] + parts[2] / 60.0
    except ValueError:
        return 0.0
    return 0.0


def _player_name(player: Mapping[str, Any]) -> str | None:
    name = _optional_text(player.get("name")) or _optional_text(player.get("nameI"))
    if name:
        return name
    full_name = " ".join(
        part
        for part in (
            _optional_text(player.get("firstName")),
            _optional_text(player.get("familyName")),
        )
        if part
    )
    return full_name or None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a player-game availability mart from official NBA box-score reasons"
    )
    parser.add_argument("seasons", nargs="+", help="NBA season labels, such as 2015-16")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for season in args.seasons:
        print(
            build_player_availability_mart(
                season, raw_dir=args.raw_dir, curated_dir=args.curated_dir
            )
        )
    print(build_player_availability_coverage(curated_dir=args.curated_dir))
