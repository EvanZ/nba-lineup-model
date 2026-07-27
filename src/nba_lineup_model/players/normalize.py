from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from nba_lineup_model.players.schema import (
    PlayerCatalog,
    PlayerIdentity,
    PlayerSeasonBio,
    PlayerSeasonBioDataset,
)
from nba_lineup_model.players.source import PlayerStatsError, PlayerStatsResponse

_HEIGHT_PATTERN = re.compile(r"^(?P<feet>\d+)-(?P<inches>\d{1,2})$")

_PLAYER_INDEX_FIELDS = frozenset(
    {
        "PERSON_ID",
        "PLAYER_LAST_NAME",
        "PLAYER_FIRST_NAME",
        "PLAYER_SLUG",
        "TEAM_ID",
        "TEAM_SLUG",
        "IS_DEFUNCT",
        "TEAM_ABBREVIATION",
        "JERSEY_NUMBER",
        "POSITION",
        "HEIGHT",
        "WEIGHT",
        "COLLEGE",
        "COUNTRY",
        "DRAFT_YEAR",
        "DRAFT_ROUND",
        "DRAFT_NUMBER",
        "ROSTER_STATUS",
        "FROM_YEAR",
        "TO_YEAR",
        "SUPPLEMENTAL_STATUS",
    }
)
_PLAYER_BIO_FIELDS = frozenset(
    {
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "AGE",
        "PLAYER_HEIGHT",
        "PLAYER_HEIGHT_INCHES",
        "PLAYER_WEIGHT",
        "COLLEGE",
        "COUNTRY",
        "DRAFT_YEAR",
        "DRAFT_ROUND",
        "DRAFT_NUMBER",
    }
)


def player_catalog_from_response(response: PlayerStatsResponse) -> PlayerCatalog:
    """Normalize the historical PlayerIndex response without lossy ID coercion."""

    source_sha256 = _response_sha256(response)
    players = []
    for row in _result_rows(
        response,
        result_name="PlayerIndex",
        required_fields=_PLAYER_INDEX_FIELDS,
    ):
        first_name = _optional_text(row["PLAYER_FIRST_NAME"])
        last_name = _optional_text(row["PLAYER_LAST_NAME"])
        display_name = " ".join(
            value for value in (first_name, last_name) if value
        ).strip()
        player_id = _required_int(row["PERSON_ID"], "PERSON_ID")
        if not display_name:
            display_name = _optional_text(row["PLAYER_SLUG"]) or str(player_id)
        draft_year, draft_round, draft_number, is_undrafted = _draft_fields(
            row,
            missing_status=None,
        )
        players.append(
            PlayerIdentity(
                player_id=player_id,
                first_name=first_name,
                last_name=last_name,
                display_name=display_name,
                player_slug=_optional_text(row["PLAYER_SLUG"]),
                listed_position=_optional_text(row["POSITION"]),
                height_raw=_optional_text(row["HEIGHT"]),
                height_inches=_height_inches(row["HEIGHT"]),
                weight_pounds=_optional_int(row["WEIGHT"]),
                college=_optional_text(row["COLLEGE"]),
                country=_optional_text(row["COUNTRY"]),
                draft_year=draft_year,
                draft_round=draft_round,
                draft_number=draft_number,
                is_undrafted=is_undrafted,
                roster_status=_optional_bool(row["ROSTER_STATUS"]),
                from_year=_optional_int(row["FROM_YEAR"]),
                to_year=_optional_int(row["TO_YEAR"]),
                latest_team_id=_positive_optional_int(row["TEAM_ID"]),
                latest_team_abbreviation=_optional_text(
                    row["TEAM_ABBREVIATION"]
                ),
                latest_team_slug=_optional_text(row["TEAM_SLUG"]),
                latest_jersey_number=_optional_text(row["JERSEY_NUMBER"]),
                is_defunct_team=_optional_bool(row["IS_DEFUNCT"]),
                supplemental_status=_optional_text(row["SUPPLEMENTAL_STATUS"]),
                source_season=response.season,
                source_url=response.url,
                source_fetched_at=response.fetched_at,
                source_sha256=source_sha256,
            )
        )
    return PlayerCatalog(
        players=sorted(players, key=lambda player: player.player_id)
    )


def player_season_bios_from_response(
    response: PlayerStatsResponse,
    catalog: PlayerCatalog,
) -> PlayerSeasonBioDataset:
    """Normalize season bio fields while excluding same-season performance."""

    if response.season_type is None:
        raise ValueError("Player bio responses require a canonical season type")
    catalog_by_id = {player.player_id: player for player in catalog.players}
    bio_source_sha256 = _response_sha256(response)
    players = []
    for row in _result_rows(
        response,
        result_name="LeagueDashPlayerBioStats",
        required_fields=_PLAYER_BIO_FIELDS,
    ):
        player_id = _required_int(row["PLAYER_ID"], "PLAYER_ID")
        identity = catalog_by_id.get(player_id)
        if identity is None:
            raise ValueError(f"Player bio ID is missing from PlayerIndex: {player_id}")
        height_raw = _required_text(row["PLAYER_HEIGHT"], "PLAYER_HEIGHT")
        height_inches = _required_int(
            row["PLAYER_HEIGHT_INCHES"],
            "PLAYER_HEIGHT_INCHES",
        )
        parsed_height = _height_inches(height_raw)
        if parsed_height != height_inches:
            raise ValueError(
                f"Player bio height fields disagree for {player_id}: "
                f"{height_raw} vs {height_inches}"
            )
        draft_year, draft_round, draft_number, is_undrafted = _draft_fields(
            row,
            missing_status=None,
        )
        if is_undrafted is None:
            raise ValueError(f"Player bio draft status is missing for {player_id}")
        players.append(
            PlayerSeasonBio(
                player_id=player_id,
                player_name=_required_text(row["PLAYER_NAME"], "PLAYER_NAME"),
                season=response.season,
                season_type=response.season_type,
                team_id=_required_int(row["TEAM_ID"], "TEAM_ID"),
                team_abbreviation=_required_text(
                    row["TEAM_ABBREVIATION"],
                    "TEAM_ABBREVIATION",
                ),
                age=_required_float(row["AGE"], "AGE"),
                listed_position=identity.listed_position,
                height_raw=height_raw,
                height_inches=height_inches,
                weight_pounds=_optional_int(row["PLAYER_WEIGHT"]),
                college=_optional_text(row["COLLEGE"]),
                country=_optional_text(row["COUNTRY"]),
                draft_year=draft_year,
                draft_round=draft_round,
                draft_number=draft_number,
                is_undrafted=is_undrafted,
                player_index_source_sha256=identity.source_sha256,
                bio_source_url=response.url,
                bio_source_fetched_at=response.fetched_at,
                bio_source_sha256=bio_source_sha256,
            )
        )
    return PlayerSeasonBioDataset(
        season=response.season,
        season_type=response.season_type,
        players=sorted(players, key=lambda player: player.player_id),
    )


def _result_rows(
    response: PlayerStatsResponse,
    *,
    result_name: str,
    required_fields: frozenset[str],
) -> list[dict[str, Any]]:
    result_sets = response.payload.get("resultSets")
    if not isinstance(result_sets, list):
        raise PlayerStatsError(
            f"NBA {response.endpoint.value} response lacks resultSets"
        )
    result = next(
        (
            candidate
            for candidate in result_sets
            if isinstance(candidate, dict) and candidate.get("name") == result_name
        ),
        None,
    )
    if result is None:
        raise PlayerStatsError(
            f"NBA {response.endpoint.value} response lacks {result_name}"
        )
    headers = result.get("headers")
    row_set = result.get("rowSet")
    if not isinstance(headers, list) or not all(
        isinstance(header, str) for header in headers
    ):
        raise PlayerStatsError(f"NBA {result_name} result has invalid headers")
    missing = required_fields - set(headers)
    if missing:
        raise PlayerStatsError(
            f"NBA {result_name} result lacks fields: {sorted(missing)}"
        )
    if not isinstance(row_set, list) or not row_set:
        raise PlayerStatsError(f"NBA {result_name} result contains no rows")
    rows: list[dict[str, Any]] = []
    for number, values in enumerate(row_set):
        if not isinstance(values, list) or len(values) != len(headers):
            raise PlayerStatsError(
                f"NBA {result_name} row {number} does not match its headers"
            )
        rows.append(dict(zip(headers, values, strict=True)))
    return rows


def _response_sha256(response: PlayerStatsResponse) -> str:
    if response.raw_body is None:
        raise ValueError("Player normalization requires exact response bytes")
    return hashlib.sha256(response.raw_body).hexdigest()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", "-", "None", "N/A"} else text


def _required_text(value: Any, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field} must contain text")
    return text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        if not value.is_integer():
            raise ValueError(f"Expected integer value, received {value}")
        return int(value)
    text = _optional_text(value)
    if text is None or text.lower() == "undrafted":
        return None
    return int(text)


def _required_int(value: Any, field: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"{field} must contain an integer")
    return parsed


def _positive_optional_int(value: Any) -> int | None:
    parsed = _optional_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _required_float(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"{field} must contain a number")
    parsed = float(value)
    if parsed != parsed:
        raise ValueError(f"{field} must contain a number")
    return parsed


def _optional_bool(value: Any) -> bool | None:
    parsed = _optional_int(value)
    if parsed is None:
        return None
    if parsed not in {0, 1}:
        raise ValueError(f"Expected zero-or-one flag, received {value}")
    return bool(parsed)


def _height_inches(value: Any) -> int | None:
    text = _optional_text(value)
    if text is None:
        return None
    match = _HEIGHT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"Unsupported NBA height value: {text}")
    feet = int(match.group("feet"))
    inches = int(match.group("inches"))
    if inches >= 12:
        raise ValueError(f"Unsupported NBA height value: {text}")
    return feet * 12 + inches


def _draft_fields(
    row: Mapping[str, Any],
    *,
    missing_status: bool | None,
) -> tuple[int | None, int | None, int | None, bool | None]:
    source_values = (
        row["DRAFT_YEAR"],
        row["DRAFT_ROUND"],
        row["DRAFT_NUMBER"],
    )
    texts = {
        text.lower()
        for value in source_values
        if (text := _optional_text(value)) is not None
    }
    is_undrafted = "undrafted" in texts
    if is_undrafted:
        if any(
            text != "undrafted"
            for text in texts
        ):
            raise ValueError("NBA draft fields mix Undrafted and numeric values")
        return None, None, None, True
    draft_year = _optional_int(row["DRAFT_YEAR"])
    draft_round = _optional_int(row["DRAFT_ROUND"])
    draft_number = _optional_int(row["DRAFT_NUMBER"])
    if draft_round == 0 or draft_number == 0:
        if (draft_round or 0) > 0 or (draft_number or 0) > 0:
            raise ValueError("NBA draft fields mix zero and positive pick values")
        return draft_year, None, None, True
    if draft_year is None and draft_round is None and draft_number is None:
        return None, None, None, missing_status
    if draft_year is not None and draft_round is None and draft_number is None:
        return draft_year, None, None, True
    return draft_year, draft_round, draft_number, False
