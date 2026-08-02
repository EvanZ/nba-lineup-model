from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nba_lineup_model.season.schema import (
    CatalogGame,
    GameCatalog,
    validate_season,
)

SCHEDULE_ENDPOINT = "https://stats.nba.com/stats/scheduleleaguev2"
_REQUIRED_GAME_FIELDS = frozenset(
    {
        "gameId",
        "gameStatus",
        "gameStatusText",
        "gameDateEst",
        "gameDateTimeUTC",
        "gameLabel",
        "gameSubLabel",
        "seriesText",
        "gameSubtype",
        "homeTeam",
        "awayTeam",
    }
)
_REQUIRED_TEAM_FIELDS = frozenset({"teamId", "teamTricode"})
_GAME_ID_PREFIX_SEASON_TYPES = {
    "001": "preseason",
    "002": "regular",
    "003": "all_star",
    "004": "playoffs",
    "005": "play_in",
    "006": "nba_cup_final",
}
_OVERTIME_RE = re.compile(
    r"(?:/|\s)(?:(?P<prefix>\d+)\s*)?OT(?P<suffix>\d+)?\b",
    re.IGNORECASE,
)


class NbaScheduleError(RuntimeError):
    """Raised when an NBA season schedule cannot be fetched or normalized."""


class ScheduleResponse(BaseModel):
    """One direct response from the NBA Stats season schedule endpoint."""

    model_config = ConfigDict(extra="forbid")

    season: str
    url: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any]
    raw_body: bytes | None = Field(default=None, exclude=True, repr=False)


class ScheduleCacheMetadata(BaseModel):
    """Provenance sidecar for one byte-preserved schedule response."""

    model_config = ConfigDict(extra="forbid")

    season: str
    url: str
    fetched_at: datetime
    sha256: str


class SeasonScheduleCache:
    """Byte-preserving cache for season-addressable NBA schedule responses."""

    def __init__(self, root: Path | str = Path("data/raw")) -> None:
        self.root = Path(root)

    def path_for(self, season: str) -> Path:
        return self.root / "scheduleleaguev2" / f"{validate_season(season)}.json"

    def metadata_path_for(self, season: str) -> Path:
        return self.path_for(season).with_suffix(".meta.json")

    def read(self, season: str) -> ScheduleResponse | None:
        path = self.path_for(season)
        if not path.exists():
            return None

        raw_body = path.read_bytes()
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NbaScheduleError(f"Cached NBA schedule is not valid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise NbaScheduleError(f"Cached NBA schedule has an unexpected JSON root: {path}")

        metadata_path = self.metadata_path_for(season)
        if not metadata_path.exists():
            raise NbaScheduleError(f"Cached NBA schedule is missing provenance metadata: {path}")
        metadata = ScheduleCacheMetadata.model_validate_json(metadata_path.read_text())
        if metadata.season != season:
            raise NbaScheduleError(f"Cached NBA schedule metadata does not match path: {path}")
        if metadata.sha256 != hashlib.sha256(raw_body).hexdigest():
            raise NbaScheduleError(f"Cached NBA schedule hash mismatch: {path}")

        return ScheduleResponse(
            season=metadata.season,
            url=metadata.url,
            fetched_at=metadata.fetched_at,
            payload=payload,
            raw_body=raw_body,
        )

    def write(self, response: ScheduleResponse) -> Path:
        validate_season(response.season)
        path = self.path_for(response.season)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_body = response.raw_body
        if raw_body is None:
            raw_body = json.dumps(
                response.payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

        _atomic_write_bytes(raw_body, path)
        metadata = ScheduleCacheMetadata(
            season=response.season,
            url=response.url,
            fetched_at=response.fetched_at,
            sha256=hashlib.sha256(raw_body).hexdigest(),
        )
        _atomic_write_text(
            metadata.model_dump_json(indent=2),
            self.metadata_path_for(response.season),
        )
        return path


class NbaScheduleClient:
    """Direct client for the season-parameterized NBA Stats schedule endpoint."""

    def __init__(
        self,
        *,
        cache: SeasonScheduleCache | None = None,
        http_client: httpx.Client | None = None,
        endpoint: str = SCHEDULE_ENDPOINT,
        timeout: float = 60.0,
    ) -> None:
        self.cache = cache or SeasonScheduleCache()
        self.endpoint = endpoint
        self._owns_http_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://www.nba.com",
                "Referer": "https://www.nba.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
                ),
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_http_client:
            self._client.close()

    def __enter__(self) -> NbaScheduleClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, season: str, *, use_cache: bool = True) -> ScheduleResponse:
        season = validate_season(season)
        if use_cache:
            cached = self.cache.read(season)
            if cached is not None:
                return cached

        try:
            http_response = self._client.get(
                self.endpoint,
                params={"LeagueID": "00", "Season": season},
            )
            http_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise NbaScheduleError(
                f"NBA Stats returned HTTP {exc.response.status_code} for {exc.request.url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise NbaScheduleError(f"NBA Stats schedule request failed for {season}") from exc

        raw_body = http_response.content
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NbaScheduleError(
                f"NBA Stats returned non-JSON schedule content for {season}"
            ) from exc
        if not isinstance(payload, dict):
            raise NbaScheduleError(
                f"NBA Stats returned an unexpected schedule JSON root for {season}"
            )

        response = ScheduleResponse(
            season=season,
            url=str(http_response.url),
            payload=payload,
            raw_body=raw_body,
        )
        self.cache.write(response)
        return response


def catalog_from_schedule(response: ScheduleResponse) -> GameCatalog:
    """Normalize one NBA schedule response into the canonical game catalog."""

    season = validate_season(response.season)
    rows = _schedule_rows(response.payload, season=season)
    if not rows:
        raise NbaScheduleError(f"NBA schedule contains no games for {season}")

    games = [
        _catalog_game(
            row,
            season=season,
            source_url=response.url,
            source_fetched_at=response.fetched_at,
        )
        for row in rows
    ]
    return GameCatalog(games=games)


def replace_catalog_season(existing: GameCatalog, discovered: GameCatalog) -> GameCatalog:
    """Replace one discovered season while retaining every other catalog season."""

    discovered_seasons = {game.season for game in discovered.games}
    if len(discovered_seasons) != 1:
        raise ValueError("A schedule discovery result must contain exactly one season")
    season = next(iter(discovered_seasons))
    retained = [game for game in existing.games if game.season != season]
    return GameCatalog(games=[*retained, *discovered.games])


def _schedule_rows(payload: dict[str, Any], *, season: str) -> list[dict[str, Any]]:
    schedule = payload.get("leagueSchedule")
    if not isinstance(schedule, dict):
        raise NbaScheduleError("NBA schedule response is missing leagueSchedule")

    source_season = _required_text(schedule, "seasonYear")
    if source_season != season:
        raise NbaScheduleError(
            f"NBA schedule season {source_season!r} does not match requested season {season!r}"
        )
    if _required_text(schedule, "leagueId") != "00":
        raise NbaScheduleError("NBA schedule response is not for the NBA league")

    game_dates = schedule.get("gameDates")
    if not isinstance(game_dates, list):
        raise NbaScheduleError("NBA schedule gameDates are invalid")

    rows: list[dict[str, Any]] = []
    for date_index, game_date in enumerate(game_dates):
        if not isinstance(game_date, dict):
            raise NbaScheduleError(f"NBA schedule game date {date_index} is invalid")
        schedule_date = _required_text(game_date, "gameDate")
        games = game_date.get("games")
        if not isinstance(games, list):
            raise NbaScheduleError(f"NBA schedule games for date {date_index} are invalid")
        for game_index, game in enumerate(games):
            if not isinstance(game, dict):
                raise NbaScheduleError(
                    f"NBA schedule game {game_index} for date {date_index} is invalid"
                )
            if _is_unidentifiable_non_catalog_placeholder(game):
                continue
            rows.append(
                _flatten_schedule_game(
                    game,
                    season=source_season,
                    schedule_date=schedule_date,
                    location=f"date {date_index}, game {game_index}",
                )
            )
    return rows


def _flatten_schedule_game(
    game: dict[str, Any],
    *,
    season: str,
    schedule_date: str,
    location: str,
) -> dict[str, Any]:
    missing = sorted(_REQUIRED_GAME_FIELDS - set(game))
    if missing:
        raise NbaScheduleError(
            f"NBA schedule {location} is missing required fields: {', '.join(missing)}"
        )
    home_team = game["homeTeam"]
    away_team = game["awayTeam"]
    if not isinstance(home_team, dict) or not isinstance(away_team, dict):
        raise NbaScheduleError(f"NBA schedule {location} has invalid team objects")
    for side, team in (("home", home_team), ("away", away_team)):
        missing_team_fields = sorted(_REQUIRED_TEAM_FIELDS - set(team))
        if missing_team_fields:
            raise NbaScheduleError(
                f"NBA schedule {location} {side} team is missing required fields: "
                f"{', '.join(missing_team_fields)}"
            )
        if not _has_identifiable_team(team):
            raise NbaScheduleError(f"NBA schedule {location} {side} team has an invalid identity")

    return {
        **game,
        "seasonYear": season,
        "gameDate": schedule_date,
        "homeTeam_teamId": home_team["teamId"],
        "homeTeam_teamTricode": home_team["teamTricode"],
        "awayTeam_teamId": away_team["teamId"],
        "awayTeam_teamTricode": away_team["teamTricode"],
    }


def _is_unidentifiable_non_catalog_placeholder(game: dict[str, Any]) -> bool:
    """Identify unplayed schedule placeholders without actual teams."""

    if _optional_int(game.get("gameStatus"), field="gameStatus") == 3:
        return False
    home_team = game.get("homeTeam")
    away_team = game.get("awayTeam")
    return not (
        isinstance(home_team, dict)
        and isinstance(away_team, dict)
        and _has_identifiable_team(home_team)
        and _has_identifiable_team(away_team)
    )


def _has_identifiable_team(team: dict[str, Any]) -> bool:
    team_id = _optional_int(team.get("teamId"), field="teamId")
    tricode = _optional_text(team.get("teamTricode"))
    return team_id is not None and team_id > 0 and tricode is not None


def _catalog_game(
    row: dict[str, Any],
    *,
    season: str,
    source_url: str,
    source_fetched_at: datetime,
) -> CatalogGame:
    source_season = _required_text(row, "seasonYear")
    if source_season != season:
        raise NbaScheduleError(
            f"NBA schedule row season {source_season!r} does not match requested season {season!r}"
        )

    game_id = _required_text(row, "gameId")
    status_code = _optional_int(row.get("gameStatus"), field="gameStatus")
    status_text = _optional_source_text(row.get("gameStatusText"))
    game_status = _game_status(status_code, status_text)
    period_count, is_overtime = _periods_from_status(game_status, status_text)

    return CatalogGame(
        game_id=game_id,
        season=season,
        season_type=_season_type(row, game_id),
        game_date=_game_date(row),
        game_time_utc=_optional_utc_datetime(row.get("gameDateTimeUTC"), season=season),
        game_status=game_status,
        home_team_id=_required_int(row, "homeTeam_teamId"),
        home_team_tricode=_required_text(row, "homeTeam_teamTricode").upper(),
        away_team_id=_required_int(row, "awayTeam_teamId"),
        away_team_tricode=_required_text(row, "awayTeam_teamTricode").upper(),
        period_count=period_count,
        is_overtime=is_overtime,
        source_status_code=status_code,
        source_status_text=status_text,
        source_url=source_url,
        source_fetched_at=source_fetched_at,
    )


def _season_type(row: dict[str, Any], game_id: str) -> str:
    season_type = _GAME_ID_PREFIX_SEASON_TYPES.get(game_id[:3])
    if season_type is not None:
        return season_type

    label = " ".join(
        filter(
            None,
            (
                _optional_text(row.get("gameLabel")),
                _optional_text(row.get("gameSubLabel")),
                _optional_text(row.get("seriesText")),
                _optional_text(row.get("gameSubtype")),
            ),
        )
    ).lower()

    if "preseason" in label or "pre-season" in label:
        return "preseason"
    if "play-in" in label or "play in" in label or "playin" in label:
        return "play_in"
    if "all-star" in label or "all star" in label:
        return "all_star"
    if "playoff" in label or "finals" in label:
        return "playoffs"

    raise NbaScheduleError(f"Cannot classify NBA game {game_id} from its labels or ID prefix")


def _game_status(status_code: int | None, status_text: str | None) -> str:
    normalized_text = (status_text or "").strip().lower()
    if "cancel" in normalized_text:
        return "cancelled"
    if "postpon" in normalized_text or normalized_text == "ppd":
        return "postponed"
    return {1: "scheduled", 2: "live", 3: "final"}.get(status_code, "unknown")


def _periods_from_status(
    game_status: str,
    status_text: str | None,
) -> tuple[int | None, bool | None]:
    if game_status != "final":
        return None, None
    overtime_match = _OVERTIME_RE.search(status_text or "")
    if overtime_match is None:
        return 4, False
    overtime_periods = int(overtime_match.group("prefix") or overtime_match.group("suffix") or "1")
    return 4 + overtime_periods, True


def _game_date(row: dict[str, Any]) -> date:
    for field in ("gameDateEst", "gameDate"):
        value = _optional_text(row.get(field))
        if value is None:
            continue
        normalized = value[:10]
        try:
            return date.fromisoformat(normalized)
        except ValueError:
            pass
        for format_string in ("%m/%d/%Y", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(value, format_string).date()
            except ValueError:
                continue
    raise NbaScheduleError("NBA schedule row has no parseable game date")


def _optional_utc_datetime(value: Any, *, season: str) -> datetime | None:
    text = _optional_text(value)
    if text is None or text.upper() in {"TBD", "TBA"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NbaScheduleError(f"NBA schedule UTC datetime is invalid: {text!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed.year < int(season[:4]):
        return None
    return parsed


def _required_text(row: dict[str, Any], field: str) -> str:
    value = _optional_text(row.get(field))
    if value is None:
        raise NbaScheduleError(f"NBA schedule row is missing {field}")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _optional_source_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return value if value.strip() else None


def _required_int(row: dict[str, Any], field: str) -> int:
    value = _optional_int(row.get(field), field=field)
    if value is None:
        raise NbaScheduleError(f"NBA schedule row is missing {field}")
    return value


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise NbaScheduleError(f"NBA schedule {field} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise NbaScheduleError(f"NBA schedule {field} is not an integer: {value!r}")


def _atomic_write_bytes(content: bytes, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _atomic_write_text(content: str, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content)
    temporary_path.replace(path)
