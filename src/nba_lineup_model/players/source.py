from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nba_lineup_model.season.schema import validate_season


class PlayerStatsEndpoint(StrEnum):
    """Direct NBA Stats endpoints used by the player reference pipeline."""

    PLAYER_INDEX = "playerindex"
    PLAYER_BIO_STATS = "leaguedashplayerbiostats"
    DRAFT_HISTORY = "drafthistory"
    TEAM_ROSTER = "commonteamroster"


class PlayerStatsError(RuntimeError):
    """Raised when direct NBA player reference data is unavailable or invalid."""


class PlayerStatsResponse(BaseModel):
    """One exact NBA Stats player endpoint response and request provenance."""

    model_config = ConfigDict(extra="forbid")

    endpoint: PlayerStatsEndpoint
    season: str
    season_type: str | None = None
    team_id: int | None = None
    params: dict[str, str]
    url: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any]
    raw_body: bytes | None = Field(default=None, exclude=True, repr=False)


class PlayerStatsCacheMetadata(BaseModel):
    """Sidecar evidence for a byte-preserved player endpoint response."""

    model_config = ConfigDict(extra="forbid")

    endpoint: PlayerStatsEndpoint
    season: str
    season_type: str | None = None
    team_id: int | None = None
    params: dict[str, str]
    url: str
    fetched_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PlayerStatsCache:
    """Byte-preserving cache for season-addressable player endpoint responses."""

    def __init__(self, root: Path | str = Path("data/raw")) -> None:
        self.root = Path(root)

    def path_for(
        self,
        endpoint: PlayerStatsEndpoint,
        season: str,
        season_type: str | None = None,
        team_id: int | None = None,
    ) -> Path:
        season = validate_season(season)
        if endpoint in {PlayerStatsEndpoint.PLAYER_INDEX, PlayerStatsEndpoint.DRAFT_HISTORY}:
            if season_type is not None or team_id is not None:
                raise ValueError(
                    "Player index and Draft History cache paths do not use extra scope"
                )
            return self.root / endpoint.value / f"{season}.json"
        if endpoint is PlayerStatsEndpoint.TEAM_ROSTER:
            if season_type is not None or team_id is None or team_id <= 0:
                raise ValueError("Team roster cache paths require a positive team ID")
            return self.root / endpoint.value / season / f"{team_id}.json"
        if team_id is not None:
            raise ValueError("Player bio cache paths do not use team ID")
        if not season_type:
            raise ValueError("Player bio cache paths require season type")
        return self.root / endpoint.value / season / f"{season_type}.json"

    def metadata_path_for(
        self,
        endpoint: PlayerStatsEndpoint,
        season: str,
        season_type: str | None = None,
        team_id: int | None = None,
    ) -> Path:
        return self.path_for(endpoint, season, season_type, team_id).with_suffix(".meta.json")

    def read(
        self,
        endpoint: PlayerStatsEndpoint,
        season: str,
        season_type: str | None,
        team_id: int | None = None,
        *,
        expected_params: dict[str, str],
    ) -> PlayerStatsResponse | None:
        path = self.path_for(endpoint, season, season_type, team_id)
        if not path.exists():
            return None
        metadata_path = self.metadata_path_for(endpoint, season, season_type, team_id)
        if not metadata_path.exists():
            raise PlayerStatsError(f"Cached player response lacks metadata: {path}")
        raw_body = path.read_bytes()
        metadata = PlayerStatsCacheMetadata.model_validate_json(
            metadata_path.read_text()
        )
        if (
            metadata.endpoint != endpoint
            or metadata.season != season
            or metadata.season_type != season_type
            or metadata.team_id != team_id
        ):
            raise PlayerStatsError(f"Cached player metadata does not match path: {path}")
        if metadata.params != expected_params:
            return None
        if hashlib.sha256(raw_body).hexdigest() != metadata.sha256:
            raise PlayerStatsError(f"Cached player response hash mismatch: {path}")
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlayerStatsError(
                f"Cached player response is not valid JSON: {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise PlayerStatsError(
                f"Cached player response has an unexpected JSON root: {path}"
            )
        return PlayerStatsResponse(
            endpoint=endpoint,
            season=season,
            season_type=season_type,
            team_id=team_id,
            params=metadata.params,
            url=metadata.url,
            fetched_at=metadata.fetched_at,
            payload=payload,
            raw_body=raw_body,
        )

    def write(self, response: PlayerStatsResponse) -> Path:
        path = self.path_for(
            response.endpoint,
            response.season,
            response.season_type,
            response.team_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_body = response.raw_body
        if raw_body is None:
            raw_body = json.dumps(
                response.payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        _atomic_write_bytes(raw_body, path)
        metadata = PlayerStatsCacheMetadata(
            endpoint=response.endpoint,
            season=response.season,
            season_type=response.season_type,
            team_id=response.team_id,
            params=response.params,
            url=response.url,
            fetched_at=response.fetched_at,
            sha256=hashlib.sha256(raw_body).hexdigest(),
        )
        _atomic_write_text(
            metadata.model_dump_json(indent=2) + "\n",
            self.metadata_path_for(
                response.endpoint,
                response.season,
                response.season_type,
                response.team_id,
            ),
        )
        return path


class PlayerStatsClient:
    """Direct client for bulk NBA player index and season bio data."""

    def __init__(
        self,
        *,
        cache: PlayerStatsCache | None = None,
        http_client: httpx.Client | None = None,
        base_url: str = "https://stats.nba.com/stats",
        timeout: float = 60.0,
    ) -> None:
        self.cache = cache or PlayerStatsCache()
        self.base_url = base_url.rstrip("/")
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

    def __enter__(self) -> PlayerStatsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_player_index(
        self,
        season: str,
        *,
        use_cache: bool = True,
    ) -> PlayerStatsResponse:
        season = validate_season(season)
        params = _player_index_params(season)
        return self._fetch(
            PlayerStatsEndpoint.PLAYER_INDEX,
            season,
            None,
            params,
            use_cache=use_cache,
        )

    def fetch_player_season_bios(
        self,
        season: str,
        *,
        season_type: str = "regular",
        use_cache: bool = True,
    ) -> PlayerStatsResponse:
        season = validate_season(season)
        nba_season_type = _nba_season_type(season_type)
        params = _player_bio_params(season, nba_season_type)
        return self._fetch(
            PlayerStatsEndpoint.PLAYER_BIO_STATS,
            season,
            season_type,
            params,
            use_cache=use_cache,
        )

    def fetch_draft_history(
        self,
        season: str,
        *,
        use_cache: bool = True,
    ) -> PlayerStatsResponse:
        """Fetch one NBA Draft class directly from the NBA Stats endpoint."""

        season = validate_season(season)
        return self._fetch(
            PlayerStatsEndpoint.DRAFT_HISTORY,
            season,
            None,
            _draft_history_params(season),
            use_cache=use_cache,
        )

    def fetch_team_roster(
        self,
        season: str,
        team_id: int,
        *,
        use_cache: bool = True,
    ) -> PlayerStatsResponse:
        """Fetch one active team roster directly from NBA Stats."""

        season = validate_season(season)
        if team_id <= 0:
            raise ValueError("Team roster requires a positive team ID")
        return self._fetch(
            PlayerStatsEndpoint.TEAM_ROSTER,
            season,
            None,
            _team_roster_params(season, team_id),
            team_id=team_id,
            use_cache=use_cache,
        )

    def _fetch(
        self,
        endpoint: PlayerStatsEndpoint,
        season: str,
        season_type: str | None,
        params: dict[str, str],
        *,
        team_id: int | None = None,
        use_cache: bool,
    ) -> PlayerStatsResponse:
        if use_cache:
            cached = self.cache.read(
                endpoint,
                season,
                season_type,
                team_id,
                expected_params=params,
            )
            if cached is not None:
                return cached
        try:
            http_response = self._client.get(
                f"{self.base_url}/{endpoint.value}",
                params=params,
            )
            http_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise PlayerStatsError(
                f"NBA Stats returned HTTP {exc.response.status_code} "
                f"for {exc.request.url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise PlayerStatsError(
                f"NBA Stats player request failed for {endpoint.value}"
            ) from exc
        raw_body = http_response.content
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PlayerStatsError(
                f"NBA Stats returned non-JSON content for {endpoint.value}"
            ) from exc
        if not isinstance(payload, dict):
            raise PlayerStatsError(
                f"NBA Stats returned an unexpected JSON root for {endpoint.value}"
            )
        response = PlayerStatsResponse(
            endpoint=endpoint,
            season=season,
            season_type=season_type,
            team_id=team_id,
            params=params,
            url=str(http_response.url),
            payload=payload,
            raw_body=raw_body,
        )
        self.cache.write(response)
        return response


def _player_index_params(season: str) -> dict[str, str]:
    return {
        "College": "",
        "Country": "",
        "DraftPick": "",
        "DraftRound": "",
        "DraftYear": "",
        "Height": "",
        "Historical": "1",
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
        "TeamID": "0",
        "Weight": "",
    }


def _player_bio_params(season: str, nba_season_type: str) -> dict[str, str]:
    return {
        "College": "",
        "Conference": "",
        "Country": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "DraftPick": "",
        "DraftRound": "",
        "DraftYear": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": "0",
        "LeagueID": "00",
        "Location": "",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": nba_season_type,
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": "0",
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }


def _draft_history_params(season: str) -> dict[str, str]:
    return {
        "College": "",
        "LeagueID": "00",
        "OverallPick": "",
        "RoundNum": "",
        "RoundPick": "",
        "Season": season[:4],
        "TeamID": "0",
        "TopX": "",
    }


def _team_roster_params(season: str, team_id: int) -> dict[str, str]:
    return {"LeagueID": "00", "Season": season, "TeamID": str(team_id)}


def _nba_season_type(season_type: str) -> str:
    values = {
        "regular": "Regular Season",
        "playoffs": "Playoffs",
        "preseason": "Pre Season",
        "all_star": "All Star",
    }
    try:
        return values[season_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported player bio season type: {season_type}") from exc


def _atomic_write_bytes(content: bytes, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _atomic_write_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content)
    temporary_path.replace(path)
