from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

GAME_ID_RE = re.compile(r"^\d{10}$")


class NbaCdnEndpoint(StrEnum):
    PLAY_BY_PLAY = "playbyplay"
    BOXSCORE = "boxscore"
    SCOREBOARD = "scoreboard"


class NbaCdnError(RuntimeError):
    """Raised when the NBA CDN returns an unusable response."""


class CachedResponse(BaseModel):
    endpoint: NbaCdnEndpoint
    game_id: str | None = None
    url: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any]
    raw_body: bytes | None = Field(default=None, exclude=True, repr=False)


class CacheMetadata(BaseModel):
    endpoint: NbaCdnEndpoint
    game_id: str | None = None
    url: str
    fetched_at: datetime
    sha256: str


class RawJsonCache:
    """Byte-preserving file cache for raw NBA source responses."""

    def __init__(self, root: Path | str = Path("data/raw")) -> None:
        self.root = Path(root)

    def path_for(self, endpoint: NbaCdnEndpoint, game_id: str | None = None) -> Path:
        if endpoint is NbaCdnEndpoint.SCOREBOARD:
            return self.root / endpoint.value / "todays_scoreboard_00.json"
        if game_id is None:
            raise ValueError(f"{endpoint.value} cache paths require a game_id")
        return self.root / endpoint.value / f"{game_id}.json"

    def metadata_path_for(
        self,
        endpoint: NbaCdnEndpoint,
        game_id: str | None = None,
    ) -> Path:
        return self.path_for(endpoint, game_id).with_suffix(".meta.json")

    def read(self, endpoint: NbaCdnEndpoint, game_id: str | None = None) -> CachedResponse | None:
        path = self.path_for(endpoint, game_id)
        if not path.exists():
            return None
        raw_body = path.read_bytes()
        try:
            document = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NbaCdnError(f"Cached NBA response is not valid JSON: {path}") from exc

        if _is_legacy_cache_envelope(document):
            return CachedResponse.model_validate(document)
        if not isinstance(document, dict):
            raise NbaCdnError(f"Cached NBA response has an unexpected JSON root: {path}")

        metadata_path = self.metadata_path_for(endpoint, game_id)
        if metadata_path.exists():
            metadata = CacheMetadata.model_validate_json(metadata_path.read_text())
            actual_sha256 = hashlib.sha256(raw_body).hexdigest()
            if metadata.sha256 != actual_sha256:
                raise NbaCdnError(f"Cached NBA response hash mismatch: {path}")
            if metadata.endpoint != endpoint or metadata.game_id != game_id:
                raise NbaCdnError(f"Cached NBA response metadata does not match path: {path}")
        else:
            metadata = CacheMetadata(
                endpoint=endpoint,
                game_id=game_id,
                url="",
                fetched_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
                sha256=hashlib.sha256(raw_body).hexdigest(),
            )

        return CachedResponse(
            endpoint=metadata.endpoint,
            game_id=metadata.game_id,
            url=metadata.url,
            fetched_at=metadata.fetched_at,
            payload=document,
            raw_body=raw_body,
        )

    def write(self, response: CachedResponse) -> Path:
        path = self.path_for(response.endpoint, response.game_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_body = response.raw_body
        if raw_body is None:
            raw_body = json.dumps(
                response.payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_bytes(raw_body)
        tmp_path.replace(path)

        metadata = CacheMetadata(
            endpoint=response.endpoint,
            game_id=response.game_id,
            url=response.url,
            fetched_at=response.fetched_at,
            sha256=hashlib.sha256(raw_body).hexdigest(),
        )
        metadata_path = self.metadata_path_for(response.endpoint, response.game_id)
        metadata_tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        metadata_tmp_path.write_text(metadata.model_dump_json(indent=2))
        metadata_tmp_path.replace(metadata_path)
        return path


class NbaCdnClient:
    """Small direct client for NBA CDN liveData endpoints."""

    def __init__(
        self,
        *,
        cache: RawJsonCache | None = None,
        http_client: httpx.Client | None = None,
        base_url: str = "https://cdn.nba.com/static/json/liveData",
        timeout: float = 30.0,
    ) -> None:
        self.cache = cache or RawJsonCache()
        self.base_url = base_url.rstrip("/")
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

    def fetch_play_by_play(self, game_id: str, *, use_cache: bool = True) -> CachedResponse:
        return self._fetch_game_endpoint(NbaCdnEndpoint.PLAY_BY_PLAY, game_id, use_cache=use_cache)

    def fetch_boxscore(self, game_id: str, *, use_cache: bool = True) -> CachedResponse:
        return self._fetch_game_endpoint(NbaCdnEndpoint.BOXSCORE, game_id, use_cache=use_cache)

    def fetch_todays_scoreboard(self, *, use_cache: bool = False) -> CachedResponse:
        endpoint = NbaCdnEndpoint.SCOREBOARD
        if use_cache:
            cached = self.cache.read(endpoint)
            if cached is not None:
                return cached

        url = f"{self.base_url}/scoreboard/todaysScoreboard_00.json"
        response = self._request(endpoint, url)
        self.cache.write(response)
        return response

    def _fetch_game_endpoint(
        self,
        endpoint: NbaCdnEndpoint,
        game_id: str,
        *,
        use_cache: bool,
    ) -> CachedResponse:
        validate_game_id(game_id)
        if use_cache:
            cached = self.cache.read(endpoint, game_id)
            if cached is not None:
                return cached

        if endpoint is NbaCdnEndpoint.PLAY_BY_PLAY:
            url = f"{self.base_url}/playbyplay/playbyplay_{game_id}.json"
        elif endpoint is NbaCdnEndpoint.BOXSCORE:
            url = f"{self.base_url}/boxscore/boxscore_{game_id}.json"
        else:
            raise ValueError(f"Unsupported game endpoint: {endpoint}")

        response = self._request(endpoint, url, game_id=game_id)
        self.cache.write(response)
        return response

    def _request(
        self,
        endpoint: NbaCdnEndpoint,
        url: str,
        *,
        game_id: str | None = None,
    ) -> CachedResponse:
        try:
            http_response = self._client.get(url)
            http_response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise NbaCdnError(
                f"NBA CDN returned HTTP {exc.response.status_code} for {url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise NbaCdnError(f"NBA CDN request failed for {url}") from exc

        raw_body = http_response.content
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NbaCdnError(f"NBA CDN returned non-JSON content for {url}") from exc

        if not isinstance(payload, dict):
            raise NbaCdnError(f"NBA CDN returned unexpected JSON root for {url}")

        return CachedResponse(
            endpoint=endpoint,
            game_id=game_id,
            url=url,
            payload=payload,
            raw_body=raw_body,
        )


def validate_game_id(game_id: str) -> str:
    if not GAME_ID_RE.match(game_id):
        raise ValueError(f"Expected a 10-digit NBA game_id, got {game_id!r}")
    return game_id


def _is_legacy_cache_envelope(document: Any) -> bool:
    return (
        isinstance(document, dict)
        and "endpoint" in document
        and "payload" in document
        and isinstance(document["payload"], dict)
    )
