from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

GAME_ID_RE = re.compile(r"^\d{10}$")
DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS = 15 * 60.0


class NbaCdnEndpoint(StrEnum):
    PLAY_BY_PLAY = "playbyplay"
    BOXSCORE = "boxscore"
    SCOREBOARD = "scoreboard"


class NbaCdnError(RuntimeError):
    """Raised when the NBA CDN returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        transient: bool = False,
        retry_after_seconds: float | None = None,
        content_type: str | None = None,
        response_body_preview: str | None = None,
        circuit_open: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.transient = transient
        self.retry_after_seconds = retry_after_seconds
        self.content_type = content_type
        self.response_body_preview = response_body_preview
        self.circuit_open = circuit_open


class NbaCdnRequestGate:
    """Coordinate request pacing and access-denial cooldown across worker threads."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        jitter_source: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._jitter_source = jitter_source
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._blocked_until = 0.0
        self._block_reason: str | None = None

    def wait_for_request_slot(
        self,
        *,
        minimum_interval_seconds: float,
        interval_jitter_seconds: float,
    ) -> None:
        """Reserve one request slot or fail fast while the CDN circuit is open."""

        while True:
            with self._lock:
                now = self._monotonic_clock()
                if now < self._blocked_until:
                    remaining = self._blocked_until - now
                    reason = self._block_reason or "an earlier access denial"
                    raise NbaCdnError(
                        "NBA CDN request circuit is open for "
                        f"{remaining:.1f} more seconds after {reason}",
                        transient=False,
                        retry_after_seconds=remaining,
                        circuit_open=True,
                    )
                if now >= self._next_request_at:
                    jitter = (
                        self._jitter_source(0.0, interval_jitter_seconds)
                        if interval_jitter_seconds > 0
                        else 0.0
                    )
                    self._next_request_at = now + minimum_interval_seconds + jitter
                    return
                delay = self._next_request_at - now
            self._sleeper(delay)

    def open_circuit(self, *, cooldown_seconds: float, reason: str) -> float:
        """Block new request slots for at least the requested cooldown."""

        with self._lock:
            now = self._monotonic_clock()
            self._blocked_until = max(self._blocked_until, now + cooldown_seconds)
            self._block_reason = reason
            return self._blocked_until - now


_DEFAULT_REQUEST_GATE = NbaCdnRequestGate()


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
        min_request_interval_seconds: float = 0.0,
        request_interval_jitter_seconds: float = 0.0,
        access_denial_cooldown_seconds: float = DEFAULT_ACCESS_DENIAL_COOLDOWN_SECONDS,
        request_gate: NbaCdnRequestGate | None = None,
    ) -> None:
        if min_request_interval_seconds < 0:
            raise ValueError("Minimum request interval cannot be negative")
        if request_interval_jitter_seconds < 0:
            raise ValueError("Request interval jitter cannot be negative")
        if access_denial_cooldown_seconds <= 0:
            raise ValueError("Access-denial cooldown must be positive")
        self.cache = cache or RawJsonCache()
        self.base_url = base_url.rstrip("/")
        self.min_request_interval_seconds = min_request_interval_seconds
        self.request_interval_jitter_seconds = request_interval_jitter_seconds
        self.access_denial_cooldown_seconds = access_denial_cooldown_seconds
        self._request_gate = request_gate or _DEFAULT_REQUEST_GATE
        self._owns_http_client = http_client is None
        self._client = http_client or self._new_http_client(timeout)

    @staticmethod
    def _new_http_client(timeout: float) -> httpx.Client:
        return httpx.Client(
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

    def __enter__(self) -> NbaCdnClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
        self._request_gate.wait_for_request_slot(
            minimum_interval_seconds=self.min_request_interval_seconds,
            interval_jitter_seconds=self.request_interval_jitter_seconds,
        )
        try:
            http_response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise NbaCdnError(
                f"NBA CDN request failed for {url}",
                url=url,
                transient=True,
            ) from exc

        if http_response.status_code != 200:
            raise self._response_error(http_response, url)

        raw_body = http_response.content
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            content_type = http_response.headers.get("content-type")
            preview = _response_body_preview(raw_body)
            raise NbaCdnError(
                f"NBA CDN returned non-JSON content for {url}"
                f"{_response_context(content_type, preview)}",
                url=url,
                content_type=content_type,
                response_body_preview=preview,
            ) from exc

        if not isinstance(payload, dict):
            raise NbaCdnError(f"NBA CDN returned unexpected JSON root for {url}")

        return CachedResponse(
            endpoint=endpoint,
            game_id=game_id,
            url=url,
            payload=payload,
            raw_body=raw_body,
        )

    def _response_error(self, response: httpx.Response, url: str) -> NbaCdnError:
        status_code = response.status_code
        content_type = response.headers.get("content-type")
        preview = _response_body_preview(response.content)
        retry_after_seconds = _parse_retry_after(response.headers.get("retry-after"))
        circuit_open = status_code in {403, 429}
        if circuit_open:
            cooldown_seconds = max(
                self.access_denial_cooldown_seconds,
                retry_after_seconds or 0.0,
            )
            retry_after_seconds = self._request_gate.open_circuit(
                cooldown_seconds=cooldown_seconds,
                reason=f"HTTP {status_code} from {url}",
            )
        reason = _response_reason(status_code, preview)
        message = (
            f"NBA CDN returned HTTP {status_code} for {url}; reason={reason}"
            f"{_response_context(content_type, preview)}"
        )
        if retry_after_seconds is not None:
            message += f"; retry_after_seconds={retry_after_seconds:.1f}"
        return NbaCdnError(
            message,
            status_code=status_code,
            url=url,
            transient=not circuit_open and (status_code in {408, 425} or status_code >= 500),
            retry_after_seconds=retry_after_seconds,
            content_type=content_type,
            response_body_preview=preview,
            circuit_open=circuit_open,
        )


def validate_game_id(game_id: str) -> str:
    if not GAME_ID_RE.match(game_id):
        raise ValueError(f"Expected a 10-digit NBA game_id, got {game_id!r}")
    return game_id


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _response_body_preview(body: bytes, *, limit: int = 200) -> str:
    text = body.decode("utf-8", errors="replace")
    return " ".join(text.split())[:limit]


def _response_reason(status_code: int, preview: str) -> str:
    normalized = preview.casefold()
    if status_code == 403 and (
        "access denied" in normalized or "permission to access" in normalized
    ):
        return "access_denied"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "upstream_error"
    return "http_error"


def _response_context(content_type: str | None, preview: str) -> str:
    context = f"; content_type={content_type}" if content_type else ""
    if preview:
        context += f"; body_preview={preview!r}"
    return context


def _is_legacy_cache_envelope(document: Any) -> bool:
    return (
        isinstance(document, dict)
        and "endpoint" in document
        and "payload" in document
        and isinstance(document["payload"], dict)
    )
