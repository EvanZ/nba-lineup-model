from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from nba_lineup_model.ingest.nba_cdn import validate_game_id

DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS = 15 * 60.0


class NbaStatsEndpoint(StrEnum):
    """Direct NBA Stats game endpoints retained by the historical cache."""

    PLAY_BY_PLAY_V3 = "playbyplayv3"
    BOXSCORE_TRADITIONAL_V3 = "boxscoretraditionalv3"
    GAME_ROTATION = "gamerotation"


class NbaStatsError(RuntimeError):
    """Raised when an NBA Stats response cannot be retained as a valid artifact."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: NbaStatsEndpoint | None = None,
        status_code: int | None = None,
        url: str | None = None,
        transient: bool = False,
        retry_after_seconds: float | None = None,
        content_type: str | None = None,
        response_body_preview: str | None = None,
        circuit_open: bool = False,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.url = url
        self.transient = transient
        self.retry_after_seconds = retry_after_seconds
        self.content_type = content_type
        self.response_body_preview = response_body_preview
        self.circuit_open = circuit_open


class NbaStatsRequestGate:
    """Coordinate Stats request pacing and access-denial cooldown across threads."""

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
        """Reserve one process-wide request slot or fail while the circuit is open."""

        while True:
            with self._lock:
                now = self._monotonic_clock()
                if now < self._blocked_until:
                    remaining = self._blocked_until - now
                    reason = self._block_reason or "an earlier access denial"
                    raise NbaStatsError(
                        "NBA Stats request circuit is open for "
                        f"{remaining:.1f} more seconds after {reason}",
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


_DEFAULT_REQUEST_GATE = NbaStatsRequestGate()


class StatsCachedResponse(BaseModel):
    """Parsed NBA Stats response plus the exact bytes received from the source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: NbaStatsEndpoint
    game_id: str
    url: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    response_headers: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any]
    raw_body: bytes = Field(exclude=True, repr=False)


class StatsCacheMetadata(BaseModel):
    """Provenance sidecar for one byte-preserved NBA Stats response."""

    model_config = ConfigDict(strict=True, extra="forbid")

    endpoint: NbaStatsEndpoint
    game_id: str
    url: str
    fetched_at: datetime
    sha256: str
    response_headers: dict[str, str] = Field(default_factory=dict)


class NbaStatsRawCache:
    """Byte-preserving cache isolated from the NBA liveData CDN namespace."""

    def __init__(self, root: Path | str = Path("data/raw/stats")) -> None:
        self.root = Path(root)

    def path_for(self, endpoint: NbaStatsEndpoint, game_id: str) -> Path:
        validate_game_id(game_id)
        return self.root / endpoint.value / f"{game_id}.json"

    def metadata_path_for(self, endpoint: NbaStatsEndpoint, game_id: str) -> Path:
        return self.path_for(endpoint, game_id).with_suffix(".meta.json")

    def read(
        self,
        endpoint: NbaStatsEndpoint,
        game_id: str,
    ) -> StatsCachedResponse | None:
        path = self.path_for(endpoint, game_id)
        if not path.exists():
            return None
        metadata_path = self.metadata_path_for(endpoint, game_id)
        if not metadata_path.exists():
            raise NbaStatsError(f"Cached NBA Stats response is missing metadata: {path}")

        raw_body = path.read_bytes()
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise NbaStatsError(f"Cached NBA Stats response is not valid JSON: {path}") from error
        if not isinstance(payload, dict):
            raise NbaStatsError(f"Cached NBA Stats response has an unexpected root: {path}")

        metadata = StatsCacheMetadata.model_validate_json(metadata_path.read_text())
        actual_sha256 = hashlib.sha256(raw_body).hexdigest()
        if metadata.sha256 != actual_sha256:
            raise NbaStatsError(f"Cached NBA Stats response hash mismatch: {path}")
        if metadata.endpoint is not endpoint or metadata.game_id != game_id:
            raise NbaStatsError(f"Cached NBA Stats metadata does not match path: {path}")
        validate_stats_payload(payload, endpoint, game_id)
        return StatsCachedResponse(
            endpoint=endpoint,
            game_id=game_id,
            url=metadata.url,
            fetched_at=metadata.fetched_at,
            response_headers=metadata.response_headers,
            payload=payload,
            raw_body=raw_body,
        )

    def write(self, response: StatsCachedResponse) -> Path:
        validate_stats_payload(response.payload, response.endpoint, response.game_id)
        path = self.path_for(response.endpoint, response.game_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_bytes(response.raw_body)
        temporary_path.replace(path)

        metadata = StatsCacheMetadata(
            endpoint=response.endpoint,
            game_id=response.game_id,
            url=response.url,
            fetched_at=response.fetched_at,
            sha256=hashlib.sha256(response.raw_body).hexdigest(),
            response_headers=response.response_headers,
        )
        metadata_path = self.metadata_path_for(response.endpoint, response.game_id)
        metadata_temporary_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        metadata_temporary_path.write_text(metadata.model_dump_json(indent=2) + "\n")
        metadata_temporary_path.replace(metadata_path)
        return path


class NbaStatsClient:
    """Direct client for archived NBA Stats game endpoints."""

    def __init__(
        self,
        *,
        cache: NbaStatsRawCache | None = None,
        http_client: httpx.Client | None = None,
        base_url: str = "https://stats.nba.com/stats",
        timeout: float = 60.0,
        min_request_interval_seconds: float = 0.0,
        request_interval_jitter_seconds: float = 0.0,
        access_denial_cooldown_seconds: float = (
            DEFAULT_STATS_ACCESS_DENIAL_COOLDOWN_SECONDS
        ),
        request_gate: NbaStatsRequestGate | None = None,
    ) -> None:
        if min_request_interval_seconds < 0:
            raise ValueError("Minimum request interval cannot be negative")
        if request_interval_jitter_seconds < 0:
            raise ValueError("Request interval jitter cannot be negative")
        if access_denial_cooldown_seconds <= 0:
            raise ValueError("Access-denial cooldown must be positive")
        self.cache = cache or NbaStatsRawCache()
        self.base_url = base_url.rstrip("/")
        self.min_request_interval_seconds = min_request_interval_seconds
        self.request_interval_jitter_seconds = request_interval_jitter_seconds
        self.access_denial_cooldown_seconds = access_denial_cooldown_seconds
        self._request_gate = request_gate or _DEFAULT_REQUEST_GATE
        self._owns_http_client = http_client is None
        self._client = http_client or self._new_http_client(timeout)
        self._client.headers.update(_stats_headers())

    @staticmethod
    def _new_http_client(timeout: float) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            headers=_stats_headers(),
            follow_redirects=True,
        )

    def close(self) -> None:
        if self._owns_http_client:
            self._client.close()

    def __enter__(self) -> NbaStatsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(
        self,
        endpoint: NbaStatsEndpoint,
        game_id: str,
        *,
        use_cache: bool = True,
    ) -> StatsCachedResponse:
        validate_game_id(game_id)
        if use_cache:
            cached = self.cache.read(endpoint, game_id)
            if cached is not None:
                return cached

        url = f"{self.base_url}/{endpoint.value}"
        self._request_gate.wait_for_request_slot(
            minimum_interval_seconds=self.min_request_interval_seconds,
            interval_jitter_seconds=self.request_interval_jitter_seconds,
        )
        try:
            http_response = self._client.get(
                url,
                params=stats_endpoint_parameters(endpoint, game_id),
            )
        except httpx.HTTPError as error:
            raise NbaStatsError(
                f"NBA Stats request failed for {url}",
                endpoint=endpoint,
                url=url,
                transient=True,
            ) from error

        if http_response.status_code != 200:
            raise self._response_error(endpoint, http_response)

        raw_body = http_response.content
        try:
            payload = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            content_type = http_response.headers.get("content-type")
            preview = _response_body_preview(raw_body)
            raise NbaStatsError(
                f"NBA Stats returned non-JSON content for {http_response.request.url}"
                f"{_response_context(content_type, preview)}",
                endpoint=endpoint,
                url=str(http_response.request.url),
                content_type=content_type,
                response_body_preview=preview,
            ) from error
        if not isinstance(payload, dict):
            raise NbaStatsError(
                f"NBA Stats returned an unexpected JSON root for {http_response.request.url}",
                endpoint=endpoint,
                url=str(http_response.request.url),
            )
        validate_stats_payload(payload, endpoint, game_id)

        response = StatsCachedResponse(
            endpoint=endpoint,
            game_id=game_id,
            url=str(http_response.request.url),
            response_headers=_provenance_headers(http_response.headers),
            payload=payload,
            raw_body=raw_body,
        )
        self.cache.write(response)
        return response

    def _response_error(
        self,
        endpoint: NbaStatsEndpoint,
        response: httpx.Response,
    ) -> NbaStatsError:
        status_code = response.status_code
        url = str(response.request.url)
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
        message = (
            f"NBA Stats returned HTTP {status_code} for {url}"
            f"{_response_context(content_type, preview)}"
        )
        if retry_after_seconds is not None:
            message += f"; retry_after_seconds={retry_after_seconds:.1f}"
        return NbaStatsError(
            message,
            endpoint=endpoint,
            status_code=status_code,
            url=url,
            transient=not circuit_open and (status_code in {408, 425} or status_code >= 500),
            retry_after_seconds=retry_after_seconds,
            content_type=content_type,
            response_body_preview=preview,
            circuit_open=circuit_open,
        )


def stats_endpoint_parameters(
    endpoint: NbaStatsEndpoint,
    game_id: str,
) -> dict[str, str]:
    """Return the explicit official query contract for one endpoint."""

    validate_game_id(game_id)
    if endpoint is NbaStatsEndpoint.PLAY_BY_PLAY_V3:
        return {
            "GameID": game_id,
            "StartPeriod": "0",
            "EndPeriod": "14",
        }
    if endpoint is NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3:
        return {
            "GameID": game_id,
            "StartPeriod": "0",
            "EndPeriod": "0",
            "StartRange": "0",
            "EndRange": "0",
            "RangeType": "0",
        }
    if endpoint is NbaStatsEndpoint.GAME_ROTATION:
        return {
            "GameID": game_id,
            "LeagueID": "00",
        }
    raise ValueError(f"Unsupported NBA Stats endpoint: {endpoint}")


def validate_stats_payload(
    payload: Mapping[str, Any],
    endpoint: NbaStatsEndpoint,
    game_id: str,
) -> None:
    """Validate identity and minimum endpoint structure before caching."""

    if endpoint is NbaStatsEndpoint.PLAY_BY_PLAY_V3:
        game = payload.get("game")
        if not isinstance(game, dict) or game.get("gameId") != game_id:
            raise NbaStatsError(f"NBA {endpoint.value} response gameId does not match {game_id}")
        if not isinstance(game.get("actions"), list):
            raise NbaStatsError(f"NBA {endpoint.value} response is missing actions")
        return

    if endpoint is NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3:
        boxscore = payload.get("boxScoreTraditional")
        if not isinstance(boxscore, dict) or boxscore.get("gameId") != game_id:
            raise NbaStatsError(f"NBA {endpoint.value} response gameId does not match {game_id}")
        if not isinstance(boxscore.get("homeTeam"), dict) or not isinstance(
            boxscore.get("awayTeam"),
            dict,
        ):
            raise NbaStatsError(f"NBA {endpoint.value} response is missing teams")
        return

    if endpoint is NbaStatsEndpoint.GAME_ROTATION:
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("GameID") != game_id:
            raise NbaStatsError(f"NBA {endpoint.value} response GameID does not match {game_id}")
        result_sets = payload.get("resultSets")
        if not isinstance(result_sets, list):
            raise NbaStatsError(f"NBA {endpoint.value} response is missing resultSets")
        names = {
            result_set.get("name")
            for result_set in result_sets
            if isinstance(result_set, dict)
        }
        if not {"AwayTeam", "HomeTeam"}.issubset(names):
            raise NbaStatsError(f"NBA {endpoint.value} response is missing team rotations")
        return

    raise ValueError(f"Unsupported NBA Stats endpoint: {endpoint}")


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
    return " ".join(body.decode("utf-8", errors="replace").split())[:limit]


def _response_context(content_type: str | None, preview: str) -> str:
    context = f"; content_type={content_type}" if content_type else ""
    if preview:
        context += f"; body_preview={preview!r}"
    return context


def _provenance_headers(headers: httpx.Headers) -> dict[str, str]:
    retained = (
        "cache-control",
        "content-type",
        "date",
        "etag",
        "last-modified",
        "x-datasource",
        "x-etag",
    )
    return {name: headers[name] for name in retained if name in headers}


def _stats_headers() -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Referer": "https://www.nba.com/",
        "Sec-Ch-Ua": (
            '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"'
        ),
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Fetch-Dest": "empty",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
    }
