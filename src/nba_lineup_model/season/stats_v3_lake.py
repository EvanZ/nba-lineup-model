from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from nba_lineup_model.ingest.nba_cdn import validate_game_id
from nba_lineup_model.ingest.nba_stats import (
    NbaStatsClient,
    NbaStatsEndpoint,
    NbaStatsError,
    NbaStatsRawCache,
    StatsCachedResponse,
    stats_endpoint_parameters,
)
from nba_lineup_model.season.schedule import (
    NbaScheduleClient,
    SeasonScheduleCache,
    catalog_from_schedule,
)
from nba_lineup_model.season.schema import validate_season

DEFAULT_STATS_V3_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_STATS_V3_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_STATS_V3_COVERAGE_PATH = Path("/var/lib/nba-gestalt/stats-v3/coverage.json")
DEFAULT_STATS_V3_SEASON = "2026-27"
EASTERN = ZoneInfo("America/New_York")
REQUIRED_ENDPOINTS = (
    NbaStatsEndpoint.PLAY_BY_PLAY_V3,
    NbaStatsEndpoint.BOXSCORE_TRADITIONAL_V3,
)


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Datetime values must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"S3 URI is invalid: {uri}")
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"S3 URI is invalid: {uri}")
    return bucket, key.rstrip("/")


def _join_s3_uri(base_uri: str, *parts: str) -> str:
    bucket, key = _parse_s3_uri(base_uri)
    clean = [part.strip("/") for part in parts]
    if any(not part or ".." in Path(part).parts for part in clean):
        raise ValueError(f"S3 key part is invalid: {parts}")
    return f"s3://{bucket}/{'/'.join([key, *clean])}"


@dataclass(frozen=True)
class CompletedScheduleGame:
    """Minimal final-game identity retained by the serving schedule state."""

    game_id: str
    espn_game_id: str
    season: str
    game_date: date
    start_time: datetime
    final_observed_at: datetime
    away_team: str
    home_team: str

    def __post_init__(self) -> None:
        validate_game_id(self.game_id)
        validate_season(self.season)
        if self.start_time.tzinfo is None or self.start_time.utcoffset() is None:
            raise ValueError("Game start time must include a timezone")
        if self.final_observed_at.tzinfo is None or self.final_observed_at.utcoffset() is None:
            raise ValueError("Final observation time must include a timezone")

    @property
    def season_type(self) -> str:
        return {
            "001": "preseason",
            "002": "regular",
            "004": "playoffs",
            "005": "play_in",
            "006": "nba_cup_final",
        }.get(self.game_id[:3], "unknown")


def discover_completed_schedule_games(
    state_dir: Path | str,
    *,
    season: str,
    start_date: date | None = None,
    end_date: date | None = None,
    game_ids: Sequence[str] | None = None,
    now: datetime | None = None,
    grace_period: timedelta = timedelta(hours=1),
    ignore_grace: bool = False,
) -> list[CompletedScheduleGame]:
    """Read final games from compact schedule state without hitting an API."""

    season = validate_season(season)
    selected_ids = {validate_game_id(game_id) for game_id in game_ids or ()}
    current = (now or datetime.now(UTC)).astimezone(UTC)
    root = Path(state_dir)
    latest_dir = root / "latest"
    games_by_id: dict[str, CompletedScheduleGame] = {}
    if not latest_dir.is_dir():
        return []

    for path in sorted(latest_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or str(payload.get("season")) != season:
            continue
        payload_date = _date_value(payload.get("game_date")) or _date_value(path.stem)
        if payload_date is None:
            continue
        if start_date is not None and payload_date < start_date:
            continue
        if end_date is not None and payload_date > end_date:
            continue
        for row in payload.get("games", []):
            if not isinstance(row, dict):
                continue
            game_id = str(row.get("game_id") or "")
            try:
                validate_game_id(game_id)
            except ValueError:
                continue
            if selected_ids and game_id not in selected_ids:
                continue
            status = row.get("status")
            if not isinstance(status, dict) or not (
                bool(status.get("completed")) or status.get("state") == "post"
            ):
                continue
            start_time = _parse_timestamp(row.get("start_time"))
            if start_time is None:
                continue
            espn_game_id = str(row.get("espn_game_id") or "")
            final_observed_at = _parse_timestamp(row.get("final_observed_at"))
            if final_observed_at is None and espn_game_id:
                final_observed_at = _final_archive_timestamp(root, espn_game_id)
            if final_observed_at is None:
                # Existing snapshots predate explicit final observation. Four hours
                # after tip is a conservative one-hour postgame approximation.
                final_observed_at = start_time + timedelta(hours=4)
            game = CompletedScheduleGame(
                game_id=game_id,
                espn_game_id=espn_game_id,
                season=season,
                game_date=payload_date,
                start_time=start_time,
                final_observed_at=final_observed_at,
                away_team=str(row.get("away_team") or ""),
                home_team=str(row.get("home_team") or ""),
            )
            if ignore_grace or current >= game.final_observed_at + grace_period:
                games_by_id[game_id] = game
    return sorted(games_by_id.values(), key=lambda game: (game.game_date, game.game_id))


def _date_value(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _final_archive_timestamp(root: Path, espn_game_id: str) -> datetime | None:
    path = root / "final" / f"{espn_game_id}.summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return _parse_timestamp(payload.get("archived_at")) if isinstance(payload, dict) else None


def discover_completed_nba_schedule_games(
    *,
    season: str,
    start_date: date | None = None,
    end_date: date | None = None,
    game_ids: Sequence[str] | None = None,
) -> list[CompletedScheduleGame]:
    """Refresh the official season catalog for a workstation-run backfill."""

    season = validate_season(season)
    selected_ids = {validate_game_id(game_id) for game_id in game_ids or ()}
    with tempfile.TemporaryDirectory(prefix="gestalt-stats-schedule-") as directory:
        with NbaScheduleClient(cache=SeasonScheduleCache(directory)) as client:
            catalog = catalog_from_schedule(client.fetch(season, use_cache=False))
    games = []
    for game in catalog.games:
        if game.game_status != "final" or game.game_time_utc is None:
            continue
        if start_date is not None and game.game_date < start_date:
            continue
        if end_date is not None and game.game_date > end_date:
            continue
        if selected_ids and game.game_id not in selected_ids:
            continue
        games.append(
            CompletedScheduleGame(
                game_id=game.game_id,
                espn_game_id="",
                season=season,
                game_date=game.game_date,
                start_time=game.game_time_utc,
                final_observed_at=game.source_fetched_at,
                away_team=game.away_team_tricode,
                home_team=game.home_team_tricode,
            )
        )
    return sorted(games, key=lambda game: (game.game_date, game.game_id))


class ImmutableObjectStore(Protocol):
    def list_keys(self, prefix_uri: str) -> list[str]: ...

    def put_bytes(self, uri: str, payload: bytes) -> bool: ...


class AwsCliImmutableObjectStore:
    """Small S3 adapter that reuses the instance's AWS CLI credential chain."""

    def list_keys(self, prefix_uri: str) -> list[str]:
        bucket, prefix = _parse_s3_uri(prefix_uri)
        result = subprocess.run(
            [
                "aws",
                "s3api",
                "list-objects-v2",
                "--bucket",
                bucket,
                "--prefix",
                prefix,
                "--query",
                "Contents[].Key",
                "--output",
                "json",
            ],
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout or b"[]")
        return [str(key) for key in payload or []]

    def put_bytes(self, uri: str, payload: bytes) -> bool:
        digest = hashlib.sha256(payload).hexdigest()
        checksum = base64.b64encode(bytes.fromhex(digest)).decode()
        bucket, key = _parse_s3_uri(uri)
        with tempfile.TemporaryDirectory(prefix="gestalt-s3-") as directory:
            source = Path(directory) / "object"
            source.write_bytes(payload)
            result = subprocess.run(
                [
                    "aws",
                    "s3api",
                    "put-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--body",
                    str(source),
                    "--if-none-match",
                    "*",
                    "--metadata",
                    f"sha256={digest}",
                    "--checksum-algorithm",
                    "SHA256",
                    "--checksum-sha256",
                    checksum,
                    "--output",
                    "json",
                ],
                capture_output=True,
            )
            error_text = result.stderr.decode(errors="replace").strip()
            already_exists = result.returncode != 0 and (
                "PreconditionFailed" in error_text or "412" in error_text
            )
            if result.returncode != 0 and not already_exists:
                raise RuntimeError(
                    f"Unable to create immutable S3 object {uri} "
                    f"(exit {result.returncode}): {error_text}"
                )
            remote_path = Path(directory) / "remote"
            verification = subprocess.run(
                [
                    "aws",
                    "s3api",
                    "get-object",
                    "--bucket",
                    bucket,
                    "--key",
                    key,
                    "--checksum-mode",
                    "ENABLED",
                    "--output",
                    "json",
                    str(remote_path),
                ],
                capture_output=True,
            )
            if verification.returncode != 0:
                error_text = verification.stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"Unable to verify immutable S3 object {uri} "
                    f"(exit {verification.returncode}): {error_text}"
                )
            remote = remote_path.read_bytes()
        if hashlib.sha256(remote).hexdigest() != digest or remote != payload:
            raise RuntimeError(f"Immutable S3 verification failed for {uri}")
        return not already_exists


@dataclass(frozen=True)
class ArchiveOutcome:
    endpoint: NbaStatsEndpoint
    sha256: str
    byte_count: int
    disposition: str
    raw_uri: str
    record_uri: str | None
    prior_sha256: str | None


class StatsV3LakeArchive:
    """Content-addressed Stats V3 archive plus append-only query records."""

    def __init__(
        self,
        base_uri: str = DEFAULT_STATS_V3_LAKE_URI,
        *,
        store: ImmutableObjectStore | None = None,
    ) -> None:
        _parse_s3_uri(base_uri)
        self.base_uri = base_uri.rstrip("/")
        self.store = store or AwsCliImmutableObjectStore()

    def inventory(self, season: str) -> dict[tuple[str, NbaStatsEndpoint], list[str]]:
        season = validate_season(season)
        found: dict[tuple[str, NbaStatsEndpoint], list[str]] = {}
        for endpoint in REQUIRED_ENDPOINTS:
            prefix_uri = _join_s3_uri(
                self.base_uri,
                "raw",
                "stats",
                endpoint.value,
                f"season={season}",
            )
            _, root_key = _parse_s3_uri(prefix_uri)
            root_prefix = root_key.rstrip("/") + "/"
            for key in self.store.list_keys(prefix_uri):
                relative = key.removeprefix(root_prefix)
                parts = relative.split("/")
                if len(parts) != 2 or not parts[0].startswith("game_id="):
                    continue
                filename = parts[1]
                if not filename.endswith(".json") or filename.endswith(".meta.json"):
                    continue
                digest = filename.removesuffix(".json")
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    continue
                game_id = parts[0].removeprefix("game_id=")
                try:
                    validate_game_id(game_id)
                except ValueError:
                    continue
                found.setdefault((game_id, endpoint), []).append(digest)
        return {key: sorted(set(values)) for key, values in found.items()}

    def archive(
        self,
        game: CompletedScheduleGame,
        response: StatsCachedResponse,
        *,
        run_id: str,
        attempt_number: int,
        known_sha256: Sequence[str],
    ) -> ArchiveOutcome:
        if response.game_id != game.game_id:
            raise ValueError("Stats response game ID does not match the selected game")
        digest = hashlib.sha256(response.raw_body).hexdigest()
        raw_uri = self._raw_uri(game, response.endpoint, digest)
        known = list(dict.fromkeys(known_sha256))
        prior = known[-1] if known else None
        if digest in known:
            return ArchiveOutcome(
                endpoint=response.endpoint,
                sha256=digest,
                byte_count=len(response.raw_body),
                disposition="unchanged",
                raw_uri=raw_uri,
                record_uri=None,
                prior_sha256=prior,
            )

        self.store.put_bytes(raw_uri, response.raw_body)
        metadata = self._record_payload(
            game,
            response.endpoint,
            run_id=run_id,
            attempt_number=attempt_number,
            status="succeeded",
            started_at=response.fetched_at,
            finished_at=datetime.now(UTC),
            source_url=response.url,
            response_headers=response.response_headers,
            sha256=digest,
            byte_count=len(response.raw_body),
            raw_uri=raw_uri,
            prior_sha256=prior,
        )
        metadata_uri = raw_uri.removesuffix(".json") + ".meta.json"
        self.store.put_bytes(metadata_uri, _canonical_json(metadata))
        record_uri = self._record_uri(game, response.endpoint, f"success-{digest}.json")
        self.store.put_bytes(record_uri, _canonical_json(metadata))
        return ArchiveOutcome(
            endpoint=response.endpoint,
            sha256=digest,
            byte_count=len(response.raw_body),
            disposition="revised" if prior else "archived",
            raw_uri=raw_uri,
            record_uri=record_uri,
            prior_sha256=prior,
        )

    def record_failure(
        self,
        game: CompletedScheduleGame,
        endpoint: NbaStatsEndpoint,
        *,
        run_id: str,
        attempt_number: int,
        started_at: datetime,
        error: Exception,
    ) -> str:
        finished_at = datetime.now(UTC)
        payload = self._record_payload(
            game,
            endpoint,
            run_id=run_id,
            attempt_number=attempt_number,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        record_uri = self._record_uri(
            game,
            endpoint,
            f"failure-{run_id}-{attempt_number}.json",
        )
        self.store.put_bytes(record_uri, _canonical_json(payload))
        return record_uri

    def write_run_summary(self, run_id: str, payload: Mapping[str, Any]) -> str:
        uri = _join_s3_uri(
            self.base_uri,
            "manifests",
            "stats_v3",
            "runs",
            f"{run_id}.json",
        )
        self.store.put_bytes(uri, _canonical_json(payload))
        return uri

    def _raw_uri(
        self,
        game: CompletedScheduleGame,
        endpoint: NbaStatsEndpoint,
        digest: str,
    ) -> str:
        return _join_s3_uri(
            self.base_uri,
            "raw",
            "stats",
            endpoint.value,
            f"season={game.season}",
            f"game_id={game.game_id}",
            f"{digest}.json",
        )

    def _record_uri(
        self,
        game: CompletedScheduleGame,
        endpoint: NbaStatsEndpoint,
        filename: str,
    ) -> str:
        return _join_s3_uri(
            self.base_uri,
            "manifests",
            "stats_v3",
            "records",
            f"season={game.season}",
            f"game_date={game.game_date.isoformat()}",
            f"game_id={game.game_id}",
            f"endpoint={endpoint.value}",
            filename,
        )

    @staticmethod
    def _record_payload(
        game: CompletedScheduleGame,
        endpoint: NbaStatsEndpoint,
        *,
        run_id: str,
        attempt_number: int,
        status: str,
        started_at: datetime,
        finished_at: datetime,
        source_url: str | None = None,
        response_headers: Mapping[str, str] | None = None,
        sha256: str | None = None,
        byte_count: int | None = None,
        raw_uri: str | None = None,
        prior_sha256: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_number": attempt_number,
            "game_id": game.game_id,
            "espn_game_id": game.espn_game_id,
            "season": game.season,
            "season_type": game.season_type,
            "game_date": game.game_date.isoformat(),
            "away_team": game.away_team,
            "home_team": game.home_team,
            "endpoint": endpoint.value,
            "source_url": source_url,
            "source_parameters": stats_endpoint_parameters(endpoint, game.game_id),
            "response_headers": dict(response_headers or {}),
            "started_at": _utc_timestamp(started_at),
            "finished_at": _utc_timestamp(finished_at),
            "duration_seconds": max(0.0, (finished_at - started_at).total_seconds()),
            "status": status,
            "sha256": sha256,
            "byte_count": byte_count,
            "raw_uri": raw_uri,
            "prior_sha256": prior_sha256,
            "error_type": error_type,
            "error_message": _redacted_error_message(error_message),
        }


class StatsV3BatchError(RuntimeError):
    def __init__(self, summary: Mapping[str, Any]) -> None:
        self.summary = dict(summary)
        super().__init__(f"Stats V3 ingestion failed for {summary['failed_count']} endpoints")


FetchResponse = Callable[[NbaStatsEndpoint, str], StatsCachedResponse]


def read_coverage_index(
    path: Path | str,
    *,
    season: str,
) -> dict[tuple[str, NbaStatsEndpoint], list[str]] | None:
    """Read the compact local accelerator; S3 remains the durable authority."""

    source = Path(path)
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("season") != validate_season(season):
        return None
    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, dict):
        return None
    result: dict[tuple[str, NbaStatsEndpoint], list[str]] = {}
    for game_id, by_endpoint in endpoints.items():
        if not isinstance(by_endpoint, dict):
            continue
        try:
            validate_game_id(str(game_id))
        except ValueError:
            continue
        for endpoint_value, raw_digests in by_endpoint.items():
            try:
                endpoint = NbaStatsEndpoint(str(endpoint_value))
            except ValueError:
                continue
            if endpoint not in REQUIRED_ENDPOINTS or not isinstance(raw_digests, list):
                continue
            digests = [
                str(digest)
                for digest in raw_digests
                if isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ]
            if digests:
                result[(str(game_id), endpoint)] = sorted(set(digests))
    return result


def write_coverage_index(
    path: Path | str,
    *,
    season: str,
    inventory: Mapping[tuple[str, NbaStatsEndpoint], Sequence[str]],
) -> Path:
    output = Path(path)
    endpoints: dict[str, dict[str, list[str]]] = {}
    for (game_id, endpoint), digests in sorted(
        inventory.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        endpoints.setdefault(game_id, {})[endpoint.value] = sorted(set(digests))
    payload = {
        "schema_version": 1,
        "season": validate_season(season),
        "updated_at": _utc_timestamp(datetime.now(UTC)),
        "endpoints": endpoints,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(payload))
    temporary.replace(output)
    return output


def ingest_completed_stats_v3(
    *,
    state_dir: Path | str = DEFAULT_STATS_V3_STATE_DIR,
    lake_uri: str = DEFAULT_STATS_V3_LAKE_URI,
    coverage_path: Path | str = DEFAULT_STATS_V3_COVERAGE_PATH,
    season: str = DEFAULT_STATS_V3_SEASON,
    start_date: date | None = None,
    end_date: date | None = None,
    game_ids: Sequence[str] | None = None,
    now: datetime | None = None,
    grace_period: timedelta = timedelta(hours=1),
    ignore_grace: bool = False,
    refresh: bool = False,
    max_retries: int = 3,
    min_request_interval_seconds: float = 1.0,
    request_interval_jitter_seconds: float = 0.25,
    discover_from_nba_schedule: bool = False,
    archive: StatsV3LakeArchive | None = None,
    fetch_response: FetchResponse | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch, content-address, and manifest all eligible completed games."""

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    started_at = (now or datetime.now(UTC)).astimezone(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "-" + f"{random.getrandbits(32):08x}"
    if discover_from_nba_schedule:
        games = discover_completed_nba_schedule_games(
            season=season,
            start_date=start_date,
            end_date=end_date,
            game_ids=game_ids,
        )
    else:
        games = discover_completed_schedule_games(
            state_dir,
            season=season,
            start_date=start_date,
            end_date=end_date,
            game_ids=game_ids,
            now=started_at,
            grace_period=grace_period,
            ignore_grace=ignore_grace,
        )
    lake = archive or StatsV3LakeArchive(lake_uri)
    inventory = read_coverage_index(coverage_path, season=season)
    if inventory is None:
        inventory = lake.inventory(season)
        write_coverage_index(coverage_path, season=season, inventory=inventory)
    requested = 0
    archived = 0
    revised = 0
    unchanged = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    circuit_open = False

    with tempfile.TemporaryDirectory(prefix="gestalt-stats-v3-") as directory:
        client: NbaStatsClient | None = None
        if fetch_response is None:
            client = NbaStatsClient(
                cache=NbaStatsRawCache(Path(directory) / "stats"),
                min_request_interval_seconds=min_request_interval_seconds,
                request_interval_jitter_seconds=request_interval_jitter_seconds,
            )

            def fetch_from_stats(endpoint: NbaStatsEndpoint, game_id: str) -> StatsCachedResponse:
                assert client is not None
                return client.fetch(endpoint, game_id, use_cache=False)

            fetch_response = fetch_from_stats
        try:
            for game in games:
                for endpoint in REQUIRED_ENDPOINTS:
                    known = inventory.get((game.game_id, endpoint), [])
                    if known and not refresh:
                        skipped += 1
                        continue
                    requested += 1
                    endpoint_started_at = datetime.now(UTC)
                    response: StatsCachedResponse | None = None
                    terminal_error: Exception | None = None
                    attempt_number = 0
                    for attempt_number in range(1, max_retries + 2):
                        try:
                            response = fetch_response(endpoint, game.game_id)
                            terminal_error = None
                            break
                        except Exception as error:  # retained in an immutable failure record
                            terminal_error = error
                            if not _retryable(error) or attempt_number > max_retries:
                                break
                            sleeper((5.0, 30.0, 120.0)[min(attempt_number - 1, 2)])
                    if terminal_error is not None or response is None:
                        error = terminal_error or RuntimeError("Stats fetch returned no response")
                        record_uri = lake.record_failure(
                            game,
                            endpoint,
                            run_id=run_id,
                            attempt_number=attempt_number,
                            started_at=endpoint_started_at,
                            error=error,
                        )
                        failure = {
                            "game_id": game.game_id,
                            "endpoint": endpoint.value,
                            "error_type": type(error).__name__,
                            "error_message": _redacted_error_message(str(error)),
                            "record_uri": record_uri,
                        }
                        failures.append(failure)
                        outcomes.append({**failure, "status": "failed"})
                        if isinstance(error, NbaStatsError) and error.circuit_open:
                            circuit_open = True
                            break
                        continue
                    outcome = lake.archive(
                        game,
                        response,
                        run_id=run_id,
                        attempt_number=attempt_number,
                        known_sha256=known,
                    )
                    inventory.setdefault((game.game_id, endpoint), []).append(outcome.sha256)
                    if outcome.disposition == "archived":
                        archived += 1
                    elif outcome.disposition == "revised":
                        revised += 1
                    else:
                        unchanged += 1
                    outcomes.append(
                        {
                            "game_id": game.game_id,
                            "endpoint": endpoint.value,
                            "status": outcome.disposition,
                            "sha256": outcome.sha256,
                            "byte_count": outcome.byte_count,
                            "raw_uri": outcome.raw_uri,
                            "record_uri": outcome.record_uri,
                        }
                    )
                if circuit_open:
                    break
        finally:
            if client is not None:
                client.close()

    write_coverage_index(coverage_path, season=season, inventory=inventory)

    finished_at = datetime.now(UTC)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "season": season,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "selected_game_count": len(games),
        "selected_game_ids": [game.game_id for game in games],
        "requested_endpoint_count": requested,
        "archived_count": archived,
        "revised_count": revised,
        "unchanged_count": unchanged,
        "skipped_count": skipped,
        "failed_count": len(failures),
        "refresh": refresh,
        "discovery_source": "nba_schedule" if discover_from_nba_schedule else "serving_state",
        "ignore_grace": ignore_grace,
        "grace_period_minutes": grace_period.total_seconds() / 60.0,
        "started_at": _utc_timestamp(started_at),
        "finished_at": _utc_timestamp(finished_at),
        "duration_seconds": max(0.0, (finished_at - started_at).total_seconds()),
        "outcomes": outcomes,
    }
    summary["manifest_uri"] = lake.write_run_summary(run_id, summary)
    if failures:
        raise StatsV3BatchError(summary)
    return summary


def pending_completed_game_ids(
    *,
    state_dir: Path | str,
    lake_uri: str,
    season: str,
    coverage_path: Path | str = DEFAULT_STATS_V3_COVERAGE_PATH,
    now: datetime | None = None,
    grace_period: timedelta = timedelta(hours=1),
) -> list[str]:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    eastern_date = current.astimezone(EASTERN).date()
    games = discover_completed_schedule_games(
        state_dir,
        season=season,
        start_date=eastern_date - timedelta(days=1),
        end_date=eastern_date,
        now=current,
        grace_period=grace_period,
    )
    del lake_uri  # retained in the public contract for stable Dagster configuration
    inventory = read_coverage_index(coverage_path, season=season) or {}
    return [
        game.game_id
        for game in games
        if any((game.game_id, endpoint) not in inventory for endpoint in REQUIRED_ENDPOINTS)
    ]


def _retryable(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, NbaStatsError):
            return current.transient and not current.circuit_open
        current = current.__cause__
    return False


def _redacted_error_message(message: str | None) -> str | None:
    if message is None:
        return None
    proxy = os.environ.get("GESTALT_NBA_STATS_PROXY")
    return message.replace(proxy, "<redacted-proxy>") if proxy else message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive completed NBA Stats V3 play-by-play and box scores to S3."
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("GESTALT_SCHEDULE_STATE_DIR", DEFAULT_STATS_V3_STATE_DIR)),
    )
    parser.add_argument(
        "--lake-uri",
        default=os.environ.get("GESTALT_STATS_V3_LAKE_URI", DEFAULT_STATS_V3_LAKE_URI),
    )
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=Path(
            os.environ.get("GESTALT_STATS_V3_COVERAGE_PATH", DEFAULT_STATS_V3_COVERAGE_PATH)
        ),
    )
    parser.add_argument(
        "--season",
        default=os.environ.get("GESTALT_STATS_V3_SEASON", DEFAULT_STATS_V3_SEASON),
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--game-id", action="append", dest="game_ids")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--discover-from-nba-schedule", action="store_true")
    parser.add_argument("--ignore-grace", action="store_true")
    parser.add_argument("--grace-minutes", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = ingest_completed_stats_v3(
            state_dir=args.state_dir,
            lake_uri=args.lake_uri,
            coverage_path=args.coverage_path,
            season=args.season,
            start_date=args.start_date,
            end_date=args.end_date,
            game_ids=args.game_ids,
            refresh=args.refresh,
            discover_from_nba_schedule=args.discover_from_nba_schedule,
            ignore_grace=args.ignore_grace,
            grace_period=timedelta(minutes=args.grace_minutes),
            max_retries=args.max_retries,
        )
    except StatsV3BatchError as error:
        print(json.dumps(error.summary, indent=2, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
