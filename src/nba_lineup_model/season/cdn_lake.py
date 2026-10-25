from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nba_lineup_model.ingest.nba_cdn import (
    CachedResponse,
    NbaCdnClient,
    NbaCdnEndpoint,
    NbaCdnError,
    RawJsonCache,
    validate_game_id,
)
from nba_lineup_model.season.schema import validate_season
from nba_lineup_model.season.stats_v3_lake import (
    AwsCliImmutableObjectStore,
    CompletedScheduleGame,
    ImmutableObjectStore,
    _canonical_json,
    _join_s3_uri,
    _parse_s3_uri,
    _redacted_error_message,
    _utc_timestamp,
    discover_completed_nba_schedule_games,
    discover_completed_schedule_games,
)

DEFAULT_CDN_LAKE_URI = "s3://nba-gestalt-data/lake"
DEFAULT_CDN_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_CDN_COVERAGE_PATH = Path("/var/lib/nba-gestalt/cdn/coverage.json")
DEFAULT_CDN_SEASON = "2026-27"
EASTERN = ZoneInfo("America/New_York")
REQUIRED_ENDPOINT = NbaCdnEndpoint.PLAY_BY_PLAY


@dataclass(frozen=True)
class CdnArchiveOutcome:
    sha256: str
    byte_count: int
    disposition: str
    raw_uri: str
    record_uri: str | None
    prior_sha256: str | None


class CdnLakeArchive:
    """Content-addressed NBA liveData play-by-play plus query records."""

    def __init__(
        self,
        base_uri: str = DEFAULT_CDN_LAKE_URI,
        *,
        store: ImmutableObjectStore | None = None,
    ) -> None:
        _parse_s3_uri(base_uri)
        self.base_uri = base_uri.rstrip("/")
        self.store = store or AwsCliImmutableObjectStore()

    def inventory(self, season: str) -> dict[str, list[str]]:
        season = validate_season(season)
        prefix_uri = _join_s3_uri(
            self.base_uri,
            "raw",
            "cdn",
            REQUIRED_ENDPOINT.value,
            f"season={season}",
        )
        _, root_key = _parse_s3_uri(prefix_uri)
        root_prefix = root_key.rstrip("/") + "/"
        found: dict[str, list[str]] = {}
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
            found.setdefault(game_id, []).append(digest)
        return {game_id: sorted(set(values)) for game_id, values in found.items()}

    def archive(
        self,
        game: CompletedScheduleGame,
        response: CachedResponse,
        *,
        run_id: str,
        attempt_number: int,
        known_sha256: Sequence[str],
    ) -> CdnArchiveOutcome:
        if response.endpoint is not REQUIRED_ENDPOINT or response.game_id != game.game_id:
            raise ValueError("CDN response does not match the selected game and endpoint")
        payload_game = response.payload.get("game")
        if not isinstance(payload_game, dict) or str(payload_game.get("gameId")) != game.game_id:
            raise ValueError("CDN payload game ID does not match the selected game")
        raw_body = response.raw_body
        if raw_body is None:
            raw_body = json.dumps(
                response.payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        digest = hashlib.sha256(raw_body).hexdigest()
        raw_uri = self._raw_uri(game, digest)
        known = list(dict.fromkeys(known_sha256))
        prior = known[-1] if known else None
        if digest in known:
            return CdnArchiveOutcome(
                sha256=digest,
                byte_count=len(raw_body),
                disposition="unchanged",
                raw_uri=raw_uri,
                record_uri=None,
                prior_sha256=prior,
            )

        raw_created = self.store.put_bytes(raw_uri, raw_body)
        if not raw_created:
            # Another process may have archived the same content after this
            # run loaded its local coverage index. Treat that race exactly as
            # an idempotent checksum hit and do not contend on sidecars.
            return CdnArchiveOutcome(
                sha256=digest,
                byte_count=len(raw_body),
                disposition="unchanged",
                raw_uri=raw_uri,
                record_uri=None,
                prior_sha256=digest,
            )
        metadata = self._record_payload(
            game,
            run_id=run_id,
            attempt_number=attempt_number,
            status="succeeded",
            started_at=response.fetched_at,
            finished_at=datetime.now(UTC),
            source_url=response.url,
            response_headers=response.response_headers,
            sha256=digest,
            byte_count=len(raw_body),
            raw_uri=raw_uri,
            prior_sha256=prior,
        )
        self.store.put_bytes(
            raw_uri.removesuffix(".json") + ".meta.json", _canonical_json(metadata)
        )
        record_uri = self._record_uri(game, f"success-{digest}.json")
        self.store.put_bytes(record_uri, _canonical_json(metadata))
        return CdnArchiveOutcome(
            sha256=digest,
            byte_count=len(raw_body),
            disposition="revised" if prior else "archived",
            raw_uri=raw_uri,
            record_uri=record_uri,
            prior_sha256=prior,
        )

    def record_failure(
        self,
        game: CompletedScheduleGame,
        *,
        run_id: str,
        attempt_number: int,
        started_at: datetime,
        error: Exception,
    ) -> str:
        payload = self._record_payload(
            game,
            run_id=run_id,
            attempt_number=attempt_number,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            error_type=type(error).__name__,
            error_message=str(error),
        )
        uri = self._record_uri(game, f"failure-{run_id}-{attempt_number}.json")
        self.store.put_bytes(uri, _canonical_json(payload))
        return uri

    def write_run_summary(self, run_id: str, payload: Mapping[str, Any]) -> str:
        uri = _join_s3_uri(
            self.base_uri,
            "manifests",
            "cdn_live_data",
            "runs",
            f"{run_id}.json",
        )
        self.store.put_bytes(uri, _canonical_json(payload))
        return uri

    def _raw_uri(self, game: CompletedScheduleGame, digest: str) -> str:
        return _join_s3_uri(
            self.base_uri,
            "raw",
            "cdn",
            REQUIRED_ENDPOINT.value,
            f"season={game.season}",
            f"game_id={game.game_id}",
            f"{digest}.json",
        )

    def _record_uri(self, game: CompletedScheduleGame, filename: str) -> str:
        return _join_s3_uri(
            self.base_uri,
            "manifests",
            "cdn_live_data",
            "records",
            f"season={game.season}",
            f"game_date={game.game_date.isoformat()}",
            f"game_id={game.game_id}",
            f"endpoint={REQUIRED_ENDPOINT.value}",
            filename,
        )

    @staticmethod
    def _record_payload(
        game: CompletedScheduleGame,
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
            "source": "nba_cdn_live_data",
            "run_id": run_id,
            "attempt_number": attempt_number,
            "game_id": game.game_id,
            "espn_game_id": game.espn_game_id,
            "season": game.season,
            "season_type": game.season_type,
            "game_date": game.game_date.isoformat(),
            "away_team": game.away_team,
            "home_team": game.home_team,
            "endpoint": REQUIRED_ENDPOINT.value,
            "source_url": source_url,
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


class CdnBatchError(RuntimeError):
    def __init__(self, summary: Mapping[str, Any]) -> None:
        self.summary = dict(summary)
        super().__init__(f"NBA CDN ingestion failed for {summary['failed_count']} games")


FetchResponse = Callable[[str], CachedResponse]


def read_coverage_index(path: Path | str, *, season: str) -> dict[str, list[str]] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("season") != validate_season(season):
        return None
    games = payload.get("games")
    if not isinstance(games, dict):
        return None
    result: dict[str, list[str]] = {}
    for game_id, raw_digests in games.items():
        try:
            validate_game_id(str(game_id))
        except ValueError:
            continue
        if not isinstance(raw_digests, list):
            continue
        digests = [
            str(digest)
            for digest in raw_digests
            if isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ]
        if digests:
            result[str(game_id)] = sorted(set(digests))
    return result


def write_coverage_index(
    path: Path | str,
    *,
    season: str,
    inventory: Mapping[str, Sequence[str]],
) -> Path:
    output = Path(path)
    payload = {
        "schema_version": 1,
        "source": "nba_cdn_live_data",
        "season": validate_season(season),
        "updated_at": _utc_timestamp(datetime.now(UTC)),
        "games": {game_id: sorted(set(digests)) for game_id, digests in sorted(inventory.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(payload))
    temporary.replace(output)
    return output


def ingest_completed_cdn(
    *,
    state_dir: Path | str = DEFAULT_CDN_STATE_DIR,
    lake_uri: str = DEFAULT_CDN_LAKE_URI,
    coverage_path: Path | str = DEFAULT_CDN_COVERAGE_PATH,
    season: str = DEFAULT_CDN_SEASON,
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
    archive: CdnLakeArchive | None = None,
    fetch_response: FetchResponse | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Archive final-game CDN play-by-play with immutable revisions."""

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
    lake = archive or CdnLakeArchive(lake_uri)
    inventory = read_coverage_index(coverage_path, season=season)
    if inventory is None:
        inventory = lake.inventory(season)
        write_coverage_index(coverage_path, season=season, inventory=inventory)

    requested = archived = revised = unchanged = skipped = 0
    failures: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    circuit_open = False
    with tempfile.TemporaryDirectory(prefix="gestalt-cdn-") as directory:
        client: NbaCdnClient | None = None
        if fetch_response is None:
            client = NbaCdnClient(
                cache=RawJsonCache(Path(directory) / "cdn"),
                min_request_interval_seconds=min_request_interval_seconds,
                request_interval_jitter_seconds=request_interval_jitter_seconds,
            )

            def fetch_from_cdn(game_id: str) -> CachedResponse:
                assert client is not None
                return client.fetch_play_by_play(game_id, use_cache=False)

            fetch_response = fetch_from_cdn
        try:
            for game in games:
                known = inventory.get(game.game_id, [])
                if known and not refresh:
                    skipped += 1
                    continue
                requested += 1
                endpoint_started_at = datetime.now(UTC)
                response: CachedResponse | None = None
                terminal_error: Exception | None = None
                attempt_number = 0
                for attempt_number in range(1, max_retries + 2):
                    try:
                        response = fetch_response(game.game_id)
                        terminal_error = None
                        break
                    except Exception as error:
                        terminal_error = error
                        if not _retryable(error) or attempt_number > max_retries:
                            break
                        sleeper((5.0, 30.0, 120.0)[min(attempt_number - 1, 2)])
                if terminal_error is not None or response is None:
                    error = terminal_error or RuntimeError("CDN fetch returned no response")
                    record_uri = lake.record_failure(
                        game,
                        run_id=run_id,
                        attempt_number=attempt_number,
                        started_at=endpoint_started_at,
                        error=error,
                    )
                    failure = {
                        "game_id": game.game_id,
                        "endpoint": REQUIRED_ENDPOINT.value,
                        "error_type": type(error).__name__,
                        "error_message": _redacted_error_message(str(error)),
                        "record_uri": record_uri,
                    }
                    failures.append(failure)
                    outcomes.append({**failure, "status": "failed"})
                    if isinstance(error, NbaCdnError) and error.circuit_open:
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
                inventory.setdefault(game.game_id, []).append(outcome.sha256)
                if outcome.disposition == "archived":
                    archived += 1
                elif outcome.disposition == "revised":
                    revised += 1
                else:
                    unchanged += 1
                outcomes.append(
                    {
                        "game_id": game.game_id,
                        "endpoint": REQUIRED_ENDPOINT.value,
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
        "source": "nba_cdn_live_data",
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
        raise CdnBatchError(summary)
    return summary


def pending_completed_cdn_game_ids(
    *,
    state_dir: Path | str,
    season: str,
    coverage_path: Path | str = DEFAULT_CDN_COVERAGE_PATH,
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
    inventory = read_coverage_index(coverage_path, season=season) or {}
    return [game.game_id for game in games if game.game_id not in inventory]


def _retryable(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, NbaCdnError):
            return current.transient and not current.circuit_open
        current = current.__cause__
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive completed NBA CDN play-by-play to S3.")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("GESTALT_SCHEDULE_STATE_DIR", DEFAULT_CDN_STATE_DIR)),
    )
    parser.add_argument(
        "--lake-uri",
        default=os.environ.get("GESTALT_CDN_LAKE_URI", DEFAULT_CDN_LAKE_URI),
    )
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=Path(os.environ.get("GESTALT_CDN_COVERAGE_PATH", DEFAULT_CDN_COVERAGE_PATH)),
    )
    parser.add_argument(
        "--season",
        default=os.environ.get("GESTALT_CDN_SEASON", DEFAULT_CDN_SEASON),
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
        summary = ingest_completed_cdn(
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
    except CdnBatchError as error:
        print(json.dumps(error.summary, indent=2, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
