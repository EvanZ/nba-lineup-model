"""Build the observed-data chain one Pacific slate date at a time.

The command is intentionally fail closed.  It processes every game on one date
to produce a useful error inventory, but it publishes no artifact and advances
no pointer unless the entire date is clean.  The next date is never attempted
after a failed date.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from nba_lineup_model.ingest.nba_cdn import NbaCdnClient, RawJsonCache
from nba_lineup_model.modeling.in_season_nail_monitor import (
    _rapm_stints_from_nba_boxscore,
)
from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.daily_data_artifact import (
    PublishedDailyDataArtifact,
    promote_daily_data_artifact,
    publish_daily_data_artifact,
    verify_daily_data_artifact,
)
from nba_lineup_model.season.stats_v3_lake import (
    AwsCliImmutableObjectStore,
    ImmutableObjectStore,
)
from nba_lineup_model.web_api.win_projections import (
    _fetch_latest_official_injury_report,
)

DEFAULT_SEASON = "2026-27"
DEFAULT_COVERAGE_PATH = Path("/var/lib/nba-gestalt/cdn/coverage.json")
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/daily-data")
DEFAULT_GAME_CATALOG_PATH = Path("/var/lib/nba-gestalt-dev/catalog/games.parquet")
DEFAULT_PLAY_BY_PLAY_S3_PREFIX = "s3://nba-gestalt-data/lake/raw/cdn/playbyplay"
DEFAULT_BOXSCORE_S3_PREFIX = "s3://nba-gestalt-data/lake/raw/cdn/boxscore"
DEFAULT_INJURY_S3_PREFIX = "s3://nba-gestalt-data/lake/raw/nba/injury-reports"
DEFAULT_INACTIVE_S3_PREFIX = "s3://nba-gestalt-data/lake/raw/nba/game-pages"
PACIFIC = ZoneInfo("America/Los_Angeles")
NBA_DURATION_RE = re.compile(
    r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$"
)


class DailyDataReplayBlocked(RuntimeError):
    """Raised after a date-level failure report has been written."""


@dataclass(frozen=True)
class OfficialBoxscore:
    payload: dict[str, Any]
    sha256: str
    immutable_uri: str
    source_url: str


BoxscoreLoader = Callable[[str], OfficialBoxscore]


@dataclass(frozen=True)
class OfficialInjuryReport:
    payload: dict[str, Any]
    sha256: str
    immutable_uri: str
    source_url: str


InjuryReportLoader = Callable[[date], OfficialInjuryReport]


@dataclass(frozen=True)
class OfficialInactiveList:
    payload: dict[str, Any]
    sha256: str
    immutable_uri: str
    source_url: str


InactiveListLoader = Callable[[Mapping[str, Any]], OfficialInactiveList]


def replay_daily_data_chain(
    *,
    start_date: date,
    end_date: date,
    coverage_path: Path | str = DEFAULT_COVERAGE_PATH,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    game_catalog_path: Path | str = DEFAULT_GAME_CATALOG_PATH,
    play_by_play_s3_prefix: str = DEFAULT_PLAY_BY_PLAY_S3_PREFIX,
    boxscore_s3_prefix: str = DEFAULT_BOXSCORE_S3_PREFIX,
    injury_s3_prefix: str = DEFAULT_INJURY_S3_PREFIX,
    inactive_s3_prefix: str = DEFAULT_INACTIVE_S3_PREFIX,
    season: str = DEFAULT_SEASON,
    boxscore_loader: BoxscoreLoader | None = None,
    injury_report_loader: InjuryReportLoader | None = None,
    inactive_list_loader: InactiveListLoader | None = None,
) -> list[PublishedDailyDataArtifact]:
    """Replay dates chronologically and stop before the first bad publication."""

    if end_date < start_date:
        raise ValueError("Replay end date cannot precede its start date")
    chain_root = Path(output_root)
    catalog_path = Path(game_catalog_path)
    games_by_date = _load_official_game_catalog(
        catalog_path,
        season=season,
        start_date=start_date,
        end_date=end_date,
    )
    coverage = _read_json(Path(coverage_path)).get("games")
    if not isinstance(coverage, dict):
        raise ValueError("CDN coverage manifest has no game map")

    parent_bundle = _current_chain_tip(chain_root)
    if parent_bundle is not None:
        pointer = _read_json(chain_root / "latest.json")
        next_date = date.fromisoformat(str(pointer["game_date_pacific"])) + timedelta(days=1)
        if next_date != start_date:
            raise ValueError(
                f"Replay must resume at {next_date.isoformat()}, not {start_date.isoformat()}"
            )

    published: list[PublishedDailyDataArtifact] = []
    with tempfile.TemporaryDirectory(prefix="nba-boxscore-cache-") as cache_root:
        store = AwsCliImmutableObjectStore()
        with NbaCdnClient(cache=RawJsonCache(cache_root)) as client:
            loader = boxscore_loader or _official_boxscore_loader(
                client=client,
                store=store,
                season=season,
                s3_prefix=boxscore_s3_prefix,
            )
            injury_loader = injury_report_loader or _official_injury_report_loader(
                store=store,
                season=season,
                s3_prefix=injury_s3_prefix,
            )
            inactive_loader = inactive_list_loader or _official_inactive_list_loader(
                store=store,
                season=season,
                s3_prefix=inactive_s3_prefix,
            )
            current_date = start_date
            while current_date <= end_date:
                games = games_by_date.get(current_date, [])
                artifact = _build_date(
                    game_date_pacific=current_date,
                    games=games,
                    coverage=coverage,
                    game_catalog_path=catalog_path,
                    output_root=chain_root,
                    parent_bundle=parent_bundle,
                    play_by_play_s3_prefix=play_by_play_s3_prefix,
                    boxscore_loader=loader,
                    injury_report_loader=injury_loader,
                    inactive_list_loader=inactive_loader,
                    season=season,
                )
                promote_daily_data_artifact(artifact.path, root=chain_root)
                write_daily_data_chain_report(chain_root)
                published.append(artifact)
                parent_bundle = artifact.path
                current_date += timedelta(days=1)
    return published


def _build_date(
    *,
    game_date_pacific: date,
    games: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, object],
    game_catalog_path: Path,
    output_root: Path,
    parent_bundle: Path | None,
    play_by_play_s3_prefix: str,
    boxscore_loader: BoxscoreLoader,
    injury_report_loader: InjuryReportLoader,
    inactive_list_loader: InactiveListLoader,
    season: str,
) -> PublishedDailyDataArtifact:
    errors: list[dict[str, str]] = []
    repairs: list[dict[str, str]] = []
    game_sources: dict[str, dict[str, dict[str, str]]] = {}
    game_quality: dict[str, dict[str, float | int]] = {}
    final_games: list[dict[str, Any]] = []
    boxscores: list[dict[str, Any]] = []
    actual_minutes: list[pd.DataFrame] = []
    stints: list[pd.DataFrame] = []
    identity_rows: list[pd.DataFrame] = []
    rosters: list[dict[str, Any]] = []
    injuries: list[dict[str, Any]] = []
    inactive_lists: list[dict[str, Any]] = []

    catalog_sha256 = file_sha256(game_catalog_path)
    official_injury = injury_report_loader(game_date_pacific) if games else None
    for catalog_game in games:
        game_id = str(catalog_game.get("game_id", "")).strip()
        if not game_id:
            errors.append({"game_id": "", "message": "Official NBA schedule row has no game ID"})
            continue
        try:
            versions = coverage.get(game_id)
            if not isinstance(versions, list) or not versions:
                raise ValueError("CDN coverage has no immutable play-by-play version")
            play_by_play_sha = str(versions[-1])
            play_by_play_uri = (
                f"{play_by_play_s3_prefix.rstrip('/')}/season={season}/"
                f"game_id={game_id}/{play_by_play_sha}.json"
            )
            play_by_play = _fetch_verified_s3_json(play_by_play_uri, play_by_play_sha)
            official_boxscore = boxscore_loader(game_id)
            official_inactives = inactive_list_loader(catalog_game)
            boxscore = official_boxscore.payload
            game_record = _official_final_game(
                catalog_game=catalog_game,
                boxscore=boxscore,
                game_date_pacific=game_date_pacific,
            )
            game_stints, _team_ids, issue_count = _rapm_stints_from_nba_boxscore(
                boxscore=boxscore,
                play_by_play=play_by_play,
                play_by_play_sha256=play_by_play_sha,
                boxscore_sha256=official_boxscore.sha256,
                game_date=game_date_pacific.isoformat(),
                game_time_utc=str(catalog_game["game_time_utc"]),
            )
            game_actual, game_identity, game_roster = _nba_actual_minutes(
                game_record=game_record,
                boxscore=boxscore,
                inactive_list=official_inactives.payload,
            )
            final_home, final_away = _terminal_play_by_play_scores(play_by_play)
            box_home, box_away = _nba_box_scores(boxscore)
            # The event reconstruction counts each team's possessions. RAPM stints
            # use paired game possessions (half that total), so conservation is
            # checked in the same unit rather than comparing 198 to 99.
            reconstructed_possessions = (
                float(game_stints.attrs.get("reconstructed_possessions", -1)) / 2.0
            )
            stint_possessions = float(game_stints["possessions"].sum())
            if issue_count:
                issue_codes = sorted(
                    {
                        str(issue["code"])
                        for issue in game_stints.attrs.get("reconstruction_issues", [])
                    }
                )
                repairs.append(
                    {
                        "game_id": game_id,
                        "action": (
                            f"accepted {issue_count} audited reconstruction warning(s): "
                            + ", ".join(issue_codes)
                        ),
                        "source": "NBA liveData play-by-play",
                    }
                )
            game_sources[game_id] = {
                "nba_schedule": {
                    "uri": game_catalog_path.resolve().as_uri(),
                    "sha256": catalog_sha256,
                },
                "nba_box_score": {
                    "uri": official_boxscore.immutable_uri,
                    "sha256": official_boxscore.sha256,
                },
                "nba_play_by_play": {
                    "uri": play_by_play_uri,
                    "sha256": play_by_play_sha,
                },
                "nba_injury_report": {
                    "uri": str(official_injury.immutable_uri),
                    "sha256": str(official_injury.sha256),
                },
                "nba_inactive_list": {
                    "uri": official_inactives.immutable_uri,
                    "sha256": official_inactives.sha256,
                },
            }
            game_quality[game_id] = {
                "final_home_score": final_home,
                "final_away_score": final_away,
                "box_home_points": box_home,
                "box_away_points": box_away,
                "play_by_play_possessions": reconstructed_possessions,
                "observed_stint_possessions": stint_possessions,
            }
            final_games.append(game_record)
            boxscores.append(
                {
                    "game_id": game_id,
                    "source_url": official_boxscore.source_url,
                    "immutable_uri": official_boxscore.immutable_uri,
                    "sha256": official_boxscore.sha256,
                    "payload": boxscore,
                }
            )
            inactive_lists.append(
                {
                    "game_id": game_id,
                    "source_url": official_inactives.source_url,
                    "immutable_uri": official_inactives.immutable_uri,
                    "sha256": official_inactives.sha256,
                    "payload": official_inactives.payload,
                }
            )
            actual_minutes.append(game_actual)
            identity_rows.append(game_identity)
            stints.append(game_stints)
            rosters.append(game_roster)
        except Exception as error:  # date-level report must inventory every failed game
            errors.append({"game_id": game_id, "message": f"{type(error).__name__}: {error}"})

    expected_game_ids = sorted(str(game["game_id"]) for game in games)
    if official_injury is not None:
        injuries.append(
            {
                "report_date": game_date_pacific.isoformat(),
                "source_url": official_injury.source_url,
                "immutable_uri": official_injury.immutable_uri,
                "sha256": official_injury.sha256,
                "payload": official_injury.payload,
            }
        )
    if errors:
        report_path = _write_failure_report(output_root, game_date_pacific, errors)
        raise DailyDataReplayBlocked(
            f"Pacific date {game_date_pacific.isoformat()} failed; report: {report_path}"
        )

    with tempfile.TemporaryDirectory(prefix="daily-data-components-") as directory:
        component_root = Path(directory)
        components = _write_components(
            component_root,
            final_games=final_games,
            boxscores=boxscores,
            actual_minutes=actual_minutes,
            stints=stints,
            identity_rows=identity_rows,
            rosters=rosters,
            injuries=injuries,
            inactive_lists=inactive_lists,
        )
        try:
            return publish_daily_data_artifact(
                root=output_root,
                season=season,
                game_date_pacific=game_date_pacific,
                game_ids=expected_game_ids,
                game_sources=game_sources,
                game_quality=game_quality,
                components=components,
                parent_bundle=parent_bundle,
                repairs=repairs,
            )
        except Exception as error:
            validation_errors = [
                {"game_id": "", "message": f"{type(error).__name__}: {error}"}
            ]
            report_path = _write_failure_report(
                output_root, game_date_pacific, validation_errors
            )
            raise DailyDataReplayBlocked(
                f"Pacific date {game_date_pacific.isoformat()} failed; report: {report_path}"
            ) from error


def _load_official_game_catalog(
    path: Path,
    *,
    season: str,
    start_date: date,
    end_date: date,
) -> dict[date, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Official NBA game catalog is missing: {path}")
    frame = pd.read_parquet(path)
    required = {
        "game_id",
        "season",
        "season_type",
        "game_time_utc",
        "home_team_id",
        "home_team_tricode",
        "away_team_id",
        "away_team_tricode",
        "source_url",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Official NBA game catalog is missing: " + ", ".join(missing))
    selected = frame.loc[
        frame["season"].astype(str).eq(season)
        & frame["season_type"].astype(str).eq("regular")
    ].copy()
    selected["game_time_utc"] = pd.to_datetime(selected["game_time_utc"], utc=True)
    selected["game_date_pacific"] = (
        selected["game_time_utc"].dt.tz_convert(PACIFIC).dt.date
    )
    selected = selected.loc[
        selected["game_date_pacific"].between(start_date, end_date)
    ].sort_values(["game_time_utc", "game_id"])
    if selected["game_id"].astype(str).duplicated().any():
        raise ValueError("Official NBA game catalog contains duplicate game IDs")
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in selected.to_dict("records"):
        grouped.setdefault(row["game_date_pacific"], []).append(row)
    return grouped


def _official_boxscore_loader(
    *,
    client: NbaCdnClient,
    store: ImmutableObjectStore,
    season: str,
    s3_prefix: str,
) -> BoxscoreLoader:
    def load(game_id: str) -> OfficialBoxscore:
        response = client.fetch_boxscore(game_id, use_cache=False)
        game = response.payload.get("game")
        if not isinstance(game, dict) or str(game.get("gameId", "")) != game_id:
            raise ValueError("NBA box score payload does not match the scheduled game")
        if int(game.get("gameStatus", 0)) != 3:
            raise ValueError(f"NBA box score is not final: {game.get('gameStatusText')}")
        raw_body = response.raw_body
        if raw_body is None:
            raw_body = json.dumps(
                response.payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        digest = hashlib.sha256(raw_body).hexdigest()
        uri = (
            f"{s3_prefix.rstrip('/')}/season={season}/"
            f"game_id={game_id}/{digest}.json"
        )
        store.put_bytes(uri, raw_body)
        return OfficialBoxscore(
            payload=response.payload,
            sha256=digest,
            immutable_uri=uri,
            source_url=response.url,
        )

    return load


def _official_injury_report_loader(
    *,
    store: ImmutableObjectStore,
    season: str,
    s3_prefix: str,
) -> InjuryReportLoader:
    def load(report_date: date) -> OfficialInjuryReport:
        report, pdf = _fetch_latest_official_injury_report(
            report_date=report_date.isoformat(),
            now=datetime.now(UTC),
        )
        digest = hashlib.sha256(pdf).hexdigest()
        uri = (
            f"{s3_prefix.rstrip('/')}/season={season}/"
            f"report_date={report_date.isoformat()}/{digest}.pdf"
        )
        store.put_bytes(uri, pdf)
        return OfficialInjuryReport(
            payload=report,
            sha256=digest,
            immutable_uri=uri,
            source_url=str(report["source_url"]),
        )

    return load


def _official_inactive_list_loader(
    *,
    store: ImmutableObjectStore,
    season: str,
    s3_prefix: str,
) -> InactiveListLoader:
    def load(catalog_game: Mapping[str, Any]) -> OfficialInactiveList:
        game_id = str(catalog_game["game_id"])
        away = str(catalog_game["away_team_tricode"]).lower()
        home = str(catalog_game["home_team_tricode"]).lower()
        source_url = f"https://www.nba.com/game/{away}-vs-{home}-{game_id}/box-score"
        request = urllib.request.Request(
            source_url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; NBA-Gestalt/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw_body = response.read()
            resolved_url = str(response.url)
        payload = _parse_official_inactive_list(
            raw_body,
            expected_game_id=game_id,
            expected_home=str(catalog_game["home_team_tricode"]),
            expected_away=str(catalog_game["away_team_tricode"]),
        )
        digest = hashlib.sha256(raw_body).hexdigest()
        uri = (
            f"{s3_prefix.rstrip('/')}/season={season}/"
            f"game_id={game_id}/{digest}.html"
        )
        store.put_bytes(uri, raw_body)
        return OfficialInactiveList(
            payload=payload,
            sha256=digest,
            immutable_uri=uri,
            source_url=resolved_url,
        )

    return load


def _parse_official_inactive_list(
    raw_html: bytes,
    *,
    expected_game_id: str,
    expected_home: str,
    expected_away: str,
) -> dict[str, Any]:
    text = raw_html.decode("utf-8")
    match = re.search(
        r"<script[^>]+id=[\"']__NEXT_DATA__[\"'][^>]*>(.*?)</script>",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Official NBA game page has no __NEXT_DATA__ payload")
    document = json.loads(html.unescape(match.group(1)))
    game = document.get("props", {}).get("pageProps", {}).get("game")
    if not isinstance(game, Mapping):
        raise ValueError("Official NBA game page has no game payload")
    if str(game.get("gameId", "")) != expected_game_id:
        raise ValueError("Official NBA inactive list does not match the scheduled game")
    output: dict[str, Any] = {"game_id": expected_game_id, "teams": []}
    for side, expected_team in (("homeTeam", expected_home), ("awayTeam", expected_away)):
        team = game.get(side)
        if not isinstance(team, Mapping) or str(team.get("teamTricode", "")) != expected_team:
            raise ValueError("Official NBA inactive-list teams differ from the schedule")
        players = team.get("inactives")
        if not isinstance(players, list):
            raise ValueError(f"Official NBA game page has no {side} inactive list")
        normalized = []
        seen: set[int] = set()
        for player in players:
            if not isinstance(player, Mapping) or player.get("personId") is None:
                raise ValueError(f"Official NBA game page has malformed {expected_team} inactive")
            player_id = int(player["personId"])
            if player_id in seen:
                raise ValueError(f"Official NBA game page duplicates inactive {player_id}")
            seen.add(player_id)
            normalized.append(
                {
                    "player_id": player_id,
                    "player_name": _nba_player_name(player),
                }
            )
        output["teams"].append(
            {
                "team_id": int(team["teamId"]),
                "team": expected_team,
                "side": "home" if side == "homeTeam" else "away",
                "players": normalized,
            }
        )
    return output


def _official_final_game(
    *,
    catalog_game: Mapping[str, Any],
    boxscore: Mapping[str, Any],
    game_date_pacific: date,
) -> dict[str, Any]:
    game = boxscore.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("NBA box score has no game payload")
    game_id = str(catalog_game["game_id"])
    if str(game.get("gameId", "")) != game_id:
        raise ValueError("NBA schedule and box score game IDs differ")
    home = game.get("homeTeam")
    away = game.get("awayTeam")
    if not isinstance(home, Mapping) or not isinstance(away, Mapping):
        raise ValueError("NBA box score has incomplete team records")
    expected = (
        str(catalog_game["home_team_tricode"]),
        str(catalog_game["away_team_tricode"]),
    )
    observed = (str(home.get("teamTricode", "")), str(away.get("teamTricode", "")))
    if observed != expected:
        raise ValueError(
            f"NBA schedule teams {expected} differ from box score teams {observed}"
        )
    home_score = int(home["score"])
    away_score = int(away["score"])
    return {
        "game_id": game_id,
        "season": str(catalog_game["season"]),
        "game_date_pacific": game_date_pacific.isoformat(),
        "start_time": pd.Timestamp(catalog_game["game_time_utc"]).isoformat(),
        "home_team_id": int(home["teamId"]),
        "home_team": observed[0],
        "away_team_id": int(away["teamId"]),
        "away_team": observed[1],
        "status": "final",
        "home_score": home_score,
        "away_score": away_score,
        "actual_home_margin": home_score - away_score,
    }


def _nba_actual_minutes(
    *,
    game_record: Mapping[str, Any],
    boxscore: Mapping[str, Any],
    inactive_list: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    actual_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    roster_teams: list[dict[str, Any]] = []
    game_id = str(game_record["game_id"])
    game = boxscore.get("game")
    if not isinstance(game, Mapping) or str(game.get("gameId", "")) != game_id:
        raise ValueError("NBA box score game ID does not match the official schedule")
    for side in ("homeTeam", "awayTeam"):
        team_row = game.get(side)
        if not isinstance(team_row, Mapping):
            raise ValueError(f"NBA box score has no {side}")
        team = str(team_row.get("teamTricode", "")).strip()
        players = team_row.get("players")
        if not team or not isinstance(players, list):
            raise ValueError(f"NBA box score has incomplete {side} roster")
        roster_players: list[dict[str, object]] = []
        roster_ids: set[int] = set()
        for player in players:
            if not isinstance(player, Mapping) or player.get("personId") is None:
                raise ValueError(f"NBA box score has malformed {team} player identity")
            player_id = int(player["personId"])
            roster_ids.add(player_id)
            player_name = _nba_player_name(player)
            statistics = player.get("statistics")
            if not isinstance(statistics, Mapping):
                statistics = {}
            minutes = _nba_minutes_value(statistics.get("minutes"))
            played = minutes > 0
            identity_rows.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "player_name": player_name,
                    "player_id": player_id,
                    "played": played,
                    "resolution": "nba_person_id",
                }
            )
            roster_players.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "played": played,
                    "starter": str(player.get("starter", "0")) == "1",
                    "status": str(player.get("status", "")),
                    "minutes": minutes,
                }
            )
            if played:
                actual_rows.append(
                    {
                        "game_id": game_id,
                        "game_date_pacific": str(game_record["game_date_pacific"]),
                        "team": team,
                        "player_id": player_id,
                        "player_name": player_name,
                        "minutes": minutes,
                    }
                )
        inactive_teams = {
            str(row.get("team")): row
            for row in (inactive_list or {}).get("teams", [])
            if isinstance(row, Mapping)
        }
        inactive_team = inactive_teams.get(team, {})
        if inactive_team and int(inactive_team.get("team_id", -1)) != int(team_row["teamId"]):
            raise ValueError(f"Official NBA inactive-list team ID differs for {team}")
        for inactive in inactive_team.get("players", []):
            player_id = int(inactive["player_id"])
            if player_id in roster_ids:
                raise ValueError(
                    f"Official NBA player {player_id} is both box-roster and inactive for {game_id}"
                )
            player_name = str(inactive["player_name"])
            identity_rows.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "player_name": player_name,
                    "player_id": player_id,
                    "played": False,
                    "resolution": "nba_inactive_person_id",
                }
            )
            roster_players.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "played": False,
                    "starter": False,
                    "status": "INACTIVE",
                    "minutes": 0.0,
                    "source": "official_nba_inactive_list",
                }
            )
        roster_teams.append(
            {
                "team_id": int(team_row["teamId"]),
                "team": team,
                "side": "home" if side == "homeTeam" else "away",
                "players": roster_players,
            }
        )
    actual = pd.DataFrame(actual_rows)
    if actual.empty:
        raise ValueError("NBA box score has no played players")
    roster = {"game_id": game_id, "teams": roster_teams}
    return actual, pd.DataFrame(identity_rows), roster


def _nba_player_name(player: Mapping[str, Any]) -> str:
    full_name = " ".join(
        str(player.get(field) or "").strip() for field in ("firstName", "familyName")
    ).strip()
    return full_name or str(player.get("name", "")).strip() or str(player["personId"])


def _nba_minutes_value(value: object) -> float:
    if value in (None, ""):
        return 0.0
    match = NBA_DURATION_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"NBA box score has invalid minutes duration: {value!r}")
    hours = float(match.group("hours") or 0)
    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    return hours * 60.0 + minutes + seconds / 60.0


def _nba_box_scores(boxscore: Mapping[str, Any]) -> tuple[int, int]:
    game = boxscore.get("game")
    if not isinstance(game, Mapping):
        raise ValueError("NBA box score has no game payload")
    return int(game["homeTeam"]["score"]), int(game["awayTeam"]["score"])


def _terminal_play_by_play_scores(play_by_play: Mapping[str, Any]) -> tuple[int, int]:
    actions = play_by_play.get("game", {}).get("actions", [])
    action_rows = actions if isinstance(actions, list) else []
    scored = [
        action
        for action in action_rows
        if isinstance(action, Mapping)
        if action.get("scoreHome") not in (None, "")
        and action.get("scoreAway") not in (None, "")
    ]
    if scored:
        # NBA corrections can be appended to the JSON after the game even when
        # they refer to an earlier period. Provider list order is therefore not
        # chronological; the canonical action ordering fields are authoritative.
        action = max(
            scored,
            key=lambda row: (
                int(row.get("period") or 0),
                int(row.get("orderNumber") or 0),
                int(row.get("actionNumber") or 0),
            ),
        )
        return int(float(action["scoreHome"])), int(float(action["scoreAway"]))
    raise ValueError("NBA play-by-play has no terminal score")


def _write_components(
    root: Path,
    *,
    final_games: Sequence[Mapping[str, Any]],
    boxscores: Sequence[Mapping[str, Any]],
    actual_minutes: Sequence[pd.DataFrame],
    stints: Sequence[pd.DataFrame],
    identity_rows: Sequence[pd.DataFrame],
    rosters: Sequence[Mapping[str, Any]],
    injuries: Sequence[Mapping[str, Any]],
    inactive_lists: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    final_games_path = root / "final_games.json"
    final_games_path.write_text(
        json.dumps(list(final_games), indent=2, sort_keys=True) + "\n"
    )
    boxscores_path = root / "nba_boxscores.json"
    boxscores_path.write_text(
        json.dumps(list(boxscores), indent=2, sort_keys=True) + "\n"
    )
    actual_path = root / "actual_minutes.parquet"
    _concat(actual_minutes).to_parquet(actual_path, index=False)
    stints_frame = _concat(stints)
    stints_path = root / "observed_stints.parquet"
    stints_frame.to_parquet(stints_path, index=False)
    lineups_path = root / "observed_lineups.parquet"
    _observed_lineups(stints_frame).to_parquet(lineups_path, index=False)
    identity_path = root / "player_identity.parquet"
    _concat(identity_rows).to_parquet(identity_path, index=False)
    rosters_path = root / "rosters.json"
    rosters_path.write_text(
        json.dumps({"games": list(rosters)}, indent=2, sort_keys=True)
        + "\n"
    )
    injuries_path = root / "injuries.json"
    injuries_path.write_text(json.dumps(list(injuries), indent=2, sort_keys=True) + "\n")
    inactive_lists_path = root / "nba_inactive_lists.json"
    inactive_lists_path.write_text(
        json.dumps(list(inactive_lists), indent=2, sort_keys=True) + "\n"
    )
    return {
        "final_games": final_games_path,
        "nba_boxscores": boxscores_path,
        "actual_minutes": actual_path,
        "observed_stints": stints_path,
        "observed_lineups": lineups_path,
        "player_identity": identity_path,
        "rosters": rosters_path,
        "injuries": injuries_path,
        "nba_inactive_lists": inactive_lists_path,
    }


def _observed_lineups(stints: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in stints.itertuples(index=False):
        for side in ("home", "away"):
            rows.append(
                {
                    "game_id": str(row.game_id),
                    "team_id": int(getattr(row, f"{side}_team_id")),
                    "team": str(getattr(row, f"{side}_team_tricode")),
                    "player_ids_json": json.dumps(
                        sorted(map(int, getattr(row, f"{side}_player_ids")))
                    ),
                    "possessions": float(row.possessions),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(
            ["game_id", "team_id", "team", "player_ids_json"], as_index=False
        )["possessions"]
        .sum()
        .sort_values(["game_id", "team", "possessions"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fetch_verified_s3_json(uri: str, expected_sha256: str) -> dict[str, Any]:
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-"],
        check=True,
        capture_output=True,
    )
    actual_sha = hashlib.sha256(result.stdout).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(f"S3 source checksum changed: {uri}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {uri}")
    return payload


def _current_chain_tip(root: Path) -> Path | None:
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        return None
    pointer = _read_json(pointer_path)
    return (root / str(pointer["bundle_path"])).resolve()


def _write_failure_report(
    root: Path,
    game_date_pacific: date,
    errors: Sequence[Mapping[str, str]],
) -> Path:
    failure_root = root / "failures" / f"game_date_pacific={game_date_pacific.isoformat()}"
    failure_root.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{html.escape(error['game_id'])}</td>"
        f"<td>{html.escape(error['message'])}</td></tr>"
        for error in errors
    )
    report = failure_root / "report.html"
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>Daily data replay blocked</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.6rem;"
        "border-bottom:1px solid #ddd;text-align:left}.bad{color:#a11}</style>"
        f"<h1 class='bad'>Replay blocked — {game_date_pacific.isoformat()}</h1>"
        "<p>No artifact was published and the data-chain pointer did not advance.</p>"
        f"<table><thead><tr><th>Game</th><th>Error</th></tr></thead><tbody>{rows}</tbody></table>",
        encoding="utf-8",
    )
    (failure_root / "errors.json").write_text(
        json.dumps(list(errors), indent=2, sort_keys=True) + "\n"
    )
    return report


def write_daily_data_chain_report(root: Path | str) -> Path:
    """Write one browser-readable index for the currently promoted data chain."""

    chain_root = Path(root).resolve()
    pointer = _read_json(chain_root / "latest.json")
    bundle = (chain_root / str(pointer["bundle_path"])).resolve()
    manifests: list[tuple[Path, dict[str, Any]]] = []
    while True:
        manifest = verify_daily_data_artifact(bundle)
        manifests.append((bundle, manifest))
        identity = manifest["identity"]
        parent_sha = identity["parent_data_sha256"]
        if parent_sha is None:
            break
        prior_date = date.fromisoformat(identity["game_date_pacific"]) - timedelta(days=1)
        bundle = (
            chain_root
            / f"season={identity['season']}"
            / f"game_date_pacific={prior_date.isoformat()}"
            / f"data_sha256={parent_sha}"
        )
    manifests.reverse()
    game_ids = [
        game_id
        for _bundle, manifest in manifests
        for game_id in manifest["identity"]["game_ids"]
    ]
    source_names = sorted(
        {
            source_name
            for _bundle, manifest in manifests
            for sources in manifest["identity"]["game_sources"].values()
            for source_name in sources
        }
    )
    rows = []
    for artifact_path, manifest in manifests:
        identity = manifest["identity"]
        relative_report = (artifact_path / "report.html").relative_to(chain_root)
        repair_count = len(identity["validation"]["repairs"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(identity['game_date_pacific'])}</td>"
            f"<td>{identity['game_count']}</td>"
            f"<td>{repair_count}</td>"
            f"<td><code>{html.escape(manifest['bundle_sha256'][:12])}</code></td>"
            f"<td><a href='{html.escape(relative_report.as_posix())}'>CLEAN</a></td>"
            "</tr>"
        )
    failure_rows = []
    for errors_path in sorted(chain_root.glob("failures/game_date_pacific=*/errors.json")):
        errors = json.loads(errors_path.read_text())
        failure_report = (errors_path.parent / "report.html").relative_to(chain_root)
        failure_rows.append(
            "<tr>"
            f"<td>{html.escape(errors_path.parent.name.split('=', 1)[-1])}</td>"
            f"<td>{len(errors) if isinstance(errors, list) else '?'}</td>"
            f"<td><a href='{html.escape(failure_report.as_posix())}'>"
            "attempt report</a></td>"
            "</tr>"
        )
    report = chain_root / "report.html"
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>Daily NBA data chain</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;width:100%;margin-bottom:2rem}"
        "td,th{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}"
        ".ok{color:#176b3a;font-weight:700}code{font-size:.9em}</style>"
        "<h1>Official NBA daily data chain</h1>"
        f"<p class='ok'>CLEAN THROUGH {html.escape(pointer['game_date_pacific'])}</p>"
        f"<p>{len(manifests)} Pacific dates · {len(game_ids)} games · "
        f"{len(set(game_ids))} unique games · ESPN dependencies: 0</p>"
        f"<p>Required sources: {html.escape(', '.join(source_names))}</p>"
        "<table><thead><tr><th>Pacific date</th><th>Games</th><th>Audited warnings</th>"
        "<th>Data SHA</th><th>Status</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table><h2>Blocked attempts retained for audit</h2>"
        "<table><thead><tr><th>Date</th><th>Errors</th><th>Report</th></tr></thead><tbody>"
        + ("".join(failure_rows) or "<tr><td colspan='3'>None</td></tr>")
        + "</tbody></table>",
        encoding="utf-8",
    )
    return report


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--coverage-path", type=Path, default=DEFAULT_COVERAGE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--game-catalog-path", type=Path, default=DEFAULT_GAME_CATALOG_PATH
    )
    parser.add_argument(
        "--play-by-play-s3-prefix", default=DEFAULT_PLAY_BY_PLAY_S3_PREFIX
    )
    parser.add_argument("--boxscore-s3-prefix", default=DEFAULT_BOXSCORE_S3_PREFIX)
    parser.add_argument("--injury-s3-prefix", default=DEFAULT_INJURY_S3_PREFIX)
    parser.add_argument("--inactive-s3-prefix", default=DEFAULT_INACTIVE_S3_PREFIX)
    return parser


def main() -> None:
    args = _parser().parse_args()
    artifacts = replay_daily_data_chain(
        start_date=args.start_date,
        end_date=args.end_date,
        coverage_path=args.coverage_path,
        output_root=args.output_root,
        game_catalog_path=args.game_catalog_path,
        play_by_play_s3_prefix=args.play_by_play_s3_prefix,
        boxscore_s3_prefix=args.boxscore_s3_prefix,
        injury_s3_prefix=args.injury_s3_prefix,
        inactive_s3_prefix=args.inactive_s3_prefix,
        season=args.season,
    )
    for artifact in artifacts:
        print(artifact.path)


if __name__ == "__main__":
    main()
