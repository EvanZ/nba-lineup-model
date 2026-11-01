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
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nba_lineup_model.modeling.in_season_nail_monitor import (
    ESPN_TEAM_ALIASES,
    _name_aliases,
    _player_name_lookup,
    _rapm_stints,
)
from nba_lineup_model.season.daily_artifact_common import file_sha256
from nba_lineup_model.season.daily_data_artifact import (
    PublishedDailyDataArtifact,
    promote_daily_data_artifact,
    publish_daily_data_artifact,
)
from nba_lineup_model.season.slate_time import summary_slate_date

DEFAULT_SEASON = "2026-27"
DEFAULT_STATE_DIR = Path("/var/lib/nba-gestalt/schedule")
DEFAULT_COVERAGE_PATH = Path("/var/lib/nba-gestalt/cdn/coverage.json")
DEFAULT_OUTPUT_ROOT = Path("/var/lib/nba-gestalt-dev/daily-data")
DEFAULT_ROSTER_POINTER = Path("/var/lib/nba-gestalt-dev/in-season-rosters/latest.json")
DEFAULT_PLAYER_CATALOG_PATH = Path(
    "/opt/nba-gestalt/current/data/catalog/players.parquet"
)
DEFAULT_CDN_S3_PREFIX = "s3://nba-gestalt-data/lake/raw/cdn/playbyplay"


class DailyDataReplayBlocked(RuntimeError):
    """Raised after a date-level failure report has been written."""


def replay_daily_data_chain(
    *,
    start_date: date,
    end_date: date,
    state_dir: Path | str = DEFAULT_STATE_DIR,
    coverage_path: Path | str = DEFAULT_COVERAGE_PATH,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    roster_pointer: Path | str = DEFAULT_ROSTER_POINTER,
    player_catalog_path: Path | str = DEFAULT_PLAYER_CATALOG_PATH,
    cdn_s3_prefix: str = DEFAULT_CDN_S3_PREFIX,
    season: str = DEFAULT_SEASON,
) -> list[PublishedDailyDataArtifact]:
    """Replay dates chronologically and stop before the first bad publication."""

    if end_date < start_date:
        raise ValueError("Replay end date cannot precede its start date")
    state_root = Path(state_dir)
    chain_root = Path(output_root)
    summaries = _load_final_summaries(state_root)
    summaries_by_date: dict[date, list[tuple[Path, dict[str, Any]]]] = {}
    for path, summary in summaries:
        slate_date = summary_slate_date(summary)
        if start_date <= slate_date <= end_date:
            summaries_by_date.setdefault(slate_date, []).append((path, summary))
    coverage = _read_json(Path(coverage_path)).get("games")
    if not isinstance(coverage, dict):
        raise ValueError("CDN coverage manifest has no game map")
    roster_manifest = _read_json(Path(roster_pointer))
    reference_path = Path(str(roster_manifest.get("artifacts", {}).get("preseason_rankings", "")))
    if not reference_path.is_file():
        raise FileNotFoundError(f"Player identity reference is missing: {reference_path}")
    reference = pd.read_parquet(reference_path)
    catalog_path = Path(player_catalog_path)
    catalog = pd.read_parquet(catalog_path)
    catalog_reference = catalog[["player_id", "display_name"]].rename(
        columns={"display_name": "player_name"}
    )
    # The historical catalog can contain retired namesakes. Prefer the current
    # season's reconciled player universe, then use globally unique catalog
    # aliases only as a fallback for inactive/DNP rows.
    name_lookup = _player_identity_lookup(
        reference[["player_id", "player_name"]], catalog_reference
    )

    parent_bundle = _current_chain_tip(chain_root)
    if parent_bundle is not None:
        pointer = _read_json(chain_root / "latest.json")
        next_date = date.fromisoformat(str(pointer["game_date_pacific"])) + timedelta(days=1)
        if next_date != start_date:
            raise ValueError(
                f"Replay must resume at {next_date.isoformat()}, not {start_date.isoformat()}"
            )

    published: list[PublishedDailyDataArtifact] = []
    current_date = start_date
    while current_date <= end_date:
        games = sorted(
            summaries_by_date.get(current_date, []),
            key=lambda item: (
                str(item[1].get("pregame", {}).get("start_time", "")),
                str(item[1].get("game_id", "")),
            ),
        )
        artifact = _build_date(
            game_date_pacific=current_date,
            games=games,
            coverage=coverage,
            name_lookup=name_lookup,
            roster_manifest=roster_manifest,
            reference_path=reference_path,
            player_catalog_path=catalog_path,
            output_root=chain_root,
            parent_bundle=parent_bundle,
            cdn_s3_prefix=cdn_s3_prefix,
            season=season,
        )
        promote_daily_data_artifact(artifact.path, root=chain_root)
        published.append(artifact)
        parent_bundle = artifact.path
        current_date += timedelta(days=1)
    return published


def _build_date(
    *,
    game_date_pacific: date,
    games: Sequence[tuple[Path, dict[str, Any]]],
    coverage: Mapping[str, object],
    name_lookup: dict[str, int],
    roster_manifest: Mapping[str, Any],
    reference_path: Path,
    player_catalog_path: Path,
    output_root: Path,
    parent_bundle: Path | None,
    cdn_s3_prefix: str,
    season: str,
) -> PublishedDailyDataArtifact:
    errors: list[dict[str, str]] = []
    repairs: list[dict[str, str]] = []
    game_sources: dict[str, dict[str, dict[str, str]]] = {}
    game_quality: dict[str, dict[str, float | int]] = {}
    summaries: list[dict[str, Any]] = []
    actual_minutes: list[pd.DataFrame] = []
    stints: list[pd.DataFrame] = []
    identity_rows: list[pd.DataFrame] = []
    rosters: list[dict[str, Any]] = []
    injuries: list[dict[str, Any]] = []

    for summary_path, summary in games:
        game_id = str(summary.get("game_id", "")).strip()
        if not game_id:
            errors.append({"game_id": "", "message": f"Missing game ID: {summary_path}"})
            continue
        try:
            espn_path = summary_path.with_name(
                summary_path.name.removesuffix(".summary.json") + ".espn.json"
            )
            summary_bytes = summary_path.read_bytes()
            espn_bytes = espn_path.read_bytes()
            espn = json.loads(espn_bytes)
            versions = coverage.get(game_id)
            if not isinstance(versions, list) or not versions:
                raise ValueError("CDN coverage has no immutable play-by-play version")
            play_by_play_sha = str(versions[-1])
            play_by_play_uri = (
                f"{cdn_s3_prefix.rstrip('/')}/season={season}/"
                f"game_id={game_id}/{play_by_play_sha}.json"
            )
            play_by_play = _fetch_verified_s3_json(play_by_play_uri, play_by_play_sha)
            game_stints, _team_ids, issue_count = _rapm_stints(
                summary=summary,
                espn=espn,
                play_by_play=play_by_play,
                play_by_play_sha256=play_by_play_sha,
                espn_sha256=hashlib.sha256(espn_bytes).hexdigest(),
                name_to_player_id=name_lookup,
            )
            game_actual, game_identity = _actual_minutes(
                summary=summary,
                espn=espn,
                name_lookup=name_lookup,
            )
            final_home, final_away = _final_scores(summary)
            box_home, box_away = _espn_scores(summary, espn)
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
                        "source": "CDN play-by-play",
                    }
                )
            game_sources[game_id] = {
                "final_summary": {
                    "uri": summary_path.resolve().as_uri(),
                    "sha256": hashlib.sha256(summary_bytes).hexdigest(),
                },
                "espn_box_score": {
                    "uri": espn_path.resolve().as_uri(),
                    "sha256": hashlib.sha256(espn_bytes).hexdigest(),
                },
                "play_by_play": {
                    "uri": play_by_play_uri,
                    "sha256": play_by_play_sha,
                },
                "player_reference": {
                    "uri": reference_path.resolve().as_uri(),
                    "sha256": file_sha256(reference_path),
                },
                "player_catalog": {
                    "uri": player_catalog_path.resolve().as_uri(),
                    "sha256": file_sha256(player_catalog_path),
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
            summaries.append(summary)
            actual_minutes.append(game_actual)
            identity_rows.append(game_identity)
            stints.append(game_stints)
            rosters.append(
                {
                    "game_id": game_id,
                    "home_team": summary["home_team"],
                    "away_team": summary["away_team"],
                    "projected_minutes": summary.get("pregame", {}).get(
                        "projected_minutes", {}
                    ),
                }
            )
            injuries.append(
                {
                    "game_id": game_id,
                    "injury_report": summary.get("pregame", {}).get("injury_report"),
                }
            )
        except Exception as error:  # date-level report must inventory every failed game
            errors.append({"game_id": game_id, "message": f"{type(error).__name__}: {error}"})

    expected_game_ids = sorted(str(summary[1]["game_id"]) for summary in games)
    if errors:
        report_path = _write_failure_report(output_root, game_date_pacific, errors)
        raise DailyDataReplayBlocked(
            f"Pacific date {game_date_pacific.isoformat()} failed; report: {report_path}"
        )

    with tempfile.TemporaryDirectory(prefix="daily-data-components-") as directory:
        component_root = Path(directory)
        components = _write_components(
            component_root,
            summaries=summaries,
            actual_minutes=actual_minutes,
            stints=stints,
            identity_rows=identity_rows,
            rosters=rosters,
            injuries=injuries,
            roster_manifest=roster_manifest,
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


def _actual_minutes(
    *,
    summary: Mapping[str, Any],
    espn: Mapping[str, Any],
    name_lookup: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    game_id = str(summary["game_id"])
    valid_teams = {str(summary["home_team"]), str(summary["away_team"])}
    for team_row in espn.get("boxscore", {}).get("players", []):
        raw_team = str(team_row.get("team", {}).get("abbreviation", ""))
        team = ESPN_TEAM_ALIASES.get(raw_team, raw_team)
        if team not in valid_teams:
            continue
        for statistic in team_row.get("statistics", []):
            labels = list(statistic.get("labels", []))
            for athlete_row in statistic.get("athletes", []):
                athlete = athlete_row.get("athlete", {})
                player_name = str(athlete.get("displayName", "")).strip()
                player_id = next(
                    (
                        name_lookup[alias]
                        for alias in _name_aliases(player_name)
                        if alias in name_lookup
                    ),
                    None,
                )
                stats = list(athlete_row.get("stats", []))
                stat_map = dict(zip(labels, stats, strict=False))
                raw_minutes = stat_map.get("MIN", stats[0] if stats else "0")
                minutes = _minutes_value(raw_minutes)
                played = minutes > 0
                if player_id is None:
                    raise ValueError(
                        f"Could not map ESPN athlete {player_name!r} for {team}"
                    )
                identity_rows.append(
                    {
                        "game_id": game_id,
                        "team": team,
                        "espn_athlete_id": str(athlete.get("id", "")),
                        "player_name": player_name,
                        "player_id": player_id,
                        "played": played,
                        "resolution": "name_alias",
                    }
                )
                if played:
                    actual_rows.append(
                        {
                            "game_id": game_id,
                            "game_date_pacific": summary_slate_date(summary).isoformat(),
                            "team": team,
                            "player_id": int(player_id),
                            "player_name": player_name,
                            "minutes": minutes,
                        }
                    )
    actual = pd.DataFrame(actual_rows)
    if actual.empty:
        raise ValueError("ESPN box score has no played athletes")
    return actual, pd.DataFrame(identity_rows)


def _player_identity_lookup(
    current_reference: pd.DataFrame,
    historical_catalog: pd.DataFrame,
) -> dict[str, int]:
    return {
        **_player_name_lookup(historical_catalog),
        **_player_name_lookup(current_reference),
    }


def _minutes_value(value: object) -> float:
    text = str(value).strip()
    if not text or text in {"--", "-"}:
        return 0.0
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60.0
    return float(text)


def _final_scores(summary: Mapping[str, Any]) -> tuple[int, int]:
    final = summary.get("final", {})
    return int(final["home_score"]), int(final["away_score"])


def _espn_scores(
    summary: Mapping[str, Any], espn: Mapping[str, Any]
) -> tuple[int, int]:
    competitions = espn.get("header", {}).get("competitions", [])
    if not competitions:
        raise ValueError("ESPN box score has no competition header")
    scores: dict[str, int] = {}
    for competitor in competitions[0].get("competitors", []):
        raw_team = str(competitor.get("team", {}).get("abbreviation", ""))
        team = ESPN_TEAM_ALIASES.get(raw_team, raw_team)
        scores[team] = int(competitor["score"])
    return scores[str(summary["home_team"])], scores[str(summary["away_team"])]


def _write_components(
    root: Path,
    *,
    summaries: Sequence[Mapping[str, Any]],
    actual_minutes: Sequence[pd.DataFrame],
    stints: Sequence[pd.DataFrame],
    identity_rows: Sequence[pd.DataFrame],
    rosters: Sequence[Mapping[str, Any]],
    injuries: Sequence[Mapping[str, Any]],
    roster_manifest: Mapping[str, Any],
) -> dict[str, Path]:
    final_games_path = root / "final_games.json"
    final_games_path.write_text(json.dumps(list(summaries), indent=2, sort_keys=True) + "\n")
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
        json.dumps(
            {"games": list(rosters), "reconciliation": dict(roster_manifest)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    injuries_path = root / "injuries.json"
    injuries_path.write_text(json.dumps(list(injuries), indent=2, sort_keys=True) + "\n")
    return {
        "final_games": final_games_path,
        "actual_minutes": actual_path,
        "observed_stints": stints_path,
        "observed_lineups": lineups_path,
        "player_identity": identity_path,
        "rosters": rosters_path,
        "injuries": injuries_path,
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


def _load_final_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    output: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "final").glob("*.summary.json")):
        summary = _read_json(path)
        if summary.get("final", {}).get("status", {}).get("completed") and str(
            summary.get("game_id", "")
        ).startswith("002"):
            output.append((path, summary))
    return output


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
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--coverage-path", type=Path, default=DEFAULT_COVERAGE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--roster-pointer", type=Path, default=DEFAULT_ROSTER_POINTER)
    parser.add_argument(
        "--player-catalog-path", type=Path, default=DEFAULT_PLAYER_CATALOG_PATH
    )
    parser.add_argument("--cdn-s3-prefix", default=DEFAULT_CDN_S3_PREFIX)
    return parser


def main() -> None:
    args = _parser().parse_args()
    artifacts = replay_daily_data_chain(
        start_date=args.start_date,
        end_date=args.end_date,
        state_dir=args.state_dir,
        coverage_path=args.coverage_path,
        output_root=args.output_root,
        roster_pointer=args.roster_pointer,
        player_catalog_path=args.player_catalog_path,
        cdn_s3_prefix=args.cdn_s3_prefix,
        season=args.season,
    )
    for artifact in artifacts:
        print(artifact.path)


if __name__ == "__main__":
    main()
