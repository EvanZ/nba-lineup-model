"""Probe NBA Game Rotation coverage for games failing lineup reconstruction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from prefect.task_runners import ThreadPoolTaskRunner

from nba_lineup_model.flows.fetch_stats_history import fetch_stats_history_flow
from nba_lineup_model.ingest.nba_stats import NbaStatsEndpoint, NbaStatsRawCache
from nba_lineup_model.season.schema import CatalogGame, GameBuildRecord
from nba_lineup_model.season.storage import read_build_ledger, read_game_catalog


@dataclass(frozen=True)
class RotationEvidence:
    """Structural availability of one cached Game Rotation response."""

    available: bool
    away_interval_count: int
    home_interval_count: int
    reason: str | None = None


def latest_failed_builds(records: list[GameBuildRecord]) -> list[GameBuildRecord]:
    """Return games whose latest terminal processing record is a failure."""

    latest_by_game: dict[str, GameBuildRecord] = {}
    for record in sorted(records, key=lambda item: item.finished_at):
        latest_by_game[record.game_id] = record
    return sorted(
        (
            record
            for record in latest_by_game.values()
            if record.status == "failed" and record.season_type == "regular"
        ),
        key=lambda item: (item.season, item.game_id),
    )


def select_probe_games(
    catalog: list[CatalogGame],
    failed_builds: list[GameBuildRecord],
    *,
    per_season: int,
) -> list[tuple[CatalogGame, GameBuildRecord]]:
    """Select a deterministic bounded sample of latest failures by season."""

    if per_season < 1:
        raise ValueError("per_season must be positive")
    catalog_by_id = {game.game_id: game for game in catalog}
    selected: list[tuple[CatalogGame, GameBuildRecord]] = []
    by_season: dict[str, list[GameBuildRecord]] = {}
    for record in failed_builds:
        if record.game_id in catalog_by_id:
            by_season.setdefault(record.season, []).append(record)
    for season in sorted(by_season):
        for record in by_season[season][:per_season]:
            selected.append((catalog_by_id[record.game_id], record))
    return selected


def game_rotation_evidence(payload: dict[str, object]) -> RotationEvidence:
    """Determine whether both teams expose structurally usable intervals."""

    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        return RotationEvidence(False, 0, 0, "missing_result_sets")
    interval_counts: dict[str, int] = {}
    for result_set in result_sets:
        if not isinstance(result_set, dict):
            continue
        name = result_set.get("name")
        if name not in {"AwayTeam", "HomeTeam"}:
            continue
        headers = result_set.get("headers")
        rows = result_set.get("rowSet")
        if not isinstance(headers, list) or not isinstance(rows, list):
            return RotationEvidence(False, 0, 0, f"malformed_{name}")
        required = {"PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"}
        if not required.issubset(headers):
            return RotationEvidence(False, 0, 0, f"missing_interval_fields_{name}")
        indexes = {field: headers.index(field) for field in required}
        valid_count = 0
        for row in rows:
            if not isinstance(row, list) or len(row) < len(headers):
                return RotationEvidence(False, 0, 0, f"malformed_row_{name}")
            player_id = row[indexes["PERSON_ID"]]
            in_time = row[indexes["IN_TIME_REAL"]]
            out_time = row[indexes["OUT_TIME_REAL"]]
            if (
                isinstance(player_id, bool)
                or not isinstance(player_id, int)
                or player_id <= 0
                or not _finite_number(in_time)
                or not _finite_number(out_time)
                or float(out_time) <= float(in_time)
            ):
                return RotationEvidence(False, 0, 0, f"invalid_interval_{name}")
            valid_count += 1
        if valid_count == 0:
            return RotationEvidence(False, 0, 0, f"empty_{name}")
        interval_counts[name] = valid_count
    if set(interval_counts) != {"AwayTeam", "HomeTeam"}:
        return RotationEvidence(False, 0, 0, "missing_team_rotation")
    return RotationEvidence(
        True,
        away_interval_count=interval_counts["AwayTeam"],
        home_interval_count=interval_counts["HomeTeam"],
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Game Rotation availability for latest failed regular-season builds."
    )
    parser.add_argument("--catalog", default="data/catalog/games.parquet")
    parser.add_argument("--builds", default="data/manifests/builds.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifests/stats_fetches.parquet")
    parser.add_argument("--output-dir", default="artifacts/reports/game_rotation_probe")
    parser.add_argument("--per-season", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--min-request-interval", type=float, default=1.0)
    parser.add_argument("--request-interval-jitter", type=float, default=0.25)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.per_season < 1 or args.max_workers < 1:
        raise SystemExit("--per-season and --max-workers must be positive")

    catalog = read_game_catalog(args.catalog).games
    failures = latest_failed_builds(read_build_ledger(args.builds).records)
    selected = select_probe_games(catalog, failures, per_season=args.per_season)
    if not selected:
        raise SystemExit("No latest failed regular-season builds are available to probe")

    started_at = datetime.now(UTC)
    run_id = args.run_id or f"game-rotation-probe-{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    games = [game for game, _record in selected]
    fetch_summary = fetch_stats_history_flow.with_options(
        task_runner=ThreadPoolTaskRunner(max_workers=args.max_workers)
    )(
        sorted({game.season for game in games}),
        catalog_path=args.catalog,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        season_types=["regular"],
        game_ids=[game.game_id for game in games],
        endpoints=[NbaStatsEndpoint.GAME_ROTATION],
        run_id=run_id,
        min_request_interval_seconds=args.min_request_interval,
        request_interval_jitter_seconds=args.request_interval_jitter,
        max_retries=0,
    )
    cache = NbaStatsRawCache(Path(args.raw_dir) / "stats")
    rows: list[dict[str, object]] = []
    for game, record in selected:
        response = cache.read(NbaStatsEndpoint.GAME_ROTATION, game.game_id)
        evidence = (
            game_rotation_evidence(response.payload)
            if response is not None
            else RotationEvidence(False, 0, 0, "response_not_cached")
        )
        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "game_date": game.game_date,
                "failure_stage": record.terminal_stage,
                "failure_message": record.error_message,
                **asdict(evidence),
            }
        )

    report_dir = Path(args.output_dir) / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_parquet(report_dir / "games.parquet", index=False)
    available_count = sum(bool(row["available"]) for row in rows)
    summary = {
        "run_id": run_id,
        "selected_games": len(rows),
        "rotation_available_games": available_count,
        "rotation_availability_rate": available_count / len(rows),
        "fetch_summary": fetch_summary.model_dump(mode="json"),
        "report_path": str(report_dir / "games.parquet"),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Game Rotation probe {run_id}: selected={len(rows)}, "
        f"available={available_count}, rate={available_count / len(rows):.1%}; "
        f"report={report_dir}"
    )


if __name__ == "__main__":
    main()
